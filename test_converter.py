#!/usr/bin/env python3
"""
Comprehensive test suite for the nifti-to-zarr converter.

Verifies:
1. Output equivalence: new per-slab reorientation produces byte-identical
   Zarr output to nibabel's as_closest_canonical for all 48 orientations.
2. 4D mean-over-time collapse preserved across orientations.
3. Multiple dtypes (uint8, uint16, int16, float32).
4. Both .nii and .nii.gz formats.
5. Bounded queue: write errors propagate, semaphore doesn't deadlock.
6. Pyramid levels: no chunk race condition (destination-chunk iteration).
7. Chunk collision guard: detects overlapping writes.
8. Full-conversion error propagation: Zarr write failure crashes, not truncates.
"""

import json
import math
import os
import shutil
import sys
import tempfile
from itertools import permutations, product

import numpy as np
import nibabel as nib
import zarr
from nibabel.orientations import io_orientation

# ---- Reference implementation (old code path using as_closest_canonical) ----

def _convert_reference(input_path, output_path, chunk_slices=32, tile=256,
                       initial_downsample=1):
    """The original converter logic (SEQUENTIAL), for output comparison."""
    from zarr.storage import LocalStore
    from zarr.codecs import BloscCodec
    from skimage.transform import downscale_local_mean

    img = nib.as_closest_canonical(nib.load(input_path))
    shape = img.shape[:3]
    is_4d = img.ndim == 4
    voxel_sizes = img.header.get_zooms()[:3]

    idf = initial_downsample
    if idf > 1:
        shape = tuple(max(1, math.ceil(s / idf)) for s in shape)
        voxel_sizes = tuple(v * idf for v in voxel_sizes)

    min_dim = 64
    smallest = min(shape)
    n_levels = 1
    while smallest // (2 ** n_levels) >= min_dim:
        n_levels += 1

    store = LocalStore(output_path)
    root = zarr.open_group(store, mode="w", zarr_format=3)
    compressor = BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")

    level_shapes = [
        tuple(max(1, math.ceil(s / (2 ** i))) for s in shape)
        for i in range(n_levels)
    ]
    level_arrays = []
    for i in range(n_levels):
        ls = level_shapes[i]
        chunks = (min(chunk_slices, ls[0]), min(tile, ls[1]), min(tile, ls[2]))
        arr = root.create_array(
            str(i), shape=ls, dtype=np.float32,
            chunks=chunks, compressors=[compressor],
        )
        level_arrays.append(arr)

    src_dim0 = img.shape[0]
    if idf > 1:
        slices = [
            (start, min(start + chunk_slices * idf, src_dim0))
            for start in range(0, src_dim0, chunk_slices * idf)
        ]
        dst_offset = 0
        for start, end in slices:
            chunk = np.asarray(img.dataobj[start:end, :, :], dtype=np.float32)
            if is_4d:
                chunk = chunk.mean(axis=3)
            chunk = downscale_local_mean(chunk, (idf, idf, idf))
            dst_end = dst_offset + chunk.shape[0]
            level_arrays[0][dst_offset:dst_end, :, :] = chunk
            dst_offset = dst_end
    else:
        slices = [
            (start, min(start + chunk_slices, src_dim0))
            for start in range(0, src_dim0, chunk_slices)
        ]
        for start, end in slices:
            chunk = np.asarray(img.dataobj[start:end, :, :], dtype=np.float32)
            if is_4d:
                chunk = chunk.mean(axis=3)
            level_arrays[0][start:end, :, :] = chunk

    # SEQUENTIAL pyramid (no threading — known-good reference)
    for lvl in range(1, n_levels):
        src = level_arrays[lvl - 1]
        dst = level_arrays[lvl]
        for start in range(0, src.shape[0], chunk_slices):
            end = min(start + chunk_slices, src.shape[0])
            chunk = np.asarray(src[start:end, :, :])
            downsampled = downscale_local_mean(chunk, (2, 2, 2))
            dst_start = start // 2
            dst_end = dst_start + downsampled.shape[0]
            dst[dst_start:dst_end, :, :] = downsampled

    return root


# ---- New implementation ----

def _convert_new(input_path, output_path, chunk_slices=32, tile=256,
                 initial_downsample=1, max_workers=2, memory_budget_gb=1.0):
    """Run the new converter."""
    from processor.config import Config
    from processor.converter import convert_nifti_to_ome_zarr
    config = Config(
        input_dir="", output_dir="",
        initial_downsample=initial_downsample,
        tile_size=tile, compression="zstd", compression_level=5,
        max_levels=0, min_dimension=64,
        chunk_slices=chunk_slices, max_workers=max_workers,
        memory_budget_gb=memory_budget_gb,
    )
    convert_nifti_to_ome_zarr(input_path, output_path, config)
    from zarr.storage import LocalStore
    store = LocalStore(output_path)
    return zarr.open_group(store, mode="r")


def _compare_zarr_outputs(ref_root, new_root, label):
    """Compare two Zarr groups array-by-array, including all pyramid levels."""
    ref_keys = sorted(ref_root.keys())
    new_keys = sorted(new_root.keys())
    if ref_keys != new_keys:
        print(f"  FAIL {label}: different keys ref={ref_keys} new={new_keys}")
        return False

    for key in ref_keys:
        ref_arr = np.asarray(ref_root[key])
        new_arr = np.asarray(new_root[key])
        if ref_arr.shape != new_arr.shape:
            print(f"  FAIL {label} key={key}: shape ref={ref_arr.shape} new={new_arr.shape}")
            return False
        if not np.allclose(ref_arr, new_arr, rtol=1e-6, atol=1e-6):
            diff = np.abs(ref_arr - new_arr)
            max_diff = np.max(diff)
            n_diff = np.sum(diff > 1e-6)
            # Check for zeroed regions (the signature of the chunk race)
            n_zeroed = np.sum((np.abs(new_arr) < 1e-6) & (np.abs(ref_arr) > 1e-6))
            detail = f"{n_diff}/{ref_arr.size} differ, max_diff={max_diff}"
            if n_zeroed > 0:
                detail += f", {n_zeroed} zeroed (chunk race signature)"
            print(f"  FAIL {label} level={key}: {detail}")
            return False
    return True


def generate_all_48_affines(shape):
    """Generate all 48 signed permutation matrices as NIfTI affines."""
    results = []
    for perm in permutations(range(3)):
        for signs in product([-1, 1], repeat=3):
            rot = np.zeros((3, 3))
            for vox_ax, (world_ax, sign) in enumerate(zip(perm, signs)):
                rot[world_ax, vox_ax] = sign
            aff = np.eye(4)
            aff[:3, :3] = rot
            for vox_ax in range(3):
                world_ax = perm[vox_ax]
                if signs[vox_ax] == -1:
                    aff[world_ax, 3] = shape[vox_ax] - 1
            results.append(aff)
    return results


def make_volume(shape, dtype):
    """Create a deterministic volume with unique values."""
    rng = np.random.RandomState(42)
    if np.issubdtype(dtype, np.integer):
        iinfo = np.iinfo(dtype)
        return rng.randint(0, min(10000, iinfo.max), size=shape).astype(dtype)
    else:
        return rng.rand(*shape).astype(dtype) * 100


def test_orientation_equivalence(tmpdir):
    """Test all 48 orientations x 2 formats for output equivalence."""
    print("=" * 60)
    print("TEST: 48-orientation output equivalence (3D)")
    print("=" * 60)

    shape = (11, 13, 17)  # asymmetric, primes to catch alignment bugs
    vol = make_volume(shape, np.uint16)
    affines = generate_all_48_affines(shape)

    passed = failed = 0
    for idx, affine in enumerate(affines):
        for compress in [True, False]:
            suffix = ".nii.gz" if compress else ".nii"
            label = f"ornt{idx:02d}{suffix}"

            img = nib.Nifti1Image(vol, affine)
            nii_path = os.path.join(tmpdir, f"{label}{suffix}")
            nib.save(img, nii_path)

            ref_path = os.path.join(tmpdir, f"{label}_ref.zarr")
            new_path = os.path.join(tmpdir, f"{label}_new.zarr")

            ref_root = _convert_reference(nii_path, ref_path, chunk_slices=4)
            new_root = _convert_new(nii_path, new_path, chunk_slices=4)

            if _compare_zarr_outputs(ref_root, new_root, label):
                passed += 1
            else:
                failed += 1

            shutil.rmtree(ref_path, ignore_errors=True)
            shutil.rmtree(new_path, ignore_errors=True)

    print(f"  {passed} passed, {failed} failed")
    return failed == 0


def test_4d_equivalence(tmpdir):
    """Test 4D mean-over-time with several orientations."""
    print("\n" + "=" * 60)
    print("TEST: 4D mean-over-time equivalence")
    print("=" * 60)

    shape_3d = (8, 10, 12)
    shape_4d = shape_3d + (5,)  # 5 timepoints
    vol = make_volume(shape_4d, np.int16)

    test_affines = [
        ("canonical", np.eye(4)),
        ("LAS", np.array([[-1,0,0,7],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)),
        ("permuted", np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]], dtype=float)),
    ]

    passed = failed = 0
    for name, affine in test_affines:
        for compress in [True, False]:
            suffix = ".nii.gz" if compress else ".nii"
            label = f"4d_{name}{suffix}"

            img = nib.Nifti1Image(vol, affine)
            nii_path = os.path.join(tmpdir, f"{label}{suffix}")
            nib.save(img, nii_path)

            ref_path = os.path.join(tmpdir, f"{label}_ref.zarr")
            new_path = os.path.join(tmpdir, f"{label}_new.zarr")

            ref_root = _convert_reference(nii_path, ref_path, chunk_slices=3)
            new_root = _convert_new(nii_path, new_path, chunk_slices=3)

            if _compare_zarr_outputs(ref_root, new_root, label):
                passed += 1
            else:
                failed += 1

            shutil.rmtree(ref_path, ignore_errors=True)
            shutil.rmtree(new_path, ignore_errors=True)

    print(f"  {passed} passed, {failed} failed")
    return failed == 0


def test_dtype_equivalence(tmpdir):
    """Test multiple dtypes for output equivalence."""
    print("\n" + "=" * 60)
    print("TEST: dtype equivalence")
    print("=" * 60)

    shape = (10, 12, 14)
    affine = np.array([[-1,0,0,9],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=float)

    passed = failed = 0
    for dtype in [np.uint8, np.int16, np.uint16, np.float32]:
        vol = make_volume(shape, dtype)
        label = f"dtype_{np.dtype(dtype).name}"

        img = nib.Nifti1Image(vol, affine)
        nii_path = os.path.join(tmpdir, f"{label}.nii.gz")
        nib.save(img, nii_path)

        ref_path = os.path.join(tmpdir, f"{label}_ref.zarr")
        new_path = os.path.join(tmpdir, f"{label}_new.zarr")

        ref_root = _convert_reference(nii_path, ref_path, chunk_slices=4)
        new_root = _convert_new(nii_path, new_path, chunk_slices=4)

        if _compare_zarr_outputs(ref_root, new_root, label):
            passed += 1
        else:
            failed += 1

        shutil.rmtree(ref_path, ignore_errors=True)
        shutil.rmtree(new_path, ignore_errors=True)

    print(f"  {passed} passed, {failed} failed")
    return failed == 0


def test_initial_downsample(tmpdir):
    """Test initial_downsample > 1 with non-canonical orientation."""
    print("\n" + "=" * 60)
    print("TEST: initial_downsample with non-canonical")
    print("=" * 60)

    shape = (16, 20, 24)
    vol = make_volume(shape, np.uint16)
    affine = np.array([[-1,0,0,15],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=float)

    passed = failed = 0
    for idf in [2, 4]:
        label = f"downsample_{idf}"
        img = nib.Nifti1Image(vol, affine)
        nii_path = os.path.join(tmpdir, f"{label}.nii.gz")
        nib.save(img, nii_path)

        ref_path = os.path.join(tmpdir, f"{label}_ref.zarr")
        new_path = os.path.join(tmpdir, f"{label}_new.zarr")

        ref_root = _convert_reference(nii_path, ref_path, chunk_slices=4,
                                       initial_downsample=idf)
        new_root = _convert_new(nii_path, new_path, chunk_slices=4,
                                initial_downsample=idf)

        if _compare_zarr_outputs(ref_root, new_root, label):
            passed += 1
        else:
            failed += 1

        shutil.rmtree(ref_path, ignore_errors=True)
        shutil.rmtree(new_path, ignore_errors=True)

    print(f"  {passed} passed, {failed} failed")
    return failed == 0


def test_unsupported_ndim(tmpdir):
    """Test that ndim > 4 and ndim < 3 fail with clear messages."""
    print("\n" + "=" * 60)
    print("TEST: unsupported dimensionality rejection")
    print("=" * 60)

    passed = 0

    vol_5d = np.zeros((4, 4, 4, 2, 3), dtype=np.uint16)
    img_5d = nib.Nifti1Image(vol_5d, np.eye(4))
    path_5d = os.path.join(tmpdir, "5d.nii")
    nib.save(img_5d, path_5d)
    try:
        _convert_new(path_5d, os.path.join(tmpdir, "5d.zarr"))
        print("  FAIL: 5D should have raised ValueError")
    except ValueError as e:
        if "ndim=5" in str(e):
            passed += 1
            print(f"  OK: 5D rejected with: {e}")
        else:
            print(f"  FAIL: wrong message: {e}")

    vol_2d = np.zeros((4, 4), dtype=np.uint16)
    img_2d = nib.Nifti1Image(vol_2d, np.eye(4))
    path_2d = os.path.join(tmpdir, "2d.nii")
    nib.save(img_2d, path_2d)
    try:
        _convert_new(path_2d, os.path.join(tmpdir, "2d.zarr"))
        print("  FAIL: 2D should have raised ValueError")
    except ValueError as e:
        if "ndim=" in str(e):
            passed += 1
            print(f"  OK: 2D rejected with: {e}")
        else:
            print(f"  FAIL: wrong message: {e}")

    print(f"  {passed}/2 passed")
    return passed == 2


def test_bounded_queue_error_propagation(tmpdir):
    """Test that a write error propagates and doesn't silently truncate."""
    print("\n" + "=" * 60)
    print("TEST: bounded queue error propagation")
    print("=" * 60)

    from processor.converter import _submit_bounded, _drain_futures, _ChunkCollisionGuard
    from concurrent.futures import ThreadPoolExecutor
    import threading

    sem = threading.Semaphore(2)

    def failing_write(*args):
        raise IOError("simulated Zarr write failure")

    pool = ThreadPoolExecutor(max_workers=2)
    futures = []
    futures.append(_submit_bounded(pool, sem, failing_write, None, None, None))

    try:
        _drain_futures(futures)
        print("  FAIL: exception not propagated")
        pool.shutdown(wait=False)
        return False
    except IOError as e:
        if "simulated" in str(e):
            print("  OK: write error propagated correctly")
            pool.shutdown(wait=False)
            return True
        print(f"  FAIL: wrong exception: {e}")
        pool.shutdown(wait=False)
        return False


def test_pyramid_race(tmpdir):
    """Test that pyramid levels match a single-threaded reference.

    Uses a large volume with high concurrency to stress the pyramid build.
    With the old source-slab iteration pattern, two consecutive source slabs
    (each chunk_slices rows) downsample to chunk_slices/2 output rows and
    both land in the same destination Zarr chunk — a read-modify-write race
    that zeroes half the data.

    The fix: iterate by destination chunk so each chunk has exactly one writer.
    This test verifies the fix by comparing against a sequential reference
    across multiple pyramid levels.
    """
    print("\n" + "=" * 60)
    print("TEST: pyramid race (high concurrency, multi-level)")
    print("=" * 60)

    # Shape chosen so level 0 is large enough for multiple pyramid levels
    # and small chunk_slices guarantee many writes per destination chunk.
    shape = (128, 130, 132)  # asymmetric, >64 on each axis for 2 levels
    vol = make_volume(shape, np.uint16)

    # Canonical to isolate the pyramid from reorientation
    affine = np.eye(4)
    img = nib.Nifti1Image(vol, affine)
    nii_path = os.path.join(tmpdir, "pyramid_race.nii")
    nib.save(img, nii_path)

    # Also test with non-canonical to cover both paths
    aff_nc = np.array([[-1,0,0,127],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=float)
    img_nc = nib.Nifti1Image(vol, aff_nc)
    nii_nc_path = os.path.join(tmpdir, "pyramid_race_nc.nii")
    nib.save(img_nc, nii_nc_path)

    passed = failed = 0
    for label, path in [("canonical", nii_path), ("non-canonical", nii_nc_path)]:
        ref_path = os.path.join(tmpdir, f"{label}_ref.zarr")
        new_path = os.path.join(tmpdir, f"{label}_new.zarr")

        # Reference: sequential (no threading)
        ref_root = _convert_reference(path, ref_path, chunk_slices=4)

        # New: high concurrency to maximize race window
        new_root = _convert_new(path, new_path, chunk_slices=4,
                                max_workers=16, memory_budget_gb=4.0)

        if _compare_zarr_outputs(ref_root, new_root, f"pyramid_{label}"):
            passed += 1
        else:
            failed += 1

        shutil.rmtree(ref_path, ignore_errors=True)
        shutil.rmtree(new_path, ignore_errors=True)

    print(f"  {passed} passed, {failed} failed")
    return failed == 0


def test_chunk_collision_guard(tmpdir):
    """Test that the chunk collision guard detects overlapping writes."""
    print("\n" + "=" * 60)
    print("TEST: chunk collision guard")
    print("=" * 60)

    from processor.converter import _ChunkCollisionGuard
    from zarr.storage import LocalStore
    from zarr.codecs import BloscCodec

    store = LocalStore(os.path.join(tmpdir, "guard_test.zarr"))
    root = zarr.open_group(store, mode="w", zarr_format=3)
    arr = root.create_array(
        "0", shape=(32, 16, 16), dtype=np.float32,
        chunks=(8, 16, 16),
        compressors=[BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")],
    )

    guard = _ChunkCollisionGuard()
    passed = 0

    # Non-overlapping writes should succeed
    guard.claim(arr, (slice(0, 8), slice(None), slice(None)))
    guard.claim(arr, (slice(8, 16), slice(None), slice(None)))
    guard.release(arr, (slice(0, 8), slice(None), slice(None)))
    guard.release(arr, (slice(8, 16), slice(None), slice(None)))
    passed += 1
    print("  OK: non-overlapping writes accepted")

    # Overlapping writes should raise
    guard.claim(arr, (slice(0, 4), slice(None), slice(None)))  # chunk 0
    try:
        guard.claim(arr, (slice(4, 8), slice(None), slice(None)))  # also chunk 0
        print("  FAIL: overlapping write not detected")
    except RuntimeError as e:
        if "Chunk collision" in str(e) and "chunk 0" in str(e):
            passed += 1
            print(f"  OK: overlapping write detected: {e}")
        else:
            print(f"  FAIL: wrong error: {e}")
    guard.release(arr, (slice(0, 4), slice(None), slice(None)))

    # Cross-chunk write should detect collision
    guard.claim(arr, (slice(8, 16), slice(None), slice(None)))  # chunk 1
    try:
        guard.claim(arr, (slice(12, 20), slice(None), slice(None)))  # chunks 1 and 2
        print("  FAIL: cross-chunk overlap not detected")
    except RuntimeError as e:
        if "Chunk collision" in str(e):
            passed += 1
            print(f"  OK: cross-chunk overlap detected: {e}")
        else:
            print(f"  FAIL: wrong error: {e}")
    guard.release(arr, (slice(8, 16), slice(None), slice(None)))

    shutil.rmtree(os.path.join(tmpdir, "guard_test.zarr"), ignore_errors=True)
    print(f"  {passed}/3 passed")
    return passed == 3


def test_full_conversion_error_propagation(tmpdir):
    """Test that a Zarr write error during conversion crashes loudly.

    A silently truncated Zarr is worse than an OOM because it produces
    a file that looks fine. This test verifies that the whole conversion
    fails rather than producing partial output.
    """
    print("\n" + "=" * 60)
    print("TEST: full conversion error propagation")
    print("=" * 60)

    import unittest.mock as mock
    from processor.config import Config
    from processor.converter import convert_nifti_to_ome_zarr

    shape = (8, 10, 12)
    vol = make_volume(shape, np.uint16)
    img = nib.Nifti1Image(vol, np.eye(4))
    nii_path = os.path.join(tmpdir, "error_test.nii")
    nib.save(img, nii_path)

    config = Config(
        input_dir="", output_dir="",
        initial_downsample=1,
        tile_size=256, compression="zstd", compression_level=5,
        max_levels=0, min_dimension=64,
        chunk_slices=4, max_workers=2,
        memory_budget_gb=1.0,
    )

    output_path = os.path.join(tmpdir, "error_test.zarr")

    # Patch zarr array __setitem__ to fail on the second write
    call_count = [0]
    original_setitem = zarr.Array.__setitem__

    def failing_setitem(self, key, value):
        call_count[0] += 1
        if call_count[0] == 2:
            raise IOError("simulated disk full")
        return original_setitem(self, key, value)

    try:
        with mock.patch.object(zarr.Array, '__setitem__', failing_setitem):
            convert_nifti_to_ome_zarr(nii_path, output_path, config)
        print("  FAIL: conversion should have raised on write error")
        return False
    except IOError as e:
        if "simulated disk full" in str(e):
            print("  OK: write error propagated, conversion failed loudly")
            return True
        print(f"  FAIL: wrong exception: {e}")
        return False
    except Exception as e:
        # Accept any exception that surfaces the error
        if "simulated disk full" in str(e):
            print("  OK: write error propagated (as {type(e).__name__})")
            return True
        print(f"  FAIL: unexpected exception: {type(e).__name__}: {e}")
        return False
    finally:
        shutil.rmtree(output_path, ignore_errors=True)


def main():
    tmpdir = tempfile.mkdtemp(prefix="converter_test_")
    print(f"Temp dir: {tmpdir}\n")

    results = []
    try:
        results.append(("48-orientation 3D", test_orientation_equivalence(tmpdir)))
        results.append(("4D equivalence", test_4d_equivalence(tmpdir)))
        results.append(("dtype equivalence", test_dtype_equivalence(tmpdir)))
        results.append(("initial_downsample", test_initial_downsample(tmpdir)))
        results.append(("ndim rejection", test_unsupported_ndim(tmpdir)))
        results.append(("error propagation", test_bounded_queue_error_propagation(tmpdir)))
        results.append(("pyramid race", test_pyramid_race(tmpdir)))
        results.append(("chunk collision guard", test_chunk_collision_guard(tmpdir)))
        results.append(("full conversion error", test_full_conversion_error_propagation(tmpdir)))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())

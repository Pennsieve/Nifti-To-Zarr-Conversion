import gc
import gzip
import math
import logging
import os
import resource
import sys
import threading
import time

import numpy as np
import nibabel as nib
import zarr
from nibabel.orientations import io_orientation, inv_ornt_aff, aff2axcodes
from zarr.storage import LocalStore
from zarr.codecs import BloscCodec
from skimage.transform import downscale_local_mean
from concurrent.futures import ThreadPoolExecutor, as_completed

from processor.config import Config

log = logging.getLogger(__name__)

# Supported on-disk dtypes. Anything else (RGB, complex, etc.) would produce
# garbage when cast to float32 and must be rejected at load time.
_SUPPORTED_DTYPES = {
    np.dtype("uint8"), np.dtype("int8"),
    np.dtype("uint16"), np.dtype("int16"),
    np.dtype("uint32"), np.dtype("int32"),
    np.dtype("float32"), np.dtype("float64"),
}

try:
    import indexed_gzip  # noqa: F401
    _HAS_INDEXED_GZIP = True
except ImportError:
    _HAS_INDEXED_GZIP = False

# Log indexed_gzip status at import time so it appears in the first lines
# of every run, before any file processing.
if _HAS_INDEXED_GZIP:
    log.info("indexed_gzip: available (will use keep_file_open=True for .nii.gz)")
else:
    log.warning(
        "indexed_gzip: NOT INSTALLED. "
        "Scattered slab reads on .nii.gz files will use O(n^2) gzip re-decompression. "
        "Install indexed_gzip for significantly faster processing."
    )


def _rss_mb():
    """Current max RSS in MB."""
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return val / (1024 * 1024)
    return val / 1024


def _scaling_is_nontrivial(slope, inter):
    """True if scl_slope/scl_inter would change raw values."""
    if slope is None or math.isnan(slope) or slope == 0:
        return False
    return slope != 1.0 or inter != 0.0


def _read_raw_scaling(input_path):
    """Read scl_slope/scl_inter from the raw on-disk header.

    nib.load() calls update_header() which recalculates these fields,
    destroying the on-disk values. We read the raw header separately.
    """
    compressed = input_path.lower().endswith(".nii.gz")
    try:
        opener = gzip.open if compressed else open
        with opener(input_path, "rb") as fobj:
            raw_header = nib.Nifti1Header.from_fileobj(fobj)
        slope = float(raw_header["scl_slope"])
        inter = float(raw_header["scl_inter"])
        return slope, inter
    except Exception:
        return float("nan"), 0.0


def _compute_num_levels(shape: tuple, min_dim: int, max_levels: int) -> int:
    """Compute number of pyramid levels based on smallest dimension."""
    smallest = min(shape)
    auto_levels = 1
    while smallest // (2 ** auto_levels) >= min_dim:
        auto_levels += 1
    if max_levels > 0:
        return min(auto_levels, max_levels)
    return auto_levels


def _compute_slab_bytes(chunk_slices, shape, on_disk_dtype, has_scaling):
    """Compute per-slab memory in bytes, accounting for dtype transients.

    nibabel returns float64 when scl_slope/scl_inter are non-trivial,
    otherwise native dtype. We then cast to float32. During the cast,
    both arrays are transiently live.
    """
    voxels = chunk_slices * shape[1] * shape[2]
    if has_scaling:
        # nibabel returns float64 → we cast to float32 → transient peak is both
        bytes_per_voxel = 8 + 4  # float64 + float32
    else:
        src_size = on_disk_dtype.itemsize
        bytes_per_voxel = src_size + 4  # native dtype + float32 cast
    return voxels * bytes_per_voxel


def _compute_max_in_flight(budget_gb, slab_bytes, overhead_reserve_gb=0.0):
    """Derive in-flight bound from a memory budget and per-slab size.

    Subtracts overhead_reserve_gb (indexed_gzip index, Python/zarr overhead,
    compression buffers) before computing how many slabs fit.
    """
    effective_gb = max(budget_gb - overhead_reserve_gb, 0.5)
    budget_bytes = int(effective_gb * (1024 ** 3))
    n = max(1, budget_bytes // slab_bytes)
    return n


class _ChunkCollisionGuard:
    """Detects concurrent writes to the same Zarr chunk along axis 0.

    Always-on. Overhead is a lock + dict lookup per slab write — negligible
    compared to I/O. This converts "we tested and it passed" into "it cannot
    happen silently," which matters because the failure mode (zeroed regions
    in pyramid levels) produces output that looks valid and passes schema checks.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._claimed = {}  # (array_id, chunk_idx) -> write_desc

    def claim(self, arr, slicing):
        """Claim all axis-0 Zarr chunks touched by slicing. Raises on overlap."""
        chunk_d0 = arr.chunks[0]
        s = slicing[0]
        first_chunk = s.start // chunk_d0
        last_chunk = (s.stop - 1) // chunk_d0
        write_desc = f"[{s.start}:{s.stop}]"
        arr_id = id(arr)
        with self._lock:
            for ci in range(first_chunk, last_chunk + 1):
                key = (arr_id, ci)
                if key in self._claimed:
                    raise RuntimeError(
                        f"Chunk collision: axis-0 chunk {ci} "
                        f"(rows {ci * chunk_d0}–{(ci + 1) * chunk_d0 - 1}) "
                        f"already claimed by write {self._claimed[key]}, "
                        f"collides with write {write_desc}"
                    )
                self._claimed[key] = write_desc

    def release(self, arr, slicing):
        """Release axis-0 chunks after write completes."""
        chunk_d0 = arr.chunks[0]
        s = slicing[0]
        first_chunk = s.start // chunk_d0
        last_chunk = (s.stop - 1) // chunk_d0
        arr_id = id(arr)
        with self._lock:
            for ci in range(first_chunk, last_chunk + 1):
                self._claimed.pop((arr_id, ci), None)


def _write_chunk(guard, arr, slicing, data):
    """Write data to a Zarr array at the given slicing, with collision detection."""
    guard.claim(arr, slicing)
    try:
        arr[slicing] = data
    finally:
        guard.release(arr, slicing)


def _submit_bounded(pool, semaphore, fn, *args):
    """Submit a task to the pool, blocking if max_in_flight are pending.

    Releases the semaphore when the future completes (success or failure).
    Returns the future. If the semaphore is acquired but submit fails,
    the semaphore is released to avoid deadlock.
    """
    semaphore.acquire()
    try:
        future = pool.submit(fn, *args)
    except Exception:
        semaphore.release()
        raise
    future.add_done_callback(lambda _: semaphore.release())
    return future


def _drain_futures(futures):
    """Wait for all futures and re-raise the first exception."""
    for f in as_completed(futures):
        f.result()


def _load_image(input_path):
    """Load a NIfTI image, using indexed_gzip for .nii.gz when available."""
    compressed = input_path.lower().endswith(".nii.gz")
    if compressed and _HAS_INDEXED_GZIP:
        img = nib.load(input_path, keep_file_open=True)
        log.info("indexed_gzip: enabled (keep_file_open=True)")
    else:
        img = nib.load(input_path)
        if compressed and not _HAS_INDEXED_GZIP:
            log.warning(
                "indexed_gzip not installed for this file. Gzip slab reads will "
                "re-decompress from the start of the stream for each slab."
            )
        elif not compressed:
            log.info("Uncompressed .nii file — indexed_gzip not applicable")
    return img


def _read_slab_reoriented(dataobj, ornt, src_axis, slab_start, slab_end,
                          transpose_order, output_dtype):
    """Read a slab from the input along src_axis, apply orientation
    transform (flips + transpose), and cast to output_dtype.

    This replicates nibabel's apply_orientation per-slab without
    materializing the full volume.
    """
    slicing = [slice(None)] * len(dataobj.shape)
    slicing[src_axis] = slice(slab_start, slab_end)
    # For 4D: slice all timepoints
    slab = np.asanyarray(dataobj[tuple(slicing)])

    # Apply flips on spatial axes (same as nibabel: before transpose)
    for ax in range(3):
        if ornt[ax, 1] == -1:
            slab = np.flip(slab, axis=ax)

    # Apply transpose on spatial axes; preserve time axis position if 4D
    if slab.ndim == 4:
        spatial_transpose = list(transpose_order) + [3]
        slab = slab.transpose(spatial_transpose)
    else:
        slab = slab.transpose(transpose_order)

    return np.asarray(slab, dtype=output_dtype)


def _read_slab_with_retry(dataobj, ornt, src_axis, slab_start, slab_end,
                           transpose_order, output_dtype, input_path,
                           max_retries=2):
    """Read a slab with retry on indexed_gzip / ZranError failures."""
    last_exc = None
    for attempt in range(1 + max_retries):
        try:
            result = _read_slab_reoriented(
                dataobj, ornt, src_axis, slab_start, slab_end,
                transpose_order, output_dtype,
            )
            if attempt > 0:
                log.info(
                    f"Slab read succeeded on attempt {attempt + 1}/{1 + max_retries}: "
                    f"slab=[{slab_start}:{slab_end}]"
                )
            return result
        except Exception as exc:
            exc_name = type(exc).__name__
            # Retry on indexed_gzip errors (ZranError, etc.)
            if "Zran" in exc_name or "indexed_gzip" in type(exc).__module__:
                last_exc = exc
                if attempt < max_retries:
                    log.warning(
                        f"Slab read failed (attempt {attempt + 1}/{1 + max_retries}): "
                        f"{exc_name}: {exc} — file={input_path}, "
                        f"slab=[{slab_start}:{slab_end}], retrying in 1s"
                    )
                    time.sleep(1)
                    continue
            raise
    raise RuntimeError(
        f"Slab read failed after {1 + max_retries} attempts: "
        f"file={input_path}, slab=[{slab_start}:{slab_end}]"
    ) from last_exc


def convert_nifti_to_ome_zarr(input_path: str, output_path: str, config: Config) -> None:
    phase_start = time.monotonic()

    # --- Phase: Load ---
    t0 = time.monotonic()
    raw_img = _load_image(input_path)
    t_load = time.monotonic() - t0

    # Canary: on-disk properties (before any reorientation)
    on_disk_dtype = raw_img.get_data_dtype()
    on_disk_shape = raw_img.header.get_data_shape()
    ndim = len(on_disk_shape)
    ornt = io_orientation(raw_img.affine)
    axcodes = "".join(aff2axcodes(raw_img.affine))
    is_canonical = np.array_equal(ornt[:3], [[0, 1], [1, 1], [2, 1]])
    compressed = input_path.lower().endswith(".nii.gz")
    scl_slope, scl_inter = _read_raw_scaling(input_path)
    has_scaling = _scaling_is_nontrivial(scl_slope, scl_inter)
    spatial_unit, _ = raw_img.header.get_xyzt_units()
    voxel_sizes = raw_img.header.get_zooms()[:3]

    file_size_mb = os.path.getsize(input_path) / (1024 ** 2)
    total_voxels = 1
    for s in on_disk_shape[:3]:
        total_voxels *= s
    log.info(
        f"Load: shape={on_disk_shape}, ndim={ndim}, dtype={on_disk_dtype}, "
        f"compressed={compressed}, indexed_gzip={_HAS_INDEXED_GZIP}, "
        f"file_size={file_size_mb:.1f}MB, total_voxels={total_voxels:,}, "
        f"load_time={t_load:.2f}s, RSS={_rss_mb():.0f}MB"
    )
    log.info(
        f"Orientation: axcodes={axcodes}, canonical={is_canonical}, "
        f"ornt={ornt[:3].tolist()}"
    )
    log.info(
        f"Header: voxel_sizes={tuple(float(v) for v in voxel_sizes)} {spatial_unit}, "
        f"scl_slope={scl_slope}, scl_inter={scl_inter}, "
        f"nontrivial_scaling={has_scaling}"
    )

    # --- Validation ---
    if on_disk_dtype.base not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype: {on_disk_dtype}. "
            f"Supported dtypes: {sorted(str(d) for d in _SUPPORTED_DTYPES)}. "
            f"RGB, complex, and other compound dtypes are not supported."
        )
    if ndim > 4:
        raise ValueError(
            f"Unsupported dimensionality: ndim={ndim}, shape={on_disk_shape}. "
            f"Only 3D and 4D NIfTI files are supported."
        )
    if ndim < 3:
        raise ValueError(
            f"Unsupported dimensionality: ndim={ndim}, shape={on_disk_shape}. "
            f"Expected at least 3 dimensions."
        )

    is_4d = ndim == 4
    if is_4d:
        log.info(
            f"4D input detected: full_shape={on_disk_shape}, "
            f"timepoints={on_disk_shape[3]}, collapsing to mean over time"
        )

    # --- Phase: Reorientation planning (no data loaded) ---
    spatial_ornt = ornt[:3]
    if is_canonical:
        src_axis = 0
        src_flip = False
        transpose_order = [0, 1, 2]
        reoriented_shape = on_disk_shape[:3]
    else:
        src_axis = int(np.where(spatial_ornt[:, 0] == 0)[0][0])
        src_flip = spatial_ornt[src_axis, 1] == -1
        transpose_order = np.argsort(spatial_ornt[:, 0].astype(int)).tolist()
        inv_map = np.argsort(spatial_ornt[:, 0].astype(int))
        reoriented_shape = tuple(on_disk_shape[inv_map[j]] for j in range(3))

    # Compute canonical affine without loading data
    if is_canonical:
        canonical_affine = raw_img.affine.copy()
    else:
        canonical_affine = raw_img.affine.dot(inv_ornt_aff(spatial_ornt, on_disk_shape[:3]))

    # Recompute voxel sizes from canonical affine
    voxel_sizes = tuple(float(np.sqrt(np.sum(canonical_affine[:3, i] ** 2))) for i in range(3))

    log.info(
        f"Reorientation plan: src_axis={src_axis}, src_flip={src_flip}, "
        f"transpose={transpose_order}, "
        f"input_shape={on_disk_shape[:3]}, output_shape={reoriented_shape}, "
        f"materialized=False, RSS={_rss_mb():.0f}MB"
    )

    shape = reoriented_shape
    log.info(f"NIfTI header voxel sizes (x,y,z): {voxel_sizes} {spatial_unit}")

    output_dtype = np.float32
    log.info(
        f"Dtype: on_disk={on_disk_dtype}, output={output_dtype}, "
        f"upcast={'yes' if on_disk_dtype != output_dtype else 'no'}"
    )

    idf = config.initial_downsample
    if idf > 1:
        shape = tuple(max(1, math.ceil(s / idf)) for s in shape)
        voxel_sizes = tuple(v * idf for v in voxel_sizes)
        log.info(
            f"Initial downsample factor {idf}: effective shape {shape}, "
            f"effective voxel sizes {tuple(float(v) for v in voxel_sizes)} {spatial_unit}"
        )

    n_levels = _compute_num_levels(shape, config.min_dimension, config.max_levels)
    log.info(f"Pyramid levels: {n_levels}, shape: {shape}")

    # Canary: memory budget
    chunk_slices = config.chunk_slices
    slab_bytes = _compute_slab_bytes(
        chunk_slices, reoriented_shape, on_disk_dtype, has_scaling,
    )
    max_in_flight = _compute_max_in_flight(
        config.memory_budget_gb, slab_bytes, config.overhead_reserve_gb,
    )
    effective_gb = max(config.memory_budget_gb - config.overhead_reserve_gb, 0.5)
    theoretical_peak_mb = max_in_flight * slab_bytes / (1024 ** 2)
    log.info(
        f"Memory budget: {config.memory_budget_gb}GB, "
        f"overhead_reserve={config.overhead_reserve_gb}GB, "
        f"effective_budget={effective_gb:.1f}GB, "
        f"slab_bytes={slab_bytes / (1024**2):.1f}MB "
        f"(scaling={'float64+float32' if has_scaling else f'{on_disk_dtype}+float32'}), "
        f"max_in_flight={max_in_flight}, "
        f"theoretical_peak={theoretical_peak_mb:.0f}MB"
    )

    # 2. Open Zarr v3 store
    store = LocalStore(output_path)
    root = zarr.open_group(store, mode="w", zarr_format=3)

    compressor = BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")
    tile = config.tile_size

    # 3. Pre-create arrays for each pyramid level
    level_shapes = [
        tuple(max(1, math.ceil(s / (2 ** i))) for s in shape)
        for i in range(n_levels)
    ]
    level_arrays = []
    for i in range(n_levels):
        ls = level_shapes[i]
        chunks = (min(chunk_slices, ls[0]), min(tile, ls[1]), min(tile, ls[2]))
        arr = root.create_array(
            str(i),
            shape=ls,
            dtype=np.float32,
            chunks=chunks,
            compressors=[compressor],
        )
        level_arrays.append(arr)
        log.info(f"Zarr level {i}: shape={ls}, chunks={chunks}, dtype={arr.dtype}")

    # Chunk collision guard — shared across all write phases.
    guard = _ChunkCollisionGuard()

    # 4. Stream level 0 from NIfTI with per-slab reorientation
    #
    # IMPORTANT: We iterate in OUTPUT chunk order so each Zarr chunk is
    # written by exactly one slab. Iterating in input order with src_flip
    # causes output slabs to straddle Zarr chunk boundaries, and concurrent
    # read-modify-write on the same chunk produces a race condition.
    t0 = time.monotonic()
    src_len = on_disk_shape[src_axis]
    out_dim0 = shape[0]  # output axis-0 size (after any initial_downsample)
    semaphore = threading.Semaphore(max_in_flight)
    total_slabs = math.ceil(out_dim0 / chunk_slices)
    log.info(f"Level 0: streaming {total_slabs} slabs (chunk_slices={chunk_slices})")

    if idf > 1:
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = []
            slab_idx = 0
            last_progress_time = t0
            for out_start in range(0, out_dim0, chunk_slices):
                # Early error detection: fail fast if a write already failed
                for f in futures:
                    if f.done() and f.exception() is not None:
                        log.error(f"Level 0: write failure detected at slab {slab_idx}/{total_slabs}, failing fast")
                        raise f.exception()

                out_end = min(out_start + chunk_slices, out_dim0)
                # Map output chunk back to input range
                in_start = out_start * idf
                in_end = min(in_start + (out_end - out_start) * idf, src_len)
                if src_flip:
                    in_start_f = src_len - in_end
                    in_end_f = src_len - in_start
                    in_start, in_end = in_start_f, in_end_f

                slab = _read_slab_with_retry(
                    raw_img.dataobj, spatial_ornt, src_axis, in_start, in_end,
                    transpose_order, output_dtype, input_path,
                )
                if is_4d:
                    slab = slab.mean(axis=3)
                slab = downscale_local_mean(slab, (idf, idf, idf))

                dst_end = out_start + slab.shape[0]
                slicing = (slice(out_start, dst_end), slice(None), slice(None))
                f = _submit_bounded(pool, semaphore, _write_chunk, guard, level_arrays[0], slicing, slab)
                futures.append(f)
                slab_idx += 1

                now = time.monotonic()
                if now - last_progress_time >= 30:
                    elapsed = now - t0
                    log.info(
                        f"Level 0 progress: slab {slab_idx}/{total_slabs} "
                        f"({100 * slab_idx / total_slabs:.0f}%), "
                        f"elapsed={elapsed:.0f}s, RSS={_rss_mb():.0f}MB"
                    )
                    last_progress_time = now
            _drain_futures(futures)
    else:
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = []
            slab_idx = 0
            last_progress_time = t0
            for out_start in range(0, out_dim0, chunk_slices):
                # Early error detection: fail fast if a write already failed
                for f in futures:
                    if f.done() and f.exception() is not None:
                        log.error(f"Level 0: write failure detected at slab {slab_idx}/{total_slabs}, failing fast")
                        raise f.exception()

                out_end = min(out_start + chunk_slices, out_dim0)
                # Map output chunk back to input range
                if src_flip:
                    in_start = src_len - out_end
                    in_end = src_len - out_start
                else:
                    in_start = out_start
                    in_end = out_end

                slab = _read_slab_with_retry(
                    raw_img.dataobj, spatial_ornt, src_axis, in_start, in_end,
                    transpose_order, output_dtype, input_path,
                )
                if is_4d:
                    slab = slab.mean(axis=3)

                slicing = (slice(out_start, out_end), slice(None), slice(None))
                f = _submit_bounded(pool, semaphore, _write_chunk, guard, level_arrays[0], slicing, slab)
                futures.append(f)
                slab_idx += 1

                now = time.monotonic()
                if now - last_progress_time >= 30:
                    elapsed = now - t0
                    log.info(
                        f"Level 0 progress: slab {slab_idx}/{total_slabs} "
                        f"({100 * slab_idx / total_slabs:.0f}%), "
                        f"elapsed={elapsed:.0f}s, RSS={_rss_mb():.0f}MB"
                    )
                    last_progress_time = now
            _drain_futures(futures)

    t_level0 = time.monotonic() - t0
    log.info(f"Level 0 written: time={t_level0:.1f}s, RSS={_rss_mb():.0f}MB")

    # Release nibabel image to free indexed_gzip seek index + file handle
    rss_before = _rss_mb()
    del raw_img
    gc.collect()
    log.info(f"Released NIfTI image: RSS {rss_before:.0f}MB -> {_rss_mb():.0f}MB")

    # 5. Build lower pyramid levels with quality downsampling
    #
    # IMPORTANT: Iterate in DESTINATION chunk order. Each source slab of
    # chunk_slices rows downsamples to chunk_slices/2 output rows. Two
    # consecutive source slabs would map to the same destination Zarr chunk,
    # creating a read-modify-write race identical to the level-0 flip bug.
    # By reading 2× chunk_slices source rows per destination chunk, each
    # chunk has exactly one writer.
    for lvl in range(1, n_levels):
        t0 = time.monotonic()
        src = level_arrays[lvl - 1]
        dst = level_arrays[lvl]
        dst_dim0 = dst.shape[0]
        dst_chunk_d0 = dst.chunks[0]

        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = []
            for dst_start in range(0, dst_dim0, dst_chunk_d0):
                dst_end = min(dst_start + dst_chunk_d0, dst_dim0)
                # Each destination row comes from 2 source rows
                src_start = dst_start * 2
                src_end = min(dst_end * 2, src.shape[0])
                chunk = np.asarray(src[src_start:src_end, :, :])
                downsampled = downscale_local_mean(chunk, (2, 2, 2))
                actual_dst_end = dst_start + downsampled.shape[0]
                slicing = (slice(dst_start, actual_dst_end), slice(None), slice(None))
                f = _submit_bounded(pool, semaphore, _write_chunk, guard, dst, slicing, downsampled)
                futures.append(f)
            _drain_futures(futures)
        t_lvl = time.monotonic() - t0
        log.info(f"Level {lvl} written: shape={dst.shape}, time={t_lvl:.1f}s, RSS={_rss_mb():.0f}MB")

    # 6. Write OME-Zarr multiscale metadata
    datasets = []
    for i in range(n_levels):
        level_scale = [
            float(voxel_sizes[0]) * (2 ** i),
            float(voxel_sizes[1]) * (2 ** i),
            float(voxel_sizes[2]) * (2 ** i),
        ]
        log.info(f"Level {i} output voxel sizes (x,y,z): {tuple(level_scale)} {spatial_unit}")
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": level_scale,
                }
            ],
        })

    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": output_path,
        "axes": [
            {"name": "x", "type": "space", "unit": "millimeter"},
            {"name": "y", "type": "space", "unit": "millimeter"},
            {"name": "z", "type": "space", "unit": "millimeter"},
        ],
        "datasets": datasets,
        "type": "downscale_local_mean",
    }]

    total_time = time.monotonic() - phase_start
    peak_rss = _rss_mb()
    budget_mb = config.memory_budget_gb * 1024
    rss_pct = 100 * peak_rss / budget_mb if budget_mb > 0 else 0
    log.info(
        f"Done: {n_levels} levels, total_time={total_time:.1f}s, "
        f"peak_RSS={peak_rss:.0f}MB, budget={budget_mb:.0f}MB ({rss_pct:.0f}% used)"
    )
    if peak_rss > budget_mb:
        log.warning(
            f"Peak RSS ({peak_rss:.0f}MB) exceeded memory budget ({budget_mb:.0f}MB). "
            f"Consider increasing MEMORY_BUDGET_GB or OVERHEAD_RESERVE_GB."
        )

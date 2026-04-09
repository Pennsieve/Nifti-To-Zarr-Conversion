import math
import logging

import numpy as np
import nibabel as nib
import zarr
from zarr.storage import LocalStore
from zarr.codecs import BloscCodec
from skimage.transform import downscale_local_mean
from concurrent.futures import ThreadPoolExecutor, as_completed

from processor.config import Config

log = logging.getLogger(__name__)


def _compute_num_levels(shape: tuple, min_dim: int, max_levels: int) -> int:
    """Compute number of pyramid levels based on smallest dimension."""
    smallest = min(shape)
    auto_levels = 1
    while smallest // (2 ** auto_levels) >= min_dim:
        auto_levels += 1
    # auto_levels is at least 1 (full resolution)
    if max_levels > 0:
        return min(auto_levels, max_levels)
    return auto_levels


def _write_chunk(arr, start, end, data):
    arr[start:end, :, :] = data


def convert_nifti_to_ome_zarr(input_path: str, output_path: str, config: Config) -> None:
    # 1. Load NIfTI and reorient to canonical (RAS+)
    img = nib.as_closest_canonical(nib.load(input_path))
    voxel_sizes = img.header.get_zooms()[:3]
    shape = img.shape[:3]

    idf = config.initial_downsample
    if idf > 1:
        shape = tuple(max(1, math.ceil(s / idf)) for s in shape)
        voxel_sizes = tuple(v * idf for v in voxel_sizes)
        log.info(f"Initial downsample factor {idf}: effective shape {shape}")

    n_levels = _compute_num_levels(shape, config.min_dimension, config.max_levels)
    log.info(f"Pyramid levels: {n_levels}, shape: {shape}")

    # 2. Open Zarr v3 store
    store = LocalStore(output_path)
    root = zarr.open_group(store, mode="w", zarr_format=3)

    compressor = BloscCodec(cname=config.compression, clevel=config.compression_level)
    tile = config.tile_size
    chunk_slices = config.chunk_slices

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

    # 4. Stream level 0 from NIfTI
    src_dim0 = img.shape[0]
    if idf > 1:
        # Read larger slabs and downsample
        slices = [
            (start, min(start + chunk_slices * idf, src_dim0))
            for start in range(0, src_dim0, chunk_slices * idf)
        ]
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = []
            dst_offset = 0
            for start, end in slices:
                chunk = np.asarray(img.dataobj[start:end, :, :], dtype=np.float32)
                chunk = downscale_local_mean(chunk, (idf, idf, idf))
                dst_end = dst_offset + chunk.shape[0]
                futures.append(pool.submit(_write_chunk, level_arrays[0], dst_offset, dst_end, chunk))
                dst_offset = dst_end
            for f in as_completed(futures):
                f.result()
    else:
        slices = [
            (start, min(start + chunk_slices, src_dim0))
            for start in range(0, src_dim0, chunk_slices)
        ]
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = []
            for start, end in slices:
                chunk = np.asarray(img.dataobj[start:end, :, :], dtype=np.float32)
                futures.append(pool.submit(_write_chunk, level_arrays[0], start, end, chunk))
            for f in as_completed(futures):
                f.result()

    log.info("Level 0 written")

    # 5. Build lower pyramid levels with quality downsampling
    for lvl in range(1, n_levels):
        src = level_arrays[lvl - 1]
        dst = level_arrays[lvl]
        src_dim0 = src.shape[0]
        src_slices = [
            (start, min(start + chunk_slices, src_dim0))
            for start in range(0, src_dim0, chunk_slices)
        ]
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = []
            for start, end in src_slices:
                chunk = np.asarray(src[start:end, :, :])
                downsampled = downscale_local_mean(chunk, (2, 2, 2))
                dst_start = start // 2
                dst_end = dst_start + downsampled.shape[0]
                futures.append(pool.submit(_write_chunk, dst, dst_start, dst_end, downsampled))
            for f in as_completed(futures):
                f.result()
        log.info(f"Level {lvl} written: {dst.shape}")

    # 6. Write OME-Zarr multiscale metadata
    datasets = []
    for i in range(n_levels):
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [
                        float(voxel_sizes[0]) * (2 ** i),
                        float(voxel_sizes[1]) * (2 ** i),
                        float(voxel_sizes[2]) * (2 ** i),
                    ],
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

    log.info(f"OME-Zarr metadata written ({n_levels} levels)")

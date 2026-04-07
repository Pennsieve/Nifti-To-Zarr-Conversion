import math
import numpy as np
import nibabel as nib
import zarr
from ome_zarr.writer import write_multiscales_metadata
from concurrent.futures import ThreadPoolExecutor, as_completed

CHUNK_SLICES = 32
MAX_WORKERS  = 8  # concurrent chunk writes

def _write_chunk(arr, start, end, data):
    arr[start:end, :, :] = data

def convert_nifti_to_ome_zarr(input_path: str, output_path: str) -> None:
    # 1. Load NIfTI and reorient to canonical (RAS) to ensure correct axis labels
    img = nib.as_closest_canonical(nib.load(input_path))
    voxel_sizes = img.header.get_zooms()  # (dim0, dim1, dim2) spacings — in RAS+ canonical form: dim0=x, dim1=y, dim2=z
    shape = img.shape[:3]  # (dim0, dim1, dim2) — drop time dim if present

    store = zarr.open(output_path, mode="w")
    root = store

    # 2. Pre-create Zarr arrays for each pyramid level (no data yet)
    n_levels = 3
    level_shapes = [
        tuple(max(1, math.ceil(s / (2 ** i))) for s in shape)
        for i in range(n_levels)
    ]
    level_arrays = [
        root.require_dataset(
            str(i),
            shape=level_shapes[i],
            dtype=np.float32,
            chunks=(min(CHUNK_SLICES, level_shapes[i][0]), min(64, level_shapes[i][1]), min(64, level_shapes[i][2])),
            overwrite=True,
        )
        for i in range(n_levels)
    ]

    # 3. Stream level 0 slice-by-slice, writing chunks in parallel
    dim0 = shape[0]
    slices = [
        (start, min(start + CHUNK_SLICES, dim0))
        for start in range(0, dim0, CHUNK_SLICES)
    ]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = []
        for start, end in slices:
            chunk = np.asarray(img.dataobj[start:end, :, :], dtype=np.float32)
            futures.append(pool.submit(_write_chunk, level_arrays[0], start, end, chunk))
        for f in as_completed(futures):
            f.result()  # re-raises any exception from worker

    # 4. Build lower pyramid levels from level 0, also in parallel
    for lvl in range(1, n_levels):
        src = level_arrays[lvl - 1]
        dst = level_arrays[lvl]
        src_dim0 = src.shape[0]
        src_slices = [
            (start, min(start + CHUNK_SLICES, src_dim0))
            for start in range(0, src_dim0, CHUNK_SLICES)
        ]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = []
            for start, end in src_slices:
                chunk = src[start:end:2, ::2, ::2]  # 2x downsample
                dst_start = start // 2
                dst_end = dst_start + chunk.shape[0]
                futures.append(pool.submit(_write_chunk, dst, dst_start, dst_end, chunk))
            for f in as_completed(futures):
                f.result()

    # 5. Write OME-Zarr multiscale metadata (no data, just .zattrs)
    datasets = []
    for i in range(n_levels):
        factor = 2 ** i
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [
                {"type": "scale", "scale": [float(voxel_sizes[0]) * factor, float(voxel_sizes[1]) * factor, float(voxel_sizes[2]) * factor]}
            ],
        })
    write_multiscales_metadata(
        group=root,
        datasets=datasets,
        axes=[
            {"name": "x", "type": "space", "unit": "millimeter"},
            {"name": "y", "type": "space", "unit": "millimeter"},
            {"name": "z", "type": "space", "unit": "millimeter"},
        ],
    )
import numpy as np
import nibabel as nib
import zarr
from ome_zarr.writer import write_multiscales_metadata

# Number of slices to load into RAM at once when writing level 0.
# Lower = less memory, more I/O. 32 is a safe default.
CHUNK_SLICES = 32

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
        tuple(max(1, s // (2 ** i)) for s in shape)
        for i in range(n_levels)
    ]
    level_arrays = [
        root.require_dataset(
            str(i),
            shape=level_shapes[i],
            dtype=np.float32,
            chunks=(min(64, level_shapes[i][0]), min(64, level_shapes[i][1]), min(64, level_shapes[i][2])),
            overwrite=True,
        )
        for i in range(n_levels)
    ]

    # 3. Stream level 0 slice-by-slice to avoid loading full volume into RAM
    dim0 = shape[0]
    for start in range(0, dim0, CHUNK_SLICES):
        end = min(start + CHUNK_SLICES, dim0)
        chunk = np.asarray(img.dataobj[start:end, :, :], dtype=np.float32)
        level_arrays[0][start:end, :, :] = chunk

    # 4. Build lower pyramid levels from level 0 (already on disk, read back in chunks)
    for lvl in range(1, n_levels):
        src = level_arrays[lvl - 1]
        dst = level_arrays[lvl]
        src_dim0 = src.shape[0]
        for start in range(0, src_dim0, CHUNK_SLICES):
            end = min(start + CHUNK_SLICES, src_dim0)
            chunk = src[start:end:2, ::2, ::2]  # 2x downsample
            dst_start = start // 2
            dst_end = dst_start + chunk.shape[0]
            dst[dst_start:dst_end, :, :] = chunk

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
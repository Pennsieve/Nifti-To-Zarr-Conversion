import numpy as np
import nibabel as nib
import zarr
from ome_zarr.writer import write_multiscale

def convert_nifti_to_ome_zarr(input_path: str, output_path: str) -> None:
    # 1. Load NIfTI and reorient to canonical (RAS) to ensure correct axis labels
    img = nib.as_closest_canonical(nib.load(input_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    voxel_sizes = img.header.get_zooms()  # (dim0, dim1, dim2) spacings — in RAS+ canonical form: dim0=x, dim1=y, dim2=z

    # 2. Build multiscale pyramid (3 levels: full, half, quarter res)
    levels = _build_pyramid(data, n_levels=3)

    # 3. Write OME-Zarr v2
    store = zarr.open(output_path, mode="w")
    write_multiscale(
        pyramid=levels,
        group=store,
        axes=[
            {"name": "z", "type": "space", "unit": "millimeter"},
            {"name": "y", "type": "space", "unit": "millimeter"},
            {"name": "x", "type": "space", "unit": "millimeter"},
        ],
        coordinate_transformations=[
            [{"type": "scale", "scale": [float(voxel_sizes[0]), float(voxel_sizes[1]), float(voxel_sizes[2])]}],
            [{"type": "scale", "scale": [float(voxel_sizes[0])*2, float(voxel_sizes[1])*2, float(voxel_sizes[2])*2]}],
            [{"type": "scale", "scale": [float(voxel_sizes[0])*4, float(voxel_sizes[1])*4, float(voxel_sizes[2])*4]}],
        ],
    )

def _build_pyramid(data: np.ndarray, n_levels: int) -> list[np.ndarray]:
    """Downsample by 2x in each spatial dimension per level."""
    levels = [data]
    for _ in range(n_levels - 1):
        prev = levels[-1]
        downsampled = prev[::2, ::2, ::2]
        levels.append(downsampled)
    return levels
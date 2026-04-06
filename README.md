# NIfTI to OME-Zarr

Converts NIfTI (`.nii`, `.nii.gz`) files to [OME-Zarr](https://ngff.openmicroscopy.org/) with a 3-level multiscale pyramid. Designed to run as a Pennsieve workflow processor.

## How it works

1. Loads each NIfTI file and reorients to canonical RAS+
2. Streams voxel data to Zarr in chunks to keep memory usage low
3. Builds a 3-level pyramid (1x, 2x, 4x downsampling)
4. Writes OME-Zarr v0.4 multiscale metadata with voxel spacing preserved

## Usage

### Docker

```bash
docker build -t nifti-to-zarr .
docker run \
  -v /path/to/nifti/files:/inputs \
  -v /path/to/output:/outputs \
  nifti-to-zarr
```

### Local

```bash
pip install -r processor/requirements.txt
INPUT_DIR=./test_inputs OUTPUT_DIR=./test_outputs python -m processor.main
```

## Configuration

| Environment Variable | Default    | Description                  |
|---------------------|------------|------------------------------|
| `INPUT_DIR`         | `/inputs`  | Directory containing NIfTI files |
| `OUTPUT_DIR`        | `/outputs` | Directory for Zarr output    |

## Output

For each input file (e.g. `brain.nii.gz`), produces `brain.zarr/` containing:

```
brain.zarr/
├── .zattrs          # OME-Zarr multiscale metadata
├── 0/               # Full resolution
├── 1/               # 2x downsampled
└── 2/               # 4x downsampled
```

## Dependencies

- Python 3.12
- nibabel
- numpy
- zarr 2.x
- ome-zarr

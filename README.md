# NIfTI to OME-Zarr

Converts NIfTI (`.nii`, `.nii.gz`) files to [OME-Zarr](https://ngff.openmicroscopy.org/) (Zarr v3) with dynamic multiscale pyramids. Designed to run as a Pennsieve workflow processor in both Lambda and ECS modes.

## How it works

1. Loads each NIfTI file and reorients to canonical RAS+
2. Streams voxel data to Zarr v3 in chunks to keep memory usage low
3. Builds a dynamic pyramid (auto-computed from data dimensions)
4. Uses quality downsampling via `downscale_local_mean`
5. Writes OME-Zarr v0.4 multiscale metadata with voxel spacing preserved

## Usage

### Docker Compose (recommended for local dev)

```bash
# Place NIfTI files in ./test_inputs/
make run
```

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
make local
```

### Make targets

| Target  | Description                          |
|---------|--------------------------------------|
| `build` | Build Docker image                   |
| `run`   | Build and run with docker compose    |
| `clean` | Remove containers, images, outputs   |
| `local` | Run locally without Docker           |

## Configuration

| Environment Variable   | Default    | Description                                      |
|-----------------------|------------|--------------------------------------------------|
| `INPUT_DIR`           | `/inputs`  | Directory containing NIfTI files                 |
| `OUTPUT_DIR`          | `/outputs` | Directory for Zarr output                        |
| `INITIAL_DOWNSAMPLE`  | `1`        | Initial downsample factor (1 = full resolution)  |
| `TILE_SIZE`           | `64`       | Y, Z chunk size                                  |
| `COMPRESSION`         | `zstd`     | Blosc compression algorithm                      |
| `COMPRESSION_LEVEL`   | `5`        | Blosc compression level                          |
| `MAX_LEVELS`          | `0`        | Max pyramid levels (0 = auto)                    |
| `MIN_DIMENSION`       | `64`       | Stop pyramid when smallest dim <= this           |
| `CHUNK_SLICES`        | `32`       | Number of slices per streaming batch             |
| `MAX_WORKERS`         | `8`        | Thread pool size for parallel chunk writes       |

Defaults can be overridden via `dev.env` or `dev.env.local` (gitignored).

## Output

For each input file (e.g. `brain.nii.gz`), produces `brain.zarr/` (Zarr v3 directory store):

```
brain.zarr/
├── zarr.json         # Zarr v3 group metadata + OME multiscale attrs
├── 0/                # Full resolution (or initial downsample)
├── 1/                # 2x downsampled
├── 2/                # 4x downsampled
└── ...               # Additional levels as needed
```

Pyramid levels are computed automatically based on data dimensions and `MIN_DIMENSION`.

## Lambda Support

The processor supports AWS Lambda via `awslambdaric`. The `entrypoint.sh` detects `$AWS_LAMBDA_RUNTIME_API` and routes to the Lambda handler, which accepts camelCase event keys (e.g. `tileSize`, `compressionLevel`).

## Dependencies

- Python 3.12
- nibabel
- numpy
- zarr 3.x
- scikit-image
- awslambdaric

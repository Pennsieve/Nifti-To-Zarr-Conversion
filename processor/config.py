import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    input_dir: str
    output_dir: str
    initial_downsample: int
    tile_size: int
    compression: str
    compression_level: int
    max_levels: int
    min_dimension: int
    chunk_slices: int
    max_workers: int


def get_config() -> Config:
    return Config(
        input_dir=os.environ.get("INPUT_DIR", "/inputs"),
        output_dir=os.environ.get("OUTPUT_DIR", "/outputs"),
        initial_downsample=int(os.environ.get("INITIAL_DOWNSAMPLE", "1")),
        tile_size=int(os.environ.get("TILE_SIZE", "256")),
        compression=os.environ.get("COMPRESSION", "zstd"),
        compression_level=int(os.environ.get("COMPRESSION_LEVEL", "5")),
        max_levels=int(os.environ.get("MAX_LEVELS", "0")),
        min_dimension=int(os.environ.get("MIN_DIMENSION", "64")),
        chunk_slices=int(os.environ.get("CHUNK_SLICES", "32")),
        max_workers=int(os.environ.get("MAX_WORKERS", "8")),
    )

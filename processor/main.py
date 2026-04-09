import os
import logging

from processor.config import get_config
from processor.converter import convert_nifti_to_ome_zarr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

VERSION = "2.0.2"


def run():
    config = get_config()

    log.info(f"nifti-to-zarr processor version {VERSION}")
    log.info(f"Config: {config}")

    input_files = [
        f.path for f in os.scandir(config.input_dir)
        if f.is_file() and os.path.splitext(f.name)[1].lower() in ('.nii', '.gz')
    ]

    if not input_files:
        raise FileNotFoundError(f"No NIfTI files found in {config.input_dir}")

    for input_path in input_files:
        log.info(f"Converting: {input_path}")
        base = os.path.basename(input_path).replace('.nii.gz', '').replace('.nii', '')
        output_path = os.path.join(config.output_dir, f"{base}.zarr")
        convert_nifti_to_ome_zarr(input_path, output_path, config)
        log.info(f"Written: {output_path}")


if __name__ == "__main__":
    run()

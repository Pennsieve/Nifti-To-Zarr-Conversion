import os
import logging
from config import Config
from converter import convert_nifti_to_ome_zarr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

VERSION = "1.2.0"  # bump this on every push
log.info(f"nifti-to-zarr processor version {VERSION}")

if __name__ == "__main__":
    config = Config()

    input_files = [
        f.path for f in os.scandir(config.INPUT_DIR)
        if f.is_file() and os.path.splitext(f.name)[1].lower() in ('.nii', '.gz')
    ]

    if not input_files:
        raise FileNotFoundError(f"No NIfTI files found in {config.INPUT_DIR}")

    for input_path in input_files:
        log.info(f"Converting: {input_path}")
        base = os.path.basename(input_path).replace('.nii.gz', '').replace('.nii', '')
        output_path = os.path.join(config.OUTPUT_DIR, f"{base}.zarr")
        convert_nifti_to_ome_zarr(input_path, output_path)
        log.info(f"Written: {output_path}")
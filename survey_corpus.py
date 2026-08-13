#!/usr/bin/env python3
"""
NIfTI corpus survey — reads headers only, no voxel data.

Usage:
    python survey_corpus.py /path/to/nifti/root [--csv output.csv] [--recursive]

Walks the given directory for .nii and .nii.gz files, reads the 352-byte
header from each, and produces a CSV plus a summary report on stdout.

For .nii.gz files, only the first gzip block is decompressed (enough for
the header). No voxel data is read or decompressed.
"""

import argparse
import csv
import gzip
import logging
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes, io_orientation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# dtypes we consider safe for processing
SUPPORTED_DTYPES = {
    np.dtype("uint8"), np.dtype("int8"),
    np.dtype("uint16"), np.dtype("int16"),
    np.dtype("uint32"), np.dtype("int32"),
    np.dtype("float32"), np.dtype("float64"),
}


@dataclass
class FileRecord:
    path: str
    filename: str
    compressed: bool  # .nii.gz vs .nii
    file_size_bytes: int
    ndim: int
    shape: str  # tuple as string
    dtype: str  # includes endianness
    dtype_base: str  # without endianness
    byteorder: str  # '<', '>', '=', '|'
    is_big_endian: bool
    is_supported_dtype: bool
    voxels: int
    uncompressed_data_bytes: int  # voxels × dtype itemsize
    compression_ratio: Optional[float]  # uncompressed / file_size, None for .nii
    is_canonical: bool  # RAS+ orientation
    axcodes: str  # e.g. "RAS", "LPI"
    orientation_matrix: str  # io_orientation result as string
    scl_slope: float
    scl_inter: float
    has_nontrivial_scaling: bool
    pixdim_x: float
    pixdim_y: float
    pixdim_z: float
    xyzt_units: int
    spatial_unit: str
    error: str  # empty if no error


def is_nifti(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def scaling_is_nontrivial(slope: float, inter: float) -> bool:
    """True if scl_slope/scl_inter would change the raw values."""
    if math.isnan(slope) or slope == 0:
        # NIfTI spec: slope=0 or NaN means "not set" → raw values used as-is
        return False
    return slope != 1.0 or inter != 0.0


def survey_file(filepath: str) -> FileRecord:
    """Read header only, return a FileRecord."""
    filename = os.path.basename(filepath)
    compressed = filepath.lower().endswith(".nii.gz")
    file_size = os.path.getsize(filepath)

    # Defaults for error path
    error_record = FileRecord(
        path=filepath, filename=filename, compressed=compressed,
        file_size_bytes=file_size, ndim=0, shape="", dtype="", dtype_base="",
        byteorder="", is_big_endian=False, is_supported_dtype=False,
        voxels=0, uncompressed_data_bytes=0, compression_ratio=None,
        is_canonical=False, axcodes="", orientation_matrix="",
        scl_slope=0, scl_inter=0, has_nontrivial_scaling=False,
        pixdim_x=0, pixdim_y=0, pixdim_z=0,
        xyzt_units=0, spatial_unit="", error="",
    )

    try:
        img = nib.load(filepath)  # header only, no voxel data
    except Exception as e:
        error_record.error = f"load failed: {e}"
        return error_record

    # Read raw header separately — nib.load() calls update_header() which
    # recalculates scl_slope/scl_inter, destroying the on-disk values.
    try:
        opener = gzip.open if compressed else open
        with opener(filepath, "rb") as fobj:
            raw_header = nib.Nifti1Header.from_fileobj(fobj)
    except Exception:
        raw_header = None

    try:
        header = img.header
        data_shape = header.get_data_shape()
        ndim = len(data_shape)
        data_dtype = img.get_data_dtype()
        dtype_str = str(data_dtype)
        dtype_base = data_dtype.base.str.lstrip("<>=|")
        byteorder = data_dtype.byteorder
        is_big = byteorder == ">"

        voxels = int(np.prod(data_shape)) if ndim > 0 else 0
        uncompressed_bytes = voxels * data_dtype.itemsize

        compression_ratio = None
        if compressed and file_size > 0:
            compression_ratio = round(uncompressed_bytes / file_size, 2)

        # Orientation
        affine = img.affine
        ornt = io_orientation(affine)
        axcodes_tuple = aff2axcodes(affine)
        axcodes_str = "".join(axcodes_tuple)
        # Canonical = RAS+ = identity orientation
        is_canonical = np.array_equal(
            ornt[:3],
            [[0, 1], [1, 1], [2, 1]],
        )

        # Scaling — use raw_header to get on-disk values, since nib.load()
        # recalculates slope/inter and may clear them.
        if raw_header is not None:
            scl_slope = float(raw_header["scl_slope"])
            scl_inter = float(raw_header["scl_inter"])
        else:
            scl_slope = float("nan")
            scl_inter = 0.0
        has_scaling = scaling_is_nontrivial(scl_slope, scl_inter)

        # Spatial info
        pixdim = header.get_zooms()
        pixdim_x = float(pixdim[0]) if len(pixdim) > 0 else 0.0
        pixdim_y = float(pixdim[1]) if len(pixdim) > 1 else 0.0
        pixdim_z = float(pixdim[2]) if len(pixdim) > 2 else 0.0

        xyzt_raw = int(header.get("xyzt_units", 0))
        spatial_unit_str, _ = header.get_xyzt_units()

        is_supported = data_dtype.base in SUPPORTED_DTYPES

        return FileRecord(
            path=filepath,
            filename=filename,
            compressed=compressed,
            file_size_bytes=file_size,
            ndim=ndim,
            shape=str(data_shape),
            dtype=dtype_str,
            dtype_base=dtype_base,
            byteorder=byteorder,
            is_big_endian=is_big,
            is_supported_dtype=is_supported,
            voxels=voxels,
            uncompressed_data_bytes=uncompressed_bytes,
            compression_ratio=compression_ratio,
            is_canonical=is_canonical,
            axcodes=axcodes_str,
            orientation_matrix=str(ornt.tolist()),
            scl_slope=scl_slope,
            scl_inter=scl_inter,
            has_nontrivial_scaling=has_scaling,
            pixdim_x=pixdim_x,
            pixdim_y=pixdim_y,
            pixdim_z=pixdim_z,
            xyzt_units=xyzt_raw,
            spatial_unit=spatial_unit_str,
            error="",
        )

    except Exception as e:
        error_record.error = f"header parse failed: {e}"
        return error_record


def find_nifti_files(root: str, recursive: bool) -> list[str]:
    """Find all .nii and .nii.gz files under root."""
    files = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                full = os.path.join(dirpath, f)
                if is_nifti(full):
                    files.append(full)
    else:
        for entry in os.scandir(root):
            if entry.is_file() and is_nifti(entry.path):
                files.append(entry.path)
    return sorted(files)


def write_csv(records: list[FileRecord], csv_path: str) -> None:
    field_names = [f.name for f in fields(FileRecord)]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))
    log.info(f"CSV written: {csv_path} ({len(records)} rows)")


def fmt_bytes(n: float) -> str:
    if n >= 1e12:
        return f"{n/1e12:.1f} TB"
    if n >= 1e9:
        return f"{n/1e9:.1f} GB"
    if n >= 1e6:
        return f"{n/1e6:.1f} MB"
    if n >= 1e3:
        return f"{n/1e3:.1f} KB"
    return f"{n:.0f} B"


def print_summary(records: list[FileRecord]) -> None:
    total = len(records)
    errors = [r for r in records if r.error]
    ok = [r for r in records if not r.error]

    print("\n" + "=" * 70)
    print(f"CORPUS SURVEY SUMMARY — {total} files, {len(errors)} errors")
    print("=" * 70)

    if not ok:
        print("No valid files to summarize.")
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for r in errors:
                print(f"  {r.filename}: {r.error}")
        return

    # 1. Canonical vs non-canonical
    canonical = sum(1 for r in ok if r.is_canonical)
    non_canonical = sum(1 for r in ok if not r.is_canonical)
    pct_non_canon = 100 * non_canonical / len(ok) if ok else 0
    print(f"\n1. ORIENTATION")
    print(f"   Canonical (RAS+): {canonical} ({100-pct_non_canon:.1f}%)")
    print(f"   Non-canonical:    {non_canonical} ({pct_non_canon:.1f}%)")
    if non_canonical:
        axcode_counts = Counter(r.axcodes for r in ok if not r.is_canonical)
        for code, count in axcode_counts.most_common(10):
            print(f"     {code}: {count}")

    # 2. Dtype distribution
    print(f"\n2. DTYPE DISTRIBUTION")
    dtype_counts = Counter(r.dtype for r in ok)
    for dtype, count in dtype_counts.most_common():
        pct = 100 * count / len(ok)
        print(f"   {dtype:>12s}: {count:>6d} ({pct:.1f}%)")
    big_endian = sum(1 for r in ok if r.is_big_endian)
    unsupported = sum(1 for r in ok if not r.is_supported_dtype)
    if big_endian:
        print(f"   Big-endian files: {big_endian}")
    if unsupported:
        print(f"   Unsupported dtypes: {unsupported}")
        for r in ok:
            if not r.is_supported_dtype:
                print(f"     {r.filename}: {r.dtype}")

    # 3. Uncompressed size distribution
    sizes = sorted(r.uncompressed_data_bytes for r in ok)
    print(f"\n3. UNCOMPRESSED DATA SIZE")
    print(f"   Min:    {fmt_bytes(sizes[0])}")
    print(f"   Median: {fmt_bytes(sizes[len(sizes)//2])}")
    p95_idx = int(0.95 * len(sizes))
    print(f"   P95:    {fmt_bytes(sizes[min(p95_idx, len(sizes)-1)])}")
    print(f"   Max:    {fmt_bytes(sizes[-1])}")
    print(f"   Total:  {fmt_bytes(sum(sizes))}")

    # Size buckets
    buckets = [
        ("< 100 MB", 0, 100e6),
        ("100 MB – 1 GB", 100e6, 1e9),
        ("1 – 10 GB", 1e9, 10e9),
        ("10 – 50 GB", 10e9, 50e9),
        ("50 – 100 GB", 50e9, 100e9),
        ("> 100 GB", 100e9, float("inf")),
    ]
    print("   Size buckets:")
    for label, lo, hi in buckets:
        count = sum(1 for s in sizes if lo <= s < hi)
        if count:
            print(f"     {label:>15s}: {count}")

    # 4. Scaling
    nontrivial_scaling = sum(1 for r in ok if r.has_nontrivial_scaling)
    print(f"\n4. SCL_SLOPE / SCL_INTER")
    print(f"   Trivial (slope=1/0/NaN, inter=0): {len(ok) - nontrivial_scaling}")
    print(f"   Non-trivial:                      {nontrivial_scaling}")
    if nontrivial_scaling:
        for r in ok:
            if r.has_nontrivial_scaling:
                print(f"     {r.filename}: slope={r.scl_slope}, inter={r.scl_inter}")

    # 5. Dimensionality
    print(f"\n5. DIMENSIONALITY")
    ndim_counts = Counter(r.ndim for r in ok)
    for nd, count in sorted(ndim_counts.items()):
        print(f"   {nd}D: {count}")
    non_3d = [r for r in ok if r.ndim != 3]
    if non_3d:
        for r in non_3d:
            print(f"     {r.filename}: {r.ndim}D, shape={r.shape}")

    # 6. Compression
    compressed = [r for r in ok if r.compressed]
    uncompressed = [r for r in ok if not r.compressed]
    print(f"\n6. COMPRESSION")
    print(f"   .nii.gz: {len(compressed)}")
    print(f"   .nii:    {len(uncompressed)}")
    if compressed:
        ratios = [r.compression_ratio for r in compressed if r.compression_ratio]
        if ratios:
            ratios.sort()
            print(f"   Compression ratio (uncompressed/compressed):")
            print(f"     Min:    {ratios[0]:.1f}x")
            print(f"     Median: {ratios[len(ratios)//2]:.1f}x")
            print(f"     Max:    {ratios[-1]:.1f}x")

    # 7. Spatial units
    print(f"\n7. SPATIAL UNITS")
    unit_counts = Counter(r.spatial_unit for r in ok)
    for unit, count in unit_counts.most_common():
        label = unit if unit else "(unset/unknown)"
        print(f"   {label:>20s}: {count}")

    # 8. Errors
    if errors:
        print(f"\n8. ERRORS ({len(errors)})")
        for r in errors:
            print(f"   {r.filename}: {r.error}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Survey NIfTI corpus — headers only, no voxel data."
    )
    parser.add_argument("root", help="Root directory to scan for NIfTI files")
    parser.add_argument("--csv", default="corpus_survey.csv",
                        help="Output CSV path (default: corpus_survey.csv)")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Recurse into subdirectories")
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        log.error(f"Not a directory: {root}")
        sys.exit(1)

    log.info(f"Scanning {root} (recursive={args.recursive})")
    t0 = time.monotonic()
    nifti_files = find_nifti_files(root, args.recursive)
    t1 = time.monotonic()
    log.info(f"Found {len(nifti_files)} NIfTI files in {t1-t0:.1f}s")

    if not nifti_files:
        log.warning("No NIfTI files found.")
        sys.exit(0)

    records = []
    for i, fpath in enumerate(nifti_files):
        if (i + 1) % 100 == 0 or i == 0:
            log.info(f"Processing {i+1}/{len(nifti_files)}: {os.path.basename(fpath)}")
        rec = survey_file(fpath)
        if rec.error:
            log.warning(f"Error on {rec.filename}: {rec.error}")
        records.append(rec)

    write_csv(records, args.csv)
    print_summary(records)


if __name__ == "__main__":
    main()

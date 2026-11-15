#!/usr/bin/env python3
"""
Fix the axis units (and, optionally, missing voxel sizes) in OME-Zarr assets
made by the NIfTI converter.

1. Units (always): before the nifti-units fix every asset was labeled
   "millimeter". The scale values are the header's voxel sizes in the header's
   own unit, so when the header says micron or meter only the unit label is
   wrong. This rewrites the label; the numbers and the image data stay.

2. Missing voxel size (opt-in, --voxel-from-filename): some headers carry no
   voxel size at all (pixdim 1, unit unknown), so the asset says 1 mm voxels.
   When the file name states the size (e.g. "B796_C3L_9um_0000.nii.gz"), this
   sets the scale from it: 9 um -> 0.009 mm per voxel at level 0, doubling per
   level.

Only the root zarr.json is rewritten. Reads go through the Pennsieve API, so a
dry run needs only a session token; --apply writes with boto3 and needs S3
write access.

Usage:
  export PENNSIEVE_SESSION_TOKEN=...
  python patch_zarr_units.py packages.txt [--voxel-from-filename]
  python patch_zarr_units.py packages.txt [--voxel-from-filename] --apply --main-bucket <bucket>

packages.txt: "N:package:... N:dataset:..." per line, or a run list with a
"# DATASET_ID=N:dataset:..." header and one package ID per line.
"""

import argparse
import datetime
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.request
import zlib

API = os.environ.get("PENNSIEVE_API_HOST", "https://api.pennsieve.io")
API2 = os.environ.get("PENNSIEVE_API2_HOST", "https://api2.pennsieve.io")
PREFIX_BUCKETS = {"b11f440c": "prd-sparc-storage-use1"}
NIFTI_UNITS = {1: "meter", 2: "millimeter", 3: "micrometer"}  # xyzt_units & 7; 0 = unknown
TO_MM = {"meter": 1000.0, "millimeter": 1.0, "micrometer": 0.001}
MARKER = "units_patch"


def _token():
    t = os.environ.get("PENNSIEVE_SESSION_TOKEN")
    if not t:
        sys.exit("Set PENNSIEVE_SESSION_TOKEN")
    return t


def api_get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_token()}"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def http_get(url, rng=None):
    headers = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120).read()


def nifti_header(pkg):
    """(filename, pixdim[1:4], unit name or None) from the source file's header."""
    srcs = [s["content"] for s in api_get(f"{API}/packages/{pkg}/sources")
            if s["content"].get("filename", "").lower().endswith((".nii", ".nii.gz"))]
    if len(srcs) != 1:
        raise ValueError(f"{len(srcs)} NIfTI source files")
    url = api_get(f"{API}/packages/{pkg}/files/{srcs[0]['id']}")["url"]
    b = http_get(url, (0, 65535))
    if b[:2] == b"\x1f\x8b":
        b = zlib.decompressobj(31).decompress(b)
    if struct.unpack("<i", b[:4])[0] == 348:
        e = "<"
    elif struct.unpack(">i", b[:4])[0] == 348:
        e = ">"
    else:
        raise ValueError("not a NIfTI-1 header")
    pixdim = struct.unpack(e + "8f", b[76:108])[1:4]
    return srcs[0]["filename"], [float(v) for v in pixdim], NIFTI_UNITS.get(b[123] & 7)


def close(a, b):
    return abs(a - b) <= 1e-5 * max(1.0, abs(a), abs(b))


def plan(attrs, filename, pixdim, header_unit, from_filename):
    """Return (new_attrs or None, note)."""
    ms = (attrs.get("multiscales") or [None])[0]
    if not ms:
        return None, "no multiscales"
    if ms.get("type") != "downscale_local_mean" or [a["name"] for a in ms["axes"]] != ["x", "y", "z"]:
        return None, "not written by the NIfTI converter"
    if ms.get("metadata", {}).get(MARKER):
        return None, "already patched"
    cur_unit = ms["axes"][0].get("unit")
    scales = [next(t for t in d["coordinateTransformations"] if t["type"] == "scale")["scale"]
              for d in ms["datasets"]]

    # Level-0 scale is the header voxel size (possibly reordered by reorientation,
    # possibly times the initial downsample factor). Check before trusting it.
    ratios = [s / p for s, p in zip(sorted(scales[0]), sorted(pixdim)) if p > 0]
    if len(ratios) != 3 or not all(close(r, ratios[0]) for r in ratios):
        return None, f"level-0 scale {scales[0]} doesn't match header pixdim {pixdim}; check by hand"

    new_ms = json.loads(json.dumps(ms))
    no_size = header_unit is None and all(close(p, 1.0) for p in pixdim)
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(um|µm|micron)", filename, re.I)

    if header_unit and header_unit != cur_unit:
        for a in new_ms["axes"]:
            a["unit"] = header_unit
        note = f"unit {cur_unit} -> {header_unit} (header unit; scale unchanged)"
    elif no_size and from_filename and m:
        mm = float(m.group(1)) / 1000.0
        for d in new_ms["datasets"]:
            t = next(t for t in d["coordinateTransformations"] if t["type"] == "scale")
            t["scale"] = [v * mm for v in t["scale"]]  # v is the pixel shrink factor here
        for a in new_ms["axes"]:
            a["unit"] = "millimeter"
        note = f"no voxel size in header; {m.group(1)} um from file name -> level 0 {[round(v * mm, 6) for v in scales[0]]} mm"
    elif no_size:
        hint = f" (file name says {m.group(0)}; use --voxel-from-filename)" if m else ""
        return None, f"header has no voxel size, asset says 1 {cur_unit}{hint}"
    else:
        return None, f"correct ({cur_unit}, voxel {pixdim})"

    new_ms.setdefault("metadata", {})[MARKER] = f"{note} ({datetime.date.today().isoformat()})"
    out = dict(attrs)
    out["multiscales"] = [new_ms] + attrs["multiscales"][1:]
    return out, note


def read_packages(path):
    out, ds = [], None
    for line in open(path):
        line = line.strip()
        m = re.match(r"#\s*DATASET_ID=(N:dataset:[0-9a-f-]+)", line)
        if m:
            ds = m.group(1)
            continue
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").replace("\t", " ").split()
        d = next((p for p in parts[1:] if p.startswith("N:dataset:")), ds)
        if not d:
            sys.exit(f"No dataset ID for {parts[0]}")
        out.append((parts[0], d))
    return out


def s3_location(asset_url, main_bucket):
    path = asset_url.split("://", 1)[1].split("/", 1)[1].strip("/")
    first = path.split("/", 1)[0]
    if first in PREFIX_BUCKETS:
        return PREFIX_BUCKETS[first], path.split("/", 1)[1]
    if re.fullmatch(r"[0-9a-f]{8}", first):
        raise ValueError(f"unknown CloudFront prefix {first}; add it to PREFIX_BUCKETS")
    if not main_bucket:
        raise ValueError("asset has no CloudFront prefix; pass --main-bucket")
    return main_bucket, path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packages")
    ap.add_argument("--voxel-from-filename", action="store_true",
                    help="for headers with no voxel size, take it from an 'NNum' in the file name")
    ap.add_argument("--apply", action="store_true", help="write the patched zarr.json files (default: dry run)")
    ap.add_argument("--main-bucket", help="bucket for assets without a CloudFront prefix")
    ap.add_argument("--backup-dir", default="zarr_units_backups")
    ap.add_argument("--invalidations", default="cloudfront_invalidation_paths.txt")
    a = ap.parse_args()

    s3 = None
    if a.apply:
        import boto3
        s3 = boto3.client("s3")
        os.makedirs(a.backup_dir, exist_ok=True)

    counts, inval, seen = {}, [], set()
    def bump(k):
        counts[k] = counts.get(k, 0) + 1

    for pkg, ds in read_packages(a.packages):
        try:
            r = api_get(f"{API2}/packages/assets?package_id={pkg}&dataset_id={ds}")
            c = r["cloudfront"]
            q = f"?Policy={c['policy']}&Signature={c['signature']}&Key-Pair-Id={c['key_pair_id']}"
            zarrs = [x for x in r["assets"] if x["asset_type"] == "ome-zarr" and x["status"] == "ready"
                     and x["id"] not in seen]
            if not zarrs:
                print(f"{pkg}\tSKIP no ready ome-zarr asset"); bump("skip")
                continue
            filename, pixdim, header_unit = nifti_header(pkg)
        except Exception as e:
            print(f"{pkg}\tERROR {type(e).__name__}: {e}"); bump("error")
            continue
        for asset in zarrs:
            seen.add(asset["id"])
            tag = f"{pkg}\t{asset['id']}\t{filename}"
            url = asset["asset_url"].rstrip("/") + "/"
            try:
                doc = json.loads(http_get(url + "zarr.json" + q))
                new_attrs, note = plan(doc.get("attributes", {}), filename, pixdim, header_unit,
                                       a.voxel_from_filename)
            except Exception as e:
                print(f"{tag}\tERROR {type(e).__name__}: {e}"); bump("error")
                continue
            if new_attrs is None:
                print(f"{tag}\tSKIP {note}"); bump("skip")
                continue
            print(f"{tag}\t{'PATCH' if a.apply else 'WOULD PATCH'} {note}")
            if a.apply:
                try:
                    bucket, prefix = s3_location(url, a.main_bucket)
                    key = prefix.rstrip("/") + "/zarr.json"
                    cur = s3.get_object(Bucket=bucket, Key=key)
                    body = cur["Body"].read()
                    if json.loads(body) != doc:
                        raise ValueError(f"s3://{bucket}/{key} differs from what CloudFront served; not touching it")
                    with open(os.path.join(a.backup_dir, f"{asset['id']}.zarr.json"), "wb") as f:
                        f.write(body)
                    extra = {k: cur[k] for k in ("ContentType", "CacheControl", "ContentEncoding") if cur.get(k)}
                    s3.put_object(Bucket=bucket, Key=key,
                                  Body=json.dumps(dict(doc, attributes=new_attrs), indent=2).encode(), **extra)
                except Exception as e:
                    print(f"{tag}\tERROR writing: {type(e).__name__}: {e}"); bump("error")
                    continue
            inval.append("/" + url.split("://", 1)[1].split("/", 1)[1] + "zarr.json")
            bump("patch")

    with open(a.invalidations, "w") as f:
        f.write("\n".join(inval) + ("\n" if inval else ""))
    print(f"\nSummary: {counts.get('patch', 0)} {'patched' if a.apply else 'to patch'}, "
          f"{counts.get('skip', 0)} skipped, {counts.get('error', 0)} errors")
    if inval:
        print(f"CloudFront paths to invalidate ({len(inval)}): {a.invalidations}"
              + ("" if a.apply else " (dry run: nothing was written)"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Download the 10x Xenium breast-cancer files named in the config.

Idempotent: skips files already present with a non-trivial size. The 1.4 GB
H&E image is optional (--skip-image) for the non-image pipeline.
"""
import os
import urllib.request

from _bootstrap import abspath, common_parser, load_config


def fetch(url: str, dst: str):
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        print(f"  [skip] {os.path.basename(dst)} ({os.path.getsize(dst)/1e6:.1f} MB)")
        return
    print(f"  [get ] {url}")
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)
    print(f"        -> {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")


def main():
    ap = common_parser(__doc__)
    ap.add_argument("--skip-image", action="store_true",
                    help="do not download the 1.4 GB H&E OME-TIFF")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = cfg["dataset"]
    raw = abspath(ds["raw_dir"]); os.makedirs(raw, exist_ok=True)
    base = ds["base_url"]
    files = dict(ds["files"])
    if args.skip_image:
        files.pop("he_image", None)
    for key, fname in files.items():
        # filename already carries the sample prefix; URL = base + suffix
        suffix = fname.replace(ds["sample"], "")
        fetch(base + suffix, os.path.join(raw, fname))
    print("download complete:", raw)


if __name__ == "__main__":
    main()

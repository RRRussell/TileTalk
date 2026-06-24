#!/usr/bin/env python
"""Build the shared candidate cell pool and crop H&E patches per cell.

Selects a (stratified) subset of QC'd cells as the common retrieval pool used
by every baseline, then crops multi-scale H&E patches centered on each cell.
Writes data/processed/{pool_cells.parquet, patches_<scale>.npy}.

If the H&E image is absent, pass --no-image to still emit the pool (patches are
skipped; image baselines will be unavailable). This is the plan's
"non-image pipeline runs first" path.
"""
import os

import numpy as np
import pandas as pd

from _bootstrap import abspath, common_parser, load_config
from tiletalk import data as D
from tiletalk import patches as P


def build_pool(cells: pd.DataFrame, size: int, stratify: bool, seed: int) -> np.ndarray:
    """Return integer row positions (into cells) for the candidate pool."""
    n = len(cells)
    if size >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    if not stratify:
        return np.sort(rng.choice(n, size, replace=False))
    pos = np.arange(n)
    chosen = []
    for ct, grp in pd.Series(cells["cell_type"].values).groupby(cells["cell_type"].values):
        idx = pos[cells["cell_type"].values == ct]
        take = max(1, int(round(size * len(idx) / n)))
        take = min(take, len(idx))
        chosen.append(rng.choice(idx, take, replace=False))
    out = np.sort(np.concatenate(chosen))
    return out


def main():
    ap = common_parser(__doc__)
    ap.add_argument("--no-image", action="store_true",
                    help="skip H&E cropping; emit pool only")
    ap.add_argument("--scales", nargs="+", default=["medium", "neighborhood"],
                    help="patch scales to crop (keys of image.patch_window_px)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds, img, cp = cfg["dataset"], cfg["image"], cfg["candidate_pool"]
    proc = abspath(ds["processed_dir"])

    cells, _, _ = D.load_processed(proc)
    pool_idx = build_pool(cells, cp["size"], cp["stratify"], cp["seed"])
    pool = cells.iloc[pool_idx].reset_index(drop=True).copy()
    pool["pool_idx"] = np.arange(len(pool))
    pool["orig_row"] = pool_idx
    pool.to_parquet(os.path.join(proc, "pool_cells.parquet"))
    print(f"candidate pool: {len(pool)} cells (stratify={cp['stratify']})")
    print(pool["cell_type"].value_counts().to_string())

    if args.no_image:
        print("--no-image: skipping patch extraction"); return

    he_path = os.path.join(proc.replace("processed", "raw"), ds["files"]["he_image"])
    he_path = os.path.join(abspath(ds["raw_dir"]), ds["files"]["he_image"])
    if not os.path.exists(he_path):
        print(f"[warn] H&E not found at {he_path}; skipping patches"); return

    pix = img["pixel_size_um"]
    print(f"loading H&E level {img['level']} (pixel size {pix} um/px) ...")
    image = D.load_he_image(he_path, img["level"])
    print("  image shape", image.shape)
    coords_um = pool[["x_centroid", "y_centroid"]].values
    if img.get("alignment_csv"):
        align_path = os.path.join(abspath(ds["raw_dir"]), img["alignment_csv"])
        affine = np.loadtxt(align_path, delimiter=",")
        print(f"  using affine alignment {img['alignment_csv']} (invert={img.get('affine_invert', False)})")
        centers_px = P.micron_to_pixel_affine(coords_um, affine, img.get("affine_invert", False))
    else:
        centers_px = P.micron_to_pixel(coords_um, pix)

    for scale in args.scales:
        win = img["patch_window_px"][scale]
        print(f"cropping {len(pool)} '{scale}' patches (window {win}px -> "
              f"{img['patch_out_size']}px) ...")
        arr = P.extract_patches(image, centers_px, win, img["patch_out_size"])
        out = os.path.join(proc, f"patches_{scale}.npy")
        np.save(out, arr)
        print(f"  saved {out} {arr.shape} ({arr.nbytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()

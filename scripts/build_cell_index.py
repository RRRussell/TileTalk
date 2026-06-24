#!/usr/bin/env python
"""Encode candidate-pool H&E patches into embeddings and build a search index.

Runs the configured image encoder (BiomedCLIP by default; falls back to the
dependency-free HashEncoder) over each patch scale and writes
data/processed/emb_<encoder>_<scale>.npy. A FAISS CosineIndex is built lazily
at retrieval time from these embeddings.
"""
import os

import numpy as np

from _bootstrap import abspath, common_parser, load_config
from tiletalk import encoders as E


def main():
    ap = common_parser(__doc__)
    ap.add_argument("--encoder", default=None, help="override encoders.image")
    ap.add_argument("--scales", nargs="+", default=["medium", "neighborhood"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = abspath(cfg["dataset"]["processed_dir"])
    enc_cfg = cfg["encoders"]
    name = args.encoder or enc_cfg["image"]

    print(f"loading image encoder: {name}")
    try:
        enc = E.get_encoder(name, hub=enc_cfg["biomedclip_hub"],
                            device=enc_cfg["device"], batch_size=enc_cfg["batch_size"])
    except Exception as e:
        print(f"[warn] could not load '{name}' ({e}); falling back to hash encoder")
        name = "hash"; enc = E.get_encoder("hash")

    for scale in args.scales:
        ppath = os.path.join(proc, f"patches_{scale}.npy")
        if not os.path.exists(ppath):
            print(f"[skip] no patches for scale '{scale}' ({ppath})"); continue
        patches = np.load(ppath)
        print(f"encoding {len(patches)} '{scale}' patches with {name} ...")
        emb = enc.encode_image(patches)
        out = os.path.join(proc, f"emb_{name}_{scale}.npy")
        np.save(out, emb)
        print(f"  saved {out} {emb.shape}")


if __name__ == "__main__":
    main()

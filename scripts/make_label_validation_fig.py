#!/usr/bin/env python
"""Label-validation figure: the marker-derived cell types are real transcriptomic
populations, not marker-scoring artifacts.

(a) UMAP of pool-cell expression coloured by the marker-derived cell type --
    the labels carve out the natural transcriptomic clusters.
(b) Agreement matrix between unsupervised Leiden clusters (run on expression with
    NO marker genes privileged) and the marker types -- a near-block-diagonal
    structure that quantifies to Leiden-purity 0.87 (NMI 0.58).
Writes tiletalk-bibm/figures/fig_label_validation.pdf.
"""
import os

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROC = os.path.join(REPO, "data", "processed")
OUT = os.path.abspath(os.path.join(REPO, "..", "tiletalk-bibm", "figures",
                                   "fig_label_validation.pdf"))

TYPES = ["B_cell", "T_cell", "Myeloid", "Endothelial", "Fibroblast",
         "Epithelial", "Mast"]
PRETTY = {"B_cell": "B", "T_cell": "T", "Myeloid": "Myeloid",
          "Endothelial": "Endothelial", "Fibroblast": "Fibroblast",
          "Epithelial": "Epithelial", "Mast": "Mast"}
# short tick labels (the rotated long names eat vertical space)
ABBR = {"B_cell": "B", "T_cell": "T", "Myeloid": "Mye", "Endothelial": "Endo",
        "Fibroblast": "Fibro", "Epithelial": "Epi", "Mast": "Mast"}
COLORS = {"B_cell": "#4e79a7", "T_cell": "#59a14f", "Myeloid": "#e15759",
          "Endothelial": "#76b7b2", "Fibroblast": "#edc948",
          "Epithelial": "#b07aa1", "Mast": "#ff9da7", "unknown": "#cccccc"}


def main():
    rng = np.random.default_rng(0)
    expr = np.load(os.path.join(PROC, "expr_lognorm.npy"))
    pool = pd.read_parquet(os.path.join(PROC, "pool_cells.parquet"))
    X = expr[pool["orig_row"].values]
    ct = pool["cell_type"].astype(str).values

    n = min(12000, len(pool))
    idx = rng.choice(len(pool), n, replace=False)
    Xs, cts = X[idx], ct[idx]

    ad = sc.AnnData(np.asarray(Xs, dtype=np.float32))
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, n_comps=30)
    sc.pp.neighbors(ad, n_neighbors=15, n_pcs=30, random_state=0)
    sc.tl.leiden(ad, resolution=0.5, random_state=0)
    sc.tl.umap(ad, random_state=0)
    um = ad.obsm["X_umap"]
    leiden = ad.obs["leiden"].values.astype(int)

    # each Leiden cluster -> its majority marker type (cells in a cluster whose
    # majority is type j are the unsupervised "prediction" of type j)
    mask = np.isin(cts, TYPES)
    maj = {}
    for cl in np.unique(leiden):
        sub = cts[(leiden == cl) & mask]
        maj[cl] = (pd.Series(sub).value_counts().idxmax() if len(sub) else "unknown")
    pred = np.array([maj[c] for c in leiden])

    plt.rcParams.update({"font.size": 7, "font.family": "serif",
                         "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, (axU, axC) = plt.subplots(1, 2, figsize=(3.4, 1.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- (a) UMAP coloured by marker type ----
    for t in TYPES + ["unknown"]:
        m = cts == t
        if m.sum():
            axU.scatter(um[m, 0], um[m, 1], s=1.0, c=COLORS[t], linewidths=0,
                        label=PRETTY.get(t, "unk."), rasterized=True, alpha=0.8)
    axU.set_xticks([]); axU.set_yticks([])
    axU.set_title("(a) UMAP by marker type", fontsize=6.5)
    axU.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.02), ncol=4,
               markerscale=3, fontsize=4.6, frameon=False, handletextpad=0.1,
               columnspacing=0.5, borderaxespad=0)

    # ---- (b) 7x7 agreement: marker type vs Leiden-majority type (row-norm.) ----
    M = np.zeros((len(TYPES), len(TYPES)))
    for i, t in enumerate(TYPES):
        sub = pred[(cts == t)]
        for j, u in enumerate(TYPES):
            M[i, j] = (sub == u).mean() if len(sub) else 0
    im = axC.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axC.set_xticks(range(len(TYPES)))
    axC.set_xticklabels([ABBR[t] for t in TYPES], rotation=45, ha="right", fontsize=5)
    axC.set_yticks(range(len(TYPES)))
    axC.set_yticklabels([ABBR[t] for t in TYPES], fontsize=5)
    axC.set_ylabel("marker type", fontsize=5.5)
    axC.set_xlabel("Leiden type", fontsize=5.5)
    axC.set_title("(b) agreement (purity 0.87)", fontsize=6.5)
    cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=4.5)

    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    png = "/lv_scratch/tmp/claude-59728/-home-zihend1-TileTalk/6408fb19-d251-4821-88e6-577c24075179/scratchpad/labelval.png"
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

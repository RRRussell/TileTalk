#!/usr/bin/env python
"""Predict-expression baseline ('predict molecules from H&E, then read them').

Regress the transcriptomic relevance signal the oracle reads -- marker score
(cell type), normalized gene expression (gene marker), proliferation score
(cell state), or neighborhood composition (niche) -- from the SAME frozen,
multi-encoder H&E features TileTalk uses, out-of-fold, and rank by the
prediction. This is the two-stage analogue of TileTalk's direct per-query
relevance head: instead of predicting relevance, it predicts the molecules and
then applies the oracle's scoring.

Writes results/retrieval/scores_predexpr.npy so evaluate_retrieval.py scores it
alongside the oracle and TileTalk (5-fold out-of-fold, no leakage).
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from _bootstrap import abspath, common_parser, load_config, results_dir
from tiletalk import data as D
from tiletalk import features as F

# niche label -> oracle composition target over the pool (same as run_retrieval)
NICHE_ORACLE = {
    "bcell_rich":   lambda c: c["frac_B_cell"].values,
    "tcell_rich":   lambda c: c["frac_T_cell"].values,
    "myeloid_rich": lambda c: c["frac_Myeloid"].values,
    "vascular":     lambda c: c["frac_Endothelial"].values,
    "immune_rich":  lambda c: (c["frac_T_cell"] + c["frac_B_cell"] + c["frac_Myeloid"]).values,
    "tumor_immune_interface":
        lambda c: (c["frac_Epithelial"] * (c["frac_T_cell"] + c["frac_Myeloid"])).values,
}


def oracle_targets(qs, pool, expr, gene_names):
    """The molecular quantity the oracle ranks by, per query (n_q x n_pool)."""
    gi = {g: i for i, g in enumerate(map(str, gene_names))}
    out = np.zeros((len(qs), len(pool)), dtype=np.float64)
    for r, q in enumerate(qs.itertuples()):
        f, lab = q.target_field, q.target_label
        if f == "cell_type":
            out[r] = pool[f"score_{lab}"].values
        elif f == "cell_state":
            out[r] = pool["proliferation_score"].values
        elif f == "gene":
            out[r] = expr[:, gi[lab]] if lab in gi else 0.0
        elif f == "niche":
            out[r] = NICHE_ORACLE[lab](pool)
    return out


def oof_ridge_multi(X, Y, nfolds=5, alpha=1.0, seed=0):
    """Closed-form 5-fold out-of-fold ridge for many targets at once.
    X: (n, d) features; Y: (n, n_q) targets. Returns OOF preds (n, n_q)."""
    n, d = X.shape
    pred = np.zeros_like(Y)
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    eye = alpha * np.eye(d)
    for tr, te in kf.split(X):
        Xtr = X[tr]
        A = Xtr.T @ Xtr + eye               # (d, d)
        B = Xtr.T @ Y[tr]                   # (d, n_q)
        W = np.linalg.solve(A, B)           # (d, n_q)
        pred[te] = X[te] @ W
    return pred


def main():
    cfg = load_config(common_parser(__doc__).parse_args().config)
    proc = abspath(cfg["dataset"]["processed_dir"])
    rdir = results_dir(cfg, "retrieval")
    seed = cfg["seed"]
    k = cfg["neighborhood"]["k"]

    cells, expr_full, gene_names = D.load_processed(proc)
    pool = pd.read_parquet(os.path.join(proc, "pool_cells.parquet"))
    expr = expr_full[pool["orig_row"].values]
    qs = pd.read_csv(os.path.join(proc, "queries.csv"))
    coords = pool[["x_centroid", "y_centroid"]].values

    blocks = F.load_blocks(proc, coords, k)
    specs = F.variant_specs(blocks)
    spec = specs.get("cellseek_all") or specs.get("cellseek_full") or []
    feat = F.assemble(blocks, spec).astype(np.float64)
    tag = "cellseek_all (+UNI2-h)" if "cellseek_all" in specs else "cellseek_full (open)"
    print(f"feature spec: {tag}, dim={feat.shape[1]}, pool={len(pool)}, queries={len(qs)}")

    X = StandardScaler().fit_transform(feat)          # standardize features (no y leakage)
    Y = oracle_targets(qs, pool, expr, gene_names).T  # (n_pool, n_q)
    pred = oof_ridge_multi(X, Y, nfolds=5, alpha=1.0, seed=seed)

    os.makedirs(rdir, exist_ok=True)
    out = pred.T.astype(np.float32)                   # (n_q, n_pool) like other baselines
    np.save(os.path.join(rdir, "scores_predexpr.npy"), out)
    print("wrote", os.path.join(rdir, "scores_predexpr.npy"), out.shape)


if __name__ == "__main__":
    main()

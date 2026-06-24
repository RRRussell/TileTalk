#!/usr/bin/env python
"""Predict-expression baseline ('predict molecules from H&E, then read them').

This is the faithful two-stage analogue of the H&E->expression literature
(ST-Net, HE2RNA, BLEEP, ...): first regress the full (z-scored) panel expression
from the SAME frozen multi-encoder H&E features TileTalk uses, out-of-fold; then
DERIVE the transcriptomic relevance signal from the PREDICTED expression -- marker
scores (cell type), the target gene (gene marker), the proliferation score (cell
state), and the neighborhood composition (niche) -- and rank by it. Predicting the
whole transcriptome and deriving the signal compounds error, unlike TileTalk's
direct per-query relevance head.

Writes results/<tag>/retrieval/scores_predexpr.npy so evaluate_retrieval.py scores
it alongside the oracle and TileTalk (5-fold out-of-fold, no leakage).
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from _bootstrap import abspath, common_parser, load_config, results_dir
from tiletalk import data as D
from tiletalk import features as F
from tiletalk import queries as Q

# niche label -> oracle composition score over a composition DataFrame
NICHE_ORACLE = {
    "bcell_rich":   lambda c: c["frac_B_cell"].values,
    "tcell_rich":   lambda c: c["frac_T_cell"].values,
    "myeloid_rich": lambda c: c["frac_Myeloid"].values,
    "vascular":     lambda c: c["frac_Endothelial"].values,
    "immune_rich":  lambda c: (c["frac_T_cell"] + c["frac_B_cell"] + c["frac_Myeloid"]).values,
    "tumor_immune_interface":
        lambda c: (c["frac_Epithelial"] * (c["frac_T_cell"] + c["frac_Myeloid"])).values,
}


def oof_ridge_multi(X, Y, nfolds=5, alpha=1.0, seed=0):
    """Closed-form 5-fold out-of-fold ridge for many targets at once.
    X: (n, d) features; Y: (n, m) targets. Returns OOF preds (n, m)."""
    n, d = X.shape
    pred = np.zeros_like(Y)
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    eye = alpha * np.eye(d)
    for tr, te in kf.split(X):
        Xtr = X[tr]
        W = np.linalg.solve(Xtr.T @ Xtr + eye, Xtr.T @ Y[tr])   # (d, m)
        pred[te] = X[te] @ W
    return pred


def main():
    cfg = load_config(common_parser(__doc__).parse_args().config)
    proc = abspath(cfg["dataset"]["processed_dir"])
    rdir = results_dir(cfg, "retrieval")
    seed = cfg["seed"]
    k = cfg["neighborhood"]["k"]
    pp = cfg["preprocess"]

    cells, expr_full, gene_names = D.load_processed(proc)
    pool = pd.read_parquet(os.path.join(proc, "pool_cells.parquet"))
    expr = expr_full[pool["orig_row"].values]                 # log-normalized, pool
    qs = pd.read_csv(os.path.join(proc, "queries.csv"))
    coords = pool[["x_centroid", "y_centroid"]].values

    blocks = F.load_blocks(proc, coords, k)
    specs = F.variant_specs(blocks)
    spec = specs.get("cellseek_all") or specs.get("cellseek_full") or []
    feat = F.assemble(blocks, spec).astype(np.float64)
    tag = "cellseek_all (+UNI2-h)" if "cellseek_all" in specs else "cellseek_full (open)"
    print(f"feature spec: {tag}, dim={feat.shape[1]}, pool={len(pool)}, genes={expr.shape[1]}")
    X = StandardScaler().fit_transform(feat)

    # ---- Stage 1: predict the full z-scored expression from H&E (out-of-fold) ----
    expr_z = Q.zscore(expr)
    ez_hat = oof_ridge_multi(X, expr_z, nfolds=5, alpha=1.0, seed=seed)
    r2 = 1.0 - ((expr_z - ez_hat) ** 2).sum() / ((expr_z - expr_z.mean(0)) ** 2).sum()
    print(f"  stage-1 expression regression: overall R^2 = {r2:.3f}")

    # ---- Stage 2: derive the relevance signal from the PREDICTED expression ----
    mk = Q.available_markers(gene_names, verbose=False)
    score_df = Q.marker_scores(ez_hat, gene_names, mk)                  # per-type marker score
    prolif_hat, _ = Q.proliferation(ez_hat, gene_names,
                                    pp.get("proliferation_quantile", 0.90))
    types_hat = Q.assign_cell_types(score_df, pp["celltype_score_threshold"],
                                    pp["celltype_margin"])["cell_type"].values
    comp_hat = Q.neighbor_composition(coords, types_hat, k=k)           # frac_<type>
    gi = {g: i for i, g in enumerate(map(str, gene_names))}

    # ---- Stage 3: rank each query by the predicted quantity (oracle logic) ----
    out = np.zeros((len(qs), len(pool)), dtype=np.float32)
    for r, q in enumerate(qs.itertuples()):
        f, lab = q.target_field, q.target_label
        if f == "cell_type":
            out[r] = score_df[lab].values
        elif f == "cell_state":
            out[r] = prolif_hat
        elif f == "gene":
            out[r] = ez_hat[:, gi[lab]] if lab in gi else 0.0
        elif f == "niche":
            out[r] = NICHE_ORACLE[lab](comp_hat)

    os.makedirs(rdir, exist_ok=True)
    np.save(os.path.join(rdir, "scores_predexpr.npy"), out)
    print("wrote", os.path.join(rdir, "scores_predexpr.npy"), out.shape)


if __name__ == "__main__":
    main()

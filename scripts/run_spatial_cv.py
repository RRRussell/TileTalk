#!/usr/bin/env python
"""Spatially-blocked cross-validation control for TileTalk.

The default 5-fold OOF uses RANDOM per-cell folds. But niche labels (kNN, k=15)
and the spatial-neighbor feature average each draw on a cell's 15 nearest
spatial neighbors, so under random folds a held-out cell's neighbors mostly sit
in the TRAINING fold -- a potential cross-fold leak that could inflate the
niche/neighbor results. This script re-scores TileTalk with SPATIALLY-BLOCKED
folds (KMeans on centroids -> 5 contiguous tissue regions; a held-out region's
neighbors are themselves held out) and reports Enrich@10 overall and by task
type, next to the random-fold version (same head, same seed). A stable result
means the gains are not a spatial-leakage artifact.
"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from _bootstrap import abspath, common_parser, load_config, results_dir
from tiletalk import data as D, features as F, queries as Q, metrics as M, probe as PB


def oof_with_folds(feat, Y, folds, head="logistic", seed=0, device="cuda"):
    """OOF relevance scores using a precomputed fold assignment (TileTalk head)."""
    import torch
    dev = PB._device(device)
    N, Dd = feat.shape
    Qn = Y.shape[1]
    Xt = torch.tensor(feat, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    out = np.zeros((Qn, N), dtype=np.float32)
    for f in np.unique(folds):
        te = np.where(folds == f)[0]
        tr = np.where(folds != f)[0]
        Xtr = Xt[torch.tensor(tr, device=dev)]
        mu = Xtr.mean(0, keepdim=True)
        sd = Xtr.std(0, keepdim=True) + 1e-6
        Xtr_n = (Xtr - mu) / sd
        Xte_n = (Xt[torch.tensor(te, device=dev)] - mu) / sd
        model = PB._make_model(head, Dd, Qn, 256, seed, dev)
        PB._train(model, Xtr_n, Yt[torch.tensor(tr, device=dev)], head, 400, 5e-2, 1e-3)
        model.eval()
        with torch.no_grad():
            pred = model(Xte_n)
        out[:, te] = pred.t().cpu().numpy()
    return out


def oof_buffered(feat, Y, coords, folds, buffer, head="logistic", seed=0, device="cuda"):
    """Block CV with a buffer: train cells within `buffer` microns of any test
    cell are dropped, so a test cell's spatial neighbors are never in training."""
    import torch
    from sklearn.neighbors import NearestNeighbors
    dev = PB._device(device)
    N, Dd = feat.shape
    Qn = Y.shape[1]
    Xt = torch.tensor(feat, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    out = np.zeros((Qn, N), dtype=np.float32)
    for f in np.unique(folds):
        te = np.where(folds == f)[0]
        d1, _ = NearestNeighbors(n_neighbors=1).fit(coords[te]).kneighbors(coords)
        near = d1[:, 0] <= buffer
        tr = np.where((folds != f) & (~near))[0]
        print(f"    fold {f}: test={len(te)} train={len(tr)} (dropped {(folds!=f).sum()-len(tr)} to buffer)")
        Xtr = Xt[torch.tensor(tr, device=dev)]
        mu = Xtr.mean(0, keepdim=True)
        sd = Xtr.std(0, keepdim=True) + 1e-6
        model = PB._make_model(head, Dd, Qn, 256, seed, dev)
        PB._train(model, (Xtr - mu) / sd, Yt[torch.tensor(tr, device=dev)],
                  head, 400, 5e-2, 1e-3)
        model.eval()
        with torch.no_grad():
            pred = model((Xt[torch.tensor(te, device=dev)] - mu) / sd)
        out[:, te] = pred.t().cpu().numpy()
    return out


def main():
    cfg = load_config(common_parser(__doc__).parse_args().config)
    proc = abspath(cfg["dataset"]["processed_dir"])
    rdir = results_dir(cfg, "retrieval")
    seed = cfg["seed"]
    k = cfg["neighborhood"]["k"]
    dev = cfg["encoders"]["device"]

    cells, expr_full, gene_names = D.load_processed(proc)
    pool = pd.read_parquet(os.path.join(proc, "pool_cells.parquet"))
    expr = expr_full[pool["orig_row"].values]
    qs = pd.read_csv(os.path.join(rdir, "queries.csv"))
    coords = pool[["x_centroid", "y_centroid"]].values

    blocks = F.load_blocks(proc, coords, k)
    specs = F.variant_specs(blocks)
    spec = specs.get("cellseek_all") or specs.get("cellseek_full") or []
    feat = F.assemble(blocks, spec)
    Y = PB.relevance_matrix(qs, pool, expr, gene_names)

    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)

    rand = np.zeros(len(pool), dtype=int)
    for i, (_, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(feat)):
        rand[te] = i
    spat = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(coords)

    # grid tiles -> random fold per tile (positives stay distributed across folds)
    def grid_folds(tile):
        tx = ((coords[:, 0] - coords[:, 0].min()) // tile).astype(int)
        ty = ((coords[:, 1] - coords[:, 1].min()) // tile).astype(int)
        tid = tx * (ty.max() + 1) + ty
        fmap = {t: int(rng.integers(5)) for t in np.unique(tid)}
        return np.array([fmap[t] for t in tid])
    grid = grid_folds(500.0)
    print("fold sizes  random:", np.bincount(rand), " kmeans:", np.bincount(spat),
          " grid500:", np.bincount(grid))

    rel = [Q.relevance_mask(q, pool, expr, gene_names) for _, q in qs.iterrows()]
    ks = cfg["retrieval"]["topk"]
    ndcg_k = cfg["retrieval"]["ndcg_k"]

    def evaldf(S):
        df = pd.DataFrame([{**M.evaluate_query(S[qi], rel[qi], ks=ks, ndcg_k=ndcg_k),
                            "tt": qs.iloc[qi]["task_type"]} for qi in range(len(qs))])
        g = df.groupby("tt")["enrich@10"].mean()
        return df["enrich@10"].mean(), g

    def report(name, folds, buffer=None):
        if buffer is None:
            S = oof_with_folds(feat, Y, folds, seed=seed, device=dev)
        else:
            S = oof_buffered(feat, Y, coords, folds, buffer, seed=seed, device=dev)
        o, g = evaldf(S)
        print(f"{name:22s} overall {o:.2f} | cell_type {g.get('cell_type',0):.2f}  "
              f"niche {g.get('niche',0):.2f}  gene {g.get('gene_marker',0):.2f}  "
              f"state {g.get('cell_state',0):.2f}")

    grid1000 = grid_folds(1000.0)
    report("random-fold", rand)
    report("kmeans-block (5 regions)", spat)
    report("grid500 block", grid)
    report("grid500 block +150um buf", grid, buffer=150.0)
    report("grid1000 block +150um buf", grid1000, buffer=150.0)


if __name__ == "__main__":
    main()

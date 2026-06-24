#!/usr/bin/env python
"""Score matrices -> metric tables.

Reads results/retrieval/scores_<baseline>.npy, resolves ground-truth relevance
per query, and writes:
  results/tables/main_retrieval_results.csv        (baseline x metrics, overall)
  results/tables/query_generalization_results.csv  (baseline x split x metrics)
  results/tables/ablation_results.csv              (image-variant comparison)
  results/tables/per_query_results.csv             (every baseline x query)
  results/tables/results_by_tasktype.csv           (baseline x task_type)
"""
import glob
import os

import numpy as np
import pandas as pd

from _bootstrap import abspath, common_parser, load_config, results_dir
from tiletalk import data as D
from tiletalk import metrics as M
from tiletalk import queries as Q

REPORT = ["hit@1", "P@5", "R@5", "R@10", "mAP", "MRR", "nDCG@10", "enrich@10"]
IMAGE_VARIANTS = ["biomedclip", "biomedclip_nbhd", "plip", "neighborhood",
                  "linear_probe", "cellseek_mlp", "cellseek"]


def main():
    args = common_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    proc = abspath(cfg["dataset"]["processed_dir"])
    rdir = results_dir(cfg, "retrieval")
    tdir = results_dir(cfg, "tables"); os.makedirs(tdir, exist_ok=True)
    ks = cfg["retrieval"]["topk"]; ndcg_k = cfg["retrieval"]["ndcg_k"]

    cells, expr_full, gene_names = D.load_processed(proc)
    pool = pd.read_parquet(os.path.join(proc, "pool_cells.parquet"))
    expr = expr_full[pool["orig_row"].values]
    qs = pd.read_csv(os.path.join(rdir, "queries.csv"))

    # precompute relevance per query
    rel = [Q.relevance_mask(q, pool, expr, gene_names) for _, q in qs.iterrows()]

    score_files = sorted(glob.glob(os.path.join(rdir, "scores_*.npy")))
    baselines = [os.path.basename(f)[len("scores_"):-len(".npy")] for f in score_files]
    print("evaluating baselines:", baselines)

    rows = []
    for b, f in zip(baselines, score_files):
        S = np.load(f)
        for qi, (_, q) in enumerate(qs.iterrows()):
            m = M.evaluate_query(S[qi], rel[qi], ks=ks, ndcg_k=ndcg_k)
            m.update(baseline=b, query_id=q["query_id"], query_text=q["query_text"],
                     task_type=q["task_type"], split=q["split"], level=q["level"])
            rows.append(m)
    per_q = pd.DataFrame(rows)
    per_q.to_csv(os.path.join(tdir, "per_query_results.csv"), index=False)

    metric_cols = [c for c in per_q.columns if any(
        c.startswith(p) for p in ("hit@", "P@", "R@", "enrich@")) or
        c in ("mAP", "MRR", f"nDCG@{ndcg_k}")]

    def agg(df, by):
        g = df.groupby(by)[metric_cols].mean().reset_index()
        return g

    # ---- main: overall per baseline ----
    main = agg(per_q, "baseline")
    main = main[["baseline"] + [c for c in REPORT if c in main.columns]]
    main = main.sort_values("mAP", ascending=False)
    main.round(4).to_csv(os.path.join(tdir, "main_retrieval_results.csv"), index=False)
    print("\n=== main_retrieval_results ===\n", main.round(4).to_string(index=False))

    # ---- generalization: baseline x split ----
    gen = agg(per_q, ["baseline", "split"])
    gen = gen[["baseline", "split"] + [c for c in REPORT if c in gen.columns]]
    gen.round(4).to_csv(os.path.join(tdir, "query_generalization_results.csv"), index=False)

    # ---- by task type ----
    byt = agg(per_q, ["baseline", "task_type"])
    byt = byt[["baseline", "task_type"] + [c for c in REPORT if c in byt.columns]]
    byt.round(4).to_csv(os.path.join(tdir, "results_by_tasktype.csv"), index=False)

    # ---- ablation: image variants only ----
    abl = per_q[per_q["baseline"].isin(IMAGE_VARIANTS)]
    if len(abl):
        abl_t = agg(abl, "baseline")
        abl_t = abl_t[["baseline"] + [c for c in REPORT if c in abl_t.columns]]
        abl_t.round(4).to_csv(os.path.join(tdir, "ablation_results.csv"), index=False)
        print("\n=== ablation_results ===\n", abl_t.round(4).to_string(index=False))

    print("\nwrote tables to", tdir)


if __name__ == "__main__":
    main()

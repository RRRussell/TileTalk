#!/usr/bin/env python
"""Preprocess a Xenium bundle into TileTalk's processed artifacts.

Steps: load cells + expression -> QC filter -> library-size + log1p normalize
-> marker z-score cell typing -> proliferation state -> spatial kNN niche
composition + labels. Writes data/processed/{cells.parquet, expr_lognorm.npy,
gene_names.json} and results/tables/dataset_stats.csv.
"""
import os

import numpy as np
import pandas as pd

from _bootstrap import abspath, common_parser, load_config, results_dir
from tiletalk import data as D
from tiletalk import queries as Q


def main():
    args = common_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    ds, pp, nb = cfg["dataset"], cfg["preprocess"], cfg["neighborhood"]
    raw = abspath(ds["raw_dir"])
    proc = abspath(ds["processed_dir"]); os.makedirs(proc, exist_ok=True)

    # ---- load ----
    print("loading cells + expression ...")
    cells = D.load_cells(os.path.join(raw, ds["files"]["cells"]))
    expr = D.load_expression(os.path.join(raw, ds["files"]["cell_feature_matrix"]))
    # align expression rows to cells.csv order
    order = pd.Index(expr.cell_ids).get_indexer(cells.index)
    assert (order >= 0).all(), "cell_id mismatch between cells.csv and matrix.h5"
    counts = expr.counts[order]
    gene_names = expr.gene_names
    print(f"  {cells.shape[0]} cells x {len(gene_names)} genes")

    # ---- QC filter ----
    keep = cells["transcript_counts"].values >= pp["min_transcript_counts"]
    if pp.get("min_cell_area", 0):
        keep &= cells["cell_area"].values >= pp["min_cell_area"]
    n0 = len(cells)
    cells = cells[keep].copy(); counts = counts[keep]
    print(f"  QC: kept {len(cells)}/{n0} cells "
          f"(>= {pp['min_transcript_counts']} transcripts)")

    # ---- normalize + marker typing ----
    print("normalizing + marker scoring ...")
    lognorm = D.normalize_expression(counts, pp.get("normalize_target_sum"))
    expr_z = Q.zscore(lognorm)
    mk = Q.available_markers(gene_names)
    scores = Q.marker_scores(expr_z, gene_names, mk)
    types = Q.assign_cell_types(scores, pp["celltype_score_threshold"], pp["celltype_margin"])
    prolif_score, is_prolif = Q.proliferation(expr_z, gene_names, pp["proliferation_quantile"])

    cells = cells.reset_index()
    cells["cell_type"] = types["cell_type"].values
    cells["cell_type_score"] = types["cell_type_score"].values
    cells["cell_type_margin"] = types["cell_type_margin"].values
    cells["proliferation_score"] = prolif_score
    cells["is_proliferating"] = is_prolif
    for ct in Q.LINEAGE_TYPES:
        cells[f"score_{ct}"] = scores[ct].values

    print("  cell-type counts:\n",
          cells["cell_type"].value_counts().to_string())

    # ---- spatial niches ----
    print(f"building spatial {nb['use']} neighborhood (k={nb['k']}) ...")
    coords = cells[["x_centroid", "y_centroid"]].values
    comp = Q.neighbor_composition(coords, cells["cell_type"].values,
                                  k=nb["k"], radius_um=nb["radius_um"], use=nb["use"])
    for c in comp.columns:
        cells[c] = comp[c].values
    niches = Q.niche_labels(comp, nb["niche_thresholds"])
    for c in niches.columns:
        cells[c] = niches[c].values
    niche_counts = {k: int(cells[f"niche_{k}"].sum()) for k in Q.NICHE_KEYS}
    print("  niche counts:", niche_counts)

    # ---- save ----
    D.save_processed(proc, cells, lognorm, gene_names)
    print("saved processed artifacts to", proc)

    # ---- dataset stats table ----
    tdir = results_dir(cfg, "tables"); os.makedirs(tdir, exist_ok=True)
    stats = [
        ("dataset", ds["name"]),
        ("sample", ds["sample"]),
        ("n_cells_raw", n0),
        ("n_cells_qc", len(cells)),
        ("n_genes_panel", len(gene_names)),
        ("median_transcripts", float(np.median(cells["transcript_counts"]))),
        ("n_proliferating", int(cells["is_proliferating"].sum())),
    ]
    stats += [(f"n_{ct}", int((cells['cell_type'] == ct).sum()))
              for ct in Q.LINEAGE_TYPES + ["unknown"]]
    stats += [(f"n_niche_{k}", niche_counts[k]) for k in Q.NICHE_KEYS]
    pd.DataFrame(stats, columns=["stat", "value"]).to_csv(
        os.path.join(tdir, "dataset_stats.csv"), index=False)
    print("wrote", os.path.join(tdir, "dataset_stats.csv"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build the natural-language query set and write data/processed/queries.csv.

Columns: query_id, query_text, target_field, target_label, task_type, level,
split. Splits = {seen, paraphrase, unseen_concept} for generalization studies.
"""
import os

from _bootstrap import abspath, common_parser, load_config
from tiletalk import queries as Q


def main():
    args = common_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    proc = abspath(cfg["dataset"]["processed_dir"]); os.makedirs(proc, exist_ok=True)
    qs = Q.build_query_set()
    out = os.path.join(proc, "queries.csv")
    qs.to_csv(out, index=False)
    print(f"wrote {len(qs)} queries -> {out}")
    print(qs.groupby(["task_type", "split"]).size().to_string())


if __name__ == "__main__":
    main()

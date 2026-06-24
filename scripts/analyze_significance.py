#!/usr/bin/env python
"""Bootstrap 95% CIs per method and paired significance vs TileTalk.

Resamples the query set (the unit of evaluation) to put error bars on the
macro-averaged metrics and to test whether TileTalk's margin over each baseline
is significant. Reads results/<tag>/tables/per_query_results.csv; writes
significance_results.csv (+ .tex).
"""
import os

import pandas as pd

from _bootstrap import common_parser, load_config, results_dir
from tiletalk import stats as S

METRICS = ["mAP", "nDCG@10", "enrich@10"]
REF = "cellseek"
ORDER = ["random", "biomedclip", "plip", "biomedclip_prompt", "plip_prompt",
         "biomedclip_nbhd", "neighborhood", "linear_probe", "cellseek_mlp",
         "predexpr", "cellseek", "oracle"]
PRETTY = {"random": "Random", "oracle": "Transcriptomic oracle",
          "biomedclip": "BiomedCLIP zero-shot", "plip": "PLIP zero-shot",
          "biomedclip_prompt": "BiomedCLIP zero-shot+prompts",
          "plip_prompt": "PLIP zero-shot+prompts",
          "biomedclip_nbhd": "BiomedCLIP (nbhd)", "neighborhood": "Neighborhood-aware",
          "linear_probe": "Linear probe", "cellseek_mlp": "TileTalk (MLP head)",
          "predexpr": "Predicted-expression", "cellseek": "TileTalk (ours)"}


def main():
    ap = common_parser(__doc__)
    ap.add_argument("--B", type=int, default=5000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t = results_dir(cfg, "tables")
    pq = pd.read_csv(os.path.join(t, "per_query_results.csv"))
    present = set(pq["baseline"])
    methods = [m for m in ORDER if m in present]  # main comparison only
    qids = sorted(pq["query_id"].unique())

    def vec(method, metric):
        return (pq[pq.baseline == method].set_index("query_id")[metric]
                .reindex(qids).values.astype(float))

    ref = {m: vec(REF, m) for m in METRICS} if REF in present else None
    rows = []
    for method in methods:
        row = {"method": PRETTY.get(method, method)}
        for m in METRICS:
            mean, lo, hi = S.bootstrap_ci(vec(method, m), B=args.B)
            row[m] = f"{mean:.3f} [{lo:.2f},{hi:.2f}]"
            if ref is not None and method not in (REF, "oracle"):
                row[m + "_sig"] = S.stars(S.paired_bootstrap(ref[m], vec(method, m),
                                                             B=args.B)["p_value"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(t, "significance_results.csv"), index=False)
    print(out.to_string(index=False))

    caption = (r"Bootstrap means with 95\% CIs over queries ($B{=}" + str(args.B)
               + r"$). Stars: paired bootstrap test of TileTalk $-$ method "
               r"(*$p{<}0.05$, **$p{<}0.01$, ***$p{<}0.001$).")
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\caption{" + caption + "}",
             r"\label{tab:sig}", r"\begin{tabular}{lcc}", r"\toprule",
             r"Method & mAP & Enrich@10 \\", r"\midrule"]
    for _, r in out.iterrows():
        def cell(metric):
            s = r.get(metric + "_sig", "")
            if pd.isna(s) or s in ("", "ns"):
                return r[metric]
            return r[metric] + f"\\textsuperscript{{{s}}}"
        name = r["method"]
        if name == PRETTY[REF]:
            name = r"\textbf{" + name + "}"
        lines.append(f"{name} & {cell('mAP')} & {cell('enrich@10')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(t, "significance_results.tex"), "w").write("\n".join(lines))
    print("\nwrote significance_results.csv/.tex to", t)


if __name__ == "__main__":
    main()

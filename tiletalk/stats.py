"""Bootstrap confidence intervals and paired significance tests.

Retrieval metrics are macro-averaged over a modest number of queries, so we
quantify uncertainty by bootstrapping over queries: resample the query set with
replacement B times and recompute the macro-average. Paired tests resample the
same query indices for two methods to test whether their mean difference is
significantly non-zero (queries are the unit of resampling).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def bootstrap_ci(per_query: np.ndarray, B: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Mean and (lo, hi) percentile CI of a per-query metric vector."""
    rng = np.random.default_rng(seed)
    n = len(per_query)
    idx = rng.integers(0, n, size=(B, n))
    means = per_query[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(per_query.mean()), float(lo), float(hi)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, B: int = 5000,
                     seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap test of mean(a) - mean(b) over shared query resamples.

    Returns the observed difference, its 95% CI, and a two-sided p-value
    (fraction of resamples whose difference crosses zero, doubled).
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(B, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    obs = float(a.mean() - b.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p: how often the resampled diff is on the opposite side of 0
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": float(min(p, 1.0))}


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"

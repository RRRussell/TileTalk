"""Retrieval metrics for TileTalk.

Every query induces a ranking over the candidate cell pool (by descending
similarity) and a binary relevance vector. Because a query like "B cells" has
many relevant cells, we report the full IR suite: Hit@K / Precision@K /
Recall@K / mAP / MRR / nDCG@K, plus label Enrichment@K (precision over the
base rate) which is the natural spatial-grounding metric.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _rank(scores: np.ndarray, relevance: np.ndarray):
    """Return relevance reordered by descending score (ties broken stably)."""
    order = np.argsort(-scores, kind="stable")
    return relevance[order].astype(np.float64), order


def hit_at_k(rel_sorted: np.ndarray, k: int) -> float:
    return float(rel_sorted[:k].sum() > 0)


def precision_at_k(rel_sorted: np.ndarray, k: int) -> float:
    return float(rel_sorted[:k].sum() / k)


def recall_at_k(rel_sorted: np.ndarray, k: int) -> float:
    total = rel_sorted.sum()
    return float(rel_sorted[:k].sum() / total) if total > 0 else 0.0


def average_precision(rel_sorted: np.ndarray) -> float:
    total = rel_sorted.sum()
    if total == 0:
        return 0.0
    hits = np.cumsum(rel_sorted)
    ranks = np.arange(1, len(rel_sorted) + 1)
    precisions = hits / ranks
    return float((precisions * rel_sorted).sum() / total)


def reciprocal_rank(rel_sorted: np.ndarray) -> float:
    nz = np.nonzero(rel_sorted)[0]
    return float(1.0 / (nz[0] + 1)) if nz.size else 0.0


def ndcg_at_k(rel_sorted: np.ndarray, k: int) -> float:
    rel_k = rel_sorted[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(rel_k) + 2))
    dcg = (rel_k * discounts).sum()
    ideal = np.sort(rel_sorted)[::-1][:k]
    idcg = (ideal * discounts).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def enrichment_at_k(rel_sorted: np.ndarray, k: int) -> float:
    """Precision@K divided by the base rate of relevant items (fold-enrichment)."""
    base = rel_sorted.mean()
    if base == 0:
        return 0.0
    return float(precision_at_k(rel_sorted, k) / base)


def evaluate_query(scores: np.ndarray,
                   relevance: np.ndarray,
                   ks: Sequence[int] = (1, 5, 10, 20, 50),
                   ndcg_k: int = 10) -> Dict[str, float]:
    """All metrics for a single query's score vector + relevance mask."""
    rel_sorted, _ = _rank(scores, relevance)
    out: Dict[str, float] = {}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(rel_sorted, k)
        out[f"P@{k}"] = precision_at_k(rel_sorted, k)
        out[f"R@{k}"] = recall_at_k(rel_sorted, k)
        out[f"enrich@{k}"] = enrichment_at_k(rel_sorted, k)
    out["mAP"] = average_precision(rel_sorted)
    out["MRR"] = reciprocal_rank(rel_sorted)
    out[f"nDCG@{ndcg_k}"] = ndcg_at_k(rel_sorted, ndcg_k)
    out["n_relevant"] = float(relevance.sum())
    out["base_rate"] = float(relevance.mean())
    return out


def aggregate(per_query: List[Dict[str, float]],
              exclude=("n_relevant", "base_rate")) -> Dict[str, float]:
    """Macro-average a list of per-query metric dicts."""
    if not per_query:
        return {}
    keys = [k for k in per_query[0] if k not in exclude]
    return {k: float(np.mean([q[k] for q in per_query])) for k in keys}

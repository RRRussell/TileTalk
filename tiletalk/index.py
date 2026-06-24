"""Similarity search over cell embeddings.

Thin wrapper that uses FAISS when available (inner-product on L2-normalized
vectors == cosine) and falls back to a numpy/sklearn cosine computation. For
the MVP pool sizes (~20k cells, ~30 queries) a dense score matrix is cheap, so
the default path just returns full cosine scores; FAISS is used for top-k.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_scores(queries: np.ndarray, items: np.ndarray) -> np.ndarray:
    """(n_queries, n_items) cosine similarity matrix."""
    return l2_normalize(queries) @ l2_normalize(items).T


class CosineIndex:
    """Cosine-similarity index with FAISS acceleration when present."""

    def __init__(self, items: np.ndarray):
        self.items = l2_normalize(items)
        self._faiss = None
        try:
            import faiss
            idx = faiss.IndexFlatIP(self.items.shape[1])
            idx.add(self.items)
            self._faiss = idx
        except Exception:
            self._faiss = None

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) of top-k items per query."""
        q = l2_normalize(queries)
        k = min(k, self.items.shape[0])
        if self._faiss is not None:
            return self._faiss.search(q, k)
        scores = q @ self.items.T
        idx = np.argsort(-scores, axis=1)[:, :k]
        top = np.take_along_axis(scores, idx, axis=1)
        return top, idx

    def full_scores(self, queries: np.ndarray) -> np.ndarray:
        return l2_normalize(queries) @ self.items.T

"""Shared frozen-feature construction for the TileTalk grounding head.

Loads precomputed encoder embeddings for a processed sample and assembles named
feature blocks (per encoder, per scale, plus a spatial-neighbor-averaged block),
so that retrieval, ablation, and cross-sample transfer all build features the
same way.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np

from .index import l2_normalize


def neighbor_average(emb: np.ndarray, coords: np.ndarray, k: int) -> np.ndarray:
    """Mean embedding over each cell's k nearest spatial neighbors (incl. self)."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords)
    _, idx = nn.kneighbors(coords)
    return emb[idx].mean(axis=1).astype(np.float32)


def load_blocks(proc: str, coords: np.ndarray, k: int = 15,
                encoders: Sequence[str] = ("biomedclip", "plip", "uni2", "gigapath", "conch"),
                scales: Sequence[str] = ("medium", "neighborhood")) -> Dict[str, np.ndarray]:
    """Return {block_name: [N, d]} of L2-normalized frozen-feature blocks.

    Block names: '<enc>_<scale>' for each available embedding, and
    '<enc>_medium_nbravg' (spatial-neighbor average of the medium embedding).
    Missing embedding files are skipped silently.
    """
    blocks: Dict[str, np.ndarray] = {}
    for enc in encoders:
        for scale in scales:
            p = os.path.join(proc, f"emb_{enc}_{scale}.npy")
            if os.path.exists(p):
                blocks[f"{enc}_{scale}"] = l2_normalize(np.load(p))
        med = os.path.join(proc, f"emb_{enc}_medium.npy")
        if os.path.exists(med):
            blocks[f"{enc}_medium_nbravg"] = l2_normalize(
                neighbor_average(np.load(med), coords, k))
    return blocks


def assemble(blocks: Dict[str, np.ndarray], names: List[str]) -> np.ndarray:
    """Concatenate the named blocks into one [N, sum_d] feature matrix."""
    chosen = [blocks[n] for n in names if n in blocks]
    if not chosen:
        return None
    return np.concatenate(chosen, axis=1).astype(np.float32)


# Named feature specifications for ablations. Keys are reported variant names.
def variant_specs(blocks: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
    have = set(blocks)
    specs = {
        "bmc_medium":        ["biomedclip_medium"],
        "bmc_multiscale":    ["biomedclip_medium", "biomedclip_neighborhood"],
        "bmc_full":          ["biomedclip_medium", "biomedclip_neighborhood",
                              "biomedclip_medium_nbravg"],
        "plip_full":         ["plip_medium", "plip_neighborhood", "plip_medium_nbravg"],
        "both_no_context":   ["biomedclip_medium", "biomedclip_neighborhood",
                              "plip_medium", "plip_neighborhood"],
        "cellseek_full":     ["biomedclip_medium", "biomedclip_neighborhood",
                              "biomedclip_medium_nbravg", "plip_medium",
                              "plip_neighborhood", "plip_medium_nbravg"],
        "uni2_full":         ["uni2_medium", "uni2_neighborhood", "uni2_medium_nbravg"],
        "cellseek_all":      ["biomedclip_medium", "biomedclip_neighborhood",
                              "biomedclip_medium_nbravg", "plip_medium",
                              "plip_neighborhood", "plip_medium_nbravg",
                              "uni2_medium", "uni2_neighborhood", "uni2_medium_nbravg"],
        # cellseek_all + the single-cell (small) scale for each encoder
        "cellseek_3scale":   ["biomedclip_small", "biomedclip_medium",
                              "biomedclip_neighborhood", "biomedclip_medium_nbravg",
                              "plip_small", "plip_medium", "plip_neighborhood",
                              "plip_medium_nbravg", "uni2_small", "uni2_medium",
                              "uni2_neighborhood", "uni2_medium_nbravg"],
        "gigapath_full":     ["gigapath_medium", "gigapath_neighborhood",
                              "gigapath_medium_nbravg"],
        "conch_full":        ["conch_medium", "conch_neighborhood", "conch_medium_nbravg"],
        # cellseek_all + CONCH (four frozen pathology encoders, incl. a VLM)
        "cellseek_all_conch": ["biomedclip_medium", "biomedclip_neighborhood",
                              "biomedclip_medium_nbravg", "plip_medium",
                              "plip_neighborhood", "plip_medium_nbravg",
                              "uni2_medium", "uni2_neighborhood", "uni2_medium_nbravg",
                              "conch_medium", "conch_neighborhood", "conch_medium_nbravg"],
        # cellseek_all + GigaPath (four frozen pathology encoders)
        "cellseek_all4":     ["biomedclip_medium", "biomedclip_neighborhood",
                              "biomedclip_medium_nbravg", "plip_medium",
                              "plip_neighborhood", "plip_medium_nbravg",
                              "uni2_medium", "uni2_neighborhood", "uni2_medium_nbravg",
                              "gigapath_medium", "gigapath_neighborhood",
                              "gigapath_medium_nbravg"],
    }
    return {k: [b for b in v if b in have] for k, v in specs.items()
            if all(b in have for b in v)}

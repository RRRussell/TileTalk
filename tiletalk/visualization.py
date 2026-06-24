"""Plotting helpers: cell-type maps, query similarity heatmaps, top-K panels.

All functions write a figure to disk (PDF/PNG) and are matplotlib-only so they
run headless. Spatial plots use micron centroids with y inverted to match
image orientation.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Stable color per coarse cell type.
TYPE_COLORS = {
    "B_cell": "#1f77b4", "T_cell": "#d62728", "Myeloid": "#ff7f0e",
    "Endothelial": "#2ca02c", "Fibroblast": "#9467bd", "Epithelial": "#8c564b",
    "Mast": "#e377c2", "unknown": "#cccccc",
}


def plot_celltype_map(cells, out_path: str, s: float = 1.0, title: str = "Cell types"):
    fig, ax = plt.subplots(figsize=(9, 7))
    for ct, color in TYPE_COLORS.items():
        sub = cells[cells["cell_type"] == ct]
        if len(sub):
            ax.scatter(sub["x_centroid"], sub["y_centroid"], s=s, c=color,
                       label=f"{ct} ({len(sub)})", linewidths=0, rasterized=True)
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_title(title)
    ax.legend(markerscale=6, fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_query_heatmap(coords: np.ndarray, scores: np.ndarray, query_text: str,
                       out_path: str, relevance: Optional[np.ndarray] = None,
                       topk_idx: Optional[np.ndarray] = None):
    """Two panels: similarity heatmap over space, and ground-truth relevant cells."""
    ncols = 2 if relevance is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6), squeeze=False)
    ax = axes[0, 0]
    order = np.argsort(scores)  # draw high scores last
    sc = ax.scatter(coords[order, 0], coords[order, 1], c=scores[order], s=3,
                    cmap="magma", linewidths=0, rasterized=True)
    if topk_idx is not None:
        ax.scatter(coords[topk_idx, 0], coords[topk_idx, 1], s=18,
                   facecolors="none", edgecolors="cyan", linewidths=0.6,
                   label=f"top-{len(topk_idx)}")
        ax.legend(loc="upper right", fontsize=8)
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_title(f'Similarity: "{query_text}"'); fig.colorbar(sc, ax=ax, shrink=0.7)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

    if relevance is not None:
        ax2 = axes[0, 1]
        ax2.scatter(coords[~relevance, 0], coords[~relevance, 1], s=2,
                    c="#dddddd", linewidths=0, rasterized=True)
        ax2.scatter(coords[relevance, 0], coords[relevance, 1], s=4,
                    c="#d62728", linewidths=0, rasterized=True,
                    label=f"relevant ({int(relevance.sum())})")
        ax2.invert_yaxis(); ax2.set_aspect("equal")
        ax2.set_title("Ground-truth relevant cells"); ax2.legend(loc="upper right", fontsize=8)
        ax2.set_xlabel("x (µm)"); ax2.set_ylabel("y (µm)")
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_retrieval_examples(patches: np.ndarray, query_text: str, out_path: str,
                            scores: Optional[Sequence[float]] = None,
                            relevant: Optional[Sequence[bool]] = None, ncol: int = 8):
    """Grid of the top retrieved H&E patches for a query."""
    n = len(patches)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.6 * ncol, 1.7 * nrow))
    axes = np.atleast_2d(axes)
    for i in range(nrow * ncol):
        ax = axes[i // ncol, i % ncol]; ax.axis("off")
        if i < n:
            ax.imshow(patches[i])
            t = f"#{i+1}"
            if scores is not None:
                t += f" {scores[i]:.2f}"
            color = "green" if (relevant is not None and relevant[i]) else "black"
            ax.set_title(t, fontsize=7, color=color)
    fig.suptitle(f'Top retrieved patches: "{query_text}"', fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_niche_map(cells, niche_col: str, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = cells[niche_col].values.astype(bool)
    ax.scatter(cells["x_centroid"][~mask], cells["y_centroid"][~mask], s=1,
               c="#e8e8e8", linewidths=0, rasterized=True)
    ax.scatter(cells["x_centroid"][mask], cells["y_centroid"][mask], s=2,
               c="#d62728", linewidths=0, rasterized=True, label=niche_col)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.legend(markerscale=6, fontsize=8)
    ax.set_title(niche_col); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)

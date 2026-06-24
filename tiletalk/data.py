"""Data loading utilities for 10x Xenium output bundles.

Reads the four core artifacts of a Xenium run:
  * cells.csv.gz            -- per-cell centroids (micron) and QC stats
  * cell_feature_matrix.h5  -- sparse transcript counts (features x cells, CSC)
  * cell_boundaries.csv.gz  -- polygon vertices per cell (optional)
  * <sample>_he_image.ome.tif -- registered H&E (optional, large)

Everything here is dependency-light (numpy/scipy/pandas/h5py) so the
non-image pipeline runs without scanpy / torch / tifffile.
"""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


# --------------------------------------------------------------------------- #
# Cell metadata
# --------------------------------------------------------------------------- #
def load_cells(path: str) -> pd.DataFrame:
    """Load cells.csv.gz -> DataFrame indexed by string cell_id."""
    df = pd.read_csv(path)
    df["cell_id"] = df["cell_id"].astype(str)
    return df.set_index("cell_id")


# --------------------------------------------------------------------------- #
# Expression matrix (10x CellRanger / Xenium .h5)
# --------------------------------------------------------------------------- #
@dataclass
class Expression:
    """Sparse cells x genes count matrix with aligned ids/names."""
    counts: sp.csr_matrix          # (n_cells, n_genes) int
    cell_ids: np.ndarray           # (n_cells,) str
    gene_names: np.ndarray         # (n_genes,) str

    @property
    def n_cells(self) -> int:
        return self.counts.shape[0]

    @property
    def n_genes(self) -> int:
        return self.counts.shape[1]


def load_expression(path: str, gex_only: bool = True) -> Expression:
    """Read a Xenium cell_feature_matrix.h5 into a cells x genes CSR matrix.

    The 10x .h5 stores a CSC matrix of shape (n_features, n_cells); we
    transpose to cells x genes and (by default) keep only the
    'Gene Expression' features, dropping control/blank codewords.
    """
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = tuple(m["shape"][:])                      # (n_features, n_cells)
        names = np.array([x.decode() for x in m["features"]["name"][:]])
        ftype = np.array([x.decode() for x in m["features"]["feature_type"][:]])
        barcodes = np.array([x.decode() for x in m["matrix" if False else "barcodes"][:]]) \
            if "barcodes" in m else np.array([x.decode() for x in f["matrix"]["barcodes"][:]])

    # features x cells (CSC) -> cells x features (CSR)
    feat_by_cell = sp.csc_matrix((data, indices, indptr), shape=shape)
    cells_by_feat = feat_by_cell.T.tocsr()

    if gex_only:
        keep = ftype == "Gene Expression"
        cells_by_feat = cells_by_feat[:, keep]
        names = names[keep]

    return Expression(counts=cells_by_feat, cell_ids=barcodes, gene_names=names)


def normalize_expression(counts: sp.csr_matrix,
                         target_sum: Optional[float] = None) -> np.ndarray:
    """Library-size normalize then log1p (scanpy-style), returns dense float32.

    Each cell is scaled to `target_sum` total counts (default: median of
    per-cell totals over cells with >0 counts), then log1p transformed.
    """
    counts = counts.astype(np.float32)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if target_sum is None:
        nz = totals[totals > 0]
        target_sum = float(np.median(nz)) if nz.size else 1.0
    scale = np.zeros_like(totals)
    nz = totals > 0
    scale[nz] = target_sum / totals[nz]
    norm = counts.multiply(scale[:, None]).tocsr()
    dense = norm.toarray()
    np.log1p(dense, out=dense)
    return dense.astype(np.float32)


# --------------------------------------------------------------------------- #
# Cell boundaries (optional; used for crisp visualization)
# --------------------------------------------------------------------------- #
def load_cell_boundaries(path: str) -> pd.DataFrame:
    """Load cell_boundaries.csv.gz -> DataFrame [cell_id, vertex_x, vertex_y]."""
    df = pd.read_csv(path)
    df["cell_id"] = df["cell_id"].astype(str)
    return df


# --------------------------------------------------------------------------- #
# H&E image (optional, requires tifffile)
# --------------------------------------------------------------------------- #
def he_pixel_size_um(path: str, default: float = 0.363788) -> float:
    """Read OME PhysicalSizeX (micron/pixel) from the registered H&E TIFF."""
    try:
        import re
        import tifffile
        ome = tifffile.TiffFile(path).ome_metadata
        if ome:
            m = re.search(r'PhysicalSizeX="([^"]+)"', ome)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return default


def load_he_image(path: str, level: int = 0) -> np.ndarray:
    """Load a pyramid level of the registered H&E as an (H, W, 3) uint8 array."""
    import tifffile
    series = tifffile.TiffFile(path).series[0]
    if level >= len(series.levels):
        level = len(series.levels) - 1
    return series.levels[level].asarray()


def he_level_shape(path: str, level: int = 0):
    import tifffile
    series = tifffile.TiffFile(path).series[0]
    level = min(level, len(series.levels) - 1)
    return series.levels[level].shape


# --------------------------------------------------------------------------- #
# Processed-artifact IO
# --------------------------------------------------------------------------- #
def save_processed(processed_dir: str,
                   cells: pd.DataFrame,
                   expr_lognorm: np.ndarray,
                   gene_names: np.ndarray) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    cells.to_parquet(os.path.join(processed_dir, "cells.parquet"))
    np.save(os.path.join(processed_dir, "expr_lognorm.npy"), expr_lognorm)
    with open(os.path.join(processed_dir, "gene_names.json"), "w") as fh:
        json.dump(list(map(str, gene_names)), fh)


def load_processed(processed_dir: str):
    cells = pd.read_parquet(os.path.join(processed_dir, "cells.parquet"))
    expr = np.load(os.path.join(processed_dir, "expr_lognorm.npy"))
    with open(os.path.join(processed_dir, "gene_names.json")) as fh:
        gene_names = np.array(json.load(fh))
    return cells, expr, gene_names

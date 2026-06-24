"""Marker-based cell typing, spatial niche labels, and the text query set.

This module owns the *label space* of TileTalk: it turns transcript counts
into coarse cell types (marker scoring), turns the spatial graph into niche
labels, builds the natural-language query set with generalization splits, and
provides the single relevance resolver that maps a query to a boolean ground
truth mask over cells. Retrieval and metrics both call `relevance_mask`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Panel-adapted marker sets
# --------------------------------------------------------------------------- #
# Genes follow the plan's marker lists, augmented with markers that are present
# in the 313-gene Xenium breast panel (several plan genes such as COL1A1, DCN,
# KRT18/KRT19 are absent from this targeted panel). `available_markers` drops
# any gene not in the loaded panel and warns, so the same dicts work on other
# panels.
MARKER_SETS: Dict[str, List[str]] = {
    "B_cell":       ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "TCL1A"],
    "T_cell":       ["CD3D", "CD3E", "CD3G", "TRAC", "CD247", "IL7R", "CD8A", "CD8B"],
    "Myeloid":      ["LYZ", "CD68", "C1QA", "C1QC", "CD163", "ITGAX", "TYROBP",
                     "AIF1", "CD14", "FCGR3A"],
    "Endothelial":  ["PECAM1", "VWF", "KDR", "CLDN5", "CLEC14A", "EGFL7",
                     "RAMP2", "MMRN2", "AQP1", "CD93"],
    "Fibroblast":   ["LUM", "PDGFRA", "PDGFRB", "POSTN", "SFRP4", "FBLN1",
                     "MMP2", "PCOLCE", "DPT", "CCDC80", "LRRC15"],
    "Epithelial":   ["EPCAM", "KRT8", "KRT7", "CDH1", "ELF3", "TACSTD2",
                     "KRT5", "KRT14", "KRT15", "CLDN4", "S100A14", "FOXA1", "GATA3"],
    "Mast":         ["TPSAB1", "CPA3", "KIT", "CTSG", "HDC"],
}

# Cell *state* (orthogonal to lineage) -- scored separately, not in argmax.
PROLIFERATION_MARKERS: List[str] = ["MKI67", "TOP2A", "CENPF", "PCLAF"]

LINEAGE_TYPES = list(MARKER_SETS.keys())


def available_markers(gene_names, verbose: bool = True) -> Dict[str, List[str]]:
    """Intersect every marker set with the loaded panel; warn on dropped genes."""
    present = set(map(str, gene_names))
    out = {}
    for ct, genes in MARKER_SETS.items():
        kept = [g for g in genes if g in present]
        dropped = [g for g in genes if g not in present]
        if verbose and dropped:
            print(f"[markers] {ct}: dropped {dropped} (not in panel); using {kept}")
        out[ct] = kept
    return out


# --------------------------------------------------------------------------- #
# Marker scoring + coarse cell typing
# --------------------------------------------------------------------------- #
def zscore(expr: np.ndarray) -> np.ndarray:
    """Per-gene z-score across cells (expr is log-normalized cells x genes)."""
    mu = expr.mean(axis=0, keepdims=True)
    sd = expr.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (expr - mu) / sd


def marker_scores(expr_z: np.ndarray, gene_names, marker_sets: Dict[str, List[str]]) -> pd.DataFrame:
    """score(cell, type) = mean z-scored expression over that type's markers."""
    gi = {g: i for i, g in enumerate(map(str, gene_names))}
    cols = {}
    for ct, genes in marker_sets.items():
        idx = [gi[g] for g in genes if g in gi]
        cols[ct] = expr_z[:, idx].mean(axis=1) if idx else np.zeros(expr_z.shape[0])
    return pd.DataFrame(cols)


def assign_cell_types(scores: pd.DataFrame,
                      threshold: float = 0.0,
                      margin: float = 0.10) -> pd.DataFrame:
    """Argmax cell typing with a confidence margin; ambiguous -> 'unknown'."""
    arr = scores[LINEAGE_TYPES].values
    order = np.argsort(-arr, axis=1)
    top = arr[np.arange(len(arr)), order[:, 0]]
    second = arr[np.arange(len(arr)), order[:, 1]]
    labels = np.array(LINEAGE_TYPES)[order[:, 0]]
    confident = (top > threshold) & ((top - second) >= margin)
    labels = np.where(confident, labels, "unknown")
    return pd.DataFrame({"cell_type": labels,
                         "cell_type_score": top,
                         "cell_type_margin": top - second})


def proliferation(expr_z: np.ndarray, gene_names, quantile: float = 0.90):
    """Proliferation score (mean z over MKI67/TOP2A/...) + boolean state flag."""
    gi = {g: i for i, g in enumerate(map(str, gene_names))}
    idx = [gi[g] for g in PROLIFERATION_MARKERS if g in gi]
    score = expr_z[:, idx].mean(axis=1) if idx else np.zeros(expr_z.shape[0])
    thr = np.quantile(score, quantile)
    return score, score >= thr


# --------------------------------------------------------------------------- #
# Spatial neighborhood composition + niche labels
# --------------------------------------------------------------------------- #
def neighbor_composition(coords: np.ndarray,
                         cell_type: np.ndarray,
                         k: int = 15,
                         radius_um: Optional[float] = None,
                         use: str = "knn") -> pd.DataFrame:
    """Per-cell fraction of spatial neighbors belonging to each lineage type.

    coords: (n,2) micron centroids. Returns DataFrame with columns
    'frac_<Type>' for each lineage type (neighbors only, self excluded).
    """
    from sklearn.neighbors import NearestNeighbors

    n = coords.shape[0]
    types = LINEAGE_TYPES
    type_to_col = {t: j for j, t in enumerate(types)}
    onehot = np.zeros((n, len(types)), dtype=np.float32)
    for i, t in enumerate(cell_type):
        if t in type_to_col:
            onehot[i, type_to_col[t]] = 1.0

    if use == "radius" and radius_um is not None:
        nn = NearestNeighbors(radius=radius_um).fit(coords)
        _, idx = nn.radius_neighbors(coords)
        comp = np.zeros((n, len(types)), dtype=np.float32)
        for i, nb in enumerate(idx):
            nb = nb[nb != i]
            if len(nb):
                comp[i] = onehot[nb].sum(axis=0) / len(nb)
    else:
        nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords)
        _, idx = nn.kneighbors(coords)
        idx = idx[:, 1:]                       # drop self
        comp = onehot[idx].mean(axis=1)        # mean one-hot over neighbors

    return pd.DataFrame(comp, columns=[f"frac_{t}" for t in types])


def niche_labels(comp: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Derive (possibly overlapping) boolean niche labels from composition.

    Niches: bcell_rich, tcell_rich, myeloid_rich, vascular, immune_rich,
    tumor_immune_interface. Also a single priority `niche_label` for plots.
    """
    f = comp
    out = pd.DataFrame(index=comp.index)
    out["niche_bcell_rich"] = f["frac_B_cell"] >= thresholds.get("bcell_rich", 0.15)
    out["niche_tcell_rich"] = f["frac_T_cell"] >= thresholds.get("tcell_rich", 0.15)
    out["niche_myeloid_rich"] = f["frac_Myeloid"] >= thresholds.get("myeloid_rich", 0.15)
    out["niche_vascular"] = (f["frac_Endothelial"] >= thresholds.get("vascular", 0.08)) & \
                            (f["frac_Fibroblast"] > 0.0)
    out["niche_immune_rich"] = (f["frac_T_cell"] + f["frac_B_cell"] +
                                f["frac_Myeloid"]) >= 0.30
    out["niche_tumor_immune_interface"] = (
        (f["frac_Epithelial"] >= thresholds.get("tumor_immune_interface_epi", 0.25)) &
        ((f["frac_T_cell"] + f["frac_Myeloid"]) >=
         thresholds.get("tumor_immune_interface_imm", 0.15)))

    # priority single label (rarer / more specific niches win)
    priority = ["niche_tumor_immune_interface", "niche_vascular", "niche_bcell_rich",
                "niche_myeloid_rich", "niche_tcell_rich", "niche_immune_rich"]
    label = np.array(["none"] * len(out), dtype=object)
    for col in reversed(priority):
        label[out[col].values] = col.replace("niche_", "")
    out["niche_label"] = label
    return out


NICHE_KEYS = ["bcell_rich", "tcell_rich", "myeloid_rich", "vascular",
              "immune_rich", "tumor_immune_interface"]


# --------------------------------------------------------------------------- #
# Natural-language query set
# --------------------------------------------------------------------------- #
# Each entry: (query_text, target_field, target_label, task_type, split)
#   target_field in {cell_type, cell_state, gene, niche} -> selects resolver path
#   split in {seen, paraphrase, unseen_concept}
_QUERY_TEMPLATES = [
    # ---- cell-type, canonical (seen) ----
    ("B cells", "cell_type", "B_cell", "cell_type", "seen"),
    ("T cells", "cell_type", "T_cell", "cell_type", "seen"),
    ("macrophage-rich areas", "cell_type", "Myeloid", "cell_type", "seen"),
    ("endothelial cells", "cell_type", "Endothelial", "cell_type", "seen"),
    ("fibroblasts", "cell_type", "Fibroblast", "cell_type", "seen"),
    ("tumor epithelial cells", "cell_type", "Epithelial", "cell_type", "seen"),
    ("mast cells", "cell_type", "Mast", "cell_type", "seen"),
    # ---- cell-type, paraphrase (unseen phrasing, seen concept) ----
    ("regions containing B cells", "cell_type", "B_cell", "cell_type", "paraphrase"),
    ("areas with T lymphocytes", "cell_type", "T_cell", "cell_type", "paraphrase"),
    ("myeloid immune cells", "cell_type", "Myeloid", "cell_type", "paraphrase"),
    ("blood vessel lining cells", "cell_type", "Endothelial", "cell_type", "paraphrase"),
    ("stromal fibroblast cells", "cell_type", "Fibroblast", "cell_type", "paraphrase"),
    ("carcinoma epithelial cells", "cell_type", "Epithelial", "cell_type", "paraphrase"),
    # ---- gene-marker (seen) ----
    ("CD79A positive B cell areas", "gene", "CD79A", "gene_marker", "seen"),
    ("CD3D positive T cell regions", "gene", "CD3D", "gene_marker", "seen"),
    ("CD68 positive macrophage regions", "gene", "CD68", "gene_marker", "seen"),
    ("MKI67 positive regions", "gene", "MKI67", "gene_marker", "seen"),
    ("EPCAM positive epithelial regions", "gene", "EPCAM", "gene_marker", "seen"),
    # ---- cell-state ----
    ("proliferating cells", "cell_state", "proliferating", "cell_state", "seen"),
    ("dividing tumor cells", "cell_state", "proliferating", "cell_state", "paraphrase"),
    # ---- niche (seen) ----
    ("B cell rich niche", "niche", "bcell_rich", "niche", "seen"),
    ("T cell rich region", "niche", "tcell_rich", "niche", "seen"),
    ("myeloid rich region", "niche", "myeloid_rich", "niche", "seen"),
    ("vascular niche", "niche", "vascular", "niche", "seen"),
    ("immune rich region", "niche", "immune_rich", "niche", "seen"),
    ("tumor immune interface", "niche", "tumor_immune_interface", "niche", "seen"),
    # ---- niche (paraphrase) ----
    ("region densely populated by B cells", "niche", "bcell_rich", "niche", "paraphrase"),
    ("T cell enriched neighborhood", "niche", "tcell_rich", "niche", "paraphrase"),
    ("perivascular region", "niche", "vascular", "niche", "paraphrase"),
    ("tumor immune boundary", "niche", "tumor_immune_interface", "niche", "paraphrase"),
    # ---- unseen biological concept (novel phrasing -> existing labels) ----
    ("antibody producing immune cells", "cell_type", "B_cell", "cell_type", "unseen_concept"),
    ("cytotoxic lymphocytes", "cell_type", "T_cell", "cell_type", "unseen_concept"),
    ("keratin expressing cells", "cell_type", "Epithelial", "cell_type", "unseen_concept"),
    ("phagocytic immune cells", "cell_type", "Myeloid", "cell_type", "unseen_concept"),
]


def build_query_set() -> pd.DataFrame:
    rows = []
    for i, (text, field, label, task, split) in enumerate(_QUERY_TEMPLATES):
        # niche / cell_state queries are inherently neighborhood-level;
        # cell_type / gene_marker queries are cell-level.
        level = "neighborhood" if task in ("niche",) else "cell"
        rows.append(dict(query_id=f"q{i:03d}", query_text=text, target_field=field,
                         target_label=label, task_type=task, level=level, split=split))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Relevance resolver -- the single source of ground truth for a query
# --------------------------------------------------------------------------- #
def relevance_mask(query_row,
                   cells: pd.DataFrame,
                   expr: Optional[np.ndarray] = None,
                   gene_names=None,
                   gene_quantile: float = 0.90) -> np.ndarray:
    """Boolean ground-truth mask over `cells` rows for one query.

    cells/expr must be row-aligned (same cell order). For gene queries a cell
    is relevant if its log-normalized expression is in the top `gene_quantile`
    for that gene.
    """
    field = query_row["target_field"]
    label = query_row["target_label"]

    if field == "cell_type":
        return (cells["cell_type"].values == label)
    if field == "cell_state":
        if label == "proliferating":
            return cells["is_proliferating"].values.astype(bool)
        raise ValueError(f"unknown cell_state {label}")
    if field == "niche":
        return cells[f"niche_{label}"].values.astype(bool)
    if field == "gene":
        if expr is None or gene_names is None:
            raise ValueError("gene query needs expr + gene_names")
        gi = {g: i for i, g in enumerate(map(str, gene_names))}
        if label not in gi:
            return np.zeros(len(cells), dtype=bool)
        col = expr[:, gi[label]]
        thr = np.quantile(col, gene_quantile)
        return col >= max(thr, 1e-9)
    raise ValueError(f"unknown target_field {field}")

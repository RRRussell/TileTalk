# TileTalk: Querying H&E at Single-Cell Resolution

TileTalk answers natural-language biological queries — a cell type, a marker
gene, a microenvironmental niche — by retrieving the individual cells and local
neighborhoods in H&E histology images that match them, supervised by paired
10x Xenium spatial transcriptomics. This repository contains the core method and
benchmark pipeline for the paper
*"TileTalk: Querying H&E at Single-Cell Resolution."*

## How it works

1. **Labeling** — derive per-cell ground truth from the paired Xenium data:
   marker-gene z-scoring for coarse cell types, and a *k*-NN composition graph for
   spatial niches (no manual annotation).
2. **Encoding** — crop multi-scale H&E patches around each cell and embed them
   with *frozen* pathology encoders (BiomedCLIP, PLIP, and the gated UNI2-h).
3. **Retrieval** — fit a lightweight per-query head over the fused frozen
   features and rank the candidate cell pool. At inference TileTalk uses H&E only.

## Installation

```bash
conda create -n tiletalk python=3.9 -y && conda activate tiletalk
pip install -r requirements.txt
```

## Reproduce

The pipeline downloads the public Xenium breast dataset, builds the benchmark,
and runs retrieval end-to-end:

```bash
bash scripts/run_all.sh                              # breast Rep 1 (open encoders)
WITH_UNI2=1 bash scripts/run_all.sh                  # also use the gated UNI2-h encoder (needs HF access)
CFG=configs/xenium_lung.yaml bash scripts/run_all.sh # cross-tissue (lung)
```

Metric tables land in `results/<tag>/`.

| step | script |
|------|--------|
| download Xenium bundle | `scripts/download_xenium.py` |
| preprocess + derive labels | `scripts/preprocess_xenium.py` |
| build the query set | `scripts/build_query_set.py` |
| crop multi-scale patches | `scripts/extract_cell_patches.py` |
| encode patches (per encoder) | `scripts/build_cell_index.py` |
| run retrieval baselines | `scripts/run_retrieval.py` |
| score with IR metrics | `scripts/evaluate_retrieval.py` |

> In the code, the **`cellseek`** baseline is **TileTalk (ours)** — the per-query
> head over fused frozen features. Other baselines: `random` (chance
> floor), `oracle` (transcriptomic upper bound), `biomedclip` / `plip` (zero-shot
> image–text), `linear_probe` (single-encoder ablation).

## Data

Uses the public *10x Xenium FFPE Human Breast Cancer* dataset
(Janesick *et al.*, *Nat. Commun.* 2023) and an independent Xenium lung section.
Large artifacts (the OME-TIFF, patch tensors, and embeddings) are regenerated
locally by the pipeline and are not tracked by git.

## License

Released under the MIT License (see [LICENSE](LICENSE)).

#!/usr/bin/env bash
# Minimal end-to-end TileTalk pipeline: download -> preprocess -> queries ->
# patches -> encode -> retrieve -> score.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${CFG:-configs/xenium_breast.yaml}"   # or configs/xenium_lung.yaml, xenium_breast_rep2.yaml
WITH_UNI2="${WITH_UNI2:-0}"                 # =1 to also use the gated UNI2-h encoder (needs HF access)

python scripts/download_xenium.py      --config "$CFG"
python scripts/preprocess_xenium.py    --config "$CFG"
python scripts/build_query_set.py      --config "$CFG"
python scripts/extract_cell_patches.py --config "$CFG"
python scripts/build_cell_index.py     --config "$CFG" --encoder biomedclip
python scripts/build_cell_index.py     --config "$CFG" --encoder plip
[ "$WITH_UNI2" = 1 ] && python scripts/build_cell_index.py --config "$CFG" --encoder uni2
python scripts/run_retrieval.py        --config "$CFG" --baselines random oracle biomedclip plip linear_probe cellseek
python scripts/evaluate_retrieval.py   --config "$CFG"
echo "Done. Metric tables are in results/<tag>/."

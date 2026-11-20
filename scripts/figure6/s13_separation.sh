#!/usr/bin/env bash
set -euo pipefail

source /srv/pharos/code/pharos_server_scripts/server_env.sh
source "$PHAROS_CONDA_SH"
conda activate "$PHAROS_ENV_NAME"

BASE=/srv/pharos/figure6_bcc
mkdir -p "$BASE/logs" "$BASE/output"

python "$BASE/subset_post_only.py" > "$BASE/logs/s13_subset.log" 2>&1

pharos admissibility separation \
  --adata "$BASE/output/malignant_bcc_log1p.post_only.SE600M.h5ad" \
  --cell-col "response_pre_post" \
  --cells-per-line 350 \
  --knn-k 50 \
  --umap-n-neighbors 50 \
  --umap-min-dist 0.0 \
  --output-dir "$BASE/output/UMAP_sep_post_only" \
  > "$BASE/logs/s13_separation.log" 2>&1

echo "S13_SEP_EXIT 0 $(date -Is)" >> "$BASE/logs/s13_separation.log"

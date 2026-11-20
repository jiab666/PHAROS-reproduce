#!/usr/bin/env bash
set -euo pipefail

source /srv/pharos/code/pharos_server_scripts/server_env.sh
source "$PHAROS_CONDA_SH"
conda activate "$PHAROS_ENV_NAME"

BASE=/srv/pharos/figure6_bcc
ST_RUN=/srv/pharos/input/ST-SE-Tahoe/fewshot/state_generalization_X_state
META=/srv/pharos/code/PHAROS_reproduce/metadata/drug_metadata_sciplex.csv

mkdir -p "$BASE/logs" "$BASE/output"
echo "OPEN_SEARCH_START $(date -Is)" > "$BASE/logs/open_search.log"

set +e
/usr/bin/time -v pharos open-search \
  --adata "$BASE/output/malignant_bcc_log1p.post_only.SE600M.h5ad" \
  --start-cell "resist_post" \
  --target-cell "response_post" \
  --cell-col "response_pre_post" \
  --model-dir "$ST_RUN" \
  --output-dir "$BASE/output/search_default" \
  --drug-metadata "$META" \
  --skip-sample-drug-report \
  --overwrite \
  >> "$BASE/logs/open_search.log" 2>&1
STATUS=$?
set -e

echo "OPEN_SEARCH_EXIT $STATUS $(date -Is)" >> "$BASE/logs/open_search.log"
exit "$STATUS"

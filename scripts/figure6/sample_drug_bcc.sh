#!/usr/bin/env bash
set -euo pipefail

source /srv/pharos/code/pharos_server_scripts/server_env.sh
source "$PHAROS_CONDA_SH"
conda activate "$PHAROS_ENV_NAME"

BASE=/srv/pharos/figure6_bcc
META=/srv/pharos/code/PHAROS_reproduce/metadata/drug_metadata_sciplex.csv
CELL_META=/srv/pharos/code/PHAROS_reproduce/metadata/cell_line_metadata.csv

mkdir -p "$BASE/logs"
echo "SAMPLE_DRUG_START $(date -Is)" > "$BASE/logs/sample_drug.log"

set +e
python -m pharos_cell.reports.sample_drug \
  --run-dir "$BASE/output/search_default" \
  --drug-metadata "$META" \
  --cell-metadata "$CELL_META" \
  --top-n-paths 100 \
  --top-n-targets 10 \
  --top-n-moas 10 \
  >> "$BASE/logs/sample_drug.log" 2>&1
STATUS=$?
set -e

echo "SAMPLE_DRUG_EXIT $STATUS $(date -Is)" >> "$BASE/logs/sample_drug.log"
exit "$STATUS"

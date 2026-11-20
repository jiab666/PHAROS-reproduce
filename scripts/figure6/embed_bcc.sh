#!/usr/bin/env bash
set -euo pipefail

source /srv/pharos/code/pharos_server_scripts/server_env.sh
source "$PHAROS_CONDA_SH"
conda activate "$PHAROS_ENV_NAME"

SE_DIR=/srv/pharos/input/SE-600M
BASE=/srv/pharos/figure6_bcc
INPUT="$BASE/input/malignant_bcc_log1p.h5ad"
OUTPUT="$BASE/output/malignant_bcc_log1p.SE600M.h5ad"
COMPAT=/srv/pharos/code/pharos_server_scripts/state_emb_compat.py

mkdir -p "$BASE/logs" "$BASE/output"
test -s "$INPUT"
test -s "$SE_DIR/se600m_epoch16.ckpt"
test ! -e "$OUTPUT" || { echo "output already exists; refusing to overwrite" >&2; exit 1; }

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 \
  > "$BASE/logs/embed_bcc_gpu.csv" &
MONITOR_PID=$!

set +e
/usr/bin/time -v python "$COMPAT" emb transform \
  --model-folder "$SE_DIR" \
  --checkpoint "$SE_DIR/se600m_epoch16.ckpt" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --embed-key X_state \
  --batch-size 64 \
  > "$BASE/logs/embed_bcc.log" 2>&1
STATUS=$?
set -e

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
echo "EMBED_BCC_EXIT $STATUS $(date -Is)" >> "$BASE/logs/embed_bcc.log"
exit "$STATUS"

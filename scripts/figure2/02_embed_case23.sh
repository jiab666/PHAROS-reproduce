#!/usr/bin/env bash
set -euo pipefail
source "${1:?Usage: 02_embed_case23.sh server_env.sh}"
source "$PHAROS_CONDA_SH"
conda activate "$PHAROS_ENV_NAME"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
positive_dir="$PHAROS_WORKDIR/positive_controls"
input="$positive_dir/GSE206741_qc_mad_scrublet_log1p.a549_case23_input.h5ad"
output="$positive_dir/GSE206741_qc_mad_scrublet_log1p.a549_case23.SE600M.h5ad"
mkdir -p "$PHAROS_RUNS_DIR/logs"
test -s "$positive_dir/GSE206741_qc_mad_scrublet_log1p.h5ad"
test -s "$PHAROS_CASE1_REFERENCE"
test ! -e "$input" && test ! -e "$output" || { echo 'Case 2/3 input or output already exists; refusing overwrite' >&2; exit 1; }
python "$script_dir/02_make_case23_input.py" \
  --processed "$positive_dir/GSE206741_qc_mad_scrublet_log1p.h5ad" \
  --case1-reference "$PHAROS_CASE1_REFERENCE" --output "$input" \
  > "$PHAROS_RUNS_DIR/logs/case23_prepare_input.log" 2>&1
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > "$PHAROS_RUNS_DIR/logs/case23_embed_gpu.csv" &
monitor_pid=$!
set +e
/usr/bin/time -v python "$script_dir/state_emb_compat.py" emb transform \
  --model-folder "$PHAROS_SE_MODEL_DIR" \
  --checkpoint "$PHAROS_SE_MODEL_DIR/se600m_epoch16.ckpt" \
  --input "$input" --output "$output" --embed-key X_state --batch-size 32 \
  > "$PHAROS_RUNS_DIR/logs/case23_embed.log" 2>&1
status=$?
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
test "$status" -eq 0
python "$script_dir/03_split_a549_cases.py" --input "$output" --output-dir "$positive_dir" \
  > "$PHAROS_RUNS_DIR/logs/case23_split.log" 2>&1
echo "CASE23_EMBED_DONE $(date -Is)" >> "$PHAROS_RUNS_DIR/logs/case23_embed_status.log"

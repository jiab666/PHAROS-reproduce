#!/usr/bin/env bash
set -euo pipefail
source "${1:?Usage: 04_run_hypothesis_cases.sh server_env.sh}"
source "$PHAROS_CONDA_SH"
conda activate "$PHAROS_ENV_NAME"
positive_dir="$PHAROS_WORKDIR/positive_controls"
mkdir -p "$PHAROS_RUNS_DIR/logs"
run_case() {
  local case_id="$1" suffix="$2" target="$3" drug_a="$4" moa_a="$5"
  local input="$positive_dir/GSE206741_qc_mad_scrublet_log1p.${suffix}.SE600M.h5ad"
  local output="$PHAROS_RUNS_DIR/${case_id}_hypothesis"
  test -s "$input"
  test ! -e "$output" || { echo "$output exists; refusing to overwrite" >&2; exit 1; }
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > "$PHAROS_RUNS_DIR/logs/${case_id}_gpu.csv" &
  local monitor_pid=$!
  set +e
  /usr/bin/time -v pharos hypothesis-driven pair \
    --adata "$input" --start-cell DMSO_DMSO --target-cell "$target" \
    --cell-col cell_type --embed-key X_state \
    --model-dir "$PHAROS_ST_MODEL_DIR" \
    --checkpoint "$PHAROS_ST_MODEL_DIR/checkpoints/final.ckpt" \
    --device cuda:0 --drug-pair "$drug_a" Panobinostat \
    --moa-pairs "$moa_a" 'HDAC inhibitor' \
    --random-pairs 100 --random-type matched --batch 5 --batch-selection standard --seed 42 \
    --drug-metadata "$PHAROS_REPRO_REPO/metadata/drug_metadata_sciplex.csv" \
    --output-dir "$output" \
    > "$PHAROS_RUNS_DIR/logs/${case_id}.log" 2>&1
  local status=$?
  set -e
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  return "$status"
}
run_case case2 pano_srt3 panobinostat_SRT3025 Resveratrol 'SIRT1 activator'
run_case case3 pano_alve panobinostat_Alvespimycin Pimitespib 'HSP90 inhibitor'

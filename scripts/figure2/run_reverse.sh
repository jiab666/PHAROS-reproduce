#!/usr/bin/env bash
set -euo pipefail
case_id="${1:?case1, case2, or case3}"
phase="${2:?smoke or formal}"
root='/mnt/d/大四上/实习'
out="$root/pharos_reverse"
env_bin='/home/oslab/miniforge3/envs/PHAROS/bin'
model="$root/models/ST-SE-Tahoe/fewshot/state_generalization_X_state"
metadata="$root/PHAROS_reproduce/metadata/drug_metadata_sciplex.csv"
case "$case_id" in
  case1) input='/home/oslab/pharos-data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad'; start='panobinostat_crizotinib'; drug='crizotinib'; moa='Multi-TK inhibitor' ;;
  case2) input="$root/pharos_server_results/embedded_inputs/GSE206741_qc_mad_scrublet_log1p.pano_srt3.SE600M.h5ad"; start='panobinostat_SRT3025'; drug='Resveratrol'; moa='SIRT1 activator' ;;
  case3) input="$root/pharos_server_results/embedded_inputs/GSE206741_qc_mad_scrublet_log1p.pano_alve.SE600M.h5ad"; start='panobinostat_Alvespimycin'; drug='Pimitespib'; moa='HSP90 inhibitor' ;;
  *) echo "unknown case: $case_id" >&2; exit 2 ;;
esac
case "$phase" in
  smoke) random=10; batches=1 ;;
  formal) random=100; batches=5 ;;
  *) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
run_dir="$out/raw/${case_id}_${phase}"
log="$out/logs/${case_id}_${phase}.log"
gpu_log="$out/logs/${case_id}_${phase}_gpu.csv"
test -s "$input"
test -s "$model/checkpoints/final.ckpt"
test -s "$metadata"
test ! -e "$run_dir" || { echo "refusing to overwrite $run_dir" >&2; exit 3; }
mkdir -p "$out/logs" "$out/raw" "$out/figures"
export PATH="$env_bin:$PATH"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > "$gpu_log" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
set +e
/usr/bin/time -v "$env_bin/pharos" hypothesis-driven pair \
  --adata "$input" --start-cell "$start" --target-cell DMSO_DMSO \
  --cell-col cell_type --embed-key X_state \
  --model-dir "$model" --checkpoint "$model/checkpoints/final.ckpt" \
  --device cuda:0 --drug-pair "$drug" Panobinostat \
  --moa-pairs "$moa" 'HDAC inhibitor' \
  --random-pairs "$random" --random-type matched --batch "$batches" \
  --batch-selection standard --seed 42 --drug-metadata "$metadata" \
  --output-dir "$run_dir" > "$log" 2>&1
status=$?
set -e
echo "$case_id $phase exit=$status log=$log"
exit "$status"

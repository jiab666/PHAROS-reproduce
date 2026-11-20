#!/usr/bin/env bash
set -u
ROOT=/srv/pharos
CASE=${1:?case2 or case3 required}
if [[ "$CASE" == case2 ]]; then
  OUT=$ROOT/runs/open_search_case2
  ADATA=$ROOT/work/positive_controls/GSE206741_qc_mad_scrublet_log1p.pano_srt3.SE600M.h5ad
  TARGET=panobinostat_SRT3025
  PROXY=Resveratrol
elif [[ "$CASE" == case3 ]]; then
  OUT=$ROOT/runs/open_search_case3
  ADATA=$ROOT/work/positive_controls/GSE206741_qc_mad_scrublet_log1p.pano_alve.SE600M.h5ad
  TARGET=panobinostat_Alvespimycin
  PROXY=Pimitespib
else
  echo "Unknown case $CASE" >&2
  exit 2
fi
mkdir -p "$OUT"
if [[ -e "$OUT/start_time.txt" || -e "$OUT/result" ]]; then
  echo "Output already used: $OUT" >&2
  exit 3
fi
avail_kib=$(df -Pk "$OUT" | awk 'NR==2{print $4}')
if (( avail_kib < 15*1024*1024 )); then
  echo "Less than 15 GiB available" >&2
  exit 4
fi
test -s "$ADATA" || exit 5
test -s "$ROOT/input/ST-SE-Tahoe/fewshot/state_generalization_X_state/checkpoints/final.ckpt" || exit 5
test -s "$ROOT/code/PHAROS_reproduce/metadata/drug_metadata_sciplex.csv" || exit 5
nvidia-smi -L > "$OUT/gpu_preflight.txt" || exit 6
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate PHAROS
export PHAROS_CHECK_ADATA="$ADATA" PHAROS_CHECK_TARGET="$TARGET" PHAROS_CHECK_PROXY="$PROXY"
python - <<'PY' > "$OUT/input_preflight.txt" 2>&1
import os
import pandas as pd
import anndata as ad
p = os.environ['PHAROS_CHECK_ADATA']
t = os.environ['PHAROS_CHECK_TARGET']
proxy = os.environ['PHAROS_CHECK_PROXY']
a = ad.read_h5ad(p, backed='r')
counts = a.obs['cell_type'].value_counts()
assert 'X_state' in a.obsm
assert counts.get('DMSO_DMSO', 0) > 0 and counts.get(t, 0) > 0
m = pd.read_csv('/srv/pharos/code/PHAROS_reproduce/metadata/drug_metadata_sciplex.csv')
assert m.astype(str).apply(lambda col: col.str.fullmatch(proxy, case=False).any()).any()
print('input', p, 'shape', a.shape, 'X_state', a.obsm['X_state'].shape)
print('start', 'DMSO_DMSO', int(counts['DMSO_DMSO']))
print('target', t, int(counts[t]))
print('proxy', proxy, 'present_in_metadata', True)
a.file.close()
PY
if [[ $? -ne 0 ]]; then
  echo "Input preflight failed; see $OUT/input_preflight.txt" >&2
  exit 7
fi
cd "$ROOT/code/PHAROS_reproduce" || exit 8
date -Is > "$OUT/start_time.txt"
printf 'timestamp,gpu_used_mib,gpu_total_mib,process_rss_kib,mem_available_kib,swap_used_kib\n' > "$OUT/resource_samples.csv"
PYTHONUNBUFFERED=1 pharos open-search \
  --adata "$ADATA" \
  --start-cell DMSO_DMSO --target-cell "$TARGET" \
  --cell-col cell_type --embed-key X_state \
  --model-dir "$ROOT/input/ST-SE-Tahoe/fewshot/state_generalization_X_state" \
  --checkpoint "$ROOT/input/ST-SE-Tahoe/fewshot/state_generalization_X_state/checkpoints/final.ckpt" \
  --drug-metadata "$ROOT/code/PHAROS_reproduce/metadata/drug_metadata_sciplex.csv" \
  --device cuda:0 --max-depth 2 --beam-size 128 \
  --converter-chunk-size 16 --robust-n-samples 5 \
  --output-dir "$OUT/result" > "$OUT/open_search.log" 2>&1 &
JOB=$!
echo "$JOB" > "$OUT/job.pid"
while kill -0 "$JOB" 2>/dev/null; do
  line=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  rss=$(ps -o rss= -p "$JOB" 2>/dev/null | tr -d ' ')
  avail=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  swap_total=$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)
  swap_free=$(awk '/SwapFree:/ {print $2}' /proc/meminfo)
  printf '%s,%s,%s,%s,%s,%s\n' "$(date -Is)" "${line%,*}" "${line#*,}" "${rss:-0}" "$avail" "$((swap_total-swap_free))" >> "$OUT/resource_samples.csv"
  sleep 0.5
done
wait "$JOB"
rc=$?
echo "$rc" > "$OUT/exit_code.txt"
date -Is > "$OUT/end_time.txt"
exit "$rc"

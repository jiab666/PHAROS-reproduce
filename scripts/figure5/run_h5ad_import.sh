#!/usr/bin/env bash
set -euo pipefail

ROOT='/mnt/d/大四上/实习'
PREP="$ROOT/figure5_preparation"
EXPORT="$PREP/raw/raw_counts_export"
OUTPUT="$PREP/raw/breast_cancer_raw_counts.h5ad"
LOG="$PREP/raw/raw_h5ad_import.log"
PYTHON='/home/oslab/miniforge3/envs/PHAROS/bin/python'

test -f "$EXPORT/manifest.tsv"
test ! -e "$OUTPUT"
ulimit -v 15728640
/usr/bin/time -v "$PYTHON" \
  "$ROOT/PHAROS_reproduce/metastatic_breast_cancer/preprocess/build_h5ad_from_export.py" \
  --export-dir "$EXPORT" --output "$OUTPUT" 2>&1 | tee "$LOG"

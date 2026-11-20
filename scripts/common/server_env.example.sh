#!/usr/bin/env bash
# Set this to the actual writable storage root before running any experiment.
export PHAROS_DATA_ROOT=/path/to/writable-storage/pharos
export PHAROS_CONDA_SH=/usr/local/miniconda3/etc/profile.d/conda.sh
export PHAROS_ENV_NAME=PHAROS
export PHAROS_WORKDIR="$PHAROS_DATA_ROOT/work"
export PHAROS_GEO_DIR="$PHAROS_DATA_ROOT/input/GSE206741_source"
export PHAROS_REPRO_REPO=/srv/pharos/code/PHAROS_reproduce
export PHAROS_SE_MODEL_DIR="$PHAROS_DATA_ROOT/input/SE-600M"
export PHAROS_ST_MODEL_DIR="$PHAROS_DATA_ROOT/input/ST-SE-Tahoe/fewshot/state_generalization_X_state"
export PHAROS_RUNS_DIR="$PHAROS_DATA_ROOT/runs"
export PHAROS_CASE1_REFERENCE="$PHAROS_DATA_ROOT/input/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad"

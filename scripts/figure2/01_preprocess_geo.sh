#!/usr/bin/env bash
set -euo pipefail
source "${1:?Usage: 01_preprocess_geo.sh server_env.sh}"
source "$PHAROS_CONDA_SH"
conda activate "$PHAROS_ENV_NAME"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
positive_dir="$PHAROS_WORKDIR/positive_controls"
mkdir -p "$positive_dir"
mkdir -p "$PHAROS_RUNS_DIR/logs"
exec > >(tee -a "$PHAROS_RUNS_DIR/logs/preprocess_geo.log") 2>&1
test -s "$PHAROS_GEO_DIR/GSE206741_count_matrix.mtx.gz"
gzip -t "$PHAROS_GEO_DIR/GSE206741_count_matrix.mtx.gz"
for name in GSE206741_cell_metadata.tsv.gz GSE206741_gene_metadata.tsv.gz; do
  test -s "$PHAROS_GEO_DIR/$name"
  gzip -t "$PHAROS_GEO_DIR/$name"
done
raw="$positive_dir/GSE206741_raw_counts.h5ad"
processed="$positive_dir/GSE206741_qc_mad_scrublet_log1p.h5ad"
test ! -e "$processed" || { echo 'Processed output already exists; refusing to overwrite' >&2; exit 1; }
if [[ -s "$raw" ]]; then
  python -c 'import anndata as ad,sys; a=ad.read_h5ad(sys.argv[1],backed="r"); assert a.shape == (63430,58347),a.shape; print("Reusing validated raw AnnData",a.shape,flush=True)' "$raw"
else
  /usr/bin/time -v python "$script_dir/build_gse206741_raw.py" --source-dir "$PHAROS_GEO_DIR" --output "$raw"
fi
cd "$PHAROS_WORKDIR"
/usr/bin/time -v python "$script_dir/run_official_preprocess_compat.py" "$PHAROS_REPRO_REPO/CPA_A549_cell_line/preprocess/prepare_CPA_positive_controls.py"
test -s "$processed"
validation_args=(--input "$processed")
if [[ -n "${PHAROS_CASE1_REFERENCE:-}" ]]; then
  validation_args+=(--case1-reference "$PHAROS_CASE1_REFERENCE")
fi
python "$script_dir/01_validate_preprocessed.py" "${validation_args[@]}"

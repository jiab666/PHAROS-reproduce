"""Prepare one sparse SE-600M input for PHAROS Figure 5c and 5d.

The 5c gene filter matches the official malignant-cell preprocessing:
Scanpy min_cells=3 on all malignant cells. The resulting gene set is then
applied to the union of cells needed by 5c and 5d. Per-cell QC, 10k
normalization, and log1p match the official operations. This all-cell union
and the detailed 5d labels are an explicit adapter because no 5d raw-count
preprocessing script is published in PHAROS_reproduce.

The output intentionally omits a duplicate raw-count layer. The separately
verified raw-count h5ad remains the immutable source of those values.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


MET_SAMPLES = (1, 3, 4, 6, 9, 11)
IMMUNE = {
    "T02 CD8 Teffectormemory-GZMK",
    "T03 Trm-ZNF683",
    "T09 CD8 Texhausted-CXCL13",
    "N01 NK-CD16",
    "B02 B Memory",
}
OBJECTIVES = (
    ("conv0", "Malignant_Metastasis", "Malignant_Primary"),
    ("conv1", "T02 CD8 Teffectormemory-GZMK_Metastasis", "T02 CD8 Teffectormemory-GZMK_Primary"),
    ("conv2", "T03 Trm-ZNF683_Metastasis", "T03 Trm-ZNF683_Primary"),
    ("conv3", "T09 CD8 Texhausted-CXCL13_Metastasis", "T03 Trm-ZNF683_Primary"),
    ("conv6", "N01 NK-CD16_Metastasis", "N01 NK-CD16_Primary"),
    ("conv7", "B02 B Memory_Metastasis", "B02 B Memory_Primary"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def assert_sparse_and_finite(matrix: sparse.spmatrix, stage: str) -> None:
    if not sparse.isspmatrix_csr(matrix):
        raise TypeError(f"{stage}: expected CSR sparse matrix, got {type(matrix)!r}")
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError(f"{stage}: nonfinite or negative expression values")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "figure5_union_log1p_for_SE600M.h5ad"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    started = time.monotonic()

    print(f"Reading {args.raw}", flush=True)
    raw = ad.read_h5ad(args.raw)
    assert_sparse_and_finite(raw.X, "raw input")
    if raw.n_obs != 99_197 or raw.n_vars != 28_217:
        raise ValueError(f"Unexpected raw shape: {raw.shape}")
    obs = raw.obs
    malignant = obs["major_celltype"].astype(str).eq("Malignant").to_numpy()
    immune = obs["minor_celltype"].astype(str).isin(IMMUNE).to_numpy()
    # T09 is needed only as the exhausted metastatic source.
    t09_primary = (
        obs["minor_celltype"].astype(str).eq("T09 CD8 Texhausted-CXCL13")
        & obs["Disease"].astype(str).eq("Primary")
    ).to_numpy()
    union_mask = malignant | (immune & ~t09_primary)
    print(f"Malignant cells: {malignant.sum():,}; union: {union_mask.sum():,}", flush=True)

    # The official malignant preprocessing applies filter_genes(min_cells=3)
    # *before* selecting patient-specific Figure 5c cells.
    malignant_counts = raw.X[malignant, :]
    gene_keep, detected_cells = sc.pp.filter_genes(malignant_counts, min_cells=3, inplace=False)
    del malignant_counts
    gene_keep = np.asarray(gene_keep, dtype=bool)
    print(f"Genes retained after official malignant min_cells=3: {gene_keep.sum():,}", flush=True)

    data = raw[union_mask, gene_keep].copy()
    del raw
    gc.collect()
    assert_sparse_and_finite(data.X, "selected raw counts")
    data.var_names_make_unique()
    gene_names = data.var["gene_id"].astype(str)
    data.var["mt"] = gene_names.str.startswith("MT-")
    data.var["ribo"] = gene_names.str.startswith(("RPS", "RPL"))
    sc.pp.calculate_qc_metrics(
        data, qc_vars=["mt", "ribo"], percent_top=None, log1p=True, inplace=True
    )
    assert_sparse_and_finite(data.X, "post-QC raw counts")

    sc.pp.normalize_total(data, target_sum=1e4)
    assert_sparse_and_finite(data.X, "10k normalization")
    sc.pp.log1p(data)
    assert_sparse_and_finite(data.X, "log1p")

    obs = data.obs
    major = obs["major_celltype"].astype(str)
    minor = obs["minor_celltype"].astype(str)
    disease = obs["Disease"].astype(str)
    sample = obs["Sample"].astype(str)
    label_base = pd.Series(np.where(major.eq("Malignant"), major, minor), index=obs.index).astype(str)
    obs["cell_type"] = label_base + "_" + disease
    obs["cell_type_merged"] = major + "_" + disease + "_" + sample
    obs["drugname_drugconc"] = "[('DMSO_TF', 0.0, 'uM')]"

    five_c = []
    for number in MET_SAMPLES:
        start = f"Malignant_Metastasis_Metastasis_{number}"
        target = "Malignant_Primary_Primary_2"
        five_c.append({
            "case": f"Met{number}",
            "start_label": start,
            "target_label": target,
            "start_cells": int(obs["cell_type_merged"].eq(start).sum()),
            "target_cells": int(obs["cell_type_merged"].eq(target).sum()),
        })
    five_d = []
    for case, start, target in OBJECTIVES:
        five_d.append({
            "case": case,
            "start_label": start,
            "target_label": target,
            "start_cells": int(obs["cell_type"].eq(start).sum()),
            "target_cells": int(obs["cell_type"].eq(target).sum()),
        })
    c_frame = pd.DataFrame(five_c)
    d_frame = pd.DataFrame(five_d)
    if (c_frame[["start_cells", "target_cells"]] < 256).any().any():
        raise ValueError("A 5c label has fewer than 256 cells")
    if (d_frame[["start_cells", "target_cells"]] < 128).any().any():
        raise ValueError("A 5d label has fewer than 128 cells")

    data.uns["figure5_preprocessing"] = (
        "Official malignant min_cells=3 gene filter; Scanpy QC, normalize_total(10000), "
        "log1p; union of malignant and required immune cells; no duplicate raw-count layer"
    )
    data.uns["raw_count_source"] = str(args.raw)
    data.uns["x_state_status"] = "not_generated"

    print(f"Writing {destination}", flush=True)
    data.write_h5ad(destination, compression="gzip")
    c_frame.to_csv(out / "figure5c_label_manifest.csv", index=False)
    d_frame.to_csv(out / "figure5d_label_manifest.csv", index=False)

    sample_rows = data.X[: min(100, data.n_obs)].copy()
    sample_rows.data = np.expm1(sample_rows.data)
    sample_sums = np.asarray(sample_rows.sum(axis=1)).ravel()
    nonzero = sample_sums > 0
    audit = {
        "raw_path": str(args.raw),
        "output_path": str(destination),
        "cells": int(data.n_obs),
        "genes": int(data.n_vars),
        "nonzero_expression_values": int(data.X.nnz),
        "matrix_type": type(data.X).__name__,
        "matrix_dtype": str(data.X.dtype),
        "matrix_data_min": float(data.X.data.min()),
        "matrix_data_max": float(data.X.data.max()),
        "all_matrix_values_finite": bool(np.isfinite(data.X.data).all()),
        "all_matrix_values_nonnegative": bool((data.X.data >= 0).all()),
        "malignant_cells_used_for_gene_filter": int(malignant.sum()),
        "genes_before_filter": 28_217,
        "genes_after_filter": int(gene_keep.sum()),
        "normalization_target": 10_000,
        "sampled_nonzero_cell_sum_min_after_expm1": float(sample_sums[nonzero].min()),
        "sampled_nonzero_cell_sum_max_after_expm1": float(sample_sums[nonzero].max()),
        "zero_count_cells_in_first_100": int((~nonzero).sum()),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "five_c": five_c,
        "five_d": five_d,
    }
    (out / "preprocessing_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({k: audit[k] for k in ("cells", "genes", "matrix_type", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()

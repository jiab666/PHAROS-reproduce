"""Low-memory verification of the official raw-count AnnData output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
PATH = BASE / "raw" / "breast_cancer_raw_counts.h5ad"
META = BASE / "Integrated_Dataset_metadata.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    data = ad.read_h5ad(PATH, backed="r")
    clinical_columns = ["Sample", "Disease", "major_celltype", "minor_celltype", "ERstatus", "HER2status"]
    source = pd.read_csv(META, usecols=["CellID"] + clinical_columns, low_memory=False)
    expected_ids = set(source["CellID"].astype(str))
    observed_ids = set(data.obs["cell_id_original"].astype(str))
    sample = data.X[:10, :100].toarray()
    result = {
        "path": str(PATH),
        "bytes": PATH.stat().st_size,
        "sha256": sha256(PATH),
        "n_cells": int(data.n_obs),
        "n_genes": int(data.n_vars),
        "matrix_dtype": str(data.X.dtype),
        "matrix_content": str(data.uns.get("matrix_content")),
        "source_object": [str(x) for x in data.uns.get("source_objects", [])],
        "raw_count_assays": [str(x) for x in data.uns.get("raw_count_assays", [])],
        "raw_count_layers": [str(x) for x in data.uns.get("raw_count_layers", [])],
        "cell_id_exact_set_match": observed_ids == expected_ids,
        "obs_index_unique": bool(data.obs_names.is_unique),
        "gene_index_unique": bool(data.var_names.is_unique),
        "sample_nonnegative_integer": bool(np.isfinite(sample).all() and (sample >= 0).all() and np.equal(sample, np.floor(sample)).all()),
    }
    checked = source.merge(
        data.obs[["cell_id_original"] + clinical_columns],
        left_on="CellID", right_on="cell_id_original", how="inner",
        suffixes=("_csv", "_h5ad"), validate="one_to_one",
    )
    result["h5ad_metadata_value_mismatches"] = {
        column: int((checked[f"{column}_csv"].astype("string").fillna("") != checked[f"{column}_h5ad"].astype("string").fillna("")).sum())
        for column in clinical_columns
    }
    if data.n_obs != 99197 or data.n_vars != 28217:
        raise ValueError(f"unexpected h5ad dimensions: {data.shape}")
    if not result["cell_id_exact_set_match"] or not result["sample_nonnegative_integer"] or any(result["h5ad_metadata_value_mismatches"].values()):
        raise ValueError("h5ad IDs or sampled raw count values failed validation")
    (BASE / "h5ad_import_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    data.file.close()


if __name__ == "__main__":
    main()

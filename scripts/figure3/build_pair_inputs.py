"""Validate new embeddings, reuse existing DMSO, and build four pair inputs."""

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np


BASE = Path("/srv/pharos/figure3_a549")
RAW = Path("/srv/pharos/work/positive_controls/GSE206741_qc_mad_scrublet_log1p.h5ad")
EXISTING = Path("/srv/pharos/work/positive_controls/GSE206741_qc_mad_scrublet_log1p.a549_case23.SE600M.h5ad")
INPUT = BASE / "input/figure3_missing_aligned_log1p.h5ad"
EMBEDDED = BASE / "embedding_aligned/figure3_missing_aligned.SE600M.h5ad"
OUT_DIR = BASE / "pair_inputs"
MANIFEST = OUT_DIR / "pair_inputs_manifest.json"
TARGETS = (
    "Alvespimycin_Pirarubicin", "Cediranib_PCI-34501",
    "Givinostat_Carmofur", "Givinostat_crizotinib",
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() or any(OUT_DIR.glob("*.h5ad")):
        raise FileExistsError("Refusing to overwrite existing pair inputs")
    source = ad.read_h5ad(RAW, backed="r")
    input_data = ad.read_h5ad(INPUT, backed="r")
    new = ad.read_h5ad(EMBEDDED, backed="r")
    old = ad.read_h5ad(EXISTING, backed="r")
    if new.n_obs != 7422 or list(input_data.obs_names) != list(new.obs_names):
        raise ValueError("Embedded output cell order/size differs from input")
    if new.obsm["X_state"].shape != (7422, 2058) or old.obsm["X_state"].shape[1] != 2058:
        raise ValueError("Unexpected X_state shape")
    if not np.isfinite(np.asarray(new.obsm["X_state"][:])).all():
        raise ValueError("New X_state has non-finite values")
    if not np.isfinite(np.asarray(old.obsm["X_state"][:])).all():
        raise ValueError("Existing X_state has non-finite values")
    if not new.var_names.equals(old.var_names):
        raise ValueError("New and existing embedding gene order differs")
    source_labels = source.obs["cell_type"].astype(str)
    dmso_ids = set(source.obs_names[source_labels == "DMSO_DMSO"])
    old_dmso_ids = set(old.obs_names[old.obs["cell_type"] == "DMSO_DMSO"])
    if dmso_ids != old_dmso_ids or len(dmso_ids) != 1400:
        raise ValueError("DMSO cell IDs do not match original preprocessing")
    output = {
        "new_embedding": str(EMBEDDED), "new_embedding_sha256": sha256(EMBEDDED),
        "new_embedding_cells": new.n_obs, "embedding_dims": 2058,
        "reused_dmso_cells": 1400, "genes": new.n_vars, "pair_files": {},
    }
    dmso = old[old.obs["cell_type"] == "DMSO_DMSO"].to_memory()
    for label in TARGETS:
        target_ids = set(source.obs_names[source_labels == label])
        target = new[new.obs["cell_type"] == label].to_memory()
        if set(target.obs_names) != target_ids:
            raise ValueError(f"Target IDs mismatch: {label}")
        pair = ad.concat([dmso, target], join="inner", merge="same", uns_merge="same")
        expected_ids = list(source.obs_names[source_labels.isin(["DMSO_DMSO", label])])
        pair = pair[expected_ids].copy()
        if list(pair.obs_names) != expected_ids:
            raise ValueError(f"Output order mismatch: {label}")
        if pair.obsm["X_state"].shape != (len(expected_ids), 2058):
            raise ValueError(f"Output shape mismatch: {label}")
        path = OUT_DIR / f"{label}.SE600M.h5ad"
        pair.write_h5ad(path)
        output["pair_files"][label] = {
            "path": str(path), "cells": pair.n_obs,
            "dmso_cells": 1400, "target_cells": len(target_ids),
            "bytes": path.stat().st_size, "sha256": sha256(path),
        }
    MANIFEST.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    for file in (source, input_data, new, old):
        file.file.close()


if __name__ == "__main__":
    main()

"""Select only missing A549 cells, aligned to the existing DMSO gene set."""

import json
from pathlib import Path

import anndata as ad


BASE = Path("/srv/pharos/figure3_a549")
RAW = Path("/srv/pharos/work/positive_controls/GSE206741_qc_mad_scrublet_log1p.h5ad")
EXISTING = Path("/srv/pharos/work/positive_controls/GSE206741_qc_mad_scrublet_log1p.a549_case23.SE600M.h5ad")
OUTPUT = BASE / "input/figure3_missing_aligned_log1p.h5ad"
MANIFEST = BASE / "input/aligned_selection_manifest.json"
TARGETS = (
    "Alvespimycin_Pirarubicin", "Cediranib_PCI-34501",
    "Givinostat_Carmofur", "Givinostat_crizotinib",
)


def main():
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("Refusing to overwrite aligned input")
    raw = ad.read_h5ad(RAW, backed="r")
    existing = ad.read_h5ad(EXISTING, backed="r")
    if not raw.var_names.is_unique or not existing.var_names.is_unique:
        raise ValueError("Gene identifiers are not unique")
    if not existing.var_names.isin(raw.var_names).all():
        raise ValueError("Existing DMSO genes are not a subset of source raw genes")
    labels = raw.obs["cell_type"].astype(str)
    selected = raw.obs_names[labels.isin(TARGETS)]
    if len(selected) != 7422 or set(selected) & set(existing.obs_names):
        raise ValueError("Missing-cell selection mismatch")
    aligned = raw[selected, existing.var_names].to_memory()
    if not aligned.var_names.equals(existing.var_names):
        raise ValueError("Gene order does not match existing embedding")
    aligned.write_h5ad(OUTPUT)
    report = {
        "source_raw": str(RAW), "existing_dmso_embedding": str(EXISTING),
        "output": str(OUTPUT), "cells": aligned.n_obs, "genes": aligned.n_vars,
        "target_counts": labels.loc[selected].value_counts().to_dict(),
        "same_gene_order_as_existing": True,
    }
    MANIFEST.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    raw.file.close()
    existing.file.close()


if __name__ == "__main__":
    main()

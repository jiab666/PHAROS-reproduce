"""Extract six 5c patient pairs; register the shared 5d input without copying it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path("/srv/pharos/work/figure5")
PARENT = ROOT / "full_embedding_61901" / "figure5_union.SE600M.h5ad"
DERIVED = ROOT / "derived_inputs"
FIVE_C = DERIVED / "figure5c"
FIVE_D = DERIVED / "figure5d_shared_input.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if DERIVED.exists():
        raise FileExistsError(f"Refusing to reuse nonempty derived input directory: {DERIVED}")
    FIVE_C.mkdir(parents=True)
    started = time.monotonic()
    parent = ad.read_h5ad(PARENT, backed="r")
    c_manifest = pd.read_csv(ROOT / "figure5c_label_manifest.csv")
    d_manifest = pd.read_csv(ROOT / "figure5d_label_manifest.csv")
    if parent.shape != (61_901, 26_597) or "X_state" not in parent.obsm:
        raise ValueError("Shared embedding missing or unexpected shape")
    if parent.obsm["X_state"].shape != (61_901, 2058):
        raise ValueError("Unexpected shared X_state shape")
    labels = parent.obs["cell_type_merged"].astype(str).to_numpy()
    rows = []
    for row in c_manifest.itertuples(index=False):
        selected = np.flatnonzero(np.isin(labels, (row.start_label, row.target_label)))
        target = FIVE_C / f"figure5c_{row.case}.SE600M.h5ad"
        subset = parent[selected, :].to_memory()
        if subset.n_obs != row.start_cells + row.target_cells:
            raise ValueError(f"Cell count mismatch for {row.case}")
        if not sparse.isspmatrix_csr(subset.X):
            raise TypeError(f"{row.case} expression matrix is not sparse CSR")
        if subset.obsm["X_state"].shape != (subset.n_obs, 2058):
            raise ValueError(f"X_state mismatch for {row.case}")
        subset.write_h5ad(target, compression="gzip")
        reread = ad.read_h5ad(target, backed="r")
        if not np.array_equal(reread.obs_names, parent.obs_names[selected]):
            raise ValueError(f"Cell order mismatch for {row.case}")
        if not reread.var_names.equals(parent.var_names):
            raise ValueError(f"Gene order mismatch for {row.case}")
        if not np.array_equal(reread.obsm["X_state"], parent.obsm["X_state"][selected]):
            raise ValueError(f"X_state changed for {row.case}")
        if int(reread.obs["cell_type_merged"].astype(str).eq(row.start_label).sum()) != row.start_cells:
            raise ValueError(f"Start label mismatch for {row.case}")
        if int(reread.obs["cell_type_merged"].astype(str).eq(row.target_label).sum()) != row.target_cells:
            raise ValueError(f"Target label mismatch for {row.case}")
        rows.append({
            "case": row.case, "path": str(target), "cells": reread.n_obs,
            "genes": reread.n_vars, "embedding_dimensions": 2058,
            "start_label": row.start_label, "start_cells": row.start_cells,
            "target_label": row.target_label, "target_cells": row.target_cells,
            "bytes": target.stat().st_size, "sha256": sha256(target),
        })
        reread.file.close()
        print(f"{row.case}: {subset.n_obs} cells; {target.stat().st_size} bytes", flush=True)
        del subset

    d_rows = []
    d_labels = parent.obs["cell_type"].astype(str)
    for row in d_manifest.itertuples(index=False):
        if int(d_labels.eq(row.start_label).sum()) != row.start_cells or int(d_labels.eq(row.target_label).sum()) != row.target_cells:
            raise ValueError(f"5d labels changed for {row.case}")
        d_rows.append({
            "case": row.case, "shared_input": str(PARENT),
            "cell_col": "cell_type", "start_label": row.start_label,
            "start_cells": row.start_cells, "target_label": row.target_label,
            "target_cells": row.target_cells,
        })
    parent_hash = json.loads((ROOT / "full_embedding_61901" / "validation.json").read_text())["output_sha256"]
    FIVE_D.write_text(json.dumps({
        "source": str(PARENT), "source_sha256": parent_hash,
        "note": "Six 5d objectives reuse this single embedded h5ad; no matrix copy.",
        "objectives": d_rows,
    }, indent=2))
    manifest = {
        "source": str(PARENT), "source_sha256": parent_hash,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "five_c": rows, "five_d_shared_manifest": str(FIVE_D),
    }
    (DERIVED / "derived_inputs_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Completed six 5c extracts and 5d shared validation in {manifest['elapsed_seconds']} s", flush=True)
    parent.file.close()


if __name__ == "__main__":
    main()

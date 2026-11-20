"""Select GEO-derived Case 2/3 cells in the released A549 gene schema.

Only the feature names/order come from the published Case 1 file. Every cell
and expression value in the output comes from the independently preprocessed
GSE206741 matrix.
"""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument("--processed", type=Path, required=True)
parser.add_argument("--case1-reference", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
source = ad.read_h5ad(args.processed, backed="r")
reference = ad.read_h5ad(args.case1_reference, backed="r")
genes = list(reference.var_names)
assert len(genes) == 31854 and set(genes).issubset(set(source.var_names))
labels = ("DMSO_DMSO", "panobinostat_SRT3025", "panobinostat_Alvespimycin")
mask = source.obs["cell_type"].isin(labels).to_numpy()
counts = source.obs.loc[mask, "cell_type"].value_counts()
assert all(counts.get(label, 0) > 0 for label in labels), counts
subset = source[mask, genes].to_memory()
assert list(subset.var_names) == genes
assert np.isfinite(subset.X.data if sparse.issparse(subset.X) else subset.X).all()
args.output.parent.mkdir(parents=True, exist_ok=True)
subset.write_h5ad(args.output)
print("Wrote", args.output, "shape", subset.shape, "states", dict(counts), flush=True)

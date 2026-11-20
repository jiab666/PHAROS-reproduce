import argparse
from pathlib import Path
import anndata as ad
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
cases = {
    'pano_srt3': 'panobinostat_SRT3025',
    'pano_alve': 'panobinostat_Alvespimycin',
}
adata = ad.read_h5ad(args.input, backed='r')
assert 'X_state' in adata.obsm
assert 'cell_type' in adata.obs
for suffix, target in cases.items():
    mask = adata.obs['cell_type'].isin(['DMSO_DMSO', target]).to_numpy()
    counts = adata.obs.loc[mask, 'cell_type'].value_counts()
    assert counts.get('DMSO_DMSO', 0) > 0 and counts.get(target, 0) > 0, (target, counts)
    output = args.output_dir / f'GSE206741_qc_mad_scrublet_log1p.{suffix}.SE600M.h5ad'
    if output.exists():
        raise FileExistsError(output)
    subset = adata[mask].to_memory()
    values = np.asarray(subset.obsm['X_state'])
    assert values.shape[0] == subset.n_obs and values.ndim == 2
    assert np.isfinite(values).all(), f'Non-finite X_state in {target}'
    subset.write_h5ad(output)
    print(output, dict(counts), subset.obsm['X_state'].shape, flush=True)

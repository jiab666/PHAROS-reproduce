"""Build the raw AnnData expected by the PHAROS_reproduce CPA preprocessing script.

Uses the three original GEO GSE206741 supplementary files without altering counts.
"""
import argparse
from pathlib import Path
import gzip
import pandas as pd
import scanpy as sc
from scipy.io import mmread

parser = argparse.ArgumentParser()
parser.add_argument('--source-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
src = args.source_dir
out = args.output
out.parent.mkdir(parents=True, exist_ok=True)
obs = pd.read_csv(src / 'GSE206741_cell_metadata.tsv.gz', sep='\t', index_col=0)
var = pd.read_csv(src / 'GSE206741_gene_metadata.tsv.gz', sep='\t', index_col=0)
with gzip.open(src / 'GSE206741_count_matrix.mtx.gz', 'rb') as fh:
    matrix = mmread(fh).tocsr()
if matrix.shape == (len(var), len(obs)):
    matrix = matrix.T.tocsr()
assert matrix.shape == (len(obs), len(var)), (matrix.shape, len(obs), len(var))
var['gene_symbol'] = var['gene_short_name'].astype(str)
var['id'] = var.index.astype(str)
symbol = var['gene_symbol'].str.upper()
var['mt'] = symbol.str.startswith('MT-')
var['ribo'] = symbol.str.startswith(('RPS', 'RPL'))
var['hb'] = symbol.str.match(r'^HB(?!P|S)')
var.index = pd.Index(var['gene_short_name'].astype(str), name=None)
var.index.name = None
adata = sc.AnnData(X=matrix, obs=obs, var=var)
adata.var_names_make_unique()
adata.var.index.name = None
adata.write_h5ad(out)
print(f'Wrote {out}: {adata.shape}, nnz={adata.X.nnz}', flush=True)

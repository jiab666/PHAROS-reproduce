import argparse
import anndata as ad

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--case1-reference')
args = parser.parse_args()
adata = ad.read_h5ad(args.input, backed='r')
counts = adata.obs['cell_type'].value_counts()
for label in ('DMSO_DMSO', 'panobinostat_crizotinib', 'panobinostat_SRT3025', 'panobinostat_Alvespimycin'):
    n = int(counts.get(label, 0))
    print(label, n)
    assert n > 0, label
print('genes', adata.n_vars)
if args.case1_reference:
    ref = ad.read_h5ad(args.case1_reference, backed='r')
    labels = ('DMSO_DMSO', 'panobinostat_crizotinib')
    for label in labels:
        actual = set(adata.obs_names[adata.obs['cell_type'] == label])
        expected = set(ref.obs_names[ref.obs['cell_type'] == label])
        print('case1', label, 'actual', len(actual), 'reference', len(expected), 'overlap', len(actual & expected))
        assert actual == expected, f'QC cell mismatch for {label}: {len(actual ^ expected)} differing IDs'
    print('case1_gene_overlap', len(set(adata.var_names) & set(ref.var_names)), 'reference_genes', ref.n_vars)
    assert list(adata.var_names) == list(ref.var_names), 'QC gene list/order differs from released Case 1 reference'

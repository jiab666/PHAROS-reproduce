"""Summarize an official PHAROS hypothesis-driven run without changing scores."""
import argparse
import json
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
table = pd.read_csv(args.run_dir / 'tables/evaluation_results.tsv', sep='\t')
score = 'score_sinkhorn_ot'
baseline = table.loc[table['group'] == 'baseline', score]
explicit = table.loc[table['group'] == 'explicit_pair', score]
random = table.loc[table['group'] == 'random_pair'].copy()
assert len(baseline) > 0 and len(explicit) > 0 and len(random) > 0
assert random['random_pair_index'].notna().all()
random_means = random.groupby('random_pair_index')[score].mean()
baseline_mean = float(baseline.mean())
explicit_mean = float(explicit.mean())
better_or_tied = int((random_means <= explicit_mean).sum())
additive = pd.read_csv(args.run_dir / 'tables/explicit_pair_additive_results.tsv', sep='\t')
ordered = additive.groupby('mode')[score].mean().to_dict()
moa = table.loc[table['group'] == 'moa_pair'].groupby('pair_id')[score].mean().nsmallest(3).to_dict()
result = {
    'run_dir': str(args.run_dir),
    'baseline_mean_sinkhorn_ot': baseline_mean,
    'explicit_pair_mean_sinkhorn_ot': explicit_mean,
    'distance_reduction_fraction': (baseline_mean - explicit_mean) / baseline_mean,
    'random_pair_count': len(random_means),
    'random_pair_scores_per_pair': random.groupby('random_pair_index').size().unique().tolist(),
    'random_pairs_better_or_tied': better_or_tied,
    'explicit_rank_among_random_plus_explicit': better_or_tied + 1,
    'one_sided_empirical_p_plus_one': (better_or_tied + 1) / (len(random_means) + 1),
    'ordered_modes_mean_sinkhorn_ot': {str(k): float(v) for k, v in ordered.items()},
    'top3_moa_pairs_mean_sinkhorn_ot': {str(k): float(v) for k, v in moa.items()},
}
payload = json.dumps(result, ensure_ascii=False, indent=2)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
print(payload)

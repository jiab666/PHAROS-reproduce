#!/usr/bin/env python3
"""Validate raw runs and build the reverse/forward comparison artifacts."""
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/mnt/d/大四上/实习')
OUT = ROOT / 'pharos_reverse'
FORWARD = {
    'case1': ROOT / 'pharos_server_results/case1_full',
    'case2': ROOT / 'pharos_server_results/case2_hypothesis',
    'case3': ROOT / 'pharos_server_results/case3_hypothesis',
}

def read(run, file):
    path = run / file
    assert path.is_file() and path.stat().st_size, path
    return pd.read_csv(path, sep='\t')

def stats(case, run):
    args = json.loads((run / 'positive_control_cli_args.json').read_text())
    baseline = read(run, 'tables/baseline_results.tsv')
    ev = read(run, 'tables/evaluation_results.tsv')
    selected = read(run, 'tables/selected_pairs.tsv')
    additive = read(run, 'tables/explicit_pair_additive_results.tsv')
    random = read(run, 'tables/random_pairs.tsv')
    # Primary scores must be present and finite in every official raw row.
    for name, df, col in [('baseline', baseline, 'score_sinkhorn_ot'),
                          ('evaluation', ev, 'score_sinkhorn_ot'),
                          ('selected', selected, 'selection_score_sinkhorn_ot'),
                          ('additive', additive, 'score_sinkhorn_ot'),
                          ('random', random, 'score_sinkhorn_ot')]:
        assert len(df) and col in df and np.isfinite(pd.to_numeric(df[col], errors='coerce')).all(), (case, name)
    exp = ev[ev.group == 'explicit_pair']
    null = ev[ev.group == 'random_pair']
    assert len(exp) == args['n_batches'] and len(null) == args['n_batches'] * args['random_pairs'], (case, len(exp), len(null))
    base_mean = baseline.score_sinkhorn_ot.mean()
    exp_mean = exp.score_sinkhorn_ot.mean()
    null_means = null.groupby('random_pair_index').score_sinkhorn_ot.mean()
    assert len(null_means) == args['random_pairs']
    better_or_equal = int((null_means <= exp_mean).sum())
    row = {
        'case': case, 'direction': 'reverse' if 'pharos_reverse' in str(run) else 'forward',
        'start_cell': args['start_cell'], 'target_cell': args['target_cell'],
        'drug_a': args['two_drug_pair'][0], 'drug_b': args['two_drug_pair'][1],
        'baseline_sinkhorn': base_mean, 'explicit_pair_sinkhorn': exp_mean,
        'distance_reduction_pct': (base_mean - exp_mean) / base_mean * 100,
        'random_rank': better_or_equal + 1, 'random_total_with_explicit': len(null_means) + 1,
        'empirical_p': (better_or_equal + 1) / (len(null_means) + 1),
        'random_mean_sinkhorn': null_means.mean(), 'random_median_sinkhorn': null_means.median(),
        'explicit_order': f"{exp.iloc[0]['first_drug']} -> {exp.iloc[0]['second_drug']}",
        'n_batches': args['n_batches'], 'n_random_pairs': args['random_pairs'],
    }
    for mode in ('single_A','single_B','additive_A_plus_B','sequential_A_to_B','sequential_B_to_A'):
        subset = additive[additive['mode'] == mode]
        row[mode + '_sinkhorn'] = subset.score_sinkhorn_ot.mean() if len(subset) else math.nan
    moa = ev[ev.group == 'moa_pair'].groupby('pair_id').score_sinkhorn_ot.mean().sort_values()
    row['best_moa_pair'] = moa.index[0] if len(moa) else ''
    row['best_moa_sinkhorn'] = moa.iloc[0] if len(moa) else math.nan
    row['moa_pairs_evaluated'] = len(moa)
    return row

def resources(case):
    log = (OUT / f'logs/{case}_formal.log').read_text(errors='replace')
    gpu = pd.read_csv(OUT / f'logs/{case}_formal_gpu.csv', skipinitialspace=True)
    match = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', log)
    rss = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', log)
    exit_match = re.search(r'Exit status:\s*(\d+)', log)
    assert match and rss and exit_match and int(exit_match.group(1)) == 0
    return {'elapsed': match.group(1), 'peak_rss_mib': int(rss.group(1))/1024,
        'peak_gpu_mib': gpu['memory.used [MiB]'].str.extract(r'(\d+)')[0].astype(int).max()}

forward = [stats(case, run) for case, run in FORWARD.items()]
reverse = []
for case in FORWARD:
    run = OUT / f'raw/{case}_formal'
    row = stats(case, run)
    row.update(resources(case))
    reverse.append(row)
summary = pd.DataFrame(reverse)
summary.to_csv(OUT / 'reverse_results_summary.csv', index=False, float_format='%.6f')
comparison = pd.DataFrame(forward + reverse)
comparison.to_csv(OUT / 'forward_reverse_comparison.csv', index=False, float_format='%.6f')

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
x = np.arange(3)
for axis, field, ylabel in [(axes[0], 'distance_reduction_pct', 'Distance reduction (%)'),
                           (axes[1], 'random_rank', 'Rank among 101 pairs (lower is better)')]:
    axis.bar(x - .18, [r[field] for r in forward], width=.36, label='Forward', color='#2879b9')
    axis.bar(x + .18, [r[field] for r in reverse], width=.36, label='Reverse', color='#d88737')
    axis.set_xticks(x, ['Case 1', 'Case 2', 'Case 3'])
    axis.set_ylabel(ylabel)
    axis.grid(axis='y', alpha=.25)
    axis.set_axisbelow(True)
axes[0].axhline(0, color='black', lw=.6)
axes[0].legend(frameon=False)
fig.savefig(OUT / 'figures/forward_reverse_comparison.png', dpi=180)
plt.close(fig)
print(summary.to_string(index=False))

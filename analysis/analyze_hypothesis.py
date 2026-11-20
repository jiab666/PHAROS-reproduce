"""Offline reanalysis of the three completed PHAROS Figure 2 forward queries.

Run from the project root: python pharos_analysis/analyze_hypothesis.py
Reads only archived formal TSV/JSON outputs; never imports PHAROS or loads a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
CASES = {
    "Case 1": "case1_full",
    "Case 2": "case2_hypothesis",
    "Case 3": "case3_hypothesis",
}
MODES = ["single_A", "single_B", "additive_A_plus_B", "sequential_A_to_B", "sequential_B_to_A"]
SCORE = "score_sinkhorn_ot"


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def summary(values) -> dict:
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Missing or non-finite numeric values")
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    sem = sd / np.sqrt(len(x))
    ci = float(stats.t.ppf(0.975, len(x) - 1) * sem) if len(x) > 1 else float("nan")
    return {"n": len(x), "mean": mean, "sd_sample": sd, "min": float(x.min()),
            "max": float(x.max()), "cv_percent": 100 * sd / abs(mean) if mean else float("nan"),
            "ci95_low": mean - ci, "ci95_high": mean + ci}


def assert_batches(frame: pd.DataFrame, expected_per_batch: int, label: str) -> None:
    counts = frame.groupby("batch_index").size().to_dict()
    if counts != {i: expected_per_batch for i in range(5)}:
        raise ValueError(f"{label}: unexpected batch counts {counts}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    batches, singles, random_summary, random_scores, moa_rows, moa_batch = [], [], [], [], [], []
    variation, crosscheck, manifest = [], [], []
    plot_data = {}
    for case, folder in CASES.items():
        base = ROOT / "pharos_server_results" / folder
        table = base / "tables"
        names = ["evaluation_results.tsv", "baseline_results.tsv", "explicit_pair_additive_results.tsv",
                 "random_pairs.tsv", "selected_pairs.tsv", "batch_metadata.tsv"]
        for path in [base / "positive_control_cli_args.json", base / "positive_control_config.used.json"] + [table / n for n in names]:
            if not path.is_file():
                raise FileNotFoundError(path)
            manifest.append({"case": case, "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
        cfg = json.loads((base / "positive_control_cli_args.json").read_text(encoding="utf-8"))
        for key, value in {"seed": 42, "random_pairs": 100, "random_type": "matched", "n_batches": 5,
                           "batch_selection": "standard"}.items():
            if cfg.get(key) != value:
                raise ValueError(f"{case}: {key}={cfg.get(key)}; expected {value}")
        eva, baseline, additive, random, selected, meta = [read_table(table / n) for n in names]
        assert_batches(baseline, 1, f"{case} baseline")
        explicit = eva[eva.group.eq("explicit_pair")].copy()
        assert_batches(explicit, 1, f"{case} explicit")
        assert_batches(additive, 5, f"{case} additive")
        assert_batches(random, 100, f"{case} random")
        assert_batches(meta, 1, f"{case} metadata")
        if set(random.random_type.dropna()) != {"matched"} or random.random_pair_index.nunique() != 100:
            raise ValueError(f"{case}: random controls do not match 100 matched pairs")
        if not (random.groupby("random_pair_index").batch_index.nunique() == 5).all():
            raise ValueError(f"{case}: random pair IDs are not repeated across all five batches")
        if set(additive["mode"]) != set(MODES):
            raise ValueError(f"{case}: unexpected additive modes")
        b = baseline.set_index("batch_index")[SCORE].sort_index()
        e = explicit.set_index("batch_index")[SCORE].sort_index()
        b2 = eva[eva.group.eq("baseline")].set_index("batch_index")[SCORE].sort_index()
        if not np.allclose(b, b2, atol=1e-6):
            raise ValueError(f"{case}: baseline tables disagree")
        a = additive.pivot(index="batch_index", columns="mode", values=SCORE).sort_index()
        if not np.allclose(e, a["sequential_A_to_B"], atol=1e-3):
            raise ValueError(f"{case}: explicit and additive A->B rows disagree")
        r = random.pivot(index="random_pair_index", columns="batch_index", values=SCORE)
        if r.shape != (100, 5) or r.isna().any().any():
            raise ValueError(f"{case}: random pair matrix incomplete or duplicate")
        pair_mean = r.mean(axis=1)
        best = a[["single_A", "single_B"]].min(axis=1)
        reduction_batch = 100 * (1 - e / b)
        gain_batch = best - e
        for i in range(5):
            row = {"case": case, "batch_index": i, "seed": int(meta.set_index("batch_index").loc[i, "seed"]),
                   "baseline_ot": b.loc[i], "explicit_ot": e.loc[i],
                   "reduction_percent_per_batch": reduction_batch.loc[i],
                   "random_mean_ot": r[i].mean(), "random_median_ot": r[i].median(),
                   "random_min_ot": r[i].min(), "random_below_or_equal_explicit": int((r[i] <= e.loc[i]).sum()),
                   "first_drug_single_ot": a.loc[i, "single_A"], "panobinostat_single_ot": a.loc[i, "single_B"],
                   "best_single_ot": best.loc[i], "additive_ot": a.loc[i, "additive_A_plus_B"],
                   "forward_combo_ot": a.loc[i, "sequential_A_to_B"],
                   "reverse_combo_ot": a.loc[i, "sequential_B_to_A"],
                   "gain_vs_best_single_ot": gain_batch.loc[i],
                   "gain_vs_best_single_percent": 100 * gain_batch.loc[i] / best.loc[i]}
            batches.append(row)
            singles.append({k: row[k] for k in ["case", "batch_index", "first_drug_single_ot",
                         "panobinostat_single_ot", "best_single_ot", "additive_ot", "forward_combo_ot",
                         "reverse_combo_ot", "gain_vs_best_single_ot", "gain_vs_best_single_percent"]})
        for metric, vals in {"baseline_ot": b, "explicit_ot": e, "reduction_percent_per_batch": reduction_batch,
                             "gain_vs_best_single_ot": gain_batch}.items():
            variation.append({"case": case, "metric": metric, **summary(vals)})
        rstats = summary(pair_mean)
        counts = int((pair_mean <= e.mean()).sum())
        random_summary.append({"case": case, "statistical_unit": "unique matched pair; mean of 5 batches",
                               "random_pairs": 100, "explicit_mean_ot": e.mean(), "random_mean_ot": rstats["mean"],
                               "random_median_ot": pair_mean.median(), "random_sd_sample_ot": rstats["sd_sample"],
                               "random_p5_ot": pair_mean.quantile(.05), "random_p25_ot": pair_mean.quantile(.25),
                               "random_p75_ot": pair_mean.quantile(.75), "random_p95_ot": pair_mean.quantile(.95),
                               "random_min_ot": pair_mean.min(), "random_max_ot": pair_mean.max(),
                               "gap_vs_random_min_ot": pair_mean.min()-e.mean(),
                               "gap_vs_random_median_ot": pair_mean.median()-e.mean(),
                               "random_below_or_equal_explicit": counts, "rank_of_101": counts+1,
                               "empirical_p_plus_one": (counts+1)/101})
        for idx, val in pair_mean.items():
            pair_meta = random[random.random_pair_index.eq(idx)].iloc[0]
            random_scores.append({"case": case, "random_pair_index": int(idx), "pair_id": pair_meta.pair_id,
                                  "mean_ot_5_batches": val, "sd_sample_ot_5_batches": r.loc[idx].std(ddof=1)})
        m = eva[eva.group.eq("moa_pair")].copy()
        assert_batches(m, m.pair_id.nunique(), f"{case} mechanism")
        ranked = m.groupby("pair_id")[SCORE].mean().sort_values()
        for rank, (name, mean) in enumerate(ranked.items(), 1):
            rows = m[m.pair_id.eq(name)]
            if rows.batch_index.nunique() != 5 or len(rows) != 5:
                raise ValueError(f"{case}: incomplete mechanism candidate {name}")
            pm = rows.iloc[0]
            rank_by_batch = m.groupby("batch_index")[SCORE].rank(method="min").loc[rows.index]
            moa_rows.append({"case": case, "rank_by_mean": rank, "pair_id": name,
                             "first_drug": pm.first_drug, "second_drug": pm.second_drug,
                             "first_moa_fine": pm.first_moa_fine, "second_moa_fine": pm.second_moa_fine,
                             "mean_ot": mean, "sd_sample_ot": rows[SCORE].std(ddof=1),
                             "delta_vs_explicit_ot": mean-e.mean(), "delta_vs_best_single_ot": mean-best.mean(),
                             "delta_vs_random_median_ot": mean-pair_mean.median(),
                             "top1_batches": int((rank_by_batch == 1).sum()),
                             "top3_batches": int((rank_by_batch <= 3).sum())})
            for _, rr in rows.iterrows():
                moa_batch.append({"case": case, "pair_id": name, "batch_index": int(rr.batch_index),
                                  "score_ot": rr[SCORE],
                                  "rank_within_mechanism_batch": int(m[m.batch_index.eq(rr.batch_index)][SCORE].rank(method="min").loc[rr.name])})
        crosscheck.append({"case": case, "baseline_mean_ot": b.mean(), "explicit_mean_ot": e.mean(),
                           "reduction_ratio_of_means_percent": 100*(1-e.mean()/b.mean()),
                           "reduction_mean_of_batch_percent": reduction_batch.mean(),
                           "reduction_method_gap_pp": reduction_batch.mean()-100*(1-e.mean()/b.mean()),
                           "first_single_mean_ot": a.single_A.mean(), "panobinostat_single_mean_ot": a.single_B.mean(),
                           "best_single_mean_ot": best.mean(), "additive_mean_ot": a.additive_A_plus_B.mean(),
                           "forward_mean_ot": a.sequential_A_to_B.mean(), "reverse_mean_ot": a.sequential_B_to_A.mean(),
                           "gain_vs_best_single_mean_ot": gain_batch.mean(),
                           "gain_vs_best_single_ratio_of_means_percent": 100*gain_batch.mean()/best.mean(),
                           "gain_vs_best_single_mean_of_batch_percent": (100*gain_batch/best).mean(),
                           "batches_combo_better_than_best_single": int((gain_batch>0).sum()),
                           "mechanism_candidates": len(ranked),
                           "mechanism_better_than_explicit": int((ranked<e.mean()).sum())})
        plot_data[case] = {"baseline": b, "explicit": e, "best_single": best,
                           "random_pair_means": pair_mean, "reduction": reduction_batch,
                           "gain": gain_batch, "random_by_batch": r}
    outputs = {"batch_statistics.csv": batches, "batch_variation_summary.csv": variation,
               "single_vs_combo.csv": singles, "random_pair_statistics.csv": random_summary,
               "random_pair_scores.csv": random_scores, "mechanism_candidates.csv": moa_rows,
               "mechanism_candidate_batch.csv": moa_batch, "crosscheck_summary.csv": crosscheck,
               "source_files.csv": manifest}
    for name, data in outputs.items():
        pd.DataFrame(data).to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.10f")
    make_figures(plot_data)
    print(pd.DataFrame(crosscheck).round(5).to_string(index=False))
    print(pd.DataFrame(random_summary).round(5).to_string(index=False))


def make_figures(data: dict) -> None:
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "savefig.dpi": 160})
    colors = {"Case 1": "#4C78A8", "Case 2": "#F58518", "Case 3": "#54A24B"}
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for i, (case, d) in enumerate(data.items()):
        ax = axes[i, 0]
        x = np.arange(1, 6)
        ax.plot(x, d["baseline"].values, "o-", label="Baseline", color="#777777")
        ax.plot(x, d["best_single"].values, "s-", label="Best single", color="#E45756")
        ax.plot(x, d["explicit"].values, "^-", label="Explicit pair", color=colors[case])
        ax.set_title(f"{case}: batch OT")
        ax.set_xlabel("Batch (1–5)")
        ax.set_ylabel("Sinkhorn distance (lower is better)")
        ax.set_xticks(x)
        ax.grid(alpha=.2)
        if i == 0:
            ax.legend(fontsize=8)
        ax = axes[i, 1]
        ax.bar(x, d["gain"].values, color=[colors[case] if v > 0 else "#E45756" for v in d["gain"].values])
        ax.axhline(0, color="black", lw=.8)
        ax.set_title(f"{case}: gain over best single")
        ax.set_xlabel("Batch (1–5)")
        ax.set_ylabel("Best single OT − pair OT")
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(FIG / "batch_variation.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, (case, d) in zip(axes, data.items()):
        x = d["random_pair_means"].values
        ax.hist(x, bins=18, color=colors[case], alpha=.7, edgecolor="white")
        ax.axvline(d["explicit"].mean(), color="#C53030", lw=2, label="Explicit pair")
        ax.axvline(x.min(), color="black", lw=1, ls="--", label="Best random")
        ax.set_title(f"{case}: 100 matched pairs")
        ax.set_xlabel("Mean OT across 5 batches")
        ax.set_ylabel("Count of unique pairs")
        ax.legend(fontsize=8)
    fig.savefig(FIG / "random_ot_histograms.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for i, (case, d) in enumerate(data.items()):
        x = d["random_pair_means"].values
        ax.boxplot(x, positions=[i], widths=.5, showfliers=True, patch_artist=True,
                   boxprops={"facecolor": colors[case], "alpha": .35}, medianprops={"color": "black"})
        ax.scatter(i, d["explicit"].mean(), color="#C53030", marker="D", zorder=3,
                   label="Explicit pair" if i == 0 else None)
    ax.set_xticks(range(3), list(data))
    ax.set_ylabel("Mean Sinkhorn distance across 5 batches")
    ax.set_title("Explicit pair versus 100 matched random pairs")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.savefig(FIG / "explicit_vs_random.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for case, d in data.items():
        x = np.sort(d["random_pair_means"].values)
        ax.step(x, np.arange(1, 101)/100, where="post", color=colors[case], label=f"{case} random")
        ax.axvline(d["explicit"].mean(), color=colors[case], ls="--", alpha=.65)
    ax.set_xlabel("Mean Sinkhorn distance across 5 batches")
    ax.set_ylabel("ECDF (100 unique pairs)")
    ax.set_title("Matched random-pair distributions; dashed lines = explicit pair")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(FIG / "random_ot_ecdf.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

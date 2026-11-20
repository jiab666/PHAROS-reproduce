"""Recompute Figure 5c/5d summaries from the 12 backed-up formal runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


BASE = Path(__file__).resolve().parent
INPUT = BASE / "formal_results"
OUTPUT = BASE / "formal_analysis"
FIGURES = OUTPUT / "figures"
PATIENTS = ["Met1", "Met3", "Met4", "Met6", "Met9", "Met11"]
OBJECTIVES = ["conv0", "conv1", "conv2", "conv3", "conv6", "conv7"]
PAIR_ORDER = ["FDA_2", "FDA_4", "FDA_7", "FDA_9", "FDA_5", "FDA_8", "FDA_6", "FDA_10", "FDA_3"]
PAIR_LABELS = ["Palbociclib + Anastrozole", "Abemaciclib + Anastrozole", "Abemaciclib + Fulvestrant",
               "Capivasertib + Fulvestrant", "Palbociclib + Fulvestrant", "Alpelisib + Fulvestrant",
               "Ribociclib + Fulvestrant", "Gemcitabine + Paclitaxel", "Ribociclib + Anastrozole"]
# Transcribed from the visual Figure 5d in local PHAROS.pdf, PDF page 14.
PAPER_5D = np.array([
    [66, 59, 72, 87, 96, 89, 38, 78, 72],
    [82, 92, 87, 80, 51, 71, 13, 36, 54],
    [84, 74, 63, 64, 95, 98, 80, 87, 87],
    [88, 96, 66, 34, 22, 81, 17, 78, 66],
    [97, 87, 54, 52, 86, 73, 32, 54, 42],
    [28, 77, 34, 78, 72, 85, 62, 50, 29],
], dtype=float)
PAPER_5C_BEST_PERCENT = dict(zip(PATIENTS, [99, 100, 98, 97, 99, 98]))


def bh(pvalues: list[float]) -> np.ndarray:
    x = np.asarray(pvalues, dtype=float)
    idx = np.argsort(x)
    ranked = np.minimum.accumulate((x[idx] * len(x) / np.arange(1, len(x) + 1))[::-1])[::-1]
    out = np.empty_like(x)
    out[idx] = np.minimum(ranked, 1)
    return out


def read_case(name: str) -> tuple[pd.DataFrame, dict, dict]:
    directory = INPUT / name
    timing = json.loads((directory / "timing.json").read_text())
    check = json.loads((directory / "validation.json").read_text())
    config = json.loads((directory / "formal_config.json").read_text())
    if timing["exit_code"] != 0 or config["status"] != "complete":
        raise ValueError(f"Incomplete formal case: {name}")
    from hashlib import sha256
    path = directory / "result/tables/evaluation_results.tsv"
    if sha256(path.read_bytes()).hexdigest() != check["evaluation_sha256"]:
        raise ValueError(f"SHA-256 mismatch: {name}")
    evaluation = pd.read_csv(path, sep="\t")
    expected_batch = config["spec"]["batches"]
    expected = {"explicit_pair": 9 * expected_batch, "random_pair": 100 * expected_batch,
                "baseline": expected_batch}
    counts = evaluation.groupby("group").size().to_dict()
    if any(counts.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Unexpected formal row counts: {name}: {counts}")
    if not np.isfinite(evaluation.score_sinkhorn_ot).all():
        raise ValueError(f"Non-finite scores: {name}")
    return evaluation, timing, config


def analyze_case(name: str) -> tuple[pd.DataFrame, dict, dict]:
    evaluation, timing, config = read_case(name)
    random = evaluation[evaluation.group.eq("random_pair")]
    explicit = evaluation[evaluation.group.eq("explicit_pair")]
    baseline = evaluation[evaluation.group.eq("baseline")]
    random_values = random.score_sinkhorn_ot.to_numpy(float)
    baseline_mean = float(baseline.score_sinkhorn_ot.mean())
    records = []
    for pair, part in explicit.groupby("explicit_panel_pair_id"):
        pair_mean = float(part.score_sinkhorn_ot.mean())
        records.append({"case": name, "pair_id": pair, "first_drug": part.first_drug.iloc[0],
                        "second_drug": part.second_drug.iloc[0], "batches": len(part),
                        "mean_sinkhorn_ot": pair_mean,
                        "sd_sinkhorn_ot": float(part.score_sinkhorn_ot.std(ddof=1)),
                        "baseline_mean_sinkhorn_ot": baseline_mean,
                        "gain_vs_baseline": baseline_mean - pair_mean,
                        "percent_random_worse_or_equal": 100 * float(np.mean(random_values >= pair_mean)),
                        "n_random_rows": len(random_values),
                        "n_random_candidates": int(random.random_pair_index.nunique())})
    pairs = pd.DataFrame(records).sort_values("mean_sinkhorn_ot")
    best = pairs.iloc[0]
    best_rows = explicit.loc[explicit.explicit_panel_pair_id.eq(best.pair_id), "score_sinkhorn_ot"].to_numpy(float)
    p_raw = float(mannwhitneyu(random_values, best_rows, alternative="greater").pvalue)
    summary = {"case": name, "selection": config["spec"]["selection"],
               "batches": config["spec"]["batches"], "sample_per_state": config["spec"]["sample"],
               "baseline_mean_sinkhorn_ot": baseline_mean,
               "random_mean_sinkhorn_ot": float(random_values.mean()),
               "random_sd_sinkhorn_ot": float(random_values.std(ddof=1)),
               "best_pair_id": str(best.pair_id), "best_pair_mean_sinkhorn_ot": float(best.mean_sinkhorn_ot),
               "best_pair_percent_random_worse_or_equal": float(best.percent_random_worse_or_equal),
               "best_pair_gain_vs_baseline": float(best.gain_vs_baseline),
               "mw_one_sided_raw_p_batch_rows": p_raw,
               "n_random_rows": len(random_values), "n_random_unique_pairs": int(random.random_pair_index.nunique()),
               "elapsed_seconds": timing["elapsed_seconds"], "peak_gpu_used_mib": timing["peak_gpu_used_mib"],
               "peak_process_rss_gib": timing["peak_process_rss_kib"] / 1024**2,
               "peak_cgroup_gib": timing["peak_cgroup_current_bytes"] / 1024**3}
    return pairs, summary, timing


def heldout_heatmap_row(name: str) -> dict[str, float]:
    """Match official make_multicellular_fda_heatmap.py defaults exactly."""
    evaluation, _, _ = read_case(name)
    heldout = evaluation[evaluation.batch_index.ne(0)]
    random_means = heldout.loc[heldout.group.eq("random_pair")].groupby("pair_id").score_sinkhorn_ot.mean()
    explicit_means = heldout.loc[heldout.group.eq("explicit_pair")].groupby("pair_id").score_sinkhorn_ot.mean()
    if len(random_means) != 100 or len(explicit_means) != 9:
        raise ValueError(f"Incomplete held-out panel: {name}")
    return {pair: 100 * float(np.mean(random_means.to_numpy() >= explicit_means.loc[pair]))
            for pair in PAIR_ORDER}


def heatmap(data: np.ndarray, title: str, path: Path, symmetric: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    lim = max(10, int(np.ceil(np.abs(data).max() / 10) * 10)) if symmetric else 100
    im = ax.imshow(data, cmap="RdBu" if symmetric else "Blues", vmin=-lim if symmetric else 0, vmax=lim)
    ax.set_xticks(range(9), PAIR_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(6), OBJECTIVES)
    ax.set_title(title)
    for i in range(6):
        for j in range(9):
            value = data[i, j]
            ax.text(j, i, f"{value:+.0f}" if symmetric else f"{value:.0f}", ha="center", va="center",
                    color="white" if (not symmetric and value >= 65) or (symmetric and abs(value) > lim * .65) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="percentage points" if symmetric else "% random OT >= FDA OT")
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pair_frames = []
    summaries = []
    for name in ["5c_" + p for p in PATIENTS] + ["5d_" + d for d in OBJECTIVES]:
        pairs, summary, _ = analyze_case(name)
        pair_frames.append(pairs)
        summaries.append(summary)
    pairs = pd.concat(pair_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    c = summary[summary.case.str.startswith("5c_")].copy()
    c["patient"] = c.case.str.removeprefix("5c_")
    c["paper_best_percent"] = c.patient.map(PAPER_5C_BEST_PERCENT)
    c["best_percent_delta_points"] = c.best_pair_percent_random_worse_or_equal - c.paper_best_percent
    c["mw_one_sided_q_bh_6"] = bh(c.mw_one_sided_raw_p_batch_rows.tolist())
    c.to_csv(OUTPUT / "figure5c_statistics.csv", index=False)
    pairs.to_csv(OUTPUT / "all_fda_pairs.csv", index=False)
    summary.to_csv(OUTPUT / "task_resources.csv", index=False)
    drows = []
    ours = np.zeros((6, 9))
    for i, objective in enumerate(OBJECTIVES):
        subset = pairs[pairs.case.eq("5d_" + objective)].set_index("pair_id")
        heldout_pct = heldout_heatmap_row("5d_" + objective)
        for j, pair in enumerate(PAIR_ORDER):
            pct = heldout_pct[pair]
            ours[i, j] = pct
            drows.append({"objective": objective, "pair_id": pair, "drug_pair": PAIR_LABELS[j],
                          "reproduced_percent": pct, "paper_percent": PAPER_5D[i, j],
                          "delta_percentage_points": pct - PAPER_5D[i, j],
                          "all_batch_mean_sinkhorn_ot": float(subset.loc[pair, "mean_sinkhorn_ot"]),
                          "evaluation_batch_mode": "heldout_exclude_batch_0",
                          "random_statistical_unit": "100_pair_means"})
    pd.DataFrame(drows).to_csv(OUTPUT / "figure5d_heatmap_values.csv", index=False)
    heatmap(ours, "Figure 5d reproduced: FDA pair vs matched random", FIGURES / "figure5d_reproduced.png")
    heatmap(PAPER_5D, "Figure 5d paper (transcribed from PDF p.14)", FIGURES / "figure5d_paper.png")
    heatmap(ours - PAPER_5D, "Reproduced minus paper (percentage points)", FIGURES / "figure5d_difference.png", symmetric=True)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x = np.arange(6)
    ax.bar(x - .18, c.best_pair_percent_random_worse_or_equal, width=.36, label="Reproduced")
    ax.bar(x + .18, c.paper_best_percent, width=.36, label="Paper Figure 5c")
    ax.set_xticks(x, c.patient)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Random rows with OT >= best FDA pair (%)")
    ax.legend()
    fig.savefig(FIGURES / "figure5c_best_percent_comparison.png", dpi=250)
    plt.close(fig)
    delta = ours - PAPER_5D
    metrics = {"completed_tasks": len(summary), "total_seconds": float(summary.elapsed_seconds.sum()),
               "peak_gpu_mib": float(summary.peak_gpu_used_mib.max()),
               "peak_process_rss_gib": float(summary.peak_process_rss_gib.max()),
               "peak_cgroup_gib": float(summary.peak_cgroup_gib.max()),
               "figure5c_best_percent_mae_points": float(np.abs(c.best_percent_delta_points).mean()),
               "figure5d_mae_points": float(np.abs(delta).mean()),
               "figure5d_max_abs_delta_points": float(np.abs(delta).max()),
               "figure5d_cells_within_10_points": int(np.sum(np.abs(delta) <= 10)),
               "figure5d_cells_total": int(delta.size),
               "figure5d_spearman_all_cells": float(spearmanr(ours.ravel(), PAPER_5D.ravel()).statistic)}
    (OUTPUT / "summary_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

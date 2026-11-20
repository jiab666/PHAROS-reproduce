"""Offline attribution audit for the Figure 5d 9x6 percentile heatmap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr


HERE = Path(__file__).resolve().parent
PREP = HERE.parent
RUNS = PREP / "formal_results"
OBJECTIVES = ("conv0", "conv1", "conv2", "conv3", "conv6", "conv7")
PAIRS = ("FDA_2", "FDA_4", "FDA_7", "FDA_9", "FDA_5", "FDA_8", "FDA_6", "FDA_10", "FDA_3")
EXPECTED = {
    "conv0": ("standard", 5, 256),
    "conv1": ("high-sensitivity", 3, 256),
    "conv2": ("high-sensitivity", 3, 128),
    "conv3": ("standard", 5, 128),
    "conv6": ("high-sensitivity", 3, 128),
    "conv7": ("high-sensitivity", 3, 128),
}
VARIANTS = (
    ("official_heldout_pair_mean", "heldout", "pair_mean"),
    ("all_batches_pair_mean", "all", "pair_mean"),
    ("heldout_pooled_random_rows", "heldout", "pooled_rows"),
    ("all_batches_pooled_random_rows", "all", "pooled_rows"),
    ("heldout_pair_median", "heldout", "pair_median"),
    ("selection_batch_only_diagnostic", "batch0", "pair_mean"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(obj: str) -> tuple[pd.DataFrame, dict, dict, dict]:
    root = RUNS / f"5d_{obj}"
    config = json.loads((root / "formal_config.json").read_text())
    used = json.loads((root / "result/positive_control_config.used.json").read_text())["config"]
    cli = json.loads((root / "result/positive_control_panel_cli_args.json").read_text())
    validation = json.loads((root / "validation.json").read_text())
    path = root / "result/tables/evaluation_results.tsv"
    if sha(path) != validation["evaluation_sha256"]:
        raise ValueError(f"TSV SHA mismatch: {obj}")
    evaluation = pd.read_csv(path, sep="\t")
    if not np.isfinite(evaluation.score_sinkhorn_ot).all():
        raise ValueError(f"Nonfinite OT: {obj}")
    return evaluation, config, used, cli


def grid_for_variant(evaluations: dict[str, pd.DataFrame], batch_mode: str, aggregation: str) -> np.ndarray:
    grid = np.empty((len(OBJECTIVES), len(PAIRS)), dtype=float)
    for i, obj in enumerate(OBJECTIVES):
        d = evaluations[obj]
        if batch_mode == "heldout":
            d = d[d.batch_index.ne(0)]
        elif batch_mode == "batch0":
            d = d[d.batch_index.eq(0)]
        explicit = d[d.group.eq("explicit_pair")]
        random = d[d.group.eq("random_pair")]
        if aggregation == "pair_median":
            pair_score = explicit.groupby("pair_id").score_sinkhorn_ot.median()
            random_score = random.groupby("pair_id").score_sinkhorn_ot.median().to_numpy()
        else:
            pair_score = explicit.groupby("pair_id").score_sinkhorn_ot.mean()
            random_score = (random.score_sinkhorn_ot.to_numpy() if aggregation == "pooled_rows"
                            else random.groupby("pair_id").score_sinkhorn_ot.mean().to_numpy())
        if len(pair_score) != 9 or (aggregation != "pooled_rows" and len(random_score) != 100):
            raise ValueError(f"Incomplete pair set: {obj}, {batch_mode}, {aggregation}")
        for j, pair in enumerate(PAIRS):
            grid[i, j] = 100 * np.mean(random_score >= pair_score.loc[pair])
    return grid


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    return (float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic),
            float(kendalltau(x, y, variant="b").statistic))


def top_overlap(x: np.ndarray, y: np.ndarray, k: int = 3) -> int:
    # Stable ID order breaks ties only for top-k set reporting; rank correlations retain ties.
    return len(set(np.argsort(-x, kind="stable")[:k]) & set(np.argsort(-y, kind="stable")[:k]))


def audit_config(obj: str, d: pd.DataFrame, config: dict, used: dict, cli: dict) -> dict:
    expected_selection, expected_batches, expected_sample = EXPECTED[obj]
    spec = config["spec"]
    selected = pd.read_csv(RUNS / f"5d_{obj}/result/tables/selected_pairs.tsv", sep="\t")
    random = pd.read_csv(RUNS / f"5d_{obj}/result/tables/random_pairs.tsv", sep="\t")
    metadata = pd.read_csv(RUNS / f"5d_{obj}/result/tables/batch_metadata.tsv", sep="\t")
    observed = {
        "objective": obj,
        "start_label": spec["start"], "target_label": spec["target"], "cell_col": spec["cell_col"],
        "selection": used["batch_selection"], "batches": int(used["n_batches"]),
        "sample_per_state": int(used["start_sample"]), "random_pairs": int(used["random_pairs"]),
        "random_type": used["random_type"], "seed": int(used["seed"]),
        "batch_candidates": int(used["batch_candidates"]),
        "batch_overlap_penalty": float(used["batch_overlap_penalty"]),
        "sinkhorn_metric": cli["sinkhorn_metric"], "sinkhorn_epsilon": float(cli["sinkhorn_epsilon"]),
        "sinkhorn_iters": int(cli["sinkhorn_iters"]),
        "fda_pairs_scored": int(selected.pair_id.nunique()),
        "random_candidates_scored": int(random.random_pair_index.nunique()),
        "selection_batch_index_unique": ",".join(map(str, sorted(selected.selection_batch_index.unique()))),
        "random_selection_batch_index_unique": ",".join(map(str, sorted(random.selection_batch_index.unique()))),
        "ordered_drug_orders_min": int(selected.n_ordered_drug_orders_scored.min()),
        "ordered_drug_orders_max": int(selected.n_ordered_drug_orders_scored.max()),
        "concentration_combinations_min": int(selected.n_concentration_combinations_scored.min()),
        "concentration_combinations_max": int(selected.n_concentration_combinations_scored.max()),
        "evaluation_batch_rows": int(metadata.batch_index.nunique()),
    }
    checks = [spec["selection"] == expected_selection, observed["selection"] == expected_selection,
              spec["batches"] == expected_batches, observed["batches"] == expected_batches,
              spec["sample"] == expected_sample, observed["sample_per_state"] == expected_sample,
              observed["random_pairs"] == 100, observed["random_type"] == "matched",
              observed["seed"] == 42, observed["fda_pairs_scored"] == 9,
              observed["random_candidates_scored"] == 100,
              observed["selection_batch_index_unique"] == "0",
              observed["random_selection_batch_index_unique"] == "0",
              observed["ordered_drug_orders_min"] == 2,
              observed["evaluation_batch_rows"] == expected_batches,
              used["start_cell"] == spec["start"], used["target_cell"] == spec["target"]]
    observed["matches_published_script_and_methods"] = bool(all(checks))
    if not all(checks):
        raise ValueError(f"Configuration mismatch for {obj}: {observed}")
    return observed


def cohort_audit(config_rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    meta = pd.read_csv(PREP / "Integrated_Dataset_metadata.csv")
    immune = {"T02 CD8 Teffectormemory-GZMK", "T03 Trm-ZNF683", "T09 CD8 Texhausted-CXCL13",
              "N01 NK-CD16", "B02 B Memory"}
    malignant = meta.major_celltype.eq("Malignant")
    selected_immune = meta.minor_celltype.isin(immune) & ~(
        meta.minor_celltype.eq("T09 CD8 Texhausted-CXCL13") & meta.Disease.eq("Primary"))
    union = meta[malignant | selected_immune].copy()
    if len(union) != 61_901:
        raise ValueError("Cohort does not reproduce shared input cell count")
    union["state_label"] = np.where(union.major_celltype.eq("Malignant"), "Malignant", union.minor_celltype) + "_" + union.Disease
    union["HR_pos_HER2_neg"] = (union.ERstatus.eq("POS") | union.PRstatus.eq("POS")) & union.HER2status.eq("NEG")
    rows = []
    for cfg in config_rows:
        for role, label in (("start", cfg["start_label"]), ("target", cfg["target_label"])):
            part = union[union.state_label.eq(label)]
            h = part.HER2status.fillna("MISSING").value_counts()
            rows.append({"objective": cfg["objective"], "role": role, "label": label,
                         "all_input_cells": len(part), "unique_samples": part.Sample.nunique(),
                         "HER2_NEG": int(h.get("NEG", 0)), "HER2_POS": int(h.get("POS", 0)),
                         "HER2_EQU": int(h.get("EQU", 0)), "HER2_MISSING": int(h.get("MISSING", 0)),
                         "HR_pos_HER2_neg": int(part.HR_pos_HER2_neg.sum())})
    overall = {"union_cells": len(union), "union_samples": int(union.Sample.nunique()),
               "her2_status_counts": union.HER2status.fillna("MISSING").value_counts().to_dict(),
               "hr_pos_her2_neg_cells": int(union.HR_pos_HER2_neg.sum())}
    return pd.DataFrame(rows), overall


def selected_seed_audit() -> pd.DataFrame:
    metadata = pd.read_csv(PREP / "Integrated_Dataset_metadata.csv").set_index("CellID")
    rows = []
    for obj in OBJECTIVES:
        batches = pd.read_csv(RUNS / f"5d_{obj}/result/tables/batch_metadata.tsv", sep="\t")
        for item in batches.itertuples(index=False):
            for role in ("start", "target"):
                cell_id = getattr(item, f"{role}_seed_obs_name")
                if not isinstance(cell_id, str) or not cell_id:
                    continue
                if cell_id not in metadata.index:
                    raise ValueError(f"Selected seed absent from metadata: {cell_id}")
                row = metadata.loc[cell_id]
                rows.append({"objective": obj, "batch_index": int(item.batch_index), "role": role,
                             "cell_id": cell_id, "sample": row.Sample,
                             "HER2status": row.HER2status if pd.notna(row.HER2status) else "MISSING",
                             "ERstatus": row.ERstatus if pd.notna(row.ERstatus) else "MISSING",
                             "PRstatus": row.PRstatus if pd.notna(row.PRstatus) else "MISSING"})
    return pd.DataFrame(rows)


def main() -> None:
    reference = pd.read_csv(HERE / "figure5d_heatmap_values.csv")
    if len(reference) != 54 or reference.duplicated(["objective", "pair_id"]).any():
        raise ValueError("Expected 54 unique paper/reference cells")
    paper = reference.pivot(index="objective", columns="pair_id", values="paper_percent").loc[list(OBJECTIVES), list(PAIRS)].to_numpy()
    official_csv = pd.read_csv(HERE / "official_figure5d/fda_pair_multicellular_percentiles.csv", index_col=0)
    evaluations = {}
    configs = []
    for obj in OBJECTIVES:
        d, config, used, cli = load_run(obj)
        evaluations[obj] = d
        configs.append(audit_config(obj, d, config, used, cli))
    config_df = pd.DataFrame(configs)
    config_df.to_csv(HERE / "figure5d_config_audit.csv", index=False)
    rows = []
    matrices = {}
    for name, batches, aggregation in VARIANTS:
        result = grid_for_variant(evaluations, batches, aggregation)
        matrices[name] = result
        p, s, k = corr(paper.ravel(), result.ravel())
        rows.append({"variant": name, "batch_mode": batches, "aggregation": aggregation,
                     "mae_points": float(np.abs(result - paper).mean()),
                     "rmse_points": float(np.sqrt(np.mean((result - paper) ** 2))),
                     "pearson_all_54": p, "spearman_all_54": s, "kendall_all_54": k,
                     "mean_objective_spearman": float(np.mean([spearmanr(paper[i], result[i]).statistic for i in range(6)])),
                     "mean_top3_overlap_of_3": float(np.mean([top_overlap(paper[i], result[i]) for i in range(6)]))})
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(HERE / "figure5d_statistical_sensitivity.csv", index=False)
    ours = matrices["official_heldout_pair_mean"]
    existing = reference.pivot(index="objective", columns="pair_id", values="reproduced_percent").loc[list(OBJECTIVES), list(PAIRS)].to_numpy()
    if not np.array_equal(ours, existing):
        raise ValueError("Independent recomputation differs from existing 54 cells")
    official = official_csv.loc[:, list(PAIRS)].to_numpy()
    if not np.allclose(ours, official):
        raise ValueError("Independent recomputation differs from official heatmap")
    pair_names = reference.drop_duplicates("pair_id").set_index("pair_id").drug_pair.to_dict()
    cell_rows = []
    for i, obj in enumerate(OBJECTIVES):
        paper_ranks = rankdata(-paper[i], method="average")
        local_ranks = rankdata(-ours[i], method="average")
        for j, pair in enumerate(PAIRS):
            cell_rows.append({"objective": obj, "pair_id": pair, "drug_pair": pair_names[pair],
                              "paper_percent": paper[i, j], "local_percent": ours[i, j],
                              "signed_error_points": ours[i, j] - paper[i, j],
                              "absolute_error_points": abs(ours[i, j] - paper[i, j]),
                              "paper_rank_within_objective": paper_ranks[j],
                              "local_rank_within_objective": local_ranks[j],
                              "absolute_rank_shift": abs(local_ranks[j] - paper_ranks[j])})
    cells = pd.DataFrame(cell_rows)
    cells.to_csv(HERE / "figure5d_cell_differences.csv", index=False)
    comparison = pd.DataFrame(index=OBJECTIVES, columns=PAIRS)
    for i, obj in enumerate(OBJECTIVES):
        for j, pair in enumerate(PAIRS):
            comparison.loc[obj, pair] = f"{paper[i,j]:.0f} / {ours[i,j]:.0f} / {abs(ours[i,j]-paper[i,j]):.0f}"
    comparison.index.name = "objective (cell = paper / local / absolute error, percentage points)"
    comparison.to_csv(HERE / "figure5d_9x6_comparison.csv")
    obj_rows = []
    for i, obj in enumerate(OBJECTIVES):
        p, s, k = corr(paper[i], ours[i])
        err = np.abs(ours[i] - paper[i])
        obj_rows.append({"objective": obj, "mae_points": err.mean(), "median_absolute_error_points": np.median(err),
                         "max_absolute_error_points": err.max(), "pearson": p, "spearman_rank": s,
                         "kendall_tau_b": k, "top3_overlap_of_3": top_overlap(paper[i], ours[i]),
                         "mean_absolute_rank_shift": cells[cells.objective.eq(obj)].absolute_rank_shift.mean()})
    pd.DataFrame(obj_rows).to_csv(HERE / "figure5d_objective_errors.csv", index=False)
    pair_rows = []
    for j, pair in enumerate(PAIRS):
        p, s, k = corr(paper[:, j], ours[:, j])
        err = np.abs(ours[:, j] - paper[:, j])
        pair_rows.append({"pair_id": pair, "drug_pair": pair_names[pair], "mae_points": err.mean(),
                          "median_absolute_error_points": np.median(err), "max_absolute_error_points": err.max(),
                          "pearson": p, "spearman_rank_across_objectives": s, "kendall_tau_b": k,
                          "top3_objective_overlap_of_3": top_overlap(paper[:, j], ours[:, j])})
    pd.DataFrame(pair_rows).to_csv(HERE / "figure5d_pair_errors.csv", index=False)
    cohort, overall = cohort_audit(configs)
    cohort.to_csv(HERE / "figure5d_input_cohort_audit.csv", index=False)
    selected_seed_audit().to_csv(HERE / "figure5d_selected_seed_audit.csv", index=False)
    audit = {"paper_pdf_sha256": sha(HERE.parent.parent / "PHAROS.pdf"),
             "paper_numbers_csv_sha256": sha(HERE / "figure5d_heatmap_values.csv"),
             "official_matrix_max_abs_difference": float(np.abs(ours - official).max()),
             "overall": overall,
             "largest_errors": cells.nlargest(10, "absolute_error_points").to_dict("records")}
    (HERE / "figure5d_discrepancy_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(sensitivity.to_string(index=False))
    print("COHORT", overall)


if __name__ == "__main__":
    main()

"""Offline comparison of Figure 3 S5 and Figure 3i official result tables."""
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent
CASES = [
    ("Givinostat_Carmofur", "givino_carmf", "fig3c_givino_carmf"),
    ("Alvespimycin_Pirarubicin", "alves_pira", "fig3b_alves_pira_retry"),
    ("Cediranib_PCI-34501", "cedi_pci", "fig3d_cedi_pci"),
]


def summarize(root):
    t = pd.read_csv(root / "tables/evaluation_results.tsv", sep="\t")
    base = t.loc[t.group == "baseline", "score_sinkhorn_ot"].mean()
    pair = t.loc[t.group == "explicit_pair", "score_sinkhorn_ot"].mean()
    random = t.loc[t.group == "random_pair"].groupby("random_pair_index")["score_sinkhorn_ot"].mean()
    assert len(random) == 100
    return base, pair, 100 * (base - pair) / base, 1 + int((random < pair).sum())


rows, counts = [], []
for target, key, old_name in CASES:
    old_root = OLD / "runs" / old_name
    new_root = ROOT / "s5_full_embedding" / key
    old = summarize(old_root)
    new = summarize(new_root)
    old_args = json.loads((old_root / "positive_control_cli_args.json").read_text())
    new_args = json.loads((new_root / "positive_control_cli_args.json").read_text())
    old_batch = pd.read_csv(old_root / "tables/batch_metadata.tsv", sep="\t").iloc[0]
    new_batch = pd.read_csv(new_root / "tables/batch_metadata.tsv", sep="\t").iloc[0]
    old_random = pd.read_csv(old_root / "tables/random_pairs.tsv", sep="\t")
    new_random = pd.read_csv(new_root / "tables/random_pairs.tsv", sep="\t")
    identity_cols = ["random_pair_index", "first_drug_norm", "second_drug_norm"]
    dose_cols = ["random_pair_index", "first_dose", "second_dose"]
    same_random_drugs = old_random[identity_cols].equals(new_random[identity_cols])
    same_random_doses = old_random[dose_cols].equals(new_random[dose_cols])
    old_drug_set = set(map(tuple, old_random[identity_cols].drop_duplicates().to_numpy()))
    new_drug_set = set(map(tuple, new_random[identity_cols].drop_duplicates().to_numpy()))
    assert old_args["target_cell"] == new_args["target_cell"] == target
    assert old_args["projection_method"] == "pca_pls_da" and new_args["projection_method"] == "none"
    assert old_args["random_pairs"] == new_args["random_pairs"] == 100
    assert old_args["n_batches"] == new_args["n_batches"] == 5
    rows.append({
        "target": target,
        "projected_baseline_ot": old[0], "projected_pair_ot": old[1],
        "projected_reduction_pct": old[2], "projected_rank_of_101": old[3],
        "full2058_baseline_ot": new[0], "full2058_pair_ot": new[1],
        "full2058_reduction_pct": new[2], "full2058_rank_of_101": new[3],
        "reduction_change_pp": new[2] - old[2], "rank_change": new[3] - old[3],
        "projected_start_available": int(old_batch.start_n_available),
        "full2058_start_available": int(new_batch.start_n_available),
        "projected_target_available": int(old_batch.target_n_available),
        "full2058_target_available": int(new_batch.target_n_available),
        "same_random_drug_rows": same_random_drugs,
        "same_random_selected_doses": same_random_doses,
        "random_drug_identities_shared": len(old_drug_set & new_drug_set),
        "projected_random_drug_identities": len(old_drug_set),
        "full2058_random_drug_identities": len(new_drug_set),
    })
    search = ROOT / "figure3i_open_search" / key
    t = pd.read_csv(search / "search/results.tsv", sep="\t")
    for depth in (1, 2):
        paths = t.loc[t.depth == depth, "drug_names_json"].map(json.loads)
        pano = paths.map(lambda p: any("panobinostat" in x.strip().casefold() for x in p))
        crizo = paths.map(lambda p: any("crizotinib" in x.strip().casefold() for x in p))
        counts.append({"target": target, "depth": depth, "retained_paths": len(paths),
                       "paths_with_panobinostat": int(pano.sum()),
                       "paths_with_crizotinib": int(crizo.sum()),
                       "paths_with_both": int((pano & crizo).sum())})

pd.DataFrame(rows).to_csv(ROOT / "s5_projection_comparison.csv", index=False)
pd.DataFrame(counts).to_csv(ROOT / "figure3i_drug_occurrences.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(counts).to_string(index=False))

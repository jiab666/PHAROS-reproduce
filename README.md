# PHAROS reproduction scripts

Project-authored orchestration and offline analysis code for selected PHAROS experiments. The PHAROS and STATE algorithms, model weights, cell data, and large result files are **not** included. The scripts were used with the official implementations and retain the paths of the original `/srv/pharos` server layout; adapt those paths for another machine.

## Upstream versions

| Dependency | Pinned commit |
|---|---|
| [PHAROS](https://github.com/jbezney61/PHAROS) | `7435bbe2a34cdec628111f13d0a40f0bf886bb91` |
| [PHAROS_reproduce](https://github.com/jbezney61/PHAROS_reproduce) | `3ad97ae0f5cb9acde5bcea17fb14fc29bca0cc3c` |
| [STATE](https://github.com/ArcInstitute/state) | `bf4fbc9ea35bf4d1e91afe201b663dec5d8bdd48` |

The environment exports are in [`environment/`](environment/). Install the official projects at the pinned commits before using these scripts. The full workflow requires the ST-SE-Tahoe checkpoint and, when generating new `X_state` embeddings, the official SE-600M checkpoint and protein embeddings. Obtain data and weights from the official releases; do not commit them to Git.

## Code layout

- [`scripts/figure2/`](scripts/figure2/): A549 GEO preprocessing, embedding, hypothesis-driven and Open Search orchestration, plus reverse state transition.
- [`scripts/figure3/`](scripts/figure3/): missing A549 state inputs, separation, nine Figure 3/S4 pair runs, Figure 3i and S5.
- [`scripts/figure5/`](scripts/figure5/): shared breast-cancer preprocessing/embedding, 5c/5d panel evaluation, and six S12 searches.
- [`scripts/figure6/`](scripts/figure6/): BCC subset, embedding, separation and searches.
- [`analysis/`](analysis/): offline result validation and comparisons.
- [`metadata/FDA_drug_pairs_9.csv`](metadata/FDA_drug_pairs_9.csv): reconstructed public nine-pair FDA panel. The authors' internal panel file was not published, so exact identity is not established.

Run scripts against the original directory layout or update their `ROOT`, `MODEL`, and metadata paths. Scripts that launch experiments may require an NVIDIA GPU and substantial RAM. Output directories and existing results should be checked before launching; some historical shell wrappers contain `--overwrite` because they were originally run in fresh directories.

## Scope and cohort differences

The scripts reproduce selected Figure 2/3/5/6 computations; they do not claim biological validation. Figure 5/S12 patient inputs were extracted from a shared 61,901-cell `X_state` file. The official S12 filenames say `HER2neg`, while the public metadata records Met6 as HER2-positive and Met9 as equivocal; `Primary_2` has no HER2 value in the import audit. The original author-specific patient subset therefore cannot be confirmed from these public inputs. Keep this distinction when comparing to the paper.

Generated `*.h5ad`, checkpoints, logs, search results and figures are intentionally excluded from this code repository. Preserve and verify them separately with the SHA-256 manifests from the experiment archive.

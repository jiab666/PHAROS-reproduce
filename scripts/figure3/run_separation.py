"""Run official PHAROS separation on four independently paired A549 states."""

import json
from pathlib import Path
import subprocess
import time


BASE = Path("/srv/pharos/figure3_a549")
PHAROS = Path("/usr/local/miniconda3/envs/PHAROS/bin/pharos")
CASES = [
    ("Givinostat_crizotinib", 1300),
    ("Alvespimycin_Pirarubicin", 470),
    ("Cediranib_PCI-34501", 1300),
    ("Givinostat_Carmofur", 1300),
]


def main():
    for label, n_cells in CASES:
        adata = BASE / "pair_inputs" / f"{label}.SE600M.h5ad"
        output = BASE / "separation" / label
        log = BASE / "logs" / f"separation_{label}.log"
        timing = BASE / "logs" / f"separation_{label}.json"
        if not adata.exists() or output.exists() or log.exists() or timing.exists():
            raise RuntimeError(f"Missing input or existing output: {label}")
        cmd = [
            str(PHAROS), "admissibility", "separation", "--adata", str(adata),
            "--cell-col", "cell_type", "--cells-per-line", str(n_cells),
            "--umap-n-neighbors", "30", "--device", "cpu", "--output-dir", str(output),
        ]
        start = time.monotonic()
        with log.open("w") as file:
            code = subprocess.run(cmd, stdout=file, stderr=subprocess.STDOUT).returncode
        data = {
            "label": label, "command": cmd, "exit_code": code,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "summary_exists": (output / "summary.md").is_file(),
            "qc_exists": (output / "tables/cell_line_qc.tsv").is_file(),
            "umap_exists": (output / "figures/01_umap_by_cell_line.png").is_file(),
        }
        timing.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(json.dumps(data), flush=True)
        if code or not all(data[k] for k in ("summary_exists", "qc_exists", "umap_exists")):
            raise SystemExit(code or 1)


if __name__ == "__main__":
    main()

"""Run the 12 official Figure 5 panel configurations sequentially, with checks."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

import numpy as np
import pandas as pd

from run_figure5_panel_smoke import sample_resources


ROOT = Path("/srv/pharos/work/figure5")
FORMAL = ROOT / "formal_panels"
MODEL = Path("/srv/pharos/input/ST-SE-Tahoe/fewshot/state_generalization_X_state")
META = Path("/srv/pharos/code/PHAROS_reproduce/metadata/drug_metadata_sciplex.csv")
PAIRS = ROOT / "FDA_drug_pairs_9.csv"
PHAROS = Path("/usr/local/miniconda3/envs/PHAROS/bin/pharos")
PYTHON = Path("/usr/local/miniconda3/envs/PHAROS/bin/python")
ORDER_5C = ["Met1", "Met4", "Met6", "Met9", "Met11", "Met3"]
ORDER_5D = ["conv0", "conv1", "conv2", "conv3", "conv6", "conv7"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cases() -> list[dict]:
    five_c = json.loads((ROOT / "derived_inputs/derived_inputs_manifest.json").read_text())
    five_d = json.loads((ROOT / "derived_inputs/figure5d_shared_input.json").read_text())
    c_lookup = {row["case"]: row for row in five_c["five_c"]}
    d_lookup = {row["case"]: row for row in five_d["objectives"]}
    result = []
    for case in ORDER_5C:
        row = c_lookup[case]
        high = case == "Met3"
        result.append({"name": "5c_" + case, "adata": row["path"], "start": row["start_label"],
                       "target": row["target_label"], "cell_col": "cell_type_merged",
                       "selection": "high-sensitivity" if high else "standard", "batches": 3 if high else 5,
                       "sample": 256})
    for case in ORDER_5D:
        row = d_lookup[case]
        high = case in {"conv1", "conv2", "conv6", "conv7"}
        result.append({"name": "5d_" + case, "adata": row["shared_input"], "start": row["start_label"],
                       "target": row["target_label"], "cell_col": "cell_type",
                       "selection": "high-sensitivity" if high else "standard", "batches": 3 if high else 5,
                       "sample": 256 if case in {"conv0", "conv1"} else 128})
    return result


def validate(run: Path, spec: dict) -> dict:
    table = pd.read_csv(run / "result/tables/evaluation_results.tsv", sep="\t")
    required = {"baseline": spec["batches"], "explicit_pair": 9 * spec["batches"],
                "random_pair": 100 * spec["batches"]}
    counts = table.groupby("group").size().to_dict()
    for group, expected in required.items():
        if counts.get(group) != expected:
            raise ValueError(f"{group}: {counts.get(group)} != {expected}")
    if sorted(table.batch_index.unique()) != list(range(spec["batches"])):
        raise ValueError("Batch indices incomplete")
    if table.loc[table.group.eq("explicit_pair"), "explicit_panel_pair_id"].nunique() != 9:
        raise ValueError("FDA panel incomplete")
    if table.loc[table.group.eq("random_pair"), "random_pair_index"].nunique() != 100:
        raise ValueError("100 random candidates incomplete")
    if not np.isfinite(table.score_sinkhorn_ot.to_numpy()).all():
        raise ValueError("Non-finite Sinkhorn value")
    metadata = pd.read_csv(run / "result/tables/batch_metadata.tsv", sep="\t")
    return {"group_counts": counts, "rows": len(table), "batch_metadata_rows": len(metadata),
            "evaluation_sha256": digest(run / "result/tables/evaluation_results.tsv"),
            "selected_pairs_sha256": digest(run / "result/tables/selected_pairs.tsv"),
            "random_pairs_sha256": digest(run / "result/tables/random_pairs.tsv")}


def run_case(spec: dict, journal) -> dict:
    run = FORMAL / spec["name"]
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite: {run}")
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise RuntimeError("Less than 15 GiB free; stop before new case")
    run.mkdir(parents=True)
    output = run / "result"
    command = [str(PYTHON), "-u", str(PHAROS), "hypothesis-driven", "panel",
               "--adata", spec["adata"], "--start-cell", spec["start"],
               "--target-cell", spec["target"], "--cell-col", spec["cell_col"],
               "--model-dir", str(MODEL), "--approved-pairs-file", str(PAIRS),
               "--output-dir", str(output), "--device", "cuda:0", "--drug-metadata", str(META),
               "--random-pairs", "100", "--random-type", "matched", "--batch", str(spec["batches"]),
               "--seed", "42", "--converter-chunk-size", "16"]
    if spec["sample"] == 128:
        command += ["--start-sample", "128", "--target-sample", "128"]
    if spec["selection"] == "high-sensitivity":
        command += ["--batch-selection", "high-sensitivity", "--batch-candidates", "1000"]
    configuration = {"spec": spec, "command": command, "status": "running", "started_unix": time.time()}
    (run / "formal_config.json").write_text(json.dumps(configuration, indent=2))
    journal.write(f"START {spec['name']} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    journal.flush()
    start = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    stop = threading.Event()
    worker = threading.Thread(target=sample_resources, args=(process.pid, stop, start, run / "resources_0p5s.csv"), daemon=True)
    worker.start()
    stages = {}
    buffer = ""
    try:
        with (run / "panel.log").open("wb") as log:
            while chunk := os.read(process.stdout.fileno(), 4096):
                log.write(chunk)
                log.flush()
                buffer += chunk.decode("utf-8", errors="replace")
                parts = re.split(r"[\r\n]", buffer)
                buffer = parts.pop()
                for line in parts:
                    match = re.search(r"\[([0-7])/8\]", line)
                    if match:
                        key = f"stage_{match.group(1)}"
                        if key not in stages:
                            stages[key] = round(time.perf_counter() - start, 3)
            exit_code = process.wait()
    finally:
        stop.set()
        worker.join(timeout=10)
    elapsed = time.perf_counter() - start
    with (run / "resources_0p5s.csv").open(newline="") as f:
        samples = list(csv.DictReader(f))
    def numeric(col):
        values = []
        for row in samples:
            try:
                values.append(float(row[col]))
            except (KeyError, ValueError, TypeError):
                pass
        return values
    timing = {"case": spec["name"], "exit_code": exit_code, "elapsed_seconds": round(elapsed, 3),
              "stage_start_seconds": stages, "sample_count": len(samples),
              "peak_gpu_used_mib": max(numeric("gpu_used_mib"), default=None),
              "peak_process_rss_kib": max(numeric("process_rss_kib"), default=None),
              "peak_cgroup_current_bytes": max(numeric("cgroup_current_bytes"), default=None),
              "min_mem_available_kib": min(numeric("mem_available_kib"), default=None),
              "min_disk_free_bytes": min(numeric("disk_free_bytes"), default=None)}
    (run / "timing.json").write_text(json.dumps(timing, indent=2))
    if exit_code != 0:
        configuration["status"] = "failed"
        (run / "formal_config.json").write_text(json.dumps(configuration, indent=2))
        raise RuntimeError(f"{spec['name']} exited {exit_code}; see {run / 'panel.log'}")
    checks = validate(run, spec)
    (run / "validation.json").write_text(json.dumps(checks, indent=2))
    configuration["status"] = "complete"
    configuration["finished_unix"] = time.time()
    (run / "formal_config.json").write_text(json.dumps(configuration, indent=2))
    journal.write(f"DONE {spec['name']} {elapsed:.1f}s GPU={timing['peak_gpu_used_mib']}MiB RSS={timing['peak_process_rss_kib']}KiB\n")
    journal.flush()
    return {"spec": spec, "timing": timing, "validation": checks, "directory": str(run)}


def main() -> None:
    if FORMAL.exists():
        raise FileExistsError(f"Formal root already exists: {FORMAL}")
    for path in [PAIRS, META, MODEL / "checkpoints/final.ckpt", ROOT / "derived_inputs/derived_inputs_manifest.json"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if digest(PAIRS) != "2c6ef91069c9ba1268404629089d6f659b21e02a351942de4f2f0830389e972c":
        raise ValueError("FDA pair panel SHA-256 changed")
    specifications = cases()
    for spec in specifications:
        if not Path(spec["adata"]).is_file():
            raise FileNotFoundError(spec["adata"])
    FORMAL.mkdir(parents=True)
    (FORMAL / "run_manifest.json").write_text(json.dumps(specifications, indent=2))
    completed = []
    with (FORMAL / "orchestrator.log").open("w") as journal:
        try:
            for spec in specifications:
                completed.append(run_case(spec, journal))
        except Exception as exc:
            journal.write(f"HALT {type(exc).__name__}: {exc}\n")
            journal.flush()
            (FORMAL / "completed.json").write_text(json.dumps(completed, indent=2))
            raise
    (FORMAL / "completed.json").write_text(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()

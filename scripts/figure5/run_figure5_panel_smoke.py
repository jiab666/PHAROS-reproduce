"""Run one official Figure 5 panel smoke test with 0.5 s resource sampling."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time


ROOT = Path("/srv/pharos/work/figure5")
MODEL = Path("/srv/pharos/input/ST-SE-Tahoe/fewshot/state_generalization_X_state")
META = Path("/srv/pharos/code/PHAROS_reproduce/metadata/drug_metadata_sciplex.csv")
PAIRS = ROOT / "FDA_drug_pairs_9.csv"
PYTHON = Path("/usr/local/miniconda3/envs/PHAROS/bin/python")
PHAROS = Path("/usr/local/miniconda3/envs/PHAROS/bin/pharos")
CASES = {
    "met1": {
        "adata": ROOT / "derived_inputs/figure5c/figure5c_Met1.SE600M.h5ad",
        "start": "Malignant_Metastasis_Metastasis_1",
        "target": "Malignant_Primary_Primary_2",
        "cell_col": "cell_type_merged",
        "batch": 5,
        "selection": "standard",
    },
    "conv1": {
        "adata": ROOT / "full_embedding_61901/figure5_union.SE600M.h5ad",
        "start": "T02 CD8 Teffectormemory-GZMK_Metastasis",
        "target": "T02 CD8 Teffectormemory-GZMK_Primary",
        "cell_col": "cell_type",
        "batch": 3,
        "selection": "high-sensitivity",
    },
}


def read_rss_kib(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return None
    match = re.search(r"^VmRSS:\s+(\d+) kB$", text, re.M)
    return int(match.group(1)) if match else None


def mem_available_kib() -> int:
    return int(re.search(r"^MemAvailable:\s+(\d+) kB$", Path("/proc/meminfo").read_text(), re.M).group(1))


def gpu_used_mib() -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=5,
    )
    return int(result.stdout.strip().splitlines()[0]) if result.returncode == 0 else None


def sample_resources(pid: int, stop: threading.Event, started: float, output: Path) -> None:
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["elapsed_s", "gpu_used_mib", "process_rss_kib", "mem_available_kib", "cgroup_current_bytes", "disk_free_bytes"])
        while not stop.is_set():
            tick = time.perf_counter()
            try:
                writer.writerow([
                    round(tick - started, 3), gpu_used_mib(), read_rss_kib(pid),
                    mem_available_kib(), int(Path("/sys/fs/cgroup/memory.current").read_text()),
                    shutil.disk_usage(ROOT).free,
                ])
            except Exception as exc:
                writer.writerow([round(tick - started, 3), f"error:{exc}"])
            stream.flush()
            stop.wait(max(0, 0.5 - (time.perf_counter() - tick)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    args = parser.parse_args()
    config = CASES[args.case]
    run = ROOT / "smoke_panels" / f"{args.case}_{config['selection'].replace('-', '_')}_r3"
    run.mkdir(parents=True, exist_ok=False)
    output_dir = run / "result"
    for path in (config["adata"], PAIRS, META, MODEL / "checkpoints/final.ckpt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    command = [
        str(PYTHON), "-u", str(PHAROS), "hypothesis-driven", "panel",
        "--adata", str(config["adata"]),
        "--start-cell", config["start"], "--target-cell", config["target"],
        "--cell-col", config["cell_col"],
        "--model-dir", str(MODEL), "--approved-pairs-file", str(PAIRS),
        "--output-dir", str(output_dir), "--device", "cuda:0",
        "--drug-metadata", str(META),
        "--random-pairs", "3", "--random-type", "matched",
        "--batch", str(config["batch"]), "--seed", "42",
    ]
    if config["selection"] == "high-sensitivity":
        command.extend(["--batch-selection", "high-sensitivity"])
    started = time.perf_counter()
    print("Running:", " ".join(command), flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    stop = threading.Event()
    worker = threading.Thread(target=sample_resources, args=(process.pid, stop, started, run / "resources_0p5s.csv"), daemon=True)
    worker.start()
    stages: dict[str, float] = {}
    buffer = ""
    with (run / "panel.log").open("wb") as stream:
        while chunk := os.read(process.stdout.fileno(), 4096):
            stream.write(chunk)
            stream.flush()
            buffer += chunk.decode("utf-8", errors="replace")
            parts = re.split(r"[\r\n]", buffer)
            buffer = parts.pop()
            for line in parts:
                match = re.search(r"\[([0-7])/8\]", line)
                if match:
                    stage = f"stage_{match.group(1)}"
                    if stage not in stages:
                        stages[stage] = round(time.perf_counter() - started, 3)
                        print(f"{stage}: {stages[stage]}s", flush=True)
                if "=== Explicit pair panel workflow complete ===" in line:
                    stages["workflow_complete"] = round(time.perf_counter() - started, 3)
    exit_code = process.wait()
    elapsed = time.perf_counter() - started
    stop.set()
    worker.join(timeout=10)
    with (run / "resources_0p5s.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    def values(column: str) -> list[float]:
        result = []
        for row in rows:
            try:
                result.append(float(row[column]))
            except (TypeError, ValueError):
                pass
        return result
    result = {
        "case": args.case, "command": command, "exit_code": exit_code,
        "elapsed_seconds": round(elapsed, 3), "stage_start_seconds": stages,
        "resource_samples": len(rows),
        "peak_gpu_used_mib": max(values("gpu_used_mib"), default=None),
        "peak_process_rss_kib": max(values("process_rss_kib"), default=None),
        "minimum_mem_available_kib": min(values("mem_available_kib"), default=None),
        "peak_cgroup_current_bytes": max(values("cgroup_current_bytes"), default=None),
        "minimum_disk_free_bytes": min(values("disk_free_bytes"), default=None),
        "output_dir": str(output_dir),
    }
    (run / "timing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("case", "exit_code", "elapsed_seconds", "peak_gpu_used_mib", "peak_process_rss_kib")}, indent=2), flush=True)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

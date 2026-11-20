"""Run nine unchanged PHAROS Figure 3/S4 pair jobs sequentially."""

import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


BASE = Path("/srv/pharos/figure3_a549")
PHAROS = Path("/usr/local/miniconda3/envs/PHAROS/bin/pharos")
MODEL = Path("/srv/pharos/input/ST-SE-Tahoe/fewshot/state_generalization_X_state")
JOBS = [
    ("fig3b_alves_pira_retry", "Alvespimycin_Pirarubicin", "crizotinib"),
    ("fig3c_givino_carmf", "Givinostat_Carmofur", "crizotinib"),
    ("fig3d_cedi_pci", "Cediranib_PCI-34501", "crizotinib"),
    ("s4_case2_alves_pira", "Alvespimycin_Pirarubicin", "Resveratrol"),
    ("s4_case2_cedi_pci", "Cediranib_PCI-34501", "Resveratrol"),
    ("s4_case2_givino_carmf", "Givinostat_Carmofur", "Resveratrol"),
    ("s4_case3_alves_pira", "Alvespimycin_Pirarubicin", "Pimitespib"),
    ("s4_case3_cedi_pci", "Cediranib_PCI-34501", "Pimitespib"),
    ("s4_case3_givino_carmf", "Givinostat_Carmofur", "Pimitespib"),
]


def gpu_mib():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=5,
    )
    return int(result.stdout.strip().splitlines()[0]) if result.returncode == 0 else None


def rss_kib(pid):
    try:
        match = re.search(r"^VmRSS:\s+(\d+) kB$", Path(f"/proc/{pid}/status").read_text(), re.MULTILINE)
        return int(match.group(1)) if match else None
    except FileNotFoundError:
        return None


def verify(output, label, drug):
    required = [
        "positive_control_cli_args.json", "tables/baseline_results.tsv",
        "tables/evaluation_results.tsv", "tables/random_pairs.tsv",
        "tables/explicit_pair_additive_results.tsv", "tables/batch_metadata.tsv",
    ]
    missing = [item for item in required if not (output / item).is_file()]
    if missing:
        return {"valid": False, "missing": missing}
    args = json.loads((output / "positive_control_cli_args.json").read_text())
    config_ok = (
        args.get("start_cell") == "DMSO_DMSO"
        and args.get("target_cell") == label
        and args.get("two_drug_pair") == [drug, "Panobinostat"]
        and args.get("random_pairs") == 100
        and args.get("random_type") == "matched"
        and args.get("n_batches") == 5
    )
    with (output / "tables/random_pairs.tsv").open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    unique_random = len({r["random_pair_index"] for r in rows})
    batches = sorted({int(r["batch_index"]) for r in rows})
    valid = config_ok and unique_random == 100 and batches == list(range(5))
    return {"valid": valid, "missing": missing, "config_ok": config_ok,
            "random_pair_rows": len(rows), "unique_random_pairs": unique_random,
            "random_batches": batches}


def main():
    for job_id, label, drug in JOBS:
        adata = BASE / "pair_inputs" / f"{label}.SE600M.h5ad"
        output = BASE / "runs" / job_id
        log = BASE / "logs" / f"{job_id}.log"
        monitor = BASE / "logs" / f"{job_id}_resources_0p5s.csv"
        status_file = BASE / "logs" / f"{job_id}_status.json"
        if shutil.disk_usage(BASE).free < 15 * 1024**3:
            raise RuntimeError("Disk free below 15 GiB; stop before next job")
        if not adata.is_file() or any(p.exists() for p in (output, log, monitor, status_file)):
            raise RuntimeError(f"Input missing or output exists: {job_id}")
        cmd = [
            str(PHAROS), "hypothesis-driven", "pair",
            "--adata", str(adata), "--start-cell", "DMSO_DMSO",
            "--target-cell", label, "--cell-col", "cell_type",
            "--model-dir", str(MODEL), "--output-dir", str(output),
            "--drug-pair", drug, "Panobinostat",
            "--moa-pairs", "Multi-TK inhibitor", "HDAC inhibitor",
            "--device", "cuda:0",
        ]
        start = time.monotonic()
        print(f"START {job_id}", flush=True)
        with log.open("wb") as log_file, monitor.open("w", newline="") as monitor_file:
            writer = csv.writer(monitor_file)
            writer.writerow(["elapsed_s", "gpu_used_mib", "process_rss_kib", "cgroup_current_bytes", "disk_free_bytes"])
            proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT,
                cwd="/srv/pharos/code/PHAROS_reproduce",
            )
            while proc.poll() is None:
                tick = time.monotonic()
                writer.writerow([
                    round(tick - start, 3), gpu_mib(), rss_kib(proc.pid),
                    int(Path("/sys/fs/cgroup/memory.current").read_text()),
                    shutil.disk_usage(BASE).free,
                ])
                monitor_file.flush()
                time.sleep(max(0, 0.5 - (time.monotonic() - tick)))
            code = proc.wait()
        rows = list(csv.DictReader(monitor.open()))
        def peak(key):
            vals = [float(r[key]) for r in rows if r[key] not in ("", None)]
            return max(vals) if vals else None
        validation = verify(output, label, drug) if code == 0 else {"valid": False}
        result = {
            "job_id": job_id, "command": cmd, "exit_code": code,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "peak_gpu_used_mib": peak("gpu_used_mib"),
            "peak_process_rss_kib": peak("process_rss_kib"),
            "peak_cgroup_current_bytes": peak("cgroup_current_bytes"),
            "minimum_disk_free_bytes": min(float(r["disk_free_bytes"]) for r in rows),
            "validation": validation,
        }
        status_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"END {job_id}: exit={code}, valid={validation['valid']}, elapsed={result['elapsed_seconds']:.1f}s", flush=True)
        if code or not validation["valid"]:
            raise SystemExit(code or 1)


if __name__ == "__main__":
    main()

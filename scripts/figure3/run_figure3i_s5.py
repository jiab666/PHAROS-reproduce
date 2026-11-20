"""Run official Figure 3i searches, pair reports, then S5 ablations."""

import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


BASE = Path("/srv/pharos/figure3_a549/figure3i_s5")
INPUTS = Path("/srv/pharos/figure3_a549/pair_inputs")
MODEL = Path("/srv/pharos/input/ST-SE-Tahoe/fewshot/state_generalization_X_state")
REPRO = Path("/srv/pharos/code/PHAROS_reproduce")
PHAROS = Path("/usr/local/miniconda3/envs/PHAROS/bin/pharos")
TARGETS = [
    ("givino_carmf", "Givinostat_Carmofur"),
    ("alves_pira", "Alvespimycin_Pirarubicin"),
    ("cedi_pci", "Cediranib_PCI-34501"),
]


def gpu_mib():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=8,
    )
    return int(result.stdout.strip().splitlines()[0]) if result.returncode == 0 and result.stdout.strip() else None


def rss_kib(pid):
    try:
        match = re.search(r"^VmRSS:\s+(\d+) kB$", Path(f"/proc/{pid}/status").read_text(), re.MULTILINE)
        return int(match.group(1)) if match else None
    except FileNotFoundError:
        return None


def run(name, cmd):
    log = BASE / "logs" / f"{name}.log"
    resources = BASE / "logs" / f"{name}_resources.csv"
    status = BASE / "logs" / f"{name}_status.json"
    if any(path.exists() for path in (log, resources, status)):
        raise FileExistsError(f"Refusing to overwrite {name}")
    if shutil.disk_usage(BASE).free < 15 * 1024**3:
        raise RuntimeError("Disk free below 15 GiB")
    start = time.monotonic()
    print(f"START {name} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
    with log.open("wb") as log_file, resources.open("w", newline="") as resource_file:
        writer = csv.writer(resource_file)
        writer.writerow(["elapsed_s", "gpu_used_mib", "process_rss_kib", "cgroup_current_bytes", "disk_free_bytes"])
        proc = subprocess.Popen(cmd, cwd=REPRO, stdout=log_file, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            tick = time.monotonic()
            writer.writerow([
                round(tick - start, 3), gpu_mib(), rss_kib(proc.pid),
                int(Path("/sys/fs/cgroup/memory.current").read_text()),
                shutil.disk_usage(BASE).free,
            ])
            resource_file.flush()
            time.sleep(max(0, 1.0 - (time.monotonic() - tick)))
        code = proc.wait()
    rows = list(csv.DictReader(resources.open()))
    def peak(column):
        values = [float(row[column]) for row in rows if row[column] not in ("", None)]
        return max(values) if values else None
    result = {
        "name": name, "command": cmd, "cwd": str(REPRO), "exit_code": code,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "peak_gpu_used_mib": peak("gpu_used_mib"),
        "peak_process_rss_kib": peak("process_rss_kib"),
        "peak_cgroup_current_bytes": peak("cgroup_current_bytes"),
        "minimum_disk_free_bytes": min(float(row["disk_free_bytes"]) for row in rows),
    }
    status.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"END {name} exit={code} elapsed={result['elapsed_seconds']:.1f}s peak_gpu={result['peak_gpu_used_mib']}", flush=True)
    if code:
        raise SystemExit(code)
    return result


def validate_search(output, target):
    required = [
        output / "run_manifest.json", output / "search/search_config.used.yaml",
        output / "search/results.tsv", output / "report/summary.md",
    ]
    if any(not item.is_file() for item in required):
        raise RuntimeError(f"Missing search outputs: {output}")
    manifest = json.loads(required[0].read_text())
    args = manifest["args"]
    if not (
        args["start_cell"] == "DMSO_DMSO" and args["target_cell"] == target
        and args["max_depth"] == 2 and args["beam_size"] == 128
        and args["robust_n_samples"] == 5 and args["max_drugs_to_consider"] is None
        and args["device"] == "cuda:0"
    ):
        raise RuntimeError(f"Search configuration mismatch: {output}")
    with required[2].open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    if not any(int(row["depth"]) == 2 for row in rows):
        raise RuntimeError(f"No depth-2 paths: {output}")
    return len(rows)


def validate_pair_report(output):
    report = output / "pair_recovery_report"
    required = [
        report / "summary.md", report / "tables/positive_control_hit_summary.tsv",
        report / "figures/01_depth2_sinkhorn_rank_curve.png",
    ]
    if any(not item.is_file() for item in required):
        raise RuntimeError(f"Pair recovery report incomplete: {report}")


def validate_s5(output, target):
    args_path = output / "positive_control_cli_args.json"
    table = output / "tables/random_pairs.tsv"
    required = [args_path, table, output / "tables/evaluation_results.tsv", output / "tables/batch_metadata.tsv"]
    if any(not item.is_file() for item in required):
        raise RuntimeError(f"S5 outputs incomplete: {output}")
    args = json.loads(args_path.read_text())
    if not (
        args["start_cell"] == "DMSO_DMSO" and args["target_cell"] == target
        and args["two_drug_pair"] == ["crizotinib", "Panobinostat"]
        and args["random_pairs"] == 100 and args["random_type"] == "matched"
        and args["n_batches"] == 5 and args["projection_method"] == "none"
        and not args["projection_auto_select_components"]
        and args["projection_target_split"] == "none"
        and args["trajectory_embedding_space"] == "full"
    ):
        raise RuntimeError(f"S5 configuration mismatch: {output}")
    with table.open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    if len(rows) != 500 or len({row["random_pair_index"] for row in rows}) != 100:
        raise RuntimeError(f"S5 random control mismatch: {output}")


def main():
    (BASE / "logs").mkdir(parents=True, exist_ok=True)
    if gpu_mib() is None:
        raise RuntimeError("GPU unavailable; refusing to start")
    if not (MODEL / "checkpoints/final.ckpt").is_file():
        raise FileNotFoundError("ST-SE-Tahoe checkpoint missing")
    for key, target in TARGETS:
        adata = INPUTS / f"{target}.SE600M.h5ad"
        if not adata.is_file():
            raise FileNotFoundError(adata)
    for key, target in TARGETS:
        adata = INPUTS / f"{target}.SE600M.h5ad"
        output = BASE / "figure3i_open_search" / key
        if output.exists():
            raise FileExistsError(output)
        cmd = [
            str(PHAROS), "open-search", "--adata", str(adata),
            "--start-cell", "DMSO_DMSO", "--target-cell", target,
            "--cell-col", "cell_type", "--model-dir", str(MODEL),
            "--output-dir", str(output),
            "--drug-metadata", str(REPRO / "metadata/drug_metadata_sciplex.csv"),
            "--device", "cuda:0", "--max-depth", "2", "--beam-size", "128",
            "--converter-chunk-size", "16", "--robust-n-samples", "5",
        ]
        run(f"search_{key}", cmd)
        n_rows = validate_search(output, target)
        print(f"VALID search_{key} retained_rows={n_rows}", flush=True)
        report_cmd = [
            str(PHAROS), "report", "open-search", "--run-dir", str(output),
            "--drug-a", "crizotinib", "--drug-b", "Panobinostat",
            "--output-dir", str(output / "pair_recovery_report"),
        ]
        run(f"pair_report_{key}", report_cmd)
        validate_pair_report(output)
        print(f"VALID pair_report_{key}", flush=True)
    for key, target in TARGETS:
        adata = INPUTS / f"{target}.SE600M.h5ad"
        output = BASE / "s5_full_embedding" / key
        if output.exists():
            raise FileExistsError(output)
        cmd = [
            str(PHAROS), "hypothesis-driven", "pair", "--adata", str(adata),
            "--start-cell", "DMSO_DMSO", "--target-cell", target,
            "--cell-col", "cell_type", "--model-dir", str(MODEL),
            "--output-dir", str(output),
            "--drug-pair", "crizotinib", "Panobinostat",
            "--moa-pairs", "Multi-TK inhibitor", "HDAC inhibitor",
            "--projection-method", "none", "--no-projection-auto-select-components",
            "--projection-target-split", "none", "--trajectory-embedding-space", "full",
            "--device", "cuda:0",
        ]
        run(f"s5_{key}", cmd)
        validate_s5(output, target)
        print(f"VALID s5_{key}", flush=True)
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

"""Embed 7,422 missing Figure 3 cells once, with resource monitoring."""

import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


BASE = Path("/srv/pharos/figure3_a549")
INPUT = BASE / "input/figure3_missing_aligned_log1p.h5ad"
OUTPUT = BASE / "embedding_aligned/figure3_missing_aligned.SE600M.h5ad"
LOG = BASE / "logs/embedding_aligned_cli.log"
MONITOR = BASE / "logs/embedding_aligned_resources_0p5s.csv"
TIMING = BASE / "logs/embedding_aligned_timing.json"
MODEL = Path("/srv/pharos/input/SE-600M")
COMPAT = Path("/srv/pharos/code/pharos_server_scripts/state_emb_compat.py")


def gpu_mib():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=5,
    )
    return int(r.stdout.strip().splitlines()[0]) if r.returncode == 0 else None


def rss_kib(pid):
    try:
        text = Path(f"/proc/{pid}/status").read_text()
        match = re.search(r"^VmRSS:\s+(\d+) kB$", text, re.MULTILINE)
        return int(match.group(1)) if match else None
    except FileNotFoundError:
        return None


def available_kib():
    match = re.search(r"^MemAvailable:\s+(\d+) kB$", Path("/proc/meminfo").read_text(), re.MULTILINE)
    return int(match.group(1))


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists() or any(p.exists() for p in (OUTPUT, LOG, MONITOR, TIMING)):
        raise RuntimeError("Missing input or existing output; refusing overwrite")
    cmd = [
        sys.executable, "-u", str(COMPAT), "emb", "transform",
        "--model-folder", str(MODEL),
        "--checkpoint", str(MODEL / "se600m_epoch16.ckpt"),
        "--input", str(INPUT), "--output", str(OUTPUT),
        "--embed-key", "X_state", "--batch-size", "32",
    ]
    start = time.monotonic()
    with LOG.open("wb") as log, MONITOR.open("w", newline="") as monitor:
        writer = csv.writer(monitor)
        writer.writerow(["elapsed_s", "gpu_used_mib", "process_rss_kib", "mem_available_kib", "cgroup_current_bytes", "disk_free_bytes"])
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            tick = time.monotonic()
            writer.writerow([
                round(tick - start, 3), gpu_mib(), rss_kib(proc.pid), available_kib(),
                int(Path("/sys/fs/cgroup/memory.current").read_text()),
                shutil.disk_usage(BASE).free,
            ])
            monitor.flush()
            time.sleep(max(0, 0.5 - (time.monotonic() - tick)))
        code = proc.wait()
    elapsed = time.monotonic() - start
    rows = list(csv.DictReader(MONITOR.open()))
    def peak(col):
        values = [float(r[col]) for r in rows if r[col] not in ("", None)]
        return max(values) if values else None
    log_text = LOG.read_text(errors="replace")
    result = {
        "command": cmd, "input_cells": 7422, "batch_size": 32,
        "exit_code": code, "elapsed_seconds": round(elapsed, 3),
        "peak_gpu_used_mib": peak("gpu_used_mib"),
        "peak_process_rss_kib": peak("process_rss_kib"),
        "peak_cgroup_current_bytes": peak("cgroup_current_bytes"),
        "minimum_mem_available_kib": min(float(r["mem_available_kib"]) for r in rows),
        "minimum_disk_free_bytes": min(float(r["disk_free_bytes"]) for r in rows),
        "cgroup_memory_max_bytes": int(Path("/sys/fs/cgroup/memory.max").read_text()),
        "resource_samples": len(rows),
        "output_exists": OUTPUT.is_file(),
        "success_marker": "Embedding computation completed successfully!" in log_text,
    }
    TIMING.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if code != 0 or not result["success_marker"]:
        raise SystemExit(code or 1)


if __name__ == "__main__":
    main()

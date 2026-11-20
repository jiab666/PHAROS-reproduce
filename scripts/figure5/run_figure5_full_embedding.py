"""Run one full 61,901-cell STATE embedding job with resource sampling."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time


ROOT = Path("/srv/pharos/work/figure5/full_embedding_61901")
INPUT = ROOT.parent / "figure5_union_log1p_for_SE600M.h5ad"
OUTPUT = ROOT / "figure5_union.SE600M.h5ad"
MODEL = Path("/srv/pharos/input/SE-600M")
COMPAT = Path("/srv/pharos/code/pharos_server_scripts/state_emb_compat.py")
LOG = ROOT / "state_cli.log"
MONITOR = ROOT / "resource_0p5s.csv"
TIMING = ROOT / "timing.json"


def rss_kib(pid: int) -> int | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        match = re.search(r"^VmRSS:\s+(\d+) kB$", status, re.MULTILINE)
        return int(match.group(1)) if match else None
    except FileNotFoundError:
        return None


def mem_available_kib() -> int:
    info = Path("/proc/meminfo").read_text()
    return int(re.search(r"^MemAvailable:\s+(\d+) kB$", info, re.MULTILINE).group(1))


def gpu_used_mib() -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5, check=False,
    )
    if result.returncode != 0:
        return None
    return int(result.stdout.strip().splitlines()[0])


def monitor(pid: int, stop: threading.Event, started: float) -> None:
    with MONITOR.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["elapsed_s", "gpu_used_mib", "process_rss_kib", "mem_available_kib", "cgroup_current_bytes", "disk_free_bytes"])
        while not stop.is_set():
            tick = time.perf_counter()
            try:
                writer.writerow([
                    round(tick - started, 3), gpu_used_mib(), rss_kib(pid),
                    mem_available_kib(),
                    int(Path("/sys/fs/cgroup/memory.current").read_text().strip()),
                    shutil.disk_usage(ROOT).free,
                ])
                handle.flush()
            except Exception as exc:
                writer.writerow([round(tick - started, 3), f"error:{exc}", "", "", ""])
                handle.flush()
            stop.wait(max(0, 0.5 - (time.perf_counter() - tick)))


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not INPUT.is_file() or OUTPUT.exists() or TIMING.exists() or LOG.exists() or MONITOR.exists():
        raise RuntimeError("Missing input or an existing test output/timing file")
    if not MODEL.joinpath("se600m_epoch16.ckpt").is_file() or not MODEL.joinpath("protein_embeddings.pt").is_file():
        raise FileNotFoundError("SE-600M weights/protein embeddings missing")
    cmd = [
        sys.executable, "-u", str(COMPAT), "emb", "transform",
        "--model-folder", str(MODEL),
        "--checkpoint", str(MODEL / "se600m_epoch16.ckpt"),
        "--input", str(INPUT), "--output", str(OUTPUT),
        "--embed-key", "X_state", "--batch-size", "32",
    ]
    started = time.perf_counter()
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    stop = threading.Event()
    worker = threading.Thread(target=monitor, args=(proc.pid, stop, started), daemon=True)
    worker.start()
    events: dict[str, float] = {}
    markers = {
        "model_load_start": "Loading model from checkpoint:",
        "model_load_end": "Created output directory:",
        "encode_adata_start": "Computing embeddings for",
        "cli_complete": "Embedding computation completed successfully!",
    }
    line_buffer = ""
    with LOG.open("wb") as log:
        while chunk := os.read(proc.stdout.fileno(), 4096):
            log.write(chunk)
            log.flush()
            line_buffer += chunk.decode("utf-8", errors="replace")
            parts = re.split(r"[\r\n]", line_buffer)
            line_buffer = parts.pop()
            for part in parts:
                now = time.perf_counter() - started
                for name, marker in markers.items():
                    if name not in events and marker in part:
                        events[name] = now
                        print(f"{name}: {now:.2f}s", flush=True)
                if "Encoding:" in part:
                    if "encoding_batches_start" not in events and re.search(r"Encoding:\s*0%", part):
                        events["encoding_batches_start"] = now
                        print(f"encoding_batches_start: {now:.2f}s", flush=True)
                    if "encoding_batches_end" not in events and re.search(r"Encoding:\s*100%", part):
                        events["encoding_batches_end"] = now
                        print(f"encoding_batches_end: {now:.2f}s", flush=True)
    code = proc.wait()
    elapsed = time.perf_counter() - started
    stop.set()
    worker.join(timeout=10)

    with MONITOR.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    def peak(column: str) -> float | None:
        values = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except (ValueError, TypeError):
                pass
        return max(values) if values else None
    result = {
        "command": cmd,
        "input_cells": 61_901,
        "batch_size": 32,
        "exit_code": code,
        "elapsed_seconds": round(elapsed, 3),
        "events_seconds_from_start": {key: round(value, 3) for key, value in events.items()},
        "model_load_seconds": round(events["model_load_end"] - events["model_load_start"], 3)
            if "model_load_start" in events and "model_load_end" in events else None,
        "encoding_batches_seconds": round(events["encoding_batches_end"] - events["encoding_batches_start"], 3)
            if "encoding_batches_start" in events and "encoding_batches_end" in events else None,
        "peak_gpu_used_mib": peak("gpu_used_mib"),
        "peak_process_rss_kib": peak("process_rss_kib"),
        "minimum_mem_available_kib": min(float(row["mem_available_kib"]) for row in rows if row["mem_available_kib"]),
        "peak_cgroup_current_bytes": peak("cgroup_current_bytes"),
        "minimum_disk_free_bytes": min(float(row["disk_free_bytes"]) for row in rows if row["disk_free_bytes"]),
        "cgroup_memory_max_bytes": int(Path("/sys/fs/cgroup/memory.max").read_text().strip()),
        "resource_samples": len(rows),
    }
    if result["encoding_batches_seconds"]:
        result["encoding_throughput_cells_per_second"] = round(61_901 / result["encoding_batches_seconds"], 3)
    TIMING.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["exit_code", "elapsed_seconds", "model_load_seconds", "encoding_batches_seconds", "encoding_throughput_cells_per_second", "peak_gpu_used_mib", "peak_process_rss_kib"] if k in result}, indent=2), flush=True)
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()

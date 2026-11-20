"""Resume the official Ozmen raw RDS with independently verified HTTP ranges.

The existing prefix from curl is preserved. Each remaining range is saved to a
separate part file and checked against Content-Range before assembly and MD5.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
from pathlib import Path
import time

import requests


URL = "https://zenodo.org/records/13743374/files/Integrated_Dataset_raw_seurobj.rds?download=1"
SIZE = 640_670_945
MD5 = "d843f850a152f641389b3e55a1e42627"
OUT = Path(__file__).parent / "raw" / "Integrated_Dataset_raw_seurobj.rds"
PARTS = OUT.parent / "rds_range_parts"
CHUNK = 16 * 1024 * 1024


def fetch(part: tuple[int, int]) -> tuple[int, int]:
    start, end = part
    target = PARTS / f"{start:012d}-{end:012d}.part"
    expected = end - start + 1
    if target.exists() and target.stat().st_size == expected:
        return start, expected
    for attempt in range(8):
        try:
            response = requests.get(
                URL,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(20, 120),
            )
            response.raise_for_status()
            content_range = response.headers.get("Content-Range")
            if response.status_code != 206 or content_range != f"bytes {start}-{end}/{SIZE}":
                raise ValueError(f"unexpected range response: {response.status_code} {content_range}")
            temp = target.with_suffix(".tmp")
            written = 0
            with temp.open("wb") as handle:
                for block in response.iter_content(chunk_size=256 * 1024):
                    if block:
                        handle.write(block)
                        written += len(block)
            if written != expected:
                raise IOError(f"short range {start}-{end}: {written}/{expected}")
            os.replace(temp, target)
            return start, written
        except Exception as exc:
            print(f"range {start}-{end}, attempt {attempt + 1}: {exc}", flush=True)
            time.sleep(min(3 * (attempt + 1), 20))
    raise RuntimeError(f"range {start}-{end} failed after retries")


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    PARTS.mkdir(exist_ok=True)
    prefix = OUT.stat().st_size if OUT.exists() else 0
    if prefix > SIZE:
        raise ValueError(f"existing prefix exceeds official size: {prefix}")
    if prefix == SIZE:
        print("Full-size file already present; checking MD5", flush=True)
    else:
        ranges = [(s, min(s + CHUNK - 1, SIZE - 1)) for s in range(prefix, SIZE, CHUNK)]
        print(f"Preserving {prefix} existing bytes; downloading {len(ranges)} ranges", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(fetch, item) for item in ranges]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                start, count = future.result()
                print(f"{index}/{len(ranges)} ranges: {start}, {count} bytes", flush=True)

        assembled = OUT.with_suffix(".assembled")
        digest = hashlib.md5()
        with assembled.open("wb") as destination:
            if prefix:
                with OUT.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
                        digest.update(block)
            for start, end in ranges:
                part = PARTS / f"{start:012d}-{end:012d}.part"
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
                        digest.update(block)
        if assembled.stat().st_size != SIZE or digest.hexdigest() != MD5:
            raise ValueError(f"assembled file failed size/MD5: {assembled.stat().st_size}, {digest.hexdigest()}")
        os.replace(assembled, OUT)
        print(f"Verified official MD5: {MD5}", flush=True)
        return

    digest = hashlib.md5()
    with OUT.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MD5:
        raise ValueError(f"full-size file failed MD5: {digest.hexdigest()}")
    print(f"Verified official MD5: {MD5}", flush=True)


if __name__ == "__main__":
    main()

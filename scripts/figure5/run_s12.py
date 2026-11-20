"""Run official Figure S12 Open Searches with independent logs and resource records."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import anndata as ad


WORK = Path('/srv/pharos/work/figure5')
ROOT = WORK / 's12_open_search'
INPUT = WORK / 'derived_inputs/figure5c'
MODEL = Path('/srv/pharos/input/ST-SE-Tahoe/fewshot/state_generalization_X_state')
REPRO = Path('/srv/pharos/code/PHAROS_reproduce')
PHAROS = Path('/usr/local/miniconda3/envs/PHAROS/bin/pharos')
CASES = ['Met1', 'Met4', 'Met6', 'Met9', 'Met11', 'Met3']


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gpu_mib():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                        '--format=csv,noheader,nounits'], capture_output=True,
                       text=True, timeout=8)
    return int(r.stdout.strip().splitlines()[0]) if r.returncode == 0 and r.stdout.strip() else None


def rss_kib(pid):
    try:
        m = re.search(r'^VmRSS:\s+(\d+) kB$', Path(f'/proc/{pid}/status').read_text(), re.M)
        return int(m.group(1)) if m else None
    except FileNotFoundError:
        return None


def preflight():
    if gpu_mib() is None:
        raise RuntimeError('GPU unavailable')
    if not PHAROS.is_file() or not (MODEL / 'checkpoints/final.ckpt').is_file():
        raise FileNotFoundError('PHAROS CLI or model checkpoint missing')
    if not (REPRO / 'metadata/drug_metadata_sciplex.csv').is_file():
        raise FileNotFoundError('Drug metadata missing')
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise RuntimeError('Less than 15 GiB disk available')
    manifest = json.loads((WORK / 'derived_inputs/derived_inputs_manifest.json').read_text())
    items = {r['case']: r for r in manifest['five_c']}
    rows = []
    for case in CASES:
        m = items[case]
        path = INPUT / f'figure5c_{case}.SE600M.h5ad'
        if not path.is_file() or path.stat().st_size != m['bytes'] or digest(path) != m['sha256']:
            raise RuntimeError(f'Input file missing or SHA-256 mismatch: {case}')
        a = ad.read_h5ad(path, backed='r')
        labels = a.obs['cell_type_merged'].astype(str)
        start_n = int(labels.eq(m['start_label']).sum())
        target_n = int(labels.eq(m['target_label']).sum())
        ok = (a.n_obs == m['cells'] == start_n + target_n and
              a.obsm['X_state'].shape == (a.n_obs, 2058) and
              start_n == m['start_cells'] and target_n == m['target_cells'] and
              start_n >= 256 and target_n >= 256)
        a.file.close()
        if not ok:
            raise RuntimeError(f'Label/embedding/count mismatch: {case}')
        rows.append({'case': case, 'input': str(path), 'input_sha256': m['sha256'],
                     'start_label': m['start_label'], 'start_cells': start_n,
                     'target_label': m['target_label'], 'target_cells': target_n,
                     'batch_selection': 'high-sensitivity' if case == 'Met3' else 'standard'})
    (ROOT / 'preflight.json').write_text(json.dumps(rows, indent=2))
    print('PREFLIGHT_OK ' + ' '.join(f"{r['case']}={r['start_cells']}/{r['target_cells']}" for r in rows), flush=True)
    return rows


def run_one(item):
    case = item['case']
    result = ROOT / case / 'result'
    log = ROOT / case / 'open_search.log'
    samples = ROOT / case / 'resources.csv'
    status = ROOT / case / 'status.json'
    if any(p.exists() for p in (result, log, samples, status)):
        raise FileExistsError(f'Refusing to overwrite {case}')
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise RuntimeError('Less than 15 GiB disk available')
    log.parent.mkdir(parents=True)
    cmd = [str(PHAROS), 'open-search', '--adata', item['input'],
           '--start-cell', item['start_label'], '--target-cell', item['target_label'],
           '--cell-col', 'cell_type_merged', '--model-dir', str(MODEL),
           '--output-dir', str(result), '--drug-metadata',
           str(REPRO / 'metadata/drug_metadata_sciplex.csv'), '--device', 'cuda:0']
    if case == 'Met3':
        cmd += ['--batch-selection', 'high-sensitivity']
    started = time.monotonic()
    print(f'START {case} {time.strftime("%Y-%m-%dT%H:%M:%S%z")}', flush=True)
    with log.open('wb') as out, samples.open('w', newline='') as resource_file:
        writer = csv.writer(resource_file)
        writer.writerow(['elapsed_s', 'gpu_used_mib', 'process_rss_kib',
                         'cgroup_current_bytes', 'disk_free_bytes'])
        proc = subprocess.Popen(cmd, cwd=REPRO, stdout=out, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            tick = time.monotonic()
            writer.writerow([round(tick - started, 3), gpu_mib(), rss_kib(proc.pid),
                             int(Path('/sys/fs/cgroup/memory.current').read_text()),
                             shutil.disk_usage(ROOT).free])
            resource_file.flush()
            time.sleep(max(0, 1.0 - (time.monotonic() - tick)))
        code = proc.wait()
    with samples.open() as f:
        records = list(csv.DictReader(f))
    def peak(col):
        values = [float(x[col]) for x in records if x[col]]
        return max(values) if values else None
    info = {'case': case, 'command': cmd, 'exit_code': code,
            'elapsed_seconds': round(time.monotonic() - started, 3),
            'peak_gpu_used_mib': peak('gpu_used_mib'),
            'peak_process_rss_kib': peak('process_rss_kib'),
            'peak_cgroup_current_bytes': peak('cgroup_current_bytes')}
    if code == 0:
        required = [result / 'run_manifest.json', result / 'search/search_config.used.yaml',
                    result / 'search/results.tsv', result / 'report/summary.md',
                    result / 'sample_drug_report/summary.md']
        info['required_files_present'] = all(p.is_file() for p in required)
        if info['required_files_present']:
            args = json.loads(required[0].read_text())['args']
            with required[2].open() as f:
                rows = list(csv.DictReader(f, delimiter='\t'))
            depth2 = [r for r in rows if int(r['depth']) == 2]
            info['depth2_paths'] = len(depth2)
            info['configuration_valid'] = (args['start_cell'] == item['start_label'] and
                    args['target_cell'] == item['target_label'] and
                    args['max_depth'] == 2 and args['beam_size'] == 128 and
                    args['max_drugs_to_consider'] is None and
                    args['batch_selection'] == item['batch_selection'])
            info['required_files_present'] &= len(depth2) == 128 and info['configuration_valid']
        if not info['required_files_present']:
            info['error'] = 'Search exited 0 but official outputs/config incomplete'
    status.write_text(json.dumps(info, indent=2))
    print(f"END {case} exit={code} seconds={info['elapsed_seconds']} peak_gpu_mib={info['peak_gpu_used_mib']} validated={info.get('required_files_present')}", flush=True)
    if code != 0 or not info.get('required_files_present'):
        raise RuntimeError(f'{case} failed; see {log}')
    return info


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    items = preflight()
    results = []
    for item in items:
        result = run_one(item)
        results.append(result)
        if item['case'] == 'Met1':
            gpu_ok = 2 * result['peak_gpu_used_mib'] < 0.85 * 24564
            mem_ok = 2 * result['peak_cgroup_current_bytes'] < 0.85 * 92341796864
            print(f'MET1_RESOURCE_DECISION pair_gpu_under_85pct={gpu_ok} pair_ram_under_85pct={mem_ok}; using serial execution', flush=True)
    (ROOT / 'run_summary.json').write_text(json.dumps(results, indent=2))
    print('ALL_COMPLETE', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--preflight-only']:
        ROOT.mkdir(parents=True, exist_ok=True)
        preflight()
    else:
        main()

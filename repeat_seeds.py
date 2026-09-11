"""Repeat training with the same cohort and different random seeds."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[43, 44])
    args = parser.parse_args()
    source = ROOT / 'artifacts'
    digest = hashlib.sha256((source / 'prepared.npz').read_bytes()).hexdigest()
    for seed in args.seeds:
        out = ROOT / 'experiments' / f'seed_{seed}'
        out.mkdir(parents=True, exist_ok=True)
        prepared = out / 'prepared.npz'
        if not prepared.exists():
            try:
                os.link(source / 'prepared.npz', prepared)
            except OSError:
                shutil.copy2(source / 'prepared.npz', prepared)
        if hashlib.sha256(prepared.read_bytes()).hexdigest() != digest:
            raise ValueError('Repeated runs must use the identical prepared cohort')
        if not (out / 'manifest.json').exists():
            shutil.copy2(source / 'manifest.json', out / 'manifest.json')
        (out / 'protocol.json').write_text(json.dumps(dict(
            cohort_seed=42, training_seed=seed, prepared_sha256=digest,
            epochs=3, all_targets=True, selection='validation NDCG@10',
            purpose='Fixed-split training-seed sensitivity; not a fresh test set'), indent=2))
        subprocess.run([sys.executable, str(ROOT / 'run.py'), 'train', '--out', str(out),
                        '--seed', str(seed), '--epochs', '3', '--all-targets',
                        '--resume', '--threads', '4'], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()

"""Recompute seed-42 paired uncertainty using the deployed dislike filter.

Does not train or select models. Refuses to write if NDCG differs from results.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from models import NextBeat
from run import neural_ranks, serving_dislikes

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    folder = ROOT / 'artifacts'
    with np.load(folder / 'demo.npz') as archive:
        data = {key: archive[key] for key in archive.files}
    blocked = [serving_dislikes(data, i, x, f)
               for i, (x, f) in enumerate(zip(data['x'], data['f']))]
    reported = json.loads((folder / 'results.json').read_text())
    values = {}
    for name, filename, feedback in [('Sequence-only GRU', 'sequence.pt', False),
                                     ('NextBeat', 'nextbeat.pt', True)]:
        model = NextBeat(len(data['counts']), feedback=feedback)
        model.load_state_dict(torch.load(folder / filename, map_location='cpu', weights_only=True))
        ranks = neural_ranks(model, data, blocked)
        values[name] = ((ranks == data['y'][:, None]) / np.log2(np.arange(10) + 2)).sum(1)
        if not np.isclose(values[name].mean(), reported[name]['ndcg10'], rtol=0, atol=1e-8):
            raise ValueError(f'{name}: recomputed NDCG does not match results.json')
    delta = values['NextBeat'] - values['Sequence-only GRU']
    rng = np.random.default_rng(42)
    samples = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
    result = {
        'nextbeat_minus_sequence_ndcg10': float(delta.mean()),
        'paired_user_bootstrap_95_percent': np.quantile(samples, [.025, .975]).tolist(),
        'replicates': 2000,
        'seed': 42,
        'evaluated_users': len(delta),
        'evaluation': 'Full prior active-dislike filtering; seed-42 frozen models; same cohort as results.json',
        'scope': 'User sampling uncertainty only; not training-seed variance or independent generalization proof',
    }
    path = folder / 'uncertainty.json'
    path.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
    check_path = ROOT / 'verification' / 'ARTIFACT_CHECKSUMS.json'
    checks = json.loads(check_path.read_text())
    checks['uncertainty.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    check_path.write_bytes((json.dumps(checks, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

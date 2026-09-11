"""Check the saved project before a presentation; never starts training."""
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    failures = []
    print(f'NextBeat installation check | {platform.system()} | Python {platform.python_version()}')
    for name in ['numpy', 'scipy', 'pandas', 'polars', 'pyarrow', 'torch', 'streamlit', 'fastapi', 'uvicorn']:
        try:
            module = importlib.import_module(name)
            print(f'OK {name}: {getattr(module, "__version__", "installed")}')
        except ImportError as error:
            failures.append(f'{name}: {error}')
    checks = json.loads((ROOT / 'verification' / 'ARTIFACT_CHECKSUMS.json').read_text())
    for name, expected in checks.items():
        path = ROOT / 'artifacts' / name
        if (name == 'prepared.npz' or name.endswith('.resume')) and not path.exists():
            print(f'INFO optional training file not installed: {name}')
            continue
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f'Missing or modified saved artifact: {name}')
    if not failures:
        import numpy as np
        import torch
        from models import NextBeat
        from run import top10
        with np.load(ROOT / 'artifacts' / 'demo.npz') as data:
            for feedback, filename in [(False, 'sequence.pt'), (True, 'nextbeat.pt')]:
                model = NextBeat(len(data['counts']), feedback=feedback)
                model.load_state_dict(torch.load(ROOT / 'artifacts' / filename,
                                                 map_location='cpu', weights_only=True))
                model.eval()
                with torch.inference_mode():
                    scores = model.scores(torch.tensor(data['x'][:1], dtype=torch.long),
                                          torch.tensor(data['f'][:1], dtype=torch.float32)).numpy()
                ranks = top10(scores.copy())
                if ranks.shape != (1, 10) or ranks.min() < 2 or not np.isfinite(scores).all():
                    failures.append(f'Invalid inference output: {filename}')
                else:
                    print(f'OK {filename}: 10 real catalogue recommendations')
    for failure in failures:
        print('FAIL', failure)
    print('READY: python -m streamlit run app.py' if not failures else 'Fix the failures above before presenting.')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())

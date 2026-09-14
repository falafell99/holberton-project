"""Read-only API for the trained NextBeat model; anonymous IDs only."""
from pathlib import Path
from functools import lru_cache
import json
import os
import numpy as np
import scipy.sparse as sp
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models import NextBeat
from run import top10, serving_dislikes

app = FastAPI(title='NextBeat', version='1.0.0')
origins = os.environ.get('ALLOWED_ORIGINS', '*')
app.add_middleware(CORSMiddleware,
                    allow_origins=[o.strip().rstrip('/') for o in origins.split(',')] if origins != '*' else ['*'],
                    allow_methods=['GET', 'POST'], allow_headers=['*'])
ROOT = Path(__file__).parent / 'artifacts'
cache = None


def resources():
    global cache
    if cache is None:
        archive = np.load(ROOT / 'demo.npz')
        data = {k: archive[k] for k in archive.files}
        selected = json.loads((ROOT/'manifest.json').read_text())['selected_model_by_validation']
        model = None
        if selected in ['NextBeat', 'Sequence-only GRU']:
            model = NextBeat(len(data['counts']), feedback=selected == 'NextBeat')
            filename = 'nextbeat.pt' if selected == 'NextBeat' else 'sequence.pt'
            model.load_state_dict(torch.load(ROOT / filename, map_location='cpu', weights_only=True))
            model.eval()
        elif selected == 'ItemKNN':
            model = sp.load_npz(ROOT/'itemknn.npz')
        torch.set_num_threads(2)
        cache = data, model, selected
    return cache


@app.get('/health')
def health():
    return {'ready': (ROOT / 'nextbeat.pt').exists() and (ROOT / 'demo.npz').exists()}


@app.get('/models')
def models():
    data, _, selected = resources()
    return {'models': ['NextBeat', 'Sequence-only GRU', 'ItemKNN', 'Most Popular'], 'default': selected}


def track_label(data, item):
    return f'Track {int(data["vocab"][item-2])}' if item >= 2 else 'Outside selected catalogue'


@app.get('/history/{uid}')
def history(uid: int):
    data, _, _ = resources()
    ix = np.flatnonzero(data['uid'] == uid)
    if len(ix) == 0:
        raise HTTPException(404, 'Anonymous user is not in the saved evaluation cohort.')
    i = int(ix[0])
    rows = []
    for item, features in zip(data['x'][i], data['f'][i]):
        if item:
            event = ['listen', 'like', 'dislike', 'unlike', 'undislike'][int(np.argmax(features[1:]))]
            rows.append({'track': track_label(data, int(item)), 'event': event,
                        'played_percent': round(float(features[0]) * 100, 1) if event == 'listen' else None})
    return {'uid': uid, 'history': rows}


@app.get('/metrics')
def metrics():
    report = json.loads((ROOT / 'results.json').read_text())
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    return {'metrics': report, 'eligible_targets': manifest['test_users'],
            'total_targets': report['NextBeat']['all_target_users']}


@app.get('/users')
def users():
    data, _, _ = resources()
    return {'anonymous_user_ids': data['uid'].tolist()}


WHAT_IF_EDITS = {
    'keep': None,
    'full_listen': [1, 1, 0, 0, 0, 0],
    'like': [0, 0, 1, 0, 0, 0],
    'dislike': [0, 0, 0, 1, 0, 0],
    'short_listen': [.1, 1, 0, 0, 0, 0],
}


class RecommendRequest(BaseModel):
    model: str | None = None
    what_if: str = 'keep'


def score_and_rank(data, model_obj, model_name, x, f, index):
    blocked = [serving_dislikes(data, index, x[0], f[0])]
    if model_name == 'Most Popular':
        scores = data['counts'][None].copy()
    elif model_name == 'ItemKNN':
        seen = x[0][x[0] >= 2]
        scores = np.asarray(model_obj[seen].sum(axis=0)) if len(seen) else data['counts'][None].copy()
        scores = scores + data['counts'][None] / data['counts'].max() * 1e-6
    else:
        with torch.no_grad():
            scores = model_obj.scores(torch.tensor(x, dtype=torch.long), torch.tensor(f, dtype=torch.float32)).numpy()
    ranking = top10(scores, blocked)[0]
    return [{'track_id': int(data['vocab'][j-2]), 'score': float(scores[0, j])} for j in ranking]


@lru_cache(maxsize=4)
def model_for(name, n_items):
    if name in ['NextBeat', 'Sequence-only GRU']:
        model = NextBeat(n_items, feedback=name == 'NextBeat')
        filename = 'nextbeat.pt' if name == 'NextBeat' else 'sequence.pt'
        model.load_state_dict(torch.load(ROOT / filename, map_location='cpu', weights_only=True))
        model.eval()
        return model
    if name == 'ItemKNN':
        return sp.load_npz(ROOT / 'itemknn.npz')
    return None


@app.get('/recommend/{uid}')
def recommend(uid: int):
    data, model, selected = resources()
    ix = np.flatnonzero(data['uid'] == uid)
    if len(ix) == 0:
        raise HTTPException(404, 'Anonymous user is not in the saved evaluation cohort.')
    i = int(ix[0])
    recommendations = score_and_rank(data, model, selected, data['x'][i:i+1], data['f'][i:i+1], i)
    return {'uid': uid, 'model': selected, 'recommendations': recommendations}


@app.post('/recommend/{uid}')
def recommend_with_options(uid: int, body: RecommendRequest):
    data, cached_model, selected = resources()
    ix = np.flatnonzero(data['uid'] == uid)
    if len(ix) == 0:
        raise HTTPException(404, 'Anonymous user is not in the saved evaluation cohort.')
    if body.what_if not in WHAT_IF_EDITS:
        raise HTTPException(422, f'Unknown what_if option: {body.what_if}')
    i = int(ix[0])
    model_name = body.model or selected
    if model_name not in ['NextBeat', 'Sequence-only GRU', 'ItemKNN', 'Most Popular']:
        raise HTTPException(422, f'Unknown model: {model_name}')
    model_obj = cached_model if model_name == selected else model_for(model_name, len(data['counts']))
    x, f = data['x'][i:i+1].copy(), data['f'][i:i+1].copy()
    edit = WHAT_IF_EDITS[body.what_if]
    if edit is not None:
        f[0, -1] = edit
    recommendations = score_and_rank(data, model_obj, model_name, x, f, i)
    return {'uid': uid, 'model': model_name, 'recommendations': recommendations}

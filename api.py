"""Read-only API for the trained NextBeat model; anonymous IDs only."""
from pathlib import Path
import json
import os
import numpy as np
import scipy.sparse as sp
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models import NextBeat
from run import top10, windowed_active_dislikes

app = FastAPI(title='NextBeat', version='1.0.0')
origins = os.environ.get('ALLOWED_ORIGINS', '*')
app.add_middleware(CORSMiddleware, allow_origins=origins.split(',') if origins != '*' else ['*'],
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


@app.get('/users')
def users():
    data, _, _ = resources()
    return {'anonymous_user_ids': data['uid'].tolist()}


@app.get('/recommend/{uid}')
def recommend(uid: int):
    data, model, selected = resources()
    ix = np.flatnonzero(data['uid'] == uid)
    if len(ix) == 0:
        raise HTTPException(404, 'Anonymous user is not in the saved evaluation cohort.')
    i = int(ix[0])
    blocked = [windowed_active_dislikes(data['x'][i], data['f'][i])]
    if selected == 'Most Popular':
        scores = data['counts'][None].copy()
    elif selected == 'ItemKNN':
        x = data['x'][i]
        scores = np.asarray(model[x[x >= 2]].sum(axis=0))
        scores += data['counts'][None] / data['counts'].max() * 1e-6
    else:
        with torch.no_grad():
            scores = model.scores(torch.tensor(data['x'][i:i+1], dtype=torch.long),
                                  torch.tensor(data['f'][i:i+1], dtype=torch.float32)).numpy()
    ranking = top10(scores, blocked)[0]
    return {'uid': uid, 'model': selected, 'recommendations': [
        {'track_id': int(data['vocab'][j-2]), 'score': float(scores[0,j])} for j in ranking]}

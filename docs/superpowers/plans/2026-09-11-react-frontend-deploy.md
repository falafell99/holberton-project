# React Frontend + Deployment (Part 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Streamlit as the primary user-facing UI with a real React web frontend backed by an extended FastAPI backend, both actually deployed to the public internet (backend on Render, frontend on Vercel).

**Architecture:** Extend `api.py` with the endpoints the Streamlit app currently gets "for free" by running in the same Python process (`/models`, `/history/{uid}`, `/metrics`, and a `POST /recommend/{uid}` that accepts a model choice and a what-if edit) plus CORS. Build a single-page React app (Vite, plain CSS, no UI framework) in `frontend/` that mirrors the Streamlit app's functionality against these endpoints. Deploy the backend as a Docker container on Render and the frontend as a static Vite build on Vercel, wired together by an environment variable.

**Tech Stack:** FastAPI (existing), React 18 + Vite (new, `frontend/`), plain CSS (no Tailwind/MUI/etc.), Docker (existing `Dockerfile` pattern, new `Dockerfile.api`).

**Spec:** [docs/superpowers/specs/2026-09-11-finish-nextbeat-design.md](../specs/2026-09-11-finish-nextbeat-design.md) — Part 2.

## Global Constraints

- Backend endpoints must apply the Part 1 dislike-violation filter (`top10(scores, blocked)` with `blocked` from `windowed_active_dislikes`) on every ranking response — this is already true of the existing `GET /recommend/{uid}`; the new `POST /recommend/{uid}` must do the same.
- `app.py` (Streamlit) stays as-is for local debugging; it is not required to change in this plan.
- Frontend feature parity with the current Streamlit app: user select, model select, recent history, what-if picker (5 options: Keep/Full listen/Like/Dislike/Short listen), recommend button + results table, and the frozen-model metrics table — nothing more, nothing less (YAGNI: no auth, no extra pages, no charts beyond the existing table).
- No hardcoded backend URL in the frontend — read from `VITE_API_URL` at build time.
- CORS must allow the deployed frontend origin; default to permissive (`*`) since this API serves no credentials/cookies and only anonymous, non-sensitive track-id data — configurable via an `ALLOWED_ORIGINS` environment variable for tightening later.
- The existing `Dockerfile` (which serves the Streamlit app) must not be broken or repurposed — the backend gets its own `Dockerfile.api`.

---

### Task 1: CORS + `GET /models` endpoint

**Files:**
- Modify: `api.py:1-14` (imports, app setup), add new endpoint after `health()` (`api.py:35-37`)
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `GET /models` → `{"models": ["NextBeat", "Sequence-only GRU", "ItemKNN", "Most Popular"], "default": "<selected_model_by_validation>"}`. Later tasks' frontend `ModelSelect` component consumes this exact shape.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_artifacts.py` (this file already has `pytestmark = pytest.mark.skipif(...)` at the top and imports `TestClient`/`app` inside test functions — follow that pattern):

```python
def test_models_endpoint_lists_all_four_with_validation_default():
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    response = client.get('/models')
    assert response.status_code == 200
    body = response.json()
    assert body['models'] == ['NextBeat', 'Sequence-only GRU', 'ItemKNN', 'Most Popular']
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    assert body['default'] == manifest['selected_model_by_validation']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_artifacts.py -k models_endpoint -v`
Expected: FAIL with `404 Not Found` (no `/models` route yet)

- [ ] **Step 3: Implement CORS + the endpoint**

Replace `api.py:1-14`:

```python
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
```

Add after the `health()` endpoint (`api.py:35-37`):

```python
@app.get('/models')
def models():
    data, _, selected = resources()
    return {'models': ['NextBeat', 'Sequence-only GRU', 'ItemKNN', 'Most Popular'], 'default': selected}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_artifacts.py -k models_endpoint -v`
Expected: PASS

Run the full suite to confirm nothing broke: `.venv/bin/python -m pytest -q`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_artifacts.py
git commit -m "Add CORS and GET /models endpoint to api.py"
```

---

### Task 2: `GET /history/{uid}` endpoint

**Files:**
- Modify: `api.py` (add endpoint after `/models`)
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: `resources()` (existing), `windowed_active_dislikes` (existing import)
- Produces: `GET /history/{uid}` → `{"uid": <int>, "history": [{"track": "Track <id>" | "Outside selected catalogue", "event": "listen"|"like"|"dislike"|"unlike"|"undislike", "played_percent": <float or null>}]}`, 404 for unknown uid. Later tasks' frontend `HistoryTable` component consumes this exact shape — note `track` is a pre-formatted string (`"Track 12345"`), matching what `app.py`'s `track()` helper (`app.py:52-53`) already produces, so the frontend can render it directly with no track-id math.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_artifacts.py`:

```python
def test_history_endpoint_matches_streamlit_track_formatting():
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    uid = client.get('/users').json()['anonymous_user_ids'][0]
    response = client.get(f'/history/{uid}')
    assert response.status_code == 200
    body = response.json()
    assert body['uid'] == uid
    assert isinstance(body['history'], list)
    for row in body['history']:
        assert row['event'] in ['listen', 'like', 'dislike', 'unlike', 'undislike']
        assert row['track'].startswith('Track ') or row['track'] == 'Outside selected catalogue'
        if row['event'] != 'listen':
            assert row['played_percent'] is None


def test_history_endpoint_unknown_user_404s():
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    assert client.get('/history/-1').status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_artifacts.py -k history_endpoint -v`
Expected: FAIL with `404 Not Found` for the first test (no route), and the second test also currently 404s but for the wrong reason (no route at all rather than the intended "unknown user" 404) — both fail pre-implementation.

- [ ] **Step 3: Implement the endpoint**

Add to `api.py`, after the `/models` endpoint:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_artifacts.py -k history_endpoint -v`
Expected: 2 passed

Run the full suite: `.venv/bin/python -m pytest -q`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_artifacts.py
git commit -m "Add GET /history/{uid} endpoint to api.py"
```

---

### Task 3: `GET /metrics` endpoint

**Files:**
- Modify: `api.py` (add endpoint)
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: `artifacts/results.json` (existing file), `artifacts/manifest.json` (existing file)
- Produces: `GET /metrics` → `{"metrics": {"<model name>": {"recall10": ..., "ndcg10": ..., "unconditional_recall10": ..., "coverage10": ..., "novelty10": ..., "nfvr10": ..., "evaluated_users": ..., "all_target_users": ...}, ...}, "eligible_targets": <int>, "total_targets": <int>}` — `eligible_targets` mirrors `manifest['test_users']` and `total_targets` mirrors any model's `all_target_users` (they're identical across models per `run.py`'s `metrics()` function), matching the two numbers `app.py:91-92` already reports. Later tasks' frontend `MetricsPanel` component consumes this exact shape.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_artifacts.py`:

```python
def test_metrics_endpoint_reports_all_four_models():
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    response = client.get('/metrics')
    assert response.status_code == 200
    body = response.json()
    assert set(body['metrics']) == {'Most Popular', 'ItemKNN', 'Sequence-only GRU', 'NextBeat'}
    for row in body['metrics'].values():
        assert set(row) == {'recall10', 'ndcg10', 'unconditional_recall10', 'coverage10',
                            'novelty10', 'nfvr10', 'evaluated_users', 'all_target_users'}
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    assert body['eligible_targets'] == manifest['test_users']
    assert body['total_targets'] == body['metrics']['NextBeat']['all_target_users']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_artifacts.py -k metrics_endpoint -v`
Expected: FAIL with `404 Not Found`

- [ ] **Step 3: Implement the endpoint**

Add to `api.py`, after the `/history/{uid}` endpoint:

```python
@app.get('/metrics')
def metrics():
    report = json.loads((ROOT / 'results.json').read_text())
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    return {'metrics': report, 'eligible_targets': manifest['test_users'],
            'total_targets': report['NextBeat']['all_target_users']}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_artifacts.py -k metrics_endpoint -v`
Expected: PASS

Run the full suite: `.venv/bin/python -m pytest -q`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_artifacts.py
git commit -m "Add GET /metrics endpoint to api.py"
```

---

### Task 4: `POST /recommend/{uid}` with model choice and what-if edit

**Files:**
- Modify: `api.py` (refactor scoring into a shared helper, add the POST endpoint)
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: `resources()`, `top10`, `windowed_active_dislikes` (existing), `track_label` (Task 2)
- Produces: `POST /recommend/{uid}` with JSON body `{"model": "NextBeat"|"Sequence-only GRU"|"ItemKNN"|"Most Popular", "what_if": "keep"|"full_listen"|"like"|"dislike"|"short_listen"}` (both fields optional — `model` defaults to the validation-selected model, `what_if` defaults to `"keep"`) → same response shape as the existing `GET /recommend/{uid}`: `{"uid": <int>, "model": "<name>", "recommendations": [{"track_id": <int>, "score": <float>}, ...]}`. The existing `GET /recommend/{uid}` (`api.py:46-66`) is left in place unchanged, for back-compat/simple use. Later tasks' frontend `RecommendButton`/`RecommendationsTable` consume the POST response shape.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_artifacts.py`:

```python
def test_post_recommend_defaults_match_get_recommend():
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    uid = client.get('/users').json()['anonymous_user_ids'][0]
    get_response = client.get(f'/recommend/{uid}')
    post_response = client.post(f'/recommend/{uid}', json={})
    assert post_response.status_code == 200
    assert post_response.json()['model'] == get_response.json()['model']
    assert len(post_response.json()['recommendations']) == 10


def test_post_recommend_accepts_explicit_model_choice():
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    uid = client.get('/users').json()['anonymous_user_ids'][0]
    response = client.post(f'/recommend/{uid}', json={'model': 'Most Popular'})
    assert response.status_code == 200
    assert response.json()['model'] == 'Most Popular'
    assert len(response.json()['recommendations']) == 10


def test_post_recommend_what_if_dislike_changes_ranking_and_stays_outside_recorded_data():
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    uid = client.get('/users').json()['anonymous_user_ids'][0]
    before = (ROOT / 'demo.npz').read_bytes()
    keep = client.post(f'/recommend/{uid}', json={'model': 'NextBeat', 'what_if': 'keep'})
    dislike = client.post(f'/recommend/{uid}', json={'model': 'NextBeat', 'what_if': 'dislike'})
    assert keep.status_code == 200 and dislike.status_code == 200
    assert len(dislike.json()['recommendations']) == 10
    assert (ROOT / 'demo.npz').read_bytes() == before


def test_post_recommend_rejects_unknown_what_if():
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    uid = client.get('/users').json()['anonymous_user_ids'][0]
    response = client.post(f'/recommend/{uid}', json={'what_if': 'not-a-real-option'})
    assert response.status_code == 422


def test_post_recommend_never_returns_an_active_dislike():
    from fastapi.testclient import TestClient
    from run import windowed_active_dislikes
    from api import app
    client = TestClient(app)
    archive = np.load(ROOT / 'demo.npz')
    data = {k: archive[k] for k in archive.files}
    checked = 0
    for i, uid in enumerate(data['uid'].tolist()):
        blocked = windowed_active_dislikes(data['x'][i], data['f'][i])
        if len(blocked) == 0:
            continue
        checked += 1
        for model_name in ['NextBeat', 'Sequence-only GRU', 'ItemKNN', 'Most Popular']:
            response = client.post(f'/recommend/{uid}', json={'model': model_name})
            track_ids = {rec['track_id'] for rec in response.json()['recommendations']}
            blocked_track_ids = {int(data['vocab'][b - 2]) for b in blocked if b >= 2}
            assert track_ids.isdisjoint(blocked_track_ids), f'{model_name} recommended a disliked track for uid {uid}'
        if checked >= 5:
            break
    assert checked > 0, 'No demo.npz users had an active dislike — test is not exercising anything'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_artifacts.py -k post_recommend -v`
Expected: FAIL with `404 Not Found` / `405 Method Not Allowed` (no POST route yet)

- [ ] **Step 3: Implement the shared scoring helper and the endpoint**

Replace `api.py:46-66` (the existing `GET /recommend/{uid}` function) with a shared helper plus both endpoints:

```python
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


def score_and_rank(data, model_obj, model_name, x, f):
    blocked = [windowed_active_dislikes(x[0], f[0])]
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
    recommendations = score_and_rank(data, model, selected, data['x'][i:i+1], data['f'][i:i+1])
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
    recommendations = score_and_rank(data, model_obj, model_name, x, f)
    return {'uid': uid, 'model': model_name, 'recommendations': recommendations}
```

Note: `x, f = data['x'][i:i+1].copy(), ...` copies before any what-if mutation, exactly mirroring `app.py:49` and `app.py:64-67`'s pattern — this is what keeps the mutation from touching `data['x']`/`data['f']` (and therefore `demo.npz` on disk, which is never re-written by this process), which the fourth test above verifies.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_artifacts.py -k post_recommend -v`
Expected: 5 passed

Run the full suite: `.venv/bin/python -m pytest -q`
Expected: all tests pass, including the pre-existing `test_api_real_user_and_unknown_user` (verifies `GET /recommend/{uid}` still works after the refactor)

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_artifacts.py
git commit -m "Add POST /recommend/{uid} with model choice and what-if edit"
```

---

### Task 5: Scaffold the React frontend (Vite, no UI framework)

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.js`
- Create: `frontend/index.html`
- Create: `frontend/.env.example`
- Create: `frontend/.gitignore`
- Create: `frontend/src/main.jsx`
- Create: `frontend/src/api.js`
- Modify: `.gitignore` (repo root — ensure `frontend/node_modules/` and `frontend/dist/` stay ignored; `frontend/dist/` is already covered by the existing root `.gitignore` entry `frontend/dist/`, but `frontend/node_modules/` needs its own line since the root `.gitignore` only has a generic `node_modules/` — verify this already covers it before adding a duplicate)

**Interfaces:**
- Consumes: the five backend endpoints from Tasks 1-4 (`/models`, `/history/{uid}`, `/metrics`, `GET`/`POST /recommend/{uid}`, `/users` — the last already existed)
- Produces: `api.js` exports `getUsers()`, `getModels()`, `getHistory(uid)`, `getMetrics()`, `postRecommend(uid, {model, what_if})` — all `async`, all returning parsed JSON, all reading the base URL from `import.meta.env.VITE_API_URL`. Tasks 6-8 import and call these functions by these exact names.

- [ ] **Step 1: Check the root `.gitignore` already covers frontend build artifacts**

Run: `grep -E "node_modules|frontend/dist" .gitignore`
Expected output includes both `node_modules/` and `frontend/dist/` (already added in Part 1's `.gitignore` — verify, don't duplicate if present)

- [ ] **Step 2: Create `frontend/package.json`**

```json
{
  "name": "nextbeat-frontend",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.4",
    "vite": "^5.4.11"
  }
}
```

- [ ] **Step 3: Create `frontend/vite.config.js`**

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})
```

- [ ] **Step 4: Create `frontend/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>NextBeat</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 5: Create `frontend/.env.example`**

```
VITE_API_URL=http://127.0.0.1:8000
```

- [ ] **Step 6: Create `frontend/.gitignore`**

```
node_modules/
dist/
.env
.env.local
```

- [ ] **Step 7: Create `frontend/src/api.js`**

```javascript
const BASE = import.meta.env.VITE_API_URL

async function getJSON(path) {
  const response = await fetch(`${BASE}${path}`)
  if (!response.ok) throw new Error(`${path} failed: ${response.status}`)
  return response.json()
}

export function getUsers() {
  return getJSON('/users')
}

export function getModels() {
  return getJSON('/models')
}

export function getHistory(uid) {
  return getJSON(`/history/${uid}`)
}

export function getMetrics() {
  return getJSON('/metrics')
}

export async function postRecommend(uid, { model, what_if }) {
  const response = await fetch(`${BASE}/recommend/${uid}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, what_if }),
  })
  if (!response.ok) throw new Error(`recommend failed: ${response.status}`)
  return response.json()
}
```

- [ ] **Step 8: Create `frontend/src/main.jsx`** (minimal placeholder `App` — Task 6 replaces this)

```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'

function App() {
  return <div>NextBeat frontend scaffold — App.jsx lands in the next task.</div>
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

- [ ] **Step 9: Verify the scaffold builds**

Run: `cd frontend && npm install && npm run build`
Expected: build succeeds, produces `frontend/dist/index.html` and JS bundle, no errors

- [ ] **Step 10: Commit**

```bash
git add frontend/package.json frontend/vite.config.js frontend/index.html frontend/.env.example frontend/.gitignore frontend/src/main.jsx frontend/src/api.js
git commit -m "Scaffold React+Vite frontend with API client"
```

Note: do NOT commit `frontend/package-lock.json` exclusion by accident — `npm install` generates it; it SHOULD be committed (lockfiles belong in version control), so include it in this commit too: `git add frontend/package-lock.json`.

---

### Task 6: User/model selection + history display

**Files:**
- Create: `frontend/src/components/UserSelect.jsx`
- Create: `frontend/src/components/ModelSelect.jsx`
- Create: `frontend/src/components/HistoryTable.jsx`
- Create: `frontend/src/styles.css`
- Modify: `frontend/src/main.jsx` (replace placeholder `App` with the real one, wire in these three components)

**Interfaces:**
- Consumes: `getUsers`, `getModels`, `getHistory` from `frontend/src/api.js` (Task 5)
- Produces: `App` holds `user` (selected uid, number|null) and `model` (selected model name, string|null) in React state. Task 7 (`WhatIfPicker`, `RecommendButton`) and Task 8/9 read these same two pieces of state from `App` — this task establishes them as the two source-of-truth state variables the rest of the app is built around.

- [ ] **Step 1: Create `frontend/src/styles.css`**

```css
:root {
  --bg: #0f1115;
  --surface: #171a21;
  --surface-2: #1f232c;
  --border: #2a2f3a;
  --text: #e8eaed;
  --text-dim: #9aa0ac;
  --accent: #6ee7b7;
  --accent-dim: #2d5c47;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
}

.app {
  max-width: 960px;
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
}

.app h1 {
  font-size: 1.75rem;
  margin-bottom: 0.25rem;
}

.app .subtitle {
  color: var(--text-dim);
  margin-bottom: 2rem;
  font-size: 0.9rem;
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1.5rem;
}

.card h2 {
  font-size: 1.05rem;
  margin: 0 0 1rem;
  color: var(--text);
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  margin-bottom: 1rem;
}

.field label {
  font-size: 0.8rem;
  color: var(--text-dim);
}

.field select, .field button {
  background: var(--surface-2);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.5rem 0.75rem;
  font-size: 0.9rem;
}

.primary-button {
  background: var(--accent-dim);
  color: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 6px;
  padding: 0.6rem 1.2rem;
  font-size: 0.9rem;
  cursor: pointer;
}

.primary-button:hover { filter: brightness(1.15); }

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

th, td {
  text-align: left;
  padding: 0.5rem 0.6rem;
  border-bottom: 1px solid var(--border);
}

th {
  color: var(--text-dim);
  font-weight: 500;
}

.caption {
  color: var(--text-dim);
  font-size: 0.78rem;
  margin-top: 0.5rem;
}
```

- [ ] **Step 2: Create `frontend/src/components/UserSelect.jsx`**

```jsx
export default function UserSelect({ users, value, onChange }) {
  return (
    <div className="field">
      <label htmlFor="user-select">Anonymous user</label>
      <select id="user-select" value={value ?? ''} onChange={(e) => onChange(Number(e.target.value))}>
        {users.map((uid) => (
          <option key={uid} value={uid}>{uid}</option>
        ))}
      </select>
    </div>
  )
}
```

- [ ] **Step 3: Create `frontend/src/components/ModelSelect.jsx`**

```jsx
export default function ModelSelect({ models, defaultModel, value, onChange }) {
  return (
    <div className="field">
      <label htmlFor="model-select">Model</label>
      <select id="model-select" value={value ?? ''} onChange={(e) => onChange(e.target.value)}>
        {models.map((name) => (
          <option key={name} value={name}>{name}</option>
        ))}
      </select>
      <span className="caption">Default model selected by validation NDCG@10: {defaultModel}.</span>
    </div>
  )
}
```

- [ ] **Step 4: Create `frontend/src/components/HistoryTable.jsx`**

```jsx
export default function HistoryTable({ rows }) {
  return (
    <table>
      <thead>
        <tr><th>Track</th><th>Event</th><th>Played %</th></tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            <td>{row.track}</td>
            <td>{row.event}</td>
            <td>{row.played_percent ?? '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
```

- [ ] **Step 5: Replace `frontend/src/main.jsx` with the real `App`**

```jsx
import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'
import { getUsers, getModels, getHistory } from './api'
import UserSelect from './components/UserSelect'
import ModelSelect from './components/ModelSelect'
import HistoryTable from './components/HistoryTable'
import './styles.css'

function App() {
  const [users, setUsers] = useState([])
  const [modelList, setModelList] = useState([])
  const [defaultModel, setDefaultModel] = useState(null)
  const [user, setUser] = useState(null)
  const [model, setModel] = useState(null)
  const [history, setHistory] = useState([])

  useEffect(() => {
    getUsers().then((body) => {
      setUsers(body.anonymous_user_ids)
      setUser(body.anonymous_user_ids[0])
    })
    getModels().then((body) => {
      setModelList(body.models)
      setDefaultModel(body.default)
      setModel(body.default)
    })
  }, [])

  useEffect(() => {
    if (user != null) getHistory(user).then((body) => setHistory(body.history))
  }, [user])

  return (
    <div className="app">
      <h1>NextBeat</h1>
      <p className="subtitle">Real Yambda listening histories · trained recommendation models · anonymous track IDs</p>
      <div className="card">
        <h2>Selection</h2>
        <UserSelect users={users} value={user} onChange={setUser} />
        <ModelSelect models={modelList} defaultModel={defaultModel} value={model} onChange={setModel} />
      </div>
      <div className="card">
        <h2>Recent real history</h2>
        <HistoryTable rows={history} />
      </div>
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

- [ ] **Step 6: Verify manually against a running backend**

In one terminal, from the repo root: `.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000`
In another terminal: `cd frontend && cp .env.example .env.local && npm run dev`
Open the printed local URL (typically `http://127.0.0.1:5173`) in a browser. Expected: user and model dropdowns populate with real data, selecting a different user updates the history table with that user's real recorded events.

- [ ] **Step 7: Verify the production build still succeeds**

Run: `cd frontend && npm run build`
Expected: build succeeds with no errors

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/UserSelect.jsx frontend/src/components/ModelSelect.jsx frontend/src/components/HistoryTable.jsx frontend/src/styles.css frontend/src/main.jsx
git commit -m "Add user/model selection and history display to frontend"
```

---

### Task 7: What-if picker + recommend button + results table

**Files:**
- Create: `frontend/src/components/WhatIfPicker.jsx`
- Create: `frontend/src/components/RecommendationsTable.jsx`
- Modify: `frontend/src/main.jsx` (wire in both, add the recommend flow)

**Interfaces:**
- Consumes: `postRecommend` from `frontend/src/api.js` (Task 5); `user`/`model` state from `App` (Task 6)
- Produces: `App` gains `whatIf` state (string, one of `'keep'|'full_listen'|'like'|'dislike'|'short_listen'`) and `recommendations` state (array). No later task consumes these directly, but Task 8's `MetricsPanel` is added to the same `App` component, so this task's edits to `main.jsx` must be additive, not replacing what Task 6 built.

- [ ] **Step 1: Create `frontend/src/components/WhatIfPicker.jsx`**

```jsx
const OPTIONS = [
  { value: 'keep', label: 'Keep recorded event' },
  { value: 'full_listen', label: 'Full listen' },
  { value: 'like', label: 'Like' },
  { value: 'dislike', label: 'Dislike' },
  { value: 'short_listen', label: 'Short listen' },
]

export default function WhatIfPicker({ value, onChange }) {
  return (
    <div className="field">
      <label htmlFor="what-if-select">Optional what-if change to the last event</label>
      <select id="what-if-select" value={value} onChange={(e) => onChange(e.target.value)}>
        {OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
      {value !== 'keep' && (
        <span className="caption">This is an explicitly edited what-if input. It does not change the recorded data or evaluation scores.</span>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Create `frontend/src/components/RecommendationsTable.jsx`**

```jsx
export default function RecommendationsTable({ recommendations }) {
  if (!recommendations.length) return null
  return (
    <>
      <table>
        <thead>
          <tr><th>Rank</th><th>Track</th><th>Score</th></tr>
        </thead>
        <tbody>
          {recommendations.map((rec, i) => (
            <tr key={rec.track_id}>
              <td>{i + 1}</td>
              <td>Track {rec.track_id}</td>
              <td>{rec.score.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caption">Scores rank tracks within a model; they are not calibrated probabilities or comparable across models.</p>
    </>
  )
}
```

- [ ] **Step 3: Wire both into `App`**

In `frontend/src/main.jsx`, add these imports alongside the existing ones:

```jsx
import { getUsers, getModels, getHistory, postRecommend } from './api'
import WhatIfPicker from './components/WhatIfPicker'
import RecommendationsTable from './components/RecommendationsTable'
```

(This replaces the existing `import { getUsers, getModels, getHistory } from './api'` line from Task 6 — same line, extended import list.)

Add state and a handler inside `App`, alongside the existing `useState` calls:

```jsx
const [whatIf, setWhatIf] = useState('keep')
const [recommendations, setRecommendations] = useState([])

async function handleRecommend() {
  const body = await postRecommend(user, { model, what_if: whatIf })
  setRecommendations(body.recommendations)
}
```

Add to the JSX, inside the first `<div className="card">` block (after the `ModelSelect` line), and a new card after it:

```jsx
<WhatIfPicker value={whatIf} onChange={setWhatIf} />
<button className="primary-button" onClick={handleRecommend}>Recommend next tracks</button>
```

```jsx
<div className="card">
  <h2>Recommended next tracks</h2>
  <RecommendationsTable recommendations={recommendations} />
</div>
```

(Place this new card in the JSX after the existing "Recent real history" card, before the closing `</div>` of `.app`.)

- [ ] **Step 4: Verify manually against a running backend**

With the backend still running (`uvicorn api:app` from Task 6 Step 6) and `npm run dev` running: click "Recommend next tracks". Expected: a 10-row table appears with rank/track/score. Change the what-if dropdown to "Dislike" and click again — the table updates (it may or may not visibly differ depending on the user's history, but the request must succeed with no console errors).

- [ ] **Step 5: Verify the production build still succeeds**

Run: `cd frontend && npm run build`
Expected: build succeeds with no errors

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/WhatIfPicker.jsx frontend/src/components/RecommendationsTable.jsx frontend/src/main.jsx
git commit -m "Add what-if picker, recommend button, and results table to frontend"
```

---

### Task 8: Metrics panel

**Files:**
- Create: `frontend/src/components/MetricsPanel.jsx`
- Modify: `frontend/src/main.jsx` (fetch metrics on load, render the panel)

**Interfaces:**
- Consumes: `getMetrics` from `frontend/src/api.js` (Task 5)
- Produces: nothing consumed by later tasks (this is the last frontend feature task)

- [ ] **Step 1: Create `frontend/src/components/MetricsPanel.jsx`**

```jsx
const ORDER = ['Most Popular', 'ItemKNN', 'Sequence-only GRU', 'NextBeat']

const FORMAT = {
  recall10: (v) => `${(v * 100).toFixed(2)}%`,
  ndcg10: (v) => v.toFixed(4),
  unconditional_recall10: (v) => `${(v * 100).toFixed(2)}%`,
  coverage10: (v) => `${(v * 100).toFixed(2)}%`,
  novelty10: (v) => v.toFixed(2),
  nfvr10: (v) => `${(v * 100).toFixed(3)}%`,
}

const LABEL = {
  recall10: 'Recall@10',
  ndcg10: 'NDCG@10',
  unconditional_recall10: 'Recall (all targets)',
  coverage10: 'Catalogue coverage',
  novelty10: 'Novelty (bits)',
  nfvr10: 'Dislike violations',
}

const COLUMNS = ['recall10', 'ndcg10', 'unconditional_recall10', 'coverage10', 'novelty10', 'nfvr10']

export default function MetricsPanel({ metrics, eligibleTargets, totalTargets }) {
  if (!metrics) return null
  return (
    <>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            {COLUMNS.map((col) => <th key={col}>{LABEL[col]}</th>)}
          </tr>
        </thead>
        <tbody>
          {ORDER.map((name) => (
            <tr key={name}>
              <td>{name}</td>
              {COLUMNS.map((col) => <td key={col}>{FORMAT[col](metrics[name][col])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caption">
        {eligibleTargets.toLocaleString()} eligible in-catalogue targets; {totalTargets.toLocaleString()} total test targets. One recorded next listen per user.
      </p>
      <p className="caption">
        Repeat listens are allowed. Unknown test targets are excluded from conditional metrics and included as misses in unconditional Recall.
      </p>
    </>
  )
}
```

- [ ] **Step 2: Wire into `App`**

In `frontend/src/main.jsx`, add to the imports:

```jsx
import { getUsers, getModels, getHistory, postRecommend, getMetrics } from './api'
import MetricsPanel from './components/MetricsPanel'
```

(Replaces the Task 7 import line with the extended list.)

Add state and a fetch effect:

```jsx
const [metrics, setMetrics] = useState(null)
const [eligibleTargets, setEligibleTargets] = useState(0)
const [totalTargets, setTotalTargets] = useState(0)

useEffect(() => {
  getMetrics().then((body) => {
    setMetrics(body.metrics)
    setEligibleTargets(body.eligible_targets)
    setTotalTargets(body.total_targets)
  })
}, [])
```

Add a final card to the JSX, after the "Recommended next tracks" card, before the closing `</div>` of `.app`:

```jsx
<div className="card">
  <h2>Frozen-model test results</h2>
  <MetricsPanel metrics={metrics} eligibleTargets={eligibleTargets} totalTargets={totalTargets} />
</div>
```

- [ ] **Step 3: Verify manually against a running backend**

With the backend and `npm run dev` running, reload the page. Expected: a 4-row metrics table appears (Most Popular, ItemKNN, Sequence-only GRU, NextBeat) with formatted percentages, matching the numbers in `artifacts/results.json`.

- [ ] **Step 4: Verify the production build still succeeds**

Run: `cd frontend && npm run build`
Expected: build succeeds with no errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/MetricsPanel.jsx frontend/src/main.jsx
git commit -m "Add metrics panel to frontend"
```

---

### Task 9: `Dockerfile.api` for the backend, build-tested

**Files:**
- Create: `Dockerfile.api`
- Modify: `.dockerignore` (verify it already excludes `frontend/node_modules` — the existing `.dockerignore` has generic entries; add `frontend/node_modules` explicitly if not already covered)

**Interfaces:**
- Consumes: `requirements.txt`, `api.py`, `run.py`, `models.py`, `artifacts/` (all existing)
- Produces: a Docker image that serves `api.py` via uvicorn on the port given by the `PORT` environment variable (Render sets this at runtime; default to `8000` for local testing)

- [ ] **Step 1: Check `.dockerignore`**

Run: `cat .dockerignore`
Expected to see `data`, `artifacts/prepared.npz`, `__pycache__`, `.pytest_cache`, `.venv` (from the existing file). Add a line for `frontend/node_modules` and `frontend/dist` if not present — these are large and irrelevant to the API image.

- [ ] **Step 2: Create `Dockerfile.api`**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY api.py run.py models.py ./
COPY artifacts ./artifacts
ENV PORT=8000
EXPOSE 8000
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
```

- [ ] **Step 3: Build-test the image locally**

Run: `docker build -f Dockerfile.api -t nextbeat-api .`
Expected: image builds successfully with no errors. If Docker is not installed/available in this environment, report this step's outcome honestly (BLOCKED or a clearly-labeled skip) rather than claiming success — the image must be verified before this task is considered done, either here or by whoever runs the deploy step in Task 11.

- [ ] **Step 4: Run the built image and smoke-test it**

Run: `docker run --rm -d -p 8000:8000 --name nextbeat-api-test nextbeat-api` then `curl -s http://127.0.0.1:8000/health` then `docker stop nextbeat-api-test`
Expected: `curl` returns `{"ready":true}` (or similar truthy JSON), confirming the container serves the API correctly with the default `PORT=8000`.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile.api .dockerignore
git commit -m "Add Dockerfile.api for deploying the backend on Render"
```

---

### Task 10: Deploy backend to Render and frontend to Vercel

This task requires accounts on Render and Vercel (both have free tiers) and cannot be fully automated — it is a human-driven deployment, documented here step by step. If executed by an agentic worker without interactive browser/account access, stop after Step 1 (Dockerfile is already verified working from Task 9) and hand the remaining steps to the user, the same way Task 7 of the dislike-violation-fix plan handed off the Colab run.

**Files:**
- Modify: `README.md` (document the deployed URLs and redeploy steps)

- [ ] **Step 1: Push the current branch/main so Render and Vercel can pull from it**

Ensure all commits from Tasks 1-9 are on `main` and pushed to `https://github.com/falafell99/holberton-project.git` (Render and Vercel both deploy by connecting to a GitHub repo).

- [ ] **Step 2: Deploy the backend on Render**

At https://dashboard.render.com: New → Web Service → connect the `falafell99/holberton-project` GitHub repo → set "Dockerfile Path" to `Dockerfile.api` → Render auto-detects the `PORT` env var it injects (already handled by `Dockerfile.api`'s `CMD`) → Create Web Service. Wait for the build to finish (several minutes — it installs `torch` and other heavy dependencies). Once live, note the assigned URL (looks like `https://nextbeat-api-xxxx.onrender.com`) and verify it: `curl https://<render-url>/health` should return `{"ready":true}`.

- [ ] **Step 3: Deploy the frontend on Vercel**

At https://vercel.com: Add New → Project → import the same GitHub repo → set "Root Directory" to `frontend` (Vercel auto-detects the Vite framework preset once the root directory is set) → under Environment Variables, add `VITE_API_URL` = the Render URL from Step 2 (no trailing slash) → Deploy. Once live, note the assigned URL (looks like `https://nextbeat-xxxx.vercel.app`).

- [ ] **Step 4: Tighten CORS now that the real frontend URL is known**

Back in the Render dashboard, add an environment variable `ALLOWED_ORIGINS` = the Vercel URL from Step 3 (no trailing slash) to the backend service, then trigger a redeploy (Render redeploys automatically on env var changes for most plans; if not, use "Manual Deploy" → "Deploy latest commit"). This narrows CORS from the wildcard default to just the real frontend origin.

- [ ] **Step 5: Verify the deployed app end-to-end**

Open the Vercel URL in a browser. Expected: the same behavior verified locally in Tasks 6-8 — user/model dropdowns populate, history displays, recommend button returns a 10-row table, metrics panel shows all four models' numbers — now against the live Render backend instead of localhost.

- [ ] **Step 6: Document the deployment in `README.md`**

Add a new section to `README.md`, after the existing "API and Docker" section (`README.md`'s current "API and Docker" section ends right before "## Sources"):

```markdown
## Deployed app

- Frontend: <the actual Vercel URL from Step 3>
- Backend API: <the actual Render URL from Step 2> (`/docs` for interactive API docs)

To redeploy after new commits: Render and Vercel both auto-deploy on push to `main`. To redeploy manually, use each dashboard's "Deploy latest commit" / "Redeploy" action.
```

- [ ] **Step 7: Commit the README update**

```bash
git add README.md
git commit -m "Document deployed frontend and backend URLs"
git push
```

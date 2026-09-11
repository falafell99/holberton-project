# Dislike-Violation Fix (Part 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recommendations from all four models (Most Popular, ItemKNN, Sequence-only GRU, NextBeat) never surface a track the user has actively disliked, without a separate full retrain — then (optionally, time permitting) improve NextBeat's own scoring behavior around disliked tracks and/or train on a larger cohort.

**Architecture:** Add an optional `blocked` per-row mask to the existing `top10()` ranking function in `run.py` and thread it through every path that calls it (batch evaluation in `train()`, live inference in `app.py`/`api.py`). Add a new lightweight `run.py evaluate` CLI stage that recomputes `results.json` from already-trained model weights + `prepared.npz`, without re-running the training loop. Add a `windowed_active_dislikes()` helper for the fixed-length (20-event) context windows used at live-inference time, mirroring the existing `active_dislikes()` helper that works on full flat history arrays.

**Tech Stack:** Python, NumPy, PyTorch, SciPy sparse, pytest (existing stack — no new dependencies for Part 1).

**Spec:** [docs/superpowers/specs/2026-09-11-finish-nextbeat-design.md](../specs/2026-09-11-finish-nextbeat-design.md) — Part 1 (G1 required, G2/G3 optional stretch).

## Global Constraints

- G1 (the inference-time filter) must ship regardless of time; G2 and G3 are optional and are cut in that order (G3 first, then G2) if time runs short.
- `blocked=None` is the default on every function signature this plan touches (`top10`, `neural_ranks`, `knn_ranks`) — existing callers that don't pass it must keep working unchanged.
- "Actively disliked" = the most recent `dislike`/`undislike` event for that track in the relevant history is `dislike` (an `undislike` after a `dislike` removes it from the blocked set). Feature column indices are fixed by `EVENTS = ['listen', 'like', 'dislike', 'unlike', 'undislike']` in `run.py`, offset by 1 for the leading play-ratio column: index 3 = dislike, index 5 = undislike.
- Regenerating `artifacts/results.json` / `experiments/*/results.json` with corrected numbers requires the full `download.py` → `run.py prepare` → `run.py evaluate` pipeline (a multi-GB dataset download is unavoidable — `prepared.npz` and the raw parquet are not shipped in this repo). This must run on **Google Colab** per the user's stated hardware constraint (Mac, no GPU) — do not attempt this locally.
- The new `run.py evaluate` stage must NOT retrain any model — it only loads existing `.pt` weights and recomputes metrics. This keeps the required G1 fix cheap even though full data must still be downloaded/prepared.
- Task 6's real-artifact test runs entirely against files already in this repo (`artifacts/demo.npz`, `artifacts/*.pt`, `artifacts/itemknn.npz`) — it needs no download and no Colab.

---

### Task 1: Add blocked-row masking to `top10()`

**Files:**
- Modify: `run.py:128-131` (the `top10` function)
- Test: `tests/test_real_pipeline.py` (add new test near `test_reserved_ids_never_recommended`, line 40-42)

**Interfaces:**
- Consumes: nothing new
- Produces: `top10(scores, blocked=None)` — `blocked` is `None` or a sequence with one entry per row of `scores`; each entry is an array (possibly empty) of item ids to exclude from that row's ranking. Every later task in this plan calls `top10` through this signature.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_real_pipeline.py` (this file already imports `top10` from `run` on line 4):

```python
def test_top10_never_returns_blocked_items():
    scores = np.arange(15, dtype=float)[None]
    ranks = top10(scores.copy(), blocked=[np.array([12, 13, 14])])
    assert set(ranks[0].tolist()).isdisjoint({12, 13, 14})


def test_top10_blocked_none_is_backward_compatible():
    scores = np.arange(15, dtype=float)[None]
    ranks = top10(scores.copy())
    assert ranks.shape == (1, 10) and ranks.min() >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_real_pipeline.py -k top10_never_returns_blocked -v`
Expected: FAIL with `TypeError: top10() got an unexpected keyword argument 'blocked'`

- [ ] **Step 3: Implement the minimal change**

Replace `run.py:128-131`:

```python
def top10(scores, blocked=None):
    scores[:, :2] = -np.inf
    if blocked is not None:
        for row, ids in zip(scores, blocked):
            if len(ids):
                row[np.asarray(ids, dtype=np.int64)] = -np.inf
    selected = np.argpartition(scores, -10, axis=1)[:, -10:]
    return np.take_along_axis(selected, np.argsort(-np.take_along_axis(scores, selected, axis=1), axis=1), axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_real_pipeline.py -k "top10_never_returns_blocked or top10_blocked_none" -v`
Expected: 2 passed

Also re-run the full existing suite to confirm nothing broke:
Run: `python -m pytest tests/test_real_pipeline.py -v`
Expected: all previously-passing tests still pass (this file is only skipped entirely if `artifacts/demo.npz` is missing, which it is not in this repo)

- [ ] **Step 5: Commit**

```bash
git add run.py tests/test_real_pipeline.py
git commit -m "Add optional blocked-row masking to top10()

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Add `windowed_active_dislikes()` for fixed-length context windows

**Files:**
- Modify: `run.py` (add new function near `active_dislikes`, `run.py:111-125`)
- Test: `tests/test_real_pipeline.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `windowed_active_dislikes(items, features)` — `items` is a 1D array of length 20 (0 = padding), `features` is a `(20, 6)` array in the same column order as `EVENTS` (offset by 1 for play ratio). Returns a sorted `np.int32` array of actively-disliked item ids within that window. Task 5 (live inference) calls this directly.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_real_pipeline.py`:

```python
def test_windowed_active_dislikes_tracks_reversal():
    from run import windowed_active_dislikes
    items = np.array([0, 2, 3, 2])
    f = np.zeros((4, 6))
    f[1, 3] = 1  # dislike item 2
    f[2, 3] = 1  # dislike item 3
    f[3, 5] = 1  # undislike item 2 (reappears in the window, so it's un-disliked)
    result = windowed_active_dislikes(items, f)
    assert result.tolist() == [3]


def test_windowed_active_dislikes_ignores_padding():
    from run import windowed_active_dislikes
    items = np.array([0, 0, 0, 2])
    f = np.zeros((4, 6))
    f[3, 3] = 1
    assert windowed_active_dislikes(items, f).tolist() == [2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_real_pipeline.py -k windowed_active_dislikes -v`
Expected: FAIL with `ImportError: cannot import name 'windowed_active_dislikes'`

- [ ] **Step 3: Implement the minimal function**

Add to `run.py` directly after `active_dislikes` (after line 125):

```python
def windowed_active_dislikes(items, features):
    """Same active-set logic as active_dislikes(), scoped to one fixed-length
    context window (items/features rows as stored in demo.npz), for live inference."""
    active = set()
    for item, row in zip(items, features):
        item = int(item)
        if item == 0:
            continue
        if row[3] == 1:
            active.add(item)
        elif row[5] == 1:
            active.discard(item)
    return np.array(sorted(active), dtype=np.int32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_real_pipeline.py -k windowed_active_dislikes -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add run.py tests/test_real_pipeline.py
git commit -m "Add windowed_active_dislikes() for live-inference context windows

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Thread `blocked` through batch ranking (`neural_ranks`, `knn_ranks`, Most Popular) in `train()`

**Files:**
- Modify: `run.py:134-142` (`neural_ranks`), `run.py:168-175` (`knn_ranks`), `run.py:192-198,264` (`train`, Most Popular + `results[name]` line for test-set metrics)
- Test: `tests/test_real_pipeline.py`

**Interfaces:**
- Consumes: `top10(scores, blocked=None)` from Task 1
- Produces: `neural_ranks(model, q, blocked=None)`, `knn_ranks(knn, q, counts, blocked=None)` — both keep their existing return shape `(len(q['y']), 10)`. Task 4's new `evaluate` stage calls both with `blocked` set.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_real_pipeline.py` (uses the small synthetic `NextBeat` model pattern already used in `test_feedback_changes_neural_state_but_not_ablation`):

```python
def test_neural_ranks_respects_blocked():
    from run import neural_ranks
    torch.manual_seed(0)
    model = NextBeat(12, feedback=False)
    q = dict(x=np.zeros((1, 2), dtype=int), f=np.zeros((1, 2, 6)), y=np.array([2]))
    unblocked = neural_ranks(model, q)
    blocked_id = int(unblocked[0, 0])
    ranks = neural_ranks(model, q, blocked=[np.array([blocked_id])])
    assert blocked_id not in ranks[0].tolist()


def test_knn_ranks_respects_blocked():
    from run import knn_ranks
    import scipy.sparse as sp
    knn = sp.csr_matrix(np.ones((14, 14)))
    counts = np.ones(14, dtype=np.float32)
    q = dict(x=np.array([[2, 3]]))
    unblocked = knn_ranks(knn, q, counts)
    blocked_id = int(unblocked[0, 0])
    ranks = knn_ranks(knn, q, counts, blocked=[np.array([blocked_id])])
    assert blocked_id not in ranks[0].tolist()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_real_pipeline.py -k "neural_ranks_respects_blocked or knn_ranks_respects_blocked" -v`
Expected: FAIL with `TypeError: ...got an unexpected keyword argument 'blocked'`

- [ ] **Step 3: Implement the minimal changes**

Replace `run.py:134-142` (`neural_ranks`):

```python
def neural_ranks(model, q, blocked=None):
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(q['y']), 128):
            x = torch.tensor(q['x'][i:i+128], dtype=torch.long)
            f = torch.tensor(q['f'][i:i+128], dtype=torch.float32)
            chunk = blocked[i:i+128] if blocked is not None else None
            rows.append(top10(model.scores(x, f).numpy(), chunk))
    return np.concatenate(rows)
```

Replace `run.py:168-175` (`knn_ranks`):

```python
def knn_ranks(knn, q, counts, blocked=None):
    output = []
    for idx, x in enumerate(q['x']):
        seen = x[x >= 2]
        scores = np.asarray(knn[seen].sum(axis=0)).ravel() if len(seen) else counts.copy()
        scores = scores + counts / max(counts.max(), 1) * 1e-6
        chunk = blocked[idx:idx+1] if blocked is not None else None
        output.append(top10(scores[None], chunk)[0])
    return np.array(output)
```

In `train()`, replace `run.py:194-195` (Most Popular):

```python
    pop = top10(np.tile(counts, (len(test['y']), 1)), blocked)
    results['Most Popular'] = metrics(pop, test, counts, blocked)
```

And in `train()`, replace `run.py:198`:

```python
    results['ItemKNN'] = metrics(knn_ranks(knn, test, counts, blocked), test, counts, blocked)
```

And in `train()`, replace `run.py:264` (only the test-set line — leave the validation-set `neural_ranks(model, valid)` call on line 246 unchanged, since validation is for epoch/checkpoint selection, not served recommendations):

```python
        results[name] = metrics(neural_ranks(model, test, blocked), test, counts, blocked)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_real_pipeline.py -k "neural_ranks_respects_blocked or knn_ranks_respects_blocked" -v`
Expected: 2 passed

Run the full suite to confirm `train()`'s internal call sites still parse/import correctly:
Run: `python -m pytest tests/test_real_pipeline.py -v`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add run.py tests/test_real_pipeline.py
git commit -m "Thread blocked-item masking through batch ranking paths

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Add `run.py evaluate` stage (recompute results.json without retraining)

**Files:**
- Modify: `run.py` (add new `evaluate(args)` function after `train()`, before the `if __name__ == '__main__':` block at line 280; extend the argparse block at `run.py:280-302`)
- Test: `tests/test_real_pipeline.py`

**Interfaces:**
- Consumes: `queries`, `active_dislikes`, `metrics`, `top10`, `neural_ranks`, `knn_ranks` (all from this file), `NextBeat` from `models.py`
- Produces: a `evaluate` CLI stage (`python run.py evaluate --out artifacts`) that overwrites `results.json` in `--out`. Nothing later in this plan calls `evaluate()` as a Python function — it's invoked only via CLI, on Colab, per Task 7.

- [ ] **Step 1: Write the failing test**

This function needs `prepared.npz`, which isn't shipped in this repo (see Global Constraints), so it can't be exercised end-to-end here. Instead, test that the CLI wiring and argument validation are correct without needing real data:

Add to `tests/test_real_pipeline.py`:

```python
def test_evaluate_stage_is_registered_in_cli():
    import subprocess
    result = subprocess.run(['python', 'run.py', '--help'], capture_output=True, text=True,
                            cwd=str(ROOT.parent))
    assert 'evaluate' in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_real_pipeline.py -k evaluate_stage_is_registered -v`
Expected: FAIL — `evaluate` not yet in the `stage` choices, so `--help` output won't list it

- [ ] **Step 3: Implement the minimal change**

Add to `run.py` immediately after the `train()` function ends (after line 277, before `if __name__ == '__main__':` on line 280):

```python
def evaluate(args):
    """Recompute results.json from already-trained weights. Never trains."""
    out = Path(args.out)
    data = np.load(out / 'prepared.npz')
    data = {k: data[k] for k in data.files}
    new_time = np.r_[True, (data['uid'][1:] != data['uid'][:-1]) | (data['ts'][1:] != data['ts'][:-1])]
    data['prefix'] = np.maximum.accumulate(np.where(new_time, np.arange(len(new_time)), 0))
    manifest = json.loads((out / 'manifest.json').read_text())
    t1, t2 = manifest['cutoffs']
    test = queries(data, t2, np.iinfo(np.uint32).max)
    blocked = active_dislikes(data, data['prefix'][test['positions']])
    counts = data['counts']
    results = {}
    pop = top10(np.tile(counts, (len(test['y']), 1)), blocked)
    results['Most Popular'] = metrics(pop, test, counts, blocked)
    knn = sp.load_npz(out / 'itemknn.npz')
    results['ItemKNN'] = metrics(knn_ranks(knn, test, counts, blocked), test, counts, blocked)
    for feedback, filename, name in [(False, 'sequence.pt', 'Sequence-only GRU'),
                                      (True, 'nextbeat.pt', 'NextBeat')]:
        model = NextBeat(len(counts), feedback=feedback)
        model.load_state_dict(torch.load(out / filename, map_location='cpu', weights_only=True))
        model.eval()
        results[name] = metrics(neural_ranks(model, test, blocked), test, counts, blocked)
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
```

Update the argparse block at `run.py:280-282`:

```python
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'train', 'evaluate'])
```

Add the dispatch branch at the end of the file (`run.py:296-302`, inside the `if a.stage == 'train':` / `else:` block) — replace it with:

```python
    if a.stage == 'train':
        # Separate output folders may train concurrently; one folder has one writer.
        from filelock import FileLock
        with FileLock(str(Path(a.out) / '.training.lock')):
            train(a)
    elif a.stage == 'evaluate':
        evaluate(a)
    else:
        prepare(a)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_real_pipeline.py -k evaluate_stage_is_registered -v`
Expected: 1 passed

Run the full suite:
Run: `python -m pytest tests/test_real_pipeline.py -v`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add run.py tests/test_real_pipeline.py
git commit -m "Add run.py evaluate stage to recompute results.json without retraining

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Apply the dislike filter to live inference in `app.py` and `api.py`

**Files:**
- Modify: `app.py:9,68-77` (import line and the recommend-button block)
- Modify: `api.py:8,46-65` (import line and the `recommend` endpoint)

**Interfaces:**
- Consumes: `top10(scores, blocked=None)` from Task 1, `windowed_active_dislikes(items, features)` from Task 2
- Produces: nothing new consumed later — this is the user-facing end of the chain

- [ ] **Step 1: Update `app.py` imports and recommend block**

Change `app.py:10`:

```python
from run import top10, windowed_active_dislikes
```

Replace `app.py:68-77`:

```python
if st.button('Recommend next tracks', type='primary'):
    blocked = [windowed_active_dislikes(x[0], f[0])]
    if choice in models:
        with torch.no_grad():
            scores = models[choice].scores(torch.tensor(x, dtype=torch.long), torch.tensor(f, dtype=torch.float32)).numpy()
    elif choice == 'ItemKNN':
        scores = np.asarray(knn[x[0][x[0] >= 2]].sum(axis=0))
        scores += data['counts'][None] / data['counts'].max() * 1e-6
    else:
        scores = data['counts'][None].copy()
    ranking = top10(scores, blocked)[0]
```

- [ ] **Step 2: Update `api.py` imports and recommend endpoint**

Change `api.py:9`:

```python
from run import top10, windowed_active_dislikes
```

Replace `api.py:46-65` (the whole `recommend` function body from `data, model, selected = resources()` onward):

```python
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
```

- [ ] **Step 3: Verify by running the existing real-artifact tests**

Run: `python -m pytest tests/test_real_pipeline.py -v`
Expected: `test_api_real_user_and_unknown_user`, `test_streamlit_real_prediction`, and `test_streamlit_all_models_and_what_if_preserve_recorded_evidence` all still pass (they exercise these exact code paths against real saved artifacts already in this repo)

- [ ] **Step 4: Commit**

```bash
git add app.py api.py
git commit -m "Apply dislike filter to live inference in app.py and api.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Add a real-artifact regression test proving the fix works end-to-end

**Files:**
- Test: `tests/test_real_pipeline.py`

This test needs no download and no Colab — it runs entirely against `artifacts/demo.npz`, `artifacts/*.pt`, and `artifacts/itemknn.npz`, which already exist in this repo.

**Interfaces:**
- Consumes: `windowed_active_dislikes` (Task 2), `top10` (Task 1), `NextBeat` (`models.py`)
- Produces: nothing consumed later

- [ ] **Step 1: Write the test**

Add to `tests/test_real_pipeline.py`:

```python
def test_no_model_ever_recommends_an_active_dislike():
    from run import windowed_active_dislikes
    from models import NextBeat
    data = np.load(ROOT / 'demo.npz')
    knn = None
    import scipy.sparse as sp
    knn = sp.load_npz(ROOT / 'itemknn.npz')
    models = {}
    for name, filename, feedback in [('NextBeat', 'nextbeat.pt', True), ('Sequence-only GRU', 'sequence.pt', False)]:
        model = NextBeat(len(data['counts']), feedback=feedback)
        model.load_state_dict(torch.load(ROOT / filename, map_location='cpu', weights_only=True))
        model.eval()
        models[name] = model
    checked = 0
    for i in range(len(data['uid'])):
        blocked = windowed_active_dislikes(data['x'][i], data['f'][i])
        if len(blocked) == 0:
            continue
        checked += 1
        x = data['x'][i:i+1]
        f = data['f'][i:i+1]
        with torch.no_grad():
            for name, model in models.items():
                scores = model.scores(torch.tensor(x, dtype=torch.long), torch.tensor(f, dtype=torch.float32)).numpy()
                ranking = top10(scores, [blocked])[0]
                assert set(ranking.tolist()).isdisjoint(set(blocked.tolist())), f'{name} recommended a disliked track'
        seen = x[0][x[0] >= 2]
        knn_scores = np.asarray(knn[seen].sum(axis=0)) if len(seen) else data['counts'][None].copy()
        knn_scores = knn_scores + data['counts'][None] / data['counts'].max() * 1e-6
        ranking = top10(knn_scores, [blocked])[0]
        assert set(ranking.tolist()).isdisjoint(set(blocked.tolist())), 'ItemKNN recommended a disliked track'
        pop_scores = data['counts'][None].copy()
        ranking = top10(pop_scores, [blocked])[0]
        assert set(ranking.tolist()).isdisjoint(set(blocked.tolist())), 'Most Popular recommended a disliked track'
    assert checked > 0, 'No demo.npz users had an active dislike — test is not exercising anything'
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_real_pipeline.py -k no_model_ever_recommends_an_active_dislike -v`
Expected: PASS (Tasks 1, 2, and 5 already made every scoring path respect the mask, so this should pass on the first run — if it fails, re-check Task 5's edits)

- [ ] **Step 3: Run the full test suite one more time**

Run: `python -m pytest tests/test_real_pipeline.py tests/test_artifacts.py -v`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_real_pipeline.py
git commit -m "Add real-artifact regression test for the dislike filter

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Regenerate `results.json` on Colab and update README

This task runs on Google Colab (no local GPU/CPU budget spent), per the user's constraint. It produces the corrected NFVR numbers this whole plan exists to fix.

**Files:**
- Create: `colab/evaluate_nfvr_fix.ipynb` (new notebook, committed to the repo so it's reproducible)
- Modify: `README.md` (the Results section, `README.md:44-52`)
- Modify: `artifacts/results.json`, `experiments/seed_43/results.json`, `experiments/seed_44/results.json` (regenerated, not hand-edited)

- [ ] **Step 1: Write the Colab notebook**

Create `colab/evaluate_nfvr_fix.ipynb` with these cells (as notebook JSON — use `nbformat` conventions: a markdown cell followed by code cells):

Cell 1 (markdown):
```
# NextBeat — regenerate results.json with the dislike-violation fix
Run all cells top to bottom. Needs no GPU. Takes a while only because of the dataset download (multi-GB), not because of training — this notebook never trains, it only re-evaluates already-trained weights.
```

Cell 2 (code — clone and install):
```python
!git clone https://github.com/falafell99/holberton-project.git
%cd holberton-project
!pip install -q -r requirements.txt
```

Cell 3 (code — download the pinned dataset):
```python
!python download.py
```

Cell 4 (code — prepare, using the same cohort settings as the shipped artifacts):
```python
import json
manifest = json.load(open('artifacts/manifest.json'))
print('Reproducing seed', manifest['seed'], 'users', manifest['selected_users'], 'catalog', manifest['catalog'])
!python run.py prepare --raw data/raw/multi_event.parquet --out artifacts --users {manifest['selected_users']} --catalog {manifest['catalog']} --seed {manifest['seed']}
```

Cell 5 (code — evaluate seed 42, no retraining):
```python
!python run.py evaluate --out artifacts
```

Cell 6 (code — repeat for seeds 43 and 44):
```python
for seed_dir in ['experiments/seed_43', 'experiments/seed_44']:
    seed_manifest = json.load(open(f'{seed_dir}/manifest.json'))
    !python run.py prepare --raw data/raw/multi_event.parquet --out {seed_dir} --users {seed_manifest['selected_users']} --catalog {seed_manifest['catalog']} --seed {seed_manifest['seed']}
    !python run.py evaluate --out {seed_dir}
```

Cell 7 (markdown):
```
## Download the regenerated results
Download `artifacts/results.json`, `experiments/seed_43/results.json`, and `experiments/seed_44/results.json` from the Colab file browser and replace the copies in your local repo, then commit.
```

- [ ] **Step 2: Run the notebook on Colab and retrieve the files**

Upload `colab/evaluate_nfvr_fix.ipynb` to https://colab.research.google.com, run all cells, then download the three regenerated `results.json` files from the Colab file browser (left sidebar → folder icon) and replace the local copies at `artifacts/results.json`, `experiments/seed_43/results.json`, `experiments/seed_44/results.json`.

- [ ] **Step 3: Regenerate `experiments/seed_results.csv` and `experiments/seed_summary.json` locally**

Run: `python summarize_seeds.py`
Expected: the script (already in this repo) reads the three `results.json` files and rewrites `experiments/seed_results.csv` and `experiments/seed_summary.json` with the corrected NFVR figures.

- [ ] **Step 4: Update `README.md`'s Results section**

Read the new `artifacts/results.json` and `experiments/seed_results.csv`, then update the NFVR sentence at `README.md:50` (currently: *"In that run, NextBeat had more dislike violations than the sequence-only GRU: 0.312% versus 0.272%."*) to state the new post-fix numbers, which should now read at or near 0% for both models. Keep the rest of the Results section's structure and tone unchanged — only the NFVR-related sentence(s) and any Recall/NDCG numbers that shifted need updating.

- [ ] **Step 5: Run the full test suite against the regenerated artifacts**

Run: `python -m pytest -q`
Expected: all tests pass, including `test_full_target_training_keeps_reference_data_and_split` (skipped unless `manifest['training']['all_targets']` is set — fine either way) and the new `test_no_model_ever_recommends_an_active_dislike` from Task 6

- [ ] **Step 6: Update `verification/ARTIFACT_CHECKSUMS.json`**

The regenerated `results.json` files have new checksums. Regenerate this file's `results.json` entries (recompute SHA256 for each changed file and update the corresponding value in the JSON) so `check_installation.py` doesn't report false "Missing or modified saved artifact" failures.

Run: `python -c "import hashlib,json,pathlib; c=json.load(open('verification/ARTIFACT_CHECKSUMS.json')); [c.update({k: hashlib.sha256(pathlib.Path('artifacts',k).read_bytes()).hexdigest()}) for k in c if k=='results.json' and pathlib.Path('artifacts',k).exists()]; json.dump(c, open('verification/ARTIFACT_CHECKSUMS.json','w'), indent=2)"`

Then run: `python check_installation.py`
Expected: `READY: python -m streamlit run app.py` with no `FAIL` lines

- [ ] **Step 7: Commit**

```bash
git add colab/evaluate_nfvr_fix.ipynb README.md artifacts/results.json experiments/seed_43/results.json experiments/seed_44/results.json experiments/seed_results.csv experiments/seed_summary.json verification/ARTIFACT_CHECKSUMS.json
git commit -m "Regenerate results.json with the dislike-violation fix applied

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
git push
```

---

## Optional stretch (G2/G3) — only if time remains after Task 7

These are intentionally left lighter-weight than Tasks 1-7: they are the first things to cut. Do not start G2/G3 until Tasks 1-7 are committed and pushed, since G1 alone already satisfies the reviewer's core complaint.

### Task 8 (optional, G2): Train-time penalty for disliked tracks, seed 42 only

**Files:**
- Modify: `run.py:229-243` (the training inner loop in `train()`)
- Create: `colab/train_with_dislike_penalty.ipynb`

- [ ] **Step 1:** In the training loop, after computing `negatives` (`run.py:234`), also sample from each example's own actively-disliked tracks when available, and add a penalty term identical in form to the existing negative-sampling loss but applied to those disliked ids specifically:

```python
                dislikes = [windowed_active_dislikes(x[k], f[k]) for k in range(len(p))]
                if feedback and any(len(d) for d in dislikes):
                    max_len = max(len(d) for d in dislikes)
                    padded = np.zeros((len(p), max_len), dtype=np.int64)
                    mask = np.zeros((len(p), max_len), dtype=np.float32)
                    for k, d in enumerate(dislikes):
                        padded[k, :len(d)] = d
                        mask[k, :len(d)] = 1
                    padded_t = torch.tensor(padded, dtype=torch.long)
                    mask_t = torch.tensor(mask)
                    dislike_scores = (state[:, None] * model.output(padded_t)).sum(-1) + model.bias(padded_t).squeeze(-1)
                    dislike_penalty = (F.softplus(dislike_scores) * mask_t).sum() / mask_t.sum().clamp(min=1)
                    loss = loss + 0.1 * dislike_penalty
```

  This only fires for the `feedback=True` (NextBeat) model — pass a `feedback` flag into scope (it's already the outer loop variable at `run.py:202`) and requires `from run import windowed_active_dislikes` to already be defined in the same file (Task 2 already added it here, so it's just a local call, no import needed).

- [ ] **Step 2:** Write a Colab notebook `colab/train_with_dislike_penalty.ipynb` mirroring Task 7's structure (clone, install, download, prepare with `--seed 42`), but calling `python run.py train --epochs 3 --all-targets --seed 42 --out artifacts_g2` (a separate output directory, so the original G1-only artifacts are never overwritten until you've confirmed G2 is actually better).

- [ ] **Step 3:** After training completes on Colab, download `artifacts_g2/results.json` and compare its `nfvr10`, `recall10`, and `ndcg10` for `NextBeat` against the G1-only `artifacts/results.json`. Only replace `artifacts/nextbeat.pt` and `artifacts/results.json` with the G2 versions if NFVR is equal-or-better AND Recall@10/NDCG@10 did not regress by more than what's already a small, mixed margin in this project's own numbers (i.e., don't accept a trade that undoes G1's guarantee-level fix for a training-time approximation that's still nonzero).

- [ ] **Step 4:** If kept, commit the same way as Task 7 Step 7 (new `nextbeat.pt`, `results.json`, updated README, updated checksums). If not kept, discard `artifacts_g2/` and do nothing further — G1 remains the shipped fix.

### Task 9 (optional, G3): Larger training cohort, seed 42 only

- [ ] **Step 1:** On the same Colab environment as Task 7/8, re-run `run.py prepare` with a larger `--users` value (e.g. `--users 5000`, double the current 2500 — pick based on how much Colab time/disk is actually available) into a fresh `--out artifacts_g3` directory, then `run.py train --epochs 3 --all-targets --seed 42 --out artifacts_g3`.
- [ ] **Step 2:** Compare `artifacts_g3/results.json` against the current shipped `results.json`. Only adopt it if it completed within a reasonable Colab session and the numbers are sensible (no NaNs, no collapsed metrics).
- [ ] **Step 3:** If adopted, replace the `artifacts/` contents and update `README.md`'s "Data and training" table (the `Selected users` row and related counts at `README.md:33-40`) and commit. If skipped or it fails/times out, do nothing — per the spec, training on the current 2500-user cohort with a documented "may be less precise with more data" caveat is an explicitly acceptable outcome.

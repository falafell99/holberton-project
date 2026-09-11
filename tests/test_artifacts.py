"""Integration checks for the saved models, API and interface."""
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'artifacts'
pytestmark = pytest.mark.skipif(not (ROOT / 'demo.npz').exists(), reason='Train real artifacts first')


def test_real_dataset_and_artifact_shapes():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    data = np.load(ROOT / 'demo.npz')
    assert manifest['audit']['rows'] > 40_000_000
    assert manifest['train_events'] > 1_000_000
    assert data['x'].shape == data['f'].shape[:2]
    assert data['x'].shape[0] == len(data['y'])


def test_api_real_user_and_unknown_user():
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    assert client.get('/health').json()['ready']
    uid = client.get('/users').json()['anonymous_user_ids'][0]
    response = client.get(f'/recommend/{uid}')
    assert response.status_code == 200
    assert response.json()['model'] == json.loads((ROOT/'manifest.json').read_text())['selected_model_by_validation']
    assert len(response.json()['recommendations']) == 10
    assert client.get('/recommend/-1').status_code == 404


def test_api_recommend_endpoint_never_returns_an_active_dislike():
    """Exercises the actual /recommend/{uid} wiring in api.py, not a re-implementation of it."""
    from fastapi.testclient import TestClient
    from api import app
    from run import windowed_active_dislikes
    client = TestClient(app)
    data = np.load(ROOT / 'demo.npz')
    uids = client.get('/users').json()['anonymous_user_ids']
    checked = 0
    for uid in uids:
        i = int(np.flatnonzero(data['uid'] == uid)[0])
        blocked = windowed_active_dislikes(data['x'][i], data['f'][i])
        if len(blocked) == 0:
            continue
        checked += 1
        response = client.get(f'/recommend/{uid}')
        assert response.status_code == 200
        recommended = {r['track_id'] for r in response.json()['recommendations']}
        # /recommend returns real vocab track ids, but windowed_active_dislikes() returns
        # internal item ids (data['x'] space) — map blocked ids into vocab space to compare.
        blocked_track_ids = {int(data['vocab'][item_id - 2]) for item_id in blocked.tolist()}
        assert recommended.isdisjoint(blocked_track_ids), f'user {uid} was recommended a disliked track'
    if checked == 0:
        pytest.skip('No demo.npz user has an active dislike — test is not exercising anything')


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


def test_streamlit_real_prediction():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT.parent / 'app.py')).run(timeout=30)
    assert not at.exception
    at.button[0].click().run(timeout=30)
    assert not at.exception
    # History, recommendation table, and real comparison table.
    assert len(at.dataframe) == 3


def test_streamlit_all_models_and_what_if_preserve_recorded_evidence():
    from streamlit.testing.v1 import AppTest
    before = (ROOT/'results.json').read_bytes(), (ROOT/'demo.npz').read_bytes()
    at = AppTest.from_file(str(ROOT.parent/'app.py')).run(timeout=30)
    for model in ['NextBeat', 'Sequence-only GRU', 'ItemKNN', 'Most Popular']:
        at.selectbox[1].set_value(model).run(timeout=30)
        at.selectbox[2].set_value('Dislike').run(timeout=30)
        at.button[0].click().run(timeout=30)
        assert not at.exception
        recommendations = at.dataframe[1].value
        assert len(recommendations) == 10
        assert recommendations['track'].str.startswith('Track ').all()
    assert before == ((ROOT/'results.json').read_bytes(), (ROOT/'demo.npz').read_bytes())


def test_installation_check_runs_real_models(capsys):
    from check_installation import main
    assert main() == 0
    assert 'READY:' in capsys.readouterr().out


def test_full_target_training_keeps_reference_data_and_split():
    m = json.loads((ROOT/'manifest.json').read_text())
    if not m['training'].get('all_targets'):
        pytest.skip('Full-target training not enabled')
    old = json.loads((Path(__file__).resolve().parent/'fixtures/reference_manifest.json').read_text())
    for key in ['source_sha256','cutoffs','catalog','selected_users','selected_events','seed']:
        assert m[key] == old[key]
    logs = json.loads((ROOT/'training_log.json').read_text())
    assert len(logs) == 6
    assert all(row['examples'] == m['strict_time_train_targets'] for row in logs)
    assert m['selected_model_by_validation'] == max(m['validation_ndcg10'], key=m['validation_ndcg10'].get)

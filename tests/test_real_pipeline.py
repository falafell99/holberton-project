from pathlib import Path

import numpy as np
import torch
from models import NextBeat
from run import contexts, queries, active_dislikes, top10, metrics

ROOT = Path(__file__).resolve().parents[1] / 'artifacts'


def test_context_does_not_cross_users_or_include_target():
    items = np.array([2,3,4,5,6])
    starts = np.array([0,0,2,2,2])
    x, _ = contexts(items, np.ones((5,6)), starts, np.array([1,4]), length=3)
    assert x.tolist() == [[0,0,2],[0,4,5]]


def test_unknown_target_not_replaced_by_later_known_track():
    data = dict(ts=np.array([1,4,5]), uid=np.array([1,1,1]), positive=np.ones(3,bool),
                items=np.array([2,1,3]), starts=np.zeros(3,int), features=np.zeros((3,6)))
    q = queries(data, 3, 6)
    assert q['total'] == 1 and len(q['y']) == 0


def test_dislike_reversal():
    f = np.zeros((4,6))
    f[0,3] = 1
    f[1,5] = 1
    f[2,3] = 1
    data = dict(starts=np.zeros(4,int), items=np.array([2,2,3,4]), features=f)
    assert active_dislikes(data, [3])[0].tolist() == [3]


def test_feedback_changes_neural_state_but_not_ablation():
    torch.manual_seed(42)
    x = torch.tensor([[2,3]])
    a, b = torch.zeros(1,2,6), torch.ones(1,2,6)
    model = NextBeat(12, feedback=True)
    assert not torch.allclose(model(x,a), model(x,b))
    model.feedback = False
    assert torch.allclose(model(x,a), model(x,b))


def test_reserved_ids_never_recommended():
    ranks = top10(np.arange(15, dtype=float)[None])
    assert ranks.shape == (1,10) and ranks.min() >= 2


def test_top10_never_returns_blocked_items():
    scores = np.arange(15, dtype=float)[None]
    ranks = top10(scores.copy(), blocked=[np.array([12, 13, 14])])
    assert set(ranks[0].tolist()).isdisjoint({12, 13, 14})


def test_top10_blocked_none_is_backward_compatible():
    scores = np.arange(15, dtype=float)[None]
    ranks = top10(scores.copy())
    assert ranks.shape == (1, 10) and ranks.min() >= 2


def test_metrics_known_answer():
    q = dict(y=np.array([2]), total=2)
    ranks = np.arange(2,12)[None]
    counts = np.r_[0,0,np.ones(10)]
    result = metrics(ranks, q, counts, [np.array([2])])
    assert result['recall10'] == 1
    assert result['unconditional_recall10'] == .5
    assert result['ndcg10'] == 1
    assert result['nfvr10'] == .1


def test_same_timestamp_feedback_is_excluded():
    data = dict(ts=np.array([1,4,4,5]), uid=np.array([1,1,1,1]), positive=np.array([True,False,True,True]),
                items=np.array([2,3,4,5]), starts=np.zeros(4,int), features=np.zeros((4,6)),
                prefix=np.array([0,1,1,3]))
    q = queries(data, 4, 6)
    assert q['y'].tolist() == [4]
    assert q['x'][0,-1] == 2
    assert 3 not in q['x'][0]


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


def test_neural_ranks_respects_blocked():
    from run import neural_ranks
    torch.manual_seed(0)
    # 13 items so blocking one still leaves >=10 valid candidates (ids 2..12);
    # with 12 items the valid pool is exactly 10 and top10 is forced to
    # backfill with an already-blocked id, making the assertion unsatisfiable.
    model = NextBeat(13, feedback=False)
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


def test_evaluate_stage_is_registered_in_cli():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, 'run.py', '--help'], capture_output=True, text=True,
                            cwd=str(ROOT.parent))
    assert 'evaluate' in result.stdout


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

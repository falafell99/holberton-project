import numpy as np
import torch
from models import NextBeat
from run import contexts, queries, active_dislikes, top10, metrics


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

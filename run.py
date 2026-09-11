"""Reproducible real-Yambda experiment. No synthetic-data fallback."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from models import NextBeat

REVISION = 'dd6f3a19eef5866e346c3270e098baa641a44948'
EVENTS = ['listen', 'like', 'dislike', 'unlike', 'undislike']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def contexts(items, features, starts, positions, length=20):
    offsets = np.arange(-length, 0)
    ix = positions[:, None] + offsets
    valid = ix >= starts[positions, None]
    ix = np.maximum(ix, 0)
    return np.where(valid, items[ix], 0), features[ix] * valid[..., None]


def prepare(args):
    raw = Path(args.raw)
    if not raw.is_file():
        raise FileNotFoundError('Real Yambda multi_event.parquet is required. Run download.py first.')
    from download import verify
    verify(raw)
    scan = pl.scan_parquet(raw)
    audit = scan.select(pl.len().alias('rows'), pl.col('uid').n_unique().alias('users'),
                        pl.col('timestamp').min().alias('min_time'),
                        pl.col('timestamp').max().alias('max_time')).collect().to_dicts()[0]
    users = scan.select('uid').unique().collect()['uid'].sort().to_numpy()
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(users, min(args.users, len(users)), replace=False))
    frame = scan.filter(pl.col('uid').is_in(chosen.tolist())).collect().sort(['uid', 'timestamp'])
    t1 = int(audit['min_time'] + .8 * (audit['max_time'] - audit['min_time']))
    t2 = int(audit['min_time'] + .9 * (audit['max_time'] - audit['min_time']))
    positive = (pl.col('event_type') == 'listen') & (pl.col('played_ratio_pct') > 50)
    vocab = (frame.filter((pl.col('timestamp') < t1) & positive)
             .group_by('item_id').len().sort(['len', 'item_id'], descending=[True, False])
             .head(args.catalog)['item_id'].to_numpy())
    mapping = {int(v): i + 2 for i, v in enumerate(vocab)}
    uid = frame['uid'].to_numpy()
    ts = frame['timestamp'].to_numpy()
    original = frame['item_id'].to_numpy()
    items = np.array([mapping.get(int(x), 1) for x in original], dtype=np.int32)
    types = frame['event_type'].cast(pl.String).to_numpy()
    ratio = frame['played_ratio_pct'].fill_null(0).to_numpy().astype(np.float32)
    features = np.stack([np.minimum(ratio / 100, 2)] +
                        [(types == name).astype(np.float32) for name in EVENTS], axis=1)
    starts = np.maximum.accumulate(np.where(np.r_[True, uid[1:] != uid[:-1]], np.arange(len(uid)), 0))
    positive_np = (types == 'listen') & (ratio > 50)
    train_positions = np.flatnonzero((ts < t1) & positive_np & (items >= 2) & (np.arange(len(uid)) > starts))
    counts = np.bincount(items[(ts < t1) & positive_np], minlength=len(vocab) + 2).astype(np.float32)
    counts[:2] = 0
    manifest = dict(source='https://huggingface.co/datasets/yandex/yambda', revision=REVISION,
                    source_sha256=digest(raw), audit=audit, seed=args.seed, selected_users=len(chosen),
                    selected_events=len(uid), train_events=int((ts < t1).sum()),
                    train_positive_targets=len(train_positions), catalog=len(vocab),
                    cutoffs=[t1, t2], sequence_length=20,
                    timestamp_units='Original Yambda globally ordered five-second bins; no invented UTC dates',
                    cohort='Seeded users with all their events; no random row deletion',
                    candidate_policy='Train-only top catalogue; repeats allowed; OOV targets counted separately',
                    event_counts=frame.group_by('event_type').len().to_dicts())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'prepared.npz', uid=uid, ts=ts, original=original, items=items,
                        features=features, starts=starts, targets=train_positions, counts=counts,
                        vocab=vocab, positive=positive_np)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


def queries(data, cutoff, end):
    positions = np.flatnonzero((data['ts'] >= cutoff) & (data['ts'] < end) & data['positive'])
    _, first = np.unique(data['uid'][positions], return_index=True)
    positions = positions[first]
    # Do not silently replace unknown first targets with easier later targets.
    prefix = data.get('prefix', np.arange(len(data['items'])))
    available = (data['items'][positions] >= 2) & (prefix[positions] > data['starts'][positions])
    eligible = positions[available]
    x, f = contexts(data['items'], data['features'], data['starts'], prefix[eligible])
    return dict(positions=eligible, x=x, f=f, y=data['items'][eligible], total=len(positions))


def metrics(ranks, q, counts, blocked):
    hit = ranks == q['y'][:, None]
    recall = hit.any(axis=1).mean()
    ndcg = (hit / np.log2(np.arange(ranks.shape[1]) + 2)).sum(axis=1).mean()
    violation = sum(np.isin(r, b).sum() for r, b in zip(ranks, blocked)) / ranks.size
    return dict(recall10=float(recall), ndcg10=float(ndcg),
                unconditional_recall10=float(recall * len(q['y']) / q['total']),
                coverage10=float(len(np.unique(ranks)) / (len(counts) - 2)),
                novelty10=float((-np.log2(np.maximum(counts[ranks], 1) / counts.sum())).mean()),
                nfvr10=float(violation), evaluated_users=len(q['y']), all_target_users=q['total'])


def active_dislikes(data, positions):
    result = []
    for p in positions:
        begin = data['starts'][p]
        f = data['features'][begin:p]
        changes = np.flatnonzero((f[:, 3] == 1) | (f[:, 5] == 1))
        active = set()
        for j in changes:
            item = int(data['items'][begin + j])
            if f[j, 3] == 1:
                active.add(item)
            else:
                active.discard(item)
        result.append(np.array(sorted(active), dtype=np.int32))
    return result


def top10(scores):
    scores[:, :2] = -np.inf
    selected = np.argpartition(scores, -10, axis=1)[:, -10:]
    return np.take_along_axis(selected, np.argsort(-np.take_along_axis(scores, selected, axis=1), axis=1), axis=1)


def neural_ranks(model, q):
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(q['y']), 128):
            x = torch.tensor(q['x'][i:i+128], dtype=torch.long)
            f = torch.tensor(q['f'][i:i+128], dtype=torch.float32)
            rows.append(top10(model.scores(x, f).numpy()))
    return np.concatenate(rows)


def fit_knn(data, t1, out):
    mask = (data['ts'] < t1) & data['positive'] & (data['items'] >= 2)
    _, user_codes = np.unique(data['uid'][mask], return_inverse=True)
    n = len(data['counts'])
    matrix = sp.coo_matrix((np.ones(mask.sum(), np.float32), (user_codes, data['items'][mask])),
                           shape=(user_codes.max()+1, n)).tocsr()
    matrix.data[:] = 1
    norms = np.sqrt(np.asarray(matrix.sum(axis=0)).ravel())
    normalized = matrix @ sp.diags(1 / np.maximum(norms, 1))
    rows = []
    for start in range(0, n, 128):
        scores = (normalized[:, start:start+128].T @ normalized).toarray()
        scores[np.arange(len(scores)), np.arange(start, start+len(scores))] = 0
        ix = np.argpartition(scores, -100, axis=1)[:, -100:]
        vals = np.take_along_axis(scores, ix, axis=1)
        rows.append(sp.csr_matrix((vals.ravel(), (np.repeat(np.arange(len(scores)), 100), ix.ravel())),
                                  shape=(len(scores), n)))
    knn = sp.vstack(rows).tocsr()
    knn.eliminate_zeros()
    sp.save_npz(out / 'itemknn.npz', knn)
    return knn


def knn_ranks(knn, q, counts):
    output = []
    for x in q['x']:
        seen = x[x >= 2]
        scores = np.asarray(knn[seen].sum(axis=0)).ravel() if len(seen) else counts.copy()
        scores = scores + counts / max(counts.max(), 1) * 1e-6
        output.append(top10(scores[None])[0])
    return np.array(output)


def train(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    data = np.load(out / 'prepared.npz')
    # Materialize once: repeated access to compressed npz decompresses every array.
    data = {k: data[k] for k in data.files}
    new_time = np.r_[True, (data['uid'][1:] != data['uid'][:-1]) | (data['ts'][1:] != data['ts'][:-1])]
    data['prefix'] = np.maximum.accumulate(np.where(new_time, np.arange(len(new_time)), 0))
    data['targets'] = data['targets'][data['prefix'][data['targets']] > data['starts'][data['targets']]]
    manifest = json.loads((out / 'manifest.json').read_text())
    t1, t2 = manifest['cutoffs']
    valid = queries(data, t1, t2)
    test = queries(data, t2, np.iinfo(np.uint32).max)
    blocked = active_dislikes(data, data['prefix'][test['positions']])
    results = {}
    counts = data['counts']
    pop = np.repeat(top10(counts[None].copy()), len(test['y']), axis=0)
    results['Most Popular'] = metrics(pop, test, counts, blocked)
    print('Fitting cosine ItemKNN on all training positives...', flush=True)
    knn = fit_knn(data, t1, out)
    results['ItemKNN'] = metrics(knn_ranks(knn, test, counts), test, counts, blocked)
    logs = []
    validation = {'Most Popular': float(((np.repeat(top10(counts[None].copy()), len(valid['y']), axis=0) == valid['y'][:, None]) / np.log2(np.arange(10)+2)).sum(1).mean()),
                  'ItemKNN': float(((knn_ranks(knn, valid, counts) == valid['y'][:, None]) / np.log2(np.arange(10)+2)).sum(1).mean())}
    for feedback in [False, True]:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        name = 'NextBeat' if feedback else 'Sequence-only GRU'
        model = NextBeat(len(counts), feedback=feedback)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        best = -1
        filename = 'nextbeat.pt' if feedback else 'sequence.pt'
        resume_path = out / (filename + '.resume')
        first_epoch = 0
        if args.resume and resume_path.exists():
            checkpoint = torch.load(resume_path, weights_only=False)
            if checkpoint['all_targets'] != args.all_targets or checkpoint['seed'] != args.seed:
                raise ValueError('Resume configuration does not match saved run')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            rng.bit_generator.state = checkpoint['numpy_rng']
            torch.set_rng_state(checkpoint['torch_rng'])
            first_epoch = checkpoint['epoch']
            best = checkpoint['best']
            logs.extend(checkpoint['logs'])
        for epoch in range(first_epoch, args.epochs):
            model.train()
            size = len(data['targets']) if args.all_targets else min(args.examples, len(data['targets']))
            positions = rng.choice(data['targets'], size=size, replace=False)
            losses = []
            began = time.time()
            for offset in range(0, size, 512):
                p = positions[offset:offset+512]
                x, f = contexts(data['items'], data['features'], data['starts'], data['prefix'][p])
                y = torch.tensor(data['items'][p], dtype=torch.long)
                state = model(torch.tensor(x, dtype=torch.long), torch.tensor(f, dtype=torch.float32))
                negatives = torch.randint(2, len(counts), (len(p), 64))
                pos = (state * model.output(y)).sum(-1) + model.bias(y).squeeze(-1)
                neg = (state[:, None] * model.output(negatives)).sum(-1) + model.bias(negatives).squeeze(-1)
                valid_neg = negatives != y[:, None]
                loss = F.softplus(-pos).mean() + (F.softplus(neg) * valid_neg).sum() / valid_neg.sum()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(loss.item())
                if (offset // 512 + 1) % 1000 == 0:
                    print(f'{name} epoch {epoch+1}: {min(offset+512,size):,}/{size:,} targets', flush=True)
            ranks = neural_ranks(model, valid)
            score = float(((ranks == valid['y'][:, None]) / np.log2(np.arange(10)+2)).sum(1).mean())
            log = dict(model=name, epoch=epoch+1, examples=size, loss=float(np.mean(losses)),
                       validation_ndcg10=score, seconds=round(time.time()-began, 1))
            logs.append(log)
            print(json.dumps(log), flush=True)
            (out / 'training_log.json').write_text(json.dumps(logs, indent=2))
            if score > best:
                best = score
                torch.save(model.state_dict(), out / filename)
            temporary = resume_path.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                            epoch=epoch+1, best=best, seed=args.seed, all_targets=args.all_targets,
                            logs=[row for row in logs if row['model'] == name]), temporary)
            temporary.replace(resume_path)
        model.load_state_dict(torch.load(out / filename, weights_only=True))
        validation[name] = best
        results[name] = metrics(neural_ranks(model, test), test, counts, blocked)
        (out / 'results.json').write_text(json.dumps(results, indent=2))
    # Real held-out histories, no invented users or songs.
    np.savez_compressed(out / 'demo.npz', x=test['x'], f=test['f'], y=test['y'],
                        uid=data['uid'][test['positions']], vocab=data['vocab'], counts=counts)
    manifest['training'] = vars(args)
    manifest['strict_time_train_targets'] = len(data['targets'])
    manifest['tie_policy'] = 'Exclude every event with timestamp equal to the target timestamp from its input'
    manifest['validation_ndcg10'] = validation
    manifest['selected_model_by_validation'] = max(validation, key=validation.get)
    manifest['validation_users'] = len(valid['y'])
    manifest['test_users'] = len(test['y'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'train'])
    p.add_argument('--raw', default='data/raw/multi_event.parquet')
    p.add_argument('--out', default='artifacts')
    p.add_argument('--users', type=int, default=2500)
    p.add_argument('--catalog', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--examples', type=int, default=1000000)
    p.add_argument('--all-targets', action='store_true')
    p.add_argument('--resume', action='store_true', help='Resume trusted locally generated epoch checkpoints')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.users < 1 or a.catalog < 100 or a.epochs < 1 or a.examples < 1 or a.threads < 1:
        p.error('Use positive user/epoch/example/thread counts and a catalogue of at least 100.')
    if a.stage == 'train':
        # Separate output folders may train concurrently; one folder has one writer.
        from filelock import FileLock
        with FileLock(str(Path(a.out) / '.training.lock')):
            train(a)
    else:
        prepare(a)

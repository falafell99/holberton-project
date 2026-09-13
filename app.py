"""Streamlit interface backed only by trained models and real held-out histories."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
import torch
from models import NextBeat
from run import top10, serving_dislikes

ROOT = Path(__file__).parent / 'artifacts'
st.set_page_config(page_title='NextBeat', page_icon='🎧', layout='wide')
st.title('NextBeat')
st.caption('Real Yambda listening histories · trained recommendation models · anonymous track IDs')
if not (ROOT / 'demo.npz').exists():
    st.error('Trained artifacts are missing. Run the real-data pipeline in README.md. No synthetic fallback is used.')
    st.stop()


@st.cache_resource
def load():
    torch.set_num_threads(2)
    data_file = np.load(ROOT / 'demo.npz')
    data = {k: data_file[k] for k in data_file.files}
    models = {}
    for name, filename, feedback in [('NextBeat', 'nextbeat.pt', True), ('Sequence-only GRU', 'sequence.pt', False)]:
        model = NextBeat(len(data['counts']), feedback=feedback)
        model.load_state_dict(torch.load(ROOT / filename, map_location='cpu', weights_only=True))
        model.eval()
        models[name] = model
    return data, models, sp.load_npz(ROOT / 'itemknn.npz')


data, models, knn = load()
report = json.loads((ROOT / 'results.json').read_text())
manifest = json.loads((ROOT / 'manifest.json').read_text())
cols = st.columns(3)
cols[0].metric('Real events in study', f"{manifest['selected_events']:,}")
cols[1].metric('Training events', f"{manifest['train_events']:,}")
cols[2].metric('Candidate tracks', f"{manifest['catalog']:,}")
user = st.selectbox('Anonymous user', data['uid'].tolist(),
                    help='Real saved evaluation histories. No names or music accounts are used.')
i = int(np.flatnonzero(data['uid'] == user)[0])
options = ['NextBeat', 'Sequence-only GRU', 'ItemKNN', 'Most Popular']
selected = manifest.get('selected_model_by_validation', 'NextBeat')
choice = st.selectbox('Model', options, index=options.index(selected))
st.caption(f'Default model selected by validation NDCG@10: {selected}. You can inspect every model.')
x, f = data['x'][i:i+1].copy(), data['f'][i:i+1].copy()


def track(item):
    return f'Track {int(data["vocab"][item-2])}' if item >= 2 else 'Outside selected catalogue'


history = []
for item, features in zip(x[0], f[0]):
    if item:
        event = ['listen', 'like', 'dislike', 'unlike', 'undislike'][int(np.argmax(features[1:]))]
        history.append(dict(track=track(item), event=event, played_percent=round(float(features[0])*100,1) if event == 'listen' else None))
st.subheader('Recent real history')
st.dataframe(pd.DataFrame(history), hide_index=True)
edit = st.selectbox('Optional what-if change to the last event', ['Keep recorded event', 'Full listen', 'Like', 'Dislike', 'Short listen'])
if edit != 'Keep recorded event':
    f[0, -1] = {'Full listen': [1,1,0,0,0,0], 'Like': [0,0,1,0,0,0],
                'Dislike': [0,0,0,1,0,0], 'Short listen': [.1,1,0,0,0,0]}[edit]
    st.info('This is an explicitly edited what-if input. It does not change the recorded data or evaluation scores.')
if st.button('Recommend next tracks', type='primary'):
    blocked = [serving_dislikes(data, i, x[0], f[0])]
    if choice in models:
        with torch.no_grad():
            scores = models[choice].scores(torch.tensor(x, dtype=torch.long), torch.tensor(f, dtype=torch.float32)).numpy()
    elif choice == 'ItemKNN':
        scores = np.asarray(knn[x[0][x[0] >= 2]].sum(axis=0))
        scores += data['counts'][None] / data['counts'].max() * 1e-6
    else:
        scores = data['counts'][None].copy()
    ranking = top10(scores, blocked)[0]
    st.dataframe(pd.DataFrame({'rank': range(1,11), 'track': [track(j) for j in ranking],
                               'score': scores[0, ranking]}), hide_index=True)
    st.caption('Scores rank tracks within a model; they are not calibrated probabilities or comparable across models.')
with st.expander('Recorded next track — held out from the input'):
    st.write(track(int(data['y'][i])))
st.subheader('Frozen-model test results')
display = pd.DataFrame(report).T.rename(columns={
    'recall10':'Recall@10', 'ndcg10':'NDCG@10',
    'unconditional_recall10':'Recall (all targets)', 'coverage10':'Catalogue coverage',
    'novelty10':'Novelty (bits)', 'nfvr10':'Dislike violations'})
formats = {'Recall@10':'{:.2%}', 'NDCG@10':'{:.4f}', 'Recall (all targets)':'{:.2%}',
           'Catalogue coverage':'{:.2%}', 'Novelty (bits)':'{:.2f}', 'Dislike violations':'{:.3%}'}
st.dataframe(display[list(formats)].style.format(formats))
st.caption(f"{manifest['test_users']:,} eligible in-catalogue targets; "
           f"{report['NextBeat']['all_target_users']:,} total test targets. One recorded next listen per user.")
st.caption('Repeat listens are allowed. Unknown test targets are excluded from conditional metrics and included as misses in unconditional Recall. All models filter active dislikes from the full prior history in evaluation and live recommendations. Zero NFVR reflects this filter, not perfect learned dislike avoidance.')

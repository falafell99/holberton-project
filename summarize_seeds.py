"""Summarize complete fixed-data runs; incomplete runs are never reported as final."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    runs=[]
    original=json.loads((ROOT/'artifacts/manifest.json').read_text())
    for seed in (42,43,44):
        folder=ROOT/'artifacts' if seed==42 else ROOT/'experiments'/f'seed_{seed}'
        logs=json.loads((folder/'training_log.json').read_text())
        manifest=json.loads((folder/'manifest.json').read_text())
        expected={(name,epoch) for name in ['Sequence-only GRU','NextBeat'] for epoch in [1,2,3]}
        assert {(r['model'],r['epoch']) for r in logs}==expected, f'Seed {seed} is incomplete'
        assert all(r['examples']==2837655 for r in logs)
        assert manifest['training']['seed']==seed and manifest['training']['all_targets']
        for key in ['source_sha256','seed','cutoffs','catalog','selected_users','selected_events']:
            assert manifest[key]==original[key],f'Seed {seed} changed {key}'
        runs.append(dict(seed=seed,results=json.loads((folder/'results.json').read_text()),
                         validation=manifest['validation_ndcg10']))
    models=list(runs[0]['results'])
    metrics=list(runs[0]['results'][models[0]])
    aggregate={model:{metric:dict(mean=float(np.mean([r['results'][model][metric] for r in runs])),
                                  std=float(np.std([r['results'][model][metric] for r in runs],ddof=1)))
                      for metric in metrics} for model in models}
    wins=sum(r['results']['NextBeat']['ndcg10']>r['results']['Sequence-only GRU']['ndcg10'] for r in runs)
    result=dict(runs=runs,aggregate=aggregate,
                short_conclusion_en=f'NextBeat has higher NDCG in {wins}/3 training seeds.',
                short_conclusion_az=f'NextBeat {wins}/3 seed-də daha yüksək NDCG göstərir.',
                interpretation='Three training seeds on one previously inspected split; descriptive stability evidence, not independent generalization proof.',
                serving_policy='Keep the original seed-42 validation-selected model; do not select a seed using test scores.')
    output=ROOT/'experiments'
    (output/'seed_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    with (output/'seed_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','model',*metrics]);writer.writeheader()
        for run in runs:
            for model,row in run['results'].items(): writer.writerow(dict(seed=run['seed'],model=model,**row))
    lines=['# Repeated training seeds','',
           'I kept the users, catalogue, time split, architecture and training budget fixed. Only the training seed changed. Each neural model used all 2,837,655 eligible targets for three epochs in every run.','',
           '| Model | Mean Recall@10 | SD | Mean NDCG@10 | SD |','|---|---:|---:|---:|---:|']
    for model,row in aggregate.items():
        lines.append(f"| {model} | {row['recall10']['mean']:.4f} | {row['recall10']['std']:.4f} | {row['ndcg10']['mean']:.4f} | {row['ndcg10']['std']:.4f} |")
    lines += ['',result['short_conclusion_en'],'',result['interpretation'],'',result['serving_policy'],
              '', 'The standard deviations describe training variability. Users and test targets are shared across runs, so these are not three independent datasets. All four models retain the same candidate and repeat-listen policy. See seed_results.csv for coverage, novelty, unconditional Recall and NFVR for every seed.']
    (ROOT/'REPEATED_SEEDS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()

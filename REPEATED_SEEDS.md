# Repeated training seeds

I kept the users, catalogue, time split, architecture and training budget fixed. Only the training seed changed. Each neural model used all 2,837,655 eligible targets for three epochs in every run.

| Model | Mean Recall@10 | SD | Mean NDCG@10 | SD |
|---|---:|---:|---:|---:|
| Most Popular | 0.0402 | 0.0000 | 0.0155 | 0.0000 |
| ItemKNN | 0.0443 | 0.0000 | 0.0223 | 0.0000 |
| Sequence-only GRU | 0.0929 | 0.0082 | 0.0519 | 0.0039 |
| NextBeat | 0.1003 | 0.0073 | 0.0534 | 0.0022 |

NextBeat has higher NDCG in 2/3 training seeds.

Three training seeds on one previously inspected split; descriptive stability evidence, not independent generalization proof.

Keep the original seed-42 validation-selected model; do not select a seed using test scores.

The standard deviations describe training variability. Users and test targets are shared across runs, so these are not three independent datasets. All four models retain the same candidate and repeat-listen policy. See seed_results.csv for coverage, novelty, unconditional Recall and NFVR for every seed.

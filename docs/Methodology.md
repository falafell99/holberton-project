# Methodology

## The comparison

| Model | What it uses |
|---|---|
| Most Popular | Training-period positive-listen counts |
| ItemKNN | Cosine similarity between binary user-item columns, top 100 neighbours |
| Sequence-only GRU | Last 20 event item IDs |
| NextBeat | Same GRU, plus play ratio and listen/like/dislike/unlike/undislike indicators |

Comparing the two GRU models tests the value of adding feedback to the same sequence architecture. The compact GRU is not SASRec and is not claimed to reproduce the published GRU4Rec architecture exactly.

Both neural models use 48-dimensional embeddings and hidden states, Adam with learning rate 0.003, batches of 512, and 64 uniform negative samples per target. The objective is sampled binary logistic loss, not full-catalogue cross entropy. Padding and unknown items are never recommended. Checkpoints are selected using validation NDCG@10, not test scores.

## Evaluation

The data is split by global timestamp: training below 20,800,000, validation from 20,800,000 to below 23,400,000, and test from 23,400,000 onward. These are the dataset's ordered time bins, not calendar dates invented from Unix timestamps.

For each period evaluation uses the first positive listen per user, using only events with a strictly earlier timestamp. All events tied at the target timestamp are excluded from its input because their ordering is uncertain. The model weights stay frozen. Test inputs can include earlier validation-period events because those would already be known at test time. Repeat recommendations are allowed because people replay music.

Conditional Recall@10 and NDCG@10 use targets inside the training catalogue with nonempty history. Unconditional Recall also counts excluded targets as misses. Always show both: the restricted catalogue makes conditional metrics easier. This is a one-target-per-user evaluation, not every future listen.

NFVR is the fraction of recommendations that repeat an actively disliked track from the known history. An undislike removes that track from the active set. All four models apply an automatic active-dislike filter using the full strictly earlier history, both in reported test evaluation and live recommendations. Zero NFVR therefore measures enforcement of this rule, not learned avoidance. The neural model still sees only its last 20 events; the filter retains older active dislikes separately. A what-if edit replaces the last event and recomputes the active set without changing the saved history. Short listens are observable behaviour, not proof that the user disliked a song.

`artifacts/results.json` contains measured test results; `training_log.json` contains actual epoch losses, validation scores, and timings. `manifest.json` records source revision, checksum, exact counts, cohort seed, and temporal boundaries.


## Validation and uncertainty limitations

`run.py train` and `run.py evaluate` now apply the same full-history dislike filter to validation NDCG@10 as they already did to the test metric and live recommendations, so future checkpoint/model selection and the served numbers use one consistent policy. The delivered checkpoints and `manifest.json`'s `selected_model_by_validation` in this artifact set were still produced by the earlier unfiltered validation run and have not yet been regenerated against the corrected code (that requires re-running `run.py evaluate` on the full prepared dataset, the same Colab step used to refresh `results.json`). Until that re-run happens, treat the currently published `validation_ndcg10` figures as pre-filter; the test results and live filtering are unaffected either way.

`artifacts/uncertainty.json` is a paired-user bootstrap for the frozen seed-42 NextBeat and sequence-only models on the same 994 eligible targets and full-history filter as `results.json`. The mean NDCG@10 difference is 0.0007834; its 95% percentile interval is approximately [-0.00812, 0.00954]. Because the interval includes zero, these results do not establish a statistically significant improvement. This interval describes user-sampling uncertainty, not training-seed variability or performance on new datasets.

Run `python refresh_uncertainty.py` to reproduce this estimate from the saved serving histories and model weights. The script checks both recomputed NDCG values against the published results before writing and updates only the uncertainty file and its checksum.

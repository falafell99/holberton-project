# NextBeat: finish, fix dislike violations, add web frontend, deploy

Status: approved by user, ready for implementation planning
Date: 2026-09-11

## Context

NextBeat is a completed Holberton School ML final project: a next-track
recommendation system comparing four approaches (Most Popular, ItemKNN,
sequence-only GRU, and NextBeat = GRU + feedback signals) on anonymized
Yambda listening histories. Trained artifacts already exist in
`artifacts/` and `experiments/`. The project currently ships as a
Streamlit app (`app.py`) plus a read-only FastAPI backend (`api.py`).

A reviewer flagged two things to improve before this is presentation-
ready:

1. Dislike violations (NFVR — the fraction of recommendations that
   repeat a track the user actively disliked) are too high; the model
   needs more focus there.
2. (Secondary, optional) Training set size could be bigger for a more
   confident model, but is explicitly *not* a hard requirement.

Separately, the user wants the project to go from "it works" to
presentable and deployed: a real web frontend instead of only
Streamlit, and both frontend and backend actually deployed online.

This spec covers three ordered pieces of work. They are sequenced, not
parallel: model fix first, then frontend + deploy.

## Goals

- G1: Recommendations never surface an actively disliked track,
  across all four models, without materially hurting Recall@10/NDCG@10.
- G2 (optional, time-permitting): NextBeat's own scores are trained to
  down-rank disliked tracks, not just filtered post-hoc.
- G3 (optional, lowest priority): Larger training cohort, only if it
  doesn't cost excessive time/compute; a smaller cohort is explicitly
  an acceptable outcome ("model is less precise" is a fine caveat).
- G4: A real web frontend (React + Vite, plain CSS) replaces Streamlit
  as the primary user-facing UI, with feature parity with the current
  Streamlit app.
- G5: Backend (FastAPI) and frontend (React) are both deployed and
  reachable over the public internet.

## Non-goals

- No change to the core GRU architecture (`models.py`) beyond an
  optional loss-term addition for G2.
- No user accounts, auth, or write-back of "what-if" edits to saved
  history — what-if stays a client-side/request-side simulation, as
  today.
- No redesign of Most Popular / ItemKNN scoring logic beyond adding
  the dislike mask.
- No CI/CD pipeline beyond what's needed to deploy (manual deploys via
  Render/Vercel dashboards or their CLIs are acceptable).
- No mobile app; web only, responsive is a nice-to-have not a
  requirement.

## Part 1 — Fix dislike violations (NFVR)

Priority order if time runs short: drop G3 first, then G2. G1 (the
filter) always ships.

### G1: Inference-time dislike filter (required)

- A track counts as "actively disliked" for a user if the most recent
  event for that track in their history is `dislike` and no later
  `undislike` event exists for it.
- Add a shared helper (co-located with `top10` in `run.py`, since both
  `app.py` and `api.py` already import `top10` from there) that takes
  the raw scores plus a user's item/feature history and masks actively
  disliked tracks' scores to `-inf` before ranking.
- Apply this mask uniformly across all four models (Most Popular,
  ItemKNN, sequence-only GRU, NextBeat) — comparisons must stay fair.
- Recompute `artifacts/results.json` and `experiments/*/results.json`
  (NFVR and any metric that depends on ranking) via the existing
  evaluate path in `run.py`, so the shipped numbers reflect the fix.
- Update `README.md`'s results table/narrative to reflect the new NFVR
  numbers.
- `app.py` and `api.py` both use the shared masking helper rather than
  re-implementing it.

### G2: Optional training-time penalty (time-permitting)

- Add an additional loss term when training NextBeat (`run.py train`):
  penalize high scores assigned to a user's actively-disliked tracks
  from their own history.
- Retrain only seed 42 (the seed the interface shows by default).
- Compute constraint: user has no GPU (Mac, no dedicated graphics
  card). Retraining happens on **Google Colab**, not locally. Deliver
  a `.ipynb` that: installs deps, pulls the pinned Yambda data (or
  reuses prepared arrays), runs `run.py train` with the new loss term,
  and produces downloadable `.pt` files to drop into `artifacts/`.
- Compare NFVR/Recall@10/NDCG@10 before vs. after. Keep this change
  only if it doesn't regress Recall/NDCG meaningfully. If it's worse
  or time runs out, ship with G1 only — this is explicitly acceptable.

### G3: Optional larger training cohort (lowest priority)

- If time/compute allow, increase `--users` (and/or other cohort
  sizing) in `run.py prepare`, then retrain seed 42 on Colab the same
  way as G2.
- Explicitly fine to skip: if skipped, note in README/report that the
  model was trained on a bounded cohort and precision may improve with
  more data. This is the first thing to cut if time is short.

### Testing

- Extend `tests/test_real_pipeline.py` with a check that, for real
  evaluation users with a nonempty active-dislike set, none of a
  model's top-10 recommendations are in that set — run across all four
  models.

## Part 2 — React frontend + deployment

### Backend changes (`api.py`)

Current endpoints: `/health`, `/users`, `/recommend/{uid}`. To reach
parity with the Streamlit app's functionality, add:

- `GET /history/{uid}` — the user's recent event history (track,
  event type, played percent), same data `app.py` currently renders
  locally from `demo.npz`.
- `GET /models` — the list of selectable models plus which one is the
  validation-selected default (from `manifest.json`).
- `GET /metrics` — the per-model metrics table currently rendered from
  `results.json` (recall10, ndcg10, unconditional_recall10, coverage10,
  novelty10, nfvr10, all_target_users), with the same field names/
  formatting semantics the Streamlit app uses today.
- `POST /recommend/{uid}` — body `{model, what_if}`, where `what_if` is
  one of `keep | full_listen | like | dislike | short_listen` (mirrors
  the Streamlit selectbox). Replaces relying on `GET /recommend/{uid}`
  alone, since model choice and what-if edits need to be passed in.
  Keep the existing `GET /recommend/{uid}` for simple/back-compat use
  (defaults to the validation-selected model, no what-if edit).
- Add CORS middleware (`fastapi.middleware.cors.CORSMiddleware`)
  allowing the deployed Vercel origin (and `localhost` during dev).
- All ranking endpoints apply the Part 1 dislike-filter helper.
- `app.py` (Streamlit) is left in place for local debugging; it is not
  required to move to the new endpoints, but if convenient it may call
  the same shared helper functions directly (no hard requirement).

### Frontend (`frontend/`, React + Vite, plain CSS — no UI framework)

Feature parity with the current Streamlit app, same screen (single
page):

- `UserSelect` — dropdown sourced from `GET /users`
- `ModelSelect` — dropdown sourced from `GET /models`, defaulting to
  the validation-selected model
- `HistoryTable` — recent real history from `GET /history/{uid}`
- `WhatIfPicker` — the 5 options (Keep / Full listen / Like / Dislike /
  Short listen), sent as part of the `POST /recommend/{uid}` body
- `RecommendButton` → calls `POST /recommend/{uid}` → renders
  `RecommendationsTable` (rank, track, score) with the same "scores
  aren't comparable across models" caveat text
- `MetricsPanel` — table from `GET /metrics`, same metrics/captions as
  the Streamlit report section (recall, ndcg, coverage, novelty, NFVR,
  eligible vs. total test targets)
- Visual design: simple, clean, not template-generic — styling pass
  happens during implementation using a dedicated design skill rather
  than default component styling.

### Deployment

- Backend (`api.py`): deploy to **Render**, using the existing
  `Dockerfile`. Since the README notes the Docker recipe is untested,
  build-test it and fix whatever breaks as part of this work.
- Frontend: deploy to **Vercel**, auto-detected Vite build.
- Frontend reads the backend URL from an env var (`VITE_API_URL`) —
  no hardcoded localhost.
- Document the deployed URLs and redeploy steps in `README.md`.

### Testing

- Add tests (in `tests/`) for the new/changed API endpoints:
  `/history/{uid}`, `/models`, `/metrics`, `POST /recommend/{uid}`
  (including a what-if case and a dislike-filter case).

## Open questions / risks

- Colab session limits (disk, idle timeout) could interrupt G2/G3
  training runs; if that happens, fall back to G1-only per the stated
  priority order.
- Render's free tier cold-starts (~30-50s first request after idle) —
  acceptable for a student project demo, worth mentioning to whoever
  presents it so they don't think it's broken.
- This directory is not currently a git repository — version control
  and history for this work is out of scope for this spec; raise
  separately if the user wants it before implementation starts.

# Architecture

## Principles
Separate concerns, version everything, train offline, serve read-only. Python code is one installable package (`src/coolroof`); the UI is an independent Vite app that talks to the API.

## Components
| Layer | Module | Responsibility |
|---|---|---|
| Configuration | `config.py` | Paths, seed, column roles, **feature contract**, task registry, assumptions (e.g. high-rating threshold) |
| Data | `data/clean.py` | Structural validation, plausibility checks, audit trail, per-task eligibility, derived targets |
| Data | `data/report.py` | Profile JSON, 7 data figures, `data_quality_report.md` |
| Features | `features.py` | Leakage checks, `FeatureEngineer`, shared preprocessing pipeline, prediction-time schema |
| Training | `ml/candidates.py`, `ml/train.py`, `ml/estimators.py`, `ml/metrics.py`, `ml/split.py` | Candidate zoo, CV search, selection, hold-out evaluation, calibration, intervals, artifact saving |
| Reporting | `ml/evaluation_report.py`, `ml/insights.py`, `ml/business_report.py` | Diagnostics, permutation importance, counterfactual effects, segment errors, narrative reports |
| Inference | `service.py` | Singleton model loader; predict, batch, compare, exact Shapley explanations, status |
| Business logic | `recommend.py` | Valid options, normalisation + weighted scoring, ROI/payback |
| Documents | `proposal.py` | PDF proposal (ReportLab) from real predictions |
| API | `api/main.py`, `api/schemas.py` | REST endpoints, strict validation, error handling, static UI hosting |
| Optional AI | `llm/assistant.py` | Tool-calling loop over the real service; disabled cleanly without a key |
| UI | `frontend/src` | React 19 + TypeScript, Tailwind 4 tokens, Recharts, React Router |

## Data flow
```
raw CSV ──clean──► cleaned CSV + audit ──┐
                                         ├─► split (20% hold-out, fixed first)
features contract ──────────────────────┤
                                         ├─► CV tuning + selection (train pool) ─► refit ─► ONE hold-out evaluation
                                         └─► models/*.joblib + model_metadata.json + reports/*
request ─► pydantic validation (extra=forbid) ─► service (loaded once) ─► response
```

## Training protocol (identical per task)
1. Hold-out (20 %, stratified on rating) fixed before any modelling. Each task intersects it with its eligible rows.
2. Each candidate: hyper-parameter search (grid / 20 random draws) with 5-fold CV on the remaining 80 %; out-of-fold predictions give validation scores.
3. Select by the task's primary metric (baselines excluded). Optionally calibrate probabilities if out-of-fold Brier improves.
4. Refit on the whole pool; score once on the hold-out; also score the baseline.
5. Fit empirical prediction intervals from out-of-fold residuals; verify hold-out coverage; check physical plausibility of predictions.
6. Save model, metadata (feature schema, params, metrics, sample counts, seed, library versions, data SHA-256).

## API surface (prefix `/api`)
`GET health · status · schema · models · data-quality · insights · reports/{name} · figures/{name} · assistant/status` and
`POST predict · predict/batch · compare · explain · economics · proposal/pdf · assistant/chat`.
Errors are `{"detail": ...}` with proper codes: 422 validation / unsupported roof–method, 503 models missing or corrupted, 502 assistant upstream failure, 500 generic (stack trace only in server log; customer inputs are never logged).

## Security & reliability
- `extra="forbid"` request models block post-installation fields.
- Report/figure endpoints use a whitelist and a strict filename regex.
- No secrets in code; `.env` is git-ignored; the key is read from the environment only.
- Models are loaded once; no training at request time. Missing/corrupt artifacts return 503, never invented values; the UI shows an offline state instead of numbers.
- Deterministic training (seed 42), pinned minimum versions in `requirements.txt`, library versions recorded in metadata.

## Frontend structure
`pages/` (Overview, Predict, Compare, Simulator, Performance, Insights, Proposal, Assistant) · `components/` (ui primitives, charts, result cards, layout) · `state/AppState.tsx` (form persisted in sessionStorage, last prediction, theme) · `lib/` (typed API client, formatters, labels). Charts always ship a text summary and a "view data table" fallback; colour is never the only encoding; light/dark themes and reduced-motion are supported.

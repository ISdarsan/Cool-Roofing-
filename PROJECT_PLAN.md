# Project Plan — Cool Roofing Intelligence

**Goal:** an end-to-end, honest, working system that answers the five questions a sales rep needs answered about a new
building *before* installation — cost, roof-temperature drop, monthly kWh saved, likelihood of a high rating, likelihood of a
referral — plus a transparent comparison of roofing options, explanations and a customer proposal.

## Source-of-truth hierarchy
1. `cool_roofing_problem.md` (organisers' prediction tasks) → 2. evaluation-metrics image (**not supplied**; metrics taken from
the brief: MAE/RMSE for regression, Accuracy+MAE for rating, Accuracy+F1 for referral) → 3. `approach.md` (pre-modelling worksheet)
→ 4. the CSV itself (`dataset/cool_roofing_data.csv`, never modified) → 5. documented engineering judgement.

## Milestones and acceptance criteria
| # | Milestone | Acceptance criteria | Status |
|---|---|---|---|
| 1 | Discovery | Files, dataset, tooling inspected; ambiguities listed (`docs/requirements.md`) | done |
| 2 | Data audit & cleaning | Reproducible `python -m coolroof data`; audit CSV, JSON, markdown report, 7 figures; per-task sample counts | done |
| 3 | Feature contract | Single config; automated leakage tests; prediction-time schema | done |
| 4 | Modelling | 6 model heads (cost, temp drop, units saved, rating, high-rating, referral) with baseline, interpretable benchmark, boosted candidates; hold-out evaluation once | done |
| 5 | Explainability & insights | Permutation importance, SHAP per prediction, segment analysis with uncertainty, method comparison | done |
| 6 | Recommendation engine | Valid method×quality options, transparent weighted score, ROI/payback with user tariff | done |
| 7 | Backend | FastAPI: health, status, predict, batch, compare, explain, economics, proposal PDF, optional assistant | done |
| 8 | Frontend | React/TS/Vite, 8 views (incl. what-if simulator), real API only, responsive, accessible | done |
| 9 | Optional LLM assistant | Tool-calling over the real prediction service; disabled cleanly without a key | done |
| 10 | Tests & docs | pytest (data, ML, API, recommendation, assistant), frontend tests, README, runbook | done |

## Order of work (value first)
Data → models → honest evaluation → prediction API/UI → comparison → explanations → proposal → optional LLM.

# Runbook

## First-time setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e . --no-deps
python -m coolroof all                                 # audit -> clean -> train -> evaluate -> insights (~10 min)
cd frontend && npm ci && npm run build && cd ..
python -m coolroof serve                               # http://127.0.0.1:8000  (UI + API), /docs for OpenAPI
```

## Everyday commands
| Task | Command |
|---|---|
| Re-run the data audit/cleaning only | `python -m coolroof data` |
| Retrain + re-evaluate | `python -m coolroof train` (then `python -m coolroof insights`) |
| Everything from the raw CSV | `python -m coolroof all` |
| Start the product | `python -m coolroof serve [--host 0.0.0.0 --port 8000]` |
| UI hot-reload development | `cd frontend && npm run dev` (needs the API running on :8000) |
| Backend tests | `python -m pytest` |
| UI tests / typecheck / build | `cd frontend && npm test` · `npm run typecheck` · `npm run build` |

## Outputs
| Path | Content |
|---|---|
| `data/processed/cleaned_dataset.csv` | cleaned data with flags, derived targets, per-task eligibility |
| `reports/cleaning_audit.csv`, `data_quality_summary.json`, `data_quality_report.md` | audit trail and findings |
| `reports/model_comparison.csv`, `model_evaluation.md`, `model_metadata.json`, `model_candidates.json` | model selection and evaluation |
| `reports/test_predictions.csv`, `oof_predictions.csv` | traceable hold-out and out-of-fold predictions (by `building_id`) |
| `reports/feature_importance.csv`, `segment_error_analysis.csv`, `insights.json`, `business_insights.md` | explainability and insights |
| `models/*.joblib`, `models/model_metadata.json`, `models/split.json` | serving artifacts, hold-out ids |
| `reports/figures/*.png` | all static figures |

## Configuration
| Variable | Purpose |
|---|---|
| `COOLROOF_LLM_PROVIDER` | `builtin` (default, free, no key) · `gemini` · `groq` · `openrouter` · `ollama` · `openai_compat` · `anthropic` · `off` · `auto` |
| `GEMINI_API_KEY` / `GROQ_API_KEY` / `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` | key for the chosen provider (never required) |
| `COOLROOF_LLM_MODEL` | override the provider's default model (Gemini default `gemini-flash-lite-latest`) |
| `COOLROOF_LLM_BASE_URL`, `COOLROOF_LLM_API_KEY` | for `openai_compat` / custom Ollama |
| `COOLROOF_CORS_ORIGINS` | comma-separated allowed UI origins (default Vite dev server) |
| `COOLROOF_ROOT` | override the project root |
| `COOLROOF_HIGH_RATING_THRESHOLD` | high-rating definition used at training time (default 4) — retrain after changing |

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| UI shows "Prediction service unreachable" | start the backend (`python -m coolroof serve`); nothing is displayed until real models respond |
| API returns 503 "Models have not been trained yet" | run `python -m coolroof all` (or `train`) |
| 503 "Model files could not be loaded" | artifact corrupted or built with an incompatible scikit-learn; retrain |
| Predictions 422 "not offered on a … roof" | that method never occurs for that roof type in the data; pick a supported method |
| Assistant error "model not found" | model names change: set `COOLROOF_LLM_MODEL` (list Gemini models with the API's `models` endpoint) |
| Assistant error "free-tier limit" / notice "busy or rate-limited" | the free tier's daily/minute request quota is used up (each chat uses 2+ requests). Wait, try another model via `COOLROOF_LLM_MODEL`, or use `builtin`; the app already falls back to the built-in assistant automatically |
| Assistant page says "not available" | provider chosen without its key, or `off`; see `.env.example` |
| Backend tests skipped/failing on model files | train first; don't run tests while training is rewriting `models/` |

## Retraining on new data
Replace/extend `dataset/cool_roofing_data.csv` (keep a copy of the old file), run `python -m coolroof all`, review `reports/data_quality_report.md` for new anomalies, compare `reports/model_evaluation.md` with the previous version (metadata stores the raw-data SHA-256, library versions and seed), then restart the server. Keep the previous `models/` folder for rollback.

## Production notes
Put the API behind TLS and authentication; restrict CORS; add rate limiting; run several workers (`uvicorn coolroof.api.main:app --workers N`) since models are read-only; monitor input drift against `models/model_metadata.json` training ranges (the API already warns on extrapolation).

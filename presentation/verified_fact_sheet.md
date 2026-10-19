# Verified fact sheet (source of truth for the deck)

Every figure below was read from the project's saved artifacts (`models/model_metadata.json`, `reports/*.json`, `reports/*.md`) when this file was generated, or from commands run during this project (test counts). Nothing is assumed.

## 1. Models and task mapping
Five required prediction tasks, **six trained models** (customer rating is both a numeric regression and a high/not-high classifier):

| Task | Model name in the repo | Type | Selected model | Baseline (hold-out) |
|---|---|---|---|---|
| A. Installation cost | cost | Regression | Log-linear ridge (multiplicative) | training mean |
| B. Temperature drop | temperature_drop | Regression | LightGBM on drop-fraction x starting temperature | training mean |
| C. Electricity saved | units_saved | Regression | LightGBM on saved-fraction x current usage | training mean |
| D1. Customer rating (1–5) | rating | Regression | LightGBM | training median |
| D2. High rating (rating ≥ 4, assumed) | high_rating | Classification | LightGBM | majority class |
| E. Referral | referral | Classification | Logistic regression | all-positive (F1), majority class (accuracy) |

## 2. Evaluation design
- Dataset: 2,500 rows × 15 columns. Hold-out: **500 buildings**, fixed before modelling (seed 42, stratified on rating). Train+validation pool: 2,000 buildings, 5-fold CV for tuning and selection.
- Task-specific usable counts (train+validation / hold-out): cost 1,979 / 493; temperature_drop 1,956 / 486; units_saved 1,958 / 494; rating 2,000 / 500; high_rating 2,000 / 500; referral 2,000 / 500.
- Models and baselines are scored on the identical hold-out rows (verified: baselines are computed from the training pool and applied to the same test rows).
- 'Better than baseline' is claimed only when the paired-bootstrap (2,000 resamples) 95% CI of the improvement excludes 0 (`reports/significance.json`).

## 3. Hold-out results
| Task | Metric | Model | Baseline | Verdict |
|---|---|---|---|---|
| Cost | MAE / RMSE | ₹25,299 / ₹38,515 (R² 0.998) | MAE ₹637,024 | better |
| Temperature drop | MAE / RMSE | 0.414 / 0.512 °C (R² 0.948) | MAE 1.797 °C | better |
| Electricity saved | MAE / RMSE | 3.617 / 5.870 kWh (R² 0.987) | MAE 38.35 kWh | better |
| Rating | MAE | 0.769 stars (R² 0.36) | MAE 0.924 | better |
| High rating (≥4) | Accuracy / F1 / AUC | 0.688 / 0.614 / 0.755 | majority accuracy 0.576, all-positive F1 0.596 | accuracy better; F1 not_distinguishable |
| Referral (cut-off 0.25) | Accuracy / F1 / AUC | 0.512 / 0.569 / 0.655 (precision 0.425, recall 0.861) | majority accuracy 0.626, all-positive F1 0.544 | accuracy worse; F1 not_distinguishable; AUC above chance True |

- Error ratios vs baseline (MAE): cost 25.2×, temperature 4.3×, savings 10.6×.
- Referral confusion matrix (hold-out): {'tn': 95, 'fp': 218, 'fn': 26, 'tp': 161}. At cut-off 0.5: accuracy 0.642, F1 0.367.
- 80% likely-range coverage on the hold-out: cost 77.7%, temperature_drop 78.8%, units_saved 75.5% (slightly narrow).
- Predictability ceiling (diagnostic, not a model): referral AUC from realised savings+temperature drop 0.68; from realised rating 0.77.
- Independent replication on 500 never-explored rows (selection rerun from scratch): cost MAE 2.594e+04; temperature_drop MAE 0.4169; units_saved MAE 3.545; rating MAE 0.8258; high_rating acc 0.678 AUC 0.747; referral acc 0.518 AUC 0.653.

## 4. Data-quality findings
- 120 °C roof temperature: **58** records (32 before, 26 after) — confirmed invalid, set to missing.
- Exactly 0 kWh: **48** readings (26 before, 22 after) — treated as placeholders: an **inference** (no values at 1–3 kWh; physically inconsistent rows), not confirmed.
- Effect: electricity-savings hold-out MAE 4.45 → 3.62 kWh and RMSE 8.82 → 5.87 kWh after removing suspected placeholders (measured in this project; the earlier figures come from the first training run).
- Adverse outcomes retained (electricity went up, excluding zero-before rows): **167**.
- Missing input cells: 106 in 4 columns (location_type 24, building_size 28, building_age 23, num_ac_units 31). Rows with ≥1 missing/invalid input after cleaning: 164.
- Eligible rows per task: cost 2,472, temperature_drop 2,442, units_saved 2,452, rating 2,500, high_rating 2,500, referral 2,500. No record is deleted outright.
- Referral rate when rating ≥ 4 is 0.61 vs 0.09 when rating ≤ 2 (`reports/audit_evidence.txt`): why rating is forbidden as a referral input.

## 5. Stack, tests and status
- Stack: Python, pandas, scikit-learn, LightGBM, FastAPI, ReportLab (PDF); React 19 + TypeScript + Vite + Tailwind 4 + Recharts.
- Tests (last full runs): **153 backend (pytest)** and **30 frontend (Vitest, including axe accessibility checks and the what-if simulator)**; TypeScript type-check clean; production build succeeds; pages verified in a real browser with zero console errors.
- Optional AI assistant: built-in free mode (works with no key); **Google Gemini verified live** (tool calling against the real prediction service); **Anthropic/Claude integration built and tested only with a simulated client — live call failed with 'credit balance too low', so it is NOT live-verified**.
- Assistant guard-rails: only validated tool calls; told it has no durability/lifespan data; transient provider errors retry once then fall back to the built-in assistant.

## 6. Differences from the brief's starting-point facts
- Test counts: **153 backend** and **30 frontend** tests. Older figures of 107 / 22 (seen in earlier briefs) are out of date: review, assistant-provider, significance and what-if simulator tests were added.
- The brief says the live LLM API is unverified; the **Gemini path is now live-verified**, the Claude path is not.
- The brief's referral baseline note holds: all-positive F1 0.544 and majority accuracy 0.626 on the same hold-out; the model's accuracy is significantly worse than majority and its F1 is not distinguishable from all-positive.
- The evaluation-metrics image was **never supplied** (searched the whole project); metrics follow the written brief.
- Hackathon name and team members are **not recorded anywhere in the project**, so the title slide omits them.
- Not implemented (roadmap only): digital-twin simulator, carbon-savings tracker, geo-climate intelligence.

## 7. Visual-verification status
The 8-slide deck was rendered with Microsoft PowerPoint itself (PNG per slide plus a PDF in `presentation_assets/render8/` and `Cool_Roofing_Hackathon_Presentation_8slides.pdf`) and inspected from contact sheets and full-size renders. The canonical file name is currently held open (locked) by the user's PowerPoint, so the new deck was written to `Cool_Roofing_Hackathon_Presentation_8slides.pptx`; the earlier 15-slide deck is preserved in `presentation/archive_15_slide/`.

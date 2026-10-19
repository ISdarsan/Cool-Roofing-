# Assumptions & Decisions Log

Each entry: decision → why → how to change it.

## Requirements / interpretation
| # | Decision | Rationale | Change via |
|---|---|---|---|
| 1 | "High rating" = rating **≥ 4** | Organisers gave no definition; 4–5 is the conventional "satisfied" band. Labelled as an assumption in the UI, API and PDFs | `COOLROOF_HIGH_RATING_THRESHOLD` / `config.HIGH_RATING_THRESHOLD`, then retrain |
| 2 | Rating is modelled two ways: numeric regression (MAE) and a separately trained high/not-high classifier (accuracy) | The two required metrics need different formulations. Accuracy of a decision *derived* from the regression is reported too | `config.TASKS` |
| 3 | Referral F1 is **binary**, positive = True | Standard for the minority class; macro-F1 is also stored | — |
| 4 | `avg_units_*` are **monthly kWh** | Stated in the problem statement | — |
| 5 | `building_size` unit unknown | Not stated; treated as an opaque area unit | — |
| 6 | The evaluation-metrics image was not supplied | Metrics follow the written brief; no other metric is used for model selection | — |
| 7 | Electricity tariff is never assumed | The data has none; payback uses only a user-entered tariff and a tariff-free break-even tariff is shown instead | — |

## Data cleaning
| # | Decision | Rationale |
|---|---|---|
| 8 | Roof temperature **120.0 °C** → missing (58 records) | Repeated sentinel far outside a tight physical band; implied drops of ±90 °C vs. a valid 2–13.5 °C range. Not capped or guessed |
| 9 | Electricity **0 kWh** → missing (26 before, 22 after) | No values at 1–3 kWh exist while 4–8 kWh occur; the rows are physically inconsistent (e.g. 0 kWh before, 113 after; 273 before, 0 after, ratio 0 vs. a smallest valid ratio of 0.078). High-confidence inference, documented as such. Removing them improved savings MAE from 4.45 to 3.62 kWh and RMSE from 8.82 to 5.87 kWh on the hold-out |
| 10 | Negative savings (use went **up**) **kept** | Real adverse outcomes, concentrated in zero-AC low-usage buildings; deleting them would flatter the model |
| 11 | Exactly-zero savings kept | Coherent: no AC → no cooling load to reduce |
| 12 | Missing size/age/AC/location **imputed inside training folds** (median / most-frequent + missing indicator) | No target information used; fitted on training data only |
| 13 | Missing `building_size` → record excluded from **cost** only | Size is the dominant cost driver; imputing it injects label noise |
| 14 | Record-level audit trail with a classification and action for every flagged value | `reports/cleaning_audit.csv` |

## Modelling
| # | Decision | Rationale |
|---|---|---|
| 15 | One 20 % hold-out fixed **before** modelling (stratified on rating); 5-fold CV inside the remaining 80 % for tuning and selection | Honest generalisation estimate; all tasks share the same unseen buildings |
| 16 | Selection metric = primary task metric on out-of-fold predictions (MAE; accuracy; F1 at best OOF threshold). Baselines are never selectable | Matches the organisers' metrics |
| 17 | Multiplicative structure exploited: log-target ridge for cost; "fraction × base" boosting for temperature drop (× starting temperature) and savings (× current usage) | Cost ≈ size × rate; savings ≈ usage × fraction. These beat plain boosting in CV (cost MAE 25.7 k vs 40.2 k) and stay interpretable |
| 18 | Rating predictions clipped to [1, 5] | Documented scale constraint; no clipping occurred on the hold-out |
| 19 | Referral threshold = F1-optimal on out-of-fold probabilities, but 0.5 and accuracy-optimal operating points are stored and selectable in the UI | F1 and accuracy conflict for this imbalanced, weakly predictable target; both are always reported against trivial baselines |
| 20 | Probability calibration (sigmoid) applied only if it lowers out-of-fold Brier | Decided on training data only |
| 21 | Prediction intervals = empirical out-of-fold residual quantiles by prediction tercile (log-ratio for cost) | Distribution-free; hold-out coverage verified (≈ 76–80 % for the nominal 80 % interval; slightly under-covers for cost and savings) |
| 22 | Stacking predicted outcomes into rating/referral models rejected | CV gain within noise (AUC +0.01); a diagnostic shows even oracle realised outcomes reach only 0.68 AUC for referral |
| 23 | Explanations = exact interventional Shapley values over the 9 raw inputs (SHAP definition, computed directly) | No approximation, no heavyweight dependency; additivity (base + Σφ = prediction) is tested |

## Review-driven decisions (see model_review.md)
| # | Decision | Rationale |
|---|---|---|
| 28 | "Better than baseline" is claimed only when the paired-bootstrap 95 % CI of the improvement excludes 0; baselines are declared before looking at the hold-out | Point-estimate "wins" over trivial baselines were not statistically supported for referral/high-rating F1 |
| 29 | Classifier reliability: "moderate" needs hold-out ROC-AUC ≥ 0.7, "high" ≥ 0.8 | Referral (AUC 0.655) was labelled "moderate"; it is "low" |
| 30 | Hold-out exposure disclosed; selection replayed from scratch on 500 never-explored rows | Early exploration overlapped 80.6 % of the production hold-out at design level |

## Product
| # | Decision | Rationale |
|---|---|---|
| 24 | Method × quality options limited to roof/method pairs with ≥ 30 historical records | Avoids recommending combinations that never occur |
| 25 | Balanced score = weighted sum of per-metric min–max normalised values across the options shown; satisfaction uses **P(high rating)**, never the predicted star value | Metrics in different units are combined only after a documented normalisation; scores are explicitly *relative* |
| 26 | Referral likelihood is never printed in the customer PDF | Internal prioritisation signal, weakly predictable |
| 27 | LLM assistant is optional, tool-calling only, and cannot compute predictions itself | Core app has no LLM dependency |

## Known gaps (accepted)
- No timestamps/locations → no drift or geographic validation.
- Hold-out is one random split (≈ 500 rows), so metrics carry sampling error.
- The data-driven method/quality comparison is observational and may be confounded by unrecorded customer choices.

# Independent Model Review

Reviewer stance: assume the reported numbers are wrong until reproduced. Every figure below comes from
`scripts/review_models.py` (plain sklearn/NumPy, independent of the project's own metric and training code),
`src/coolroof/ml/significance.py` and `scripts/review_replication.py`. Raw output: `reports/review_models_output.txt`,
`reports/significance.json`, `reports/review_replication_pristine.json`. The checks that matter are also
permanent tests in `tests/test_model_review.py` (24 tests).

## Verdict
**No label leakage and no train/test contamination was found. Two evaluation defects were found and fixed; one
design-level exposure of the hold-out is disclosed and was neutralised by a from-scratch replication.**
The headline numbers stand for cost, cooling, savings, rating and high-rating. The referral model has a real but weak
ranking signal and **does not demonstrably beat the trivial classifiers on either required metric** — the product was
over-claiming this and has been corrected.

## 1. What was verified (all passed)
| Check | Method | Result |
|---|---|---|
| Hold-out is disjoint from training | id-set intersection; `split.json` equals a fresh re-derivation | 500 hold-out / 2,000 train pool, **0 overlap**; rating mix identical (e.g. 27.7 % vs 27.8 % at 4★) |
| No duplicates across the split | exact 9-feature-vector match of hold-out rows to training rows | **0 of 500** |
| Hold-out rows never in OOF/validation files | id intersection | 0 |
| Reported metrics are genuine hold-out metrics | reload each saved model, predict the hold-out rows rebuilt from the **raw** CSV, recompute MAE / RMSE / accuracy / F1 / AUC with sklearn | all six tasks reproduce to ≥ 9 significant digits; `test_predictions.csv` equals the model output (max diff ≤ 4e-9) |
| Baselines | recomputed from the training pool only (mean; median for rating; majority class of the training pool; predict-all-positive) | identical to the reported baselines |
| Models were trained on the training pool only | refit the selected configuration on train rows → compare with the saved model; also refit on train+hold-out | saved model == train-only refit (max diff ≤ 4e-8); a train+test refit differs (up to ₹9,620 on cost, 18 kWh on savings) |
| Preprocessing fitted on training rows only | read each saved imputer's `statistics_` and compare with medians of train rows vs train+hold-out rows | match train medians for all six models; do **not** match train+test medians |
| Validation scores reproducible | recompute validation MAE/accuracy from `oof_predictions.csv` | identical to metadata; hold-out/validation MAE ratio 0.98×–1.09× (no optimism gap) |
| Referral threshold chosen without the hold-out | recompute F1-optimal threshold from out-of-fold train predictions | 0.25 = stored value |
| Leakage control: shuffled labels | refit the selected config on permuted training labels, score the hold-out | R² −0.11…−0.02, AUC 0.47–0.48 (chance) |
| Leakage control: `building_id` only | LightGBM on the id alone | R² −0.05…−0.07, AUC 0.48–0.53 (chance) |
| Target definitions | `temperature_drop`, `units_saved`, `high_rating` recomputed from raw columns | exact |
| Feature contract | automated checks + API `extra="forbid"` | holds |

## 2. Investigation of suspiciously good results
| Result | Why it is plausible, not leaked |
|---|---|
| **Cost R² 0.998** (MAE 1.9 % median, 9.1 % worst) | Ablation (refit on train, score hold-out): without `building_size` R² falls to **0.22**; without method+quality **0.82**; without age 0.974; without roof type 0.985; removing location/AC/usage/start-temperature changes nothing (MAE ₹25,052 vs ₹25,299). Residuals are uncorrelated with the post-installation outcomes (r = 0.04 with realised savings, 0.03 with realised cooling). The log-residual std is 0.029: the data behave like `size × rate(method, quality, age, roof)` with ≈ 3 % multiplicative noise. A plain boosted model needs 60 % more error because it cannot extrapolate the product structure. |
| **Savings R² 0.987** | Realised savings correlate 0.93 with current usage; ablation: without `avg_units_before` R² 0.92, without method+quality 0.85. The jump from R² 0.95 to 0.99 came from **removing 48 placeholder rows**, not from any model change. |
| **Cooling R² 0.948** | Without method+quality R² collapses to **0.25**; starting temperature adds little (0.937 without). Valid drops span only 2.1–13.5 °C with small noise. |
| Rating R² 0.36, high-rating AUC 0.76, referral AUC 0.66 | Modest, as expected for outcomes driven by post-installation experience; diagnostic with *realised* outcomes (not a model input) reaches only 0.79 / 0.68 AUC, so these models are near their ceiling. |

## 3. Defects found and fixed
1. **"Beats baseline" was a point-estimate claim (evaluation defect).** Paired bootstrap (2,000 resamples, baselines declared before looking at the hold-out) shows:
   | Result | Verdict |
   |---|---|
   | cost / cooling / savings / rating MAE vs baseline | **significantly better** (CIs far above 0) |
   | high-rating accuracy +0.112 (CI +0.056…+0.166) | **significantly better** than the majority class |
   | high-rating F1 vs "everyone is high" | CI −0.041…+0.077 → **not distinguishable** |
   | referral accuracy −0.114 (CI −0.190…−0.038) | **significantly worse** than the majority class |
   | referral F1 vs "everyone refers" | CI −0.005…+0.053 → **not distinguishable** |
   | ROC-AUC high rating 0.755 (0.712–0.795); referral 0.655 (0.606–0.703) | both **above chance** |

   *Fix:* new `ml/significance.py` runs inside `coolroof train`, stores verdicts and CIs in the metadata, the evaluation report gained a significance table, the API status exposes the verdicts, the Performance/Overview pages now show "significantly better / not distinguishable / significantly worse" instead of green point-estimate badges, and the referral warning states that no operating point beats the trivial classifiers on both metrics. Reliability labels for classifiers now require ROC-AUC ≥ 0.7 for "moderate" (referral, AUC 0.655, is now "low").
2. **Mislabelled and over-confident wording about referral in docs/UI** (same cause) — corrected in README, evaluation report and UI.

## 4. Design-level exposure of the hold-out (disclosed, mitigated)
My earliest exploratory experiments used `train_test_split(seed=42, no stratification)` and ran CV on its 80 % part. The
production hold-out (stratified, same seed) overlaps that exploration pool for **403 of 500 rows (80.6 %)**. No parameter was
fitted on hold-out rows, but model-*family* ideas (log target, ratio target, boosted trees) were informed by CV that included
them. Two further decisions were made after looking at hold-out results: (i) the discovery of the 0 kWh placeholder was triggered by a
hold-out residual (actual −113, predicted 0); it was then verified from the full-data value distribution and physical
consistency independently of any metric, and applied uniformly to all rows; (ii) the presentation of referral operating points and
baselines was changed after seeing the numbers (reporting only; the threshold itself is chosen from out-of-fold training data).

*Mitigation:* `scripts/review_replication.py` takes the 500 rows that exploration never touched (**0 % inside the exploration pool**), uses them as the
hold-out, and reruns hyper-parameter tuning, model selection and evaluation **from scratch** on the other 2,000 rows.
| Task | Pristine hold-out | Production hold-out |
|---|---|---|
| Cost MAE / R² | ₹25,943 / 0.997 | ₹25,299 / 0.998 |
| Cooling MAE / R² | 0.417 °C / 0.938 | 0.414 °C / 0.948 |
| Savings MAE / R² | 3.54 kWh / 0.988 | 3.62 kWh / 0.987 |
| Rating MAE (baseline) | 0.826 (0.958) | 0.769 (0.924) |
| High-rating accuracy / AUC | 0.678 / 0.747 | 0.688 / 0.755 |
| Referral F1 / AUC | 0.579 / 0.653 | 0.569 / 0.655 |
Same model families win in five of six tasks (referral: logistic vs LightGBM, tied in validation). No optimism attributable to the exposure is visible; the
rating result varies most (MAE 0.77 vs 0.83), consistent with sampling error (CI ±0.045).

## 5. Residual risks (not defects)
- One 500-row hold-out: sampling error of the order of the CIs above.
- Interval coverage for the nominal 80 % interval is 75.5–79.8 % (savings 75.5 %: ~2.5 standard errors below nominal) — slightly narrow for savings and cost; documented.
- Validation scores of tuned models are best-of-search; the hold-out (and the replication) are the unbiased numbers and agree within ≈ 10 %.
- Rating/referral remain weakly predictable from quote-time facts; the product states this.

import json
from pathlib import Path

ROOT = Path(r"D:\bia hackathon")
M = json.loads((ROOT / "models/model_metadata.json").read_text(encoding="utf-8"))
T = M["tasks"]
S = json.loads((ROOT / "reports/data_quality_summary.json").read_text(encoding="utf-8"))
P = json.loads((ROOT / "reports/data_profile.json").read_text(encoding="utf-8"))
I = json.loads((ROOT / "reports/insights.json").read_text(encoding="utf-8"))
rep = json.loads((ROOT / "reports/review_replication_pristine.json").read_text(encoding="utf-8"))["tasks"]
inr = lambda v: "₹" + f"{v:,.0f}"
c, d, u, r, h, f = (T[k] for k in ("cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral"))
z, ts = S["zero_consumption_placeholders"], S["temperature_sentinel_rows"]
miss = {x["column"]: x["before"] for x in P["missingness"] if x["before"] > 0}
ceil = I["predictability_ceiling"]

L = ["# Verified fact sheet (source of truth for the deck)", "",
     "Every figure below was read from the project's saved artifacts (`models/model_metadata.json`, `reports/*.json`, `reports/*.md`) when this file was generated, "
     "or from commands run during this project (test counts). Nothing is assumed.", "",
     "## 1. Models and task mapping",
     "Five required prediction tasks, **six trained models** (customer rating is both a numeric regression and a high/not-high classifier):", "",
     "| Task | Model name in the repo | Type | Selected model | Baseline (hold-out) |", "|---|---|---|---|---|",
     f"| A. Installation cost | cost | Regression | {c['selected_label']} | training mean |",
     f"| B. Temperature drop | temperature_drop | Regression | {d['selected_label']} | training mean |",
     f"| C. Electricity saved | units_saved | Regression | {u['selected_label']} | training mean |",
     f"| D1. Customer rating (1–5) | rating | Regression | {r['selected_label']} | training median |",
     f"| D2. High rating (rating ≥ 4, assumed) | high_rating | Classification | {h['selected_label']} | majority class |",
     f"| E. Referral | referral | Classification | {f['selected_label']} | all-positive (F1), majority class (accuracy) |", "",
     "## 2. Evaluation design",
     f"- Dataset: {M['data']['n_rows']:,} rows × 15 columns. Hold-out: **{M['data']['n_test_buildings']} buildings**, fixed before modelling (seed {M['seed']}, stratified on rating). Train+validation pool: {M['data']['n_train_validation_buildings']:,} buildings, {M['cv_folds']}-fold CV for tuning and selection.",
     "- Task-specific usable counts (train+validation / hold-out): " + "; ".join(f"{k} {T[k]['n_train_validation']:,} / {T[k]['n_test']}" for k in ("cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral")) + ".",
     "- Models and baselines are scored on the identical hold-out rows (verified: baselines are computed from the training pool and applied to the same test rows).",
     "- 'Better than baseline' is claimed only when the paired-bootstrap (2,000 resamples) 95% CI of the improvement excludes 0 (`reports/significance.json`).", "",
     "## 3. Hold-out results",
     "| Task | Metric | Model | Baseline | Verdict |", "|---|---|---|---|---|",
     f"| Cost | MAE / RMSE | {inr(c['test']['mae'])} / {inr(c['test']['rmse'])} (R² {c['test']['r2']:.3f}) | MAE {inr(c['test_baseline']['mae'])} | {c['significance']['verdict']} |",
     f"| Temperature drop | MAE / RMSE | {d['test']['mae']:.3f} / {d['test']['rmse']:.3f} °C (R² {d['test']['r2']:.3f}) | MAE {d['test_baseline']['mae']:.3f} °C | {d['significance']['verdict']} |",
     f"| Electricity saved | MAE / RMSE | {u['test']['mae']:.3f} / {u['test']['rmse']:.3f} kWh (R² {u['test']['r2']:.3f}) | MAE {u['test_baseline']['mae']:.2f} kWh | {u['significance']['verdict']} |",
     f"| Rating | MAE | {r['test']['mae']:.3f} stars (R² {r['test']['r2']:.2f}) | MAE {r['test_baseline']['mae']:.3f} | {r['significance']['verdict']} |",
     f"| High rating (≥4) | Accuracy / F1 / AUC | {h['test']['accuracy']:.3f} / {h['test']['f1']:.3f} / {h['test']['roc_auc']:.3f} | majority accuracy {h['baseline_accuracy']:.3f}, all-positive F1 {h['baseline_f1']:.3f} | accuracy {h['significance']['accuracy_verdict']}; F1 {h['significance']['f1_verdict']} |",
     f"| Referral (cut-off {f['threshold']}) | Accuracy / F1 / AUC | {f['test']['accuracy']:.3f} / {f['test']['f1']:.3f} / {f['test']['roc_auc']:.3f} (precision {f['test']['precision']:.3f}, recall {f['test']['recall']:.3f}) | majority accuracy {f['baseline_accuracy']:.3f}, all-positive F1 {f['baseline_f1']:.3f} | accuracy {f['significance']['accuracy_verdict']}; F1 {f['significance']['f1_verdict']}; AUC above chance {f['significance']['auc_above_chance']} |", "",
     f"- Error ratios vs baseline (MAE): cost {c['test_baseline']['mae'] / c['test']['mae']:.1f}×, temperature {d['test_baseline']['mae'] / d['test']['mae']:.1f}×, savings {u['test_baseline']['mae'] / u['test']['mae']:.1f}×.",
     f"- Referral confusion matrix (hold-out): {f['test']['confusion_matrix']}. At cut-off 0.5: accuracy {f['test_at_0_5']['accuracy']:.3f}, F1 {f['test_at_0_5']['f1']:.3f}.",
     "- 80% likely-range coverage on the hold-out: " + ", ".join(f"{k} {T[k]['interval_coverage_test']['0.8']:.1%}" for k in ("cost", "temperature_drop", "units_saved")) + " (slightly narrow).",
     f"- Predictability ceiling (diagnostic, not a model): referral AUC from realised savings+temperature drop {ceil['referral_auc']['from_realised_savings_and_temperature_drop']:.2f}; from realised rating {ceil['referral_auc']['from_realised_rating']:.2f}.",
     "- Independent replication on 500 never-explored rows (selection rerun from scratch): " + "; ".join(
         f"{k} {('MAE ' + format(rep[k]['test']['mae'], '.4g')) if 'mae' in rep[k]['test'] else ('acc ' + format(rep[k]['test']['accuracy'], '.3f') + ' AUC ' + format(rep[k]['test']['roc_auc'], '.3f'))}"
         for k in ("cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral")) + ".", "",
     "## 4. Data-quality findings",
     f"- 120 °C roof temperature: **{ts['before'] + ts['after']}** records ({ts['before']} before, {ts['after']} after) — confirmed invalid, set to missing.",
     f"- Exactly 0 kWh: **{z['before'] + z['after']}** readings ({z['before']} before, {z['after']} after) — treated as placeholders: an **inference** (no values at 1–3 kWh; physically inconsistent rows), not confirmed.",
     f"- Effect: electricity-savings hold-out MAE 4.45 → {u['test']['mae']:.2f} kWh and RMSE 8.82 → {u['test']['rmse']:.2f} kWh after removing suspected placeholders (measured in this project; the earlier figures come from the first training run).",
     f"- Adverse outcomes retained (electricity went up, excluding zero-before rows): **{S['negative_savings_rows']}**.",
     f"- Missing input cells: {sum(miss.values())} in {len(miss)} columns (" + ", ".join(f"{k} {v}" for k, v in miss.items()) + f"). Rows with ≥1 missing/invalid input after cleaning: {S['rows_with_any_missing_feature']}.",
     "- Eligible rows per task: " + ", ".join(f"{k} {v:,}" for k, v in S["eligible_rows_by_task"].items()) + ". No record is deleted outright.",
     "- Referral rate when rating ≥ 4 is 0.61 vs 0.09 when rating ≤ 2 (`reports/audit_evidence.txt`): why rating is forbidden as a referral input.", "",
     "## 5. Stack, tests and status",
     "- Stack: Python, pandas, scikit-learn, LightGBM, FastAPI, ReportLab (PDF); React 19 + TypeScript + Vite + Tailwind 4 + Recharts.",
     "- Tests (last full runs): **153 backend (pytest)** and **30 frontend (Vitest, including axe accessibility checks and the what-if simulator)**; TypeScript type-check clean; production build succeeds; pages verified in a real browser with zero console errors.",
     "- Optional AI assistant: built-in free mode (works with no key); **Google Gemini verified live** (tool calling against the real prediction service); **Anthropic/Claude integration built and tested only with a simulated client — live call failed with 'credit balance too low', so it is NOT live-verified**.",
     "- Assistant guard-rails: only validated tool calls; told it has no durability/lifespan data; transient provider errors retry once then fall back to the built-in assistant.", "",
     "## 6. Differences from the brief's starting-point facts",
     "- Test counts: **153 backend** and **30 frontend** tests. Older figures of 107 / 22 (seen in earlier briefs) are out of date: review, assistant-provider, significance and what-if simulator tests were added.",
     "- The brief says the live LLM API is unverified; the **Gemini path is now live-verified**, the Claude path is not.",
     "- The brief's referral baseline note holds: all-positive F1 0.544 and majority accuracy 0.626 on the same hold-out; the model's accuracy is significantly worse than majority and its F1 is not distinguishable from all-positive.",
     "- The evaluation-metrics image was **never supplied** (searched the whole project); metrics follow the written brief.",
     "- Hackathon name and team members are **not recorded anywhere in the project**, so the title slide omits them.",
     "- Not implemented (roadmap only): digital-twin simulator, carbon-savings tracker, geo-climate intelligence.", "",
     "## 7. Visual-verification status",
     "The 8-slide deck was rendered with Microsoft PowerPoint itself (PNG per slide plus a PDF in `presentation_assets/render8/` and `Cool_Roofing_Hackathon_Presentation_8slides.pdf`) and inspected from contact sheets and full-size renders. The canonical file name is currently held open (locked) by the user's PowerPoint, so the new deck was written to `Cool_Roofing_Hackathon_Presentation_8slides.pptx`; the earlier 15-slide deck is preserved in `presentation/archive_15_slide/`."]
(ROOT / "presentation/verified_fact_sheet.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("fact sheet written;", len(L), "lines")

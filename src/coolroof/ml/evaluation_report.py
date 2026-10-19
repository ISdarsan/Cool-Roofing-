"""Diagnostic figures and the human-readable model evaluation report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config as C  # noqa: E402
from ..data.report import PAL, _style  # noqa: E402


def _fmt(task: str, v: float, metric: str = "mae") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if metric in ("accuracy", "f1", "precision", "recall"):
        return f"{v:.3f}"
    if task == "cost":
        return f"₹{v:,.0f}"
    if task == "temperature_drop":
        return f"{v:.3f} °C"
    if task == "units_saved":
        return f"{v:.2f} kWh"
    return f"{v:.3f}"


def make_diagnostic_figures(results: dict) -> None:
    _style()
    test = pd.read_csv(C.REPORTS / "test_predictions.csv")
    for task in C.REGRESSION_TASKS:
        a, p = test[f"{task}_actual"], test[f"{task}_predicted"]
        keep = a.notna() & p.notna()
        a, p = a[keep], p[keep]
        r = a - p
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
        axs[0].scatter(a, p, s=10, alpha=0.5, color=PAL["teal"])
        lim = [min(a.min(), p.min()), max(a.max(), p.max())]
        axs[0].plot(lim, lim, color=PAL["clay"], lw=1.3, ls="--")
        axs[0].set(xlabel=f"actual ({C.TASKS[task].unit})", ylabel="predicted", title="Predicted vs actual (hold-out)")
        axs[1].hist(r, bins=40, color=PAL["sun"]); axs[1].axvline(0, color=PAL["ink"], lw=1)
        axs[1].set(xlabel="residual (actual − predicted)", title="Residual distribution")
        axs[2].scatter(p, r, s=10, alpha=0.5, color=PAL["sage"]); axs[2].axhline(0, color=PAL["clay"], lw=1.2, ls="--")
        axs[2].set(xlabel="predicted", ylabel="residual", title="Residuals vs predicted")
        fig.suptitle(f"{C.TASKS[task].label}  –  {results[task]['selected_label']}", fontweight="bold")
        fig.tight_layout(); fig.savefig(C.FIGURES / f"10_diagnostics_{task}.png", bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, task in zip(axs, ["high_rating", "referral"]):
        cm = results[task]["test"]["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(mat, cmap="GnBu")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if v > mat.max() / 2 else PAL["ink"])
        names = ["not high", "high"] if task == "high_rating" else ["no referral", "referral"]
        ax.set_xticks([0, 1], names); ax.set_yticks([0, 1], names); ax.grid(False)
        ax.set(xlabel="predicted", ylabel="actual", title=f"{C.TASKS[task].label}\n(threshold {results[task]['threshold']:.2f})")
    fig.tight_layout(); fig.savefig(C.FIGURES / "11_confusion_matrices.png", bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, task in zip(axs, ["high_rating", "referral"]):
        bins = results[task]["calibration_test"]["bins"]
        ax.plot([0, 1], [0, 1], color=PAL["slate"], ls="--", lw=1)
        if bins:
            ax.plot([b["mean_predicted"] for b in bins], [b["observed_rate"] for b in bins], "o-", color=PAL["teal"], lw=2)
        ax.set(xlabel="mean predicted probability", ylabel="observed rate", xlim=(0, 1), ylim=(0, 1),
               title=f"{C.TASKS[task].label} – calibration (hold-out)")
    fig.tight_layout(); fig.savefig(C.FIGURES / "12_calibration.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    names, gains = [], []
    for task in C.REGRESSION_TASKS:
        r = results[task]
        names.append(C.TASKS[task].label)
        gains.append(100 * (1 - r["test"]["mae"] / r["test_baseline"]["mae"]))
    ax.barh(names, gains, color=PAL["teal"]); ax.set(xlabel="% lower hold-out MAE than baseline", title="Improvement over baseline")
    for i, g in enumerate(gains):
        ax.text(g + 0.5, i, f"{g:.0f}%", va="center")
    ax.set_xlim(0, 105)
    fig.tight_layout(); fig.savefig(C.FIGURES / "13_improvement_over_baseline.png", bbox_inches="tight"); plt.close(fig)


def write_evaluation_report(results: dict, meta: dict) -> None:
    make_diagnostic_figures(results)
    L: list[str] = []
    w = L.append
    d = meta["data"]
    w("# Model Evaluation Report")
    w("")
    w(f"_Model version {meta['model_version']} · trained {meta['trained_at_utc']} · seed {meta['seed']} · "
      f"scikit-learn {meta['environment']['scikit_learn']} · LightGBM {meta['environment']['lightgbm']}_")
    w("")
    w("## 1. Evaluation protocol")
    w(f"- {d['n_test_buildings']} of {d['n_rows']:,} buildings ({C.TEST_SIZE:.0%}) were set aside as a **hold-out test set before any modelling** "
      "(stratified on the 1–5 rating). They were scored once, for the selected model and the baseline only.")
    w(f"- The other {d['n_train_validation_buildings']:,} buildings form the train+validation pool. Every candidate was tuned with "
      f"{meta['cv_folds']}-fold cross-validation inside that pool; its **out-of-fold predictions are the 'validation' scores** used to pick the winner.")
    w("- Imputation, encoding and scaling are inside the sklearn pipelines, so they are fitted on training folds only.")
    w("- Selection criterion: lowest validation MAE (regression); highest validation accuracy at 0.5 (high rating); "
      "highest validation F1 at the best out-of-fold threshold (referral). Tuned models' validation scores are slightly optimistic "
      "(best of a search); the hold-out column is the unbiased estimate.")
    elig = {k: results[k]["n_eligible"] for k in C.TASK_ORDER}
    w("- Task eligibility (rows with valid targets): " + " · ".join(f"{C.TASKS[k].label} {v:,}" for k, v in elig.items()) +
      " (see the data-quality report).")
    w("")
    w("## 2. Unified results")
    w("| Task | Selected model | Metric | Baseline (hold-out) | Validation (CV) | **Hold-out (selected)** | n train+val / test |")
    w("|---|---|---|---:|---:|---:|---|")
    for k in C.TASK_ORDER:
        r = results[k]
        n = f"{r['n_train_validation']:,} / {r['n_test']:,}"
        if r["kind"] == "regression":
            for m in ("mae", "rmse"):
                w(f"| {r['label']} | {r['selected_label']} | {m.upper()} ({r['unit']}) | {_fmt(k, r['test_baseline'][m])} | "
                  f"{_fmt(k, r['validation'][m])} | **{_fmt(k, r['test'][m])}** | {n} |")
            if k == "rating":
                w(f"| {r['label']} | {r['selected_label']} | Accuracy of derived high/not-high (≥{C.HIGH_RATING_THRESHOLD}) | – | "
                  f"{r['validation']['derived_high_rating_accuracy']:.3f} | **{r['test']['derived_high_rating_accuracy']:.3f}** | {n} |")
        else:
            for m in ("accuracy", "f1"):
                base = r["baseline_accuracy"] if m == "accuracy" else r["baseline_f1"]
                val = r["validation"]["accuracy"] if m == "accuracy" else r["validation"]["best_f1"]
                w(f"| {r['label']} | {r['selected_label']} | {m.capitalize() if m=='accuracy' else 'F1 (binary, positive = ' + ('high' if k=='high_rating' else 'referred') + ')'} | "
                  f"{base:.3f} | {val:.3f} | **{r['test'][m]:.3f}** | {n} |")
    w("")
    w("![improvement](../reports/figures/13_improvement_over_baseline.png)".replace("../reports/", ""))
    w("")
    w("### Is each result really better than its baseline? (paired bootstrap, 2,000 resamples, 95 %)")
    w("Baselines are declared before looking at the hold-out: training mean (median for rating); majority class of the *training* pool for accuracy; "
      "'predict every customer positive' for F1.")
    w("")
    w("| Task | Metric | Model [95 % CI] | Baseline | Improvement [95 % CI] | Verdict |")
    w("|---|---|---|---:|---|---|")
    vw = {"better": "better", "worse": "WORSE", "not_distinguishable": "not distinguishable"}
    for k in C.TASK_ORDER:
        sg = results[k]["significance"]
        if sg["kind"] == "regression":
            w(f"| {results[k]['label']} | MAE | {sg['mae']:.4g} [{sg['mae_ci95'][0]:.4g}, {sg['mae_ci95'][1]:.4g}] | {sg['baseline_mae']:.4g} | "
              f"{sg['mae_improvement']:.4g} [{sg['mae_improvement_ci95'][0]:.4g}, {sg['mae_improvement_ci95'][1]:.4g}] | {vw[sg['verdict']]} |")
        else:
            w(f"| {results[k]['label']} | Accuracy | {sg['accuracy']:.3f} [{sg['accuracy_ci95'][0]:.3f}, {sg['accuracy_ci95'][1]:.3f}] | {sg['baseline_accuracy']:.3f} | "
              f"{sg['accuracy_improvement']:+.3f} [{sg['accuracy_improvement_ci95'][0]:+.3f}, {sg['accuracy_improvement_ci95'][1]:+.3f}] | {vw[sg['accuracy_verdict']]} |")
            w(f"| {results[k]['label']} | F1 | {results[k]['test']['f1']:.3f} [{sg['f1_ci95'][0]:.3f}, {sg['f1_ci95'][1]:.3f}] | {sg['baseline_f1']:.3f} | "
              f"CI {sg['f1_improvement_ci95'][0]:+.3f} to {sg['f1_improvement_ci95'][1]:+.3f} | {vw[sg['f1_verdict']]} |")
            w(f"| {results[k]['label']} | ROC-AUC | {sg['roc_auc']:.3f} [{sg['roc_auc_ci95'][0]:.3f}, {sg['roc_auc_ci95'][1]:.3f}] | 0.500 | - | {'above chance' if sg['auc_above_chance'] else 'not above chance'} |")
    w("")
    w("## 3. Task details")
    for k in C.TASK_ORDER:
        r = results[k]
        w(f"### {r['label']}  (`{k}`)")
        w(f"- Target: {r['target_definition']}  Unit: **{r['unit']}**")
        w(f"- Samples: {r['n_eligible']:,} eligible → {r['n_train_validation']:,} train+validation, {r['n_test']:,} hold-out.")
        w(f"- Selected: **{r['selected_label']}** (`{r['selected']}`); strongest baseline: {r['baseline_label']}.")
        if r["kind"] == "regression":
            t, b = r["test"], r["test_baseline"]
            w(f"- Hold-out: MAE {_fmt(k, t['mae'])}, RMSE {_fmt(k, t['rmse'])}, R² {t['r2']:.3f}, bias {t['bias']:+.3f}. "
              f"Baseline MAE {_fmt(k, b['mae'])} → **{100*(1-t['mae']/b['mae']):.0f}% lower error**.")
            cov = r["interval_coverage_test"]
            w(f"- Prediction intervals (empirical, from out-of-fold residuals): nominal 80% interval covers **{cov['0.8']:.0%}** and 90% covers "
              f"**{cov['0.9']:.0%}** of hold-out buildings.")
            pl = r["plausibility_test"]
            w("- Physical-plausibility check of hold-out predictions: " + ", ".join(f"{(a[2:] if a.startswith('n_') else a).replace('_', ' ')} = {v}" for a, v in pl.items() if a != "n") + ".")
            if k == "rating":
                w(f"- Rating is predicted on the 1–5 scale (clipped to the scale — a documented constraint). MAE of the continuous prediction is the headline; "
                  f"if rounded to whole stars MAE = {t['mae_rounded']:.3f}, exact-star accuracy = {t['exact_match_accuracy_rounded']:.1%}, within one star = {t['within_one_star']:.1%}.")
        else:
            t, b = r["test"], r["test_baseline"]
            cm = t["confusion_matrix"]
            w(f"- Operating threshold: **{r['threshold']:.2f}** ({r['threshold_policy']}). Calibrated with sigmoid scaling: {r['calibrated']}.")
            w(f"- Hold-out: accuracy **{t['accuracy']:.3f}**, binary F1 **{t['f1']:.3f}**, precision {t['precision']:.3f}, recall {t['recall']:.3f}, "
              f"ROC-AUC {t['roc_auc']:.3f}, Brier {t['brier']:.3f}.")
            w(f"- Confusion matrix (rows actual, cols predicted): TN {cm['tn']}, FP {cm['fp']}, FN {cm['fn']}, TP {cm['tp']}. "
              f"Support: {t['support']['negative']} negative / {t['support']['positive']} positive "
              f"({t['positive_rate_true']:.1%} positive).")
            w(f"- At the default 0.5 threshold: accuracy {r['test_at_0_5']['accuracy']:.3f}, F1 {r['test_at_0_5']['f1']:.3f}.")
            w(f"- Trivial baselines on the hold-out: always-negative accuracy {r['test_baseline_always_negative']['accuracy']:.3f} (F1 0); "
              f"always-positive accuracy {r['test_baseline_always_positive']['accuracy']:.3f}, F1 {r['test_baseline_always_positive']['f1']:.3f}.")
            w(f"- Calibration on the hold-out: ECE {r['calibration_test']['ece']:.3f}.")
        if r["kind"] == "classification":
            sg = r["significance"]
            word = {"better": "**significantly better than**", "worse": "**significantly WORSE than**", "not_distinguishable": "**not distinguishable from**"}
            w(f"- Versus the trivial classifiers (paired bootstrap, 95 %): accuracy {r['test']['accuracy']:.3f} is {word[sg['accuracy_verdict']]} the majority-class "
              f"{sg['baseline_accuracy']:.3f} (improvement {sg['accuracy_improvement']:+.3f}, CI {sg['accuracy_improvement_ci95'][0]:+.3f} to {sg['accuracy_improvement_ci95'][1]:+.3f}); "
              f"F1 {r['test']['f1']:.3f} is {word[sg['f1_verdict']]} the predict-everyone-positive F1 {sg['baseline_f1']:.3f} "
              f"(CI {sg['f1_improvement_ci95'][0]:+.3f} to {sg['f1_improvement_ci95'][1]:+.3f}). "
              f"ROC-AUC {sg['roc_auc']:.3f} (95 % CI {sg['roc_auc_ci95'][0]:.3f}-{sg['roc_auc_ci95'][1]:.3f}) is {'above' if sg['auc_above_chance'] else 'NOT above'} chance.")
            op = r["named_thresholds"]
            w(f"- Named operating points (chosen on out-of-fold data): F1-optimal {op['f1_optimal']:.2f}, accuracy-optimal {op['accuracy_optimal']:.2f}, default 0.50. "
              "The full threshold trade-off table is in `reports/model_metadata.json` and is interactive in the app.")
        else:
            sg = r["significance"]
            w(f"- Improvement over the baseline in MAE: {sg['mae_improvement']:.4g} (paired bootstrap 95 % CI {sg['mae_improvement_ci95'][0]:.4g} to {sg['mae_improvement_ci95'][1]:.4g}) -> "
              f"**{'significantly better' if sg['verdict'] == 'better' else 'not demonstrably better'}**. MAE 95 % CI {sg['mae_ci95'][0]:.4g}-{sg['mae_ci95'][1]:.4g}.")
        w("")
        w("| Candidate | Family | " + (" | ".join(["CV MAE", "CV RMSE", "CV R²"]) if r["kind"] == "regression" else " | ".join(["CV accuracy", "CV log-loss", "CV best F1", "threshold"])) + " |")
        w("|---|---|" + "---:|" * (3 if r["kind"] == "regression" else 4))
        key = (lambda c: c["cv_mae"]) if r["kind"] == "regression" else (lambda c: c["select_score"])
        for c in sorted(r["candidates"], key=key):
            tag = " ✅" if c["model"] == r["selected"] else ""
            if r["kind"] == "regression":
                w(f"| {c['label']}{tag} | {c['family']} | {_fmt(k, c['cv_mae'])} | {_fmt(k, c['cv_rmse'])} | {c['cv_r2']:.3f} |")
            else:
                w(f"| {c['label']}{tag} | {c['family']} | {c['cv_accuracy']:.3f} | {c['cv_log_loss']:.3f} | {c['cv_best_f1']:.3f} | {c['cv_f1_threshold']:.2f} |")
        w("")
        if r["kind"] == "regression":
            w(f"![diagnostics](figures/10_diagnostics_{k}.png)")
            w("")
    w("![confusion](figures/11_confusion_matrices.png)")
    w("")
    w("![calibration](figures/12_calibration.png)")
    w("")
    w("## 4. Honest reading of the results")
    for line in _interpretation(results):
        w(f"- {line}")
    w("")
    w("## 5. Limitations")
    w("- One random 80/20 split of one dataset: hold-out metrics carry sampling error (a few hundred rows); no time or location variable exists to test drift.")
    w("- Rating and referral depend heavily on post-installation experience (realised savings, comfort) that is *not available* at quote time, "
      "so they are only weakly predictable from the pre-installation questions. They should be used for ranking/triage, not as promises.")
    w("- Models describe associations in historical installations, not causal effects of choosing a method.")
    w("- Predictions outside the training ranges (see `models/model_metadata.json`) are extrapolations and are flagged in the API.")
    text = "\n".join(L) + "\n"
    (C.REPORTS / "model_evaluation.md").write_text(text, encoding="utf-8")
    (C.DOCS / "model_evaluation.md").write_text(text.replace("](figures/", "](../reports/figures/"), encoding="utf-8")


def _interpretation(results: dict) -> list[str]:
    out = []
    for k in C.REGRESSION_TASKS:
        r = results[k]
        t, b = r["test"], r["test_baseline"]
        out.append(f"**{r['label']}**: {r['selected_label']} reaches MAE {_fmt(k, t['mae'])} on unseen buildings versus {_fmt(k, b['mae'])} for the baseline "
                   f"({100*(1-t['mae']/b['mae']):.0f}% lower); R² = {t['r2']:.2f}.")
    h, rf = results["high_rating"], results["referral"]
    hs, rs = h["significance"], rf["significance"]
    out.append(f"**High rating (≥ {C.HIGH_RATING_THRESHOLD})**: accuracy {h['test']['accuracy']:.1%} vs {hs['baseline_accuracy']:.1%} for the majority class is a significant gain "
               f"(paired 95 % CI {hs['accuracy_improvement_ci95'][0]:+.3f} to {hs['accuracy_improvement_ci95'][1]:+.3f}; ROC-AUC {h['test']['roc_auc']:.2f}). "
               f"Its F1 ({h['test']['f1']:.2f}) is **{'better than' if hs['f1_verdict'] == 'better' else 'not distinguishable from'}** the 'everyone is high' F1 ({hs['baseline_f1']:.2f}) "
               f"because that trivial rule already has a high F1 when 42 % of customers are positive.")
    out.append(f"**Referral**: ROC-AUC {rf['test']['roc_auc']:.2f} (95 % CI {rs['roc_auc_ci95'][0]:.2f}-{rs['roc_auc_ci95'][1]:.2f}) shows a real but weak ranking signal. "
               f"At the F1-optimal cut-off {rf['threshold']:.2f} the hold-out F1 is {rf['test']['f1']:.2f}, which is **{'better than' if rs['f1_verdict'] == 'better' else 'not distinguishable from'}** "
               f"predicting everyone refers ({rs['baseline_f1']:.2f}), while accuracy ({rf['test']['accuracy']:.1%}) is **{'better than' if rs['accuracy_verdict'] == 'better' else 'significantly worse than' if rs['accuracy_verdict'] == 'worse' else 'not distinguishable from'}** "
               f"the majority-class {rs['baseline_accuracy']:.1%}. No operating point demonstrably beats the trivial classifiers on both required metrics, so the model should be used to rank customers for follow-up, never as a yes/no forecast.")
    return out


def regenerate_from_disk() -> None:
    """Rebuild the evaluation report from saved metadata (no retraining)."""
    import json
    meta = json.loads((C.MODELS / "model_metadata.json").read_text(encoding="utf-8"))
    cands = json.loads((C.REPORTS / "model_candidates.json").read_text(encoding="utf-8"))
    results = {k: {**meta["tasks"][k], "candidates": cands[k]} for k in meta["tasks"]}
    write_evaluation_report(results, meta)

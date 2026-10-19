"""Train, compare, select and evaluate all models.

Protocol (identical for every task)
-----------------------------------
1. A hold-out test set (20 % of buildings) is fixed *before* any modelling.
2. The remaining 80 % is the train+validation pool. Every candidate is tuned with
   5-fold cross-validation inside that pool; its out-of-fold (OOF) predictions give the
   *validation* scores used for model selection.
3. The winner is refit on the whole pool and scored exactly once on the hold-out set.
   The baseline is scored on the hold-out set too, for comparison.
4. Preprocessing (imputation, encoding, scaling) lives inside the pipelines, so it is
   fitted on training folds only.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_predict

from .. import config as C
from ..data.clean import load_cleaned
from ..features import build_pipeline, design_matrix, validate_all_contracts
from .candidates import Candidate, get_candidates
from .metrics import best_f1_threshold, calibration_table, classification_metrics, regression_metrics

N_JOBS = -1
INTERVAL_LEVELS = (0.80, 0.90)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _target(df: pd.DataFrame, task_key: str) -> np.ndarray:
    spec = C.TASKS[task_key]
    y = df[spec.target]
    if spec.kind == "classification":
        return y.astype(float).astype(int).to_numpy()
    return y.astype(float).to_numpy()


def _splits(task_key: str, X: pd.DataFrame, y: np.ndarray):
    if C.TASKS[task_key].kind == "classification":
        cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    else:
        cv = KFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    return list(cv.split(X, y))


def _search(cand: Candidate, est, X, y, splits, task_key: str):
    scoring = "neg_mean_absolute_error" if C.TASKS[task_key].kind == "regression" else "neg_log_loss"
    if cand.search == "grid":
        s = GridSearchCV(est, cand.param_space(), cv=splits, scoring=scoring, n_jobs=N_JOBS, refit=False)
    else:
        s = RandomizedSearchCV(est, cand.param_space(), n_iter=cand.n_iter, cv=splits, scoring=scoring,
                               n_jobs=N_JOBS, refit=False, random_state=C.SEED)
    s.fit(X, y)
    return s.best_params_, float(-s.best_score_)


def _fold_mae(y, oof, splits) -> tuple[float, float]:
    maes = [np.mean(np.abs(oof[v] - y[v])) for _, v in splits]
    return float(np.mean(maes)), float(np.std(maes, ddof=1))


# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------
def fit_interval_model(y: np.ndarray, oof: np.ndarray, log_ratio: bool) -> dict:
    """Empirical residual quantiles from OOF predictions, by tercile of the prediction.

    cost uses log(y / prediction) (errors scale with price); other targets use y - prediction.
    """
    res = np.log(np.clip(y, 1, None) / np.clip(oof, 1, None)) if log_ratio else (y - oof)
    edges = [float(q) for q in np.quantile(oof, [1 / 3, 2 / 3])]
    bins = np.digitize(oof, edges)
    model = {"kind": "log_ratio" if log_ratio else "additive", "edges": edges, "levels": {}}
    for lvl in INTERVAL_LEVELS:
        lo_q, hi_q = (1 - lvl) / 2, 1 - (1 - lvl) / 2
        model["levels"][str(lvl)] = [
            [float(np.quantile(res[bins == b], lo_q)), float(np.quantile(res[bins == b], hi_q))] for b in range(3)]
    return model


def apply_interval(model: dict, pred: np.ndarray, level: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    b = np.digitize(pred, model["edges"])
    q = np.array(model["levels"][str(level)])[b]
    if model["kind"] == "log_ratio":
        return pred * np.exp(q[:, 0]), pred * np.exp(q[:, 1])
    return pred + q[:, 0], pred + q[:, 1]


# --------------------------------------------------------------------------
# per-task training
# --------------------------------------------------------------------------
def _plausibility(task_key: str, pred: np.ndarray, X: pd.DataFrame) -> dict:
    out = {"n": int(len(pred)), "n_nan_or_inf": int((~np.isfinite(pred)).sum())}
    if task_key == "cost":
        out["n_non_positive"] = int((pred <= 0).sum())
    elif task_key == "temperature_drop":
        out["n_negative_drop"] = int((pred < 0).sum())
        out["n_drop_exceeds_start_temp"] = int((pred > X["avg_roof_temperature_before"].fillna(np.inf)).sum())
    elif task_key == "units_saved":
        out["n_saving_exceeds_usage"] = int((pred > X["avg_units_before"].fillna(np.inf)).sum())
        out["n_negative_saving"] = int((pred < 0).sum())
    elif task_key == "rating":
        out["n_outside_1_5"] = int(((pred < 1) | (pred > 5)).sum())
    return out


def train_task(task_key: str, df: pd.DataFrame, tv_idx: np.ndarray, te_idx: np.ndarray, log=print, fast: bool = False) -> dict:
    spec = C.TASKS[task_key]
    elig = df[df[spec.eligible_col]]
    tr = elig.loc[elig.index.intersection(tv_idx)]
    te = elig.loc[elig.index.intersection(te_idx)]
    X_tr, X_te = design_matrix(tr, task_key), design_matrix(te, task_key)
    y_tr, y_te = _target(tr, task_key), _target(te, task_key)
    splits = _splits(task_key, X_tr, y_tr)
    is_reg = spec.kind == "regression"
    log(f"\n=== {spec.label} [{task_key}]  train+val={len(tr)}  test={len(te)} ===")

    cands = get_candidates(task_key, fast=fast)
    if task_key == "referral":
        from sklearn.dummy import DummyClassifier as DC
        cands.insert(1, Candidate("dummy_always_positive", "Baseline: predict everyone refers", "baseline",
                                  lambda: build_pipeline(DC(strategy="constant", constant=1))))

    rows: list[dict] = []
    fitted: dict[str, dict] = {}
    for cand in cands:
        t0 = time.time()
        est = cand.factory()
        params: dict = {}
        if cand.search != "none":
            params, search_score = _search(cand, est, X_tr, y_tr, splits, task_key)
            est.set_params(**params)
        method = "predict" if is_reg else "predict_proba"
        oof = cross_val_predict(est, X_tr, y_tr, cv=splits, n_jobs=N_JOBS, method=method)
        row = {"task": task_key, "model": cand.name, "label": cand.label, "family": cand.family,
               "params": {k.split("model__")[-1]: v for k, v in params.items()}, "fit_seconds": round(time.time() - t0, 1)}
        if is_reg:
            m = regression_metrics(y_tr, oof)
            mae_mean, mae_std = _fold_mae(y_tr, oof, splits)
            row.update({"cv_mae": m["mae"], "cv_rmse": m["rmse"], "cv_r2": m["r2"], "cv_mae_fold_std": mae_std})
            row["select_score"] = m["mae"]
            if task_key == "rating":
                row["cv_mae_rounded"] = float(np.mean(np.abs(np.rint(oof) - y_tr)))
        else:
            p = oof[:, 1] if oof.ndim == 2 and oof.shape[1] == 2 else oof.ravel()
            m05 = classification_metrics(y_tr, (p >= 0.5).astype(int), p)
            thr, f1_best = best_f1_threshold(y_tr, p)
            mt = classification_metrics(y_tr, (p >= thr).astype(int), p)
            row.update({"cv_accuracy": m05["accuracy"], "cv_f1": m05["f1"], "cv_log_loss": m05["log_loss"],
                        "cv_brier": m05["brier"], "cv_roc_auc": m05["roc_auc"],
                        "cv_best_f1": f1_best, "cv_f1_threshold": thr, "cv_accuracy_at_f1_threshold": mt["accuracy"]})
            if cand.family == "baseline":      # trivial classifiers are scored on their own labels, not a tuned threshold
                row["select_score"] = -(m05["accuracy"] if task_key == "high_rating" else m05["f1"])
            else:
                row["select_score"] = -(m05["accuracy"] if task_key == "high_rating" else f1_best)  # lower is better
        fitted[cand.name] = {"est": est, "oof": oof}
        rows.append(row)
        log(f"  {cand.name:24s} " + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k.startswith("cv_") and isinstance(v, float)))

    contenders = [r for r in rows if r["family"] != "baseline"]
    best = min(contenders, key=lambda r: r["select_score"])
    baselines = [r for r in rows if r["family"] == "baseline"]
    base_row = min(baselines, key=lambda r: r["select_score"])
    best_est = clone(fitted[best["model"]]["est"])
    oof_best = fitted[best["model"]]["oof"]
    best_cand = next(c for c in cands if c.name == best["model"])
    log(f"  -> selected: {best['model']}   (baseline: {base_row['model']})")

    calibrated = False
    thr_used = 0.5
    result: dict = {"task": task_key, "label": spec.label, "kind": spec.kind, "unit": spec.unit,
                    "target_definition": spec.description, "features": list(spec.features),
                    "n_train_validation": int(len(tr)), "n_test": int(len(te)), "n_eligible": int(len(elig)),
                    "selected": best["model"], "selected_label": best["label"], "selected_family": best["family"],
                    "selected_params": best["params"], "baseline": base_row["model"], "baseline_label": base_row["label"],
                    "candidates": rows}

    if is_reg:
        best_est.fit(X_tr, y_tr)
        pred_te = best_est.predict(X_te)
        base_est = clone(next(c for c in cands if c.name == base_row["model"]).factory()).fit(X_tr, y_tr)
        base_te = base_est.predict(X_te)
        result["test"] = regression_metrics(y_te, pred_te)
        result["test_baseline"] = regression_metrics(y_te, base_te)
        result["validation"] = {"mae": best["cv_mae"], "rmse": best["cv_rmse"], "r2": best["cv_r2"]}
        result["validation_baseline"] = {"mae": base_row["cv_mae"], "rmse": base_row["cv_rmse"]}
        result["beats_baseline"] = bool(result["test"]["mae"] < result["test_baseline"]["mae"])
        interval = fit_interval_model(y_tr, oof_best, log_ratio=(task_key == "cost"))
        result["interval_model"] = interval
        result["interval_coverage_test"] = {}
        for lvl in INTERVAL_LEVELS:
            lo, hi = apply_interval(interval, pred_te, lvl)
            result["interval_coverage_test"][str(lvl)] = float(np.mean((y_te >= lo) & (y_te <= hi)))
        result["plausibility_test"] = _plausibility(task_key, pred_te, X_te)
        result["plausibility_train_oof"] = _plausibility(task_key, oof_best, X_tr)
        if task_key == "rating":
            result["test"]["mae_rounded"] = float(np.mean(np.abs(np.rint(pred_te) - y_te)))
            result["test"]["exact_match_accuracy_rounded"] = float(np.mean(np.rint(pred_te) == y_te))
            result["test"]["within_one_star"] = float(np.mean(np.abs(np.rint(pred_te) - y_te) <= 1))
            # accuracy of the high/not-high decision *derived* from the numeric prediction
            thr_r = C.HIGH_RATING_THRESHOLD - 0.5
            high_true = (y_te >= C.HIGH_RATING_THRESHOLD).astype(int)
            derived = classification_metrics(high_true, (pred_te >= thr_r).astype(int))
            result["test"]["derived_high_rating_accuracy"] = derived["accuracy"]
            result["test"]["derived_high_rating_f1"] = derived["f1"]
            high_tr = (y_tr >= C.HIGH_RATING_THRESHOLD).astype(int)
            result["validation"]["derived_high_rating_accuracy"] = float(np.mean((oof_best >= thr_r).astype(int) == high_tr))
        test_df = pd.DataFrame({"building_id": te[C.ID_COL].to_numpy(), f"{task_key}_actual": y_te,
                                f"{task_key}_predicted": pred_te, f"{task_key}_baseline": base_te})
        oof_df = pd.DataFrame({"building_id": tr[C.ID_COL].to_numpy(), f"{task_key}_actual": y_tr, f"{task_key}_oof": oof_best})
    else:
        p_oof = oof_best[:, 1]
        # Optional probability calibration, decided on OOF Brier only (never on the test set).
        if best["family"] != "interpretable":
            cal_est = CalibratedClassifierCV(clone(best_est), method="sigmoid", cv=5)
            p_cal = cross_val_predict(cal_est, X_tr, y_tr, cv=splits, n_jobs=N_JOBS, method="predict_proba")[:, 1]
            brier_raw = classification_metrics(y_tr, (p_oof >= 0.5).astype(int), p_oof)["brier"]
            brier_cal = classification_metrics(y_tr, (p_cal >= 0.5).astype(int), p_cal)["brier"]
            result["calibration_decision"] = {"brier_uncalibrated_oof": brier_raw, "brier_sigmoid_calibrated_oof": brier_cal}
            if brier_cal < brier_raw - 1e-4:
                calibrated, p_oof, best_est = True, p_cal, cal_est
        if task_key == "referral":
            thr_used, f1_oof = best_f1_threshold(y_tr, p_oof)
            result["threshold_policy"] = "F1-optimal threshold chosen on out-of-fold probabilities (train+validation only)"
        else:
            thr_used = 0.5
            result["threshold_policy"] = "fixed 0.5 (accuracy is the required metric)"
        best_est.fit(X_tr, y_tr)
        p_te = best_est.predict_proba(X_te)[:, 1]
        base_est = clone(next(c for c in cands if c.name == base_row["model"]).factory()).fit(X_tr, y_tr)
        base_p = base_est.predict_proba(X_te)
        base_p = base_p[:, 1] if base_p.shape[1] == 2 else np.full(len(X_te), float(base_est.classes_[0] == 1))
        base_pred = base_est.predict(X_te)
        result["threshold"] = float(thr_used)
        result["calibrated"] = calibrated
        pred_label = (p_te >= thr_used).astype(int)
        result["test"] = classification_metrics(y_te, pred_label, p_te)
        result["test_at_0_5"] = classification_metrics(y_te, (p_te >= 0.5).astype(int), p_te)
        result["test_baseline"] = classification_metrics(y_te, base_pred, base_p)
        # both trivial baselines are always reported: majority class is the accuracy baseline,
        # "everyone refers / is high" is the F1 baseline.
        result["test_baseline_always_positive"] = classification_metrics(y_te, np.ones_like(y_te))
        result["test_baseline_always_negative"] = classification_metrics(y_te, np.zeros_like(y_te))
        grid = [round(t, 2) for t in np.arange(0.10, 0.81, 0.05)]
        def _op(y, p):
            rows_ = []
            for t in grid:
                m_ = classification_metrics(y, (p >= t).astype(int))
                rows_.append({"threshold": t, **{k: m_[k] for k in ("accuracy", "f1", "precision", "recall")}})
            return rows_
        result["operating_points"] = {"oof": _op(y_tr, p_oof), "test": _op(y_te, p_te)}
        acc_thr = max(grid, key=lambda t: (classification_metrics(y_tr, (p_oof >= t).astype(int))["accuracy"], -abs(t - 0.5)))
        result["named_thresholds"] = {"f1_optimal": float(best_f1_threshold(y_tr, p_oof)[0]), "accuracy_optimal": float(acc_thr), "default": 0.5}
        result["calibration_test"] = calibration_table(y_te, p_te)
        result["calibration_oof"] = calibration_table(y_tr, p_oof)
        result["validation"] = {"accuracy": best["cv_accuracy"], "f1": best["cv_f1"], "best_f1": best["cv_best_f1"],
                                "log_loss": best["cv_log_loss"], "roc_auc": best["cv_roc_auc"]}
        result["validation_baseline"] = {"accuracy": base_row["cv_accuracy"], "f1": base_row["cv_f1"]}
        primary = "accuracy" if task_key == "high_rating" else "f1"
        result["beats_baseline"] = bool(result["test"][primary] > result["test_baseline"][primary])
        best_trivial_acc = max(result["test_baseline_always_positive"]["accuracy"], result["test_baseline_always_negative"]["accuracy"])
        best_trivial_f1 = max(result["test_baseline_always_positive"]["f1"], result["test_baseline_always_negative"]["f1"])
        result["baseline_accuracy"] = float(best_trivial_acc)
        result["baseline_f1"] = float(best_trivial_f1)
        result["beats_baseline_accuracy"] = bool(result["test"]["accuracy"] > best_trivial_acc)
        result["beats_baseline_f1"] = bool(result["test"]["f1"] > best_trivial_f1)
        test_df = pd.DataFrame({"building_id": te[C.ID_COL].to_numpy(), f"{task_key}_actual": y_te,
                                f"{task_key}_probability": p_te, f"{task_key}_predicted": pred_label})
        oof_df = pd.DataFrame({"building_id": tr[C.ID_COL].to_numpy(), f"{task_key}_actual": y_tr, f"{task_key}_oof": p_oof})
    log(f"  test: " + ", ".join(f"{k}={result['test'][k]:.4f}" for k in (spec.metrics if not is_reg else ("mae", "rmse"))
                                  if k in result["test"]) + f"   | baseline: " +
        ", ".join(f"{k}={result['test_baseline'][k]:.4f}" for k in (("mae", "rmse") if is_reg else ("accuracy", "f1"))))
    return {"result": result, "estimator": best_est, "test_df": test_df, "oof_df": oof_df, "best_cand": best_cand}


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(log=print) -> dict:
    C.ensure_dirs()
    validate_all_contracts()
    df = load_cleaned()
    tv_idx, te_idx = _holdout(df)
    outputs, results, tests, oofs = {}, {}, [], []
    for task_key in C.TASK_ORDER:
        out = train_task(task_key, df, tv_idx, te_idx, log=log)
        outputs[task_key] = out
        results[task_key] = out["result"]
        tests.append(out["test_df"].set_index("building_id"))
        oofs.append(out["oof_df"].set_index("building_id"))
        joblib.dump(out["estimator"], C.MODELS / f"{task_key}.joblib", compress=3)

    # reports ---------------------------------------------------------------
    pd.concat(tests, axis=1).reset_index().to_csv(C.REPORTS / "test_predictions.csv", index=False)
    pd.concat(oofs, axis=1).reset_index().to_csv(C.REPORTS / "oof_predictions.csv", index=False)
    _write_comparison_csv(results)

    pairs = (df.groupby(["roofing_type", "cool_roofing_method"]).size().reset_index(name="n"))
    meta = {
        "model_version": C.MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": C.SEED, "test_size": C.TEST_SIZE, "cv_folds": C.CV_FOLDS,
        "high_rating_threshold": C.HIGH_RATING_THRESHOLD,
        "data": {"raw_csv_sha256": _sha256(C.RAW_CSV), "n_rows": int(len(df)),
                 "n_train_validation_buildings": int(len(tv_idx)), "n_test_buildings": int(len(te_idx))},
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "feature_schema": {"features": C.PRE_INSTALLATION_FEATURES, "categorical": C.CATEGORICAL_COLS,
                           "numeric": C.NUMERIC_FEATURE_COLS, "decision_variables": C.DECISION_VARIABLES,
                           "categories": C.EXPECTED_CATEGORIES,
                           "training_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in C.NUMERIC_FEATURE_COLS}},
        "method_roof_support": pairs.to_dict(orient="records"),
        "tasks": {k: {kk: vv for kk, vv in v.items() if kk != "candidates"} for k, v in results.items()},
        "task_order": C.TASK_ORDER,
    }
    from ..utils import write_json
    write_json(C.MODELS / "model_metadata.json", meta, indent=2)
    write_json(C.REPORTS / "model_metadata.json", meta, indent=2)
    write_json(C.REPORTS / "model_candidates.json", {k: v["candidates"] for k, v in results.items()})
    (C.MODELS / "split.json").write_text(json.dumps({
        "seed": C.SEED, "test_size": C.TEST_SIZE,
        "test_building_ids": sorted(int(i) for i in df.loc[te_idx, C.ID_COL])}), encoding="utf-8")
    from .evaluation_report import write_evaluation_report
    from .significance import run as run_significance
    sig = run_significance(log=log)                      # paired bootstrap vs pre-declared trivial baselines
    for k in results:
        results[k]["significance"] = sig[k]
        meta["tasks"][k]["significance"] = sig[k]
    write_evaluation_report(results, meta)
    log("\nsaved models ->", C.MODELS)
    return {"results": results, "meta": meta, "outputs": outputs}


def _holdout(df):
    from .split import holdout_split
    return holdout_split(df)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_comparison_csv(results: dict) -> None:
    rows = []
    for k, r in results.items():
        for c in r["candidates"]:
            role = "selected" if c["model"] == r["selected"] else ("baseline" if c["family"] == "baseline" else "candidate")
            row = {"task": k, "task_label": r["label"], "model": c["model"], "label": c["label"], "family": c["family"],
                   "role": role, "unit": r["unit"], "n_train_validation": r["n_train_validation"]}
            row.update({kk: vv for kk, vv in c.items() if kk.startswith("cv_")})
            row["params"] = json.dumps(c["params"], default=_json_default)
            if role == "selected":
                row.update({f"test_{m}": v for m, v in r["test"].items() if isinstance(v, (int, float))})
            if c["model"] == r["baseline"]:
                row.update({f"test_{m}": v for m, v in r["test_baseline"].items() if isinstance(v, (int, float))})
            rows.append(row)
    pd.DataFrame(rows).to_csv(C.REPORTS / "model_comparison.csv", index=False)


if __name__ == "__main__":
    run()

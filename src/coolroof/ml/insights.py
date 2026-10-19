"""Explainability and business insights, all computed from the trained models and real data.

Global importance      permutation importance on the hold-out set
Method/quality effects counterfactual ("set everyone's method to m") averages with bootstrap CIs
Segment error analysis out-of-fold errors by building/customer segment, with sample sizes
Predictability ceiling diagnostic using realised outcomes (analysis only - never a model input)

Everything describes associations in historical installations, not causal effects.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .. import config as C
from ..data.clean import load_cleaned
from ..features import design_matrix
from ..service import ModelService
from .split import holdout_split

MIN_SEGMENT_N = 30
BOOT = 400


def _pred(model, kind: str, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X) if kind == "regression" else model.predict_proba(X)[:, 1], dtype=float)


# --------------------------------------------------------------------------
def permutation_table(svc: ModelService, df: pd.DataFrame, te_idx) -> pd.DataFrame:
    rows = []
    for task in C.TASK_ORDER:
        spec = C.TASKS[task]
        d = df.loc[df.index.intersection(te_idx)]
        d = d[d[spec.eligible_col]]
        X = design_matrix(d, task)
        y = d[spec.target].astype(float).to_numpy()
        scoring = "neg_mean_absolute_error" if spec.kind == "regression" else "roc_auc"
        res = permutation_importance(svc.models[task], X, y, scoring=scoring, n_repeats=25, random_state=C.SEED, n_jobs=1)
        imp = np.maximum(res.importances_mean, 0)
        total = imp.sum() or 1.0
        for f, m, s, i in zip(spec.features, res.importances_mean, res.importances_std, imp):
            rows.append({"task": task, "task_label": spec.label, "feature": f, "importance": float(m), "std": float(s),
                         "share": float(i / total), "metric": "increase in MAE" if spec.kind == "regression" else "drop in ROC-AUC",
                         "n_holdout": int(len(d))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def _boot_ci(vals: np.ndarray, rng, stat=np.mean) -> list[float]:
    n = len(vals)
    bs = [stat(vals[rng.integers(0, n, n)]) for _ in range(BOOT)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def marginal_effects(svc: ModelService, df: pd.DataFrame) -> dict:
    """Counterfactual averages: force the method/quality for every comparable building and
    average the model's prediction. Uses concrete roofs for the method comparison because only
    concrete roofs accept all five methods (so the building mix is identical for every method)."""
    rng = np.random.default_rng(C.SEED)
    methods = C.EXPECTED_CATEGORIES["cool_roofing_method"]
    qualities = C.EXPECTED_CATEGORIES["cool_roofing_quality"]
    pool = df[df["roofing_type"].isin(["concrete_flat", "concrete_slanted"])].copy()
    X0 = pool[C.PRE_INSTALLATION_FEATURES].reset_index(drop=True)
    outputs = ["cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral"]
    out: dict = {"population": {"method_comparison_rows": int(len(X0)), "roof_types": ["concrete_flat", "concrete_slanted"],
                                "note": "Only concrete roofs accept all five methods, so the building mix is identical for each method."},
                 "method": {}, "quality": {}, "grid": []}

    def block(key_vals, col, Xbase, store):
        preds = {}
        for v in key_vals:
            Xv = Xbase.copy()
            Xv[col] = v
            preds[v] = {t: _pred(svc.models[t], C.TASKS[t].kind, Xv) for t in outputs}
        ref = key_vals[0]
        for t in outputs:
            store[t] = []
            for v in key_vals:
                arr = preds[v][t]
                diff = arr - preds[ref][t]
                store[t].append({"level": v, "mean": float(arr.mean()), "ci95": _boot_ci(arr, rng),
                                 "diff_vs_reference": float(diff.mean()), "diff_ci95": _boot_ci(diff, rng), "reference": ref})
        return preds

    block(methods, "cool_roofing_method", X0, out["method"])
    Xall = df[C.PRE_INSTALLATION_FEATURES].reset_index(drop=True)
    block(qualities, "cool_roofing_quality", Xall, out["quality"])
    out["quality_population_rows"] = int(len(Xall))

    # per existing roof type: only the methods the data supports for that roof
    from ..recommend import MIN_PAIR_SUPPORT
    sup = {(r["roofing_type"], r["cool_roofing_method"]): r["n"] for r in svc.pair_support}
    out["by_roof"] = {}
    for rt in C.EXPECTED_CATEGORIES["roofing_type"]:
        Xr = df[df["roofing_type"] == rt][C.PRE_INSTALLATION_FEATURES].reset_index(drop=True)
        rows_ = []
        for m in methods:
            if sup.get((rt, m), 0) < MIN_PAIR_SUPPORT:
                continue
            Xv = Xr.copy()
            Xv["cool_roofing_method"] = m
            rec = {"cool_roofing_method": m, "n_buildings": int(len(Xr))}
            for t in outputs:
                rec[t] = float(_pred(svc.models[t], C.TASKS[t].kind, Xv).mean())
            rows_.append(rec)
        out["by_roof"][rt] = rows_

    # method x quality grid on concrete roofs
    for m in methods:
        for q in qualities:
            Xv = X0.copy()
            Xv["cool_roofing_method"], Xv["cool_roofing_quality"] = m, q
            rec = {"cool_roofing_method": m, "cool_roofing_quality": q}
            for t in outputs:
                rec[t] = float(_pred(svc.models[t], C.TASKS[t].kind, Xv).mean())
            rec["cost_per_celsius_drop"] = rec["cost"] / rec["temperature_drop"]
            rec["cost_per_kwh_month_saved"] = rec["cost"] / rec["units_saved"] if rec["units_saved"] > 0 else None
            out["grid"].append(rec)
    return out


# --------------------------------------------------------------------------
def segment_errors(df: pd.DataFrame, svc: ModelService) -> pd.DataFrame:
    oof = pd.read_csv(C.REPORTS / "oof_predictions.csv")
    d = df.merge(oof, on="building_id", how="inner")
    d["size_band"] = pd.qcut(d["building_size"], 3, labels=["small", "medium", "large"])
    segs = ["location_type", "roofing_type", "cool_roofing_method", "cool_roofing_quality", "size_band"]
    rows = []
    for task in C.TASK_ORDER:
        spec = C.TASKS[task]
        col_a, col_p = f"{task}_actual", f"{task}_oof"
        if col_a not in d or col_p not in d:
            continue
        for sg in segs:
            for lvl, g in d.dropna(subset=[col_a, col_p]).groupby(sg, observed=True):
                n = len(g)
                if spec.kind == "regression":
                    rows.append({"task": task, "segment": sg, "level": str(lvl), "n": n, "mae": float(np.abs(g[col_a] - g[col_p]).mean()),
                                 "bias": float((g[col_p] - g[col_a]).mean()), "accuracy": None, "small_sample": n < MIN_SEGMENT_N})
                else:
                    thr = svc.tasks[task].get("threshold", 0.5)
                    acc = float(((g[col_p] >= thr).astype(int) == g[col_a].astype(int)).mean())
                    rows.append({"task": task, "segment": sg, "level": str(lvl), "n": n, "mae": None, "bias": float(g[col_p].mean() - g[col_a].mean()),
                                 "accuracy": acc, "small_sample": n < MIN_SEGMENT_N})
    return pd.DataFrame(rows)


def predictability_ceiling(df: pd.DataFrame, tv_idx) -> dict:
    """Diagnostic only. How much *could* referral / high-rating be predicted even if the realised
    post-installation outcomes were known? Uses the train+validation pool and cross-validation."""
    d = df.loc[df.index.intersection(tv_idx)].dropna(subset=["units_saved", "temperature_drop"]).copy()
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)

    def auc(cols, y):
        m = LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=4, min_child_samples=30, verbose=-1,
                           random_state=C.SEED, n_jobs=1)
        p = cross_val_predict(m, d[cols], y, cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, p))

    yr = d["customer_referral"].astype(float).astype(int).to_numpy()
    yh = d["high_rating"].astype(int).to_numpy()
    d["rating_num"] = d["customer_given_rating"].astype(float)
    return {
        "note": "DIAGNOSTIC ONLY - realised post-installation values are unavailable at quote time and are never used by any model.",
        "n": int(len(d)),
        "referral_auc": {
            "from_realised_savings_and_temperature_drop": auc(["units_saved", "temperature_drop"], yr),
            "from_realised_rating": auc(["rating_num"], yr),
            "from_realised_rating_savings_drop": auc(["rating_num", "units_saved", "temperature_drop"], yr),
        },
        "high_rating_auc": {"from_realised_savings_and_temperature_drop": auc(["units_saved", "temperature_drop"], yh)},
    }


# --------------------------------------------------------------------------
def observed_segments(df: pd.DataFrame) -> dict:
    """Plain observed rates (no model) by who the customer is, with Wilson intervals and n."""
    from ..data.report import _wilson

    d = df.copy()
    d["usage_band"] = pd.cut(d["avg_units_before"], [0, 25, 75, 150, 10_000], labels=["≤25 kWh", "26–75 kWh", "76–150 kWh", ">150 kWh"])
    d["ac_group"] = d["num_ac_units"].map(lambda v: "unknown" if pd.isna(v) else f"{int(v)} AC")
    out = {}
    for seg in ("ac_group", "usage_band", "location_type"):
        rows = []
        for lvl, g in d.groupby(seg, observed=True):
            n = len(g)
            hi, rf = g["high_rating"].dropna(), g["customer_referral"].dropna().astype(bool)
            hl, hh = _wilson(int(hi.sum()), len(hi))
            rl, rh = _wilson(int(rf.sum()), len(rf))
            us = g["units_saved"].dropna()
            rows.append({"level": str(lvl), "n": n, "mean_units_saved": float(us.mean()) if len(us) else None,
                         "mean_rating": float(g["customer_given_rating"].mean()),
                         "high_rating_rate": float(hi.mean()), "high_rating_ci95": [hl, hh],
                         "referral_rate": float(rf.mean()), "referral_ci95": [rl, rh], "small_sample": n < MIN_SEGMENT_N})
        out[seg] = rows
    return out


def run(log=print) -> dict:
    C.ensure_dirs()
    df = load_cleaned()
    svc = ModelService()
    tv_idx, te_idx = holdout_split(df)
    imp = permutation_table(svc, df, te_idx)
    imp.to_csv(C.REPORTS / "feature_importance.csv", index=False)
    eff = marginal_effects(svc, df)
    seg = segment_errors(df, svc)
    seg.to_csv(C.REPORTS / "segment_error_analysis.csv", index=False)
    ceil = predictability_ceiling(df, tv_idx)
    insights = {
        "feature_importance": imp.to_dict(orient="records"),
        "marginal_effects": eff,
        "segment_errors": seg.to_dict(orient="records"),
        "predictability_ceiling": ceil,
        "observed_segments": observed_segments(df),
        "min_segment_n": MIN_SEGMENT_N,
    }
    from ..utils import write_json
    write_json(C.REPORTS / "insights.json", insights)
    from .business_report import write_business_insights
    write_business_insights(insights, svc, df)
    log("insights written to", C.REPORTS)
    return insights


if __name__ == "__main__":
    run()

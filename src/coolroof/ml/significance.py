"""Statistical significance of hold-out results (paired bootstrap, 95 %).

"Better than the baseline" is only claimed when the paired bootstrap interval of the improvement excludes 0.
Trivial baselines are declared BEFORE looking at the hold-out:
  * regression         : training mean (training median for the 1-5 rating)
  * accuracy           : majority class of the TRAINING pool
  * F1 / precision etc.: predict every customer positive
  * ROC-AUC            : chance (0.5)
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from .. import config as C
from ..data.clean import load_cleaned
from ..features import design_matrix
from ..utils import write_json

B = 2000


def _verdict(lo: float, hi: float) -> str:
    return "better" if lo > 0 else "worse" if hi < 0 else "not_distinguishable"


def _ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def _counts(y, lab, idx):
    yy, ll = y[idx], lab[idx]
    tp = (yy * ll).sum(1); fp = ((1 - yy) * ll).sum(1); fn = (yy * (1 - ll)).sum(1); tn = ((1 - yy) * (1 - ll)).sum(1)
    return tp, fp, fn, tn


def compute(models_dir=C.MODELS, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    meta = json.loads((models_dir / "model_metadata.json").read_text(encoding="utf-8"))
    df = load_cleaned()
    te_ids = set(json.loads((models_dir / "split.json").read_text())["test_building_ids"])
    is_te = df[C.ID_COL].isin(te_ids)
    out: dict = {}
    for k in C.TASK_ORDER:
        spec = C.TASKS[k]
        e = df[spec.eligible_col]
        tr, te = df[e & ~is_te], df[e & is_te]
        model = joblib.load(models_dir / f"{k}.joblib")
        X = design_matrix(te, k)
        n = len(te)
        idx = rng.integers(0, n, (B, n))
        if spec.kind == "regression":
            y = te[spec.target].astype(float).to_numpy()
            p = np.asarray(model.predict(X), dtype=float)
            ytr = tr[spec.target].astype(float).to_numpy()
            base = float(np.median(ytr)) if k == "rating" else float(ytr.mean())
            em, eb = np.abs(y - p), np.abs(y - base)
            d = eb[idx].mean(1) - em[idx].mean(1)
            lo, hi = _ci(d)
            out[k] = {"kind": "regression", "n": n, "baseline_rule": "training median" if k == "rating" else "training mean",
                      "mae": float(em.mean()), "mae_ci95": _ci(em[idx].mean(1)), "baseline_mae": float(eb.mean()),
                      "mae_improvement": float(d.mean()), "mae_improvement_ci95": [lo, hi], "verdict": _verdict(lo, hi)}
        else:
            y = te[spec.target].astype(float).astype(int).to_numpy()
            ytr = tr[spec.target].astype(float).astype(int).to_numpy()
            pr = model.predict_proba(X)[:, 1]
            thr = meta["tasks"][k]["threshold"]
            lab = (pr >= thr).astype(int)
            majority = int(ytr.mean() >= 0.5)                         # from TRAINING prevalence only
            tp, fp, fn, tn = _counts(y, lab, idx)
            acc_m = (tp + tn) / n
            f1_m = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
            maj = np.full(n, majority); pos = np.ones(n, int)
            tpb, fpb, fnb, tnb = _counts(y, maj, idx)
            acc_b = (tpb + tnb) / n
            tpp, fpp, fnp, _ = _counts(y, pos, idx)
            f1_p = 2 * tpp / np.maximum(2 * tpp + fpp + fnp, 1)
            da, df1 = acc_m - acc_b, f1_m - f1_p
            auc_b = np.array([roc_auc_score(y[i], pr[i]) for i in idx[:1000] if 0 < y[i].sum() < n])
            ca, cf = _ci(da), _ci(df1)
            out[k] = {"kind": "classification", "n": n, "threshold": thr, "accuracy_baseline_rule": f"majority class of the training pool (= {majority})",
                      "accuracy": float((lab == y).mean()), "accuracy_ci95": _ci(acc_m),
                      "baseline_accuracy": float((maj == y).mean()), "accuracy_improvement": float((lab == y).mean() - (maj == y).mean()),
                      "accuracy_improvement_ci95": ca, "accuracy_verdict": _verdict(*ca),
                      "f1_baseline_rule": "predict every customer positive", "f1_ci95": _ci(f1_m),
                      "baseline_f1": float(2 * y.sum() / (y.sum() + n)),
                      "f1_improvement_ci95": cf, "f1_verdict": _verdict(*cf),
                      "roc_auc": float(roc_auc_score(y, pr)), "roc_auc_ci95": _ci(auc_b),
                      "auc_above_chance": bool(np.percentile(auc_b, 2.5) > 0.5)}
    return out


def run(log=print) -> dict:
    res = compute()
    for path in (C.MODELS / "model_metadata.json", C.REPORTS / "model_metadata.json"):
        m = json.loads(path.read_text(encoding="utf-8"))
        for k, v in res.items():
            m["tasks"][k]["significance"] = v
        write_json(path, m, indent=2)
    write_json(C.REPORTS / "significance.json", res)
    for k, v in res.items():
        if v["kind"] == "regression":
            log(f"{k:17s} MAE improvement vs baseline {v['mae_improvement']:.4g} CI {v['mae_improvement_ci95']} -> {v['verdict']}")
        else:
            log(f"{k:17s} accuracy vs majority: {v['accuracy_improvement']:+.3f} CI {np.round(v['accuracy_improvement_ci95'], 3)} -> {v['accuracy_verdict']} | "
                f"F1 vs all-positive: CI {np.round(v['f1_improvement_ci95'], 3)} -> {v['f1_verdict']} | AUC {v['roc_auc']:.3f} CI {np.round(v['roc_auc_ci95'], 3)} above chance: {v['auc_above_chance']}")
    return res


if __name__ == "__main__":
    run()

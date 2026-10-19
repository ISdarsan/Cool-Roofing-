"""Metric helpers (thin, well-tested wrappers over scikit-learn / NumPy)."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_ae": float(np.median(np.abs(err))),
        "r2": float(skm.r2_score(y_true, y_pred)),
        "bias": float(np.mean(err)),
        "n": int(len(y_true)),
    }


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = skm.confusion_matrix(y_true, y_pred, labels=[0, 1])
    out = {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "f1": float(skm.f1_score(y_true, y_pred, pos_label=1, zero_division=0)),   # binary, positive class = 1
        "precision": float(skm.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(skm.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "support": {"negative": int((y_true == 0).sum()), "positive": int((y_true == 1).sum())},
        "positive_rate_true": float(y_true.mean()),
        "positive_rate_pred": float(y_pred.mean()),
        "n": int(len(y_true)),
    }
    if y_proba is not None:
        p = np.clip(np.asarray(y_proba, dtype=float), 1e-6, 1 - 1e-6)
        out["brier"] = float(skm.brier_score_loss(y_true, p))
        out["log_loss"] = float(skm.log_loss(y_true, p, labels=[0, 1]))
        out["roc_auc"] = float(skm.roc_auc_score(y_true, p)) if len(np.unique(y_true)) == 2 else None
    return out


def best_f1_threshold(y_true, proba, grid=None) -> tuple[float, float]:
    """Threshold maximising binary F1 on the supplied (out-of-fold) probabilities."""
    y_true = np.asarray(y_true).astype(int)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = skm.f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f > best_f + 1e-12:
            best_t, best_f = float(t), float(f)
    return best_t, best_f


def calibration_table(y_true, proba, n_bins: int = 5) -> dict:
    """Quantile-binned reliability table and expected calibration error."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    edges = np.unique(np.quantile(proba, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return {"bins": [], "ece": None}
    idx = np.clip(np.digitize(proba, edges[1:-1], right=True), 0, len(edges) - 2)
    bins, ece = [], 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        mp, ob = float(proba[m].mean()), float(y_true[m].mean())
        bins.append({"mean_predicted": mp, "observed_rate": ob, "n": int(m.sum())})
        ece += m.mean() * abs(mp - ob)
    return {"bins": bins, "ece": float(ece)}

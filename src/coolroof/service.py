"""Inference service: loads the trained artifacts once and serves predictions,
option comparisons and explanations. No training ever happens here."""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import recommend as R
from .features import design_matrix
from .ml.train import apply_interval

DERIVED_NOTE = "derived: measured/entered 'before' value minus the predicted change"


class ModelNotReady(RuntimeError):
    """Models are missing or unreadable (the message is safe to show to clients)."""


class InvalidCombination(ValueError):
    """Requested roof/method combination has no support in the historical data."""


def _reliability(task: dict) -> dict:
    t = task["test"]
    if task["kind"] == "regression":
        r2 = t["r2"]
        lvl = "high" if r2 >= 0.9 else "moderate" if r2 >= 0.5 else "low"
        why = f"hold-out R² = {r2:.2f}"
    else:
        auc = t.get("roc_auc") or 0.5
        lvl = "high" if auc >= 0.8 else "moderate" if auc >= 0.7 else "low"
        why = f"hold-out ROC-AUC = {auc:.2f}"
    return {"level": lvl, "basis": why}


class ModelService:
    def __init__(self, models_dir: Path = C.MODELS) -> None:
        self.dir = Path(models_dir)
        meta_path = self.dir / "model_metadata.json"
        if not meta_path.exists():
            raise ModelNotReady("Models have not been trained yet. Run `python -m coolroof all`.")
        try:
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.models = {k: joblib.load(self.dir / f"{k}.joblib") for k in C.TASK_ORDER}
        except Exception as exc:  # corrupted / incompatible artifact
            raise ModelNotReady("Model files could not be loaded; retrain with `python -m coolroof train`.") from exc
        self.tasks = self.meta["tasks"]
        self.features = self.meta["feature_schema"]["features"]
        self.ranges = self.meta["feature_schema"]["training_ranges"]
        self.categories = self.meta["feature_schema"]["categories"]
        self.pair_support = self.meta["method_roof_support"]
        self._combo_support = self._load_combo_support()
        self._bg_cache: dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ data helpers
    def _load_combo_support(self) -> dict:
        try:
            from .data.clean import load_cleaned
            df = load_cleaned()
            g = df.groupby(["roofing_type", "cool_roofing_method", "cool_roofing_quality"]).size()
            return {k: int(v) for k, v in g.items()}
        except Exception:
            return {}

    def _background(self, task: str, k: int) -> pd.DataFrame:
        key = f"{task}:{k}"
        if key not in self._bg_cache:
            from .data.clean import load_cleaned
            df = load_cleaned()
            df = df[df[C.TASKS[task].eligible_col]]
            self._bg_cache[key] = design_matrix(df.sample(n=min(k, len(df)), random_state=C.SEED), task).reset_index(drop=True)
        return self._bg_cache[key]

    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        recs = []
        for r in rows:
            recs.append({f: (np.nan if r.get(f) is None else r.get(f)) for f in self.features})
        return pd.DataFrame(recs, columns=self.features)

    # ------------------------------------------------------------------ validation / warnings
    def check_combination(self, roofing_type: str, method: str) -> None:
        n = {(r["roofing_type"], r["cool_roofing_method"]): r["n"] for r in self.pair_support}.get((roofing_type, method), 0)
        if n < R.MIN_PAIR_SUPPORT:
            raise InvalidCombination(
                f"'{method}' is not offered on a '{roofing_type}' roof in the historical data "
                f"({n} records); choose a supported method for this roof type.")

    def warnings_for(self, inp: dict) -> list[str]:
        w = []
        imputed = [f for f in ("building_age", "avg_roof_temperature_before", "num_ac_units", "avg_units_before", "location_type", "building_size")
                   if inp.get(f) is None]
        if imputed:
            w.append("Not provided, so filled with typical training values (reliability is reduced): " + ", ".join(imputed) + ".")
        for f, (lo, hi) in self.ranges.items():
            v = inp.get(f)
            if v is not None and (v < lo or v > hi):
                w.append(f"{f} = {v:g} is outside the range seen in training ({lo:g}–{hi:g}); this is an extrapolation.")
        key = (inp["roofing_type"], inp["cool_roofing_method"], inp["cool_roofing_quality"])
        n = self._combo_support.get(key)
        if n is not None and n < R.LOW_COMBO_SUPPORT:
            w.append(f"Only {n} historical installations match this roof/method/quality combination; its predictions are less reliable.")
        return w

    # ------------------------------------------------------------------ prediction
    def _predict_frame(self, X: pd.DataFrame, level: float = 0.8, referral_cutoff: float | None = None,
                       high_cutoff: float = 0.5) -> list[dict]:
        n = len(X)
        raw: dict[str, np.ndarray] = {}
        for k in C.TASK_ORDER:
            m = self.models[k]
            if C.TASKS[k].kind == "regression":
                raw[k] = np.asarray(m.predict(X), dtype=float)
            else:
                raw[k] = np.asarray(m.predict_proba(X)[:, 1], dtype=float)
        out = []
        ivals = {k: apply_interval(self.tasks[k]["interval_model"], raw[k], level) for k in C.REGRESSION_TASKS}
        thr_ref = self.tasks["referral"]["threshold"] if referral_cutoff is None else referral_cutoff
        for i in range(n):
            preds = {}
            for k, unit in (("cost", "INR"), ("temperature_drop", "°C"), ("units_saved", "kWh/month")):
                preds[k] = {"value": float(raw[k][i]), "unit": unit,
                            "interval": {"level": level, "low": float(ivals[k][0][i]), "high": float(ivals[k][1][i])},
                            "reliability": _reliability(self.tasks[k])}
            preds["rating"] = {"value": float(raw["rating"][i]), "unit": "stars (1–5 scale)", "scale": [1, 5],
                               "reliability": _reliability(self.tasks["rating"]),
                               "note": "Expected star rating (a numeric estimate, not a probability)."}
            preds["high_rating"] = {"probability": float(raw["high_rating"][i]), "cutoff": high_cutoff,
                                    "predicted_high": bool(raw["high_rating"][i] >= high_cutoff),
                                    "definition": f"customer rating ≥ {self.meta['high_rating_threshold']} (assumption: organisers did not define 'high')",
                                    "reliability": _reliability(self.tasks["high_rating"])}
            preds["referral"] = {"probability": float(raw["referral"][i]), "cutoff": float(thr_ref),
                                 "predicted_referral": bool(raw["referral"][i] >= thr_ref),
                                 "reliability": _reliability(self.tasks["referral"])}
            out.append(preds)
        return out

    def predict(self, inp: dict, level: float = 0.8, referral_cutoff: float | None = None, high_cutoff: float = 0.5) -> dict:
        self.check_combination(inp["roofing_type"], inp["cool_roofing_method"])
        X = self._frame([inp])
        preds = self._predict_frame(X, level, referral_cutoff, high_cutoff)[0]
        derived = {}
        if inp.get("avg_roof_temperature_before") is not None:
            derived["avg_roof_temperature_after"] = {"value": inp["avg_roof_temperature_before"] - preds["temperature_drop"]["value"],
                                                     "unit": "°C", "note": DERIVED_NOTE}
        if inp.get("avg_units_before") is not None:
            derived["avg_units_after"] = {"value": inp["avg_units_before"] - preds["units_saved"]["value"],
                                          "unit": "kWh/month", "note": DERIVED_NOTE}
        return {"model_version": self.meta["model_version"], "predictions": preds, "derived": derived,
                "warnings": self.warnings_for(inp),
                "assumptions": [
                    f"'High rating' means a customer rating of {self.meta['high_rating_threshold']} or more (organisers gave no definition).",
                    "Electricity figures are monthly kWh; building size is in the dataset's (unstated) area unit.",
                    "Predictions describe historical associations, not guaranteed outcomes.",
                    f"Intervals are empirical {int(level*100)}% ranges from out-of-fold errors, not guarantees."]}

    def predict_batch(self, rows: list[dict], level: float = 0.8) -> list[dict]:
        for r in rows:
            self.check_combination(r["roofing_type"], r["cool_roofing_method"])
        X = self._frame(rows)
        return self._predict_frame(X, level)

    # ------------------------------------------------------------------ comparison
    def options_for(self, inp: dict) -> list[dict]:
        return R.valid_options(inp["roofing_type"], self.pair_support, self.categories["cool_roofing_method"],
                               self.categories["cool_roofing_quality"],
                               {k: v for k, v in self._combo_support.items()})

    def compare(self, inp: dict, objective: str = "balanced", weights: dict | None = None, level: float = 0.8) -> dict:
        opts = self.options_for(inp)
        if not opts:
            raise InvalidCombination("No supported roofing methods exist for this roof type.")
        rows = [{**inp, "cool_roofing_method": o["cool_roofing_method"], "cool_roofing_quality": o["cool_roofing_quality"]} for o in opts]
        X = self._frame(rows)
        preds = self._predict_frame(X, level)
        options = []
        for o, p in zip(opts, preds):
            options.append({
                "cool_roofing_method": o["cool_roofing_method"], "cool_roofing_quality": o["cool_roofing_quality"],
                "support": o["support"], "low_support": o["low_support"],
                "predictions": {"cost": p["cost"]["value"], "temperature_drop": p["temperature_drop"]["value"],
                                "units_saved": p["units_saved"]["value"], "rating": p["rating"]["value"],
                                "high_rating": p["high_rating"]["probability"], "referral": p["referral"]["probability"]},
                "intervals": {k: [p[k]["interval"]["low"], p[k]["interval"]["high"]] for k in ("cost", "temperature_drop", "units_saved")},
                "is_current_selection": (o["cool_roofing_method"] == inp.get("cool_roofing_method")
                                         and o["cool_roofing_quality"] == inp.get("cool_roofing_quality")),
            })
        model_mae = {k: self.tasks[k]["test"]["mae"] for k in ("cost", "temperature_drop", "units_saved")}
        rec = R.recommend(options, objective, weights, model_mae)
        # keep a stable, id-addressable list as well as the ranking
        for o in options:
            o["id"] = f"{o['cool_roofing_method']}|{o['cool_roofing_quality']}"
        rec["options"] = sorted(options, key=lambda o: o["rank"])
        rec["warnings"] = self.warnings_for({**inp, "cool_roofing_method": rec["recommended"]["cool_roofing_method"],
                                             "cool_roofing_quality": rec["recommended"]["cool_roofing_quality"]})
        rec["metric_info"] = {k: {"label": v[0], "direction": v[1], "unit": v[2]} for k, v in R.METRICS.items()}
        rec["normalisation_note"] = ("Scores are relative: each metric is min–max scaled across the options shown, so 100 means "
                                     "'best of these options on every weighted metric', not an absolute grade.")
        return rec

    # ------------------------------------------------------------------ explanations
    def explain(self, inp: dict, task: str, background_size: int = 48) -> dict:
        """Exact interventional Shapley values over the raw input features.

        phi_j is the average marginal effect of feature j on the prediction, relative to a
        background of typical historical buildings: base_value + sum(phi) = prediction.
        These describe the *model's* reasoning, not causal effects.
        """
        if task not in self.models:
            raise KeyError(task)
        self.check_combination(inp["roofing_type"], inp["cool_roofing_method"])
        feats = self.features
        n = len(feats)
        bg = self._background(task, background_size)
        K = len(bg)
        x = {f: (np.nan if inp.get(f) is None else inp.get(f)) for f in feats}
        masks = np.array(list(product([0, 1], repeat=n)), dtype=bool)          # (2^n, n) True = take from x
        cols = {}
        for j, f in enumerate(feats):
            xv = np.full((len(masks), K), x[f], dtype=object if isinstance(x[f], str) else float)
            bv = np.tile(bg[f].to_numpy(dtype=object if bg[f].dtype == object else float), (len(masks), 1))
            cols[f] = np.where(masks[:, j][:, None], xv, bv).ravel()
        big = pd.DataFrame(cols)
        for f in feats:
            if f in C.NUMERIC_FEATURE_COLS:
                big[f] = pd.to_numeric(big[f], errors="coerce")
        m = self.models[task]
        pred = (m.predict(big) if C.TASKS[task].kind == "regression" else m.predict_proba(big)[:, 1]).reshape(len(masks), K)
        v = pred.mean(axis=1)                                                    # v(S) for every coalition
        from math import factorial
        # masks are enumerated in lexicographic order, so a coalition's row index equals its bit pattern
        # (feature 0 = most significant bit); adding feature j therefore means `idx | bit_j`.
        sizes = masks.sum(axis=1)
        w_size = np.array([factorial(int(k)) * factorial(n - int(k) - 1) / factorial(n) if k < n else 0.0 for k in range(n + 1)])
        idx = np.arange(len(masks))
        phi = np.zeros(n)
        for j in range(n):
            bit = 1 << (n - 1 - j)
            off = idx[(idx & bit) == 0]
            phi[j] = float(np.sum(w_size[sizes[off]] * (v[off | bit] - v[off])))
        base, full = float(v[0]), float(v[-1])
        total = float(pred[-1].mean())
        contribs = sorted(({"feature": f, "value": (None if (isinstance(x[f], float) and np.isnan(x[f])) else x[f]),
                            "contribution": float(p)} for f, p in zip(feats, phi)), key=lambda d: -abs(d["contribution"]))
        return {"task": task, "unit": C.TASKS[task].unit, "base_value": base, "prediction": total,
                "contributions": contribs, "background_size": K,
                "method": "exact interventional Shapley values over the 9 input features (SHAP definition), background = typical historical buildings",
                "caveat": "Explains the model, not the world: correlation in historical data, not causation."}

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        t = self.tasks
        headline = {}
        for k in C.TASK_ORDER:
            r = t[k]
            headline[k] = {"label": r["label"], "kind": r["kind"], "unit": r["unit"], "selected": r["selected_label"],
                           "n_train_validation": r["n_train_validation"], "n_test": r["n_test"],
                           "test": {m: r["test"][m] for m in ("mae", "rmse", "r2", "accuracy", "f1", "roc_auc", "precision", "recall") if m in r["test"]},
                           "baseline": {m: r["test_baseline"][m] for m in ("mae", "rmse", "accuracy", "f1") if m in r["test_baseline"]},
                           "beats_baseline": r["beats_baseline"], "reliability": _reliability(r)}
            sg = r.get("significance")
            if sg:
                headline[k]["significance"] = ({"verdict": sg["verdict"]} if sg["kind"] == "regression" else
                                               {"accuracy": sg["accuracy_verdict"], "f1": sg["f1_verdict"], "auc_above_chance": sg["auc_above_chance"],
                                                "auc_ci95": sg["roc_auc_ci95"]})
            if r["kind"] == "classification":
                headline[k]["best_trivial"] = {"accuracy": r["baseline_accuracy"], "f1": r["baseline_f1"]}
        return {"model_version": self.meta["model_version"], "trained_at_utc": self.meta["trained_at_utc"],
                "data": self.meta["data"], "headline": headline, "high_rating_threshold": self.meta["high_rating_threshold"]}


# singleton ---------------------------------------------------------------------
_service: ModelService | None = None
_svc_lock = threading.Lock()


def get_service(reload: bool = False) -> ModelService:
    global _service
    with _svc_lock:
        if _service is None or reload:
            _service = ModelService()
        return _service

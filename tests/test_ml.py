import json

import joblib
import numpy as np
import pandas as pd
import pytest

from coolroof import config as C
from coolroof.data import clean
from coolroof.ml import metrics as M
from coolroof.ml import train
from coolroof.ml.split import holdout_split


# ---------------------------------------------------------------- metrics are correct
def test_regression_metrics_known_values():
    m = M.regression_metrics([0, 0, 10], [1, -1, 7])
    assert m["mae"] == pytest.approx((1 + 1 + 3) / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 1 + 9) / 3))
    assert m["bias"] == pytest.approx((1 - 1 - 3) / 3)


def test_classification_metrics_known_values():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [1, 1, 0, 1, 0, 0, 0, 0]
    m = M.classification_metrics(y, p)
    assert m["confusion_matrix"] == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)           # binary F1, positive class = 1
    assert m["support"] == {"negative": 5, "positive": 3}


def test_best_f1_threshold_picks_separating_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    t, f = M.best_f1_threshold(y, p)
    assert 0.3 < t <= 0.6 and f == pytest.approx(1.0)


def test_calibration_table_perfect_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert M.calibration_table(y, p, 5)["ece"] < 0.05


# ---------------------------------------------------------------- split
def test_holdout_split_is_reproducible_and_disjoint(clean_res):
    d = clean_res.cleaned
    a_tv, a_te = holdout_split(d)
    b_tv, b_te = holdout_split(d)
    assert (a_tv == b_tv).all() and (a_te == b_te).all()
    assert not set(a_tv) & set(a_te)
    assert len(a_tv) + len(a_te) == len(d) and len(a_te) == int(round(C.TEST_SIZE * len(d)))


# ---------------------------------------------------------------- real training (fast mode, temp dir)
@pytest.fixture(scope="module")
def fast_runs(clean_res):
    d = clean_res.cleaned
    tv, te = holdout_split(d)
    return {k: train.train_task(k, d, tv, te, log=lambda *a, **k: None, fast=True) for k in ("units_saved", "referral")}


def test_training_completes_and_baseline_is_evaluated(fast_runs):
    reg = fast_runs["units_saved"]["result"]
    assert reg["test_baseline"]["mae"] > reg["test"]["mae"]            # boosted/linear model beats the mean
    assert reg["baseline"].startswith("dummy") and reg["beats_baseline"]
    clf = fast_runs["referral"]["result"]
    assert "dummy_always_positive" in {c["model"] for c in clf["candidates"]}
    assert 0 < clf["test"]["f1"] <= 1 and set(clf["test"]["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}


def test_saved_model_reloads_and_predictions_are_finite_and_shaped(fast_runs, tmp_path, clean_res):
    d = clean_res.cleaned
    for k, out in fast_runs.items():
        joblib.dump(out["estimator"], tmp_path / f"{k}.joblib")
        m = joblib.load(tmp_path / f"{k}.joblib")
        X = d[list(C.TASKS[k].features)].head(25)
        p = m.predict(X) if C.TASKS[k].kind == "regression" else m.predict_proba(X)
        assert np.isfinite(p).all() and len(p) == 25
        if C.TASKS[k].kind == "classification":
            assert p.shape == (25, 2) and np.allclose(p.sum(axis=1), 1)


def test_test_rows_never_used_for_training(fast_runs, clean_res):
    d = clean_res.cleaned
    _, te = holdout_split(d)
    te_ids = set(d.loc[te, C.ID_COL])
    for out in fast_runs.values():
        assert not te_ids & set(out["oof_df"]["building_id"])
        assert set(out["test_df"]["building_id"]) <= te_ids


def test_interval_model_coverage_close_to_nominal(fast_runs):
    r = fast_runs["units_saved"]["result"]
    assert 0.65 < r["interval_coverage_test"]["0.8"] < 0.95


# ---------------------------------------------------------------- production artifacts
def test_production_metadata_complete(service):
    meta = service.meta
    assert set(meta["tasks"]) == set(C.TASK_ORDER)
    assert meta["feature_schema"]["features"] == C.PRE_INSTALLATION_FEATURES
    for k, t in meta["tasks"].items():
        assert t["n_test"] > 300 and t["features"] == list(C.TASKS[k].features)
        assert "baseline" in t and "selected" in t and "validation" in t and "test" in t
    assert meta["data"]["raw_csv_sha256"]


def test_training_and_inference_schemas_agree(service):
    from coolroof.api.schemas import BuildingInput
    api_fields = set(BuildingInput.model_fields)
    assert api_fields == set(C.PRE_INSTALLATION_FEATURES)
    assert service.features == C.PRE_INSTALLATION_FEATURES


def test_production_models_beat_baselines_on_holdout(service):
    for k in C.REGRESSION_TASKS:
        t = service.tasks[k]
        assert t["test"]["mae"] < t["test_baseline"]["mae"], k
    assert service.tasks["high_rating"]["test"]["accuracy"] > service.tasks["high_rating"]["baseline_accuracy"]


def test_holdout_predictions_file_traceable_and_consistent(service):
    tp = pd.read_csv(C.REPORTS / "test_predictions.csv")
    split = json.loads((C.MODELS / "split.json").read_text())
    assert set(tp.building_id) == set(split["test_building_ids"])
    cost = tp.dropna(subset=["cost_actual", "cost_predicted"])
    mae = float(np.abs(cost.cost_actual - cost.cost_predicted).mean())
    assert mae == pytest.approx(service.tasks["cost"]["test"]["mae"], rel=1e-6)   # reported metric == recomputed from file


def test_physical_plausibility_of_production_predictions(service, clean_res):
    d = clean_res.cleaned.sample(300, random_state=1)
    X = d[C.PRE_INSTALLATION_FEATURES]
    assert (service.models["cost"].predict(X) > 0).all()
    assert (service.models["temperature_drop"].predict(X) > 0).all()
    r = service.models["rating"].predict(X)
    assert ((r >= 1) & (r <= 5)).all()
    for k in ("high_rating", "referral"):
        p = service.models[k].predict_proba(X)[:, 1]
        assert ((p >= 0) & (p <= 1)).all()


def test_missing_optional_inputs_are_imputed_not_crashing(service, building):
    b = {**building, "building_age": None, "num_ac_units": None, "avg_units_before": None, "avg_roof_temperature_before": None}
    r = service.predict(b)
    assert np.isfinite(r["predictions"]["cost"]["value"]) and r["warnings"]

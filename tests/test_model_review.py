"""Independent-review guard rails: reported hold-out metrics must be reproducible with plain sklearn from the
saved models, the models must be trained on the training pool only, leakage controls must sit at chance, and
'better than baseline' claims must be statistically supported."""
import json

import joblib
import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from coolroof import config as C
from coolroof.features import design_matrix
from coolroof.ml.candidates import get_candidates

META = None


@pytest.fixture(scope="module")
def ctx(clean_res, service):
    ids = set(json.loads((C.MODELS / "split.json").read_text())["test_building_ids"])
    df = clean_res.cleaned
    return df, df[C.ID_COL].isin(ids), service.tasks


def task_split(ctx, k):
    df, is_te, _ = ctx
    e = df[C.TASKS[k].eligible_col]
    return df[e & ~is_te], df[e & is_te]


def target(sub, k):
    y = sub[C.TASKS[k].target]
    return y.astype(float).astype(int).to_numpy() if C.TASKS[k].kind == "classification" else y.astype(float).to_numpy()


def rebuild(meta, k):
    m = meta[k]
    cand = next(c for c in get_candidates(k) if c.name == m["selected"])
    est = cand.factory()
    est.set_params(**{f"{cand.prefix}model__{p}": v for p, v in m["selected_params"].items()})
    return CalibratedClassifierCV(est, method="sigmoid", cv=5) if m.get("calibrated") else est


@pytest.mark.parametrize("k", C.REGRESSION_TASKS)
def test_regression_metrics_recomputed_with_plain_sklearn(ctx, k):
    _, _, meta = ctx
    _, te = task_split(ctx, k)
    p = joblib.load(C.MODELS / f"{k}.joblib").predict(design_matrix(te, k))
    y = target(te, k)
    assert skm.mean_absolute_error(y, p) == pytest.approx(meta[k]["test"]["mae"], rel=1e-9)
    assert np.sqrt(skm.mean_squared_error(y, p)) == pytest.approx(meta[k]["test"]["rmse"], rel=1e-9)
    assert len(y) == meta[k]["n_test"]


@pytest.mark.parametrize("k", C.CLASSIFICATION_TASKS)
def test_classification_metrics_recomputed_with_plain_sklearn(ctx, k):
    _, _, meta = ctx
    _, te = task_split(ctx, k)
    pr = joblib.load(C.MODELS / f"{k}.joblib").predict_proba(design_matrix(te, k))[:, 1]
    y = target(te, k)
    lab = (pr >= meta[k]["threshold"]).astype(int)
    assert skm.accuracy_score(y, lab) == pytest.approx(meta[k]["test"]["accuracy"], rel=1e-9)
    assert skm.f1_score(y, lab, pos_label=1) == pytest.approx(meta[k]["test"]["f1"], rel=1e-9)
    assert skm.roc_auc_score(y, pr) == pytest.approx(meta[k]["test"]["roc_auc"], rel=1e-9)


@pytest.mark.parametrize("k", C.TASK_ORDER)
def test_saved_models_equal_a_refit_on_training_rows_only(ctx, k):
    """Provenance: refitting the selected configuration on TRAIN reproduces the saved predictions; adding the
    hold-out rows to the fit would change them."""
    _, _, meta = ctx
    tr, te = task_split(ctx, k)
    Xte = design_matrix(te, k)
    f = (lambda e: e.predict(Xte)) if C.TASKS[k].kind == "regression" else (lambda e: e.predict_proba(Xte)[:, 1])
    saved = f(joblib.load(C.MODELS / f"{k}.joblib"))
    scale = max(1.0, float(np.abs(saved).max()))
    train_only = f(clone(rebuild(meta, k)).fit(design_matrix(tr, k), target(tr, k)))
    assert np.abs(saved - train_only).max() < 1e-6 * scale
    import pandas as pd
    both = pd.concat([tr, te])
    with_test = f(clone(rebuild(meta, k)).fit(design_matrix(both, k), target(both, k)))
    assert np.abs(saved - with_test).max() > 1e-4 * scale


def test_saved_preprocessing_statistics_come_from_training_rows(ctx):
    from coolroof.features import NUMERIC_MODEL_COLS, FeatureEngineer
    from sklearn.pipeline import Pipeline
    tr, te = task_split(ctx, "cost")
    est = joblib.load(C.MODELS / "cost.joblib")
    pipe = est.regressor_ if hasattr(est, "regressor_") else est
    assert isinstance(pipe, Pipeline)
    stats = pipe.named_steps["prep"].named_transformers_["num"].named_steps["impute"].statistics_
    fe = FeatureEngineer()
    assert np.allclose(stats, fe.fit_transform(design_matrix(tr, "cost"))[NUMERIC_MODEL_COLS].median().to_numpy(), equal_nan=True)


@pytest.mark.parametrize("k", ["cost", "units_saved", "referral"])
def test_shuffled_labels_give_no_skill(ctx, k):
    """If a feature leaked the target, a model fitted on permuted labels would still score above chance."""
    _, _, meta = ctx
    tr, te = task_split(ctx, k)
    y_tr, y_te = target(tr, k), target(te, k)
    rng = np.random.default_rng(1)
    e = rebuild(meta, k).fit(design_matrix(tr, k), rng.permutation(y_tr))
    Xte = design_matrix(te, k)
    if C.TASKS[k].kind == "regression":
        assert skm.r2_score(y_te, e.predict(Xte)) < 0.05
    else:
        assert skm.roc_auc_score(y_te, e.predict_proba(Xte)[:, 1]) < 0.58


@pytest.mark.parametrize("k", ["cost", "temperature_drop", "high_rating"])
def test_building_id_alone_has_no_predictive_power(ctx, k):
    from lightgbm import LGBMClassifier, LGBMRegressor
    tr, te = task_split(ctx, k)
    y_tr, y_te = target(tr, k), target(te, k)
    if C.TASKS[k].kind == "regression":
        e = LGBMRegressor(n_estimators=100, num_leaves=8, verbose=-1, random_state=0).fit(tr[[C.ID_COL]], y_tr)
        assert skm.r2_score(y_te, e.predict(te[[C.ID_COL]])) < 0.05
    else:
        e = LGBMClassifier(n_estimators=100, num_leaves=8, verbose=-1, random_state=0).fit(tr[[C.ID_COL]], y_tr)
        assert skm.roc_auc_score(y_te, e.predict_proba(te[[C.ID_COL]])[:, 1]) < 0.58


def test_accuracy_depends_on_legitimate_inputs_not_on_everything(ctx):
    """Cost must collapse without building size and degrade without method/quality (ablation)."""
    _, _, meta = ctx
    tr, te = task_split(ctx, "cost")
    y_te = target(te, "cost")

    def r2_without(cols):
        Xtr, Xte = design_matrix(tr, "cost").copy(), design_matrix(te, "cost").copy()
        for c in cols:
            const = Xtr[c].mode().iloc[0] if Xtr[c].dtype == object else Xtr[c].median()
            Xtr[c], Xte[c] = const, const
        return skm.r2_score(y_te, rebuild(meta, "cost").fit(Xtr, target(tr, "cost")).predict(Xte))

    assert r2_without([]) > 0.99
    assert r2_without(["building_size"]) < 0.5
    assert r2_without(["cool_roofing_method", "cool_roofing_quality"]) < 0.9


def test_significance_verdicts_are_consistent_with_their_intervals(ctx):
    _, _, meta = ctx
    for k in C.TASK_ORDER:
        sg = meta[k]["significance"]
        if sg["kind"] == "regression":
            lo, hi = sg["mae_improvement_ci95"]
            assert (sg["verdict"] == "better") == (lo > 0)
            assert sg["mae_ci95"][0] <= sg["mae"] <= sg["mae_ci95"][1]
        else:
            for name in ("accuracy", "f1"):
                lo, hi = sg[f"{name}_improvement_ci95"]
                v = sg[f"{name}_verdict"]
                assert v == ("better" if lo > 0 else "worse" if hi < 0 else "not_distinguishable")
            assert sg["roc_auc_ci95"][0] <= sg["roc_auc"] <= sg["roc_auc_ci95"][1]


def test_models_claimed_better_than_baseline_really_are(ctx):
    _, _, meta = ctx
    for k in C.REGRESSION_TASKS:
        assert meta[k]["significance"]["verdict"] == "better", k
    assert meta["high_rating"]["significance"]["accuracy_verdict"] == "better"
    assert meta["high_rating"]["significance"]["auc_above_chance"] and meta["referral"]["significance"]["auc_above_chance"]


def test_ui_status_exposes_verdicts_instead_of_point_estimate_claims(client):
    h = client.get("/api/status").json()["headline"]
    assert h["referral"]["significance"]["accuracy"] in {"better", "worse", "not_distinguishable"}
    assert "f1" in h["referral"]["significance"] and h["cost"]["significance"]["verdict"] == "better"


def test_validation_scores_are_not_wildly_optimistic_versus_holdout(ctx):
    """Selection uses out-of-fold scores; a large validation->hold-out degradation would signal selection leakage."""
    _, _, meta = ctx
    for k in C.REGRESSION_TASKS:
        assert meta[k]["test"]["mae"] < 1.25 * meta[k]["validation"]["mae"], k
    assert abs(meta["high_rating"]["test"]["accuracy"] - meta["high_rating"]["validation"]["accuracy"]) < 0.05

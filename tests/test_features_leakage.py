import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from coolroof import config as C
from coolroof import features as F


def test_all_task_contracts_pass():
    F.validate_all_contracts()


def test_features_are_only_pre_installation_and_exclude_id_and_targets():
    for k, t in C.TASKS.items():
        assert set(t.features) <= set(C.PRE_INSTALLATION_FEATURES)
        assert C.ID_COL not in t.features and t.target not in t.features
        assert not set(t.features) & set(C.POST_INSTALLATION_COLS)


@pytest.mark.parametrize("leak", ["cool_roofing_cost", "avg_roof_temperature_after", "avg_units_after",
                                  "customer_given_rating", "customer_referral", "building_id", "units_saved"])
def test_leaking_feature_is_rejected(monkeypatch, leak):
    bad = dataclasses.replace(C.TASKS["cost"], features=tuple(C.PRE_INSTALLATION_FEATURES) + (leak,))
    monkeypatch.setitem(C.TASKS, "cost", bad)
    with pytest.raises(F.LeakageError):
        F.validate_feature_contract("cost")


def test_target_as_feature_rejected(monkeypatch):
    bad = dataclasses.replace(C.TASKS["rating"], features=("customer_given_rating",))
    monkeypatch.setitem(C.TASKS, "rating", bad)
    with pytest.raises(F.LeakageError):
        F.validate_feature_contract("rating")


def test_frame_guard_blocks_leaking_columns(clean_res):
    d = clean_res.cleaned
    with pytest.raises(F.LeakageError):
        F.assert_frame_has_no_leakage(d[C.PRE_INSTALLATION_FEATURES + ["cool_roofing_cost"]])
    with pytest.raises(F.LeakageError):
        F.assert_frame_has_no_leakage(d[C.PRE_INSTALLATION_FEATURES + ["customer_referral"]])


@pytest.mark.parametrize("task", list(C.TASKS))
def test_design_matrix_matches_contract(clean_res, task):
    X = F.design_matrix(clean_res.cleaned, task)
    assert list(X.columns) == list(C.TASKS[task].features)


def test_preprocessing_is_fitted_on_training_data_only(clean_res):
    d = clean_res.cleaned
    train = d.iloc[:1500]
    pipe = F.build_pipeline(Ridge(), scale=True)
    y = train["cool_roofing_cost"].fillna(train["cool_roofing_cost"].median())
    pipe.fit(train[C.PRE_INSTALLATION_FEATURES], y)
    imp = pipe.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    cols = C.NUMERIC_FEATURE_COLS + F.ENGINEERED_FEATURES
    expected = pd.DataFrame(F.FeatureEngineer().fit_transform(train[C.PRE_INSTALLATION_FEATURES]))[cols].median().to_numpy()
    assert np.allclose(imp.statistics_, expected, equal_nan=True)
    full = pd.DataFrame(F.FeatureEngineer().fit_transform(d[C.PRE_INSTALLATION_FEATURES]))[cols].median().to_numpy()
    assert not np.allclose(imp.statistics_, full)   # would be equal had the full data leaked into the fit


def test_input_schema_covers_every_model_feature():
    s = F.input_schema()
    assert set(s["features"]) == set(C.PRE_INSTALLATION_FEATURES)
    assert set(s["required"]) | set(s["optional_imputed"]) == set(C.PRE_INSTALLATION_FEATURES)


def test_engineer_rejects_missing_inputs():
    with pytest.raises(KeyError):
        F.FeatureEngineer().fit_transform(pd.DataFrame({"building_size": [1.0]}))

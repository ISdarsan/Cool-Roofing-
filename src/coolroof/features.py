"""Feature contract, leakage checks and the shared preprocessing pipeline.

Training and inference both build their inputs through the functions in this
module, so the two can never drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

ENGINEERED_FEATURES = ["log_building_size"]
NUMERIC_MODEL_COLS = C.NUMERIC_FEATURE_COLS + ENGINEERED_FEATURES


class LeakageError(ValueError):
    """Raised when a feature set violates the feature contract."""


# --------------------------------------------------------------------------
# Contract checks
# --------------------------------------------------------------------------
def validate_feature_contract(task_key: str) -> None:
    """Raise LeakageError if the task's features break the contract."""
    spec = C.TASKS[task_key]
    feats = list(spec.features)
    forbidden = sorted(set(feats) & set(C.FORBIDDEN_AS_FEATURES))
    if forbidden:
        raise LeakageError(f"{task_key}: forbidden (post-installation / target / id) features used: {forbidden}")
    if spec.target in feats:
        raise LeakageError(f"{task_key}: target '{spec.target}' is listed as a feature")
    unknown = sorted(set(feats) - set(C.PRE_INSTALLATION_FEATURES))
    if unknown:
        raise LeakageError(f"{task_key}: features not on the pre-installation allow-list: {unknown}")


def validate_all_contracts() -> None:
    for k in C.TASKS:
        validate_feature_contract(k)


def assert_frame_has_no_leakage(X: pd.DataFrame, task_key: str | None = None) -> None:
    """Guard used immediately before fitting: the frame handed to a model may only
    contain allow-listed columns."""
    bad = sorted(set(X.columns) & set(C.FORBIDDEN_AS_FEATURES))
    if bad:
        raise LeakageError(f"leaking columns present in model input: {bad}")
    extra = sorted(set(X.columns) - set(C.PRE_INSTALLATION_FEATURES) - set(ENGINEERED_FEATURES))
    if extra:
        raise LeakageError(f"unexpected columns present in model input: {extra}")


def design_matrix(df: pd.DataFrame, task_key: str) -> pd.DataFrame:
    """The raw (un-preprocessed) feature frame for a task, in contract order."""
    validate_feature_contract(task_key)
    X = df[list(C.TASKS[task_key].features)].copy()
    assert_frame_has_no_leakage(X, task_key)
    return X


# --------------------------------------------------------------------------
# Transformers
# --------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Coerces raw inputs to a stable schema and adds derived (input-only) features."""

    def fit(self, X, y=None):
        self.feature_names_in_ = list(C.PRE_INSTALLATION_FEATURES)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        missing = [c for c in C.PRE_INSTALLATION_FEATURES if c not in X.columns]
        if missing:
            raise KeyError(f"missing required model inputs: {missing}")
        X = X[C.PRE_INSTALLATION_FEATURES].reset_index(drop=True)
        for c in C.NUMERIC_FEATURE_COLS:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        for c in C.CATEGORICAL_COLS:
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            X["log_building_size"] = np.where(X["building_size"] > 0, np.log(X["building_size"].clip(lower=1e-9)), np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(C.PRE_INSTALLATION_FEATURES + ENGINEERED_FEATURES, dtype=object)


def make_preprocessor(scale: bool = False) -> ColumnTransformer:
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("cat", cat, C.CATEGORICAL_COLS), ("num", Pipeline(num_steps), NUMERIC_MODEL_COLS)],
        verbose_feature_names_out=False)


def build_pipeline(model, scale: bool = False) -> Pipeline:
    """FeatureEngineer -> imputation/encoding (fitted on training data only) -> model."""
    return Pipeline([("fe", FeatureEngineer()), ("prep", make_preprocessor(scale)), ("model", model)])


# --------------------------------------------------------------------------
# Prediction-time schema helpers
# --------------------------------------------------------------------------
def input_schema() -> dict:
    """Machine-readable description of the prediction-time input (for docs/UI)."""
    fields = {}
    for c in C.CATEGORICAL_COLS:
        fields[c] = {"type": "category", "allowed": C.EXPECTED_CATEGORIES[c],
                     "decision_variable": c in C.DECISION_VARIABLES}
    for c in C.NUMERIC_FEATURE_COLS:
        lo, hi = C.API_BOUNDS[c]
        fields[c] = {"type": "number", "min": lo, "max": hi, "decision_variable": False}
    return {"features": fields, "required": ["location_type", "building_size", "roofing_type",
                                             "cool_roofing_quality", "cool_roofing_method"],
            "optional_imputed": ["building_age", "avg_roof_temperature_before", "num_ac_units", "avg_units_before"]}

"""Candidate models per task: a baseline, an interpretable benchmark and stronger models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from .. import config as C
from ..features import FeatureEngineer, make_preprocessor
from .estimators import ClippedRegressor, RatioTargetRegressor


@dataclass
class Candidate:
    name: str
    label: str
    family: str                       # baseline | interpretable | forest | boosting
    factory: Callable[[], object]
    space: dict = field(default_factory=dict)   # keys relative to the 'model' step
    search: str = "none"              # none | grid | random
    n_iter: int = 20
    prefix: str = ""                  # path from the top-level estimator down to the Pipeline
    notes: str = ""

    def param_space(self) -> dict:
        return {f"{self.prefix}model__{k}": v for k, v in self.space.items()}


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
def _pipe(model, scale: bool = False, poly: bool = False) -> Pipeline:
    steps = [("fe", FeatureEngineer()), ("prep", make_preprocessor(scale))]
    if poly:
        steps.append(("poly", PolynomialFeatures(2, interaction_only=True, include_bias=False)))
    steps.append(("model", model))
    return Pipeline(steps)


def _lgbm_reg(**kw) -> LGBMRegressor:
    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=10, subsample=0.85,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=C.SEED, n_jobs=1, verbose=-1)
    base.update(kw)
    return LGBMRegressor(**base)


def _lgbm_clf(**kw) -> LGBMClassifier:
    base = dict(n_estimators=250, learning_rate=0.04, num_leaves=6, min_child_samples=20, subsample=0.85,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=C.SEED, n_jobs=1, verbose=-1)
    base.update(kw)
    return LGBMClassifier(**base)


LGBM_SPACE = {
    "n_estimators": [150, 250, 400, 600], "learning_rate": [0.02, 0.04, 0.08],
    "num_leaves": [4, 6, 8, 12, 16], "min_child_samples": [5, 10, 20, 40],
    "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0.0, 1.0, 5.0, 20.0],
}
RF_SPACE = {"min_samples_leaf": [1, 2, 4, 8], "max_features": [0.4, 0.7, 1.0]}
RIDGE_SPACE = {"alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]}
LOGIT_SPACE = {"C": [0.01, 0.1, 1.0, 10.0]}


def _rf_reg() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, random_state=C.SEED, n_jobs=1)


def _log_target(inner) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=inner, func=np.log, inverse_func=np.exp, check_inverse=False)


# --------------------------------------------------------------------------
# regression tasks
# --------------------------------------------------------------------------
def _cost() -> list[Candidate]:
    return [
        Candidate("dummy_mean", "Baseline: training mean", "baseline", lambda: _pipe(DummyRegressor(strategy="mean"))),
        Candidate("linear", "Linear regression", "interpretable", lambda: _pipe(LinearRegression(), scale=True)),
        Candidate("loglinear_ridge", "Log-linear ridge (multiplicative)", "interpretable",
                  lambda: _log_target(_pipe(Ridge(1e-3), scale=True)), RIDGE_SPACE, "grid", prefix="regressor__",
                  notes="cost = exp(linear in log-size, age, method, quality...)"),
        Candidate("random_forest", "Random forest", "forest", lambda: _pipe(_rf_reg()), RF_SPACE, "grid"),
        Candidate("lightgbm", "LightGBM", "boosting", lambda: _pipe(_lgbm_reg()), LGBM_SPACE, "random"),
        Candidate("lightgbm_log", "LightGBM (log target)", "boosting", lambda: _log_target(_pipe(_lgbm_reg())),
                  LGBM_SPACE, "random", prefix="regressor__"),
        Candidate("ratio_lightgbm", "LightGBM on cost-per-area x area", "boosting",
                  lambda: RatioTargetRegressor(_pipe(_lgbm_reg()), denom="building_size"),
                  LGBM_SPACE, "random", prefix="regressor__"),
    ]


def _temperature() -> list[Candidate]:
    return [
        Candidate("dummy_mean", "Baseline: training mean", "baseline", lambda: _pipe(DummyRegressor(strategy="mean"))),
        Candidate("linear", "Ridge regression", "interpretable", lambda: _pipe(Ridge(1e-3), scale=True), RIDGE_SPACE, "grid"),
        Candidate("poly2_ridge", "Ridge with pairwise interactions", "interpretable",
                  lambda: _pipe(Ridge(10.0), scale=True, poly=True), RIDGE_SPACE, "grid"),
        Candidate("random_forest", "Random forest", "forest", lambda: _pipe(_rf_reg()), RF_SPACE, "grid"),
        Candidate("lightgbm", "LightGBM", "boosting", lambda: _pipe(_lgbm_reg()), LGBM_SPACE, "random"),
        Candidate("ratio_lightgbm", "LightGBM on drop-fraction x starting temperature", "boosting",
                  lambda: RatioTargetRegressor(_pipe(_lgbm_reg()), denom="avg_roof_temperature_before"),
                  LGBM_SPACE, "random", prefix="regressor__"),
    ]


def _units() -> list[Candidate]:
    return [
        Candidate("dummy_mean", "Baseline: training mean", "baseline", lambda: _pipe(DummyRegressor(strategy="mean"))),
        Candidate("linear", "Ridge regression", "interpretable", lambda: _pipe(Ridge(1e-3), scale=True), RIDGE_SPACE, "grid"),
        Candidate("poly2_ridge", "Ridge with pairwise interactions", "interpretable",
                  lambda: _pipe(Ridge(10.0), scale=True, poly=True), RIDGE_SPACE, "grid"),
        Candidate("random_forest", "Random forest", "forest", lambda: _pipe(_rf_reg()), RF_SPACE, "grid"),
        Candidate("lightgbm", "LightGBM", "boosting", lambda: _pipe(_lgbm_reg()), LGBM_SPACE, "random"),
        Candidate("ratio_lightgbm", "LightGBM on saved-fraction x current usage", "boosting",
                  lambda: RatioTargetRegressor(_pipe(_lgbm_reg()), denom="avg_units_before", weighted=True),
                  LGBM_SPACE, "random", prefix="regressor__"),
    ]


def _rating() -> list[Candidate]:
    clip = lambda m, scale=False: ClippedRegressor(_pipe(m, scale=scale), 1.0, 5.0)  # noqa: E731
    return [
        Candidate("dummy_mean", "Baseline: training mean rating", "baseline", lambda: clip(DummyRegressor(strategy="mean")),
                  prefix="regressor__"),
        Candidate("dummy_median", "Baseline: training median rating", "baseline", lambda: clip(DummyRegressor(strategy="median")),
                  prefix="regressor__"),
        Candidate("linear", "Ridge regression", "interpretable", lambda: clip(Ridge(10.0), True), RIDGE_SPACE, "grid", prefix="regressor__"),
        Candidate("random_forest", "Random forest", "forest", lambda: clip(RandomForestRegressor(300, min_samples_leaf=5, random_state=C.SEED, n_jobs=1)),
                  RF_SPACE, "grid", prefix="regressor__"),
        Candidate("lightgbm", "LightGBM", "boosting", lambda: clip(_lgbm_reg(num_leaves=4, min_child_samples=30, n_estimators=200, learning_rate=0.03)),
                  LGBM_SPACE, "random", prefix="regressor__"),
        Candidate("lightgbm_l1", "LightGBM (absolute-error objective)", "boosting",
                  lambda: clip(_lgbm_reg(objective="regression_l1", num_leaves=4, min_child_samples=30, n_estimators=200, learning_rate=0.03)),
                  LGBM_SPACE, "random", prefix="regressor__"),
    ]


# --------------------------------------------------------------------------
# classification tasks
# --------------------------------------------------------------------------
def _classifiers(tag: str) -> list[Candidate]:
    cw_lr = {**LOGIT_SPACE, "class_weight": [None, "balanced"]} if tag == "referral" else LOGIT_SPACE
    cw_rf = {**RF_SPACE, "class_weight": [None, "balanced_subsample"]} if tag == "referral" else RF_SPACE
    cw_gb = {**LGBM_SPACE, "class_weight": [None, "balanced"]} if tag == "referral" else LGBM_SPACE
    return [
        Candidate("dummy_prior", "Baseline: majority class (class prior)", "baseline",
                  lambda: _pipe(DummyClassifier(strategy="prior"))),
        Candidate("logistic", "Logistic regression", "interpretable",
                  lambda: _pipe(LogisticRegression(max_iter=3000), scale=True), cw_lr, "grid"),
        Candidate("random_forest", "Random forest", "forest",
                  lambda: _pipe(RandomForestClassifier(300, min_samples_leaf=5, random_state=C.SEED, n_jobs=1)), cw_rf, "grid"),
        Candidate("lightgbm", "LightGBM", "boosting", lambda: _pipe(_lgbm_clf()), cw_gb, "random"),
    ]


REGISTRY: dict[str, Callable[[], list[Candidate]]] = {
    "cost": _cost,
    "temperature_drop": _temperature,
    "units_saved": _units,
    "rating": _rating,
    "high_rating": lambda: _classifiers("high_rating"),
    "referral": lambda: _classifiers("referral"),
}


def get_candidates(task_key: str, fast: bool = False) -> list[Candidate]:
    """`fast=True` trims every search space (used by the automated tests, never for the real models)."""
    cands = REGISTRY[task_key]()
    if fast:
        for c in cands:
            c.space = {k: v[:2] for k, v in c.space.items()}
            c.n_iter = 3
    return cands

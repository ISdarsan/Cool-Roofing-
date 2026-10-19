"""Small sklearn-compatible wrappers used by the candidate models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone


class RatioTargetRegressor(BaseEstimator, RegressorMixin):
    """Models ``y / denom`` and multiplies back by ``denom`` at prediction time.

    Captures multiplicative structure (cost ~ size x rate, savings ~ usage x fraction,
    temperature drop ~ starting temperature x fraction) while the reported error is
    still in the original units. Rows whose denominator is not > 0 cannot be expressed
    as a ratio and are left out of *fitting*; at prediction time a zero denominator
    yields a zero prediction. A missing denominator is replaced by the training median.
    """

    def __init__(self, regressor, denom: str = "building_size", weighted: bool = False):
        self.regressor = regressor
        self.denom = denom
        self.weighted = weighted

    def _d(self, X: pd.DataFrame) -> np.ndarray:
        d = pd.to_numeric(pd.DataFrame(X)[self.denom], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isnan(d), self.fill_, d)

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        raw = pd.to_numeric(X[self.denom], errors="coerce")
        self.fill_ = float(raw.median())
        d = np.where(raw.isna(), self.fill_, raw)
        mask = d > 0
        self.regressor_ = clone(self.regressor)
        r = y[mask] / d[mask]
        if self.weighted:
            self.regressor_.fit(X[mask], r, model__sample_weight=d[mask])
        else:
            self.regressor_.fit(X[mask], r)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        d = self._d(X)
        return np.where(d > 0, self.regressor_.predict(X) * d, 0.0)


class ClippedRegressor(BaseEstimator, RegressorMixin):
    """Clips predictions to the valid target scale (e.g. ratings live in [1, 5])."""

    def __init__(self, regressor, low: float = 1.0, high: float = 5.0):
        self.regressor = regressor
        self.low = low
        self.high = high

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.regressor_.predict(X), self.low, self.high)

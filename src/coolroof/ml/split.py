"""Hold-out split, fixed once before any model selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .. import config as C


def holdout_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (train_validation_index, test_index) over *all* rows.

    The test set is chosen at the building level, stratified on the 1-5 rating so that
    it has the same outcome mix. Each task then intersects it with its eligible rows,
    so every task is evaluated on the same untouched buildings.
    """
    strat = df["customer_given_rating"].fillna(0).astype(int)
    tv, te = train_test_split(df.index.to_numpy(), test_size=C.TEST_SIZE, random_state=C.SEED, stratify=strat)
    return np.sort(tv), np.sort(te)

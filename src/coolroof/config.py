"""Central configuration: paths, column roles, feature contract and task registry.

Everything that defines *what the models may see* lives here so that the
training pipeline, the inference service and the automated leakage tests all
read the very same contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("COOLROOF_ROOT", Path(__file__).resolve().parents[2]))
RAW_CSV = ROOT / "dataset" / "cool_roofing_data.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
CLEANED_CSV = DATA_PROCESSED / "cleaned_dataset.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"

SEED = 42
TEST_SIZE = 0.20          # holdout share, reserved before any model selection
CV_FOLDS = 5
MODEL_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# Raw schema
# --------------------------------------------------------------------------
ID_COL = "building_id"
RAW_COLUMNS = [
    "building_id", "location_type", "building_size", "building_age",
    "cool_roofing_quality", "roofing_type", "cool_roofing_method",
    "cool_roofing_cost", "avg_roof_temperature_before", "num_ac_units",
    "avg_units_before", "avg_roof_temperature_after", "avg_units_after",
    "customer_given_rating", "customer_referral",
]
CATEGORICAL_COLS = ["location_type", "roofing_type", "cool_roofing_quality", "cool_roofing_method"]
NUMERIC_FEATURE_COLS = ["building_size", "building_age", "avg_roof_temperature_before",
                        "num_ac_units", "avg_units_before"]

# Categories observed in the supplied data (verified by the audit, which fails
# loudly if anything else appears).
EXPECTED_CATEGORIES = {
    "location_type": ["rural", "intermediate", "urban", "dense_urban"],
    "roofing_type": ["concrete_flat", "concrete_slanted", "sheet", "tiles"],
    "cool_roofing_quality": ["basic", "intermediate", "high"],
    "cool_roofing_method": ["basic_coating", "sheet", "sheet_with_coating", "tiles", "tiles_with_coating"],
}

# Physical / business plausibility bounds used by the data audit.
# Roof *surface* temperature: a monthly average above 80 C or below 10 C is not
# physically credible for the Indian climate; 120.0 appears as a repeated sentinel.
TEMP_PLAUSIBLE_RANGE = (10.0, 80.0)
TEMP_SENTINELS = (120.0,)
RATING_RANGE = (1, 5)
AC_UNITS_RANGE = (0, 10)
BUILDING_AGE_RANGE = (0, 100)
BUILDING_SIZE_RANGE = (50, 50_000)
# Hard bounds accepted by the prediction API (wider than training support;
# values outside training support trigger an extrapolation warning).
API_BOUNDS = {
    "building_size": (50.0, 50_000.0),
    "building_age": (0.0, 100.0),
    "avg_roof_temperature_before": (10.0, 80.0),
    "num_ac_units": (0, 10),
    "avg_units_before": (0.0, 5_000.0),
}

# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------
HIGH_RATING_THRESHOLD = int(os.environ.get("COOLROOF_HIGH_RATING_THRESHOLD", 4))  # ASSUMPTION: organisers did not define "high"; >=4 is used.
DEFAULT_TARIFF_INR_PER_KWH = None  # never invented: the user supplies the tariff.

# --------------------------------------------------------------------------
# Feature contract
# --------------------------------------------------------------------------
# Every feature below is knowable when a sales rep sits across the table, i.e.
# BEFORE installation.  `cool_roofing_method` and `cool_roofing_quality` are
# *decision variables*: the rep proposes them, so they can be varied to compare
# alternatives.
PRE_INSTALLATION_FEATURES = [
    "location_type", "building_size", "building_age", "roofing_type",
    "cool_roofing_quality", "cool_roofing_method",
    "avg_roof_temperature_before", "num_ac_units", "avg_units_before",
]
DECISION_VARIABLES = ["cool_roofing_method", "cool_roofing_quality"]

# Columns that only exist after the work is done (or that are targets).
POST_INSTALLATION_COLS = [
    "cool_roofing_cost", "avg_roof_temperature_after", "avg_units_after",
    "customer_given_rating", "customer_referral",
]
DERIVED_TARGET_COLS = ["temperature_drop", "units_saved", "high_rating"]
FORBIDDEN_AS_FEATURES = [ID_COL, *POST_INSTALLATION_COLS, *DERIVED_TARGET_COLS]


@dataclass(frozen=True)
class TaskSpec:
    key: str                 # stable machine key
    label: str               # human label
    target: str              # column in the cleaned dataset
    kind: str                # "regression" | "classification"
    unit: str
    features: tuple[str, ...]
    eligible_col: str        # boolean column in the cleaned dataset
    primary_metric: str      # metric used for model selection
    metrics: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


TASKS: dict[str, TaskSpec] = {
    "cost": TaskSpec(
        "cost", "Installation cost", "cool_roofing_cost", "regression", "INR",
        tuple(PRE_INSTALLATION_FEATURES), "eligible_cost", "mae", ("mae", "rmse"),
        "Total cost of the cool-roofing intervention in INR."),
    "temperature_drop": TaskSpec(
        "temperature_drop", "Roof temperature drop", "temperature_drop", "regression", "°C",
        tuple(PRE_INSTALLATION_FEATURES), "eligible_temperature_drop", "mae", ("mae", "rmse"),
        "avg_roof_temperature_before - avg_roof_temperature_after."),
    "units_saved": TaskSpec(
        "units_saved", "Monthly electricity saved", "units_saved", "regression", "kWh/month",
        tuple(PRE_INSTALLATION_FEATURES), "eligible_units_saved", "mae", ("mae", "rmse"),
        "avg_units_before - avg_units_after (monthly kWh)."),
    "rating": TaskSpec(
        "rating", "Customer rating (1-5)", "customer_given_rating", "regression", "rating points",
        tuple(PRE_INSTALLATION_FEATURES), "eligible_rating", "mae", ("mae", "accuracy"),
        "Numeric 1-5 rating (MAE) plus a high-rating (>= threshold) classification (accuracy)."),
    "high_rating": TaskSpec(
        "high_rating", "High rating (>= threshold)", "high_rating", "classification", "probability",
        tuple(PRE_INSTALLATION_FEATURES), "eligible_rating", "accuracy", ("accuracy", "f1"),
        "Separately trained classifier for customer_given_rating >= HIGH_RATING_THRESHOLD."),
    "referral": TaskSpec(
        "referral", "Customer referral", "customer_referral", "classification", "probability",
        tuple(PRE_INSTALLATION_FEATURES), "eligible_referral", "f1", ("accuracy", "f1"),
        "Whether the customer referred the company to someone else (positive class = True)."),
}
# Order in which tasks are trained / shown.
TASK_ORDER = ["cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral"]
REGRESSION_TASKS = [k for k in TASK_ORDER if TASKS[k].kind == "regression"]
CLASSIFICATION_TASKS = [k for k in TASK_ORDER if TASKS[k].kind == "classification"]


def ensure_dirs() -> None:
    for p in (DATA_PROCESSED, REPORTS, FIGURES, MODELS, DOCS):
        p.mkdir(parents=True, exist_ok=True)

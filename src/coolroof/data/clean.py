"""Data audit and cleaning.

Principles
----------
* The raw file is never modified. Cleaning returns a *new* frame.
* Nothing is dropped silently: every suspicious value becomes a row in the audit
  trail with a classification (invalid / uncertain / unusual / valid_adverse /
  missing) and the action taken.
* Invalid *feature* values become NaN and are imputed later with statistics fitted
  on training folds only. Invalid *target* values make the record ineligible for
  that task only; the record stays usable for every other task.
* Unusual-but-possible outcomes (e.g. electricity use that went UP after
  installation) are preserved because they are real adverse outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .. import config as C

# (roofing_type, method) pairs that never occur in the 2,500 records and are
# physically contradictory (a tile product cannot be laid on a sheet roof and
# vice versa).  The audit verifies no record violates the rule.
INCOMPATIBLE_ROOF_METHOD = {
    ("sheet", "tiles"), ("sheet", "tiles_with_coating"),
    ("tiles", "sheet"), ("tiles", "sheet_with_coating"),
}

AUDIT_COLUMNS = [
    "check", "column", "building_id", "observed_value", "reason", "classification",
    "action", "changes_target", "changes_feature", "tasks_affected",
]


@dataclass
class CleaningResult:
    raw: pd.DataFrame
    cleaned: pd.DataFrame
    audit: pd.DataFrame
    summary: dict[str, Any]


# --------------------------------------------------------------------------
# Loading / standardising
# --------------------------------------------------------------------------
def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def _to_bool(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False, "y": True, "n": False}
    if series.dtype == bool:
        return series
    out = series.astype("string").str.strip().str.lower().map(mapping)
    return out.astype("object").where(out.notna(), None)


def standardize(df: pd.DataFrame, notes: list[dict]) -> pd.DataFrame:
    """Trim/normalise labels, coerce numerics and booleans. Records what changed."""
    d = df.copy()
    for col in C.CATEGORICAL_COLS:
        before = d[col].copy()
        d[col] = d[col].astype("string").str.strip().str.lower().str.replace(" ", "_")
        d[col] = d[col].astype(object).where(d[col].notna(), np.nan)
        changed = int(((before.astype("string") != d[col].astype("string")) & before.notna()).sum())
        notes.append({"check": "label_standardisation", "column": col, "n_changed": changed})
    numeric_cols = ["building_size", "building_age", "cool_roofing_cost", "avg_roof_temperature_before",
                    "num_ac_units", "avg_units_before", "avg_roof_temperature_after", "avg_units_after",
                    "customer_given_rating"]
    for col in numeric_cols:
        coerced = pd.to_numeric(d[col], errors="coerce")
        newly_na = int((coerced.isna() & d[col].notna()).sum())
        notes.append({"check": "numeric_coercion", "column": col, "n_unparseable": newly_na})
        d[col] = coerced
    d["customer_referral"] = _to_bool(d["customer_referral"])
    return d


# --------------------------------------------------------------------------
# Audit helpers
# --------------------------------------------------------------------------
class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, check: str, column: str, idx, df: pd.DataFrame, reason: str, classification: str,
            action: str, changes_target: bool, changes_feature: bool, tasks: str) -> None:
        for i in idx:
            val = df.at[i, column] if column in df.columns else None
            self.rows.append({
                "check": check, "column": column,
                "building_id": df.at[i, C.ID_COL] if C.ID_COL in df.columns else i,
                "observed_value": "" if pd.isna(val) else val,
                "reason": reason, "classification": classification, "action": action,
                "changes_target": changes_target, "changes_feature": changes_feature,
                "tasks_affected": tasks,
            })

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=AUDIT_COLUMNS)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def clean(raw: pd.DataFrame) -> CleaningResult:
    notes: list[dict] = []
    audit = _Audit()
    n_raw = len(raw)

    # ---- structural validation -------------------------------------------
    missing_cols = [c for c in C.RAW_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Raw data is missing required columns: {missing_cols}")
    extra_cols = [c for c in raw.columns if c not in C.RAW_COLUMNS]

    df = standardize(raw[C.RAW_COLUMNS], notes)
    structural = {
        "rows": n_raw, "columns": int(raw.shape[1]), "extra_columns": extra_cols,
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_building_ids": int(raw[C.ID_COL].duplicated().sum()),
        "duplicate_rows_excluding_id": int(raw.drop(columns=[C.ID_COL]).duplicated().sum()),
        "id_is_sequential": bool((raw[C.ID_COL].sort_values().to_numpy() == np.arange(1, n_raw + 1)).all()),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
    }
    unexpected: dict[str, list[str]] = {}
    for col, allowed in C.EXPECTED_CATEGORIES.items():
        bad = sorted(set(df[col].dropna().unique()) - set(allowed))
        if bad:
            unexpected[col] = bad
            idx = df.index[df[col].isin(bad)]
            audit.add("unexpected_category", col, idx, df, f"label not in {allowed}", "invalid",
                      "set to missing; imputed from training-fold mode", False, True, "all (feature)")
            df.loc[idx, col] = np.nan
    structural["unexpected_categories"] = unexpected

    missing_before = {c: int(raw[c].isna().sum()) for c in C.RAW_COLUMNS}

    # ---- missing feature values (kept; imputed inside the training pipeline) ----
    for col, why in [("location_type", "location unknown"), ("building_size", "floor area unknown"),
                     ("building_age", "age unknown"), ("num_ac_units", "AC count unknown")]:
        idx = df.index[df[col].isna()]
        tasks = "all (feature imputed in pipeline)"
        action = "kept; imputed with training-fold statistic + missing-indicator"
        if col == "building_size":
            tasks = "cost excluded; others imputed"
            action = "excluded from cost task (size is the dominant cost driver, imputing it would inject label noise); imputed for other tasks"
        audit.add("missing_value", col, idx, df, why, "missing", action, col == "building_size", True, tasks)

    # ---- temperature plausibility ---------------------------------------
    lo, hi = C.TEMP_PLAUSIBLE_RANGE
    for col in ("avg_roof_temperature_before", "avg_roof_temperature_after"):
        s = df[col]
        sentinel = s.isin(C.TEMP_SENTINELS)
        outside = (s < lo) | (s > hi)
        bad = sentinel | outside
        idx = df.index[bad]
        reason = (f"value {C.TEMP_SENTINELS[0]:.0f} repeated {int(sentinel.sum())}x and outside the plausible "
                  f"{lo:.0f}-{hi:.0f} C range for an average roof-surface temperature (placeholder/sentinel)")
        is_before = col.endswith("before")
        audit.add("temperature_sentinel_or_out_of_range", col, idx, df, reason, "invalid",
                  "set to missing (true value cannot be recovered); "
                  + ("feature imputed in pipeline; " if is_before else "") + "excluded from temperature-drop task",
                  True, is_before, "temperature_drop" + (" (+feature imputation elsewhere)" if is_before else ""))
        df.loc[idx, col] = np.nan

    both = df["avg_roof_temperature_before"].notna() & df["avg_roof_temperature_after"].notna()
    hotter = both & (df["avg_roof_temperature_after"] > df["avg_roof_temperature_before"])
    audit.add("roof_hotter_after", "avg_roof_temperature_after", df.index[hotter], df,
              "roof surface hotter after installation", "uncertain", "kept (could be a real adverse outcome)", False, False, "none")

    # ---- electricity -------------------------------------------------------
    for col in ("avg_units_before", "avg_units_after"):
        neg = df[col] < 0
        audit.add("negative_consumption", col, df.index[neg], df, "negative kWh is impossible", "invalid",
                  "set to missing; units_saved task excluded", True, col == "avg_units_before", "units_saved")
        df.loc[neg, col] = np.nan
        s_ = df[col]
        zero = s_ == 0
        n_1_3 = int(((s_ >= 1) & (s_ <= 3)).sum())
        n_4_8 = int(((s_ >= 4) & (s_ <= 8)).sum())
        evidence = (f"exactly 0 kWh appears {int(zero.sum())}x while 1-3 kWh never occurs ({n_1_3}x) and 4-8 kWh occurs {n_4_8}x "
                    f"(the natural lower tail starts near 4-5); the zero rows are also physically inconsistent with their other readings "
                    f"(see report) - a placeholder for a missing reading, like the 120 C temperature")
        audit.add("zero_consumption_placeholder", col, df.index[zero], df, evidence, "invalid",
                  "set to missing (true value cannot be recovered)" + ("; feature imputed in pipeline" if col == "avg_units_before" else "")
                  + "; units_saved task excludes the record", True, col == "avg_units_before", "units_saved")
        df.loc[zero, col] = np.nan
    saved = df["avg_units_before"] - df["avg_units_after"]
    audit.add("negative_savings", "avg_units_after", df.index[saved < 0], df,
              "electricity use increased after installation (after > before); concentrated in low-usage, "
              "zero-AC buildings, consistent with measurement noise around a small base load or seasonal change",
              "valid_adverse", "kept as a genuine (undesirable) outcome", False, False, "none")
    ac0_heavy = (df["num_ac_units"] == 0) & (df["avg_units_before"] > 150)
    audit.add("zero_ac_high_usage", "avg_units_before", df.index[ac0_heavy], df,
              "no AC units but very high consumption", "uncertain", "kept", False, False, "none")

    # ---- cost ---------------------------------------------------------------
    nonpos = df["cool_roofing_cost"] <= 0
    audit.add("non_positive_cost", "cool_roofing_cost", df.index[nonpos], df, "cost must be > 0", "invalid",
              "set to missing; cost task excluded", True, False, "cost")
    df.loc[nonpos, "cool_roofing_cost"] = np.nan
    cps = df["cool_roofing_cost"] / df["building_size"]
    grp = cps.groupby([df["cool_roofing_method"], df["cool_roofing_quality"]]).transform("median")
    ratio = cps / grp
    odd = (ratio < 1 / 3) | (ratio > 3)
    audit.add("cost_per_area_outlier", "cool_roofing_cost", df.index[odd.fillna(False)], df,
              "cost per unit area is >3x or <1/3 of the median for the same method and quality", "unusual",
              "kept (not provably wrong)", False, False, "none")

    # ---- other numeric ranges ------------------------------------------------
    def range_check(col, lo_, hi_, name, tasks_feature=True):
        bad = df[col].notna() & ((df[col] < lo_) | (df[col] > hi_))
        audit.add(name, col, df.index[bad], df, f"outside plausible range [{lo_}, {hi_}]", "invalid",
                  "set to missing; imputed in pipeline", False, True, "all (feature)")
        df.loc[bad, col] = np.nan

    range_check("building_size", *C.BUILDING_SIZE_RANGE, "building_size_range")
    range_check("building_age", *C.BUILDING_AGE_RANGE, "building_age_range")
    range_check("num_ac_units", *C.AC_UNITS_RANGE, "ac_units_range")
    frac_ac = df["num_ac_units"].notna() & (df["num_ac_units"] % 1 != 0)
    audit.add("ac_units_non_integer", "num_ac_units", df.index[frac_ac], df, "AC unit count must be an integer",
              "invalid", "set to missing", False, True, "all (feature)")
    df.loc[frac_ac, "num_ac_units"] = np.nan

    # ---- rating / referral ----------------------------------------------------
    r = df["customer_given_rating"]
    bad_r = r.notna() & ((r < C.RATING_RANGE[0]) | (r > C.RATING_RANGE[1]) | (r % 1 != 0))
    audit.add("rating_range", "customer_given_rating", df.index[bad_r], df, "rating must be an integer 1-5",
              "invalid", "set to missing; rating/high-rating tasks excluded", True, False, "rating, high_rating")
    df.loc[bad_r, "customer_given_rating"] = np.nan
    bad_ref = df["customer_referral"].isna()
    audit.add("referral_invalid", "customer_referral", df.index[bad_ref], df, "referral flag missing/unparseable",
              "invalid", "excluded from referral task", True, False, "referral")

    # ---- structural business rule ----------------------------------------------
    pair = list(zip(df["roofing_type"], df["cool_roofing_method"]))
    incompatible = pd.Series([p in INCOMPATIBLE_ROOF_METHOD for p in pair], index=df.index)
    audit.add("incompatible_roof_method", "cool_roofing_method", df.index[incompatible], df,
              "roof type and installation method are physically incompatible", "invalid", "excluded from all tasks",
              True, True, "all")

    # ---- derived targets & eligibility ------------------------------------------
    out = df.copy()
    out["temperature_drop"] = out["avg_roof_temperature_before"] - out["avg_roof_temperature_after"]
    out["units_saved"] = out["avg_units_before"] - out["avg_units_after"]
    out["high_rating"] = np.where(out["customer_given_rating"].notna(),
                                  (out["customer_given_rating"] >= C.HIGH_RATING_THRESHOLD).astype(float), np.nan)
    out["customer_referral"] = out["customer_referral"].astype("boolean")

    keep = ~incompatible
    out["eligible_cost"] = keep & out["cool_roofing_cost"].notna() & out["building_size"].notna()
    out["eligible_temperature_drop"] = keep & out["temperature_drop"].notna()
    out["eligible_units_saved"] = keep & out["units_saved"].notna()
    out["eligible_rating"] = keep & out["customer_given_rating"].notna()
    out["eligible_referral"] = keep & out["customer_referral"].notna()
    out["flag_temp_before_invalid"] = raw["avg_roof_temperature_before"].reset_index(drop=True).isin(C.TEMP_SENTINELS) | \
        out["avg_roof_temperature_before"].isna()
    out["flag_temp_after_invalid"] = out["avg_roof_temperature_after"].isna()
    out["flag_negative_savings"] = (out["units_saved"] < 0).fillna(False)
    out["flag_units_placeholder"] = ((raw["avg_units_before"] == 0) | (raw["avg_units_after"] == 0)).to_numpy()
    out["n_feature_missing"] = out[C.PRE_INSTALLATION_FEATURES].isna().sum(axis=1)

    audit_df = audit.frame()
    missing_after = {c: int(out[c].isna().sum()) for c in C.RAW_COLUMNS}

    # ---- summary -------------------------------------------------------------------
    eligible_counts = {k: int(out[t.eligible_col].sum()) for k, t in C.TASKS.items()}
    excl_reasons = {
        "cost": {"missing_building_size": int((out["building_size"].isna()).sum()),
                 "non_positive_cost": int(nonpos.sum())},
        "temperature_drop": {
            "temperature_before_invalid": int(out["flag_temp_before_invalid"].sum()),
            "temperature_after_invalid": int(out["flag_temp_after_invalid"].sum()),
            "either_invalid": int((~out["eligible_temperature_drop"]).sum())},
        "units_saved": {"zero_or_negative_consumption_placeholder": int((~out["eligible_units_saved"]).sum()),
                        "avg_units_before_zero": int((raw["avg_units_before"] == 0).sum()),
                        "avg_units_after_zero": int((raw["avg_units_after"] == 0).sum())},
        "rating": {"invalid_rating": int((~out["eligible_rating"]).sum())},
        "referral": {"invalid_referral": int((~out["eligible_referral"]).sum())},
    }
    by_class = audit_df.groupby(["check", "classification"]).size().reset_index(name="n_records")
    summary = {
        "n_raw_rows": n_raw,
        "structural": structural,
        "label_and_type_notes": notes,
        "missing_before": missing_before,
        "missing_after_cleaning": missing_after,
        "rows_with_any_missing_feature": int((out["n_feature_missing"] > 0).sum()),
        "eligible_rows_by_task": eligible_counts,
        "exclusions_by_task": excl_reasons,
        "records_excluded_from_every_task": int((~out[[t.eligible_col for t in C.TASKS.values()]].any(axis=1)).sum()),
        "audit_findings": by_class.to_dict(orient="records"),
        "zero_consumption_placeholders": {
            "before": int((raw["avg_units_before"] == 0).sum()), "after": int((raw["avg_units_after"] == 0).sum()),
            "both": int(((raw["avg_units_before"] == 0) & (raw["avg_units_after"] == 0)).sum()),
            "after_when_before_is_zero_max": float(raw.loc[raw["avg_units_before"] == 0, "avg_units_after"].max()),
            "before_when_after_is_zero_max": float(raw.loc[raw["avg_units_after"] == 0, "avg_units_before"].max()),
            "min_after_over_before_ratio_among_valid": float((raw.loc[(raw.avg_units_before > 0) & (raw.avg_units_after > 0), "avg_units_after"]
                                                             / raw.loc[(raw.avg_units_before > 0) & (raw.avg_units_after > 0), "avg_units_before"]).min()),
        },
        "negative_savings_rows": int(out["flag_negative_savings"].sum()),
        "zero_savings_rows": int((out["units_saved"] == 0).sum()),
        "temperature_sentinel_rows": {
            "before": int(raw["avg_roof_temperature_before"].isin(C.TEMP_SENTINELS).sum()),
            "after": int(raw["avg_roof_temperature_after"].isin(C.TEMP_SENTINELS).sum()),
            "both": int((raw["avg_roof_temperature_before"].isin(C.TEMP_SENTINELS)
                         & raw["avg_roof_temperature_after"].isin(C.TEMP_SENTINELS)).sum()),
        },
        "valid_temperature_drop_range": [float(out["temperature_drop"].min()), float(out["temperature_drop"].max())],
        "assumptions": {
            "units": "avg_units_* are monthly kWh (stated in the problem statement: 'average monthly electricity consumption (kWh)')",
            "building_size_unit": "not stated in the problem statement ('floor area'); treated as an opaque area unit, shown as 'area units' / sq ft in the UI",
            "high_rating_threshold": C.HIGH_RATING_THRESHOLD,
        },
    }
    return CleaningResult(raw=raw, cleaned=out, audit=audit_df, summary=summary)


def save_artifacts(res: CleaningResult) -> None:
    import json

    C.ensure_dirs()
    res.cleaned.to_csv(C.CLEANED_CSV, index=False)
    res.audit.to_csv(C.REPORTS / "cleaning_audit.csv", index=False)
    from ..utils import write_json
    write_json(C.REPORTS / "data_quality_summary.json", res.summary, indent=2)


def load_cleaned() -> pd.DataFrame:
    """Load the cleaned dataset, regenerating it from the raw file when absent."""
    if C.CLEANED_CSV.exists():
        df = pd.read_csv(C.CLEANED_CSV)
        df["customer_referral"] = df["customer_referral"].astype("boolean")
        return df
    res = clean(load_raw())
    save_artifacts(res)
    return res.cleaned

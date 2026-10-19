import numpy as np
import pandas as pd
import pytest

from coolroof import config as C
from coolroof.data import clean
from conftest import sha256, tiny_frame


def run(df):
    return clean.clean(df)


def test_required_columns_validated(raw_df):
    with pytest.raises(ValueError, match="missing required columns"):
        clean.clean(raw_df.drop(columns=["cool_roofing_cost"]))


def test_raw_file_is_preserved_and_matches_archive():
    assert sha256(C.RAW_CSV) == sha256(C.ROOT / "data" / "raw" / "cool_roofing_data.original.csv")


def test_structure_counts(clean_res):
    s = clean_res.summary["structural"]
    assert s["rows"] == 2500 and s["columns"] == 15
    assert s["duplicate_rows"] == 0 and s["duplicate_building_ids"] == 0


def test_temperature_sentinel_becomes_missing_and_only_blocks_temperature_task():
    res = run(tiny_frame(avg_roof_temperature_after=120.0))
    r = res.cleaned.iloc[0]
    assert np.isnan(r.avg_roof_temperature_after)
    assert not r.eligible_temperature_drop
    assert r.eligible_cost and r.eligible_units_saved and r.eligible_rating and r.eligible_referral
    assert (res.audit.check == "temperature_sentinel_or_out_of_range").sum() == 1


def test_invalid_before_temperature_is_also_imputed_feature():
    r = run(tiny_frame(avg_roof_temperature_before=120.0)).cleaned.iloc[0]
    assert np.isnan(r.avg_roof_temperature_before) and r.n_feature_missing >= 1 and not r.eligible_temperature_drop


def test_zero_consumption_placeholder_invalid_but_other_tasks_kept():
    r = run(tiny_frame(avg_units_before=0, avg_units_after=64)).cleaned.iloc[0]
    assert np.isnan(r.avg_units_before) and not r.eligible_units_saved and r.eligible_cost


def test_negative_savings_are_preserved_as_valid_adverse_outcome():
    res = run(tiny_frame(avg_units_before=20, avg_units_after=27))
    r = res.cleaned.iloc[0]
    assert r.units_saved == -7 and r.eligible_units_saved
    assert (res.audit[res.audit.check == "negative_savings"].classification == "valid_adverse").all()


def test_missing_building_size_excludes_only_cost_task():
    r = run(tiny_frame(building_size=np.nan)).cleaned.iloc[0]
    assert not r.eligible_cost and r.eligible_units_saved and r.eligible_temperature_drop


def test_non_positive_cost_excluded():
    r = run(tiny_frame(cool_roofing_cost=-5)).cleaned.iloc[0]
    assert not r.eligible_cost and r.eligible_rating


def test_invalid_rating_and_referral_handling():
    r = run(tiny_frame(customer_given_rating=9)).cleaned.iloc[0]
    assert not r.eligible_rating and np.isnan(r.high_rating)
    r = run(tiny_frame(customer_referral=None)).cleaned.iloc[0]
    assert not r.eligible_referral


def test_unexpected_category_flagged_and_set_missing():
    res = run(tiny_frame(location_type="coastal"))
    assert pd.isna(res.cleaned.iloc[0].location_type)
    assert "unexpected_category" in set(res.audit.check)


def test_label_standardisation_trims_and_lowercases():
    r = run(tiny_frame(location_type="  Urban ")).cleaned.iloc[0]
    assert r.location_type == "urban"


def test_incompatible_roof_method_excluded_everywhere():
    res = run(tiny_frame(roofing_type="sheet", cool_roofing_method="tiles"))
    r = res.cleaned.iloc[0]
    assert not any(r[t.eligible_col] for t in C.TASKS.values())


def test_target_calculation(clean_res):
    d = clean_res.cleaned
    ok = d.eligible_temperature_drop
    assert np.allclose(d.loc[ok, "temperature_drop"], d.loc[ok, "avg_roof_temperature_before"] - d.loc[ok, "avg_roof_temperature_after"])
    ok = d.eligible_units_saved
    assert np.allclose(d.loc[ok, "units_saved"], d.loc[ok, "avg_units_before"] - d.loc[ok, "avg_units_after"])
    assert (d.high_rating == (d.customer_given_rating >= C.HIGH_RATING_THRESHOLD).astype(float)).all()


def test_real_data_known_findings(clean_res):
    s = clean_res.summary
    assert s["temperature_sentinel_rows"] == {"before": 32, "after": 26, "both": 0}
    assert s["zero_consumption_placeholders"]["before"] == 26 and s["zero_consumption_placeholders"]["after"] == 22
    assert s["eligible_rows_by_task"]["temperature_drop"] == 2442
    assert s["eligible_rows_by_task"]["units_saved"] == 2452
    assert s["eligible_rows_by_task"]["cost"] == 2472
    assert s["negative_savings_rows"] > 100          # adverse outcomes are kept, not deleted
    d = clean_res.cleaned
    assert d.temperature_drop.max() < 20 and d.temperature_drop.min() > 0


def test_cleaning_is_reproducible(raw_df):
    a, b = clean.clean(raw_df).cleaned, clean.clean(raw_df).cleaned
    pd.testing.assert_frame_equal(a, b)


def test_audit_trail_has_required_fields(clean_res):
    a = clean_res.audit
    assert list(a.columns) == clean.AUDIT_COLUMNS
    assert set(a.classification) <= {"invalid", "uncertain", "unusual", "valid_adverse", "missing"}
    assert a.reason.str.len().min() > 0 and a.action.str.len().min() > 0


def test_task_sample_counts_match_flags(clean_res):
    d, s = clean_res.cleaned, clean_res.summary
    for k, t in C.TASKS.items():
        assert int(d[t.eligible_col].sum()) == s["eligible_rows_by_task"][k]

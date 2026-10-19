import copy

import pytest

from coolroof import recommend as R

PAIRS = [{"roofing_type": "sheet", "cool_roofing_method": m, "n": n} for m, n in
         [("basic_coating", 500), ("sheet", 500), ("sheet_with_coating", 500), ("tiles", 0), ("tiles_with_coating", 0)]]
METHODS = ["basic_coating", "sheet", "sheet_with_coating", "tiles", "tiles_with_coating"]
QUALS = ["basic", "intermediate", "high"]


def opt(cost, drop, kwh, hi, ref):
    return {"predictions": {"cost": cost, "temperature_drop": drop, "units_saved": kwh, "high_rating": hi, "referral": ref}}


def test_valid_options_respect_roof_support():
    opts = R.valid_options("sheet", PAIRS, METHODS, QUALS)
    assert {o["cool_roofing_method"] for o in opts} == {"basic_coating", "sheet", "sheet_with_coating"}
    assert len(opts) == 9


def test_normalisation_is_direction_aware():
    assert R._normalise([100, 200, 300], "min") == [1.0, 0.5, 0.0]
    assert R._normalise([1, 2, 3], "max") == [0.0, 0.5, 1.0]
    assert R._normalise([5, 5, 5], "max") == [0.5, 0.5, 0.5]       # no spread -> no discrimination


def test_weights_are_rescaled_and_score_is_0_to_100():
    options = [opt(100, 5, 40, 0.3, 0.3), opt(200, 9, 80, 0.6, 0.5)]
    R.score_options(options, {"cost": 2, "temperature_drop": 2})
    for o in options:
        assert 0 <= o["score"] <= 100
        assert sum(o["contributions"].values()) == pytest.approx(o["score"] / 100)
    assert options[0]["score"] == pytest.approx(50) and options[1]["score"] == pytest.approx(50)   # cost vs cooling tie


def test_objectives_pick_expected_option():
    options = [opt(100, 5, 40, 0.3, 0.2), opt(200, 9, 80, 0.6, 0.4), opt(150, 7, 90, 0.4, 0.5)]
    pick = lambda obj: R.recommend(copy.deepcopy(options), obj)["recommended"]["predictions"]  # noqa: E731
    assert pick("cost")["cost"] == 100
    assert pick("temperature_drop")["temperature_drop"] == 9
    assert pick("units_saved")["units_saved"] == 90
    assert pick("high_rating")["high_rating"] == 0.6
    assert pick("referral")["referral"] == 0.5


def test_balanced_recommendation_changes_with_priorities():
    options = [opt(100, 5, 40, 0.3, 0.2), opt(300, 10, 90, 0.7, 0.5)]
    cheap = R.recommend(copy.deepcopy(options), "balanced", {"cost": 1.0})
    quality = R.recommend(copy.deepcopy(options), "balanced", {"temperature_drop": 1, "units_saved": 1, "high_rating": 1})
    assert cheap["recommended"]["predictions"]["cost"] == 100
    assert quality["recommended"]["predictions"]["cost"] == 300
    assert cheap["weights"] == {"cost": 1.0}


def test_rejects_unknown_objective_and_empty_options():
    with pytest.raises(ValueError):
        R.recommend([opt(1, 1, 1, .1, .1)], "cheapest-ish")
    with pytest.raises(ValueError):
        R.recommend([], "cost")
    with pytest.raises(ValueError):
        R.score_options([opt(1, 1, 1, .1, .1)], {"cost": 0})


def test_close_call_is_flagged_within_model_error():
    options = [opt(100, 5.0, 40, 0.3, 0.2), opt(101, 5.1, 40.5, 0.3, 0.2)]
    rec = R.recommend(options, "temperature_drop", model_mae={"temperature_drop": 0.5})
    assert rec["close_call"] and any("equivalent" in n for n in rec["notes"])


def test_economics_known_numbers():
    e = R.economics(R.EconomicsInput(cost_inr=120_000, monthly_kwh_saved=100, tariff_inr_per_kwh=10))
    assert e["central"]["annual_bill_saving_inr"] == pytest.approx(12_000)
    assert e["central"]["simple_payback_years"] == pytest.approx(10.0)
    assert e["central"]["breakeven_tariff_inr_per_kwh"] == pytest.approx(10.0)    # 120000/10yr/1200kWh


def test_economics_handles_zero_negative_savings_and_maintenance():
    e = R.economics(R.EconomicsInput(100_000, 0, 8))
    assert e["central"]["simple_payback_years"] is None and e["warnings"]
    e = R.economics(R.EconomicsInput(100_000, -20, 8))
    assert e["central"]["simple_payback_years"] is None
    e = R.economics(R.EconomicsInput(100_000, 100, 10, annual_maintenance_inr=12_000))
    assert e["central"]["simple_payback_years"] is None            # maintenance eats the whole saving


def test_economics_uncertainty_range_and_escalation():
    e = R.economics(R.EconomicsInput(100_000, 100, 10, tariff_escalation_pct=5), kwh_low=60, kwh_high=140)
    assert e["range"]["optimistic"]["simple_payback_years"] < e["central"]["simple_payback_years"] < e["range"]["pessimistic"]["simple_payback_years"]
    assert e["central"]["payback_with_escalation_years"] < e["central"]["simple_payback_years"]
    assert any("not a guarantee" in a.lower() for a in e["assumptions"])


def test_economics_requires_user_supplied_tariff():
    with pytest.raises(ValueError):
        R.economics(R.EconomicsInput(100_000, 100, None))
    with pytest.raises(ValueError):
        R.economics(R.EconomicsInput(100_000, 100, 0))

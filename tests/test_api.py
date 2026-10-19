import math

import pytest

from coolroof import config as C


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["models_loaded"] is True


def test_status_has_real_holdout_metrics(client):
    s = client.get("/api/status").json()
    assert set(s["headline"]) == set(C.TASK_ORDER)
    assert s["headline"]["cost"]["test"]["mae"] > 0 and s["headline"]["referral"]["test"]["f1"] > 0
    assert s["headline"]["cost"]["n_test"] > 300
    assert "assistant" in s


def test_schema_endpoint(client):
    s = client.get("/api/schema").json()
    assert set(s["features"]) == set(C.PRE_INSTALLATION_FEATURES)
    assert s["high_rating_threshold"] == C.HIGH_RATING_THRESHOLD
    assert abs(sum(s["default_weights"].values()) - 1) < 1e-9


def test_predict_all_five_tasks(client, building):
    r = client.post("/api/predict", json={"building": building})
    assert r.status_code == 200
    p = r.json()["predictions"]
    assert finite(p["cost"]["value"]) and p["cost"]["value"] > 0
    assert p["cost"]["interval"]["low"] < p["cost"]["value"] < p["cost"]["interval"]["high"]
    assert p["temperature_drop"]["value"] > 0
    assert finite(p["units_saved"]["value"])
    assert 1 <= p["rating"]["value"] <= 5
    assert 0 <= p["high_rating"]["probability"] <= 1 and "definition" in p["high_rating"]
    assert 0 <= p["referral"]["probability"] <= 1 and isinstance(p["referral"]["predicted_referral"], bool)
    assert r.json()["assumptions"]


def test_predict_derived_after_values_labelled(client, building):
    d = client.post("/api/predict", json={"building": building}).json()["derived"]
    assert d["avg_units_after"]["note"].startswith("derived")


def test_predict_reports_warnings_for_missing_and_extrapolated(client, building):
    b = {**building, "building_age": None, "building_size": 20000}
    w = client.post("/api/predict", json={"building": b}).json()["warnings"]
    assert any("building_age" in x for x in w) and any("extrapolation" in x for x in w)


@pytest.mark.parametrize("field,value", [("location_type", "coastal"), ("building_size", -5), ("building_size", "big"),
                                         ("num_ac_units", 99), ("avg_roof_temperature_before", 500), ("cool_roofing_method", "paint")])
def test_validation_errors_are_422_and_do_not_echo_input(client, building, field, value):
    r = client.post("/api/predict", json={"building": {**building, field: value}})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"] == "Invalid request." and body["errors"][0]["field"].startswith("building.")
    assert str(value) not in r.text or isinstance(value, int)     # input value not echoed back (ints may coincide with text)


def test_missing_required_field_422(client, building):
    b = dict(building)
    del b["building_size"]
    assert client.post("/api/predict", json={"building": b}).status_code == 422


@pytest.mark.parametrize("leak", ["cool_roofing_cost", "customer_given_rating", "customer_referral",
                                  "avg_units_after", "avg_roof_temperature_after", "building_id"])
def test_post_installation_fields_are_rejected(client, building, leak):
    r = client.post("/api/predict", json={"building": {**building, leak: 1}})
    assert r.status_code == 422


def test_unsupported_roof_method_combination_is_422(client, building):
    r = client.post("/api/predict", json={"building": {**building, "roofing_type": "sheet", "cool_roofing_method": "tiles"}})
    assert r.status_code == 422 and "not offered" in r.json()["detail"]


def test_batch_predict(client, building):
    items = [building, {**building, "building_size": 900, "cool_roofing_quality": "basic"}]
    r = client.post("/api/predict/batch", json={"items": items})
    assert r.status_code == 200 and r.json()["count"] == 2
    assert r.json()["predictions"][0]["cost"]["value"] > r.json()["predictions"][1]["cost"]["value"]
    assert client.post("/api/predict/batch", json={"items": []}).status_code == 422


def test_compare_options_balanced_and_objectives(client, building):
    body = {"building": {k: v for k, v in building.items() if k not in ("cool_roofing_method", "cool_roofing_quality")},
            "current_method": "basic_coating", "current_quality": "basic"}
    r = client.post("/api/compare", json={**body, "objective": "balanced"})
    assert r.status_code == 200
    j = r.json()
    assert len(j["options"]) == 15                       # concrete roof accepts all 5 methods x 3 qualities
    assert j["options"][0]["rank"] == 1 and abs(sum(j["weights"].values()) - 1) < 1e-6
    assert sum(o["is_current_selection"] for o in j["options"]) == 1
    cheapest = client.post("/api/compare", json={**body, "objective": "cost"}).json()
    assert cheapest["recommended"]["predictions"]["cost"] == min(o["predictions"]["cost"] for o in cheapest["options"])
    best_drop = client.post("/api/compare", json={**body, "objective": "temperature_drop"}).json()
    assert best_drop["recommended"]["predictions"]["temperature_drop"] == max(o["predictions"]["temperature_drop"] for o in best_drop["options"])


def test_compare_only_returns_supported_methods_for_roof(client, building):
    body = {"building": {k: v for k, v in building.items() if k not in ("cool_roofing_method", "cool_roofing_quality")}}
    body["building"]["roofing_type"] = "sheet"
    methods = {o["cool_roofing_method"] for o in client.post("/api/compare", json=body).json()["options"]}
    assert methods == {"basic_coating", "sheet", "sheet_with_coating"}


def test_compare_custom_weights_change_recommendation(client, building):
    base = {"building": {k: v for k, v in building.items() if k not in ("cool_roofing_method", "cool_roofing_quality")}, "objective": "balanced"}
    cheap = client.post("/api/compare", json={**base, "weights": {"cost": 1.0}}).json()["recommended"]
    cool = client.post("/api/compare", json={**base, "weights": {"temperature_drop": 1.0}}).json()["recommended"]
    assert cheap["predictions"]["cost"] < cool["predictions"]["cost"]
    assert client.post("/api/compare", json={**base, "weights": {"cost": 0}}).status_code == 422


def test_explain_is_additive(client, building):
    for task in ("cost", "units_saved", "referral"):
        r = client.post("/api/explain", json={"building": building, "task": task})
        assert r.status_code == 200
        j = r.json()
        total = j["base_value"] + sum(c["contribution"] for c in j["contributions"])
        assert total == pytest.approx(j["prediction"], rel=1e-6, abs=1e-6)
        assert "caveat" in j
    assert client.post("/api/explain", json={"building": building, "task": "nonsense"}).status_code == 422


def test_economics_endpoint(client):
    ok = client.post("/api/economics", json={"cost_inr": 120000, "monthly_kwh_saved": 100, "tariff_inr_per_kwh": 10})
    assert ok.status_code == 200 and ok.json()["central"]["simple_payback_years"] == pytest.approx(10)
    zero = client.post("/api/economics", json={"cost_inr": 120000, "monthly_kwh_saved": 0, "tariff_inr_per_kwh": 10}).json()
    assert zero["central"]["simple_payback_years"] is None and zero["warnings"]
    assert client.post("/api/economics", json={"cost_inr": 120000, "monthly_kwh_saved": 100}).status_code == 422   # no tariff -> no invented default
    assert client.post("/api/economics", json={"cost_inr": 120000, "monthly_kwh_saved": 100, "tariff_inr_per_kwh": 0}).status_code == 422


def test_proposal_pdf(client, building):
    r = client.post("/api/proposal/pdf", json={"building": building, "customer_name": "A & B <Test>", "tariff_inr_per_kwh": 9})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")


def test_reports_and_figures_are_whitelisted(client):
    assert client.get("/api/reports/model_evaluation").status_code == 200
    traversal = client.get("/api/reports/../../pyproject")
    assert "[build-system]" not in traversal.text and "coolroof" not in traversal.text.lower().replace("cool roofing", "")   # never the file
    assert client.get("/api/reports/secrets").status_code == 404
    assert client.get("/api/figures/01_target_distributions.png").headers["content-type"] == "image/png"
    assert client.get("/api/figures/..%2f..%2fpyproject.toml").status_code == 404
    assert client.get("/api/figures/notes.txt").status_code == 404


def test_data_quality_and_insights_endpoints(client):
    dq = client.get("/api/data-quality").json()
    assert dq["summary"]["n_raw_rows"] == 2500 and dq["profile"]["targets"]["referral"]["true"] > 0
    ins = client.get("/api/insights").json()
    assert ins["feature_importance"] and ins["marginal_effects"]["method"]["cost"]


def test_models_endpoint(client):
    m = client.get("/api/models").json()
    assert set(m["tasks"]) == set(C.TASK_ORDER) and m["candidates"]["cost"]


def test_models_missing_returns_503_without_internals(monkeypatch):
    from fastapi.testclient import TestClient
    from coolroof.api import main
    from coolroof.service import ModelNotReady

    def boom():
        raise ModelNotReady("Models have not been trained yet. Run `python -m coolroof all`.")

    monkeypatch.setattr(main, "get_service", boom)
    c = TestClient(main.app)
    r = c.post("/api/predict", json={"building": {"location_type": "urban", "building_size": 1000, "roofing_type": "tiles",
                                                  "cool_roofing_quality": "basic", "cool_roofing_method": "tiles"}})
    assert r.status_code == 503 and "trained" in r.json()["detail"] and "Traceback" not in r.text
    h = c.get("/api/health").json()
    assert h["status"] == "degraded" and h["models_loaded"] is False


def test_corrupted_model_file_is_handled(tmp_path, service):
    import shutil

    from coolroof.service import ModelNotReady, ModelService
    for f in C.MODELS.glob("*"):
        if f.is_file():
            shutil.copy(f, tmp_path / f.name)
    (tmp_path / "cost.joblib").write_bytes(b"not a model")
    with pytest.raises(ModelNotReady):
        ModelService(tmp_path)
    with pytest.raises(ModelNotReady):
        ModelService(tmp_path / "nowhere")


def test_unexpected_server_error_is_generic(monkeypatch, client, building):
    from coolroof.api import main

    class Broken:
        def predict(self, *a, **k):
            raise RuntimeError("secret path D:/internal/stuff")

    monkeypatch.setattr(main, "get_service", lambda: Broken())
    from fastapi.testclient import TestClient
    r = TestClient(main.app, raise_server_exceptions=False).post("/api/predict", json={"building": building})
    assert r.status_code == 500 and "secret" not in r.text and "internal" not in r.text.lower().replace("internal error", "")


def test_requests_are_not_logged_with_customer_data(client, building, caplog):
    caplog.set_level("DEBUG")
    client.post("/api/predict", json={"building": {**building, "building_size": 1234.5}})
    assert "1234.5" not in caplog.text

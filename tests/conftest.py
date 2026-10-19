from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coolroof import config as C  # noqa: E402
from coolroof.data import clean  # noqa: E402

GOOD_BUILDING = dict(location_type="urban", building_size=1500, roofing_type="concrete_flat",
                     cool_roofing_quality="high", cool_roofing_method="sheet_with_coating",
                     building_age=20, avg_roof_temperature_before=34.0, num_ac_units=2, avg_units_before=140)


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return clean.load_raw()


@pytest.fixture(scope="session")
def clean_res(raw_df):
    return clean.clean(raw_df)


@pytest.fixture(scope="session")
def service():
    from coolroof.service import ModelService
    try:
        return ModelService()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"trained models not available: {exc}. Run `python -m coolroof all` first.")


@pytest.fixture(scope="session")
def client(service):
    from fastapi.testclient import TestClient
    from coolroof.api.main import app
    return TestClient(app)


@pytest.fixture()
def building() -> dict:
    return dict(GOOD_BUILDING)


def tiny_frame(**overrides) -> pd.DataFrame:
    """One valid synthetic record, with optional overrides, for rule-level tests."""
    base = {"building_id": 1, "location_type": "urban", "building_size": 1000.0, "building_age": 10.0,
            "cool_roofing_quality": "basic", "roofing_type": "concrete_flat", "cool_roofing_method": "basic_coating",
            "cool_roofing_cost": 600000, "avg_roof_temperature_before": 34.0, "num_ac_units": 1.0, "avg_units_before": 80,
            "avg_roof_temperature_after": 29.0, "avg_units_after": 50, "customer_given_rating": 4, "customer_referral": True}
    base.update(overrides)
    return pd.DataFrame([base])


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

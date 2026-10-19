"""Request/response models for the HTTP API.

`extra="forbid"` is deliberate: a client cannot send post-installation values
(cost, rating, after-measurements...) - they are rejected, not silently ignored.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .. import config as C

Location = Literal["rural", "intermediate", "urban", "dense_urban"]
RoofType = Literal["concrete_flat", "concrete_slanted", "sheet", "tiles"]
Quality = Literal["basic", "intermediate", "high"]
Method = Literal["basic_coating", "sheet", "sheet_with_coating", "tiles", "tiles_with_coating"]
TaskKey = Literal["cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral"]
Objective = Literal["cost", "temperature_drop", "units_saved", "high_rating", "referral", "balanced"]

B = C.API_BOUNDS


class BuildingBase(BaseModel):
    """What a sales rep can learn about the building before installation."""
    model_config = ConfigDict(extra="forbid")

    location_type: Location = Field(description="Urbanisation level of the site")
    building_size: float = Field(ge=B["building_size"][0], le=B["building_size"][1], description="Floor area (dataset unit)")
    roofing_type: RoofType = Field(description="Existing roof construction")
    building_age: Optional[float] = Field(None, ge=B["building_age"][0], le=B["building_age"][1], description="Years")
    avg_roof_temperature_before: Optional[float] = Field(None, ge=B["avg_roof_temperature_before"][0],
                                                         le=B["avg_roof_temperature_before"][1], description="°C, average roof surface temperature")
    num_ac_units: Optional[int] = Field(None, ge=B["num_ac_units"][0], le=B["num_ac_units"][1])
    avg_units_before: Optional[float] = Field(None, ge=B["avg_units_before"][0], le=B["avg_units_before"][1], description="Average monthly kWh now")

    def as_features(self) -> dict:
        return self.model_dump()


class BuildingInput(BuildingBase):
    cool_roofing_quality: Quality
    cool_roofing_method: Method


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    building: BuildingInput
    interval_level: Literal[0.8, 0.9] = 0.8
    referral_cutoff: Optional[float] = Field(None, gt=0, lt=1, description="Override the F1-optimal referral probability cut-off")
    high_rating_cutoff: float = Field(0.5, gt=0, lt=1)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[BuildingInput] = Field(min_length=1, max_length=500)
    interval_level: Literal[0.8, 0.9] = 0.8


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    building: BuildingBase
    current_method: Optional[Method] = None
    current_quality: Optional[Quality] = None
    objective: Objective = "balanced"
    weights: Optional[dict[Literal["cost", "temperature_drop", "units_saved", "high_rating", "referral"], float]] = Field(
        None, description="Only used for the 'balanced' objective; each in [0, 1], re-scaled to sum to 1")
    interval_level: Literal[0.8, 0.9] = 0.8


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    building: BuildingInput
    task: TaskKey


class EconomicsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cost_inr: float = Field(gt=0, le=1e9)
    monthly_kwh_saved: float = Field(ge=-5000, le=5000)
    tariff_inr_per_kwh: float = Field(gt=0, le=100, description="Supplied by the user - never assumed")
    annual_maintenance_inr: float = Field(0, ge=0, le=1e8)
    tariff_escalation_pct: float = Field(0, ge=-20, le=30)
    horizon_years: int = Field(10, ge=1, le=30)
    monthly_kwh_saved_low: Optional[float] = Field(None, ge=-5000, le=5000)
    monthly_kwh_saved_high: Optional[float] = Field(None, ge=-5000, le=5000)


class ProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    building: BuildingInput
    customer_name: Optional[str] = Field(None, max_length=80)
    prepared_by: Optional[str] = Field(None, max_length=80)
    tariff_inr_per_kwh: Optional[float] = Field(None, gt=0, le=100)
    interval_level: Literal[0.8, 0.9] = 0.8


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[ChatMessage] = Field(min_length=1, max_length=30)

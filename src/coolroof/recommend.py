"""Roofing-option comparison, transparent scoring and economics.

Nothing here knows about the ML models: it consumes plain per-option prediction dicts,
which keeps the logic unit-testable and the scoring auditable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# metric key -> (label, direction, unit)
METRICS = {
    "cost": ("Installation cost", "min", "INR"),
    "temperature_drop": ("Roof temperature drop", "max", "°C"),
    "units_saved": ("Monthly electricity saved", "max", "kWh/month"),
    "high_rating": ("Probability of a high rating", "max", "probability"),
    "referral": ("Referral probability", "max", "probability"),
}

# The satisfaction criterion uses P(high rating) – a probability – never the predicted star value.
BALANCED_WEIGHTS = {"cost": 0.25, "temperature_drop": 0.15, "units_saved": 0.25, "high_rating": 0.15, "referral": 0.20}

OBJECTIVES = {
    "cost": "Lowest estimated installation cost",
    "temperature_drop": "Greatest estimated temperature drop",
    "units_saved": "Greatest estimated electricity savings",
    "high_rating": "Best predicted customer satisfaction",
    "referral": "Best predicted referral likelihood",
    "balanced": "Balanced overall value (weighted)",
}
MIN_PAIR_SUPPORT = 30       # records needed before a roof-type × method pairing counts as 'supported'
LOW_COMBO_SUPPORT = 15      # method × quality × roof cells below this get a reliability warning


def valid_options(roofing_type: str, pair_support: list[dict], methods: list[str], qualities: list[str],
                  combo_support: dict | None = None) -> list[dict]:
    """Method × quality combinations that the data supports for this existing roof type."""
    sup = {(r["roofing_type"], r["cool_roofing_method"]): r["n"] for r in pair_support}
    out = []
    for m in methods:
        if sup.get((roofing_type, m), 0) < MIN_PAIR_SUPPORT:
            continue
        for q in qualities:
            n = (combo_support or {}).get((roofing_type, m, q))
            out.append({"cool_roofing_method": m, "cool_roofing_quality": q, "support": n,
                        "low_support": (n is not None and n < LOW_COMBO_SUPPORT)})
    return out


def _normalise(values: list[float], direction: str) -> list[float]:
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.5] * len(values)            # no spread → no discrimination between options
    return [(v - lo) / (hi - lo) if direction == "max" else (hi - v) / (hi - lo) for v in values]


def score_options(options: list[dict], weights: dict[str, float] | None = None) -> list[dict]:
    """Adds 'score' (0–100) and per-metric 'contributions'.

    Each metric is min–max normalised *across the options being compared* to [0, 1]
    (direction-aware: cheaper = better, larger drop = better, …) and combined with the
    weights (re-scaled to sum to 1). The score is therefore **relative to this option set**.
    """
    weights = {k: float(v) for k, v in (weights or BALANCED_WEIGHTS).items() if k in METRICS and float(v) > 0}
    if not weights:
        raise ValueError("at least one metric weight must be positive")
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    norm = {}
    for k in weights:
        vals = [o["predictions"][k] for o in options]
        norm[k] = _normalise(vals, METRICS[k][1])
    for i, o in enumerate(options):
        contrib = {k: weights[k] * norm[k][i] for k in weights}
        o["normalised"] = {k: norm[k][i] for k in weights}
        o["contributions"] = contrib
        o["score"] = 100 * sum(contrib.values())
    return options


def recommend(options: list[dict], objective: str = "balanced", weights: dict[str, float] | None = None,
              model_mae: dict[str, float] | None = None) -> dict:
    """Pick the best option for an objective and explain the basis."""
    if not options:
        raise ValueError("no valid options to compare")
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective '{objective}'")
    if objective == "balanced":
        score_options(options, weights)
        ranked = sorted(options, key=lambda o: -o["score"])
        basis = "Weighted sum of min–max normalised metrics across the compared options."
        top, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
        margin = top["score"] - second["score"] if second else None
        close = margin is not None and margin < 3.0
    else:
        direction = METRICS[objective][1]
        ranked = sorted(options, key=lambda o: o["predictions"][objective], reverse=(direction == "max"))
        score_options(options, {objective: 1.0})
        basis = f"{OBJECTIVES[objective]} (single-metric ranking of model predictions)."
        top, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
        margin, close = None, False
        if second is not None and model_mae and objective in model_mae:
            diff = abs(top["predictions"][objective] - second["predictions"][objective])
            close = diff < 0.5 * model_mae[objective]
    notes = []
    if close:
        notes.append("The top two options differ by less than the model's typical error – treat them as equivalent.")
    if top.get("low_support"):
        notes.append("This combination has few historical examples, so its prediction is less reliable.")
    for i, o in enumerate(ranked):
        o["rank"] = i + 1
    return {"objective": objective, "objective_label": OBJECTIVES[objective], "basis": basis,
            "weights": ({k: round(v, 4) for k, v in (_norm_w(weights or BALANCED_WEIGHTS)).items()} if objective == "balanced" else {objective: 1.0}),
            "recommended": top, "ranking": ranked, "score_margin": margin, "close_call": close, "notes": notes}


def _norm_w(weights: dict[str, float]) -> dict[str, float]:
    w = {k: float(v) for k, v in weights.items() if k in METRICS and float(v) > 0}
    t = sum(w.values())
    return {k: v / t for k, v in w.items()}


# --------------------------------------------------------------------------
# Economics
# --------------------------------------------------------------------------
@dataclass
class EconomicsInput:
    cost_inr: float
    monthly_kwh_saved: float
    tariff_inr_per_kwh: float
    annual_maintenance_inr: float = 0.0
    tariff_escalation_pct: float = 0.0     # per year
    horizon_years: int = 10


def economics(e: EconomicsInput, kwh_low: float | None = None, kwh_high: float | None = None) -> dict:
    """Simple, assumption-based payback. Never a guarantee.

    annual bill saving = monthly kWh saved × tariff × 12  (constant all year, as stated)
    simple payback     = installation cost / (annual bill saving − annual maintenance)
    """
    if e.tariff_inr_per_kwh is None or e.tariff_inr_per_kwh <= 0:
        raise ValueError("tariff must be a positive INR/kWh value supplied by the user")
    if e.cost_inr <= 0:
        raise ValueError("cost must be positive")

    def one(kwh: float) -> dict:
        annual_gross = kwh * e.tariff_inr_per_kwh * 12
        annual_net = annual_gross - e.annual_maintenance_inr
        payback = e.cost_inr / annual_net if annual_net > 0 else None
        cum, pay_escal, t = 0.0, None, 0
        for yr in range(1, e.horizon_years + 1):
            cum += annual_net * ((1 + e.tariff_escalation_pct / 100) ** (yr - 1)) if annual_net > 0 else annual_net
            if pay_escal is None and cum >= e.cost_inr:
                pay_escal = yr - (cum - e.cost_inr) / (annual_net * ((1 + e.tariff_escalation_pct / 100) ** (yr - 1)))
            t = yr
        # tariff that would make the cost pay back exactly over the horizon (no escalation) - a tariff-free reference point
        breakeven = ((e.cost_inr / e.horizon_years + e.annual_maintenance_inr) / (kwh * 12)) if kwh > 0 else None
        return {"monthly_kwh_saved": kwh, "annual_bill_saving_inr": annual_gross, "annual_net_saving_inr": annual_net,
                "breakeven_tariff_inr_per_kwh": breakeven,
                "simple_payback_years": payback, "payback_with_escalation_years": pay_escal,
                "net_position_after_horizon_inr": cum - e.cost_inr}

    central = one(e.monthly_kwh_saved)
    out = {"inputs": e.__dict__.copy(), "central": central, "range": None, "assumptions": [
        "Estimate, not a guarantee: monthly kWh saved is a model prediction with the stated uncertainty.",
        "Savings are assumed constant across all 12 months and years (no seasonality or degradation).",
        "Tariff is the user-supplied INR/kWh figure; future tariff changes only enter via the escalation input.",
        f"Maintenance cost (user-supplied): INR {e.annual_maintenance_inr:,.0f}/year.",
        "Costs and savings are not discounted (no time value of money).",
    ], "warnings": []}
    if central["annual_net_saving_inr"] <= 0:
        out["warnings"].append("Estimated net annual saving is zero or negative, so the installation does not pay back on electricity savings alone.")
    if kwh_low is not None and kwh_high is not None:
        lo, hi = one(max(kwh_low, 0.0)), one(kwh_high)
        out["range"] = {"pessimistic": lo, "optimistic": hi}
        if lo["simple_payback_years"] is None:
            out["warnings"].append("In the pessimistic case the savings never pay back the cost.")
    return out

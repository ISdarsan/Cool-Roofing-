"""Built-in (free, offline, no API key) sales assistant.

It understands plain-language descriptions of a building with a fixed set of patterns, asks for whatever required
facts are missing, and then calls exactly the same validated tools as the LLM assistants. Every number in its reply is
copied from a tool result - it cannot calculate or invent predictions.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from .. import recommend as R

WORDS = {"no": 0, "zero": 0, "none": 0, "one": 1, "a single": 1, "single": 1, "two": 2, "three": 3}
LABEL_M = {"basic_coating": "Basic coating", "sheet": "Sheet", "sheet_with_coating": "Sheet + coating",
           "tiles": "Tiles", "tiles_with_coating": "Tiles + coating"}
LABEL_LOC = {"rural": "rural", "intermediate": "intermediate-town", "urban": "urban", "dense_urban": "dense-urban"}
LABEL_ROOF = {"concrete_flat": "flat concrete", "concrete_slanted": "slanted concrete", "sheet": "sheet", "tiles": "tiled"}
OBJ = [
    ("cost", r"cheap|lowest cost|least expensive|budget|lowest price|minimi[sz]e cost"),
    ("temperature_drop", r"cool(est|ing)|temperature|coolest|heat"),
    ("units_saved", r"saving|save (the )?most|electricity|bill|kwh saved"),
    ("high_rating", r"satisf|rating|happy|happiest|review"),
    ("referral", r"referral|refer\b|recommend us|word of mouth"),
]
OBJ_LABEL = {"cost": "lowest cost", "temperature_drop": "greatest cooling", "units_saved": "greatest electricity savings",
             "high_rating": "best predicted satisfaction", "referral": "best referral likelihood", "balanced": "balanced value"}

INTRO = ("Hello! I can estimate cost, roof cooling, monthly electricity savings, rating and referral likelihood, and compare roofing options. "
         "Tell me about the building, for example: \"Urban house, flat concrete roof, 1500 sq ft, 2 ACs, about 140 kWh a month.\" "
         "I need at least the location type (rural, intermediate, urban, dense urban), the floor area and the existing roof type "
         "(flat/slanted concrete, sheet or tiles). Optional but helpful: building age, current roof temperature, number of ACs and monthly kWh.")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse(text: str) -> dict[str, Any]:
    """Extract building facts from one message. Returns only what was actually found."""
    t = text.lower()
    out: dict[str, Any] = {}

    if re.search(r"dense[\s_-]*urban|metro\b|downtown|city cent", t):
        out["location_type"] = "dense_urban"
    elif re.search(r"semi[\s-]*urban|intermediate|small town|\btown\b", t):
        out["location_type"] = "intermediate"
    elif re.search(r"\brural\b|village|countryside", t):
        out["location_type"] = "rural"
    elif re.search(r"\burban\b|\bcity\b", t):
        out["location_type"] = "urban"

    if re.search(r"concrete|rcc|slab", t):
        out["roofing_type"] = "concrete_slanted" if re.search(r"slant|slop", t) else "concrete_flat"
    elif re.search(r"tile[sd]?\s+(roof|house|building|terrace)|roof\s+(is\s+|of\s+|made of\s+)?tiles?\b|tiled|mangalore|clay tile", t):
        out["roofing_type"] = "tiles"
    elif re.search(r"sheet\s+(roof|house|building|shed)|roof\s+(is\s+|of\s+|made of\s+)?sheet|tin\b|asbestos|corrugated|metal roof|\bgi\b", t):
        out["roofing_type"] = "sheet"

    m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|square\s*(?:feet|foot|ft)|sq\.?\s*m\b|sqm|area\s*units?|units?\s*of\s*area)", t) \
        or re.search(r"(?:floor\s*area|area|size)\s*(?:of|is|:|=)?\s*(\d[\d,]*(?:\.\d+)?)", t)
    if m:
        out["building_size"] = _num(m.group(1))

    m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(?:kwh|units?\s*(?:a|per|/|each)\s*month|units?\s*monthly)", t) \
        or re.search(r"(?:consum\w*|uses?|bills?|usage)\s*(?:is|of|about|around|roughly)?\s*(\d[\d,]*)\s*(?:units|kwh)", t)
    if m:
        out["avg_units_before"] = _num(m.group(1))

    if re.search(r"\b(?:no|without|zero)\s+(?:ac|a/c|acs|air[\s-]*condition\w*)", t):
        out["num_ac_units"] = 0
    else:
        m = re.search(r"(\d+|one|two|three|a single|single)\s*(?:ac\b|acs\b|a/c|air[\s-]*condition\w*)", t)
        if m:
            out["num_ac_units"] = int(m.group(1)) if m.group(1).isdigit() else WORDS[m.group(1)]

    m = re.search(r"(\d+)\s*(?:years?|yrs?)[\s-]*old", t) or re.search(r"(?:built|age|aged)\D{0,14}(\d+)\s*(?:years?|yrs?)", t)
    if m:
        out["building_age"] = float(m.group(1))

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:°\s*c|degrees?\s*c?|deg\b)", t)
    if m and 10 <= float(m.group(1)) <= 80:
        out["avg_roof_temperature_before"] = float(m.group(1))

    m = re.search(r"(?:₹|rs\.?|inr)\s*(\d+(?:\.\d+)?)\s*(?:/|per)\s*(?:kwh|unit)", t) or re.search(r"tariff\D{0,20}?(\d+(?:\.\d+)?)", t)
    if m:
        out["_tariff"] = float(m.group(1))

    if re.search(r"(sheet|tiles?)\s*(\+|with|and|plus)\s*coating", t):
        out["_method"] = "sheet_with_coating" if re.search(r"sheet\s*(\+|with|and|plus)\s*coating", t) else "tiles_with_coating"
    elif re.search(r"basic coating|coating only|just (a )?coating", t):
        out["_method"] = "basic_coating"
    elif re.search(r"(?:use|with|install|go for|option)\s+(?:the\s+)?(sheet|tiles?)\b(?!\s*(?:roof|house))", t):
        out["_method"] = "sheet" if "sheet" in re.search(r"(sheet|tiles?)", t).group(1) else "tiles"
    m = re.search(r"\b(basic|intermediate|high|premium|top)\s*(?:quality|grade)", t)
    if m:
        out["_quality"] = {"premium": "high", "top": "high"}.get(m.group(1), m.group(1))

    for key, pat in OBJ:
        if re.search(pat, t):
            out["_objective"] = key
            break
    if re.search(r"\bwhy\b|explain|reason", t):
        out["_explain"] = True
    return out


def _inr(v: float) -> str:
    return "₹" + f"{v:,.0f}"


def _inr_in(v: float) -> str:
    s = f"{int(round(v))}"
    if len(s) <= 3:
        return "₹" + s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return "₹" + ",".join(parts + [tail])


def _opt_line(o: dict) -> str:
    return (f"{LABEL_M[o['method']]}, {o['quality']} quality: {_inr_in(o['cost'])}, cooling {o['temperature_drop']:.1f} °C, "
            f"{o['units_saved']:.0f} kWh/month, {o['high_rating'] * 100:.0f}% chance of a high rating, {o['referral'] * 100:.0f}% referral chance")


def respond(messages: list[dict], tool_runner: Callable[[str, dict], dict]) -> dict:
    """One assistant turn. Facts accumulate across the whole conversation; later messages override earlier ones."""
    users = [m["content"] for m in messages if m["role"] == "user"]
    facts: dict[str, Any] = {}
    for u in users:
        facts.update(parse(u))
    last = parse(users[-1]) if users else {}

    building = {k: v for k, v in facts.items() if not k.startswith("_")}
    required = {"location_type": "the location type (rural, intermediate, urban or dense urban)",
                "building_size": "the floor area (e.g. 1500 sq ft)",
                "roofing_type": "the existing roof type (flat concrete, slanted concrete, sheet or tiles)"}
    missing = [d for k, d in required.items() if k not in building]
    if not building:
        return {"reply": INTRO, "tool_results": []}
    if missing:
        got = ", ".join(f"{k.replace('_', ' ')} = {v}" for k, v in building.items())
        return {"reply": f"Thanks - so far I have: {got}. To give you real estimates I still need " + " and ".join(missing) + ".", "tool_results": []}

    trace: list[dict] = []

    def call(name: str, args: dict) -> dict:
        out = tool_runner(name, dict(args))
        trace.append({"tool": name, "input": dict(args), "output": out})
        return out

    objective = last.get("_objective") or facts.get("_objective") or "balanced"
    cmp_ = call("compare_options", {**building, "objective": objective})
    if "error" in cmp_:
        return {"reply": f"I couldn't compare the options: {cmp_['error']} Please correct that and try again.", "tool_results": trace}

    lines = [f"For a {LABEL_LOC[building['location_type']]} building of {building['building_size']:g} area units with a {LABEL_ROOF[building['roofing_type']]} roof"
             + (f", {building['num_ac_units']} AC unit(s)" if "num_ac_units" in building else "")
             + (f" and about {building['avg_units_before']:g} kWh a month" if "avg_units_before" in building else "") + ", the prediction service says:", ""]
    rec = cmp_["recommended"]
    if re.search(r"durab|lifespan|life span|last long|warranty|maintenance|reappl|waterproof|leak", users[-1].lower()):
        lines.insert(1, "Note: the historical data has no information about durability, lifespan, warranty or maintenance, so I can't compare options on that. "
                        "What it does support is cost, cooling, electricity savings, satisfaction and referral, shown below.")
    lines.append(f"Recommended for {OBJ_LABEL.get(objective, objective)}: {_opt_line(rec)}.")
    others = [o for o in cmp_["options"] if o["rank"] != rec["rank"]][:2]
    for o in others:
        lines.append(f"  • Next best: {_opt_line(o)}.")

    chosen = None
    if facts.get("_method"):
        q = facts.get("_quality") or rec["quality"]
        pred = call("predict_outcomes", {**building, "cool_roofing_method": facts["_method"], "cool_roofing_quality": q})
        if "error" in pred:
            lines += ["", f"About your requested option: {pred['error']}"]
        else:
            p = pred["predictions"]
            chosen = p
            lines += ["", f"For {LABEL_M[facts['_method']]}, {q} quality specifically: {_inr_in(p['cost']['value'])} "
                          f"(likely {_inr_in(p['cost']['interval']['low'])} - {_inr_in(p['cost']['interval']['high'])}), cooling {p['temperature_drop']['value']:.1f} °C, "
                          f"{p['units_saved']['value']:.0f} kWh/month saved, expected rating {p['rating']['value']:.1f}/5, {p['high_rating']['probability'] * 100:.0f}% chance of a high rating, "
                          f"{p['referral']['probability'] * 100:.0f}% referral chance."]

    if facts.get("_explain"):
        args = {**building, "cool_roofing_method": rec["method"], "cool_roofing_quality": rec["quality"], "task": "cost"}
        ex = call("explain_prediction", args)
        if "error" not in ex:
            top = ex["contributions"][:3]
            lines += ["", "Biggest drivers of the cost estimate for the recommended option: "
                      + "; ".join(f"{c['feature'].replace('_', ' ')} ({'+' if c['contribution'] >= 0 else '-'}{_inr_in(abs(c['contribution']))})" for c in top) + "."]

    if facts.get("_tariff"):
        cost = chosen["cost"]["value"] if chosen else rec["cost"]
        kwh = chosen["units_saved"]["value"] if chosen else rec["units_saved"]
        pay = call("calculate_payback", {"cost_inr": cost, "monthly_kwh_saved": kwh, "tariff_inr_per_kwh": facts["_tariff"]})
        if "error" in pay:
            lines += ["", f"Payback: {pay['error']}"]
        else:
            c = pay["central"]
            pb = f"{c['simple_payback_years']:.1f} years" if c["simple_payback_years"] else "no payback from electricity savings alone"
            lines += ["", f"At ₹{facts['_tariff']:g}/kWh (your figure) the simple payback is {pb}; an assumption-based estimate, not a guarantee."]

    notes = list(cmp_.get("notes", [])) + list(cmp_.get("warnings", []))
    if notes:
        lines += [""] + [f"Note: {n}" for n in notes[:3]]
    lines += ["", "These are statistical estimates from past installations, not guarantees. 'High rating' means 4 stars or more (an assumption). "
              "Ask me to optimise for cost, cooling, savings, satisfaction or referrals, to explain why, or give an electricity tariff for payback."]
    return {"reply": "\n".join(lines), "tool_results": trace}

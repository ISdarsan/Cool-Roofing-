"""Sales assistant with pluggable providers.

Providers (COOLROOF_LLM_PROVIDER, default "auto"):
  builtin        free, offline, no key - pattern-based understanding (see builtin.py)
  gemini         Google Gemini free tier via its OpenAI-compatible endpoint  (GEMINI_API_KEY)
  groq           Groq free tier                                              (GROQ_API_KEY)
  openrouter     OpenRouter (has free models)                                (OPENROUTER_API_KEY)
  ollama         local models, no key, no cost                               (COOLROOF_LLM_BASE_URL optional)
  openai_compat  any OpenAI-compatible server  (COOLROOF_LLM_BASE_URL + COOLROOF_LLM_API_KEY)
  anthropic      Claude                                                      (ANTHROPIC_API_KEY)
  off            disable the assistant
"auto" picks the first configured provider (anthropic > gemini > groq > openrouter > openai_compat > builtin) and falls
back to the built-in assistant if that provider fails, so the page keeps working.

Design rules (all providers): the model never produces official numbers; every figure comes from a tool call into the
real prediction service and tool arguments are validated with the same pydantic schemas as the REST API; no code
execution; keys are read from the environment only and never logged or returned.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from pydantic import ValidationError

from .. import recommend as R
from ..api import schemas as S
from ..service import InvalidCombination, ModelNotReady, get_service
from . import builtin

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOOL_ROUNDS = 6

# base_url, env var holding the key, default model (model names change over time: override with COOLROOF_LLM_MODEL)
PRESETS: dict[str, dict] = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "key_env": "GEMINI_API_KEY", "model": "gemini-flash-lite-latest", "free": True},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile", "free": True},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY", "model": "meta-llama/llama-3.3-70b-instruct:free", "free": True},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None, "model": "llama3.1", "free": True},
}

SYSTEM_PROMPT = """You are the sales assistant for a cool-roofing company in India. You help a sales representative \
talk to a customer.

Rules you must follow:
1. You NEVER calculate, estimate or invent predictions yourself. All cost, temperature, electricity-saving, rating and \
referral figures MUST come from the tools. Quote the tool values faithfully. You may round for readability only: rupees to whole numbers \
with Indian digit grouping (e.g. ₹16,60,593), temperatures to one decimal, kWh to whole numbers, probabilities as whole percentages. \
Never change a value or round in a way that flatters the result.
2. Collect these required inputs from the user (ask for any that are missing, one short question at a time): \
location_type (rural, intermediate, urban, dense_urban), building_size (floor area number), roofing_type (concrete_flat, \
concrete_slanted, sheet, tiles). Optional but valuable: building_age, avg_roof_temperature_before (°C), num_ac_units, \
avg_units_before (monthly kWh). Never invent measurements the customer has not given you.
3. To compare roofing options call compare_options. To get predictions for one specific method and quality call predict_outcomes. \
Use explain_prediction when asked why.
4. Payback only if the customer gives an electricity tariff in INR/kWh; call calculate_payback. Never assume a tariff.
5. Never promise or guarantee savings, payback or satisfaction. Say they are estimates with uncertainty and mention any \
warnings returned by the tools.
6. 'High rating' means a customer rating of 4 or more (an assumption); referral and rating are weakly predictable, say so.
7. The historical data contains ONLY: cost, roof temperature before/after, monthly electricity before/after, customer rating and referral. \
It has NO information about durability, lifespan, warranty, maintenance, reapplication, waterproofing or any other property. If asked about \
those, say clearly that this data cannot answer it, do not state anything about them from general knowledge, and offer what the data \
does support (cost, cooling, electricity savings, predicted satisfaction, referral).
8. Be concise and friendly: short paragraphs or a short bullet list, a recommendation first, then the key figures. \
If a tool returns an error, explain it plainly and ask for corrected input."""

BUILDING_PROPS = {
    "location_type": {"type": "string", "enum": ["rural", "intermediate", "urban", "dense_urban"]},
    "building_size": {"type": "number", "description": "floor area"},
    "roofing_type": {"type": "string", "enum": ["concrete_flat", "concrete_slanted", "sheet", "tiles"]},
    "building_age": {"type": "number", "description": "years (optional)"},
    "avg_roof_temperature_before": {"type": "number", "description": "°C (optional)"},
    "num_ac_units": {"type": "integer", "description": "optional"},
    "avg_units_before": {"type": "number", "description": "monthly kWh now (optional)"},
}
_METHOD = {"type": "string", "enum": ["basic_coating", "sheet", "sheet_with_coating", "tiles", "tiles_with_coating"]}
_QUALITY = {"type": "string", "enum": ["basic", "intermediate", "high"]}
TOOLS = [
    {"name": "predict_outcomes",
     "description": "Predict cost, roof temperature drop, monthly kWh saved, rating, high-rating probability and referral probability "
                    "for ONE specific roofing method and quality, using the trained models.",
     "input_schema": {"type": "object", "properties": {**BUILDING_PROPS, "cool_roofing_method": _METHOD, "cool_roofing_quality": _QUALITY},
                      "required": ["location_type", "building_size", "roofing_type", "cool_roofing_method", "cool_roofing_quality"]}},
    {"name": "compare_options",
     "description": "Compare every supported roofing method x quality for this building and rank them for an objective.",
     "input_schema": {"type": "object", "properties": {**BUILDING_PROPS,
                      "objective": {"type": "string", "enum": ["cost", "temperature_drop", "units_saved", "high_rating", "referral", "balanced"]}},
                      "required": ["location_type", "building_size", "roofing_type"]}},
    {"name": "explain_prediction",
     "description": "Explain which inputs pushed one prediction up or down (Shapley values) for a specific option.",
     "input_schema": {"type": "object", "properties": {**BUILDING_PROPS, "cool_roofing_method": _METHOD, "cool_roofing_quality": _QUALITY,
                      "task": {"type": "string", "enum": ["cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral"]}},
                      "required": ["location_type", "building_size", "roofing_type", "cool_roofing_method", "cool_roofing_quality", "task"]}},
    {"name": "calculate_payback",
     "description": "Simple payback from a user-supplied electricity tariff. Only call when the customer has given a tariff.",
     "input_schema": {"type": "object", "properties": {
         "cost_inr": {"type": "number"}, "monthly_kwh_saved": {"type": "number"},
         "tariff_inr_per_kwh": {"type": "number"}, "annual_maintenance_inr": {"type": "number"}},
         "required": ["cost_inr", "monthly_kwh_saved", "tariff_inr_per_kwh"]}},
    {"name": "get_input_schema", "description": "List allowed values and ranges for building inputs.",
     "input_schema": {"type": "object", "properties": {}}},
]


# --------------------------------------------------------------------------
# provider resolution / status
# --------------------------------------------------------------------------
def _key_for(provider: str) -> str | None:
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY") or None
    if provider == "openai_compat":
        return os.environ.get("COOLROOF_LLM_API_KEY") or None
    p = PRESETS[provider]
    return (os.environ.get("COOLROOF_LLM_API_KEY") or (os.environ.get(p["key_env"]) if p["key_env"] else "ollama")) or None


def resolve_provider() -> tuple[str, bool]:
    """Returns (provider, is_auto)."""
    chosen = (os.environ.get("COOLROOF_LLM_PROVIDER") or "auto").strip().lower()
    if chosen != "auto":
        return chosen, False
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", True
    for name in ("gemini", "groq", "openrouter"):
        if os.environ.get(PRESETS[name]["key_env"]):
            return name, True
    if os.environ.get("COOLROOF_LLM_BASE_URL") and os.environ.get("COOLROOF_LLM_API_KEY"):
        return "openai_compat", True
    return "builtin", True


def _model_for(provider: str) -> str | None:
    if provider == "builtin":
        return None
    env = os.environ.get("COOLROOF_LLM_MODEL")
    if provider in PRESETS:
        # a leftover Claude model id must never be sent to Gemini/Groq/...
        return env if env and not env.startswith("claude") else PRESETS[provider]["model"]
    return env or DEFAULT_MODEL


def llm_status() -> dict:
    provider, auto = resolve_provider()
    if provider == "off":
        return {"enabled": False, "provider": "off", "model": None, "free": False,
                "message": "The assistant is switched off (COOLROOF_LLM_PROVIDER=off). Remove that setting, or choose builtin, to turn it on."}
    if provider == "builtin":
        return {"enabled": True, "provider": "builtin", "model": None, "free": True, "message":
                "Built-in assistant: free, offline, no account needed. It understands plain-language building descriptions using fixed patterns. "
                "For free conversational AI add a Gemini, Groq or Ollama setting in .env (see .env.example)."}
    if provider not in PRESETS and provider not in ("anthropic", "openai_compat"):
        return {"enabled": False, "provider": provider, "model": None, "free": False, "message": f"Unknown COOLROOF_LLM_PROVIDER '{provider}'."}
    if provider == "anthropic":
        try:
            import anthropic  # noqa: F401
        except Exception:
            return {"enabled": False, "provider": provider, "model": None, "free": False, "message": "Install the Anthropic SDK (`pip install anthropic`) or choose another provider."}
    if provider == "openai_compat" and not os.environ.get("COOLROOF_LLM_BASE_URL"):
        return {"enabled": False, "provider": provider, "model": None, "free": False, "message": "Set COOLROOF_LLM_BASE_URL (and COOLROOF_LLM_API_KEY) for openai_compat."}
    if not _key_for(provider):
        env_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else PRESETS.get(provider, {}).get("key_env") or "COOLROOF_LLM_API_KEY"
        return {"enabled": False, "provider": provider, "model": None, "free": False, "message": f"Provider '{provider}' needs {env_name} in .env. Or use COOLROOF_LLM_PROVIDER=builtin (free, no key)."}
    return {"enabled": True, "provider": provider, "model": _model_for(provider), "free": bool(PRESETS.get(provider, {}).get("free")),
            "message": None if not auto else "Automatic mode: if this provider fails, the free built-in assistant answers instead."}


# --------------------------------------------------------------------------
# tool execution (validated; returns JSON-serialisable dicts, errors become {"error": ...})
# --------------------------------------------------------------------------
def run_tool(name: str, args: dict[str, Any]) -> dict:
    try:
        svc = get_service()
        if name == "get_input_schema":
            from ..features import input_schema
            return input_schema()
        if name == "predict_outcomes":
            b = S.BuildingInput(**args)
            return svc.predict(b.as_features())
        if name == "compare_options":
            objective = args.pop("objective", "balanced")
            b = S.BuildingBase(**args)
            rec = svc.compare(b.as_features(), objective=S.CompareRequest(building=b, objective=objective).objective)
            slim = [{"method": o["cool_roofing_method"], "quality": o["cool_roofing_quality"], "rank": o["rank"], "score": round(o["score"], 1),
                     **o["predictions"]} for o in rec["options"]]
            return {"objective": rec["objective_label"], "recommended": slim[0], "options": slim, "notes": rec["notes"],
                    "warnings": rec["warnings"], "normalisation_note": rec["normalisation_note"]}
        if name == "explain_prediction":
            task = args.pop("task")
            b = S.BuildingInput(**args)
            S.ExplainRequest(building=b, task=task)
            return svc.explain(b.as_features(), task)
        if name == "calculate_payback":
            e = S.EconomicsRequest(cost_inr=args["cost_inr"], monthly_kwh_saved=args["monthly_kwh_saved"],
                                   tariff_inr_per_kwh=args["tariff_inr_per_kwh"], annual_maintenance_inr=args.get("annual_maintenance_inr", 0))
            return R.economics(R.EconomicsInput(e.cost_inr, e.monthly_kwh_saved, e.tariff_inr_per_kwh, e.annual_maintenance_inr))
        return {"error": f"unknown tool '{name}'"}
    except ValidationError as exc:
        return {"error": "invalid input: " + "; ".join(f"{'.'.join(map(str, er['loc']))}: {er['msg']}" for er in exc.errors())}
    except (InvalidCombination, ValueError, KeyError) as exc:
        return {"error": str(exc)}
    except ModelNotReady as exc:
        return {"error": str(exc)}
    except Exception:
        return {"error": "the prediction service failed unexpectedly"}


# --------------------------------------------------------------------------
# provider loops
# --------------------------------------------------------------------------
def _chat_anthropic(messages: list[dict], client: Any, tool_runner: Callable) -> dict:
    model = _model_for("anthropic") or DEFAULT_MODEL
    convo = [{"role": m["role"], "content": m["content"]} for m in messages]
    tool_trace: list[dict] = []
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.messages.create(model=model, max_tokens=1200, system=SYSTEM_PROMPT, tools=TOOLS, messages=convo)
        blocks = list(resp.content)
        uses = [b for b in blocks if getattr(b, "type", "") == "tool_use"]
        if not uses:
            text = "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text").strip()
            return {"reply": text, "tool_results": tool_trace}
        convo.append({"role": "assistant", "content": [_block_dict(b) for b in blocks]})
        results = []
        for u in uses:
            out = tool_runner(u.name, dict(u.input))
            tool_trace.append({"tool": u.name, "input": dict(u.input), "output": out})
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(out, default=str)[:20000], "is_error": "error" in out})
        convo.append({"role": "user", "content": results})
    return {"reply": "I could not finish that request within the allowed number of steps. Please narrow it down.", "tool_results": tool_trace}


def _openai_tools() -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in TOOLS]


def _chat_openai_compat(messages: list[dict], provider: str, tool_runner: Callable, http_client: Any = None) -> dict:
    import httpx

    preset = PRESETS.get(provider, {})
    base = (os.environ.get("COOLROOF_LLM_BASE_URL") or preset.get("base_url", "")).rstrip("/")
    key = _key_for(provider)
    model = _model_for(provider) or preset.get("model")
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    convo: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}] + [{"role": m["role"], "content": m["content"]} for m in messages]
    tool_trace: list[dict] = []
    own = http_client is None
    client = http_client or httpx.Client(timeout=60)
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            r = client.post(f"{base}/chat/completions", headers=headers,
                            json={"model": model, "messages": convo, "tools": _openai_tools(), "tool_choice": "auto", "temperature": 0.2, "max_tokens": 1200})
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                return {"reply": (msg.get("content") or "").strip(), "tool_results": tool_trace}
            convo.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
            for c in calls:
                fn = c.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be an object")
                    out = tool_runner(fn.get("name", ""), dict(args))
                except (ValueError, TypeError) as exc:
                    args, out = {}, {"error": f"could not read tool arguments: {exc}"}
                tool_trace.append({"tool": fn.get("name", ""), "input": args, "output": out})
                convo.append({"role": "tool", "tool_call_id": c.get("id", ""), "name": fn.get("name", ""), "content": json.dumps(out, default=str)[:20000]})
        return {"reply": "I could not finish that request within the allowed number of steps. Please narrow it down.", "tool_results": tool_trace}
    finally:
        if own:
            client.close()


TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def is_transient(exc: Exception) -> bool:
    """Temporary provider problems that are worth one retry and then a graceful fallback."""
    name = type(exc).__name__
    if name == "HTTPStatusError":
        return getattr(getattr(exc, "response", None), "status_code", 0) in TRANSIENT_STATUS
    return name in ("ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError", "RateLimitError", "APIConnectionError",
                    "APITimeoutError", "InternalServerError", "ServiceUnavailableError", "OverloadedError")


def chat(messages: list[dict], client: Any | None = None, tool_runner: Callable[[str, dict], dict] = run_tool, http_client: Any | None = None) -> dict:
    """One assistant turn. `client` (Anthropic-style) and `http_client` (httpx) are injectable for tests."""
    if client is not None:
        return {**_chat_anthropic(messages, client, tool_runner), "provider": "anthropic"}
    st = llm_status()
    if not st["enabled"]:
        raise RuntimeError(st["message"])
    provider, auto = resolve_provider()
    if provider == "builtin":
        return {**builtin.respond(messages, tool_runner), "provider": "builtin"}
    def attempt() -> dict:
        if provider == "anthropic":
            import anthropic
            return {**_chat_anthropic(messages, anthropic.Anthropic(), tool_runner), "provider": "anthropic"}
        return {**_chat_openai_compat(messages, provider, tool_runner, http_client), "provider": provider}

    try:
        try:
            return attempt()
        except Exception as exc:
            if not is_transient(exc):
                raise
            time.sleep(float(os.environ.get("COOLROOF_LLM_RETRY_DELAY", "1.5")))     # one short retry for brief overloads
            return attempt()
    except Exception as exc:
        # Configuration problems (bad key, wrong model, no credit) are shown to the user when they chose the provider
        # explicitly. Temporary problems (rate limit, overload, timeout) and automatic mode fall back to the built-in
        # assistant so the page keeps working.
        if not auto and not is_transient(exc):
            raise
        out = builtin.respond(messages, tool_runner)
        why = "is busy or rate-limited right now" if is_transient(exc) else f"could not be used ({type(exc).__name__})"
        return {**out, "provider": "builtin", "notice": f"The {provider} provider {why}; this answer came from the free built-in assistant."}


def _block_dict(b) -> dict:
    t = getattr(b, "type", "")
    if t == "text":
        return {"type": "text", "text": b.text}
    if t == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)}
    return {"type": t}

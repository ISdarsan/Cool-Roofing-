"""The optional LLM assistant is tested with a scripted fake client: these tests verify the
grounding/validation machinery, not any real model's behaviour."""
from types import SimpleNamespace as NS

import pytest

from coolroof.llm import assistant


class FakeClient:
    """Scripted responses: first a tool call, then a final text turn."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return NS(content=self.script.pop(0))


def tool_use(name, inp, id_="t1"):
    return NS(type="tool_use", name=name, input=inp, id=id_)


def text(t):
    return NS(type="text", text=t)


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "COOLROOF_LLM_API_KEY",
              "COOLROOF_LLM_BASE_URL", "COOLROOF_LLM_PROVIDER", "COOLROOF_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)


def test_default_is_the_free_builtin_assistant_without_any_key():
    st = assistant.llm_status()
    assert st["enabled"] is True and st["provider"] == "builtin" and st["free"] is True


def test_provider_resolution(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert assistant.llm_status()["provider"] == "gemini" and assistant.llm_status()["free"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert assistant.llm_status()["provider"] == "anthropic"             # first configured wins in auto mode
    monkeypatch.setenv("COOLROOF_LLM_PROVIDER", "groq")
    st = assistant.llm_status()
    assert st["enabled"] is False and "GROQ_API_KEY" in st["message"]    # explicit choice without its key
    monkeypatch.setenv("COOLROOF_LLM_PROVIDER", "off")
    assert assistant.llm_status()["enabled"] is False


def test_leftover_claude_model_name_is_not_sent_to_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("COOLROOF_LLM_MODEL", "claude-sonnet-5-5")
    assert assistant.llm_status()["model"] == "gemini-flash-lite-latest"


def test_status_enabled_with_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    assert assistant.llm_status()["enabled"] is True


def test_tool_loop_calls_real_prediction_service(service, building):
    args = {k: v for k, v in building.items()}
    fake = FakeClient([[tool_use("predict_outcomes", args)], [text("Estimated cost is shown above.")]])
    out = assistant.chat([{"role": "user", "content": "quote please"}], client=fake)
    assert out["reply"].startswith("Estimated cost")
    trace = out["tool_results"][0]
    assert trace["tool"] == "predict_outcomes" and trace["output"]["predictions"]["cost"]["value"] > 0
    # the tool result sent back to the LLM carries the real model numbers
    sent = fake.calls[1]["messages"][-1]["content"][0]
    assert sent["type"] == "tool_result" and str(round(trace["output"]["predictions"]["cost"]["value"]))[:4] in sent["content"]
    assert "system" in fake.calls[0] and "NEVER calculate" in fake.calls[0]["system"]


def test_invalid_tool_arguments_become_errors_not_predictions(service):
    r = assistant.run_tool("predict_outcomes", {"location_type": "mars", "building_size": -1, "roofing_type": "sheet",
                                                "cool_roofing_method": "sheet", "cool_roofing_quality": "high"})
    assert "error" in r and "invalid input" in r["error"]
    r = assistant.run_tool("predict_outcomes", {"location_type": "urban", "building_size": 1000, "roofing_type": "sheet",
                                                "cool_roofing_method": "tiles", "cool_roofing_quality": "high"})
    assert "error" in r and "not offered" in r["error"]
    assert "error" in assistant.run_tool("not_a_tool", {})
    leak = assistant.run_tool("predict_outcomes", {"location_type": "urban", "building_size": 1000, "roofing_type": "tiles",
                                                   "cool_roofing_method": "tiles", "cool_roofing_quality": "high", "cool_roofing_cost": 5})
    assert "error" in leak


def test_compare_and_payback_tools(service, building):
    b = {k: v for k, v in building.items() if k not in ("cool_roofing_method", "cool_roofing_quality")}
    cmp_ = assistant.run_tool("compare_options", {**b, "objective": "cost"})
    assert len(cmp_["options"]) == 15 and cmp_["recommended"]["rank"] == 1
    pay = assistant.run_tool("calculate_payback", {"cost_inr": 100000, "monthly_kwh_saved": 100, "tariff_inr_per_kwh": 10})
    assert pay["central"]["simple_payback_years"] == pytest.approx(100000 / 12000)
    assert "error" in assistant.run_tool("calculate_payback", {"cost_inr": 100000, "monthly_kwh_saved": 100, "tariff_inr_per_kwh": 0})


def test_runaway_tool_loop_is_bounded(service):
    endless = FakeClient([[tool_use("get_input_schema", {}, f"t{i}")] for i in range(20)])
    out = assistant.chat([{"role": "user", "content": "loop"}], client=endless)
    assert "could not finish" in out["reply"] and len(endless.calls) == assistant.MAX_TOOL_ROUNDS


def test_chat_endpoint_503_only_when_switched_off_and_core_still_works(client, building, monkeypatch):
    monkeypatch.setenv("COOLROOF_LLM_PROVIDER", "off")
    r = client.post("/api/assistant/chat", json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 503 and "switched off" in r.json()["detail"]
    assert client.post("/api/predict", json={"building": building}).status_code == 200


def test_chat_endpoint_works_with_no_key_via_builtin(client):
    r = client.post("/api/assistant/chat", json={"messages": [{"role": "user", "content":
        "Urban house with a flat concrete roof, 1500 sq ft, 2 ACs and about 140 kWh a month"}]})
    assert r.status_code == 200
    j = r.json()
    assert j["provider"] == "builtin" and [t["tool"] for t in j["tool_results"]] == ["compare_options"]
    assert "Recommended" in j["reply"] and "not guarantees" in j["reply"]


def test_chat_endpoint_failure_is_graceful(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    monkeypatch.setenv("COOLROOF_LLM_PROVIDER", "anthropic")

    def boom(*a, **k):
        raise RuntimeError("upstream exploded with key sk-secret")

    monkeypatch.setattr(assistant, "chat", boom)
    r = client.post("/api/assistant/chat", json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 502 and "sk-secret" not in r.text


def test_no_secrets_in_repository():
    import re
    from coolroof import config as C
    pat = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")
    for p in list(C.ROOT.glob("src/**/*.py")) + list(C.ROOT.glob("*.md")) + list(C.ROOT.glob("*.example")):
        assert not pat.search(p.read_text(encoding="utf-8", errors="ignore")), p
    assert not (C.ROOT / ".env").exists() or ".env" in (C.ROOT / ".gitignore").read_text()


@pytest.mark.parametrize("exc_name,text,code,needle", [
    ("AuthenticationError", "Error code: 401 - invalid x-api-key", 502, "rejected the API key"),
    ("BadRequestError", "Your credit balance is too low to access the Anthropic API", 502, "no credit"),
    ("RateLimitError", "429 slow down", 429, "rate-limiting"),
    ("APIConnectionError", "boom", 502, "temporarily unavailable"),
])
def test_upstream_llm_errors_get_actionable_messages_without_secrets(client, monkeypatch, exc_name, text, code, needle):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-should-never-appear")
    Err = type(exc_name, (Exception,), {})

    def boom(*a, **k):
        raise Err(text + " sk-ant-secret-should-never-appear")

    monkeypatch.setattr(assistant, "chat", boom)
    r = client.post("/api/assistant/chat", json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == code and needle in r.json()["detail"]
    assert "sk-ant" not in r.text


# ------------------------------------------------------------------ built-in assistant
from coolroof.llm import builtin


def test_builtin_parses_plain_language():
    f = builtin.parse("A 20 years old tiled house in a village, 900 sq ft, no AC, uses about 60 kWh a month, roof at 36 degrees. Tariff Rs 8 per kWh")
    assert f["roofing_type"] == "tiles" and f["location_type"] == "rural" and f["building_size"] == 900
    assert f["num_ac_units"] == 0 and f["avg_units_before"] == 60 and f["building_age"] == 20
    assert f["avg_roof_temperature_before"] == 36 and f["_tariff"] == 8


def test_builtin_does_not_invent_missing_facts_and_asks_for_them():
    out = builtin.respond([{"role": "user", "content": "I have a flat concrete roof"}], tool_runner=lambda *a: pytest.fail("no tool call without required facts"))
    assert out["tool_results"] == [] and "location type" in out["reply"] and "floor area" in out["reply"]
    greet = builtin.respond([{"role": "user", "content": "hey"}], tool_runner=lambda *a: pytest.fail("no tools for a greeting"))
    assert "Hello" in greet["reply"]


def test_builtin_conversation_accumulates_and_numbers_come_only_from_tools(service):
    msgs = [{"role": "user", "content": "Urban building, flat concrete roof"}]
    first = builtin.respond(msgs, assistant.run_tool)
    assert "floor area" in first["reply"] and not first["tool_results"]
    msgs += [{"role": "assistant", "content": first["reply"]}, {"role": "user", "content": "1500 sq ft, 2 ACs, 140 kWh a month. Which is cheapest?"}]
    out = builtin.respond(msgs, assistant.run_tool)
    cmp_ = out["tool_results"][0]
    assert cmp_["tool"] == "compare_options" and cmp_["input"]["objective"] == "cost"
    rec = cmp_["output"]["recommended"]
    assert f"{rec['temperature_drop']:.1f} °C" in out["reply"] and f"{rec['units_saved']:.0f} kWh" in out["reply"]
    cheapest = min(cmp_["output"]["options"], key=lambda o: o["cost"])
    assert rec["method"] == cheapest["method"] and rec["quality"] == cheapest["quality"]


def test_builtin_requested_option_payback_and_explanation(service):
    out = builtin.respond([{"role": "user", "content":
        "Urban, flat concrete roof, 1500 sq ft, 2 ACs, 140 kWh a month. What about sheet with coating, high quality? Why? Tariff Rs 8 per kWh"}], assistant.run_tool)
    names = [t["tool"] for t in out["tool_results"]]
    assert names == ["compare_options", "predict_outcomes", "explain_prediction", "calculate_payback"]
    pred = out["tool_results"][1]["output"]["predictions"]
    assert f"{pred['units_saved']['value']:.0f} kWh/month" in out["reply"]
    pay = out["tool_results"][3]["output"]["central"]
    assert pay["annual_bill_saving_inr"] > 0


def test_builtin_reports_unsupported_combination_instead_of_guessing(service):
    out = builtin.respond([{"role": "user", "content": "Urban, sheet roof, 1000 sq ft. Use tiles with coating, high quality"}], assistant.run_tool)
    assert "not offered" in out["reply"]


# ------------------------------------------------------------------ OpenAI-compatible providers (Gemini / Groq / Ollama ...)
import httpx


def _mock_gemini(script, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "auth": request.headers.get("authorization"), "body": json.loads(request.content)})
        return script.pop(0)(request)
    return httpx.Client(transport=httpx.MockTransport(handler))


def _msg(content=None, tool_calls=None):
    m = {"role": "assistant", "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return httpx.Response(200, json={"choices": [{"message": m}]})


import json


def test_gemini_tool_loop_via_openai_compatible_endpoint(service, monkeypatch, building):
    monkeypatch.setenv("COOLROOF_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = []
    call = {"id": "c1", "type": "function", "function": {"name": "predict_outcomes", "arguments": json.dumps(building)}}
    http = _mock_gemini([lambda r: _msg(tool_calls=[call]), lambda r: _msg("The estimate is shown above.")], seen)
    out = assistant.chat([{"role": "user", "content": "quote please"}], http_client=http)
    assert out["provider"] == "gemini" and out["reply"] == "The estimate is shown above."
    assert out["tool_results"][0]["output"]["predictions"]["cost"]["value"] > 0
    assert seen[0]["url"].startswith("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    assert seen[0]["auth"] == "Bearer test-key" and seen[0]["body"]["model"] == "gemini-flash-lite-latest"
    assert seen[0]["body"]["tools"][0]["type"] == "function" and seen[0]["body"]["messages"][0]["role"] == "system"
    tool_msg = seen[1]["body"]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1" and "cost" in tool_msg["content"]


def test_bad_tool_arguments_from_an_llm_become_errors(service, monkeypatch):
    monkeypatch.setenv("COOLROOF_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    bad = {"id": "c1", "type": "function", "function": {"name": "predict_outcomes", "arguments": "{not json"}}
    http = _mock_gemini([lambda r: _msg(tool_calls=[bad]), lambda r: _msg("Sorry, could you repeat the details?")], [])
    out = assistant.chat([{"role": "user", "content": "x"}], http_client=http)
    assert "error" in out["tool_results"][0]["output"] and out["reply"].startswith("Sorry")


def test_auto_mode_falls_back_to_builtin_when_provider_fails(service, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")                      # auto -> gemini
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429, json={"error": "quota"})))
    out = assistant.chat([{"role": "user", "content": "Urban, flat concrete roof, 1500 sq ft"}], http_client=http)
    assert out["provider"] == "builtin" and "free built-in assistant" in out["notice"] and out["tool_results"]


def test_explicit_provider_configuration_errors_are_not_hidden(service, monkeypatch):
    monkeypatch.setenv("COOLROOF_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": "bad key"})))
    with pytest.raises(httpx.HTTPStatusError):
        assistant.chat([{"role": "user", "content": "x"}], http_client=http)


@pytest.mark.parametrize("code", [429, 500, 503])
def test_temporary_provider_problems_retry_once_then_fall_back(service, monkeypatch, code):
    monkeypatch.setenv("COOLROOF_LLM_PROVIDER", "gemini")           # even when chosen explicitly
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("COOLROOF_LLM_RETRY_DELAY", "0")
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(code, json={"error": "busy"})

    out = assistant.chat([{"role": "user", "content": "Urban, flat concrete roof, 1500 sq ft"}], http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert len(calls) == 2                                           # one retry
    assert out["provider"] == "builtin" and "busy or rate-limited" in out["notice"] and out["tool_results"]


def test_a_transient_failure_that_recovers_on_retry_uses_the_real_provider(service, monkeypatch, building):
    monkeypatch.setenv("COOLROOF_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("COOLROOF_LLM_RETRY_DELAY", "0")
    script = [lambda r: httpx.Response(503, json={"error": "overloaded"}), lambda r: _msg("Here you go.")]
    out = assistant.chat([{"role": "user", "content": "hi"}], http_client=_mock_gemini(script, []))
    assert out["provider"] == "gemini" and out["reply"] == "Here you go." and "notice" not in out


def test_free_tier_rate_limit_message_is_actionable(client, monkeypatch):
    monkeypatch.setenv("COOLROOF_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
    req = httpx.Request("POST", "https://x")
    err = httpx.HTTPStatusError("429 rate limit super-secret-key", request=req, response=httpx.Response(429, request=req))

    def boom(*a, **k):
        raise err

    monkeypatch.setattr(assistant, "chat", boom)
    r = client.post("/api/assistant/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 429 and "free-tier limit" in r.json()["detail"] and "secret" not in r.text


def test_assistants_never_claim_things_the_data_cannot_support(service):
    assert "NO information about durability" in assistant.SYSTEM_PROMPT and "do not state anything about them" in assistant.SYSTEM_PROMPT
    out = builtin.respond([{"role": "user", "content": "Urban, flat concrete roof, 1500 sq ft. Which is cheap and has the best durability?"}], assistant.run_tool)
    assert "no information about durability" in out["reply"] and out["tool_results"]

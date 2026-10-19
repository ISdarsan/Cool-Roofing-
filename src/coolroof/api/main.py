"""FastAPI backend. Start with:  python -m coolroof serve   (or uvicorn coolroof.api.main:app)

* Models are loaded once (lazy singleton) - requests never retrain.
* Customer inputs are never logged.
* Errors are returned as {"detail": "..."} without internal paths or stack traces.
"""
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import config as C
from .. import recommend as R
from ..features import input_schema
from ..llm import assistant
from ..proposal import build_proposal_pdf
from ..service import InvalidCombination, ModelNotReady, ModelService, get_service
from . import schemas as S

def _load_dotenv() -> None:
    """Minimal .env reader (no extra dependency). Existing environment variables win."""
    f = C.ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
log = logging.getLogger("coolroof.api")
app = FastAPI(title="Cool Roofing Intelligence API", version=C.MODEL_VERSION,
              description="Cost, comfort and energy-savings predictions for cool-roofing installations.")

origins = [o.strip() for o in os.environ.get("COOLROOF_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


# --------------------------------------------------------------------------
# error handling (never echoes inputs or internals)
# --------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def _validation(_: Request, exc: RequestValidationError):
    items = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": "Invalid request.", "errors": items})


@app.exception_handler(ModelNotReady)
async def _not_ready(_: Request, exc: ModelNotReady):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(InvalidCombination)
async def _bad_combo(_: Request, exc: InvalidCombination):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception):
    log.exception("unhandled error")           # stack trace stays in the server log, request body is not logged
    return JSONResponse(status_code=500, content={"detail": "Internal error. Please try again or contact support."})


def svc() -> ModelService:
    return get_service()


@lru_cache(maxsize=32)
def _read_json(path: str, mtime: float):
    from ..utils import to_jsonable
    with open(path, encoding="utf-8") as f:
        return to_jsonable(json.load(f))      # tolerate legacy NaN literals in older report files


def _json_report(name: str):
    p = C.REPORTS / name
    if not p.exists():
        raise HTTPException(404, "Report not generated yet. Run `python -m coolroof all`.")
    return _read_json(str(p), p.stat().st_mtime)


# --------------------------------------------------------------------------
# health / status / metadata
# --------------------------------------------------------------------------
@app.get("/api/health", tags=["system"])
def health():
    try:
        s = get_service()
        return {"status": "ok", "models_loaded": True, "model_version": s.meta["model_version"]}
    except ModelNotReady as exc:
        return {"status": "degraded", "models_loaded": False, "detail": str(exc)}


@app.get("/api/status", tags=["system"])
def status():
    s = svc().status()
    s["assistant"] = assistant.llm_status()
    return s


@app.get("/api/schema", tags=["system"])
def schema():
    s = svc()
    return {**input_schema(), "method_roof_support": s.pair_support, "training_ranges": s.ranges,
            "min_pair_support": R.MIN_PAIR_SUPPORT, "objectives": R.OBJECTIVES, "default_weights": R.BALANCED_WEIGHTS,
            "metrics": {k: {"label": v[0], "direction": v[1], "unit": v[2]} for k, v in R.METRICS.items()},
            "high_rating_threshold": s.meta["high_rating_threshold"]}


@app.get("/api/models", tags=["system"])
def models():
    s = svc()
    cands = _json_report("model_candidates.json")
    return {"meta": {k: s.meta[k] for k in ("model_version", "trained_at_utc", "seed", "test_size", "cv_folds", "data", "environment")},
            "tasks": s.tasks, "candidates": cands, "task_order": C.TASK_ORDER}


@app.get("/api/data-quality", tags=["insights"])
def data_quality():
    return {"profile": _json_report("data_profile.json"), "summary": _json_report("data_quality_summary.json")}


@app.get("/api/insights", tags=["insights"])
def insights():
    return _json_report("insights.json")


REPORT_NAMES = {"data_quality_report", "model_evaluation", "business_insights", "business_recommendations"}


@app.get("/api/reports/{name}", tags=["insights"])
def report(name: str):
    if name not in REPORT_NAMES:
        raise HTTPException(404, "Unknown report.")
    p = C.REPORTS / f"{name}.md"
    if not p.exists():
        raise HTTPException(404, "Report not generated yet.")
    return {"name": name, "markdown": p.read_text(encoding="utf-8")}


@app.get("/api/figures/{name}", tags=["insights"])
def figure(name: str):
    if not re.fullmatch(r"[\w\-]+\.png", name):
        raise HTTPException(404, "Unknown figure.")
    p = C.FIGURES / name
    if not p.exists():
        raise HTTPException(404, "Unknown figure.")
    return FileResponse(p, media_type="image/png")


# --------------------------------------------------------------------------
# predictions
# --------------------------------------------------------------------------
@app.post("/api/predict", tags=["predict"])
def predict(req: S.PredictRequest):
    return svc().predict(req.building.as_features(), level=req.interval_level, referral_cutoff=req.referral_cutoff,
                         high_cutoff=req.high_rating_cutoff)


@app.post("/api/predict/batch", tags=["predict"])
def predict_batch(req: S.BatchRequest):
    preds = svc().predict_batch([b.as_features() for b in req.items], level=req.interval_level)
    return {"count": len(preds), "model_version": svc().meta["model_version"], "predictions": preds}


@app.post("/api/compare", tags=["predict"])
def compare(req: S.CompareRequest):
    b = req.building.as_features()
    b["cool_roofing_method"], b["cool_roofing_quality"] = req.current_method, req.current_quality
    try:
        return svc().compare(b, objective=req.objective, weights=req.weights, level=req.interval_level)
    except ValueError as exc:
        if isinstance(exc, InvalidCombination):
            raise
        raise HTTPException(422, str(exc))


@app.post("/api/explain", tags=["predict"])
def explain(req: S.ExplainRequest):
    return svc().explain(req.building.as_features(), req.task)


@app.post("/api/economics", tags=["predict"])
def economics(req: S.EconomicsRequest):
    e = R.EconomicsInput(req.cost_inr, req.monthly_kwh_saved, req.tariff_inr_per_kwh, req.annual_maintenance_inr,
                         req.tariff_escalation_pct, req.horizon_years)
    return R.economics(e, req.monthly_kwh_saved_low, req.monthly_kwh_saved_high)


@app.post("/api/proposal/pdf", tags=["proposal"])
def proposal(req: S.ProposalRequest):
    pdf = build_proposal_pdf(svc(), req.building.as_features(), req.customer_name, req.prepared_by, req.tariff_inr_per_kwh, req.interval_level)
    return Response(content=pdf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="cool-roofing-proposal.pdf"'})


# --------------------------------------------------------------------------
# optional assistant
# --------------------------------------------------------------------------
def _explain_llm_error(exc: Exception) -> tuple[int, str]:
    """Maps upstream LLM failures to actionable, secret-free messages."""
    tail = "The prediction tools and the rest of the app still work."
    name = type(exc).__name__
    text = str(exc).lower()
    if name == "HTTPStatusError":                               # OpenAI-compatible providers (Gemini, Groq, Ollama...)
        code = getattr(getattr(exc, "response", None), "status_code", 0)
        if code in (401, 403):
            return 502, f"The AI provider rejected the API key. Check the key in .env and restart the server. {tail}"
        if code == 404:
            return 502, f"The AI provider could not find the configured model. Check COOLROOF_LLM_MODEL (or the provider's model list). {tail}"
        if code == 429:
            return 429, f"The AI provider's free-tier limit was reached. Wait a minute and try again. {tail}"
        if code == 400:
            return 502, f"The AI provider rejected the request (often a wrong model name). Check COOLROOF_LLM_MODEL. {tail}"
        if code >= 500:
            return 502, f"The AI provider is overloaded or down right now. Try again in a moment, or set COOLROOF_LLM_PROVIDER=builtin. {tail}"
    if name in ("ConnectError", "ConnectTimeout", "ReadTimeout"):
        return 502, f"Could not reach the AI provider (is Ollama running / is the network up?). {tail}"
    if name == "AuthenticationError" or "invalid x-api-key" in text:
        return 502, f"The AI provider rejected the API key. Check ANTHROPIC_API_KEY in .env and restart the server. {tail}"
    if "credit balance is too low" in text:
        return 502, f"The AI provider account has no credit. Add credits under Plans & Billing, then try again. {tail}"
    if name == "PermissionDeniedError":
        return 502, f"The API key is not allowed to use this model. Check the key's permissions or COOLROOF_LLM_MODEL. {tail}"
    if name == "NotFoundError":
        return 502, f"The configured model was not found. Check COOLROOF_LLM_MODEL. {tail}"
    if name == "RateLimitError":
        return 429, f"The AI provider is rate-limiting requests. Wait a moment and try again. {tail}"
    return 502, f"The AI assistant is temporarily unavailable. {tail}"

@app.get("/api/assistant/status", tags=["assistant"])
def assistant_status():
    return assistant.llm_status()


@app.post("/api/assistant/chat", tags=["assistant"])
def assistant_chat(req: S.ChatRequest):
    st = assistant.llm_status()
    if not st["enabled"]:
        raise HTTPException(503, st["message"])
    try:
        return assistant.chat([m.model_dump() for m in req.messages])
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("assistant failure")
        status_code, message = _explain_llm_error(exc)
        raise HTTPException(status_code, message)


# --------------------------------------------------------------------------
# optional: serve the built frontend so one process runs the whole product
# --------------------------------------------------------------------------
_dist = C.ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Not found.")
        f = (_dist / path).resolve()
        if path and f.is_file() and _dist.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(_dist / "index.html")

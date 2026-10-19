import re
from pathlib import Path

R = Path(r"D:\bia hackathon")


def edit(path, pairs):
    p = R / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:70])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")


# 1. httpx is used at runtime by the Gemini/Groq/Ollama assistant path
edit("pyproject.toml", [('  "reportlab>=4.0",\n]', '  "reportlab>=4.0",\n  "httpx>=0.27",\n]')])

# 2. README: current test counts + simulator
edit("README.md", [
    ("python -m pytest            # 148 backend tests (data, leakage, ML, independent model review, recommendation, API, assistant providers)",
     "python -m pytest            # 153 backend tests (data, leakage, ML, independent model review, recommendation, API, assistant providers)"),
    ("cd frontend && npm test     # 22 UI tests incl. axe accessibility checks", "cd frontend && npm test     # 30 UI tests incl. axe accessibility checks and the what-if simulator"),
])
s = (R / "README.md").read_text(encoding="utf-8")
if "What-if simulator" not in s:
    s = s.replace("## Quick start", "## What-if simulator\n`What-if simulator` (sidebar) starts from the customer on the prediction page, lets you change inputs or the roofing method/quality, and **Run simulation** shows baseline vs scenario for cost, temperature reduction and electricity saved plus a chart of every valid roofing option. "
                  "Every number comes from the existing `/api/predict` and `/api/compare` endpoints and trained models; nothing is computed in the browser.\n\n## Quick start", 1)
    (R / "README.md").write_text(s, encoding="utf-8")

# 3. plan + architecture page counts
edit("PROJECT_PLAN.md", [("React/TS/Vite, 7 views, real API only, responsive, accessible", "React/TS/Vite, 8 views (incl. what-if simulator), real API only, responsive, accessible")])
arch = (R / "docs/architecture.md").read_text(encoding="utf-8")
arch = arch.replace("`pages/` (Overview, Predict, Compare, Performance, Insights, Proposal, Assistant)", "`pages/` (Overview, Predict, Compare, Simulator, Performance, Insights, Proposal, Assistant)")
(R / "docs/architecture.md").write_text(arch, encoding="utf-8")

# 4. deck + fact sheet constants
edit("presentation/build_deck.py", [("BACKEND_TESTS, FRONTEND_TESTS = 153, 22 ", "BACKEND_TESTS, FRONTEND_TESTS = 153, 30 "),
                                    ('"7 pages · light/dark · accessible"', '"8 pages · light/dark · accessible"')])
edit("presentation/make_fact_sheet.py", [("**22 frontend (Vitest, including axe accessibility checks)**", "**30 frontend (Vitest, including axe accessibility checks and the what-if simulator)**"),
                                         ("Frontend: 22 (unchanged).", "Frontend: now 30 (22 at the time of the brief, plus 8 what-if simulator tests).")])
print("edited")

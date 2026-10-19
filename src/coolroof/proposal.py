"""Customer proposal PDF built from the real model predictions (never from typed-in numbers)."""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import recommend as R
from .service import ModelService

INK, TEAL, SUN, MIST = colors.HexColor("#14262B"), colors.HexColor("#0E7C86"), colors.HexColor("#E8A33D"), colors.HexColor("#E8F0F0")
LABEL = {"basic_coating": "Basic coating", "sheet": "Sheet", "sheet_with_coating": "Sheet + coating", "tiles": "Tiles",
         "tiles_with_coating": "Tiles + coating"}


def _fonts() -> tuple[str, str, str]:
    """Registers DejaVu (has the ₹ glyph) if bundled with matplotlib; otherwise falls back to Helvetica + 'Rs.'."""
    try:
        import matplotlib

        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont("DejaVu", str(d / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(d / "DejaVuSans-Bold.ttf")))
        return "DejaVu", "DejaVu-Bold", "₹"
    except Exception:
        return "Helvetica", "Helvetica-Bold", "Rs. "


def _inr(v: float, sym: str) -> str:
    return f"{sym}{v:,.0f}"


def build_proposal_pdf(svc: ModelService, building: dict, customer_name: str | None = None, prepared_by: str | None = None,
                       tariff: float | None = None, level: float = 0.8) -> bytes:
    base, bold, sym = _fonts()
    pred = svc.predict(building, level=level)
    p = pred["predictions"]
    cmp_ = svc.compare(building, objective="balanced", level=level)
    obs = _historical(svc, building)

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontName=bold, fontSize=22, textColor=INK, alignment=0, spaceAfter=2)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName=bold, fontSize=13, textColor=TEAL, spaceBefore=10, spaceAfter=4)
    P = ParagraphStyle("P", parent=ss["BodyText"], fontName=base, fontSize=9.5, leading=13.5, textColor=INK)
    SM = ParagraphStyle("SM", parent=P, fontSize=8, leading=11, textColor=colors.HexColor("#5B6B70"))
    CELL = ParagraphStyle("CELL", parent=P, fontSize=9, leading=12)

    def tbl(rows, widths, header=True):
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        st = [("FONTNAME", (0, 0), (-1, -1), base), ("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D6D8")),
              ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
        if header:
            st += [("BACKGROUND", (0, 0), (-1, 0), TEAL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), bold)]
        t.setStyle(TableStyle(st))
        return t

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Cool Roofing Proposal", author=prepared_by or "Cool Roofing Intelligence")
    S: list = []
    S.append(Paragraph("Cool Roofing Proposal", H1))
    who = customer_name or "Customer"
    S.append(Paragraph(f"Prepared for <b>{_esc(who)}</b>{(' · by ' + _esc(prepared_by)) if prepared_by else ''} · {date.today():%d %B %Y}", P))
    S.append(Spacer(1, 6))

    S.append(Paragraph("1. Your building", H2))
    inp = building
    missing = "not provided – typical value assumed"
    rows = [["Detail", "Value"],
            ["Location type", inp["location_type"].replace("_", " ")],
            ["Building size", f"{inp['building_size']:,.0f}"],
            ["Existing roof", inp["roofing_type"].replace("_", " ")],
            ["Building age (years)", _v(inp.get("building_age"), missing)],
            ["Current roof temperature (°C)", _v(inp.get("avg_roof_temperature_before"), missing)],
            ["Air-conditioning units", _v(inp.get("num_ac_units"), missing)],
            ["Current electricity use (kWh/month)", _v(inp.get("avg_units_before"), missing)]]
    S.append(tbl(rows, [70 * mm, 100 * mm]))

    S.append(Paragraph("2. Proposed option", H2))
    S.append(Paragraph(f"<b>{LABEL[inp['cool_roofing_method']]}</b>, <b>{inp['cool_roofing_quality']}</b> quality grade.", P))
    lv = int(level * 100)
    rows = [["Estimate", "Expected value", f"{lv}% likely range"],
            ["Installation cost", _inr(p["cost"]["value"], sym), f"{_inr(p['cost']['interval']['low'], sym)} – {_inr(p['cost']['interval']['high'], sym)}"],
            ["Roof temperature reduction", f"{p['temperature_drop']['value']:.1f} °C",
             f"{p['temperature_drop']['interval']['low']:.1f} – {p['temperature_drop']['interval']['high']:.1f} °C"],
            ["Electricity saved per month", f"{p['units_saved']['value']:.0f} kWh",
             f"{p['units_saved']['interval']['low']:.0f} – {p['units_saved']['interval']['high']:.0f} kWh"]]
    S.append(tbl(rows, [70 * mm, 45 * mm, 55 * mm]))
    if "avg_roof_temperature_after" in pred["derived"]:
        S.append(Spacer(1, 3))
        S.append(Paragraph(f"That would take the roof from {inp['avg_roof_temperature_before']:.1f} °C to about "
                           f"{pred['derived']['avg_roof_temperature_after']['value']:.1f} °C.", P))
    if obs:
        S.append(Paragraph(f"In our records, {obs['rate']:.0%} of {obs['n']} comparable installations (same method and quality) were rated "
                           f"{obs['threshold']} stars or higher by customers.", P))

    S.append(Paragraph("3. Options we compared", H2))
    S.append(Paragraph("All supported method and quality combinations for your roof type were evaluated with the same models. "
                       "Ranked by a balanced score across cost, cooling, savings, satisfaction and referral likelihood.", P))
    rows = [["Option", "Cost", "Cooling", "kWh/month"]]
    for o in cmp_["options"][:5]:
        tag = " ◄ proposed" if (o["cool_roofing_method"] == inp["cool_roofing_method"] and o["cool_roofing_quality"] == inp["cool_roofing_quality"]) else ""
        rows.append([f"{LABEL[o['cool_roofing_method']]}, {o['cool_roofing_quality']}{tag}", _inr(o["predictions"]["cost"], sym),
                     f"{o['predictions']['temperature_drop']:.1f} °C", f"{o['predictions']['units_saved']:.0f}"])
    S.append(tbl(rows, [72 * mm, 36 * mm, 28 * mm, 34 * mm]))

    assumptions = list(pred["assumptions"])
    if tariff:
        e = R.economics(R.EconomicsInput(p["cost"]["value"], p["units_saved"]["value"], tariff),
                        p["units_saved"]["interval"]["low"], p["units_saved"]["interval"]["high"])
        c = e["central"]
        S.append(Paragraph("4. Indicative payback (assumption-based)", H2))
        pb = f"{c['simple_payback_years']:.1f} years" if c["simple_payback_years"] else "does not pay back through electricity savings alone"
        S.append(Paragraph(f"At an electricity tariff of {sym}{tariff:g}/kWh (supplied by you), estimated bill savings are about "
                           f"{_inr(c['annual_bill_saving_inr'], sym)} per year, giving a simple payback of <b>{pb}</b>.", P))
        be = c.get("breakeven_tariff_inr_per_kwh")
        if be:
            S.append(Paragraph(f"For this option to pay back within {e['inputs']['horizon_years']} years from electricity savings alone, the tariff would need to be "
                               f"about {sym}{be:,.1f}/kWh or higher.", P))
        if e["range"]:
            lo, hi = e["range"]["pessimistic"], e["range"]["optimistic"]
            f = lambda x: f"{x['simple_payback_years']:.1f} yrs" if x["simple_payback_years"] else "no payback"  # noqa: E731
            S.append(Paragraph(f"Range using the {lv}% savings interval: {f(hi)} (optimistic) to {f(lo)} (pessimistic).", P))
        assumptions += e["assumptions"] + e["warnings"]

    S.append(Paragraph("Assumptions and limitations" if not tariff else "5. Assumptions and limitations", H2))
    for w_ in pred["warnings"]:
        S.append(Paragraph("• " + _esc(w_), CELL))
    for a in assumptions:
        S.append(Paragraph("• " + _esc(a), CELL))
    S.append(Paragraph("• These are statistical estimates from past installations, not guarantees. Actual results depend on the site, weather, "
                       "usage patterns and installation quality.", CELL))
    S.append(Spacer(1, 8))
    S.append(Paragraph(f"Model version {svc.meta['model_version']} · trained on {svc.meta['data']['n_rows']:,} historical installations.", SM))
    doc.build(S)
    return buf.getvalue()


def _v(x, missing):
    return missing if x is None else (f"{x:g}")


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _historical(svc: ModelService, b: dict) -> dict | None:
    try:
        from .data.clean import load_cleaned
        df = load_cleaned()
        g = df[(df.cool_roofing_method == b["cool_roofing_method"]) & (df.cool_roofing_quality == b["cool_roofing_quality"])]
        n = int(g["customer_given_rating"].notna().sum())
        if n < 15:
            return None
        return {"n": n, "rate": float(g["high_rating"].mean()), "threshold": svc.meta["high_rating_threshold"]}
    except Exception:
        return None

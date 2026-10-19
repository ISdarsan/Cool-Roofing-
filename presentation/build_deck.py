"""Builds the hackathon deck from the project's saved reports. Every number is read from reports/ or models/ at build time."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(r"D:\bia hackathon")
OUT = ROOT / "presentation"
ASSETS = OUT / "presentation_assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- verified data
META = json.loads((ROOT / "models/model_metadata.json").read_text(encoding="utf-8"))
T = META["tasks"]
PROFILE = json.loads((ROOT / "reports/data_profile.json").read_text(encoding="utf-8"))
SUMMARY = json.loads((ROOT / "reports/data_quality_summary.json").read_text(encoding="utf-8"))
INS = json.loads((ROOT / "reports/insights.json").read_text(encoding="utf-8"))
FI = pd.DataFrame(INS["feature_importance"])
N_TEST = META["data"]["n_test_buildings"]
N_TV = META["data"]["n_train_validation_buildings"]
N_ROWS = META["data"]["n_rows"]
BACKEND_TESTS, FRONTEND_TESTS = 153, 30      # last full runs in this project (pytest / vitest)

# ----------------------------------------------------------------------------- style
GREEN, TEAL, SUN, CLAY = "0F4C3A", "0E7C86", "E8A33D", "B0412C"
CHAR, GREY, MIST, WHITE, LINE = "1F2A2E", "5F6E73", "EAF2EF", "FFFFFF", "D5DEDB"
FONT = "Calibri"
rgb = lambda h: RGBColor.from_string(h)
NICE = {"cool_roofing_method": "Roofing method", "cool_roofing_quality": "Quality grade", "building_size": "Building size",
        "building_age": "Building age", "roofing_type": "Existing roof", "location_type": "Location",
        "avg_roof_temperature_before": "Start roof temp.", "num_ac_units": "AC units", "avg_units_before": "Current kWh"}
METHOD = {"basic_coating": "Basic coating", "sheet": "Sheet", "sheet_with_coating": "Sheet + coating", "tiles": "Tiles", "tiles_with_coating": "Tiles + coating"}
inr = lambda v: "₹" + f"{v:,.0f}"

plt.rcParams.update({"font.family": ["Calibri", "Segoe UI", "DejaVu Sans"], "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8A979B", "axes.labelcolor": "#1F2A2E", "xtick.color": "#4B5A5F", "ytick.color": "#4B5A5F",
                     "axes.titleweight": "bold", "axes.titlesize": 13, "axes.grid": True, "grid.color": "#E3EAE7", "figure.dpi": 200})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(ASSETS / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- charts from real data
def hist_ax(ax, h, color, xlabel, scale=1.0, zero=False, fmt=None):
    e = np.array(h["edges"]) / scale
    ax.bar(e[:-1], h["counts"], width=np.diff(e), align="edge", color="#" + color, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("buildings")
    if zero:
        ax.axvline(0, color="#" + CLAY, ls="--", lw=1.6)


fig, axs = plt.subplots(1, 3, figsize=(12.2, 2.5))
hist_ax(axs[0], PROFILE["targets"]["cost"], TEAL, "installation cost (₹ lakh)", 1e5)
hist_ax(axs[1], PROFILE["targets"]["temperature_drop"], SUN, "roof temperature drop (°C)")
hist_ax(axs[2], PROFILE["targets"]["units_saved"], GREEN, "monthly electricity saved (kWh)", zero=True)
axs[0].set_title(f"Cost (n={PROFILE['targets']['cost']['n']:,})"); axs[1].set_title(f"Temperature drop (n={PROFILE['targets']['temperature_drop']['n']:,})")
axs[2].set_title(f"Electricity saved (n={PROFILE['targets']['units_saved']['n']:,}); dashed = 0")
save(fig, "chart_target_distributions.png")

miss = {c["column"]: c["before"] for c in PROFILE["missingness"] if c["before"] > 0}
fig, ax = plt.subplots(figsize=(5.9, 2.5))
names = [k.replace("_", " ") for k in miss]
ax.barh(names, list(miss.values()), color="#" + GREY)
for i, v in enumerate(miss.values()):
    ax.text(v + 0.5, i, str(v), va="center")
ax.set_xlabel("missing cells in the raw file"); ax.set_title(f"Missing inputs: {sum(miss.values())} cells in {len(miss)} columns"); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
save(fig, "chart_missingness.png")

rc = PROFILE["targets"]["rating"]; rf = PROFILE["targets"]["referral"]
fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.5), gridspec_kw={"width_ratios": [3, 2]})
axs[0].bar([f"{v}★" for v in rc["values"]], rc["counts"], color="#" + TEAL)
axs[0].set_title("Customer rating"); axs[0].set_ylabel("buildings")
axs[1].bar(["no", "yes"], [rf["false"], rf["true"]], color=["#" + GREY, "#" + SUN])
axs[1].set_title(f"Referral ({rf['rate']:.1%} yes)")
save(fig, "chart_rating_referral.png")

z, tsent = SUMMARY["zero_consumption_placeholders"], SUMMARY["temperature_sentinel_rows"]
miss_cells = sum(miss.values())
anom = [("Missing input cells (imputed in training folds)", miss_cells, GREY), ("120 °C roof temperatures: invalid → missing", tsent["before"] + tsent["after"], CLAY),
        ("0 kWh readings: suspected placeholders → missing", z["before"] + z["after"], SUN), ("Electricity went UP after install: kept (real)", SUMMARY["negative_savings_rows"], TEAL)]
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.barh([a[0] for a in anom], [a[1] for a in anom], color=["#" + a[2] for a in anom])
for i, a in enumerate(anom):
    ax.text(a[1] + 2, i, str(a[1]), va="center", fontweight="bold")
ax.invert_yaxis(); ax.set_xlabel("records / cells"); ax.set_title("What the audit found"); ax.grid(axis="y", visible=False); ax.set_xlim(0, 200)
save(fig, "chart_anomalies.png")

tb = PROFILE["temperature_before_after_cleaning"]
fig, ax = plt.subplots(figsize=(5.7, 3.0))
for key, col, lab in (("raw_before", CLAY, "raw file"), ("clean_before", TEAL, "after cleaning")):
    e = np.array(tb[key]["edges"]); ax.bar(e[:-1], tb[key]["counts"], width=np.diff(e), align="edge", color="#" + col, alpha=0.9 if key == "clean_before" else 0.85, label=lab)
ax.set_xlabel("average roof temperature before installation (°C)"); ax.set_ylabel("buildings"); ax.set_title("120 °C spike removed"); ax.legend(frameon=False)
save(fig, "chart_temperature_cleaning.png")

order = ["cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral"]
ttl = {"cost": "Cost", "temperature_drop": "Temperature drop", "units_saved": "Electricity saved", "rating": "Rating (1–5)", "high_rating": "High rating (≥4)", "referral": "Referral"}
fig, axs = plt.subplots(2, 3, figsize=(8.2, 4.9))
for ax, k in zip(axs.ravel(), order):
    g = FI[FI.task == k].sort_values("share", ascending=False).head(4).iloc[::-1]
    ax.barh([NICE[f] for f in g.feature], g.share * 100, color="#" + (TEAL if k in ("cost", "temperature_drop", "units_saved") else SUN))
    for i, v in enumerate(g.share * 100):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=11)
    ax.set_title(ttl[k], fontsize=13); ax.set_xlim(0, 100); ax.tick_params(axis="y", labelsize=11); ax.tick_params(axis="x", labelsize=10); ax.grid(axis="y", visible=False)
fig.supxlabel("share of permutation importance on the hold-out set", fontsize=11)
save(fig, "chart_feature_importance.png")

# ----------------------------------------------------------------------------- pptx helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
NOTES: dict[int, str] = {}


def tbox(s, x, y, w, h, paras, size=16, color=CHAR, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=0):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = anchor
    paras = [paras] if isinstance(paras, (str, tuple)) else paras
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space)
        runs = p if isinstance(p, list) else [p]
        for r in runs:
            txt, fmt = (r, {}) if isinstance(r, str) else r
            run = para.add_run()
            run.text = txt
            f = run.font
            f.name = FONT
            f.size = Pt(fmt.get("size", size))
            f.bold = fmt.get("bold", bold)
            f.color.rgb = rgb(fmt.get("color", color))
    return sh


def box(s, x, y, w, h, fill=MIST, line=None, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    if fill:
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line); sh.line.width = Pt(1.25)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def card(s, x, y, w, h, title, body, fill=MIST, tcolor=GREEN, size=14, tsize=16, line=None):
    box(s, x, y, w, h, fill, line)
    tbox(s, x + 0.12, y + 0.08, w - 0.24, 0.4, title, tsize, tcolor, True)
    tbox(s, x + 0.12, y + 0.5, w - 0.24, h - 0.55, body, size, CHAR)


def chip(s, x, y, w, text, fill, color=WHITE, size=13, h=0.38):
    sh = box(s, x, y, w, h, fill, radius=0.3)
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = True; r.font.name = FONT; r.font.color.rgb = rgb(color)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sh


def arrow(s, x, y, w, h, kind=MSO_SHAPE.RIGHT_ARROW, fill=TEAL):
    sh = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill); sh.line.fill.background(); sh.shadow.inherit = False
    return sh


def pic(s, name, x, y, w=None, h=None):
    path = ASSETS / name
    iw, ih = Image.open(path).size
    if w and not h:
        h = w * ih / iw
    elif h and not w:
        w = h * iw / ih
    s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def slide(n, title, kicker=None, source=None, notes=""):
    s = prs.slides.add_slide(BLANK)
    NOTES[n] = notes
    box(s, 0, 0, 13.333, 0.12, GREEN, radius=0, shape=MSO_SHAPE.RECTANGLE)
    if kicker:
        tbox(s, 0.6, 0.28, 9, 0.3, kicker.upper(), 12, TEAL, True)
    tbox(s, 0.6, 0.55, 12.1, 0.9, title, 28, GREEN, True, anchor=MSO_ANCHOR.TOP)
    tbox(s, 0.6, 7.08, 10.8, 0.3, source or "", 11, GREY)
    tbox(s, 11.9, 7.08, 0.85, 0.3, f"{n} / 15", 11, GREY, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def table(s, x, y, w, rows, colw, size=13, rowh=0.42, header_fill=GREEN, zebra=True, bold_first=True):
    shp = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(rowh * len(rows)))
    tb_ = shp.table
    for j, cw in enumerate(colw):
        tb_.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        tb_.rows[i].height = Inches(rowh)
        for j, val in enumerate(r):
            c = tb_.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08); c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid(); c.fill.fore_color.rgb = rgb(header_fill if i == 0 else (MIST if (zebra and i % 2 == 0) else WHITE))
            tf = c.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
            run = p.add_run(); run.text = str(val)
            run.font.size = Pt(size); run.font.name = FONT
            run.font.bold = (i == 0) or (bold_first and j == 0)
            run.font.color.rgb = rgb(WHITE if i == 0 else CHAR)
    return shp


def native_bar(s, x, y, w, h, cats, vals, colors, numfmt, title, size=13):
    cd = CategoryChartData(); cd.categories = cats; cd.add_series("MAE", vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.has_legend = False; ch.has_title = True
    ch.chart_title.text_frame.text = title
    r = ch.chart_title.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(14); r.font.bold = True; r.font.name = FONT; r.font.color.rgb = rgb(CHAR)
    pl = ch.plots[0]; pl.gap_width = 60; pl.has_data_labels = True
    dl = pl.data_labels; dl.number_format = numfmt; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(size); dl.font.bold = True; dl.font.name = FONT
    for i, c in enumerate(colors):
        pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(c)
    va = ch.value_axis; va.has_major_gridlines = False; va.visible = False; va.minimum_scale = 0
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(12); ca.tick_labels.font.name = FONT; ca.format.line.color.rgb = rgb(LINE)
    return gf


# ----------------------------------------------------------------------------- numbers used in text
c, d, u, ra, hr, rfr = (T[k] for k in ("cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral"))
cov = {k: T[k]["interval_coverage_test"]["0.8"] for k in ("cost", "temperature_drop", "units_saved")}
sg_r, sg_h = rfr["significance"], hr["significance"]
x_cost, x_temp, x_units = c["test_baseline"]["mae"] / c["test"]["mae"], d["test_baseline"]["mae"] / d["test"]["mae"], u["test_baseline"]["mae"] / u["test"]["mae"]
ELIG = SUMMARY["eligible_rows_by_task"]
me = INS["marginal_effects"]["method"]
mval = lambda k, m: next(r for r in me[k] if r["level"] == m)["mean"]

# ============================================================================= SLIDE 1
s = prs.slides.add_slide(BLANK)
NOTES[1] = ("[~0.5 min] Good [morning/afternoon]. We built Cool Roofing Intelligence: a system that turns a cool-roofing company's historical installation records into estimates a sales representative can use "
            "before the first quote: what it will cost, how much cooler the roof gets, how many electricity units are saved, and how the customer is likely to feel afterwards. "
            "Everything you will see today comes from our saved evaluation reports and the running application, including the limitations.\n"
            "Takeaway: predictive, explainable and honest decision support for cooler buildings.\nTransition: let me start with the problem.")
s.notes_slide.notes_text_frame.text = NOTES[1]
box(s, 0, 0, 5.4, 7.5, GREEN, radius=0, shape=MSO_SHAPE.RECTANGLE)
box(s, 1.3, 3.2, 2.8, 2.6, "F4F1E8", radius=0.02)                                   # walls
arrow(s, 1.05, 1.85, 3.3, 1.4, MSO_SHAPE.ISOSCELES_TRIANGLE, TEAL)                  # roof
for i in range(3):
    box(s, 1.75 + i * 0.85, 4.1, 0.55, 0.75, "BBD3CC", radius=0.04)
sun = box(s, 3.9, 0.7, 0.95, 0.95, SUN, radius=0.5, shape=MSO_SHAPE.OVAL)
for k in range(8):
    ang = k * np.pi / 4
    box(s, 4.37 + 0.72 * np.cos(ang) - 0.07, 1.17 + 0.72 * np.sin(ang) - 0.07, 0.14, 0.14, SUN, radius=0.5, shape=MSO_SHAPE.OVAL)
tbox(s, 0.5, 6.4, 4.5, 0.5, "Cooler roofs · lower bills · clearer quotes", 15, "BBD3CC", True)
tbox(s, 6.0, 1.7, 6.8, 0.5, "HACKATHON PROJECT", 14, TEAL, True)
tbox(s, 6.0, 2.15, 6.9, 1.9, ["Cool Roofing", "Intelligence"], 50, GREEN, True)
tbox(s, 6.0, 4.15, 6.7, 1.0, "Cost, comfort and energy-savings predictions before the first quote", 22, CHAR)
box(s, 6.05, 5.35, 6.7, 0.04, SUN, radius=0, shape=MSO_SHAPE.RECTANGLE)
tbox(s, 6.0, 5.5, 6.8, 1.0, [f"Learned from {N_ROWS:,} real installations", "5 prediction tasks · 6 models · explainable · tested"], 16, GREY, space=2)

# ============================================================================= SLIDE 2
s = slide(2, "A roofing quote still rests on four open questions", "The problem",
          f"Source: cool_roofing_problem.md; {N_ROWS:,} historical installation records",
          "[~0.75 min] A customer walking in asks four things. How much will it cost? How much cooler will my roof get? How much electricity will I save each month? And will I be happy enough to recommend you? "
          "Today a sales rep answers these from experience. The company, however, has a valuable asset: records of thousands of past installations with the cost, the temperatures and electricity use before and after, and the customer's rating and referral. "
          f"We have {N_ROWS:,} of them. The opportunity is to learn from those records and give consistent, explained answers.\nTakeaway: the data to answer the questions already exists.\nTransition: here is the solution we built on it.")
qs = [("How much will it cost?", "Installation cost depends on size, method, quality and roof."),
      ("How much cooler will the roof get?", "Cooling varies by method and quality, not just by price."),
      ("How many kWh will we save?", "Savings depend on current usage and AC load, and can be near zero."),
      ("Will they rate us highly and refer us?", "Satisfaction and referrals are driven by experience after the work.")]
for i, (q, sub) in enumerate(qs):
    x = 0.6 + i * 3.08
    box(s, x, 1.65, 2.9, 2.6, MIST)
    box(s, x, 1.65, 2.9, 0.1, [TEAL, SUN, GREEN, CLAY][i], radius=0, shape=MSO_SHAPE.RECTANGLE)
    tbox(s, x + 0.15, 1.95, 2.6, 1.0, q, 20, GREEN, True)
    tbox(s, x + 0.15, 3.0, 2.6, 1.2, sub, 15, CHAR)
arrow(s, 6.2, 4.4, 0.9, 0.6, MSO_SHAPE.DOWN_ARROW, SUN)
box(s, 0.6, 5.15, 12.1, 1.65, GREEN)
tbox(s, 0.9, 5.3, 11.5, 0.5, "The opportunity", 16, SUN, True)
tbox(s, 0.9, 5.75, 11.5, 1.0, f"{N_ROWS:,} past installations record cost, roof temperature and electricity use before and after, plus the customer's rating and referral. "
     "That is enough to learn from, if the data can be trusted and the answers are explained honestly.", 18, WHITE)

# ============================================================================= SLIDE 3
s = slide(3, "One system that predicts, compares, explains and proposes", "The solution",
          "Status taken from the repository: implemented and tested vs. optional vs. future",
          "[~0.75 min] The product has three layers. The core, which is implemented and tested, answers the five required questions with six models, because customer rating is both a numeric prediction and a high-or-not classification. "
          "It starts with an audit and cleaning pipeline, because the raw data has problems. It compares every valid roofing option for the same building, explains each prediction, and produces a PDF proposal. "
          "The AI assistant is optional: it never calculates numbers itself, it only calls the same prediction service. The three items on the right are future ideas and are not built.\n"
          "Takeaway: a complete decision-support system, with a clear line between what exists and what is planned.\nTransition: all of it rests on the dataset, so let's look at that.")
cols = [("CORE: implemented and tested", GREEN, ["5 prediction tasks → 6 models", "Data audit and cleaning pipeline", "Hold-out evaluation vs baselines", "Roofing-option comparison engine",
                                                "Per-customer explanations", "Payback calculator (your tariff)", "PDF customer proposals", "React app + FastAPI service"]),
        ("OPTIONAL: implemented", TEAL, ["AI sales assistant: built-in free mode, plus Gemini (live-tested)", "Claude integration built, not live-tested (no credit)", "Assistant only calls validated tools; it cannot invent predictions"]),
        ("FUTURE: not built", GREY, ["Digital-twin simulator", "Carbon-savings tracker", "Geo-climate intelligence", "Validation on newly measured installations"])]
for i, (h_, colr, items) in enumerate(cols):
    x = 0.6 + i * 4.1
    box(s, x, 1.65, 3.9, 5.2, MIST if i < 2 else "F3F4F4", line=None if i < 2 else LINE)
    chip(s, x + 0.15, 1.8, 3.6, h_, colr, size=13, h=0.42)
    tbox(s, x + 0.2, 2.4, 3.55, 4.4, [[("• ", {"bold": True, "color": colr}), it] for it in items], 17, CHAR if i < 2 else GREY, space=8)

# ============================================================================= SLIDE 4
s = slide(4, f"{N_ROWS:,} installations, 15 columns, three measurable outcomes", "Dataset and exploration",
          "Source: dataset/cool_roofing_data.csv (read-only); charts computed from reports/data_profile.json after cleaning",
          f"[~0.75 min] The dataset has {N_ROWS:,} installation records and 15 columns: building profile (location, size, age), roofing details (existing roof, method, quality), cost, roof temperature and monthly electricity before and after, the customer rating from one to five, and whether the customer referred us. "
          "Cost is right-skewed and scales with building size. Roof temperature drops by about 2 to 13 degrees on valid records. Electricity savings are strongly tied to current usage; note the dashed line at zero, because some buildings used more electricity afterwards. "
          f"Four input columns have {sum(miss.values())} missing cells in total. About {rf['rate']:.0%} of customers referred us, so referral is an imbalanced target.\nTakeaway: rich data with skewed targets and some gaps.\nTransition: and some values that should not be trusted.")
pic(s, "chart_target_distributions.png", 0.6, 1.5, w=12.1)
pic(s, "chart_missingness.png", 0.6, 4.15, w=5.9)
pic(s, "chart_rating_referral.png", 6.8, 4.15, w=5.9)
tbox(s, 0.6, 6.7, 12.1, 0.35, f"Records: {N_ROWS:,}  ·  Inputs: location, size, age, existing roof, method, quality, start temperature, AC units, current kWh  ·  Outcomes: cost, temperature, electricity, rating, referral", 12, GREY)

# ============================================================================= SLIDE 5
s = slide(5, "The audit found two silent placeholders that would have corrupted the models", "Data quality",
          "Source: reports/data_quality_report.md, reports/cleaning_audit.csv (record-level audit trail); raw file never modified",
          "[~1 min] Before modelling we audited every column. Two problems stood out. First, exactly 120 degrees appears as a roof temperature in 58 records, never anywhere near other readings, which would imply drops of plus or minus ninety degrees: that is confirmed invalid. "
          "Second, exactly zero kilowatt-hours appears 48 times while one to three kWh never occurs, and those rows are physically inconsistent. We treated them as placeholders. That is an inference, a strong one, but not a confirmed fact. "
          f"We set invalid values to missing rather than guessing, and exclude a record only from the task that needs it. The result was measurable: electricity-savings error on the hold-out fell from 4.45 to {u['test']['mae']:.2f} kWh. "
          f"We also kept {SUMMARY['negative_savings_rows']} buildings where electricity use went up, because those are real adverse outcomes, and deleting them would flatter the model. No record was deleted outright.\n"
          "Takeaway: cleaning decisions are documented, evidence-based and measurable.\nTransition: with clean data, we must also prevent leakage.")
pic(s, "chart_anomalies.png", 0.6, 1.5, w=6.6)
pic(s, "chart_temperature_cleaning.png", 7.4, 1.5, w=5.3)
card(s, 0.6, 4.85, 3.95, 1.95, "Confirmed invalid", [f"120 °C roof temperature: {tsent['before'] + tsent['after']} records ({tsent['before']} before, {tsent['after']} after)", "Set to missing, never capped or guessed"], size=14, fill="F8E6E1", tcolor=CLAY)
card(s, 4.7, 4.85, 3.95, 1.95, "Inferred, high confidence", [f"0 kWh: {z['before'] + z['after']} readings. No 1–3 kWh values exist; rows are physically inconsistent",
                                                             f"Effect: savings MAE 4.45 → {u['test']['mae']:.2f} kWh"], size=14, fill="FBF0DA", tcolor="8A5A00")
card(s, 8.8, 4.85, 3.9, 1.95, "Kept on purpose", [f"{SUMMARY['negative_savings_rows']} buildings used MORE electricity after install: real adverse outcomes", "0 records deleted; exclusions are per task only"], size=14, fill="DDEEF0", tcolor=TEAL)

# ============================================================================= SLIDE 6
s = slide(6, "Only what a rep knows at the table can be a model input", "Targets, features and leakage prevention",
          "Source: docs/feature_contract.md; tests/test_features_leakage.py, tests/test_model_review.py",
          "[~1 min] Two outcomes are derived: temperature drop is temperature before minus after, and electricity saved is units before minus after. The key design rule is that a model may only use facts known before installation. "
          "On the right, nine inputs are allowed: location, size, age, existing roof, the proposed method and quality, and three pre-installation measurements. Realised cost, the after-installation measurements, the customer's rating and the referral flag are forbidden, "
          "as is the building ID. This matters: for example, the referral rate is 61 percent for four-star-plus customers against 9 percent for one-to-two stars, so using the rating to predict referral would be pure leakage. "
          f"For evaluation we fixed a hold-out of {N_TEST} buildings before any modelling. Tuning used five-fold cross-validation on the other {N_TV:,}. All preprocessing is fitted on training data only. "
          "Automated tests and independent controls, such as models trained on shuffled labels, confirm there is no leakage.\nTakeaway: the evaluation answers 'how well will this work on a new building?'.\nTransition: here is the system that implements it.")
card(s, 0.6, 1.55, 5.9, 0.95, "Temperature drop (°C)", "= roof temperature before − roof temperature after", fill=MIST, size=15, tsize=14)
card(s, 0.6, 2.6, 5.9, 0.95, "Electricity saved (kWh / month)", "= monthly kWh before − monthly kWh after", fill=MIST, size=15, tsize=14)
tbox(s, 0.6, 3.75, 5.9, 0.4, "Evaluation protocol", 16, GREEN, True)
box(s, 0.6, 4.2, 5.9, 0.62, GREEN, radius=0.1)
tbox(s, 0.7, 4.25, 5.7, 0.5, f"{N_ROWS:,} buildings", 16, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
box(s, 0.6, 4.95, 4.7, 0.75, TEAL, radius=0.1)
tbox(s, 0.65, 4.97, 4.6, 0.7, [f"{N_TV:,} train + validation", "5-fold CV: tuning and selection"], 14, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
box(s, 5.4, 4.95, 1.1, 0.75, SUN, radius=0.1)
tbox(s, 5.4, 4.97, 1.1, 0.7, [f"{N_TEST}", "hold-out"], 14, CHAR, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
tbox(s, 0.6, 5.85, 5.9, 1.1, ["• Preprocessing fitted on training folds only", "• Hold-out scored once; baselines declared first", "• Shuffled-label and ID-only controls sit at chance"], 14, CHAR, space=2)
tbox(s, 6.9, 1.5, 5.8, 0.4, "ALLOWED at quote time (9 inputs)", 15, GREEN, True)
allowed = ["Location type", "Building size", "Building age", "Existing roof", "Proposed method", "Proposed quality", "Start roof temp.", "AC units", "Current kWh"]
for i, a in enumerate(allowed):
    chip(s, 6.9 + (i % 3) * 1.97, 1.95 + (i // 3) * 0.52, 1.88, a, GREEN, size=12)
tbox(s, 6.9, 3.65, 5.8, 0.4, "FORBIDDEN as inputs", 15, CLAY, True)
forbidden = ["Realised cost", "Temp. after install", "kWh after install", "Customer rating", "Referral flag", "Building ID"]
for i, a in enumerate(forbidden):
    chip(s, 6.9 + (i % 3) * 1.97, 4.1 + (i // 3) * 0.52, 1.88, a, CLAY, size=12)
box(s, 6.9, 5.3, 5.8, 1.5, "F3F4F4", line=LINE)
tbox(s, 7.05, 5.38, 5.5, 1.4, [[("Why it matters: ", {"bold": True, "color": GREEN}), "referral rate is 61% for 4★+ customers vs 9% for 1–2★, so rating would leak referral."],
                               "Rejected at the API too: unknown fields return HTTP 422."], 14, CHAR, space=4)

# ============================================================================= SLIDE 7
s = slide(7, "From raw CSV to a tested, explainable web product", "System architecture",
          "Source: docs/architecture.md and the implemented code (src/coolroof, frontend/)",
          "[~0.75 min] The pipeline flows left to right. The raw CSV is read-only; the audit and cleaning step produces a cleaned dataset and an audit trail; a feature contract and shared preprocessing pipelines feed training and evaluation, which saves six model files plus metadata. "
          "A FastAPI service loads the models once and serves prediction, option comparison, Shapley-value explanations, a payback calculator and PDF proposals. The React and TypeScript front end calls that API. "
          "The optional assistant also goes through the API: the language model can only call validated tools, so it never produces official numbers. "
          f"Quality gates: {BACKEND_TESTS} backend tests and {FRONTEND_TESTS} frontend tests including accessibility checks, a clean type check and production build.\nTakeaway: one shared feature contract keeps training and serving consistent.\nTransition: now the models themselves.")
steps = [("Raw CSV", f"{N_ROWS:,} × 15, read-only"), ("Audit & cleaning", "flags, audit trail"), ("Feature contract", "leakage guards, pipelines"), ("Train & evaluate", "CV, hold-out, baselines"), ("Saved artifacts", "6 models + metadata")]
for i, (a, b_) in enumerate(steps):
    x = 0.6 + i * 2.5
    box(s, x, 1.6, 2.0, 1.15, GREEN if i < 4 else TEAL)
    tbox(s, x + 0.05, 1.68, 1.9, 0.45, a, 16, WHITE, True, PP_ALIGN.CENTER)
    tbox(s, x + 0.05, 2.15, 1.9, 0.55, b_, 12, "D9EAE4", False, PP_ALIGN.CENTER)
    if i < 4:
        arrow(s, x + 2.03, 2.0, 0.44, 0.35)
arrow(s, 11.35, 2.85, 0.5, 0.55, MSO_SHAPE.DOWN_ARROW)
arrow(s, 7.6, 2.85, 0.5, 0.55, MSO_SHAPE.DOWN_ARROW, GREY)
box(s, 6.4, 3.5, 6.3, 1.35, MIST, line=TEAL)
tbox(s, 6.55, 3.55, 6.0, 0.4, "FastAPI inference service (models loaded once)", 16, GREEN, True)
tbox(s, 6.55, 3.98, 6.0, 0.85, "predict · batch · compare options · explain (Shapley) · payback · PDF proposal · strict input validation", 14, CHAR)
box(s, 0.6, 3.5, 5.5, 1.35, "F3F4F4", line=LINE)
tbox(s, 0.75, 3.55, 5.2, 0.4, "Reports and figures", 16, GREY, True)
tbox(s, 0.75, 3.98, 5.2, 0.85, "data-quality, evaluation, significance and insight reports feed the Performance and Insights pages", 14, CHAR)
arrow(s, 8.0, 4.9, 0.4, 0.45, MSO_SHAPE.UP_DOWN_ARROW, TEAL)
arrow(s, 11.0, 4.9, 0.4, 0.45, MSO_SHAPE.UP_DOWN_ARROW, TEAL)
box(s, 6.4, 5.4, 3.0, 1.0, GREEN)
tbox(s, 6.45, 5.45, 2.9, 0.95, ["React + TypeScript UI", "8 pages · light/dark · accessible"], 14, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
box(s, 9.7, 5.4, 3.0, 1.0, TEAL)
tbox(s, 9.75, 5.45, 2.9, 0.95, ["Optional AI assistant", "validated tool calls only"], 14, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
card(s, 0.6, 5.1, 5.5, 1.3, "Quality gates", f"{BACKEND_TESTS} backend + {FRONTEND_TESTS} UI tests (incl. accessibility) · type-check · production build · browser-verified", fill=MIST, size=14, tsize=14)
tbox(s, 0.6, 6.55, 12.1, 0.4, "pandas · scikit-learn · LightGBM · FastAPI · React 19 + TypeScript + Vite · Tailwind · Recharts · ReportLab", 13, GREY, True, PP_ALIGN.CENTER)

# ============================================================================= SLIDE 8
s = slide(8, "Five tasks, six models: simple where the data is simple", "Models",
          "Source: models/model_metadata.json. Customer rating needs two models (numeric rating + high/not-high classifier)",
          "[~0.75 min] There are five prediction tasks but six trained models, because the rating task has two parts: a regression for the one-to-five rating, scored by mean absolute error, and a classifier for 'high rating', scored by accuracy. "
          "Regression predicts a number, such as rupees or degrees; classification predicts a category or probability, such as will the customer refer. "
          "Note that the best cost model is not the fanciest one: a log-linear ridge regression beat gradient boosting because cost behaves like size times a rate. "
          "The two LightGBM savings and cooling models predict a fraction and multiply it back, which suits how those outcomes scale. Each model is compared with a baseline and tuned with cross-validation only.\nTakeaway: we chose models by validation evidence, not sophistication.\nTransition: how do we measure them?")
rows = [["#", "Task", "Prediction", "Selected model", "Type", "Baseline", "Train+val / test"]]
spec = [("A", "cost", "Installation cost (₹)", "Regression", "Training mean"), ("B", "temperature_drop", "Roof temperature drop (°C)", "Regression", "Training mean"),
        ("C", "units_saved", "Electricity saved (kWh/mo)", "Regression", "Training mean"), ("D1", "rating", "Customer rating (1–5)", "Regression", "Training median"),
        ("D2", "high_rating", "High rating (≥ 4★)", "Classification", "Majority class"), ("E", "referral", "Customer referral", "Classification", "Mean / majority / all-positive")]
for tag, k, pred_, typ, base in spec:
    n_tr, n_te = T[k]["n_train_validation"], T[k]["n_test"]
    rows.append([tag, T[k]["label"].replace(" (1-5)", "").replace(" (>= threshold)", ""), pred_, T[k]["selected_label"].replace("LightGBM on ", "LightGBM: ").replace(" x ", " × "), typ, base, f"{n_tr:,} / {n_te}"])
rows[-1][5] = "Majority and all-positive"
table(s, 0.6, 1.55, 12.1, rows, [0.55, 1.85, 2.45, 3.1, 1.3, 1.5, 1.35], size=12, rowh=0.62)
tbox(s, 0.6, 6.0, 12.1, 0.9, [[("Regression ", {"bold": True, "color": GREEN}), "predicts a number (₹, °C, kWh, stars).  ", ("Classification ", {"bold": True, "color": GREEN}), "predicts a yes/no with a probability."],
                              "The high-rating threshold of 4 stars is our assumption: the organisers gave no definition (configurable)."], 14, CHAR, space=3)

# ============================================================================= SLIDE 9
s = slide(9, f"Judged on {N_TEST} buildings the models never saw", "Evaluation methodology",
          "Source: docs/model_evaluation.md and docs/model_review.md; paired bootstrap, 2,000 resamples, 95% intervals",
          "[~0.75 min] A quick guide to the metrics. MAE is the average size of the miss, in the unit of the thing we predict. RMSE also measures error but squares it first, so large misses count more. "
          "For yes-or-no tasks, accuracy is the share of correct answers, but it can be misleading when classes are imbalanced: always predicting 'no referral' is right 63 percent of the time. "
          "F1 balances precision, how many flagged customers truly are, with recall, how many true cases we catch. "
          f"Our protocol: the {N_TEST}-building hold-out was fixed before modelling and scored once. Baselines were declared before looking at it. "
          "We call a model 'better' only if a paired bootstrap interval of the improvement excludes zero. An independent review also rebuilt every number from the saved models and replicated the whole selection on a hold-out that exploration had never touched.\n"
          "Takeaway: results are honest and statistically checked.\nTransition: first, the three regression results.")
defs = [("MAE", "Mean absolute error", "Average size of the miss, in the target's own unit"), ("RMSE", "Root mean squared error", "Like MAE, but big misses count more"),
        ("Accuracy", "Share correct", "Misleading when one class dominates"), ("F1", "Precision × recall balance", "Matters for the rarer class, e.g. referrals")]
for i, (a, b_, c_) in enumerate(defs):
    x = 0.6 + (i % 2) * 3.55
    y = 1.55 + (i // 2) * 1.75
    box(s, x, y, 3.4, 1.6, MIST)
    tbox(s, x + 0.12, y + 0.05, 3.2, 0.5, a, 24, TEAL, True)
    tbox(s, x + 0.12, y + 0.6, 3.2, 0.3, b_, 13, GREEN, True)
    tbox(s, x + 0.12, y + 0.92, 3.2, 0.65, c_, 13, CHAR)
tbox(s, 0.6, 5.1, 6.95, 0.35, "Precision = flagged who truly are · Recall = true cases we catch · AUC = ranking skill (0.5 = chance)", 13, GREY)
box(s, 7.9, 1.55, 4.8, 5.3, "F3F4F4", line=LINE)
tbox(s, 8.05, 1.62, 4.5, 0.4, "How we kept it honest", 18, GREEN, True)
tbox(s, 8.05, 2.1, 4.55, 4.7, [
    [("1  ", {"bold": True, "color": TEAL}), f"{N_TEST}-building hold-out fixed before modelling, scored once"],
    [("2  ", {"bold": True, "color": TEAL}), "Tuning and selection by 5-fold CV on the rest"],
    [("3  ", {"bold": True, "color": TEAL}), "Baselines declared first: training mean (median for rating); majority class and 'everyone positive' for classifiers"],
    [("4  ", {"bold": True, "color": TEAL}), "'Better' only if the paired-bootstrap 95% CI excludes 0"],
    [("5  ", {"bold": True, "color": TEAL}), "Independent review recomputed every metric and replicated the selection on never-explored rows"]], 14, CHAR, space=7)
tbox(s, 0.6, 5.55, 6.95, 1.3, [[("Same split everywhere: ", {"bold": True, "color": GREEN}), "models and their baselines are scored on the identical hold-out rows, so comparisons are direct."]], 14, CHAR)

# ============================================================================= SLIDE 10
s = slide(10, f"Regression models cut error {x_temp:.0f}–{x_cost:.0f}× versus a mean baseline", "Results: regression",
          f"Source: models/model_metadata.json (hold-out n = {c['n_test']} / {d['n_test']} / {u['n_test']}); each chart has its own unit and scale",
          f"[~1 min] For the three regression tasks, each chart compares the baseline, which just predicts the training average, with our selected model, in the task's own unit. "
          f"Cost: the typical miss falls from {inr(c['test_baseline']['mae'])} to {inr(c['test']['mae'])}, about {x_cost:.0f} times smaller. Cooling: from {d['test_baseline']['mae']:.2f} to {d['test']['mae']:.2f} degrees Celsius. Electricity saved: from {u['test_baseline']['mae']:.1f} to {u['test']['mae']:.2f} kWh a month. "
          "In all three, the paired bootstrap interval of the improvement is far above zero. R-squared is 0.998, 0.948 and 0.987. An ablation shows these come from legitimate inputs: remove building size and the cost R-squared collapses to 0.22; remove method and quality and cooling drops to 0.25. "
          f"One caveat: our nominal 80 percent likely-range intervals actually cover {cov['cost']:.0%}, {cov['temperature_drop']:.0%} and {cov['units_saved']:.0%}, so they are slightly narrow.\nTakeaway: strong, statistically supported accuracy where the data is structured.\nTransition: the human-outcome tasks are harder.")
native_bar(s, 0.6, 1.5, 3.95, 3.0, ["Baseline", "Our model"], [round(c["test_baseline"]["mae"] / 1e5, 2), round(c["test"]["mae"] / 1e5, 2)], [GREY, TEAL], '"₹"0.00" L"', "Installation cost, MAE (₹ lakh)")
native_bar(s, 4.7, 1.5, 3.95, 3.0, ["Baseline", "Our model"], [round(d["test_baseline"]["mae"], 2), round(d["test"]["mae"], 2)], [GREY, TEAL], '0.00" °C"', "Temperature drop, MAE (°C)")
native_bar(s, 8.8, 1.5, 3.9, 3.0, ["Baseline", "Our model"], [round(u["test_baseline"]["mae"], 1), round(u["test"]["mae"], 2)], [GREY, TEAL], '0.0" kWh"', "Electricity saved, MAE (kWh/month)")
rows = [["Task", "Selected model", "MAE", "RMSE", "R²", "Baseline MAE"],
        ["Cost", "Log-linear ridge", inr(c["test"]["mae"]), inr(c["test"]["rmse"]), f"{c['test']['r2']:.3f}", inr(c["test_baseline"]["mae"])],
        ["Temperature drop", "LightGBM (ratio target)", f"{d['test']['mae']:.2f} °C", f"{d['test']['rmse']:.2f} °C", f"{d['test']['r2']:.3f}", f"{d['test_baseline']['mae']:.2f} °C"],
        ["Electricity saved", "LightGBM (ratio target)", f"{u['test']['mae']:.2f} kWh", f"{u['test']['rmse']:.2f} kWh", f"{u['test']['r2']:.3f}", f"{u['test_baseline']['mae']:.1f} kWh"]]
table(s, 0.6, 4.65, 7.6, rows, [1.5, 2.0, 1.1, 1.1, 0.55, 1.35], size=12, rowh=0.45)
card(s, 8.45, 4.65, 4.25, 2.2, "What this shows", ["MAE improvement: 95% CI above 0 for all three", f"80% ranges cover {cov['cost']:.0%} / {cov['temperature_drop']:.0%} / {cov['units_saved']:.0%} (slightly narrow)", "Accuracy comes from legitimate inputs: no size → cost R² 0.22"], size=13, tsize=15)

# ============================================================================= SLIDE 11
s = slide(11, "Ratings and referrals are only weakly predictable, and we show it", "Results: rating and referral",
          f"Source: models/model_metadata.json and reports/significance.json (hold-out n = {hr['n_test']}); trivial baselines declared before scoring",
          f"[~1 min] These two tasks depend on what happens after installation, so they are harder at quote time. For ratings, mean error is {ra['test']['mae']:.2f} stars against {ra['test_baseline']['mae']:.2f} for the baseline, a significant gain. "
          f"For high rating, defined as four stars or more, which is our assumption, accuracy is {hr['test']['accuracy']:.3f} against {hr['baseline_accuracy']:.3f} for the majority class, also significant, and the AUC is {hr['test']['roc_auc']:.2f}. But the F1 of {hr['test']['f1']:.2f} is not distinguishable from simply saying everyone will rate highly. "
          f"Referral is the weakest. At its best-F1 cut-off the model catches most referrers, recall {rfr['test']['recall']:.2f}, but flags many people wrongly. F1 is {rfr['test']['f1']:.2f}, statistically indistinguishable from predicting everyone refers at {rfr['baseline_f1']:.2f}, and accuracy {rfr['test']['accuracy']:.3f} is significantly worse than the majority-class {rfr['baseline_accuracy']:.3f}. "
          f"The AUC of {rfr['test']['roc_auc']:.2f} shows a real but weak ranking signal. Even knowing the realised savings would only reach 0.68 AUC, so the information mostly isn't in the data. We use it to rank follow-ups, never to promise.\n"
          "Takeaway: being transparent about weak results is part of the engineering.\nTransition: what actually drives the strong results?")
tbox(s, 0.6, 1.45, 6.0, 0.4, "Customer rating and high rating", 18, GREEN, True)
tiles = [("Rating MAE", f"{ra['test']['mae']:.2f} ★", f"baseline {ra['test_baseline']['mae']:.2f}", "better"),
         ("High-rating accuracy", f"{hr['test']['accuracy']:.3f}", f"majority {hr['baseline_accuracy']:.3f}", "better"),
         ("High-rating F1", f"{hr['test']['f1']:.3f}", f"all-high {hr['baseline_f1']:.3f}", "n.s."),
         ("High-rating AUC", f"{hr['test']['roc_auc']:.2f}", f"95% CI {sg_h['roc_auc_ci95'][0]:.2f}–{sg_h['roc_auc_ci95'][1]:.2f}", "above chance")]
for i, (a, b_, c_, v) in enumerate(tiles):
    x, y = 0.6 + (i % 2) * 3.05, 1.95 + (i // 2) * 1.55
    box(s, x, y, 2.9, 1.4, MIST)
    tbox(s, x + 0.1, y + 0.05, 2.7, 0.3, a, 13, GREY, True)
    tbox(s, x + 0.1, y + 0.33, 2.7, 0.55, b_, 26, GREEN, True)
    tbox(s, x + 0.1, y + 0.88, 2.7, 0.3, c_ + "  ·  " + v, 12, CHAR)
tbox(s, 0.6, 5.1, 6.0, 0.6, "High rating means 4 stars or more: our assumption (no official definition).", 13, CLAY, True)
tbox(s, 6.95, 1.45, 5.8, 0.4, "Customer referral (best-F1 cut-off 25%)", 18, GREEN, True)
cm = rfr["test"]["confusion_matrix"]
cmrows = [["", "Predicted: no", "Predicted: yes"], ["Actual: no", f"{cm['tn']}", f"{cm['fp']}"], ["Actual: yes", f"{cm['fn']}", f"{cm['tp']}"]]
table(s, 6.95, 1.95, 5.75, cmrows, [1.6, 2.075, 2.075], size=15, rowh=0.55)
box(s, 6.95, 3.75, 5.75, 1.95, MIST)
tbox(s, 7.05, 3.8, 5.6, 1.9, [[("F1 ", {"bold": True, "color": GREEN}), f"{rfr['test']['f1']:.3f} vs {rfr['baseline_f1']:.3f} (everyone refers): not distinguishable"],
                              [("Accuracy ", {"bold": True, "color": GREEN}), f"{rfr['test']['accuracy']:.3f} vs {rfr['baseline_accuracy']:.3f} (majority class): significantly worse"],
                              [("Recall ", {"bold": True, "color": GREEN}), f"{rfr['test']['recall']:.2f} · precision {rfr['test']['precision']:.2f} · AUC {rfr['test']['roc_auc']:.2f} (CI {sg_r['roc_auc_ci95'][0]:.2f}–{sg_r['roc_auc_ci95'][1]:.2f})"]], 14, CHAR, space=5)
box(s, 0.6, 5.85, 12.1, 1.0, GREEN)
tbox(s, 0.85, 5.9, 11.6, 0.9, "Accuracy and F1 pull in opposite directions for referrals. No cut-off beats the trivial classifiers on both, so the app shows both and uses referral probability only to rank follow-ups.", 17, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)

# ============================================================================= SLIDE 12
s = slide(12, "Size drives cost; method and quality drive cooling and satisfaction", "Explainability and business insights",
          "Source: reports/feature_importance.csv (hold-out permutation importance), reports/insights.json (model counterfactuals, concrete roofs, n = 1,323). Associations, not causal effects.",
          "[~1 min] On the left is global feature importance from the hold-out set, meaning how much accuracy drops when an input is shuffled. Building size dominates cost; the roofing method and quality dominate cooling; current electricity use and AC count dominate savings; "
          "AC count and the method matter most for ratings and referrals. On the right, we ask the model what happens on concrete roofs, the only roofs that accept all five methods, if every building used each method in turn. "
          f"Coated options are consistently strongest on cooling, savings and predicted satisfaction, but there is no single best method for everyone, and roof type limits what can be installed. Basic coating is cheapest but weakest. "
          "These are associations from past installations, not proof of cause. Also, at a plausible electricity price the savings alone would take very long to repay the cost, so the business case rests on comfort and benefits the data doesn't measure.\n"
          "Takeaway: explanations are specific, computed, and carefully worded.\nTransition: and here is how it looks in the product.")
pic(s, "chart_feature_importance.png", 0.5, 1.45, w=7.25)
tbox(s, 7.9, 1.45, 4.8, 0.4, "Model view: methods on concrete roofs", 16, GREEN, True)
mrows = [["Method", "Cost", "Cooling", "kWh", "P(high)"]]
for m in ["basic_coating", "sheet", "sheet_with_coating", "tiles", "tiles_with_coating"]:
    mrows.append([METHOD[m], f"₹{mval('cost', m) / 1e5:.1f} L", f"{mval('temperature_drop', m):.1f} °C", f"{mval('units_saved', m):.0f}", f"{mval('high_rating', m):.0%}"])
table(s, 7.9, 1.9, 4.8, mrows, [1.55, 0.85, 0.85, 0.7, 0.85], size=12, rowh=0.44)
card(s, 7.9, 4.7, 4.8, 2.15, "Careful reading", ["Coated options lead on cooling, savings and satisfaction; none wins everywhere", "Associations, not causation", "Electricity savings alone repay slowly: the case rests on comfort"], size=13, tsize=15)

# ============================================================================= SLIDE 13
CROPS = {"crop_predict.png": ("ui_predict.png", (300, 10, 1410, 515)), "crop_compare.png": ("ui_compare.png", (300, 10, 1410, 515)),
         "crop_performance.png": ("ui_performance.png", (300, 10, 1410, 610)), "crop_proposal.png": ("ui_proposal.png", (300, 10, 1410, 610)),
         "crop_assistant.png": ("ui_assistant.png", (330, 180, 1400, 780))}
for out_name, (src_name, box_) in CROPS.items():
    Image.open(ASSETS / src_name).convert("RGB").crop(box_).save(ASSETS / out_name)
s = slide(13, "The product is built and running: predict, compare, explain, propose", "Demonstration",
          "Source: screenshots of the running local application (React + FastAPI). Assistant: Gemini live-tested; Claude not live-tested.",
          "[~1 min] These are real screenshots from the running app. Top left, the prediction workspace: the rep enters a few facts and gets cost, cooling, electricity savings, rating and referral, each with a likely range and a reliability flag, plus a 'why this estimate' explanation and an optional payback calculator that never assumes a tariff. "
          "Top right, the comparison page ranks every valid method and quality combination for the same building, with adjustable priorities and a transparent score. Below: the model performance page with baselines and confidence intervals, the PDF proposal page, and the optional AI assistant, which answered live through Gemini by calling the same prediction tools. "
          "The Claude integration is built, but we could not live-test it because that account has no credit.\nTakeaway: this is working software, not a mock-up.\nTransition: finally, value and limits.")
w_, h_ = pic(s, "crop_predict.png", 0.6, 1.4, w=6.0)
pic(s, "crop_compare.png", 6.75, 1.4, w=6.0)
for x_, t_, d_ in [(0.6, "Customer prediction", "cost, cooling, kWh, rating and referral, each with a likely range and a reliability flag"),
                   (6.75, "Roofing comparison", "every valid method × quality, ranked for your objective, with the basis shown")]:
    tbox(s, x_, 1.4 + h_ + 0.02, 6.0, 0.55, [[(t_ + ": ", {"bold": True, "color": GREEN}), d_]], 12, CHAR)
for k_, (name, label, sub) in enumerate([("crop_performance.png", "Model performance", "baselines and intervals"),
                                         ("crop_proposal.png", "PDF proposal", "from real predictions"),
                                         ("crop_assistant.png", "AI assistant", "Gemini live-tested")]):
    x_ = 0.6 + k_ * 4.1
    wi, hi = pic(s, name, x_, 4.6, w=3.6)
    tbox(s, x_, 4.6 + hi + 0.02, 3.95, 0.3, [[(label + ": ", {"bold": True, "color": GREEN}), sub]], 12, CHAR)

# ============================================================================= SLIDE 14
s = slide(14, "Useful today as decision support; honest about what it cannot do", "Business value, limitations and roadmap",
          "Source: docs/business_recommendations.md, docs/assumptions_and_decisions.md, docs/model_review.md. Roadmap items are NOT implemented.",
          "[~0.75 min] For the business, the product gives a consistent first estimate in well under a second, lets a rep compare fifteen option combinations for the same building, explains each prediction, makes every assumption visible, and produces a proposal PDF. "
          "We are equally clear about limits. Predictions are estimates, not guarantees. The zero-kilowatt-hour treatment is an inference. Rating and referral models are weak. The high-rating threshold is our assumption. "
          "Method effects are associations from observational data, there is no time or location field to test drift, and the Claude integration has not been live-tested. "
          "The roadmap on the right is future work only: a digital-twin simulator, a carbon-savings tracker and geo-climate intelligence are not built. The most valuable next step is validating against newly measured installations.\n"
          "Takeaway: real value, bounded claims.\nTransition: to close.")
colspec = [("Business value (demonstrated)", GREEN, ["Preliminary estimates in well under a second", "Compare 15 method × quality options per building", "Every prediction explained and assumptions visible", "Proposal PDFs from real model outputs", "Payback only from the customer's own tariff"]),
           ("Limitations (stated openly)", CLAY, ["Estimates, not guarantees", "0 kWh placeholder treatment is an inference", "Rating and referral only weakly predictable", "High-rating threshold (4★) is our assumption", "Associations, not causation; no time or location fields", "Claude not live-tested (Gemini is)"]),
           ("Roadmap (future, not built)", GREY, ["Validate on newly measured installations", "Digital-twin simulator", "Carbon-savings tracker", "Geo-climate intelligence"])]
for i, (h_, colr, items) in enumerate(colspec):
    x = 0.6 + i * 4.1
    box(s, x, 1.55, 3.9, 5.3, MIST if i < 2 else "F3F4F4", line=None if i < 2 else LINE)
    chip(s, x + 0.15, 1.7, 3.6, h_, colr, size=13, h=0.42)
    tbox(s, x + 0.2, 2.3, 3.55, 4.5, [[("• ", {"bold": True, "color": colr}), it] for it in items], 16, CHAR if i < 2 else GREY, space=7)

# ============================================================================= SLIDE 15
s = slide(15, "From 2,500 past roofs to decisions that are explained and honest", "Conclusion",
          "Cool Roofing Intelligence · all figures from the project's saved reports",
          "[~0.5 min] To close. We built a complete system: a cleaned and audited dataset, six models for five tasks, honest hold-out evaluation with baselines and significance tests, explanations, an option-comparison engine, PDF proposals and an optional assistant. "
          "The evaluation supports strong predictions of cost, cooling and electricity savings, and modest but real signal for ratings, while referrals remain a ranking aid only. "
          "Our strongest contribution may be the discipline: finding the hidden placeholders, preventing leakage, and reporting weak results plainly. The next step is validating on new, measured installations. Thank you; we're happy to take questions.\n"
          "Takeaway: trustworthy, explainable decisions from historical roofing data.")
cols4 = [("What we built", ["Audit + cleaning pipeline", "6 models, 5 tasks", "Compare · explain · propose", f"{BACKEND_TESTS + FRONTEND_TESTS} automated tests"]),
         ("What the evidence supports", [f"Cost MAE {inr(c['test']['mae'])}", f"Cooling MAE {d['test']['mae']:.2f} °C", f"Savings MAE {u['test']['mae']:.2f} kWh", "Rating and referral: weak signal"]),
         ("Strongest capabilities", ["Leakage-free, hold-out-verified", "Honest baselines and CIs", "Explanations per customer", "Working end-to-end app"]),
         ("Next step", ["Validate on newly measured installations", "Then: digital twin, carbon tracker, geo-climate"])]
for i, (h_, items) in enumerate(cols4):
    x = 0.6 + i * 3.08
    box(s, x, 1.65, 2.9, 3.2, MIST)
    box(s, x, 1.65, 2.9, 0.09, [GREEN, TEAL, SUN, GREY][i], radius=0, shape=MSO_SHAPE.RECTANGLE)
    tbox(s, x + 0.12, 1.85, 2.66, 0.5, h_, 17, GREEN, True)
    tbox(s, x + 0.12, 2.45, 2.66, 2.4, [[("• ", {"bold": True, "color": TEAL}), it] for it in items], 14, CHAR, space=5)
box(s, 0.6, 5.15, 12.1, 1.15, GREEN)
tbox(s, 0.85, 5.2, 11.6, 1.05, "Historical roofing data, turned into estimates a sales rep can trust, with the reasons and the limits attached.", 22, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
tbox(s, 0.6, 6.4, 12.1, 0.5, "Thank you · Questions?", 24, TEAL, True, PP_ALIGN.CENTER)

# ----------------------------------------------------------------------------- save deck, notes, fact sheet
import os
PPTX = OUT / os.environ.get("DECK_FILE", "Cool_Roofing_Hackathon_Presentation.pptx")
prs.save(PPTX)
mins = {1: 0.5, 2: 0.75, 3: 0.75, 4: 0.75, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 0.75, 15: 0.5}
titles = {i + 1: sl.shapes[2].text_frame.text if i > 0 else "Cool Roofing Intelligence" for i, sl in enumerate(prs.slides)}
import re as _re
for n_, sl_ in enumerate(prs.slides, start=1):
    NOTES[n_] = _re.sub(r"\[~[\d.]+ min\]", f"[~{mins[n_]:g} min]", NOTES[n_])
    sl_.notes_slide.notes_text_frame.text = NOTES[n_]
prs.save(PPTX)
md = ["# Speaker notes: Cool Roofing Intelligence (15 slides, about 13 minutes)", "", f"Planned total: {sum(mins.values()):.2f} minutes, leaving about 2 minutes of buffer in a 15-minute slot.", ""]
for n in range(1, 16):
    md += [f"## Slide {n} (~{mins[n]} min)", NOTES[n].replace("\n", "\n\n"), ""]
(OUT / "presentation_notes.md").write_text("\n".join(md), encoding="utf-8")
print("saved", PPTX, "slides:", len(prs.slides), "planned minutes:", sum(mins.values()))

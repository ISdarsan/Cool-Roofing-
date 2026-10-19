"""8-slide hackathon deck. Every number is read from models/ and reports/ at build time."""
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
N_ROWS, N_TEST, N_TV = META["data"]["n_rows"], META["data"]["n_test_buildings"], META["data"]["n_train_validation_buildings"]
BACKEND_TESTS, FRONTEND_TESTS = 153, 30        # last full runs in this project (pytest / vitest)
c, d, u, ra, hr, rfr = (T[k] for k in ("cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral"))
sg_h, sg_r = hr["significance"], rfr["significance"]
z, tsent = SUMMARY["zero_consumption_placeholders"], SUMMARY["temperature_sentinel_rows"]
miss_cells = sum(x["before"] for x in PROFILE["missingness"] if x["before"] > 0)
n_cols = 15
assert N_ROWS == 2500 and tsent["before"] + tsent["after"] == 58 and z["before"] + z["after"] == 48 and SUMMARY["negative_savings_rows"] == 167
drop_lo, drop_hi = SUMMARY["valid_temperature_drop_range"]

# ----------------------------------------------------------------------------- style
GREEN, TEAL, SUN, CLAY = "0F4C3A", "0E7C86", "E8A33D", "B0412C"
CHAR, GREY, MIST, WHITE, LINE = "1F2A2E", "5F6E73", "EAF2EF", "FFFFFF", "D5DEDB"
FONT = "Calibri"
rgb = lambda h: RGBColor.from_string(h)
inr = lambda v: "₹" + f"{v:,.0f}"
plt.rcParams.update({"font.family": ["Calibri", "Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8A979B", "axes.labelcolor": "#1F2A2E", "xtick.color": "#4B5A5F", "ytick.color": "#4B5A5F",
                     "axes.titleweight": "bold", "axes.titlesize": 12, "axes.grid": True, "grid.color": "#E3EAE7", "figure.dpi": 220})


def save(fig, name):
    fig.savefig(ASSETS / name, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)


# ----------------------------------------------------------------------------- charts at their exact placement size (so fonts stay readable)
def hist(ax, h, color, xlabel, scale=1.0, zero=False):
    e = np.array(h["edges"]) / scale
    ax.bar(e[:-1], h["counts"], width=np.diff(e), align="edge", color="#" + color, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(xlabel); ax.set_ylabel("buildings")
    if zero:
        ax.axvline(0, color="#" + CLAY, ls="--", lw=1.6)


fig, axs = plt.subplots(1, 3, figsize=(12.0, 2.3), constrained_layout=True)
hist(axs[0], PROFILE["targets"]["cost"], TEAL, "installation cost (₹ lakh)", 1e5)
hist(axs[1], PROFILE["targets"]["temperature_drop"], SUN, "roof temperature drop (°C)")
hist(axs[2], PROFILE["targets"]["units_saved"], GREEN, "electricity saved (kWh / month)", zero=True)
axs[0].set_title(f"Cost is right-skewed (n = {PROFILE['targets']['cost']['n']:,})")
axs[1].set_title(f"Cooling: {drop_lo:.1f}–{drop_hi:.1f} °C (n = {PROFILE['targets']['temperature_drop']['n']:,})")
axs[2].set_title(f"Savings: some ≤ 0, dashed (n = {PROFILE['targets']['units_saved']['n']:,})")
save(fig, "d8_target_distributions.png")

anom = [("Usage went UP: kept (real)", SUMMARY["negative_savings_rows"], TEAL), ("Missing input cells", miss_cells, GREY),
        ("120 °C readings: invalid", tsent["before"] + tsent["after"], CLAY), ("0 kWh readings: suspected", z["before"] + z["after"], SUN)]
fig, ax = plt.subplots(figsize=(4.4, 2.6), constrained_layout=True)
ax.barh([a[0] for a in anom], [a[1] for a in anom], color=["#" + a[2] for a in anom])
for i, a in enumerate(anom):
    ax.text(a[1] + 3, i, str(a[1]), va="center", fontweight="bold")
ax.invert_yaxis(); ax.set_xlim(0, 230); ax.set_xticks([0, 100, 200]); ax.set_xlabel("records / cells"); ax.set_title("What the audit found"); ax.grid(axis="y", visible=False)
save(fig, "d8_anomalies.png")

tb = PROFILE["temperature_before_after_cleaning"]
fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
for key, col, lab in (("raw_before", CLAY, "raw"), ("clean_before", TEAL, "cleaned")):
    e = np.array(tb[key]["edges"]); ax.bar(e[:-1], tb[key]["counts"], width=np.diff(e), align="edge", color="#" + col, label=lab)
ax.set_xlabel("roof temp. before (°C)"); ax.set_ylabel("buildings"); ax.set_title("120 °C spike removed"); ax.legend(frameon=False, loc="upper right")
save(fig, "d8_temperature_cleaning.png")

# ----------------------------------------------------------------------------- pptx helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
NOTES: dict[int, str] = {}
MINS = {1: 1.25, 2: 1.5, 3: 2.0, 4: 1.25, 5: 1.25, 6: 2.5, 7: 1.75, 8: 1.0}


def tbox(s, x, y, w, h, paras, size=16, color=CHAR, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=0):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = anchor
    paras = [paras] if isinstance(paras, (str, tuple)) else paras
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.space_after = Pt(space)
        for r in (p if isinstance(p, list) else [p]):
            txt, fmt = (r, {}) if isinstance(r, str) else r
            run = para.add_run(); run.text = txt
            f = run.font; f.name = FONT; f.size = Pt(fmt.get("size", size)); f.bold = fmt.get("bold", bold); f.color.rgb = rgb(fmt.get("color", color))
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


def card(s, x, y, w, h, title, body, fill=MIST, tcolor=GREEN, size=13, tsize=15, line=None):
    box(s, x, y, w, h, fill, line)
    tbox(s, x + 0.12, y + 0.07, w - 0.24, 0.4, title, tsize, tcolor, True)
    tbox(s, x + 0.12, y + 0.47, w - 0.24, h - 0.5, body, size, CHAR, space=2)


def chip(s, x, y, w, text, fill, color=WHITE, size=13, h=0.38):
    sh = box(s, x, y, w, h, fill, radius=0.3)
    tf = sh.text_frame; tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = True; r.font.name = FONT; r.font.color.rgb = rgb(color)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sh


def arrow(s, x, y, w, h, kind=MSO_SHAPE.RIGHT_ARROW, fill=TEAL):
    sh = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill); sh.line.fill.background(); sh.shadow.inherit = False
    return sh


def pic(s, name, x, y, w=None, h=None):
    iw, ih = Image.open(ASSETS / name).size
    if w and not h:
        h = w * ih / iw
    elif h and not w:
        w = h * iw / ih
    s.shapes.add_picture(str(ASSETS / name), Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def slide(n, title, kicker, source, notes):
    s = prs.slides.add_slide(BLANK)
    NOTES[n] = notes
    box(s, 0, 0, 13.333, 0.12, GREEN, radius=0, shape=MSO_SHAPE.RECTANGLE)
    tbox(s, 0.6, 0.26, 9, 0.3, kicker.upper(), 12, TEAL, True)
    tbox(s, 0.6, 0.5, 12.1, 0.75, title, 28, GREEN, True)
    tbox(s, 0.6, 7.08, 10.8, 0.3, source, 11, GREY)
    tbox(s, 11.9, 7.08, 0.85, 0.3, f"{n} / 8", 11, GREY, align=PP_ALIGN.RIGHT)
    return s


def table(s, x, y, rows, colw, size=12, rowh=0.42, bold_first=True):
    shp = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(sum(colw)), Inches(rowh * len(rows)))
    tb_ = shp.table
    for j, cw in enumerate(colw):
        tb_.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        tb_.rows[i].height = Inches(rowh)
        for j, val in enumerate(r):
            cell = tb_.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08); cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid(); cell.fill.fore_color.rgb = rgb(GREEN if i == 0 else (MIST if i % 2 == 0 else WHITE))
            tf = cell.text_frame; tf.word_wrap = True
            run = tf.paragraphs[0].add_run(); run.text = str(val)
            run.font.size = Pt(size); run.font.name = FONT; run.font.bold = (i == 0) or (bold_first and j == 0)
            run.font.color.rgb = rgb(WHITE if i == 0 else CHAR)
    return shp


def native_bar(s, x, y, w, h, vals, numfmt, title):
    cd = CategoryChartData(); cd.categories = ["Baseline", "Our model"]; cd.add_series("MAE", vals)
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.has_legend = False; ch.has_title = True; ch.chart_title.text_frame.text = title
    r = ch.chart_title.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(13); r.font.bold = True; r.font.name = FONT; r.font.color.rgb = rgb(CHAR)
    pl = ch.plots[0]; pl.gap_width = 60; pl.has_data_labels = True
    dl = pl.data_labels; dl.number_format = numfmt; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(13); dl.font.bold = True; dl.font.name = FONT
    for i, col in enumerate([GREY, TEAL]):
        pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(col)
    va = ch.value_axis; va.has_major_gridlines = False; va.visible = False; va.minimum_scale = 0
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(12); ca.tick_labels.font.name = FONT; ca.format.line.color.rgb = rgb(LINE)


# ============================================================================= SLIDE 1: Introduction
s = prs.slides.add_slide(BLANK)
NOTES[1] = ("[~1.25 min] Good [morning/afternoon]. Our project is Cool Roofing Intelligence. Cool roofing means heat-reflective roofing: coatings, sheets or tiles that keep the roof surface cooler, "
            f"which should make the building more comfortable and cut cooling electricity. In our data, buildings across India that received these roofs cooled by about {drop_lo:.0f} to {drop_hi:.0f} degrees Celsius. "
            "The business problem is that a company wants to tell a new customer, before the quote, what it will cost and what they will get. Our objective is to turn historical installation records into that answer. "
            f"In one sentence: we built a tested web system that predicts cost, cooling, electricity savings, rating and referral from {N_ROWS:,} past installations, compares roofing options, and explains every estimate.\n"
            "Takeaway: predictive, explainable decision support for cool roofing.\nTransition: let me state the five problems precisely.")
box(s, 0, 0, 13.333, 0.12, GREEN, radius=0, shape=MSO_SHAPE.RECTANGLE)
box(s, 0, 0.12, 4.7, 7.38, GREEN, radius=0, shape=MSO_SHAPE.RECTANGLE)
box(s, 1.1, 3.55, 2.5, 2.3, "F4F1E8", radius=0.02)
arrow(s, 0.85, 2.3, 3.0, 1.3, MSO_SHAPE.ISOSCELES_TRIANGLE, TEAL)
for i in range(3):
    box(s, 1.45 + i * 0.72, 4.3, 0.48, 0.7, "BBD3CC", radius=0.04)
box(s, 3.3, 1.0, 0.85, 0.85, SUN, radius=0.5, shape=MSO_SHAPE.OVAL)
for k in range(8):
    ang = k * np.pi / 4
    box(s, 3.725 + 0.65 * np.cos(ang) - 0.065, 1.425 + 0.65 * np.sin(ang) - 0.065, 0.13, 0.13, SUN, radius=0.5, shape=MSO_SHAPE.OVAL)
tbox(s, 0.4, 6.4, 4.0, 0.5, "Cooler roofs · lower bills · clearer quotes", 14, "BBD3CC", True)
tbox(s, 5.2, 0.55, 7.5, 0.4, "HACKATHON PROJECT", 13, TEAL, True)
tbox(s, 5.2, 0.9, 7.6, 1.5, ["Cool Roofing Intelligence"], 38, GREEN, True)
tbox(s, 5.2, 1.85, 7.5, 0.7, "Cost, comfort and energy-savings predictions before the first quote", 18, CHAR)
box(s, 5.25, 2.65, 7.4, 0.04, SUN, radius=0, shape=MSO_SHAPE.RECTANGLE)
card(s, 5.2, 2.85, 7.5, 1.1, "What is cool roofing?", "Heat-reflective roofing (coatings, sheets or tiles) that lowers roof-surface temperature, so buildings stay cooler and need less cooling electricity.", size=14, tsize=15)
card(s, 5.2, 4.05, 3.65, 1.45, "Objective and relevance", "Give roofing companies data-backed answers before quoting: what will it cost, and what will the customer gain?", size=13, tsize=15)
card(s, 9.05, 4.05, 3.65, 1.45, "The solution", f"One tested web system that learns from {N_ROWS:,} past installations to predict, compare and explain.", size=13, tsize=15, fill="DDEEF0", tcolor=TEAL)
for i, (big, small) in enumerate([(f"{N_ROWS:,}", "historical installations"), ("5 tasks", "6 trained models"), ("8 pages", "web app with PDF proposals")]):
    x = 5.2 + i * 2.55
    box(s, x, 5.7, 2.4, 1.1, GREEN)
    tbox(s, x + 0.05, 5.74, 2.3, 0.55, big, 24, SUN, True, PP_ALIGN.CENTER)
    tbox(s, x + 0.05, 6.3, 2.3, 0.45, small, 12, WHITE, False, PP_ALIGN.CENTER)
s.notes_slide.notes_text_frame.text = NOTES[1]

# ============================================================================= SLIDE 2: Problem statement
s = slide(2, "Five questions a roofing customer asks, and the data to answer them", "Problem statement",
          "Source: cool_roofing_problem.md; targets as defined in docs/data_dictionary.md",
          "[~1.5 min] The organisers' problem statement asks for five predictions for a new building. One: the installation cost in rupees. Two: how much the roof temperature drops, which we compute as temperature before minus after. "
          "Three: the monthly electricity saved in kilowatt-hours, units before minus units after. Four: the customer's rating from one to five. Five: whether the customer will refer the company. "
          "Why does historical data help? Each past installation records what was known at quote time, such as building size, roof type, method and quality, and what actually happened afterwards. "
          "A model can learn that pattern and give a consistent, explained estimate instead of relying on experience alone. We also had to check whether the data could be trusted, which I will show next.\n"
          "Takeaway: five well-defined prediction tasks, supported by real historical outcomes.\nTransition: first, the dataset.")
probs = [("1", "Installation cost", "Total cost of the job in ₹", "Regression"), ("2", "Roof temperature drop", "Temperature before − after (°C)", "Regression"),
         ("3", "Electricity savings", "Monthly kWh before − after", "Regression"), ("4", "Customer rating", "1–5 stars; high = 4★ or more (our assumption)", "Regression + classification"),
         ("5", "Customer referral", "Did the customer refer the company? (yes/no)", "Classification")]
for i, (n_, t_, def_, kind) in enumerate(probs):
    x = 0.6 + i * 2.45
    box(s, x, 1.55, 2.3, 3.3, MIST)
    box(s, x, 1.55, 2.3, 0.1, [TEAL, SUN, GREEN, CLAY, GREY][i], radius=0, shape=MSO_SHAPE.RECTANGLE)
    tbox(s, x + 0.12, 1.75, 0.6, 0.6, n_, 30, [TEAL, "B07A10", GREEN, CLAY, GREY][i], True)
    tbox(s, x + 0.12, 2.4, 2.1, 0.8, t_, 18, GREEN, True)
    tbox(s, x + 0.12, 3.25, 2.1, 0.95, def_, 14, CHAR)
    chip(s, x + 0.12, 4.3, 2.06, kind, [TEAL, TEAL, TEAL, CLAY, GREY][i], size=11, h=0.4)
box(s, 0.6, 5.1, 12.1, 1.75, GREEN)
tbox(s, 0.9, 5.2, 11.5, 0.4, "Why historical installation data helps", 17, SUN, True)
tbox(s, 0.9, 5.65, 11.5, 1.2, [[("• ", {"color": SUN, "bold": True}), f"Each of the {N_ROWS:,} past jobs records what was known at quote time and what happened afterwards."],
                               [("• ", {"color": SUN, "bold": True}), "Models learn how size, roof, method and quality moved cost, cooling, savings and customer feedback."],
                               [("• ", {"color": SUN, "bold": True}), "Result: consistent, explained estimates instead of guesswork, with limits stated."]], 15, WHITE, space=3)

# ============================================================================= SLIDE 3: Dataset evaluation and cleaning
s = slide(3, "The audit found hidden placeholders; no data was deleted", "Dataset evaluation and cleaning",
          "Source: dataset/cool_roofing_data.csv (read-only); reports/data_quality_report.md, cleaning_audit.csv; charts from reports/data_profile.json",
          f"[~2 min] The dataset has {N_ROWS:,} records and {n_cols} columns. Inputs known before installation are location, size, age, existing roof, proposed method and quality, starting roof temperature, number of air conditioners and current monthly electricity use. "
          "The outcomes are cost, temperatures and electricity after installation, the rating and the referral. The bottom charts show the exploration: cost is right-skewed and scales with size; valid cooling is between about two and thirteen degrees; "
          "and electricity savings track current usage, with some buildings at or below zero. "
          f"The audit found two silent problems. Exactly 120 degrees appears as a roof temperature in 58 records, far from every other reading, so it is invalid. Exactly zero kilowatt-hours appears 48 times while one to three never occurs, "
          "and those rows are physically inconsistent, so we treated them as placeholders. That second one is an inference, a strong one, not a confirmed fact. "
          f"We set both to missing instead of guessing, and exclude a record only from the task that needs that value; nothing was deleted outright. We also kept {SUMMARY['negative_savings_rows']} buildings whose electricity use actually went up, because those are real adverse outcomes. "
          f"The cleaning had a measurable effect: electricity-savings error on unseen data fell from 4.45 to {u['test']['mae']:.2f} kilowatt-hours. There are also {miss_cells} missing input cells, which are filled using training data only.\n"
          "Takeaway: the data was audited, and every cleaning decision is documented and evidence-based.\nTransition: next, the models.")
card(s, 0.6, 1.45, 3.75, 2.9, f"{N_ROWS:,} records · {n_cols} columns",
     [[("Inputs: ", {"bold": True, "color": GREEN}), "location, size, age, existing roof, method, quality, start temp., AC units, current kWh"],
      [("Targets: ", {"bold": True, "color": GREEN}), "cost, temperature drop, kWh saved, rating, referral"],
      [("Decisions: ", {"bold": True, "color": GREEN}), f"120 °C → missing; 0 kWh → missing (inferred); {SUMMARY['negative_savings_rows']} increases kept; {miss_cells} gaps imputed in training only; 0 rows deleted"]],
     size=13, tsize=15)
pic(s, "d8_anomalies.png", 4.45, 1.45, h=2.9)
pic(s, "d8_temperature_cleaning.png", 9.15, 1.45, h=2.9)
pic(s, "d8_target_distributions.png", 0.6, 4.45, w=12.1)
tbox(s, 0.6, 6.75, 12.1, 0.3, f"Effect of cleaning: electricity-savings hold-out MAE 4.45 → {u['test']['mae']:.2f} kWh. Zero-kWh treatment is an inference; the 120 °C finding is confirmed.", 12, CLAY, True)

# ============================================================================= SLIDE 4: Models used
s = slide(4, "Five tasks, six models, each chosen by validation evidence", "Models used",
          "Source: models/model_metadata.json (selected by cross-validated score; baselines are never selectable)",
          "[~1.25 min] There are five prediction tasks and six trained models, because customer rating is treated two ways: a numerical rating, and a high-or-not classification. "
          "We use regression when the answer is a number, such as rupees, degrees or kilowatt-hours, and classification when the answer is a category, such as will the customer refer, or is the rating high. "
          "For cost we used a log-linear ridge regression, because cost behaves like size times a rate, and that simple model actually beat gradient boosting. "
          "For cooling and savings we used LightGBM, a gradient-boosted tree model, to predict a fraction and multiply it back by the starting temperature or usage. "
          "Rating and high rating use LightGBM too, and referral uses logistic regression. Every model was compared against baselines and simpler alternatives using cross-validation only, never the test set.\n"
          "Takeaway: simple models where the data is simple, with every choice backed by validation.\nTransition: and the technology that delivers them.")
mrows = [["Task", "Output", "Selected model", "Why it was chosen"],
         ["Installation cost", "₹ (regression)", "Log-linear Ridge", "Cost ≈ size × rate: beat boosting in CV"],
         ["Roof temperature drop", "°C (regression)", "LightGBM", "Predicts a fraction of start temperature"],
         ["Electricity savings", "kWh/month (regression)", "LightGBM", "Predicts a fraction of current usage"],
         ["Customer rating", "1–5 stars (regression)", "LightGBM", "Best CV error; clipped to 1–5"],
         ["High rating (≥ 4★)", "yes / no (classification)", "LightGBM", "Best CV accuracy"],
         ["Customer referral", "yes / no (classification)", "Logistic Regression", "Best CV F1; weak signal overall"]]
for k, name_, sel in zip(("cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral"),
                        ["Log-linear Ridge", "LightGBM", "LightGBM", "LightGBM", "LightGBM", "Logistic Regression"], range(6)):
    assert name_.split()[0].lower() in T[k]["selected_label"].lower() or (name_ == "Log-linear Ridge" and "log-linear" in T[k]["selected_label"].lower()), k
table(s, 0.6, 1.5, mrows, [2.5, 2.85, 2.45, 4.3], size=13, rowh=0.6)
card(s, 0.6, 5.85, 5.95, 1.1, "Regression: predicts a number", "Cost (₹), temperature drop (°C), electricity saved (kWh), rating (stars).", size=13, tsize=14)
card(s, 6.75, 5.85, 5.95, 1.1, "Classification: predicts a yes / no", "High rating (≥ 4★) and customer referral, with a probability.", size=13, tsize=14)

# ============================================================================= SLIDE 5: Technology stack
s = slide(5, "A tested web product built on a standard, open technology stack", "Technology stack",
          f"Source: requirements.txt, package.json, docs/architecture.md; {BACKEND_TESTS} backend + {FRONTEND_TESTS} UI tests last run",
          "[~1.25 min] The data flows left to right. The raw CSV is read-only and processed with pandas. The audit and cleaning step produces a cleaned dataset and an audit trail. Training and evaluation use scikit-learn and LightGBM, and save six model files with metadata. "
          "A FastAPI service loads those models once and serves predictions, option comparison, explanations and PDF proposals generated with ReportLab. The React, TypeScript and Vite front end, styled with Tailwind and charted with Recharts, calls that API. "
          "There is also an optional AI assistant. It has a free built-in mode, and a Gemini integration that we tested live. A Claude integration is built but we could not live-test it, because that account has no credit. "
          "In every mode the language model can only call our validated tools, so it never invents the numbers. "
          f"For quality, we have {BACKEND_TESTS} backend tests and {FRONTEND_TESTS} frontend tests including accessibility checks, a clean type check and a production build.\n"
          "Takeaway: a conventional, maintainable stack with one shared feature contract between training and serving.\nTransition: now the results.")
steps = [("Raw CSV", "pandas · read-only"), ("Audit & cleaning", "pandas · audit trail"), ("Train & evaluate", "scikit-learn · LightGBM"), ("Saved models", "joblib · 6 models")]
for i, (a, b_) in enumerate(steps):
    x = 0.6 + i * 3.2
    box(s, x, 1.55, 2.5, 1.2, GREEN if i < 3 else TEAL)
    tbox(s, x + 0.05, 1.62, 2.4, 0.45, a, 17, WHITE, True, PP_ALIGN.CENTER)
    tbox(s, x + 0.05, 2.12, 2.4, 0.5, b_, 13, "D9EAE4", False, PP_ALIGN.CENTER)
    if i < 3:
        arrow(s, x + 2.55, 1.95, 0.6, 0.4)
arrow(s, 11.5, 2.85, 0.5, 0.5, MSO_SHAPE.DOWN_ARROW)
arrow(s, 7.95, 2.85, 0.5, 0.5, MSO_SHAPE.DOWN_ARROW, GREY)
box(s, 6.8, 3.45, 5.9, 1.35, MIST, line=TEAL)
tbox(s, 6.95, 3.5, 5.6, 0.4, "FastAPI backend", 17, GREEN, True)
tbox(s, 6.95, 3.93, 5.6, 0.85, "predict · compare options · explain · payback · PDF proposals (ReportLab) · strict input validation", 13, CHAR)
box(s, 0.6, 3.45, 5.9, 1.35, "F3F4F4", line=LINE)
tbox(s, 0.75, 3.5, 5.6, 0.4, "Reports and figures", 17, GREY, True)
tbox(s, 0.75, 3.93, 5.6, 0.85, "data-quality, evaluation and insight reports feed the performance and insight pages", 13, CHAR)
arrow(s, 8.0, 4.85, 0.4, 0.45, MSO_SHAPE.UP_DOWN_ARROW, TEAL)
arrow(s, 11.0, 4.85, 0.4, 0.45, MSO_SHAPE.UP_DOWN_ARROW, TEAL)
box(s, 6.8, 5.35, 2.85, 1.55, GREEN)
tbox(s, 6.85, 5.4, 2.75, 1.45, [("React + TypeScript + Vite", {"bold": True, "size": 14}), "Tailwind CSS · Recharts", "8 pages · accessible"], 13, WHITE, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, 2)
box(s, 9.85, 5.35, 2.85, 1.55, TEAL)
tbox(s, 9.9, 5.4, 2.75, 1.45, [("Optional AI assistant", {"bold": True, "size": 14}), "Built-in (free): tested", "Gemini: live-tested", "Claude: built, not live-tested"], 12, WHITE, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE, 1)
card(s, 0.6, 5.35, 5.9, 1.55, "Quality gates", f"{BACKEND_TESTS} backend + {FRONTEND_TESTS} frontend tests (incl. accessibility) · TypeScript type-check · production build · leakage and provenance checks", size=13, tsize=15)

# ============================================================================= SLIDE 6: Evaluation and results
s = slide(6, "Strong on cost, cooling and savings; weak on rating and referral", "Model evaluation and results",
          f"Source: models/model_metadata.json, reports/significance.json. All results on {N_TEST} held-out buildings never used in training or tuning.",
          f"[~2.5 min] First the metrics. MAE, mean absolute error, is the average size of the miss in the unit being predicted. RMSE also measures error but squares it, so large misses count more. R-squared is the share of variation explained. "
          "For yes-or-no tasks, accuracy is the share of correct answers and F1 balances precision and recall; ROC-AUC measures how well the model ranks cases, where 0.5 is chance. "
          f"All results are on {N_TEST} buildings that were set aside before modelling and never used for training or tuning. Each chart has its own unit and scale. "
          f"Cost: the typical miss is {inr(c['test']['mae'])} against {inr(c['test_baseline']['mae'])} for a baseline that just predicts the average. Cooling: {d['test']['mae']:.2f} against {d['test_baseline']['mae']:.2f} degrees. Electricity saved: {u['test']['mae']:.2f} against {u['test_baseline']['mae']:.1f} kilowatt-hours. R-squared is 0.998, 0.948 and 0.987. "
          f"Rating and referral are weaker. Rating error is {ra['test']['mae']:.2f} stars against {ra['test_baseline']['mae']:.2f}. High-rating accuracy is {hr['test']['accuracy']:.1%} against {hr['baseline_accuracy']:.1%} for always guessing the majority, which is a significant gain, but its F1 is not distinguishable from a trivial rule. "
          f"For referral, F1 is {rfr['test']['f1']:.3f}, statistically indistinguishable from predicting everyone refers at {rfr['baseline_f1']:.3f}, and accuracy of {rfr['test']['accuracy']:.1%} is significantly worse than the majority-class {rfr['baseline_accuracy']:.1%}. "
          f"The AUC of {rfr['test']['roc_auc']:.2f} shows a real but weak ranking signal. These two outcomes depend on what happens after installation, which isn't available at quote time, so we use them only to rank, never to promise.\n"
          "Takeaway: strong where the data is structured, honest where it isn't.\nTransition: how it looks in practice.")
tbox(s, 0.6, 1.2, 12.1, 0.4, [[("MAE ", {"bold": True, "color": GREEN}), "average miss  ·  ", ("RMSE ", {"bold": True, "color": GREEN}), "big misses weigh more  ·  ", ("R² ", {"bold": True, "color": GREEN}), "variation explained  ·  ",
                               ("Accuracy ", {"bold": True, "color": GREEN}), "% correct  ·  ", ("F1 ", {"bold": True, "color": GREEN}), "precision/recall balance  ·  ", ("ROC-AUC ", {"bold": True, "color": GREEN}), "ranking skill (0.5 = chance)"]], 12, CHAR)
native_bar(s, 0.6, 1.6, 3.95, 2.2, [round(c["test_baseline"]["mae"] / 1e5, 2), round(c["test"]["mae"] / 1e5, 2)], '"₹"0.00" L"', "Cost, MAE (₹ lakh)")
native_bar(s, 4.7, 1.6, 3.95, 2.2, [round(d["test_baseline"]["mae"], 2), round(d["test"]["mae"], 2)], '0.00" °C"', "Temperature drop, MAE (°C)")
native_bar(s, 8.8, 1.6, 3.9, 2.2, [round(u["test_baseline"]["mae"], 1), round(u["test"]["mae"], 2)], '0.0" kWh"', "Electricity saved, MAE (kWh/mo)")
rrows = [["Task", "MAE", "RMSE", "R²", "Baseline MAE"],
         ["Installation cost", inr(c["test"]["mae"]), inr(c["test"]["rmse"]), f"{c['test']['r2']:.3f}", inr(c["test_baseline"]["mae"])],
         ["Roof temperature drop", f"{d['test']['mae']:.2f} °C", f"{d['test']['rmse']:.2f} °C", f"{d['test']['r2']:.3f}", f"{d['test_baseline']['mae']:.2f} °C"],
         ["Electricity savings", f"{u['test']['mae']:.2f} kWh", f"{u['test']['rmse']:.2f} kWh", f"{u['test']['r2']:.3f}", f"{u['test_baseline']['mae']:.1f} kWh"]]
table(s, 0.6, 3.9, rrows, [2.35, 1.35, 1.35, 0.8, 1.75], size=12, rowh=0.38)
short = {"better": "better", "worse": "worse", "not_distinguishable": "n.s."}
crows = [["Task", "Our model", "Trivial baseline", "Verdict"],
         ["Rating (MAE)", f"{ra['test']['mae']:.2f} ★ · R² {ra['test']['r2']:.2f}", f"{ra['test_baseline']['mae']:.2f} ★ (median)", "better"],
         ["High rating ≥4★", f"acc {hr['test']['accuracy']:.1%} · F1 {hr['test']['f1']:.3f} · AUC {hr['test']['roc_auc']:.2f}", f"acc {hr['baseline_accuracy']:.1%} · F1 {hr['baseline_f1']:.3f}", f"acc {short[sg_h['accuracy_verdict']]} · F1 {short[sg_h['f1_verdict']]}"],
         ["Referral", f"F1 {rfr['test']['f1']:.3f} · acc {rfr['test']['accuracy']:.1%} · AUC {rfr['test']['roc_auc']:.2f}", f"F1 {rfr['baseline_f1']:.3f} · acc {rfr['baseline_accuracy']:.1%}", f"F1 {short[sg_r['f1_verdict']]} · acc {short[sg_r['accuracy_verdict']]}"]]
table(s, 0.6, 5.55, crows, [1.5, 3.05, 2.1, 1.5], size=11, rowh=0.36)
card(s, 8.95, 3.9, 3.75, 1.55, "Held-out evaluation", f"{N_TEST} buildings set aside before modelling, scored once. Baselines scored on the same rows. 'Better' only if the 95% bootstrap interval excludes 0; n.s. = not statistically distinguishable.", size=12, tsize=14)
card(s, 8.95, 5.55, 3.75, 1.4, "Why rating/referral are weaker", "They depend on experience after installation, not known at quote time. Use to rank, not promise.", size=12, tsize=14, fill="F8E6E1", tcolor=CLAY)

# ============================================================================= SLIDE 7: Demonstration and business value
s = slide(7, "Working software: predict, compare, check quality, propose", "System demonstration and business value",
          "Source: screenshots of the running local application. Assistant: Gemini live-tested; Claude built but not live-tested.",
          "[~1.75 min] These are real screenshots from the running application. Top left, customer prediction: a rep enters a few facts about the building and gets cost, cooling, electricity savings, rating and referral, each with a likely range and a reliability flag, "
          "plus a 'why this estimate' explanation. Next, the roofing-option comparison ranks every valid method and quality for the same building using a transparent, adjustable score. "
          "Third, the model performance dashboard, with baselines, confidence intervals and a confusion matrix. On the bottom row, the data-quality insights page shows the audit, then the PDF proposal page, built from real predictions, "
          "and the optional AI assistant. The assistant answered live through Gemini by calling the same prediction tools; the Claude integration is built but not live-tested. "
          "The business value is simple: a company can prepare a preliminary estimate in seconds, compare options before quoting, and hand the customer a proposal that states its assumptions. "
          "These are estimates, not guarantees, and we say so in the product.\nTakeaway: this is working software that supports quoting.\nTransition: to wrap up.")
CROPS = {"d8_predict.png": ("ui_predict.png", (300, 10, 1410, 610)), "d8_compare.png": ("ui_compare.png", (300, 10, 1410, 610)),
         "d8_performance.png": ("ui_performance.png", (300, 10, 1410, 610)), "d8_insights.png": ("ui_insights.png", (300, 10, 1410, 610)),
         "d8_proposal.png": ("ui_proposal.png", (300, 10, 1410, 610)), "d8_assistant.png": ("ui_assistant.png", (320, 180, 1430, 780))}
for out_name, (src_name, bx) in CROPS.items():
    Image.open(ASSETS / src_name).convert("RGB").crop(bx).save(ASSETS / out_name)
grid = [("d8_predict.png", "Customer prediction", "cost, cooling, kWh, rating, referral"), ("d8_compare.png", "Option comparison", "15 method × quality options ranked"),
        ("d8_performance.png", "Model performance", "baselines, intervals, confusion matrix"), ("d8_insights.png", "Data-quality insights", "audit counts and cleaning decisions"),
        ("d8_proposal.png", "PDF proposal", "built from real predictions"), ("d8_assistant.png", "AI assistant", "Gemini live-tested; Claude untested")]
for i, (name, label, sub) in enumerate(grid):
    x, y = 0.6 + (i % 3) * 4.1, 1.35 + (i // 3) * 2.58
    w_, h_ = pic(s, name, x, y, w=3.95)
    tbox(s, x, y + h_ + 0.01, 3.95, 0.3, [[(label + ": ", {"bold": True, "color": GREEN}), sub]], 11, CHAR)
box(s, 0.6, 6.55, 12.1, 0.45, GREEN)
tbox(s, 0.75, 6.57, 11.8, 0.4, "Business value: prepare preliminary estimates in seconds, compare options before quoting, hand over a proposal with its assumptions stated.", 13, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)

# ============================================================================= SLIDE 8: Conclusion and future scope
s = slide(8, "Explained, honest estimates from historical roofing data", "Conclusion and future scope",
          "Cool Roofing Intelligence · all figures from the project's saved reports. Future scope items are NOT implemented.",
          "[~1 min] To conclude. The project turns 2,500 historical installations into a working system: an audited dataset, six models for five tasks, honest evaluation on unseen buildings, explanations, option comparison and PDF proposals. "
          f"The strongest results are cost with an average miss of {inr(c['test']['mae'])}, cooling within {d['test']['mae']:.1f} degrees and electricity savings within {u['test']['mae']:.1f} kilowatt-hours. "
          "The value for a roofing company is faster, better-informed quotes. The limits are real: the dataset has no time or location fields, the zero-kilowatt-hour treatment is an inference, rating and referral are weak, "
          "the four-star threshold is our assumption, and predictions are estimates, not guarantees. Future scope, none of it built yet: a digital-twin simulator, a carbon-savings tracker, geo-climate intelligence, "
          "and above all validation using newly measured installations. Thank you, we're happy to take questions.\nTakeaway: trustworthy, explainable decisions from historical data.")
cols3 = [("What the project achieves", GREEN, [f"Audited {N_ROWS:,}-record dataset; 0 rows deleted", "5 tasks, 6 models, held-out evaluation", f"Cost MAE {inr(c['test']['mae'])}", f"Cooling MAE {d['test']['mae']:.2f} °C · savings MAE {u['test']['mae']:.2f} kWh", "Compare options · explain · PDF proposals"]),
         ("Value and limitations", TEAL, ["Faster preliminary estimates; options compared before quoting", "Dataset: no time or location fields; 0 kWh fix is inferred", "Rating and referral only weakly predictable", "High rating = 4★ is our assumption", "Estimates, not guarantees"]),
         ("Future scope (not implemented)", GREY, ["Validate on newly measured installations", "Digital-twin simulator", "Carbon-savings tracker", "Geo-climate intelligence"])]
for i, (h_, colr, items) in enumerate(cols3):
    x = 0.6 + i * 4.1
    box(s, x, 1.45, 3.9, 3.95, MIST if i < 2 else "F3F4F4", line=None if i < 2 else LINE)
    chip(s, x + 0.15, 1.58, 3.6, h_, colr, size=13, h=0.42)
    tbox(s, x + 0.2, 2.15, 3.55, 3.2, [[("• ", {"bold": True, "color": colr}), it] for it in items], 14, CHAR if i < 2 else GREY, space=5)
box(s, 0.6, 5.55, 12.1, 0.95, GREEN)
tbox(s, 0.85, 5.58, 11.6, 0.9, "Historical roofing data, turned into estimates a sales rep can trust, with the reasons and the limits attached.", 20, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
tbox(s, 0.6, 6.55, 12.1, 0.5, "Thank you · Questions?", 22, TEAL, True, PP_ALIGN.CENTER)

# ----------------------------------------------------------------------------- save (canonical path, with a safe fallback if PowerPoint has it locked)
for n_, sl_ in enumerate(prs.slides, start=1):
    NOTES[n_] = re.sub(r"\[~[\d.]+ min\]", f"[~{MINS[n_]:g} min]", NOTES[n_])
    sl_.notes_slide.notes_text_frame.text = NOTES[n_]
target = OUT / "Cool_Roofing_Hackathon_Presentation.pptx"
try:
    prs.save(target)
except PermissionError:
    target = OUT / "Cool_Roofing_Hackathon_Presentation_8slides.pptx"
    prs.save(target)
    print("NOTE: canonical file is locked (open in PowerPoint); saved as", target.name)
md = ["# Speaker notes: Cool Roofing Intelligence (8 slides)", "", f"Planned speaking time: {sum(MINS.values()):.2f} minutes (target 12-13), leaving buffer in a 15-minute slot.", ""]
for n in range(1, 9):
    md += [f"## Slide {n} (~{MINS[n]:g} min)", NOTES[n].replace("\n", "\n\n"), ""]
(OUT / "presentation_notes.md").write_text("\n".join(md), encoding="utf-8")
print("saved", target, "| slides:", len(prs.slides), "| planned minutes:", sum(MINS.values()))

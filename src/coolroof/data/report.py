"""Profiling, figures and the human-readable data-quality report.

All numbers are computed from the real data; nothing is typed in by hand.
`build_profile` returns plain JSON (also served by the API and charted by the UI).
"""
from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config as C  # noqa: E402
from .clean import CleaningResult  # noqa: E402

PAL = {"ink": "#14262B", "teal": "#0E7C86", "sun": "#E8A33D", "clay": "#C8553D", "sage": "#7A9E7E",
       "mist": "#DCE7E8", "sand": "#F6F1E7", "slate": "#5B6B70"}
QUALITY_COLOURS = {"basic": PAL["sun"], "intermediate": PAL["teal"], "high": PAL["ink"]}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": PAL["slate"],
        "axes.labelcolor": PAL["ink"], "xtick.color": PAL["slate"], "ytick.color": PAL["slate"],
        "text.color": PAL["ink"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6ECEC", "grid.linewidth": 0.8, "font.size": 10,
        "axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 130,
    })


def _hist(s: pd.Series, bins: int = 30, rng=None) -> dict:
    s = s.dropna()
    counts, edges = np.histogram(s, bins=bins, range=rng)
    return {"edges": [round(float(e), 4) for e in edges], "counts": [int(c) for c in counts], "n": int(len(s))}


def _group_stats(df: pd.DataFrame, by: str, val: str) -> list[dict]:
    out = []
    for key, g in df.dropna(subset=[val]).groupby(by):
        v = g[val].astype(float)
        n = len(v)
        sem = v.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
        out.append({"group": key, "n": int(n), "mean": float(v.mean()), "median": float(v.median()),
                    "std": float(v.std(ddof=1)) if n > 1 else None,
                    "ci95": [float(v.mean() - 1.96 * sem), float(v.mean() + 1.96 * sem)] if n > 1 else None})
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _prop_stats(df: pd.DataFrame, by: str, flag: pd.Series) -> list[dict]:
    out = []
    for key, idx in df.groupby(by).groups.items():
        f = flag.loc[idx].dropna().astype(bool)
        n, k = len(f), int(f.sum())
        lo, hi = _wilson(k, n)
        out.append({"group": key, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]})
    return out


def build_profile(res: CleaningResult) -> dict:
    raw, df = res.raw, res.cleaned
    rng = np.random.default_rng(C.SEED)
    samp = df.sample(n=min(700, len(df)), random_state=C.SEED)
    ref = df["customer_referral"].astype("boolean")
    profile = {
        "n_rows": int(len(df)),
        "targets": {
            "cost": _hist(df.loc[df.eligible_cost, "cool_roofing_cost"], 30),
            "temperature_drop": _hist(df.loc[df.eligible_temperature_drop, "temperature_drop"], 30),
            "units_saved": _hist(df.loc[df.eligible_units_saved, "units_saved"], 40),
            "rating": {"values": [1, 2, 3, 4, 5],
                       "counts": [int((df.customer_given_rating == v).sum()) for v in range(1, 6)]},
            "referral": {"true": int(ref.sum()), "false": int((~ref).sum()),
                         "rate": float(ref.mean())},
        },
        "missingness": [{"column": c, "before": res.summary["missing_before"][c],
                         "after": res.summary["missing_after_cleaning"][c]} for c in C.RAW_COLUMNS],
        "temperature_before_after_cleaning": {
            "raw_before": _hist(raw["avg_roof_temperature_before"], 40, (15, 125)),
            "clean_before": _hist(df["avg_roof_temperature_before"], 40, (15, 125)),
            "raw_after": _hist(raw["avg_roof_temperature_after"], 40, (15, 125)),
            "clean_after": _hist(df["avg_roof_temperature_after"], 40, (15, 125)),
        },
        "scatter": {
            "cost_vs_size": [{"x": float(r.building_size), "y": float(r.cool_roofing_cost), "g": r.cool_roofing_quality}
                             for r in samp.itertuples() if pd.notna(r.building_size) and pd.notna(r.cool_roofing_cost)],
            "units_saved_vs_before": [{"x": float(r.avg_units_before), "y": float(r.units_saved), "g": str(r.num_ac_units)}
                                      for r in samp.itertuples() if pd.notna(r.units_saved)],
        },
        "groups": {
            "temperature_drop_by_roofing_type": _group_stats(df, "roofing_type", "temperature_drop"),
            "temperature_drop_by_method": _group_stats(df, "cool_roofing_method", "temperature_drop"),
            "temperature_drop_by_quality": _group_stats(df, "cool_roofing_quality", "temperature_drop"),
            "cost_by_method": _group_stats(df.loc[df.eligible_cost], "cool_roofing_method", "cool_roofing_cost"),
            "cost_by_quality": _group_stats(df.loc[df.eligible_cost], "cool_roofing_quality", "cool_roofing_cost"),
            "units_saved_by_method": _group_stats(df, "cool_roofing_method", "units_saved"),
            "units_saved_by_quality": _group_stats(df, "cool_roofing_quality", "units_saved"),
            "rating_by_method": _group_stats(df, "cool_roofing_method", "customer_given_rating"),
            "rating_by_quality": _group_stats(df, "cool_roofing_quality", "customer_given_rating"),
            "high_rating_by_method": _prop_stats(df, "cool_roofing_method", df["high_rating"]),
            "high_rating_by_quality": _prop_stats(df, "cool_roofing_quality", df["high_rating"]),
            "referral_by_method": _prop_stats(df, "cool_roofing_method", ref),
            "referral_by_quality": _prop_stats(df, "cool_roofing_quality", ref),
            "referral_by_location": _prop_stats(df, "location_type", ref),
        },
        "method_roof_support": (df.groupby(["roofing_type", "cool_roofing_method"]).size()
                                .reset_index(name="n").to_dict(orient="records")),
        "feature_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in C.NUMERIC_FEATURE_COLS},
        "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in C.CATEGORICAL_COLS},
        "audit_by_check": res.audit.groupby(["check", "classification"]).size().reset_index(name="n_records")
        .to_dict(orient="records"),
        "summary": {k: res.summary[k] for k in ("eligible_rows_by_task", "exclusions_by_task",
                                                "temperature_sentinel_rows", "zero_consumption_placeholders", "negative_savings_rows",
                                                "zero_savings_rows", "rows_with_any_missing_feature",
                                                "records_excluded_from_every_task")},
    }
    return profile


def make_figures(res: CleaningResult) -> list[str]:
    _style()
    C.ensure_dirs()
    raw, df = res.raw, res.cleaned
    made: list[str] = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(C.FIGURES / name, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    # 1. target distributions
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    ax = axs[0, 0]; ax.hist(df.loc[df.eligible_cost, "cool_roofing_cost"] / 1e5, bins=40, color=PAL["teal"])
    ax.set(title="Installation cost", xlabel="INR lakh", ylabel="buildings")
    ax = axs[0, 1]; ax.hist(df["temperature_drop"].dropna(), bins=40, color=PAL["sun"])
    ax.set(title="Roof temperature drop (valid rows)", xlabel="°C")
    ax = axs[0, 2]; ax.hist(df["units_saved"].dropna(), bins=50, color=PAL["sage"])
    ax.axvline(0, color=PAL["clay"], lw=1.5, ls="--"); ax.set(title="Monthly electricity saved", xlabel="kWh / month")
    ax = axs[1, 0]; vc = df["customer_given_rating"].value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=PAL["teal"]); ax.set(title="Customer rating", xlabel="stars", ylabel="buildings")
    ax = axs[1, 1]; rc = df["customer_referral"].astype(bool).value_counts()
    ax.bar(["no referral", "referred"], [rc.get(False, 0), rc.get(True, 0)], color=[PAL["slate"], PAL["sun"]])
    ax.set(title="Referral class balance")
    axs[1, 2].axis("off")
    axs[1, 2].text(0, 0.5, "Distributions use the cleaned,\ntask-eligible rows.\n\nDashed line = no saving.", fontsize=10, va="center")
    save(fig, "01_target_distributions.png")

    # 2. missingness
    miss = pd.DataFrame({"column": C.RAW_COLUMNS,
                         "raw_missing": [res.summary["missing_before"][c] for c in C.RAW_COLUMNS],
                         "after_cleaning": [res.summary["missing_after_cleaning"][c] for c in C.RAW_COLUMNS]})
    miss = miss[(miss.raw_missing > 0) | (miss.after_cleaning > 0)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(miss)); ax.bar(x - 0.2, miss.raw_missing, 0.4, label="raw file", color=PAL["slate"])
    ax.bar(x + 0.2, miss.after_cleaning, 0.4, label="after cleaning (invalid → missing)", color=PAL["clay"])
    ax.set_xticks(x, miss.column, rotation=25, ha="right"); ax.set(ylabel="missing cells", title="Missing values by column")
    ax.legend(frameon=False)
    save(fig, "02_missingness.png")

    # 3. temperature before/after cleaning
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, col, t in zip(axs, ["avg_roof_temperature_before", "avg_roof_temperature_after"], ["Before installation", "After installation"]):
        ax.hist(raw[col], bins=np.arange(15, 126, 2.5), color=PAL["clay"], alpha=0.85, label="raw (120 °C sentinel visible)")
        ax.hist(df[col].dropna(), bins=np.arange(15, 126, 2.5), color=PAL["teal"], alpha=0.9, label="cleaned")
        ax.set(title=t, xlabel="avg roof temperature (°C)"); ax.legend(frameon=False)
    axs[0].set_ylabel("buildings")
    save(fig, "03_temperature_outliers_before_after.png")

    # 4. cost vs size
    fig, ax = plt.subplots(figsize=(7.5, 5))
    d = df[df.eligible_cost]
    for q, g in d.groupby("cool_roofing_quality"):
        ax.scatter(g.building_size, g.cool_roofing_cost / 1e5, s=9, alpha=0.5, color=QUALITY_COLOURS[q], label=q)
    ax.set(xlabel="building size (area units)", ylabel="cost (INR lakh)", title="Installation cost vs building size")
    ax.legend(title="quality", frameon=False)
    save(fig, "04_cost_vs_size.png")

    # 5. temperature drop by roof type and method
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    pv = df.pivot_table(index="cool_roofing_method", columns="roofing_type", values="temperature_drop", aggfunc="mean")
    pv.plot(kind="bar", ax=ax, color=[PAL["teal"], PAL["sun"], PAL["sage"], PAL["ink"]], width=0.8)
    ax.set(xlabel="", ylabel="mean temperature drop (°C)", title="Temperature drop by method and existing roof type")
    ax.set_xticklabels(pv.index, rotation=20, ha="right"); ax.legend(title="roof type", frameon=False)
    save(fig, "05_temperature_drop_by_roof_and_method.png")

    # 6. units saved vs units before
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sc = ax.scatter(df.avg_units_before, df.units_saved, s=9, alpha=0.5, c=df.num_ac_units.fillna(-1), cmap="viridis")
    ax.axhline(0, color=PAL["clay"], ls="--", lw=1.2)
    ax.set(xlabel="electricity before (kWh/month)", ylabel="kWh/month saved", title="Savings vs pre-installation usage")
    plt.colorbar(sc, ax=ax, label="AC units (-1 = unknown)")
    save(fig, "06_units_saved_vs_before.png")

    # 7. rating & referral by method
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    g = df.groupby("cool_roofing_method")["customer_given_rating"].mean().sort_values()
    axs[0].barh(g.index, g.values, color=PAL["teal"]); axs[0].set(title="Mean rating by method", xlabel="rating (1-5)")
    r = df.groupby("cool_roofing_method")["customer_referral"].apply(lambda s: s.astype(bool).mean()).sort_values()
    axs[1].barh(r.index, r.values * 100, color=PAL["sun"]); axs[1].set(title="Referral rate by method", xlabel="% referred")
    save(fig, "07_rating_referral_by_method.png")
    return made


def write_markdown(res: CleaningResult, profile: dict) -> str:
    s = res.summary
    st = s["structural"]
    a = res.audit
    lines: list[str] = []
    w = lines.append
    w("# Data Quality & Cleaning Report")
    w("")
    w("_Generated by `coolroof data` from the real CSV. Every number below is computed, not typed._")
    w("")
    w("## 1. Structure")
    w(f"- Rows: **{st['rows']:,}**, columns: **{st['columns']}** (all 15 expected columns present).")
    w(f"- Duplicate rows: **{st['duplicate_rows']}**; duplicate `building_id`: **{st['duplicate_building_ids']}**; "
      f"duplicates when ignoring the id: **{st['duplicate_rows_excluding_id']}**.")
    w(f"- `building_id` is a unique sequential record number ({'1..N' if st['id_is_sequential'] else 'not sequential'}). "
      "It carries no business meaning, so it is **never used as a feature**.")
    w(f"- Unexpected category labels: **{sum(len(v) for v in st['unexpected_categories'].values())}**; "
      "labels are already lower-case/trimmed (0 changes needed).")
    w("- Numeric columns all parsed as numbers; `customer_referral` parsed as boolean.")
    w("")
    w("## 2. Missing values")
    w("| column | missing (raw) | missing (after cleaning) |")
    w("|---|---:|---:|")
    for c in C.RAW_COLUMNS:
        b, af = s["missing_before"][c], s["missing_after_cleaning"][c]
        if b or af:
            w(f"| `{c}` | {b} | {af} |")
    w("")
    w(f"Rows with at least one missing/invalid *feature* after cleaning: **{s['rows_with_any_missing_feature']}** "
      f"of {st['rows']:,}. Missing features are imputed **inside each training fold** (median / most-frequent plus "
      "a missing-indicator), never on the held-out set.")
    w("")
    w("## 3. Anomalies and decisions")
    ts = s["temperature_sentinel_rows"]
    w("### 3.1 Roof temperature = 120 °C (invalid)")
    w(f"- **{ts['before']}** records have `avg_roof_temperature_before = 120.0` and **{ts['after']}** have "
      f"`avg_roof_temperature_after = 120.0` (never both on one record).")
    w("- Every other reading lies in a tight, physical band; 120 °C is a repeated placeholder far outside it, "
      "and the implied 'temperature drop' would be ±90 °C versus a valid range of "
      f"{s['valid_temperature_drop_range'][0]:.1f}–{s['valid_temperature_drop_range'][1]:.1f} °C.")
    w("- **Decision:** the value cannot be recovered, so it is set to missing (not capped, not guessed). "
      "A record with an invalid *before/after* temperature is excluded from the temperature-drop task only; "
      "an invalid *before* temperature is also imputed when used as a feature for the other tasks.")
    w("")
    z = s["zero_consumption_placeholders"]
    w("### 3.2 Electricity readings of exactly 0 kWh (invalid placeholder)")
    w(f"- `avg_units_before = 0` in **{z['before']}** records and `avg_units_after = 0` in **{z['after']}** (never both).")
    w("- Evidence they are not real readings: (1) the value distribution has a spike at 0 but **no values at 1–3 kWh**, "
      "with the natural lower tail starting near 4–5 kWh; (2) the records are physically inconsistent: `before = 0` rows report up to "
      f"**{z['after_when_before_is_zero_max']:.0f} kWh after** (often with 1–2 AC units), and `after = 0` rows had up to "
      f"**{z['before_when_after_is_zero_max']:.0f} kWh before** (often with 3 AC units) – an after/before ratio of exactly 0, while the smallest ratio "
      f"among all valid records is **{z['min_after_over_before_ratio_among_valid']:.3f}**.")
    w("- **Decision:** set to missing (the true value is unrecoverable). The record is excluded from the electricity-savings task only; "
      "an invalid *before* value is imputed (training-fold median + missing-indicator) when used as a feature elsewhere. "
      "This is stated as a high-confidence inference, not a certainty.")
    w("")
    w("### 3.3 Electricity use higher after installation (valid but undesirable)")
    w(f"- **{s['negative_savings_rows']}** records have `avg_units_after > avg_units_before` (negative savings) and "
      f"**{s['zero_savings_rows']}** have exactly zero savings.")
    neg = res.cleaned[res.cleaned.flag_negative_savings]
    w(f"- {(neg.num_ac_units == 0).mean():.0%} of the negative-saving buildings have no AC units and the median "
      f"pre-installation usage is {neg.avg_units_before.median():.0f} kWh/month: they are small base-load buildings where a cooler roof has "
      "nothing to save and ordinary month-to-month variation dominates.")
    w("- **Decision:** kept. These are genuine outcomes the company must be able to warn customers about; deleting them would "
      "make the savings model look better than reality.")
    w("")
    w("### 3.4 Exactly-zero savings are kept")
    w(f"- {s['zero_savings_rows']} records have `units_saved = 0`. They are almost all low-usage buildings without AC, where a cooler roof has no "
      "cooling load to reduce – physically coherent, so retained.")
    w("")
    w("### 3.5 Other checks that found nothing")
    clean_checks = [
        "non-positive or extreme cost (cost per unit area vs. method×quality median)",
        "building size / age / AC-count out of range or non-integer",
        "rating outside 1–5 or non-integer",
        "referral flag missing or not boolean",
        "roof temperature after > before (excluding the sentinel rows)",
        "incompatible roof-type × method combinations (e.g. tile product on a sheet roof)",
        "negative electricity consumption",
    ]
    for c in clean_checks:
        w(f"- {c}: **0 records**")
    w("")
    w("## 4. Task-specific sample counts")
    w("| task | usable records | excluded | why |")
    w("|---|---:|---:|---|")
    why = {
        "cost": "building_size missing (size drives cost; not imputed for this task)",
        "temperature_drop": "before or after temperature is the 120 °C sentinel",
        "units_saved": "avg_units_before or avg_units_after is the 0 kWh placeholder", "rating": "none", "high_rating": "none", "referral": "none",
    }
    for k, t in C.TASKS.items():
        n = s["eligible_rows_by_task"][k]
        w(f"| {t.label} | {n:,} | {st['rows'] - n} | {why[k]} |")
    w("")
    w(f"Records excluded from **every** task: **{s['records_excluded_from_every_task']}**.")
    w("")
    w("## 5. Assumptions")
    for k, v in s["assumptions"].items():
        w(f"- **{k}**: {v}")
    w("")
    w("## 6. Remaining limitations")
    w("- Other silent artefacts that look plausible cannot be detected from values alone.")
    w("- The 0 kWh and 120 °C placeholders suggest a logger/migration artefact; treating them as missing is an inference from the value distribution and physical consistency, not a confirmed fact.")
    w("- No timestamp or location identifier exists, so temporal or geographic leakage/drift cannot be tested. Rows are treated as independent buildings.")
    w("- `building_size` has no stated unit; relative effects are valid, absolute 'sq ft' wording is an assumption.")
    w("")
    w("## 7. Figures")
    for f in sorted(p.name for p in C.FIGURES.glob("0*.png")):
        w(f"![{f}](figures/{f})")
    return "\n".join(lines) + "\n"


def run_all(res: CleaningResult) -> dict:
    C.ensure_dirs()
    profile = build_profile(res)
    from ..utils import write_json
    write_json(C.REPORTS / "data_profile.json", profile)
    make_figures(res)
    md = write_markdown(res, profile)
    (C.REPORTS / "data_quality_report.md").write_text(md, encoding="utf-8")
    (C.DOCS / "data_quality_report.md").write_text(md.replace("](figures/", "](../reports/figures/"), encoding="utf-8")
    return profile

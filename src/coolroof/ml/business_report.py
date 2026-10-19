"""Turns the computed insights into business_insights.md / business_recommendations.md.

Every figure in the text comes from `insights` (itself computed from the real models and data);
the wording only describes associations and never claims causation or guarantees.
"""
from __future__ import annotations

import pandas as pd

from .. import config as C

NICE = {"cool_roofing_method": "roofing method", "cool_roofing_quality": "quality grade", "building_size": "building size",
        "building_age": "building age", "roofing_type": "existing roof type", "location_type": "location type",
        "avg_roof_temperature_before": "starting roof temperature", "num_ac_units": "number of AC units",
        "avg_units_before": "current electricity use"}
M = {"basic_coating": "basic coating", "sheet": "sheet", "sheet_with_coating": "sheet + coating", "tiles": "tiles",
     "tiles_with_coating": "tiles + coating"}


def _top(fi: pd.DataFrame, task: str, k: int = 3) -> str:
    g = fi[fi.task == task].sort_values("share", ascending=False).head(k)
    return ", ".join(f"{NICE[r.feature]} ({r.share:.0%})" for r in g.itertuples())


def _row(lst, level):
    return next(r for r in lst if r["level"] == level)


def write_business_insights(insights: dict, svc, df: pd.DataFrame) -> None:
    fi = pd.DataFrame(insights["feature_importance"])
    me = insights["marginal_effects"]
    grid = pd.DataFrame(me["grid"])
    obs = insights["observed_segments"]
    ceil = insights["predictability_ceiling"]
    seg = pd.DataFrame(insights["segment_errors"])
    methods = C.EXPECTED_CATEGORIES["cool_roofing_method"]
    q = me["quality"]
    L: list[str] = []
    w = L.append

    w("# Business Insights")
    w("")
    w("_Computed by `coolroof insights` from the trained models and the cleaned data. These are **associations in historical installations**, "
      "not proven causal effects: method and quality may have been chosen by customers for reasons the data does not record._")
    w("")
    w("## 1. What drives each outcome (permutation importance, hold-out)")
    w("Share of the total drop in model skill when a feature is shuffled. This is a *global* view; per-customer explanations are in the app.")
    w("")
    w("| Outcome | Top drivers (share of importance) |")
    w("|---|---|")
    for t in C.TASK_ORDER:
        w(f"| {C.TASKS[t].label} | {_top(fi, t, 4)} |")
    w("")
    w("![importance](figures/20_feature_importance.png)")
    w("")
    w("## 2. How method and quality compare (model-based counterfactuals)")
    pop = me["population"]
    w(f"For each option we set every comparable building's method (or quality) to that option and average the model's prediction. "
      f"The method comparison uses the **{pop['method_comparison_rows']:,} concrete-roof buildings**, the only roofs that accept all five methods, "
      "so the building mix is identical for each method. Intervals are bootstrap intervals over buildings and **exclude model error and confounding**; read them as "
      "'how much the average shift varies across buildings', not as certainty.")
    w("")
    w("| Method (concrete roofs) | Avg cost (₹) | Temp drop (°C) | kWh saved / month | P(high rating) | P(referral) |")
    w("|---|---:|---:|---:|---:|---:|")
    for m in methods:
        c, t, u = (_row(me["method"][k], m) for k in ("cost", "temperature_drop", "units_saved"))
        h, r = _row(me["method"]["high_rating"], m), _row(me["method"]["referral"], m)
        w(f"| {M[m]} | {c['mean']:,.0f} | {t['mean']:.2f} | {u['mean']:.1f} | {h['mean']:.1%} | {r['mean']:.1%} |")
    w("")
    w("| Quality (all buildings) | Avg cost (₹) | Temp drop (°C) | kWh saved / month | P(high rating) | P(referral) |")
    w("|---|---:|---:|---:|---:|---:|")
    for lv in C.EXPECTED_CATEGORIES["cool_roofing_quality"]:
        c, t, u = (_row(q[k], lv) for k in ("cost", "temperature_drop", "units_saved"))
        h, r = _row(q["high_rating"], lv), _row(q["referral"], lv)
        w(f"| {lv} | {c['mean']:,.0f} | {t['mean']:.2f} | {u['mean']:.1f} | {h['mean']:.1%} | {r['mean']:.1%} |")
    w("")
    w("### Value for money (method × quality, concrete roofs)")
    w("Tariff-free efficiency: rupees paid per °C of temperature drop and per kWh/month saved (lower is better).")
    w("")
    w("| Method | Quality | Cost (₹) | ₹ per °C drop | ₹ per kWh/month saved | P(high rating) | P(referral) |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for r in grid.sort_values("cost_per_celsius_drop").itertuples():
        w(f"| {M[r.cool_roofing_method]} | {r.cool_roofing_quality} | {r.cost:,.0f} | {r.cost_per_celsius_drop:,.0f} | "
          f"{(r.cost_per_kwh_month_saved or float('nan')):,.0f} | {r.high_rating:.1%} | {r.referral:.1%} |")
    w("")
    w("![options](figures/21_method_quality_effects.png)")
    w("")
    be = grid.assign(breakeven=lambda x: x.cost / (x.units_saved.clip(lower=1e-9) * 12 * 10)).sort_values("breakeven")
    lo_, hi_ = be.iloc[0], be.iloc[-1]
    w("### Can electricity savings alone pay for the roof?")
    w("The data contains no tariff, so none is assumed. Instead, the **break-even tariff** is the electricity price (₹/kWh) at which the predicted monthly savings "
      "would repay the average installation cost in 10 years (no maintenance, no escalation):")
    w("")
    w(f"- lowest across the 15 method × quality options: **₹{lo_.breakeven:,.0f}/kWh** ({M[lo_.cool_roofing_method]}, {lo_.cool_roofing_quality}); "
      f"highest: ₹{hi_.breakeven:,.0f}/kWh ({M[hi_.cool_roofing_method]}, {hi_.cool_roofing_quality}).")
    w("- Compare that with the customer's actual tariff in the app. If it is far below these values, the investment has to be justified by benefits this dataset does "
      "not measure (comfort, roof protection, sustainability goals) rather than by the electricity bill.")
    w("")
    w("### Method comparison on each existing roof type")
    w("Same counterfactual idea, restricted to the methods the data supports for that roof (quality left at each building's own value).")
    for rt, rows_ in me["by_roof"].items():
        w("")
        w(f"**{rt}** (n = {rows_[0]['n_buildings']} buildings)")
        w("")
        w("| Method | Avg cost (₹) | Temp drop (°C) | kWh saved / month | P(high rating) | P(referral) |")
        w("|---|---:|---:|---:|---:|---:|")
        for r_ in rows_:
            w(f"| {M[r_['cool_roofing_method']]} | {r_['cost']:,.0f} | {r_['temperature_drop']:.2f} | {r_['units_saved']:.1f} | {r_['high_rating']:.1%} | {r_['referral']:.1%} |")
    w("")
    w("## 3. Who is good business — knowable before installation")
    w("Observed rates (no model) by customer segment, with 95% Wilson intervals. Segments with fewer than "
      f"{insights['min_segment_n']} buildings are marked and should not drive decisions.")
    for seg_name, title in (("ac_group", "Number of AC units"), ("usage_band", "Current monthly electricity use")):
        w("")
        w(f"**{title}**")
        w("")
        w("| Segment | n | Mean kWh saved | Mean rating | High-rating rate | Referral rate |")
        w("|---|---:|---:|---:|---:|---:|")
        for r in obs[seg_name]:
            flag = " ⚠ small" if r["small_sample"] else ""
            w(f"| {r['level']}{flag} | {r['n']} | {r['mean_units_saved']:.1f} | {r['mean_rating']:.2f} | "
              f"{r['high_rating_rate']:.1%} ({r['high_rating_ci95'][0]:.0%}–{r['high_rating_ci95'][1]:.0%}) | "
              f"{r['referral_rate']:.1%} ({r['referral_ci95'][0]:.0%}–{r['referral_ci95'][1]:.0%}) |")
    w("")
    w("## 4. Predictability limits (honesty check)")
    ra, ha = ceil["referral_auc"], ceil["high_rating_auc"]
    w(f"- Using only quote-time information, referral has hold-out ROC-AUC {svc.tasks['referral']['test']['roc_auc']:.2f} and high rating {svc.tasks['high_rating']['test']['roc_auc']:.2f}.")
    w(f"- **Diagnostic (not a model):** even if the *realised* savings and temperature drop were known, referral AUC would only reach "
      f"{ra['from_realised_savings_and_temperature_drop']:.2f}; adding the realised rating gives {ra['from_realised_rating_savings_drop']:.2f}. "
      f"High-rating AUC from realised outcomes is {ha['from_realised_savings_and_temperature_drop']:.2f}, barely above the quote-time model. "
      "So referral behaviour is mostly driven by factors outside this dataset; the pre-installation model is close to the practical ceiling and should be used for ranking/triage, not promises.")
    w("")
    w("## 5. Where the models are less reliable (out-of-fold error by segment)")
    for task in ("cost", "temperature_drop", "units_saved"):
        s = seg[(seg.task == task) & (~seg.small_sample)].dropna(subset=["mae"])
        worst = s.sort_values("mae", ascending=False).head(3)
        best = s.sort_values("mae").head(1)
        unit = C.TASKS[task].unit
        w(f"- **{C.TASKS[task].label}**: largest errors in " + "; ".join(f"{r.segment} = {r.level} (MAE {r.mae:,.2f} {unit}, n={r.n})" for r in worst.itertuples())
          + f". Smallest: {best.iloc[0].segment} = {best.iloc[0].level} ({best.iloc[0].mae:,.2f}).")
    w("")
    w("(Full table: `reports/segment_error_analysis.csv`.)")
    text = "\n".join(L) + "\n"
    (C.REPORTS / "business_insights.md").write_text(text, encoding="utf-8")
    (C.DOCS / "business_insights.md").write_text(text.replace("](figures/", "](../reports/figures/"), encoding="utf-8")

    # ---------------------------------------------------------------- recommendations
    cost = {m: _row(me["method"]["cost"], m)["mean"] for m in methods}
    drop = {m: _row(me["method"]["temperature_drop"], m)["mean"] for m in methods}
    sav = {m: _row(me["method"]["units_saved"], m)["mean"] for m in methods}
    hi = {m: _row(me["method"]["high_rating"], m)["mean"] for m in methods}
    rf = {m: _row(me["method"]["referral"], m)["mean"] for m in methods}
    cpd = {m: cost[m] / drop[m] for m in methods}
    best_out = max(methods, key=lambda m: hi[m] + rf[m])
    cheap = min(methods, key=lambda m: cost[m])
    ac = {r["level"]: r for r in obs["ac_group"]}
    R: list[str] = []
    r = R.append
    r("# Business Recommendations")
    r("")
    r("_Written from the computed results in `business_insights.md`; every number is traceable to the trained models and the real data._")
    r("")
    r("## Headline answer: which roofing method should the company recommend?")
    r(f"**There is no single best method for every customer, but the coated products are consistently stronger on the outcomes customers notice.**")
    r("")
    r(f"- On concrete roofs, **{M[best_out]}** has the highest predicted satisfaction ({hi[best_out]:.0%} chance of a high rating) and referral ({rf[best_out]:.0%}), "
      f"with a {drop[best_out]:.1f} °C average temperature drop and {sav[best_out]:.0f} kWh/month saved, at an average ₹{cost[best_out]:,.0f}.")
    r(f"- **{M['sheet_with_coating']}** is close behind ({hi['sheet_with_coating']:.0%} high rating, {rf['sheet_with_coating']:.0%} referral) and about "
      f"{(1 - cost['sheet_with_coating']/cost['tiles_with_coating']):.0%} cheaper than {M['tiles_with_coating']} (₹{cost['sheet_with_coating']:,.0f} vs ₹{cost['tiles_with_coating']:,.0f}).")
    worst = max(methods, key=lambda m: cpd[m])
    worst_txt = "the worst" if worst == "basic_coating" else "among the highest"
    r(f"- **{M['basic_coating']}** is the cheapest (₹{cost[cheap]:,.0f}) {'but ' if cheap == 'basic_coating' else ''}gives the weakest outcomes ({drop['basic_coating']:.1f} °C, {hi['basic_coating']:.0%} high rating, {rf['basic_coating']:.0%} referral) "
      f"and {worst_txt} cost per °C (₹{cpd['basic_coating']:,.0f} vs ₹{cpd['sheet_with_coating']:,.0f} for {M['sheet_with_coating']}), so it is not good value despite the low ticket price.")
    r("- Method must match the existing roof: tile products need tile or concrete roofs; sheet products need sheet or concrete roofs. By roof type "
      "(highest combined chance of a high rating and a referral):")
    for rt, rows_ in me["by_roof"].items():
        top_ = max(rows_, key=lambda x: x["high_rating"] + x["referral"])
        cheapest_ = min(rows_, key=lambda x: x["cost"])
        r(f"  - **{rt}**: {M[top_['cool_roofing_method']]} (P(high rating) {top_['high_rating']:.0%}, referral {top_['referral']:.0%}, avg ₹{top_['cost']:,.0f}); "
          f"cheapest is {M[cheapest_['cool_roofing_method']]} (₹{cheapest_['cost']:,.0f}).")
    r("- Practical rule: **lead with the strongest option that fits the roof** (above), then use the comparison tool to confirm budget fit.")
    r("")
    r("## Quality grade")
    qc, qh = _row(q["cost"], "high"), _row(q["high_rating"], "high")
    qb = _row(q["high_rating"], "basic")
    r(f"High quality costs about ₹{qc['diff_vs_reference']:,.0f} more than basic on average but is associated with +{_row(q['temperature_drop'], 'high')['diff_vs_reference']:.1f} °C more cooling, "
      f"+{_row(q['units_saved'], 'high')['diff_vs_reference']:.0f} kWh/month and a {qh['mean']:.0%} (vs {qb['mean']:.0%}) chance of a high rating. Recommend high or intermediate "
      "quality to customers whose priority is comfort or satisfaction; basic quality only where budget is binding.")
    r("")
    r("## Which customers to prioritise (knowable at the table)")
    if "3 AC" in ac and "0 AC" in ac:
        r(f"- Customers with 3 AC units show a {ac['3 AC']['high_rating_rate']:.0%} high-rating rate and {ac['3 AC']['referral_rate']:.0%} referral rate; "
          f"customers with no AC show {ac['0 AC']['high_rating_rate']:.0%} and {ac['0 AC']['referral_rate']:.0%}, and their average saving is essentially nil ({ac['0 AC']['mean_units_saved']:+.1f} kWh/month).")
    r("- Set expectations honestly with low-usage, no-AC buildings: little electricity saving is possible (a minority even see a small increase), although roof temperature still falls. "
      "Lead with comfort and roof-life messages rather than bill savings for them.")
    r("- Current electricity use, AC count and starting roof temperature are the strongest day-one signals; they are the questions to ask first.")
    r("")
    be = grid.assign(breakeven=lambda x: x.cost / (x.units_saved.clip(lower=1e-9) * 12 * 10))
    r("## Economics: be upfront about payback")
    r(f"- Predicted electricity savings are modest relative to installation cost. For a 10-year payback from electricity alone the tariff would need to be at least "
      f"₹{be.breakeven.min():,.0f}/kWh even for the best-value option ({M[be.sort_values('breakeven').iloc[0].cool_roofing_method]}, {be.sort_values('breakeven').iloc[0].cool_roofing_quality}). "
      "Use the app's payback calculator with the customer's real tariff; do not sell this on bill savings alone.")
    r("- The value case rests on cooling (roof temperature drop) and on benefits the data does not measure; position the product accordingly.")
    r("")
    r("## What not to promise")
    r(f"- Referral is only weakly predictable (hold-out ROC-AUC {svc.tasks['referral']['test']['roc_auc']:.2f}); use it to rank follow-ups, never to forecast individual customers.")
    r("- Payback depends on a tariff the data does not contain; the app computes it only from a tariff the user enters and labels it an assumption.")
    r("- Savings and satisfaction are historical associations; they are not guarantees for a new building.")
    r("")
    r("## Limits of this evidence")
    r("- Observational data: customers who chose a method may differ in unrecorded ways.")
    r("- No time or location identifiers, so drift and regional effects cannot be checked.")
    r(f"- Temperature, savings and cost models are strong (see `model_evaluation.md`); rating and referral models carry real but modest signal.")
    text = "\n".join(R) + "\n"
    (C.DOCS / "business_recommendations.md").write_text(text, encoding="utf-8")
    (C.REPORTS / "business_recommendations.md").write_text(text, encoding="utf-8")
    _figures(fi, me, grid)


def _figures(fi: pd.DataFrame, me: dict, grid: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..data.report import PAL, _style

    _style()
    tasks = C.TASK_ORDER
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for ax, t in zip(axs.ravel(), tasks):
        g = fi[fi.task == t].sort_values("share")
        ax.barh([NICE[f] for f in g.feature], g.share * 100, color=PAL["teal"])
        ax.set(title=C.TASKS[t].label, xlabel="% of importance")
    fig.suptitle("Permutation importance on the hold-out set", fontweight="bold")
    fig.tight_layout(); fig.savefig(C.FIGURES / "20_feature_importance.png", bbox_inches="tight"); plt.close(fig)

    methods = C.EXPECTED_CATEGORIES["cool_roofing_method"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    cols = {"basic": PAL["sun"], "intermediate": PAL["teal"], "high": PAL["ink"]}
    for ax, (key, ttl) in zip(axs, [("temperature_drop", "Temperature drop (°C)"), ("units_saved", "kWh saved / month"), ("high_rating", "P(high rating)")]):
        for qv in C.EXPECTED_CATEGORIES["cool_roofing_quality"]:
            g = grid[grid.cool_roofing_quality == qv].set_index("cool_roofing_method").loc[methods]
            ax.plot([M[m] for m in methods], g[key], "o-", color=cols[qv], label=qv, lw=2)
        ax.set(title=ttl); ax.tick_params(axis="x", rotation=25)
    axs[0].legend(title="quality", frameon=False)
    fig.suptitle("Predicted outcome by method and quality (concrete roofs)", fontweight="bold")
    fig.tight_layout(); fig.savefig(C.FIGURES / "21_method_quality_effects.png", bbox_inches="tight"); plt.close(fig)

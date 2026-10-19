import json
import numpy as np, pandas as pd
from scipy import stats
from coolroof import config as C
from coolroof.data import clean
from coolroof.ml.split import holdout_split

raw = pd.read_csv(C.RAW_CSV)
res = clean.clean(raw)
d, a, s = res.cleaned, res.audit, res.summary

print("== A. Is missingness informative? (two-sided tests vs rows where the field is present) ==")
for c in ["location_type", "building_size", "building_age", "num_ac_units"]:
    m = raw[c].isna()
    t_rating = stats.mannwhitneyu(raw.customer_given_rating[m], raw.customer_given_rating[~m]).pvalue
    t_ref = stats.fisher_exact([[raw.customer_referral[m].sum(), (~raw.customer_referral[m]).sum()],
                                [raw.customer_referral[~m].sum(), (~raw.customer_referral[~m]).sum()]])[1]
    t_cost = stats.mannwhitneyu(raw.cool_roofing_cost[m], raw.cool_roofing_cost[~m]).pvalue
    print(f"  {c:14s} n={m.sum():2d}  p(rating)={t_rating:.3f}  p(referral)={t_ref:.3f}  p(cost)={t_cost:.3f}")
print("  (with 4 fields x 3 outcomes = 12 tests, a Bonferroni threshold is 0.05/12 = 0.004)")
mask = raw.building_size.isna()
print("  building_size missing: median cost %d vs %d; median size-driven cost/area ratio unaffected? cost per area on observed rows only" % (raw.cool_roofing_cost[mask].median(), raw.cool_roofing_cost[~mask].median()))

print("\n== B. Cross-check: independent audit vs cleaning pipeline ==")
checks = {
    "rows": (len(raw), s["n_raw_rows"]),
    "temp 120 before": (int((raw.avg_roof_temperature_before == 120).sum()), s["temperature_sentinel_rows"]["before"]),
    "temp 120 after": (int((raw.avg_roof_temperature_after == 120).sum()), s["temperature_sentinel_rows"]["after"]),
    "0 kWh before": (int((raw.avg_units_before == 0).sum()), s["zero_consumption_placeholders"]["before"]),
    "0 kWh after": (int((raw.avg_units_after == 0).sum()), s["zero_consumption_placeholders"]["after"]),
    "missing size": (int(raw.building_size.isna().sum()), s["missing_before"]["building_size"]),
    "missing cells (4 fields)": (int(raw[["location_type", "building_size", "building_age", "num_ac_units"]].isna().sum().sum()), int((a.check == "missing_value").sum())),
    "negative savings kept (after>before, before>0)": (int(((raw.avg_units_after > raw.avg_units_before) & (raw.avg_units_before > 0)).sum()), s["negative_savings_rows"]),
}
ok = True
for k, (indep, pipe) in checks.items():
    flag = "OK " if indep == pipe else "MISMATCH"
    ok &= indep == pipe
    print(f"  {flag} {k:48s} independent={indep:5d} pipeline={pipe:5d}")
print("  ALL CROSS-CHECKS AGREE:", ok)

print("\n== C. Cleaning decisions: audit trail (from reports/cleaning_audit.csv) ==")
g = a.groupby(["check", "classification", "action"]).size().reset_index(name="records")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110)
print(g.to_string(index=False))
print("  records deleted outright from the dataset:", len(raw) - len(d), "(cleaning only sets invalid VALUES to missing; eligibility is decided per task)")
print("  raw file unchanged, cleaned output rows:", len(d))

print("\n== D. Task-specific sample counts (eligible / excluded, then train+val vs hold-out) ==")
tv, te = holdout_split(d)
rows = []
for k, t in C.TASKS.items():
    e = d[t.eligible_col]
    n_tr = int(e.loc[e.index.intersection(tv)].sum()); n_te = int(e.loc[e.index.intersection(te)].sum())
    pos = ""
    if t.kind == "classification":
        y = d.loc[e, t.target].astype(float)
        pos = f"positive rate {y.mean():.3f}"
    rows.append((k, int(e.sum()), int((~e).sum()), n_tr, n_te, pos))
print(pd.DataFrame(rows, columns=["task", "usable", "excluded", "train+val", "hold-out", "class balance"]).to_string(index=False))
print("  hold-out = %d buildings fixed before modelling (seed %d, stratified on rating); rows excluded from every task: %d" % (len(te), C.SEED, s["records_excluded_from_every_task"]))
print("  rows with at least one missing/invalidated INPUT after cleaning:", s["rows_with_any_missing_feature"], "(imputed inside training folds only)")

print("\n== E. Feature contract ==")
print("  allowed inputs:", C.PRE_INSTALLATION_FEATURES)
print("  forbidden:", C.FORBIDDEN_AS_FEATURES)

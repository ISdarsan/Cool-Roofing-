import re, hashlib
import numpy as np, pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
RAW = "dataset/cool_roofing_data.csv"
raw_bytes = open(RAW, "rb").read()
print("SHA256", hashlib.sha256(raw_bytes).hexdigest()[:16], "| archived copy identical:",
      raw_bytes == open("data/raw/cool_roofing_data.original.csv", "rb").read())

# ---------- 1. documents vs data ----------
prob = open("cool_roofing_problem.md", encoding="utf-8").read()
appr = open("approach.md", encoding="utf-8").read()
doc_cols = re.findall(r"`([a-z_]+)`", appr.split("Dataset columns for reference:")[1])
df = pd.read_csv(RAW)
print("\n== 1. DOCUMENTS vs DATA ==")
print("problem statement says 2,500 records:", "2,500" in prob, "| actual rows:", len(df))
print("approach.md column list == CSV columns (same order):", doc_cols == list(df.columns), "| n =", len(doc_cols))
print("statement says monthly electricity consumption (kWh):", "monthly electricity consumption (kWh)" in prob)
print("statement gives rating scale 1-5:", "(1-5)" in prob, "| defines 'high rating' numerically:", bool(re.search(r"high rating\W+(is|means|=|>=)", prob + appr, re.I)))
print("statement gives building_size unit:", bool(re.search(r"sq\.? ?(ft|m)|square", prob + appr, re.I)), "| gives electricity tariff:", bool(re.search(r"tariff|per unit|rs\.?/", prob + appr, re.I)))

# ---------- 2. schema ----------
print("\n== 2. SCHEMA ==")
sch = pd.DataFrame({"dtype": df.dtypes.astype(str), "non_null": df.notna().sum(), "missing": df.isna().sum(), "n_unique": df.nunique()})
print(sch)
print("shape", df.shape, "| duplicate rows", df.duplicated().sum(), "| duplicate ids", df.building_id.duplicated().sum(),
      "| ids == 1..N", (df.building_id.sort_values().to_numpy() == np.arange(1, len(df) + 1)).all(),
      "| dup rows ignoring id", df.drop(columns="building_id").duplicated().sum())
for c in ["location_type", "cool_roofing_quality", "roofing_type", "cool_roofing_method"]:
    s = df[c].dropna()
    print(f"{c}: {s.value_counts().to_dict()} | whitespace/case issues: {(s != s.str.strip().str.lower()).sum()}")
print("customer_referral values:", df.customer_referral.value_counts().to_dict(), "| dtype", df.customer_referral.dtype)
print("integer checks: rating non-int", (df.customer_given_rating % 1 != 0).sum(), "| ac non-int", (df.num_ac_units.dropna() % 1 != 0).sum(),
      "| age non-int", (df.building_age.dropna() % 1 != 0).sum())
num = ["building_size", "building_age", "cool_roofing_cost", "avg_roof_temperature_before", "num_ac_units", "avg_units_before",
       "avg_roof_temperature_after", "avg_units_after", "customer_given_rating"]
print(df[num].describe().T[["count", "min", "25%", "50%", "75%", "max", "mean", "std"]].round(2))

# ---------- 3. derived targets ----------
df["temperature_drop"] = df.avg_roof_temperature_before - df.avg_roof_temperature_after
df["units_saved"] = df.avg_units_before - df.avg_units_after
print("\n== 3. RAW TARGET DISTRIBUTIONS (before any cleaning) ==")
for c in ["cool_roofing_cost", "temperature_drop", "units_saved"]:
    s = df[c]
    print(f"{c}: min {s.min():.1f} p1 {s.quantile(.01):.1f} median {s.median():.1f} p99 {s.quantile(.99):.1f} max {s.max():.1f} skew {s.skew():.2f}")
print("rating counts", df.customer_given_rating.value_counts().sort_index().to_dict(), "| P(rating>=4)=%.3f" % (df.customer_given_rating >= 4).mean())
print("referral rate %.3f" % df.customer_referral.mean(), "| referral rate by rating:", df.groupby("customer_given_rating").customer_referral.mean().round(3).to_dict())

# ---------- 4. missingness ----------
print("\n== 4. MISSINGNESS ==")
m = df[["location_type", "building_size", "building_age", "num_ac_units"]].isna()
print("per column:", m.sum().to_dict(), "| rows with any:", m.any(axis=1).sum(), "| rows with >1 missing:", (m.sum(axis=1) > 1).sum())
print("outcomes when the field is missing vs present (cost | rating | referral rate):")
for c in m.columns:
    print(f"  {c:14s} n_missing={m[c].sum():3d} cost {df.cool_roofing_cost[m[c]].mean():>10,.0f} vs {df.cool_roofing_cost[~m[c]].mean():>10,.0f} | "
          f"rating {df.customer_given_rating[m[c]].mean():.2f} vs {df.customer_given_rating[~m[c]].mean():.2f} | "
          f"referral {df.customer_referral[m[c]].mean():.2f} vs {df.customer_referral[~m[c]].mean():.2f}")

# ---------- 5. anomalies ----------
print("\n== 5. ANOMALOUS RECORDS ==")
for col in ["avg_roof_temperature_before", "avg_roof_temperature_after"]:
    s = df[col]; vc = s.value_counts()
    print(f"{col}: most repeated value {vc.index[0]} x{vc.iloc[0]} (next most repeated x{vc.iloc[1]}) | max excluding it {s[s != vc.index[0]].max():.2f} | min {s.min():.2f}")
t120b = df.avg_roof_temperature_before == 120
t120a = df.avg_roof_temperature_after == 120
print("  120.0 rows: before", t120b.sum(), "after", t120a.sum(), "both", (t120b & t120a).sum(),
      "| implied temperature_drop on those rows: %.1f..%.1f" % (df.temperature_drop[t120b | t120a].min(), df.temperature_drop[t120b | t120a].max()))
v = df[~(t120b | t120a)]
print("  valid-row temperature_drop: min %.2f max %.2f | roof hotter after (drop<=0) among valid rows: %d" % (v.temperature_drop.min(), v.temperature_drop.max(), (v.temperature_drop <= 0).sum()))
for col in ["avg_units_before", "avg_units_after"]:
    s = df[col]
    print(f"{col}: counts of values <= 8 -> {s[s <= 8].value_counts().sort_index().to_dict()}")
z_b = df.avg_units_before == 0
z_a = df.avg_units_after == 0
ok = df[(df.avg_units_before > 0) & (df.avg_units_after > 0)]
ratio = ok.avg_units_after / ok.avg_units_before
print("  zero before: %d (max units_after on those rows: %d; with >=1 AC: %d)" % (z_b.sum(), df.avg_units_after[z_b].max(), (df.num_ac_units[z_b] >= 1).sum()))
print("  zero after : %d (units_before on those rows: min %d median %d max %d; after/before ratio is 0 vs smallest valid ratio %.3f) | both zero: %d" %
      (z_a.sum(), df.avg_units_before[z_a].min(), df.avg_units_before[z_a].median(), df.avg_units_before[z_a].max(), ratio.min(), (z_b & z_a).sum()))
neg = df[df.units_saved < 0]
print("  negative savings (after>before): %d rows; excluding zero-before rows: %d; zero-AC among them: %d; median units_before %d; max units_before %d; savings exactly 0: %d" %
      (len(neg), (neg.avg_units_before > 0).sum(), (neg.num_ac_units == 0).sum(), neg.avg_units_before.median(), neg.avg_units_before.max(), (df.units_saved == 0).sum()))
frac = (1 - ratio)
print("  mean saved-fraction by usage band (valid rows):", frac.groupby(pd.cut(ok.avg_units_before, [0, 10, 25, 50, 100, 600]), observed=True).mean().round(3).to_dict())
cps = df.cool_roofing_cost / df.building_size
g = cps.groupby([df.cool_roofing_method, df.cool_roofing_quality]).transform("median")
r = cps / g
print("  cost: min %d max %d non-positive %d | cost-per-area / (method x quality median): min %.2f max %.2f -> beyond 1/3x..3x: %d" %
      (df.cool_roofing_cost.min(), df.cool_roofing_cost.max(), (df.cool_roofing_cost <= 0).sum(), r.min(), r.max(), ((r < 1 / 3) | (r > 3)).sum()))
print("  ranges: size", (df.building_size.min(), df.building_size.max()), "age", (df.building_age.min(), df.building_age.max()),
      "ac", (df.num_ac_units.min(), df.num_ac_units.max()), "| rating outside 1-5:", ((df.customer_given_rating < 1) | (df.customer_given_rating > 5)).sum())
print("  roof x method crosstab (zero cells = never observed):")
print(pd.crosstab(df.roofing_type, df.cool_roofing_method))
print("  zero-AC but >150 kWh before:", ((df.num_ac_units == 0) & (df.avg_units_before > 150)).sum())
for c in ["building_size", "building_age", "cool_roofing_cost"]:
    vc = df[c].value_counts()
    print(f"  spike scan {c}: most repeated value {vc.index[0]} appears {vc.iloc[0]}x")

# ---------- 6. leakage-risk evidence ----------
print("\n== 6. LEAKAGE-RISK EVIDENCE ==")
d = df.copy()
d["high"] = (d.customer_given_rating >= 4).astype(int)
d["ref"] = d.customer_referral.astype(int)
print("corr(cost, building_size)=%.2f | corr(units_saved, units_before)=%.2f | corr(units_saved, units_after)=%.2f | corr(temp_drop, temp_after)=%.2f" %
      (d.cool_roofing_cost.corr(d.building_size), d.units_saved.corr(d.avg_units_before), d.units_saved.corr(d.avg_units_after), d.temperature_drop.corr(d.avg_roof_temperature_after)))
print("post-install vs rating: corr(rating, units_saved)=%.2f corr(rating, temp_drop)=%.2f corr(rating, cost)=%.2f" %
      (d.customer_given_rating.corr(d.units_saved), d.customer_given_rating.corr(d.temperature_drop), d.customer_given_rating.corr(d.cool_roofing_cost)))
print("P(referral | rating>=4)=%.2f vs P(referral | rating<=2)=%.2f" % (d.ref[d.high == 1].mean(), d.ref[d.customer_given_rating <= 2].mean()))
print("building_id vs targets (corr): cost %.3f rating %.3f referral %.3f" % (d.building_id.corr(d.cool_roofing_cost), d.building_id.corr(d.customer_given_rating), d.building_id.corr(d.ref)))

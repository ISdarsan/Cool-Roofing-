"""Independent replication of the WHOLE selection procedure on a hold-out that never influenced any decision.

Why: my earliest exploratory experiments used train_test_split(seed=42, no stratification) and ran CV on its 80 %
part. The production hold-out (stratified, seed 42) overlaps that exploration pool by ~81 %, so it is not pristine
at the design level (model-family ideas were informed by it; no parameter was fitted on it).
The 500 rows of the exploration split's *untouched* 20 % were never used for any decision. Here they become the
hold-out and hyper-parameter tuning + model selection + evaluation are rerun from scratch on the other 2,000 rows.
Models are not saved; production artifacts are untouched.

usage: python scripts/review_replication.py pristine | <seed>
"""
import json
import sys
import numpy as np
from sklearn.model_selection import train_test_split

from coolroof import config as C
from coolroof.data.clean import load_cleaned
from coolroof.ml.train import train_task

mode = sys.argv[1] if len(sys.argv) > 1 else "pristine"
df = load_cleaned()
if mode == "pristine":
    tv, te = train_test_split(df.index.to_numpy(), test_size=0.2, random_state=42)   # the exploration split
    tag = "pristine"
else:
    strat = df["customer_given_rating"].fillna(0).astype(int)
    tv, te = train_test_split(df.index.to_numpy(), test_size=C.TEST_SIZE, random_state=int(mode), stratify=strat)
    tag = f"seed{mode}"
orig_te = set(json.load(open(C.MODELS / "split.json"))["test_building_ids"])
ex_pool = set(train_test_split(df.index.to_numpy(), test_size=0.2, random_state=42)[0])
new_te_ids = set(df.loc[te, C.ID_COL])
print(f"[{tag}] hold-out {len(te)} buildings | overlap with production hold-out: {len(orig_te & new_te_ids)} | "
      f"share inside the early-exploration CV pool: {100 * np.mean([i in ex_pool for i in te]):.1f}%", flush=True)

out = {"mode": tag, "overlap_with_production_holdout": len(orig_te & new_te_ids), "tasks": {}}
for k in C.TASK_ORDER:
    r = train_task(k, df, np.sort(tv), np.sort(te), log=lambda *a, **kw: None)["result"]
    out["tasks"][k] = {"selected": r["selected"], "n_test": r["n_test"],
                       "test": {m: v for m, v in r["test"].items() if isinstance(v, (int, float))},
                       "baseline": {m: v for m, v in r["test_baseline"].items() if isinstance(v, (int, float))},
                       "validation": r["validation"]}
    t = r["test"]
    print(k, r["selected"], {m: round(t[m], 4) for m in ("mae", "rmse", "r2", "accuracy", "f1", "roc_auc") if m in t}, flush=True)
json.dump(out, open(f"reports/review_replication_{tag}.json", "w"), indent=1, default=float)
print("DONE", flush=True)

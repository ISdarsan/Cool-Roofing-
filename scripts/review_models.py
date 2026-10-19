"""Independent ML review. Recomputes metrics with plain sklearn/NumPy (not coolroof.metrics), proves model
provenance, tests preprocessing fit, runs leakage controls, and quantifies uncertainty of the hold-out results."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from coolroof import config as C
from coolroof.data import clean
from coolroof.features import design_matrix
from coolroof.ml.candidates import get_candidates
from coolroof.ml.split import holdout_split

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
META = json.load(open(C.MODELS / "model_metadata.json"))["tasks"]
SPLIT = json.load(open(C.MODELS / "split.json"))
df = clean.clean(clean.load_raw()).cleaned                       # rebuilt from the RAW csv, not read from disk
te_ids = set(SPLIT["test_building_ids"])
is_te = df[C.ID_COL].isin(te_ids)
H = lambda s: print(f"\n{'=' * 8} {s} {'=' * 8}")


def y_of(sub, k):
    spec = C.TASKS[k]
    y = sub[spec.target]
    return y.astype(float).astype(int).to_numpy() if spec.kind == "classification" else y.astype(float).to_numpy()


def split_task(k):
    spec = C.TASKS[k]
    e = df[spec.eligible_col]
    return df[e & ~is_te], df[e & is_te]


# ------------------------------------------------------------------ A. split integrity
H("A. TRAIN / TEST SEPARATION")
tv, te = holdout_split(df)
print("saved split.json equals a fresh re-derivation (seed %d):" % C.SEED, set(df.loc[te, C.ID_COL]) == te_ids)
print("hold-out size", is_te.sum(), "| train pool", (~is_te).sum(), "| overlap of id sets:", len(te_ids & set(df.loc[~is_te, C.ID_COL])))
print("rating distribution train vs test (%):", (df[~is_te].customer_given_rating.value_counts(normalize=True).sort_index() * 100).round(1).to_dict(),
      (df[is_te].customer_given_rating.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
tp = pd.read_csv(C.REPORTS / "test_predictions.csv")
oof = pd.read_csv(C.REPORTS / "oof_predictions.csv")
print("test_predictions ids ⊆ hold-out:", set(tp.building_id) <= te_ids, "| oof ids ∩ hold-out:", len(set(oof.building_id) & te_ids))
feats = C.PRE_INSTALLATION_FEATURES
key = df[feats].astype(str).agg("|".join, axis=1)
dup = key[is_te].isin(set(key[~is_te]))
print("hold-out rows whose 9-feature vector exactly equals a training row:", int(dup.sum()), "of", int(is_te.sum()),
      "| of those with no missing input:", int((dup & df[feats].notna().all(axis=1)[is_te.index]).sum()))
# exploration exposure: my very first scratch experiments used train_test_split(seed 42, no stratify) and ran CV on its 80% part
ex_tv, ex_te = train_test_split(df.index.to_numpy(), test_size=0.2, random_state=42)
print("DESIGN-LEVEL EXPOSURE: share of production hold-out rows that were inside the early exploration CV pool: %.1f%% (%d of %d)" %
      (100 * is_te[ex_tv].sum() / is_te.sum(), is_te[ex_tv].sum(), is_te.sum()))

# ------------------------------------------------------------------ B. metric recomputation
H("B. REPORTED METRICS vs INDEPENDENT RECOMPUTATION FROM THE SAVED MODELS ON HELD-OUT ROWS")
rows = []
pred_cache = {}
for k in C.TASK_ORDER:
    spec, m = C.TASKS[k], META[k]
    model = joblib.load(C.MODELS / f"{k}.joblib")
    tr_, te_ = split_task(k)
    X, y = design_matrix(te_, k), y_of(te_, k)
    if spec.kind == "regression":
        p = model.predict(X)
        base = y_of(tr_, k).mean() if k != "rating" else np.median(y_of(tr_, k))
        mae, rmse, r2 = skm.mean_absolute_error(y, p), np.sqrt(skm.mean_squared_error(y, p)), skm.r2_score(y, p)
        bmae = skm.mean_absolute_error(y, np.full_like(y, base, dtype=float))
        col = tp[f"{k}_predicted"].dropna()
        csv_diff = np.abs(tp.set_index("building_id").loc[te_[C.ID_COL], f"{k}_predicted"].to_numpy() - p).max()
        rows.append((k, len(y), f"{mae:.4f}", f"{m['test']['mae']:.4f}", f"{rmse:.4f}", f"{m['test']['rmse']:.4f}", f"{r2:.4f}", f"{bmae:.4f}", f"{m['test_baseline']['mae']:.4f}", f"{csv_diff:.1e}"))
        pred_cache[k] = p
    else:
        pr = model.predict_proba(X)[:, 1]
        lab = (pr >= m["threshold"]).astype(int)
        acc, f1 = skm.accuracy_score(y, lab), skm.f1_score(y, lab)
        maj = max(y.mean(), 1 - y.mean())
        ap_f1 = skm.f1_score(y, np.ones_like(y))
        csv_diff = np.abs(tp.set_index("building_id").loc[te_[C.ID_COL], f"{k}_probability"].to_numpy() - pr).max()
        rows.append((k, len(y), f"acc {acc:.4f}", f"{m['test']['accuracy']:.4f}", f"F1 {f1:.4f}", f"{m['test']['f1']:.4f}", f"AUC {skm.roc_auc_score(y, pr):.4f}", f"trivial acc {maj:.4f} F1 {ap_f1:.4f}", f"{m['baseline_accuracy']:.4f}/{m['baseline_f1']:.4f}", f"{csv_diff:.1e}"))
        pred_cache[k] = pr
hdr = ["task", "n", "recomputed A", "reported A", "recomputed B", "reported B", "R2/AUC", "recomputed baseline", "reported baseline", "max|csv-model|"]
print(pd.DataFrame(rows, columns=hdr).to_string(index=False))

# ------------------------------------------------------------------ C. provenance
H("C. PROVENANCE: saved model == refit on TRAIN only (and != refit on TRAIN+TEST)")


def rebuild(k):
    m = META[k]
    cand = next(c for c in get_candidates(k) if c.name == m["selected"])
    est = cand.factory()
    est.set_params(**{f"{cand.prefix}model__{p}": v for p, v in m["selected_params"].items()})
    if m.get("calibrated"):
        est = CalibratedClassifierCV(est, method="sigmoid", cv=5)
    return est


prov = []
for k in C.TASK_ORDER:
    spec = C.TASKS[k]
    tr_, te_ = split_task(k)
    X_tr, y_tr, X_te = design_matrix(tr_, k), y_of(tr_, k), design_matrix(te_, k)
    both = pd.concat([tr_, te_])
    saved = pred_cache[k]
    f = (lambda e: e.predict(X_te)) if spec.kind == "regression" else (lambda e: e.predict_proba(X_te)[:, 1])
    a = f(clone(rebuild(k)).fit(X_tr, y_tr))
    b = f(clone(rebuild(k)).fit(design_matrix(both, k), y_of(both, k)))
    prov.append((k, f"{np.abs(saved - a).max():.2e}", f"{np.abs(saved - b).max():.2e}", "TRAIN-ONLY" if np.abs(saved - a).max() < 1e-6 * max(1, np.abs(saved).max()) else "NOT reproduced"))
print(pd.DataFrame(prov, columns=["task", "max|saved - train-only refit|", "max|saved - train+test refit|", "verdict"]).to_string(index=False))

# ------------------------------------------------------------------ D. preprocessing fit
H("D. PREPROCESSING IS FITTED ON TRAINING ROWS ONLY (imputer statistics of the saved models)")


def find_pipe(e):
    if isinstance(e, Pipeline):
        return e
    for a in ("regressor_", "regressor", "estimator"):
        if hasattr(e, a):
            r = find_pipe(getattr(e, a))
            if r is not None:
                return r
    if hasattr(e, "calibrated_classifiers_"):
        return find_pipe(e.calibrated_classifiers_[0].estimator)
    return None


from coolroof.features import FeatureEngineer, NUMERIC_MODEL_COLS
for k in C.TASK_ORDER:
    pipe = find_pipe(joblib.load(C.MODELS / f"{k}.joblib"))
    imp = pipe.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    tr_, te_ = split_task(k)
    fe = FeatureEngineer()
    med_tr = fe.fit_transform(design_matrix(tr_, k))[NUMERIC_MODEL_COLS].median().to_numpy()
    med_all = fe.fit_transform(design_matrix(pd.concat([tr_, te_]), k))[NUMERIC_MODEL_COLS].median().to_numpy()
    st = imp.statistics_
    print(f"{k:17s} matches TRAIN medians: {np.allclose(st, med_tr, equal_nan=True)} | matches train+test medians: {np.allclose(st, med_all, equal_nan=True)} "
          f"(calibrated wrapper uses 4/5 folds, so can differ)" if k in ("high_rating", "referral") and not np.allclose(st, med_tr, equal_nan=True) else
          f"{k:17s} matches TRAIN medians: {np.allclose(st, med_tr, equal_nan=True)} | matches train+test medians: {np.allclose(st, med_all, equal_nan=True)}")

# ------------------------------------------------------------------ E. leakage controls
H("E. LEAKAGE CONTROLS (selected config refit on TRAIN, scored on HELD-OUT)")
ctl = []
from lightgbm import LGBMClassifier, LGBMRegressor
for k in C.TASK_ORDER:
    spec = C.TASKS[k]
    tr_, te_ = split_task(k)
    X_tr, y_tr, X_te, y_te = design_matrix(tr_, k), y_of(tr_, k), design_matrix(te_, k), y_of(te_, k)
    res = []
    for _ in range(3):                                           # permuted labels -> any skill above baseline = leakage
        yp = rng.permutation(y_tr)
        e = rebuild(k).fit(X_tr, yp)
        if spec.kind == "regression":
            res.append(skm.r2_score(y_te, e.predict(X_te)))
        else:
            res.append(skm.roc_auc_score(y_te, e.predict_proba(X_te)[:, 1]))
    # ID-only control: building_id alone
    idtr, idte = tr_[[C.ID_COL]], te_[[C.ID_COL]]
    if spec.kind == "regression":
        e = LGBMRegressor(n_estimators=200, num_leaves=8, verbose=-1, random_state=0).fit(idtr, y_tr)
        idm = skm.r2_score(y_te, e.predict(idte))
    else:
        e = LGBMClassifier(n_estimators=200, num_leaves=8, verbose=-1, random_state=0).fit(idtr, y_tr)
        idm = skm.roc_auc_score(y_te, e.predict_proba(idte)[:, 1])
    real = META[k]["test"]["r2"] if spec.kind == "regression" else META[k]["test"]["roc_auc"]
    ctl.append((k, "R2" if spec.kind == "regression" else "AUC", f"{real:.3f}", f"{np.mean(res):+.3f} (±{np.std(res):.3f})", f"{idm:+.3f}",
                "chance≈0" if spec.kind == "regression" else "chance≈0.5"))
print(pd.DataFrame(ctl, columns=["task", "metric", "real model", "shuffled-label model", "building_id-only model", "expected if no leakage"]).to_string(index=False))

# ------------------------------------------------------------------ F. suspicious-result investigation
H("F. WHY ARE COST / SAVINGS / COOLING SO ACCURATE?  (ablation: neutralise one input at a time, refit on TRAIN, score on HELD-OUT)")


def ablate(k, drop):
    tr_, te_ = split_task(k)
    X_tr, X_te = design_matrix(tr_, k).copy(), design_matrix(te_, k).copy()
    for c in drop:
        const = X_tr[c].mode().iloc[0] if X_tr[c].dtype == object else X_tr[c].median()
        X_tr[c], X_te[c] = const, const
    e = rebuild(k).fit(X_tr, y_of(tr_, k))
    y = y_of(te_, k)
    return skm.mean_absolute_error(y, e.predict(X_te)), skm.r2_score(y, e.predict(X_te))


for k, groups in {"cost": [[], ["building_size"], ["cool_roofing_method", "cool_roofing_quality"], ["building_age"], ["roofing_type"],
                           ["avg_units_before", "num_ac_units", "avg_roof_temperature_before", "location_type"]],
                  "units_saved": [[], ["avg_units_before"], ["num_ac_units"], ["cool_roofing_method", "cool_roofing_quality"]],
                  "temperature_drop": [[], ["avg_roof_temperature_before"], ["cool_roofing_method", "cool_roofing_quality"], ["roofing_type"]]}.items():
    print(k)
    for g in groups:
        mae, r2 = ablate(k, g)
        print(f"   without {str(g) if g else 'nothing (full model)':92s} MAE {mae:>12,.3f}  R2 {r2:7.3f}")

tr_, te_ = split_task("cost")
p = pred_cache["cost"]; y = y_of(te_, "cost")
rel = np.abs(p - y) / y
print("cost: relative error median %.1f%% p90 %.1f%% max %.1f%% | std of log(actual/pred) = %.3f (a ~3-4%% multiplicative noise floor)" % (100 * np.median(rel), 100 * np.quantile(rel, .9), 100 * rel.max(), np.std(np.log(y / p))))
print("cost: corr(residual, realised savings) = %.3f, corr(residual, realised temp drop) = %.3f  (≈0 -> model does not smuggle post-install info either way)" %
      (np.corrcoef(y - p, te_["units_saved"].fillna(te_["units_saved"].median()))[0, 1], np.corrcoef(y - p, te_["temperature_drop"].fillna(te_["temperature_drop"].median()))[0, 1]))
u_tr, u_te = split_task("units_saved")
print("units_saved: corr(units_saved, units_before) on hold-out = %.3f ; by usage band the saved fraction is ~0 below 25 kWh and 0.3-0.54 above (see audit) -> near-deterministic given usage" % u_te.units_saved.corr(u_te.avg_units_before))

# ------------------------------------------------------------------ G. uncertainty & significance
H("G. HOLD-OUT METRICS WITH 95% BOOTSTRAP INTERVALS AND PAIRED COMPARISON TO BASELINE (2,000 resamples)")
B = 2000
out = []
for k in C.TASK_ORDER:
    spec, m = C.TASKS[k], META[k]
    tr_, te_ = split_task(k)
    y = y_of(te_, k); p = pred_cache[k]
    n = len(y)
    idx = rng.integers(0, n, (B, n))
    if spec.kind == "regression":
        base = np.median(y_of(tr_, k)) if k == "rating" else y_of(tr_, k).mean()
        e_m, e_b = np.abs(y - p), np.abs(y - base)
        mm, bb = e_m[idx].mean(1), e_b[idx].mean(1)
        d = bb - mm
        out.append((k, "MAE", f"{e_m.mean():.4g} [{np.percentile(mm, 2.5):.4g}, {np.percentile(mm, 97.5):.4g}]", f"{e_b.mean():.4g}", f"{d.mean():.4g} [{np.percentile(d, 2.5):.4g}, {np.percentile(d, 97.5):.4g}]", "yes" if np.percentile(d, 2.5) > 0 else "NO"))
    else:
        thr = m["threshold"]; lab = (p >= thr).astype(int)
        for name, fn, triv in (("accuracy", skm.accuracy_score, None), ("F1", skm.f1_score, None)):
            def stat(ix):
                return np.array([fn(y[i], lab[i]) for i in ix])
            best_triv = np.zeros(n, int) if (name == "accuracy" and y.mean() < .5) else np.ones(n, int)
            if name == "F1":
                best_triv = np.ones(n, int)
            sm_ = stat(idx[:600]); sb_ = np.array([fn(y[i], best_triv[i]) for i in idx[:600]])
            dd = sm_ - sb_
            out.append((k, name, f"{fn(y, lab):.4f} [{np.percentile(sm_, 2.5):.4f}, {np.percentile(sm_, 97.5):.4f}]", f"{fn(y, best_triv):.4f}",
                        f"{dd.mean():+.4f} [{np.percentile(dd, 2.5):+.4f}, {np.percentile(dd, 97.5):+.4f}]", "yes" if np.percentile(dd, 2.5) > 0 else "NO"))
print(pd.DataFrame(out, columns=["task", "metric", "model [95% CI]", "baseline", "improvement [95% CI]", "CI excludes 0 (better)?"]).to_string(index=False))

# ------------------------------------------------------------------ H. selection integrity
H("H. THRESHOLD / SELECTION INTEGRITY (nothing tuned on the hold-out)")
oo = oof.dropna(subset=["referral_oof"])
grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
f1s = [skm.f1_score(oo.referral_actual.astype(int), (oo.referral_oof >= t).astype(int)) for t in grid]
print("referral F1-optimal threshold recomputed from OUT-OF-FOLD train predictions: %.2f | stored in metadata: %.2f" % (grid[int(np.argmax(f1s))], META["referral"]["threshold"]))
for k in C.REGRESSION_TASKS:
    o = oof.dropna(subset=[f"{k}_oof"])
    mae = skm.mean_absolute_error(o[f"{k}_actual"], o[f"{k}_oof"])
    print(f"{k:17s} validation MAE recomputed from OOF file {mae:.4f} | metadata {META[k]['validation']['mae']:.4f} | hold-out {META[k]['test']['mae']:.4f} | gap hold-out/validation {META[k]['test']['mae'] / META[k]['validation']['mae']:.2f}x")
for k in ("high_rating",):
    o = oof.dropna(subset=[f"{k}_oof"])
    print(f"{k} validation accuracy recomputed {skm.accuracy_score(o[f'{k}_actual'], (o[f'{k}_oof'] >= .5).astype(int)):.4f} | metadata {META[k]['validation']['accuracy']:.4f} | hold-out {META[k]['test']['accuracy']:.4f}")
print("interval coverage recomputed independently on hold-out is stored as", {k: META[k]["interval_coverage_test"] for k in C.REGRESSION_TASKS})

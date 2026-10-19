// Minimal MOCK API payloads for UI tests. They exercise rendering only; they are not model results.
const rel = { level: "high", basis: "mock" };
const reg = (v: number, u: string) => ({ value: v, unit: u, interval: { level: 0.8, low: v * 0.9, high: v * 1.1 }, reliability: rel });

export const status = {
  model_version: "9.9.9", trained_at_utc: "2026-01-02T03:04:05+00:00",
  data: { raw_csv_sha256: "x", n_rows: 2500, n_train_validation_buildings: 2000, n_test_buildings: 500 },
  high_rating_threshold: 4,
  assistant: { enabled: false, model: null, message: "The AI sales assistant is optional and currently disabled. Set ANTHROPIC_API_KEY." },
  headline: Object.fromEntries(["cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral"].map((k) => [k, {
    label: k, kind: "regression", unit: "u", selected: "Mock model", n_train_validation: 2000, n_test: 500, beats_baseline: true, reliability: rel,
    test: { mae: 1000, rmse: 1500, r2: 0.99, accuracy: 0.7, f1: 0.6, roc_auc: 0.75, precision: 0.5, recall: 0.5 },
    baseline: { mae: 5000, rmse: 6000, accuracy: 0.5, f1: 0.5 }, best_trivial: { accuracy: 0.6, f1: 0.54 },
  }])),
};

export const schema = {
  features: {}, required: [], optional_imputed: [], min_pair_support: 30, high_rating_threshold: 4, training_ranges: {},
  method_roof_support: [
    ...["basic_coating", "sheet", "sheet_with_coating", "tiles", "tiles_with_coating"].map((m) => ({ roofing_type: "concrete_flat", cool_roofing_method: m, n: 100 })),
    ...["basic_coating", "sheet", "sheet_with_coating"].map((m) => ({ roofing_type: "sheet", cool_roofing_method: m, n: 100 })),
    ...["tiles", "tiles_with_coating"].map((m) => ({ roofing_type: "sheet", cool_roofing_method: m, n: 0 })),
  ],
  objectives: { balanced: "Balanced overall value (weighted)", cost: "Lowest estimated installation cost", temperature_drop: "Greatest estimated temperature drop",
    units_saved: "Greatest estimated electricity savings", high_rating: "Best predicted customer satisfaction", referral: "Best predicted referral likelihood" },
  default_weights: { cost: 0.25, temperature_drop: 0.15, units_saved: 0.25, high_rating: 0.15, referral: 0.2 },
  metrics: {},
};

export const predict = {
  model_version: "9.9.9",
  predictions: {
    cost: reg(1_500_000, "INR"), temperature_drop: reg(9.5, "°C"), units_saved: reg(100, "kWh/month"),
    rating: { value: 4.1, unit: "stars", scale: [1, 5], reliability: { level: "low", basis: "mock" }, note: "Expected star rating." },
    high_rating: { probability: 0.77, cutoff: 0.5, predicted_high: true, definition: "customer rating ≥ 4", reliability: { level: "moderate", basis: "mock" } },
    referral: { probability: 0.58, cutoff: 0.25, predicted_referral: true, reliability: { level: "moderate", basis: "mock" } },
  },
  derived: { avg_roof_temperature_after: { value: 24.5, unit: "°C", note: "derived" }, avg_units_after: { value: 40, unit: "kWh", note: "derived" } },
  warnings: ["Not provided, so filled with typical training values: building_age."],
  assumptions: ["'High rating' means a customer rating of 4 or more."],
};

const opt = (m: string, q: string, rank: number, cost: number, drop: number) => ({
  id: `${m}|${q}`, cool_roofing_method: m, cool_roofing_quality: q, support: 50, low_support: false, is_current_selection: rank === 2, rank, score: 100 - rank * 10,
  predictions: { cost, temperature_drop: drop, units_saved: 80, rating: 3.5, high_rating: 0.6, referral: 0.4 },
  intervals: { cost: [cost * 0.9, cost * 1.1], temperature_drop: [drop - 1, drop + 1], units_saved: [70, 90] },
  normalised: {}, contributions: { cost: 0.1, temperature_drop: 0.1, units_saved: 0.1, high_rating: 0.1, referral: 0.1 },
});
export const compare = {
  objective: "balanced", objective_label: "Balanced overall value (weighted)", basis: "Weighted sum of min–max normalised metrics.",
  weights: { cost: 0.25 }, score_margin: 10, close_call: false, notes: [], warnings: [], normalisation_note: "Scores are relative.",
  metric_info: {
    cost: { label: "Installation cost", direction: "min", unit: "INR" }, temperature_drop: { label: "Temperature drop", direction: "max", unit: "°C" },
    units_saved: { label: "Electricity saved", direction: "max", unit: "kWh" }, high_rating: { label: "P(high)", direction: "max", unit: "p" }, referral: { label: "P(ref)", direction: "max", unit: "p" },
  },
  recommended: opt("tiles_with_coating", "high", 1, 1_650_000, 10.2),
  options: [opt("tiles_with_coating", "high", 1, 1_650_000, 10.2), opt("sheet_with_coating", "high", 2, 1_500_000, 9.5), opt("basic_coating", "basic", 3, 650_000, 3.6)],
};

export const models = { meta: { data: { n_test_buildings: 500, n_train_validation_buildings: 2000 }, cv_folds: 5 }, tasks: {}, candidates: {}, task_order: [] };
export const dataQuality = {
  profile: {}, summary: { n_raw_rows: 2500, zero_consumption_placeholders: { before: 26, after: 22 }, temperature_sentinel_rows: { before: 32, after: 26 },
    negative_savings_rows: 167, rows_with_any_missing_feature: 164 },
};
export const insights = { marginal_effects: { method: {}, grid: [] }, feature_importance: [] };

export const FORM_OK = {
  location_type: "urban", building_size: "1500", roofing_type: "concrete_flat", building_age: "20",
  avg_roof_temperature_before: "34", num_ac_units: "2", avg_units_before: "140", cool_roofing_method: "sheet_with_coating", cool_roofing_quality: "high",
};

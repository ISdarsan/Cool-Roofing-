export type Location = "rural" | "intermediate" | "urban" | "dense_urban";
export type RoofType = "concrete_flat" | "concrete_slanted" | "sheet" | "tiles";
export type Quality = "basic" | "intermediate" | "high";
export type Method = "basic_coating" | "sheet" | "sheet_with_coating" | "tiles" | "tiles_with_coating";
export type TaskKey = "cost" | "temperature_drop" | "units_saved" | "rating" | "high_rating" | "referral";
export type MetricKey = "cost" | "temperature_drop" | "units_saved" | "high_rating" | "referral";
export type Objective = MetricKey | "balanced";

export interface BuildingBase {
  location_type: Location;
  building_size: number;
  roofing_type: RoofType;
  building_age: number | null;
  avg_roof_temperature_before: number | null;
  num_ac_units: number | null;
  avg_units_before: number | null;
}
export interface BuildingInput extends BuildingBase {
  cool_roofing_quality: Quality;
  cool_roofing_method: Method;
}

export interface Reliability { level: "high" | "moderate" | "low"; basis: string }
export interface Interval { level: number; low: number; high: number }
export interface RegPrediction { value: number; unit: string; interval: Interval; reliability: Reliability }
export interface Predictions {
  cost: RegPrediction;
  temperature_drop: RegPrediction;
  units_saved: RegPrediction;
  rating: { value: number; unit: string; scale: [number, number]; reliability: Reliability; note: string };
  high_rating: { probability: number; cutoff: number; predicted_high: boolean; definition: string; reliability: Reliability };
  referral: { probability: number; cutoff: number; predicted_referral: boolean; reliability: Reliability };
}
export interface PredictResponse {
  model_version: string;
  predictions: Predictions;
  derived: Record<string, { value: number; unit: string; note: string }>;
  warnings: string[];
  assumptions: string[];
}

export interface OptionRow {
  id: string;
  cool_roofing_method: Method;
  cool_roofing_quality: Quality;
  support: number | null;
  low_support: boolean;
  is_current_selection: boolean;
  rank: number;
  score: number;
  predictions: Record<MetricKey | "rating", number>;
  intervals: Record<"cost" | "temperature_drop" | "units_saved", [number, number]>;
  normalised: Record<string, number>;
  contributions: Record<string, number>;
}
export interface CompareResponse {
  objective: Objective;
  objective_label: string;
  basis: string;
  weights: Record<string, number>;
  recommended: OptionRow;
  options: OptionRow[];
  score_margin: number | null;
  close_call: boolean;
  notes: string[];
  warnings: string[];
  metric_info: Record<MetricKey, { label: string; direction: "min" | "max"; unit: string }>;
  normalisation_note: string;
}

export interface Contribution { feature: string; value: string | number | null; contribution: number }
export interface ExplainResponse {
  task: TaskKey;
  unit: string;
  base_value: number;
  prediction: number;
  contributions: Contribution[];
  method: string;
  caveat: string;
}

export interface PaybackCase {
  monthly_kwh_saved: number;
  annual_bill_saving_inr: number;
  annual_net_saving_inr: number;
  breakeven_tariff_inr_per_kwh: number | null;
  simple_payback_years: number | null;
  payback_with_escalation_years: number | null;
  net_position_after_horizon_inr: number;
}
export interface EconomicsResponse {
  inputs: Record<string, number>;
  central: PaybackCase;
  range: { pessimistic: PaybackCase; optimistic: PaybackCase } | null;
  assumptions: string[];
  warnings: string[];
}

export interface SchemaResponse {
  features: Record<string, { type: string; allowed?: string[]; min?: number; max?: number; decision_variable: boolean }>;
  required: string[];
  optional_imputed: string[];
  method_roof_support: { roofing_type: RoofType; cool_roofing_method: Method; n: number }[];
  training_ranges: Record<string, [number, number]>;
  min_pair_support: number;
  objectives: Record<Objective, string>;
  default_weights: Record<MetricKey, number>;
  metrics: Record<MetricKey, { label: string; direction: "min" | "max"; unit: string }>;
  high_rating_threshold: number;
}

export interface HeadlineTask {
  label: string;
  kind: "regression" | "classification";
  unit: string;
  selected: string;
  n_train_validation: number;
  n_test: number;
  test: Record<string, number>;
  baseline: Record<string, number>;
  beats_baseline: boolean;
  reliability: Reliability;
  best_trivial?: { accuracy: number; f1: number };
  significance?: { verdict?: string; accuracy?: string; f1?: string; auc_above_chance?: boolean; auc_ci95?: [number, number] };
}
export interface StatusResponse {
  model_version: string;
  trained_at_utc: string;
  data: { raw_csv_sha256: string; n_rows: number; n_train_validation_buildings: number; n_test_buildings: number };
  headline: Record<TaskKey, HeadlineTask>;
  high_rating_threshold: number;
  assistant: { enabled: boolean; provider?: string; free?: boolean; model: string | null; message: string | null };
}

// Loosely typed: these come straight from generated report JSON.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Json = any;

export interface ModelsResponse {
  meta: Json;
  tasks: Record<TaskKey, Json>;
  candidates: Record<TaskKey, Json[]>;
  task_order: TaskKey[];
}
export interface DataQualityResponse { profile: Json; summary: Json }

export interface ChatToolResult { tool: string; input: Json; output: Json }
export interface ChatResponse { reply: string; tool_results: ChatToolResult[]; provider?: string; notice?: string }

import type { Location, MetricKey, Method, Quality, RoofType, TaskKey } from "./types";

export const METHOD_LABEL: Record<Method, string> = {
  basic_coating: "Basic coating", sheet: "Sheet", sheet_with_coating: "Sheet + coating",
  tiles: "Tiles", tiles_with_coating: "Tiles + coating",
};
export const METHOD_HINT: Record<Method, string> = {
  basic_coating: "Reflective coating applied on any roof type",
  sheet: "Reflective sheet product",
  sheet_with_coating: "Reflective sheet plus a top coating",
  tiles: "Reflective tile product",
  tiles_with_coating: "Reflective tiles plus a top coating",
};
export const QUALITY_LABEL: Record<Quality, string> = { basic: "Basic", intermediate: "Intermediate", high: "High" };
export const ROOF_LABEL: Record<RoofType, string> = {
  concrete_flat: "Concrete – flat", concrete_slanted: "Concrete – slanted", sheet: "Sheet", tiles: "Tiles",
};
export const LOCATION_LABEL: Record<Location, string> = {
  rural: "Rural", intermediate: "Intermediate town", urban: "Urban", dense_urban: "Dense urban",
};
export const TASK_LABEL: Record<TaskKey, string> = {
  cost: "Installation cost", temperature_drop: "Roof temperature drop", units_saved: "Monthly electricity saved",
  rating: "Customer rating", high_rating: "High-rating probability", referral: "Referral probability",
};
export const METRIC_LABEL: Record<MetricKey, string> = {
  cost: "Installation cost", temperature_drop: "Temperature drop", units_saved: "Electricity saved",
  high_rating: "P(high rating)", referral: "P(referral)",
};
export const FEATURE_LABEL: Record<string, string> = {
  location_type: "Location type", building_size: "Building size", building_age: "Building age", roofing_type: "Existing roof",
  cool_roofing_quality: "Quality grade", cool_roofing_method: "Roofing method",
  avg_roof_temperature_before: "Starting roof temperature", num_ac_units: "AC units", avg_units_before: "Current electricity use",
};
export const METHOD_ORDER: Method[] = ["basic_coating", "sheet", "sheet_with_coating", "tiles", "tiles_with_coating"];
export const QUALITY_ORDER: Quality[] = ["basic", "intermediate", "high"];
export const LOCATION_ORDER: Location[] = ["rural", "intermediate", "urban", "dense_urban"];
export const ROOF_ORDER: RoofType[] = ["concrete_flat", "concrete_slanted", "sheet", "tiles"];
export const TASK_ORDER: TaskKey[] = ["cost", "temperature_drop", "units_saved", "rating", "high_rating", "referral"];

/** Method options valid for a roof type, from the schema's data-derived support table. */
export function supportedMethods(
  roof: RoofType,
  support: { roofing_type: RoofType; cool_roofing_method: Method; n: number }[] | undefined,
  minSupport = 30,
): Method[] {
  if (!support) return METHOD_ORDER;
  return METHOD_ORDER.filter((m) =>
    (support.find((s) => s.roofing_type === roof && s.cool_roofing_method === m)?.n ?? 0) >= minSupport);
}

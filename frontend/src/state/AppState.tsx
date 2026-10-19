import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { api } from "../lib/api";
import { safeGet, safeSet, useAsync } from "../lib/hooks";
import type { AsyncState } from "../lib/hooks";
import type {
  BuildingBase, BuildingInput, Location, Method, PredictResponse, Quality, RoofType, SchemaResponse, StatusResponse,
} from "../lib/types";

/* ------------------------------------------------------------------ form model */
export interface FormState {
  location_type: Location;
  building_size: string;
  roofing_type: RoofType;
  building_age: string;
  avg_roof_temperature_before: string;
  num_ac_units: string;
  avg_units_before: string;
  cool_roofing_method: Method;
  cool_roofing_quality: Quality;
}

export const EMPTY_FORM: FormState = {
  location_type: "urban", building_size: "", roofing_type: "concrete_flat", building_age: "",
  avg_roof_temperature_before: "", num_ac_units: "", avg_units_before: "",
  cool_roofing_method: "sheet_with_coating", cool_roofing_quality: "intermediate",
};
export const EXAMPLE_FORM: FormState = {
  location_type: "urban", building_size: "1500", roofing_type: "concrete_flat", building_age: "20",
  avg_roof_temperature_before: "34", num_ac_units: "2", avg_units_before: "140",
  cool_roofing_method: "sheet_with_coating", cool_roofing_quality: "high",
};

export type FieldErrors = Partial<Record<keyof FormState, string>>;

const RANGES: Record<string, { min: number; max: number; label: string; integer?: boolean }> = {
  building_size: { min: 50, max: 50000, label: "Building size" },
  building_age: { min: 0, max: 100, label: "Building age" },
  avg_roof_temperature_before: { min: 10, max: 80, label: "Roof temperature" },
  num_ac_units: { min: 0, max: 10, label: "AC units", integer: true },
  avg_units_before: { min: 0, max: 5000, label: "Electricity use" },
};

export function validateField(name: keyof FormState, value: string, schema?: SchemaResponse | null): string | null {
  const r = RANGES[name];
  if (!r) return null;
  const required = name === "building_size";
  if (value.trim() === "") return required ? `${r.label} is required.` : null;
  const n = Number(value);
  if (!Number.isFinite(n)) return `${r.label} must be a number.`;
  if (r.integer && !Number.isInteger(n)) return `${r.label} must be a whole number.`;
  if (n < r.min || n > r.max) return `${r.label} must be between ${r.min} and ${r.max}.`;
  void schema;
  return null;
}

export function validateForm(f: FormState, schema?: SchemaResponse | null): FieldErrors {
  const errs: FieldErrors = {};
  (Object.keys(RANGES) as (keyof FormState)[]).forEach((k) => {
    const e = validateField(k, String(f[k]), schema);
    if (e) errs[k] = e;
  });
  return errs;
}

const toNum = (s: string): number | null => (s.trim() === "" ? null : Number(s));

export function toBase(f: FormState): BuildingBase {
  return {
    location_type: f.location_type, building_size: Number(f.building_size), roofing_type: f.roofing_type,
    building_age: toNum(f.building_age), avg_roof_temperature_before: toNum(f.avg_roof_temperature_before),
    num_ac_units: toNum(f.num_ac_units), avg_units_before: toNum(f.avg_units_before),
  };
}
export const toInput = (f: FormState): BuildingInput => ({
  ...toBase(f), cool_roofing_method: f.cool_roofing_method, cool_roofing_quality: f.cool_roofing_quality,
});

/* ------------------------------------------------------------------ context */
export interface PredictionSnapshot { input: BuildingInput; response: PredictResponse; at: number }

interface Ctx {
  status: AsyncState<StatusResponse>;
  schema: AsyncState<SchemaResponse>;
  form: FormState;
  setForm: (patch: Partial<FormState>) => void;
  resetForm: (f?: FormState) => void;
  prediction: PredictionSnapshot | null;
  setPrediction: (p: PredictionSnapshot | null) => void;
  theme: "system" | "light" | "dark";
  setTheme: (t: "system" | "light" | "dark") => void;
}
const AppCtx = createContext<Ctx | null>(null);
const FORM_KEY = "coolroof.form.v1";

function loadForm(): FormState {
  const raw = safeGet(FORM_KEY);
  if (!raw) return EMPTY_FORM;
  try { return { ...EMPTY_FORM, ...JSON.parse(raw) }; } catch { return EMPTY_FORM; }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const status = useAsync(() => api.status(), []);
  const schema = useAsync(() => api.schema(), []);
  const [form, setFormState] = useState<FormState>(loadForm);
  const [prediction, setPrediction] = useState<PredictionSnapshot | null>(null);
  const [theme, setThemeState] = useState<"system" | "light" | "dark">(() => (safeGet("coolroof.theme", "local") as "system" | "light" | "dark") ?? "system");

  useEffect(() => { safeSet(FORM_KEY, JSON.stringify(form)); }, [form]);
  useEffect(() => {
    const el = document.documentElement;
    if (theme === "system") el.removeAttribute("data-theme"); else el.setAttribute("data-theme", theme);
  }, [theme]);

  const setForm = useCallback((patch: Partial<FormState>) => setFormState((f) => ({ ...f, ...patch })), []);
  const resetForm = useCallback((f: FormState = EMPTY_FORM) => { setFormState(f); setPrediction(null); }, []);
  const setTheme = useCallback((t: "system" | "light" | "dark") => { setThemeState(t); safeSet("coolroof.theme", t, "local"); }, []);

  const value = useMemo<Ctx>(() => ({ status, schema, form, setForm, resetForm, prediction, setPrediction, theme, setTheme }),
    [status, schema, form, setForm, resetForm, prediction, theme, setTheme]);
  return <AppCtx.Provider value={value}>{children}</AppCtx.Provider>;
}

export function useApp(): Ctx {
  const c = useContext(AppCtx);
  if (!c) throw new Error("useApp must be used inside <AppProvider>");
  return c;
}

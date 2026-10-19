import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowDownRight, ArrowUpRight, Minus, Play, RotateCcw, SlidersHorizontal } from "lucide-react";
import { api } from "../lib/api";
import { inr, inrCompact, num } from "../lib/format";
import {
  FEATURE_LABEL, LOCATION_LABEL, LOCATION_ORDER, METHOD_LABEL, QUALITY_LABEL, QUALITY_ORDER, ROOF_LABEL, ROOF_ORDER, supportedMethods,
} from "../lib/labels";
import type { CompareResponse, PredictResponse, Quality, RoofType } from "../lib/types";
import { toBase, toInput, useApp, validateField, validateForm } from "../state/AppState";
import type { FieldErrors, FormState } from "../state/AppState";
import { Alert, Badge, Button, Card, EmptyState, ErrorState, LoadingBlock, NumberField, PageHeader, Segmented, SelectField, Tabs, useDocumentTitle } from "../components/ui";
import { CategoryBars, ChartCard } from "../components/charts";

type MetricKey = "cost" | "temperature_drop" | "units_saved";
interface RunResult { base: PredictResponse; scen: PredictResponse; cmp: CompareResponse; baseForm: FormState; scenForm: FormState }

const METRICS: { key: MetricKey; label: string; short: string; good: "down" | "up"; fmt: (v: number) => string; compact: (v: number) => string }[] = [
  { key: "cost", label: "Installation cost", short: "Cost", good: "down", fmt: inr, compact: inrCompact },
  { key: "temperature_drop", label: "Roof temperature reduction", short: "Cooling", good: "up", fmt: (v) => `${num(v, 1)} °C`, compact: (v) => `${num(v, 1)} °C` },
  { key: "units_saved", label: "Electricity saved per month", short: "Savings", good: "up", fmt: (v) => `${num(v, 0)} kWh`, compact: (v) => `${num(v, 0)} kWh` },
];
const FIELDS: (keyof FormState)[] = ["location_type", "building_size", "roofing_type", "building_age", "avg_roof_temperature_before", "num_ac_units", "avg_units_before", "cool_roofing_method", "cool_roofing_quality"];
const optName = (m: string, q: string) => `${METHOD_LABEL[m as keyof typeof METHOD_LABEL]} · ${QUALITY_LABEL[q as Quality]}`;

function describe(k: keyof FormState, v: string): string {
  if (v === "") return "not given";
  if (k === "location_type") return LOCATION_LABEL[v as keyof typeof LOCATION_LABEL] ?? v;
  if (k === "roofing_type") return ROOF_LABEL[v as RoofType] ?? v;
  if (k === "cool_roofing_method") return METHOD_LABEL[v as keyof typeof METHOD_LABEL] ?? v;
  if (k === "cool_roofing_quality") return QUALITY_LABEL[v as Quality] ?? v;
  return v;
}

function Delta({ metric, base, scen, id }: { metric: (typeof METRICS)[number]; base: number; scen: number; id: string }) {
  const d = scen - base;
  const pctChange = Math.abs(base) >= 1 ? (d / Math.abs(base)) * 100 : null;
  const same = Math.abs(d) < 1e-9 || (pctChange !== null && Math.abs(pctChange) < 0.5);
  const better = metric.good === "down" ? d < 0 : d > 0;
  const tone = same ? "neutral" : better ? "good" : "bad";
  const Icon = same ? Minus : d > 0 ? ArrowUpRight : ArrowDownRight;
  const text = same ? "no meaningful change" : `${d > 0 ? "+" : "−"}${metric.compact(Math.abs(d))}${pctChange !== null ? ` (${d > 0 ? "+" : "−"}${Math.abs(pctChange).toFixed(0)}%)` : ""}`;
  return (
    <Badge tone={tone}>
      <Icon size={13} aria-hidden /> <span data-testid={id}>{text}</span>
    </Badge>
  );
}

export default function Simulator() {
  useDocumentTitle("What-if simulator");
  const { form, schema } = useApp();
  const baselineValid = Object.keys(validateForm(form, schema.data)).length === 0;
  const [scen, setScen] = useState<FormState>(form);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [metric, setMetric] = useState<MetricKey>("cost");

  const set = (patch: Partial<FormState>) => { setScen((s) => ({ ...s, ...patch })); };
  const methods = useMemo(() => supportedMethods(scen.roofing_type, schema.data?.method_roof_support, schema.data?.min_pair_support), [scen.roofing_type, schema.data]);
  useEffect(() => { if (!methods.includes(scen.cool_roofing_method)) set({ cool_roofing_method: methods[0] }); }, [methods, scen.cool_roofing_method]);

  const changed = FIELDS.filter((k) => String(scen[k]) !== String(form[k]));
  const blur = (k: keyof FormState) => setErrors((e) => ({ ...e, [k]: validateField(k, String(scen[k]), schema.data) ?? undefined }));
  const numField = (k: keyof FormState, label: string, extra: Partial<Parameters<typeof NumberField>[0]> = {}) => (
    <NumberField id={`sim-${k}`} label={label} value={String(scen[k])} onChange={(v) => set({ [k]: v } as Partial<FormState>)} onBlur={() => blur(k)} error={errors[k] ?? null} {...extra} />
  );

  async function run() {
    const errs = validateForm(scen, schema.data);
    setErrors(errs);
    if (Object.keys(errs).length) return;
    setBusy(true); setError(null);
    try {
      const [base, sc, cmp] = await Promise.all([
        api.predict(toInput(form)), api.predict(toInput(scen)),
        api.compare(toBase(scen), "balanced", null, { method: scen.cool_roofing_method, quality: scen.cool_roofing_quality }),
      ]);
      setResult({ base, scen: sc, cmp, baseForm: { ...form }, scenForm: { ...scen } });
    } catch (e) {
      setError(e); setResult(null);
    } finally {
      setBusy(false);
    }
  }

  if (!baselineValid) {
    return (
      <>
        <PageHeader title="What-if simulator" subtitle="Change a building input or the roofing choice and see how the model's estimates move." />
        <EmptyState icon={<SlidersHorizontal size={34} />} title="Start from a baseline customer">
          The simulator compares a scenario against the building on <Link to="/predict" className="font-semibold text-primary underline">Customer prediction</Link>. Enter those details there first (or load the example), then come back.
        </EmptyState>
      </>
    );
  }

  const sameAsBase = result && JSON.stringify(result.baseForm) === JSON.stringify(result.scenForm);
  const chartRows = result ? [...result.cmp.options]
    .sort((a, b) => (metric === "cost" ? a.predictions.cost - b.predictions.cost : b.predictions[metric] - a.predictions[metric])) : [];
  const m = METRICS.find((x) => x.key === metric)!;

  return (
    <>
      <PageHeader title="What-if simulator"
        subtitle="Change inputs or the roofing choice, then run the simulation. Every number comes from the trained models, not from a formula on this page."
        actions={<Button variant="ghost" onClick={() => { setScen(form); setErrors({}); setResult(null); setError(null); }}><RotateCcw size={16} aria-hidden /> Reset to baseline</Button>} />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <form onSubmit={(e) => { e.preventDefault(); void run(); }} noValidate aria-label="Scenario inputs" className="space-y-5">
          <Card className="space-y-4 p-4 sm:p-5">
            <div>
              <h2 className="text-lg font-semibold">Scenario</h2>
              <p className="text-sm text-muted">Starts from the customer on the prediction page. Change anything; the baseline stays fixed.</p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <SelectField id="sim-location_type" label="Location type" value={scen.location_type} onChange={(v) => set({ location_type: v as FormState["location_type"] })}
                options={LOCATION_ORDER.map((l) => ({ value: l, label: LOCATION_LABEL[l] }))} />
              {numField("building_size", "Building size", { unit: "area", min: 50, max: 50000 })}
              <SelectField id="sim-roofing_type" label="Existing roof" value={scen.roofing_type} onChange={(v) => set({ roofing_type: v as RoofType })}
                options={ROOF_ORDER.map((r) => ({ value: r, label: ROOF_LABEL[r] }))} />
              {numField("building_age", "Building age", { unit: "years", optional: true, min: 0, max: 100 })}
              {numField("avg_roof_temperature_before", "Roof temperature", { unit: "°C", optional: true, min: 10, max: 80 })}
              {numField("num_ac_units", "AC units", { optional: true, step: "1", min: 0, max: 10 })}
              {numField("avg_units_before", "Electricity use", { unit: "kWh/mo", optional: true, min: 0 })}
            </div>
            <Segmented legend="Roofing method" name="sim-method" value={scen.cool_roofing_method} onChange={(v) => set({ cool_roofing_method: v })}
              options={methods.map((x) => ({ value: x, label: METHOD_LABEL[x] }))} hint={`Only methods supported for a ${ROOF_LABEL[scen.roofing_type].toLowerCase()} roof.`} />
            <Segmented legend="Quality grade" name="sim-quality" value={scen.cool_roofing_quality} onChange={(v: Quality) => set({ cool_roofing_quality: v })}
              options={QUALITY_ORDER.map((q) => ({ value: q, label: QUALITY_LABEL[q] }))} />
            <div className="flex flex-wrap items-center gap-3">
              <Button type="submit" loading={busy}><Play size={16} aria-hidden /> Run simulation</Button>
              <span className="text-sm text-muted" aria-live="polite">{changed.length === 0 ? "No changes from the baseline yet" : `${changed.length} change${changed.length > 1 ? "s" : ""} from the baseline`}</span>
            </div>
          </Card>
        </form>

        <div className="min-w-0 space-y-5" aria-live="polite">
          {busy && <LoadingBlock label="Running the simulation…" rows={3} />}
          {!busy && !!error && <ErrorState error={error} onRetry={() => void run()} />}
          {!busy && !error && !result && (
            <EmptyState icon={<SlidersHorizontal size={34} />} title="No simulation yet">
              Adjust the scenario and choose <strong>Run simulation</strong>. Results are the models' predictions for both the baseline and your scenario, side by side.
            </EmptyState>
          )}
          {!busy && !error && result && (
            <>
              {sameAsBase && <Alert tone="neutral">The scenario is identical to the baseline, so both columns match. Change an input to see an effect.</Alert>}
              {result.scen.warnings.map((w) => <Alert key={w} tone="warn" title="Reliability note">{w}</Alert>)}
              {!sameAsBase && (
                <p className="text-sm text-muted">
                  Changed: {FIELDS.filter((k) => String(result.scenForm[k]) !== String(result.baseForm[k])).map((k) =>
                    `${FEATURE_LABEL[k]} ${describe(k, String(result.baseForm[k]))} → ${describe(k, String(result.scenForm[k]))}`).join("; ")}
                </p>
              )}
              <div className="grid gap-4 md:grid-cols-3 xl:grid-cols-1 2xl:grid-cols-3">
                {METRICS.map((mt) => {
                  const b = result.base.predictions[mt.key].value, sc = result.scen.predictions[mt.key];
                  return (
                    <Card key={mt.key} className="p-4">
                      <h3 className="text-sm font-semibold">{mt.label}</h3>
                      <div className="mt-2 grid grid-cols-2 gap-3">
                        <div><p className="text-xs uppercase tracking-wide text-muted">Baseline</p><p className="num text-xl font-semibold text-muted" data-testid={`sim-base-${mt.key}`}>{mt.fmt(b)}</p></div>
                        <div><p className="text-xs uppercase tracking-wide text-primary">Scenario</p><p className="num text-xl font-semibold" data-testid={`sim-scen-${mt.key}`}>{mt.fmt(sc.value)}</p></div>
                      </div>
                      <div className="mt-2"><Delta metric={mt} base={b} scen={sc.value} id={`sim-delta-${mt.key}`} /></div>
                      <p className="mt-2 text-xs text-muted">{Math.round(sc.interval.level * 100)}% likely range: {mt.compact(sc.interval.low)} – {mt.compact(sc.interval.high)}</p>
                    </Card>
                  );
                })}
              </div>

              <Tabs label="Metric to compare across options" value={metric} onChange={setMetric} tabs={METRICS.map((x) => ({ value: x.key, label: x.short }))} />
              <div id={`panel-${metric}`} role="tabpanel" aria-labelledby={`tab-${metric}`}>
              <ChartCard title="Roofing options for this scenario's building" subtitle={`Model predictions for every supported method × quality. Gold = your scenario's choice.`}
                summary={`Bar chart of predicted ${m.label.toLowerCase()} for ${chartRows.length} roofing options.`} height={Math.max(260, chartRows.length * 30)}
                table={{ headers: ["Option", "Cost (₹)", "Cooling (°C)", "kWh/month"], rows: chartRows.map((o) => [optName(o.cool_roofing_method, o.cool_roofing_quality), Math.round(o.predictions.cost), num(o.predictions.temperature_drop, 2), num(o.predictions.units_saved, 0)]) }}>
                <CategoryBars layout="horizontal" fmt={m.compact} name={m.label}
                  data={chartRows.map((o) => ({ name: optName(o.cool_roofing_method, o.cool_roofing_quality), value: o.predictions[metric],
                    color: o.cool_roofing_method === result.scenForm.cool_roofing_method && o.cool_roofing_quality === result.scenForm.cool_roofing_quality ? "var(--c2)" : "var(--c4)" }))} />
              </ChartCard>
              </div>
              <Alert tone="neutral" title="How to read this">
                Estimates come from the same models as the rest of the app (model v{result.scen.model_version}). They describe how similar past buildings turned out, so a change in a scenario shows how the estimate moves, not a guaranteed result.
              </Alert>
            </>
          )}
        </div>
      </div>
    </>
  );
}

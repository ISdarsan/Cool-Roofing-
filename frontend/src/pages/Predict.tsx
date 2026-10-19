import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Calculator, FileText, RotateCcw, Scale, Wand2 } from "lucide-react";
import { api } from "../lib/api";
import { LOCATION_LABEL, LOCATION_ORDER, METHOD_HINT, METHOD_LABEL, QUALITY_LABEL, QUALITY_ORDER, ROOF_LABEL, ROOF_ORDER, supportedMethods } from "../lib/labels";
import { EXAMPLE_FORM, toInput, useApp, validateField, validateForm } from "../state/AppState";
import type { FieldErrors, FormState } from "../state/AppState";
import { Alert, Button, Card, EmptyState, ErrorState, LoadingBlock, NumberField, PageHeader, Segmented, SelectField, useDocumentTitle } from "../components/ui";
import { ExplainPanel, ResultsGrid, ThresholdPanel } from "../components/PredictionResults";
import PaybackPanel from "../components/PaybackPanel";
import type { Method, Quality, RoofType } from "../lib/types";

export default function Predict() {
  useDocumentTitle("Customer prediction");
  const { form, setForm, resetForm, schema, prediction, setPrediction, status } = useApp();
  const [errors, setErrors] = useState<FieldErrors>({});
  const [serverErrors, setServerErrors] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [refCut, setRefCut] = useState<number | null>(null);
  const [highCut, setHighCut] = useState(0.5);
  const summaryRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const methods = useMemo<Method[]>(() => supportedMethods(form.roofing_type, schema.data?.method_roof_support, schema.data?.min_pair_support), [form.roofing_type, schema.data]);

  // keep the proposed method valid when the roof type changes
  useEffect(() => {
    if (!methods.includes(form.cool_roofing_method)) setForm({ cool_roofing_method: methods[0] });
  }, [methods, form.cool_roofing_method, setForm]);

  const defaultRef = prediction?.response.predictions.referral.cutoff ?? 0.25;
  useEffect(() => { setRefCut(null); }, [prediction]);
  const refCutoff = refCut ?? defaultRef;

  const blur = (k: keyof FormState) => setErrors((e) => ({ ...e, [k]: validateField(k, String(form[k]), schema.data) ?? undefined }));
  const num = (k: keyof FormState, label: string, extra: Partial<Parameters<typeof NumberField>[0]> = {}) => (
    <NumberField id={k} label={label} value={String(form[k])} onChange={(v) => setForm({ [k]: v } as Partial<FormState>)} onBlur={() => blur(k)} error={errors[k] ?? null} {...extra} />
  );

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const errs = validateForm(form, schema.data);
    setErrors(errs);
    setServerErrors([]);
    if (Object.keys(errs).length) {
      requestAnimationFrame(() => summaryRef.current?.focus());
      return;
    }
    setBusy(true); setError(null);
    try {
      const input = toInput(form);
      const response = await api.predict(input, { high_rating_cutoff: 0.5 });
      setPrediction({ input, response, at: Date.now() });
      requestAnimationFrame(() => resultsRef.current?.scrollIntoView?.({ behavior: "smooth", block: "start" }));
    } catch (err) {
      setError(err);
      const fe = (err as { fieldErrors?: { field: string; message: string }[] }).fieldErrors ?? [];
      setServerErrors(fe.map((x) => `${x.field}: ${x.message}`));
    } finally {
      setBusy(false);
    }
  }

  const errList = Object.entries(errors).filter(([, v]) => v) as [string, string][];
  const modelsOffline = !!status.error;

  return (
    <>
      <PageHeader title="Customer prediction"
        subtitle="Answer a few day-one questions about the building. Every number comes from the trained models, with a likely range, a reliability flag and a plain-language “why”."
        actions={<><Button variant="secondary" onClick={() => { resetForm(EXAMPLE_FORM); setErrors({}); }}><Wand2 size={16} aria-hidden /> Load example</Button>
          <Button variant="ghost" onClick={() => { resetForm(); setErrors({}); }}><RotateCcw size={16} aria-hidden /> Reset</Button></>} />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <form onSubmit={submit} noValidate aria-label="Building and installation details" className="space-y-5">
          {(errList.length > 0 || serverErrors.length > 0) && (
            <div ref={summaryRef} tabIndex={-1} role="alert" aria-labelledby="err-title" className="rounded-xl border border-clay/50 bg-clay-soft p-3 text-sm">
              <h2 id="err-title" className="font-semibold text-clay">Please fix {errList.length + serverErrors.length} thing{errList.length + serverErrors.length > 1 ? "s" : ""}</h2>
              <ul className="mt-1 list-disc pl-5">
                {errList.map(([k, m]) => <li key={k}><a className="underline" href={`#${k}`}>{m}</a></li>)}
                {serverErrors.map((m) => <li key={m}>{m}</li>)}
              </ul>
            </div>
          )}

          <Card className="space-y-4 p-4 sm:p-5">
            <h2 className="text-lg font-semibold">1 · The building</h2>
            <div className="grid gap-4 sm:grid-cols-2">
              <SelectField id="location_type" label="Location type" value={form.location_type} onChange={(v) => setForm({ location_type: v as FormState["location_type"] })}
                options={LOCATION_ORDER.map((l) => ({ value: l, label: LOCATION_LABEL[l] }))} />
              {num("building_size", "Building size", { unit: "area", hint: "Floor area (the data does not state the unit)", min: 50, max: 50000 })}
              <SelectField id="roofing_type" label="Existing roof" value={form.roofing_type} onChange={(v) => setForm({ roofing_type: v as RoofType })}
                options={ROOF_ORDER.map((r) => ({ value: r, label: ROOF_LABEL[r] }))} hint="Decides which methods can be installed" />
              {num("building_age", "Building age", { unit: "years", optional: true, min: 0, max: 100 })}
            </div>
          </Card>

          <Card className="space-y-4 p-4 sm:p-5">
            <h2 className="text-lg font-semibold">2 · Proposed roofing</h2>
            <Segmented legend="Method" name="method" value={form.cool_roofing_method} onChange={(v) => setForm({ cool_roofing_method: v })}
              options={methods.map((m) => ({ value: m, label: METHOD_LABEL[m], sub: METHOD_HINT[m] }))}
              hint={`Only methods the history supports for a ${ROOF_LABEL[form.roofing_type].toLowerCase()} roof are shown.`} />
            <Segmented legend="Quality grade" name="quality" value={form.cool_roofing_quality} onChange={(v: Quality) => setForm({ cool_roofing_quality: v })}
              options={QUALITY_ORDER.map((q) => ({ value: q, label: QUALITY_LABEL[q] }))} />
          </Card>

          <Card className="space-y-4 p-4 sm:p-5">
            <div>
              <h2 className="text-lg font-semibold">3 · Today’s performance</h2>
              <p className="text-sm text-muted">Optional, but the savings estimate is much sharper with current usage. Blanks are filled with typical values and flagged.</p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              {num("avg_roof_temperature_before", "Roof temperature", { unit: "°C", optional: true, min: 10, max: 80, hint: "Average surface temp" })}
              {num("num_ac_units", "AC units", { optional: true, step: "1", min: 0, max: 10 })}
              {num("avg_units_before", "Electricity use", { unit: "kWh/mo", optional: true, min: 0, hint: "Average monthly bill units" })}
            </div>
          </Card>

          <Button type="submit" loading={busy} disabled={modelsOffline} className="w-full sm:w-auto"><Calculator size={17} aria-hidden /> Predict outcomes</Button>
          {modelsOffline && <p className="text-sm text-clay">Models are offline, so predictions are disabled rather than guessed.</p>}
        </form>

        <div ref={resultsRef} className="min-w-0 space-y-5 scroll-mt-4" aria-live="polite">
          {busy && <LoadingBlock label="Running the models…" rows={4} />}
          {!busy && !!error && <ErrorState error={error} onRetry={() => document.querySelector<HTMLFormElement>("form")?.requestSubmit()} />}
          {!busy && !error && !prediction && (
            <EmptyState icon={<Calculator size={34} />} title="No prediction yet">
              Fill in the building details and choose <strong>Predict outcomes</strong>. You will see cost, cooling, electricity savings, rating and referral estimates here — or try <strong>Load example</strong>.
            </EmptyState>
          )}
          {!busy && !error && prediction && (
            <>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-sm text-muted">{METHOD_LABEL[prediction.input.cool_roofing_method]} · {QUALITY_LABEL[prediction.input.cool_roofing_quality]} quality · model v{prediction.response.model_version}</p>
                <div className="flex gap-2">
                  <Link to="/compare" className="inline-flex min-h-11 items-center gap-1.5 rounded-xl border border-line px-3 text-sm font-semibold text-primary hover:bg-surface2"><Scale size={16} aria-hidden /> Compare options</Link>
                  <Link to="/proposal" className="inline-flex min-h-11 items-center gap-1.5 rounded-xl bg-primary px-3 text-sm font-semibold text-on-primary hover:bg-primary-hover"><FileText size={16} aria-hidden /> Create proposal <ArrowRight size={15} aria-hidden /></Link>
                </div>
              </div>
              {prediction.response.warnings.map((w) => <Alert key={w} tone="warn" title="Reliability note">{w}</Alert>)}
              <ResultsGrid response={prediction.response} input={prediction.input} referralCutoff={refCutoff} highCutoff={highCut} />
              <details className="rounded-card border border-line bg-surface p-4">
                <summary className="cursor-pointer font-semibold">Adjust decision cut-offs <span className="text-sm font-normal text-muted">(how sure before we flag a customer)</span></summary>
                <div className="mt-3 grid gap-3 md:grid-cols-2">
                  <ThresholdPanel task="high_rating" label="High-rating" value={highCut} onChange={setHighCut} defaultValue={0.5} />
                  <ThresholdPanel task="referral" label="Referral" value={refCutoff} onChange={setRefCut} defaultValue={defaultRef} />
                </div>
              </details>
              <ExplainPanel input={prediction.input} />
              <PaybackPanel cost={prediction.response.predictions.cost.value} kwh={prediction.response.predictions.units_saved.value}
                kwhLow={prediction.response.predictions.units_saved.interval.low} kwhHigh={prediction.response.predictions.units_saved.interval.high} />
              <details className="rounded-card border border-line bg-surface p-4 text-sm">
                <summary className="cursor-pointer font-semibold">Assumptions behind these numbers</summary>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-muted">{prediction.response.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>
              </details>
            </>
          )}
        </div>
      </div>
    </>
  );
}

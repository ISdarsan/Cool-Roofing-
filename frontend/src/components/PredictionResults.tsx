import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { ResponsiveContainer } from "recharts";
import { Flame, HelpCircle, Sparkles, ThermometerSnowflake, ThumbsUp, Wallet, Zap } from "lucide-react";
import { api } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { inr, inrCompact, num, pct, signed, titleCase } from "../lib/format";
import { FEATURE_LABEL, LOCATION_LABEL, METHOD_LABEL, QUALITY_LABEL, ROOF_LABEL, TASK_LABEL, TASK_ORDER } from "../lib/labels";
import type { BuildingInput, ExplainResponse, Predictions, PredictResponse, TaskKey } from "../lib/types";
import { Alert, Badge, Button, Card, ErrorState, LoadingBlock, ReliabilityBadge, Tabs } from "./ui";
import { ContributionBars, IntervalBar, ProbabilityMeter } from "./charts";

/* ------------------------------------------------------------------ stars */
export function Stars({ value, max = 5 }: { value: number; max?: number }) {
  return (
    <span className="inline-flex gap-0.5" role="img" aria-label={`${value.toFixed(1)} out of ${max} stars`}>
      {Array.from({ length: max }).map((_, i) => {
        const fill = Math.max(0, Math.min(1, value - i));
        return (
          <svg key={i} width="22" height="22" viewBox="0 0 24 24" aria-hidden>
            <defs><clipPath id={`s${i}-${Math.round(value * 10)}`}><rect x="0" y="0" width={24 * fill} height="24" /></clipPath></defs>
            <path d="m12 2.5 2.9 6.1 6.6.8-4.9 4.6 1.3 6.6L12 17.3 6.1 20.6l1.3-6.6L2.5 9.4l6.6-.8z" fill="none" stroke="var(--sun)" strokeWidth="1.6" strokeLinejoin="round" />
            <path d="m12 2.5 2.9 6.1 6.6.8-4.9 4.6 1.3 6.6L12 17.3 6.1 20.6l1.3-6.6L2.5 9.4l6.6-.8z" fill="var(--sun)" clipPath={`url(#s${i}-${Math.round(value * 10)})`} />
          </svg>
        );
      })}
    </span>
  );
}

/* ------------------------------------------------------------------ one result card */
function ResultCard({ icon, title, kind, children, badge, foot }: {
  icon: ReactNode; title: string; kind: "Estimate" | "Probability" | "Expected score"; children: ReactNode; badge?: ReactNode; foot?: ReactNode;
}) {
  return (
    <Card className="flex flex-col p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-primary-soft text-primary" aria-hidden>{icon}</span>
          <div>
            <h3 className="text-sm font-semibold leading-tight">{title}</h3>
            <p className="text-xs text-muted">{kind}</p>
          </div>
        </div>
        {badge}
      </div>
      <div className="mt-3 flex-1">{children}</div>
      {foot && <div className="mt-3 border-t border-line pt-2 text-xs text-muted">{foot}</div>}
    </Card>
  );
}

export function ResultsGrid({ response, input, referralCutoff, highCutoff }: {
  response: PredictResponse; input: BuildingInput; referralCutoff: number; highCutoff: number;
}) {
  const p: Predictions = response.predictions;
  const lv = Math.round(p.cost.interval.level * 100);
  const afterT = response.derived.avg_roof_temperature_after;
  const afterU = response.derived.avg_units_after;
  const highLikely = p.high_rating.probability >= highCutoff;
  const refLikely = p.referral.probability >= referralCutoff;
  return (
    <div className="grid gap-4 sm:grid-cols-2" aria-label="Prediction results">
      <ResultCard icon={<Wallet size={18} />} title="Installation cost" kind="Estimate" badge={<ReliabilityBadge r={p.cost.reliability} />}
        foot={<>{lv}% likely range comes from the model’s past errors on unseen buildings.</>}>
        <p className="num text-3xl font-semibold" data-testid="cost-value">{inr(p.cost.value)}</p>
        <p className="text-sm text-muted">{lv}% likely range {inrCompact(p.cost.interval.low)} – {inrCompact(p.cost.interval.high)}</p>
        <IntervalBar low={p.cost.interval.low} value={p.cost.value} high={p.cost.interval.high} fmt={inrCompact} label="Installation cost" />
      </ResultCard>

      <ResultCard icon={<ThermometerSnowflake size={18} />} title="Roof temperature drop" kind="Estimate" badge={<ReliabilityBadge r={p.temperature_drop.reliability} />}
        foot={afterT ? <>Roof would go from {num(input.avg_roof_temperature_before ?? 0)} °C to ≈ {num(afterT.value)} °C (derived).</> : <>Add the current roof temperature to see the expected “after” value.</>}>
        <p className="num text-3xl font-semibold" data-testid="temp-value">−{num(p.temperature_drop.value, 1)} °C</p>
        <p className="text-sm text-muted">{lv}% likely range {num(p.temperature_drop.interval.low)} – {num(p.temperature_drop.interval.high)} °C</p>
        <IntervalBar low={p.temperature_drop.interval.low} value={p.temperature_drop.value} high={p.temperature_drop.interval.high} fmt={(v) => `${num(v)}°`} label="Temperature drop" />
      </ResultCard>

      <ResultCard icon={<Zap size={18} />} title="Electricity saved per month" kind="Estimate" badge={<ReliabilityBadge r={p.units_saved.reliability} />}
        foot={afterU ? <>Monthly use would go from {num(input.avg_units_before ?? 0, 0)} to ≈ {num(afterU.value, 0)} kWh (derived).</> : <>Add current monthly kWh to see the expected “after” value.</>}>
        <p className="num text-3xl font-semibold" data-testid="units-value">{signed(p.units_saved.value, 0)} kWh</p>
        <p className="text-sm text-muted">{lv}% likely range {num(p.units_saved.interval.low, 0)} – {num(p.units_saved.interval.high, 0)} kWh/month</p>
        <IntervalBar low={p.units_saved.interval.low} value={p.units_saved.value} high={p.units_saved.interval.high} fmt={(v) => num(v, 0)} label="Electricity saved" />
        {p.units_saved.value < 5 && <p className="mt-2 text-xs text-warn-ink">Low-usage buildings may see little or no saving; a few even use more.</p>}
      </ResultCard>

      <ResultCard icon={<Sparkles size={18} />} title="Customer rating" kind="Expected score" badge={<ReliabilityBadge r={p.rating.reliability} />}
        foot={<>{p.rating.note} It is not the probability below.</>}>
        <p className="num text-3xl font-semibold" data-testid="rating-value">{num(p.rating.value, 1)} <span className="text-lg text-muted">/ 5</span></p>
        <div className="mt-1"><Stars value={p.rating.value} /></div>
      </ResultCard>

      <ResultCard icon={<ThumbsUp size={18} />} title="Chance of a high rating" kind="Probability" badge={<ReliabilityBadge r={p.high_rating.reliability} />}
        foot={<>“High” = rating ≥ {response.predictions.high_rating.definition.match(/\d/)?.[0] ?? 4} (assumed; organisers gave no definition).</>}>
        <p className="num text-3xl font-semibold" data-testid="high-value">{pct(p.high_rating.probability)}</p>
        <p className="text-sm">{highLikely ? <Badge tone="good">Likely to rate {response.predictions.high_rating.definition.match(/\d/)?.[0] ?? 4}★ or more</Badge> : <Badge tone="neutral">Below the {pct(highCutoff)} cut-off</Badge>}</p>
        <ProbabilityMeter value={p.high_rating.probability} cutoff={highCutoff} label="Probability of a high rating" />
      </ResultCard>

      <ResultCard icon={<Flame size={18} />} title="Chance of a referral" kind="Probability" badge={<ReliabilityBadge r={p.referral.reliability} />}
        foot={<>Weakly predictable from quote-time facts: use to prioritise follow-ups, not to promise.</>}>
        <p className="num text-3xl font-semibold" data-testid="referral-value">{pct(p.referral.probability)}</p>
        <p className="text-sm">{refLikely ? <Badge tone="good">Flagged as likely referrer</Badge> : <Badge tone="neutral">Below the {pct(referralCutoff)} cut-off</Badge>}</p>
        <ProbabilityMeter value={p.referral.probability} cutoff={referralCutoff} label="Probability of a referral" />
      </ResultCard>
    </div>
  );
}

/* ------------------------------------------------------------------ cut-off trade-off */
export function ThresholdPanel({ task, label, value, onChange, defaultValue }: {
  task: "referral" | "high_rating"; label: string; value: number; onChange: (v: number) => void; defaultValue: number;
}) {
  const models = useAsync(() => api.models(), []);
  const op = useMemo(() => {
    const rows = models.data?.tasks?.[task]?.operating_points?.test as { threshold: number; accuracy: number; f1: number; precision: number; recall: number }[] | undefined;
    if (!rows?.length) return null;
    return rows.reduce((b, r) => (Math.abs(r.threshold - value) < Math.abs(b.threshold - value) ? r : b), rows[0]);
  }, [models.data, task, value]);
  const id = `cut-${task}`;
  return (
    <div className="rounded-xl border border-line p-3">
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={id} className="text-sm font-semibold">{label} cut-off</label>
        <span className="num text-sm">{pct(value)}</span>
      </div>
      <input id={id} type="range" min={0.1} max={0.8} step={0.01} value={value} onChange={(e) => onChange(Number(e.target.value))} className="mt-1 w-full" />
      <div className="mt-1 flex justify-between text-xs text-muted">
        <span>catch more (lower)</span>
        <button type="button" className="text-primary underline-offset-2 hover:underline" onClick={() => onChange(defaultValue)}>reset to {pct(defaultValue)}</button>
        <span>be stricter (higher)</span>
      </div>
      {op ? (
        <dl className="mt-2 grid grid-cols-4 gap-2 text-center text-xs" aria-label={`Hold-out performance at about ${pct(op.threshold)} cut-off`}>
          {([["Accuracy", op.accuracy], ["F1", op.f1], ["Precision", op.precision], ["Recall", op.recall]] as const).map(([k, v]) => (
            <div key={k} className="rounded-lg bg-surface2 p-1.5"><dt className="text-muted">{k}</dt><dd className="num text-sm font-semibold">{v.toFixed(2)}</dd></div>
          ))}
        </dl>
      ) : <p className="mt-2 text-xs text-muted">{models.loading ? "Loading hold-out trade-offs…" : "Trade-off table unavailable."}</p>}
      <p className="mt-1 text-xs text-muted">Measured on held-out buildings at the nearest 0.05 cut-off. Lowering it flags more customers but with more false alarms.</p>
    </div>
  );
}

/* ------------------------------------------------------------------ explanations */
function fmtContribution(task: TaskKey) {
  return (v: number) => {
    if (v === 0) return "0";
    switch (task) {
      case "cost": return `${v >= 0 ? "+" : "−"}${inrCompact(Math.abs(v))}`;
      case "temperature_drop": return `${signed(v, 2)} °C`;
      case "units_saved": return `${signed(v, 1)} kWh`;
      case "rating": return `${signed(v, 2)} ★`;
      default: return `${signed(v * 100, 1)} pp`;
    }
  };
}
function fmtTotal(task: TaskKey, v: number) {
  switch (task) {
    case "cost": return inr(v);
    case "temperature_drop": return `${num(v, 2)} °C`;
    case "units_saved": return `${num(v, 1)} kWh`;
    case "rating": return `${num(v, 2)} ★`;
    default: return pct(v, 1);
  }
}
function featureValue(f: string, v: string | number | null): string {
  if (v === null || v === undefined) return "not provided (typical value used)";
  if (f === "location_type") return LOCATION_LABEL[v as keyof typeof LOCATION_LABEL] ?? String(v);
  if (f === "roofing_type") return ROOF_LABEL[v as keyof typeof ROOF_LABEL] ?? String(v);
  if (f === "cool_roofing_method") return METHOD_LABEL[v as keyof typeof METHOD_LABEL] ?? String(v);
  if (f === "cool_roofing_quality") return QUALITY_LABEL[v as keyof typeof QUALITY_LABEL] ?? String(v);
  return String(v);
}

export function ExplainPanel({ input }: { input: BuildingInput }) {
  const [task, setTask] = useState<TaskKey>("cost");
  const [asked, setAsked] = useState(false);
  const [state, setState] = useState<{ loading: boolean; error: unknown; data: ExplainResponse | null }>({ loading: false, error: null, data: null });

  useEffect(() => { setState({ loading: false, error: null, data: null }); setAsked(false); }, [input]);
  useEffect(() => {
    if (!asked) return;
    let live = true;
    setState({ loading: true, error: null, data: null });
    api.explain(input, task).then((d) => live && setState({ loading: false, error: null, data: d }), (e) => live && setState({ loading: false, error: e, data: null }));
    return () => { live = false; };
  }, [asked, task, input]);

  const f = fmtContribution(task);
  return (
    <Card className="p-4 sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="flex items-center gap-2 text-lg font-semibold"><HelpCircle size={19} className="text-primary" aria-hidden /> Why this estimate?</h3>
          <p className="text-sm text-muted">How each input moved the prediction for <strong>this</strong> customer, relative to a typical past building.</p>
        </div>
        {!asked && <Button onClick={() => setAsked(true)}>Explain this prediction</Button>}
      </div>
      {asked && (
        <div className="mt-4 space-y-4">
          <Tabs label="Prediction to explain" value={task} onChange={setTask} tabs={TASK_ORDER.map((t) => ({ value: t, label: TASK_LABEL[t] }))} />
          <div id={`panel-${task}`} role="tabpanel" aria-labelledby={`tab-${task}`} aria-live="polite">
            {state.loading && <LoadingBlock label="Computing explanation…" rows={2} />}
            {!!state.error && <ErrorState error={state.error} onRetry={() => setAsked(true)} />}
            {state.data && (
              <div className="space-y-3">
                <p className="text-sm">
                  A typical past building would get <strong className="num">{fmtTotal(task, state.data.base_value)}</strong>; for this customer the model says{" "}
                  <strong className="num">{fmtTotal(task, state.data.prediction)}</strong>.
                </p>
                <div style={{ height: Math.max(220, state.data.contributions.length * 34) }} role="img"
                  aria-label={`Contribution of each input: ${state.data.contributions.slice(0, 3).map((c) => `${FEATURE_LABEL[c.feature]} ${f(c.contribution)}`).join(", ")}`}>
                  <ContributionsChart data={state.data} f={f} />
                </div>
                <ul className="grid gap-1 text-sm sm:grid-cols-2">
                  {state.data.contributions.slice(0, 6).map((c) => (
                    <li key={c.feature} className="rounded-lg bg-surface2 px-3 py-1.5">
                      <span className="font-semibold">{FEATURE_LABEL[c.feature]}</span>
                      <span className="text-muted"> ({featureValue(c.feature, c.value)})</span>{" "}
                      <span className={c.contribution >= 0 ? "text-primary" : "text-clay"}>{c.contribution >= 0 ? "pushes up" : "pushes down"} {f(Math.abs(c.contribution)).replace(/^[+−]/, "")}</span>
                    </li>
                  ))}
                </ul>
                <Alert tone="neutral">{state.data.caveat} Method: {state.data.method}.</Alert>
              </div>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}

function ContributionsChart({ data, f }: { data: ExplainResponse; f: (v: number) => string }) {
  const items = data.contributions.map((c) => ({ label: FEATURE_LABEL[c.feature] ?? titleCase(c.feature), value: c.contribution }));
  return (
    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
      <ContributionBars items={items} fmt={f} />
    </ResponsiveContainer>
  );
}

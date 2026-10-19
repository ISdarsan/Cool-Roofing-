import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowDown, ArrowUp, Check, FileText, Info, RotateCcw, Trophy } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../lib/api";
import { useAsync, useDebounced } from "../lib/hooks";
import { inr, inrCompact, num, pct } from "../lib/format";
import { METHOD_LABEL, METRIC_LABEL, QUALITY_LABEL } from "../lib/labels";
import type { CompareResponse, MetricKey, Objective, OptionRow } from "../lib/types";
import { toBase, useApp, validateForm } from "../state/AppState";
import { Alert, Badge, Button, Card, EmptyState, ErrorState, LoadingBlock, PageHeader, SectionTitle, cx, useDocumentTitle } from "../components/ui";
import { ChartCard, SERIES, ScatterPlot } from "../components/charts";

const OBJ_ORDER: Objective[] = ["balanced", "cost", "temperature_drop", "units_saved", "high_rating", "referral"];
const METRICS: MetricKey[] = ["cost", "temperature_drop", "units_saved", "high_rating", "referral"];
const Q_COLOR: Record<string, string> = { basic: "var(--c2)", intermediate: "var(--c1)", high: "var(--c4)" };
const optName = (o: OptionRow) => `${METHOD_LABEL[o.cool_roofing_method]} · ${QUALITY_LABEL[o.cool_roofing_quality]}`;

type SortKey = "rank" | MetricKey | "score";

export default function Compare() {
  useDocumentTitle("Roofing comparison");
  const { form, setForm, schema } = useApp();
  const [objective, setObjective] = useState<Objective>("balanced");
  const [weights, setWeights] = useState<Record<MetricKey, number> | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "rank", dir: 1 });
  const [chosen, setChosen] = useState<string | null>(null);

  const w = weights ?? (schema.data ? (Object.fromEntries(METRICS.map((m) => [m, Math.round(schema.data!.default_weights[m] * 100)])) as Record<MetricKey, number>) : null);
  const dw = useDebounced(w, 300);
  const errs = validateForm(form, schema.data);
  const valid = Object.keys(errs).length === 0;
  const baseKey = JSON.stringify(toBase(form));

  const cmp = useAsync<CompareResponse | null>(async () => {
    if (!valid || !dw) return null;
    const ww = objective === "balanced" ? Object.fromEntries(METRICS.map((m) => [m, dw[m] / 100])) : null;
    if (ww && Object.values(ww).every((v) => v === 0)) throw new Error("Give at least one priority a weight above zero.");
    return api.compare(toBase(form), objective, ww, { method: form.cool_roofing_method, quality: form.cool_roofing_quality });
  }, [baseKey, objective, JSON.stringify(dw), valid, form.cool_roofing_method, form.cool_roofing_quality]);

  const rows = useMemo(() => {
    const o = cmp.data?.options ?? [];
    const val = (r: OptionRow) => (sort.key === "rank" ? r.rank : sort.key === "score" ? r.score : r.predictions[sort.key]);
    return [...o].sort((a, b) => (val(a) - val(b)) * sort.dir);
  }, [cmp.data, sort]);

  const winners = useMemo(() => {
    const o = cmp.data?.options ?? [];
    if (!o.length || !cmp.data) return [];
    return METRICS.map((m) => {
      const dir = cmp.data!.metric_info[m].direction;
      const best = o.reduce((b, r) => (dir === "min" ? r.predictions[m] < b.predictions[m] : r.predictions[m] > b.predictions[m]) ? r : b);
      return { metric: m, option: best };
    });
  }, [cmp.data]);

  if (!valid) {
    return (
      <>
        <PageHeader title="Roofing comparison" subtitle="Compare every supported method and quality for the same building." />
        <EmptyState icon={<Info size={34} />} title="Enter the building details first">
          The comparison needs the customer’s building details. <Link to="/predict" className="font-semibold text-primary underline">Go to Customer prediction</Link> and fill in the form (or load the example).
        </EmptyState>
      </>
    );
  }

  const rec = cmp.data?.recommended;
  const sumW = w ? METRICS.reduce((s, m) => s + w[m], 0) : 0;

  return (
    <>
      <PageHeader title="Roofing comparison"
        subtitle="Same building, every method and quality the history supports. Choose what matters; the ranking and the reasons update."
        actions={<Link to="/predict" className="text-sm font-semibold text-primary underline-offset-2 hover:underline">Edit building details</Link>} />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,4fr)_minmax(0,8fr)]">
        <div className="space-y-5">
          <Card className="p-4">
            <fieldset>
              <legend className="mb-2 text-lg font-semibold">Decision objective</legend>
              <div className="space-y-1.5">
                {OBJ_ORDER.map((o) => (
                  <label key={o} className={cx("flex min-h-11 items-center gap-3 rounded-xl border px-3 py-2 text-sm has-[:focus-visible]:outline has-[:focus-visible]:outline-[3px] has-[:focus-visible]:outline-primary",
                    objective === o ? "border-primary bg-primary-soft font-semibold" : "border-line hover:border-primary/60")}>
                    <input type="radio" name="objective" className="sr-only" checked={objective === o} onChange={() => setObjective(o)} />
                    <span aria-hidden className={cx("grid h-4 w-4 place-items-center rounded-full border", objective === o ? "border-primary bg-primary text-on-primary" : "border-muted")}>{objective === o && <Check size={11} />}</span>
                    {schema.data?.objectives[o] ?? o}
                  </label>
                ))}
              </div>
            </fieldset>
          </Card>

          {objective === "balanced" && w && (
            <Card className="p-4">
              <div className="mb-1 flex items-center justify-between">
                <h2 className="text-lg font-semibold">Priorities</h2>
                <Button variant="ghost" className="min-h-9 px-2" onClick={() => setWeights(null)}><RotateCcw size={14} aria-hidden /> Default</Button>
              </div>
              <p className="mb-3 text-sm text-muted">Weights are re-scaled to 100%. Each metric is first normalised (0–1, cheaper/better = higher) across the options shown, so rupees, °C, kWh and probabilities are comparable.</p>
              <div className="space-y-3">
                {METRICS.map((m) => (
                  <div key={m}>
                    <div className="flex justify-between text-sm"><label htmlFor={`w-${m}`} className="font-medium">{METRIC_LABEL[m]}</label>
                      <span className="num text-muted">{sumW ? Math.round((w[m] / sumW) * 100) : 0}%</span></div>
                    <input id={`w-${m}`} type="range" min={0} max={100} step={5} value={w[m]} className="w-full"
                      onChange={(e) => setWeights({ ...w, [m]: Number(e.target.value) })} aria-valuetext={`${sumW ? Math.round((w[m] / sumW) * 100) : 0} percent`} />
                  </div>
                ))}
              </div>
            </Card>
          )}
          <Alert tone="neutral" title="How the score works">{cmp.data?.normalisation_note ?? "Scores are relative to the options shown."}</Alert>
        </div>

        <div className="min-w-0 space-y-5" aria-live="polite">
          {cmp.loading && <LoadingBlock label="Comparing options…" rows={4} />}
          {!!cmp.error && <ErrorState error={cmp.error} onRetry={cmp.reload} />}
          {cmp.data && rec && !cmp.loading && (
            <>
              <Card className="border-primary/40 p-5">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="flex items-center gap-1.5 text-sm font-semibold text-primary"><Trophy size={16} aria-hidden /> Recommended · {cmp.data.objective_label}</p>
                    <h2 className="mt-1 text-2xl font-bold">{optName(rec)}</h2>
                    <p className="mt-1 text-sm text-muted">{cmp.data.basis}</p>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <Badge tone="primary">Score {num(rec.score, 0)} / 100 (relative)</Badge>
                    {rec.is_current_selection && <Badge tone="good">Your current selection</Badge>}
                  </div>
                </div>
                <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-5">
                  <Fig k="Cost" v={inrCompact(rec.predictions.cost)} />
                  <Fig k="Temp drop" v={`${num(rec.predictions.temperature_drop)} °C`} />
                  <Fig k="kWh saved / mo" v={num(rec.predictions.units_saved, 0)} />
                  <Fig k="P(high rating)" v={pct(rec.predictions.high_rating)} />
                  <Fig k="P(referral)" v={pct(rec.predictions.referral)} />
                </dl>
                {cmp.data.notes.map((n) => <div key={n} className="mt-3"><Alert tone="warn">{n}</Alert></div>)}
                {cmp.data.warnings.map((n) => <div key={n} className="mt-3"><Alert tone="warn">{n}</Alert></div>)}
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button onClick={() => { setForm({ cool_roofing_method: rec.cool_roofing_method, cool_roofing_quality: rec.cool_roofing_quality }); setChosen(rec.id); }}>
                    {chosen === rec.id ? <><Check size={16} aria-hidden /> Selected for this customer</> : "Use this option"}
                  </Button>
                  <Link to="/proposal" className="inline-flex min-h-11 items-center gap-2 rounded-xl border border-line px-4 text-sm font-semibold hover:bg-surface2"><FileText size={16} aria-hidden /> Create proposal</Link>
                </div>
              </Card>

              <div className="grid gap-5 xl:grid-cols-2">
                <ChartCard title="Cost vs cooling trade-off" subtitle="Each dot is one method × quality; bigger dot = more kWh saved."
                  summary={`Scatter of ${cmp.data.options.length} options: predicted cost against predicted temperature drop, coloured by quality.`}
                  table={{ headers: ["Option", "Cost (₹)", "Drop (°C)", "kWh/mo"], rows: cmp.data.options.map((o) => [optName(o), Math.round(o.predictions.cost), num(o.predictions.temperature_drop, 2), num(o.predictions.units_saved, 0)]) }}
                  height={300}>
                  <ScatterPlot xLabel="Predicted cost" yLabel="Temperature drop (°C)" xFmt={inrCompact} yFmt={(v) => num(v)} zKey="kWh saved"
                    series={(["basic", "intermediate", "high"] as const).map((q) => ({
                      name: `${QUALITY_LABEL[q]} quality`, color: Q_COLOR[q],
                      points: cmp.data!.options.filter((o) => o.cool_roofing_quality === q).map((o) => ({ x: o.predictions.cost, y: o.predictions.temperature_drop, z: Math.max(o.predictions.units_saved, 1), label: optName(o) })),
                    }))} />
                </ChartCard>

                <ChartCard title="Why the top options rank where they do" subtitle="Each bar is the option’s score, split by what contributed (weights × normalised metric)."
                  summary="Stacked bars of score contributions for the top six options."
                  table={{ headers: ["Option", ...METRICS.map((m) => METRIC_LABEL[m]), "Score"], rows: cmp.data.options.slice(0, 6).map((o) => [optName(o), ...METRICS.map((m) => num((o.contributions[m] ?? 0) * 100, 1)), num(o.score, 1)]) }}
                  height={300}>
                  <BarChart layout="vertical" margin={{ top: 4, right: 16, left: 4, bottom: 4 }}
                    data={cmp.data.options.slice(0, 6).map((o) => ({ name: optName(o), ...Object.fromEntries(METRICS.map((m) => [m, (o.contributions[m] ?? 0) * 100])) }))}>
                    <CartesianGrid stroke="var(--grid)" horizontal={false} />
                    <XAxis type="number" tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--line)" domain={[0, 100]} />
                    <YAxis type="category" dataKey="name" width={150} tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--line)" />
                    <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 12, fontSize: 12 }} formatter={(v) => `${Number(v).toFixed(1)} pts`} cursor={{ fill: "var(--surface-2)" }} />
                    <Legend verticalAlign="top" height={44} wrapperStyle={{ fontSize: 11 }} />
                    {METRICS.map((m, i) => <Bar key={m} dataKey={m} name={METRIC_LABEL[m]} stackId="s" fill={SERIES[i]} isAnimationActive={false} />)}
                  </BarChart>
                </ChartCard>
              </div>

              <Card className="p-4">
                <SectionTitle title="All options" hint="Click a column to sort. Ranked for the objective above." />
                <div className="table-wrap">
                  <table className="w-full min-w-[600px] text-left text-sm">
                    <caption className="sr-only">Predicted outcomes for every supported roofing option</caption>
                    <thead>
                      <tr className="border-b border-line text-xs text-muted">
                        <th scope="col" className="py-2 pr-2">Option</th>
                        {([["cost", "Cost"], ["temperature_drop", "Drop °C"], ["units_saved", "kWh/mo"], ["high_rating", "P(high)"], ["referral", "P(ref.)"], ["score", "Score"]] as [SortKey, string][]).map(([k, label]) => (
                          <th key={k} scope="col" aria-sort={sort.key === k ? (sort.dir === 1 ? "ascending" : "descending") : "none"} className="px-1.5 py-2">
                            <button type="button" className="inline-flex min-h-9 items-center gap-1 font-semibold hover:text-ink"
                              onClick={() => setSort((s) => ({ key: k, dir: s.key === k ? (s.dir === 1 ? -1 : 1) : (k === "rank" || k === "cost" ? 1 : -1) }))}>
                              {label}{sort.key === k && (sort.dir === 1 ? <ArrowUp size={12} aria-hidden /> : <ArrowDown size={12} aria-hidden />)}
                            </button>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((o) => (
                        <tr key={o.id} className={cx("border-b border-line/60", o.id === rec.id && "bg-primary-soft/60")}>
                          <th scope="row" className="py-2 pr-2 font-medium"><span className="num mr-1.5 text-muted">{o.rank}.</span>
                            {optName(o)}
                            <span className="ml-1 inline-flex flex-wrap gap-1 align-middle">
                              {o.is_current_selection && <Badge tone="primary">current</Badge>}
                              {o.low_support && <Badge tone="warn" title={`Only ${o.support} similar historical installations`}>few examples</Badge>}
                            </span>
                          </th>
                          <td className="num px-2" title={`likely ${inr(o.intervals.cost[0])} – ${inr(o.intervals.cost[1])}`}>{inrCompact(o.predictions.cost)}</td>
                          <td className="num px-2">{num(o.predictions.temperature_drop)}</td>
                          <td className="num px-2">{num(o.predictions.units_saved, 0)}</td>
                          <td className="num px-2">{pct(o.predictions.high_rating)}</td>
                          <td className="num px-2">{pct(o.predictions.referral)}</td>
                          <td className="num px-2 font-semibold">{num(o.score, 0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              <Card className="p-4">
                <SectionTitle title="Winner by single objective" hint="How the answer changes with what you optimise. No option wins everything." />
                <ul className="grid gap-2 sm:grid-cols-2">
                  {winners.map((x) => (
                    <li key={x.metric} className="flex items-center justify-between gap-2 rounded-xl bg-surface2 px-3 py-2 text-sm">
                      <span className="text-muted">{METRIC_LABEL[x.metric]}</span>
                      <span className="text-right font-semibold">{optName(x.option)}</span>
                    </li>
                  ))}
                </ul>
              </Card>
            </>
          )}
        </div>
      </div>
    </>
  );
}

const Fig = ({ k, v }: { k: string; v: string }) => (
  <div className="rounded-xl bg-surface2 p-2.5"><dt className="text-xs text-muted">{k}</dt><dd className="num text-lg font-semibold">{v}</dd></div>
);

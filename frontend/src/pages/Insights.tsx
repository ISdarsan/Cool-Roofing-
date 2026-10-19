import { useMemo, useState } from "react";
import { api } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { gridTable } from "../lib/insights";
import { inr, inrCompact, num, pct } from "../lib/format";
import { FEATURE_LABEL, METHOD_LABEL, METHOD_ORDER, QUALITY_LABEL, ROOF_LABEL, TASK_LABEL, TASK_ORDER } from "../lib/labels";
import type { Json, TaskKey } from "../lib/types";
import { Alert, Badge, Card, ErrorState, LoadingBlock, PageHeader, SectionTitle, Tabs, useDocumentTitle } from "../components/ui";
import { CategoryBars, ChartCard, HistogramBars, ScatterPlot, SERIES } from "../components/charts";
import type { Hist } from "../components/charts";
import { ReportViewer } from "../components/Markdown";

type Tab = "quality" | "drivers" | "methods" | "reports";
const TABS = [
  { value: "quality" as const, label: "Data quality" }, { value: "drivers" as const, label: "What drives outcomes" },
  { value: "methods" as const, label: "Methods & customers" }, { value: "reports" as const, label: "Full reports" },
];

export default function Insights() {
  useDocumentTitle("Data quality & insights");
  const [tab, setTab] = useState<Tab>("quality");
  const dq = useAsync(() => api.dataQuality(), []);
  const ins = useAsync(() => api.insights(), []);
  return (
    <>
      <PageHeader title="Data quality & insights" subtitle="What was wrong with the raw data, what we did about it, and what the cleaned data and models say about drivers and roofing methods." />
      <Tabs label="Insight sections" value={tab} onChange={setTab} tabs={TABS} />
      <div id={`panel-${tab}`} role="tabpanel" aria-labelledby={`tab-${tab}`} className="mt-5 space-y-6">
        {tab === "quality" && (dq.loading ? <LoadingBlock rows={4} /> : dq.error ? <ErrorState error={dq.error} onRetry={dq.reload} /> : dq.data && <Quality profile={dq.data.profile} summary={dq.data.summary} />)}
        {tab === "drivers" && (ins.loading ? <LoadingBlock rows={4} /> : ins.error ? <ErrorState error={ins.error} onRetry={ins.reload} /> : ins.data && <Drivers ins={ins.data} />)}
        {tab === "methods" && (ins.loading ? <LoadingBlock rows={4} /> : ins.error ? <ErrorState error={ins.error} onRetry={ins.reload} /> : ins.data && <Methods ins={ins.data} />)}
        {tab === "reports" && (
          <div className="space-y-3">
            <p className="text-sm text-muted">The complete generated reports. Nothing is fetched until you open one.</p>
            <ReportViewer name="data_quality_report" title="data-quality report" />
            <ReportViewer name="model_evaluation" title="model evaluation" />
            <ReportViewer name="business_insights" title="business insights" />
            <ReportViewer name="business_recommendations" title="business recommendations" />
          </div>
        )}
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ data quality */
const histTable = (h: Hist, fmt: (v: number) => string) => ({ headers: ["Range", "Buildings"], rows: h.counts.map((c, i) => [`${fmt(h.edges[i])}–${fmt(h.edges[i + 1])}`, c]) });

function Quality({ profile: p, summary: s }: { profile: Json; summary: Json }) {
  const z = s.zero_consumption_placeholders, t = s.temperature_sentinel_rows;
  const tasks: [TaskKey, string][] = [
    ["cost", "building size missing (not imputed for cost)"], ["temperature_drop", "before or after temperature is the 120 °C placeholder"],
    ["units_saved", "before or after kWh is the 0 placeholder"], ["rating", "—"], ["high_rating", "—"], ["referral", "—"],
  ];
  const tb = p.temperature_before_after_cleaning;
  const miss = (p.missingness as Json[]).filter((r) => r.before > 0 || r.after > 0);
  return (
    <>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["Raw records", s.n_raw_rows.toLocaleString("en-IN"), "no duplicate rows or IDs"],
          ["120 °C placeholder temperatures", t.before + t.after, `${t.before} before · ${t.after} after → set to missing`],
          ["0 kWh placeholder readings", z.before + z.after, `${z.before} before · ${z.after} after → set to missing`],
          ["Adverse outcomes kept", s.negative_savings_rows, "electricity use went up; real, so retained"],
        ].map(([k, v, sub]) => (
          <Card key={String(k)} className="p-4"><p className="text-xs font-semibold uppercase tracking-wide text-muted">{k}</p><p className="num text-3xl font-semibold">{v}</p><p className="text-xs text-muted">{sub}</p></Card>
        ))}
      </div>

      <Alert tone="neutral" title="Why 0 kWh and 120 °C are treated as placeholders">
        120.0 °C repeats {t.before + t.after} times far outside every other reading. Exactly 0 kWh appears {z.before + z.after} times while 1–3 kWh never occurs, and those rows are physically inconsistent
        (e.g. 0 kWh before but up to {z.after_when_before_is_zero_max} kWh after). The true values cannot be recovered, so they are set to missing — not capped, not guessed — and the record is excluded only from the task that needs it.
        Removing them cut electricity-savings error substantially. This is a high-confidence inference, not a certainty.
      </Alert>

      <Card className="p-5">
        <SectionTitle title="Usable records per task" hint="A record with one invalid target stays usable for every other task." />
        <div className="table-wrap"><table className="w-full min-w-[560px] text-left text-sm">
          <caption className="sr-only">Usable and excluded records for each prediction task</caption>
          <thead><tr className="border-b border-line text-xs text-muted"><th scope="col" className="py-2">Task</th><th scope="col">Usable</th><th scope="col">Excluded</th><th scope="col">Why excluded</th></tr></thead>
          <tbody>{tasks.map(([k, why]) => {
            const n = s.eligible_rows_by_task[k];
            return <tr key={k} className="border-b border-line/60"><th scope="row" className="py-2 font-medium">{TASK_LABEL[k].replace(" probability", "")}</th><td className="num">{n.toLocaleString("en-IN")}</td><td className="num">{s.n_raw_rows - n}</td><td className="text-muted">{why}</td></tr>;
          })}</tbody></table></div>
      </Card>

      <div className="grid gap-5 xl:grid-cols-2">
        <ChartCard title="Roof temperature, before vs after cleaning" subtitle="Raw data shows the 120 °C spike; cleaned data is physical." summary="Histograms of roof temperature before installation, raw versus cleaned, with a spike at 120 degrees in the raw data."
          table={histTable(tb.raw_before, (v) => v.toFixed(0))}>
          <HistogramBars xLabel="avg roof temperature before (°C)" series={[{ key: "raw", label: "raw", hist: tb.raw_before, color: "var(--c3)" }, { key: "clean", label: "cleaned", hist: tb.clean_before, color: "var(--c1)" }]} />
        </ChartCard>
        <ChartCard title="Missing values by column" subtitle="Raw file vs after invalid values were set to missing." summary="Bar chart of missing cell counts per column, raw versus after cleaning."
          table={{ headers: ["Column", "Raw", "After cleaning"], rows: miss.map((r) => [r.column, r.before, r.after]) }}>
          <CategoryBars data={miss.map((r) => ({ name: r.column.replace("avg_", "").replace("_temperature", " temp").replace(/_/g, " "), value: r.after, color: r.after > r.before ? "var(--c3)" : "var(--c4)" }))} name="missing after cleaning" fmt={(v) => String(v)} />
        </ChartCard>
        <ChartCard title="Installation cost" subtitle="Distribution of valid costs (INR)" summary="Histogram of installation cost." table={histTable(p.targets.cost, (v) => inrCompact(v))}>
          <HistogramBars xLabel="cost" fmt={inrCompact} series={[{ key: "c", label: "buildings", hist: p.targets.cost, color: "var(--c1)" }]} />
        </ChartCard>
        <ChartCard title="Roof temperature drop" subtitle="Valid rows only (°C)" summary="Histogram of temperature drop." table={histTable(p.targets.temperature_drop, (v) => v.toFixed(1))}>
          <HistogramBars xLabel="temperature drop (°C)" fmt={(v) => v.toFixed(1)} series={[{ key: "c", label: "buildings", hist: p.targets.temperature_drop, color: "var(--c2)" }]} />
        </ChartCard>
        <ChartCard title="Monthly electricity saved" subtitle="Dashed line = no saving; negative values are genuine adverse outcomes" summary="Histogram of units saved per month." table={histTable(p.targets.units_saved, (v) => v.toFixed(0))}>
          <HistogramBars xLabel="kWh saved per month" refX={0} series={[{ key: "c", label: "buildings", hist: p.targets.units_saved, color: "var(--c5)" }]} />
        </ChartCard>
        <ChartCard title="Customer ratings and referrals" subtitle={`Rating counts; referral rate ${pct(p.targets.referral.rate, 1)} (${p.targets.referral.true} of ${p.targets.referral.true + p.targets.referral.false})`}
          summary="Bar chart of customer rating counts." table={{ headers: ["Rating", "Buildings"], rows: p.targets.rating.values.map((v: number, i: number) => [v, p.targets.rating.counts[i]]) }}>
          <CategoryBars data={p.targets.rating.values.map((v: number, i: number) => ({ name: `${v}★`, value: p.targets.rating.counts[i] }))} name="buildings" />
        </ChartCard>
        <ChartCard title="Cost vs building size" subtitle="Cost scales with size; quality sets the rate." summary="Scatter of cost against building size coloured by quality grade."
          table={{ headers: ["Size", "Cost", "Quality"], rows: p.scatter.cost_vs_size.slice(0, 40).map((r: Json) => [r.x, r.y, r.g]) }}>
          <ScatterPlot xLabel="building size" yLabel="cost" yFmt={inrCompact} series={(["basic", "intermediate", "high"] as const).map((q, i) => ({ name: QUALITY_LABEL[q], color: [SERIES[1], SERIES[0], SERIES[3]][i], points: p.scatter.cost_vs_size.filter((r: Json) => r.g === q).map((r: Json) => ({ x: r.x, y: r.y })) }))} />
        </ChartCard>
        <ChartCard title="Savings vs current electricity use" subtitle="Savings grow with usage; low-use buildings hover around zero." summary="Scatter of units saved against units before, coloured by AC count."
          table={{ headers: ["kWh before", "kWh saved", "AC units"], rows: p.scatter.units_saved_vs_before.slice(0, 40).map((r: Json) => [r.x, r.y, r.g]) }}>
          <ScatterPlot xLabel="kWh/month before" yLabel="kWh/month saved" series={["0.0", "1.0", "2.0", "3.0"].map((g, i) => ({ name: `${g.slice(0, 1)} AC`, color: SERIES[i], points: p.scatter.units_saved_vs_before.filter((r: Json) => r.g === g).map((r: Json) => ({ x: r.x, y: r.y })) }))} />
        </ChartCard>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        {(["temperature_drop_by_method", "temperature_drop_by_roofing_type"] as const).map((k) => {
          const g: Json[] = p.groups[k];
          const lab = (x: string) => (METHOD_LABEL as Record<string, string>)[x] ?? (ROOF_LABEL as Record<string, string>)[x] ?? x;
          return (
            <ChartCard key={k} title={`Temperature drop by ${k.endsWith("method") ? "method" : "existing roof type"}`} subtitle="Mean with 95% interval; n shown in the table" summary="Mean temperature drop per group with confidence intervals."
              table={{ headers: ["Group", "n", "Mean °C", "95% low", "95% high"], rows: g.map((r) => [lab(r.group), r.n, num(r.mean, 2), num(r.ci95[0], 2), num(r.ci95[1], 2)]) }}>
              <CategoryBars data={g.map((r) => ({ name: lab(r.group), value: r.mean, lo: r.ci95[0], hi: r.ci95[1] }))} fmt={(v) => num(v, 1)} name="mean °C" domain={[0, "auto"]} />
            </ChartCard>
          );
        })}
      </div>

      <Card className="p-5">
        <SectionTitle title="Audit trail summary" hint="Every flagged value is listed record-by-record in reports/cleaning_audit.csv" />
        <div className="table-wrap"><table className="w-full min-w-[520px] text-left text-sm">
          <caption className="sr-only">Count of flagged records by check and classification</caption>
          <thead><tr className="border-b border-line text-xs text-muted"><th scope="col" className="py-2">Check</th><th scope="col">Classification</th><th scope="col">Records</th></tr></thead>
          <tbody>{(p.audit_by_check as Json[]).map((r, i) => (
            <tr key={i} className="border-b border-line/60"><th scope="row" className="py-2 font-medium">{String(r.check).replace(/_/g, " ")}</th>
              <td><Badge tone={r.classification === "invalid" ? "bad" : r.classification === "valid_adverse" ? "warn" : "neutral"}>{String(r.classification).replace("_", " ")}</Badge></td><td className="num">{r.n_records}</td></tr>))}</tbody></table></div>
      </Card>
    </>
  );
}

/* ------------------------------------------------------------------ drivers */
function Drivers({ ins }: { ins: Json }) {
  const [task, setTask] = useState<TaskKey>("cost");
  const imp = useMemo(() => (ins.feature_importance as Json[]).filter((r) => r.task === task).sort((a, b) => b.share - a.share), [ins, task]);
  const seg = useMemo(() => (ins.segment_errors as Json[]).filter((r) => r.task === task), [ins, task]);
  const reg = !["high_rating", "referral"].includes(task);
  const segs = ["cool_roofing_method", "cool_roofing_quality", "roofing_type", "location_type", "size_band"];
  const lab = (x: string) => (METHOD_LABEL as Record<string, string>)[x] ?? (ROOF_LABEL as Record<string, string>)[x] ?? (QUALITY_LABEL as Record<string, string>)[x] ?? x.replace(/_/g, " ");
  return (
    <>
      <Tabs label="Outcome" value={task} onChange={setTask} tabs={TASK_ORDER.map((k) => ({ value: k, label: TASK_LABEL[k].replace(" probability", "").replace("Monthly electricity saved", "Electricity saved") }))} />
      <div className="grid gap-5 xl:grid-cols-2">
        <ChartCard title={`What drives: ${TASK_LABEL[task].replace(" probability", "")}`} subtitle={`Share of model skill lost when each input is shuffled (${imp[0]?.metric}; hold-out, n=${imp[0]?.n_holdout})`}
          summary="Bar chart of permutation importance share per input." height={320}
          table={{ headers: ["Input", "Share", "Importance", "± std"], rows: imp.map((r) => [FEATURE_LABEL[r.feature], pct(r.share, 1), num(r.importance, 4), num(r.std, 4)]) }}>
          <CategoryBars layout="horizontal" fmt={(v) => `${(v * 100).toFixed(0)}%`} data={imp.map((r) => ({ name: FEATURE_LABEL[r.feature], value: r.share }))} name="share of importance" />
        </ChartCard>
        <Card className="p-5">
          <SectionTitle title="Reading this" />
          <ul className="list-disc space-y-2 pl-5 text-sm text-muted">
            <li><strong className="text-ink">Global, not personal.</strong> This shows what matters across all buildings. Per-customer reasons are on the prediction page (“Why this estimate?”).</li>
            <li><strong className="text-ink">Association, not causation.</strong> An influential input is useful for prediction; it does not prove that changing it would change the outcome.</li>
            <li>Inputs with ~0% share add no usable signal for this outcome in the data.</li>
          </ul>
        </Card>
      </div>
      <Card className="p-5">
        <SectionTitle title="Where the model is less reliable" hint="Out-of-fold error by segment. Small groups (< 30) are flagged and should not drive decisions." />
        <div className="table-wrap"><table className="w-full min-w-[560px] text-left text-sm">
          <caption className="sr-only">Model error by customer and building segment</caption>
          <thead><tr className="border-b border-line text-xs text-muted"><th scope="col" className="py-2">Segment</th><th scope="col">Group</th><th scope="col">n</th><th scope="col">{reg ? `MAE (${task === "cost" ? "₹" : task === "temperature_drop" ? "°C" : task === "units_saved" ? "kWh" : "★"})` : "Accuracy"}</th></tr></thead>
          <tbody>{segs.flatMap((sg) => seg.filter((r) => r.segment === sg).map((r) => (
            <tr key={sg + r.level} className="border-b border-line/60"><td className="py-1.5 text-muted">{sg.replace(/_/g, " ")}</td><th scope="row" className="font-medium">{lab(r.level)} {r.small_sample && <Badge tone="warn">small n</Badge>}</th><td className="num">{r.n}</td>
              <td className="num">{reg ? (task === "cost" ? inr(r.mae) : num(r.mae, 2)) : pct(r.accuracy, 1)}</td></tr>)))}</tbody></table></div>
      </Card>
    </>
  );
}

/* ------------------------------------------------------------------ methods & customers */
function Methods({ ins }: { ins: Json }) {
  const me = ins.marginal_effects;
  const grid = gridTable(ins).sort((a, b) => a.cost / a.drop - b.cost / b.drop);
  const series: [string, string, (v: number) => string][] = [["cost", "Average cost", inrCompact], ["temperature_drop", "Temperature drop (°C)", (v) => num(v)], ["units_saved", "kWh saved / month", (v) => num(v, 0)], ["high_rating", "P(high rating)", (v) => pct(v)], ["referral", "P(referral)", (v) => pct(v)]];
  const obs = ins.observed_segments;
  const ceil = ins.predictability_ceiling;
  return (
    <>
      <Alert tone="neutral" title="How to read the method comparison">
        We set every concrete-roof building ({me.population.method_comparison_rows.toLocaleString("en-IN")} of them — the only roofs accepting all five methods) to each method in turn and averaged the model’s prediction. Bars show the mean; whiskers are bootstrap intervals over buildings and exclude model error and confounding.
      </Alert>
      <div className="grid gap-5 lg:grid-cols-2 xl:grid-cols-3">
        {series.map(([k, title, fmt]) => (
          <ChartCard key={k} title={title} subtitle="By method (quality at each building’s own value)" summary={`${title} by roofing method on concrete roofs.`} height={250}
            table={{ headers: ["Method", "Mean", "95% low", "95% high"], rows: me.method[k].map((r: Json) => [METHOD_LABEL[r.level as keyof typeof METHOD_LABEL], fmt(r.mean), fmt(r.ci95[0]), fmt(r.ci95[1])]) }}>
            <CategoryBars data={METHOD_ORDER.map((m, i) => { const r = me.method[k].find((x: Json) => x.level === m); return { name: METHOD_LABEL[m].replace(" + coating", "+c").replace("Basic coating", "Basic c."), value: r.mean, lo: r.ci95[0], hi: r.ci95[1], color: SERIES[i] }; })} fmt={fmt} name={title} domain={[0, "auto"]} />
          </ChartCard>
        ))}
      </div>

      <Card className="p-5">
        <SectionTitle title="Value for money" hint="Tariff-free: rupees per °C of cooling and per kWh/month saved (lower is better). Sorted by ₹ per °C." />
        <div className="table-wrap"><table className="w-full min-w-[700px] text-left text-sm">
          <caption className="sr-only">Cost efficiency by method and quality on concrete roofs</caption>
          <thead><tr className="border-b border-line text-xs text-muted"><th scope="col" className="py-2">Option</th><th scope="col">Cost</th><th scope="col">₹ / °C</th><th scope="col">₹ / kWh-month</th><th scope="col">P(high)</th><th scope="col">P(ref.)</th><th scope="col">Break-even tariff (10 yr)</th></tr></thead>
          <tbody>{grid.map((g) => (
            <tr key={g.method + g.quality} className="border-b border-line/60"><th scope="row" className="py-1.5 font-medium">{METHOD_LABEL[g.method]} · {QUALITY_LABEL[g.quality as keyof typeof QUALITY_LABEL]}</th>
              <td className="num">{inrCompact(g.cost)}</td><td className="num">{inr(g.cost / g.drop)}</td><td className="num">{inr(g.cost / Math.max(g.kwh, 1e-9))}</td><td className="num">{pct(g.high)}</td><td className="num">{pct(g.referral)}</td><td className="num">₹{num(g.breakeven10, 0)}/kWh</td></tr>))}</tbody></table></div>
        <p className="mt-2 text-xs text-muted">Break-even tariff = the electricity price at which predicted savings repay the average cost in 10 years (no maintenance, no growth). Compare it with the customer’s real tariff.</p>
      </Card>

      <Card className="p-5">
        <SectionTitle title="Who is good business — knowable at the table" hint="Observed rates with 95% Wilson intervals. Groups under 30 are flagged." />
        {([["ac_group", "Number of AC units"], ["usage_band", "Current monthly electricity use"], ["location_type", "Location type"]] as const).map(([k, title]) => (
          <div key={k} className="mb-5 last:mb-0">
            <h3 className="mb-1 font-semibold">{title}</h3>
            <div className="table-wrap"><table className="w-full min-w-[560px] text-left text-sm">
              <caption className="sr-only">{title}: observed outcomes</caption>
              <thead><tr className="border-b border-line text-xs text-muted"><th scope="col" className="py-2">Group</th><th scope="col">n</th><th scope="col">Mean kWh saved</th><th scope="col">Mean rating</th><th scope="col">High-rating rate</th><th scope="col">Referral rate</th></tr></thead>
              <tbody>{(obs[k] as Json[]).map((r) => (
                <tr key={r.level} className="border-b border-line/60"><th scope="row" className="py-1.5 font-medium">{r.level} {r.small_sample && <Badge tone="warn">small n</Badge>}</th><td className="num">{r.n}</td>
                  <td className="num">{r.mean_units_saved == null ? "—" : num(r.mean_units_saved, 1)}</td><td className="num">{num(r.mean_rating, 2)}</td>
                  <td className="num">{pct(r.high_rating_rate)} <span className="text-xs text-muted">({pct(r.high_rating_ci95[0])}–{pct(r.high_rating_ci95[1])})</span></td>
                  <td className="num">{pct(r.referral_rate)} <span className="text-xs text-muted">({pct(r.referral_ci95[0])}–{pct(r.referral_ci95[1])})</span></td></tr>))}</tbody></table></div>
          </div>
        ))}
      </Card>

      <Alert tone="warn" title="Why referral is hard to predict (diagnostic, not a model)">
        Even with the <em>realised</em> savings and temperature drop known, referral ROC-AUC would only be {ceil.referral_auc.from_realised_savings_and_temperature_drop.toFixed(2)}
        {" "}({ceil.referral_auc.from_realised_rating_savings_drop.toFixed(2)} adding the realised rating). Referral behaviour is driven largely by factors outside this dataset, so a quote-time model can only prioritise.
      </Alert>
    </>
  );
}

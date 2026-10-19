import { useId } from "react";
import type { ReactNode } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter,
  ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { cx } from "./ui";

const AXIS = { stroke: "var(--line)", tick: { fill: "var(--muted)", fontSize: 12 } } as const;
const tooltipStyle = {
  contentStyle: { background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 12, color: "var(--ink)", fontSize: 13, boxShadow: "var(--shadow)" },
  labelStyle: { color: "var(--ink)", fontWeight: 600 },
  itemStyle: { color: "var(--ink)" },
};
export const SERIES = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)"];

/* ------------------------------------------------------------------ frame with accessible data fallback */
export interface TableData { headers: string[]; rows: (string | number)[][] }

export function ChartCard({ title, subtitle, summary, table, children, height = 280, className }: {
  title: string; subtitle?: ReactNode; summary?: string; table?: TableData; children: ReactNode; height?: number; className?: string;
}) {
  const id = useId();
  return (
    <figure className={cx("min-w-0 rounded-card border border-line bg-surface p-4 shadow-card", className)} aria-labelledby={`${id}-t`} aria-describedby={summary ? `${id}-s` : undefined}>
      <figcaption>
        <p id={`${id}-t`} className="font-semibold">{title}</p>
        {subtitle && <p className="text-sm text-muted">{subtitle}</p>}
        {summary && <p id={`${id}-s`} className="sr-only">{summary}</p>}
      </figcaption>
      <div className="mt-2 w-full" style={{ height }} role="img" aria-label={summary ?? title}>
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>{children as React.ReactElement}</ResponsiveContainer>
      </div>
      {table && (
        <details className="mt-2 text-sm">
          <summary className="inline-flex min-h-8 items-center text-primary underline-offset-2 hover:underline">View data table</summary>
          <div className="table-wrap mt-2 max-h-64 overflow-y-auto rounded-xl border border-line">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-surface2"><tr>{table.headers.map((h) => <th key={h} scope="col" className="px-3 py-2 font-semibold">{h}</th>)}</tr></thead>
              <tbody>{table.rows.map((r, i) => <tr key={i} className="border-t border-line">{r.map((c, j) => <td key={j} className="num px-3 py-1.5">{c}</td>)}</tr>)}</tbody>
            </table>
          </div>
        </details>
      )}
    </figure>
  );
}

/* ------------------------------------------------------------------ histogram(s) */
export interface Hist { edges: number[]; counts: number[]; n?: number }

export function binLabel(h: Hist, i: number, fmt: (v: number) => string) {
  return `${fmt(h.edges[i])}–${fmt(h.edges[i + 1])}`;
}

export function HistogramBars({ series, fmt = (v) => String(Math.round(v)), xLabel, refX }: {
  series: { key: string; label: string; hist: Hist; color: string }[]; fmt?: (v: number) => string; xLabel?: string; refX?: number;
}) {
  const base = series[0].hist;
  const data = base.counts.map((_, i) => ({
    x: (base.edges[i] + base.edges[i + 1]) / 2, range: binLabel(base, i, fmt),
    ...Object.fromEntries(series.map((s) => [s.key, s.hist.counts[i] ?? 0])),
  }));
  return (
    <BarChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: xLabel ? 18 : 4 }} barCategoryGap={1}>
      <CartesianGrid stroke="var(--grid)" vertical={false} />
      <XAxis dataKey="x" type="number" domain={[base.edges[0], base.edges[base.edges.length - 1]]} tickFormatter={fmt} {...AXIS}
        label={xLabel ? { value: xLabel, position: "insideBottom", offset: -10, fill: "var(--muted)", fontSize: 12 } : undefined} />
      <YAxis {...AXIS} allowDecimals={false} />
      <Tooltip {...tooltipStyle} labelFormatter={(_, p) => (p?.[0]?.payload?.range as string) ?? ""} cursor={{ fill: "var(--surface-2)" }} />
      {series.length > 1 && <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 12 }} />}
      {refX !== undefined && <ReferenceLine x={refX} stroke="var(--clay)" strokeDasharray="4 3" />}
      {series.map((s) => <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[2, 2, 0, 0]} />)}
    </BarChart>
  );
}

/* ------------------------------------------------------------------ bars with optional CI */
export interface BarDatum { name: string; value: number; lo?: number; hi?: number; color?: string }

export function CategoryBars({ data, fmt = (v) => String(v), color = "var(--c1)", layout = "vertical", name = "value", domain }: {
  data: BarDatum[]; fmt?: (v: number) => string; color?: string; layout?: "vertical" | "horizontal"; name?: string; domain?: [number | "auto", number | "auto"];
}) {
  const withErr = data.map((d) => ({ ...d, err: d.lo !== undefined && d.hi !== undefined ? [d.value - d.lo, d.hi - d.value] : undefined }));
  const horizontal = layout === "horizontal";
  return (
    <BarChart data={withErr} layout={horizontal ? "vertical" : "horizontal"} margin={{ top: 8, right: 24, left: horizontal ? 8 : -8, bottom: 4 }}>
      <CartesianGrid stroke="var(--grid)" horizontal={!horizontal} vertical={horizontal} />
      {horizontal ? (
        <>
          <XAxis type="number" tickFormatter={fmt} domain={domain} {...AXIS} />
          <YAxis type="category" dataKey="name" width={200} {...AXIS} />
        </>
      ) : (
        <>
          <XAxis dataKey="name" {...AXIS} interval={0} />
          <YAxis tickFormatter={fmt} domain={domain} {...AXIS} />
        </>
      )}
      <Tooltip {...tooltipStyle} formatter={(v) => fmt(Number(v))} cursor={{ fill: "var(--surface-2)" }} />
      <Bar dataKey="value" name={name} radius={horizontal ? [0, 6, 6, 0] : [6, 6, 0, 0]} isAnimationActive={false}>
        {withErr.map((d, i) => <Cell key={i} fill={d.color ?? color} />)}
        {withErr.some((d) => d.err) && <ErrorBar dataKey="err" width={5} stroke="var(--c-ink)" strokeWidth={1.5} direction={horizontal ? "x" : "y"} />}
      </Bar>
    </BarChart>
  );
}

/* ------------------------------------------------------------------ diverging contributions */
export function ContributionBars({ items, fmt }: { items: { label: string; value: number }[]; fmt: (v: number) => string }) {
  const data = items.map((d) => ({ ...d, abs: d.value }));
  return (
    <BarChart data={data} layout="vertical" margin={{ top: 4, right: 28, left: 8, bottom: 4 }}>
      <CartesianGrid stroke="var(--grid)" horizontal={false} />
      <XAxis type="number" tickFormatter={fmt} {...AXIS} />
      <YAxis type="category" dataKey="label" width={170} {...AXIS} />
      <ReferenceLine x={0} stroke="var(--c-ink)" />
      <Tooltip {...tooltipStyle} formatter={(v) => fmt(Number(v))} cursor={{ fill: "var(--surface-2)" }} />
      <Bar dataKey="abs" name="Effect on prediction" radius={4} isAnimationActive={false}>
        {data.map((d, i) => <Cell key={i} fill={d.value >= 0 ? "var(--c1)" : "var(--c3)"} />)}
      </Bar>
    </BarChart>
  );
}

/* ------------------------------------------------------------------ scatter */
export function ScatterPlot({ series, xLabel, yLabel, xFmt = String, yFmt = String, zKey, zRange = [40, 360] }: {
  series: { name: string; color: string; points: { x: number; y: number; z?: number; label?: string }[] }[];
  xLabel: string; yLabel: string; xFmt?: (v: number) => string; yFmt?: (v: number) => string; zKey?: string; zRange?: [number, number];
}) {
  return (
    <ScatterChart margin={{ top: 10, right: 16, left: 0, bottom: 22 }}>
      <CartesianGrid stroke="var(--grid)" />
      <XAxis type="number" dataKey="x" name={xLabel} tickFormatter={xFmt} {...AXIS} domain={["auto", "auto"]}
        label={{ value: xLabel, position: "insideBottom", offset: -12, fill: "var(--muted)", fontSize: 12 }} />
      <YAxis type="number" dataKey="y" name={yLabel} tickFormatter={yFmt} {...AXIS} width={64} domain={["auto", "auto"]}
        label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 6, fill: "var(--muted)", fontSize: 12 }} />
      {zKey && <ZAxis type="number" dataKey="z" name={zKey} range={zRange} />}
      <Tooltip {...tooltipStyle} cursor={{ strokeDasharray: "3 3" }}
        formatter={(v, n) => [n === xLabel ? xFmt(Number(v)) : n === yLabel ? yFmt(Number(v)) : String(v), n]} />
      <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 12 }} />
      {series.map((s) => <Scatter key={s.name} name={s.name} data={s.points} fill={s.color} fillOpacity={0.75} isAnimationActive={false} />)}
    </ScatterChart>
  );
}

/* ------------------------------------------------------------------ lines */
export function LineSeries({ data, xKey, lines, xLabel, yFmt = String, xFmt = String, refX, yDomain }: {
  data: Record<string, number>[]; xKey: string; lines: { key: string; label: string; color: string; dashed?: boolean }[];
  xLabel?: string; yFmt?: (v: number) => string; xFmt?: (v: number) => string; refX?: number; yDomain?: [number, number];
}) {
  return (
    <LineChart data={data} margin={{ top: 8, right: 16, left: -8, bottom: xLabel ? 20 : 4 }}>
      <CartesianGrid stroke="var(--grid)" />
      <XAxis dataKey={xKey} type="number" domain={["dataMin", "dataMax"]} tickFormatter={xFmt} {...AXIS}
        label={xLabel ? { value: xLabel, position: "insideBottom", offset: -12, fill: "var(--muted)", fontSize: 12 } : undefined} />
      <YAxis tickFormatter={yFmt} domain={yDomain} {...AXIS} />
      <Tooltip {...tooltipStyle} formatter={(v) => yFmt(Number(v))} labelFormatter={(l) => `${xLabel ?? xKey}: ${xFmt(Number(l))}`} />
      <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 12 }} />
      {refX !== undefined && <ReferenceLine x={refX} stroke="var(--c-ink)" strokeDasharray="4 3" />}
      {lines.map((l) => (
        <Line key={l.key} type="monotone" dataKey={l.key} name={l.label} stroke={l.color} strokeWidth={2.2}
          strokeDasharray={l.dashed ? "6 4" : undefined} dot={{ r: 2.5 }} isAnimationActive={false} />
      ))}
    </LineChart>
  );
}

/* ------------------------------------------------------------------ small non-recharts visuals */
/** Range bar: value marker inside a likely range, scaled to the range with padding. */
export function IntervalBar({ low, value, high, fmt, label }: { low: number; value: number; high: number; fmt: (v: number) => string; label: string }) {
  const span = Math.max(high - low, 1e-9);
  const pad = span * 0.25;
  const lo = low - pad, hi = high + pad;
  const p = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  return (
    <div className="mt-2" role="img" aria-label={`${label}: expected ${fmt(value)}, likely range ${fmt(low)} to ${fmt(high)}`}>
      <div className="relative h-2.5 rounded-full bg-surface2">
        <div className="absolute top-0 h-full rounded-full bg-primary/30" style={{ left: p(low), width: `calc(${p(high)} - ${p(low)})` }} />
        <div className="absolute top-1/2 h-4 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded bg-primary" style={{ left: p(value) }} />
      </div>
      <div className="num mt-1 flex justify-between text-xs text-muted"><span>{fmt(low)}</span><span>{fmt(high)}</span></div>
    </div>
  );
}

export function ProbabilityMeter({ value, cutoff, label }: { value: number; cutoff?: number; label: string }) {
  return (
    <div role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)} aria-label={label} className="mt-2">
      <div className="relative h-3 rounded-full bg-surface2">
        <div className="h-full rounded-full bg-primary" style={{ width: `${value * 100}%` }} />
        {cutoff !== undefined && <div className="absolute top-[-3px] h-[18px] w-0.5 bg-ink" style={{ left: `${cutoff * 100}%` }} title={`cut-off ${Math.round(cutoff * 100)}%`} />}
      </div>
      {cutoff !== undefined && <p className="mt-1 text-xs text-muted">Marker = decision cut-off ({Math.round(cutoff * 100)}%)</p>}
    </div>
  );
}

export function ConfusionMatrix({ tn, fp, fn, tp, negative, positive }: { tn: number; fp: number; fn: number; tp: number; negative: string; positive: string }) {
  const total = tn + fp + fn + tp;
  const cell = (v: number, good: boolean, label: string) => (
    <div className={cx("rounded-xl p-3 text-center", good ? "bg-primary-soft" : "bg-clay-soft")}>
      <p className="num text-2xl font-semibold">{v}</p>
      <p className="text-xs text-muted">{label} · {((v / total) * 100).toFixed(0)}%</p>
    </div>
  );
  return (
    <div role="table" aria-label="Confusion matrix on the hold-out set" className="grid grid-cols-[auto_1fr_1fr] items-stretch gap-2 text-sm">
      <div />
      <p className="text-center text-xs font-semibold text-muted">Predicted: {negative}</p>
      <p className="text-center text-xs font-semibold text-muted">Predicted: {positive}</p>
      <p className="self-center text-xs font-semibold text-muted [writing-mode:vertical-rl] rotate-180">Actual: {negative}</p>
      {cell(tn, true, "true negative")}{cell(fp, false, "false positive")}
      <p className="self-center text-xs font-semibold text-muted [writing-mode:vertical-rl] rotate-180">Actual: {positive}</p>
      {cell(fn, false, "false negative")}{cell(tp, true, "true positive")}
    </div>
  );
}

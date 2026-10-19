import { useEffect, useId, useRef } from "react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, Loader2, OctagonAlert, RefreshCw } from "lucide-react";
import { ApiError } from "../lib/api";
import type { Reliability } from "../lib/types";

export const cx = (...a: (string | false | null | undefined)[]) => a.filter(Boolean).join(" ");

export function useDocumentTitle(title: string) {
  useEffect(() => {
    document.title = `${title} · Cool Roofing Intelligence`;
  }, [title]);
}

/* ------------------------------------------------------------------ layout */
export function Card({ className, children, ...rest }: { className?: string; children: ReactNode } & React.HTMLAttributes<HTMLElement>) {
  return (
    <section className={cx("rounded-card border border-line bg-surface shadow-card", className)} {...rest}>
      {children}
    </section>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

export function SectionTitle({ title, hint, right }: { title: string; hint?: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
      <div>
        <h2 className="text-lg font-semibold">{title}</h2>
        {hint && <p className="text-sm text-muted">{hint}</p>}
      </div>
      {right}
    </div>
  );
}

/* ------------------------------------------------------------------ badges / alerts */
const TONES = {
  neutral: "bg-surface2 text-ink border-line",
  primary: "bg-primary-soft text-primary border-transparent",
  good: "bg-good-soft text-good border-transparent",
  warn: "bg-sun-soft text-warn-ink border-transparent",
  bad: "bg-clay-soft text-clay border-transparent",
} as const;
export type Tone = keyof typeof TONES;

export function Badge({ tone = "neutral", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={cx("inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold", TONES[tone])}>
      {children}
    </span>
  );
}

export function ReliabilityBadge({ r }: { r: Reliability }) {
  const tone: Tone = r.level === "high" ? "good" : r.level === "moderate" ? "warn" : "bad";
  return (
    <Badge tone={tone} title={r.basis}>
      {r.level === "high" ? <CheckCircle2 size={13} aria-hidden /> : <AlertTriangle size={13} aria-hidden />}
      {r.level === "high" ? "High" : r.level === "moderate" ? "Moderate" : "Low"} reliability
    </Badge>
  );
}

export function Alert({ tone = "neutral", title, children, role }: { tone?: Tone; title?: string; children?: ReactNode; role?: string }) {
  const Icon = tone === "bad" ? OctagonAlert : tone === "warn" ? AlertTriangle : tone === "good" ? CheckCircle2 : Info;
  return (
    <div role={role} className={cx("flex gap-3 rounded-xl border p-3 text-sm", TONES[tone])}>
      <Icon size={18} className="mt-0.5 shrink-0" aria-hidden />
      <div className="min-w-0">
        {title && <p className="font-semibold">{title}</p>}
        {children && <div className={cx(title && "mt-0.5", "text-ink/90")}>{children}</div>}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ buttons */
export function Button({
  variant = "primary", loading, children, className, ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost"; loading?: boolean }) {
  const v = {
    primary: "bg-primary text-on-primary hover:bg-primary-hover",
    secondary: "border border-line bg-surface text-ink hover:bg-surface2",
    ghost: "text-primary hover:bg-primary-soft",
  }[variant];
  return (
    <button
      {...rest}
      disabled={rest.disabled || loading}
      aria-busy={loading || undefined}
      className={cx("inline-flex min-h-11 items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition-colors duration-200 disabled:opacity-60", v, className)}
    >
      {loading && <Loader2 size={16} className="animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------ forms */
interface ShellProps { id: string; label: string; hint?: ReactNode; error?: string | null; optional?: boolean; children: ReactNode }
export function FieldShell({ id, label, hint, error, optional, children }: ShellProps) {
  return (
    <div className="min-w-0">
      <label htmlFor={id} className="mb-1 flex items-baseline justify-between gap-2 text-sm font-semibold">
        <span>{label}</span>
        {optional && <span className="text-xs font-normal text-muted">optional</span>}
      </label>
      {children}
      {hint && !error && <p id={`${id}-hint`} className="mt-1 text-xs text-muted">{hint}</p>}
      {error && <p id={`${id}-err`} role="alert" className="mt-1 text-xs font-semibold text-clay">{error}</p>}
    </div>
  );
}
const inputCls = (err?: string | null) =>
  cx("min-h-11 w-full rounded-xl border bg-surface px-3 py-2 text-base text-ink placeholder:text-muted/70", err ? "border-clay" : "border-line hover:border-primary/60");

export function SelectField({ id, label, value, onChange, options, hint, error, optional, disabled }: {
  id: string; label: string; value: string; onChange: (v: string) => void; options: { value: string; label: string }[];
  hint?: ReactNode; error?: string | null; optional?: boolean; disabled?: boolean;
}) {
  return (
    <FieldShell id={id} label={label} hint={hint} error={error} optional={optional}>
      <select id={id} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)} className={inputCls(error)}
        aria-invalid={!!error} aria-describedby={error ? `${id}-err` : hint ? `${id}-hint` : undefined}>
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </FieldShell>
  );
}

export function NumberField({ id, label, value, onChange, onBlur, hint, error, optional, unit, step = "any", min, max, placeholder }: {
  id: string; label: string; value: string; onChange: (v: string) => void; onBlur?: () => void; hint?: ReactNode;
  error?: string | null; optional?: boolean; unit?: string; step?: string; min?: number; max?: number; placeholder?: string;
}) {
  return (
    <FieldShell id={id} label={label} hint={hint} error={error} optional={optional}>
      <div className="relative">
        <input id={id} type="number" inputMode="decimal" value={value} step={step} min={min} max={max} placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)} onBlur={onBlur} style={unit ? { paddingRight: `${unit.length * 8 + 22}px` } : undefined} className={inputCls(error)}
          aria-invalid={!!error} aria-describedby={error ? `${id}-err` : hint ? `${id}-hint` : undefined} />
        {unit && <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted">{unit}</span>}
      </div>
    </FieldShell>
  );
}

export function Segmented<T extends string>({ legend, name, value, onChange, options, hint }: {
  legend: string; name: string; value: T; onChange: (v: T) => void; hint?: ReactNode;
  options: { value: T; label: string; sub?: string; disabled?: boolean }[];
}) {
  return (
    <fieldset className="min-w-0">
      <legend className="mb-1 text-sm font-semibold">{legend}</legend>
      <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(options.length, 3)}, minmax(0, 1fr))` }}>
        {options.map((o) => (
          <label key={o.value} className={cx(
            "relative flex min-h-11 flex-col justify-center rounded-xl border px-3 py-2 text-sm transition-colors duration-200 has-[:focus-visible]:outline has-[:focus-visible]:outline-[3px] has-[:focus-visible]:outline-primary",
            value === o.value ? "border-primary bg-primary-soft font-semibold text-ink" : "border-line bg-surface hover:border-primary/60",
            o.disabled && "cursor-not-allowed opacity-50")}>
            <input type="radio" name={name} value={o.value} checked={value === o.value} disabled={o.disabled}
              onChange={() => onChange(o.value)} className="sr-only" />
            <span>{o.label}</span>
            {o.sub && <span className="text-xs font-normal text-muted">{o.sub}</span>}
          </label>
        ))}
      </div>
      {hint && <p className="mt-1 text-xs text-muted">{hint}</p>}
    </fieldset>
  );
}

/* ------------------------------------------------------------------ tabs */
export function Tabs<T extends string>({ label, tabs, value, onChange }: {
  label: string; tabs: { value: T; label: string }[]; value: T; onChange: (v: T) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const onKey = (e: React.KeyboardEvent) => {
    const i = tabs.findIndex((t) => t.value === value);
    let n = i;
    if (e.key === "ArrowRight") n = (i + 1) % tabs.length;
    else if (e.key === "ArrowLeft") n = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") n = 0;
    else if (e.key === "End") n = tabs.length - 1;
    else return;
    e.preventDefault();
    onChange(tabs[n].value);
    (ref.current?.querySelectorAll<HTMLElement>("[role=tab]")[n])?.focus();
  };
  return (
    <div ref={ref} role="tablist" aria-label={label} onKeyDown={onKey} className="flex flex-wrap gap-1 rounded-2xl bg-surface2 p-1">
      {tabs.map((t) => (
        <button key={t.value} role="tab" id={`tab-${t.value}`} aria-selected={value === t.value} aria-controls={`panel-${t.value}`}
          tabIndex={value === t.value ? 0 : -1} onClick={() => onChange(t.value)}
          className={cx("min-h-11 rounded-xl px-3.5 py-2 text-sm font-semibold transition-colors duration-200",
            value === t.value ? "bg-surface text-primary shadow-sm" : "text-muted hover:text-ink")}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ states */
export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cx("animate-pulse rounded-xl bg-surface2", className)} />;
}

export function LoadingBlock({ label = "Loading…", rows = 3 }: { label?: string; rows?: number }) {
  return (
    <div role="status" aria-live="polite" className="space-y-3">
      <span className="sr-only">{label}</span>
      {Array.from({ length: rows }).map((_, i) => <Skeleton key={i} className="h-16 w-full" />)}
    </div>
  );
}

export function ErrorState({ error, onRetry, title = "Something went wrong" }: { error: unknown; onRetry?: () => void; title?: string }) {
  const e = error instanceof ApiError ? error : null;
  const message = e ? e.message : error instanceof Error ? error.message : "Unexpected error.";
  const offline = e?.status === 0;
  const notReady = e?.status === 503;
  return (
    <div role="alert" className="rounded-card border border-clay/40 bg-clay-soft p-5">
      <div className="flex items-start gap-3">
        <OctagonAlert className="mt-0.5 shrink-0 text-clay" size={20} aria-hidden />
        <div className="min-w-0">
          <p className="font-semibold text-clay">{offline ? "Prediction service unreachable" : notReady ? "Models not ready" : title}</p>
          <p className="mt-1 text-sm text-ink">{message}</p>
          {offline && <p className="mt-2 text-sm text-muted">Start the backend with <code className="num rounded bg-surface px-1">python -m coolroof serve</code>, then retry. No values are shown until the real models respond.</p>}
          {onRetry && <Button variant="secondary" className="mt-3" onClick={onRetry}><RefreshCw size={15} aria-hidden /> Try again</Button>}
        </div>
      </div>
    </div>
  );
}

export function EmptyState({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center rounded-card border border-dashed border-line bg-surface/60 p-10 text-center">
      {icon && <div className="mb-3 text-primary" aria-hidden>{icon}</div>}
      <p className="text-lg font-semibold">{title}</p>
      {children && <div className="mt-1 max-w-md text-sm text-muted">{children}</div>}
    </div>
  );
}

export function Stat({ label, value, sub, className }: { label: string; value: ReactNode; sub?: ReactNode; className?: string }) {
  return (
    <div className={className}>
      <p className="text-xs font-semibold uppercase tracking-wide text-muted">{label}</p>
      <p className="num mt-0.5 text-2xl font-semibold">{value}</p>
      {sub && <p className="mt-0.5 text-xs text-muted">{sub}</p>}
    </div>
  );
}

/** Stable id helper for aria wiring. */
export function useUid(prefix = "f") {
  return `${prefix}-${useId().replace(/:/g, "")}`;
}

import { useState } from "react";
import { PiggyBank } from "lucide-react";
import { api } from "../lib/api";
import { inr, num, years } from "../lib/format";
import type { EconomicsResponse } from "../lib/types";
import { Alert, Button, Card, ErrorState, NumberField } from "./ui";

export default function PaybackPanel({ cost, kwh, kwhLow, kwhHigh, onTariff }: {
  cost: number; kwh: number; kwhLow?: number; kwhHigh?: number; onTariff?: (t: number | null) => void;
}) {
  const [tariff, setTariff] = useState("");
  const [maint, setMaint] = useState("");
  const [esc, setEsc] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [state, setState] = useState<{ loading: boolean; error: unknown; data: EconomicsResponse | null }>({ loading: false, error: null, data: null });

  async function run() {
    const t = Number(tariff);
    if (tariff.trim() === "" || !Number.isFinite(t) || t <= 0 || t > 100) {
      setErr("Enter the customer’s electricity tariff in ₹ per kWh (between 0 and 100). It is never assumed.");
      return;
    }
    setErr(null);
    onTariff?.(t);
    setState({ loading: true, error: null, data: null });
    try {
      const data = await api.economics({
        cost_inr: cost, monthly_kwh_saved: kwh, tariff_inr_per_kwh: t,
        annual_maintenance_inr: maint.trim() === "" ? 0 : Number(maint), tariff_escalation_pct: esc.trim() === "" ? 0 : Number(esc), horizon_years: 10,
        ...(kwhLow !== undefined ? { monthly_kwh_saved_low: kwhLow } : {}), ...(kwhHigh !== undefined ? { monthly_kwh_saved_high: kwhHigh } : {}),
      });
      setState({ loading: false, error: null, data });
    } catch (e) {
      setState({ loading: false, error: e, data: null });
    }
  }
  const c = state.data?.central;
  return (
    <Card className="p-4 sm:p-5">
      <h3 className="flex items-center gap-2 text-lg font-semibold"><PiggyBank size={19} className="text-primary" aria-hidden /> Payback from electricity savings <span className="text-sm font-normal text-muted">(optional, assumption-based)</span></h3>
      <p className="text-sm text-muted">The data has no electricity prices, so enter the customer’s tariff. Savings are assumed constant through the year.</p>
      <form className="mt-3 grid gap-3 sm:grid-cols-3" onSubmit={(e) => { e.preventDefault(); void run(); }} noValidate>
        <NumberField id="tariff" label="Tariff" unit="₹/kWh" value={tariff} onChange={setTariff} error={err} hint="Required for payback" min={0} max={100} />
        <NumberField id="maint" label="Maintenance" unit="₹/year" value={maint} onChange={setMaint} optional min={0} />
        <NumberField id="esc" label="Tariff growth" unit="%/year" value={esc} onChange={setEsc} optional min={-20} max={30} />
        <div className="sm:col-span-3"><Button type="submit" loading={state.loading}>Calculate payback</Button></div>
      </form>
      {!!state.error && <div className="mt-3"><ErrorState error={state.error} /></div>}
      {state.data && c && (
        <div className="mt-4 space-y-3" aria-live="polite">
          <dl className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl bg-surface2 p-3"><dt className="text-xs text-muted">Simple payback</dt><dd className="num text-xl font-semibold" data-testid="payback-years">{years(c.simple_payback_years)}</dd></div>
            <div className="rounded-xl bg-surface2 p-3"><dt className="text-xs text-muted">Annual bill saving</dt><dd className="num text-xl font-semibold">{inr(c.annual_bill_saving_inr)}</dd></div>
            <div className="rounded-xl bg-surface2 p-3"><dt className="text-xs text-muted">Break-even tariff (10 yr)</dt><dd className="num text-xl font-semibold">{c.breakeven_tariff_inr_per_kwh ? `₹${num(c.breakeven_tariff_inr_per_kwh, 0)}/kWh` : "—"}</dd></div>
          </dl>
          {state.data.range && (
            <p className="text-sm">Using the model’s likely savings range: <strong className="num">{years(state.data.range.optimistic.simple_payback_years)}</strong> (optimistic) to <strong className="num">{years(state.data.range.pessimistic.simple_payback_years)}</strong> (pessimistic).</p>
          )}
          {c.payback_with_escalation_years != null && Number(esc) !== 0 && <p className="text-sm">With {esc}%/year tariff growth: <strong className="num">{years(c.payback_with_escalation_years)}</strong>.</p>}
          {state.data.warnings.map((w) => <Alert key={w} tone="warn">{w}</Alert>)}
          <details className="text-sm"><summary className="cursor-pointer font-semibold text-primary">Assumptions</summary>
            <ul className="mt-1 list-disc space-y-0.5 pl-5 text-muted">{state.data.assumptions.map((a) => <li key={a}>{a}</li>)}</ul></details>
        </div>
      )}
    </Card>
  );
}

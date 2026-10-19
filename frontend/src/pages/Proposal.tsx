import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { CheckCircle2, Download, FileText } from "lucide-react";
import { api } from "../lib/api";
import { inr, num } from "../lib/format";
import { LOCATION_LABEL, METHOD_LABEL, QUALITY_LABEL, ROOF_LABEL } from "../lib/labels";
import type { PredictResponse } from "../lib/types";
import { toInput, useApp, validateForm } from "../state/AppState";
import { Alert, Button, Card, EmptyState, ErrorState, LoadingBlock, NumberField, PageHeader, SectionTitle, useDocumentTitle } from "../components/ui";
import { FieldShell } from "../components/ui";

export default function Proposal() {
  useDocumentTitle("Proposal");
  const { form, prediction, schema } = useApp();
  const [customer, setCustomer] = useState("");
  const [preparedBy, setPreparedBy] = useState("");
  const [tariff, setTariff] = useState("");
  const [tariffErr, setTariffErr] = useState<string | null>(null);
  const [pred, setPred] = useState<{ loading: boolean; error: unknown; data: PredictResponse | null }>({ loading: true, error: null, data: null });
  const [pdf, setPdf] = useState<{ busy: boolean; error: unknown; done: string | null }>({ busy: false, error: null, done: null });

  const valid = Object.keys(validateForm(form, schema.data)).length === 0;
  const key = JSON.stringify(form);

  useEffect(() => {
    if (!valid) return;
    const input = toInput(form);
    if (prediction && JSON.stringify(prediction.input) === JSON.stringify(input)) { setPred({ loading: false, error: null, data: prediction.response }); return; }
    let live = true;
    setPred({ loading: true, error: null, data: null });
    api.predict(input).then((d) => live && setPred({ loading: false, error: null, data: d }), (e) => live && setPred({ loading: false, error: e, data: null }));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, valid]);

  if (!valid) {
    return (
      <>
        <PageHeader title="Customer proposal" />
        <EmptyState icon={<FileText size={34} />} title="No building to propose for yet">
          Enter the customer’s building details on <Link to="/predict" className="font-semibold text-primary underline">Customer prediction</Link> first. The proposal is built from the real model outputs for that building.
        </EmptyState>
      </>
    );
  }

  async function download() {
    let t: number | undefined;
    if (tariff.trim() !== "") {
      t = Number(tariff);
      if (!Number.isFinite(t) || t <= 0 || t > 100) { setTariffErr("Tariff must be between 0 and 100 ₹/kWh."); return; }
    }
    setTariffErr(null);
    setPdf({ busy: true, error: null, done: null });
    try {
      const blob = await api.proposalPdf({
        building: toInput(form), ...(customer.trim() ? { customer_name: customer.trim() } : {}),
        ...(preparedBy.trim() ? { prepared_by: preparedBy.trim() } : {}), ...(t ? { tariff_inr_per_kwh: t } : {}),
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = "cool-roofing-proposal.pdf"; document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
      setPdf({ busy: false, error: null, done: `${(blob.size / 1024).toFixed(0)} KB` });
    } catch (e) {
      setPdf({ busy: false, error: e, done: null });
    }
  }

  const p = pred.data?.predictions;
  return (
    <>
      <PageHeader title="Customer proposal" subtitle="A clear, honest PDF built from the actual model predictions for this building: the proposed option, likely ranges, alternatives, assumptions and limitations." />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Card className="space-y-4 p-5">
          <h2 className="text-lg font-semibold">Proposal details</h2>
          <FieldShell id="customer" label="Customer name" optional>
            <input id="customer" value={customer} maxLength={80} onChange={(e) => setCustomer(e.target.value)} className="min-h-11 w-full rounded-xl border border-line bg-surface px-3" />
          </FieldShell>
          <FieldShell id="prepared" label="Prepared by" optional>
            <input id="prepared" value={preparedBy} maxLength={80} onChange={(e) => setPreparedBy(e.target.value)} className="min-h-11 w-full rounded-xl border border-line bg-surface px-3" />
          </FieldShell>
          <NumberField id="proposal-tariff" label="Electricity tariff" unit="₹/kWh" optional value={tariff} onChange={setTariff} error={tariffErr}
            hint="Adds a payback section. Left blank, no payback is shown — it is never assumed." min={0} max={100} />
          <Button onClick={download} loading={pdf.busy} disabled={pred.loading || !!pred.error}><Download size={17} aria-hidden /> Download PDF proposal</Button>
          {pdf.done && <Alert tone="good" role="status"><span className="inline-flex items-center gap-1"><CheckCircle2 size={15} aria-hidden /> Proposal downloaded ({pdf.done}).</span></Alert>}
          {!!pdf.error && <ErrorState error={pdf.error} title="Could not create the proposal" />}
        </Card>

        <div className="min-w-0 space-y-4" aria-live="polite">
          <Card className="p-5">
            <SectionTitle title="What the proposal will say" right={<Link to="/predict" className="text-sm font-semibold text-primary hover:underline">Edit inputs</Link>} />
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm sm:grid-cols-3">
              <div><dt className="text-muted">Location</dt><dd className="font-medium">{LOCATION_LABEL[form.location_type]}</dd></div>
              <div><dt className="text-muted">Size</dt><dd className="num font-medium">{num(Number(form.building_size), 0)}</dd></div>
              <div><dt className="text-muted">Existing roof</dt><dd className="font-medium">{ROOF_LABEL[form.roofing_type]}</dd></div>
            </dl>
            <p className="mt-3 rounded-xl bg-primary-soft p-3 font-semibold">Proposed: {METHOD_LABEL[form.cool_roofing_method]} · {QUALITY_LABEL[form.cool_roofing_quality]} quality</p>
            {pred.loading && <div className="mt-3"><LoadingBlock label="Preparing figures…" rows={2} /></div>}
            {!!pred.error && <div className="mt-3"><ErrorState error={pred.error} /></div>}
            {p && (
              <dl className="mt-3 grid gap-3 sm:grid-cols-3">
                <div className="rounded-xl bg-surface2 p-3"><dt className="text-xs text-muted">Estimated cost</dt><dd className="num text-xl font-semibold">{inr(p.cost.value)}</dd><dd className="text-xs text-muted">{inr(p.cost.interval.low)} – {inr(p.cost.interval.high)}</dd></div>
                <div className="rounded-xl bg-surface2 p-3"><dt className="text-xs text-muted">Roof cooling</dt><dd className="num text-xl font-semibold">−{num(p.temperature_drop.value)} °C</dd><dd className="text-xs text-muted">{num(p.temperature_drop.interval.low)} – {num(p.temperature_drop.interval.high)} °C</dd></div>
                <div className="rounded-xl bg-surface2 p-3"><dt className="text-xs text-muted">Electricity saved</dt><dd className="num text-xl font-semibold">{num(p.units_saved.value, 0)} kWh/mo</dd><dd className="text-xs text-muted">{num(p.units_saved.interval.low, 0)} – {num(p.units_saved.interval.high, 0)}</dd></div>
              </dl>
            )}
          </Card>
          <Alert tone="neutral" title="What is deliberately left out">Referral likelihood is internal and not printed. Satisfaction is shown only as the observed share of comparable past customers who rated 4★+. The PDF lists assumptions and states that results are estimates, not guarantees.</Alert>
          {pred.data?.warnings.map((w) => <Alert key={w} tone="warn" title="Will be disclosed in the proposal">{w}</Alert>)}
        </div>
      </div>
    </>
  );
}

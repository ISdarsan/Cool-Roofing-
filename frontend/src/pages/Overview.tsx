import { Link } from "react-router-dom";
import { ArrowRight, Calculator, Database, Gauge, Scale, ShieldCheck } from "lucide-react";
import { api } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { gridTable, methodTable, topDrivers } from "../lib/insights";
import { inr, inrCompact, num, pct } from "../lib/format";
import { FEATURE_LABEL, METHOD_LABEL, TASK_LABEL } from "../lib/labels";
import { useApp } from "../state/AppState";
import { Alert, Badge, Card, ErrorState, LoadingBlock, ReliabilityBadge, SectionTitle, useDocumentTitle } from "../components/ui";
import type { HeadlineTask } from "../lib/types";

function Kpi({ title, hl, children }: { title: string; hl: HeadlineTask; children: React.ReactNode }) {
  return (
    <Card className="p-4">
      <div className="flex items-start justify-between gap-2">
        <h3 className="font-semibold leading-tight">{title}</h3>
        <ReliabilityBadge r={hl.reliability} />
      </div>
      <div className="mt-3 space-y-1">{children}</div>
      <p className="mt-3 border-t border-line pt-2 text-xs text-muted">{hl.selected} · tested on {hl.n_test} unseen buildings</p>
    </Card>
  );
}
const Row = ({ k, v, sub }: { k: string; v: string; sub?: string }) => (
  <div className="flex items-baseline justify-between gap-3">
    <span className="text-sm text-muted">{k}</span>
    <span className="text-right"><span className="num text-lg font-semibold">{v}</span>{sub && <span className="block text-xs text-muted">{sub}</span>}</span>
  </div>
);

export default function Overview() {
  useDocumentTitle("Overview");
  const { status } = useApp();
  const dq = useAsync(() => api.dataQuality(), []);
  const ins = useAsync(() => api.insights(), []);

  const s = status.data;
  const h = s?.headline;
  const methods = ins.data ? methodTable(ins.data) : [];
  const grid = ins.data ? gridTable(ins.data) : [];
  const bestOut = methods.length ? methods.reduce((b, m) => (m.high + m.referral > b.high + b.referral ? m : b)) : null;
  const cheapest = methods.length ? methods.reduce((b, m) => (m.cost < b.cost ? m : b)) : null;
  const be = grid.length ? grid.reduce((b, g) => (g.breakeven10 < b.breakeven10 ? g : b)) : null;

  return (
    <>
      <section className="mb-8 rounded-card border border-line bg-surface p-6 shadow-card sm:p-8">
        <p className="text-sm font-semibold uppercase tracking-wider text-primary">Cool Roofing Intelligence</p>
        <h1 className="mt-1 max-w-3xl text-3xl font-bold tracking-tight sm:text-4xl">Know the cost, comfort and savings before the first quote.</h1>
        <p className="mt-3 max-w-2xl text-lg text-muted">
          Trained on {s ? s.data.n_rows.toLocaleString("en-IN") : "2,500"} past installations, this tool turns a few day-one questions into honest estimates — with likely ranges, reliability flags and the reasons behind them.
        </p>
        <div className="mt-5 flex flex-wrap gap-3">
          <Link to="/predict" className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-primary px-5 text-sm font-semibold text-on-primary hover:bg-primary-hover"><Calculator size={17} aria-hidden /> Predict for a customer <ArrowRight size={16} aria-hidden /></Link>
          <Link to="/compare" className="inline-flex min-h-11 items-center gap-2 rounded-xl border border-line px-5 text-sm font-semibold hover:bg-surface2"><Scale size={17} aria-hidden /> Compare roofing options</Link>
        </div>
      </section>

      <SectionTitle title="How well do the models work?" hint={<>Hold-out results on buildings the models never saw. These are the models’ <strong>typical errors</strong> — not customer savings.</>} />
      {status.loading && <LoadingBlock label="Loading model status…" rows={2} />}
      {!!status.error && <ErrorState error={status.error} onRetry={status.reload} />}
      {h && (
        <div className="mb-8 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          <Kpi title={TASK_LABEL.cost} hl={h.cost}>
            <Row k="Typical miss (MAE)" v={inr(h.cost.test.mae)} sub={`baseline ${inrCompact(h.cost.baseline.mae)}`} />
            <Row k="RMSE" v={inr(h.cost.test.rmse)} /><Row k="R²" v={h.cost.test.r2.toFixed(3)} />
          </Kpi>
          <Kpi title={TASK_LABEL.temperature_drop} hl={h.temperature_drop}>
            <Row k="Typical miss (MAE)" v={`${num(h.temperature_drop.test.mae, 2)} °C`} sub={`baseline ${num(h.temperature_drop.baseline.mae, 2)} °C`} />
            <Row k="RMSE" v={`${num(h.temperature_drop.test.rmse, 2)} °C`} /><Row k="R²" v={h.temperature_drop.test.r2.toFixed(3)} />
          </Kpi>
          <Kpi title={TASK_LABEL.units_saved} hl={h.units_saved}>
            <Row k="Typical miss (MAE)" v={`${num(h.units_saved.test.mae, 2)} kWh`} sub={`baseline ${num(h.units_saved.baseline.mae, 1)} kWh`} />
            <Row k="RMSE" v={`${num(h.units_saved.test.rmse, 2)} kWh`} /><Row k="R²" v={h.units_saved.test.r2.toFixed(3)} />
          </Kpi>
          <Kpi title="Customer rating (1–5)" hl={h.rating}>
            <Row k="Rating MAE" v={`${num(h.rating.test.mae, 2)} ★`} sub={`baseline ${num(h.rating.baseline.mae, 2)} ★`} />
            <Row k={`High-rating accuracy (≥${s?.high_rating_threshold})`} v={pct(h.high_rating.test.accuracy, 1)} sub={`best trivial guess ${pct(h.high_rating.best_trivial?.accuracy ?? 0, 1)}`} />
            <Row k="High-rating ROC-AUC" v={h.high_rating.test.roc_auc.toFixed(2)} sub={h.high_rating.significance?.f1 === "better" ? undefined : "F1 not distinguishable from trivial"} />
          </Kpi>
          <Kpi title="Customer referral" hl={h.referral}>
            <Row k="F1 (referral = positive)" v={h.referral.test.f1.toFixed(2)} sub={`${h.referral.significance?.f1 === "better" ? "better than" : "not distinguishable from"} ${(h.referral.best_trivial?.f1 ?? 0).toFixed(2)} (everyone refers)`} />
            <Row k="Accuracy at that cut-off" v={pct(h.referral.test.accuracy, 1)}
              sub={`${h.referral.significance?.accuracy === "worse" ? "significantly below" : h.referral.significance?.accuracy === "better" ? "above" : "not distinguishable from"} the majority-class ${pct(h.referral.best_trivial?.accuracy ?? 0, 1)}`} />
            <Row k="ROC-AUC" v={h.referral.test.roc_auc.toFixed(2)} />
          </Kpi>
          <Card className="flex flex-col justify-between bg-primary-soft p-4">
            <div><h3 className="flex items-center gap-2 font-semibold"><ShieldCheck size={18} aria-hidden /> Read this honestly</h3>
              <p className="mt-2 text-sm">Cost, cooling and savings are predicted precisely. Ratings and referrals depend on what happens <em>after</em> installation, so they are only weakly predictable at quote time — use them to prioritise, not to promise.</p></div>
            <Link to="/performance" className="mt-3 inline-flex items-center gap-1 text-sm font-semibold text-primary hover:underline">Full evaluation <ArrowRight size={15} aria-hidden /></Link>
          </Card>
        </div>
      )}

      <div className="mb-8 grid gap-6 lg:grid-cols-2">
        <Card className="p-5">
          <SectionTitle title="Dataset status" hint="Checked and cleaned before any modelling" />
          {dq.loading && <LoadingBlock rows={2} />}
          {!!dq.error && <ErrorState error={dq.error} onRetry={dq.reload} />}
          {dq.data && s && (() => {
            const sm = dq.data.summary;
            const z = sm.zero_consumption_placeholders, t = sm.temperature_sentinel_rows;
            return (
              <dl className="grid grid-cols-2 gap-3 text-sm">
                <div><dt className="text-muted">Records</dt><dd className="num text-xl font-semibold">{sm.n_raw_rows.toLocaleString("en-IN")}</dd></div>
                <div><dt className="text-muted">Train+val / hold-out buildings</dt><dd className="num text-xl font-semibold">{s.data.n_train_validation_buildings} / {s.data.n_test_buildings}</dd></div>
                <div><dt className="text-muted">120 °C placeholder temps</dt><dd className="num text-xl font-semibold">{t.before + t.after}</dd></div>
                <div><dt className="text-muted">0 kWh placeholder readings</dt><dd className="num text-xl font-semibold">{z.before + z.after}</dd></div>
                <div><dt className="text-muted">Adverse outcomes kept</dt><dd className="num text-xl font-semibold">{sm.negative_savings_rows}</dd></div>
                <div><dt className="text-muted">Rows with a missing input</dt><dd className="num text-xl font-semibold">{sm.rows_with_any_missing_feature}</dd></div>
              </dl>
            );
          })()}
          <Link to="/insights" className="mt-4 inline-flex items-center gap-1 text-sm font-semibold text-primary hover:underline"><Database size={15} aria-hidden /> Data-quality details <ArrowRight size={15} aria-hidden /></Link>
        </Card>

        <Card className="p-5">
          <SectionTitle title="What the data says" hint="Computed from the trained models on past buildings (associations, not causation)" />
          {ins.loading && <LoadingBlock rows={3} />}
          {!!ins.error && <ErrorState error={ins.error} onRetry={ins.reload} />}
          {ins.data && bestOut && cheapest && (
            <ul className="space-y-3 text-sm">
              <li><Badge tone="primary">Method</Badge>{" "}On concrete roofs, <strong>{METHOD_LABEL[bestOut.method]}</strong> has the best predicted satisfaction ({pct(bestOut.high)} high-rating chance, {pct(bestOut.referral)} referral) at about {inrCompact(bestOut.cost)}; <strong>{METHOD_LABEL[cheapest.method]}</strong> is cheapest ({inrCompact(cheapest.cost)}) with the weakest outcomes.</li>
              <li><Badge tone="primary">Drivers</Badge>{" "}Cost: {topDrivers(ins.data, "cost").map((d) => `${FEATURE_LABEL[d.feature]} ${pct(d.share)}`).join(", ")}. Savings: {topDrivers(ins.data, "units_saved").map((d) => `${FEATURE_LABEL[d.feature]} ${pct(d.share)}`).join(", ")}.</li>
              {be && <li><Badge tone="warn">Economics</Badge>{" "}Electricity savings alone repay the roof in 10 years only if the tariff is at least about <strong className="num">₹{num(be.breakeven10, 0)}/kWh</strong> even for the best-value option — compare with the customer’s real tariff.</li>}
            </ul>
          )}
          <Link to="/insights" className="mt-4 inline-flex items-center gap-1 text-sm font-semibold text-primary hover:underline"><Gauge size={15} aria-hidden /> All insights <ArrowRight size={15} aria-hidden /></Link>
        </Card>
      </div>

      <Alert tone="neutral" title="Assumptions in force">“High rating” means a customer rating of {s?.high_rating_threshold ?? 4} or more (the organisers gave no definition). Electricity is monthly kWh. Building size uses the dataset’s unstated area unit.</Alert>
    </>
  );
}

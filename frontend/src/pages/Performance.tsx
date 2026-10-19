import { useState } from "react";
import { CheckCircle2, XCircle } from "lucide-react";
import { api, figureUrl } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { inr, num, pct } from "../lib/format";
import { TASK_LABEL, TASK_ORDER } from "../lib/labels";
import type { Json, TaskKey } from "../lib/types";
import { Alert, Badge, Card, ErrorState, LoadingBlock, PageHeader, SectionTitle, Tabs, useDocumentTitle } from "../components/ui";
import { ChartCard, CategoryBars, ConfusionMatrix, LineSeries } from "../components/charts";
import { ReportViewer } from "../components/Markdown";
import { useApp } from "../state/AppState";

const REG_FMT: Record<string, (v: number) => string> = {
  cost: (v) => inr(v), temperature_drop: (v) => `${num(v, 3)} °C`, units_saved: (v) => `${num(v, 2)} kWh`, rating: (v) => `${num(v, 3)} ★`,
};

function VerdictBadge({ v, what }: { v?: string; what: string }) {
  if (v === "better") return <Badge tone="good"><CheckCircle2 size={13} aria-hidden /> significantly better than {what}</Badge>;
  if (v === "worse") return <Badge tone="bad"><XCircle size={13} aria-hidden /> significantly worse than {what}</Badge>;
  return <Badge tone="warn">not distinguishable from {what}</Badge>;
}

function Metric({ label, value, base, hint }: { label: string; value: string; base?: string; hint?: string }) {
  return (
    <div className="rounded-xl bg-surface2 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted">{label}</p>
      <p className="num text-xl font-semibold" data-testid={`m-${label}`}>{value}</p>
      {base && <p className="text-xs text-muted">{base}</p>}
      {hint && <p className="text-xs text-muted">{hint}</p>}
    </div>
  );
}

export default function Performance() {
  useDocumentTitle("Model performance");
  const [task, setTask] = useState<TaskKey>("cost");
  const models = useAsync(() => api.models(), []);
  const { status } = useApp();

  if (models.loading) return <><PageHeader title="Model performance" /><LoadingBlock rows={4} /></>;
  if (models.error || !models.data) return <><PageHeader title="Model performance" /><ErrorState error={models.error} onRetry={models.reload} /></>;

  const m = models.data;
  const t: Json = m.tasks[task];
  const cands: Json[] = m.candidates[task];
  const reg = t.kind === "regression";
  const f = REG_FMT[task];
  const data = m.meta.data;

  return (
    <>
      <PageHeader title="Model performance" subtitle="Every figure on this page was computed by the pipeline on real hold-out predictions. Nothing is typed in." />

      <Card className="mb-6 grid gap-4 p-5 md:grid-cols-3">
        <div><p className="text-xs font-semibold uppercase tracking-wide text-muted">Hold-out test set</p><p className="num text-2xl font-semibold">{data.n_test_buildings} buildings</p><p className="text-sm text-muted">Reserved before any modelling; each model is scored on it once.</p></div>
        <div><p className="text-xs font-semibold uppercase tracking-wide text-muted">Train + validation pool</p><p className="num text-2xl font-semibold">{data.n_train_validation_buildings.toLocaleString("en-IN")} buildings</p><p className="text-sm text-muted">{m.meta.cv_folds}-fold cross-validation for tuning and model selection.</p></div>
        <div><p className="text-xs font-semibold uppercase tracking-wide text-muted">Leakage control</p><p className="text-sm">Only facts known before installation are inputs. Cost, after-measurements, rating, referral and IDs are excluded and enforced by automated tests.</p></div>
      </Card>

      <Tabs label="Prediction task" value={task} onChange={setTask} tabs={TASK_ORDER.map((k) => ({ value: k, label: TASK_LABEL[k].replace(" probability", "").replace("Monthly electricity saved", "Electricity saved") }))} />

      <div id={`panel-${task}`} role="tabpanel" aria-labelledby={`tab-${task}`} className="mt-5 space-y-5">
        <Card className="p-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold">{t.label}</h2>
              <p className="text-sm text-muted">{t.target_definition} · unit: <strong>{t.unit}</strong></p>
              <p className="mt-1 text-sm">Selected: <strong>{t.selected_label}</strong> · baseline: {t.baseline_label}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge tone="neutral">{t.n_train_validation} train+val</Badge><Badge tone="neutral">{t.n_test} hold-out</Badge>
              {reg ? <VerdictBadge v={t.significance?.verdict} what="baseline MAE" /> : (<>
                <VerdictBadge v={t.significance?.accuracy_verdict} what="majority-class accuracy" />
                <VerdictBadge v={t.significance?.f1_verdict} what="trivial F1" />
                <Badge tone={t.significance?.auc_above_chance ? "good" : "bad"}>ROC-AUC {t.significance?.auc_above_chance ? "above" : "not above"} chance</Badge></>)}
            </div>
          </div>

          {reg ? (
            <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metric label="MAE (hold-out)" value={f(t.test.mae)} base={`baseline ${f(t.test_baseline.mae)}`} hint={`validation ${f(t.validation.mae)}`} />
              <Metric label="RMSE (hold-out)" value={f(t.test.rmse)} base={`baseline ${f(t.test_baseline.rmse)}`} hint={`validation ${f(t.validation.rmse)}`} />
              <Metric label="R²" value={t.test.r2.toFixed(3)} hint="share of variation explained" />
              <Metric label="Bias" value={num(t.test.bias, 3)} hint="average over/under-prediction" />
              {task === "rating" && (
                <>
                  <Metric label="MAE if rounded to whole stars" value={num(t.test.mae_rounded, 3)} />
                  <Metric label="Exact-star accuracy" value={pct(t.test.exact_match_accuracy_rounded, 1)} />
                  <Metric label="Within one star" value={pct(t.test.within_one_star, 1)} />
                  <Metric label={`High/not-high accuracy (≥${m.meta.high_rating_threshold}), derived`} value={pct(t.test.derived_high_rating_accuracy, 1)} hint="from this numeric prediction" />
                </>
              )}
            </div>
          ) : (
            <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metric label="Accuracy" value={pct(t.test.accuracy, 1)} base={`trivial best ${pct(t.baseline_accuracy, 1)}`} hint={`validation ${pct(t.validation.accuracy, 1)}`} />
              <Metric label="F1 (binary)" value={t.test.f1.toFixed(3)} base={`trivial best ${t.baseline_f1.toFixed(3)}`} hint={`positive class = ${task === "referral" ? "referred" : "high rating"}`} />
              <Metric label="Precision" value={t.test.precision.toFixed(3)} hint="flagged who truly are" />
              <Metric label="Recall" value={t.test.recall.toFixed(3)} hint="true cases we catch" />
              <Metric label="ROC-AUC" value={t.test.roc_auc.toFixed(3)} hint="0.5 = chance" />
              <Metric label="Brier score" value={t.test.brier.toFixed(3)} hint="lower is better" />
              <Metric label="Cut-off used" value={pct(t.threshold, 0)} hint={t.threshold_policy} />
              <Metric label="Class balance" value={pct(t.test.positive_rate_true, 1)} hint={`${t.test.support.positive} positive / ${t.test.support.negative} negative`} />
            </div>
          )}
        </Card>

        {task === "referral" && (
          <Alert tone="warn" title="Accuracy and F1 pull in opposite directions here">
            The F1-optimal cut-off ({pct(t.threshold)}) flags most customers: F1 is {t.test.f1.toFixed(2)} (95% CI {t.significance.f1_ci95[0].toFixed(2)}–{t.significance.f1_ci95[1].toFixed(2)}), statistically indistinguishable from the always-flag baseline ({t.baseline_f1.toFixed(2)}), while accuracy is {pct(t.test.accuracy, 1)} against {pct(t.baseline_accuracy, 1)} for always predicting “no referral”.
            At the default 50% cut-off accuracy is {pct(t.test_at_0_5.accuracy, 1)} but F1 drops to {t.test_at_0_5.f1.toFixed(2)}. The model ranks customers only weakly (ROC-AUC {t.test.roc_auc.toFixed(2)}), so use it for follow-up priority, not as a forecast.
          </Alert>
        )}

        <div className="grid gap-5 xl:grid-cols-2">
          <ChartCard title="Candidate models (validation)" subtitle={reg ? "Cross-validated MAE, lower is better" : task === "referral" ? "Cross-validated best F1, higher is better" : "Cross-validated accuracy, higher is better"}
            summary={`Validation comparison of ${cands.length} candidate models; selected: ${t.selected_label}.`}
            table={{ headers: ["Model", "Family", ...(reg ? ["CV MAE", "CV RMSE", "CV R²"] : ["CV accuracy", "CV best F1", "CV log-loss"])],
              rows: cands.map((c) => [c.label, c.family, ...(reg ? [num(c.cv_mae, 3), num(c.cv_rmse, 3), num(c.cv_r2, 3)] : [num(c.cv_accuracy, 3), num(c.cv_best_f1, 3), num(c.cv_log_loss, 3)])]) }}
            height={Math.max(240, cands.length * 44)}>
            <CategoryBars layout="horizontal" fmt={reg ? (v) => (task === "cost" ? inr(v) : num(v, 3)) : (v) => v.toFixed(3)} domain={reg ? [0, "auto"] : [0, 1]}
              data={[...cands].sort((a, b) => (reg ? a.cv_mae - b.cv_mae : b.select_score === a.select_score ? 0 : a.select_score - b.select_score)).map((c) => ({
                name: `${c.label}${c.model === t.selected ? " ✓" : ""}`,
                value: reg ? c.cv_mae : task === "referral" ? c.cv_best_f1 : c.cv_accuracy,
                color: c.model === t.selected ? "var(--c1)" : c.family === "baseline" ? "var(--c3)" : "var(--c4)",
              }))} />
          </ChartCard>

          {reg ? (
            <Card className="p-4">
              <h3 className="font-semibold">Predicted vs actual, residuals</h3>
              <p className="text-sm text-muted">Diagnostics on the hold-out buildings (generated by the evaluation pipeline).</p>
              <img src={figureUrl(`10_diagnostics_${task}.png`)} loading="lazy" className="mt-2 w-full rounded-xl border border-line bg-white"
                alt={`Predicted versus actual values, residual histogram and residuals versus prediction for ${t.label} on the hold-out set. R squared ${t.test.r2.toFixed(2)}.`} />
            </Card>
          ) : (
            <Card className="p-4">
              <h3 className="font-semibold">Confusion matrix (hold-out)</h3>
              <p className="mb-3 text-sm text-muted">Cut-off {pct(t.threshold)} · F1 positive class = {task === "referral" ? "referred" : "high rating"}</p>
              <ConfusionMatrix {...t.test.confusion_matrix} negative={task === "referral" ? "no referral" : "not high"} positive={task === "referral" ? "referral" : "high rating"} />
              <p className="mt-3 text-xs text-muted">Precision {t.test.precision.toFixed(2)} · recall {t.test.recall.toFixed(2)} · macro-F1 {t.test.macro_f1.toFixed(2)}</p>
            </Card>
          )}
        </div>

        {!reg && (
          <div className="grid gap-5 xl:grid-cols-2">
            <ChartCard title="Cut-off trade-off (hold-out)" subtitle="Raising the cut-off trades recall for precision; accuracy and F1 peak at different places."
              summary="Accuracy, F1, precision and recall as the decision cut-off changes."
              table={{ headers: ["Cut-off", "Accuracy", "F1", "Precision", "Recall"], rows: t.operating_points.test.map((r: Json) => [r.threshold.toFixed(2), r.accuracy.toFixed(3), r.f1.toFixed(3), r.precision.toFixed(3), r.recall.toFixed(3)]) }}>
              <LineSeries data={t.operating_points.test} xKey="threshold" xLabel="decision cut-off" xFmt={(v) => v.toFixed(2)} yFmt={(v) => v.toFixed(2)} refX={t.threshold} yDomain={[0, 1]}
                lines={[{ key: "accuracy", label: "Accuracy", color: "var(--c1)" }, { key: "f1", label: "F1", color: "var(--c2)" }, { key: "precision", label: "Precision", color: "var(--c4)", dashed: true }, { key: "recall", label: "Recall", color: "var(--c3)", dashed: true }]} />
            </ChartCard>
            <ChartCard title="Calibration (hold-out)" subtitle={`Do predicted probabilities match reality? ECE ${t.calibration_test.ece.toFixed(3)} (lower is better).`}
              summary="Observed rate versus mean predicted probability across probability bins; the diagonal is perfect calibration."
              table={{ headers: ["Mean predicted", "Observed rate", "n"], rows: t.calibration_test.bins.map((b: Json) => [b.mean_predicted.toFixed(3), b.observed_rate.toFixed(3), b.n]) }}>
              <LineSeries data={[{ x: 0, perfect: 0, observed: NaN }, ...t.calibration_test.bins.map((b: Json) => ({ x: b.mean_predicted, perfect: b.mean_predicted, observed: b.observed_rate })), { x: 1, perfect: 1, observed: NaN }].sort((a, b) => a.x - b.x) as Record<string, number>[]}
                xKey="x" xLabel="mean predicted probability" xFmt={(v) => v.toFixed(1)} yFmt={(v) => v.toFixed(2)} yDomain={[0, 1]}
                lines={[{ key: "perfect", label: "Perfect calibration", color: "var(--muted)", dashed: true }, { key: "observed", label: "Observed", color: "var(--c1)" }]} />
            </ChartCard>
          </div>
        )}

        {reg && (
          <Card className="p-5">
            <SectionTitle title="Reliability checks" />
            <ul className="grid gap-2 text-sm md:grid-cols-2">
              <li className="rounded-xl bg-surface2 p-3">Likely-range intervals (from out-of-fold errors) cover <strong>{pct(t.interval_coverage_test["0.8"])}</strong> (nominal 80%) and <strong>{pct(t.interval_coverage_test["0.9"])}</strong> (nominal 90%) of hold-out buildings.</li>
              <li className="rounded-xl bg-surface2 p-3">Physical plausibility of hold-out predictions: {Object.entries(t.plausibility_test).filter(([k]) => k !== "n").map(([k, v]) => `${k.replace(/^n_/, "").replace(/_/g, " ")} = ${v}`).join(", ")}.</li>
            </ul>
          </Card>
        )}

        <Card className="p-5">
          <SectionTitle title="Limitations" />
          <ul className="list-disc space-y-1 pl-5 text-sm text-muted">
            <li>One random 80/20 split of {status.data?.data.n_rows.toLocaleString("en-IN") ?? ""} buildings: hold-out numbers carry sampling error, and no time or location field exists to test drift.</li>
            <li>Models capture associations in historical installations, not the causal effect of choosing a method.</li>
            {!reg && <li>Rating and referral depend on post-installation experience that is not available at quote time.</li>}
            <li>Inputs outside the training ranges are flagged as extrapolations by the API.</li>
          </ul>
        </Card>
        <ReportViewer name="model_evaluation" title="evaluation report" />
      </div>
    </>
  );
}

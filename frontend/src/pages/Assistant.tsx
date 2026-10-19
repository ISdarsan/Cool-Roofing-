import { useEffect, useRef, useState } from "react";
import { Bot, KeyRound, Send, ShieldCheck, User } from "lucide-react";
import { api } from "../lib/api";
import { inr, inrCompact, num, pct, years } from "../lib/format";
import { METHOD_LABEL, QUALITY_LABEL } from "../lib/labels";
import type { ChatToolResult, Json } from "../lib/types";
import { useApp } from "../state/AppState";
import { Alert, Badge, Button, Card, ErrorState, LoadingBlock, PageHeader, cx, useDocumentTitle } from "../components/ui";
import { Markdown } from "../components/Markdown";

interface Msg { role: "user" | "assistant"; content: string; tools?: ChatToolResult[]; provider?: string; notice?: string }

const SUGGESTIONS = [
  "A customer in an urban area has a flat concrete roof of 1,800 area units, 3 AC units and about 190 kWh a month. Which option do you recommend?",
  "Compare the cheapest and the best-cooling options for a 1,200 unit tiled house in a rural area.",
];

function ToolCard({ t }: { t: ChatToolResult }) {
  const o: Json = t.output;
  if (o?.error) return <Alert tone="warn" title={`${t.tool}: could not run`}>{o.error}</Alert>;
  if (t.tool === "predict_outcomes") {
    const p = o.predictions;
    return (
      <div className="rounded-xl border border-line bg-surface p-3 text-sm">
        <p className="mb-1 flex items-center gap-1 font-semibold"><ShieldCheck size={14} aria-hidden /> Prediction service · {METHOD_LABEL[t.input.cool_roofing_method as keyof typeof METHOD_LABEL]} · {QUALITY_LABEL[t.input.cool_roofing_quality as keyof typeof QUALITY_LABEL]}</p>
        <p className="num">{inr(p.cost.value)} · −{num(p.temperature_drop.value)} °C · {num(p.units_saved.value, 0)} kWh/mo · P(high) {pct(p.high_rating.probability)} · P(referral) {pct(p.referral.probability)}</p>
      </div>
    );
  }
  if (t.tool === "compare_options") {
    return (
      <div className="rounded-xl border border-line bg-surface p-3 text-sm">
        <p className="mb-1 flex items-center gap-1 font-semibold"><ShieldCheck size={14} aria-hidden /> Option comparison · {o.objective}</p>
        <ol className="space-y-0.5">{(o.options as Json[]).slice(0, 4).map((x) => (
          <li key={x.method + x.quality} className="num">{x.rank}. {METHOD_LABEL[x.method as keyof typeof METHOD_LABEL]} · {QUALITY_LABEL[x.quality as keyof typeof QUALITY_LABEL]} — {inrCompact(x.cost)}, −{num(x.temperature_drop)} °C, {num(x.units_saved, 0)} kWh</li>))}</ol>
      </div>
    );
  }
  if (t.tool === "calculate_payback") {
    return <div className="rounded-xl border border-line bg-surface p-3 text-sm"><p className="font-semibold">Payback (assumption-based)</p><p className="num">{years(o.central.simple_payback_years)} · annual saving {inr(o.central.annual_bill_saving_inr)}</p></div>;
  }
  if (t.tool === "explain_prediction") {
    return <div className="rounded-xl border border-line bg-surface p-3 text-sm"><p className="font-semibold">Explanation · {o.task}</p><p className="num">{(o.contributions as Json[]).slice(0, 3).map((c) => `${c.feature} ${c.contribution >= 0 ? "+" : ""}${num(c.contribution, 2)}`).join(" · ")}</p></div>;
  }
  return null;
}

export default function Assistant() {
  useDocumentTitle("AI assistant");
  const { status } = useApp();
  const a = status.data?.assistant;
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView?.({ block: "end" }); }, [msgs, busy]);

  async function send(content: string) {
    const c = content.trim();
    if (!c || busy) return;
    const next: Msg[] = [...msgs, { role: "user", content: c }];
    setMsgs(next); setText(""); setBusy(true); setError(null);
    try {
      const r = await api.assistantChat(next.map(({ role, content: ct }) => ({ role, content: ct })));
      setMsgs([...next, { role: "assistant", content: r.reply || "(no reply)", tools: r.tool_results, provider: r.provider, notice: r.notice }]);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader title="AI sales assistant" subtitle="Optional. Describe the customer in plain words; the assistant asks for anything missing and calls the same prediction service as the rest of the app. It never makes up numbers." />
      {status.loading && <LoadingBlock rows={2} />}
      {a && !a.enabled && (
        <Card className="p-6">
          <div className="flex items-start gap-3">
            <KeyRound className="mt-1 shrink-0 text-primary" size={22} aria-hidden />
            <div>
              <h2 className="text-lg font-semibold">The assistant is not available</h2>
              <p className="mt-1 text-muted">{a.message}</p>
              <ol className="mt-3 list-decimal space-y-1 pl-5 text-sm">
                <li>Copy <code className="num rounded bg-surface2 px-1">.env.example</code> to <code className="num rounded bg-surface2 px-1">.env</code>.</li>
                <li>Choose a free option: <code className="num rounded bg-surface2 px-1">COOLROOF_LLM_PROVIDER=builtin</code> (no key), or <code className="num rounded bg-surface2 px-1">gemini</code> / <code className="num rounded bg-surface2 px-1">groq</code> with the key variable named in the file (never commit keys).</li>
                <li>Restart the backend: <code className="num rounded bg-surface2 px-1">python -m coolroof serve</code>.</li>
              </ol>
              <p className="mt-3 text-sm text-muted">Everything else — predictions, comparison, explanations, proposals — works without it.</p>
            </div>
          </div>
        </Card>
      )}
      {a?.enabled && (
        <div className="grid gap-4">
          <Alert tone={a.provider === "builtin" ? "good" : "neutral"} title={a.provider === "builtin" ? "Built-in assistant (free, offline, no account)" : `Provider: ${a.provider}${a.model ? ` · ${a.model}` : ""}${a.free ? " (free tier)" : ""}`}>
            {a.provider === "builtin"
              ? "It understands plain-language building descriptions with fixed patterns, asks for what is missing and uses the same prediction tools. For free conversational AI, add a Gemini, Groq or Ollama setting in .env."
              : a.message ?? "Every number still comes from the prediction tools."}
          </Alert>
          <Card className="flex min-h-[420px] flex-col p-4">
            <div className="flex-1 space-y-4 overflow-y-auto" role="log" aria-live="polite" aria-label="Conversation" style={{ maxHeight: "60vh" }}>
              {msgs.length === 0 && (
                <div className="space-y-2">
                  <p className="text-sm text-muted">Try one of these, or describe your own customer:</p>
                  {SUGGESTIONS.map((s) => <button key={s} type="button" onClick={() => send(s)} className="block w-full rounded-xl border border-line p-3 text-left text-sm hover:bg-surface2">{s}</button>)}
                </div>
              )}
              {msgs.map((m, i) => (
                <div key={i} className={cx("flex gap-2", m.role === "user" && "flex-row-reverse")}>
                  <span className={cx("mt-1 grid h-8 w-8 shrink-0 place-items-center rounded-full", m.role === "user" ? "bg-sun text-on-sun" : "bg-primary-soft text-primary")} aria-hidden>{m.role === "user" ? <User size={16} /> : <Bot size={16} />}</span>
                  <div className={cx("max-w-[85%] space-y-2 rounded-2xl px-4 py-2.5 text-[15px]", m.role === "user" ? "bg-primary text-on-primary" : "bg-surface2")}>
                    {m.role === "assistant" ? <div className="text-[15px] [&_.prose-report]:text-[15px] [&_ul]:my-1 [&_p]:my-1"><Markdown text={m.content} /></div> : <p className="whitespace-pre-wrap">{m.content}</p>}
                    {m.tools?.map((t, j) => <ToolCard key={j} t={t} />)}
                    {m.notice && <Alert tone="warn">{m.notice}</Alert>}
                    {m.tools && m.tools.length > 0 && <Badge tone="primary">Figures come from the prediction service{m.provider ? ` · answered by ${m.provider === "builtin" ? "the built-in assistant" : m.provider}` : ""}</Badge>}
                  </div>
                </div>
              ))}
              {busy && <LoadingBlock label="Assistant is working…" rows={1} />}
              <div ref={end} />
            </div>
            {!!error && <div className="mt-3"><ErrorState error={error} title="The assistant could not answer" /></div>}
            <form className="mt-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); void send(text); }}>
              <label htmlFor="chat" className="sr-only">Message to the assistant</label>
              <textarea id="chat" value={text} onChange={(e) => setText(e.target.value)} rows={2} maxLength={4000} placeholder="Describe the customer’s building…"
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(text); } }}
                className="min-h-11 flex-1 resize-none rounded-xl border border-line bg-surface px-3 py-2" />
              <Button type="submit" loading={busy} aria-label="Send message"><Send size={16} aria-hidden /> Send</Button>
            </form>
          </Card>
          <Alert tone="neutral" title="Guard-rails">Every number comes from a validated tool call into the real models. The assistant is told never to guarantee savings or payback, to ask rather than assume, and cannot run code.</Alert>
        </div>
      )}
    </>
  );
}

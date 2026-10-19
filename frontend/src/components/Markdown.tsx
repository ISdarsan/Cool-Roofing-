import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText } from "lucide-react";
import { api, figureUrl } from "../lib/api";
import { ErrorState, LoadingBlock, Button } from "./ui";

export function Markdown({ text }: { text: string }) {
  return (
    <div className="prose-report text-[15px]">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        urlTransform={(url) => (url.startsWith("figures/") ? figureUrl(url.slice("figures/".length)) : url)}
        components={{
          table: ({ children }) => <div className="table-wrap"><table>{children}</table></div>,
          img: ({ src, alt }) => <img src={src} alt={alt ?? ""} loading="lazy" />,
        }}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

/** Loads a generated report on demand (nothing is fetched until the user asks). */
export function ReportViewer({ name, title }: { name: string; title: string }) {
  const [state, setState] = useState<{ open: boolean; loading: boolean; error: unknown; md: string | null }>({ open: false, loading: false, error: null, md: null });
  async function open() {
    setState({ open: true, loading: true, error: null, md: null });
    try {
      const r = await api.report(name);
      setState({ open: true, loading: false, error: null, md: r.markdown });
    } catch (e) {
      setState({ open: true, loading: false, error: e, md: null });
    }
  }
  if (!state.open) return <Button variant="secondary" onClick={open}><FileText size={16} aria-hidden /> Read the full {title}</Button>;
  return (
    <div className="rounded-card border border-line bg-surface p-5">
      {state.loading && <LoadingBlock label={`Loading ${title}…`} rows={4} />}
      {!!state.error && <ErrorState error={state.error} onRetry={open} />}
      {state.md && <Markdown text={state.md} />}
    </div>
  );
}

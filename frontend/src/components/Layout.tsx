import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { motion } from "motion/react";
import {
  Bot, Calculator, Database, FileText, Gauge, LayoutDashboard, Menu, Monitor, Moon, Scale, SlidersHorizontal, Sun, X,
} from "lucide-react";
import { useApp } from "../state/AppState";
import { dateShort } from "../lib/format";
import { cx } from "./ui";

export const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/predict", label: "Customer prediction", icon: Calculator },
  { to: "/compare", label: "Roofing comparison", icon: Scale },
  { to: "/simulate", label: "What-if simulator", icon: SlidersHorizontal },
  { to: "/performance", label: "Model performance", icon: Gauge },
  { to: "/insights", label: "Data quality & insights", icon: Database },
  { to: "/proposal", label: "Proposal", icon: FileText },
  { to: "/assistant", label: "AI assistant", icon: Bot },
];

export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <svg width="34" height="34" viewBox="0 0 32 32" aria-hidden>
        <rect width="32" height="32" rx="9" fill="var(--primary)" />
        <path d="M5 17 16 7l11 10" fill="none" stroke="var(--on-primary)" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M9 16v9h14v-9" fill="none" stroke="var(--on-primary)" strokeWidth="2.6" strokeLinejoin="round" />
        <circle cx="25" cy="7" r="3.2" fill="var(--sun)" />
      </svg>
      {!compact && (
        <div className="leading-tight">
          <p className="text-[15px] font-bold tracking-tight">Cool Roofing</p>
          <p className="text-xs font-medium text-muted">Intelligence</p>
        </div>
      )}
    </div>
  );
}

function NavList({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Main">
      <ul className="space-y-1">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <li key={to}>
            <NavLink to={to} end={end} onClick={onNavigate}
              className={({ isActive }) => cx(
                "flex min-h-11 items-center gap-3 rounded-xl px-3 py-2 text-[15px] font-medium transition-colors duration-200",
                isActive ? "bg-primary-soft font-semibold text-primary" : "text-ink/80 hover:bg-surface2 hover:text-ink")}>
              <Icon size={19} aria-hidden />
              <span>{label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}

function ThemeToggle() {
  const { theme, setTheme } = useApp();
  const order = ["system", "light", "dark"] as const;
  const next = order[(order.indexOf(theme) + 1) % 3];
  const Icon = theme === "dark" ? Moon : theme === "light" ? Sun : Monitor;
  return (
    <button type="button" onClick={() => setTheme(next)}
      className="inline-flex min-h-11 min-w-11 items-center justify-center gap-2 rounded-xl border border-line bg-surface px-3 text-sm font-medium text-ink hover:bg-surface2"
      aria-label={`Theme: ${theme}. Switch to ${next}`}>
      <Icon size={17} aria-hidden /> <span className="hidden sm:inline capitalize">{theme}</span>
    </button>
  );
}

function StatusChip() {
  const { status } = useApp();
  if (status.loading) return <span className="text-xs text-muted" role="status">Checking models…</span>;
  if (status.error || !status.data)
    return <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-clay" role="status"><span className="h-2 w-2 rounded-full bg-clay" aria-hidden />Models offline</span>;
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted" role="status">
      <span className="h-2 w-2 rounded-full bg-good" aria-hidden />
      <span><span className="font-semibold text-ink">Models ready</span> · v{status.data.model_version} · trained {dateShort(status.data.trained_at_utc)}</span>
    </span>
  );
}

export default function Layout() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  const menuBtn = useRef<HTMLButtonElement>(null);
  const drawer = useRef<HTMLDivElement>(null);

  useEffect(() => { setOpen(false); window.scrollTo?.(0, 0); }, [loc.pathname]);
  useEffect(() => {
    if (!open) return;
    const first = drawer.current?.querySelector<HTMLElement>("a,button");
    first?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setOpen(false); menuBtn.current?.focus(); }
      if (e.key === "Tab" && drawer.current) {
        const f = Array.from(drawer.current.querySelectorAll<HTMLElement>("a,button")).filter((x) => !x.hasAttribute("disabled"));
        if (!f.length) return;
        const a = f[0], z = f[f.length - 1];
        if (e.shiftKey && document.activeElement === a) { e.preventDefault(); z.focus(); }
        else if (!e.shiftKey && document.activeElement === z) { e.preventDefault(); a.focus(); }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[272px_minmax(0,1fr)]">
      <a href="#main" className="skip-link">Skip to content</a>

      <aside className="sticky top-0 hidden h-screen flex-col gap-6 border-r border-line bg-surface/80 p-5 backdrop-blur lg:flex">
        <Brand />
        <NavList />
        <div className="mt-auto space-y-3 border-t border-line pt-4 text-xs text-muted">
          <StatusChip />
          <p>Estimates describe historical installations, not guarantees.</p>
        </div>
      </aside>

      <div className="min-w-0">
        <header className="sticky top-0 z-30 flex items-center justify-between gap-2 border-b border-line bg-surface/90 px-4 py-2 backdrop-blur lg:hidden">
          <button ref={menuBtn} type="button" onClick={() => setOpen(true)} aria-label="Open navigation menu" aria-expanded={open}
            className="inline-flex min-h-11 min-w-11 items-center justify-center rounded-xl border border-line"><Menu size={20} aria-hidden /></button>
          <Brand />
          <ThemeToggle />
        </header>

        {open && (
          <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation menu">
            <div className="absolute inset-0 bg-ink/40" onClick={() => { setOpen(false); menuBtn.current?.focus(); }} aria-hidden />
            <div ref={drawer} className="absolute left-0 top-0 flex h-full w-[84%] max-w-sm flex-col gap-5 bg-surface p-5 shadow-2xl">
              <div className="flex items-center justify-between">
                <Brand />
                <button type="button" onClick={() => { setOpen(false); menuBtn.current?.focus(); }} aria-label="Close navigation menu"
                  className="inline-flex min-h-11 min-w-11 items-center justify-center rounded-xl border border-line"><X size={20} aria-hidden /></button>
              </div>
              <NavList onNavigate={() => setOpen(false)} />
              <div className="mt-auto border-t border-line pt-4"><StatusChip /></div>
            </div>
          </div>
        )}

        <div className="hidden items-center justify-end gap-3 px-10 pt-5 lg:flex"><ThemeToggle /></div>
        <main id="main" tabIndex={-1} className="mx-auto w-full max-w-[1200px] px-4 pb-16 pt-5 sm:px-6 lg:px-10 lg:pt-3">
          <motion.div key={loc.pathname} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, ease: "easeOut" }}>
            <Outlet />
          </motion.div>
        </main>
      </div>
    </div>
  );
}

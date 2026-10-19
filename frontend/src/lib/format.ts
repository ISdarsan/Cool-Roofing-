export const inr = (v: number, digits = 0) =>
  `₹${v.toLocaleString("en-IN", { maximumFractionDigits: digits, minimumFractionDigits: digits })}`;

/** ₹15.7 L / ₹1.24 Cr style for tight spaces (lakh = 1e5, crore = 1e7). */
export function inrCompact(v: number): string {
  const a = Math.abs(v);
  const s = v < 0 ? "-" : "";
  if (a >= 1e7) return `${s}₹${(a / 1e7).toFixed(2)} Cr`;
  if (a >= 1e5) return `${s}₹${(a / 1e5).toFixed(a >= 1e6 ? 1 : 2)} L`;
  if (a >= 1e3) return `${s}₹${(a / 1e3).toFixed(1)} k`;
  return `${s}₹${a.toFixed(0)}`;
}
export const num = (v: number, d = 1) => v.toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d });
export const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`;
export const signed = (v: number, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
export const titleCase = (s: string) => s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
export const years = (y: number | null | undefined) =>
  y == null ? "no payback" : y > 99 ? ">99 years" : `${y.toFixed(1)} years`;
export const dateShort = (iso: string) =>
  new Date(iso).toLocaleDateString("en-IN", { year: "numeric", month: "short", day: "numeric" });

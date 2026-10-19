import { METHOD_ORDER } from "./labels";
import type { Json, Method } from "./types";

export interface MethodRow { method: Method; cost: number; drop: number; kwh: number; high: number; referral: number }

/** Counterfactual method comparison on concrete roofs, straight from insights.json. */
export function methodTable(ins: Json): MethodRow[] {
  const me = ins?.marginal_effects?.method;
  if (!me || !Array.isArray(me.cost) || !me.cost.length) return [];
  const get = (k: string, m: string): number => me[k]?.find?.((r: Json) => r.level === m)?.mean ?? NaN;
  return METHOD_ORDER.map((m) => ({
    method: m, cost: get("cost", m), drop: get("temperature_drop", m), kwh: get("units_saved", m),
    high: get("high_rating", m), referral: get("referral", m),
  }));
}

export interface GridRow { method: Method; quality: string; cost: number; drop: number; kwh: number; high: number; referral: number; breakeven10: number }
export function gridTable(ins: Json): GridRow[] {
  return (ins?.marginal_effects?.grid ?? []).map((g: Json) => ({
    method: g.cool_roofing_method, quality: g.cool_roofing_quality, cost: g.cost, drop: g.temperature_drop, kwh: g.units_saved,
    high: g.high_rating, referral: g.referral, breakeven10: g.cost / (Math.max(g.units_saved, 1e-9) * 12 * 10),
  }));
}

export function topDrivers(ins: Json, task: string, k = 3): { feature: string; share: number }[] {
  return (ins?.feature_importance ?? [])
    .filter((r: Json) => r.task === task)
    .sort((a: Json, b: Json) => b.share - a.share)
    .slice(0, k)
    .map((r: Json) => ({ feature: r.feature, share: r.share }));
}

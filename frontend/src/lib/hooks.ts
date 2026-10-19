import { useCallback, useEffect, useRef, useState } from "react";

export interface AsyncState<T> {
  data: T | null;
  error: unknown;
  loading: boolean;
  reload: () => void;
}

/** Runs `fn` on mount / when deps change; ignores stale responses. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const latest = useRef(0);

  useEffect(() => {
    const id = ++latest.current;
    setLoading(true);
    setError(null);
    fn().then(
      (d) => { if (id === latest.current) { setData(d); setLoading(false); } },
      (e) => { if (id === latest.current) { setError(e); setLoading(false); } },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}

export function useDebounced<T>(value: T, ms = 350): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

/** localStorage/sessionStorage can throw (private mode, blocked data): always wrapped. */
export function safeGet(key: string, store: "local" | "session" = "session"): string | null {
  try { return (store === "local" ? localStorage : sessionStorage).getItem(key); } catch { return null; }
}
export function safeSet(key: string, value: string, store: "local" | "session" = "session") {
  try { (store === "local" ? localStorage : sessionStorage).setItem(key, value); } catch { /* ignore */ }
}

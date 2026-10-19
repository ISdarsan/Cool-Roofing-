import type {
  BuildingBase, BuildingInput, ChatResponse, CompareResponse, DataQualityResponse, EconomicsResponse, ExplainResponse,
  Json, ModelsResponse, Objective, PredictResponse, SchemaResponse, StatusResponse, TaskKey,
} from "./types";

export class ApiError extends Error {
  status: number;
  fieldErrors: { field: string; message: string }[];
  constructor(status: number, message: string, fieldErrors: { field: string; message: string }[] = []) {
    super(message);
    this.status = status;
    this.fieldErrors = fieldErrors;
  }
}

const BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(0, "Cannot reach the prediction service. Is the backend running? (python -m coolroof serve)");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    let fieldErrors: { field: string; message: string }[] = [];
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      if (Array.isArray(body.errors)) fieldErrors = body.errors;
    } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, detail, fieldErrors);
  }
  return (await res.json()) as T;
}

const post = <T>(path: string, body: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  status: () => request<StatusResponse>("/status"),
  schema: () => request<SchemaResponse>("/schema"),
  models: () => request<ModelsResponse>("/models"),
  dataQuality: () => request<DataQualityResponse>("/data-quality"),
  insights: () => request<Json>("/insights"),
  report: (name: string) => request<{ name: string; markdown: string }>(`/reports/${name}`),
  predict: (building: BuildingInput, opts: { referral_cutoff?: number | null; high_rating_cutoff?: number; interval_level?: 0.8 | 0.9 } = {}) =>
    post<PredictResponse>("/predict", {
      building,
      interval_level: opts.interval_level ?? 0.8,
      high_rating_cutoff: opts.high_rating_cutoff ?? 0.5,
      ...(opts.referral_cutoff != null ? { referral_cutoff: opts.referral_cutoff } : {}),
    }),
  compare: (building: BuildingBase, objective: Objective, weights: Record<string, number> | null,
            current?: { method?: string; quality?: string }) =>
    post<CompareResponse>("/compare", {
      building, objective, ...(objective === "balanced" && weights ? { weights } : {}),
      current_method: current?.method ?? null, current_quality: current?.quality ?? null,
    }),
  explain: (building: BuildingInput, task: TaskKey) => post<ExplainResponse>("/explain", { building, task }),
  economics: (body: Record<string, number>) => post<EconomicsResponse>("/economics", body),
  assistantChat: (messages: { role: "user" | "assistant"; content: string }[]) => post<ChatResponse>("/assistant/chat", { messages }),
  async proposalPdf(body: Record<string, unknown>): Promise<Blob> {
    let res: Response;
    try {
      res = await fetch(`${BASE}/proposal/pdf`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch {
      throw new ApiError(0, "Cannot reach the prediction service.");
    }
    if (!res.ok) {
      let detail = `Could not create the proposal (${res.status}).`;
      try { detail = (await res.json()).detail ?? detail; } catch { /* ignore */ }
      throw new ApiError(res.status, detail);
    }
    return res.blob();
  },
};

export const figureUrl = (name: string) => `${BASE}/figures/${name}`;

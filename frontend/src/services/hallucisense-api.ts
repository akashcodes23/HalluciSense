// ─────────────────────────────────────────────────────────────────────────────
// HalluciSense v1.0 — Production API Client
// Consumes real backend endpoints. Zero mocks. Zero placeholders.
// ─────────────────────────────────────────────────────────────────────────────

import type {
  AnalysisRequest,
  AnalysisResponse,
  ExplainRequest,
  ExplainResponse,
  MetricsResponse,
  HealthResponse,
  ReadinessResponse,
  TraceData,
} from "@/types/hallucisense";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "https://hallucisense-production.up.railway.app";

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new HalluciSenseAPIError(
      body.detail || body.message || `HTTP ${res.status}`,
      res.status,
      body
    );
  }

  return res.json() as Promise<T>;
}

// ── Error Class ──────────────────────────────────────────────────────────────

export class HalluciSenseAPIError extends Error {
  constructor(
    message: string,
    public status: number,
    public body: Record<string, unknown>
  ) {
    super(message);
    this.name = "HalluciSenseAPIError";
  }
}

// ── API Functions ────────────────────────────────────────────────────────────

export async function analyzeResponse(
  payload: AnalysisRequest
): Promise<AnalysisResponse> {
  return request<AnalysisResponse>("/api/v1/analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function explainResponse(
  payload: ExplainRequest
): Promise<ExplainResponse> {
  return request<ExplainResponse>("/api/v1/explain", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getMetrics(): Promise<MetricsResponse> {
  return request<MetricsResponse>("/api/v1/metrics");
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export async function getReady(): Promise<ReadinessResponse> {
  return request<ReadinessResponse>("/ready");
}

export async function getLatestDebug(): Promise<TraceData> {
  return request<TraceData>("/api/v1/debug/latest");
}

export async function getDebugTrace(traceId: string): Promise<TraceData> {
  return request<TraceData>(`/api/v1/debug/${traceId}`);
}

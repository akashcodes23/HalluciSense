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
  TraceData,
} from "@/types/hallucisense";

const getBaseUrl = (): string => {
  if (typeof window !== "undefined") {
    return process.env.NEXT_PUBLIC_API_BASE_URL ||
      process.env.NEXT_PUBLIC_API_URL ||
      "http://localhost:8000";
  }
  return process.env.NEXT_PUBLIC_API_BASE_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:8000";
};

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const url = `${getBaseUrl()}${path}`;
  try {
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
  } catch (err: unknown) {
    if (err instanceof HalluciSenseAPIError) {
      throw err;
    }
    throw new HalluciSenseAPIError(
      err instanceof Error ? err.message : "Verification service unavailable",
      503,
      { error: "NETWORK_ERROR" }
    );
  }
}

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

export async function getHealth(): Promise<{ status: string }> {
  return request<{ status: string }>("/health");
}

export async function getReady(): Promise<{ status: string; components: Record<string, boolean> }> {
  return request<{ status: string; components: Record<string, boolean> }>("/ready");
}

export async function getLatestDebug(): Promise<TraceData> {
  return request<TraceData>("/api/v1/debug/latest");
}

export async function getDebugTrace(traceId: string): Promise<TraceData> {
  return request<TraceData>(`/api/v1/debug/${traceId}`);
}

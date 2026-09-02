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
  CorrectionRequest,
  CorrectionResponse,
} from "@/types/hallucisense";

const getBaseUrl = (): string => {
  if (typeof window !== "undefined") {
    const savedUrl = localStorage.getItem("hallucisense_api_url");
    if (savedUrl) {
      return savedUrl;
    }
  }
  if (process.env.NEXT_PUBLIC_API_BASE_URL) {
    return process.env.NEXT_PUBLIC_API_BASE_URL;
  }
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (process.env.NODE_ENV === "production" || process.env.NEXT_PUBLIC_APP_ENV === "production") {
    return "https://hallucisense-production.up.railway.app";
  }
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host !== "localhost" && host !== "127.0.0.1") {
      return "https://hallucisense-production.up.railway.app";
    }
  }
  return "http://localhost:8000";
};

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const url = `${getBaseUrl()}${path}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
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
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new HalluciSenseAPIError(
        "Request timed out. The verification service did not respond within 60 seconds.",
        408,
        { error: "TIMEOUT" }
      );
    }
    throw new HalluciSenseAPIError(
      err instanceof Error ? err.message : "Verification service unavailable",
      503,
      { error: "NETWORK_ERROR" }
    );
  } finally {
    clearTimeout(timeout);
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

export async function correctResponse(
  payload: CorrectionRequest
): Promise<CorrectionResponse> {
  return request<CorrectionResponse>("/api/v1/correct", {
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

export interface ClosedLoopChatPayload {
  message: string;
  enable_verification?: boolean;
  auto_correct?: boolean;
  model_name?: string;
  conversation_id?: string;
}

export interface ClosedLoopChatResponse {
  conversation_id: string;
  message_id: string;
  original_response: string;
  final_response: string;
  verification: {
    status: string;
    h_score: number | null;
    risk_level: string | null;
    claims_total: number | null;
    claims_flagged: number | null;
    error_message?: string;
  };
  correction: {
    performed: boolean;
    reason: string;
    claims_corrected: Array<Record<string, unknown>>;
    original_to_corrected: Array<Record<string, unknown>>;
  };
  evidence: Array<{
    source_name: string;
    snippet: string;
    claim: string;
  }>;
  sources: string[];
  trace_id: string;
  latency_ms: number;
}

export async function sendClosedLoopChat(
  payload: ClosedLoopChatPayload
): Promise<ClosedLoopChatResponse> {
  return request<ClosedLoopChatResponse>("/api/v1/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

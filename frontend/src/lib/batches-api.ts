import { handleUnauthorized } from "./api-client";
import { getApiKey, getAuthRequestCredentials } from "./auth";
import { UnauthorizedError, buildUrl, createTimeoutSignal, get } from "./api-client";
import type { BatchListResponse, BatchStatusResponse, BatchSubmitResponse } from "./types";

function authHeaders(): Headers {
  const headers = new Headers({ Accept: "application/json" });
  const key = getApiKey();
  if (key) headers.set("X-API-Key", key);
  return headers;
}

export async function fetchBatches(limit = 25, offset = 0): Promise<BatchListResponse> {
  return get<BatchListResponse>(`/api/v1/jobs/batch?limit=${limit}&offset=${offset}`);
}

export async function fetchBatch(batchId: string, signal?: AbortSignal): Promise<BatchStatusResponse> {
  return get<BatchStatusResponse>(`/api/v1/jobs/batch/${encodeURIComponent(batchId)}`, { signal });
}

export async function submitBatch(files: File[], priority = "normal"): Promise<BatchSubmitResponse> {
  const body = new FormData();
  for (const file of files) {
    body.append("files", file);
  }
  body.set("priority", priority);

  const { signal, cleanup } = createTimeoutSignal();
  try {
    const response = await fetch(buildUrl("/api/v1/jobs/batch"), {
      method: "POST",
      headers: authHeaders(),
      credentials: getAuthRequestCredentials(),
      body,
      signal,
    });

    if (response.status === 401) {
      handleUnauthorized();
      throw new UnauthorizedError(response.status, null);
    }

    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      const detail =
        payload && typeof payload === "object" && "detail" in payload
          ? JSON.stringify(payload.detail)
          : `Batch submit failed with status ${response.status}`;
      throw new Error(detail);
    }

    return (await response.json()) as BatchSubmitResponse;
  } finally {
    cleanup();
  }
}

/**
 * Audit-specific API helpers.
 *
 * The backend exposes the custody log as a JSONL artifact via:
 *   GET /api/v1/jobs/{job_id}/outputs/custody
 *
 * The response body is newline-delimited JSON (one event per line). This
 * helper fetches it as text, splits, and parses each line into a typed
 * CustodyEvent. Errors on malformed lines are surfaced via reject so the
 * UI can render a clear failure state instead of silently dropping events.
 */

import { ApiError, UnauthorizedError, get } from "./api-client";
import { getApiKey, clearApiKey } from "./auth";
import type { CustodyEvent } from "./audit-verify";

const API_BASE_URL =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_API_BASE_URL) ||
  "http://localhost:8000";

function buildUrl(path: string): string {
  const base = API_BASE_URL.replace(/\/+$/, "");
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
}

export interface JobSummary {
  job_id: string;
  status: string;
  filename?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface JobListResponse {
  jobs: JobSummary[];
  total: number;
  limit: number;
  offset: number;
}

/**
 * Fetch the most recent jobs. Used by the /audit search picker.
 */
export async function listRecentJobs(limit = 20): Promise<JobListResponse> {
  const safeLimit = Math.max(1, Math.min(100, Math.trunc(limit)));
  return get<JobListResponse>(`/api/v1/jobs?limit=${safeLimit}&offset=0`);
}

/**
 * Fetch the raw custody JSONL for a job and parse it into events.
 *
 * The custody endpoint returns the file as ``application/jsonl`` so we can't
 * use the JSON-only api-client.get helper -- we need raw text.
 */
export async function fetchCustodyLog(jobId: string): Promise<CustodyEvent[]> {
  const headers = new Headers({ Accept: "application/jsonl, text/plain;q=0.9, */*;q=0.5" });
  const apiKey = getApiKey();
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }
  const response = await fetch(buildUrl(`/api/v1/jobs/${encodeURIComponent(jobId)}/outputs/custody`), {
    method: "GET",
    headers,
    cache: "no-store",
  });
  if (response.status === 401 || response.status === 403) {
    clearApiKey();
    throw new UnauthorizedError(response.status, null);
  }
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = String((body as { detail: unknown }).detail);
      }
    } catch {
      // Non-JSON error body; keep generic message.
    }
    throw new ApiError(response.status, detail, null);
  }
  const text = await response.text();
  return parseJsonlEvents(text);
}

export function parseJsonlEvents(text: string): CustodyEvent[] {
  const events: CustodyEvent[] = [];
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i].trim();
    if (raw.length === 0) {
      continue;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      throw new Error(
        `Malformed custody event on line ${i + 1}: ${err instanceof Error ? err.message : String(err)}`
      );
    }
    if (!parsed || typeof parsed !== "object") {
      throw new Error(`Custody event on line ${i + 1} is not a JSON object`);
    }
    events.push(parsed as CustodyEvent);
  }
  return events;
}

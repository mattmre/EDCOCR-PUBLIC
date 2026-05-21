/**
 * Admin alerts API helpers.
 *
 * Mirrors the (not-yet-provisioned) FastAPI endpoints under
 * `/api/v1/admin/alerts/*` and `/api/v1/admin/alert-channels/*`.
 *
 * The base `lib/api-client.ts` only exports `get`/`post` and is owned by a
 * different agent for this wave; we therefore mirror the request/handle
 * pattern locally (same shape `lib/tenant-api.ts` uses) to gain PATCH/PUT.
 *
 * IMPORTANT: We do NOT mask 403 responses. Pages are expected to render an
 * "access denied" empty-state when a non-admin caller hits these endpoints.
 */

import { ApiError, UnauthorizedError } from "./api-client";
import { clearApiKey, getApiKey } from "./auth";
import type {
  Alert,
  AlertMuteRequest,
  AlertRule,
  AlertRuleUpdate,
  NotificationChannel,
} from "./types";

const API_BASE_URL =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_API_BASE_URL) ||
  "http://localhost:8000";

function buildUrl(path: string): string {
  const base = API_BASE_URL.replace(/\/+$/, "");
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
}

function buildHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  if (!headers.has("Accept")) headers.set("Accept", "application/json");
  const apiKey = getApiKey();
  if (apiKey) headers.set("X-API-Key", apiKey);
  return headers;
}

async function parseBody(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }
  try {
    return await response.text();
  } catch {
    return null;
  }
}

async function handle<T>(response: Response): Promise<T> {
  if (response.status === 401) {
    // 401 -> drop cached key so we don't loop on a bad credential. The
    // route boundary will redirect to /login.
    clearApiKey();
    const body = await parseBody(response);
    throw new UnauthorizedError(response.status, body);
  }
  // NOTE: 403 is intentionally NOT masked. The alerts page renders an
  // "access denied" empty-state when this fires; clearing the key and
  // bouncing to /login would be wrong because the user IS logged in --
  // they just lack platform-admin scope.
  if (response.status === 403) {
    const body = await parseBody(response);
    throw new ApiError(403, "Forbidden", body);
  }
  if (!response.ok) {
    const body = await parseBody(response);
    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : `Request failed with status ${response.status}`;
    throw new ApiError(response.status, detail, body);
  }
  if (response.status === 204) return undefined as T;
  const body = await parseBody(response);
  return body as T;
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  init?: { signal?: AbortSignal }
): Promise<T> {
  const headers = buildHeaders();
  if (body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(buildUrl(path), {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
    signal: init?.signal,
  });
  return handle<T>(response);
}

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------

export async function listAlerts(signal?: AbortSignal): Promise<Alert[]> {
  return request<Alert[]>("GET", "/api/v1/admin/alerts", undefined, { signal });
}

export async function muteAlert(
  alertId: string,
  payload: AlertMuteRequest
): Promise<Alert> {
  return request<Alert>(
    "POST",
    `/api/v1/admin/alerts/${encodeURIComponent(alertId)}/mute`,
    payload
  );
}

export async function unmuteAlert(alertId: string): Promise<Alert> {
  return request<Alert>(
    "POST",
    `/api/v1/admin/alerts/${encodeURIComponent(alertId)}/unmute`,
    {}
  );
}

// ---------------------------------------------------------------------------
// Alert rules
// ---------------------------------------------------------------------------

export async function listRules(signal?: AbortSignal): Promise<AlertRule[]> {
  return request<AlertRule[]>("GET", "/api/v1/admin/alerts/rules", undefined, { signal });
}

export async function getRule(
  ruleId: string,
  signal?: AbortSignal
): Promise<AlertRule> {
  return request<AlertRule>(
    "GET",
    `/api/v1/admin/alerts/rules/${encodeURIComponent(ruleId)}`,
    undefined,
    { signal }
  );
}

/**
 * PATCH the operator-editable subset of a rule. Backend rejects any
 * attempt to mutate the PromQL expression with 400.
 */
export async function updateRuleThreshold(
  ruleId: string,
  payload: AlertRuleUpdate
): Promise<AlertRule> {
  return request<AlertRule>(
    "PATCH",
    `/api/v1/admin/alerts/rules/${encodeURIComponent(ruleId)}`,
    payload
  );
}

// ---------------------------------------------------------------------------
// Notification channels
// ---------------------------------------------------------------------------

export async function listChannels(
  signal?: AbortSignal
): Promise<NotificationChannel[]> {
  return request<NotificationChannel[]>(
    "GET",
    "/api/v1/admin/alert-channels",
    undefined,
    { signal }
  );
}

export interface ChannelTestResult {
  ok: boolean;
  /** ISO 8601 UTC timestamp the test was attempted at. */
  tested_at: string;
  message?: string;
}

export async function testChannel(channelId: string): Promise<ChannelTestResult> {
  return request<ChannelTestResult>(
    "POST",
    `/api/v1/admin/alert-channels/${encodeURIComponent(channelId)}/test`,
    {}
  );
}

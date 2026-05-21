/**
 * Tenant + glossary admin API helpers.
 *
 * Mirrors the FastAPI endpoints declared in `api/routers/translation_admin.py`
 * and `api/routers/admin.py`. These wrappers exist alongside `lib/api-client.ts`
 * because the admin flows need PUT/PATCH/DELETE verbs that the base helper
 * does not currently expose.
 */

import { ApiError, UnauthorizedError } from "./api-client";
import { clearApiKey, getApiKey } from "./auth";
import type {
  GlossaryEntry,
  GlossaryFilters,
  GlossaryListResponse,
  TenantConfig,
  TenantConfigUpdate,
  TenantSummary,
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
  if (response.status === 401 || response.status === 403) {
    clearApiKey();
    const body = await parseBody(response);
    throw new UnauthorizedError(response.status, body);
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
// Tenant directory (api/routers/admin.py)
// ---------------------------------------------------------------------------

/**
 * List known tenants from /api/v1/admin/tenants. The endpoint is gated on
 * multi-tenancy being enabled and on the caller having admin scope; on
 * 404/403/501 the caller should fall back to manual tenant_id input.
 */
export async function listTenants(): Promise<TenantSummary[]> {
  return request<TenantSummary[]>("GET", "/api/v1/admin/tenants");
}

// ---------------------------------------------------------------------------
// Tenant translation config (api/routers/translation_admin.py)
// ---------------------------------------------------------------------------

export async function getTenantConfig(tenantId: string): Promise<TenantConfig> {
  return request<TenantConfig>(
    "GET",
    `/api/v1/translation/tenants/${encodeURIComponent(tenantId)}/config`
  );
}

export async function upsertTenantConfig(
  tenantId: string,
  payload: TenantConfigUpdate
): Promise<TenantConfig> {
  return request<TenantConfig>(
    "PUT",
    `/api/v1/translation/tenants/${encodeURIComponent(tenantId)}/config`,
    payload
  );
}

// ---------------------------------------------------------------------------
// Glossary (api/routers/translation_admin.py)
// ---------------------------------------------------------------------------

export async function listGlossary(
  tenantId: string,
  filters?: GlossaryFilters
): Promise<GlossaryListResponse> {
  const params = new URLSearchParams();
  if (filters?.source_lang) params.set("source_lang", filters.source_lang);
  if (filters?.target_lang) params.set("target_lang", filters.target_lang);
  if (filters?.page) params.set("page", String(filters.page));
  if (filters?.page_size) params.set("page_size", String(filters.page_size));
  const qs = params.toString();
  const path =
    `/api/v1/translation/tenants/${encodeURIComponent(tenantId)}/glossary` +
    (qs ? `?${qs}` : "");
  return request<GlossaryListResponse>("GET", path);
}

export interface GlossaryEntryInput {
  source_term: string;
  target_term: string;
  source_lang: string;
  target_lang: string;
  case_sensitive?: boolean;
  is_regex?: boolean;
  priority?: number;
  notes?: string | null;
}

export async function createGlossaryEntry(
  tenantId: string,
  payload: GlossaryEntryInput
): Promise<GlossaryEntry> {
  return request<GlossaryEntry>(
    "POST",
    `/api/v1/translation/tenants/${encodeURIComponent(tenantId)}/glossary`,
    payload
  );
}

export async function updateGlossaryEntry(
  tenantId: string,
  entryId: number,
  payload: Partial<GlossaryEntryInput>
): Promise<GlossaryEntry> {
  return request<GlossaryEntry>(
    "PATCH",
    `/api/v1/translation/tenants/${encodeURIComponent(tenantId)}/glossary/${entryId}`,
    payload
  );
}

export async function deleteGlossaryEntry(
  tenantId: string,
  entryId: number
): Promise<void> {
  await request<void>(
    "DELETE",
    `/api/v1/translation/tenants/${encodeURIComponent(tenantId)}/glossary/${entryId}`
  );
}

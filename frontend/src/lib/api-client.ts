import { clearApiKey, getApiKey, getAuthRequestCredentials } from "./auth";
import { isValidApiBaseUrl, loadSettings } from "./settings-store";
import { configuredApiBaseUrl } from "./runtime-config";

const DEFAULT_TIMEOUT_MS = 30000;
const TIMEOUT_MIN_MS = 1000;
const TIMEOUT_MAX_MS = 120000;
const DEFAULT_API_BASE_URL = "http://localhost:8000";

function clampTimeoutMs(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_TIMEOUT_MS;
  return Math.min(TIMEOUT_MAX_MS, Math.max(TIMEOUT_MIN_MS, Math.floor(value)));
}

/**
 * Resolve the FastAPI base URL.
 *
 * Priority (browser only for settings):
 *   1. Non-empty, valid `settings.api.baseUrl` from localStorage
 *   2. Public deployment runtime configuration (same-origin by default)
 *   3. `NEXT_PUBLIC_API_BASE_URL` build/development fallback
 *   4. `http://localhost:8000` for clients without server configuration
 *
 * `loadSettings` is SSR-safe (returns defaults when `window` is undefined),
 * so this never touches localStorage during SSR/tests without a window.
 */
export function resolveApiBaseUrl(): string {
  return resolveConfiguredApiBaseUrl() ?? DEFAULT_API_BASE_URL;
}

export function resolveConfiguredApiBaseUrl(): string | undefined {
  // Only consult persisted settings in the browser so SSR/Node stay on env.
  if (typeof window !== "undefined") {
    try {
      const settings = loadSettings();
      const candidate =
        typeof settings.api?.baseUrl === "string" ? settings.api.baseUrl.trim() : "";
      if (candidate.length > 0 && isValidApiBaseUrl(candidate)) {
        return candidate.replace(/\/+$/, "");
      }
    } catch {
      // Corrupt settings or storage failure — fall through to env.
    }
  }
  return configuredApiBaseUrl()?.replace(/\/+$/, "");
}

/**
 * Resolve request timeout in ms from settings (clamped), else 30000.
 * Settings are only read in the browser.
 */
export function resolveTimeoutMs(): number {
  if (typeof window !== "undefined") {
    try {
      const settings = loadSettings();
      if (typeof settings.api?.timeoutMs === "number") {
        return clampTimeoutMs(settings.api.timeoutMs);
      }
    } catch {
      // Fall through.
    }
  }
  return DEFAULT_TIMEOUT_MS;
}

/**
 * Combine an optional caller AbortSignal with a timeout derived from settings.
 * Returns the signal to pass to fetch and a cleanup that must run after the
 * request settles (clears the timeout timer and detaches listeners).
 */
export function createTimeoutSignal(callerSignal?: AbortSignal): {
  signal: AbortSignal;
  cleanup: () => void;
} {
  const timeoutMs = resolveTimeoutMs();
  const controller = new AbortController();
  let settled = false;

  const abortFromTimeout = () => {
    if (!settled) {
      controller.abort();
    }
  };
  const timer = setTimeout(abortFromTimeout, timeoutMs);

  const abortFromCaller = () => {
    if (!settled) {
      controller.abort();
    }
  };

  if (callerSignal) {
    if (callerSignal.aborted) {
      controller.abort();
    } else {
      callerSignal.addEventListener("abort", abortFromCaller, { once: true });
    }
  }

  const cleanup = () => {
    settled = true;
    clearTimeout(timer);
    if (callerSignal) {
      callerSignal.removeEventListener("abort", abortFromCaller);
    }
  };

  return { signal: controller.signal, cleanup };
}

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, message: string, body: unknown = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

/**
 * UnauthorizedError is thrown only when authentication fails with 401. A 403
 * is an authorization denial and must preserve the caller's valid credential.
 */
export class UnauthorizedError extends ApiError {
  constructor(status: number, body: unknown = null) {
    super(status, status === 401 ? "Unauthorized" : "Forbidden", body);
    this.name = "UnauthorizedError";
  }
}

export function buildUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  const base = resolveApiBaseUrl().replace(/\/+$/, "");
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
}

function setJsonAccept(headers: Headers): void {
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json");
  }
}

function buildHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  setJsonAccept(headers);
  const apiKey = getApiKey();
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }
  return headers;
}

/**
 * Build probe headers for a candidate key WITHOUT touching the stored key.
 *
 * Single header-construction site (with `buildHeaders`) for `X-API-Key`:
 * the login probe uses this instead of hand-building headers so future
 * header-contract changes apply in one place.
 */
export function buildProbeHeaders(candidateKey: string): Headers {
  const headers = new Headers();
  setJsonAccept(headers);
  const trimmed = candidateKey.trim();
  if (trimmed.length > 0) {
    headers.set("X-API-Key", trimmed);
  }
  return headers;
}

/**
 * Credential-gated probe that never persists the candidate key.
 *
 * Uses the centralized timeout signal (`createTimeoutSignal`) so a hung
 * backend aborts the probe instead of pinning the caller indefinitely.
 * Returns the raw `Response` so the caller can branch on status
 * (401/403 vs ok) without triggering stored-key side effects.
 */
export async function probeApiKey(
  path: string,
  candidateKey: string,
  options?: RequestOptions
): Promise<Response> {
  const { signal, cleanup } = createTimeoutSignal(options?.signal);
  try {
    return await fetch(buildUrl(path), {
      method: "GET",
      headers: buildProbeHeaders(candidateKey),
      credentials: getAuthRequestCredentials(),
      signal,
      cache: "no-store",
    });
  } finally {
    cleanup();
  }
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

/**
 * Central route-boundary redirect for revoked/expired credentials.
 *
 * Fulfills the UnauthorizedError contract above: a 401 observed by the
 * shared client sends an authenticated page back to /login instead of
 * leaving it rendering as if authenticated. A 403 can mean a valid key lacks
 * the requested capability, so it never clears credentials or redirects.
 * Browser-only; skips when
 * already on /login so a failing login probe cannot loop.
 */
function redirectToLogin(): void {
  if (typeof window === "undefined") return;
  try {
    const loc = window.location;
    if (!loc || typeof loc.assign !== "function") return;
    if (loc.pathname === "/login") return;
    loc.assign("/login");
  } catch {
    // No navigation available (SSR/test runtimes) — key is already cleared.
  }
}

export function handleUnauthorized(): void {
  clearApiKey();
  redirectToLogin();
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (response.status === 401) {
    // Drop a rejected credential, then redirect centrally to /login.
    handleUnauthorized();
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
  if (response.status === 204) {
    return undefined as T;
  }
  const body = await parseBody(response);
  return body as T;
}

export interface RequestOptions {
  signal?: AbortSignal;
  headers?: HeadersInit;
}

export async function get<T = unknown>(path: string, options?: RequestOptions): Promise<T> {
  const { signal, cleanup } = createTimeoutSignal(options?.signal);
  try {
    const response = await fetch(buildUrl(path), {
      method: "GET",
      headers: buildHeaders(options?.headers),
      credentials: getAuthRequestCredentials(),
      signal,
      cache: "no-store",
    });
    return await handleResponse<T>(response);
  } finally {
    cleanup();
  }
}

export async function post<T = unknown>(
  path: string,
  body: unknown,
  options?: RequestOptions
): Promise<T> {
  const headers = buildHeaders(options?.headers);
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const { signal, cleanup } = createTimeoutSignal(options?.signal);
  try {
    const response = await fetch(buildUrl(path), {
      method: "POST",
      headers,
      credentials: getAuthRequestCredentials(),
      signal,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return await handleResponse<T>(response);
  } finally {
    cleanup();
  }
}

export async function put<T = unknown>(
  path: string,
  body: unknown,
  options?: RequestOptions
): Promise<T> {
  const headers = buildHeaders(options?.headers);
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const { signal, cleanup } = createTimeoutSignal(options?.signal);
  try {
    const response = await fetch(buildUrl(path), {
      method: "PUT",
      headers,
      credentials: getAuthRequestCredentials(),
      signal,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return await handleResponse<T>(response);
  } finally {
    cleanup();
  }
}

export const apiClient = {
  get,
  post,
  put,
  /** Dynamic base URL — re-reads settings/env on every access. */
  get baseUrl(): string {
    return resolveApiBaseUrl();
  },
};

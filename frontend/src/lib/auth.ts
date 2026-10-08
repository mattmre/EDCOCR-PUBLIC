"use client";

import { useEffect } from "react";
import { LOGIN_PROBE_PATH } from "./login-contract";
import { browserRuntimeConfig } from "./runtime-config";

const STORAGE_KEY = "ocr_local_api_key";
export const AUTH_STATE_EVENT = "ocr-local:auth-state";
const AUTH_REQUEST_TIMEOUT_MS = 30_000;

async function fetchAuthRequest(
  input: RequestInfo | URL,
  init: RequestInit
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), AUTH_REQUEST_TIMEOUT_MS);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export interface AuthStateEventDetail {
  authenticated: boolean;
}

/**
 * Prefer HttpOnly cookie session (LT-10) when ``NEXT_PUBLIC_OCR_HTTPONLY_SESSION``
 * is truthy: keys are POSTed to ``/api/v1/auth/session`` and not kept in
 * sessionStorage. Fallback remains sessionStorage (tab-scoped) over localStorage.
 *
 * sessionStorage clears when the tab closes, reducing long-lived XSS
 * key-theft window vs localStorage. XSS can still read sessionStorage while
 * the tab is open — use HttpOnly session for production consoles.
 */

export function isHttpOnlySessionEnabled(): boolean {
  const runtime = browserRuntimeConfig();
  if (runtime) return runtime.httpOnlySession;
  const value = process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION?.trim().toLowerCase();
  return value === "1" || value === "true";
}

export function getAuthRequestCredentials(): RequestCredentials {
  // Fetch defaults to same-origin cookies when this is undefined. Off mode
  // must explicitly suppress an old HttpOnly cookie after a mode rollback.
  return isHttpOnlySessionEnabled() ? "include" : "omit";
}

function emitAuthState(authenticated: boolean): void {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(
    new CustomEvent<AuthStateEventDetail>(AUTH_STATE_EVENT, {
      detail: { authenticated },
    })
  );
}

function clearStoredApiKey(): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}

function _storage(): Storage | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return window.sessionStorage;
  } catch {
    try {
      return window.localStorage;
    } catch {
      return null;
    }
  }
}

/**
 * Read the operator API key from sessionStorage (preferred) or localStorage.
 *
 * Migrates any legacy localStorage key into sessionStorage once, then removes
 * the long-lived copy.
 *
 * Returns null if the key is missing, empty, or the runtime is non-browser
 * (SSR / Node test runner without a window stub).
 */
export function getApiKey(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  if (isHttpOnlySessionEnabled()) {
    // Cookie mode must never fall back to a stale JS-readable credential.
    clearStoredApiKey();
    return null;
  }
  try {
    // Migrate legacy localStorage → sessionStorage (one-shot).
    try {
      const legacy = window.localStorage.getItem(STORAGE_KEY);
      if (legacy && legacy.trim().length > 0) {
        window.sessionStorage.setItem(STORAGE_KEY, legacy);
        window.localStorage.removeItem(STORAGE_KEY);
      }
    } catch {
      // ignore migration failures
    }
    const store = _storage();
    if (!store) {
      return null;
    }
    const value = store.getItem(STORAGE_KEY);
    if (!value || value.trim().length === 0) {
      return null;
    }
    return value;
  } catch {
    // Storage may be denied (Safari private mode, etc.) -- treat as missing.
    return null;
  }
}

/**
 * Persist the operator API key in sessionStorage (tab-scoped).
 *
 * Throws on empty input so callers cannot silently store a blank credential.
 */
export function setApiKey(key: string): void {
  if (typeof window === "undefined") {
    return;
  }
  const trimmed = key.trim();
  if (trimmed.length === 0) {
    throw new Error("API key cannot be empty");
  }
  // LT-10: when HttpOnly session preferred, clear JS-readable storage; the
  // caller should also POST /api/v1/auth/session (see setApiKeyHttpOnly).
  if (isHttpOnlySessionEnabled()) {
    clearStoredApiKey();
    return;
  }
  const store = _storage();
  if (!store) {
    throw new Error("Browser storage unavailable");
  }
  store.setItem(STORAGE_KEY, trimmed);
  // Ensure legacy long-lived key is gone.
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
  emitAuthState(true);
}

/**
 * LT-10: store API key via HttpOnly cookie endpoint (credentials include).
 * Returns true only after a cookie-only protected request succeeds. It never
 * falls back to browser-readable storage.
 */
export class HttpOnlySessionError extends Error {
  readonly status: number;
  readonly stage: "create" | "validate" | "logout";

  constructor(
    status: number,
    message = `HttpOnly session failed: HTTP ${status}`,
    stage: "create" | "validate" | "logout" = "create"
  ) {
    super(message);
    this.name = "HttpOnlySessionError";
    this.status = status;
    this.stage = stage;
  }
}

function sessionUrl(baseUrl: string): string {
  return `${baseUrl.replace(/\/+$/, "")}/api/v1/auth/session`;
}

export async function setApiKeyHttpOnly(key: string, baseUrl: string = ""): Promise<boolean> {
  const trimmed = key.trim();
  if (trimmed.length === 0) {
    throw new Error("API key cannot be empty");
  }
  const resp = await fetchAuthRequest(sessionUrl(baseUrl), {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: trimmed }),
  });
  if (!resp.ok) {
    if (resp.status === 401) {
      clearStoredApiKey();
      emitAuthState(false);
    }
    throw new HttpOnlySessionError(resp.status);
  }
  clearStoredApiKey();
  try {
    const valid = await hasValidHttpOnlySession(baseUrl);
    if (!valid) {
      throw new HttpOnlySessionError(
        401,
        "HttpOnly session cookie was rejected by protected authentication",
        "validate"
      );
    }
  } catch (error) {
    if (error instanceof HttpOnlySessionError && error.status === 401) {
      emitAuthState(false);
    }
    throw error;
  }
  emitAuthState(true);
  return true;
}

/**
 * Validate the cookie against a credential-gated production endpoint.
 *
 * GET /api/v1/auth/session only reports cookie presence and is auth-exempt, so
 * it cannot distinguish a valid session from a stale or revoked cookie.
 */
export async function hasValidHttpOnlySession(baseUrl: string = ""): Promise<boolean> {
  const resp = await fetchAuthRequest(
    `${baseUrl.replace(/\/+$/, "")}${LOGIN_PROBE_PATH}`,
    {
      method: "GET",
      credentials: "include",
      headers: { Accept: "application/json" },
      cache: "no-store",
    }
  );
  if (resp.status === 401) {
    return false;
  }
  if (!resp.ok) {
    throw new HttpOnlySessionError(
      resp.status,
      `Session validation failed: HTTP ${resp.status}`,
      "validate"
    );
  }
  return true;
}

async function deleteHttpOnlySessionCookie(baseUrl: string): Promise<void> {
  const resp = await fetchAuthRequest(sessionUrl(baseUrl), {
    method: "DELETE",
    credentials: "include",
    headers: { Accept: "application/json" },
  });
  if (!resp.ok) {
    throw new HttpOnlySessionError(
      resp.status,
      `Session logout failed: HTTP ${resp.status}`,
      "logout"
    );
  }
}

export async function clearHttpOnlySession(baseUrl: string = ""): Promise<void> {
  await deleteHttpOnlySessionCookie(baseUrl);
  clearStoredApiKey();
  emitAuthState(false);
}

export async function signOut(baseUrl: string = ""): Promise<void> {
  if (isHttpOnlySessionEnabled()) {
    await clearHttpOnlySession(baseUrl);
    return;
  }
  try {
    await deleteHttpOnlySessionCookie(baseUrl);
  } finally {
    // Header-key mode still signs out locally if the optional cookie endpoint
    // is unavailable. The caller retains an unknown/retryable state on error.
    clearApiKey();
  }
}

/**
 * Remove the API key from sessionStorage and localStorage. Used on logout
 * and on 401 from the API client. A 403 can mean the authenticated request
 * lacks permission and does not prove that the credential is invalid.
 */
export function clearApiKey(): void {
  clearStoredApiKey();
  emitAuthState(false);
}

/**
 * Derive a non-empty reviewer identity label for forensic decision records.
 *
 * Priority:
 * 1. Explicit non-blank label (prop / form override)
 * 2. Redacted API-key preview (first 4 + last 4) when the cached key is long enough
 * 3. Fallback string `"operator"` — never returns empty
 *
 * The console authenticates with an API key, not a named user, so decisions
 * must still carry an attributable label instead of an empty reviewer field.
 */
export function deriveReviewerLabel(explicit?: string | null): string {
  const trimmed = (explicit ?? "").trim();
  if (trimmed.length > 0) {
    return trimmed;
  }
  const key = getApiKey();
  if (key) {
    const k = key.trim();
    // Match settings-store redaction: short keys would leak nearly the whole value.
    if (k.length > 8) {
      return `${k.slice(0, 4)}…${k.slice(-4)}`;
    }
  }
  return "operator";
}

/**
 * Hook that redirects to /login when no API key is present.
 *
 * Wave 1 implementation: client-side gate only. Server-side enforcement
 * happens at the FastAPI layer. The hook intentionally runs effects only
 * once on mount; subsequent key changes during a session are uncommon and
 * handled by the api-client throwing on 401.
 */
export function useRequireAuth(): void {
  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    if (isHttpOnlySessionEnabled()) {
      // HttpOnly cookie state is not synchronously readable. Protected API
      // responses and the topbar protected probe remain the runtime authority.
      return;
    }
    const key = getApiKey();
    if (!key) {
      window.location.assign("/login");
    }
  }, []);
}

export const AUTH_STORAGE_KEY = STORAGE_KEY;

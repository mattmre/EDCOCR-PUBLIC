"use client";

import { useEffect } from "react";

const STORAGE_KEY = "ocr_local_api_key";

/**
 * Read the operator API key from localStorage.
 *
 * Returns null if the key is missing, empty, or the runtime is non-browser
 * (SSR / Node test runner without a window stub).
 */
export function getApiKey(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
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
 * Persist the operator API key in localStorage.
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
  window.localStorage.setItem(STORAGE_KEY, trimmed);
}

/**
 * Remove the API key from localStorage. Used on logout and on 401/403 from
 * the API client.
 */
export function clearApiKey(): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.removeItem(STORAGE_KEY);
}

/**
 * Hook that redirects to /login when no API key is present.
 *
 *  implementation: client-side gate only. Server-side enforcement
 * happens at the FastAPI layer. The hook intentionally runs effects only
 * once on mount; subsequent key changes during a session are uncommon and
 * handled by the api-client throwing on 401.
 */
export function useRequireAuth(): void {
  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const key = getApiKey();
    if (!key) {
      window.location.assign("/login");
    }
  }, []);
}

export const AUTH_STORAGE_KEY = STORAGE_KEY;

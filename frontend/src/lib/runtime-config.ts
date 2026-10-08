/** The only server settings intentionally exposed to the browser. */
export interface PublicRuntimeConfig {
  apiBaseUrl: string;
  httpOnlySession: boolean;
}

export const RUNTIME_CONFIG_ID = "ocr-local-runtime-config";

/** A backend origin or root-relative path, without credentials/query/fragment. */
export function isValidApiBaseUrl(value: string): boolean {
  if (value === "") return true;
  if (/[\\\s]/.test(value) || value.startsWith("//")) return false;
  try {
    const url = value.startsWith("/") ? new URL(value, "https://config.invalid") : new URL(value);
    return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password && !url.search && !url.hash;
  } catch {
    return false;
  }
}

/** Read at request time on the server; never serialize the environment itself. */
export function publicRuntimeConfig(env: Record<string, string | undefined>): PublicRuntimeConfig {
  const apiBaseUrl = (env.OCR_FRONTEND_API_BASE_URL ?? env.NEXT_PUBLIC_API_BASE_URL ??
    (env.NODE_ENV === "development" ? "http://localhost:8000" : "/")).trim();
  if (!apiBaseUrl || !isValidApiBaseUrl(apiBaseUrl)) {
    throw new Error("Invalid frontend API base URL configuration");
  }
  const mode = (env.OCR_FRONTEND_HTTPONLY_SESSION ?? env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION ?? "false").trim().toLowerCase();
  if (!["true", "false", "1", "0"].includes(mode)) {
    throw new Error("Invalid frontend HttpOnly session configuration");
  }
  return { apiBaseUrl, httpOnlySession: mode === "true" || mode === "1" };
}

/** JSON script contents must not be able to end their HTML element. */
export function serializeRuntimeConfig(config: PublicRuntimeConfig): string {
  return JSON.stringify(config).replace(/</g, "\\u003c").replace(/\u2028/g, "\\u2028").replace(/\u2029/g, "\\u2029");
}

export function browserRuntimeConfig(): PublicRuntimeConfig | undefined {
  // Server rendering must use the same values as the injected document to
  // avoid an authentication-mode hydration mismatch on the login page.
  if (typeof document === "undefined") return publicRuntimeConfig(process.env);
  const element = document.getElementById(RUNTIME_CONFIG_ID);
  if (!element) return undefined;
  const value: unknown = JSON.parse(element.textContent ?? "");
  if (!value || typeof value !== "object" || !("apiBaseUrl" in value) ||
      !("httpOnlySession" in value) || typeof value.apiBaseUrl !== "string" ||
      !value.apiBaseUrl || !isValidApiBaseUrl(value.apiBaseUrl) ||
      typeof value.httpOnlySession !== "boolean") {
    throw new Error("Invalid browser runtime configuration");
  }
  return { apiBaseUrl: value.apiBaseUrl, httpOnlySession: value.httpOnlySession };
}

export function configuredApiBaseUrl(): string | undefined {
  const runtime = browserRuntimeConfig();
  if (runtime) return runtime.apiBaseUrl;
  // Keep direct references: Next can inline these for development/static clients.
  const value = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (value && isValidApiBaseUrl(value)) return value;
  return undefined;
}

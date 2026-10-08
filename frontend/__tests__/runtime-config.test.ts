import { afterEach, describe, expect, it, vi } from "vitest";
import { browserRuntimeConfig, isValidApiBaseUrl, publicRuntimeConfig, RUNTIME_CONFIG_ID, serializeRuntimeConfig } from "@/lib/runtime-config";
import { buildUrl } from "@/lib/api-client";
import { getApiKey, getAuthRequestCredentials, isHttpOnlySessionEnabled } from "@/lib/auth";
import { buildJobWebSocketUrl } from "@/lib/ws-client";

function install(config: unknown) {
  const script = document.createElement("script");
  script.id = RUNTIME_CONFIG_ID;
  script.type = "application/json";
  script.textContent = JSON.stringify(config);
  document.head.appendChild(script);
}

afterEach(() => {
  document.getElementById(RUNTIME_CONFIG_ID)?.remove();
  window.localStorage.clear();
  window.sessionStorage.clear();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("public deployment runtime configuration", () => {
  it("exposes only the two public fields and reads deployment values before legacy build settings", () => {
    const config = publicRuntimeConfig({
      OCR_FRONTEND_API_BASE_URL: "/runtime",
      OCR_FRONTEND_HTTPONLY_SESSION: "true",
      NEXT_PUBLIC_API_BASE_URL: "http://build.invalid",
      NEXT_PUBLIC_OCR_HTTPONLY_SESSION: "false",
      OCR_API_KEY: "secret-must-stay-server-side",
    });
    expect(config).toEqual({ apiBaseUrl: "/runtime", httpOnlySession: true });
    expect(serializeRuntimeConfig(config)).not.toContain("secret");
  });

  it("uses same-origin defaults for a deployed server", () => {
    expect(publicRuntimeConfig({ NODE_ENV: "production" })).toEqual({ apiBaseUrl: "/", httpOnlySession: false });
  });

  it("uses the same runtime cookie mode during server rendering", () => {
    vi.stubGlobal("document", undefined);
    vi.stubEnv("OCR_FRONTEND_HTTPONLY_SESSION", "true");
    expect(isHttpOnlySessionEnabled()).toBe(true);
    vi.unstubAllGlobals();
  });

  it("configures HTTP, WebSocket and cookie mode consistently over baked defaults", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "http://build.invalid");
    vi.stubEnv("NEXT_PUBLIC_OCR_HTTPONLY_SESSION", "false");
    install({ apiBaseUrl: "https://runtime.example/ocr/", httpOnlySession: true });
    window.sessionStorage.setItem("ocr_local_api_key", "stale-key");
    expect(buildUrl("/api/v1/jobs")).toBe("https://runtime.example/ocr/api/v1/jobs");
    expect(buildJobWebSocketUrl("job-1")).toBe("wss://runtime.example/ocr/ws/jobs/job-1");
    expect(isHttpOnlySessionEnabled()).toBe(true);
    expect(getAuthRequestCredentials()).toBe("include");
    expect(getApiKey()).toBeNull();
    expect(window.sessionStorage.getItem("ocr_local_api_key")).toBeNull();
  });

  it("supports runtime cookie rollback and same-origin requests", () => {
    vi.stubEnv("NEXT_PUBLIC_OCR_HTTPONLY_SESSION", "true");
    install({ apiBaseUrl: "/", httpOnlySession: false });
    expect(buildUrl("/api/v1/jobs")).toBe("/api/v1/jobs");
    expect(getAuthRequestCredentials()).toBe("omit");
    expect(buildJobWebSocketUrl("job-1")).toBe(`${window.location.origin.replace(/^http/, "ws")}/ws/jobs/job-1`);
  });

  it("fails closed on a malformed injected config instead of downgrading cookie auth", () => {
    install({ apiBaseUrl: "/", httpOnlySession: "true" });
    expect(browserRuntimeConfig).toThrow("Invalid browser runtime configuration");
    expect(isHttpOnlySessionEnabled).toThrow();
  });

  it("escapes HTML script terminators and preserves their JSON value", () => {
    const config = { apiBaseUrl: "https://example.test/</script><script>alert(1)</script>", httpOnlySession: false };
    const result = serializeRuntimeConfig(config);
    expect(result).not.toContain("<");
    expect(JSON.parse(result)).toEqual(config);
  });

  it.each(["//evil.test", "javascript:alert(1)", "ftp://files.test", "https://user:pass@api.test", "/\\evil.test", "/api?q=x", "https://api.test/#fragment"])("rejects unsupported base %s", (base) => {
    expect(isValidApiBaseUrl(base)).toBe(false);
    expect(() => publicRuntimeConfig({ OCR_FRONTEND_API_BASE_URL: base })).toThrow();
  });

  it("rejects a misspelled auth mode", () => {
    expect(() => publicRuntimeConfig({ OCR_FRONTEND_HTTPONLY_SESSION: "ture" })).toThrow();
  });
});

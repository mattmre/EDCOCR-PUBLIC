import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  UnauthorizedError,
  apiClient,
  buildUrl,
  createTimeoutSignal,
  get,
  post,
  probeApiKey,
  put,
  resolveApiBaseUrl,
  resolveTimeoutMs,
} from "@/lib/api-client";
import { AUTH_STORAGE_KEY, getApiKey, setApiKey } from "@/lib/auth";
import { SETTINGS_STORAGE_KEY, getDefaultSettings, saveSettings } from "@/lib/settings-store";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("api-client", () => {
  const originalEnv = process.env.NEXT_PUBLIC_API_BASE_URL;
  const originalHttpOnlySession = process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION;

  beforeEach(() => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "false";
    window.sessionStorage.clear();
    setApiKey("test-key");
    window.localStorage.removeItem(SETTINGS_STORAGE_KEY);
    delete process.env.NEXT_PUBLIC_API_BASE_URL;
    vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.removeItem(SETTINGS_STORAGE_KEY);
    if (originalEnv === undefined) {
      delete process.env.NEXT_PUBLIC_API_BASE_URL;
    } else {
      process.env.NEXT_PUBLIC_API_BASE_URL = originalEnv;
    }
    if (originalHttpOnlySession === undefined) {
      delete process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION;
    } else {
      process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = originalHttpOnlySession;
    }
  });

  it("injects the X-API-Key header from sessionStorage in default mode", async () => {
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(200, { ok: true })
    );
    await get("/api/v1/health");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    const headers = init.headers as Headers;
    expect(headers.get("X-API-Key")).toBe("test-key");
    expect(init.credentials).toBe("omit");
  });

  it("omits X-API-Key when no key is stored", async () => {
    window.sessionStorage.removeItem(AUTH_STORAGE_KEY);
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(200, { ok: true })
    );
    await get("/api/v1/health");
    const headers = (fetchMock.mock.calls[0][1] as RequestInit).headers as Headers;
    expect(headers.has("X-API-Key")).toBe(false);
  });

  it("sends credentials and omits the readable key in HttpOnly mode", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, "stale-key");
    window.localStorage.setItem(AUTH_STORAGE_KEY, "legacy-key");
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(200, { ok: true })
    );

    await get("/api/v1/jobs");

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    const headers = init.headers as Headers;
    expect(init.credentials).toBe("include");
    expect(headers.has("X-API-Key")).toBe(false);
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it("clears the cached key and throws UnauthorizedError on 401", async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(401, { detail: "bad key" })
    );
    await expect(get("/api/v1/health")).rejects.toBeInstanceOf(UnauthorizedError);
    expect(getApiKey()).toBeNull();
  });

  it("preserves a valid key and throws ApiError on capability-denied 403", async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(403, { detail: "forbidden" })
    );
    await expect(get("/api/v1/jobs")).rejects.toMatchObject({
      name: "ApiError",
      status: 403,
      message: "forbidden",
    });
    expect(getApiKey()).toBe("test-key");
  });

  it("throws ApiError with detail on 500", async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(500, { detail: "boom" })
    );
    await expect(get("/api/v1/health")).rejects.toMatchObject({
      name: "ApiError",
      status: 500,
      message: "boom",
    });
    // Non-401 must NOT clear the key.
    expect(getApiKey()).toBe("test-key");
  });

  it("returns parsed JSON on 200", async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(200, { status: "ok" })
    );
    const body = await get<{ status: string }>("/api/v1/health");
    expect(body).toEqual({ status: "ok" });
  });

  it("sends Content-Type and serialized body on POST", async () => {
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(200, { ok: true })
    );
    await post("/api/v1/jobs", { foo: "bar" });
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    const headers = init.headers as Headers;
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(init.body).toBe(JSON.stringify({ foo: "bar" }));
  });

  it("omits ambient cookies in off mode for every shared request path", async () => {
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(200, { ok: true })
    );

    await get("/api/v1/health");
    await post("/api/v1/jobs", { foo: "bar" });
    await put("/api/v1/jobs/job-1", { foo: "bar" });
    await probeApiKey("/api/v1/auth/session/probe", "candidate-key");

    expect(fetchMock).toHaveBeenCalledTimes(4);
    for (const [, init] of fetchMock.mock.calls) {
      expect((init as RequestInit).credentials).toBe("omit");
    }
    expect(
      new Headers((fetchMock.mock.calls[3][1] as RequestInit).headers).get("X-API-Key")
    ).toBe("candidate-key");
  });

  it("preserves ApiError class for non-2xx without JSON body", async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response("plain text error", { status: 502 })
    );
    await expect(get("/api/v1/health")).rejects.toBeInstanceOf(ApiError);
  });

  describe("resolveApiBaseUrl / settings wire", () => {
    it("defaults to localhost when settings empty and env unset", () => {
      expect(resolveApiBaseUrl()).toBe("http://localhost:8000");
    });

    it("uses NEXT_PUBLIC_API_BASE_URL when settings baseUrl is empty", () => {
      process.env.NEXT_PUBLIC_API_BASE_URL = "https://env.example.com/";
      expect(resolveApiBaseUrl()).toBe("https://env.example.com");
    });

    it("prefers non-empty settings baseUrl over env", () => {
      process.env.NEXT_PUBLIC_API_BASE_URL = "https://env.example.com";
      const settings = getDefaultSettings();
      settings.api.baseUrl = "https://settings.example.com/";
      expect(saveSettings(settings)).toBe(true);
      expect(resolveApiBaseUrl()).toBe("https://settings.example.com");
      expect(apiClient.baseUrl).toBe("https://settings.example.com");
      expect(buildUrl("/api/v1/health")).toBe("https://settings.example.com/api/v1/health");
    });

    it("ignores invalid settings baseUrl and falls back to env", () => {
      process.env.NEXT_PUBLIC_API_BASE_URL = "https://env.example.com";
      // Bypass coerceSettings URL checks by writing raw invalid JSON shape that
      // coerce still accepts as a string (isValidApiBaseUrl rejects it at resolve).
      window.localStorage.setItem(
        SETTINGS_STORAGE_KEY,
        JSON.stringify({
          ...getDefaultSettings(),
          api: { baseUrl: "not a url", timeoutMs: 30000, apiKeyRedactedPreview: "" },
        })
      );
      expect(resolveApiBaseUrl()).toBe("https://env.example.com");
    });

    it("uses settings baseUrl on subsequent get() fetch calls", async () => {
      const settings = getDefaultSettings();
      settings.api.baseUrl = "https://settings-fetch.example.com";
      saveSettings(settings);
      const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
        jsonResponse(200, { ok: true })
      );
      await get("/api/v1/jobs");
      expect(fetchMock.mock.calls[0][0]).toBe("https://settings-fetch.example.com/api/v1/jobs");
    });
  });

  describe("resolveTimeoutMs / AbortSignal timeout", () => {
    it("defaults timeout to 30000", () => {
      expect(resolveTimeoutMs()).toBe(30000);
    });

    it("reads timeoutMs from settings", () => {
      const settings = getDefaultSettings();
      settings.api.timeoutMs = 45000;
      saveSettings(settings);
      expect(resolveTimeoutMs()).toBe(45000);
    });

    it("clamps out-of-range timeout values", () => {
      // coerceSettings already clamps on save; resolve also clamps for safety.
      const settings = getDefaultSettings();
      settings.api.timeoutMs = 45000;
      saveSettings(settings);
      expect(resolveTimeoutMs()).toBe(45000);

      // Direct storage write with an out-of-range value (coerce clamps on load).
      window.localStorage.setItem(
        SETTINGS_STORAGE_KEY,
        JSON.stringify({
          ...getDefaultSettings(),
          api: { baseUrl: "", timeoutMs: 5, apiKeyRedactedPreview: "" },
        })
      );
      // loadSettings → coerceSettings clamps 5 → 1000
      expect(resolveTimeoutMs()).toBe(1000);
    });

    it("attaches an AbortSignal to fetch", async () => {
      const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
        jsonResponse(200, { ok: true })
      );
      await get("/api/v1/health");
      const init = fetchMock.mock.calls[0][1] as RequestInit;
      expect(init.signal).toBeInstanceOf(AbortSignal);
      expect(init.signal?.aborted).toBe(false);
    });

    it("aborts when caller signal aborts", async () => {
      const controller = new AbortController();
      (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockImplementation(
        (_url: string, init?: RequestInit) => {
          return new Promise((_resolve, reject) => {
            const signal = init?.signal;
            if (signal?.aborted) {
              reject(new DOMException("Aborted", "AbortError"));
              return;
            }
            signal?.addEventListener("abort", () => {
              reject(new DOMException("Aborted", "AbortError"));
            });
          });
        }
      );
      const pending = get("/api/v1/health", { signal: controller.signal });
      controller.abort();
      await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    });

    it("createTimeoutSignal combines timeout with caller abort", () => {
      const settings = getDefaultSettings();
      settings.api.timeoutMs = 5000;
      saveSettings(settings);
      const caller = new AbortController();
      const { signal, cleanup } = createTimeoutSignal(caller.signal);
      expect(signal.aborted).toBe(false);
      caller.abort();
      expect(signal.aborted).toBe(true);
      cleanup();
    });

    it("createTimeoutSignal aborts after timeoutMs", () => {
      vi.useFakeTimers();
      try {
        const settings = getDefaultSettings();
        settings.api.timeoutMs = 2000;
        saveSettings(settings);
        const { signal, cleanup } = createTimeoutSignal();
        expect(signal.aborted).toBe(false);
        vi.advanceTimersByTime(1999);
        expect(signal.aborted).toBe(false);
        vi.advanceTimersByTime(1);
        expect(signal.aborted).toBe(true);
        cleanup();
      } finally {
        vi.useRealTimers();
      }
    });
  });
});

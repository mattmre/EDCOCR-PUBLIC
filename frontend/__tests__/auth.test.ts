import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AUTH_STATE_EVENT,
  AUTH_STORAGE_KEY,
  clearApiKey,
  clearHttpOnlySession,
  deriveReviewerLabel,
  getApiKey,
  getAuthRequestCredentials,
  hasValidHttpOnlySession,
  HttpOnlySessionError,
  isHttpOnlySessionEnabled,
  setApiKey,
  setApiKeyHttpOnly,
} from "@/lib/auth";

const originalHttpOnlySession = process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION;

function jsonResponse(status: number, body: unknown = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

beforeEach(() => {
  process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "false";
});

afterEach(() => {
  vi.restoreAllMocks();
  if (originalHttpOnlySession === undefined) {
    delete process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION;
  } else {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = originalHttpOnlySession;
  }
});

describe("auth helpers", () => {
  beforeEach(() => {
    clearApiKey();
    try {
      window.sessionStorage.clear();
      window.localStorage.clear();
    } catch {
      // ignore
    }
  });

  it("returns null when no key is stored", () => {
    expect(getApiKey()).toBeNull();
  });

  it("round-trips a key through sessionStorage (not long-lived localStorage)", () => {
    setApiKey("ocr-secret-key");
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBe("ocr-secret-key");
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(getApiKey()).toBe("ocr-secret-key");
  });

  it("trims whitespace before storing", () => {
    setApiKey("  ocr-trimmed  ");
    expect(getApiKey()).toBe("ocr-trimmed");
  });

  it("rejects empty keys", () => {
    expect(() => setApiKey("   ")).toThrow(/empty/i);
  });

  it("uses cookie credentials only when the HttpOnly flag is enabled", () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    expect(isHttpOnlySessionEnabled()).toBe(true);
    expect(getAuthRequestCredentials()).toBe("include");

    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "false";
    expect(isHttpOnlySessionEnabled()).toBe(false);
    expect(getAuthRequestCredentials()).toBe("omit");
  });

  it.each(["create", "validate", "logout"] as const)(
    "aborts a stalled HttpOnly %s request after its deadline",
    async (stage) => {
      vi.useFakeTimers();
      try {
        vi.spyOn(globalThis, "fetch").mockImplementation(
          (_input, init) =>
            new Promise((_resolve, reject) => {
              const signal = init?.signal;
              const abort = () => reject(new DOMException("Aborted", "AbortError"));
              if (signal?.aborted) {
                abort();
              } else {
                signal?.addEventListener("abort", abort, { once: true });
              }
            })
        );
        process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
        const request =
          stage === "create"
            ? setApiKeyHttpOnly("candidate-key", "https://api.example.com")
            : stage === "validate"
              ? hasValidHttpOnlySession("https://api.example.com")
              : clearHttpOnlySession("https://api.example.com");
        const rejected = expect(request).rejects.toMatchObject({ name: "AbortError" });

        await vi.advanceTimersByTimeAsync(30_001);
        await rejected;
      } finally {
        vi.useRealTimers();
      }
    }
  );

  it("does not retain a browser-readable key when HttpOnly mode is enabled", () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    window.localStorage.setItem(AUTH_STORAGE_KEY, "legacy-key");
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, "stale-key");

    setApiKey("cookie-backed-key");

    expect(getApiKey()).toBeNull();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it("validates a newly-created HttpOnly session before announcing success", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, "stale-key");
    const authEvents: boolean[] = [];
    const onAuthState = (event: Event) => {
      authEvents.push((event as CustomEvent<{ authenticated: boolean }>).detail.authenticated);
    };
    window.addEventListener(AUTH_STATE_EVENT, onAuthState);
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, { storage: "httpOnly_cookie" }))
      .mockResolvedValueOnce(jsonResponse(200, { jobs: [] }));

    await expect(
      setApiKeyHttpOnly("  cookie-backed-key  ", "https://api.example.com///")
    ).resolves.toBe(true);

    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.example.com/api/v1/auth/session",
      expect.objectContaining({
        method: "POST",
        credentials: "include",
        body: JSON.stringify({ api_key: "cookie-backed-key" }),
      })
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.example.com/api/v1/auth/session/probe",
      expect.objectContaining({ method: "GET", credentials: "include" })
    );
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(authEvents).toEqual([true]);
    window.removeEventListener(AUTH_STATE_EVENT, onAuthState);
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it("reports an invalid HttpOnly session response without storing the key", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(401, { error: "invalid_key" }));

    await expect(setApiKeyHttpOnly("bad-key", "https://api.example.com")).rejects.toEqual(
      expect.objectContaining({
        name: "HttpOnlySessionError",
        status: 401,
      }) satisfies Partial<HttpOnlySessionError>
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(getApiKey()).toBeNull();
  });

  it("preserves prior auth state when HttpOnly session creation is forbidden", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, "previous-key");
    const authEvents: boolean[] = [];
    const onAuthState = (event: Event) => {
      authEvents.push((event as CustomEvent<{ authenticated: boolean }>).detail.authenticated);
    };
    window.addEventListener(AUTH_STATE_EVENT, onAuthState);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(403, { error: "forbidden" }));

    await expect(setApiKeyHttpOnly("candidate-key", "https://api.example.com")).rejects.toMatchObject({
      name: "HttpOnlySessionError",
      status: 403,
      stage: "create",
    });

    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBe("previous-key");
    expect(authEvents).toEqual([]);
    window.removeEventListener(AUTH_STATE_EVENT, onAuthState);
  });

  it("rejects a cookie that fails the protected session probe", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    const authEvents: boolean[] = [];
    window.addEventListener(
      AUTH_STATE_EVENT,
      ((event: CustomEvent<{ authenticated: boolean }>) => {
        authEvents.push(event.detail.authenticated);
      }) as EventListener,
      { once: true }
    );
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, { storage: "httpOnly_cookie" }))
      .mockResolvedValueOnce(jsonResponse(401, { error: "invalid_api_key" }));

    await expect(setApiKeyHttpOnly("revoked-key", "https://api.example.com")).rejects.toMatchObject({
      name: "HttpOnlySessionError",
      status: 401,
      stage: "validate",
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(authEvents).toEqual([false]);
  });

  it("preserves auth state when protected session validation is forbidden", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, "legacy-key");
    const authEvents: boolean[] = [];
    const onAuthState = (event: Event) => {
      authEvents.push((event as CustomEvent<{ authenticated: boolean }>).detail.authenticated);
    };
    window.addEventListener(AUTH_STATE_EVENT, onAuthState);
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, { storage: "httpOnly_cookie" }))
      .mockResolvedValueOnce(jsonResponse(403, { error: "origin_forbidden" }));

    await expect(setApiKeyHttpOnly("accepted-key", "https://api.example.com")).rejects.toMatchObject({
      name: "HttpOnlySessionError",
      status: 403,
      stage: "validate",
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(authEvents).toEqual([]);
    // Cookie mode clears any legacy browser-readable key after session creation.
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    window.removeEventListener(AUTH_STATE_EVENT, onAuthState);
  });

  it("does not interpret a forbidden protected probe as an invalid cookie", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(403, { error: "forbidden" }));

    await expect(hasValidHttpOnlySession("https://api.example.com")).rejects.toMatchObject({
      name: "HttpOnlySessionError",
      status: 403,
      stage: "validate",
    });
  });

  it("validates and clears an HttpOnly session using credentialed requests", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, { jobs: [] }))
      .mockResolvedValueOnce(jsonResponse(200, { cleared: true }));

    await expect(hasValidHttpOnlySession("https://api.example.com")).resolves.toBe(true);
    await expect(clearHttpOnlySession("https://api.example.com")).resolves.toBeUndefined();
    expect(fetchMock.mock.calls[0]).toEqual([
      "https://api.example.com/api/v1/auth/session/probe",
      expect.objectContaining({ method: "GET", credentials: "include" }),
    ]);
    expect(fetchMock.mock.calls[1]).toEqual([
      "https://api.example.com/api/v1/auth/session",
      expect.objectContaining({ method: "DELETE", credentials: "include" }),
    ]);
  });

  it("migrates legacy localStorage blank values as missing", () => {
    window.localStorage.setItem(AUTH_STORAGE_KEY, "   ");
    expect(getApiKey()).toBeNull();
  });

  it("migrates legacy localStorage key into sessionStorage", () => {
    window.localStorage.setItem(AUTH_STORAGE_KEY, "legacy-key-value");
    expect(getApiKey()).toBe("legacy-key-value");
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBe("legacy-key-value");
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it("clears the stored key from both stores", () => {
    setApiKey("ocr-clear-me");
    clearApiKey();
    expect(getApiKey()).toBeNull();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });
});

describe("deriveReviewerLabel", () => {
  it("prefers an explicit non-blank reviewer label", () => {
    setApiKey("long-enough-api-key-value");
    expect(deriveReviewerLabel("alice@example.com")).toBe("alice@example.com");
    expect(deriveReviewerLabel("  bob  ")).toBe("bob");
  });

  it("uses a redacted key preview when no explicit label is given", () => {
    setApiKey("sk-live-abcdefghijklmnop");
    expect(deriveReviewerLabel()).toBe("sk-l…mnop");
    expect(deriveReviewerLabel("")).toBe("sk-l…mnop");
    expect(deriveReviewerLabel("   ")).toBe("sk-l…mnop");
  });

  it("falls back to 'operator' when key is missing or too short to redact", () => {
    clearApiKey();
    expect(deriveReviewerLabel()).toBe("operator");
    expect(deriveReviewerLabel(null)).toBe("operator");
    expect(deriveReviewerLabel("")).toBe("operator");

    setApiKey("short"); // length <= 8 — would leak if previewed
    expect(deriveReviewerLabel()).toBe("operator");
  });

  it("never returns an empty string", () => {
    clearApiKey();
    expect(deriveReviewerLabel()).not.toBe("");
    expect(deriveReviewerLabel("   ")).not.toBe("");
    setApiKey("test-key"); // 8 chars — short key path
    expect(deriveReviewerLabel()).not.toBe("");
  });
});

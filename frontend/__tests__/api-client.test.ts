import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, UnauthorizedError, get, post } from "@/lib/api-client";
import { AUTH_STORAGE_KEY, getApiKey, setApiKey } from "@/lib/auth";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("api-client", => {
  beforeEach(() => {
    setApiKey("test-key");
    vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("injects the X-API-Key header from localStorage", async => {
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(200, { ok: true })
    );
    await get("/api/v1/health");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    const headers = init.headers as Headers;
    expect(headers.get("X-API-Key")).toBe("test-key");
  });

  it("omits X-API-Key when no key is stored", async => {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(200, { ok: true })
    );
    await get("/api/v1/health");
    const headers = (fetchMock.mock.calls[0][1] as RequestInit).headers as Headers;
    expect(headers.has("X-API-Key")).toBe(false);
  });

  it("clears the cached key and throws UnauthorizedError on 401", async => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(401, { detail: "bad key" })
    );
    await expect(get("/api/v1/health")).rejects.toBeInstanceOf(UnauthorizedError);
    expect(getApiKey()).toBeNull();
  });

  it("clears the cached key and throws on 403", async => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(403, { detail: "forbidden" })
    );
    await expect(get("/api/v1/health")).rejects.toBeInstanceOf(UnauthorizedError);
    expect(getApiKey()).toBeNull();
  });

  it("throws ApiError with detail on 500", async => {
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

  it("returns parsed JSON on 200", async => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(200, { status: "ok" })
    );
    const body = await get<{ status: string }>("/api/v1/health");
    expect(body).toEqual({ status: "ok" });
  });

  it("sends Content-Type and serialized body on POST", async => {
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(200, { ok: true })
    );
    await post("/api/v1/jobs", { foo: "bar" });
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    const headers = init.headers as Headers;
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(init.body).toBe(JSON.stringify({ foo: "bar" }));
  });

  it("preserves ApiError class for non-2xx without JSON body", async => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response("plain text error", { status: 502 })
    );
    await expect(get("/api/v1/health")).rejects.toBeInstanceOf(ApiError);
  });
});

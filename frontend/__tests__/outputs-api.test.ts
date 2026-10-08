import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, UnauthorizedError } from "@/lib/api-client";
import { getApiKey, setApiKey } from "@/lib/auth";
import { downloadAuthenticated, parseContentDispositionFilename } from "@/lib/outputs-api";

describe("parseContentDispositionFilename", () => {
  it("parses quoted filename", () => {
    expect(parseContentDispositionFilename('attachment; filename="report.pdf"')).toBe("report.pdf");
  });

  it("parses RFC 5987 filename*", () => {
    expect(parseContentDispositionFilename("attachment; filename*=UTF-8''job%20bundle.zip")).toBe(
      "job bundle.zip"
    );
  });

  it("returns null for missing header", () => {
    expect(parseContentDispositionFilename(null)).toBeNull();
  });
});

describe("downloadAuthenticated", () => {
  const originalFetch = global.fetch;
  const originalCreateObjectURL = URL.createObjectURL;
  const originalRevokeObjectURL = URL.revokeObjectURL;

  beforeEach(() => {
    setApiKey("download-key");
    URL.createObjectURL = vi.fn(() => "blob:mock-url");
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    URL.createObjectURL = originalCreateObjectURL;
    URL.revokeObjectURL = originalRevokeObjectURL;
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("fetches with X-API-Key and triggers a temporary anchor download", async () => {
    global.fetch = vi.fn(
      async () =>
        new Response(new Blob(["payload"]), {
          status: 200,
          headers: {
            "content-type": "application/octet-stream",
            "content-disposition": 'attachment; filename="artifact.bin"',
          },
        })
    ) as unknown as typeof fetch;

    // Avoid jsdom "navigation not implemented" noise on real anchor.click().
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    const appendSpy = vi.spyOn(document.body, "appendChild");
    const removeSpy = vi.spyOn(document.body, "removeChild");

    await downloadAuthenticated("http://api.test/api/v1/jobs/j1/outputs/ocr_text", "fallback.bin");

    expect(global.fetch).toHaveBeenCalledWith(
      "http://api.test/api/v1/jobs/j1/outputs/ocr_text",
      expect.objectContaining({ method: "GET", cache: "no-store" })
    );
    const call = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    const headers = new Headers(call[1].headers as HeadersInit);
    expect(headers.get("X-API-Key")).toBe("download-key");
    expect(URL.createObjectURL).toHaveBeenCalled();
    expect(clickSpy).toHaveBeenCalled();
    expect(appendSpy).toHaveBeenCalled();
    expect(removeSpy).toHaveBeenCalled();

    const anchor = appendSpy.mock.calls.find(([node]) => node instanceof HTMLAnchorElement)?.[0] as
      | HTMLAnchorElement
      | undefined;
    expect(anchor).toBeDefined();
    expect(anchor?.download).toBe("artifact.bin");
    expect(anchor?.href).toContain("blob:mock-url");
  });

  it("uses fallback filename when Content-Disposition is absent", async () => {
    global.fetch = vi.fn(
      async () =>
        new Response(new Blob(["payload"]), {
          status: 200,
          headers: { "content-type": "application/octet-stream" },
        })
    ) as unknown as typeof fetch;

    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    const appendSpy = vi.spyOn(document.body, "appendChild");
    await downloadAuthenticated("http://api.test/file", "fallback.bin");

    const anchor = appendSpy.mock.calls.find(([node]) => node instanceof HTMLAnchorElement)?.[0] as
      | HTMLAnchorElement
      | undefined;
    expect(anchor?.download).toBe("fallback.bin");
  });

  it("clears API key and throws UnauthorizedError on 401", async () => {
    global.fetch = vi.fn(
      async () => new Response("nope", { status: 401 })
    ) as unknown as typeof fetch;

    await expect(downloadAuthenticated("http://api.test/file")).rejects.toBeInstanceOf(
      UnauthorizedError
    );
    expect(getApiKey()).toBeNull();
  });

  it("preserves API key and throws ApiError on capability-denied 403", async () => {
    global.fetch = vi.fn(
      async () => new Response("nope", { status: 403 })
    ) as unknown as typeof fetch;

    await expect(downloadAuthenticated("http://api.test/file")).rejects.toMatchObject({
      name: "ApiError",
      status: 403,
    });
    expect(getApiKey()).toBe("download-key");
  });

  it("throws ApiError on non-auth failures", async () => {
    global.fetch = vi.fn(
      async () =>
        new Response(JSON.stringify({ detail: "missing_output" }), {
          status: 404,
          headers: { "content-type": "application/json" },
        })
    ) as unknown as typeof fetch;

    await expect(downloadAuthenticated("http://api.test/file")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      message: "missing_output",
    } satisfies Partial<ApiError>);
    expect(getApiKey()).toBe("download-key");
  });
});

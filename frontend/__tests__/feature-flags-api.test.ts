import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api-client";
import {
  getFlag,
  getFlagHistory,
  listFlags,
  submitChangeRequest,
} from "@/lib/feature-flags-api";
import { setApiKey } from "@/lib/auth";
import type { FeatureFlag } from "@/lib/types";

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const flagFixture: FeatureFlag = {
  key: "ENABLE_TRANSLATION",
  category: "translation",
  value_type: "boolean",
  current_value: false,
  default_value: false,
  source: "default",
  description: "Master toggle for the translation pipeline.",
  requires_strong_auth: true,
  requires_bake_hours: 48,
};

describe("feature-flags-api", => {
  beforeEach(() => {
    setApiKey("test-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("listFlags normalizes a bare-array response", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(json(200, [flagFixture]));
    const out = await listFlags();
    expect(out).toEqual([flagFixture]);
  });

  it("listFlags normalizes a {flags:[]} envelope", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      json(200, { flags: [flagFixture] })
    );
    const out = await listFlags();
    expect(out).toEqual([flagFixture]);
  });

  it("getFlag returns null on 404", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      json(404, { detail: "not found" })
    );
    const out = await getFlag("UNKNOWN_FLAG");
    expect(out).toBeNull();
  });

  it("getFlag rethrows non-404 errors", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(json(500, { detail: "boom" }));
    await expect(getFlag("X")).rejects.toBeInstanceOf(ApiError);
  });

  it("getFlagHistory normalizes envelope and array shapes", async => {
    const entry = {
      request_id: "req_1",
      flag_key: "ENABLE_TRANSLATION",
      previous_value: false,
      new_value: true,
      reason: "test reason long enough",
      requested_by: "ops@example.com",
      requested_at: "2026-04-25T10:00:00Z",
      status: "pending" as const,
    };
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock.mockResolvedValueOnce(
      json(200, { entries: [entry], total: 1 })
    );
    const wrapped = await getFlagHistory("ENABLE_TRANSLATION");
    expect(wrapped).toEqual([entry]);

    fetchMock.mockResolvedValueOnce(json(200, [entry]));
    const bare = await getFlagHistory("ENABLE_TRANSLATION");
    expect(bare).toEqual([entry]);
  });

  it("submitChangeRequest posts to the change-request endpoint with the flag-key in the path", async => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      json(202, {
        request_id: "req_2",
        flag_key: "ENABLE_TRANSLATION",
        previous_value: false,
        new_value: true,
        reason: "valid reason text",
        requested_by: "ops@example.com",
        requested_at: "2026-04-25T10:00:00Z",
        status: "pending",
      })
    );
    const out = await submitChangeRequest("ENABLE_TRANSLATION", {
      flag_key: "ENABLE_TRANSLATION",
      new_value: true,
      reason: "valid reason text",
      auth_method: "piv_cac",
      auth_token: "tok",
    });
    expect(out.status).toBe("pending");
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(String(url)).toMatch(
      /\/api\/v1\/admin\/feature-flags\/ENABLE_TRANSLATION\/change-request$/
    );
    expect((init as RequestInit).method).toBe("POST");
    const body = JSON.parse(String((init as RequestInit).body));
    expect(body).toMatchObject({
      flag_key: "ENABLE_TRANSLATION",
      new_value: true,
      reason: "valid reason text",
      auth_method: "piv_cac",
    });
  });

  it("submitChangeRequest surfaces ApiError on 403", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      json(403, { detail: "strong auth", error_code: "strong_auth_required" })
    );
    await expect(
      submitChangeRequest("ENABLE_TRANSLATION", {
        flag_key: "ENABLE_TRANSLATION",
        new_value: true,
        reason: "x".repeat(30),
      })
    ).rejects.toBeInstanceOf(ApiError);
  });
});

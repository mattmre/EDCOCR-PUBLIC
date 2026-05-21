import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  listAlerts,
  listChannels,
  listRules,
  getRule,
  muteAlert,
  unmuteAlert,
  testChannel,
  updateRuleThreshold,
} from "@/lib/alerts-api";
import { ApiError, UnauthorizedError } from "@/lib/api-client";
import { setApiKey, getApiKey } from "@/lib/auth";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("alerts-api", => {
  beforeEach(() => {
    setApiKey("admin-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("listAlerts GETs /admin/alerts and returns the parsed body", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, [{ id: "a1", rule_id: "r1" }]));

    const out = await listAlerts();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = String(fetchMock.mock.calls[0][0]);
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(url).toContain("/api/v1/admin/alerts");
    expect(init.method).toBe("GET");
    const headers = init.headers as Headers;
    expect(headers.get("X-API-Key")).toBe("admin-key");
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ id: "a1", rule_id: "r1" });
  });

  it("listRules GETs /admin/alerts/rules", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, [{ id: "r1", name: "queue depth" }]));

    await listRules();

    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("/api/v1/admin/alerts/rules");
  });

  it("getRule URL-encodes the rule id", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, { id: "r 1", name: "x" }));

    await getRule("r 1");

    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("/api/v1/admin/alerts/rules/r%201");
  });

  it("updateRuleThreshold PATCHes the operator-editable subset", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, { id: "r1", threshold_value: 200 }));

    await updateRuleThreshold("r1", { threshold_value: 200, enabled: false });

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("PATCH");
    const headers = init.headers as Headers;
    expect(headers.get("Content-Type")).toBe("application/json");
    const body = JSON.parse(String(init.body));
    expect(body).toEqual({ threshold_value: 200, enabled: false });
  });

  it("muteAlert POSTs the reason payload to /mute", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, { id: "a1", state: "muted" }));

    await muteAlert("a1", { reason: "investigating" });

    const url = String(fetchMock.mock.calls[0][0]);
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(url).toContain("/api/v1/admin/alerts/a1/mute");
    expect(init.method).toBe("POST");
    const body = JSON.parse(String(init.body));
    expect(body).toEqual({ reason: "investigating" });
  });

  it("unmuteAlert POSTs to /unmute", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, { id: "a1", state: "firing" }));

    await unmuteAlert("a1");

    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("/api/v1/admin/alerts/a1/unmute");
  });

  it("listChannels GETs /admin/alert-channels", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, []));

    await listChannels();

    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("/api/v1/admin/alert-channels");
  });

  it("testChannel POSTs to the channel-specific test endpoint", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        jsonResponse(200, { ok: true, tested_at: "2026-04-27T00:00:00Z" })
      );

    const result = await testChannel("ch1");

    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("/api/v1/admin/alert-channels/ch1/test");
    expect(result.ok).toBe(true);
  });

  it("does not mask 403 responses (page renders access-denied)", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(403, { detail: "platform admin required" })
    );

    let caught: unknown = null;
    try {
      await listAlerts();
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(ApiError);
    expect((caught as ApiError).status).toBe(403);
    // The 401 path clears the cached key; 403 must NOT.
    expect(getApiKey()).toBe("admin-key");
    // It is NOT an UnauthorizedError -- the page differentiates.
    expect(caught).not.toBeInstanceOf(UnauthorizedError);
  });

  it("treats 404 like other non-OK responses (page handles it)", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(404, { detail: "not provisioned" })
    );

    let caught: unknown = null;
    try {
      await listAlerts();
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(ApiError);
    expect((caught as ApiError).status).toBe(404);
  });
});

const { test, expect } = require("@playwright/test");
const { hasEnv } = require("../../helpers/runtime");
const {
  expectJobSubmitResponse,
  submitJobUpload,
} = require("../../helpers/request");
const {
  buildWebSocketUrl,
  collectWebSocketMessages,
} = require("../../helpers/websocket");

function configuredApiKey() {
  const raw = process.env.PLAYWRIGHT_OCR_API_KEY || "";
  return raw.trim() || null;
}

test.describe("OCR job websocket coverage", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_API_BASE_URL"),
    "Set PLAYWRIGHT_API_BASE_URL to run browser WebSocket coverage.");

  test("@smoke websocket connects to a submitted job and handles ping/pong", async ({ page, request, baseURL }) => {
    const submitResponse = await submitJobUpload(request, baseURL);
    const submitted = await expectJobSubmitResponse(submitResponse);

    await page.goto("about:blank");
    const messages = await collectWebSocketMessages(page, {
      wsUrl: buildWebSocketUrl(baseURL, `/ws/jobs/${submitted.job_id}`),
      apiKey: configuredApiKey(),
      sendPingOnConnected: true,
      stopOnTypes: ["pong", "completed", "failed", "cancelled"],
      timeoutMs: 10_000,
    });

    expect(messages.length).toBeGreaterThan(0);
    expect(messages[0].type).toBe("connected");
    expect(messages[0].job_id).toBe(submitted.job_id);
    expect(messages[0].status).toMatch(/submitted|processing|completed|failed|cancelled/);
    expect(
      messages.some((message) =>
        ["pong", "completed", "failed", "cancelled"].includes(message.type))).toBeTruthy();
  });

  test("@smoke websocket returns a structured error for missing jobs", async ({ page, baseURL }) => {
    await page.goto("about:blank");
    const messages = await collectWebSocketMessages(page, {
      wsUrl: buildWebSocketUrl(baseURL, "/ws/jobs/playwright-missing-job"),
      apiKey: configuredApiKey(),
      sendPingOnConnected: false,
      stopOnTypes: ["error"],
      timeoutMs: 10_000,
    });

    expect(messages.length).toBeGreaterThan(0);
    expect(messages[0].type).toBe("error");
    expect(messages[0].message).toMatch(/not found/i);
  });
});

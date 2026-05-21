const { test, expect } = require("@playwright/test");
const { buildApiHeaders, hasEnv } = require("../../helpers/runtime");

test.describe("FastAPI health endpoint", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_API_BASE_URL"),
    "Set PLAYWRIGHT_API_BASE_URL to run request-level API smoke tests.");

  test("@smoke returns a healthy status payload", async ({ request, baseURL }) => {
    const response = await request.get(`${baseURL}/api/v1/health`, {
      headers: buildApiHeaders(),
    });

    expect(response.ok()).toBeTruthy();

    const payload = await response.json();
    expect(payload.status).toBe("healthy");
    expect(typeof payload.version).toBe("string");
    expect(typeof payload.uptime_seconds).toBe("number");
    expect(payload.jobs).toBeTruthy();
  });
});

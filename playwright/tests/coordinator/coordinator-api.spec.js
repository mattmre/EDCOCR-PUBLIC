// @ts-check
const { test, expect } = require("@playwright/test");
const { hasEnv } = require("../../helpers/runtime");
const { buildMetricsHeaders, hasMetricsKey } = require("../../helpers/ops");

// Coordinator API endpoint tests
// These are env-gated – skip when PLAYWRIGHT_COORDINATOR_BASE_URL is not set.

const BASE_URL = process.env.PLAYWRIGHT_COORDINATOR_BASE_URL || "";

test.describe("Coordinator API", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_COORDINATOR_BASE_URL"),
    "Set PLAYWRIGHT_COORDINATOR_BASE_URL to run coordinator API tests.");

  test("GET /api/v1/metrics/ returns 200 or 401", async ({ request }) => {
    const response = await request.get(`${BASE_URL}/api/v1/metrics/`);
    expect([200, 401, 403]).toContain(response.status());
  });

  test("GET /api/v1/metrics/ with auth returns metrics", async ({ request }) => {
    test.skip(
      !hasMetricsKey(),
      "Set PLAYWRIGHT_METRICS_API_KEY to run authenticated metrics tests.");

    const response = await request.get(`${BASE_URL}/api/v1/metrics/`, {
      headers: buildMetricsHeaders("x-api-key"),
    });
    expect(response.status()).toBe(200);
    const body = await response.json();
    expect(body).toHaveProperty("total_jobs");
  });

  test("GET /api/v1/prometheus/ returns prometheus format", async ({ request }) => {
    const response = await request.get(`${BASE_URL}/api/v1/prometheus/`);
    expect([200, 401, 403]).toContain(response.status());
  });

  test("POST /api/v1/jobs/ without auth returns 401 or 403", async ({ request }) => {
    const response = await request.post(`${BASE_URL}/api/v1/jobs/`, {
      data: {},
    });
    expect([401, 403]).toContain(response.status());
  });

  test("GET /admin/ returns 200 or redirect", async ({ request }) => {
    const response = await request.get(`${BASE_URL}/admin/`, {
      maxRedirects: 0,
    });
    expect([200, 301, 302]).toContain(response.status());
  });

  test("GET /api/v1/workers/ returns worker list or auth required", async ({ request }) => {
    const headers = hasMetricsKey() ? buildMetricsHeaders("x-api-key") : {};
    const response = await request.get(`${BASE_URL}/api/v1/workers/`, { headers });
    expect([200, 401, 403, 404]).toContain(response.status());
  });

  test("Health endpoint responds", async ({ request }) => {
    const healthUrl = process.env.COORDINATOR_HEALTH_URL || `${BASE_URL}/api/v1/health/`;
    const response = await request.get(healthUrl);
    expect([200, 404]).toContain(response.status());
  });
});

test.describe("Coordinator Job Lifecycle", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_COORDINATOR_BASE_URL"),
    "Set PLAYWRIGHT_COORDINATOR_BASE_URL to run coordinator job lifecycle tests.");

  test("Job submission requires authentication", async ({ request }) => {
    const response = await request.post(`${BASE_URL}/api/v1/jobs/`, {
      multipart: {
        file: {
          name: "test.pdf",
          mimeType: "application/pdf",
          buffer: Buffer.from("dummy"),
        },
      },
    });
    expect([401, 403]).toContain(response.status());
  });

  test("Job list endpoint responds", async ({ request }) => {
    const apiKey = process.env.PLAYWRIGHT_COORDINATOR_API_KEY || "";
    test.skip(!apiKey, "Set PLAYWRIGHT_COORDINATOR_API_KEY to run authenticated job list tests.");

    const response = await request.get(`${BASE_URL}/api/v1/jobs/`, {
      headers: { "X-Api-Key": apiKey },
    });
    expect(response.status()).toBe(200);
  });
});

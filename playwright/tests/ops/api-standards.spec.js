const { test, expect } = require("@playwright/test");
const { hasEnv } = require("../../helpers/runtime");
const {
  buildMetricsHeaders,
  buildPrimaryApiHeaders,
  getExpectedFlag,
  hasMetricsKey,
  hasPrimaryApiKey,
} = require("../../helpers/ops");
const {
  expectBatchSubmitResponse,
  expectJobSubmitResponse,
  submitBatchUpload,
  submitJobUpload,
} = require("../../helpers/request");

const API_AUTH_REQUIRED = getExpectedFlag("PLAYWRIGHT_EXPECT_API_AUTH_REQUIRED");
const METRICS_AUTH_REQUIRED = getExpectedFlag("PLAYWRIGHT_EXPECT_METRICS_AUTH_REQUIRED");

test.describe("Phase 8 operational standards", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_API_BASE_URL"),
    "Set PLAYWRIGHT_API_BASE_URL to run Phase 8 API operational tests.");

  test.skip(
    API_AUTH_REQUIRED === null,
    "Set PLAYWRIGHT_EXPECT_API_AUTH_REQUIRED to declare the local API auth posture.");

  test("FastAPI exempt and protected paths reflect expected auth policy", async ({ request, baseURL }) => {
    const healthResponse = await request.get(`${baseURL}/api/v1/health`);
    expect(healthResponse.status()).toBe(200);

    const docsResponse = await request.get(`${baseURL}/docs`);
    expect(docsResponse.status()).toBe(200);

    const openApiResponse = await request.get(`${baseURL}/openapi.json`);
    expect(openApiResponse.status()).toBe(200);

    const redocResponse = await request.get(`${baseURL}/redoc`);
    expect(redocResponse.status()).toBe(200);

    const jobsResponse = await request.get(`${baseURL}/api/v1/jobs`);
    if (API_AUTH_REQUIRED) {
      expect(jobsResponse.status()).toBe(401);
      const payload = await jobsResponse.json();
      expect(payload.error).toBe("unauthorized");
      return;
    }

    expect(jobsResponse.status()).toBe(200);
  });

  test("job submit preserves processing timeout override", async ({ request, baseURL }) => {
    test.skip(
      !hasPrimaryApiKey() && API_AUTH_REQUIRED,
      "Set PLAYWRIGHT_OCR_API_KEY or PLAYWRIGHT_FEATURE_FLAGS_API_KEY for auth-required timeout coverage.");

    const submitResponse = await submitJobUpload(request, baseURL, {
      processing_timeout_minutes: "7",
    }, {
      headers: buildPrimaryApiHeaders(),
    });
    const submitted = await expectJobSubmitResponse(submitResponse);

    const statusResponse = await request.get(`${baseURL}/api/v1/jobs/${submitted.job_id}`, {
      headers: buildPrimaryApiHeaders(),
    });
    expect(statusResponse.status()).toBe(200);
    const payload = await statusResponse.json();
    expect(payload.settings.processing_timeout_minutes).toBe(7);
  });

  test("batch submit preserves processing timeout override", async ({ request, baseURL }) => {
    test.skip(
      !hasPrimaryApiKey() && API_AUTH_REQUIRED,
      "Set PLAYWRIGHT_OCR_API_KEY or PLAYWRIGHT_FEATURE_FLAGS_API_KEY for auth-required timeout coverage.");

    const submitResponse = await submitBatchUpload(request, baseURL, {
      processing_timeout_minutes: "9",
    }, {
      headers: buildPrimaryApiHeaders(),
    });
    const submitted = await expectBatchSubmitResponse(submitResponse);

    const statusResponse = await request.get(
      `${baseURL}/api/v1/jobs/batch/${submitted.batch_id}`,
      { headers: buildPrimaryApiHeaders() });
    expect(statusResponse.status()).toBe(200);
    const payload = await statusResponse.json();
    expect(payload.settings.processing_timeout_minutes).toBe(9);
  });
});

test.describe("Phase 8 coordinator metrics auth", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_COORDINATOR_BASE_URL"),
    "Set PLAYWRIGHT_COORDINATOR_BASE_URL to run coordinator operational tests.");

  test.skip(
    METRICS_AUTH_REQUIRED === null,
    "Set PLAYWRIGHT_EXPECT_METRICS_AUTH_REQUIRED to declare the local coordinator metrics auth posture.");

  test("metrics endpoint reflects expected auth mode", async ({ request }) => {
    const metricsUrl = `${process.env.PLAYWRIGHT_COORDINATOR_BASE_URL}/api/v1/metrics/`;
    const unauthenticatedResponse = await request.get(metricsUrl);

    if (!METRICS_AUTH_REQUIRED) {
      expect(unauthenticatedResponse.status()).toBe(200);
      const payload = await unauthenticatedResponse.json();
      expect(payload.jobs).toBeTruthy();
      expect(payload.workers).toBeTruthy();
      return;
    }

    test.skip(
      !hasMetricsKey(),
      "Set PLAYWRIGHT_METRICS_API_KEY for metrics auth-required coverage.");

    expect(unauthenticatedResponse.status()).toBe(401);
    expect((await unauthenticatedResponse.json()).error).toBe("Unauthorized");

    const xApiKeyResponse = await request.get(metricsUrl, {
      headers: buildMetricsHeaders("x-api-key"),
    });
    expect(xApiKeyResponse.status()).toBe(200);
    expect((await xApiKeyResponse.json()).timestamp).toBeTruthy();

    const bearerResponse = await request.get(metricsUrl, {
      headers: buildMetricsHeaders("bearer"),
    });
    expect(bearerResponse.status()).toBe(200);
    expect((await bearerResponse.json()).pages).toBeTruthy();
  });

  test("prometheus endpoint reflects expected auth mode", async ({ request }) => {
    const prometheusUrl = `${process.env.PLAYWRIGHT_COORDINATOR_BASE_URL}/api/v1/prometheus/`;
    const unauthenticatedResponse = await request.get(prometheusUrl);

    if (!METRICS_AUTH_REQUIRED) {
      expect(unauthenticatedResponse.status()).toBe(200);
      expect(unauthenticatedResponse.headers()["content-type"]).toContain("text/plain");
      return;
    }

    test.skip(
      !hasMetricsKey(),
      "Set PLAYWRIGHT_METRICS_API_KEY for Prometheus auth-required coverage.");

    expect(unauthenticatedResponse.status()).toBe(401);
    expect((await unauthenticatedResponse.json()).error).toBe("Unauthorized");

    const xApiKeyResponse = await request.get(prometheusUrl, {
      headers: buildMetricsHeaders("x-api-key"),
    });
    expect(xApiKeyResponse.status()).toBe(200);
    expect(xApiKeyResponse.headers()["content-type"]).toContain("text/plain");

    const bearerResponse = await request.get(prometheusUrl, {
      headers: buildMetricsHeaders("bearer"),
    });
    expect(bearerResponse.status()).toBe(200);
    expect(await bearerResponse.text()).toContain("#");
  });
});

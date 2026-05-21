const { test, expect } = require("@playwright/test");
const { hasEnv } = require("../../helpers/runtime");
const {
  buildFeatureFlagHeaders,
  buildServerOutputPath,
  getExpectedFlag,
  getFeatureInputPath,
  hasFeatureExecutionPaths,
} = require("../../helpers/feature-flags");

const TRANSFORMS_ENABLED = getExpectedFlag("PLAYWRIGHT_EXPECT_TRANSFORMS_ENABLED");
const STAMPING_ENABLED = getExpectedFlag("PLAYWRIGHT_EXPECT_STAMPING_ENABLED");

test.describe("Phase 7 transform and stamp feature matrix", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_API_BASE_URL"),
    "Set PLAYWRIGHT_API_BASE_URL to run feature-flag request tests.");

  test.describe("transforms", => {
    test.skip(
      TRANSFORMS_ENABLED === null,
      "Set PLAYWRIGHT_EXPECT_TRANSFORMS_ENABLED to declare the local transform matrix state.");

    test("transform list reflects expected flag state", async ({ request, baseURL }) => {
      const response = await request.get(`${baseURL}/api/v1/transforms`);

      if (!TRANSFORMS_ENABLED) {
        expect(response.status()).toBe(403);
        const payload = await response.json();
        expect(payload.detail.error).toBe("feature_disabled");
        return;
      }

      expect(response.status()).toBe(200);
      const payload = await response.json();
      expect(Array.isArray(payload.operations)).toBeTruthy();
      expect(payload.total).toBeGreaterThan(0);
      expect(payload.operations.some((operation) => operation.name === "pdf_extract")).toBeTruthy();
    });

    test("transform metadata reflects expected flag state", async ({ request, baseURL }) => {
      const response = await request.get(`${baseURL}/api/v1/transforms/pdf_extract`);

      if (!TRANSFORMS_ENABLED) {
        expect(response.status()).toBe(403);
        const payload = await response.json();
        expect(payload.detail.error).toBe("feature_disabled");
        return;
      }

      expect(response.status()).toBe(200);
      const payload = await response.json();
      expect(payload.name).toBe("pdf_extract");
      expect(payload.supported_formats).toContain("pdf");
    });

    test("transform execute succeeds when enabled with explicit auth and server paths", async ({ request, baseURL }) => {
      test.skip(!TRANSFORMS_ENABLED, "Transform execution only applies when transforms are enabled.");
      test.skip(
        Object.keys(buildFeatureFlagHeaders()).length === 0,
        "Set PLAYWRIGHT_FEATURE_FLAGS_API_KEY or PLAYWRIGHT_OCR_API_KEY for transform execution coverage.");
      test.skip(
        !hasFeatureExecutionPaths(),
        "Set PLAYWRIGHT_FEATURE_SAMPLE_INPUT_PATH and PLAYWRIGHT_FEATURE_OUTPUT_DIR for transform execution coverage.");

      const outputPath = buildServerOutputPath("playwright-transform");
      const response = await request.post(`${baseURL}/api/v1/transforms/execute`, {
        headers: buildFeatureFlagHeaders(),
        data: {
          operation_id: "pdf_extract",
          input_path: getFeatureInputPath(),
          output_path: outputPath,
          params: { pages: [1] },
        },
      });

      expect(response.status()).toBe(200);
      const payload = await response.json();
      expect(payload.success).toBeTruthy();
      expect(payload.operation_id).toBe("pdf_extract");
      expect(payload.output_path).toBe(outputPath);
      expect(payload.pages_processed).toBeGreaterThanOrEqual(1);
      expect(payload.metadata.custody).toBeTruthy();
    });
  });

  test.describe("stamps", => {
    test.skip(
      STAMPING_ENABLED === null,
      "Set PLAYWRIGHT_EXPECT_STAMPING_ENABLED to declare the local stamping matrix state.");

    test("stamp list reflects expected flag state", async ({ request, baseURL }) => {
      const response = await request.get(`${baseURL}/api/v1/stamps`);

      if (!STAMPING_ENABLED) {
        expect(response.status()).toBe(403);
        const payload = await response.json();
        expect(payload.detail.error).toBe("feature_disabled");
        return;
      }

      expect(response.status()).toBe(200);
      const payload = await response.json();
      expect(Array.isArray(payload.operations)).toBeTruthy();
      expect(payload.total).toBeGreaterThan(0);
      expect(payload.operations.some((operation) => operation.name === "bates")).toBeTruthy();
    });

    test("stamp metadata reflects expected flag state", async ({ request, baseURL }) => {
      const response = await request.get(`${baseURL}/api/v1/stamps/bates`);

      if (!STAMPING_ENABLED) {
        expect(response.status()).toBe(403);
        const payload = await response.json();
        expect(payload.detail.error).toBe("feature_disabled");
        return;
      }

      expect(response.status()).toBe(200);
      const payload = await response.json();
      expect(payload.name).toBe("bates");
      expect(payload.supported_formats).toContain("pdf");
    });

    test("stamp execute succeeds when enabled with explicit auth and server paths", async ({ request, baseURL }) => {
      test.skip(!STAMPING_ENABLED, "Stamp execution only applies when stamping is enabled.");
      test.skip(
        Object.keys(buildFeatureFlagHeaders()).length === 0,
        "Set PLAYWRIGHT_FEATURE_FLAGS_API_KEY or PLAYWRIGHT_OCR_API_KEY for stamp execution coverage.");
      test.skip(
        !hasFeatureExecutionPaths(),
        "Set PLAYWRIGHT_FEATURE_SAMPLE_INPUT_PATH and PLAYWRIGHT_FEATURE_OUTPUT_DIR for stamp execution coverage.");

      const outputPath = buildServerOutputPath("playwright-stamp");
      const response = await request.post(`${baseURL}/api/v1/stamps/execute`, {
        headers: buildFeatureFlagHeaders(),
        data: {
          operation_id: "bates",
          input_path: getFeatureInputPath(),
          output_path: outputPath,
          placement: "bottom_right",
          params: {
            prefix: "PW",
            start: 1000,
            width: 4,
          },
        },
      });

      expect(response.status()).toBe(200);
      const payload = await response.json();
      expect(payload.success).toBeTruthy();
      expect(payload.operation_id).toBe("bates");
      expect(payload.output_path).toBe(outputPath);
      expect(payload.pages_stamped).toBeGreaterThanOrEqual(1);
      expect(Array.isArray(payload.stamp_values)).toBeTruthy();
      expect(payload.stamp_values.length).toBeGreaterThan(0);
      expect(payload.metadata.custody).toBeTruthy();
    });
  });
});

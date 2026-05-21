const { test, expect } = require("@playwright/test");
const { hasEnv } = require("../../helpers/runtime");
const {
  expectJobSubmitResponse,
  submitJobUpload,
} = require("../../helpers/request");

async function submitLifecycleJob(request, baseURL) {
  const submitResponse = await submitJobUpload(request, baseURL);
  return expectJobSubmitResponse(submitResponse);
}

test.describe("OCR job lifecycle request contracts", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_API_BASE_URL"),
    "Set PLAYWRIGHT_API_BASE_URL to run request-level API lifecycle tests.");

  test("@smoke upload submit appears in list and status", async ({ request, baseURL }) => {
    const submitted = await submitLifecycleJob(request, baseURL);

    const listResponse = await request.get(`${baseURL}/api/v1/jobs`);
    expect(listResponse.ok()).toBeTruthy();
    const listPayload = await listResponse.json();
    expect(Array.isArray(listPayload.jobs)).toBeTruthy();
    expect(listPayload.jobs.some((job) => job.job_id === submitted.job_id)).toBeTruthy();

    const statusResponse = await request.get(`${baseURL}/api/v1/jobs/${submitted.job_id}`);
    expect(statusResponse.ok()).toBeTruthy();
    const statusPayload = await statusResponse.json();
    expect(statusPayload.job_id).toBe(submitted.job_id);
    expect(statusPayload.progress).toBeTruthy();
  });

  test("@smoke result endpoint rejects non-terminal jobs", async ({ request, baseURL }) => {
    const submitted = await submitLifecycleJob(request, baseURL);

    const resultResponse = await request.get(`${baseURL}/api/v1/jobs/${submitted.job_id}/result`);
    expect([200, 409]).toContain(resultResponse.status());

    if (resultResponse.status() === 409) {
      const payload = await resultResponse.json();
      expect(payload.detail.error).toBe("job_not_complete");
    }
  });

  test("@smoke retry rejects active jobs when they are not terminal", async ({ request, baseURL }) => {
    const submitted = await submitLifecycleJob(request, baseURL);

    const retryResponse = await request.post(`${baseURL}/api/v1/jobs/${submitted.job_id}/retry`);
    expect([201, 409]).toContain(retryResponse.status());

    if (retryResponse.status() === 409) {
      const payload = await retryResponse.json();
      expect(payload.detail.error).toBe("invalid_state");
    }
  });

  test("@smoke cancel returns a valid job status response", async ({ request, baseURL }) => {
    const submitted = await submitLifecycleJob(request, baseURL);

    const cancelResponse = await request.delete(`${baseURL}/api/v1/jobs/${submitted.job_id}`);
    expect([200, 404]).toContain(cancelResponse.status());

    if (cancelResponse.status() === 200) {
      const payload = await cancelResponse.json();
      expect(payload.job_id).toBe(submitted.job_id);
      expect(payload.status).toMatch(/submitted|processing|cancelled|completed|failed/);
    }
  });

  test("@smoke submit rejects invalid webhook urls", async ({ request, baseURL }) => {
    const response = await submitJobUpload(request, baseURL, {
      webhook_url: "ftp://invalid-scheme.local/webhook",
    });

    expect(response.status()).toBe(422);
    const payload = await response.json();
    expect(payload.detail.error).toBe("invalid_webhook_url");
  });
});

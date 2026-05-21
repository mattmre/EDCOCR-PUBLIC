const { test, expect } = require("@playwright/test");
const { hasEnv } = require("../../helpers/runtime");
const {
  expectBatchSubmitResponse,
  submitBatchUpload,
} = require("../../helpers/request");

test.describe("OCR batch request contracts", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_API_BASE_URL"),
    "Set PLAYWRIGHT_API_BASE_URL to run request-level batch tests.");

  test("@smoke batch submit appears in list and status", async ({ request, baseURL }) => {
    const submitResponse = await submitBatchUpload(request, baseURL);
    const submitted = await expectBatchSubmitResponse(submitResponse);

    const listResponse = await request.get(`${baseURL}/api/v1/jobs/batch`);
    expect(listResponse.ok()).toBeTruthy();
    const listPayload = await listResponse.json();
    expect(Array.isArray(listPayload.batches)).toBeTruthy();
    expect(
      listPayload.batches.some((batch) => batch.batch_id === submitted.batch_id)).toBeTruthy();

    const statusResponse = await request.get(
      `${baseURL}/api/v1/jobs/batch/${submitted.batch_id}`);
    expect(statusResponse.ok()).toBeTruthy();
    const statusPayload = await statusResponse.json();
    expect(statusPayload.batch_id).toBe(submitted.batch_id);
    expect(Array.isArray(statusPayload.jobs)).toBeTruthy();
    expect(statusPayload.progress).toBeTruthy();
  });

  test("@smoke batch cancel returns a valid batch status payload", async ({ request, baseURL }) => {
    const submitResponse = await submitBatchUpload(request, baseURL);
    const submitted = await expectBatchSubmitResponse(submitResponse);

    const cancelResponse = await request.delete(
      `${baseURL}/api/v1/jobs/batch/${submitted.batch_id}`);
    expect(cancelResponse.status()).toBe(200);
    const payload = await cancelResponse.json();
    expect(payload.batch_id).toBe(submitted.batch_id);
    expect(payload.status).toMatch(/submitted|processing|cancelled|completed|failed/);
  });

  test("@smoke batch retry handles retryable and non-retryable states explicitly", async ({ request, baseURL }) => {
    const submitResponse = await submitBatchUpload(request, baseURL);
    const submitted = await expectBatchSubmitResponse(submitResponse);

    await request.delete(`${baseURL}/api/v1/jobs/batch/${submitted.batch_id}`);

    const retryResponse = await request.post(
      `${baseURL}/api/v1/jobs/batch/${submitted.batch_id}/retry`);
    expect([200, 409]).toContain(retryResponse.status());

    if (retryResponse.status() === 409) {
      const payload = await retryResponse.json();
      expect(payload.detail.error).toBe("no_retryable_jobs");
      return;
    }

    const payload = await retryResponse.json();
    expect(payload.batch_id).toBe(submitted.batch_id);
    expect(Array.isArray(payload.jobs)).toBeTruthy();
  });

  test("@smoke batch submit rejects invalid webhook urls", async ({ request, baseURL }) => {
    const response = await submitBatchUpload(request, baseURL, {
      webhook_url: "ftp://invalid-scheme.local/webhook",
    });

    expect(response.status()).toBe(422);
    const payload = await response.json();
    expect(payload.detail.error).toBe("invalid_webhook_url");
  });
});

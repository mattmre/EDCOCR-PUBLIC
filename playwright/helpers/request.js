const fs = require("fs");
const path = require("path");
const { expect } = require("@playwright/test");
const { buildApiHeaders } = require("./runtime");

function samplePdfPath() {
  return path.join(__dirname, "..", "fixtures", "files", "sample-upload.pdf");
}

function samplePdfPayload(name = "sample-upload.pdf") {
  return {
    name,
    mimeType: "application/pdf",
    buffer: fs.readFileSync(samplePdfPath()),
  };
}

async function submitJobUpload(request, baseURL, extraFields = {}, options = {}) {
  return request.post(`${baseURL}/api/v1/jobs`, {
    headers: options.headers || buildApiHeaders(options.headerEnvName),
    multipart: {
      file: samplePdfPayload(),
      priority: "normal",
      ...extraFields,
    },
  });
}

async function submitBatchUpload(request, baseURL, extraFields = {}, options = {}) {
  return request.post(`${baseURL}/api/v1/jobs/batch`, {
    headers: options.headers || buildApiHeaders(options.headerEnvName),
    multipart: {
      files: samplePdfPayload(),
      priority: "normal",
      ...extraFields,
    },
  });
}

async function expectJobSubmitResponse(response) {
  expect(response.status()).toBe(201);
  const payload = await response.json();
  expect(payload.job_id).toMatch(/^job_[0-9a-f]{12}$/);
  expect(payload.status).toBe("submitted");
  expect(payload.source_file).toBe("sample-upload.pdf");
  expect(payload.links.self).toContain(`/api/v1/jobs/${payload.job_id}`);
  return payload;
}

async function expectBatchSubmitResponse(response) {
  expect(response.status()).toBe(201);
  const payload = await response.json();
  expect(payload.batch_id).toMatch(/^batch_[0-9a-f]{12}$/);
  expect(payload.status).toBe("submitted");
  expect(payload.total_jobs).toBeGreaterThanOrEqual(1);
  expect(Array.isArray(payload.jobs)).toBeTruthy();
  expect(payload.links.self).toContain(`/api/v1/jobs/batch/${payload.batch_id}`);
  return payload;
}

module.exports = {
  expectBatchSubmitResponse,
  expectJobSubmitResponse,
  samplePdfPath,
  samplePdfPayload,
  submitBatchUpload,
  submitJobUpload,
};

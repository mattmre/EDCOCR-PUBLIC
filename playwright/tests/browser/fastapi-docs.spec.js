const { test, expect } = require("@playwright/test");
const { hasEnv } = require("../../helpers/runtime");
const {
  expectNoUnexpectedConsoleErrors,
  trackConsoleErrors,
} = require("../../helpers/browser");

test.describe("FastAPI browser docs", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_API_BASE_URL"),
    "Set PLAYWRIGHT_API_BASE_URL to run browser docs smoke tests.");

  test("@smoke swagger ui loads and lists core OCR routes", async ({ page }) => {
    const consoleTracker = trackConsoleErrors(page);
    await page.goto("/docs");

    await expect(page.getByText("EDCOCR API")).toBeVisible();
    await expect(
      page.locator("#operations-jobs-submit_job_api_v1_jobs_post").getByRole("link", {
        name: "/api/v1/jobs",
      })).toBeVisible();
    await expect(
      page.locator("#operations-health-health_check_api_v1_health_get").getByRole("link", {
        name: "/api/v1/health",
      })).toBeVisible();
    await expectNoUnexpectedConsoleErrors(consoleTracker);
    consoleTracker.stop();
  });

  test("@smoke redoc loads and exposes health and jobs surfaces", async ({ page }) => {
    const consoleTracker = trackConsoleErrors(page);
    await page.goto("/redoc");

    await expect(page).toHaveTitle(/EDCOCR API - ReDoc/i);
    await expect(page.locator("redoc")).toHaveAttribute("spec-url", "/openapi.json");
    await expectNoUnexpectedConsoleErrors(consoleTracker);
    consoleTracker.stop();
  });
});

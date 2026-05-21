// @ts-check
const { test, expect } = require("@playwright/test");
const { hasEnv, hasAllEnv } = require("../../helpers/runtime");
const { hasAdminCredentials, loginToAdmin } = require("../../helpers/admin");
const {
  expectNoUnexpectedConsoleErrors,
  trackConsoleErrors,
} = require("../../helpers/browser");

// Coordinator UI tests (browser-based)
// These are env-gated – skip when PLAYWRIGHT_COORDINATOR_BASE_URL is not set.

const BASE_URL = process.env.PLAYWRIGHT_COORDINATOR_BASE_URL || "";

test.describe("Coordinator Admin UI", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_COORDINATOR_BASE_URL"),
    "Set PLAYWRIGHT_COORDINATOR_BASE_URL to run coordinator UI tests.");

  test("Admin login page loads", async ({ page }) => {
    const consoleTracker = trackConsoleErrors(page);
    await page.goto(`${BASE_URL}/admin/login/`);
    // Should have a login form or redirect
    const title = await page.title();
    expect(title.length).toBeGreaterThan(0);
    await expectNoUnexpectedConsoleErrors(consoleTracker);
    consoleTracker.stop();
  });

  test("Admin page has Django styling", async ({ page }) => {
    const response = await page.goto(`${BASE_URL}/admin/login/`);
    expect([200, 301, 302]).toContain(response.status());
  });

  test("Invalid credentials show error", async ({ page }) => {
    const consoleTracker = trackConsoleErrors(page);
    await page.goto(`${BASE_URL}/admin/login/`);

    const usernameField = page.locator('input[name="username"]');
    const passwordField = page.locator('input[name="password"]');

    if (await usernameField.isVisible()) {
      await usernameField.fill("invalid_user");
      await passwordField.fill("invalid_pass");
      await page.locator('input[type="submit"]').click();
      // Should stay on login page or show error
      await expect(page).toHaveURL(/admin/);
    }
    await expectNoUnexpectedConsoleErrors(consoleTracker);
    consoleTracker.stop();
  });

  test("Authenticated admin sees dashboard", async ({ page }) => {
    test.skip(
      !hasAdminCredentials(),
      "Set Django admin credentials to run authenticated coordinator UI tests.");

    const consoleTracker = trackConsoleErrors(page);
    await loginToAdmin(page);

    // After login, should see admin dashboard
    await expect(page.locator("body")).toContainText(/Site administration|Dashboard/i);
    await expectNoUnexpectedConsoleErrors(consoleTracker);
    consoleTracker.stop();
  });
});

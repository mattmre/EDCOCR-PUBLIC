const { test, expect } = require("@playwright/test");
const { hasEnv } = require("../../helpers/runtime");
const {
  expectNoUnexpectedConsoleErrors,
  trackConsoleErrors,
} = require("../../helpers/browser");
const {
  hasAdminCredentials,
  hasSeededAdminData,
  loginToAdmin,
  openAdminChangelist,
  runAdminAction,
  selectChangelistRowByText,
} = require("../../helpers/admin");

const ADMIN_SURFACES = [
  {
    name: "jobs changelist",
    path: "/admin/jobs/job/",
    expectedHeading: /select job to change/i,
    addLabel: /add job/i,
  },
  {
    name: "workers changelist",
    path: "/admin/jobs/worker/",
    expectedHeading: /select worker to change/i,
    addLabel: /add worker/i,
  },
  {
    name: "page results changelist",
    path: "/admin/jobs/pageresult/",
    expectedHeading: /select page result to change/i,
    addLabel: /add page result/i,
  },
  {
    name: "custody events changelist",
    path: "/admin/jobs/custodyevent/",
    expectedHeading: /select custody event to change/i,
    addLabel: /add custody event/i,
  },
];

const SEEDED_JOBS = {
  failed: "playwright-failed.pdf",
  processing: "playwright-processing.pdf",
  completed: "playwright-completed.pdf",
};

const SEEDED_WORKERS = {
  online: "playwright-worker-online",
};

test.describe("Django admin smoke", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_COORDINATOR_BASE_URL"),
    "Set PLAYWRIGHT_COORDINATOR_BASE_URL to run Django admin browser tests.");

  test("@smoke admin login page loads", async ({ page }) => {
    const consoleTracker = trackConsoleErrors(page);
    await page.goto("/admin/login/?next=/admin/");

    await expect(page.getByLabel(/username/i)).toBeVisible();
    await expect(page.getByLabel(/password/i)).toBeVisible();
    await expect(page.getByRole("button", { name: /log in/i })).toBeVisible();
    await expectNoUnexpectedConsoleErrors(consoleTracker);
    consoleTracker.stop();
  });

  test.skip(
    !hasAdminCredentials(),
    "Set Django admin credentials to run authenticated admin smoke coverage.");

  test("@smoke authenticated admin index loads", async ({ page }) => {
    const consoleTracker = trackConsoleErrors(page);
    await loginToAdmin(page);

    await expect(page.getByText(/site administration/i)).toBeVisible();
    await expect(page.getByRole("link", { name: /^OCR Jobs$/i })).toBeVisible();
    await expect(page.getByRole("link", { name: /workers/i })).toBeVisible();
    await expectNoUnexpectedConsoleErrors(consoleTracker);
    consoleTracker.stop();
  });

  for (const surface of ADMIN_SURFACES) {
    test(`@smoke authenticated ${surface.name} loads`, async ({ page }) => {
      const consoleTracker = trackConsoleErrors(page);
      await loginToAdmin(page);
      await openAdminChangelist(page, surface.path, {
        expectedHeading: surface.expectedHeading,
        addLabel: surface.addLabel,
      });
      await expectNoUnexpectedConsoleErrors(consoleTracker);
      consoleTracker.stop();
    });
  }

  test.skip(
    !hasSeededAdminData(),
    "Set PLAYWRIGHT_ADMIN_SEEDED=true after running the coordinator seed command to run seeded admin workflows.");

  test("@smoke seeded failed job can be retried", async ({ page }) => {
    const consoleTracker = trackConsoleErrors(page);
    await loginToAdmin(page);
    await page.goto(`/admin/jobs/job/?q=${SEEDED_JOBS.failed}`);
    await selectChangelistRowByText(page, SEEDED_JOBS.failed);
    await runAdminAction(page, "retry_failed_jobs");

    await expect(page.locator("body")).toContainText(/1 jobs queued for retry\./i);
    await expect(page.locator("#result_list")).toContainText(/submitted/i);
    await expectNoUnexpectedConsoleErrors(consoleTracker);
    consoleTracker.stop();
  });

  test("@smoke seeded processing job can be cancelled", async ({ page }) => {
    const consoleTracker = trackConsoleErrors(page);
    await loginToAdmin(page);
    await page.goto(`/admin/jobs/job/?q=${SEEDED_JOBS.processing}`);
    await selectChangelistRowByText(page, SEEDED_JOBS.processing);
    await runAdminAction(page, "cancel_running_jobs");

    await expect(page.locator("body")).toContainText(/1 jobs cancelled\./i);
    await expect(page.locator("#result_list")).toContainText(/cancelled/i);
    await expectNoUnexpectedConsoleErrors(consoleTracker);
    consoleTracker.stop();
  });

  test("@smoke seeded completed job detail shows page and custody inlines", async ({ page }) => {
    const consoleTracker = trackConsoleErrors(page);
    await loginToAdmin(page);
    await page.goto(`/admin/jobs/job/?q=${SEEDED_JOBS.completed}`);
    const row = page.locator("#result_list tbody tr").filter({ hasText: SEEDED_JOBS.completed }).first();
    await expect(row).toBeVisible();
    await row.getByRole("link").first().click();

    await expect(page.locator("body")).toContainText(/page results/i);
    await expect(page.locator("body")).toContainText(/custody events/i);
    await expect(page.locator("body")).toContainText(/playwright-doc-001/i);
    await expectNoUnexpectedConsoleErrors(consoleTracker);
    consoleTracker.stop();
  });

  test("@smoke seeded worker can be marked offline", async ({ page }) => {
    const consoleTracker = trackConsoleErrors(page);
    await loginToAdmin(page);
    await page.goto("/admin/jobs/worker/");
    await selectChangelistRowByText(page, SEEDED_WORKERS.online);
    await runAdminAction(page, "mark_workers_offline");

    await expect(page.locator("body")).toContainText(/1 workers marked offline\./i);
    await expect(page.locator("#result_list")).toContainText(/offline/i);
    await expectNoUnexpectedConsoleErrors(consoleTracker);
    consoleTracker.stop();
  });
});

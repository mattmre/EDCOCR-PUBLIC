const { expect } = require("@playwright/test");
const { hasAllEnv, isTruthyEnv } = require("./runtime");

const ADMIN_ENV_VARS = [
  "PLAYWRIGHT_COORDINATOR_BASE_URL",
  "PLAYWRIGHT_DJANGO_ADMIN_USER",
  "PLAYWRIGHT_DJANGO_ADMIN_PASSWORD",
];

function hasSeededAdminData() {
  return isTruthyEnv("PLAYWRIGHT_ADMIN_SEEDED");
}

function hasAdminCredentials() {
  return hasAllEnv(ADMIN_ENV_VARS);
}

async function loginToAdmin(page, credentials = {}) {
  const username = credentials.username || process.env.PLAYWRIGHT_DJANGO_ADMIN_USER;
  const password = credentials.password || process.env.PLAYWRIGHT_DJANGO_ADMIN_PASSWORD;
  await page.goto("/admin/login/?next=/admin/");
  await page.getByLabel(/username/i).fill(username);
  await page.getByLabel(/password/i).fill(password);
  await page.getByRole("button", { name: /log in/i }).click();
  await expect(page.getByText(/site administration/i)).toBeVisible();
}

async function openAdminChangelist(page, path, options = {}) {
  const expectedHeading = options.expectedHeading;
  const addLabel = options.addLabel;

  await page.goto(path);
  await expect(page.locator("#changelist-form")).toBeVisible();

  if (expectedHeading) {
    await expect(page.locator("body")).toContainText(expectedHeading);
  }

  if (addLabel) {
    await expect(page.getByRole("link", { name: addLabel })).toBeVisible();
  }
}

async function selectChangelistRowByText(page, text) {
  const row = page.locator("#result_list tbody tr").filter({ hasText: text }).first();
  await expect(row).toBeVisible();
  await row.locator("input.action-select").check();
  return row;
}

async function runAdminAction(page, actionValue) {
  await page.locator("select[name='action']").first().selectOption(actionValue);
  await page.locator("button[name='index']").first().click();
}

module.exports = {
  ADMIN_ENV_VARS,
  hasAdminCredentials,
  hasSeededAdminData,
  loginToAdmin,
  openAdminChangelist,
  selectChangelistRowByText,
  runAdminAction,
};

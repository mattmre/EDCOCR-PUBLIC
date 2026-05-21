const path = require("path");

function makeRunId() {
  const iso = new Date().toISOString();
  return iso.replace(/[:.]/g, "-");
}

const runId = process.env.PLAYWRIGHT_RUN_ID || makeRunId();
const artifactRoot = path.join(__dirname, "playwright-artifacts", "runs", runId);
const isCI = Boolean(process.env.CI);

// When PLAYWRIGHT_API_BASE_URL is not explicitly set but we are in CI,
// fall back to the local webServer URL so smoke tests can run against it.
const WEB_SERVER_PORT = 8765;
const defaultApiUrl = process.env.PLAYWRIGHT_API_BASE_URL || (isCI ? `http://127.0.0.1:${WEB_SERVER_PORT}` : undefined);

// Start an auto-managed FastAPI server only when CI explicitly opts in.
// Some CI workflows run Playwright without Python API dependencies or
// provisioned service URLs; those should remain in the blocked/skip posture
// instead of hard-failing on webServer startup.
const needsWebServer =
  !process.env.PLAYWRIGHT_API_BASE_URL &&
  isCI &&
  process.env.PLAYWRIGHT_AUTO_API === "true";
const webServerConfig = needsWebServer
  ? {
      command: `python -m uvicorn api.main:app --host 127.0.0.1 --port ${WEB_SERVER_PORT}`,
      port: WEB_SERVER_PORT,
      reuseExistingServer: !isCI,
      timeout: 30_000,
      env: {
        ALLOW_UNAUTHENTICATED: "true",
        OCR_SUBMIT_RATE_LIMIT: "10000/minute",
        OCR_RATE_LIMIT: "10000/minute",
        MAX_CONCURRENT_JOBS: "100",
        OCR_SOURCE_DIR: path.join(__dirname, "ocr_source"),
        OCR_OUTPUT_DIR: path.join(__dirname, "ocr_output"),
      },
    }
  : undefined;

const config = {
  testDir: path.join(__dirname, "playwright", "tests"),
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  fullyParallel: true,
  forbidOnly: isCI,
  retries: isCI ? 2 : 0,
  workers: isCI ? 2 : undefined,
  outputDir: path.join(artifactRoot, "test-results"),
  reporter: [
    ["list"],
    ["html", { open: "never", outputFolder: path.join(artifactRoot, "html-report") }],
    ["json", { outputFile: path.join(artifactRoot, "results.json") }],
    ["junit", { outputFile: path.join(artifactRoot, "results.xml") }],
    ["blob", { outputDir: path.join(artifactRoot, "blob-report") }]
  ],
  use: {
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    ignoreHTTPSErrors: true,
    viewport: { width: 1440, height: 960 },
  },
  metadata: {
    runId,
    artifactRoot,
    apiBaseUrl: defaultApiUrl || "",
    coordinatorBaseUrl: process.env.PLAYWRIGHT_COORDINATOR_BASE_URL || "",
  },
  projects: [
    {
      name: "request-api",
      testMatch: /request\/.*\.spec\.js$/,
      use: {
        baseURL: defaultApiUrl,
      },
    },
    {
      name: "browser-api-docs",
      testMatch: /browser\/.*\.spec\.js$/,
      use: {
        browserName: "chromium",
        baseURL: defaultApiUrl,
      },
    },
    {
      name: "browser-admin",
      testMatch: /admin\/.*\.spec\.js$/,
      use: {
        browserName: "chromium",
        baseURL: process.env.PLAYWRIGHT_COORDINATOR_BASE_URL,
      },
    },
    {
      name: "feature-flags-api",
      testMatch: /feature-flags\/.*\.spec\.js$/,
      use: {
        baseURL: defaultApiUrl,
      },
    },
    {
      name: "ops-api",
      testMatch: /ops\/.*\.spec\.js$/,
      use: {
        baseURL: defaultApiUrl,
      },
    },
    {
      name: "coordinator-api",
      testMatch: /coordinator\/.*\.spec\.js$/,
      use: {
        baseURL: process.env.PLAYWRIGHT_COORDINATOR_BASE_URL,
      },
    },
  ],
};

if (webServerConfig) {
  config.webServer = webServerConfig;
}

module.exports = config;

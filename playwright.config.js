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
const needsApiWebServer =
  !process.env.PLAYWRIGHT_API_BASE_URL &&
  isCI &&
  process.env.PLAYWRIGHT_AUTO_API === "true";

if (needsApiWebServer) {
  process.env.PLAYWRIGHT_API_BASE_URL = defaultApiUrl;
}

// Auto-managed Next operator console: when PLAYWRIGHT_AUTO_FRONTEND=true and
// no external frontend URL is supplied, default to a local Next dev server.
const FRONTEND_WEB_SERVER_PORT = 3100;
const hasExplicitFrontendUrl = Boolean(
  process.env.PLAYWRIGHT_FRONTEND_BASE_URL &&
    process.env.PLAYWRIGHT_FRONTEND_BASE_URL.trim(),
);
const autoFrontend = process.env.PLAYWRIGHT_AUTO_FRONTEND === "true";
const needsFrontendWebServer = !hasExplicitFrontendUrl && autoFrontend;
const frontendBaseUrl = hasExplicitFrontendUrl
  ? process.env.PLAYWRIGHT_FRONTEND_BASE_URL.trim()
  : autoFrontend
    ? `http://127.0.0.1:${FRONTEND_WEB_SERVER_PORT}`
    : undefined;

if (needsFrontendWebServer) {
  process.env.PLAYWRIGHT_FRONTEND_BASE_URL = frontendBaseUrl;
}

const apiWebServerConfig = needsApiWebServer
  ? {
      command: `python -m uvicorn api.main:app --host 127.0.0.1 --port ${WEB_SERVER_PORT}`,
      port: WEB_SERVER_PORT,
      reuseExistingServer: !isCI,
      timeout: 30_000,
      env: {
        ALLOW_UNAUTHENTICATED: "true",
        // Request smoke tests submit, retry, and cancel jobs, so the
        // unauthenticated local harness must exercise operator-scoped paths.
        ANONYMOUS_ROLE: "operator",
        OCR_SUBMIT_RATE_LIMIT: "10000/minute",
        OCR_RATE_LIMIT: "10000/minute",
        MAX_CONCURRENT_JOBS: "100",
        OCR_SOURCE_DIR: path.join(__dirname, "ocr_source"),
        OCR_OUTPUT_DIR: path.join(__dirname, "ocr_output"),
      },
    }
  : undefined;

const frontendWebServerConfig = needsFrontendWebServer
  ? {
      command: `npm run dev -- --port ${FRONTEND_WEB_SERVER_PORT} --hostname 127.0.0.1`,
      cwd: path.join(__dirname, "frontend"),
      url: frontendBaseUrl,
      reuseExistingServer: !isCI,
      timeout: 120_000,
    }
  : undefined;

const webServers = [apiWebServerConfig, frontendWebServerConfig].filter(Boolean);

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
    apiBaseUrl: process.env.PLAYWRIGHT_API_BASE_URL || defaultApiUrl || "",
    coordinatorBaseUrl: process.env.PLAYWRIGHT_COORDINATOR_BASE_URL || "",
    frontendBaseUrl: process.env.PLAYWRIGHT_FRONTEND_BASE_URL || frontendBaseUrl || "",
  },
  projects: [
    {
      name: "request-api",
      testMatch: /request\/.*\.spec\.js$/,
      use: {
        baseURL: process.env.PLAYWRIGHT_API_BASE_URL || defaultApiUrl,
      },
    },
    {
      name: "browser-api-docs",
      testMatch: /browser\/.*\.spec\.js$/,
      use: {
        browserName: "chromium",
        baseURL: process.env.PLAYWRIGHT_API_BASE_URL || defaultApiUrl,
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
        baseURL: process.env.PLAYWRIGHT_API_BASE_URL || defaultApiUrl,
      },
    },
    {
      name: "ops-api",
      testMatch: /ops\/.*\.spec\.js$/,
      use: {
        baseURL: process.env.PLAYWRIGHT_API_BASE_URL || defaultApiUrl,
      },
    },
    {
      name: "coordinator-api",
      testMatch: /coordinator\/.*\.spec\.js$/,
      use: {
        baseURL: process.env.PLAYWRIGHT_COORDINATOR_BASE_URL,
      },
    },
    {
      name: "operator-console",
      testMatch: /operator-console\/.*\.spec\.js$/,
      use: {
        browserName: "chromium",
        baseURL: process.env.PLAYWRIGHT_FRONTEND_BASE_URL || frontendBaseUrl,
      },
    },
  ],
};

// Preserve a single webServer object when only one server is enabled so
// existing contract probes that read webServer.env continue to work.
if (webServers.length === 1) {
  config.webServer = webServers[0];
} else if (webServers.length > 1) {
  config.webServer = webServers;
}

module.exports = config;

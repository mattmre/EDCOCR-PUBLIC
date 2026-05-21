# Playwright CI Gate Promotion Guide

This document explains how the Playwright PR gate workflow was promoted from an
informational check to a **conditional blocking CI gate**, and how to configure
it in your GitHub repository.

## What Changed

The `.github/workflows/playwright-pr.yml` workflow now uses a three-job
pattern:

1. **`check-env`** -- Detects whether the `PLAYWRIGHT_API_BASE_URL` repository
   variable is set and outputs `has_base_url` (`true` or `false`).

2. **`playwright-pr`** -- The existing Playwright test job. It now has
   `needs: [check-env]` and only runs when `has_base_url == 'true'`. When the
   variable is not configured, this job is skipped entirely (no Node/browser
   install, no test execution).

3. **`playwright-gate`** -- A lightweight final job that always runs
   (`if: always`). It evaluates the combined result:
   - If `PLAYWRIGHT_API_BASE_URL` was not set, the gate **passes** (graceful
     skip).
   - If `PLAYWRIGHT_API_BASE_URL` was set and tests passed, the gate
     **passes**.
   - If `PLAYWRIGHT_API_BASE_URL` was set and tests failed, the gate
     **fails**.

This pattern means:
- **Without env vars**: the gate always passes. PRs are never blocked by
  unconfigured Playwright infrastructure.
- **With env vars**: all Playwright tests must pass. Failures are blocking.

## Configuring Branch Protection

To make the Playwright gate a required status check:

1. Go to **Settings > Branches** in your GitHub repository.
2. Edit (or create) a branch protection rule for `main`.
3. Enable **Require status checks to pass before merging**.
4. In the search box, find and add: **`Playwright Gate`**
   (this is the `playwright-gate` job name).
5. Save the rule.

The `Playwright Gate` check will now appear on every PR that touches
Playwright-related paths (see the `paths` filter in the workflow `on` trigger).

## Setting the Required Secrets and Variables

### Repository Variables (Settings > Secrets and variables > Actions > Variables)

| Variable | Required | Description |
|---|---|---|
| `PLAYWRIGHT_API_BASE_URL` | Yes (to activate) | Base URL of the FastAPI server (e.g. `https://ocr.example.com`) |
| `PLAYWRIGHT_COORDINATOR_BASE_URL` | For admin/coordinator tests | Base URL of the Django coordinator (e.g. `https://coordinator.example.com`) |
| `PLAYWRIGHT_ADMIN_SEEDED` | For admin tests | Set to `true` if the Django admin has been seeded |
| `PLAYWRIGHT_EXPECT_TRANSFORMS_ENABLED` | Optional | Set to `true` to assert transform features |
| `PLAYWRIGHT_EXPECT_STAMPING_ENABLED` | Optional | Set to `true` to assert Bates stamping features |
| `PLAYWRIGHT_EXPECT_MULTITENANCY_ENABLED` | Optional | Set to `true` to assert multi-tenancy features |
| `PLAYWRIGHT_EXPECT_API_AUTH_REQUIRED` | Optional | Set to `true` to assert API auth enforcement |
| `PLAYWRIGHT_EXPECT_METRICS_AUTH_REQUIRED` | Optional | Set to `true` to assert metrics auth enforcement |
| `PLAYWRIGHT_FEATURE_SAMPLE_INPUT_PATH` | Optional | Path to a sample input file on the target server |
| `PLAYWRIGHT_FEATURE_OUTPUT_DIR` | Optional | Path to the output directory on the target server |

### Repository Secrets (Settings > Secrets and variables > Actions > Secrets)

| Secret | Required | Description |
|---|---|---|
| `PLAYWRIGHT_OCR_API_KEY` | For authenticated API tests | OCR API key (`X-API-Key` header) |
| `PLAYWRIGHT_FEATURE_FLAGS_API_KEY` | For feature-flag tests | Feature flags API key |
| `PLAYWRIGHT_MULTITENANCY_PLATFORM_ADMIN_KEY` | For multi-tenancy tests | Platform admin key |
| `PLAYWRIGHT_METRICS_API_KEY` | For metrics tests | Metrics endpoint API key |
| `PLAYWRIGHT_DJANGO_ADMIN_USER` | For admin tests | Django admin username |
| `PLAYWRIGHT_DJANGO_ADMIN_PASSWORD` | For admin tests | Django admin password |

## Running Playwright Locally

Before pushing changes to Playwright tests, run them locally:

```bash
# Install dependencies (first time or after package.json changes)
npm ci
npm run pw:install

# Run the full suite (tests skip gracefully when env vars are unset)
npm run pw:test

# Run specific projects
npm run pw:test:request          # Request API tests
npm run pw:test:browser          # Browser-based tests (API docs + admin)
npm run pw:test:feature-flags    # Feature flag tests
npm run pw:test:ops              # Ops/standards tests

# Run with a local server
export PLAYWRIGHT_API_BASE_URL=http://127.0.0.1:8765
export PLAYWRIGHT_COORDINATOR_BASE_URL=http://127.0.0.1:8000
npm run pw:test
```

## Test Projects

The Playwright configuration (`playwright.config.js`) defines six test
projects. Each project targets a specific area of the application:

| Project | Test Directory | Base URL Source | What It Covers |
|---|---|---|---|
| `request-api` | `playwright/tests/request/` | `PLAYWRIGHT_API_BASE_URL` | Health check, job submission, batch requests |
| `browser-api-docs` | `playwright/tests/browser/` | `PLAYWRIGHT_API_BASE_URL` | FastAPI docs UI, WebSocket connections |
| `browser-admin` | `playwright/tests/admin/` | `PLAYWRIGHT_COORDINATOR_BASE_URL` | Django admin login, model browsing |
| `feature-flags-api` | `playwright/tests/feature-flags/` | `PLAYWRIGHT_API_BASE_URL` | Multi-tenancy admin, transforms/stamps |
| `ops-api` | `playwright/tests/ops/` | `PLAYWRIGHT_API_BASE_URL` | API standards validation |
| `coordinator-api` | `playwright/tests/coordinator/` | `PLAYWRIGHT_COORDINATOR_BASE_URL` | Coordinator API and UI |

## Environment Variables Reference

All environment variables consumed by the Playwright config and test helpers:

| Variable | Purpose |
|---|---|
| `PLAYWRIGHT_RUN_ID` | Unique run identifier (auto-generated in CI) |
| `PLAYWRIGHT_API_BASE_URL` | FastAPI server base URL |
| `PLAYWRIGHT_COORDINATOR_BASE_URL` | Django coordinator base URL |
| `PLAYWRIGHT_OCR_API_KEY` | OCR API authentication key |
| `PLAYWRIGHT_FEATURE_FLAGS_API_KEY` | Feature flags authentication key |
| `PLAYWRIGHT_MULTITENANCY_PLATFORM_ADMIN_KEY` | Platform admin authentication key |
| `PLAYWRIGHT_METRICS_API_KEY` | Metrics endpoint authentication key |
| `PLAYWRIGHT_DJANGO_ADMIN_USER` | Django admin username |
| `PLAYWRIGHT_DJANGO_ADMIN_PASSWORD` | Django admin password |
| `PLAYWRIGHT_ADMIN_SEEDED` | Whether Django admin has been seeded (`true`/`false`) |
| `PLAYWRIGHT_AUTO_API` | Set to `true` to auto-start FastAPI in CI |
| `PLAYWRIGHT_EXPECT_TRANSFORMS_ENABLED` | Feature flag: transforms feature |
| `PLAYWRIGHT_EXPECT_STAMPING_ENABLED` | Feature flag: Bates stamping feature |
| `PLAYWRIGHT_EXPECT_MULTITENANCY_ENABLED` | Feature flag: multi-tenancy feature |
| `PLAYWRIGHT_EXPECT_API_AUTH_REQUIRED` | Feature flag: API auth enforcement |
| `PLAYWRIGHT_EXPECT_METRICS_AUTH_REQUIRED` | Feature flag: metrics auth enforcement |
| `PLAYWRIGHT_FEATURE_SAMPLE_INPUT_PATH` | Sample input file path on target |
| `PLAYWRIGHT_FEATURE_OUTPUT_DIR` | Output directory path on target |
| `CI` | Set automatically by GitHub Actions |

## CI Smoke Tests vs PR Gate

There are two separate Playwright workflows in this repository:

1. **`ci.yml` / `playwright-smoke`** -- Runs on every push/PR against a
   locally-started FastAPI + Django stack. This is a self-contained smoke test
   that does not require external URLs. It runs the `request-api`,
   `browser-api-docs`, and `browser-admin` projects against `127.0.0.1`.

2. **`playwright-pr.yml` / Playwright Gate** -- Runs only when
   Playwright-related paths change. Tests against externally-provisioned
   services via `PLAYWRIGHT_API_BASE_URL`. This is the gate that can be made
   a required status check.

Both workflows can coexist. The smoke tests provide baseline coverage on every
PR, while the full gate validates against a real deployment when configured.

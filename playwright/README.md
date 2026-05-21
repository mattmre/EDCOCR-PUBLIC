# Playwright Scaffold

This directory contains the first Playwright scaffold for EDCOCR.

## Current Scope

- Browser smoke coverage for:
  - FastAPI Swagger UI (`/docs`)
  - FastAPI ReDoc (`/redoc`)
  - Django admin login and index pages (`/admin/*`)
  - Django admin changelists for jobs, workers, page results, and custody events
- Request-level smoke coverage for:
  - FastAPI health endpoint (`/api/v1/health`)
  - OCR job lifecycle request contracts (`/api/v1/jobs/*`)
  - OCR batch lifecycle contracts (`/api/v1/jobs/batch*`)
  - browser-driven WebSocket job coverage (`/ws/jobs/{job_id}`)

The larger plan lives in:

- `docs/testing/playwright-testing-roadmap.md`
- `docs/testing/playwright-page-coverage-matrix.md`
- `docs/testing/playwright-run-template.md`

## Environment Variables

Set these before running any suite:

- `PLAYWRIGHT_API_BASE_URL`
  - Example: `http://127.0.0.1:8000`
- `PLAYWRIGHT_COORDINATOR_BASE_URL`
  - Example: `http://127.0.0.1:8001`
- `PLAYWRIGHT_OCR_API_KEY`
  - Optional. Used for API routes that require OCR API auth.
- `PLAYWRIGHT_FEATURE_FLAGS_API_KEY`
  - Optional. Used for . This may be
    the legacy OCR API key or a seeded tenant/platform-admin key.
- `PLAYWRIGHT_EXPECT_TRANSFORMS_ENABLED`
  - Optional. Explicitly declares whether the target API is running with
    `ENABLE_TRANSFORMS=true`.
- `PLAYWRIGHT_EXPECT_STAMPING_ENABLED`
  - Optional. Explicitly declares whether the target API is running with
    `ENABLE_STAMPING=true`.
- `PLAYWRIGHT_EXPECT_MULTITENANCY_ENABLED`
  - Optional. Explicitly declares whether the target API is running with
    `ENABLE_MULTITENANCY=true`.
- `PLAYWRIGHT_EXPECT_API_AUTH_REQUIRED`
  - Optional. Explicitly declares whether protected FastAPI routes should
    reject unauthenticated requests.
- `PLAYWRIGHT_EXPECT_METRICS_AUTH_REQUIRED`
  - Optional. Explicitly declares whether coordinator metrics endpoints require
    auth in the target environment.
- `PLAYWRIGHT_FEATURE_SAMPLE_INPUT_PATH`
  - Optional. Server-visible input PDF path for execution requests.
- `PLAYWRIGHT_FEATURE_OUTPUT_DIR`
  - Optional. Server-visible output directory for execution requests.
- `PLAYWRIGHT_MULTITENANCY_PLATFORM_ADMIN_KEY`
  - Optional. Enables live multi-tenancy admin workflow coverage for .
- `PLAYWRIGHT_METRICS_API_KEY`
  - Optional. Reserved for coordinator metrics coverage.
- `PLAYWRIGHT_DJANGO_ADMIN_USER`
  - Optional. Enables authenticated admin smoke tests and changelist coverage.
- `PLAYWRIGHT_DJANGO_ADMIN_PASSWORD`
  - Optional. Enables authenticated admin smoke tests and changelist coverage.
- `PLAYWRIGHT_ADMIN_SEEDED`
  - Optional. Set to `true` after running the coordinator seed command to enable
    deterministic admin workflow tests.
- `PLAYWRIGHT_RUN_ID`
  - Optional. If unset, `playwright.config.js` generates a timestamp-based run id.

## Run Id Convention

Each run writes artifacts under:

`playwright-artifacts/runs/<run-id>/`

Recommended human-readable format for manual runs:

`PW-YYYYMMDD-HHMMSS-<scope>`

Examples:

- `PW-20260306-153000-smoke`
- `PW-20260306-161500-pr-ocr-api`
- `PW-20260306-170000-admin-auth`

## Commands

```bash
npm install
npm run pw:test:request
npm run pw:test:feature-flags
npm run pw:test:ops
npm run pw:test:smoke
npm run pw:test
```

Generate a normalized markdown and JSON summary from a completed Playwright run:

```bash
python scripts/summarize_playwright_run.py ^
  --results-json playwright-artifacts/runs/<run-id>/results.json ^
  --summary-markdown docs/testing/runs/<run-id>.md ^
  --summary-json playwright-artifacts/runs/<run-id>/summary.json ^
  --suite-scope "manual Playwright run" ^
  --trigger "local validation" ^
  --branch <branch-name>
```

Feed the normalized Playwright summary into the staged release gate:

```bash
python scripts/run_phase8_release_gate.py ^
  --repo mattmre/EDCOCR-PUBLIC ^
  --pull-request 257 ^
  --playwright-summary-json playwright-artifacts/runs/<run-id>/summary.json ^
  --playwright-gate-policy require-executed ^
  --playwright-required-project request-api ^
  --playwright-required-project browser-admin
```

### Seeded Admin Workflow Bootstrapping

To enable the seeded admin workflow tests, first seed the coordinator:

```bash
cd coordinator
python manage.py seed_playwright_admin --username playwright-admin --password playwright-admin-pass --reset
```

The command requires an explicit `--password` and refuses to run when
`DEPLOYMENT_ENV=production`.

Then set:

```bash
PLAYWRIGHT_ADMIN_SEEDED=true
PLAYWRIGHT_DJANGO_ADMIN_USER=playwright-admin
PLAYWRIGHT_DJANGO_ADMIN_PASSWORD=playwright-admin-pass
```

### The , stamps, and multi-tenancy admin routes.
Declare the expected server-side feature state explicitly before running it:

```bash
PLAYWRIGHT_EXPECT_TRANSFORMS_ENABLED=true
PLAYWRIGHT_EXPECT_STAMPING_ENABLED=true
PLAYWRIGHT_EXPECT_MULTITENANCY_ENABLED=true
```

Transform and stamp execution also need server-visible file paths:

```bash
PLAYWRIGHT_FEATURE_SAMPLE_INPUT_PATH=C:\OCR\sample.pdf
PLAYWRIGHT_FEATURE_OUTPUT_DIR=C:\OCR\playwright-output
```

If multi-tenancy is enabled, seed a deterministic platform-admin key:

```bash
python scripts/seed_playwright_multitenancy.py --db-path C:\OCR\jobs.db --reset
```

Use the returned JSON payload to set:

```bash
PLAYWRIGHT_MULTITENANCY_PLATFORM_ADMIN_KEY=<seeded-api-key>
PLAYWRIGHT_FEATURE_FLAGS_API_KEY=<seeded-api-key>
```

### The , timeout override persistence, and
coordinator metrics/prometheus auth behavior.

Declare the expected auth posture explicitly:

```bash
PLAYWRIGHT_EXPECT_API_AUTH_REQUIRED=true
PLAYWRIGHT_EXPECT_METRICS_AUTH_REQUIRED=true
```

If coordinator metrics auth is enabled, also set:

```bash
PLAYWRIGHT_METRICS_API_KEY=<metrics-api-key>
```

, rate-limit burst, and failover
checks until a more controlled orchestration harness is available.

## Artifact Retention

- Heavy run artifacts stay in `playwright-artifacts/runs/<run-id>/` and are gitignored.
- Historical summaries should be recorded in `docs/testing/playwright-run-ledger.md`.
- Review-ready markdown summaries should be created from `docs/testing/playwright-run-template.md`.

## Design Notes

- This repo is API/admin heavy, not a custom SPA. The Playwright plan therefore mixes browser checks with request-level and operational coverage.
- Browser/admin smoke now asserts that the loaded pages do not emit unexpected
  browser console `error` messages during initial render.
- Request coverage now includes upload submit/list/status/result/retry/cancel
  lifecycle contracts, while completed-artifact assertions remain deferred until
  a more controlled local processing environment is available.
- WebSocket message flow. Webhook delivery itself remains deferred to a later
  controlled harness, but submit-time validation is covered.
- can distinguish "feature intentionally disabled" from "feature not configured
  for this run."
- metrics posture, and currently limits itself to deterministic request-level
  checks rather than topology-dependent failover or allowlist scenarios.
- Seeded admin action coverage is limited to deterministic actions in the local
  coordinator environment; broker-dependent worker actions remain deferred.
- `.github/workflows/playwright-pr.yml`
  that always uploads artifacts and emits a normalized markdown summary. Missing
  environment configuration is reported as `BLOCKED`, not silently green.
- `scripts/run_phase8_release_gate.py` with staged Playwright
  policy levels so release gating can progress from advisory to strict without
  rewriting the gate wrapper.

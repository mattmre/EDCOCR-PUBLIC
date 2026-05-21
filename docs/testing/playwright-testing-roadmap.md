# Playwright Testing Roadmap

This document defines the first repo-specific Playwright strategy for EDCOCR.

## Why This Needs A Hybrid Strategy

EDCOCR is not a classic frontend-heavy app. The browser-visible surfaces are
mostly operational:

- FastAPI Swagger UI (`/docs`)
- FastAPI ReDoc (`/redoc`)
- Django admin (`/admin/*`)
- coordinator observability surfaces

The product itself is primarily exercised through API, background-processing,
WebSocket, and operational workflows. The Playwright program therefore needs to
cover:

- browser pages that exist today
- request-level API flows
- admin/operator workflows
- enterprise OCR operational standards
- feature-flagged surfaces
- PR reporting and historical run preservation

## Scope Principle

When this plan says "all pages," it means every browser-accessible page that
exists plus every critical operational surface that drives OCR behavior.

That includes Relativity-agent-like or enterprise-imaging-like capabilities that
already exist in this repo, such as:

- OCR job submission and status tracking
- result retrieval and retry behavior
- batch processing
- transforms and stamps
- admin/operator controls
- observability, auth, rate limiting, and recovery checks

It does not assume UI surfaces that do not yet exist in the codebase.

## Ten Phases

### Foundation And Conventions

- Add Playwright package/config scaffold.
- Define run-id convention and artifact layout.
- Establish the initial suite taxonomy:
  - `browser`
  - `admin`
  - `request`
  - later: `ops`, `batch`, `websocket`, `feature-flags`
- Decide how summaries are preserved:
  - heavy artifacts in `playwright-artifacts/runs/<run-id>/`
  - markdown summaries in `docs/testing/`

### Surface Inventory And Testability Hardening

- Freeze the route inventory for current pages and operational surfaces.
- Add stable selectors or accessibility-friendly hooks where custom UI appears.
- Document environment prerequisites for:
  - FastAPI
  - coordinator/Django admin
  - auth-enabled variants
  - feature-flagged variants

### Browser Smoke Coverage

- Cover every current page that a browser can open:
  - `/docs`
  - `/redoc`
  - `/admin/login/`
  - `/admin/`
  - admin changelists and detail pages
- Validate:
  - page load
  - navigation
  - visible actions/buttons
  - basic accessibility landmarks
  - console errors

### Admin Workflow Coverage

- Add operator workflow tests for Django admin:
  - view job list
  - view job detail
  - retry failed jobs
  - cancel running jobs
  - drain workers
  - mark workers offline
  - ping workers
- Build seed-data fixtures so admin tests are deterministic.

### OCR Job Lifecycle API Coverage

- Cover the full local API lifecycle:
  - `GET /api/v1/health`
  - `POST /api/v1/jobs`
  - `GET /api/v1/jobs`
  - `GET /api/v1/jobs/{job_id}`
  - `GET /api/v1/jobs/{job_id}/result`
  - `GET /api/v1/jobs/{job_id}/result/download`
  - `POST /api/v1/jobs/{job_id}/retry`
  - `DELETE /api/v1/jobs/{job_id}`
- Assert enterprise OCR outcomes, not just HTTP codes:
  - artifacts created
  - custody outputs present when enabled
  - failure/retry paths behave correctly
  - timeouts and error states are explicit

### Batch, WebSocket, And Webhook Coverage

- Cover:
  - batch submit/list/status/cancel/retry
  - WebSocket auth and status updates
  - webhook delivery expectations
- Add route-level assertions for:
  - auth frame flow
  - terminal job events
  - rate and connection guardrails

### Feature-Flagged Enterprise Surfaces

- Add explicit matrix runs for:
  - `ENABLE_TRANSFORMS=true`
  - `ENABLE_STAMPING=true`
  - `ENABLE_MULTITENANCY=true`
- Cover:
  - transform listing and execution
  - stamp listing and execution
  - tenant admin routes
  - tenant usage and API key management

### Operational Standards And Resilience

- Translate the existing operational docs into executable checks for:
  - auth required vs exempt paths
  - rate limiting
  - IP allowlist behavior
  - metrics endpoint auth modes
  - Prometheus endpoint behavior
  - job timeout overrides
  - crash/restart/recovery expectations where practical
- Use `docs/production-validation.md` and `docs/FAILOVER-RUNBOOK.md` as source material.

### PR Gate, Reporting, And Historical Preservation

- Create a PR-facing test workflow that produces:
  - HTML report
  - JUnit XML
  - JSON summary
  - blob report for merged multi-shard runs
- Add a markdown summary per meaningful run.
- Maintain a run ledger so results remain reviewable over time.
- Define statuses:
  - `PASS`
  - `PASS_WITH_FOLLOWUPS`
  - `BLOCKED`
  - `FAIL`

Status: implemented for PR workflow, normalized summary generation, and artifact
upload. Automatic in-repo ledger commits from CI remain intentionally deferred.

### Release-Gate Integration And Coverage Expansion

- Promote the suite from informative to gating in stages:
  - smoke only
  - admin + OCR lifecycle
  - feature flags
  - operational checks
- Add richer coverage:
  - accessibility scans
  - visual snapshots where stable
  - cross-browser expansion
  - sharded CI execution

Status: staged release-gate integration is implemented through normalized
summary input plus policy levels in `run_phase8_release_gate.py`. The broader
coverage-expansion items remain future work.

## Coverage Layers

Use these layers together:

1. Browser smoke
2. Authenticated admin workflows
3. Request/API lifecycle validation
4. WebSocket and async-event coverage
5. Operational and resilience assertions
6. Reporting, artifacts, and historical retention

## PR Workflow Placeholder

For each PR, the eventual Playwright workflow should answer:

1. What surfaces changed?
2. What Playwright suites ran?
3. What failed, and was it product, environment, or expected deferral?
4. What follow-up tests are now required?
5. Is the PR safe to merge?

Use `docs/testing/playwright-run-template.md` for the summary shape.

## Historical Preservation Rules

- Never overwrite artifact directories; use unique run ids.
- Never rely only on ephemeral CI reports.
- Preserve lightweight markdown summaries in-repo.
- Preserve the latest run and the most recent PR run in an easy-to-find ledger.

## External References

- Playwright API testing: https://playwright.dev/docs/api-testing
- Playwright projects: https://playwright.dev/docs/test-projects
- Playwright reporters: https://playwright.dev/docs/test-reporters
- Playwright sharding: https://playwright.dev/docs/test-sharding
- Playwright trace viewer: https://playwright.dev/docs/trace-viewer
- Playwright locators: https://playwright.dev/docs/locators
- GitHub Actions artifact storage: https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/storing-and-sharing-data-from-a-workflow

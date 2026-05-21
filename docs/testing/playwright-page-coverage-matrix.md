# Playwright Page And Surface Coverage Matrix

This is the current page-by-page and surface-by-surface plan for Playwright.

Implemented status reflects what is actually present in the repo today, not the
eventual roadmap target.

## Browser Pages

| Surface | Type | What To Check | Phase | Implemented |
|---|---|---|---|---|
| `/docs` | FastAPI Swagger UI | page loads, core endpoints visible, no critical console errors, auth docs accurate | 3 | yes |
| `/redoc` | FastAPI ReDoc | page loads, OCR lifecycle endpoints visible, no critical console errors | 3 | yes |
| `/admin/login/` | Django admin page | username/password fields, login button, redirect behavior, accessibility basics | 3 | yes |
| `/admin/` | Django admin index | authenticated load, app/model cards visible, navigation to model changelists | 3-4 | yes |
| `/admin/jobs/job/` | Django admin changelist | filters, search, row rendering, action dropdowns | 4 | yes (smoke + seeded retry/cancel) |
| `/admin/jobs/job/<id>/change/` | Django admin detail page | status, progress, inline page results, inline custody events | 4 | yes (seeded) |
| `/admin/jobs/worker/` | Django admin changelist | worker list renders, action dropdowns work, status badge visibility | 4 | yes (smoke + seeded offline) |
| `/admin/jobs/pageresult/` | Django admin changelist | list rendering, filters, raw-id navigation | 4 | yes (smoke) |
| `/admin/jobs/custodyevent/` | Django admin changelist | list rendering, search/filter behavior, hash display | 4 | yes (smoke) |

## API And Async Surfaces

| Surface | Type | What To Check | Phase | Implemented |
|---|---|---|---|---|
| `GET /api/v1/health` | request | status/version/uptime payload, auth exemption, basic availability | 5 | yes (smoke) |
| `POST /api/v1/jobs` | request | file upload, source path submit, validation failures, auth, timeout override | 5 | yes (request contract) |
| `GET /api/v1/jobs` | request | pagination, auth, visibility of created jobs | 5 | yes (request contract) |
| `GET /api/v1/jobs/{job_id}` | request | progress/status model, settings exposure, webhook status | 5 | yes (request contract) |
| `GET /api/v1/jobs/{job_id}/result` | request | artifact metadata contract | 5 | yes (non-terminal contract) |
| `GET /api/v1/jobs/{job_id}/result/download` | request | download headers, file presence, auth | 5 | no |
| `POST /api/v1/jobs/{job_id}/retry` | request | failure-path retry, new job id, guardrails | 5 | yes (active-job contract) |
| `DELETE /api/v1/jobs/{job_id}` | request | cancel behavior, status transition | 5 | yes (request contract) |
| `POST /api/v1/jobs/batch` | request | batch submit contract and validation | 6 | yes (request contract) |
| `GET /api/v1/jobs/batch` | request | batch listing and pagination | 6 | yes (request contract) |
| `GET /api/v1/jobs/batch/{batch_id}` | request | batch status/progress | 6 | yes (request contract) |
| `DELETE /api/v1/jobs/batch/{batch_id}` | request | batch cancel behavior | 6 | yes (request contract) |
| `POST /api/v1/jobs/batch/{batch_id}/retry` | request | retry failed items only | 6 | yes (request guardrail contract) |
| `/ws/jobs/{job_id}` | websocket | auth, connected event, progress events, terminal events, connection limits | 6 | yes (connected/ping/error browser coverage) |
| webhook validation on submit routes | request | invalid URL rejection and explicit async-delivery deferral | 6 | yes (validation contract) |

## Feature-Flagged Enterprise Surfaces

| Surface | Type | What To Check | Phase | Implemented |
|---|---|---|---|---|
| `GET /api/v1/transforms` | request | feature-disabled vs enabled behavior, metadata listing | 7 | yes (env-gated request matrix) |
| `GET /api/v1/transforms/{operation_id}` | request | metadata contract, 404 handling | 7 | yes (env-gated request matrix) |
| `POST /api/v1/transforms/execute` | request | role enforcement, path safety, output validation, custody diagnostics | 7 | yes (env-gated execution contract) |
| `GET /api/v1/stamps` | request | feature-disabled vs enabled behavior, metadata listing | 7 | yes (env-gated request matrix) |
| `GET /api/v1/stamps/{operation_id}` | request | metadata contract, 404 handling | 7 | yes (env-gated request matrix) |
| `POST /api/v1/stamps/execute` | request | role enforcement, validation gates, Bates/designation paths | 7 | yes (env-gated execution contract) |
| `POST /api/v1/admin/tenants` | request | platform-admin enforcement, create flow | 7 | yes (env-gated request matrix) |
| `GET /api/v1/admin/tenants` | request | tenant scoping, admin permissions | 7 | yes (env-gated request matrix) |
| `GET /api/v1/admin/tenants/{tenant_id}` | request | tenant detail + usage contract | 7 | yes (env-gated request matrix) |
| `PUT /api/v1/admin/tenants/{tenant_id}` | request | update validation and permission model | 7 | yes (env-gated request matrix) |
| `POST /api/v1/admin/tenants/{tenant_id}/suspend` | request | lifecycle transition rules | 7 | yes (env-gated request matrix) |
| `POST /api/v1/admin/tenants/{tenant_id}/activate` | request | lifecycle transition rules | 7 | yes (env-gated request matrix) |
| `POST /api/v1/admin/tenants/{tenant_id}/keys` | request | key creation, permission enforcement | 7 | yes (env-gated request matrix) |
| `DELETE /api/v1/admin/tenants/{tenant_id}/keys/{key_id}` | request | key revocation | 7 | yes (env-gated request matrix) |
| `GET /api/v1/admin/tenants/{tenant_id}/usage` | request | usage contract, empty-period behavior | 7 | yes (env-gated request matrix) |

## Operational Surfaces

| Surface | Type | What To Check | Phase | Implemented |
|---|---|---|---|---|
| `GET /api/v1/metrics/` | request | 200/401 behavior, `X-Api-Key` vs bearer auth, payload contract | 8 | yes (env-gated ops matrix) |
| `GET /api/v1/prometheus/` | request | 200/401 behavior, content type, auth modes | 8 | yes (env-gated ops matrix) |
| `OCR_API_KEY` protected paths | policy | required vs exempt paths | 8 | yes (env-gated ops matrix) |
| `API_ALLOWED_IPS` | policy | allowlist enforcement in non-proxy-safe scenarios | 8 | deferred |
| `JOB_PROCESSING_TIMEOUT_MINUTES` | behavior | default timeout and per-job override behavior | 8 | yes (submit/status override contract) |
| rate limiting | behavior | 429 responses and error payloads | 8 | deferred |
| recovery/failover checks | ops | health, metrics, and representative recovery probes from runbook | 8 | deferred |

## OCR/Enterprise Assertions To Add Over Time

These are not separate pages, but they should become first-class assertions in
the request and operational suites:

- searchable PDF and text artifacts produced
- image-only fallback preserved when OCR fails
- custody artifacts and lifecycle events recorded correctly
- batch fan-out/fan-in behavior remains intact
- transform/stamp outputs remain deterministic
- metrics accurately reflect worker/job/page state
- admin/operator actions produce the expected backend state changes

## Reporting And Gate Automation

| Surface | Type | What To Check | Phase | Implemented |
|---|---|---|---|---|
| `.github/workflows/playwright-pr.yml` | CI | PR-triggered run, artifact upload, normalized summary emission | 9 | yes |
| `scripts/summarize_playwright_run.py` | reporting | markdown summary, compact JSON summary, ledger append support | 9 | yes |
| GitHub Actions step summary | reporting | retained markdown surfaced in CI job summary | 9 | yes |
| `scripts/run_phase8_release_gate.py` Playwright integration | release gate | staged policy enforcement against normalized summary and required projects | 10 | yes |

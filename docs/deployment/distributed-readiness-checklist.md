# Distributed Coordinator Readiness Checklist

Use this checklist before setting `DEPLOYMENT_ENV=production` for `coordinator/`.

## 1. Environment and Secrets
- [ ] `coordinator/.env` exists and contains non-placeholder values.
- [ ] `DJANGO_SECRET_KEY`, `POSTGRES_PASSWORD`, `RABBITMQ_PASSWORD`, `REDIS_PASSWORD` are rotated and stored in a secret manager.
- [ ] `DEPLOYMENT_ENV=staging` for validation runs.
- [ ] `PRODUCTION_READINESS_ACK=false` during staging.

## 2. Baseline Validation (Staging)
- [ ] Run coordinator unit/integration tests:
  - `pytest -c coordinator/pytest.ini coordinator/jobs/tests -m "not integration"`
- [ ] Validate env completeness:
  - `python scripts/validate_phase7c_env.py --env-file coordinator/.env --strict-placeholders`
- [ ] Capture baseline metrics:
  - `python scripts/capture_phase7c_metrics.py --api-url http://localhost:8000/api/v1/metrics/ --env-file coordinator/.env --report docs/reports/phase7c-baseline-metrics.md --json-output docs/reports/phase7c-baseline-metrics.json`

## 3. Canary Validation
- [ ] Execute staged workload (known representative document set).
- [ ] Capture canary metrics:
  - `python scripts/capture_phase7c_metrics.py --api-url http://localhost:8000/api/v1/metrics/ --env-file coordinator/.env --report docs/reports/phase7c-canary-metrics.md --json-output docs/reports/phase7c-canary-metrics.json`
- [ ] Evaluate go/no-go:
  - `python scripts/evaluate_phase7c_canary.py --baseline-json docs/reports/phase7c-baseline-metrics.json --canary-json docs/reports/phase7c-canary-metrics.json --report docs/reports/phase7c-canary-decision.md --json-output docs/reports/phase7c-canary-decision.json`
- [ ] Decision is `GO` and no unresolved blocker remains.

## 4. Production Cutover Guard
- [ ] Set `DEPLOYMENT_ENV=production`.
- [ ] Set `PRODUCTION_READINESS_ACK=true`.
- [ ] Archive evidence:
  - baseline metrics report
  - canary metrics report
  - canary decision report
  - test execution logs

## 5. Post-Cutover Checks
- [ ] Worker heartbeats stable (`jobs_worker` table updates).
- [ ] Queue lag remains within SLO.
- [ ] Error rate and page latency within canary thresholds.
- [ ] Rollback command validated and documented for on-call.


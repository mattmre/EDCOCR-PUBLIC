# SOC 2 Type II Readiness -- EDCOCR v1.0

## Overview

This document maps EDCOCR security controls to AICPA Trust Service Criteria for SOC 2 Type II compliance readiness. It covers all nine Common Criteria (CC1-CC9) plus the Availability, Processing Integrity, and Confidentiality supplemental criteria.

**Last Updated**: 2026-05-20

---

## CC1: Control Environment

### CC1.1 -- Integrity and Ethical Values
- [x] SECURITY.md documents responsible disclosure
- [x] CODEOWNERS enforces code review requirements
- [x] Pre-commit hooks for linting and security checks

### CC1.2 -- Board/Management Oversight
- [ ] Risk assessment documentation
- [x] Formal security audit checklist (docs/security-audit-checklist.md)

### CC1.3 -- Organizational Structure and Authority
- [x] CODEOWNERS defines code ownership and review responsibility
- [x] Django admin RBAC separates admin, operator, and viewer roles
- [x] Multi-tenant isolation enforces organizational boundaries

---

## CC2: Communication and Information

### CC2.1 -- Internal Communication
- [x] `docs/` suite serves as comprehensive internal dev guide
- [x] Extensive documentation in docs/ directory (200+ files)
- [x] Architecture decision records in docs/architecture/

### CC2.2 -- External Communication
- [x] API stability contract (docs/api-stability-contract.md)
- [x] CHANGELOG.md tracks all changes
- [x] README.md with quickstart and feature documentation

### CC2.3 -- Internal Control Communication
- [x] Session logs document all significant development decisions
- [x] Compliance readiness docs in docs/compliance/
- [x] Data retention policy (docs/compliance/data-retention-policy.md)

---

## CC3: Risk Assessment

### CC3.1 -- Risk Identification
- [x] Security audit with 13 security domains assessed
- [x] Automated security scanner (scripts/security_scan.py)
- [ ] Formal risk register with likelihood/impact ratings
- [ ] Annual risk assessment schedule

### CC3.2 -- Fraud Risk Assessment
- [x] CTC-only OCR prevents text hallucination/fabrication
- [x] Hash-chained custody prevents evidence tampering
- [x] Timing-safe key comparison prevents timing attacks
- [ ] Formal fraud risk assessment document

### CC3.3 -- Risk from Changes
- [x] CI/CD gates prevent untested changes from reaching production
- [x] Pre-commit hooks catch issues before commit
- [x] CODEOWNERS ensures appropriate review of changes

---

## CC4: Monitoring Activities

### CC4.1 -- Ongoing Monitoring
- [x] Prometheus metrics collection (7 metric families)
- [x] 5 PrometheusRule alerts configured
- [x] Grafana dashboard (7 panels + 10-panel canary dashboard)
- [x] Docker healthcheck (30-second intervals)
- [x] Worker heartbeat monitoring
- [x] SLA monitoring with sliding-window metrics
- [x] Queue depth alerting (`api/queue_alerting.py`)

### CC4.2 -- Deficiency Evaluation
- [x] CI/CD pipeline (GitHub Actions with Python 3.10/3.11 matrix)
- [x] 7,500+ automated tests (1,680 root + coordinator suite)
- [x] Ruff linting with strict configuration (0 issues)
- [x] Playwright E2E test suite (phases 2-10)
- [x] Helm chart lint in CI

---

## CC5: Control Activities

### CC5.1 -- Logical Access
- [x] API key authentication (X-API-Key header)
- [x] RBAC with 3 roles (viewer, operator, admin)
- [x] Multi-tenant key isolation with SHA-256 hashing
- [x] OAuth2/OIDC JWT validation with JWKS
- [x] Rate limiting per endpoint (slowapi)
- [x] Per-tenant quotas (concurrent jobs, pages/month, storage)
- [ ] MFA for admin access (TOTP or hardware key)

### CC5.2 -- Physical Access
- N/A (cloud-hosted; physical access managed by cloud provider)

### CC5.3 -- Data Protection
- [x] CTC-only OCR (zero hallucination -- no generated text)
- [x] Chain of custody (SHA-256 hash-chained JSONL audit trail)
- [x] Audit trail integrity verification (`verify_chain`)
- [x] Path traversal prevention with multi-layer validation
- [x] SSRF protection with redirect blocking
- [x] File type validation (magic bytes + extension)
- [x] Data retention policy documented (docs/compliance/data-retention-policy.md)
- [x] Cleanup commands for data disposal (`cleanup_old_jobs.py`, `purge_temp_files.py`)
- [ ] Encryption at rest for PII entities (field-level encryption for `PiiEntity` model)
- [ ] Encryption at rest for webhook secrets (encrypt `webhook_secret` column)

### CC5.4 -- System Development
- [x] Semantic versioning contract (docs/api-stability-contract.md)
- [x] API stability tiers (stable/beta/experimental)
- [x] Breaking change definition and deprecation policy
- [x] Pre-commit hooks (ruff + pytest gates)
- [x] CI/CD with automated testing (GitHub Actions)
- [x] Playwright browser-level E2E testing

---

## CC6: Logical and Physical Access Controls

### CC6.1 -- Access Security (Infrastructure Logical Access)
- [x] API key transport via header (not URL parameters)
- [x] Timing-safe key comparison (`secrets.compare_digest`)
- [x] Key expiration and revocation support (`expires_at` field)
- [x] IP-based allowlisting (opt-in via `API_ALLOWED_IPS`)
- [x] WebSocket authentication (3 methods: API key, token, query param)

### CC6.2 -- Access Provisioning (New User Registration)
- [x] Multi-tenant API key management via admin endpoints
- [x] Tenant suspension/activation controls
- [x] Per-tenant usage tracking and resource limits
- [x] Key format: `ocr_{urlsafe44}` -- unique, unpredictable

### CC6.3 -- Access Modification (Role and Permission Changes)
- [x] API key revocation endpoint
- [x] Tenant status management (active/suspended)
- [x] Django admin interface for user/role management

### CC6.4 -- Access Removal (Deprovisioning)
- [x] API key deletion permanently revokes access
- [x] Tenant suspension immediately blocks all API requests
- [x] `cleanup_old_jobs.py` removes data associated with terminated processing

### CC6.5 -- Physical Access Restrictions
- N/A (delegated to cloud provider / data center operator)

### CC6.6 -- Logical Access Security Measures
- [x] Kubernetes NetworkPolicy templates restrict pod-to-pod communication
- [x] Worker mTLS for distributed pipeline inter-service authentication
- [x] Presigned URLs with expiry for S3 object access (no long-lived credentials on workers)

### CC6.7 -- Restriction and Authorization of System Changes
- [x] CODEOWNERS requires designated reviewer approval before merge
- [x] Branch protection rules on main branch
- [x] CI gates block merge on test failure
- [x] Squash merge strategy enforces clean commit history
- [x] Admin merge requires explicit `--admin` flag

### CC6.8 -- Access Review
- [ ] Periodic API key access review process
- [ ] Automated unused key detection and notification
- [x] Worker heartbeat monitoring identifies stale/inactive workers
- [x] Fleet status command for operational visibility (`fleet_status.py`)

---

## CC7: System Operations

### CC7.1 -- Detection of Changes to Infrastructure and Software
- [x] Git-based version control with full commit history
- [x] Pull request workflow with CODEOWNERS review
- [x] Semantic versioning (`version.py`)
- [x] CHANGELOG maintenance
- [x] CI/CD automated testing gate (GitHub Actions)
- [x] Dependency pinning (`requirements.txt`)

### CC7.2 -- Monitoring of Infrastructure and Software
- [x] Prometheus metrics collection with custom ORM-backed collector
- [x] 5 PrometheusRule alerts (queue depth, worker offline, error rate, latency, disk)
- [x] Grafana dashboards (7-panel operations + 10-panel canary)
- [x] Worker fleet status monitoring with heartbeat tracking
- [x] Queue depth tracking and alerting (`api/queue_alerting.py`)
- [x] GPU utilization monitoring
- [x] SLA/SLO monitoring with sliding-window metrics and violation reports
- [x] Docker healthcheck (30-second interval, monitor heartbeat file)
- [x] OpenTelemetry distributed tracing support (`api/tracing.py`)

### CC7.3 -- Evaluation of Security Events (Incident Response)
- [x] Failover drill framework (`failover_drill.py`)
- [x] Page-level crash resume for processing continuity
- [x] Graceful shutdown (SIGTERM/SIGINT handling)
- [x] Failover runbook (`docs/FAILOVER-RUNBOOK.md`)
- [x] Production cutover runbook (`docs/operations/production-cutover-runbook.md`)
- [ ] Formal incident response plan document (IRP)
- [ ] Incident severity classification matrix (SEV1-SEV4)
- [ ] Post-incident review (PIR) template

### CC7.4 -- Implementation of Incident Response Activities
- [x] Alert rules trigger webhook notifications for incident events
- [x] Prometheus alerting pipeline for automated escalation
- [x] Queue alerting for anomaly detection
- [ ] Documented escalation procedures with contact chains
- [ ] Incident communication templates (internal and external)

### CC7.5 -- Communication of System Changes
- [x] CHANGELOG.md documents all user-facing changes
- [x] API stability contract defines breaking change communication
- [x] Session logs capture technical decisions and rationale
- [x] PR descriptions provide change context and review artifacts

---

## CC8: Change Management

### CC8.1 -- Change Authorization and Approval
- [x] CODEOWNERS requires designated reviewer for all changes
- [x] CI gates block merge on test failure (7,500+ tests)
- [x] Admin merge requires explicit `--admin` action
- [x] Squash merge strategy with clean commit messages
- [x] Pre-commit hooks enforce code quality before commit

### CC8.2 -- Baseline Configuration
- [x] Helm `values.yaml` defines infrastructure baseline configuration
- [x] Docker images provide reproducible build environments
- [x] `requirements.txt` pins all Python dependency versions
- [x] Environment variable configuration with validation (`validate_phase7c_env.py`)

### CC8.3 -- Change Detection and Monitoring
- [x] Git tracks all code changes with author attribution
- [x] CI/CD pipeline runs on every PR and push to main
- [x] Dependabot (when enabled) monitors dependency vulnerabilities
- [x] Ruff linting detects code quality regressions
- [ ] Runtime configuration drift detection

---

## CC9: Risk Mitigation

### CC9.1 -- Risk Identification and Assessment
- [x] security reviews identified and remediated 28+ findings
- [x] Automated security scanner (`scripts/security_scan.py`)
- [x] Input validation across all entry points (API, CLI, file watcher)

### CC9.2 -- Vendor and Third-Party Risk
- [x] Air-gapped deployment support (pre-baked models, no runtime downloads)
- [x] Credential manager with Vault/AWS SM/KMS backends (`credential_manager.py`)
- [x] Placeholder secret detection prevents deployment with default credentials
- [x] Production credential validation in release gate
- [x] CTC-only OCR (no data sent to external AI services)

### CC9.3 -- Risk Mitigation Controls
- [x] Kubernetes PDBs prevent simultaneous pod disruption
- [x] Redis Sentinel failover for cache/broker resilience
- [x] PostgreSQL backup CronJobs for database recovery
- [x] RabbitMQ quorum queues for message durability
- [x] Page-level crash resume for processing continuity

---

## Additional Trust Services Criteria

### Availability (A1)

#### A1.1 -- Processing Capacity and Availability
- [x] SLA monitoring (`sla_monitoring.py`) with per-tenant overrides
- [x] KEDA autoscaling for GPU and CPU workers
- [x] Kubernetes PDBs (Pod Disruption Budgets)
- [x] Docker healthcheck with monitor heartbeat

#### A1.2 -- Recovery and Continuity
- [x] Redis Sentinel failover (opt-in)
- [x] PostgreSQL backup CronJobs (opt-in)
- [x] Page-level crash resume for processing continuity
- [x] Failover drill framework with documented runbook

#### A1.3 -- Environmental Protections
- N/A (delegated to cloud provider / data center operator)

### Processing Integrity (PI1)

#### PI1.1 -- Accuracy and Completeness of Processing
- [x] CTC-only OCR guarantees zero hallucination
- [x] Per-page confidence tracking with quality classification (good/acceptable/poor/failed)
- [x] Forensic image preservation (OCR failure fallback to image-only -- never discard evidence)
- [x] Processing validation (`validation.py`) with integrity reports

#### PI1.2 -- Timeliness of Processing
- [x] SLA monitoring with throughput tracking (pages per minute, docs per hour)
- [x] Queue depth alerting for backlog detection
- [x] Configurable processing timeout (`JOB_PROCESSING_TIMEOUT_MINUTES`)

#### PI1.3 -- Audit Trail for Processing
- [x] Hash-chained custody logging for document chain of evidence
- [x] API request audit logging with correlation IDs
- [x] Per-page OCR result tracking in coordinator database

### Confidentiality (C1)

#### C1.1 -- Identification of Confidential Information
- [x] PII/PHI detection via NER module (`ner.py`)
- [x] PII spatial extraction with bounding box coordinates
- [x] Entity consolidation across pages (`entity_consolidator.py`)
- [x] Document classification for sensitivity categorization

#### C1.2 -- Disposal of Confidential Information
- [x] Cleanup commands for data disposal (`cleanup_old_jobs.py`, `purge_temp_files.py`)
- [x] Data retention policy with per-class retention periods
- [x] Configurable retention via environment variables

#### C1.3 -- Confidentiality of System Output
- [x] HMAC-SHA256 signed webhook payloads
- [x] SSRF protection prevents data exfiltration via webhooks
- [x] Presigned URLs with expiry for S3 access

---

## Comprehensive Gap Analysis

### CC Control Gap Table

| CC | Sub-Control | Description | Status | Evidence | Gap ID |
|----|-------------|-------------|--------|----------|--------|
| CC1 | CC1.1 | Integrity and ethical values | Met | SECURITY.md, CODEOWNERS, pre-commit | -- |
| CC1 | CC1.2 | Management oversight | Partial | Security audit checklist | SOC2-G1 |
| CC1 | CC1.3 | Organizational structure | Met | CODEOWNERS, RBAC | -- |
| CC2 | CC2.1 | Internal communication | Met | `docs/` | -- |
| CC2 | CC2.2 | External communication | Met | API contract, CHANGELOG | -- |
| CC2 | CC2.3 | Internal control communication | Met | Session logs, compliance docs | -- |
| CC3 | CC3.1 | Risk identification | Partial | Security scan, but no formal register | SOC2-G2 |
| CC3 | CC3.2 | Fraud risk assessment | Partial | CTC-only, hash chain, but no formal doc | SOC2-G3 |
| CC3 | CC3.3 | Risk from changes | Met | CI/CD, pre-commit, CODEOWNERS | -- |
| CC4 | CC4.1 | Ongoing monitoring | Met | Prometheus, Grafana, alerts | -- |
| CC4 | CC4.2 | Deficiency evaluation | Met | CI/CD, 7,500+ tests, ruff | -- |
| CC5 | CC5.1 | Logical access | Partial | API key, RBAC, but no MFA | SOC2-G4 |
| CC5 | CC5.2 | Physical access | N/A | Cloud provider | -- |
| CC5 | CC5.3 | Data protection | Partial | Strong controls, missing encryption at rest | SOC2-G5 |
| CC5 | CC5.4 | System development | Met | Versioning, CI/CD, testing | -- |
| CC6 | CC6.1 | Access security | Met | API key, timing-safe, IP allowlist | -- |
| CC6 | CC6.2 | Access provisioning | Met | Multi-tenant keys, quotas | -- |
| CC6 | CC6.3 | Access modification | Met | Key revocation, tenant status | -- |
| CC6 | CC6.4 | Access removal | Met | Key deletion, tenant suspension | -- |
| CC6 | CC6.5 | Physical access restrictions | N/A | Cloud provider | -- |
| CC6 | CC6.6 | Logical access security | Met | NetworkPolicy, mTLS, presigned URLs | -- |
| CC6 | CC6.7 | System change restriction | Met | CODEOWNERS, CI gates, admin merge | -- |
| CC6 | CC6.8 | Access review | Partial | Heartbeat monitoring, but no key review | SOC2-G6 |
| CC7 | CC7.1 | Change detection | Met | Git, PRs, CI/CD, dependency pinning | -- |
| CC7 | CC7.2 | Infrastructure monitoring | Met | Prometheus, Grafana, SLA, tracing | -- |
| CC7 | CC7.3 | Security event evaluation | Partial | Failover drills, but no formal IRP | SOC2-G7 |
| CC7 | CC7.4 | Incident response activities | Partial | Alerting, but no escalation docs | SOC2-G8 |
| CC7 | CC7.5 | System change communication | Met | CHANGELOG, API contract, session logs | -- |
| CC8 | CC8.1 | Change authorization | Met | CODEOWNERS, CI, admin merge | -- |
| CC8 | CC8.2 | Baseline configuration | Met | Helm, Docker, pinned deps | -- |
| CC8 | CC8.3 | Change detection | Partial | Git, CI, but no runtime drift detection | SOC2-G9 |
| CC9 | CC9.1 | Risk assessment | Met |  reviews, security scanner | -- |
| CC9 | CC9.2 | Vendor risk | Met | Air-gapped, credential manager, CTC-only | -- |
| CC9 | CC9.3 | Risk mitigation | Met | PDBs, Sentinel, backups, crash resume | -- |

---

## Readiness Summary

| Category | Controls Assessed | Implemented | Partial | Gaps |
|----------|-------------------|-------------|---------|------|
| CC1: Control Environment | 3 | 2 | 1 | 0 |
| CC2: Communication | 3 | 3 | 0 | 0 |
| CC3: Risk Assessment | 3 | 1 | 2 | 0 |
| CC4: Monitoring | 2 | 2 | 0 | 0 |
| CC5: Control Activities | 4 | 2 | 2 | 0 |
| CC6: Access Controls | 8 | 7 | 1 | 0 |
| CC7: System Operations | 5 | 3 | 2 | 0 |
| CC8: Change Management | 3 | 2 | 1 | 0 |
| CC9: Risk Mitigation | 3 | 3 | 0 | 0 |
| Availability (A1) | 3 | 3 | 0 | 0 |
| Processing Integrity (PI1) | 3 | 3 | 0 | 0 |
| Confidentiality (C1) | 3 | 3 | 0 | 0 |
| **Total** | **43** | **34** | **9** | **0** |

**Readiness: 79% fully implemented, 100% partially or fully addressed (34/43 full, 9/43 partial)**

No controls are entirely absent. All 9 partial controls have existing technical capabilities that need supplemental organizational documentation or configuration.

---

## Gap Remediation Priority

| ID | Gap | CC Control | Severity | Remediation | Effort |
|----|-----|-----------|----------|-------------|--------|
| SOC2-G1 | No formal risk assessment documentation | CC1.2 | Medium | Create risk assessment template and conduct initial RA | 2-3 days |
| SOC2-G2 | No formal risk register | CC3.1 | Medium | Create risk register with likelihood/impact ratings | 1-2 days |
| SOC2-G3 | No formal fraud risk assessment | CC3.2 | Low | Document existing anti-fraud controls in formal assessment | 1 day |
| SOC2-G4 | No MFA for admin access | CC5.1 | Medium | Integrate TOTP or hardware key for Django admin | 3-5 days |
| SOC2-G5 | PII/webhook encryption at rest | CC5.3 | Medium | Implement field-level encryption for PiiEntity and webhook_secret | 3-5 days |
| SOC2-G6 | No periodic access review process | CC6.8 | Medium | Implement automated unused key detection and review workflow | 2-3 days |
| SOC2-G7 | No formal incident response plan | CC7.3 | Medium | Document IRP with severity matrix and escalation procedures | 2-3 days |
| SOC2-G8 | No documented escalation procedures | CC7.4 | Medium | Create escalation chain and communication templates | 1-2 days |
| SOC2-G9 | No runtime config drift detection | CC8.3 | Low | Add periodic configuration verification check | 2-3 days |

**Total estimated remediation effort: 17-27 days of engineering/documentation work.**

---

## Revision History

| Date | Version | Change |
|------|---------|--------|
| 2026-03-15 | 1.0 | Initial SOC 2 readiness assessment |
| 2026-03-24 | 2.0 | Expanded CC6-CC9 controls, added CC control gap table, updated readiness summary |

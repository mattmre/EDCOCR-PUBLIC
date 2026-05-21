# Risk Register -- EDCOCR v1.0

## Purpose

This document catalogues identified risks to the EDCOCR forensic document processing platform, their likelihood, impact, and mitigation status. It satisfies SOC 2 CC3.1-CC3.4 (risk assessment), HIPAA 164.308(a)(1)(ii)(A) (risk analysis), and FedRAMP RA-2/RA-3 (risk assessment) requirements.

The risk register serves as the central artifact for ongoing risk management. All identified risks are tracked from initial identification through mitigation and periodic reassessment. It is reviewed quarterly and updated after any SEV1/SEV2 incident.

**Last Updated**: 2026-05-20

---

## Risk Assessment Methodology

### Overview

Risks are evaluated using a quantitative Likelihood x Impact matrix. Each risk receives a numeric score from 1 to 25, which maps to a qualitative rating (Low, Medium, High, Critical). The methodology aligns with NIST SP 800-30 Rev. 1 (Guide for Conducting Risk Assessments) and ISO 27005:2022 (Information Security Risk Management).

### Risk Categories

| Category | Code | Description |
|----------|------|-------------|
| Security | SEC | Unauthorized access, data breach, credential compromise, injection attacks |
| Availability | AVL | Service outage, capacity exhaustion, infrastructure failure |
| Compliance | CMP | Regulatory non-compliance, audit findings, legal exposure |
| Data Integrity | DAT | Data loss, corruption, chain-of-custody compromise |
| Operational | OPS | Process failures, misconfigurations, human error, model degradation |

### Likelihood Scale

| Level | Label | Frequency |
|-------|-------|-----------|
| 1 | Rare | Less than once per year |
| 2 | Unlikely | Once per year |
| 3 | Possible | Once per quarter |
| 4 | Likely | Once per month |
| 5 | Almost Certain | Once per week or more |

### Impact Scale

| Level | Label | Description |
|-------|-------|-------------|
| 1 | Negligible | No data impact, no user disruption, cosmetic issue |
| 2 | Minor | Brief degradation, single user affected, quick recovery |
| 3 | Moderate | Partial outage, multiple users affected, hours to recover |
| 4 | Major | Full outage, data loss possible, regulatory notification may be required |
| 5 | Critical | Data breach confirmed, regulatory penalties, litigation, reputational damage |

### Risk Score Calculation

Risk Score = Likelihood x Impact

| Score | Rating | Response Required |
|-------|--------|-------------------|
| 1-4 | Low | Monitor, address during normal maintenance |
| 5-9 | Medium | Mitigate within current release cycle |
| 10-15 | High | Prioritize remediation, track in sprint |
| 16-25 | Critical | Immediate action required, escalate to leadership |

### Heat Map

|  | Impact 1 | Impact 2 | Impact 3 | Impact 4 | Impact 5 |
|--|----------|----------|----------|----------|----------|
| **Likelihood 5** | 5 Med | 10 High | 15 High | 20 Crit | 25 Crit |
| **Likelihood 4** | 4 Low | 8 Med | 12 High | 16 Crit | 20 Crit |
| **Likelihood 3** | 3 Low | 6 Med | 9 Med | 12 High | 15 High |
| **Likelihood 2** | 2 Low | 4 Low | 6 Med | 8 Med | 10 High |
| **Likelihood 1** | 1 Low | 2 Low | 3 Low | 4 Low | 5 Med |

---

## Risk Register

### RISK-001: PII/PHI Data Exposure via Unsecured API

| Field | Value |
|-------|-------|
| **ID** | RISK-001 |
| **Category** | SEC |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 5 (Critical) |
| **Risk Score** | 10 (High) |
| **Description** | An attacker exploits a misconfigured API key or authentication bypass to access PII/PHI entity data or job results. |
| **Existing Controls** | API key authentication (X-API-Key header), rate limiting (slowapi), SSRF protection on webhooks, IP allowlist (opt-in `API_ALLOWED_IPS`), input validation (Pydantic), network policies (Kubernetes) |
| **Residual Risk** | API key rotation is manual; IP allowlist is opt-in |
| **Mitigation Status** | Controlled -- all upstream security findings remediated; Django admin MFA added |
| **Owner** | Security officer |
| **Review Date** | 2026-06-24 |

### RISK-002: Chain-of-Custody Integrity Compromise

| Field | Value |
|-------|-------|
| **ID** | RISK-002 |
| **Category** | DAT |
| **Likelihood** | 1 (Rare) |
| **Impact** | 5 (Critical) |
| **Risk Score** | 5 (Medium) |
| **Description** | A custody JSONL file or CustodyEvent record is modified, breaking the SHA-256 hash chain and invalidating the forensic audit trail. |
| **Existing Controls** | SHA-256 hash chaining, append-only JSONL format, `verify_custody_file` verification function, CustodyEvent model with immutable hash fields, file-based and DB dual storage |
| **Residual Risk** | Custody files on NFS/S3 are writable by operators with storage access; no tamper-proof storage (WORM) configured by default |
| **Mitigation Status** | Controlled -- chain verification is available but not automatically scheduled |
| **Owner** | Engineering lead |
| **Review Date** | 2026-06-24 |

### RISK-003: GPU Worker Pool Exhaustion

| Field | Value |
|-------|-------|
| **ID** | RISK-003 |
| **Category** | AVL |
| **Likelihood** | 3 (Possible) |
| **Impact** | 3 (Moderate) |
| **Risk Score** | 9 (Medium) |
| **Description** | All GPU workers become unavailable due to VRAM exhaustion, driver crashes, or infrastructure failure, causing a complete processing halt. |
| **Existing Controls** | KEDA autoscaling, multi-GPU queue affinity, CPU fallback workers (ONNX backend), worker heartbeat monitoring, Prometheus alerts for worker heartbeat staleness |
| **Residual Risk** | CPU fallback is slower (4-7x); KEDA autoscaling depends on functioning metrics pipeline |
| **Mitigation Status** | Controlled -- CPU worker mode provides degraded-but-functional fallback |
| **Owner** | Engineering lead |
| **Review Date** | 2026-06-24 |

### RISK-004: PostgreSQL Data Loss

| Field | Value |
|-------|-------|
| **ID** | RISK-004 |
| **Category** | DAT |
| **Likelihood** | 1 (Rare) |
| **Impact** | 5 (Critical) |
| **Risk Score** | 5 (Medium) |
| **Description** | PostgreSQL primary fails without recoverable backup, causing loss of job metadata, custody events, and PII entity records. |
| **Existing Controls** | PostgreSQL backup CronJob (opt-in), WAL archiving support, Helm chart PVC storage, `pg_dump` backup scripts in failover runbook |
| **Residual Risk** | Backup CronJob is opt-in (`postgresql.backup.enabled=false` by default); backup verification is manual |
| **Mitigation Status** | Partially controlled -- backup infrastructure exists but requires operator opt-in |
| **Owner** | Operations |
| **Review Date** | 2026-06-24 |

### RISK-005: Credential Leakage in Logs or Configuration

| Field | Value |
|-------|-------|
| **ID** | RISK-005 |
| **Category** | SEC |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 4 (Major) |
| **Risk Score** | 8 (Medium) |
| **Description** | Database passwords, API keys, S3 credentials, or other secrets appear in application logs, Docker image layers, or configuration files committed to version control. |
| **Existing Controls** | `credential_manager.py` (Vault/KMS/env backend), Kubernetes Secrets, Helm `values-secret.yaml` pattern, GitGuardian integration, `.gitignore` for `.env` files, sanitized logging |
| **Residual Risk** | Coordinator `.env` file still uses placeholder credentials (`minioadmin`); no automated secret rotation |
| **Mitigation Status** | Controlled -- credential manager is wired in; live credential cutover is deferred |
| **Owner** | Security officer |
| **Review Date** | 2026-06-24 |

### RISK-006: Denial of Service via API Abuse

| Field | Value |
|-------|-------|
| **ID** | RISK-006 |
| **Category** | AVL |
| **Likelihood** | 3 (Possible) |
| **Impact** | 3 (Moderate) |
| **Risk Score** | 9 (Medium) |
| **Description** | An attacker submits a large volume of OCR jobs or oversized documents, exhausting processing capacity and blocking legitimate users. |
| **Existing Controls** | Rate limiting (slowapi per-IP and per-key), max concurrent jobs (`MAX_CONCURRENT_JOBS`), job queue capacity check, file size validation in API, per-tenant cost tracking |
| **Residual Risk** | Distributed attacks from multiple IPs can bypass per-IP rate limits; no WAF configured by default |
| **Mitigation Status** | Controlled -- rate limiting and capacity checks are enforced |
| **Owner** | Engineering lead |
| **Review Date** | 2026-06-24 |

### RISK-007: Compliance Violation from Data Over-Retention

| Field | Value |
|-------|-------|
| **ID** | RISK-007 |
| **Category** | CMP |
| **Likelihood** | 3 (Possible) |
| **Impact** | 4 (Major) |
| **Risk Score** | 12 (High) |
| **Description** | PII/PHI data or documents are retained beyond the configured retention period due to disabled cleanup jobs, misconfigured env vars, or lack of operator awareness. |
| **Existing Controls** | `cleanup_old_jobs` management command, `PII_ENTITY_RETENTION_DAYS` env var, `LITIGATION_HOLD` enforcement, deletion custody events, data retention policy document, `cleanup_output` management command, `rotate_audit_logs` management command, `purge_pii` command |
| **Residual Risk** | Celery Beat must be configured to run cleanup periodically; retention env vars default to conservative (long) periods |
| **Mitigation Status** | Controlled -- env vars and commands now implemented; requires operator scheduling |
| **Owner** | Compliance officer |
| **Review Date** | 2026-06-24 |

### RISK-008: Supply Chain Vulnerability in Dependencies

| Field | Value |
|-------|-------|
| **ID** | RISK-008 |
| **Category** | SEC |
| **Likelihood** | 3 (Possible) |
| **Impact** | 3 (Moderate) |
| **Risk Score** | 9 (Medium) |
| **Description** | A vulnerability in a Python dependency (PaddleOCR, FastAPI, Django, etc.) is exploited before the project updates to a patched version. |
| **Existing Controls** | Pinned dependencies in `requirements.txt`, Dependabot alerts (GitHub), air-gapped deployment option (pre-baked images), CI security scanning, Trivy container scanning, SBOM generation |
| **Residual Risk** | PaddleOCR/PaddlePaddle are large packages with complex native code; patching cadence is slower than web frameworks |
| **Mitigation Status** | Controlled -- pinned versions, CI scanning, and Trivy container scanning; quarterly review cadence for PaddlePaddle 3.x migration |
| **Owner** | Engineering lead |
| **Review Date** | 2026-06-24 |

### RISK-009: Message Broker Failure Causing Job Loss

| Field | Value |
|-------|-------|
| **ID** | RISK-009 |
| **Category** | AVL |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 4 (Major) |
| **Risk Score** | 8 (Medium) |
| **Description** | RabbitMQ or Redis failure causes in-flight Celery tasks to be lost, resulting in jobs stuck in processing state with no completion or error notification. |
| **Existing Controls** | RabbitMQ quorum queues (durable, replicated), Redis Sentinel failover, stale job cleanup (coordinator), page-level crash resume, Celery `acks_late` configuration, randomized Erlang cookie |
| **Residual Risk** | HA configuration is opt-in; default single-node deployments have no failover |
| **Mitigation Status** | Controlled -- HA infrastructure exists; requires operator enablement |
| **Owner** | Operations |
| **Review Date** | 2026-06-24 |

### RISK-010: Forensic Evidence Loss from Storage Failure

| Field | Value |
|-------|-------|
| **ID** | RISK-010 |
| **Category** | DAT |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 5 (Critical) |
| **Risk Score** | 10 (High) |
| **Description** | NFS or S3 storage failure causes loss of source documents, OCR output PDFs, or extracted text files that cannot be regenerated. |
| **Existing Controls** | Dual NFS/S3 storage backend, S3 versioning support, `nfs_to_s3_migration.py` with checksum verification, Kubernetes PVC with storage class redundancy |
| **Residual Risk** | NFS single-point-of-failure in non-HA deployments; S3 cross-region replication not configured by default |
| **Mitigation Status** | Partially controlled -- S3 backend provides durability; NFS deployments need external backup |
| **Owner** | Operations |
| **Review Date** | 2026-06-24 |

### RISK-011: Injection Attack via Malicious Document Content

| Field | Value |
|-------|-------|
| **ID** | RISK-011 |
| **Category** | SEC |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 4 (Major) |
| **Risk Score** | 8 (Medium) |
| **Description** | A crafted document containing embedded scripts, path traversal sequences, or XSS payloads in text content is processed by the pipeline, leading to command injection, directory traversal in output paths, or stored XSS in downstream consumers of extracted text or HTML table exports. |
| **Existing Controls** | Path segment sanitization (`sanitize_filename` in `ocr_utils.py`), XSS sanitization in HTML table export, Pydantic input validation on API endpoints, filename length enforcement, path traversal protection |
| **Residual Risk** | Extracted text is stored as-is (not sanitized for downstream HTML contexts); PDF metadata fields are not scrubbed |
| **Mitigation Status** | Controlled -- input sanitization and path traversal protection are enforced; downstream consumers must apply their own output encoding |
| **Owner** | Engineering lead |
| **Review Date** | 2026-06-24 |

### RISK-012: Unauthorized Django Admin Access

| Field | Value |
|-------|-------|
| **ID** | RISK-012 |
| **Category** | SEC |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 4 (Major) |
| **Risk Score** | 8 (Medium) |
| **Description** | An attacker gains access to the Django admin interface through credential stuffing, session hijacking, or exploitation of a missing MFA requirement, allowing direct modification of job records, worker registrations, or PII entity data. |
| **Existing Controls** | Django admin MFA via django-otp (, `ADMIN_MFA_REQUIRED` env var), Django session framework, CSRF protection, Kubernetes network policies restricting admin port access, `is_staff` permission requirement |
| **Residual Risk** | MFA enforcement depends on `ADMIN_MFA_REQUIRED` env var being set in production; admin session timeout is Django default (2 weeks) |
| **Mitigation Status** | Controlled -- MFA is implemented and auto-enabled when `DEPLOYMENT_ENV=production` |
| **Owner** | Security officer |
| **Review Date** | 2026-06-24 |

### RISK-013: API Key Compromise

| Field | Value |
|-------|-------|
| **ID** | RISK-013 |
| **Category** | SEC |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 4 (Major) |
| **Risk Score** | 8 (Medium) |
| **Description** | An API key is leaked through logs, client-side code, or configuration sharing, granting an unauthorized party full access to the OCR API including job submission, result retrieval, and PII entity access. |
| **Existing Controls** | `list_api_keys` and `revoke_api_key` management commands, `audit_api_access` command for usage review, API audit logging, rate limiting per key, IP allowlist (opt-in) |
| **Residual Risk** | No automated key expiration policy; leaked keys remain valid until manually revoked; no per-key scope restrictions |
| **Mitigation Status** | Controlled -- key review and revocation commands are available; periodic access review is operator responsibility |
| **Owner** | Security officer |
| **Review Date** | 2026-06-24 |

### RISK-014: OCR Model Drift and Accuracy Degradation

| Field | Value |
|-------|-------|
| **ID** | RISK-014 |
| **Category** | OPS |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 3 (Moderate) |
| **Risk Score** | 6 (Medium) |
| **Description** | The pre-baked PaddleOCR models produce degraded accuracy over time as document types evolve (new fonts, layouts, degradation patterns) or after a model version update introduces regressions, leading to reduced OCR quality without detection. |
| **Existing Controls** | Per-page confidence tracking (`validation.py`), quality classification (high/medium/low/failed), adaptive DPI escalation for low-confidence pages, engine selection routing (quality-based Tesseract/PaddleOCR), benchmark harness (`benchmark_ocr.py`), benchmark comparison framework (`benchmark_comparison.py`) |
| **Residual Risk** | No automated accuracy regression detection in production; confidence thresholds are static; no ground-truth corpus for continuous validation |
| **Mitigation Status** | Partially controlled -- confidence tracking detects per-page issues; systematic regression detection requires manual benchmark runs |
| **Owner** | Engineering lead |
| **Review Date** | 2026-06-24 |

### RISK-015: Storage Exhaustion Halting Pipeline

| Field | Value |
|-------|-------|
| **ID** | RISK-015 |
| **Category** | AVL |
| **Likelihood** | 3 (Possible) |
| **Impact** | 3 (Moderate) |
| **Risk Score** | 9 (Medium) |
| **Description** | Disk or object storage capacity is exhausted due to accumulated output files, temp directories from incomplete jobs, or uncleaned log files, causing the pipeline to fail on new document processing and potentially corrupting in-progress jobs. |
| **Existing Controls** | `cleanup_old_jobs` management command, `purge_temp_files` management command, `cleanup_output` management command, `rotate_audit_logs` management command, Prometheus disk usage monitoring, KEDA autoscaling (prevents unbounded queue growth) |
| **Residual Risk** | Cleanup commands must be scheduled via Celery Beat or cron; default retention periods are conservative; temp directory cleanup is manual |
| **Mitigation Status** | Controlled -- cleanup infrastructure exists; operator must schedule periodic execution |
| **Owner** | Operations |
| **Review Date** | 2026-06-24 |

### RISK-016: Network Partition in Distributed Deployment

| Field | Value |
|-------|-------|
| **ID** | RISK-016 |
| **Category** | AVL |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 3 (Moderate) |
| **Risk Score** | 6 (Medium) |
| **Description** | A network partition between coordinator, workers, and message broker causes split-brain conditions where workers continue processing pages but cannot report results, leading to duplicate processing, stale job states, or orphaned page results. |
| **Existing Controls** | Worker heartbeat monitoring (30s interval), stale job cleanup (coordinator), page-level crash resume (temp directory), RabbitMQ quorum queues (partition-tolerant), Redis Sentinel (automatic failover), Celery `acks_late` (re-delivery on worker failure) |
| **Residual Risk** | Brief partitions may cause duplicate page processing (idempotent but wasteful); extended partitions require manual intervention to reconcile job states |
| **Mitigation Status** | Controlled -- heartbeat and crash resume handle transient partitions; extended partitions require operator intervention per failover runbook |
| **Owner** | Operations |
| **Review Date** | 2026-06-24 |

### RISK-017: PII Encryption Key Compromise or Loss

| Field | Value |
|-------|-------|
| **ID** | RISK-017 |
| **Category** | SEC |
| **Likelihood** | 1 (Rare) |
| **Impact** | 5 (Critical) |
| **Risk Score** | 5 (Medium) |
| **Description** | The Fernet encryption key (`PII_ENCRYPTION_KEY`) used to encrypt PII entity values at rest is compromised (exposing all encrypted PII) or lost (rendering all encrypted PII unrecoverable). |
| **Existing Controls** | Fernet symmetric encryption for `PiiEntity.entity_value` (+), `rotate_pii_key` management command for key rotation, `credential_manager.py` for key storage (Vault/KMS/env backends), Kubernetes Secrets for deployment |
| **Residual Risk** | Key is stored as environment variable by default (not Vault/KMS); no automated key rotation schedule; key loss without backup means permanent PII data loss |
| **Mitigation Status** | Partially controlled -- encryption and rotation tooling exist; production deployments should use Vault or KMS backend |
| **Owner** | Security officer |
| **Review Date** | 2026-06-24 |

### RISK-018: Audit Trail Gaps from Logging Failures

| Field | Value |
|-------|-------|
| **ID** | RISK-018 |
| **Category** | CMP |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 4 (Major) |
| **Risk Score** | 8 (Medium) |
| **Description** | Gaps in the audit trail occur when custody logging fails silently (disk full, permission errors, database unavailable), creating periods where document processing actions are not recorded. This compromises forensic chain-of-custody requirements and regulatory compliance. |
| **Existing Controls** | Dual-write audit logging (filesystem JSONL and PostgreSQL `CustodyEvent`), FlushStreamHandler for reliable Docker stdout logging, `verify_custody_file` for chain integrity verification, audit logging completeness assessment document |
| **Residual Risk** | If both filesystem and database writes fail simultaneously, events are lost; no real-time alerting on custody write failures; 7 audit logging gaps identified in completeness assessment |
| **Mitigation Status** | Partially controlled -- dual-write provides redundancy; identified gaps documented in `audit-logging-completeness.md` |
| **Owner** | Compliance officer |
| **Review Date** | 2026-06-24 |

### RISK-019: Single Point of Failure in Coordinator

| Field | Value |
|-------|-------|
| **ID** | RISK-019 |
| **Category** | AVL |
| **Likelihood** | 2 (Unlikely) |
| **Impact** | 4 (Major) |
| **Risk Score** | 8 (Medium) |
| **Description** | The Django coordinator is a single point of failure in the distributed pipeline. If the coordinator process crashes or its host becomes unavailable, no new jobs can be submitted, in-progress job status cannot be updated, and worker registration/heartbeat monitoring stops. |
| **Existing Controls** | Helm chart supports configurable coordinator replicas, Kubernetes liveness/readiness probes, PodDisruptionBudget, Gunicorn multi-worker process model, Docker restart policies |
| **Residual Risk** | Default Helm deployment uses 1 coordinator replica; SQLite-backed local API is single-process; no active-active coordinator failover |
| **Mitigation Status** | Partially controlled -- horizontal scaling is supported but not default; operator must configure replicas > 1 for HA |
| **Owner** | Operations |
| **Review Date** | 2026-06-24 |

### RISK-020: Container Image Tampering or Unauthorized Modification

| Field | Value |
|-------|-------|
| **ID** | RISK-020 |
| **Category** | SEC |
| **Likelihood** | 1 (Rare) |
| **Impact** | 5 (Critical) |
| **Risk Score** | 5 (Medium) |
| **Description** | A malicious actor modifies a Docker image in the registry or during air-gapped deployment transfer, injecting backdoors or data exfiltration code into the OCR pipeline runtime. |
| **Existing Controls** | Multi-stage Docker builds (minimal attack surface), Trivy container scanning in CI, air-gapped bundle/deploy scripts with image checksums, Kubernetes image pull policies, SBOM generation for dependency transparency |
| **Residual Risk** | Docker images are not cryptographically signed (no Docker Content Trust / cosign); air-gapped transfer checksums are advisory |
| **Mitigation Status** | Partially controlled -- scanning detects known vulnerabilities; image signing is not yet implemented |
| **Owner** | Engineering lead |
| **Review Date** | 2026-06-24 |

---

## Risk Summary by Rating

| Rating | Count | Risk IDs |
|--------|-------|----------|
| Critical | 0 | -- |
| High | 3 | RISK-001, RISK-007, RISK-010 |
| Medium | 17 | RISK-002 through RISK-006, RISK-008 through RISK-009, RISK-011 through RISK-020 |
| Low | 0 | -- |

---

## Risk Acceptance Register

Risks that have been formally accepted (not mitigated) with justification:

| Risk ID | Description | Accepted By | Date | Justification |
|---------|-------------|-------------|------|---------------|
| N/A | GIL limits CPU extractor parallelism (TD-004) | Engineering | 2026-02-14 | By design; GPU workers are the bottleneck, not CPU extractors |

---

## Risk Review Cadence

| Activity | Frequency | Owner | Next Due |
|----------|-----------|-------|----------|
| Risk register review | Quarterly | Security officer | 2026-06-24 |
| New risk identification | Every sprint | Engineering lead | Ongoing |
| Risk score reassessment | After each SEV1/SEV2 incident | Incident commander | As needed |
| Full risk assessment | Annually | Compliance officer | 2027-03-24 |
| Dependency vulnerability review | Monthly | Engineering lead | Ongoing |
| PII encryption key rotation review | Quarterly | Security officer | 2026-06-24 |

### Quarterly Review Process

1. **Preparation** (1 week before): Engineering lead compiles list of new risks identified during the quarter, incident reports, and dependency vulnerability alerts.
2. **Review meeting**: Security officer, compliance officer, engineering lead, and operations representative assess each risk for changes in likelihood, impact, or mitigation status.
3. **Updates**: Risk register is updated with revised scores, new risks, and closed risks. Revision history is appended.
4. **Sign-off**: Security officer approves the updated register and distributes to stakeholders.
5. **Tracking**: Action items from the review are tracked in the project backlog with owner assignments and target dates.

---

## Appendix A: Risk Level Definitions

### Low (Score 1-4)

Risks with minimal potential impact that are unlikely to materialize. These are monitored during normal operations and addressed opportunistically during maintenance cycles. No dedicated resources are allocated for mitigation unless the risk score increases.

### Medium (Score 5-9)

Risks with moderate potential impact or reasonable likelihood of occurrence. These require planned mitigation within the current or next release cycle. Engineering and operations teams track these risks and implement controls as part of routine development work.

### High (Score 10-15)

Risks with significant potential impact that require prioritized attention. These are tracked in sprint planning, have assigned owners, and receive dedicated engineering effort. Progress on mitigation is reviewed at each sprint retrospective.

### Critical (Score 16-25)

Risks with severe potential impact that require immediate executive attention and resource allocation. These trigger the incident response plan if they materialize. Mitigation is the highest priority, and leadership receives regular status updates until the risk is reduced to High or below.

---

## Appendix B: Related Documents

| Document | Relevance |
|----------|-----------|
| [Incident Response Plan](incident-response-plan.md) | Response procedures when risks materialize |
| [Data Retention Policy](data-retention-policy.md) | Controls for RISK-007 (over-retention) |
| [Audit Logging Completeness](audit-logging-completeness.md) | Gap analysis for RISK-018 (audit trail gaps) |
| [SOC 2 Readiness](soc2-readiness.md) | CC3.1-CC3.4 risk assessment mapping |
| [HIPAA Readiness](hipaa-readiness.md) | 164.308(a)(1)(ii)(A) risk analysis mapping |
| [FedRAMP Readiness](fedramp-readiness.md) | RA-2/RA-3 risk assessment mapping |
| [Failover Runbook](../FAILOVER-RUNBOOK.md) | Operational procedures for RISK-016, RISK-019 |
| [Production Cutover Runbook](../operations/production-cutover-runbook.md) | Deployment procedures addressing multiple risks |

---

## Revision History

| Date | Version | Author | Change |
|------|---------|--------|--------|
| 2026-03-24 | 1.0 | Compliance remediation | Initial risk register with top 10 risks |
| 2026-03-24 | 2.0 | C-09 compliance task | Expanded to 20 risks; added methodology, heat map, summary, appendices, review process |

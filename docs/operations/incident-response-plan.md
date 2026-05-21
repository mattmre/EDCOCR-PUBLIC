# Incident Response Plan -- EDCOCR v1.0

## Purpose

This document defines the incident classification, escalation, communication, and post-incident review procedures for the EDCOCR forensic document processing platform. It satisfies SOC 2 CC7.3 (incident response), HIPAA 164.308(a)(6) (security incident procedures), and FedRAMP IR-1 through IR-8 control requirements.

## Scope

This plan covers incidents affecting:

- The OCR processing pipeline (local and distributed modes)
- The coordinator service (Django + Celery + PostgreSQL)
- Storage backends (NFS, S3/MinIO)
- Message brokers (RabbitMQ, Redis)
- The REST API and WebSocket endpoints
- Chain-of-custody audit integrity
- PII/PHI data exposure or loss

---

## 1. Severity Classification

### SEV1 -- Critical

**Definition**: Complete service outage, data breach involving PII/PHI, or chain-of-custody integrity compromise.

**Examples**:
- All GPU workers are offline and no documents can be processed
- Database corruption or unrecoverable data loss
- Confirmed PII/PHI exposure to unauthorized parties
- Chain-of-custody hash chain tampered or broken in production
- Ransomware or unauthorized access to production systems

**Response time**: Immediate (within 15 minutes)
**Resolution target**: 4 hours
**Notification**: Engineering lead, security officer, compliance officer, executive sponsor

### SEV2 -- High

**Definition**: Degraded processing capacity, partial data loss with recovery possible, or security vulnerability actively exploited.

**Examples**:
- 50% or more GPU workers offline
- RabbitMQ or Redis failover triggered
- PostgreSQL replication lag exceeding 60 seconds
- Elevated error rate (>10% of jobs failing)
- Authentication bypass or privilege escalation discovered
- S3 bucket permissions misconfigured (not yet confirmed exfiltration)

**Response time**: Within 30 minutes
**Resolution target**: 8 hours
**Notification**: Engineering lead, on-call engineer, security officer

### SEV3 -- Medium

**Definition**: Non-critical degradation, isolated failures, or potential security concerns under investigation.

**Examples**:
- Single GPU worker offline
- Intermittent OCR failures for specific document types
- Disk space warning (>80% utilization)
- Failed webhook deliveries
- Suspicious but unconfirmed access patterns in API audit logs
- Certificate expiration within 30 days

**Response time**: Within 2 hours
**Resolution target**: 24 hours
**Notification**: On-call engineer, team channel

### SEV4 -- Low

**Definition**: Minor issues with no user impact, informational findings, or planned maintenance follow-ups.

**Examples**:
- Non-critical log warnings
- Performance degradation below alerting threshold
- Documentation gaps discovered during audit
- Dependency vulnerability with no known exploit path
- Cosmetic issues in monitoring dashboards

**Response time**: Next business day
**Resolution target**: 5 business days
**Notification**: Team channel, tracked in issue backlog

---

## 2. Escalation Procedures

### 2.1 Initial Detection

Incidents may be detected via:

1. **Prometheus alerts** (5 rules configured: queue depth, worker heartbeat, error rate, processing latency, disk usage)
2. **Grafana dashboard** anomalies (7-panel operational dashboard + 10-panel canary dashboard)
3. **API health endpoint** (`GET /health`) returning non-200
4. **Docker healthcheck** failure (monitor heartbeat age >60s)
5. **Manual report** from operator or end user
6. **Automated chain-of-custody verification** failure
7. **Security scanning tools** (GitGuardian, Dependabot)

### 2.2 Escalation Matrix

| From | To | Trigger |
|------|----|---------|
| Monitoring alert | On-call engineer | Any SEV1-SEV3 alert fires |
| On-call engineer | Engineering lead | SEV1 confirmed, or SEV2 unresolved after 2 hours |
| Engineering lead | Security officer | Any PII/PHI exposure, auth bypass, or custody chain compromise |
| Security officer | Compliance officer | Confirmed data breach, regulatory notification required |
| Compliance officer | Legal counsel | Litigation-triggering events, regulatory fines, mandatory disclosure |
| Engineering lead | Executive sponsor | SEV1 unresolved after 4 hours, customer-impacting outage |

### 2.3 On-Call Responsibilities

The on-call engineer must:

1. Acknowledge the alert within the response time for the severity level
2. Assess severity using the classification matrix above
3. Begin a timeline log in the incident channel
4. Activate `LITIGATION_HOLD=true` if a data breach is suspected
5. Execute the appropriate runbook (see `docs/FAILOVER-RUNBOOK.md`)
6. Escalate per the matrix if resolution stalls

---

## 3. Response Procedures

### 3.1 SEV1 Data Breach Response

1. **Contain**: Immediately isolate affected systems (revoke API keys, disable ingress, stop workers)
2. **Preserve**: Set `LITIGATION_HOLD=true` to prevent automated data deletion
3. **Assess**: Determine scope of exposure using API audit logs and chain-of-custody records
4. **Notify**: Inform security officer and compliance officer within 1 hour
5. **Remediate**: Patch vulnerability, rotate credentials, rebuild affected images
6. **Report**: Prepare breach notification per HIPAA (60-day rule) or GDPR (72-hour rule) if applicable
7. **Review**: Conduct post-incident review within 5 business days

### 3.2 SEV1 Service Outage Response

1. **Assess**: Check Prometheus/Grafana for root cause indicators
2. **Triage**: Follow `docs/FAILOVER-RUNBOOK.md` for component-specific recovery
   - PostgreSQL: Promote replica, restore from backup CronJob
   - RabbitMQ: Quorum queue leader election, message replay
   - Redis: Sentinel automatic failover, verify Sentinel quorum
   - GPU workers: Scale via KEDA, check NVIDIA driver status
3. **Communicate**: Post status update to stakeholders within 30 minutes
4. **Recover**: Bring services back online, verify processing pipeline end-to-end
5. **Validate**: Run chain-of-custody verification on any documents processed during the incident window

### 3.3 SEV2/SEV3 Degradation Response

1. **Investigate**: Check component health via `fleet_status` command and Prometheus metrics
2. **Mitigate**: Apply temporary workaround (e.g., route traffic to healthy workers)
3. **Fix**: Deploy permanent fix via normal PR workflow
4. **Verify**: Confirm fix via monitoring and test document processing

---

## 4. Communication Templates

### 4.1 Internal Incident Declaration

```
INCIDENT DECLARED: [SEV1/SEV2/SEV3/SEV4]
Time: [ISO 8601 timestamp]
Component: [affected system]
Impact: [description of user/data impact]
Current status: [investigating / mitigating / resolved]
Incident commander: [name]
Next update: [time]
```

### 4.2 Stakeholder Status Update

```
INCIDENT UPDATE: [SEV level] - [short title]
Time: [ISO 8601 timestamp]
Status: [investigating / mitigating / monitoring / resolved]
Impact: [current user/data impact]
Root cause: [known / under investigation]
ETA to resolution: [estimate or "TBD"]
Actions taken: [bulleted list]
Next update: [time]
```

### 4.3 Data Breach Notification (External)

```
SECURITY NOTICE

Date of discovery: [date]
Nature of incident: [description]
Data potentially affected: [types of PII/PHI]
Number of individuals affected: [count or estimate]
Steps taken to contain: [actions]
Steps taken to prevent recurrence: [actions]
Point of contact: [name, email, phone]
```

### 4.4 Incident Resolution

```
INCIDENT RESOLVED: [SEV level] - [short title]
Time detected: [ISO 8601]
Time resolved: [ISO 8601]
Duration: [hours:minutes]
Root cause: [description]
Impact: [final assessment]
Documents affected: [count, if any]
Chain-of-custody impact: [none / verified / remediation required]
Post-incident review scheduled: [date]
```

---

## 5. Post-Incident Review Template

### Metadata

| Field | Value |
|-------|-------|
| Incident ID | INC-YYYY-NNN |
| Severity | SEV1/SEV2/SEV3/SEV4 |
| Date detected | YYYY-MM-DD HH:MM UTC |
| Date resolved | YYYY-MM-DD HH:MM UTC |
| Duration | HH:MM |
| Incident commander | [name] |
| Author | [name] |
| Review date | YYYY-MM-DD |

### Timeline

| Time (UTC) | Event |
|------------|-------|
| HH:MM | Alert fired / incident reported |
| HH:MM | On-call engineer acknowledged |
| HH:MM | Root cause identified |
| HH:MM | Mitigation applied |
| HH:MM | Service restored |
| HH:MM | Incident declared resolved |

### Root Cause Analysis

**What happened**: [description]

**Why it happened**: [technical root cause]

**Contributing factors**:
- [factor 1]
- [factor 2]

### Impact Assessment

- **Users affected**: [count / scope]
- **Documents affected**: [count]
- **Data integrity impact**: [none / compromised / unknown]
- **Chain-of-custody impact**: [none / broken chains identified / all verified intact]
- **SLA impact**: [any SLA violations]

### Action Items

| ID | Action | Owner | Due Date | Status |
|----|--------|-------|----------|--------|
| 1 | [action] | [name] | YYYY-MM-DD | Open |
| 2 | [action] | [name] | YYYY-MM-DD | Open |

### Lessons Learned

1. **What went well**: [description]
2. **What could be improved**: [description]
3. **Process changes recommended**: [description]

---

## 6. Periodic Review

This incident response plan must be reviewed and tested:

- **Quarterly**: Tabletop exercise with engineering team
- **Annually**: Full plan review and update
- **After every SEV1/SEV2 incident**: Plan updates based on post-incident review findings

---

## 7. Related Documents

- `docs/FAILOVER-RUNBOOK.md` -- Component-specific failover procedures
- `docs/compliance/data-retention-policy.md` -- Data retention and litigation hold
- `docs/compliance/soc2-readiness.md` -- SOC 2 control mapping
- `docs/compliance/hipaa-readiness.md` -- HIPAA compliance readiness
- `docs/operations/production-cutover-runbook.md` -- Production deployment guide

---

## Revision History

| Date | Version | Author | Change |
|------|---------|--------|--------|
| 2026-03-24 | 1.0 | Compliance remediation | Initial incident response plan |

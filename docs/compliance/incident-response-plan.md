# Incident Response Plan -- EDCOCR v1.0

## 1. Purpose and Scope

This document establishes the formal incident response plan for the EDCOCR forensic-grade document processing pipeline. It defines severity classifications, escalation paths, communication procedures, and post-incident review requirements for all incidents affecting the system.

### Scope

This plan covers incidents across the following categories:

- **Data breach or loss**: Unauthorized access to PII/PHI, custody chain tampering, unintended data exposure.
- **Service outage**: Full or partial pipeline unavailability, coordinator failure, worker pool exhaustion.
- **Infrastructure failure**: PostgreSQL, RabbitMQ, Redis, S3/MinIO, or Kubernetes cluster degradation.
- **Security events**: API key compromise, unauthorized access attempts, container escape, supply chain attacks.
- **Data integrity failures**: OCR output corruption, audit log chain breaks, checksum verification failures.

### Out of Scope

- Planned maintenance windows (covered by change management procedures).
- Individual document processing failures handled by the pipeline's built-in retry and fallback mechanisms.
- Development environment issues.

---

## 2. Severity Levels

All incidents are classified into one of four severity levels. Classification determines response urgency, escalation requirements, and communication obligations.

| Level | Name | Description | Response Time | Update Cadence | Examples |
|-------|------|-------------|---------------|----------------|----------|
| P0 | Critical | Data breach, data loss, or complete compromise of forensic integrity | 15 minutes | Every 30 minutes | PII/PHI exfiltration, custody chain tampered, encryption key compromise, unauthorized bulk data access |
| P1 | Major | Full service outage or severe degradation affecting all users | 30 minutes | Every 1 hour | All workers offline, coordinator unreachable, database corruption, S3 storage inaccessible |
| P2 | Degraded | Partial service degradation affecting some users or capabilities | 2 hours | Every 4 hours | Single worker pool failure, elevated error rates (>5%), queue backlog exceeding SLA, single-region outage |
| P3 | Minor | Limited impact, workaround available, no data integrity risk | 8 hours (next business day) | Daily | Intermittent API timeouts, non-critical monitoring gaps, single document class failures, slow processing |

### Severity Escalation Criteria

An incident's severity level must be escalated when:

- A P2 incident is not resolved within 4 hours.
- A P3 incident is not resolved within 24 hours.
- New evidence reveals broader impact than initially assessed.
- The incident involves any PII/PHI exposure, regardless of initial classification.

---

## 3. Incident Response Roles

Each active incident must have the following roles assigned. A single person may hold multiple roles for P2/P3 incidents. P0 and P1 incidents require distinct individuals for Incident Commander and Technical Lead.

| Role | Responsibilities | Required For |
|------|------------------|--------------|
| **Incident Commander (IC)** | Overall incident ownership, decision authority, escalation, timeline management | All severity levels |
| **Technical Lead (TL)** | Root cause investigation, remediation implementation, system recovery | All severity levels |
| **Communications Lead (CL)** | Internal/external notifications, status page updates, stakeholder communication | P0, P1 (optional P2/P3) |
| **Scribe** | Real-time incident timeline documentation, action item tracking | P0, P1 (optional P2/P3) |

### Role Assignment

1. The first responder to an alert or report assumes the Incident Commander role until formally handed off.
2. The IC assigns remaining roles within the first 15 minutes for P0/P1 or within the first hour for P2/P3.
3. Role assignments are recorded in the incident channel or tracking system with timestamps.
4. Handoffs between shifts must include a verbal briefing and written summary in the incident timeline.

---

## 4. Detection

Incidents are detected through three primary channels. Each channel has defined owners and expected detection latency.

### 4.1 Automated Monitoring Alerts

The pipeline's Prometheus-based monitoring generates alerts for the following conditions. Alert rules are defined in the Helm chart's `PrometheusRule` template and the Grafana dashboard.

| Alert | Condition | Severity Mapping |
|-------|-----------|------------------|
| `OCRWorkerPoolExhausted` | 0 online workers for >5 minutes | P1 |
| `OCRQueueBacklogCritical` | Queue depth >1000 for >10 minutes | P2 |
| `OCRJobFailureRateHigh` | Failure rate >10% over 15-minute window | P2 |
| `OCRCoordinatorUnreachable` | Health endpoint fails for >2 minutes | P1 |
| `OCRDatabaseConnectionFailure` | PostgreSQL connection pool exhausted | P1 |
| `OCRRedisDown` | Redis Sentinel reports no reachable primary | P1 |
| `OCRRabbitMQQuorumLost` | RabbitMQ quorum queue loses majority | P1 |
| `OCRS3Unreachable` | S3/MinIO health check fails for >5 minutes | P2 |
| `OCRSLABreachImminent` | Tenant SLA threshold >80% consumed | P2 |
| `OCRCustodyChainBroken` | Hash chain verification failure detected | P0 |
| `OCRPIIExposureDetected` | Unencrypted PII detected in output path | P0 |

### 4.2 User Reports

- API consumers report issues via the support channel or by contacting the on-call engineer.
- Reports are triaged within the response time for the assessed severity level.
- All user reports are logged in the incident tracking system regardless of severity.

### 4.3 Automated Health Checks

- **Docker healthcheck**: `healthcheck.sh` monitors the pipeline heartbeat file every 30 seconds. Three consecutive failures trigger container restart.
- **Kubernetes liveness/readiness probes**: Configured in Helm chart deployments with 10-second intervals.
- **Celery Beat periodic tasks**: `cleanup_old_jobs`, `purge_temp_files`, and stale job detection run on schedule and log anomalies.
- **Chain-of-custody verification**: Periodic `verify_custody_file` runs detect tampering or corruption.

---

## 5. Response Procedures

### 5.1 P0: Critical (Data Breach / Data Loss)

**Objective**: Contain the breach, preserve evidence, notify affected parties.

| Step | Action | Owner | Timeframe |
|------|--------|-------|-----------|
| 1 | Acknowledge alert, declare P0 incident, open incident channel | IC | 0-15 min |
| 2 | Isolate affected systems (revoke compromised API keys, block network access) | TL | 15-30 min |
| 3 | Preserve forensic evidence (snapshot logs, custody chains, database state) | TL | 15-30 min |
| 4 | Activate litigation hold (`LITIGATION_HOLD=true`) to prevent evidence destruction | TL | 15-30 min |
| 5 | Send internal notification (see Communication Templates below) | CL | Within 30 min |
| 6 | Assess scope: which data, how many records, which tenants affected | TL | 30-60 min |
| 7 | Begin external notification process per regulatory requirements | CL | Per regulation |
| 8 | Implement containment measures (rotate keys, patch vulnerability) | TL | As available |
| 9 | Verify containment (confirm no ongoing exfiltration or access) | TL | After step 8 |
| 10 | Send status update to stakeholders | CL | Every 30 min |
| 11 | Begin recovery: restore from backup if needed, re-enable services | TL | After containment verified |
| 12 | Schedule post-incident review within 48 hours | IC | After resolution |

**Regulatory notification deadlines**:

| Framework | Notification Deadline | Authority |
|-----------|----------------------|-----------|
| HIPAA | 60 days from discovery | HHS OCR (>500 individuals) |
| GDPR | 72 hours from awareness | Supervisory authority |
| SOC 2 | Per client SLA | Affected clients |
| FedRAMP | 1 hour (US-CERT) | US-CERT / agency AO |

### 5.2 P1: Major (Full Outage)

**Objective**: Restore service availability as quickly as possible.

| Step | Action | Owner | Timeframe |
|------|--------|-------|-----------|
| 1 | Acknowledge alert, declare P1 incident, open incident channel | IC | 0-30 min |
| 2 | Identify failing component(s) using monitoring dashboards | TL | 0-30 min |
| 3 | Execute relevant failover runbook procedure (see Related Documents) | TL | 30-60 min |
| 4 | Send internal notification | CL | Within 30 min |
| 5 | If failover resolves: verify full pipeline health end-to-end | TL | After failover |
| 6 | If failover does not resolve: escalate to P0 if data integrity at risk | IC | After 2 hours |
| 7 | Send status updates to stakeholders | CL | Every 1 hour |
| 8 | Verify queue drain and job completion after recovery | TL | After resolution |
| 9 | Schedule post-incident review within 5 business days | IC | After resolution |

### 5.3 P2: Degraded

**Objective**: Identify root cause, restore full capacity, prevent recurrence.

| Step | Action | Owner | Timeframe |
|------|--------|-------|-----------|
| 1 | Acknowledge alert, open incident tracking ticket | IC | 0-2 hours |
| 2 | Assess impact: which tenants, what throughput degradation, SLA risk | TL | 0-2 hours |
| 3 | Apply immediate mitigation (scale workers, restart unhealthy pods, reroute traffic) | TL | 2-4 hours |
| 4 | Send internal notification if impact exceeds 1 hour | CL | As needed |
| 5 | Investigate root cause | TL | 4-8 hours |
| 6 | Implement permanent fix or schedule follow-up work | TL | Next business day |
| 7 | Close incident ticket with summary | IC | After resolution |
| 8 | Post-incident review for incidents lasting >4 hours | IC | Within 10 business days |

### 5.4 P3: Minor

**Objective**: Document, investigate, and resolve within normal workflow.

| Step | Action | Owner | Timeframe |
|------|--------|-------|-----------|
| 1 | Create incident tracking ticket | IC | Next business day |
| 2 | Investigate root cause during normal work hours | TL | 1-5 business days |
| 3 | Implement fix and deploy | TL | Per sprint schedule |
| 4 | Close incident ticket with summary | IC | After resolution |

---

## 6. Escalation Paths

### 6.1 Escalation Matrix

| Severity | Initial Responder | 30-min Escalation | 2-hour Escalation | 4-hour Escalation |
|----------|-------------------|-------------------|--------------------|--------------------|
| P0 | On-call engineer | Engineering Manager + Security Lead | VP Engineering + Legal | Executive team + external counsel |
| P1 | On-call engineer | Engineering Manager | VP Engineering | CTO |
| P2 | On-call engineer | Team Lead | Engineering Manager | -- |
| P3 | Assigned engineer | Team Lead | -- | -- |

### 6.2 Contact Methods

All contacts are reached through the following channels in order of priority:

1. **Incident channel** (Slack/Teams) -- primary communication during active incidents.
2. **PagerDuty / on-call rotation** -- for P0/P1 after-hours escalation.
3. **Phone** -- for P0 escalation when digital channels are unresponsive.
4. **Email** -- for non-urgent updates and post-incident coordination.

### 6.3 External Escalation

| Scenario | External Contact | Trigger |
|----------|-----------------|---------|
| Data breach involving PII/PHI | Legal counsel, regulatory authority | Any confirmed PII/PHI exposure |
| Infrastructure provider failure | Cloud provider support (priority ticket) | Provider-side outage lasting >30 min |
| Supply chain compromise | Security vendor, CISA | Compromised dependency detected |
| Law enforcement request | Legal counsel | Any law enforcement data request |

---

## 7. Communication Templates

### 7.1 Internal Incident Notification

Use this template when notifying the engineering and leadership team of an active incident.

```
Subject: [P{SEVERITY}] EDCOCR Incident -- {SHORT_DESCRIPTION}

Incident ID: INC-{YYYY}-{NNN}
Severity: P{SEVERITY} ({LEVEL_NAME})
Status: Investigating | Identified | Mitigating | Resolved
Declared: {YYYY-MM-DD HH:MM UTC}

Summary:
{1-2 sentence description of what is happening and its impact}

Impact:
- Affected components: {list of affected services/components}
- Affected tenants: {all / specific tenant IDs / none confirmed}
- Data integrity: {confirmed intact / under investigation / compromised}

Current Actions:
- {action 1 -- owner -- status}
- {action 2 -- owner -- status}

Incident Commander: {name}
Technical Lead: {name}
Next Update: {YYYY-MM-DD HH:MM UTC}
```

### 7.2 External Stakeholder Notification

Use this template when notifying external stakeholders (API consumers, clients) of a service-affecting incident.

```
Subject: EDCOCR Service Notification -- {SHORT_DESCRIPTION}

Date: {YYYY-MM-DD}
Status: Investigating | Identified | Mitigating | Resolved

We are aware of an issue affecting {description of affected functionality}.

Impact:
{Plain-language description of what users may experience}

What we are doing:
{Plain-language description of remediation efforts}

Estimated resolution: {time estimate or "under investigation"}

We will provide updates every {cadence}. For questions, contact {support channel}.

Reference: INC-{YYYY}-{NNN}
```

### 7.3 Data Breach Notification (P0 Only)

Use this template when a confirmed data breach requires notification to affected parties. Legal counsel must review before sending.

```
Subject: Important Notice Regarding Your Data -- EDCOCR Security Incident

Date: {YYYY-MM-DD}

We are writing to inform you of a security incident that may have involved
your data processed by EDCOCR.

What happened:
{Factual description of the incident, dates, and discovery}

What data was involved:
{Types of data potentially affected -- PII, PHI, document content}

What we are doing:
{Remediation steps taken and planned}

What you can do:
{Recommended protective actions for affected parties}

For questions:
{Dedicated contact information for incident inquiries}

Reference: INC-{YYYY}-{NNN}
```

### 7.4 Incident Status Update

Use this template for periodic updates during an active incident.

```
Subject: [UPDATE {N}] [P{SEVERITY}] EDCOCR Incident -- {SHORT_DESCRIPTION}

Incident ID: INC-{YYYY}-{NNN}
Status: Investigating | Identified | Mitigating | Resolved
Updated: {YYYY-MM-DD HH:MM UTC}

Summary of changes since last update:
- {change 1}
- {change 2}

Current status:
{Brief description of where things stand}

Next steps:
- {planned action 1 -- owner -- ETA}
- {planned action 2 -- owner -- ETA}

Next update: {YYYY-MM-DD HH:MM UTC}
```

### 7.5 Incident Resolution Notification

Use this template when closing an incident.

```
Subject: [RESOLVED] [P{SEVERITY}] EDCOCR Incident -- {SHORT_DESCRIPTION}

Incident ID: INC-{YYYY}-{NNN}
Status: Resolved
Resolved: {YYYY-MM-DD HH:MM UTC}
Duration: {total duration}

Resolution summary:
{What was the root cause and how it was resolved}

Impact summary:
- Duration of impact: {time}
- Affected tenants: {count or list}
- Data integrity: {confirmed intact / remediation performed}
- Jobs affected: {count, if applicable}

Follow-up:
- Post-incident review scheduled: {date}
- Action items: {count} identified, tracked in {system}
```

---

## 8. Post-Incident Review

A post-incident review (PIR) is required for all P0 and P1 incidents and for any P2 incident lasting longer than 4 hours. The review must be completed within the timeframe specified in the response procedures.

### 8.1 Review Process

1. **Schedule**: The Incident Commander schedules the PIR and invites all incident participants plus relevant stakeholders.
2. **Prepare**: The Scribe (or IC if no Scribe was assigned) prepares the PIR document using the template below, pre-populating the timeline from incident notes.
3. **Conduct**: The review meeting walks through the timeline, identifies root causes, and develops action items. The meeting is blameless -- it focuses on systemic improvements, not individual fault.
4. **Publish**: The completed PIR is published to the team within 2 business days of the review meeting.
5. **Track**: All action items are entered into the project tracking system with owners and due dates.
6. **Verify**: Action items are reviewed in the next team retrospective to confirm completion.

### 8.2 Post-Incident Review Template

```
# Post-Incident Review: INC-{YYYY}-{NNN}

## Metadata

| Field | Value |
|-------|-------|
| Incident ID | INC-{YYYY}-{NNN} |
| Severity | P{N} |
| Date | {YYYY-MM-DD} |
| Duration | {total duration} |
| Incident Commander | {name} |
| Technical Lead | {name} |
| Review Date | {YYYY-MM-DD} |
| Review Author | {name} |

## Executive Summary

{2-3 sentence summary of what happened, impact, and resolution}

## Timeline

| Time (UTC) | Event |
|------------|-------|
| HH:MM | {First detection or report} |
| HH:MM | {Incident declared, IC assigned} |
| HH:MM | {Key investigation milestone} |
| HH:MM | {Mitigation applied} |
| HH:MM | {Service restored} |
| HH:MM | {Incident resolved} |

## Root Cause Analysis

### Proximate Cause
{What directly caused the incident}

### Contributing Factors
- {Factor 1: description}
- {Factor 2: description}

### Root Cause
{The underlying systemic issue that allowed the incident to occur}

## Impact Assessment

| Metric | Value |
|--------|-------|
| Total downtime / degradation | {duration} |
| Jobs affected | {count} |
| Tenants affected | {count or "all"} |
| Data integrity impact | {none / description} |
| SLA breaches | {count and details} |
| Financial impact | {estimate if applicable} |

## What Went Well

- {Positive aspect 1}
- {Positive aspect 2}

## What Could Be Improved

- {Improvement area 1}
- {Improvement area 2}

## Action Items

| ID | Action | Owner | Priority | Due Date | Status |
|----|--------|-------|----------|----------|--------|
| AI-1 | {action description} | {name} | {P0-P3} | {date} | Open |
| AI-2 | {action description} | {name} | {P0-P3} | {date} | Open |

## Monitoring Gaps

{Were there alerts that should have fired but did not? Were there metrics
that would have provided earlier detection?}

## Lessons Learned

{Key takeaways that should inform future incident handling, system design,
or operational procedures}
```

---

## 9. Incident Record Keeping

All incident records are subject to the data retention policy defined in `data-retention-policy.md`.

- **Incident tracking records**: Retained for 7 years (aligned with audit log retention).
- **Communication records**: Retained with the incident record.
- **Post-incident review documents**: Retained permanently as institutional knowledge.
- **Forensic evidence collected during incidents**: Retained under litigation hold until explicitly released.

Incident records must be stored in a system that provides:

- Immutable audit trail of changes to incident records.
- Access controls limiting incident details to authorized personnel.
- Search capability by date, severity, component, and tenant.

---

## 10. Plan Maintenance

This incident response plan is reviewed and updated:

- **Quarterly**: General review for accuracy and completeness.
- **After every P0/P1 incident**: To incorporate lessons learned.
- **After significant architecture changes**: To reflect new components or procedures.
- **Annually**: Full tabletop exercise to validate procedures.

### Tabletop Exercise Requirements

At least one tabletop exercise per year must cover:

1. A P0 data breach scenario involving PII/PHI exposure.
2. A P1 full outage scenario requiring failover procedures.
3. Communication procedures including external notification.
4. Handoff between on-call shifts during an active incident.

---

## 11. Related Documents

| Document | Location | Relevance |
|----------|----------|-----------|
| Failover Runbook | [docs/FAILOVER-RUNBOOK.md](../FAILOVER-RUNBOOK.md) | Step-by-step failover procedures for PostgreSQL, RabbitMQ, Redis, workers |
| Data Retention Policy | [docs/compliance/data-retention-policy.md](data-retention-policy.md) | Retention periods, deletion procedures, litigation hold |
| Production Cutover Runbook | [docs/operations/production-cutover-runbook.md](../operations/production-cutover-runbook.md) | Production deployment procedures |
| SOC 2 Readiness | [docs/compliance/soc2-readiness.md](soc2-readiness.md) | SOC 2 control mapping and gap analysis |
| HIPAA Readiness | [docs/compliance/hipaa-readiness.md](hipaa-readiness.md) | HIPAA Security Rule compliance assessment |
| FedRAMP Readiness | [docs/compliance/fedramp-readiness.md](fedramp-readiness.md) | FedRAMP Moderate control assessment |
| Audit Logging Assessment | [docs/compliance/audit-logging-completeness.md](audit-logging-completeness.md) | Audit logging gap analysis |
| API Reference | [docs/API-REFERENCE.md](../API-REFERENCE.md) | REST API endpoint documentation |

---

## Revision History

| Date | Version | Author | Change |
|------|---------|--------|--------|
| 2026-03-24 | 1.0 | Compliance remediation | Initial incident response plan |

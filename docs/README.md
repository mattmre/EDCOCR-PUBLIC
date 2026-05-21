# EDCOCR — Documentation Suite

Welcome to the documentation for **EDCOCR**, the forensic-grade OCR platform engineered for electronic discovery, regulated industries, and any workflow where every page must be accounted for.

**Version**: 4.1.0  ·  **License**: Apache 2.0  ·  **Python**: 3.10+  ·  **GPU/CPU/K8s**

---

## Start Here

If you are new to EDCOCR, follow this reading order.

| Order | Document | Audience | Reading Time |
|---|---|---|---|
| 1 | [EXECUTIVE-SUMMARY.md](EXECUTIVE-SUMMARY.md) | Decision-makers | 5 min |
| 2 | [00-SYSTEM-BLUEPRINT.md](00-SYSTEM-BLUEPRINT.md) | Architects | 15 min |
| 3 | [01-TECH-STACK-DNA.md](01-TECH-STACK-DNA.md) | Engineers | 15 min |
| 4 | [02-QUICKSTART-5-MINUTE-SUCCESS.md](02-QUICKSTART-5-MINUTE-SUCCESS.md) | First-time users | 5 min |
| 5 | [03-INFORMATION-FLOWS.md](03-INFORMATION-FLOWS.md) | Integrators | 15 min |
| 6 | [04-USE-CASES.md](04-USE-CASES.md) | Product / legal | 10 min |
| 7 | [05-INTERACTIVE-WALKTHROUGH.md](05-INTERACTIVE-WALKTHROUGH.md) | Hands-on operators | 20 min |
| 8 | [06-CONFIGURATION-REFERENCE.md](06-CONFIGURATION-REFERENCE.md) | Configurers | reference |
| 9 | [07-TRANSFORMS-STAMPING.md](07-TRANSFORMS-STAMPING.md) | eDiscovery teams | 15 min |
| 10 | [08-SDK-REFERENCE.md](08-SDK-REFERENCE.md) | Application developers | reference |
| 11 | [09-TROUBLESHOOTING.md](09-TROUBLESHOOTING.md) | Operators | reference |
| 12 | [10-MONITORING-OPERATIONS.md](10-MONITORING-OPERATIONS.md) | SRE / DevOps | 20 min |
| 13 | [11-ML-TRAINING-GUIDE.md](11-ML-TRAINING-GUIDE.md) | ML engineers | 30 min |

For the API surface, see [API-REFERENCE.md](API-REFERENCE.md) and the canonical [openapi.json](openapi.json).

For the overall narrative and design rationale, see [WHITE-PAPER.md](WHITE-PAPER.md).

For choosing how to deploy, see [DEPLOYMENT-DECISION-GUIDE.md](DEPLOYMENT-DECISION-GUIDE.md).

---

## Top-Level Repository Docs

These live in the repository root and frame the project.

| File | Purpose |
|---|---|
| [`../README.md`](../README.md) | Repository landing page |
| [`../INSTALL.md`](../INSTALL.md) | Local install, Docker, Kubernetes paths |
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | System architecture with diagrams |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Contribution workflow |
| [`../DEVELOPMENT.md`](../DEVELOPMENT.md) | Local dev environment |
| [`../SECURITY.md`](../SECURITY.md) | Security policy and reporting |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Versioned release notes |
| [`../LICENSE`](../LICENSE) | Apache 2.0 license |

---

## Architecture Deep Dives

Located in [`architecture/`](architecture/).

| File | Topic |
|---|---|
| [pipeline-design.md](architecture/pipeline-design.md) | Producer-consumer pipeline internals |
| [data-flow.md](architecture/data-flow.md) | End-to-end data flow |
| [thread-model.md](architecture/thread-model.md) | Concurrency model and synchronization |
| [forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md) | Forensic-core vs AI-adjacent boundary |
| [transform-stamping-support-design.md](architecture/transform-stamping-support-design.md) | Transform + Bates stamping design |
| [adr-fasttext-assessment.md](architecture/adr-fasttext-assessment.md) | FastText language detection assessment |
| [adr-paddlepaddle-upgrade-path.md](architecture/adr-paddlepaddle-upgrade-path.md) | PaddlePaddle upgrade ADR |
| [adr-sqlite-to-postgresql-migration.md](architecture/adr-sqlite-to-postgresql-migration.md) | SQLite → PostgreSQL ADR |

---

## Operations & Runbooks

Located in [`operations/`](operations/) and at the docs root.

| File | Purpose |
|---|---|
| [FAILOVER-RUNBOOK.md](FAILOVER-RUNBOOK.md) | Component failover and recovery |
| [operations/production-cutover-runbook.md](operations/production-cutover-runbook.md) | Production deployment step-by-step |
| [operations/incident-response-plan.md](operations/incident-response-plan.md) | Incident response procedures |
| [operations/redis-sentinel-drill-guide.md](operations/redis-sentinel-drill-guide.md) | Redis Sentinel failover drill |
| [operations/pg-backup-validation-guide.md](operations/pg-backup-validation-guide.md) | PostgreSQL backup validation |
| [operations/opentelemetry-production-config.md](operations/opentelemetry-production-config.md) | OpenTelemetry production setup |
| [operations/event-bus-production-guide.md](operations/event-bus-production-guide.md) | Event bus operational guide |
| [operations/pipeline-optimization-tuning-guide.md](operations/pipeline-optimization-tuning-guide.md) | Performance tuning |
| [operations/vault-kms-hardening.md](operations/vault-kms-hardening.md) | Vault / KMS hardening |
| [operations/billing-sla-policy.md](operations/billing-sla-policy.md) | Billing and SLA policy |
| [operations/cloud-native-validation-guide.md](operations/cloud-native-validation-guide.md) | Cloud-native validation |
| [operations/deployment-topology-matrix.md](operations/deployment-topology-matrix.md) | Topology decision matrix |
| [operations/layoutlm-finetuning-guide.md](operations/layoutlm-finetuning-guide.md) | LayoutLMv3 fine-tuning |
| [operations/llm-rag-integration-guide.md](operations/llm-rag-integration-guide.md) | LLM/RAG integration |
| [operations/terraform-deployment-guide.md](operations/terraform-deployment-guide.md) | Terraform deployment |
| [operations/terraform-gke-guide.md](operations/terraform-gke-guide.md) | Terraform on GKE |
| [operations/terraform-validation-guide.md](operations/terraform-validation-guide.md) | Terraform validation |

---

## Deployment

Located in [`deployment/`](deployment/).

| File | Purpose |
|---|---|
| [deployment/docker-guide.md](deployment/docker-guide.md) | Docker / Compose deployment |
| [deployment/distributed-readiness-checklist.md](deployment/distributed-readiness-checklist.md) | Distributed deployment checklist |
| [deployment/s3-rollout-runbook.md](deployment/s3-rollout-runbook.md) | S3 storage rollout |
| [deployment/s3-observability-runbook.md](deployment/s3-observability-runbook.md) | S3 observability runbook |

---

## Compliance & Forensics

Located in [`compliance/`](compliance/) and [`forensic/`](forensic/).

| File | Purpose |
|---|---|
| [compliance/README.md](compliance/README.md) | Compliance index |
| [compliance/soc2-readiness.md](compliance/soc2-readiness.md) | SOC 2 readiness |
| [compliance/hipaa-readiness.md](compliance/hipaa-readiness.md) | HIPAA readiness |
| [compliance/fedramp-readiness.md](compliance/fedramp-readiness.md) | FedRAMP readiness |
| [compliance/audit-logging-completeness.md](compliance/audit-logging-completeness.md) | Audit logging requirements |
| [compliance/data-retention-policy.md](compliance/data-retention-policy.md) | Data retention policy |
| [compliance/incident-response-plan.md](compliance/incident-response-plan.md) | Incident response (compliance lens) |
| [compliance/risk-register.md](compliance/risk-register.md) | Risk register |
| [forensic/evidence-bundle-specification.md](forensic/evidence-bundle-specification.md) | Evidence bundle spec |
| [forensic/llm-mt-admissibility-playbook.md](forensic/llm-mt-admissibility-playbook.md) | LLM/MT admissibility playbook |

---

## Testing & Validation

| File | Purpose |
|---|---|
| [testing/playwright-testing-roadmap.md](testing/playwright-testing-roadmap.md) | Playwright test plan |
| [testing/playwright-ci-promotion-guide.md](testing/playwright-ci-promotion-guide.md) | Playwright CI promotion |
| [testing/playwright-page-coverage-matrix.md](testing/playwright-page-coverage-matrix.md) | Page coverage matrix |
| [testing/playwright-run-template.md](testing/playwright-run-template.md) | Playwright run template |
| [validation/cjk-vertical-text-validation.md](validation/cjk-vertical-text-validation.md) | CJK vertical text validation |
| [validation/language-support-validation.md](validation/language-support-validation.md) | Language support validation |
| [validation/signature-verification-validation.md](validation/signature-verification-validation.md) | Signature verification validation |
| [production-validation.md](production-validation.md) | Production validation procedures |

---

## Reference

| File | Purpose |
|---|---|
| [WHITE-PAPER.md](WHITE-PAPER.md) | Technical white paper: motivation, design, posture |
| [DEPLOYMENT-DECISION-GUIDE.md](DEPLOYMENT-DECISION-GUIDE.md) | Topology decision tree (T1–T7) |
| [API-REFERENCE.md](API-REFERENCE.md) | REST API surface reference |
| [openapi.json](openapi.json) | Machine-readable OpenAPI 3.x spec |
| [api-stability-contract.md](api-stability-contract.md) | API stability guarantees |
| [SDK-VERSIONING-POLICY.md](SDK-VERSIONING-POLICY.md) | SDK versioning policy |
| [benchmarking-methodology.md](benchmarking-methodology.md) | Benchmark methodology |
| [cpu-vs-gpu-analysis.md](cpu-vs-gpu-analysis.md) | CPU vs GPU deployment guide |
| [hardware-tuning-profiles.md](hardware-tuning-profiles.md) | Hardware tuning profiles |
| [known-issues.md](known-issues.md) | Known issues |
| [limitations/format-support.md](limitations/format-support.md) | Format support limitations |
| [limitations/international-ocr.md](limitations/international-ocr.md) | International OCR limitations |
| [security-audit-checklist.md](security-audit-checklist.md) | Security audit checklist |

---

## Presentation Materials

For meetings, demos, and stakeholder briefings, see [`../presentation/`](../presentation/).

**Decision-maker briefings (one-page narratives):**

| File | Audience | Reading Time |
|---|---|---|
| [`presentation/executive-summary.html`](../presentation/executive-summary.html) | Legal, compliance, ops leadership | ~5 min |
| [`presentation/technical-brief.html`](../presentation/technical-brief.html) | Engineers, SRE, integrators | ~15 min |
| [`presentation/white-paper.html`](../presentation/white-paper.html) | Architects, evaluators, counsel | ~25 min |
| [`presentation/use-cases.html`](../presentation/use-cases.html) | Product, legal, sales engineering | ~10 min |

**Interactive decks:**

| File | Purpose |
|---|---|
| [`presentation/index.html`](../presentation/index.html) | Marketing landing page |
| [`presentation/slides.html`](../presentation/slides.html) | Slide deck (keyboard nav) |
| [`presentation/architecture.html`](../presentation/architecture.html) | Architecture deep-dive deck |

All seven are self-contained HTML files using a CDN-loaded Mermaid renderer. No build step.

---

## Conventions

- **Forensic-core vs AI-adjacent**: The system enforces a contract between deterministic forensic operations and AI-adjacent sidecars. See [architecture/forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md).
- **CTC-only OCR**: PaddleOCR with CTC decoding is the primary engine. There is no generative-LLM substitution path in the core OCR loop.
- **Image-only fallback**: When OCR fails, the original page image is preserved in the output. Evidence is never discarded.
- **Chain of custody**: Every job emits a hash-chained JSONL audit log (SHA-256). See [forensic/evidence-bundle-specification.md](forensic/evidence-bundle-specification.md).

---

## Feedback

Found a documentation gap? Open an issue on [GitHub](https://github.com/mattmre/EDCOCR-PUBLIC/issues) or submit a PR following [CONTRIBUTING.md](../CONTRIBUTING.md).

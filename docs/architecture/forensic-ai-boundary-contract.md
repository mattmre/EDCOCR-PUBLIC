# Forensic-Core vs AI-Adjacent Boundary Contract

**Date**: 2026-03-27
**Status**: Active contract for Track 4 boundary hardening
**Purpose**: Define which EDCOCR capabilities are part of the forensic-grade core and which are optional AI-adjacent extensions.

---

## 1. Contract Summary

EDCOCR's default product promise is a **forensic-grade OCR pipeline**:

- deterministic OCR extraction
- exhaustive fallback handling
- custody and audit evidence
- validation and failure reporting
- service-oriented outputs for downstream consumers

Advanced document intelligence, semantic extraction, and model-driven enrichment remain **AI-adjacent**. They may be valuable, but they must not silently redefine the default forensic-core processing path.

### Boundary Definition

The boundary exists so EDCOCR can keep one explicit contract for deterministic
OCR evidence while still allowing optional analyst-assist and semantic workflows.

---

## 2. Forensic-Core Scope

The forensic-core path is the default path a caller gets when they submit a job without enabling optional intelligence flags.

### In Scope

- OCR execution and fallback behavior
- chain-of-custody logging
- API audit logging
- processing validation and failure handling
- transform and stamping operations with custody diagnostics
- result artifact delivery for OCR outputs
- operator-safe deployment and rollback behavior

### Core Guarantees

- no generative replacement of source text
- no requirement for optional NLP or LayoutLMv3 workers
- default-safe behavior for ordinary job submission
- evidence surfaces remain available even when AI-adjacent features are disabled

---

## 3. AI-Adjacent Scope

AI-adjacent capabilities add semantic understanding, classification, or native-text bypass behavior on top of the forensic core.

### In Scope

- DocIntel / PP-Structure-derived structure extraction
- native-text bypass via `skip_ocr`
- NER sidecars
- classification sidecars
- structured extraction sidecars
- specialist routing
- handwriting detection and signature verification advisory signals
- LayoutLMv3 token classification and domain-specific fine-tuning
- semantic search / VLM-backed analyst-assist endpoints

### Required Handling Rules

- these capabilities are opt-in, not default-on
- they must degrade safely when disabled or unavailable
- they must emit additive sidecars or metadata rather than redefining custody guarantees
- they should run on isolated queues or workers when the distributed stack is used

---

## 4. Current Enforcement Points

### Request Defaults

- `api/models.py` keeps `enable_docintel=False`
- `api/models.py` keeps `skip_ocr=False`
- `api/deps.py` describes `skip_ocr` as an NLP/DocIntel-only path, not the normal OCR path

### Runtime Defaults

- `docs/06-CONFIGURATION-REFERENCE.md` keeps `ENABLE_VALIDATION=true`
- `docs/06-CONFIGURATION-REFERENCE.md` keeps `API_AUDIT_LOG_ENABLED=true`
- AI-adjacent feature toggles such as `ENABLE_CLASSIFICATION`, `ENABLE_EXTRACTION`, and `ENABLE_SPECIALIST_ROUTING` remain default-off
- `coordinator/jobs/layoutlm_config.py` keeps `ENABLE_LAYOUTLM=false`

### Queue Isolation

- `jobs.tasks.extract_structured_data` routes to `nlp_general`
- `jobs.tasks_layoutlm.run_layoutlm_extraction` routes to `ocr_layoutlm`
- `jobs.tasks.process_document` and `jobs.tasks.process_page` remain on OCR queues
- `jobs.tasks.process_text_only` stays off the OCR queue because it is a bypass path

### Operator / Consumer Visibility

- `docs/06-CONFIGURATION-REFERENCE.md` distinguishes forensic-core defaults from AI-adjacent toggles
- `docs/API-REFERENCE.md` marks `enable_docintel` and `skip_ocr` as AI-adjacent request controls
- `docs/07-TRANSFORMS-STAMPING.md` already keeps support-agent document operations out of full platform scope

### Current Capability Map

| Capability | Current Classification | Current Guardrail |
|---|---|---|
| OCR, fallback, PDF/TXT export, custody, validation | Forensic-core | Default pipeline contract |
| Transforms and stamping | Forensic-core | Explicit feature gates with custody diagnostics |
| DocIntel / PP-Structure sidecars | AI-adjacent | `enable_docintel` request flag |
| NER, extraction, classification, specialist routing | AI-adjacent | Default-off `ENABLE_*` toggles |
| LayoutLMv3 semantic extraction | AI-adjacent | `ENABLE_LAYOUTLM`, dedicated `ocr_layoutlm` queue |
| Semantic search / VLM endpoints | AI-adjacent | `VLM_ENABLED` feature gate |
| Signature verification | AI-adjacent advisory signal | Validation docs keep it advisory-only |

### Current Implementation Anchors

- [`api/job_manager.py`](../../api/job_manager.py)
- [`coordinator/coordinator/celery.py`](../../coordinator/coordinator/celery.py)
- [`api/routers/semantic.py`](../../api/routers/semantic.py)
- [`coordinator/jobs/tasks.py`](../../coordinator/jobs/tasks.py)
- [`coordinator/jobs/tasks_layoutlm.py`](../../coordinator/jobs/tasks_layoutlm.py)

---

## 5. Output Contract

### Contract Rules

AI-adjacent features may add:

- `.structure.json`
- `.entities.json`
- `.extraction.json`
- routing or classification sidecars

They must not:

- remove the standard OCR text/PDF outputs
- weaken custody or audit evidence by default
- convert the product promise from OCR pipeline to autonomous review system

---

## 6. Change Checklist

Any future work that touches AI-adjacent features must answer all of the following before merge:

1. Does the change preserve the default forensic-core path with no extra flags?
2. Does the change keep AI-adjacent behavior opt-in?
3. Does the change keep additive outputs separate from core OCR artifacts?
4. Does the change preserve queue or worker isolation where applicable?
5. Does the change update operator or API docs if the boundary moved?
6. Does `scripts/validate_feature_boundary.py` still pass?

---

## 7. Immediate Implication for Remaining Strategic Work

- PaddleOCR quarterly reassessment remains a strategic engine decision, not a default-boundary change.
- LayoutLMv3 domain training remains a dedicated optional lane.
- Any future VLM, RAG, or downstream semantic workflows must continue to sit outside the forensic-core promise unless the roadmap is explicitly re-scoped.

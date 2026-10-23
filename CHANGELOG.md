# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.0-rc.1] - Unpublished candidate

### Release candidate

- Git-free source builds bind the extracted root rather than an enclosing
  checkout; revision labels use an explicit revision or the runtime content hash.
- Keep the standalone TypeScript client on the public candidate version and
  align its buffer upload type annotation with the packaged client. Legacy
  dashboard deployment notes explicitly disclose the missing application bundle.
- Repair single-host API ownership, background drain, and CPU SOLO installation.
- Separate API upload staging from batch input; preserve job/output state on restart.
- Require native-PDF completeness and preserve searchable Latin lines with script-specific fonts.
- Add refreshable IAM credentials, control-plane identity separation, and presigned hostname policy.
- Reject unsupported federation mutual TLS; correct NFS mount paths and PostgreSQL TLS/schema restore handling.
- Preserve staged capacity during deletion and retain cleanup authority for delayed S3 uploads.
- Update frontend and SDK dependencies while preserving public SDK package names.
- Refreshed reviewed Python/coordinator/processing and frontend/tooling dependencies;
  retained the qualified inference bundle and Python3.9 SDK build/install support.
- Reject visible native glyphs mapped to whitespace or zero-width text before
  native publication, using printable-character ink coverage; retain source evidence.
- Bind presigned downloads to the configured bucket and exact artifact key;
  reject virtual-hosted bucket-segment confusion and missing endpoint prefixes.
- Preserve litigation-held rasters without starving active recovery; isolate
  retirement query/bookkeeping failures from staged-page recovery.
- Reject unsupported bundled client-only TLS topologies and hydrate saved Jobs
  preferences before the first request.
- Remove internal review identifiers from the user-visible legacy SDK deprecation warning.
- Remove newly introduced internal identifiers from the public GPU-OOM log and two assertion diagnostics using exact export transformations; rebuild and runtime-qualify affected root/API images without changing forensic behavior.
- Remove the unsupported upgrade-readiness blocker for the qualified PaddlePaddle 3.0/PaddleOCR 2.9 bundle; retain NumPy/OpenCV conflict checks.
- Linux ARM64 CPython3.10 now has an actual fasttext source-build, inference/HTTP
  transport and production CPU OCR canary. Shipping root/coordinator remain3.12;
  SDK Python3.9 support is separate. This is not all-language/GPU/system-fallback proof.
- Release preparation reads the canonical version owner, admits ASCII SemVer
  prereleases/build metadata, and preserves an existing candidate changelog section.
  Writes are atomic per file, not a multi-file rollback transaction.
- Frontend Docker contexts exclude host dependencies/caches; runtime fingerprints
  include nested business routes. Jobs preferences explicitly exclude Dashboard,
  Batches and Review; Node26 browser/API qualification covers the declared scope.
- Preserve optional legacy UIE and lightweight coordinator API requirements in
  source inventories; the legacy UIE dependency lane remains disabled/unqualified.
- Launch frontend Next directly with Node26; exclude global npm/npx from the
  copied runtime rootfs and shipping layers without changing the donor/application
  versions. Donor/build-cache retention and OS/Node advisory coverage remain separate.

See [candidate scope and limitations](docs/release-candidate.md). This is a bounded
release candidate; actual cloud, physical GPU/farm, mesh, NFS and replicated
recovery qualifications remain target-specific. Package/image publication is separate.

## [4.1.0] - 2026-05-21

### Summary

First public GitHub release of EDCOCR under the `mattmre/EDCOCR-PUBLIC` repository.
This release aligns the public version to the operator's internal release line
(v4.1) and ships the full forensic-grade OCR platform under Apache 2.0.

### Versioning

- Public release line begins at v4.1.0 to track the operator's internal release
  cadence; previous internal pre-release `2.2.0` is retained below for
  historical reference.

### Highlights

- Zero-hallucination OCR pipeline based on PaddleOCR 2.9.1 (CTC, no generative
  AI in the recognition path).
- Adaptive multi-language detection via FastText with per-span language sidecar
  output (opt-in).
- Forensic chain of custody with SHA-256 hash-chained JSONL audit logs.
- Page-level crash resume with deterministic recovery.
- Document Intelligence (PP-StructureV3 layout / table analysis, opt-in).
- REST API with API-key authentication, rate limiting, Pydantic validation,
  SSE streaming, and WebSocket progress.
- Distributed pipeline with Celery, RabbitMQ, Redis, PostgreSQL, and dynamic
  multi-capability workers.
- Production Helm chart with KEDA autoscaling, PDBs, network policies, and
  Prometheus + Grafana wiring.
- Optional enrichment layers: NER, document classification, structured field
  extraction, handwriting detection, signature verification, barcode / QR,
  OMR checkboxes, VLM gateway, semantic search.
- Translation seam for external translation services (default OFF).
- Python (`edcocr-sdk`) and TypeScript (`@edcocr/sdk`) SDKs.
- Air-gapped deployment support (model preload, bundle / deploy scripts).

### Installation

- Python SDK: `pip install edcocr-sdk==4.1.0`
- TypeScript SDK: `npm install @edcocr/sdk@4.1.0`
- Helm chart: `helm install edcocr ./helm/ocr-local --version 4.1.0`

## [2.2.0] - 2026-05-20

### Summary

First public release of EDCOCR -- a forensic-grade OCR platform for high-volume,
mixed-format document processing.

### Highlights

- Zero-hallucination OCR pipeline based on PaddleOCR 2.9.1 (CTC, no generative
  AI in the recognition path).
- Adaptive multi-language detection via FastText with per-span language sidecar
  output (opt-in).
- Forensic chain of custody with SHA-256 hash-chained JSONL audit logs.
- Page-level crash resume with deterministic recovery.
- Document Intelligence (PP-StructureV3 layout / table analysis, opt-in).
- REST API with API-key authentication, rate limiting, Pydantic validation,
  SSE streaming, and WebSocket progress.
- Distributed pipeline with Celery, RabbitMQ, Redis, PostgreSQL, and dynamic
  multi-capability workers.
- Production Helm chart with KEDA autoscaling, PDBs, network policies, and
  Prometheus + Grafana wiring.
- Optional enrichment layers: NER, document classification, structured field
  extraction, handwriting detection, signature verification, barcode / QR,
  OMR checkboxes, VLM gateway, semantic search.
- Translation seam for external translation services (default OFF).
- Python and TypeScript SDKs.

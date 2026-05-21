# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

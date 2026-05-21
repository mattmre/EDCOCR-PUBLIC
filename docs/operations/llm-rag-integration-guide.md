# LLM and RAG Integration Guide

Guide for integrating EDCOCR output with large language models, retrieval-augmented
generation pipelines, embedding stores, and semantic search systems.

**Audience**: Engineers building downstream AI/ML workflows that consume EDCOCR output.

**Version**: 4.1.0

---

## Table of Contents

1. [The Zero-Hallucination Boundary](#1-the-zero-hallucination-boundary)
2. [OCR Output Formats](#2-ocr-output-formats)
3. [Embedding Generation Patterns](#3-embedding-generation-patterns)
4. [Vector Store Integration](#4-vector-store-integration)
5. [RAG Pipeline Architecture](#5-rag-pipeline-architecture)
6. [Semantic Search](#6-semantic-search)
7. [Best Practices](#7-best-practices)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. The Zero-Hallucination Boundary

### Why EDCOCR Does Not Include LLMs

EDCOCR is a forensic-grade document processing system. Its core design principle
is **zero hallucination**: the OCR pipeline uses only Connectionist Temporal
Classification (CTC) decoders, never generative language models. Every character
in the output is decoded directly from the input image signal. Nothing is invented,
predicted, or interpolated by a language model.

This is a deliberate architectural choice for three reasons:

1. **Forensic integrity**. In legal, compliance, and investigative contexts, the
   provenance of every character matters. A CTC decoder can only produce characters
   that it detects in the pixel data. Generative models can produce plausible but
   factually incorrect text, which is unacceptable when documents serve as evidence.

2. **Auditability**. EDCOCR maintains a chain of custody (hash-chained JSONL
   audit log) and per-page confidence scores. Every output can be traced back to
   the source image and the OCR engine that produced it. Introducing a generative
   model would break this chain because the model's output cannot be deterministically
   reproduced from the input alone.

3. **Air-gapped deployment**. Many forensic and government deployments run in
   isolated networks. CTC-only inference requires only PaddleOCR model weights
   (pre-baked into the Docker image). No API calls to external LLM services are
   needed for the core OCR pipeline.

### Where the Boundary Lies

```
+---------------------------------------------------------------+
|                     EDCOCR (CTC-only)                         |
|                                                               |
|  Input Documents  -->  OCR Pipeline  -->  Structured Output   |
|  (PDF, TIFF, ...)     (PaddleOCR +       (TEXT, PDF, JSON,    |
|                        Tesseract          NER, extraction,    |
|                        fallback)          classification)     |
+---------------------------------------------------------------+
                             |
                    Zero-hallucination boundary
                             |
                             v
+---------------------------------------------------------------+
|               Downstream AI/ML (your code)                    |
|                                                               |
|  Chunking  -->  Embeddings  -->  Vector Store  -->  RAG/LLM   |
|  (consume        (vendor-          (pgvector,        (any     |
|   OCR output)     agnostic         Pinecone,         LLM)     |
|                   callable)         Weaviate, ...)            |
+---------------------------------------------------------------+
```

Everything above the boundary is deterministic, auditable, CTC-only OCR.
Everything below is where generative AI belongs. This guide covers how to
build the "below the boundary" layer using EDCOCR outputs as input.

### What EDCOCR Provides for Downstream AI

While EDCOCR does not include generative LLMs in its core pipeline, it does
produce rich structured outputs that are specifically designed to feed into
downstream AI systems:

- **Plain text** (`EXPORT/TEXT/`) for direct embedding and RAG ingestion
- **Layout structure** (`EXPORT/STRUCTURE/`) for semantic chunk boundary detection
- **Named entities** (`EXPORT/NER/`) for metadata enrichment in vector stores
- **Document classification** (`EXPORT/CLASSIFICATION/`) for routing and filtering
- **Structured fields** (`EXPORT/EXTRACTION/`) for knowledge base population
- **Per-page confidence scores** (`EXPORT/VALIDATION/`) for quality-based filtering
- **Webhook notifications** for event-driven embedding pipeline triggers
- **VLM gateway** (opt-in) for semantic search via external inference servers

---

## 2. OCR Output Formats

### 2.1 Plain Text (`EXPORT/TEXT/`)

The simplest downstream input. UTF-8 text files, one per document, suitable for
direct chunking and embedding.

```python
from pathlib import Path

text_path = Path("ocr_output/EXPORT/TEXT/contract.txt")
content = text_path.read_text(encoding="utf-8")
```

### 2.2 Layout Structure (`EXPORT/STRUCTURE/`)

When the pipeline runs with `--enable-docintel`, each document gets a sidecar
JSON describing its layout regions, tables, and figures. Useful for chunking on
semantic boundaries (paragraph, table row, list item) rather than fixed token
windows.

```json
{
  "pages": [
    {
      "page_number": 1,
      "layout_regions": [
        {"type": "title", "bbox": [...], "confidence": 0.97},
        {"type": "paragraph", "bbox": [...], "text": "..."}
      ],
      "tables": [...]
    }
  ]
}
```

### 2.3 Named Entities (`EXPORT/NER/`)

Pre-extracted entities for metadata enrichment in your vector store.

### 2.4 Per-Page Validation (`EXPORT/VALIDATION/`)

Confidence and quality classifications you can use to filter low-quality pages
out of an index.

---

## 3. Embedding Generation Patterns

EDCOCR is vendor-agnostic about embeddings. The recommended pattern is to
declare an `embed_fn` callable in your code and inject whichever embedding
provider you want to use.

```python
from typing import Callable, Iterable

EmbedFn = Callable[[Iterable[str]], list[list[float]]]

def embed_chunks(chunks: list[str], embed_fn: EmbedFn) -> list[list[float]]:
    """Generate embeddings for a list of text chunks.

    `embed_fn` is supplied by the caller and may wrap any provider:
    a local sentence-transformers model, a self-hosted vLLM endpoint, or
    any cloud embedding API. EDCOCR does not bundle a specific vendor.
    """
    return embed_fn(chunks)
```

For air-gapped deployments, use a locally hosted embedding model
(e.g. `sentence-transformers/all-MiniLM-L6-v2`, BGE, or any HuggingFace
encoder you can host on your own infrastructure).

For networked deployments, any embedding provider that returns vectors via
HTTP works -- wire it into `embed_fn` and the rest of the pipeline is
unchanged.

---

## 4. Vector Store Integration

Similarly, EDCOCR does not assume a specific vector store. The recommended
pattern is a `vector_search_fn` callable.

```python
from typing import Callable, Sequence

SearchFn = Callable[[list[float], int], list[dict]]

def retrieve(query_vector: list[float], k: int, search_fn: SearchFn) -> list[dict]:
    """Return the top-k matches for `query_vector`.

    `search_fn` is supplied by the caller and may wrap pgvector,
    Pinecone, Weaviate, Qdrant, FAISS, Milvus, or any other store.
    """
    return search_fn(query_vector, k)
```

Useful metadata to store alongside each chunk:

- `document_id`, `page_number`, `bbox` (from layout structure)
- `document_class` (from classification)
- `entities` (from NER)
- `confidence` (from validation; lets you filter out low-quality pages)

---

## 5. RAG Pipeline Architecture

A canonical EDCOCR-backed RAG pipeline:

```
1. Ingest source documents into EDCOCR.
2. EDCOCR emits TEXT + STRUCTURE + NER + CLASSIFICATION + VALIDATION.
3. Downstream pipeline:
   a. Chunk text on layout boundaries (paragraphs, table rows).
   b. Filter chunks by page confidence threshold.
   c. Embed chunks with embed_fn.
   d. Upsert vectors + metadata into vector store via search_fn.
4. Query time:
   a. Embed the user query with embed_fn.
   b. Retrieve top-k chunks via search_fn.
   c. Pass chunks + query to the LLM of your choice via llm_complete_fn.
```

Reference callable interface:

```python
from typing import Callable

LlmCompleteFn = Callable[[str, list[str]], str]

def rag_answer(
    query: str,
    embed_fn: EmbedFn,
    search_fn: SearchFn,
    llm_complete_fn: LlmCompleteFn,
    k: int = 5) -> str:
    [query_vec] = embed_fn([query])
    hits = search_fn(query_vec, k)
    context = [hit["text"] for hit in hits]
    return llm_complete_fn(query, context)
```

This shape lets operators swap any LLM backend without changing EDCOCR or the
retrieval layer.

---

## 6. Semantic Search

For semantic search at the document level, two patterns:

1. **Index every chunk** and run nearest-neighbor search over the chunks.
   Standard RAG retrieval.
2. **Use the optional VLM gateway** (`vlm_gateway.py`) to delegate the search
   to an external inference server that accepts page images + text and returns
   ranked passages. Disabled by default; enable with `VLM_ENABLED=true` and
   configure `VLM_ENDPOINT_URL` to point at any
   `v1/chat/completions`-compatible inference server (vLLM, TensorRT-LLM, or
   another compatible runtime).

---

## 7. Best Practices

- **Filter on page confidence**. EDCOCR's per-page validation report flags
  low-quality pages. Skip pages below your confidence floor (commonly 0.75)
  before embedding.
- **Chunk on layout boundaries**. The layout structure sidecar gives you
  paragraph/table/list boundaries; chunk on those rather than fixed token
  windows for better retrieval quality.
- **Carry forensic metadata into the vector store**. `document_id`,
  `page_number`, `bbox`, and the chunk's source page confidence let you
  trace any retrieved answer back to a specific page region.
- **Keep generative work outside the OCR pipeline**. EDCOCR's
  zero-hallucination guarantee depends on never mixing generative output into
  the OCR text layer. Treat the EDCOCR output as evidence; treat downstream
  LLM output as a working artifact.
- **Use webhooks for event-driven indexing**. EDCOCR emits HMAC-signed
  completion webhooks; subscribe them to your embedding worker queue rather
  than polling.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Retrieval surfaces low-quality pages | No confidence filter on ingest | Skip pages with validation confidence < 0.75 |
| Chunks split mid-sentence | Fixed-window chunking on raw text | Use layout structure boundaries |
| Answers cite the wrong document | Missing `document_id` metadata | Carry `document_id` + `page_number` into each vector |
| Embeddings re-generated on re-ingest | No idempotency key | Use `document_id + chunk_index + text_sha256` as the upsert key |
| VLM gateway times out | Endpoint slow under load | Tune `VLM_TIMEOUT_SECONDS` and `VLM_MAX_CONTEXT_PAGES` in `vlm_config.py` |

For deeper integration questions, see the API reference (`docs/API-REFERENCE.md`)
and the system blueprint (`docs/00-SYSTEM-BLUEPRINT.md`).

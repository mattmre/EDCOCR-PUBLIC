"""Semantic search endpoint for document chunks.

Phase 10 — Persistent Intelligence Platform: provides natural-language
search over stored document chunks using embedding similarity or text
fallback.

Endpoints:
- POST /api/v1/search/semantic/

Authentication: uses the same METRICS_API_KEY mechanism as the metrics endpoint.
"""

import heapq
import json
import logging
import math

from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .extraction_models import DocumentChunk
from .metrics_auth import extract_tenant_scope, require_metrics_api_key

logger = logging.getLogger(__name__)

_MAX_SEARCH_BODY_BYTES = 64 * 1024
_MAX_QUERY_CHARS = 4_096
_MAX_TENANT_ID_CHARS = 128
_MAX_EMBEDDING_DIMENSION = 4_096
_MAX_RESULT_SNIPPET_CHARS = 8_192

# ---------------------------------------------------------------------------
# Embedding service bridge
# ---------------------------------------------------------------------------
# Import at module level with graceful fallback so the view module loads
# even when sentence-transformers is not installed.
try:
    import os
    import sys

    # Add repo root to path so embedding_service is importable from coordinator
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from embedding_service import ChunkEmbedder

    _EMBEDDER_AVAILABLE = True
except ImportError:
    _EMBEDDER_AVAILABLE = False
    ChunkEmbedder = None


def _cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two lists of floats."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    if len(vec_a) > _MAX_EMBEDDING_DIMENSION:
        return 0.0
    try:
        values_a = [float(value) for value in vec_a]
        values_b = [float(value) for value in vec_b]
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if any(not math.isfinite(value) for value in values_a + values_b):
        return 0.0
    dot = math.fsum(a * b for a, b in zip(values_a, values_b))
    mag_a = math.sqrt(math.fsum(a * a for a in values_a))
    mag_b = math.sqrt(math.fsum(b * b for b in values_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    score = dot / (mag_a * mag_b)
    return score if math.isfinite(score) else 0.0


def _result_snippet(value) -> str:
    """Keep stored OCR text from turning a bounded result page into a dump."""
    text = value if isinstance(value, str) else str(value or "")
    return text[:_MAX_RESULT_SNIPPET_CHARS]


def _text_search_fallback(query, limit, *, tenant_id: str):
    """Fall back to text-based search when embeddings are unavailable."""
    qs = DocumentChunk.objects.filter(
        chunk_text__icontains=query,
        job__tenant_id=tenant_id,
    ).select_related("job")[:limit]

    results = []
    for chunk in qs:
        results.append(
            {
                "job_id": str(chunk.job_id),
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "chunk_text": _result_snippet(chunk.chunk_text),
                "score": 1.0,  # exact text match
                "match_type": "text",
            }
        )
    return results


def _semantic_search(query_embedding, limit, *, tenant_id: str):
    """Search chunks by cosine similarity against stored embeddings."""
    # Fetch chunks that have embeddings (tenant-scoped — R3-RC-02 / R4-RC-02).
    chunks = (
        DocumentChunk.objects.exclude(embedding_json__isnull=True)
        .filter(job__tenant_id=tenant_id)
        .select_related("job")
    )

    # Compute cosine similarity for each chunk
    scored = []
    for sequence, chunk in enumerate(chunks.iterator(chunk_size=256)):
        embedding = chunk.embedding_json
        if not isinstance(embedding, list):
            continue
        score = _cosine_similarity(query_embedding, embedding)
        if not score and score != 0.0:
            continue
        candidate = (score, sequence, chunk)
        if len(scored) < limit:
            heapq.heappush(scored, candidate)
        elif candidate[:2] > scored[0][:2]:
            heapq.heapreplace(scored, candidate)

    # Sort by descending similarity
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    results = []
    for score, _sequence, chunk in scored[:limit]:
        results.append(
            {
                "job_id": str(chunk.job_id),
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "chunk_text": _result_snippet(chunk.chunk_text),
                "score": round(score, 4),
                "match_type": "semantic",
            }
        )
    return results


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@require_POST
def semantic_search(request):
    """Search documents using natural language query.

    POST /api/v1/search/semantic/
    Body: {"query": "find invoices from January 2025", "limit": 10}

    When embeddings are available, the query is embedded and compared against
    stored chunk embeddings using cosine similarity. When embeddings are not
    available, falls back to text-based search.

    Returns ranked results with confidence scores, snippets, and job references.
    """
    if not require_metrics_api_key(request):
        return JsonResponse({"error": "Unauthorized"}, status=401)

    # Reject oversized bodies before JSON parsing. Django may already enforce
    # a global upload limit, but this endpoint also needs a local ceiling when
    # deployed behind a permissive or chunked request parser.
    try:
        declared_length = int(request.META.get("CONTENT_LENGTH") or 0)
    except (TypeError, ValueError):
        declared_length = 0
    if declared_length > _MAX_SEARCH_BODY_BYTES:
        return JsonResponse({"error": "Search request exceeds size limit"}, status=413)
    raw_body = request.body
    if len(raw_body) > _MAX_SEARCH_BODY_BYTES:
        return JsonResponse({"error": "Search request exceeds size limit"}, status=413)

    # Parse JSON body
    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return JsonResponse(
            {"error": "Invalid JSON body"},
            status=400,
        )
    if not isinstance(body, dict):
        return JsonResponse({"error": "JSON body must be an object"}, status=400)

    query_value = body.get("query", "")
    if not isinstance(query_value, str):
        return JsonResponse({"error": 'Field "query" must be a string'}, status=400)
    query = query_value.strip()
    if not query:
        return JsonResponse(
            {"error": 'Field "query" is required'},
            status=400,
        )
    if len(query) > _MAX_QUERY_CHARS:
        return JsonResponse({"error": 'Field "query" exceeds size limit'}, status=413)

    tenant_id = extract_tenant_scope(request, body)
    if not tenant_id:
        return JsonResponse(
            {
                "error": "tenant_id is required",
                "message": (
                    "Semantic search requires tenant_id "
                    "(JSON body, query param, or X-Tenant-Id header)."
                ),
            },
            status=400,
        )
    if len(tenant_id) > _MAX_TENANT_ID_CHARS:
        return JsonResponse({"error": "tenant_id exceeds size limit"}, status=400)

    try:
        limit = int(body.get("limit", 10))
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, 100))

    # Attempt embedding-based search
    search_mode = "text"
    if _EMBEDDER_AVAILABLE:
        try:
            embedder = ChunkEmbedder()
            if embedder.is_available:
                query_embedding = embedder.embed_text(query)
                if query_embedding:
                    results = _semantic_search(
                        query_embedding, limit, tenant_id=tenant_id
                    )
                    search_mode = "semantic"
                else:
                    results = _text_search_fallback(query, limit, tenant_id=tenant_id)
            else:
                results = _text_search_fallback(query, limit, tenant_id=tenant_id)
        except Exception as exc:
            logger.warning(
                "Semantic search embedding failed, using text fallback: %s", exc
            )
            results = _text_search_fallback(query, limit, tenant_id=tenant_id)
    else:
        results = _text_search_fallback(query, limit, tenant_id=tenant_id)

    return JsonResponse(
        {
            "query": query,
            "tenant_id": tenant_id,
            "search_mode": search_mode,
            "results": results,
            "total": len(results),
            "limit": limit,
        }
    )

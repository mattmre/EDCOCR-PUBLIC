"""Semantic search endpoint for document chunks.

Phase 10 — Persistent Intelligence Platform: provides natural-language
search over stored document chunks using embedding similarity or text
fallback.

Endpoints:
- POST /api/v1/search/semantic/

Authentication: uses the same METRICS_API_KEY mechanism as the metrics endpoint.
"""

import json
import logging
import math

from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .extraction_models import DocumentChunk
from .metrics_auth import has_valid_metrics_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding service bridge
# ---------------------------------------------------------------------------
# Import at module level with graceful fallback so the view module loads
# even when sentence-transformers is not installed.
try:
    import os
    import sys

    # Add repo root to path so embedding_service is importable from coordinator
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _text_search_fallback(query, limit):
    """Fall back to text-based search when embeddings are unavailable."""
    qs = DocumentChunk.objects.filter(
        chunk_text__icontains=query
    ).select_related('job')[:limit]

    results = []
    for chunk in qs:
        results.append({
            'job_id': str(chunk.job_id),
            'page_number': chunk.page_number,
            'chunk_index': chunk.chunk_index,
            'chunk_text': chunk.chunk_text,
            'score': 1.0,  # exact text match
            'match_type': 'text',
        })
    return results


def _semantic_search(query_embedding, limit):
    """Search chunks by cosine similarity against stored embeddings."""
    # Fetch chunks that have embeddings
    chunks = DocumentChunk.objects.exclude(
        embedding_json__isnull=True
    ).select_related('job')

    # Compute cosine similarity for each chunk
    scored = []
    for chunk in chunks:
        embedding = chunk.embedding_json
        if not isinstance(embedding, list):
            continue
        score = _cosine_similarity(query_embedding, embedding)
        scored.append((score, chunk))

    # Sort by descending similarity
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, chunk in scored[:limit]:
        results.append({
            'job_id': str(chunk.job_id),
            'page_number': chunk.page_number,
            'chunk_index': chunk.chunk_index,
            'chunk_text': chunk.chunk_text,
            'score': round(score, 4),
            'match_type': 'semantic',
        })
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
    if not has_valid_metrics_key(request):
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    # Parse JSON body
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse(
            {'error': 'Invalid JSON body'},
            status=400,
        )

    query = body.get('query', '').strip()
    if not query:
        return JsonResponse(
            {'error': 'Field "query" is required'},
            status=400,
        )

    try:
        limit = int(body.get('limit', 10))
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, 100))

    # Attempt embedding-based search
    search_mode = 'text'
    if _EMBEDDER_AVAILABLE:
        try:
            embedder = ChunkEmbedder()
            if embedder.is_available:
                query_embedding = embedder.embed_text(query)
                if query_embedding:
                    results = _semantic_search(query_embedding, limit)
                    search_mode = 'semantic'
                else:
                    results = _text_search_fallback(query, limit)
            else:
                results = _text_search_fallback(query, limit)
        except Exception as exc:
            logger.warning("Semantic search embedding failed, using text fallback: %s", exc)
            results = _text_search_fallback(query, limit)
    else:
        results = _text_search_fallback(query, limit)

    return JsonResponse({
        'query': query,
        'search_mode': search_mode,
        'results': results,
        'total': len(results),
        'limit': limit,
    })

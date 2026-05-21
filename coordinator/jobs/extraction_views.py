"""REST API views for entity and form-value queries.

Phase 10 — Persistent Intelligence Platform: provides paginated,
filterable query endpoints for extracted entities and form values.

Endpoints:
- GET /api/v1/jobs/{job_id}/entities/  — entities for a specific job
- GET /api/v1/jobs/{job_id}/forms/     — form values for a specific job
- GET /api/v1/search/entities/         — cross-job entity search

Authentication: uses the same METRICS_API_KEY mechanism as the metrics endpoint.
"""

import logging

from django.db.models import Q
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .extraction_models import ExtractedEntity, ExtractedFormValue
from .metrics_auth import has_valid_metrics_key
from .models import Job

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pagination defaults
# ---------------------------------------------------------------------------
DEFAULT_LIMIT = 50
MAX_LIMIT = 500


def _parse_pagination(request):
    """Parse limit/offset from query params with validation."""
    try:
        limit = int(request.GET.get('limit', DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT
    limit = max(1, min(limit, MAX_LIMIT))

    try:
        offset = int(request.GET.get('offset', 0))
    except (TypeError, ValueError):
        offset = 0
    offset = max(0, offset)

    return limit, offset


def _parse_float(value, default=None):
    """Parse a float query parameter safely."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _entity_to_dict(entity):
    """Serialize an ExtractedEntity to a JSON-compatible dict."""
    result = {
        'id': entity.id,
        'job_id': str(entity.job_id),
        'page_number': entity.page_number,
        'entity_type': entity.entity_type,
        'entity_text': entity.entity_text,
        'confidence': entity.confidence,
        'source_module': entity.source_module,
        'metadata': entity.metadata_json,
        'created_at': entity.created_at.isoformat() if entity.created_at else None,
    }
    # Include bbox only when present
    if entity.bbox_x1 is not None:
        result['bbox'] = [
            entity.bbox_x1,
            entity.bbox_y1,
            entity.bbox_x2,
            entity.bbox_y2,
        ]
    return result


def _form_value_to_dict(fv):
    """Serialize an ExtractedFormValue to a JSON-compatible dict."""
    return {
        'id': fv.id,
        'job_id': str(fv.job_id),
        'page_number': fv.page_number,
        'field_key': fv.field_key,
        'field_value': fv.field_value,
        'confidence': fv.confidence,
        'source_module': fv.source_module,
        'metadata': fv.metadata_json,
        'created_at': fv.created_at.isoformat() if fv.created_at else None,
    }


# ---------------------------------------------------------------------------
# Job-scoped entity endpoint
# ---------------------------------------------------------------------------


@require_GET
def job_entities(request, job_id):
    """Return extracted entities for a specific job.

    GET /api/v1/jobs/{job_id}/entities/

    Query params:
    - entity_type: filter by entity type (e.g., PERSON, DATE)
    - page_number: filter by page number
    - min_confidence: minimum confidence threshold (0.0-1.0)
    - source_module: filter by extraction source
    - q: text search in entity_text
    - limit: max results (default 50, max 500)
    - offset: pagination offset
    """
    if not has_valid_metrics_key(request):
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    # Validate job exists
    try:
        Job.objects.get(job_id=job_id)
    except (Job.DoesNotExist, ValueError):
        return JsonResponse({'error': 'Job not found'}, status=404)

    qs = ExtractedEntity.objects.filter(job_id=job_id)

    # Filters
    entity_type = request.GET.get('entity_type')
    if entity_type:
        qs = qs.filter(entity_type=entity_type)

    page_number = request.GET.get('page_number')
    if page_number:
        try:
            qs = qs.filter(page_number=int(page_number))
        except (TypeError, ValueError):
            pass

    min_confidence = _parse_float(request.GET.get('min_confidence'))
    if min_confidence is not None:
        qs = qs.filter(confidence__gte=min_confidence)

    source_module = request.GET.get('source_module')
    if source_module:
        qs = qs.filter(source_module=source_module)

    q = request.GET.get('q')
    if q:
        qs = qs.filter(entity_text__icontains=q)

    total = qs.count()
    limit, offset = _parse_pagination(request)
    entities = qs[offset:offset + limit]

    return JsonResponse({
        'job_id': str(job_id),
        'entities': [_entity_to_dict(e) for e in entities],
        'total': total,
        'limit': limit,
        'offset': offset,
    })


# ---------------------------------------------------------------------------
# Job-scoped form values endpoint
# ---------------------------------------------------------------------------


@require_GET
def job_form_values(request, job_id):
    """Return extracted form key-value pairs for a specific job.

    GET /api/v1/jobs/{job_id}/forms/

    Query params:
    - field_key: filter by key name (exact match)
    - page_number: filter by page number
    - min_confidence: minimum confidence threshold
    - source_module: filter by extraction source
    - q: text search in field_key and field_value
    - limit/offset: pagination
    """
    if not has_valid_metrics_key(request):
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    try:
        Job.objects.get(job_id=job_id)
    except (Job.DoesNotExist, ValueError):
        return JsonResponse({'error': 'Job not found'}, status=404)

    qs = ExtractedFormValue.objects.filter(job_id=job_id)

    field_key = request.GET.get('field_key')
    if field_key:
        qs = qs.filter(field_key=field_key)

    page_number = request.GET.get('page_number')
    if page_number:
        try:
            qs = qs.filter(page_number=int(page_number))
        except (TypeError, ValueError):
            pass

    min_confidence = _parse_float(request.GET.get('min_confidence'))
    if min_confidence is not None:
        qs = qs.filter(confidence__gte=min_confidence)

    source_module = request.GET.get('source_module')
    if source_module:
        qs = qs.filter(source_module=source_module)

    q = request.GET.get('q')
    if q:
        qs = qs.filter(Q(field_key__icontains=q) | Q(field_value__icontains=q))

    total = qs.count()
    limit, offset = _parse_pagination(request)
    form_values = qs[offset:offset + limit]

    return JsonResponse({
        'job_id': str(job_id),
        'form_values': [_form_value_to_dict(fv) for fv in form_values],
        'total': total,
        'limit': limit,
        'offset': offset,
    })


# ---------------------------------------------------------------------------
# Cross-job entity search
# ---------------------------------------------------------------------------


@require_GET
def search_entities(request):
    """Search entities across all jobs.

    GET /api/v1/search/entities/

    Query params:
    - q: text search in entity_text (required)
    - entity_type: filter by entity type
    - min_confidence: minimum confidence threshold
    - source_module: filter by extraction source
    - limit/offset: pagination
    """
    if not has_valid_metrics_key(request):
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    q = request.GET.get('q', '').strip()
    if not q:
        return JsonResponse(
            {'error': 'Query parameter "q" is required'},
            status=400,
        )

    qs = ExtractedEntity.objects.filter(entity_text__icontains=q)

    entity_type = request.GET.get('entity_type')
    if entity_type:
        qs = qs.filter(entity_type=entity_type)

    min_confidence = _parse_float(request.GET.get('min_confidence'))
    if min_confidence is not None:
        qs = qs.filter(confidence__gte=min_confidence)

    source_module = request.GET.get('source_module')
    if source_module:
        qs = qs.filter(source_module=source_module)

    total = qs.count()
    limit, offset = _parse_pagination(request)
    entities = qs[offset:offset + limit]

    return JsonResponse({
        'query': q,
        'entities': [_entity_to_dict(e) for e in entities],
        'total': total,
        'limit': limit,
        'offset': offset,
    })

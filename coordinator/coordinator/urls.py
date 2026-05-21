from django.contrib import admin
from django.urls import path
from jobs.extraction_views import job_entities, job_form_values, search_entities
from jobs.semantic_search_views import semantic_search
from jobs.views import dashboard, metrics, prometheus_metrics

urlpatterns = [
    path('admin/', admin.site.urls),
    path('dashboard/', dashboard, name='dashboard'),
    path('api/v1/metrics/', metrics, name='metrics'),
    path('api/v1/prometheus/', prometheus_metrics, name='prometheus-metrics'),
    # Phase 10: Extraction query endpoints
    path(
        'api/v1/jobs/<uuid:job_id>/entities/',
        job_entities,
        name='job-entities',
    ),
    path(
        'api/v1/jobs/<uuid:job_id>/forms/',
        job_form_values,
        name='job-form-values',
    ),
    path(
        'api/v1/search/entities/',
        search_entities,
        name='search-entities',
    ),
    # Phase 10: Semantic search endpoint
    path(
        'api/v1/search/semantic/',
        semantic_search,
        name='semantic-search',
    ),
]

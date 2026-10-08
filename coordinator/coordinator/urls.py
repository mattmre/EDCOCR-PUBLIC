from django.contrib import admin
from django.urls import path
from jobs.extraction_views import job_entities, job_form_values, search_entities
from jobs.processing_api import (
    detailed_health,
    get_document_bundle,
    get_job,
    submit_job,
)
from jobs.semantic_search_views import semantic_search
from jobs.views import dashboard, health_probe, metrics, prometheus_metrics

# R2-RC-03: wire real OTPAdminSite so ADMIN_MFA_REQUIRED has a TOTP login surface
# (django-otp OTPMiddleware alone does not replace password-only admin login).
try:
    from django_otp.admin import OTPAdminSite
    from jobs.mfa import is_mfa_required

    if is_mfa_required():
        # Swap class on the default site so existing model registrations remain.
        # PA-A15: do not swallow errors — fail startup if MFA required but OTP
        # admin cannot install (no silent password-only admin).
        admin.site.__class__ = OTPAdminSite  # type: ignore[misc]
except ImportError:
    from jobs.mfa import is_mfa_required as _mfa_required

    if _mfa_required():
        raise RuntimeError(
            "ADMIN_MFA_REQUIRED but django_otp.admin.OTPAdminSite is unavailable"
        ) from None

urlpatterns = [
    path("admin/", admin.site.urls),
    path("dashboard/", dashboard, name="dashboard"),
    path("api/v1/metrics/", metrics, name="metrics"),
    path("api/v1/prometheus/", prometheus_metrics, name="prometheus-metrics"),
    path("api/v1/health", health_probe, name="health-probe"),
    path("api/v1/health/detailed", detailed_health, name="processing-health-detailed"),
    path("api/v1/jobs", submit_job, name="processing-submit-job"),
    path("api/v1/jobs/<uuid:job_id>", get_job, name="processing-get-job"),
    path(
        "api/v1/jobs/<uuid:job_id>/document-bundle",
        get_document_bundle,
        name="processing-get-document-bundle",
    ),
    # Phase 10: Extraction query endpoints
    path(
        "api/v1/jobs/<uuid:job_id>/entities/",
        job_entities,
        name="job-entities",
    ),
    path(
        "api/v1/jobs/<uuid:job_id>/forms/",
        job_form_values,
        name="job-form-values",
    ),
    path(
        "api/v1/search/entities/",
        search_entities,
        name="search-entities",
    ),
    # Phase 10: Semantic search endpoint
    path(
        "api/v1/search/semantic/",
        semantic_search,
        name="semantic-search",
    ),
]

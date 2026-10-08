"""HTTP middleware: pin Postgres RLS GUC from request tenant scope.

Runs after auth-capable middleware so ``extract_tenant_scope`` can read
headers / query params used by metrics and API views.
"""

from __future__ import annotations

import logging

from django.db import transaction

from .rls import clear_tenant_rls_context, set_tenant_rls_context

logger = logging.getLogger(__name__)


class TenantRLSMiddleware:
    """SET LOCAL app.tenant_id from request tenant scope for each request."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        tenant_id = ""
        try:
            from .metrics_auth import extract_tenant_scope

            tenant_id = extract_tenant_scope(request) or ""
        except Exception:
            logger.debug("extract_tenant_scope failed in RLS middleware", exc_info=True)

        # ``set_config(..., true)`` is transaction-local.  Wrapping the whole
        # request is therefore mandatory; setting the GUC in autocommit mode
        # leaves every ORM query outside the intended RLS scope.
        with transaction.atomic():
            set_tenant_rls_context(tenant_id)
            try:
                return self.get_response(request)
            finally:
                clear_tenant_rls_context()

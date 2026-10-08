"""Repeat cleanup of retired S3 job prefixes after delayed publication."""

from datetime import timedelta

from django.db import models
from django.utils import timezone

from .models import Job, JobArtifactRetirement
from .rls import control_plane_rls_context


def retain_cleanup_authority(job, backend):
    """Call inside the transaction deleting Job, after verified initial sweep."""
    if backend.backend_name != "s3":
        return False
    from .tasks import _storage_binding_fingerprint

    binding = _storage_binding_fingerprint("s3")
    persisted = (job.settings_json or {}).get("storage_binding_fingerprint")
    if persisted and persisted != binding:
        raise RuntimeError("retired job storage binding changed")
    row, _created = JobArtifactRetirement.objects.get_or_create(
        job_uuid=job.pk,
        defaults={
            "tenant_id": job.tenant_id or "",
            "backend_binding": binding,
            "next_sweep_at": timezone.now(),
        },
    )
    if row.backend_binding != binding or row.tenant_id != (job.tenant_id or ""):
        raise RuntimeError("retired job identity must not be reused")
    return True


def sweep_retired_job_artifacts(*, limit=100):
    """Sweep exact UUID prefixes forever; empty passes cannot exclude late PUTs.

    Credentials may rotate, but endpoint/bucket/region must stay bound. This
    covers current objects, not historical versions or object-lock copies.
    UUIDs must never be reused; any live Job prevents deletion of its prefix.
    """
    from .tasks import (
        _get_storage_backend,
        _storage_binding_fingerprint,
        is_litigation_hold_active,
    )

    if is_litigation_hold_active():
        return {"swept": 0, "failed": 0, "litigation_hold": True}
    now = timezone.now()
    swept = failed = 0
    with control_plane_rls_context():
        rows = list(JobArtifactRetirement.objects.filter(next_sweep_at__lte=now)
                    .order_by("next_sweep_at")[:limit])
        for row in rows:
            try:
                if Job.objects.filter(pk=row.job_uuid).exists():
                    raise RuntimeError("retired_job_uuid_still_exists")
                if row.backend_binding != _storage_binding_fingerprint("s3"):
                    raise RuntimeError("storage_binding_changed")
                backend = _get_storage_backend()
                if backend.backend_name != "s3":
                    raise RuntimeError("retired_job_backend_is_not_s3")
                prefix = f"jobs/{row.job_uuid}/"
                keys = backend.list_objects(prefix)
                if any(not key.startswith(prefix) for key in keys):
                    raise RuntimeError("storage_listing_outside_retired_prefix")
                if keys and backend.delete_many(keys) != len(keys):
                    raise RuntimeError("retired_job_partial_delete")
                if backend.list_objects(prefix):
                    raise RuntimeError("retired_job_objects_remain")
            except Exception as exc:
                # Provider errors may contain customer paths or credentials.
                JobArtifactRetirement.objects.filter(pk=row.pk).update(
                    last_error=type(exc).__name__, next_sweep_at=now + timedelta(minutes=5),
                )
                failed += 1
            else:
                JobArtifactRetirement.objects.filter(pk=row.pk).update(
                    last_error="", last_swept_at=now,
                    sweep_count=models.F("sweep_count") + 1,
                    next_sweep_at=now + timedelta(hours=1),
                )
                swept += 1
    return {"swept": swept, "failed": failed}

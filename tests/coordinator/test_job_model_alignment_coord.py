"""Tests for Coordinator Job model alignment.

Verifies the api_job_id cross-reference field exists on the coordinator
Job model.  Requires Django to be configured — skipped otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

django = pytest.importorskip("django", reason="Django required for coordinator model tests")

COORDINATOR_ROOT = Path(__file__).resolve().parents[2] / "coordinator"
sys.path.insert(0, str(COORDINATOR_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coordinator.settings_test")

import django as _django  # noqa: E402

_django.setup()

from django.db import models as django_models  # noqa: E402

from jobs.models import Job  # noqa: E402


class TestCoordinatorJobApiJobId:
    """Verify api_job_id field on coordinator Job model."""

    def test_api_job_id_field_exists(self):
        field = Job._meta.get_field("api_job_id")
        assert field is not None

    def test_api_job_id_max_length(self):
        field = Job._meta.get_field("api_job_id")
        assert field.max_length == 64

    def test_api_job_id_has_db_index(self):
        field = Job._meta.get_field("api_job_id")
        assert field.db_index is True

    def test_api_job_id_blank_default(self):
        field = Job._meta.get_field("api_job_id")
        assert field.default == ""
        assert field.blank is True

    def test_api_job_id_is_charfield(self):
        field = Job._meta.get_field("api_job_id")
        assert isinstance(field, django_models.CharField)

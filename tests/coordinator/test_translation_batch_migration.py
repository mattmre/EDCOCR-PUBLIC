"""Tests for the BatchTranslation migration (Plan B Wave M2 -- B17).

Verifies:
- Migration ``0012_batch_translation`` exists with the expected name and
  declared dependency on the prior migration in the chain.
- ``BatchTranslationJob`` and ``BatchTranslationInput`` Django models
  expose the contract relied on by ``ocr_local.translation.batch``
  (status enum, terminal-set, related_name, FK, unique-together, index).
- ``makemigrations --check`` reports no pending model changes (the model
  state is fully reflected in 0012).

Requires Django to be configured -- skipped otherwise.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

django = pytest.importorskip(
    "django", reason="Django required for coordinator migration tests",
)

COORDINATOR_ROOT = Path(__file__).resolve().parents[2] / "coordinator"
if Path.cwd().resolve() != COORDINATOR_ROOT.resolve():
    pytest.skip(
        "coordinator.settings_test is importable only from the coordinator test context",
        allow_module_level=True,
    )

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coordinator.settings_test")

import django as _django  # noqa: E402

_django.setup()

from django.db import models as django_models  # noqa: E402

from jobs.models import (  # noqa: E402
    BatchTranslationInput,
    BatchTranslationJob,
)

# ---------------------------------------------------------------------------
# Migration file presence
# ---------------------------------------------------------------------------


class TestMigrationFile:
    def test_migration_module_importable(self):
        mod = importlib.import_module(
            "jobs.migrations.0012_batch_translation",
        )
        assert hasattr(mod, "Migration")

    def test_migration_dependencies(self):
        mod = importlib.import_module(
            "jobs.migrations.0012_batch_translation",
        )
        deps = mod.Migration.dependencies
        assert any(
            app == "jobs" and "0011" in name for app, name in deps
        ), f"missing dependency on 0011_*: {deps}"


# ---------------------------------------------------------------------------
# BatchTranslationJob model contract
# ---------------------------------------------------------------------------


class TestBatchTranslationJob:
    def test_status_constants(self):
        assert BatchTranslationJob.STATUS_PENDING == "pending"
        assert BatchTranslationJob.STATUS_RUNNING == "running"
        assert BatchTranslationJob.STATUS_COMPLETED == "completed"
        assert BatchTranslationJob.STATUS_FAILED == "failed"
        assert BatchTranslationJob.STATUS_CANCELLED == "cancelled"

    def test_terminal_statuses_set(self):
        ts = BatchTranslationJob.TERMINAL_STATUSES
        assert "completed" in ts
        assert "failed" in ts
        assert "cancelled" in ts
        assert "pending" not in ts
        assert "running" not in ts

    def test_batch_id_is_primary_key(self):
        f = BatchTranslationJob._meta.get_field("batch_id")
        assert f.primary_key is True
        assert isinstance(f, django_models.CharField)
        assert f.max_length == 64

    def test_tenant_id_indexed(self):
        f = BatchTranslationJob._meta.get_field("tenant_id")
        assert f.db_index is True

    def test_status_field_indexed(self):
        f = BatchTranslationJob._meta.get_field("status")
        assert f.db_index is True

    def test_total_inputs_default(self):
        f = BatchTranslationJob._meta.get_field("total_inputs")
        assert f.default == 0


# ---------------------------------------------------------------------------
# BatchTranslationInput model contract
# ---------------------------------------------------------------------------


class TestBatchTranslationInput:
    def test_fk_to_batch_with_related_name(self):
        f = BatchTranslationInput._meta.get_field("batch")
        assert isinstance(f, django_models.ForeignKey)
        assert f.related_model is BatchTranslationJob
        assert f.remote_field.related_name == "inputs"

    def test_client_ref_field_exists(self):
        f = BatchTranslationInput._meta.get_field("client_ref")
        assert isinstance(f, django_models.CharField)

    def test_input_index_field(self):
        f = BatchTranslationInput._meta.get_field("input_index")
        assert isinstance(f, django_models.IntegerField)

    def test_unique_together_batch_client_ref(self):
        constraints = list(BatchTranslationInput._meta.constraints)
        names = {c.name for c in constraints}
        # Either a UniqueConstraint or unique_together expressing
        # (batch, client_ref) uniqueness.
        unique_present = (
            "unique_batch_client_ref" in names
            or any(
                set(getattr(c, "fields", ())) == {"batch", "client_ref"}
                for c in constraints
            )
            or BatchTranslationInput._meta.unique_together
        )
        assert unique_present

    def test_index_on_batch_status(self):
        idx_fields = [
            tuple(idx.fields) for idx in BatchTranslationInput._meta.indexes
        ]
        # Either ("batch","status") or ("batch_id","status") depending on
        # how the migration was written.
        assert any(
            tuple(f) == ("batch", "status")
            or tuple(f) == ("batch_id", "status")
            for f in idx_fields
        )


# ---------------------------------------------------------------------------
# makemigrations --check (no pending model changes)
# ---------------------------------------------------------------------------


class TestMakeMigrationsCheck:
    def test_no_pending_model_changes(self):
        """``makemigrations --check --dry-run`` exits 0 when no changes pending."""
        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["DJANGO_SETTINGS_MODULE"] = "coordinator.settings_test"
        proc = subprocess.run(
            [
                sys.executable,
                "manage.py",
                "makemigrations",
                "jobs",
                "--check",
                "--dry-run",
                "--verbosity",
                "1",
            ],
            cwd=repo_root / "coordinator",
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, (
            "makemigrations --check reports pending changes; the "
            "BatchTranslation migration may not match the model state\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

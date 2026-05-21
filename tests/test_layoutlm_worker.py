"""
Unit tests for LayoutLMv3 Celery worker queue and task integration.

Tests cover:
- Config loading and defaults (layoutlm_config.py)
- Lazy import behavior (no torch required at import time)
- Queue routing logic (ocr_layoutlm isolation)
- Task execution with mocked LayoutLMv3Extractor
- Enable/disable toggle behavior
- Page image rendering helper
- Sidecar output format validation

Run with: python -m pytest tests/test_layoutlm_worker.py -v
"""

import json
import os
import sys
import tempfile
import types
from unittest import mock

import pytest

from layoutlm_model_registry import ResolvedModelSelection

# Ensure coordinator package is importable from root tests.
# When running from repo root, 'coordinator' is a directory but not on sys.path
# as a package.  We add the repo root so `coordinator.jobs.layoutlm_config`
# can resolve without Django being configured.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Check if Django coordinator models can be imported (requires Django setup).
# Tests that need coordinator.jobs.tasks_layoutlm are skipped when Django
# is not properly configured (e.g. running root tests without coordinator setup).
# IMPORTANT: Do NOT set DJANGO_SETTINGS_MODULE here -- that would trigger
# pytest-django's django_test_environment fixture and break all other tests.
_DJANGO_AVAILABLE = False
try:
    # Only attempt if Django is already configured (e.g. coordinator test suite)
    if os.environ.get("DJANGO_SETTINGS_MODULE"):
        import django
        if not django.conf.settings.configured:
            django.setup()
        import coordinator.jobs.tasks_layoutlm  # noqa: F401
        _DJANGO_AVAILABLE = True
except Exception:
    _DJANGO_AVAILABLE = False

_skip_no_django = pytest.mark.skipif(
    not _DJANGO_AVAILABLE,
    reason="Coordinator Django setup not available in this test environment",
)


# ---------------------------------------------------------------------------
# Config module tests
# ---------------------------------------------------------------------------


class TestLayoutLMConfig:
    """Tests for coordinator/jobs/layoutlm_config.py configuration."""

    def _import_config(self, env_overrides=None):
        """Import layoutlm_config with optional env var overrides.

        Uses importlib.util to load the config module directly from its file
        path, bypassing the ``coordinator.jobs`` package which may fail to
        import when Django is not configured.
        """
        import importlib.util
        env = env_overrides or {}
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "coordinator", "jobs", "layoutlm_config.py",
        )
        with mock.patch.dict(os.environ, env, clear=False):
            spec = importlib.util.spec_from_file_location(
                "layoutlm_config", config_path,
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return {
                "ENABLE_LAYOUTLM": mod.ENABLE_LAYOUTLM,
                "LAYOUTLM_ACTIVE_MODEL": mod.LAYOUTLM_ACTIVE_MODEL,
                "LAYOUTLM_MODEL_PATH": mod.LAYOUTLM_MODEL_PATH,
                "LAYOUTLM_REGISTRY_DIR": mod.LAYOUTLM_REGISTRY_DIR,
                "LAYOUTLM_DEVICE": mod.LAYOUTLM_DEVICE,
                "LAYOUTLM_BATCH_SIZE": mod.LAYOUTLM_BATCH_SIZE,
                "LAYOUTLM_CONFIDENCE_THRESHOLD": mod.LAYOUTLM_CONFIDENCE_THRESHOLD,
                "LAYOUTLM_MAX_LENGTH": mod.LAYOUTLM_MAX_LENGTH,
                "LAYOUTLM_TASK_TIMEOUT": mod.LAYOUTLM_TASK_TIMEOUT,
                "LAYOUTLM_QUEUE": mod.LAYOUTLM_QUEUE,
                "LAYOUTLM_ENTITY_LABELS": mod.LAYOUTLM_ENTITY_LABELS,
                "LAYOUTLM_ID2LABEL": mod.LAYOUTLM_ID2LABEL,
                "LAYOUTLM_LABEL2ID": mod.LAYOUTLM_LABEL2ID,
            }

    def test_defaults(self):
        """Config defaults are sensible when no env vars are set."""
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in list(os.environ):
                if k.startswith("LAYOUTLM_") or k == "ENABLE_LAYOUTLM":
                    os.environ.pop(k, None)
            cfg = self._import_config()

        assert cfg["ENABLE_LAYOUTLM"] is False
        assert cfg["LAYOUTLM_ACTIVE_MODEL"] == ""
        assert cfg["LAYOUTLM_MODEL_PATH"] == "microsoft/layoutlmv3-base"
        assert cfg["LAYOUTLM_REGISTRY_DIR"] == "./models/registry"
        assert cfg["LAYOUTLM_DEVICE"] == "auto"
        assert cfg["LAYOUTLM_BATCH_SIZE"] == 1
        assert cfg["LAYOUTLM_CONFIDENCE_THRESHOLD"] == 0.5
        assert cfg["LAYOUTLM_MAX_LENGTH"] == 512
        assert cfg["LAYOUTLM_TASK_TIMEOUT"] == 120
        assert cfg["LAYOUTLM_QUEUE"] == "ocr_layoutlm"

    def test_enable_toggle_true(self):
        """ENABLE_LAYOUTLM=true activates the feature."""
        cfg = self._import_config({"ENABLE_LAYOUTLM": "true"})
        assert cfg["ENABLE_LAYOUTLM"] is True

    def test_enable_toggle_one(self):
        """ENABLE_LAYOUTLM=1 activates the feature."""
        cfg = self._import_config({"ENABLE_LAYOUTLM": "1"})
        assert cfg["ENABLE_LAYOUTLM"] is True

    def test_enable_toggle_yes(self):
        """ENABLE_LAYOUTLM=yes activates the feature."""
        cfg = self._import_config({"ENABLE_LAYOUTLM": "yes"})
        assert cfg["ENABLE_LAYOUTLM"] is True

    def test_custom_model_path(self):
        """Custom model path is respected."""
        cfg = self._import_config({"LAYOUTLM_MODEL_PATH": "/models/custom-lmv3"})
        assert cfg["LAYOUTLM_MODEL_PATH"] == "/models/custom-lmv3"

    def test_active_model_and_registry_dir(self):
        """Registry-related env vars are surfaced in config."""
        cfg = self._import_config({
            "LAYOUTLM_ACTIVE_MODEL": "forensic:1.0.0",
            "LAYOUTLM_REGISTRY_DIR": "/models/registry",
        })
        assert cfg["LAYOUTLM_ACTIVE_MODEL"] == "forensic:1.0.0"
        assert cfg["LAYOUTLM_REGISTRY_DIR"] == "/models/registry"

    def test_custom_device(self):
        """Custom device is respected."""
        cfg = self._import_config({"LAYOUTLM_DEVICE": "cuda:1"})
        assert cfg["LAYOUTLM_DEVICE"] == "cuda:1"

    def test_batch_size_minimum(self):
        """Batch size has a minimum of 1."""
        cfg = self._import_config({"LAYOUTLM_BATCH_SIZE": "0"})
        assert cfg["LAYOUTLM_BATCH_SIZE"] == 1

    def test_max_length_capped(self):
        """Max length is capped at 512."""
        cfg = self._import_config({"LAYOUTLM_MAX_LENGTH": "1024"})
        assert cfg["LAYOUTLM_MAX_LENGTH"] == 512

    def test_timeout_minimum(self):
        """Task timeout has a minimum of 10 seconds."""
        cfg = self._import_config({"LAYOUTLM_TASK_TIMEOUT": "5"})
        assert cfg["LAYOUTLM_TASK_TIMEOUT"] == 10

    def test_entity_labels_bio_format(self):
        """Entity labels follow BIO tagging scheme."""
        cfg = self._import_config()
        labels = cfg["LAYOUTLM_ENTITY_LABELS"]
        assert labels[0] == "O"
        # Check BIO pairs exist
        b_labels = [lbl for lbl in labels if lbl.startswith("B-")]
        i_labels = [lbl for lbl in labels if lbl.startswith("I-")]
        assert len(b_labels) == len(i_labels)
        for b_label in b_labels:
            entity_type = b_label[2:]
            assert f"I-{entity_type}" in labels

    def test_label_id_mappings_bijective(self):
        """Label-to-ID and ID-to-label mappings are consistent."""
        cfg = self._import_config()
        for label, idx in cfg["LAYOUTLM_LABEL2ID"].items():
            assert cfg["LAYOUTLM_ID2LABEL"][idx] == label


# ---------------------------------------------------------------------------
# Task module import tests (no Django/Celery required)
# ---------------------------------------------------------------------------


class TestTaskModuleImport:
    """Test that the tasks_layoutlm module handles imports gracefully."""

    def test_config_imports_without_torch(self):
        """Config module can be imported without torch/transformers."""
        import importlib.util
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "coordinator", "jobs", "layoutlm_config.py",
        )
        spec = importlib.util.spec_from_file_location("layoutlm_config", config_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.LAYOUTLM_QUEUE == "ocr_layoutlm"

    def test_config_entity_labels_match_semantic_extraction(self):
        """Config entity labels match the canonical list in semantic_extraction."""
        import importlib.util
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "coordinator", "jobs", "layoutlm_config.py",
        )
        spec = importlib.util.spec_from_file_location("layoutlm_config", config_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from semantic_extraction import SEMANTIC_ENTITY_LABELS
        assert mod.LAYOUTLM_ENTITY_LABELS == SEMANTIC_ENTITY_LABELS


# ---------------------------------------------------------------------------
# Queue routing tests (tested via _route_task function, imported lazily)
# ---------------------------------------------------------------------------


def _get_route_task():
    """Import _route_task from celery config.

    This triggers DJANGO_SETTINGS_MODULE but avoids issues with
    pytest-django since we call it inline, not at module level.
    """
    from coordinator.coordinator.celery import _route_task
    return _route_task


@_skip_no_django
class TestLayoutLMQueueRouting:
    """Test that Celery task routing correctly maps LayoutLMv3 tasks.

    These tests exercise the _route_task callable router directly,
    without requiring a running Celery broker.
    """

    def test_static_route_in_celery_config(self):
        """The LayoutLMv3 task is in the static route table."""
        route_task = _get_route_task()
        result = route_task(
            "jobs.tasks_layoutlm.run_layoutlm_extraction",
            args=(), kwargs={}, options={},
        )
        assert result == {"queue": "ocr_layoutlm"}

    def test_layoutlm_task_not_routed_to_ocr_gpu(self):
        """LayoutLMv3 tasks never route to ocr_gpu."""
        route_task = _get_route_task()
        result = route_task(
            "jobs.tasks_layoutlm.run_layoutlm_extraction",
            args=(), kwargs={}, options={},
        )
        assert result["queue"] != "ocr_gpu"

    def test_layoutlm_task_not_routed_to_ocr_cpu(self):
        """LayoutLMv3 tasks never route to ocr_cpu."""
        route_task = _get_route_task()
        result = route_task(
            "jobs.tasks_layoutlm.run_layoutlm_extraction",
            args=(), kwargs={}, options={},
        )
        assert result["queue"] != "ocr_cpu"

    def test_layoutlm_task_not_routed_to_nlp_general(self):
        """LayoutLMv3 tasks never route to nlp_general."""
        route_task = _get_route_task()
        result = route_task(
            "jobs.tasks_layoutlm.run_layoutlm_extraction",
            args=(), kwargs={}, options={},
        )
        assert result["queue"] != "nlp_general"

    def test_gpu_tasks_unaffected(self):
        """GPU task routing is unaffected by LayoutLMv3 addition."""
        route_task = _get_route_task()
        result = route_task(
            "jobs.tasks.process_page",
            args=(), kwargs={}, options={},
        )
        assert result["queue"] in (
            "ocr_gpu", "ocr_cpu",
        ) or result["queue"].startswith("ocr_gpu_")

    def test_coordinator_tasks_unaffected(self):
        """Coordinator task routing is unaffected by LayoutLMv3 addition."""
        route_task = _get_route_task()
        result = route_task(
            "jobs.tasks.ingest_document",
            args=(), kwargs={}, options={},
        )
        assert result == {"queue": "coordinator"}

    def test_layoutlm_queue_is_distinct(self):
        """The ocr_layoutlm queue name is distinct from all other queues."""
        route_task = _get_route_task()

        all_known_task_names = [
            "jobs.tasks.ingest_document",
            "jobs.tasks.assemble_document",
            "jobs.tasks.finalize_job",
            "jobs.tasks.process_document",
            "jobs.tasks.process_page",
            "jobs.tasks.compress_pdf",
            "jobs.tasks.extract_entities",
            "jobs.tasks.extract_structured_data",
            "jobs.tasks.process_text_only",
        ]

        other_queues = set()
        for task_name in all_known_task_names:
            result = route_task(task_name, args=(), kwargs={}, options={})
            if result:
                other_queues.add(result["queue"])

        layoutlm_result = route_task(
            "jobs.tasks_layoutlm.run_layoutlm_extraction",
            args=(), kwargs={}, options={},
        )
        assert layoutlm_result["queue"] not in other_queues


# ---------------------------------------------------------------------------
# Task execution tests (direct function call, mocked Django/models)
# ---------------------------------------------------------------------------


@_skip_no_django
class TestRunLayoutLMExtractionDirect:
    """Test the run_layoutlm_extraction task logic directly.

    Instead of calling task.apply() (which requires Celery backend),
    we import and call the underlying function with a mocked self.
    """

    def _make_mock_self(self):
        """Create a mock task self object."""
        mock_self = mock.MagicMock()
        mock_self.request = mock.MagicMock()
        mock_self.request.id = "test-celery-id-123"
        return mock_self

    def _make_mock_job(self, job_id="test-job-123", source_file="doc.pdf",
                       source_type="pdf"):
        """Create a mock Job object."""
        job = mock.MagicMock()
        job.job_id = job_id
        job.source_file = source_file
        job.source_type = source_type
        job.source_hash = "abcdef1234567890" * 4
        job.nfs_job_path = None
        job.storage_backend_used = "nfs"
        job.status = "processing"
        job.Status = mock.MagicMock()
        job.Status.CANCELLED = "cancelled"
        job.settings_json = {}
        return job

    def _import_task_function(self):
        """Import the task module and return the function + module."""
        import coordinator.jobs.tasks_layoutlm as task_module
        return task_module

    def test_skipped_when_disabled(self):
        """Task returns skipped when ENABLE_LAYOUTLM is False."""
        task_mod = self._import_task_function()
        mock_self = self._make_mock_self()

        with mock.patch.object(task_mod, "ENABLE_LAYOUTLM", False):
            result = task_mod.run_layoutlm_extraction.__wrapped__(
                mock_self, "test-job-123", 1,
            )

        assert result["status"] == "skipped"
        assert result["reason"] == "layoutlm_disabled"

    def test_error_when_job_not_found(self):
        """Task returns error when job does not exist."""
        task_mod = self._import_task_function()
        mock_self = self._make_mock_self()

        mock_job_cls = mock.MagicMock()
        mock_job_cls.DoesNotExist = type("DoesNotExist", (Exception,), {})
        mock_job_cls.objects.get.side_effect = mock_job_cls.DoesNotExist("not found")

        with mock.patch.object(task_mod, "ENABLE_LAYOUTLM", True), \
             mock.patch.object(task_mod, "Job", mock_job_cls):
            result = task_mod.run_layoutlm_extraction.__wrapped__(
                mock_self, "nonexistent-job", 1,
            )

        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_skipped_when_no_text(self):
        """Task returns skipped when page has no OCR text."""
        task_mod = self._import_task_function()
        mock_self = self._make_mock_self()
        mock_job = self._make_mock_job()

        with tempfile.TemporaryDirectory() as tmpdir:
            job_path = os.path.join(tmpdir, "jobs", "test-job-123")
            doc_id = mock_job.source_hash[:16]
            temp_dir = os.path.join(job_path, "temp", doc_id)
            source_dir = os.path.join(job_path, "source")
            entities_dir = os.path.join(job_path, "output", "EXPORT", "ENTITIES")
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(source_dir, exist_ok=True)
            os.makedirs(entities_dir, exist_ok=True)

            # Write empty text file
            text_path = os.path.join(temp_dir, "1.txt")
            with open(text_path, "w") as f:
                f.write("")

            mock_job.nfs_job_path = job_path

            with mock.patch.object(task_mod, "ENABLE_LAYOUTLM", True), \
                 mock.patch.object(task_mod, "Job") as mock_job_cls, \
                 mock.patch.object(task_mod, "_get_backend_for_job",
                                   return_value=mock.MagicMock(backend_name="nfs")), \
                 mock.patch.object(task_mod, "Worker"):
                mock_job_cls.objects.get.return_value = mock_job

                result = task_mod.run_layoutlm_extraction.__wrapped__(
                    mock_self, "test-job-123", 1,
                )

        assert result["status"] == "skipped"
        assert result["reason"] == "no_text"

    def test_successful_extraction_with_mocked_extractor(self):
        """Task completes successfully with mocked LayoutLMv3 extractor."""
        task_mod = self._import_task_function()
        mock_self = self._make_mock_self()
        mock_job = self._make_mock_job()

        # Create mock entity
        mock_entity = mock.MagicMock()
        mock_entity.text = "Invoice #12345"
        mock_entity.label = "INVOICE_NUMBER"
        mock_entity.field_type = "reference_number"
        mock_entity.confidence = 0.95
        mock_entity.bbox = [100, 200, 300, 250]
        mock_entity.page_num = 1

        mock_extractor_class = mock.MagicMock()
        mock_extractor_instance = mock.MagicMock()
        mock_extractor_instance.extract_entities.return_value = [mock_entity]
        mock_extractor_class.return_value = mock_extractor_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            job_path = os.path.join(tmpdir, "jobs", "test-job-123")
            doc_id = mock_job.source_hash[:16]
            temp_dir = os.path.join(job_path, "temp", doc_id)
            source_dir = os.path.join(job_path, "source")
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(source_dir, exist_ok=True)

            text_path = os.path.join(temp_dir, "1.txt")
            with open(text_path, "w") as f:
                f.write("Invoice #12345 dated 2025-01-15 for $1,234.56")

            source_path = os.path.join(source_dir, "doc.pdf")
            with open(source_path, "wb") as f:
                f.write(b"%PDF-1.4 dummy")

            mock_job.nfs_job_path = job_path
            mock_image = mock.MagicMock()
            mock_image.size = (612, 792)

            with mock.patch.object(task_mod, "ENABLE_LAYOUTLM", True), \
                 mock.patch.object(task_mod, "Job") as mock_job_cls, \
                 mock.patch.object(task_mod, "_get_backend_for_job",
                                   return_value=mock.MagicMock(backend_name="nfs")), \
                 mock.patch.object(task_mod, "_render_page_image",
                                   return_value=mock_image), \
                 mock.patch.object(task_mod, "Worker"), \
                 mock.patch.object(task_mod, "CustodyEvent"), \
                 mock.patch.object(task_mod, "LAYOUTLM_CONFIDENCE_THRESHOLD", 0.5), \
                 mock.patch.dict("sys.modules",
                                 {"semantic_extraction": types.ModuleType("semantic_extraction")}):
                mock_job_cls.objects.get.return_value = mock_job
                sys.modules["semantic_extraction"].LayoutLMv3Extractor = mock_extractor_class

                result = task_mod.run_layoutlm_extraction.__wrapped__(
                    mock_self, "test-job-123", 1,
                )

            assert result["status"] == "completed"
            assert result["entity_count"] == 1
            assert result["page_number"] == 1
            assert "processing_time_seconds" in result

            # Verify sidecar JSON was written
            entities_files = []
            for root, _dirs, files in os.walk(job_path):
                for fname in files:
                    if fname.endswith(".entities.json"):
                        entities_files.append(os.path.join(root, fname))

            assert len(entities_files) == 1
            with open(entities_files[0], "r") as f:
                sidecar = json.load(f)

            assert sidecar["source"] == "layoutlmv3"
            assert sidecar["entity_count"] == 1
            assert sidecar["entities"][0]["text"] == "Invoice #12345"
            assert sidecar["entities"][0]["label"] == "INVOICE_NUMBER"

    def test_uses_active_registry_model_when_configured(self):
        """Worker prefers the resolved active registry model."""
        task_mod = self._import_task_function()
        mock_self = self._make_mock_self()
        mock_job = self._make_mock_job()

        mock_entity = mock.MagicMock()
        mock_entity.text = "Invoice #12345"
        mock_entity.label = "INVOICE_NUMBER"
        mock_entity.field_type = "reference_number"
        mock_entity.confidence = 0.95
        mock_entity.bbox = [100, 200, 300, 250]
        mock_entity.page_num = 1

        mock_extractor_class = mock.MagicMock()
        mock_extractor_instance = mock.MagicMock()
        mock_extractor_instance.extract_entities.return_value = [mock_entity]
        mock_extractor_class.return_value = mock_extractor_instance

        selection = ResolvedModelSelection(
            model_path="/models/registry/forensic-1.0.0",
            source="registry",
            active_model_spec="forensic:1.0.0",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            job_path = os.path.join(tmpdir, "jobs", "test-job-123")
            doc_id = mock_job.source_hash[:16]
            temp_dir = os.path.join(job_path, "temp", doc_id)
            source_dir = os.path.join(job_path, "source")
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(source_dir, exist_ok=True)

            text_path = os.path.join(temp_dir, "1.txt")
            with open(text_path, "w") as f:
                f.write("Invoice #12345 dated 2025-01-15 for $1,234.56")

            source_path = os.path.join(source_dir, "doc.pdf")
            with open(source_path, "wb") as f:
                f.write(b"%PDF-1.4 dummy")

            mock_job.nfs_job_path = job_path
            mock_image = mock.MagicMock()
            mock_image.size = (612, 792)

            with mock.patch.object(task_mod, "ENABLE_LAYOUTLM", True), \
                 mock.patch.object(task_mod, "Job") as mock_job_cls, \
                 mock.patch.object(task_mod, "_get_backend_for_job",
                                   return_value=mock.MagicMock(backend_name="nfs")), \
                 mock.patch.object(task_mod, "_render_page_image",
                                   return_value=mock_image), \
                 mock.patch.object(task_mod, "Worker"), \
                 mock.patch.object(task_mod, "CustodyEvent"), \
                 mock.patch.object(task_mod, "LAYOUTLM_CONFIDENCE_THRESHOLD", 0.5), \
                 mock.patch.object(
                     task_mod,
                     "resolve_active_model_selection",
                     return_value=selection,
                 ), \
                 mock.patch.dict("sys.modules",
                                 {"semantic_extraction": types.ModuleType("semantic_extraction")}):
                mock_job_cls.objects.get.return_value = mock_job
                sys.modules["semantic_extraction"].LayoutLMv3Extractor = mock_extractor_class

                result = task_mod.run_layoutlm_extraction.__wrapped__(
                    mock_self, "test-job-123", 1,
                )

            assert result["status"] == "completed"
            assert result["model"] == "/models/registry/forensic-1.0.0"
            assert result["model_source"] == "registry"
            assert result["active_model_spec"] == "forensic:1.0.0"
            mock_extractor_class.assert_called_once_with(
                model_path="/models/registry/forensic-1.0.0",
                device=None,
            )

            entities_files = []
            for root, _dirs, files in os.walk(job_path):
                for fname in files:
                    if fname.endswith(".entities.json"):
                        entities_files.append(os.path.join(root, fname))

            assert len(entities_files) == 1
            with open(entities_files[0], "r") as f:
                sidecar = json.load(f)
            assert sidecar["model"] == "/models/registry/forensic-1.0.0"
            assert sidecar["model_source"] == "registry"
            assert sidecar["active_model_spec"] == "forensic:1.0.0"

    def test_falls_back_when_active_model_is_unresolved(self):
        """Worker falls back to the configured env model path."""
        task_mod = self._import_task_function()
        mock_self = self._make_mock_self()
        mock_job = self._make_mock_job()

        mock_extractor_class = mock.MagicMock()
        mock_extractor_instance = mock.MagicMock()
        mock_extractor_instance.extract_entities.return_value = []
        mock_extractor_class.return_value = mock_extractor_instance

        selection = ResolvedModelSelection(
            model_path="/models/fallback-layoutlm",
            source="fallback",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            job_path = os.path.join(tmpdir, "jobs", "test-job-123")
            doc_id = mock_job.source_hash[:16]
            temp_dir = os.path.join(job_path, "temp", doc_id)
            source_dir = os.path.join(job_path, "source")
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(source_dir, exist_ok=True)

            text_path = os.path.join(temp_dir, "1.txt")
            with open(text_path, "w") as f:
                f.write("Some text here")

            source_path = os.path.join(source_dir, "doc.pdf")
            with open(source_path, "wb") as f:
                f.write(b"%PDF-1.4 dummy")

            mock_job.nfs_job_path = job_path
            mock_image = mock.MagicMock()
            mock_image.size = (612, 792)

            with mock.patch.object(task_mod, "ENABLE_LAYOUTLM", True), \
                 mock.patch.object(task_mod, "Job") as mock_job_cls, \
                 mock.patch.object(task_mod, "_get_backend_for_job",
                                   return_value=mock.MagicMock(backend_name="nfs")), \
                 mock.patch.object(task_mod, "_render_page_image",
                                   return_value=mock_image), \
                 mock.patch.object(task_mod, "Worker"), \
                 mock.patch.object(task_mod, "CustodyEvent"), \
                 mock.patch.object(
                     task_mod,
                     "resolve_active_model_selection",
                     return_value=selection,
                 ), \
                 mock.patch.dict("sys.modules",
                                 {"semantic_extraction": types.ModuleType("semantic_extraction")}):
                mock_job_cls.objects.get.return_value = mock_job
                sys.modules["semantic_extraction"].LayoutLMv3Extractor = mock_extractor_class

                result = task_mod.run_layoutlm_extraction.__wrapped__(
                    mock_self, "test-job-123", 1,
                )

            assert result["status"] == "completed"
            assert result["model"] == "/models/fallback-layoutlm"
            assert result["model_source"] == "fallback"
            assert result["active_model_spec"] == ""
            mock_extractor_class.assert_called_once_with(
                model_path="/models/fallback-layoutlm",
                device=None,
            )

    def test_module_unavailable_returns_skipped(self):
        """Task returns skipped when semantic_extraction is not importable."""
        task_mod = self._import_task_function()
        mock_self = self._make_mock_self()
        mock_job = self._make_mock_job()

        with tempfile.TemporaryDirectory() as tmpdir:
            job_path = os.path.join(tmpdir, "jobs", "test-job-123")
            doc_id = mock_job.source_hash[:16]
            temp_dir = os.path.join(job_path, "temp", doc_id)
            source_dir = os.path.join(job_path, "source")
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(source_dir, exist_ok=True)

            text_path = os.path.join(temp_dir, "1.txt")
            with open(text_path, "w") as f:
                f.write("Some text here")

            source_path = os.path.join(source_dir, "doc.pdf")
            with open(source_path, "wb") as f:
                f.write(b"%PDF-1.4 dummy")

            mock_job.nfs_job_path = job_path
            mock_image = mock.MagicMock()
            mock_image.size = (612, 792)

            with mock.patch.object(task_mod, "ENABLE_LAYOUTLM", True), \
                 mock.patch.object(task_mod, "Job") as mock_job_cls, \
                 mock.patch.object(task_mod, "_get_backend_for_job",
                                   return_value=mock.MagicMock(backend_name="nfs")), \
                 mock.patch.object(task_mod, "_render_page_image",
                                   return_value=mock_image), \
                 mock.patch.object(task_mod, "Worker"), \
                 mock.patch("builtins.__import__",
                            side_effect=_import_blocker("semantic_extraction")):
                mock_job_cls.objects.get.return_value = mock_job

                result = task_mod.run_layoutlm_extraction.__wrapped__(
                    mock_self, "test-job-123", 1,
                )

        assert result["status"] == "skipped"
        assert result["reason"] == "module_unavailable"

    def test_confidence_filtering(self):
        """Entities below confidence threshold are filtered out."""
        task_mod = self._import_task_function()
        mock_self = self._make_mock_self()
        mock_job = self._make_mock_job()

        high_conf_entity = mock.MagicMock()
        high_conf_entity.text = "High"
        high_conf_entity.label = "DATE"
        high_conf_entity.field_type = "date"
        high_conf_entity.confidence = 0.9
        high_conf_entity.bbox = [10, 20, 30, 40]
        high_conf_entity.page_num = 1

        low_conf_entity = mock.MagicMock()
        low_conf_entity.text = "Low"
        low_conf_entity.label = "AMOUNT"
        low_conf_entity.field_type = "amount"
        low_conf_entity.confidence = 0.2
        low_conf_entity.bbox = [50, 60, 70, 80]
        low_conf_entity.page_num = 1

        mock_extractor_class = mock.MagicMock()
        mock_extractor_instance = mock.MagicMock()
        mock_extractor_instance.extract_entities.return_value = [
            high_conf_entity, low_conf_entity,
        ]
        mock_extractor_class.return_value = mock_extractor_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            job_path = os.path.join(tmpdir, "jobs", "test-job-123")
            doc_id = mock_job.source_hash[:16]
            temp_dir = os.path.join(job_path, "temp", doc_id)
            source_dir = os.path.join(job_path, "source")
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(source_dir, exist_ok=True)

            text_path = os.path.join(temp_dir, "1.txt")
            with open(text_path, "w") as f:
                f.write("Some text with date and amount")

            source_path = os.path.join(source_dir, "doc.pdf")
            with open(source_path, "wb") as f:
                f.write(b"%PDF-1.4 dummy")

            mock_job.nfs_job_path = job_path
            mock_image = mock.MagicMock()
            mock_image.size = (612, 792)

            with mock.patch.object(task_mod, "ENABLE_LAYOUTLM", True), \
                 mock.patch.object(task_mod, "Job") as mock_job_cls, \
                 mock.patch.object(task_mod, "_get_backend_for_job",
                                   return_value=mock.MagicMock(backend_name="nfs")), \
                 mock.patch.object(task_mod, "_render_page_image",
                                   return_value=mock_image), \
                 mock.patch.object(task_mod, "Worker"), \
                 mock.patch.object(task_mod, "CustodyEvent"), \
                 mock.patch.object(task_mod, "LAYOUTLM_CONFIDENCE_THRESHOLD", 0.5), \
                 mock.patch.dict("sys.modules",
                                 {"semantic_extraction": types.ModuleType("semantic_extraction")}):
                mock_job_cls.objects.get.return_value = mock_job
                sys.modules["semantic_extraction"].LayoutLMv3Extractor = mock_extractor_class

                result = task_mod.run_layoutlm_extraction.__wrapped__(
                    mock_self, "test-job-123", 1,
                )

            assert result["entity_count"] == 1

            entities_files = []
            for root, _dirs, files in os.walk(job_path):
                for fname in files:
                    if fname.endswith(".entities.json"):
                        entities_files.append(os.path.join(root, fname))

            assert len(entities_files) == 1
            with open(entities_files[0], "r") as f:
                sidecar = json.load(f)
            assert len(sidecar["entities"]) == 1
            assert sidecar["entities"][0]["text"] == "High"

    def test_cancelled_job_returns_cancelled(self):
        """Task returns cancelled for cancelled jobs."""
        task_mod = self._import_task_function()
        mock_self = self._make_mock_self()
        mock_job = self._make_mock_job()
        mock_job.status = "cancelled"

        # Make job.status == job.Status.CANCELLED return True
        mock_job.Status.CANCELLED = "cancelled"

        with mock.patch.object(task_mod, "ENABLE_LAYOUTLM", True), \
             mock.patch.object(task_mod, "Job") as mock_job_cls, \
             mock.patch.object(task_mod, "Worker"):
            mock_job_cls.objects.get.return_value = mock_job

            result = task_mod.run_layoutlm_extraction.__wrapped__(
                mock_self, "test-job-123", 1,
            )

        assert result["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Page image rendering tests
# ---------------------------------------------------------------------------


@_skip_no_django
class TestRenderPageImage:
    """Test the _render_page_image helper."""

    def _get_render_func(self):
        from coordinator.jobs.tasks_layoutlm import _render_page_image
        return _render_page_image

    def test_returns_none_when_pillow_missing(self):
        """Returns None when Pillow is not importable."""
        render = self._get_render_func()
        with mock.patch(
            "builtins.__import__",
            side_effect=_import_blocker("PIL"),
        ):
            result = render("/fake/path.pdf", 1, "pdf")
        assert result is None

    def test_returns_none_for_missing_file(self):
        """Returns None for a non-existent source file."""
        render = self._get_render_func()
        result = render("/nonexistent/path.pdf", 1, "pdf")
        assert result is None

    def test_returns_image_for_image_source(self):
        """Returns PIL Image for image source type."""
        render = self._get_render_func()

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not available")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (100, 100), (255, 255, 255))
            img.save(f.name)
            tmp_path = f.name

        try:
            result = render(tmp_path, 1, "image")
            assert result is not None
            assert result.size == (100, 100)
        finally:
            os.unlink(tmp_path)

    def test_returns_none_for_out_of_range_page(self):
        """Returns None when page number is out of range."""
        render = self._get_render_func()

        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not available")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            doc = fitz.open()
            doc.new_page()
            doc.save(f.name)
            doc.close()
            tmp_path = f.name

        try:
            # Page 999 should be out of range for a 1-page PDF
            result = render(tmp_path, 999, "pdf")
            assert result is None
        finally:
            os.unlink(tmp_path)

    def test_renders_pdf_page(self):
        """Successfully renders a PDF page to an image."""
        render = self._get_render_func()

        try:
            import fitz
            from PIL import Image  # noqa: F401
        except ImportError:
            pytest.skip("PyMuPDF or Pillow not available")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            doc = fitz.open()
            doc.new_page(width=612, height=792)
            doc.save(f.name)
            doc.close()
            tmp_path = f.name

        try:
            result = render(tmp_path, 1, "pdf")
            assert result is not None
            assert result.size[0] > 0
            assert result.size[1] > 0
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Queue isolation tests
# ---------------------------------------------------------------------------


@_skip_no_django
class TestQueueIsolation:
    """Verify LayoutLMv3 tasks are isolated from other queues."""

    def test_ocr_gpu_route_unchanged(self):
        """Standard OCR GPU routing is not affected."""
        route_task = _get_route_task()
        result = route_task(
            "jobs.tasks.process_document",
            args=(), kwargs={}, options={},
        )
        assert "layoutlm" not in result["queue"]

    def test_cpu_general_route_unchanged(self):
        """CPU general routing is not affected."""
        route_task = _get_route_task()
        result = route_task(
            "jobs.tasks.compress_pdf",
            args=(), kwargs={}, options={},
        )
        assert result == {"queue": "cpu_general"}

    def test_nlp_general_route_unchanged(self):
        """NLP general routing is not affected."""
        route_task = _get_route_task()
        result = route_task(
            "jobs.tasks.extract_structured_data",
            args=(), kwargs={}, options={},
        )
        assert result == {"queue": "nlp_general"}

    def test_unknown_tasks_return_none(self):
        """Unknown task names still return None from router."""
        route_task = _get_route_task()
        result = route_task(
            "some.unknown.task",
            args=(), kwargs={}, options={},
        )
        assert result is None


# ---------------------------------------------------------------------------
# Sidecar output format tests
# ---------------------------------------------------------------------------


class TestSidecarOutputFormat:
    """Validate the .entities.json sidecar schema."""

    def test_sidecar_schema_fields(self):
        """Sidecar JSON has all required top-level fields."""
        required_fields = [
            "schema_version",
            "source",
            "model",
            "job_id",
            "page_number",
            "document_id",
            "processing_time_seconds",
            "confidence_threshold",
            "entity_count",
            "entities",
        ]

        sidecar = {
            "schema_version": "1.0",
            "source": "layoutlmv3",
            "model": "microsoft/layoutlmv3-base",
            "job_id": "test-123",
            "page_number": 1,
            "document_id": "abc123",
            "processing_time_seconds": 0.5,
            "confidence_threshold": 0.5,
            "entity_count": 0,
            "entities": [],
        }

        for field in required_fields:
            assert field in sidecar, f"Missing field: {field}"

    def test_entity_dict_fields(self):
        """Each entity in sidecar has required fields."""
        entity = {
            "text": "Invoice #12345",
            "label": "INVOICE_NUMBER",
            "field_type": "reference_number",
            "confidence": 0.95,
            "bbox": [100, 200, 300, 250],
            "page_num": 1,
        }

        required = ["text", "label", "field_type", "confidence", "bbox", "page_num"]
        for field in required:
            assert field in entity, f"Missing entity field: {field}"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _import_blocker(blocked_module):
    """Return a side_effect function that blocks import of a specific module."""
    _real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def _blocked_import(name, *args, **kwargs):
        if name == blocked_module or name.startswith(f"{blocked_module}."):
            raise ImportError(f"Mocked: {name} is not available")
        return _real_import(name, *args, **kwargs)

    return _blocked_import

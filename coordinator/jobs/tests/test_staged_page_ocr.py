"""Durable staged OCR checks using a real rendered PDF and stub inference."""

import hashlib
import os
import tempfile
import types
from contextlib import nullcontext
from unittest.mock import patch

import fitz
from django.test import TestCase, override_settings
from django.utils import timezone

from jobs.models import (
    Job,
    PageOCRByteBudget,
    PageOCRRasterRetirement,
    PageOCRStage,
    PageResult,
    TaskDispatchOutbox,
)
from jobs.storage import NFSBackend
from jobs.tasks import (
    _ensure_job_dirs,
    _ProcessingLeaseLost,
    _stage_cleanup_artifact,
    _stage_reserve,
    _sweep_retired_stage_rasters,
    finalize_page,
    infer_page,
    render_page,
)
from ocr_local.visual_coverage import CoverageAssessment


class StagedPageTests(TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = self.temporary.name
        self.job_path = os.path.join(self.root, "job")
        _ensure_job_dirs(self.job_path)
        source = os.path.join(self.job_path, "source", "input.pdf")
        with fitz.open() as document:
            document.new_page(width=595, height=842)
            document.save(source)
        with open(source, "rb") as handle:
            source_hash = hashlib.sha256(handle.read()).hexdigest()
        self.job = Job.objects.create(
            source_file="input.pdf", source_hash=source_hash, source_type="pdf",
            status=Job.Status.PROCESSING, total_pages=1,
            nfs_job_path=self.job_path, dispatch_generation=1,
        )
        self.stage = PageOCRStage.objects.create(
            job=self.job, page_num=1, generation=1,
            tenant_id="", source_sha256=source_hash,
        )
        PageResult.objects.create(
            job=self.job, page_num=1, document_id=source_hash[:16], status="pending",
            admitted_generation=1,
        )
        self.backend = NFSBackend(root=self.root)

    def test_render_infer_finalize_uses_ids_and_releases_budget(self):
        with (
            override_settings(NFS_ROOT=self.root),
            patch("jobs.tasks._get_backend_for_job", return_value=self.backend),
            patch("jobs.tasks._page_script_evidence", return_value=()),
            patch("jobs.tasks._page_text_language", return_value=None),
            patch("jobs.tasks._run_paddle_ocr", return_value=object()),
            patch("ocr_distributed.ocr_utils.extract_paddle_lines", return_value=[
                ("Evidence", [400, 500, 800, 570], 0.9)
            ]),
            patch("jobs.tasks.assess_candidate_coverage", return_value=CoverageAssessment(
                "measured", 1.0, 1.0, 1.0, 1.0, (), False, False, "supported"
            )),
            patch("jobs.tasks._gpu_available", return_value=False),
        ):
            assert render_page.run(str(self.job.pk), 1, 1)["status"] == "rendered"
            self.stage.refresh_from_db()
            assert self.stage.raster_sha256
            assert PageOCRByteBudget.objects.get(pk=1).reserved_bytes > 0
            assert infer_page.run(str(self.job.pk), 1, 1)["status"] == "inferred"
            dispatch = TaskDispatchOutbox.objects.get(
                job=self.job, stage__startswith="finalize_page:1:retry:render:"
            )
            assert dispatch.args == [str(self.job.pk), 1, 1]
            assert dispatch.queue == "ocr_finalize"
            assert finalize_page.run(str(self.job.pk), 1, 1)["status"] == "page_processed"

        self.stage.refresh_from_db()
        row = PageResult.objects.get(job=self.job, page_num=1)
        assert self.stage.status == "done"
        assert self.stage.reserved_bytes == 0
        assert PageOCRByteBudget.objects.get(pk=1).reserved_bytes == 0
        assert row.status == "ok"
        with fitz.open(row.temp_pdf_path) as output:
            assert abs(output[0].rect.width - 595) < 0.5
            assert output[0].search_for("Evidence")

    def test_cpu_script_evidence_hands_unsupported_route_to_tesseract_lane(self):
        with (
            override_settings(NFS_ROOT=self.root),
            patch("jobs.tasks._get_backend_for_job", return_value=self.backend),
            patch("jobs.tasks._page_script_evidence", return_value=("greek",)) as probe,
        ):
            assert render_page.run(str(self.job.pk), 1, 1)["status"] == "rendered"
            assert probe.call_count == 1
        self.stage.refresh_from_db()
        assert self.stage.script_evidence == ["greek"]
        with (
            override_settings(NFS_ROOT=self.root),
            patch("jobs.tasks._get_backend_for_job", return_value=self.backend),
            patch.dict(os.environ, {"OCR_WORKER_CAPABILITY": "paddle"}),
            patch("jobs.tasks._page_script_evidence", side_effect=AssertionError("GPU OSD call")),
            patch("jobs.tasks._run_paddle_ocr", side_effect=AssertionError("unsupported Paddle model")),
        ):
            assert infer_page.run(str(self.job.pk), 1, 1)["status"] == "tesseract_handoff"
        self.stage.refresh_from_db()
        assert self.stage.status == "rendered"
        assert self.stage.last_error == "cross_engine_pending"
        dispatch = TaskDispatchOutbox.objects.filter(
            job=self.job, queue="ocr_tesseract", stage__startswith="infer_page:1:retry:fallback:"
        ).get()
        assert dispatch.args == [str(self.job.pk), 1, 1]
        assert dispatch.kwargs["ocr_engine_lane"] == "tesseract"

    def test_global_byte_budget_backpressures_before_render(self):
        budget = PageOCRByteBudget.objects.get(pk=1)
        budget.reserved_bytes = 16 * 1024 * 1024 * 1024
        budget.save(update_fields=["reserved_bytes"])
        with patch.dict(os.environ, {"OCR_STAGE_GLOBAL_BYTES": str(16 * 1024 * 1024 * 1024)}):
            from jobs.tasks import _stage_reserve

            assert _stage_reserve(self.stage.pk, self.job.pk) == "saturated"
        self.stage.refresh_from_db()
        assert self.stage.status == "pending"
        assert self.stage.reserved_bytes == 0

    def test_late_raster_put_is_deleted_before_reservation_reclaim(self):
        objects = {}

        class Backend:
            backend_name = "s3"

            def delete(self, key):
                objects.pop(key, None)

        backend = Backend()
        old_token = _stage_reserve(self.stage.pk, self.job.pk, backend)
        self.stage.refresh_from_db()
        old_locator = self.stage.raster_locator
        assert old_token and old_locator
        self.stage.status = "cleanup"
        self.stage.lease_token = ""
        self.stage.lease_expires_at = None
        self.stage.last_error = "retry_render:render_lease_expired"
        self.stage.publish_safe_after = timezone.now() + timezone.timedelta(hours=1)
        self.stage.save()
        _stage_cleanup_artifact(self.stage.pk, backend)
        self.stage.refresh_from_db()
        retired = PageOCRRasterRetirement.objects.get(locator=old_locator)
        assert self.stage.status == "pending"
        assert self.stage.reserved_bytes == 0
        assert retired.reserved_bytes > 0
        assert PageOCRByteBudget.objects.get(pk=1).reserved_bytes == retired.reserved_bytes
        new_token = _stage_reserve(self.stage.pk, self.job.pk, backend)
        self.stage.refresh_from_db()
        assert new_token != old_token
        assert self.stage.raster_locator != old_locator
        with patch("jobs.tasks._get_storage_backend", return_value=backend):
            assert _sweep_retired_stage_rasters() == 1
        retired.refresh_from_db()
        assert retired.reserved_bytes > 0
        objects[old_locator] = b"late PUT after first cleanup attempt"
        retired.safe_to_release_after = timezone.now() - timezone.timedelta(seconds=1)
        retired.next_sweep_at = timezone.now() - timezone.timedelta(seconds=1)
        retired.save(update_fields=["safe_to_release_after", "next_sweep_at"])
        with patch("jobs.tasks._get_storage_backend", return_value=backend):
            assert _sweep_retired_stage_rasters() == 1
        retired.refresh_from_db()
        assert old_locator not in objects
        assert retired.reserved_bytes == 0
        assert PageOCRByteBudget.objects.get(pk=1).reserved_bytes == self.stage.reserved_bytes

    def test_render_expiring_during_put_cannot_commit_rendered(self):
        class Backend:
            backend_name = "s3"

            def delete(self, key):
                pass

        backend = Backend()
        source = os.path.join(self.job_path, "source", "input.pdf")

        def expired_put(stage, job, _backend, payload, token, presigned_urls=None):
            assert payload and token and presigned_urls is None
            PageOCRStage.objects.filter(pk=stage.pk).update(
                lease_expires_at=timezone.now() - timezone.timedelta(seconds=1)
            )
            return stage.raster_locator

        with (
            override_settings(NFS_ROOT=self.root),
            patch("jobs.tasks._get_backend_for_job", return_value=backend),
            patch("jobs.tasks._stage_source_lease", return_value=nullcontext(source)),
            patch("jobs.tasks._page_script_evidence", return_value=()),
            patch("jobs.tasks._stage_publish_raster", side_effect=expired_put),
        ):
            with self.assertRaises(_ProcessingLeaseLost):
                render_page.run(str(self.job.pk), 1, 1)
        self.stage.refresh_from_db()
        assert self.stage.status != "rendered"
        assert self.stage.raster_sha256 == ""
        assert PageOCRByteBudget.objects.get(pk=1).reserved_bytes == 0

    def test_lost_raster_put_ack_creates_durable_exact_key_retirement(self):
        objects = {}

        class Backend:
            backend_name = "s3"

            def delete(self, key):
                objects.pop(key, None)

        backend = Backend()
        source = os.path.join(self.job_path, "source", "input.pdf")

        def lost_ack(stage, job, _backend, payload, token, presigned_urls=None):
            objects[stage.raster_locator] = payload
            raise RuntimeError("remote PUT acknowledged no response")

        with (
            override_settings(NFS_ROOT=self.root),
            patch("jobs.tasks._get_backend_for_job", return_value=backend),
            patch("jobs.tasks._stage_source_lease", return_value=nullcontext(source)),
            patch("jobs.tasks._page_script_evidence", return_value=()),
            patch("jobs.tasks._stage_publish_raster", side_effect=lost_ack),
        ):
            with self.assertRaisesRegex(RuntimeError, "acknowledged no response"):
                render_page.run(str(self.job.pk), 1, 1)
        self.stage.refresh_from_db()
        retired = PageOCRRasterRetirement.objects.get(job_uuid=self.job.pk)
        assert retired.locator in objects
        assert retired.reserved_bytes > 0
        assert self.stage.status == "pending"
        assert self.stage.raster_locator == ""
        assert PageOCRByteBudget.objects.get(pk=1).reserved_bytes == retired.reserved_bytes

    def test_explicit_gpu_request_does_not_depend_on_caller_default_place(self):
        from jobs import tasks

        fake_paddle = types.SimpleNamespace(
            device=types.SimpleNamespace(
                get_device=lambda: "cpu",
                is_compiled_with_cuda=lambda: True,
                cuda=types.SimpleNamespace(device_count=lambda: 1),
            ),
        )

        class Pool:
            def run(self, key, factory, invoke, raster):
                assert key.device.startswith("gpu:0|visible:CUDA:GPU-test")
                return invoke(factory(), raster)

        class Engine:
            def ocr(self, raster):
                return raster

        with (
            patch.dict("sys.modules", {"paddle": fake_paddle}),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test"}),
            patch("jobs.tasks._ocr_runtime_pool", return_value=Pool()),
        ):
            assert tasks._run_paddle_ocr(
                "en", "gpu:0", "raster", lambda lang, device: Engine()
            ) == "raster"

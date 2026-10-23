"""Owned PDF fixtures exercise native completeness and durable OCR handoff."""

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from django.test import TestCase, override_settings
from django.utils import timezone

from jobs import tasks
from jobs.models import CustodyEvent, Job, PageResult, TaskDispatchOutbox
from jobs.storage import NFSBackend
from ocr_local.native_pdf import assess_native_pdf_page
from ocr_local.visual_coverage import CoverageAssessment


def make_pdf(path, *, mixed=False, pages=1, rotation=0, cropped=False, unmapped=False):
    with fitz.open() as document:
        for index in range(pages):
            page = document.new_page(width=595, height=842)
            page.insert_text((72, 72), "NATIVE HEADER", fontsize=24)
            page.insert_text((72, 112), "SECOND NATIVE LINE", fontsize=24)
            if mixed and index == pages - 1:
                with fitz.open() as raster_document:
                    raster_page = raster_document.new_page(width=1000, height=150)
                    raster_page.insert_text((20, 85), "RASTER EVIDENCE LINE", fontsize=44)
                    raster_bytes = raster_page.get_pixmap().tobytes("png")
                page.insert_image(fitz.Rect(72, 170, 520, 237), stream=raster_bytes)
            if unmapped:
                page.insert_font(fontname="owned", fontbuffer=fitz.Font("cjk").buffer)
                text = "A42B" if unmapped == "zero_width" else "42"
                page.insert_text((40, 650), text, fontsize=10, fontname="owned")
                characters = page.get_texttrace()[-1]["chars"]
                glyphs = {
                    char[1]: (0x200B if unmapped == "zero_width" else 0x20)
                    if char[0] in (ord("4"), ord("2")) else char[0]
                    for char in characters
                }
                font_xref = next(font[0] for font in page.get_fonts() if font[4] == "owned")
                _, reference = document.xref_get_key(font_xref, "ToUnicode")
                cmap_xref = int(reference.split()[0])
                mappings = " ".join(
                    f"<{glyph:04x}> <{unicode:04x}>" for glyph, unicode in glyphs.items()
                )
                document.update_stream(cmap_xref, (
                    "/CIDInit /ProcSet findresource begin 12 dict begin begincmap "
                    "/CIDSystemInfo <</Registry (Adobe) /Ordering (UCS) /Supplement 0>> def "
                    "/CMapName /Owned def /CMapType 2 def "
                    "1 begincodespacerange <0000> <FFFF> endcodespacerange "
                    f"{len(glyphs)} beginbfchar {mappings} endbfchar "
                    "endcmap CMapName currentdict /CMap defineresource pop end end"
                ).encode())
            if cropped:
                page.set_cropbox(fitz.Rect(20, 30, 580, 800))
            page.set_rotation(rotation)
        document.save(path)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("cropped", [False, True])
def test_native_coverage_coordinates(tmp_path, rotation, cropped):
    path = tmp_path / "native.pdf"
    make_pdf(path, rotation=rotation, cropped=cropped)
    with fitz.open(path) as document:
        assessment = assess_native_pdf_page(document[0])
    assert assessment.status == "measured"
    assert assessment.coverage == 1.0
    assert not assessment.needs_retry


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize("fontsize", [4, 10, 24])
def test_native_small_and_accented_text_remains_admissible(tmp_path, rotation, cropped, fontsize):
    path = tmp_path / "native.pdf"
    make_pdf(path, rotation=rotation, cropped=cropped)
    with fitz.open(path) as document:
        page = document[0]
        page.insert_text((72, 170), "Café naïve Ångström gyjp 42", fontsize=fontsize)
        assessment = assess_native_pdf_page(page)
    assert assessment.status == "measured"
    assert assessment.coverage == 1.0
    assert not assessment.needs_retry
    assert not assessment.needs_review

def test_native_render_is_bounded_before_allocation():
    with fitz.open() as document:
        page = document.new_page(width=5000, height=5000)
        with patch.object(fitz.Page, "get_pixmap", side_effect=AssertionError("must not allocate")) as render:
            assessment = assess_native_pdf_page(page)
    render.assert_not_called()
    assert assessment.needs_retry
    assert assessment.reason == "native_preflight_pixel_limit"


@pytest.mark.parametrize("content", ["short_raster", "vector", "annotation"])
def test_small_non_native_content_cannot_be_declared_complete(tmp_path, content):
    path = tmp_path / "native.pdf"
    make_pdf(path)
    with fitz.open(path) as document:
        page = document[0]
        if content == "short_raster":
            with fitz.open() as raster_document:
                raster_page = raster_document.new_page(width=80, height=50)
                raster_page.insert_text((5, 32), "42", fontsize=24)
                page.insert_image(fitz.Rect(400, 90, 480, 140), stream=raster_page.get_pixmap().tobytes("png"))
        elif content == "vector":
            page.draw_line((400, 90), (410, 110), color=(0, 0, 0))
        else:
            page.add_freetext_annot(fitz.Rect(400, 90, 480, 140), "42")
        if content != "annotation":
            assert "42" not in page.get_text()
        assessment = assess_native_pdf_page(page)
    assert assessment.status == "unknown"
    assert assessment.needs_retry


class NativePDFCoverageTests(TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.job_path = self.root / "job"
        (self.job_path / "source").mkdir(parents=True)
        self.source = self.job_path / "source" / "input.pdf"

    def make_job(self, *, mixed=False, pages=1, unmapped=False):
        make_pdf(self.source, mixed=mixed, pages=pages, unmapped=unmapped)
        self.original_bytes = self.source.read_bytes()
        self.job = Job.objects.create(
            source_file="input.pdf", source_type="pdf", total_pages=pages,
            source_hash=hashlib.sha256(self.original_bytes).hexdigest(),
            status=Job.Status.PROCESSING, dispatch_generation=1,
            nfs_job_path=str(self.job_path), settings_json={"skip_ocr": True},
        )
        return self.job

    def run_native(self):
        with (
            override_settings(NFS_ROOT=str(self.root)),
            patch.object(tasks, "_get_backend_for_job", return_value=NFSBackend(root=str(self.root))),
        ):
            return tasks.process_text_only.run(str(self.job.pk), dispatch_generation=1)

    def test_complete_native_pdf_preserves_text_and_coverage_custody(self):
        self.make_job(pages=2)
        assert self.run_native()["status"] == "processed"
        self.job.refresh_from_db()
        assert self.job.pages_completed == 2
        assert self.job.dispatch_generation == 1
        for page in PageResult.objects.filter(job=self.job):
            assert page.status == "ok"
            assert page.ocr_method == "NativePDF"
            assert page.ocr_provenance["native_coverage"]["coverage"] == 1.0
            with fitz.open(page.temp_pdf_path) as pdf:
                assert pdf[0].get_text() == Path(page.ocr_provenance["temp_text_path"]).read_text()
        assert CustodyEvent.objects.filter(job=self.job, event_type="text_extracted_natively").count() == 2
        assert TaskDispatchOutbox.objects.filter(job=self.job, stage="assemble_document").count() == 1

    def test_partial_last_page_hands_off_before_any_native_publication(self):
        self.make_job(mixed=True, pages=2)
        assert self.run_native()["status"] == "ocr_queued"
        self.job.refresh_from_db()
        assert self.job.status == Job.Status.PROCESSING
        assert self.job.dispatch_generation == 2
        assert not self.job.processing_attempt_token
        assert not self.job.settings_json["skip_ocr"]
        assert self.job.pages_completed == self.job.pages_failed == 0
        assert not PageResult.objects.filter(job=self.job).exists()
        assert not CustodyEvent.objects.filter(job=self.job, event_type="text_extracted_natively").exists()
        decision = CustodyEvent.objects.get(job=self.job, event_type="native_pdf_ocr_required").data
        assert decision["page_num"] == 2
        assert decision["requested_skip_ocr"] is True
        assert decision["coverage"]["needs_retry"]
        row = TaskDispatchOutbox.objects.get(job=self.job)
        assert (row.task_name, row.queue, row.generation) == ("jobs.tasks.extract_pages", "coordinator", "2")
        assert self.source.read_bytes() == self.original_bytes
        assert not list(self.job_path.glob("temp/**/*.pdf"))
        assert self.run_native()["status"] == "stale"
        assert TaskDispatchOutbox.objects.filter(job=self.job).count() == 1

    def test_unmapped_visible_digits_handoff_without_native_publication(self):
        for mapping in (True, "zero_width"):
            with self.subTest(mapping=mapping):
                self.make_job(unmapped=mapping)
                with fitz.open(self.source) as document:
                    page = document[0]
                    assert "42" not in page.get_text()
                    assert {entry[0] for entry in page.get_bboxlog()} == {"fill-text"}
                    # Real visible glyphs, not a patched coverage assessment.
                    assert min(page.get_pixmap(clip=fitz.Rect(40, 635, 90, 655)).samples) < 128
                    assessment = assess_native_pdf_page(page)
                assert assessment.needs_retry
                assert assessment.reason == "native_uncovered_visible_ink"
                assert self.run_native()["status"] == "ocr_queued"
                self.job.refresh_from_db()
                assert self.job.dispatch_generation == 2
                assert not PageResult.objects.filter(job=self.job).exists()
                assert not CustodyEvent.objects.filter(
                    job=self.job, event_type="text_extracted_natively",
                ).exists()
                assert TaskDispatchOutbox.objects.filter(
                    job=self.job, task_name="jobs.tasks.extract_pages", generation="2",
                ).count() == 1
                assert self.source.read_bytes() == self.original_bytes

    def test_unknown_render_routes_to_ocr(self):
        self.make_job()
        with patch.object(fitz.Page, "get_pixmap", side_effect=RuntimeError("owned render failure")):
            assert self.run_native()["status"] == "ocr_queued"
        assert not PageResult.objects.filter(job=self.job).exists()
        assert self.source.read_bytes() == self.original_bytes

    def test_cancel_or_replace_generation_during_preflight_cannot_handoff(self):
        self.make_job(mixed=True)
        real_assess = tasks.assess_native_pdf_page

        def revoke(page):
            Job.objects.filter(pk=self.job.pk).update(status=Job.Status.CANCELLED, dispatch_generation=2)
            return real_assess(page)

        with patch.object(tasks, "assess_native_pdf_page", side_effect=revoke):
            assert self.run_native()["status"] == "stale"
        assert not TaskDispatchOutbox.objects.filter(job=self.job).exists()
        assert not PageResult.objects.filter(job=self.job).exists()
        assert self.source.read_bytes() == self.original_bytes

    def test_outbox_insert_failure_rolls_back_handoff_and_fails_truthfully(self):
        self.make_job(mixed=True)
        with patch.object(tasks, "enqueue_task_dispatch", side_effect=RuntimeError("owned insert failure")):
            assert self.run_native()["status"] == "error"
        self.job.refresh_from_db()
        assert self.job.status == Job.Status.FAILED
        assert self.job.dispatch_generation == 1
        assert self.job.settings_json["skip_ocr"] is True
        assert not TaskDispatchOutbox.objects.filter(job=self.job).exists()
        assert not CustodyEvent.objects.filter(job=self.job, event_type="native_pdf_ocr_required").exists()
        assert self.source.read_bytes() == self.original_bytes

    def test_broker_failure_keeps_committed_handoff_for_relay(self):
        self.make_job(mixed=True)
        with patch.object(
            tasks.extract_pages, "apply_async", side_effect=RuntimeError("owned broker unavailable"),
        ), self.captureOnCommitCallbacks(execute=True):
            assert self.run_native()["status"] == "ocr_queued"
        self.job.refresh_from_db()
        row = TaskDispatchOutbox.objects.get(job=self.job)
        assert self.job.status == Job.Status.PROCESSING
        assert row.status == TaskDispatchOutbox.Status.FAILED
        assert not PageResult.objects.filter(job=self.job).exists()

    def test_expired_native_lease_cannot_enqueue_successor(self):
        self.make_job()
        token = tasks._claim_processing_lease(self.job, "owned-delivery")
        Job.objects.filter(pk=self.job.pk).update(processing_lease_expires_at=timezone.now())
        unknown = CoverageAssessment("unknown", None, None, None, None, (), True, True, "owned_unknown")
        with pytest.raises(tasks._ProcessingLeaseLost):
            tasks._handoff_native_pdf_to_ocr(self.job, token, 1, unknown)
        assert not TaskDispatchOutbox.objects.filter(job=self.job).exists()

    def test_legacy_native_prefix_is_readmitted_without_deleting_artifacts(self):
        self.make_job(mixed=True, pages=2)
        prior = self.job_path / "prior.pdf"
        prior.write_bytes(b"preserved prior attempt")
        PageResult.objects.create(
            job=self.job, page_num=1, status="ok", ocr_method="NativePDF",
            admitted_generation=1, temp_pdf_path=str(prior),
        )
        assert self.run_native()["status"] == "ocr_queued"
        self.job.refresh_from_db()
        with patch.dict(os.environ, {"OCR_FAIR_FANOUT_ENABLED": "true"}):
            assert tasks.extract_pages.run(str(self.job.pk), dispatch_generation=2)["status"] == "fanout_dispatched"
        assert PageResult.objects.filter(job=self.job, status="pending", admitted_generation=2).count() == 2
        assert prior.read_bytes() == b"preserved prior attempt"

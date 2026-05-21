"""Comprehensive tests for failure reprocessing pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestFailureStore:
    """Tests for FailureStore CSV handling."""

    def test_read_legacy_csv(self, tmp_path):
        """Test reading legacy 4-column CSV format."""
        from reprocess.failures import FailureStore

        csv_path = tmp_path / "failures.csv"
        csv_path.write_text(
            "Timestamp,SourcePath,PageNum,Error\n"
            "2024-01-01T10:00:00,/path/doc.pdf,1,No text extracted\n"
            "2024-01-01T10:05:00,/path/doc2.pdf,3,GPU error occurred\n"
        )

        store = FailureStore(csv_path)
        records = store.read_failures()

        assert len(records) == 2
        assert records[0].source_path == "/path/doc.pdf"
        assert records[0].page_num == 1
        assert records[0].error == "No text extracted"
        assert records[0].failure_type == "ocr_text"  # auto-classified
        assert records[0].status == "failed"
        assert records[0].retry_count == 0
        assert records[0].dpi_used == 300

    def test_read_enhanced_csv(self, tmp_path):
        """Test reading enhanced 10-column CSV format."""
        from reprocess.failures import FailureStore

        csv_path = tmp_path / "failures.csv"
        csv_path.write_text(
            "Timestamp,SourcePath,PageNum,Error,FailureType,Status,RetryCount,DPIUsed,LastRetryTimestamp,ResolutionMethod\n"
            "2024-01-01T10:00:00,/path/doc.pdf,1,No text,ocr_text,pending_retry,1,450,2024-01-01T11:00:00,paddle\n"
        )

        store = FailureStore(csv_path)
        records = store.read_failures()

        assert len(records) == 1
        assert records[0].retry_count == 1
        assert records[0].dpi_used == 450
        assert records[0].status == "pending_retry"
        assert records[0].resolution_method == "paddle"

    def test_classify_ocr_text_failure(self):
        """Test OCR text failure classification."""
        from reprocess.failures import FailureStore, FailureType

        store = FailureStore("dummy.csv")
        
        assert store.classify_failure("No text extracted") == FailureType.OCR_TEXT
        assert store.classify_failure("OCR returned empty result") == FailureType.OCR_TEXT
        assert store.classify_failure("Blank page detected") == FailureType.OCR_TEXT
        assert store.classify_failure("Insufficient text found") == FailureType.OCR_TEXT

    def test_classify_critical_failure(self):
        """Test critical failure classification."""
        from reprocess.failures import FailureStore, FailureType

        store = FailureStore("dummy.csv")
        
        assert store.classify_failure("Out of memory") == FailureType.CRITICAL
        assert store.classify_failure("CUDA error detected") == FailureType.CRITICAL
        assert store.classify_failure("GPU error occurred") == FailureType.CRITICAL
        assert store.classify_failure("System error: permission denied") == FailureType.CRITICAL

    def test_classify_extract_failed(self):
        """Test extract failed classification."""
        from reprocess.failures import FailureStore, FailureType

        store = FailureStore("dummy.csv")
        
        assert store.classify_failure("Extract failed for page") == FailureType.EXTRACT_FAILED
        assert store.classify_failure("PDF corrupt or damaged") == FailureType.EXTRACT_FAILED
        assert store.classify_failure("Cannot render page") == FailureType.EXTRACT_FAILED
        assert store.classify_failure("Invalid PDF structure") == FailureType.EXTRACT_FAILED

    def test_classify_unknown_failure(self):
        """Test unknown failure classification."""
        from reprocess.failures import FailureStore, FailureType

        store = FailureStore("dummy.csv")
        
        assert store.classify_failure("Something went wrong") == FailureType.UNKNOWN
        assert store.classify_failure("Random error message") == FailureType.UNKNOWN

    def test_get_retriable_failures(self, tmp_path):
        """Test filtering retriable failures."""
        from reprocess.failures import FailureStore

        csv_path = tmp_path / "failures.csv"
        csv_path.write_text(
            "Timestamp,SourcePath,PageNum,Error,FailureType,Status,RetryCount,DPIUsed,LastRetryTimestamp,ResolutionMethod\n"
            "2024-01-01T10:00:00,/doc1.pdf,1,No text,ocr_text,failed,0,300,,\n"
            "2024-01-01T10:05:00,/doc2.pdf,1,No text,ocr_text,failed,1,450,,\n"
            "2024-01-01T10:10:00,/doc3.pdf,1,No text,ocr_text,resolved,1,450,,paddle\n"
            "2024-01-01T10:15:00,/doc4.pdf,1,No text,ocr_text,retry_exhausted,2,600,,\n"
        )

        store = FailureStore(csv_path)
        retriable = store.get_retriable_failures(max_retries=2)

        assert len(retriable) == 2  # Only first two
        assert retriable[0].source_path == "/doc1.pdf"
        assert retriable[1].source_path == "/doc2.pdf"

    def test_retriable_excludes_extract_failed(self, tmp_path):
        """Test that EXTRACT_FAILED is not retriable."""
        from reprocess.failures import FailureStore

        csv_path = tmp_path / "failures.csv"
        csv_path.write_text(
            "Timestamp,SourcePath,PageNum,Error,FailureType,Status,RetryCount,DPIUsed,LastRetryTimestamp,ResolutionMethod\n"
            "2024-01-01T10:00:00,/doc1.pdf,1,Extract failed,extract_failed,failed,0,300,,\n"
        )

        store = FailureStore(csv_path)
        retriable = store.get_retriable_failures(max_retries=2)

        assert len(retriable) == 0

    def test_retriable_excludes_max_retries(self, tmp_path):
        """Test that failures at max retries are excluded."""
        from reprocess.failures import FailureStore

        csv_path = tmp_path / "failures.csv"
        csv_path.write_text(
            "Timestamp,SourcePath,PageNum,Error,FailureType,Status,RetryCount,DPIUsed,LastRetryTimestamp,ResolutionMethod\n"
            "2024-01-01T10:00:00,/doc1.pdf,1,No text,ocr_text,failed,2,600,,\n"
        )

        store = FailureStore(csv_path)
        retriable = store.get_retriable_failures(max_retries=2)

        assert len(retriable) == 0

    def test_write_and_read_roundtrip(self, tmp_path):
        """Test writing and reading records maintains data."""
        from reprocess.failures import FailureRecord, FailureStore

        csv_path = tmp_path / "failures.csv"
        store = FailureStore(csv_path)

        original_records = [
            FailureRecord(
                timestamp="2024-01-01T10:00:00",
                source_path="/doc1.pdf",
                page_num=1,
                error="No text",
                failure_type="ocr_text",
                status="failed",
                retry_count=0,
                dpi_used=300,
                last_retry_timestamp="",
                resolution_method="",
            ),
            FailureRecord(
                timestamp="2024-01-01T10:05:00",
                source_path="/doc2.pdf",
                page_num=2,
                error="GPU error",
                failure_type="critical",
                status="not_retriable",
                retry_count=0,
                dpi_used=300,
                last_retry_timestamp="",
                resolution_method="",
            ),
        ]

        store.write_failures(original_records)
        read_records = store.read_failures()

        assert len(read_records) == 2
        assert read_records[0].source_path == "/doc1.pdf"
        assert read_records[0].failure_type == "ocr_text"
        assert read_records[1].source_path == "/doc2.pdf"
        assert read_records[1].failure_type == "critical"

    def test_read_malformed_csv_skips_bad_rows(self, tmp_path):
        """Test that malformed rows (non-numeric PageNum) are skipped gracefully."""
        from reprocess.failures import FailureStore

        csv_path = tmp_path / "failures.csv"
        csv_path.write_text(
            "Timestamp,SourcePath,PageNum,Error\n"
            "2024-01-01T10:00:00,/path/doc.pdf,1,No text extracted\n"
            "2024-01-01T10:05:00,/path/bad.pdf,NOT_A_NUMBER,GPU error occurred\n"
            "2024-01-01T10:10:00,/path/good.pdf,3,OCR returned empty result\n"
        )

        store = FailureStore(csv_path)
        records = store.read_failures()

        # Malformed row should be skipped, so only 2 valid records
        assert len(records) == 2
        assert records[0].source_path == "/path/doc.pdf"
        assert records[1].source_path == "/path/good.pdf"


class TestDPIRenderer:
    """Tests for DPI escalation and rendering."""

    def test_dpi_schedule(self):
        """Test DPI schedule constant."""
        from reprocess.renderer import DPI_SCHEDULE

        assert DPI_SCHEDULE == [300, 450, 600]

    def test_get_next_dpi_300(self):
        """Test getting next DPI from 300."""
        from reprocess.renderer import get_next_dpi

        assert get_next_dpi(300) == 450

    def test_get_next_dpi_450(self):
        """Test getting next DPI from 450."""
        from reprocess.renderer import get_next_dpi

        assert get_next_dpi(450) == 600

    def test_get_next_dpi_600_returns_none(self):
        """Test that 600 DPI is maximum."""
        from reprocess.renderer import get_next_dpi

        assert get_next_dpi(600) is None

    def test_get_next_dpi_unknown_returns_none(self):
        """Test that unknown DPI returns None."""
        from reprocess.renderer import get_next_dpi

        assert get_next_dpi(150) is None
        assert get_next_dpi(1200) is None

    def test_render_page_missing_file(self, tmp_path):
        """Test render_page with missing PDF."""
        from reprocess.renderer import RenderError, render_page

        pdf_path = tmp_path / "missing.pdf"

        with patch("reprocess.renderer.convert_from_path", create=True):
            with pytest.raises(RenderError, match="PDF file not found"):
                render_page(pdf_path, 1, 300)

    def test_render_page_success(self, tmp_path):
        """Test successful page rendering (mocked)."""
        from reprocess.renderer import render_page

        # Create dummy PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        # Mock pdf2image
        mock_image = Mock()
        with patch("reprocess.renderer.convert_from_path", return_value=[mock_image], create=True):
            result = render_page(pdf_path, 1, 300)
            assert result == mock_image

    def test_render_page_no_pdf2image(self, tmp_path):
        """Test render_page when pdf2image not available."""
        from reprocess.renderer import RenderError, render_page

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")

        with patch("reprocess.renderer.convert_from_path", None):
            with pytest.raises(RenderError, match="pdf2image not available"):
                render_page(pdf_path, 1, 300)


class TestOCREngine:
    """Tests for OCR engine with mocking."""

    def test_paddle_success(self):
        """Test successful PaddleOCR extraction."""
        from reprocess.ocr_core import OCREngine

        mock_image = Mock()
        mock_paddle_instance = MagicMock()
        mock_paddle_instance.ocr.return_value = [
            [
                [None, ("First line", 0.95)],
                [None, ("Second line", 0.92)],
            ]
        ]
        mock_paddle_cls = MagicMock(return_value=mock_paddle_instance)

        with patch("reprocess.ocr_core.PaddleOCR", mock_paddle_cls):
            engine = OCREngine(use_paddle=True)
            engine._paddle_available = True
            engine._paddle_ocr = mock_paddle_instance

            text, method = engine.run_ocr(mock_image)

            assert "First line" in text
            assert "Second line" in text
            assert method == "paddle"

    def test_paddle_fails_tesseract_fallback(self):
        """Test fallback to Tesseract when PaddleOCR fails."""
        from reprocess.ocr_core import OCREngine

        mock_image = Mock()
        mock_tess = MagicMock()
        mock_tess.image_to_string.return_value = "Tesseract text"

        with patch("reprocess.ocr_core.PaddleOCR", None):
            with patch("reprocess.ocr_core.pytesseract", mock_tess):
                engine = OCREngine(use_paddle=True)
                engine._paddle_available = True
                engine._tesseract_available = True

                # Force paddle to fail
                with patch.object(engine, "_run_paddle", side_effect=Exception("Paddle failed")):
                    text, method = engine.run_ocr(mock_image)

                    assert text == "Tesseract text"
                    assert method == "tesseract"

    def test_all_fail_image_only(self):
        """Test image_only fallback when all OCR fails."""
        from reprocess.ocr_core import OCREngine

        mock_image = Mock()

        engine = OCREngine(use_paddle=True)
        engine._paddle_available = False
        engine._tesseract_available = False

        text, method = engine.run_ocr(mock_image)

        assert text == ""
        assert method == "image_only"

    def test_no_paddle_installed(self):
        """Test graceful handling when PaddleOCR not installed."""
        from reprocess.ocr_core import OCREngine

        with patch("reprocess.ocr_core.PaddleOCR", None):
            engine = OCREngine(use_paddle=True)

            assert engine.use_paddle is True
            assert engine._paddle_available is False


class TestReporter:
    """Tests for report generation."""

    def test_empty_report(self):
        """Test report generation with no results."""
        from reprocess.reporter import ReportGenerator

        reporter = ReportGenerator("/tmp/report.md")
        report = reporter.generate_report()

        assert "Total Processed**: 0" in report
        assert "Resolved**: 0" in report
        assert "No failures were processed" in report

    def test_report_with_resolved(self):
        """Test report with resolved failures."""
        from reprocess.reporter import ReportGenerator, ReprocessResult

        reporter = ReportGenerator("/tmp/report.md")
        reporter.add_result(
            ReprocessResult(
                source_path="/docs/file1.pdf",
                page_num=1,
                original_error="No text",
                original_dpi=300,
                retry_dpi=450,
                success=True,
                resolution_method="paddle",
            )
        )

        report = reporter.generate_report()

        assert "Total Processed**: 1" in report
        assert "Resolved**: 1" in report
        assert "Retry Exhausted**: 0" in report
        assert "file1.pdf" in report
        assert "450" in report
        assert "paddle" in report

    def test_report_with_exhausted(self):
        """Test report with retry exhausted failures."""
        from reprocess.reporter import ReportGenerator, ReprocessResult

        reporter = ReportGenerator("/tmp/report.md")
        reporter.add_result(
            ReprocessResult(
                source_path="/docs/file1.pdf",
                page_num=1,
                original_error="No text",
                original_dpi=300,
                retry_dpi=600,
                success=False,
                resolution_method="",
                new_error="Still no text at 600 DPI",
            )
        )

        report = reporter.generate_report()

        assert "Total Processed**: 1" in report
        assert "Resolved**: 0" in report
        assert "Retry Exhausted**: 1" in report
        assert "Still no text" in report

    def test_report_with_mixed_results(self):
        """Test report with mixed success/failure."""
        from reprocess.reporter import ReportGenerator, ReprocessResult

        reporter = ReportGenerator("/tmp/report.md")
        reporter.add_result(
            ReprocessResult(
                source_path="/docs/success.pdf",
                page_num=1,
                original_error="No text",
                original_dpi=300,
                retry_dpi=450,
                success=True,
                resolution_method="tesseract",
            )
        )
        reporter.add_result(
            ReprocessResult(
                source_path="/docs/failed.pdf",
                page_num=2,
                original_error="No text",
                original_dpi=300,
                retry_dpi=450,
                success=False,
                resolution_method="",
                new_error="Still failed",
            )
        )

        report = reporter.generate_report()

        assert "Total Processed**: 2" in report
        assert "Resolved**: 1" in report
        assert "Retry Exhausted**: 1" in report
        assert "success.pdf" in report
        assert "failed.pdf" in report

    def test_save_report_writes_file(self, tmp_path):
        """Test that save_report writes to disk."""
        from reprocess.reporter import ReportGenerator, ReprocessResult

        report_path = tmp_path / "test_report.md"
        reporter = ReportGenerator(report_path)
        reporter.add_result(
            ReprocessResult(
                source_path="/docs/file1.pdf",
                page_num=1,
                original_error="No text",
                original_dpi=300,
                retry_dpi=450,
                success=True,
                resolution_method="paddle",
            )
        )

        reporter.save_report()

        assert report_path.exists()
        content = report_path.read_text()
        assert "Reprocessing Report" in content
        assert "file1.pdf" in content


class TestCLI:
    """Tests for CLI functionality."""

    def _import_reprocess_script(self):
        """Import reprocess.py script (not the reprocess/ package)."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "reprocess_cli", str(Path(__file__).parent.parent / "reprocess.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_dry_run_no_changes(self, tmp_path):
        """Test dry run mode doesn't modify CSV."""
        import sys

        csv_path = tmp_path / "failures.csv"
        csv_path.write_text(
            "Timestamp,SourcePath,PageNum,Error,FailureType,Status,RetryCount,DPIUsed,LastRetryTimestamp,ResolutionMethod\n"
            "2024-01-01T10:00:00,/doc1.pdf,1,No text,ocr_text,failed,0,300,,\n"
        )

        original_content = csv_path.read_text()

        mod = self._import_reprocess_script()

        test_args = [
            "reprocess.py",
            "--failures", str(csv_path),
            "--dry-run",
            "--output-report", str(tmp_path / "report.md"),
        ]

        mock_engine = MagicMock()
        with patch.object(sys, "argv", test_args), \
             patch.object(mod, "OCREngine", return_value=mock_engine):
            exit_code = mod.main()

        assert exit_code == 0
        # In dry run, CSV should be unchanged
        assert csv_path.read_text() == original_content

    def test_main_with_empty_failures(self, tmp_path):
        """Test main with empty failures file."""
        from reprocess.failures import FailureStore

        csv_path = tmp_path / "failures.csv"
        csv_path.write_text(
            "Timestamp,SourcePath,PageNum,Error,FailureType,Status,RetryCount,DPIUsed,LastRetryTimestamp,ResolutionMethod\n"
        )

        store = FailureStore(csv_path)
        retriable = store.get_retriable_failures()

        assert len(retriable) == 0

    def test_main_processes_failures(self, tmp_path):
        """Test main processing workflow (mocked)."""
        import sys

        csv_path = tmp_path / "failures.csv"
        csv_path.write_text(
            "Timestamp,SourcePath,PageNum,Error,FailureType,Status,RetryCount,DPIUsed,LastRetryTimestamp,ResolutionMethod\n"
            "2024-01-01T10:00:00,/doc1.pdf,1,No text,ocr_text,failed,0,300,,\n"
        )

        mod = self._import_reprocess_script()

        test_args = [
            "reprocess.py",
            "--failures", str(csv_path),
            "--output-report", str(tmp_path / "report.md"),
        ]

        mock_image = Mock()
        mock_engine = MagicMock()
        mock_engine.run_ocr.return_value = ("Extracted text", "paddle")

        with patch.object(sys, "argv", test_args), \
             patch.object(mod, "OCREngine", return_value=mock_engine), \
             patch.object(mod, "render_page", return_value=mock_image):
            exit_code = mod.main()

        assert exit_code == 0
        # Verify CSV was updated
        from reprocess.failures import FailureStore
        store = FailureStore(csv_path)
        records = store.read_failures()
        assert records[0].status == "resolved"
        assert records[0].retry_count == 1

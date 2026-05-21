"""Tests for log_failure() in ocr_gpu_async.py.

Verifies that write errors are logged as warnings instead of being
silently swallowed.
"""

import logging
from unittest.mock import patch

import ocr_gpu_async as pipe  # noqa: E402


class TestLogFailureHappyPath:
    """Verify log_failure writes CSV rows correctly."""

    def test_writes_csv_row(self, tmp_path):
        report = tmp_path / "failures.csv"
        with patch.object(pipe, "FAILURE_REPORT", str(report)):
            pipe.log_failure("/docs/test.pdf", 3, "some error")

        content = report.read_text(encoding="utf-8")
        assert "/docs/test.pdf" in content
        assert ",3," in content
        assert "some error" in content

    def test_sanitizes_commas_and_newlines(self, tmp_path):
        report = tmp_path / "failures.csv"
        with patch.object(pipe, "FAILURE_REPORT", str(report)):
            pipe.log_failure("file.pdf", 1, "bad,error\nwith newline")

        content = report.read_text(encoding="utf-8")
        # Commas replaced with semicolons, newlines with spaces
        assert "bad;error with newline" in content
        assert "\n" in content  # only the trailing newline from the write


class TestLogFailureWriteError:
    """Verify write errors emit a warning instead of being swallowed."""

    def test_logs_warning_on_permission_error(self, caplog):
        """When the failure report cannot be opened, a warning is logged."""
        bad_path = "/nonexistent/dir/that/does/not/exist/failures.csv"
        with patch.object(pipe, "FAILURE_REPORT", bad_path):
            with caplog.at_level(logging.WARNING, logger="ocr_pipeline"):
                pipe.log_failure("test.pdf", 1, "original error")

        # Must have logged a warning about the write failure
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        msg = warning_msgs[0].message
        assert "test.pdf" in msg
        assert "p1" in msg

    def test_logs_warning_on_write_ioerror(self, caplog):
        """When write() raises IOError, a warning is logged."""
        with patch.object(pipe, "FAILURE_REPORT", "/dev/null"):
            mock_open = patch(
                "builtins.open",
                side_effect=IOError("disk full"),
            )
            with mock_open, caplog.at_level(logging.WARNING, logger="ocr_pipeline"):
                pipe.log_failure("report.pdf", 5, "ocr failed")

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) >= 1
        assert "disk full" in warning_msgs[0].message
        assert "report.pdf" in warning_msgs[0].message
        assert "p5" in warning_msgs[0].message

    def test_does_not_raise_on_write_error(self):
        """log_failure must never propagate exceptions to the caller."""
        bad_path = "/nonexistent/dir/failures.csv"
        with patch.object(pipe, "FAILURE_REPORT", bad_path):
            # Should not raise
            pipe.log_failure("test.pdf", 1, "some error")

    def test_no_warning_on_successful_write(self, tmp_path, caplog):
        """Happy path must not emit any warnings."""
        report = tmp_path / "failures.csv"
        with patch.object(pipe, "FAILURE_REPORT", str(report)):
            with caplog.at_level(logging.WARNING, logger="ocr_pipeline"):
                pipe.log_failure("good.pdf", 1, "expected error")

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) == 0

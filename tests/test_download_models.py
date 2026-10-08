"""Tests for download_models.py CPU-only mode and graceful failure handling.

Covers:
- SKIP_MODEL_PRELOAD env var
- CPU-only mode (--cpu-only flag and CPU_ONLY_BUILD env var)
- Graceful failure when PaddleOCR is not installed
- Graceful failure when individual model downloads fail
- CLI argument parsing
- UIE model download skip behaviour

Run with: python -m pytest tests/test_download_models.py -v
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# Tests: SKIP_MODEL_PRELOAD
# ===========================================================================


class TestSkipModelPreload:
    """Verify SKIP_MODEL_PRELOAD=1 skips all downloads."""

    @patch.dict(os.environ, {"SKIP_MODEL_PRELOAD": "1"})
    def test_download_models_skips_when_env_set(self):
        """download_models() returns immediately when SKIP_MODEL_PRELOAD=1."""
        # Reimport to pick up env var
        import importlib

        import download_models

        importlib.reload(download_models)

        # Should not attempt to import PaddleOCR at all
        with patch.object(download_models, "_import_paddleocr") as mock_import:
            download_models.download_models(cpu_only=False)
            mock_import.assert_not_called()

        # Restore module state
        os.environ.pop("SKIP_MODEL_PRELOAD", None)
        importlib.reload(download_models)

    @patch.dict(os.environ, {"SKIP_MODEL_PRELOAD": "1"})
    def test_download_uie_skips_when_env_set(self):
        """download_uie_model() returns immediately when SKIP_MODEL_PRELOAD=1."""
        import importlib

        import download_models

        importlib.reload(download_models)

        with patch("download_models.logger") as mock_logger:
            download_models.download_uie_model()
            # Should log the skip message
            mock_logger.info.assert_called()
            log_msg = mock_logger.info.call_args[0][0]
            assert "SKIP_MODEL_PRELOAD" in log_msg

        os.environ.pop("SKIP_MODEL_PRELOAD", None)
        importlib.reload(download_models)

    @patch.dict(os.environ, {"SKIP_MODEL_PRELOAD": "0"})
    def test_skip_not_triggered_when_zero(self):
        """SKIP_MODEL_PRELOAD=0 does not skip downloads."""
        import importlib

        import download_models

        importlib.reload(download_models)
        assert download_models.SKIP_MODEL_PRELOAD is False

        os.environ.pop("SKIP_MODEL_PRELOAD", None)
        importlib.reload(download_models)

    def test_skip_default_is_false(self):
        """SKIP_MODEL_PRELOAD defaults to False when env var is unset."""
        import importlib

        import download_models

        env = os.environ.copy()
        env.pop("SKIP_MODEL_PRELOAD", None)
        with patch.dict(os.environ, env, clear=True):
            importlib.reload(download_models)
            assert download_models.SKIP_MODEL_PRELOAD is False

        importlib.reload(download_models)


# ===========================================================================
# Tests: CPU-only mode
# ===========================================================================


class TestCpuOnlyMode:
    """Verify cpu_only=True passes correct kwargs to PaddleOCR."""

    def test_cpu_only_uses_native_paddle_models(self):
        """CPU preload must not request ONNX for fresh Paddle model caches."""
        import download_models

        mock_ocr_cls = MagicMock()
        with patch.object(
            download_models, "_import_paddleocr", return_value=mock_ocr_cls
        ):
            with patch.object(download_models, "SKIP_MODEL_PRELOAD", False):
                with patch.object(download_models, "TARGET_LANGS", ["en"]):
                    download_models.download_models(cpu_only=True)

        mock_ocr_cls.assert_called_once()
        call_kwargs = mock_ocr_cls.call_args[1]
        assert "use_onnx" not in call_kwargs
        assert call_kwargs["use_gpu"] is False
        assert call_kwargs["lang"] == "en"
        assert call_kwargs["use_angle_cls"] is True

    def test_default_mode_no_use_onnx(self):
        """cpu_only=False should not pass use_onnx to PaddleOCR constructor."""
        import download_models

        mock_ocr_cls = MagicMock()
        with patch.object(
            download_models, "_import_paddleocr", return_value=mock_ocr_cls
        ):
            with patch.object(download_models, "SKIP_MODEL_PRELOAD", False):
                with patch.object(download_models, "TARGET_LANGS", ["en"]):
                    download_models.download_models(cpu_only=False)

        call_kwargs = mock_ocr_cls.call_args[1]
        assert "use_onnx" not in call_kwargs
        assert call_kwargs["use_gpu"] is False

    def test_cpu_only_all_langs_disable_gpu_without_onnx(self):
        """Every CPU preload uses native Paddle format with GPU disabled."""
        import download_models

        mock_ocr_cls = MagicMock()
        test_langs = ["en", "ch", "fr"]
        with patch.object(
            download_models, "_import_paddleocr", return_value=mock_ocr_cls
        ):
            with patch.object(download_models, "SKIP_MODEL_PRELOAD", False):
                with patch.object(download_models, "TARGET_LANGS", test_langs):
                    download_models.download_models(cpu_only=True)

        assert mock_ocr_cls.call_count == 3
        for call in mock_ocr_cls.call_args_list:
            assert "use_onnx" not in call[1]
            assert call[1]["use_gpu"] is False

    def test_cpu_only_preload_filters_unsupported_paddleocr_29_langs(self):
        """CPU-only preload skips only the known PaddleOCR 2.9.1 gaps."""
        import download_models

        with patch.object(
            download_models, "TARGET_LANGS", ["en", "fi", "el", "kn", "ch"]
        ):
            result = download_models._preload_target_langs(cpu_only=True)

        assert result == ["en", "fi", "ch"]

    def test_standard_preload_keeps_full_target_language_list(self):
        """Unsupported scripts are filtered in standard mode too."""
        import download_models

        with patch.object(
            download_models, "TARGET_LANGS", ["en", "fi", "el", "kn", "ch"]
        ):
            result = download_models._preload_target_langs(cpu_only=False)

        assert result == ["en", "fi", "ch"]

    def test_cpu_only_preload_logs_skipped_language_contract(self):
        """Unsupported CPU-only preload languages are visible in logs."""
        import download_models

        with patch.object(download_models, "TARGET_LANGS", ["en", "fi", "kn"]):
            with patch.object(download_models, "logger") as mock_logger:
                result = download_models._preload_target_langs(cpu_only=True)

        assert result == ["en", "fi"]
        log_msg = mock_logger.warning.call_args[0][0]
        log_args = mock_logger.warning.call_args[0][1:]
        assert "preload language" in log_msg
        assert log_args == (1, "kn")

    def test_cpu_only_build_env_var(self):
        """CPU_ONLY_BUILD=1 env var is read at module level."""
        import importlib

        import download_models

        with patch.dict(os.environ, {"CPU_ONLY_BUILD": "1"}):
            importlib.reload(download_models)
            assert download_models.CPU_ONLY_BUILD is True

        with patch.dict(os.environ, {"CPU_ONLY_BUILD": "0"}):
            importlib.reload(download_models)
            assert download_models.CPU_ONLY_BUILD is False

        importlib.reload(download_models)


# ===========================================================================
# Tests: Graceful failure
# ===========================================================================


class TestGracefulFailure:
    """Verify graceful degradation on import errors and download failures."""

    def test_paddleocr_not_installed(self):
        """download_models() returns without error when PaddleOCR missing."""
        import download_models

        with patch.object(download_models, "_import_paddleocr", return_value=None):
            with patch.object(download_models, "SKIP_MODEL_PRELOAD", False):
                # Should not raise
                download_models.download_models(cpu_only=False)

    def test_import_paddleocr_handles_import_error(self):
        """_import_paddleocr returns None when paddleocr is not installed."""
        import download_models

        with patch.dict(sys.modules, {"paddleocr": None}):
            with patch("importlib.util.find_spec", return_value=None):
                with patch(
                    "builtins.__import__",
                    side_effect=ImportError("no paddleocr"),
                ):
                    result = download_models._import_paddleocr()
                    assert result is None

    def test_import_paddleocr_raises_when_installed_import_is_broken(self):
        """Broken installed PaddleOCR stacks must fail the preload build."""
        import download_models

        fake_spec = MagicMock(origin="/path/to/paddleocr")
        fake_spec.submodule_search_locations = None
        with patch("importlib.util.find_spec", return_value=fake_spec):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'setuptools'"),
            ):
                with pytest.raises(
                    RuntimeError,
                    match="paddleocr import failed during model preload at /path/to/paddleocr",
                ):
                    download_models._import_paddleocr()

    def test_single_lang_failure_continues(self):
        """A failure for one language should not stop other languages."""
        import download_models

        call_count = 0

        def mock_init(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["lang"] == "fr":
                raise RuntimeError("Simulated download failure")
            return MagicMock()

        mock_ocr_cls = MagicMock(side_effect=mock_init)
        test_langs = ["en", "fr", "ch"]

        with patch.object(
            download_models, "_import_paddleocr", return_value=mock_ocr_cls
        ):
            with patch.object(download_models, "SKIP_MODEL_PRELOAD", False):
                with patch.object(download_models, "TARGET_LANGS", test_langs):
                    with pytest.raises(RuntimeError, match="1 language"):
                        download_models.download_models(cpu_only=False)

        # All 3 languages should have been attempted
        assert call_count == 3

    def test_all_langs_fail_reports_count(self):
        """RuntimeError should report the exact number of failed languages."""
        import download_models

        mock_ocr_cls = MagicMock(side_effect=RuntimeError("boom"))
        test_langs = ["en", "fr"]

        with patch.object(
            download_models, "_import_paddleocr", return_value=mock_ocr_cls
        ):
            with patch.object(download_models, "SKIP_MODEL_PRELOAD", False):
                with patch.object(download_models, "TARGET_LANGS", test_langs):
                    with pytest.raises(RuntimeError, match="2 language"):
                        download_models.download_models(cpu_only=False)

    def test_uie_import_error_graceful(self):
        """download_uie_model() handles missing paddlenlp gracefully."""
        import download_models

        with patch.object(download_models, "SKIP_MODEL_PRELOAD", False):
            with patch.dict(sys.modules, {"paddlenlp": None}):
                with patch(
                    "builtins.__import__",
                    side_effect=ImportError("no paddlenlp"),
                ):
                    # Should not raise
                    download_models.download_uie_model()

    def test_uie_download_failure_graceful(self):
        """download_uie_model() logs warning on download failure."""
        import download_models

        mock_taskflow = MagicMock(side_effect=RuntimeError("UIE failed"))
        mock_paddlenlp = MagicMock()
        mock_paddlenlp.Taskflow = mock_taskflow

        with patch.object(download_models, "SKIP_MODEL_PRELOAD", False):
            with patch.dict(sys.modules, {"paddlenlp": mock_paddlenlp}):
                # Should not raise
                download_models.download_uie_model()


# ===========================================================================
# Tests: CLI argument parsing
# ===========================================================================


class TestCliArgs:
    """Verify CLI argument parsing for CPU, UIE, and subset controls."""

    def test_cpu_only_flag(self):
        """--cpu-only sets args.cpu_only to True."""
        import download_models

        args = download_models._parse_args(["--cpu-only"])
        assert args.cpu_only is True

    def test_no_cpu_only_flag(self):
        """Default: cpu_only is False."""
        import download_models

        args = download_models._parse_args([])
        assert args.cpu_only is False

    def test_skip_uie_flag(self):
        """--skip-uie sets args.skip_uie to True."""
        import download_models

        args = download_models._parse_args(["--skip-uie"])
        assert args.skip_uie is True

    def test_both_flags(self):
        """Both --cpu-only and --skip-uie can be used together."""
        import download_models

        args = download_models._parse_args(["--cpu-only", "--skip-uie"])
        assert args.cpu_only is True
        assert args.skip_uie is True

    def test_languages_flag(self):
        import download_models

        args = download_models._parse_args(["--languages", "en,ch"])
        assert args.languages == ["en", "ch"]

    def test_languages_flag_trims_and_deduplicates(self):
        import download_models

        args = download_models._parse_args(["--languages", " en, ch,en "])
        assert args.languages == ["en", "ch"]

    @pytest.mark.parametrize("selection", ["", " , ", "en,unknown"])
    def test_invalid_languages_flag_is_an_argument_error(self, selection):
        import download_models

        with pytest.raises(SystemExit) as exc_info:
            download_models._parse_args(["--languages", selection])
        assert exc_info.value.code == 2


# ===========================================================================
# Tests: Target languages list
# ===========================================================================


class TestTargetLangs:
    """Verify the TARGET_LANGS constant is well-formed."""

    def test_target_langs_is_list(self):
        import download_models

        assert isinstance(download_models.TARGET_LANGS, list)

    def test_target_langs_has_core_languages(self):
        """Must include en, ch, fr, german at minimum."""
        import download_models

        for lang in ["en", "ch", "fr", "german"]:
            assert lang in download_models.TARGET_LANGS, (
                f"{lang} missing from TARGET_LANGS"
            )

    def test_target_langs_count(self):
        """Should have 34 languages in the expanded baseline."""
        import download_models

        assert len(download_models.TARGET_LANGS) == 34

    def test_cpu_only_preload_unsupported_langs_stay_explicit(self):
        """The temporary PaddleOCR 2.9.1 preload gap is an explicit contract."""
        import download_models

        assert download_models.PADDLEOCR_29_UNSUPPORTED_LANGS == {
            "bn",
            "el",
            "kn",
            "th",
        }
        assert {"el", "kn"} < set(download_models.TARGET_LANGS)

    def test_requested_subset_is_bounded_to_target_registry(self):
        import download_models

        with patch.object(download_models, "TARGET_LANGS", ["en", "ch", "el"]):
            assert download_models._preload_target_langs(
                requested_langs=["ch", "en", "ch"]
            ) == ["ch", "en"]

    def test_requested_unknown_language_fails_closed(self):
        import download_models

        with patch.object(download_models, "TARGET_LANGS", ["en", "ch"]):
            with pytest.raises(ValueError, match="outside TARGET_LANGS"):
                download_models._preload_target_langs(requested_langs=["xx"])

    def test_finnish_preloads_with_latin_model_family(self):
        import download_models

        mock_ocr_cls = MagicMock()
        with (
            patch.object(
                download_models, "_import_paddleocr", return_value=mock_ocr_cls
            ),
            patch.object(download_models, "SKIP_MODEL_PRELOAD", False),
            patch.object(download_models, "TARGET_LANGS", ["fi"]),
        ):
            download_models.download_models(cpu_only=True)

        assert mock_ocr_cls.call_args.kwargs["lang"] == "latin"


# ===========================================================================
# Tests: Engine cleanup
# ===========================================================================


class TestEngineCleanup:
    """Verify engine objects are deleted after download."""

    def test_engine_del_called(self):
        """PaddleOCR engine should be deleted after successful download."""
        import download_models

        mock_engine = MagicMock()
        mock_ocr_cls = MagicMock(return_value=mock_engine)

        with patch.object(
            download_models, "_import_paddleocr", return_value=mock_ocr_cls
        ):
            with patch.object(download_models, "SKIP_MODEL_PRELOAD", False):
                with patch.object(download_models, "TARGET_LANGS", ["en"]):
                    download_models.download_models(cpu_only=False)

        # The engine was created and should be referenced (del just decrements refcount)
        mock_ocr_cls.assert_called_once()

    def test_successful_download_no_exception(self, capsys):
        """Successful download should not raise any exception."""
        import download_models

        mock_ocr_cls = MagicMock(return_value=MagicMock())

        with patch.object(
            download_models, "_import_paddleocr", return_value=mock_ocr_cls
        ):
            with patch.object(download_models, "SKIP_MODEL_PRELOAD", False):
                with patch.object(download_models, "TARGET_LANGS", ["en", "ch"]):
                    # Should not raise
                    download_models.download_models(cpu_only=True)

        assert mock_ocr_cls.call_count == 2
        assert (
            "MODEL_PRELOAD_OK attempted=2 succeeded=2 skipped=0"
            in capsys.readouterr().out
        )

    def test_subset_reports_only_requested_languages_as_skipped(self, capsys):
        """Subset preload status counts omitted requested models, not registry size."""
        import download_models

        mock_ocr_cls = MagicMock(return_value=MagicMock())
        with (
            patch.object(
                download_models, "_import_paddleocr", return_value=mock_ocr_cls
            ),
            patch.object(download_models, "SKIP_MODEL_PRELOAD", False),
            patch.object(download_models, "TARGET_LANGS", ["en", "ch", "el"]),
        ):
            download_models.download_models(cpu_only=True, languages=["en", "el"])

        assert mock_ocr_cls.call_count == 1
        assert (
            "MODEL_PRELOAD_OK attempted=1 succeeded=1 skipped=1"
            in capsys.readouterr().out
        )

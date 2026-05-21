"""Tests for OCR inference backend selection (ocr_inference_backend.py).

Tests cover:
- Default backend resolution (paddle)
- Explicit backend selection (onnx, openvino)
- Invalid backend fallback to paddle
- Auto-detection priority (openvino > onnx > paddle)
- Engine creation with backend-specific kwargs
- ONNX backend skipped on GPU device
- Backend info reporting with optional dependency probing
- OpenVINO environment variable setup
- TypeError fallback for older PaddleOCR versions

Run with: python -m pytest tests/test_ocr_inference_backend.py -v
"""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch


class TestGetBackend:
    """Tests for backend resolution logic."""

    def test_default_is_paddle(self):
        """Default backend is 'paddle' when env var not set."""
        import ocr_inference_backend

        with patch.object(ocr_inference_backend, "INFERENCE_BACKEND", "paddle"):
            assert ocr_inference_backend.get_backend() == "paddle"

    def test_onnx_backend(self):
        """Explicit onnx backend is returned as-is."""
        import ocr_inference_backend

        with patch.object(ocr_inference_backend, "INFERENCE_BACKEND", "onnx"):
            assert ocr_inference_backend.get_backend() == "onnx"

    def test_openvino_backend(self):
        """Explicit openvino backend is returned as-is."""
        import ocr_inference_backend

        with patch.object(ocr_inference_backend, "INFERENCE_BACKEND", "openvino"):
            assert ocr_inference_backend.get_backend() == "openvino"

    def test_invalid_backend_falls_back_to_paddle(self):
        """Unknown backend name falls back to paddle with warning."""
        import ocr_inference_backend

        with patch.object(ocr_inference_backend, "INFERENCE_BACKEND", "invalid"):
            assert ocr_inference_backend.get_backend() == "paddle"

    def test_empty_string_falls_back_to_paddle(self):
        """Empty string backend falls back to paddle."""
        import ocr_inference_backend

        with patch.object(ocr_inference_backend, "INFERENCE_BACKEND", ""):
            assert ocr_inference_backend.get_backend() == "paddle"

    def test_auto_delegates_to_detect(self):
        """Auto backend delegates to _detect_best_backend."""
        import ocr_inference_backend

        with patch.object(ocr_inference_backend, "INFERENCE_BACKEND", "auto"):
            with patch.object(
                ocr_inference_backend, "_detect_best_backend", return_value="onnx"
            ):
                assert ocr_inference_backend.get_backend() == "onnx"


class TestDetectBestBackend:
    """Tests for auto-detection priority logic."""

    def test_auto_detects_openvino_first(self):
        """OpenVINO is preferred when both openvino and onnxruntime are available."""
        import ocr_inference_backend

        mock_openvino = MagicMock()
        mock_ort = MagicMock()
        with patch.dict(
            "sys.modules", {"openvino": mock_openvino, "onnxruntime": mock_ort}
        ):
            assert ocr_inference_backend._detect_best_backend() == "openvino"

    def test_auto_detects_onnx_when_no_openvino(self):
        """ONNX Runtime is selected when openvino is not importable."""
        import ocr_inference_backend

        # Temporarily remove openvino from sys.modules and make import fail
        saved = sys.modules.pop("openvino", None)
        try:
            mock_ort = MagicMock()

            def fake_import(name, *args, **kwargs):
                if name == "openvino":
                    raise ImportError("no openvino")
                if name == "onnxruntime":
                    return mock_ort
                return original_import(name, *args, **kwargs)

            import builtins

            original_import = builtins.__import__
            with patch.object(builtins, "__import__", side_effect=fake_import):
                result = ocr_inference_backend._detect_best_backend()
                assert result == "onnx"
        finally:
            if saved is not None:
                sys.modules["openvino"] = saved

    def test_auto_falls_back_to_paddle(self):
        """Falls back to paddle when neither openvino nor onnxruntime are available."""
        import ocr_inference_backend

        saved_ov = sys.modules.pop("openvino", None)
        saved_ort = sys.modules.pop("onnxruntime", None)
        try:
            import builtins

            original_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name in ("openvino", "onnxruntime"):
                    raise ImportError(f"no {name}")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=fake_import):
                result = ocr_inference_backend._detect_best_backend()
                assert result == "paddle"
        finally:
            if saved_ov is not None:
                sys.modules["openvino"] = saved_ov
            if saved_ort is not None:
                sys.modules["onnxruntime"] = saved_ort


class TestCreateOcrEngine:
    """Tests for engine creation with backend selection."""

    def test_creates_paddle_engine(self):
        """Paddle backend creates engine without use_onnx kwarg (native paddle)."""
        import ocr_inference_backend

        mock_paddle_mod = MagicMock()
        with patch.object(ocr_inference_backend, "get_backend", return_value="paddle"):
            with patch.dict("sys.modules", {"paddleocr": mock_paddle_mod}):
                # Force re-import of paddleocr in create_ocr_engine
                ocr_inference_backend.create_ocr_engine("en", "cpu")
                mock_paddle_mod.PaddleOCR.assert_called_once()
                call_kwargs = mock_paddle_mod.PaddleOCR.call_args[1]
                assert call_kwargs["lang"] == "en"
                assert call_kwargs["use_gpu"] is False
                assert "use_onnx" not in call_kwargs

    def test_creates_onnx_engine_on_cpu(self):
        """ONNX backend sets use_onnx=True when device is cpu."""
        import ocr_inference_backend

        mock_paddle_mod = MagicMock()
        with patch.object(ocr_inference_backend, "get_backend", return_value="onnx"):
            with patch.dict("sys.modules", {"paddleocr": mock_paddle_mod}):
                ocr_inference_backend.create_ocr_engine("en", "cpu")
                call_kwargs = mock_paddle_mod.PaddleOCR.call_args[1]
                assert call_kwargs["use_onnx"] is True
                assert call_kwargs["lang"] == "en"
                assert call_kwargs["use_gpu"] is False

    def test_onnx_not_used_on_gpu(self):
        """ONNX backend is skipped when device is GPU -- native paddle used instead."""
        import ocr_inference_backend

        mock_paddle_mod = MagicMock()
        with patch.object(ocr_inference_backend, "get_backend", return_value="onnx"):
            with patch.dict("sys.modules", {"paddleocr": mock_paddle_mod}):
                ocr_inference_backend.create_ocr_engine("en", "gpu")
                call_kwargs = mock_paddle_mod.PaddleOCR.call_args[1]
                assert "use_onnx" not in call_kwargs

    def test_openvino_sets_onnx_on_cpu(self):
        """OpenVINO backend sets use_onnx=True on CPU device."""
        import ocr_inference_backend

        mock_paddle_mod = MagicMock()
        with patch.object(
            ocr_inference_backend, "get_backend", return_value="openvino"
        ):
            with patch.dict("sys.modules", {"paddleocr": mock_paddle_mod}):
                ocr_inference_backend.create_ocr_engine("fr", "cpu")
                call_kwargs = mock_paddle_mod.PaddleOCR.call_args[1]
                assert call_kwargs["use_onnx"] is True

    def test_openvino_not_used_on_gpu(self):
        """OpenVINO backend is skipped when device is GPU."""
        import ocr_inference_backend

        mock_paddle_mod = MagicMock()
        with patch.object(
            ocr_inference_backend, "get_backend", return_value="openvino"
        ):
            with patch.dict("sys.modules", {"paddleocr": mock_paddle_mod}):
                ocr_inference_backend.create_ocr_engine("en", "gpu")
                call_kwargs = mock_paddle_mod.PaddleOCR.call_args[1]
                assert "use_onnx" not in call_kwargs

    def test_type_error_fallback(self):
        """Falls back to minimal kwargs when PaddleOCR raises TypeError."""
        import ocr_inference_backend

        mock_paddle_mod = MagicMock()
        # First call raises TypeError, second call succeeds
        mock_paddle_mod.PaddleOCR.side_effect = [TypeError("bad kwarg"), MagicMock()]
        with patch.object(ocr_inference_backend, "get_backend", return_value="paddle"):
            with patch.dict("sys.modules", {"paddleocr": mock_paddle_mod}):
                ocr_inference_backend.create_ocr_engine("en", "cpu")
                assert mock_paddle_mod.PaddleOCR.call_count == 2
                # Second call should use minimal kwargs
                fallback_kwargs = mock_paddle_mod.PaddleOCR.call_args_list[1][1]
                assert fallback_kwargs["lang"] == "en"
                assert fallback_kwargs["use_gpu"] is False
                assert fallback_kwargs["use_angle_cls"] is True

    def test_type_error_fallback_preserves_onnx(self):
        """TypeError fallback preserves use_onnx when backend is onnx."""
        import ocr_inference_backend

        mock_paddle_mod = MagicMock()
        mock_paddle_mod.PaddleOCR.side_effect = [TypeError("bad kwarg"), MagicMock()]
        with patch.object(ocr_inference_backend, "get_backend", return_value="onnx"):
            with patch.dict("sys.modules", {"paddleocr": mock_paddle_mod}):
                ocr_inference_backend.create_ocr_engine("en", "cpu")
                fallback_kwargs = mock_paddle_mod.PaddleOCR.call_args_list[1][1]
                assert fallback_kwargs["use_onnx"] is True

    def test_shared_base_config(self):
        """All backends share the same base configuration keys."""
        import ocr_inference_backend

        mock_paddle_mod = MagicMock()
        with patch.object(ocr_inference_backend, "get_backend", return_value="paddle"):
            with patch.dict("sys.modules", {"paddleocr": mock_paddle_mod}):
                ocr_inference_backend.create_ocr_engine("ch", "gpu")
                call_kwargs = mock_paddle_mod.PaddleOCR.call_args[1]
                assert call_kwargs["use_angle_cls"] is True
                assert call_kwargs["show_log"] is False
                assert call_kwargs["lang"] == "ch"
                assert call_kwargs["use_gpu"] is True


class TestGetBackendInfo:
    """Tests for backend info reporting."""

    def test_returns_dict_with_required_keys(self):
        """Info dict always contains the required keys."""
        import ocr_inference_backend

        info = ocr_inference_backend.get_backend_info()
        assert "backend" in info
        assert "configured" in info
        assert "onnx_available" in info
        assert "openvino_available" in info

    def test_detects_onnx_availability(self):
        """Reports onnx_available=True and version when onnxruntime is importable."""
        import ocr_inference_backend

        mock_ort = MagicMock()
        mock_ort.__version__ = "1.17.0"
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            info = ocr_inference_backend.get_backend_info()
            assert info["onnx_available"] is True
            assert info["onnx_version"] == "1.17.0"

    def test_detects_openvino_availability(self):
        """Reports openvino_available=True and version when openvino is importable."""
        import ocr_inference_backend

        mock_ov = MagicMock()
        mock_ov.__version__ = "2024.0.0"
        with patch.dict("sys.modules", {"openvino": mock_ov}):
            info = ocr_inference_backend.get_backend_info()
            assert info["openvino_available"] is True
            assert info["openvino_version"] == "2024.0.0"

    def test_reports_configured_value(self):
        """Info dict reflects the raw configured env var value."""
        import ocr_inference_backend

        with patch.object(ocr_inference_backend, "INFERENCE_BACKEND", "auto"):
            with patch.object(
                ocr_inference_backend, "_detect_best_backend", return_value="paddle"
            ):
                info = ocr_inference_backend.get_backend_info()
                assert info["configured"] == "auto"
                assert info["backend"] == "paddle"

    def test_unavailable_packages_reported_false(self):
        """When optional packages are not importable, availability is False."""
        import ocr_inference_backend

        saved_ov = sys.modules.pop("openvino", None)
        saved_ort = sys.modules.pop("onnxruntime", None)
        try:
            import builtins

            original_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name in ("openvino", "onnxruntime"):
                    raise ImportError(f"no {name}")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=fake_import):
                info = ocr_inference_backend.get_backend_info()
                assert info["onnx_available"] is False
                assert info["openvino_available"] is False
                assert "onnx_version" not in info
                assert "openvino_version" not in info
        finally:
            if saved_ov is not None:
                sys.modules["openvino"] = saved_ov
            if saved_ort is not None:
                sys.modules["onnxruntime"] = saved_ort


class TestModuleConstants:
    """Tests for module-level constants and environment variable handling."""

    def test_valid_backends_tuple(self):
        """VALID_BACKENDS contains all supported backend names."""
        import ocr_inference_backend

        assert "paddle" in ocr_inference_backend.VALID_BACKENDS
        assert "onnx" in ocr_inference_backend.VALID_BACKENDS
        assert "openvino" in ocr_inference_backend.VALID_BACKENDS
        assert "auto" in ocr_inference_backend.VALID_BACKENDS
        assert len(ocr_inference_backend.VALID_BACKENDS) == 4

    def test_env_var_controls_backend(self):
        """OCR_INFERENCE_BACKEND env var is read at module load time."""
        with patch.dict(os.environ, {"OCR_INFERENCE_BACKEND": "onnx"}):
            # Force module reload to pick up env var
            import ocr_inference_backend

            importlib.reload(ocr_inference_backend)
            assert ocr_inference_backend.INFERENCE_BACKEND == "onnx"

        # Restore default
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCR_INFERENCE_BACKEND", None)
            importlib.reload(ocr_inference_backend)

    def test_env_var_strips_whitespace(self):
        """Backend env var value is stripped of whitespace."""
        with patch.dict(os.environ, {"OCR_INFERENCE_BACKEND": "  onnx  "}):
            import ocr_inference_backend

            importlib.reload(ocr_inference_backend)
            assert ocr_inference_backend.INFERENCE_BACKEND == "onnx"

        # Restore default
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCR_INFERENCE_BACKEND", None)
            importlib.reload(ocr_inference_backend)

    def test_env_var_case_insensitive(self):
        """Backend env var value is lowercased."""
        with patch.dict(os.environ, {"OCR_INFERENCE_BACKEND": "ONNX"}):
            import ocr_inference_backend

            importlib.reload(ocr_inference_backend)
            assert ocr_inference_backend.INFERENCE_BACKEND == "onnx"

        # Restore default
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCR_INFERENCE_BACKEND", None)
            importlib.reload(ocr_inference_backend)


class TestOcrUtilsDelegation:
    """Tests that ocr_distributed.ocr_utils delegates to the backend module."""

    def test_ocr_utils_delegates_to_backend(self):
        """create_paddle_engine in ocr_utils delegates to ocr_inference_backend."""
        mock_engine = MagicMock()
        mock_backend_mod = MagicMock()
        mock_backend_mod.create_ocr_engine.return_value = mock_engine

        with patch.dict("sys.modules", {"ocr_inference_backend": mock_backend_mod}):
            from ocr_distributed.ocr_utils import create_paddle_engine

            # Force the function to re-execute the import by calling it
            result = create_paddle_engine("en", "cpu")
            mock_backend_mod.create_ocr_engine.assert_called_once_with("en", "cpu")
            assert result is mock_engine

    def test_ocr_utils_fallback_without_backend(self):
        """create_paddle_engine falls back to direct PaddleOCR when backend missing."""
        mock_paddle_mod = MagicMock()
        mock_engine = MagicMock()
        mock_paddle_mod.PaddleOCR.return_value = mock_engine

        # Ensure the backend module import fails
        with patch.dict(
            "sys.modules",
            {"ocr_inference_backend": None, "paddleocr": mock_paddle_mod},
        ):
            from ocr_distributed.ocr_utils import create_paddle_engine

            result = create_paddle_engine("en", "cpu")
            mock_paddle_mod.PaddleOCR.assert_called_once()
            assert result is mock_engine

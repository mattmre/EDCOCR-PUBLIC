"""Tests for PaddlePaddle version compatibility shim (paddle_compat.py).

Tests cover:
- Version detection (get_paddle_version, is_paddle_v3)
- Device normalization for PaddleOCR 2.x
- OCR engine kwargs generation (use_gpu, use_onnx, use_angle_cls)
- Result format normalization (v2 list-of-list, defensive dict handling)
- Structure engine class resolution (PPStructure only, v2)
- Edge cases: missing paddle, malformed versions, empty results

Run with: python -m pytest tests/test_paddle_compat.py -v
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

import paddle_compat


class TestGetPaddleVersion:
    """Tests for get_paddle_version() version detection."""

    def test_returns_2_for_paddle_2x(self):
        """PaddlePaddle 2.6.2 returns major version 2."""
        mock_paddle = MagicMock()
        mock_paddle.__version__ = "2.6.2"
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            assert paddle_compat.get_paddle_version() == 2

    def test_returns_3_for_paddle_3x(self):
        """PaddlePaddle 3.0.0 returns major version 3."""
        mock_paddle = MagicMock()
        mock_paddle.__version__ = "3.0.0"
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            assert paddle_compat.get_paddle_version() == 3

    def test_returns_3_for_paddle_3_1(self):
        """PaddlePaddle 3.1.0 returns major version 3."""
        mock_paddle = MagicMock()
        mock_paddle.__version__ = "3.1.0"
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            assert paddle_compat.get_paddle_version() == 3

    def test_returns_none_when_paddle_not_installed(self):
        """Returns None when paddle cannot be imported."""
        saved = sys.modules.pop("paddle", None)
        try:
            import builtins

            original_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "paddle":
                    raise ImportError("No module named 'paddle'")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=fake_import):
                assert paddle_compat.get_paddle_version() is None
        finally:
            if saved is not None:
                sys.modules["paddle"] = saved

    def test_returns_none_for_malformed_version(self):
        """Returns None when version string is not parseable."""
        mock_paddle = MagicMock()
        mock_paddle.__version__ = "not-a-version"
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            assert paddle_compat.get_paddle_version() is None

    def test_returns_zero_when_version_attr_missing(self):
        """Returns 0 when paddle has no __version__ attribute."""
        mock_paddle = MagicMock(spec=[])  # No attributes
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            # getattr returns default "0.0.0", which parses to 0
            result = paddle_compat.get_paddle_version()
            assert result == 0

    def test_returns_none_for_empty_version_string(self):
        """Returns None when version string is empty."""
        mock_paddle = MagicMock()
        mock_paddle.__version__ = ""
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            assert paddle_compat.get_paddle_version() is None


class TestIsPaddleV3:
    """Tests for is_paddle_v3() -- always returns False for v2-only module."""

    def test_always_returns_false(self):
        """is_paddle_v3() always returns False regardless of installed version."""
        assert paddle_compat.is_paddle_v3() is False

    def test_returns_false_even_with_v3_installed(self):
        """Returns False even when PaddlePaddle 3.x is detected by get_paddle_version."""
        mock_paddle = MagicMock()
        mock_paddle.__version__ = "3.0.0"
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            assert paddle_compat.is_paddle_v3() is False

    def test_returns_false_for_v2(self):
        """Returns False when PaddlePaddle 2.x is detected."""
        mock_paddle = MagicMock()
        mock_paddle.__version__ = "2.6.2"
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            assert paddle_compat.is_paddle_v3() is False

    def test_returns_false_when_not_installed(self):
        """Returns False when PaddlePaddle is not installed."""
        assert paddle_compat.is_paddle_v3() is False


class TestNormalizeDevice:
    """Tests for normalize_device() -- always returns None for v2."""

    def test_returns_none_for_gpu(self):
        """PaddleOCR 2.x returns None (caller should use use_gpu=True)."""
        assert paddle_compat.normalize_device(use_gpu=True) is None

    def test_returns_none_for_cpu(self):
        """PaddleOCR 2.x returns None for CPU too."""
        assert paddle_compat.normalize_device(use_gpu=False) is None

    def test_default_is_gpu_still_returns_none(self):
        """Default argument is use_gpu=True, but always returns None."""
        assert paddle_compat.normalize_device() is None


class TestGetOcrEngineKwargs:
    """Tests for get_ocr_engine_kwargs() v2 constructor args."""

    def test_includes_use_gpu_true(self):
        """kwargs include use_gpu=True, no device key."""
        kwargs = paddle_compat.get_ocr_engine_kwargs(use_gpu=True, lang="en")
        assert kwargs["use_gpu"] is True
        assert kwargs["lang"] == "en"
        assert kwargs["show_log"] is False
        assert "device" not in kwargs
        assert "enable_hpi" not in kwargs

    def test_includes_use_gpu_false(self):
        """kwargs include use_gpu=False for CPU."""
        kwargs = paddle_compat.get_ocr_engine_kwargs(use_gpu=False, lang="ch")
        assert kwargs["use_gpu"] is False
        assert kwargs["lang"] == "ch"

    def test_use_onnx_adds_key_when_true(self):
        """use_onnx=True adds use_onnx=True to kwargs."""
        kwargs = paddle_compat.get_ocr_engine_kwargs(
            use_gpu=False, lang="en", use_onnx=True
        )
        assert kwargs["use_onnx"] is True

    def test_use_onnx_omitted_when_false(self):
        """use_onnx=False does not add use_onnx to kwargs."""
        kwargs = paddle_compat.get_ocr_engine_kwargs(
            use_gpu=True, lang="en", use_onnx=False
        )
        assert "use_onnx" not in kwargs

    def test_includes_use_angle_cls(self):
        """kwargs include use_angle_cls=True by default."""
        kwargs = paddle_compat.get_ocr_engine_kwargs(use_gpu=True, lang="en")
        assert kwargs["use_angle_cls"] is True

    def test_extra_kwargs_passed_through(self):
        """Extra keyword arguments are included in the result."""
        kwargs = paddle_compat.get_ocr_engine_kwargs(
            use_gpu=True,
            lang="en",
            use_doc_orientation_classify=False,
            rec_batch_num=6,
        )
        assert kwargs["use_doc_orientation_classify"] is False
        assert kwargs["rec_batch_num"] == 6

    def test_default_lang_is_en(self):
        """Default language is 'en'."""
        kwargs = paddle_compat.get_ocr_engine_kwargs()
        assert kwargs["lang"] == "en"

    def test_no_device_key(self):
        """v2 kwargs never include a device key."""
        kwargs = paddle_compat.get_ocr_engine_kwargs(use_gpu=True, lang="en")
        assert "device" not in kwargs

    def test_no_enable_hpi_key(self):
        """v2 kwargs never include an enable_hpi key."""
        kwargs = paddle_compat.get_ocr_engine_kwargs(use_gpu=True, lang="en")
        assert "enable_hpi" not in kwargs


class TestNormalizeOcrResult:
    """Tests for normalize_ocr_result() format normalization."""

    def test_v2_format_list_of_list(self):
        """Normalizes PaddleOCR 2.x list-of-list format correctly."""
        v2_result = [
            [
                [[[10, 20], [100, 20], [100, 40], [10, 40]], ("Hello world", 0.95)],
                [[[10, 50], [100, 50], [100, 70], [10, 70]], ("Second line", 0.88)],
            ]
        ]
        normalized = paddle_compat.normalize_ocr_result(v2_result)
        assert len(normalized) == 2
        assert normalized[0][1] == "Hello world"
        assert normalized[0][2] == 0.95
        assert normalized[1][1] == "Second line"
        assert normalized[1][2] == 0.88
        # Boxes are passed through
        assert normalized[0][0] == [[10, 20], [100, 20], [100, 40], [10, 40]]

    def test_dict_format_handled_defensively(self):
        """Defensively handles dict format (not expected in 2.x but supported)."""
        dict_result = [
            {
                "rec_texts": ["Hello world", "Second line"],
                "rec_scores": [0.95, 0.88],
                "dt_polys": [
                    np.array([[10, 20], [100, 20], [100, 40], [10, 40]]),
                    np.array([[10, 50], [100, 50], [100, 70], [10, 70]]),
                ],
            }
        ]
        normalized = paddle_compat.normalize_ocr_result(dict_result)
        assert len(normalized) == 2
        assert normalized[0][1] == "Hello world"
        assert normalized[0][2] == 0.95
        assert normalized[1][1] == "Second line"
        assert normalized[1][2] == 0.88
        # numpy arrays are converted to lists
        assert normalized[0][0] == [[10, 20], [100, 20], [100, 40], [10, 40]]

    def test_dict_format_plain_list_polys(self):
        """Handles dict format with plain list polys (no tolist needed)."""
        dict_result = [
            {
                "rec_texts": ["Hello"],
                "rec_scores": [0.90],
                "dt_polys": [[[10, 20], [100, 20], [100, 40], [10, 40]]],
            }
        ]
        normalized = paddle_compat.normalize_ocr_result(dict_result)
        assert len(normalized) == 1
        assert normalized[0][1] == "Hello"
        assert normalized[0][0] == [[10, 20], [100, 20], [100, 40], [10, 40]]

    def test_none_result(self):
        """Returns empty list for None input."""
        assert paddle_compat.normalize_ocr_result(None) == []

    def test_empty_list_result(self):
        """Returns empty list for empty list input."""
        assert paddle_compat.normalize_ocr_result([]) == []

    def test_empty_nested_list(self):
        """Returns empty list for nested empty list."""
        assert paddle_compat.normalize_ocr_result([[]]) == []

    def test_dict_empty_texts(self):
        """Handles dict format with empty text lists gracefully."""
        dict_result = [
            {
                "rec_texts": [],
                "rec_scores": [],
                "dt_polys": [],
            }
        ]
        normalized = paddle_compat.normalize_ocr_result(dict_result)
        assert normalized == []

    def test_dict_mismatched_lengths(self):
        """Handles dict format where polys are shorter than texts."""
        dict_result = [
            {
                "rec_texts": ["Hello", "World", "Extra"],
                "rec_scores": [0.95, 0.88],
                "dt_polys": [
                    np.array([[10, 20], [100, 40]]),
                ],
            }
        ]
        normalized = paddle_compat.normalize_ocr_result(dict_result)
        assert len(normalized) == 3
        assert normalized[0][1] == "Hello"
        assert normalized[0][2] == 0.95
        assert normalized[1][1] == "World"
        assert normalized[1][2] == 0.88
        # Third entry has no poly and no score
        assert normalized[2][1] == "Extra"
        assert normalized[2][2] == 0.0
        assert normalized[2][0] == []

    def test_v2_single_line(self):
        """Normalizes a single-line v2 result."""
        v2_result = [
            [
                [[[10, 20], [100, 40]], ("Single line", 0.92)],
            ]
        ]
        normalized = paddle_compat.normalize_ocr_result(v2_result)
        assert len(normalized) == 1
        assert normalized[0][1] == "Single line"
        assert normalized[0][2] == 0.92

    def test_v2_malformed_line_skipped(self):
        """Malformed lines in v2 format are silently skipped."""
        v2_result = [
            [
                [[[10, 20], [100, 40]], ("Good line", 0.92)],
                ["not a tuple pair"],  # malformed
                [[[10, 50], [100, 70]], ("Another good", 0.85)],
            ]
        ]
        normalized = paddle_compat.normalize_ocr_result(v2_result)
        assert len(normalized) == 2
        assert normalized[0][1] == "Good line"
        assert normalized[1][1] == "Another good"

    def test_false_result(self):
        """Returns empty list for False/0/empty-string input."""
        assert paddle_compat.normalize_ocr_result(False) == []
        assert paddle_compat.normalize_ocr_result(0) == []
        assert paddle_compat.normalize_ocr_result("") == []


class TestGetStructureEngineClass:
    """Tests for get_structure_engine_class() -- PPStructure only."""

    def test_returns_ppstructure(self):
        """Returns PPStructure class from paddleocr."""
        mock_ppstructure = MagicMock()
        mock_paddleocr = MagicMock()
        mock_paddleocr.PPStructure = mock_ppstructure

        with patch.dict("sys.modules", {"paddleocr": mock_paddleocr}):
            engine_class, engine_type = paddle_compat.get_structure_engine_class()
            assert engine_class is mock_ppstructure
            assert engine_type == "ppstructure"

    def test_ppstructure_import_fails(self):
        """Returns (None, None) when PPStructure cannot be imported."""
        saved = sys.modules.pop("paddleocr", None)
        try:
            import builtins

            original_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "paddleocr":
                    raise ImportError("No paddleocr")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=fake_import):
                engine_class, engine_type = (
                    paddle_compat.get_structure_engine_class()
                )
                assert engine_class is None
                assert engine_type is None
        finally:
            if saved is not None:
                sys.modules["paddleocr"] = saved

    def test_never_returns_paddlex(self):
        """get_structure_engine_class never returns paddlex engine type."""
        mock_ppstructure = MagicMock()
        mock_paddleocr = MagicMock()
        mock_paddleocr.PPStructure = mock_ppstructure

        with patch.dict("sys.modules", {"paddleocr": mock_paddleocr}):
            _, engine_type = paddle_compat.get_structure_engine_class()
            assert engine_type != "paddlex"


class TestModuleLevelLogging:
    """Tests for module-level version logging on import."""

    def test_version_logged_on_import(self):
        """Module logs version info when paddle is available."""
        import importlib

        mock_paddle = MagicMock()
        mock_paddle.__version__ = "2.6.2"

        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            with patch.object(paddle_compat.logger, "info") as mock_log:
                importlib.reload(paddle_compat)
                # Should have logged the version
                mock_log.assert_called()
                # Check the log message contains version info
                log_args = mock_log.call_args_list[-1]
                assert "PaddlePaddle version" in log_args[0][0]

    def test_no_log_when_paddle_missing(self):
        """Module does not log when paddle is not available."""
        import importlib

        saved = sys.modules.pop("paddle", None)
        try:
            import builtins

            original_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "paddle":
                    raise ImportError("No paddle")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=fake_import):
                with patch.object(paddle_compat.logger, "info") as mock_log:
                    importlib.reload(paddle_compat)
                    # No version info call (there may be no calls at all,
                    # or only unrelated calls from the module)
                    for call in mock_log.call_args_list:
                        assert "PaddlePaddle version" not in call[0][0]
        finally:
            if saved is not None:
                sys.modules["paddle"] = saved
            # Restore module to clean state
            importlib.reload(paddle_compat)


class TestIntegrationScenarios:
    """End-to-end scenarios combining multiple shim functions."""

    def test_v2_full_workflow(self):
        """Full v2 workflow: kwargs generation + result normalization."""
        # Build engine kwargs
        kwargs = paddle_compat.get_ocr_engine_kwargs(
            use_gpu=True, lang="en"
        )
        assert "use_gpu" in kwargs
        assert kwargs["use_gpu"] is True
        assert "device" not in kwargs
        assert kwargs["use_angle_cls"] is True

        # Normalize a v2-style result
        v2_result = [
            [
                [
                    [[10, 20], [100, 20], [100, 40], [10, 40]],
                    ("Test text", 0.95),
                ],
            ]
        ]
        normalized = paddle_compat.normalize_ocr_result(v2_result)
        assert len(normalized) == 1
        assert normalized[0][1] == "Test text"

    def test_v2_with_onnx_workflow(self):
        """Full v2 workflow with ONNX inference backend."""
        kwargs = paddle_compat.get_ocr_engine_kwargs(
            use_gpu=False, lang="ch", use_onnx=True
        )
        assert kwargs["use_gpu"] is False
        assert kwargs["use_onnx"] is True
        assert kwargs["lang"] == "ch"
        assert "device" not in kwargs
        assert "enable_hpi" not in kwargs

    def test_is_paddle_v3_never_true_in_full_workflow(self):
        """Verify is_paddle_v3() is always False regardless of installed paddle."""
        mock_paddle = MagicMock()
        mock_paddle.__version__ = "3.0.0"
        with patch.dict("sys.modules", {"paddle": mock_paddle}):
            # Even with paddle 3.x installed, is_paddle_v3 returns False
            assert paddle_compat.is_paddle_v3() is False
            # normalize_device always returns None
            assert paddle_compat.normalize_device(use_gpu=True) is None
            # kwargs always use v2 format
            kwargs = paddle_compat.get_ocr_engine_kwargs(use_gpu=True, lang="en")
            assert "use_gpu" in kwargs
            assert "device" not in kwargs

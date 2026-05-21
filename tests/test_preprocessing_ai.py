"""Unit tests for AI-powered preprocessing backends in preprocessing.py.

Tests cover:
- NAFNet ONNX denoising (model loading, tile inference, fallback)
- U-Net ONNX binarization (model loading, tile inference, fallback)
- Tile-based inference mechanics (tiling, overlap, reassembly)
- Backend selection routing (DENOISE_BACKEND, BINARIZE_BACKEND)
- ONNX model cache (thread-safe, load-once semantics)
- Graceful degradation (onnxruntime missing, model file missing)
- Integration with preprocess_for_ocr pipeline levels

All ONNX operations are mocked since onnxruntime and model files are
not guaranteed to be available in CI.

Run with: python -m pytest tests/test_preprocessing_ai.py -v
"""

import os
import threading
from unittest.mock import MagicMock, patch

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import preprocessing  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(width=100, height=80):
    """Create a small test RGB PIL Image."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_gray_image(width=100, height=80):
    """Create a small test grayscale PIL Image."""
    arr = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _make_mock_session(output_shape_fn=None):
    """Create a mock ONNX InferenceSession.

    The mock session returns an array of the same shape as input,
    filled with 0.5 (mid-gray when multiplied by 255).
    """
    session = MagicMock()
    input_meta = MagicMock()
    input_meta.name = "input"
    session.get_inputs.return_value = [input_meta]

    def run_fn(names, feed):
        blob = feed["input"]  # (1, C, H, W)
        # Return same shape, filled with 0.5
        out = np.full_like(blob, 0.5, dtype=np.float32)
        return [out]

    session.run.side_effect = run_fn
    return session


# ---------------------------------------------------------------------------
# Tests: ONNX model loading
# ---------------------------------------------------------------------------


class TestOnnxModelLoading:
    """Tests for _load_onnx_model()."""

    def test_validate_model_path_rejects_relative_paths(self):
        assert preprocessing._validate_onnx_model_path("models/test.onnx") is None

    def test_validate_model_path_rejects_non_onnx_extension(self):
        bad_path = os.path.join(os.sep, "app", "models", "test.bin")
        assert preprocessing._validate_onnx_model_path(bad_path) is None

    def test_validate_model_path_rejects_outside_allowed_roots(self, tmp_path):
        outside_path = str(tmp_path / "test_model.onnx")
        assert preprocessing._validate_onnx_model_path(outside_path) is None

    def test_returns_none_when_onnx_unavailable(self):
        """Returns None when onnxruntime is not installed."""
        with patch.object(preprocessing, "_ONNX_AVAILABLE", False):
            result = preprocessing._load_onnx_model("/fake/model.onnx")
            assert result is None

    def test_returns_none_when_file_missing(self):
        """Returns None when model file does not exist."""
        with patch.object(preprocessing, "_ONNX_AVAILABLE", True):
            # Clear cache to avoid false positive from cached entry
            preprocessing._onnx_model_cache.clear()
            result = preprocessing._load_onnx_model("/nonexistent/model.onnx")
            assert result is None

    def test_loads_model_when_available(self, tmp_path):
        """Loads model and returns InferenceSession."""
        model_path = str(tmp_path / "test_model.onnx")
        # Create a dummy file
        with open(model_path, "wb") as f:
            f.write(b"fake_onnx_data")

        mock_session = _make_mock_session()
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session

        with patch.object(preprocessing, "_ONNX_AVAILABLE", True):
            with patch.object(
                preprocessing,
                "_ALLOWED_ONNX_MODEL_ROOTS",
                (str(tmp_path.resolve()),),
            ):
                preprocessing._onnx_model_cache.clear()
                with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
                    result = preprocessing._load_onnx_model(model_path)
                    assert result is mock_session

    def test_caches_loaded_model(self, tmp_path):
        """Model is loaded once and cached for subsequent calls."""
        model_path = str(tmp_path / "cached_model.onnx")
        with open(model_path, "wb") as f:
            f.write(b"fake_onnx_data")

        mock_session = _make_mock_session()
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session

        with patch.object(preprocessing, "_ONNX_AVAILABLE", True):
            with patch.object(
                preprocessing,
                "_ALLOWED_ONNX_MODEL_ROOTS",
                (str(tmp_path.resolve()),),
            ):
                preprocessing._onnx_model_cache.clear()
                with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
                    result1 = preprocessing._load_onnx_model(model_path)
                    result2 = preprocessing._load_onnx_model(model_path)
                    assert result1 is result2
                # InferenceSession should be called only once
                assert mock_ort.InferenceSession.call_count == 1

    def test_returns_cached_model_without_onnx_check(self):
        """Cached model is returned even if file check would fail."""
        fake_path = os.path.realpath(
            os.path.join(os.sep, "app", "models", "already.onnx")
        )
        mock_session = _make_mock_session()
        preprocessing._onnx_model_cache[fake_path] = mock_session

        with patch.object(preprocessing, "_ONNX_AVAILABLE", True):
            result = preprocessing._load_onnx_model(fake_path)
            assert result is mock_session

        # Cleanup
        del preprocessing._onnx_model_cache[fake_path]

    def test_handles_session_creation_error(self, tmp_path):
        """Returns None when InferenceSession constructor fails."""
        model_path = str(tmp_path / "bad_model.onnx")
        with open(model_path, "wb") as f:
            f.write(b"corrupt_data")

        mock_ort = MagicMock()
        mock_ort.InferenceSession.side_effect = RuntimeError("bad model")

        with patch.object(preprocessing, "_ONNX_AVAILABLE", True):
            with patch.object(
                preprocessing,
                "_ALLOWED_ONNX_MODEL_ROOTS",
                (str(tmp_path.resolve()),),
            ):
                preprocessing._onnx_model_cache.clear()
                with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
                    result = preprocessing._load_onnx_model(model_path)
                    assert result is None


# ---------------------------------------------------------------------------
# Tests: Tile-based inference
# ---------------------------------------------------------------------------


class TestTileInference:
    """Tests for _tile_inference()."""

    def test_processes_rgb_image(self):
        """Processes an RGB (H, W, 3) image through tiles."""
        img = np.random.randint(0, 256, (100, 80, 3), dtype=np.uint8)
        session = _make_mock_session()

        result = preprocessing._tile_inference(img, session, tile_size=64, overlap=8)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_processes_grayscale_image(self):
        """Processes a grayscale (H, W) image through tiles."""
        img = np.random.randint(0, 256, (100, 80), dtype=np.uint8)
        session = _make_mock_session()

        result = preprocessing._tile_inference(img, session, tile_size=64, overlap=8)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_single_tile_no_overlap(self):
        """Image smaller than tile_size is processed in a single tile."""
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        session = _make_mock_session()

        result = preprocessing._tile_inference(img, session, tile_size=64, overlap=0)
        assert result.shape == img.shape

    def test_output_values_in_valid_range(self):
        """Output values are clipped to [0, 255]."""
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        session = _make_mock_session()

        result = preprocessing._tile_inference(img, session, tile_size=32, overlap=4)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_overlap_averaging(self):
        """Overlapping regions are averaged (not summed)."""
        # Use a constant input so we can verify the output
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        session = _make_mock_session()  # Returns 0.5 -> 127 or 128

        result = preprocessing._tile_inference(img, session, tile_size=32, overlap=8)
        # The mock returns 0.5 for all pixels, so result should be ~127
        assert result.shape == img.shape
        # All values should be close to 127 (0.5 * 255)
        assert np.all(result >= 120)
        assert np.all(result <= 135)

    def test_calls_session_run(self):
        """Session.run is called at least once per tile."""
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        session = _make_mock_session()

        preprocessing._tile_inference(img, session, tile_size=64, overlap=0)
        assert session.run.call_count >= 4  # 2x2 grid minimum


# ---------------------------------------------------------------------------
# Tests: NAFNet denoising
# ---------------------------------------------------------------------------


class TestDenoiseNafnet:
    """Tests for denoise_nafnet()."""

    def test_falls_back_when_model_missing(self):
        """Falls back to bilateral when ONNX model is not found."""
        img = _make_rgb_image()
        with patch.object(preprocessing, "_load_onnx_model", return_value=None):
            with patch.object(
                preprocessing, "denoise_bilateral", return_value=img
            ) as mock_bilateral:
                preprocessing.denoise_nafnet(img)
                mock_bilateral.assert_called_once_with(img)

    def test_uses_onnx_model_when_available(self):
        """Uses ONNX inference when model is loaded."""
        img = _make_rgb_image(64, 64)
        session = _make_mock_session()

        with patch.object(preprocessing, "_load_onnx_model", return_value=session):
            result = preprocessing.denoise_nafnet(img)
            assert result is not None
            assert result.mode == "RGB"

    def test_falls_back_on_inference_error(self):
        """Falls back to bilateral when inference raises an exception."""
        img = _make_rgb_image()
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input")]
        session.run.side_effect = RuntimeError("inference failed")

        with patch.object(preprocessing, "_load_onnx_model", return_value=session):
            with patch.object(
                preprocessing, "denoise_bilateral", return_value=img
            ) as mock_bilateral:
                preprocessing.denoise_nafnet(img)
                mock_bilateral.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: U-Net binarization
# ---------------------------------------------------------------------------


class TestBinarizeUnet:
    """Tests for binarize_unet()."""

    def test_falls_back_when_model_missing(self):
        """Falls back to adaptive when ONNX model is not found."""
        img = _make_gray_image()
        with patch.object(preprocessing, "_load_onnx_model", return_value=None):
            with patch.object(
                preprocessing, "binarize_adaptive", return_value=img
            ) as mock_adaptive:
                preprocessing.binarize_unet(img)
                mock_adaptive.assert_called_once_with(img)

    def test_uses_onnx_model_when_available(self):
        """Uses ONNX inference when model is loaded."""
        img = _make_gray_image(64, 64)
        session = _make_mock_session()

        with patch.object(preprocessing, "_load_onnx_model", return_value=session):
            result = preprocessing.binarize_unet(img)
            assert result is not None
            assert result.mode == "L"

    def test_output_is_binary(self):
        """Output contains only 0 and 255 values."""
        img = _make_gray_image(64, 64)
        session = _make_mock_session()

        with patch.object(preprocessing, "_load_onnx_model", return_value=session):
            result = preprocessing.binarize_unet(img)
            arr = np.array(result)
            unique_vals = set(np.unique(arr))
            assert unique_vals.issubset({0, 255})

    def test_falls_back_on_inference_error(self):
        """Falls back to adaptive when inference raises an exception."""
        img = _make_gray_image()
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input")]
        session.run.side_effect = RuntimeError("inference failed")

        with patch.object(preprocessing, "_load_onnx_model", return_value=session):
            with patch.object(
                preprocessing, "binarize_adaptive", return_value=img
            ) as mock_adaptive:
                preprocessing.binarize_unet(img)
                mock_adaptive.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Backend selection routing
# ---------------------------------------------------------------------------


class TestBackendSelection:
    """Tests for denoise_image() and binarize_image() backend routing."""

    def test_denoise_default_is_bilateral(self):
        """Default DENOISE_BACKEND='bilateral' routes to bilateral."""
        img = _make_rgb_image()
        with patch.object(preprocessing, "DENOISE_BACKEND", "bilateral"):
            with patch.object(
                preprocessing, "denoise_bilateral", return_value=img
            ) as mock_fn:
                preprocessing.denoise_image(img)
                mock_fn.assert_called_once_with(img)

    def test_denoise_nafnet_backend(self):
        """DENOISE_BACKEND='nafnet' routes to NAFNet."""
        img = _make_rgb_image()
        with patch.object(preprocessing, "DENOISE_BACKEND", "nafnet"):
            with patch.object(
                preprocessing, "denoise_nafnet", return_value=img
            ) as mock_fn:
                preprocessing.denoise_image(img)
                mock_fn.assert_called_once_with(img)

    def test_denoise_auto_backend_tries_nafnet(self):
        """DENOISE_BACKEND='auto' tries NAFNet first."""
        img = _make_rgb_image()
        with patch.object(preprocessing, "DENOISE_BACKEND", "auto"):
            with patch.object(
                preprocessing, "denoise_nafnet", return_value=img
            ) as mock_fn:
                preprocessing.denoise_image(img)
                mock_fn.assert_called_once_with(img)

    def test_binarize_default_is_adaptive(self):
        """Default BINARIZE_BACKEND='adaptive' routes to adaptive."""
        img = _make_gray_image()
        with patch.object(preprocessing, "BINARIZE_BACKEND", "adaptive"):
            with patch.object(
                preprocessing, "binarize_adaptive", return_value=img
            ) as mock_fn:
                preprocessing.binarize_image(img)
                mock_fn.assert_called_once_with(img)

    def test_binarize_unet_backend(self):
        """BINARIZE_BACKEND='unet' routes to U-Net."""
        img = _make_gray_image()
        with patch.object(preprocessing, "BINARIZE_BACKEND", "unet"):
            with patch.object(
                preprocessing, "binarize_unet", return_value=img
            ) as mock_fn:
                preprocessing.binarize_image(img)
                mock_fn.assert_called_once_with(img)

    def test_binarize_auto_backend_tries_unet(self):
        """BINARIZE_BACKEND='auto' tries U-Net first."""
        img = _make_gray_image()
        with patch.object(preprocessing, "BINARIZE_BACKEND", "auto"):
            with patch.object(
                preprocessing, "binarize_unet", return_value=img
            ) as mock_fn:
                preprocessing.binarize_image(img)
                mock_fn.assert_called_once_with(img)

    def test_denoise_skips_tiny_image(self):
        """denoise_image returns original for images below _MIN_DIM."""
        tiny = Image.new("RGB", (4, 4))
        result = preprocessing.denoise_image(tiny)
        assert result is tiny

    def test_binarize_skips_tiny_image(self):
        """binarize_image returns original for images below _MIN_DIM."""
        tiny = Image.new("L", (4, 4))
        result = preprocessing.binarize_image(tiny)
        assert result is tiny


# ---------------------------------------------------------------------------
# Tests: ONNX model cache thread safety
# ---------------------------------------------------------------------------


class TestOnnxModelCache:
    """Tests for thread-safe ONNX model caching."""

    def test_cache_is_dict(self):
        """Cache is a dictionary."""
        assert isinstance(preprocessing._onnx_model_cache, dict)

    def test_cache_lock_is_lock(self):
        """Cache lock is a threading.Lock."""
        assert isinstance(preprocessing._onnx_cache_lock, type(threading.Lock()))

    def test_cache_cleared_between_tests(self):
        """Verify we can clear the cache without side effects."""
        preprocessing._onnx_model_cache.clear()
        assert len(preprocessing._onnx_model_cache) == 0


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests for behaviour when onnxruntime is not installed."""

    def test_onnx_unavailable_nafnet_falls_back(self):
        """NAFNet falls back to bilateral when ONNX is not available."""
        img = _make_rgb_image()
        with patch.object(preprocessing, "_ONNX_AVAILABLE", False):
            preprocessing._onnx_model_cache.clear()
            with patch.object(
                preprocessing, "denoise_bilateral", return_value=img
            ) as mock_fn:
                preprocessing.denoise_nafnet(img)
                mock_fn.assert_called_once()

    def test_onnx_unavailable_unet_falls_back(self):
        """U-Net falls back to adaptive when ONNX is not available."""
        img = _make_gray_image()
        with patch.object(preprocessing, "_ONNX_AVAILABLE", False):
            preprocessing._onnx_model_cache.clear()
            with patch.object(
                preprocessing, "binarize_adaptive", return_value=img
            ) as mock_fn:
                preprocessing.binarize_unet(img)
                mock_fn.assert_called_once()

    def test_preprocess_for_ocr_aggressive_uses_binarize_image(self):
        """preprocess_for_ocr aggressive level calls binarize_image."""
        img = _make_rgb_image(100, 100)
        with patch.object(preprocessing, "_CV2_AVAILABLE", True), \
             patch.object(preprocessing, "deskew_image", return_value=img), \
             patch.object(preprocessing, "denoise_image", return_value=img), \
             patch.object(preprocessing, "enhance_contrast", return_value=img), \
             patch.object(preprocessing, "binarize_image", return_value=img) as mock_bin:
            preprocessing.preprocess_for_ocr(img, level="aggressive")
            mock_bin.assert_called_once()

    def test_preprocess_for_ocr_enhanced_uses_denoise_image(self):
        """preprocess_for_ocr enhanced level calls denoise_image."""
        img = _make_rgb_image(100, 100)
        with patch.object(preprocessing, "_CV2_AVAILABLE", True), \
             patch.object(preprocessing, "deskew_image", return_value=img), \
             patch.object(preprocessing, "denoise_image", return_value=img) as mock_dn, \
             patch.object(preprocessing, "enhance_contrast", return_value=img):
            preprocessing.preprocess_for_ocr(img, level="enhanced")
            mock_dn.assert_called_once()

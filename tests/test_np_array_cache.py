"""Tests for np.array() caching for DocIntel when preprocessing is disabled.

Verifies the conditional reuse logic in the GPU worker thread's Document
Intelligence block. When preprocessing is disabled, ``ocr_img is task.image``
holds and the already-computed ``img_np`` can be reused instead of calling
``np.array()`` a second time.

Covers:
- Preprocessing disabled: di_img_np reuses img_np (identity check)
- Preprocessing enabled: di_img_np is a fresh conversion of task.image
- PaddleOCR failed (img_np is None): di_img_np is a fresh conversion
- img_np is None AND ocr_img is task.image: still converts fresh (safe guard)

Run with: python -m pytest tests/test_np_array_cache.py -v
"""

import os

import numpy as np
import pytest
from PIL import Image

# Add project root to path


def _resolve_di_img_np(img_np, ocr_img, task_image):
    """Reproduce the conditional from ocr_gpu_async.py DocIntel block.

    This mirrors the exact logic at the Document Intelligence (Phase 3A)
    section of ``gpu_worker_thread``:

        if img_np is not None and ocr_img is task.image:
            di_img_np = img_np
        else:
            di_img_np = np.array(task.image)
    """
    if img_np is not None and ocr_img is task_image:
        return img_np
    return np.array(task_image)


@pytest.fixture
def sample_image():
    """Create a small RGB test image."""
    return Image.new("RGB", (100, 80), color=(200, 150, 100))


class TestNpArrayCachePreprocessingDisabled:
    """When preprocessing is disabled, ocr_img IS task.image (same object)."""

    def test_reuses_img_np_identity(self, sample_image):
        """di_img_np should be the exact same numpy array object as img_np."""
        task_image = sample_image
        ocr_img = task_image  # Preprocessing disabled: same object
        img_np = np.array(ocr_img)

        di_img_np = _resolve_di_img_np(img_np, ocr_img, task_image)

        assert di_img_np is img_np, (
            "Expected di_img_np to be the same object as img_np "
            "when preprocessing is disabled"
        )

    def test_no_extra_allocation(self, sample_image):
        """Verify no np.array() call happens (the returned object is img_np)."""
        task_image = sample_image
        ocr_img = task_image
        img_np = np.array(ocr_img)

        original_id = id(img_np)
        di_img_np = _resolve_di_img_np(img_np, ocr_img, task_image)

        assert id(di_img_np) == original_id

    def test_array_content_matches(self, sample_image):
        """The cached array has identical content to a fresh conversion."""
        task_image = sample_image
        ocr_img = task_image
        img_np = np.array(ocr_img)

        di_img_np = _resolve_di_img_np(img_np, ocr_img, task_image)
        fresh = np.array(task_image)

        np.testing.assert_array_equal(di_img_np, fresh)


class TestNpArrayCachePreprocessingEnabled:
    """When preprocessing is enabled, ocr_img is a different object."""

    def test_creates_fresh_array(self, sample_image):
        """di_img_np must be a NEW array from task.image, not img_np."""
        task_image = sample_image
        # Simulate preprocessing: ocr_img is a modified copy
        ocr_img = task_image.copy()
        assert ocr_img is not task_image  # Verify our test setup
        img_np = np.array(ocr_img)

        di_img_np = _resolve_di_img_np(img_np, ocr_img, task_image)

        assert di_img_np is not img_np, (
            "Expected di_img_np to be a fresh array when preprocessing "
            "is enabled (ocr_img is not task.image)"
        )

    def test_fresh_array_matches_task_image(self, sample_image):
        """The fresh array should match task.image, not ocr_img."""
        task_image = sample_image
        # Create a preprocessed version with different pixel values
        ocr_img = Image.new("RGB", (100, 80), color=(0, 0, 0))
        img_np = np.array(ocr_img)

        di_img_np = _resolve_di_img_np(img_np, ocr_img, task_image)
        expected = np.array(task_image)

        np.testing.assert_array_equal(di_img_np, expected)


class TestNpArrayCachePaddleFailed:
    """When PaddleOCR fails before numpy conversion, img_np is None."""

    def test_img_np_none_creates_fresh(self, sample_image):
        """Must create fresh array even when ocr_img is task.image."""
        task_image = sample_image
        ocr_img = task_image  # Same object (no preprocessing)
        img_np = None  # PaddleOCR failed

        di_img_np = _resolve_di_img_np(img_np, ocr_img, task_image)

        assert di_img_np is not None
        expected = np.array(task_image)
        np.testing.assert_array_equal(di_img_np, expected)

    def test_img_np_none_with_preprocessing(self, sample_image):
        """img_np is None AND preprocessing enabled: fresh conversion."""
        task_image = sample_image
        ocr_img = task_image.copy()
        img_np = None

        di_img_np = _resolve_di_img_np(img_np, ocr_img, task_image)

        expected = np.array(task_image)
        np.testing.assert_array_equal(di_img_np, expected)


class TestNpArrayCacheEdgeCases:
    """Edge cases for the caching conditional."""

    def test_grayscale_image(self):
        """Works with grayscale (mode 'L') images."""
        task_image = Image.new("L", (50, 50), color=128)
        ocr_img = task_image
        img_np = np.array(ocr_img)

        di_img_np = _resolve_di_img_np(img_np, ocr_img, task_image)

        assert di_img_np is img_np

    def test_rgba_image(self):
        """Works with RGBA images."""
        task_image = Image.new("RGBA", (50, 50), color=(100, 150, 200, 255))
        ocr_img = task_image
        img_np = np.array(ocr_img)

        di_img_np = _resolve_di_img_np(img_np, ocr_img, task_image)

        assert di_img_np is img_np

    def test_large_image(self):
        """Verify behavior with a larger image (closer to real-world sizes)."""
        task_image = Image.new("RGB", (2550, 3300), color=(255, 255, 255))
        ocr_img = task_image
        img_np = np.array(ocr_img)

        di_img_np = _resolve_di_img_np(img_np, ocr_img, task_image)

        assert di_img_np is img_np
        # Verify dimensions match (height, width, channels for RGB)
        assert di_img_np.shape == (3300, 2550, 3)


class TestSourceCodeConditional:
    """Verify the actual conditional exists in ocr_gpu_async.py source."""

    def test_docintel_block_has_cache_conditional(self):
        """The DocIntel block must contain the identity-check optimization."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ocr_gpu_async.py",
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Verify the conditional reuse pattern exists
        assert "if img_np is not None and ocr_img is task.image:" in source, (
            "Expected identity-check conditional for np.array cache in DocIntel block"
        )
        assert "di_img_np = img_np" in source, (
            "Expected reuse assignment 'di_img_np = img_np'"
        )

    def test_img_np_initialized_to_none(self):
        """img_np must be initialized to None before the OCR try block."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ocr_gpu_async.py",
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()

        assert "img_np = None" in source, (
            "img_np must be pre-initialized to None for safe guard when PaddleOCR fails"
        )

    def test_two_pass_documentation_comment(self):
        """Two-pass language detection has an explanatory comment."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ocr_gpu_async.py",
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()

        assert "paddle_lines is NOT" in source, (
            "Expected two-pass documentation comment explaining paddle_lines scope"
        )
        assert "second pass is only" in source, (
            "Expected two-pass comment explaining the purpose of the second pass"
        )

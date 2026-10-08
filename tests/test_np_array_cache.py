"""Tests for maintenance review: np.array() caching for DocIntel when preprocessing is disabled.

Verifies the conditional reuse logic in the GPU worker thread's Document
Intelligence block. When preprocessing is disabled,
``attempt.ocr_img is task.image`` holds and the already-computed
``attempt.img_np`` can be reused instead of calling ``np.array()`` a second
time. ``img_np`` is initialized to None on ``_WorkerAttempt`` at bind time.

Covers:
- Preprocessing disabled: di_img_np reuses img_np (identity check)
- Preprocessing enabled: di_img_np is a fresh conversion of task.image
- PaddleOCR failed (attempt.img_np is None): di_img_np is a fresh conversion
- attempt.img_np is None AND attempt.ocr_img is task.image: still converts fresh

Run with: python -m pytest tests/test_np_array_cache.py -v
"""

import ast
import inspect
import os

import numpy as np
import pytest
from PIL import Image

import ocr_gpu_async as pipe

# Add project root to path


def _resolve_di_img_np(img_np, ocr_img, task_image):
    """Reproduce the DocIntel np.array reuse predicate from ocr_gpu_async.py.

    Production (``worker_thread`` / ``_WorkerAttempt``) is:

        if attempt.img_np is not None and attempt.ocr_img is task.image:
            di_img_np = attempt.img_np
        else:
            di_img_np = np.array(task.image)

    This helper models the same boolean logic with bare locals.
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


def _is_attempt_attr(node, name):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "attempt"
        and node.attr == name
    )


class TestSourceCodeConditional:
    """Verify the actual conditional exists in ocr_gpu_async.py source."""

    def test_docintel_block_has_cache_conditional(self):
        """The DocIntel block must contain the identity-check optimization."""
        source = inspect.getsource(pipe.worker_thread)
        assert (
            "if attempt.img_np is not None and attempt.ocr_img is task.image:" in source
        ), "Expected identity-check conditional for np.array cache in DocIntel block"
        assert (
            "di_img_np = attempt.img_np" in source
        ), "Expected reuse assignment 'di_img_np = attempt.img_np'"

        tree = ast.parse(source)
        reuse_if = False
        reuse_assign = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
                test = node.test
                if not isinstance(test.op, ast.And) or len(test.values) != 2:
                    continue
                left, right = test.values
                left_ok = (
                    isinstance(left, ast.Compare)
                    and _is_attempt_attr(left.left, "img_np")
                    and len(left.ops) == 1
                    and isinstance(left.ops[0], ast.IsNot)
                    and isinstance(left.comparators[0], ast.Constant)
                    and left.comparators[0].value is None
                )
                right_ok = (
                    isinstance(right, ast.Compare)
                    and _is_attempt_attr(right.left, "ocr_img")
                    and len(right.ops) == 1
                    and isinstance(right.ops[0], ast.Is)
                    and isinstance(right.comparators[0], ast.Attribute)
                    and isinstance(right.comparators[0].value, ast.Name)
                    and right.comparators[0].value.id == "task"
                    and right.comparators[0].attr == "image"
                )
                if left_ok and right_ok:
                    reuse_if = True
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if (
                    isinstance(target, ast.Name)
                    and target.id == "di_img_np"
                    and _is_attempt_attr(node.value, "img_np")
                ):
                    reuse_assign = True
        assert reuse_if, (
            "DocIntel identity-check must be an AST If on attempt.img_np / "
            "attempt.ocr_img, not a comment"
        )
        assert reuse_assign, (
            "DocIntel reuse must assign di_img_np = attempt.img_np in AST, "
            "not a comment"
        )

    def test_img_np_initialized_to_none(self):
        """img_np must start as None on each _WorkerAttempt after queue receive."""
        tree = ast.parse(inspect.getsource(pipe._WorkerAttempt))
        img_np_default = None
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "img_np":
                    found = True
                    img_np_default = node.value
        assert found, "img_np must be a _WorkerAttempt field"
        assert isinstance(img_np_default, ast.Constant) and img_np_default.value is None
        worker_src = inspect.getsource(pipe.worker_thread)
        assert "attempt = _bind_worker_attempt(task)" in worker_src

    def test_two_pass_documentation_comment(self):
        """maintenance review: Two-pass language detection has an explanatory comment."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ocr_gpu_async.py",
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()

        assert (
            "replaces attempt.paddle_lines" in source
        ), "Expected two-pass comment stating accepted rerun replaces paddle_lines"
        assert (
            "paddle_lines is NOT" not in source
        ), "Two-pass must not claim first-pass paddle_lines are left in place"

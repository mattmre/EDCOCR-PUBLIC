"""
Gap detection regression tests for crash-resume scheduler logic.

Verifies ``compute_resume_gap_chunks`` — the set-difference-based helper
that the scheduler uses to decide which pages to re-extract after a crash.

Correctness invariants:
  1. Every dispatched chunk ``(start, end)`` is CONTIGUOUS in the missing
     set (never spans across an already-processed page).
  2. The union of all dispatched pages equals exactly the set of missing
     pages (no over-dispatch, no under-dispatch).
  3. No chunk exceeds ``chunk_target_size`` pages.
  4. Already-processed pages are never re-dispatched.

These invariants protect against the class of bugs described in the
expert panel review ("non-contiguous gaps dispatch extra pages"), where
naive range-based logic ``(min(missing), max(missing))`` would redundantly
re-extract already-processed pages.

Run with: python -m pytest tests/test_gap_detection.py -v
"""

import ast
import os
import types

import pytest


# ---------------------------------------------------------------------------
# Import strategy: extract the helper from source without triggering heavy
# imports (PaddleOCR, Tesseract, etc.).
# ---------------------------------------------------------------------------
def _load_compute_resume_gap_chunks():
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "ocr_gpu_async.py"
    )
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    mod = types.ModuleType("ocr_gap_helper")
    mod.__file__ = src_path

    source_tree = ast.parse(source)
    for node in source_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_resume_gap_chunks":
            func_source = ast.get_source_segment(source, node)
            exec(compile(func_source, src_path, "exec"), mod.__dict__)
            return mod.compute_resume_gap_chunks

    raise RuntimeError("compute_resume_gap_chunks not found in ocr_gpu_async.py")


try:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from ocr_gpu_async import compute_resume_gap_chunks  # type: ignore
except Exception:
    compute_resume_gap_chunks = _load_compute_resume_gap_chunks()


# ---------------------------------------------------------------------------
# Invariant checker — runs on every returned chunk list
# ---------------------------------------------------------------------------
def _assert_chunks_valid(chunks, total_pages, existing, chunk_target_size):
    """Assert the four correctness invariants of compute_resume_gap_chunks."""
    all_pages = set(range(1, total_pages + 1))
    expected_missing = all_pages - {p for p in existing if 1 <= p <= total_pages}

    dispatched = set()
    for start, end in chunks:
        # Invariant 3: chunk_target_size bound
        assert end - start + 1 <= chunk_target_size, (
            f"Chunk ({start}, {end}) exceeds chunk_target_size={chunk_target_size}"
        )
        # start <= end
        assert start <= end, f"Chunk ({start}, {end}) is inverted"
        # Within range
        assert 1 <= start <= total_pages
        assert 1 <= end <= total_pages

        # Invariant 1: contiguous in missing set (no already-processed page
        # falls inside the chunk range)
        for p in range(start, end + 1):
            assert p in expected_missing, (
                f"Chunk ({start}, {end}) includes already-processed page {p}"
            )
            assert p not in dispatched, (
                f"Page {p} dispatched in multiple chunks"
            )
            dispatched.add(p)

    # Invariant 2: union equals missing set
    assert dispatched == expected_missing, (
        f"Dispatched pages {sorted(dispatched)} != "
        f"expected missing {sorted(expected_missing)}"
    )


# ---------------------------------------------------------------------------
# Core scenarios
# ---------------------------------------------------------------------------
class TestEmptyExisting:
    """When no pages exist yet, all pages must be dispatched."""

    def test_empty_existing_small_doc(self):
        chunks = compute_resume_gap_chunks(6, set(), chunk_target_size=10)
        _assert_chunks_valid(chunks, 6, set(), 10)
        assert chunks == [(1, 6)]

    def test_empty_existing_splits_by_target_size(self):
        chunks = compute_resume_gap_chunks(10, set(), chunk_target_size=3)
        _assert_chunks_valid(chunks, 10, set(), 3)
        assert chunks == [(1, 3), (4, 6), (7, 9), (10, 10)]

    def test_empty_existing_single_page_doc(self):
        chunks = compute_resume_gap_chunks(1, set(), chunk_target_size=10)
        _assert_chunks_valid(chunks, 1, set(), 10)
        assert chunks == [(1, 1)]


class TestFullyCompleted:
    """When all pages are already processed, no chunks should be dispatched."""

    def test_all_pages_existing(self):
        chunks = compute_resume_gap_chunks(6, {1, 2, 3, 4, 5, 6}, chunk_target_size=10)
        assert chunks == []

    def test_all_pages_existing_large_doc(self):
        existing = set(range(1, 101))
        chunks = compute_resume_gap_chunks(100, existing, chunk_target_size=5)
        assert chunks == []


class TestContiguousGap:
    """Single contiguous missing range."""

    def test_contiguous_gap_at_start(self):
        # Pages 1, 2, 3 missing; 4, 5, 6 existing
        chunks = compute_resume_gap_chunks(6, {4, 5, 6}, chunk_target_size=10)
        _assert_chunks_valid(chunks, 6, {4, 5, 6}, 10)
        assert chunks == [(1, 3)]

    def test_contiguous_gap_at_end(self):
        # Pages 1, 2, 3 existing; 4, 5, 6 missing
        chunks = compute_resume_gap_chunks(6, {1, 2, 3}, chunk_target_size=10)
        _assert_chunks_valid(chunks, 6, {1, 2, 3}, 10)
        assert chunks == [(4, 6)]

    def test_contiguous_gap_in_middle(self):
        # Pages 1, 2 existing, 3, 4 missing, 5, 6 existing
        chunks = compute_resume_gap_chunks(6, {1, 2, 5, 6}, chunk_target_size=10)
        _assert_chunks_valid(chunks, 6, {1, 2, 5, 6}, 10)
        assert chunks == [(3, 4)]

    def test_contiguous_gap_split_by_target(self):
        # All 10 pages missing, target=3 -> 4 chunks
        chunks = compute_resume_gap_chunks(10, set(), chunk_target_size=3)
        _assert_chunks_valid(chunks, 10, set(), 3)
        assert chunks == [(1, 3), (4, 6), (7, 9), (10, 10)]


class TestNonContiguousGaps:
    """The critical  regression scenario: non-contiguous gaps must
    NEVER dispatch chunks that span across an already-processed page."""

    def test_alternating_missing_existing(self):
        # Pages 1, 3, 5 existing; 2, 4, 6 missing
        existing = {1, 3, 5}
        chunks = compute_resume_gap_chunks(6, existing, chunk_target_size=10)
        _assert_chunks_valid(chunks, 6, existing, 10)
        # Each missing page is its own chunk (broken by existing pages)
        assert chunks == [(2, 2), (4, 4), (6, 6)]

    def test_alternating_reversed(self):
        # Pages 2, 4, 6 existing; 1, 3, 5 missing
        existing = {2, 4, 6}
        chunks = compute_resume_gap_chunks(6, existing, chunk_target_size=10)
        _assert_chunks_valid(chunks, 6, existing, 10)
        assert chunks == [(1, 1), (3, 3), (5, 5)]

    def test_two_disjoint_runs(self):
        # Pages 3, 7 existing; missing = {1,2, 4,5,6, 8,9,10}
        existing = {3, 7}
        chunks = compute_resume_gap_chunks(10, existing, chunk_target_size=10)
        _assert_chunks_valid(chunks, 10, existing, 10)
        assert chunks == [(1, 2), (4, 6), (8, 10)]

    def test_two_disjoint_runs_with_chunk_split(self):
        # Pages 3, 7 existing; target=2 forces sub-splits
        existing = {3, 7}
        chunks = compute_resume_gap_chunks(10, existing, chunk_target_size=2)
        _assert_chunks_valid(chunks, 10, existing, 2)
        # Missing: 1,2 | 4,5,6 | 8,9,10
        # Split by target=2: (1,2) | (4,5), (6,6) | (8,9), (10,10)
        assert chunks == [(1, 2), (4, 5), (6, 6), (8, 9), (10, 10)]

    def test_many_isolated_gaps(self):
        # Pages 2, 4, 6, 8 existing; 1, 3, 5, 7, 9, 10 missing
        existing = {2, 4, 6, 8}
        chunks = compute_resume_gap_chunks(10, existing, chunk_target_size=10)
        _assert_chunks_valid(chunks, 10, existing, 10)
        assert chunks == [(1, 1), (3, 3), (5, 5), (7, 7), (9, 10)]

    def test_single_missing_page_in_middle(self):
        # Only page 5 missing out of 10
        existing = {1, 2, 3, 4, 6, 7, 8, 9, 10}
        chunks = compute_resume_gap_chunks(10, existing, chunk_target_size=10)
        _assert_chunks_valid(chunks, 10, existing, 10)
        assert chunks == [(5, 5)]


class TestEdgeCases:
    """Boundary conditions and defensive inputs."""

    def test_zero_total_pages(self):
        assert compute_resume_gap_chunks(0, set(), chunk_target_size=10) == []

    def test_negative_total_pages(self):
        assert compute_resume_gap_chunks(-1, set(), chunk_target_size=10) == []

    def test_chunk_target_size_one(self):
        # Every missing page becomes its own chunk
        chunks = compute_resume_gap_chunks(5, set(), chunk_target_size=1)
        _assert_chunks_valid(chunks, 5, set(), 1)
        assert chunks == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

    def test_chunk_target_size_zero_is_clamped(self):
        # Zero should be treated as 1 (defensive)
        chunks = compute_resume_gap_chunks(3, set(), chunk_target_size=0)
        # Still each page emitted individually, no crash
        _assert_chunks_valid(chunks, 3, set(), 1)

    def test_chunk_target_size_negative_is_clamped(self):
        chunks = compute_resume_gap_chunks(3, set(), chunk_target_size=-5)
        _assert_chunks_valid(chunks, 3, set(), 1)

    def test_existing_pages_beyond_total(self):
        # Stale temp file "7.pdf" when total_pages=5: must be ignored
        existing = {1, 2, 7, 99}
        chunks = compute_resume_gap_chunks(5, existing, chunk_target_size=10)
        # Expected missing: {3, 4, 5}
        _assert_chunks_valid(chunks, 5, existing, 10)
        assert chunks == [(3, 5)]

    def test_existing_pages_zero_or_negative_ignored(self):
        # Defensive: page 0 or negative treated as invalid
        existing = {0, -1, 2, 4}
        chunks = compute_resume_gap_chunks(5, existing, chunk_target_size=10)
        # Expected missing: {1, 3, 5}
        _assert_chunks_valid(chunks, 5, existing, 10)
        assert chunks == [(1, 1), (3, 3), (5, 5)]

    def test_existing_as_list_not_set(self):
        # Helper must accept any iterable
        chunks = compute_resume_gap_chunks(5, [1, 3, 5], chunk_target_size=10)
        _assert_chunks_valid(chunks, 5, {1, 3, 5}, 10)
        assert chunks == [(2, 2), (4, 4)]

    def test_existing_as_tuple(self):
        chunks = compute_resume_gap_chunks(5, (1, 3, 5), chunk_target_size=10)
        _assert_chunks_valid(chunks, 5, {1, 3, 5}, 10)
        assert chunks == [(2, 2), (4, 4)]


class TestChunkSizeWithinContiguousRun:
    """A single contiguous missing run longer than chunk_target_size must
    split cleanly without dropping or duplicating pages."""

    def test_single_run_exact_multiple_of_target(self):
        # 6 missing, target=3
        chunks = compute_resume_gap_chunks(6, set(), chunk_target_size=3)
        _assert_chunks_valid(chunks, 6, set(), 3)
        assert chunks == [(1, 3), (4, 6)]

    def test_single_run_remainder(self):
        # 7 missing, target=3
        chunks = compute_resume_gap_chunks(7, set(), chunk_target_size=3)
        _assert_chunks_valid(chunks, 7, set(), 3)
        assert chunks == [(1, 3), (4, 6), (7, 7)]

    def test_mid_document_run_split(self):
        # Existing: {1, 10}; missing: {2..9}; target=3
        existing = {1, 10}
        chunks = compute_resume_gap_chunks(10, existing, chunk_target_size=3)
        _assert_chunks_valid(chunks, 10, existing, 3)
        assert chunks == [(2, 4), (5, 7), (8, 9)]


# ---------------------------------------------------------------------------
# Property-based style parametric tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "total,existing,target",
    [
        (10, set(), 4),
        (10, {1, 5, 10}, 3),
        (20, {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}, 2),
        (100, set(range(1, 51)), 7),
        (50, {i for i in range(1, 51) if i % 3 == 0}, 5),
        (1, {1}, 1),
        (1, set(), 1),
    ],
)
def test_invariants_hold_parametric(total, existing, target):
    chunks = compute_resume_gap_chunks(total, existing, target)
    _assert_chunks_valid(chunks, total, existing, target)

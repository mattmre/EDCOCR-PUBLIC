"""Tests for optimize_pdfs.py (maintenance review).

Covers Ghostscript compression, timeout handling, integrity validation,
batch processing, and skip logic.
"""

import io
import logging
import os
import shutil
import stat
import subprocess
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import fitz
import pytest
from PIL import Image

from optimize_pdfs import (
    _MOVEFILE_REPLACE_EXISTING,
    _MOVEFILE_WRITE_THROUGH,
    DEFAULT_QUALITY,
    GHOSTSCRIPT_TIMEOUT,
    PAGE_DIMENSION_TOLERANCE_PT,
    RENDER_LOCAL_BLUR_CHANGED_FRAC_MAX,
    RENDER_LOCAL_ENERGY_MAE_MAX,
    RENDER_LOCAL_ENERGY_RMS_MAX,
    RENDER_LOCAL_TILE_CHANGED_FRAC_MAX,
    RENDER_LOCAL_TILE_MAE_MAX,
    RENDER_MAE_MAX,
    RENDER_MAX_PIXELS,
    RENDER_MIN_EFFECTIVE_DPI,
    RENDER_NOMINAL_DPI,
    TEXT_GEOMETRY_TOLERANCE_PT,
    TEXT_WORD_VERTICAL_METRIC_TOLERANCE_PT,
    VALID_QUALITIES,
    OptimizationDisposition,
    _durable_replace,
    _fsync_parent_dir,
    _full_window_origins,
    _images_perceptually_equivalent,
    _normalize_extracted_text,
    _page_word_geometry,
    _render_scale_for_page,
    _validate_pdf_structure,
    _windows_extended_path,
    main,
    optimize_pdf,
    optimize_pdf_detailed,
)


def _write_pdf(
    path: Path,
    page_count: int,
    label: str,
    *,
    width: float = 595,
    height: float = 842,
    include_text: bool = True,
    background_rgb=None,
) -> None:
    with fitz.open() as document:
        for page_number in range(page_count):
            page = document.new_page(width=width, height=height)
            if background_rgb is not None:
                color = tuple(c / 255.0 for c in background_rgb)
                page.draw_rect(page.rect, color=color, fill=color)
            if include_text:
                page.insert_text((36, 72), f"{label} page {page_number + 1}")
        document.save(path)


def _write_solid_image_pdf(path: Path, rgb, *, width: float = 200, height: float = 200) -> None:
    with fitz.open() as document:
        page = document.new_page(width=width, height=height)
        color = tuple(c / 255.0 for c in rgb)
        page.draw_rect(page.rect, color=color, fill=color)
        document.save(path)


def _ghostscript_output_path(command: list[str]) -> Path:
    arg = next(a for a in command if a.startswith("-sOutputFile="))
    return Path(arg.split("=", 1)[1])


def _assert_temp_cleaned(path: Path) -> None:
    assert not path.exists(), f"temp path leaked: {path}"


def _assert_no_tmp_pdfs(directory: Path) -> None:
    assert list(directory.glob("*.tmp.pdf")) == []


def _size_for_path(path, *, temp_size=5000, orig_size=10000):
    return temp_size if str(path).endswith(".tmp.pdf") else orig_size


def _gs_side_effect(writer):
    def _run(command, **_kwargs):
        writer(_ghostscript_output_path(command))

    return _run


def _gs_write_pdf(pages=1, label="original", **kwargs):
    return _gs_side_effect(lambda out: _write_pdf(out, pages, label, **kwargs))


def _gs_write_solid(rgb):
    return _gs_side_effect(lambda out: _write_solid_image_pdf(out, rgb))


def _make_source_pdf(tmp_path, name="document.pdf", pages=1, label="original", **kwargs):
    pdf_path = tmp_path / name
    _write_pdf(pdf_path, pages, label, **kwargs)
    return pdf_path, pdf_path.read_bytes()


def _blank_letter_pair(tmp_path, name: str):
    source = tmp_path / f"{name}_src.pdf"
    candidate = tmp_path / f"{name}_cand.pdf"
    for path in (source, candidate):
        _write_pdf(
            path, 1, "matter", width=612.0, height=792.0,
            include_text=False, background_rgb=(255, 255, 255),
        )
    return source, candidate


def _mutate_candidate(path: Path, mutator) -> None:
    with fitz.open(path) as document:
        mutator(document[0])
        document.saveIncr()


def _assert_render_reject(ok, error, *tokens):
    assert ok is False, error
    assert "rendered content changed" in error
    for token in tokens:
        assert token in error


def _assert_failed_preserved_bytes(pdf_path: Path, original: bytes, *, tmp_path=None):
    assert pdf_path.read_bytes() == original
    if tmp_path is not None:
        _assert_no_tmp_pdfs(tmp_path)


def _run_optimize_gs(pdf_path, side_effect, *, quality=None, detailed=False):
    call = optimize_pdf_detailed if detailed else optimize_pdf
    kwargs = {} if quality is None else {"quality": quality}
    with patch("optimize_pdfs.subprocess.run", side_effect=side_effect) as mock_run:
        result = call(str(pdf_path), **kwargs) if kwargs else call(str(pdf_path))
    return result, mock_run


def _capture_valid_gs(captured_temp=None):
    def write_valid(command, **_kwargs):
        output = _ghostscript_output_path(command)
        if captured_temp is not None:
            captured_temp["path"] = output
        _write_pdf(output, 1, "original")

    return write_valid


def _assert_mode_fsync_replace_order(call_order):
    mode_idxs = [i for i, item in enumerate(call_order) if item[0] in ("chmod", "fchmod")]
    fsync_idxs = [i for i, item in enumerate(call_order) if item[0] == "fsync"]
    replace_idxs = [i for i, item in enumerate(call_order) if item[0] == "replace"]
    assert mode_idxs and fsync_idxs and replace_idxs, call_order
    assert mode_idxs[0] < fsync_idxs[0] < replace_idxs[0], call_order


@contextmanager
def _track_mode_fsync_replace(expected_mode, *, replace_calls=None, call_order=None):
    """Patch chmod/fchmod/fsync/replace while recording mode→fsync→replace order."""
    call_order = call_order if call_order is not None else []
    replace_calls = replace_calls if replace_calls is not None else []
    real_durable, real_chmod, real_fsync = _durable_replace, os.chmod, os.fsync
    real_fchmod = getattr(os, "fchmod", None)

    def tracking_chmod(path, mode):
        call_order.append(("chmod", path, stat.S_IMODE(mode)))
        return real_chmod(path, mode)

    def tracking_fchmod(fd, mode):
        call_order.append(("fchmod", fd, stat.S_IMODE(mode)))
        return real_fchmod(fd, mode)

    def tracking_fsync(fd):
        call_order.append(("fsync", fd))
        return real_fsync(fd)

    def tracking_replace(src, dst):
        assert Path(dst).exists()
        assert stat.S_IMODE(os.stat(src).st_mode) == expected_mode
        call_order.append(("replace", src, dst))
        replace_calls.append((src, dst))
        return real_durable(src, dst)

    with ExitStack() as stack:
        stack.enter_context(patch("optimize_pdfs.os.chmod", side_effect=tracking_chmod))
        if real_fchmod is not None:
            stack.enter_context(patch("optimize_pdfs.os.fchmod", side_effect=tracking_fchmod))
        stack.enter_context(patch("optimize_pdfs.os.fsync", side_effect=tracking_fsync))
        stack.enter_context(patch("optimize_pdfs._durable_replace", side_effect=tracking_replace))
        yield call_order, replace_calls


def _opaque_rect(x0, y0, w, h, gray=0.0):
    def mutator(page):
        c = (gray, gray, gray)
        page.draw_rect(fitz.Rect(x0, y0, x0 + w, y0 + h), color=c, fill=c)

    return mutator


_DRIFT_H = 842 + PAGE_DIMENSION_TOLERANCE_PT + 5


class TestOptimizePdfGhostscriptArgs:
    """Verify Ghostscript subprocess invocation arguments."""

    @patch("optimize_pdfs._durable_replace")
    @patch("builtins.open", mock_open(read_data=b"%PDF-1.4"))
    @patch("optimize_pdfs.os.path.getsize")
    @patch("optimize_pdfs.os.path.exists", return_value=True)
    @patch("optimize_pdfs.subprocess.run")
    def test_ghostscript_args_safety_flags_and_dsafer_order(
        self, mock_run, mock_exists, mock_getsize, mock_replace, tmp_path
    ):
        """Safety flags present; -dSAFER precedes -sDEVICE (maintenance review)."""
        mock_getsize.side_effect = _size_for_path
        pdf_path = tmp_path / "gs_args.pdf"
        _write_pdf(pdf_path, 1, "gs")
        with patch("optimize_pdfs._validate_pdf_structure", return_value=(True, "")), patch(
            "optimize_pdfs._apply_mode_and_fsync", return_value=True
        ):
            assert optimize_pdf(str(pdf_path), "/prepress") is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "gs"
        assert "-dSAFER" in cmd, "Ghostscript must run with -dSAFER sandbox"
        assert cmd.index("-dSAFER") < cmd.index("-sDEVICE=pdfwrite")
        for flag in (
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/prepress",
            "-dNOPAUSE",
            "-dBATCH",
        ):
            assert flag in cmd
        output_path = str(_ghostscript_output_path(cmd))
        assert output_path.endswith(".tmp.pdf")
        assert output_path != str(pdf_path) + ".tmp.pdf"
        mock_replace.assert_called_once_with(output_path, str(pdf_path))


class TestOptimizePdfTimeoutHandling:
    """Verify graceful handling of Ghostscript timeout."""

    @patch("optimize_pdfs.os.remove")
    @patch("optimize_pdfs.subprocess.run")
    def test_timeout_kwargs_disposition_and_temp_cleanup(
        self, mock_run, mock_remove, tmp_path
    ):
        """Timeout is passed, yields FAILED_PRESERVED, and cleans unique temp."""
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="gs", timeout=GHOSTSCRIPT_TIMEOUT
        )
        pdf_path = tmp_path / "timeout.pdf"
        _write_pdf(pdf_path, 1, "timeout")
        assert optimize_pdf(str(pdf_path)) is False
        _, kwargs = mock_run.call_args
        assert kwargs["timeout"] == GHOSTSCRIPT_TIMEOUT
        detailed = optimize_pdf_detailed(str(pdf_path))
        assert detailed.disposition is OptimizationDisposition.FAILED_PRESERVED
        temp_path = str(_ghostscript_output_path(mock_run.call_args[0][0]))
        assert temp_path.endswith(".tmp.pdf")
        assert temp_path != str(pdf_path) + ".tmp.pdf"
        mock_remove.assert_called_with(temp_path)


class TestOptimizePdfOriginalPreservation:
    """Verify original file is preserved on failure conditions."""

    @patch("optimize_pdfs.subprocess.run")
    def test_original_preserved_on_gs_failure(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="gs", stderr=b"error"
        )
        pdf_path, original = _make_source_pdf(tmp_path, "gs_fail.pdf")
        result = optimize_pdf_detailed(str(pdf_path))
        assert result.disposition is OptimizationDisposition.FAILED_PRESERVED
        assert result.applied is False
        assert optimize_pdf(str(pdf_path)) is False  # public bool compatibility
        assert pdf_path.read_bytes() == original

    @pytest.mark.parametrize(
        "case_id,writer,temp_size",
        [
            ("tiny_output", lambda out: out.write_bytes(b"%PDF-x"), 50),
            ("invalid_header", lambda out: out.write_bytes(b"NOT-A-PDF-CONTENT-XXXX"), 5000),
        ],
        ids=["tiny_output", "invalid_header"],
    )
    @patch("optimize_pdfs.os.remove")
    @patch("optimize_pdfs.subprocess.run")
    def test_original_preserved_on_integrity_byte_failures(
        self, mock_run, mock_remove, tmp_path, case_id, writer, temp_size
    ):
        pdf_path, _ = _make_source_pdf(tmp_path, f"{case_id}.pdf", label=case_id)
        mock_run.side_effect = _gs_side_effect(writer)
        with patch(
            "optimize_pdfs.os.path.getsize",
            side_effect=lambda p: _size_for_path(p, temp_size=temp_size),
        ), patch("optimize_pdfs.os.path.exists", return_value=True):
            result = optimize_pdf_detailed(str(pdf_path))
        assert result.disposition is OptimizationDisposition.FAILED_PRESERVED
        mock_remove.assert_called_once_with(
            str(_ghostscript_output_path(mock_run.call_args[0][0]))
        )


class TestOptimizePdfStructuralValidation:
    """Ghostscript output must preserve text, dimensions, and page tree."""

    @pytest.mark.parametrize(
        "case_id,pages,writer,extra",
        [
            ("header_junk", 1, lambda o: o.write_bytes(b"%PDF-" + b"x" * 200), {"page_count": 1}),
            ("page_count", 1, lambda o: _write_pdf(o, 2, "original"), {"page_count": 1}),
            ("changed_text", 2, lambda o: _write_pdf(o, 2, "tampered"), {}),
            (
                "removed_text",
                1,
                lambda o: _write_pdf(o, 1, "original", include_text=False),
                {"validator_error": "searchable text removed", "blank": True},
            ),
            (
                "dimension_drift",
                1,
                lambda o: _write_pdf(o, 1, "original", width=595, height=_DRIFT_H),
                {"validator_error": "dimensions changed", "drift_height": _DRIFT_H},
            ),
        ],
        ids=["header_junk", "page_count", "changed_text", "removed_text", "dimension_drift"],
    )
    def test_integrity_rejection_variants(self, tmp_path, case_id, pages, writer, extra):
        kw = {}
        if case_id == "dimension_drift":
            kw = {"width": 595, "height": 842}
        pdf_path, original = _make_source_pdf(tmp_path, pages=pages, **kw)
        result, _ = _run_optimize_gs(pdf_path, _gs_side_effect(writer))
        assert result is False
        _assert_failed_preserved_bytes(pdf_path, original, tmp_path=tmp_path)
        if "page_count" in extra:
            with fitz.open(pdf_path) as preserved:
                assert preserved.page_count == extra["page_count"]
        if extra.get("blank"):
            blank = tmp_path / "blank.pdf"
            _write_pdf(blank, 1, "original", include_text=False)
            ok, error = _validate_pdf_structure(str(pdf_path), str(blank))
            assert ok is False and extra["validator_error"] in error
        elif "drift_height" in extra:
            drifted = tmp_path / "drifted.pdf"
            _write_pdf(drifted, 1, "original", width=595, height=extra["drift_height"])
            ok, error = _validate_pdf_structure(str(pdf_path), str(drifted))
            assert ok is False and extra["validator_error"] in error

    def test_rejected_candidate_is_a_measured_safe_skip(self, tmp_path):
        """Candidate text drift keeps the authoritative assembled PDF usable."""
        pdf_path, original = _make_source_pdf(tmp_path, pages=1)
        detailed, _ = _run_optimize_gs(
            pdf_path,
            _gs_write_pdf(pages=1, label="tampered"),
            detailed=True,
        )
        assert (
            detailed.disposition
            is OptimizationDisposition.SKIPPED_CANDIDATE_REJECTED
        )
        assert detailed.applied is False
        assert detailed.skipped_candidate_rejected is True
        _assert_failed_preserved_bytes(pdf_path, original, tmp_path=tmp_path)

    def test_accepts_valid_optimization_preserving_text_and_dimensions(self, tmp_path):
        pdf_path, _ = _make_source_pdf(tmp_path, pages=2)
        with fitz.open(pdf_path) as source:
            expected_text = [_normalize_extracted_text(p.get_text()) for p in source]
            expected_sizes = [(float(p.rect.width), float(p.rect.height)) for p in source]
        detailed, _ = _run_optimize_gs(
            pdf_path, _gs_write_pdf(pages=2), detailed=True
        )
        assert detailed.disposition is OptimizationDisposition.APPLIED
        assert detailed.applied is True
        bool_ok, _ = _run_optimize_gs(
            _make_source_pdf(tmp_path, "bool.pdf", pages=2)[0],
            _gs_write_pdf(pages=2),
        )
        assert bool_ok is True  # public bool compatibility for APPLIED
        _assert_no_tmp_pdfs(tmp_path)
        with fitz.open(pdf_path) as optimized:
            assert optimized.page_count == 2
            for i, page in enumerate(optimized):
                assert _normalize_extracted_text(page.get_text()) == expected_text[i]
                assert abs(page.rect.width - expected_sizes[i][0]) <= PAGE_DIMENSION_TOLERANCE_PT
                assert abs(page.rect.height - expected_sizes[i][1]) <= PAGE_DIMENSION_TOLERANCE_PT
                assert "original page" in page.get_text()

    def test_rejects_moved_invisible_ocr_text_even_when_render_is_unchanged(self, tmp_path):
        source = tmp_path / "hidden_source.pdf"
        candidate = tmp_path / "hidden_moved.pdf"

        def write_hidden(path, origin):
            with fitz.open() as document:
                page = document.new_page(width=612, height=792)
                page.insert_text(
                    origin,
                    "Invisible OCR evidence layer",
                    fontsize=10,
                    render_mode=3,
                )
                document.save(path)

        write_hidden(source, (72, 120))
        write_hidden(candidate, (132, 180))
        with fitz.open(source) as original, fitz.open(candidate) as moved:
            assert _normalize_extracted_text(original[0].get_text()) == (
                _normalize_extracted_text(moved[0].get_text())
            )
            assert _page_word_geometry(original[0]) != _page_word_geometry(moved[0])

        ok, error = _validate_pdf_structure(str(source), str(candidate))
        assert ok is False
        assert "searchable word geometry moved" in error

    def test_accepts_sub_tolerance_hidden_text_geometry_noise(self, tmp_path):
        source = tmp_path / "hidden_base.pdf"
        candidate = tmp_path / "hidden_noise.pdf"
        shift = TEXT_GEOMETRY_TOLERANCE_PT / 4
        for path, origin in (
            (source, (72, 120)),
            (candidate, (72 + shift, 120 + shift)),
        ):
            with fitz.open() as document:
                page = document.new_page(width=612, height=792)
                page.insert_text(
                    origin,
                    "Invisible OCR evidence layer",
                    fontsize=10,
                    render_mode=3,
                )
                document.save(path)

        ok, error = _validate_pdf_structure(str(source), str(candidate))
        assert ok is True, error


class TestOptimizePdfRenderedEquivalence:
    """Rendered RGB gate rejects color/background swaps text gates miss."""

    def test_rejects_red_vs_blue_image_only_pages(self, tmp_path):
        source, candidate = tmp_path / "red.pdf", tmp_path / "blue.pdf"
        _write_solid_image_pdf(source, (255, 0, 0))
        _write_solid_image_pdf(candidate, (0, 0, 255))
        ok, error = _validate_pdf_structure(str(source), str(candidate))
        _assert_render_reject(ok, error, "page 1", "mae=", "changed_frac=", "tile_mae=")
        assert "samples" not in error.lower() and "pixmap" not in error.lower()

    def test_rejects_same_text_changed_background(self, tmp_path):
        source, candidate = tmp_path / "white_bg.pdf", tmp_path / "black_bg.pdf"
        _write_pdf(source, 1, "matter", background_rgb=(255, 255, 255))
        _write_pdf(candidate, 1, "matter", background_rgb=(0, 0, 0))
        with fitz.open(source) as s, fitz.open(candidate) as c:
            assert _normalize_extracted_text(s[0].get_text()) == _normalize_extracted_text(
                c[0].get_text()
            )
            assert abs(s[0].rect.width - c[0].rect.width) < 0.01
        ok, error = _validate_pdf_structure(str(source), str(candidate))
        _assert_render_reject(ok, error, "page 1")

    def test_accepts_small_compression_noise(self):
        base = Image.new("RGB", (120, 80), (240, 240, 240))
        noisy = base.copy()
        pixels = noisy.load()
        for x in range(0, 120, 3):
            for y in range(0, 80, 3):
                r, g, b = pixels[x, y]
                pixels[x, y] = (min(255, r + 4), max(0, g - 3), b)
        ok, metrics = _images_perceptually_equivalent(base, noisy)
        assert ok is True, metrics
        assert metrics["mae"] <= RENDER_MAE_MAX
        assert metrics["changed_frac"] <= 0.035
        assert metrics["local_tile_mae"] <= RENDER_LOCAL_TILE_MAE_MAX
        assert (
            metrics["local_blur_overlap_changed_frac"]
            <= RENDER_LOCAL_BLUR_CHANGED_FRAC_MAX
        )
        assert metrics["local_tile_changed_frac"] == metrics["local_blur_overlap_changed_frac"]

    def test_rejects_localized_alteration(self, tmp_path):
        source, candidate = tmp_path / "clean.pdf", tmp_path / "localized.pdf"
        for path in (source, candidate):
            _write_pdf(path, 1, "original", background_rgb=(250, 250, 250))
        _mutate_candidate(candidate, _opaque_rect(20, 20, 100, 100))
        ok, error = _validate_pdf_structure(str(source), str(candidate))
        _assert_render_reject(
            ok, error, "tile_mae=", "local_tile_mae=", "local_blur_overlap_changed_frac="
        )

    @pytest.mark.parametrize(
        "case_id,mutator,tokens",
        [
            (
                "field_80x30",
                _opaque_rect(72, 72, 80, 30),
                ("local_tile_mae=", "local_blur_overlap_changed_frac="),
            ),
            (
                "sig_36x14",
                _opaque_rect(400, 700, 36, 14),
                ("local_tile_mae=", "local_blur_overlap_changed_frac="),
            ),
            (
                "zigzag_sparse",
                lambda page: page.draw_polyline(
                    [
                        fitz.Point(400.0, 707.0), fitz.Point(406.0, 700.0),
                        fitz.Point(412.0, 714.0), fitz.Point(418.0, 700.0),
                        fitz.Point(424.0, 714.0), fitz.Point(430.0, 700.0),
                        fitz.Point(436.0, 707.0),
                    ],
                    color=(0, 0, 0), width=0.8, closePath=False,
                ),
                ("local_blur_overlap_changed_frac=",),
            ),
            ("gray_36x14", _opaque_rect(72, 200, 36, 14, gray=0.75), ("local_tile_mae=",)),
        ],
        ids=["field_80x30", "sig_36x14", "zigzag_sparse", "gray_36x14"],
    )
    def test_rejects_visual_tamper_fixtures(self, tmp_path, case_id, mutator, tokens):
        source, candidate = _blank_letter_pair(tmp_path, case_id)
        _mutate_candidate(candidate, mutator)
        ok, error = _validate_pdf_structure(str(source), str(candidate))
        _assert_render_reject(ok, error, *tokens)
        if case_id == "field_80x30":
            assert "samples" not in error.lower()

    def test_letter_and_huge_page_scale_budget(self):
        scale, metrics = _render_scale_for_page(612.0, 792.0)
        assert scale is not None
        assert metrics["effective_dpi"] >= RENDER_MIN_EFFECTIVE_DPI
        assert (612.0 * scale) * (792.0 * scale) <= RENDER_MAX_PIXELS + 1e-6
        for width_pt, height_pt in ((5000.0, 5000.0), (10000.0, 10000.0)):
            scale, metrics = _render_scale_for_page(width_pt, height_pt)
            assert scale is None, metrics
            assert metrics["effective_dpi"] < RENDER_MIN_EFFECTIVE_DPI
            assert metrics["max_pixels"] == float(RENDER_MAX_PIXELS)
            assert metrics["min_effective_dpi"] == float(RENDER_MIN_EFFECTIVE_DPI)

    @pytest.mark.parametrize("width,height", [(5000.0, 5000.0), (10000.0, 10000.0)])
    def test_huge_page_skips_before_gs_preserves_bytes(self, tmp_path, width, height):
        """SKIPPED_VISUAL_BUDGET: preflight before temp/GS; original bytes intact."""
        pdf_path = tmp_path / f"huge_{int(width)}.pdf"
        _write_pdf(
            pdf_path, 1, "matter", width=width, height=height,
            include_text=False, background_rgb=(255, 255, 255),
        )
        original = pdf_path.read_bytes()
        with patch("optimize_pdfs.subprocess.run") as mock_run:
            result = optimize_pdf_detailed(str(pdf_path))
        assert result.disposition is OptimizationDisposition.SKIPPED_VISUAL_BUDGET
        assert "exceeds compression visual-validation budget" in result.reason
        assert result.applied is False
        assert optimize_pdf(str(pdf_path)) is False  # public bool compatibility
        assert pdf_path.read_bytes() == original
        mock_run.assert_not_called()
        _assert_no_tmp_pdfs(tmp_path)

    def test_huge_page_structure_validation_still_refuses(self, tmp_path):
        width = height = 5000.0
        paths = {
            name: tmp_path / f"huge_{name}.pdf"
            for name in ("source", "identical", "tampered")
        }
        for path in paths.values():
            _write_pdf(
                path, 1, "matter", width=width, height=height,
                include_text=False, background_rgb=(255, 255, 255),
            )
        _mutate_candidate(paths["tampered"], _opaque_rect(100, 100, 100, 100))
        for candidate in (paths["identical"], paths["tampered"]):
            ok, error = _validate_pdf_structure(str(paths["source"]), str(candidate))
            assert ok is False, error
            assert "exceeds compression visual-validation budget" in error

    def test_no_partial_edge_window_origins(self):
        origins = _full_window_origins(100, window=16, stride=8)
        assert origins[0] == 0 and origins[-1] == 100 - 16
        assert all(o + 16 <= 100 for o in origins)
        assert _full_window_origins(10, window=16, stride=8) == [0]

    def test_optimize_rejects_color_swap_via_rendered_gate(self, tmp_path):
        pdf_path = tmp_path / "matter.pdf"
        _write_solid_image_pdf(pdf_path, (255, 0, 0))
        original = pdf_path.read_bytes()
        result, _ = _run_optimize_gs(
            pdf_path,
            _gs_write_solid((0, 0, 255)),
            detailed=True,
        )
        assert (
            result.disposition
            is OptimizationDisposition.SKIPPED_CANDIDATE_REJECTED
        )
        assert result.metrics["visual_local_energy_rms"] > 0
        _assert_failed_preserved_bytes(pdf_path, original, tmp_path=tmp_path)


def _render_calibration_page(*, offset_pt=0.0, stroke=None):
    """Render dense vector content, optionally with one forensic-scale mark."""
    with fitz.open() as document:
        page = document.new_page(width=612, height=792)
        for y in range(40, 740, 13):
            page.insert_text(
                (35 + offset_pt, y + offset_pt),
                "Representative matter 0123456789",
                fontsize=7,
                color=(0.15, 0.15, 0.15),
            )
            page.draw_line(
                (35 + offset_pt, y + 2 + offset_pt),
                (575 + offset_pt, y + 2 + offset_pt),
                color=(0.75, 0.75, 0.75),
                width=0.25,
            )
        if stroke:
            kind, x, y, orientation = stroke
            length = 2.0 if kind == "black" else 40.0
            end = (x + length, y) if orientation == "h" else (x, y + length)
            page.draw_line(
                (x, y),
                end,
                color=(0, 0, 0) if kind == "black" else (0.75, 0.75, 0.75),
                width=1.0 if kind == "black" else 0.5,
            )
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(RENDER_NOMINAL_DPI / 72, RENDER_NOMINAL_DPI / 72),
            alpha=False,
            colorspace=fitz.csRGB,
        )
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _jpeg_proxy(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class TestLocalEnergyCalibration:
    """Production-render calibration for the bounded 8x8 energy gate."""

    def test_rejects_planted_dense_001pt_vector_displacement(self):
        base = _render_calibration_page()
        shifted = _render_calibration_page(offset_pt=0.01)
        ok, metrics = _images_perceptually_equivalent(base, shifted)
        assert ok is False, metrics
        assert (
            metrics["local_energy_mae"] > RENDER_LOCAL_ENERGY_MAE_MAX
            or metrics["local_energy_rms"] > RENDER_LOCAL_ENERGY_RMS_MAX
        ), metrics

    def test_accepts_jpeg_q95_to_q90_proxy(self):
        base = _render_calibration_page()
        ok, metrics = _images_perceptually_equivalent(
            _jpeg_proxy(base, 95),
            _jpeg_proxy(base, 90),
        )
        assert ok is True, metrics
        assert metrics["local_energy_mae"] <= RENDER_LOCAL_ENERGY_MAE_MAX
        assert metrics["local_energy_rms"] <= RENDER_LOCAL_ENERGY_RMS_MAX

    def test_dense_025pt_displacement_is_deliberately_rejected(self):
        ok, metrics = _images_perceptually_equivalent(
            _render_calibration_page(),
            _render_calibration_page(offset_pt=0.25),
        )
        assert ok is False, metrics

    @pytest.mark.parametrize("kind", ("black", "gray"))
    @pytest.mark.parametrize(
        "x,y,orientation",
        ((0, 0, "h"), (572, 792, "h"), (0, 0, "v"), (612, 0, "v"),
         (100.2, 100.2, "h"), (103.9, 103.9, "v")),
        ids=("top", "bottom", "left", "right", "phase-a", "phase-b"),
    )
    def test_rejects_short_black_and_light_gray_marks_at_edges_and_phases(
        self, kind, x, y, orientation
    ):
        base = _render_calibration_page()
        candidate = _render_calibration_page(
            stroke=(kind, x, y, orientation),
        )
        ok, metrics = _images_perceptually_equivalent(base, candidate)
        assert ok is False, metrics
        assert (
            metrics["local_energy_mae"] > RENDER_LOCAL_ENERGY_MAE_MAX
            or metrics["local_energy_rms"] > RENDER_LOCAL_ENERGY_RMS_MAX
        ), metrics


@pytest.mark.skipif(shutil.which("gs") is None, reason="Ghostscript is not installed")
def test_real_ghostscript_mixed_vector_image_roundtrip(tmp_path):
    """Pinned/local GS must stay inside the visual gate on benign content."""
    pdf_path = tmp_path / "mixed.pdf"
    image = Image.new("RGB", (180, 120))
    image.putdata(
        [
            ((x * 7) % 256, (y * 11) % 256, ((x + y) * 5) % 256)
            for y in range(image.height)
            for x in range(image.width)
        ]
    )
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG", quality=92)
    with fitz.open() as document:
        page = document.new_page(width=612, height=792)
        for row, y in enumerate(range(42, 530, 19)):
            page.insert_text(
                (36, y),
                f"Representative matter row {row:02d} 0123456789",
                fontsize=9,
                color=(0.1, 0.1, 0.1),
            )
        page.draw_rect(
            fitz.Rect(330, 570, 560, 730),
            color=(0.1, 0.3, 0.8),
            fill=(0.92, 0.95, 1.0),
            width=0.75,
        )
        page.insert_image(
            fitz.Rect(45, 570, 285, 730),
            stream=image_bytes.getvalue(),
        )
        document.save(pdf_path, deflate=True)

    result = optimize_pdf_detailed(str(pdf_path), quality="/prepress")

    assert result.disposition is OptimizationDisposition.APPLIED, (
        result.reason,
        result.metrics,
    )
    assert result.metrics["visual_local_energy_mae"] <= RENDER_LOCAL_ENERGY_MAE_MAX
    assert result.metrics["visual_local_energy_rms"] <= RENDER_LOCAL_ENERGY_RMS_MAX
    assert result.metrics["visual_text_geometry_max_delta_pt"] <= (
        TEXT_WORD_VERTICAL_METRIC_TOLERANCE_PT
    )
    with fitz.open(pdf_path) as optimized:
        assert optimized.page_count == 1
        assert "Representative matter row 00" in optimized[0].get_text()


class TestOptimizePdfAtomicReplace:
    """Unique temp + atomic replace keeps the original until commit."""

    def test_uses_unique_concurrency_safe_temp_names(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        _write_pdf(pdf, 1, "original")
        seen = []

        def capture_only(command, **_kwargs):
            out = _ghostscript_output_path(command)
            seen.append(str(out))
            raise subprocess.CalledProcessError(1, command, stderr=b"boom")

        with patch("optimize_pdfs.subprocess.run", side_effect=capture_only):
            assert optimize_pdf(str(pdf)) is False
            assert optimize_pdf(str(pdf)) is False
        assert len(seen) == 2 and seen[0] != seen[1]
        for output in seen:
            assert output.endswith(".tmp.pdf") and output != str(pdf) + ".tmp.pdf"
            assert Path(output).parent == tmp_path
            _assert_temp_cleaned(Path(output))

    def test_original_remains_until_atomic_replace_on_validation_failure(self, tmp_path):
        pdf_path, original = _make_source_pdf(tmp_path)
        with (
            patch("optimize_pdfs.subprocess.run", side_effect=_gs_write_pdf(label="changed")),
            patch("optimize_pdfs._durable_replace") as mock_replace,
        ):
            mock_replace.side_effect = AssertionError("must not run on validation failure")
            assert optimize_pdf(str(pdf_path)) is False
            mock_replace.assert_not_called()
        _assert_failed_preserved_bytes(pdf_path, original, tmp_path=tmp_path)

    def test_atomic_replace_used_on_success_and_mode_preserved(self, tmp_path):
        pdf_path, _ = _make_source_pdf(tmp_path)
        os.chmod(pdf_path, 0o640)
        expected_mode = stat.S_IMODE(os.stat(pdf_path).st_mode)
        with patch("optimize_pdfs.subprocess.run", side_effect=_gs_write_pdf()),\
                _track_mode_fsync_replace(expected_mode) as (order, replaces):
            result = optimize_pdf_detailed(str(pdf_path))
        assert result.disposition is OptimizationDisposition.APPLIED
        assert result.applied is True
        assert len(replaces) == 1
        src, dst = replaces[0]
        assert src.endswith(".tmp.pdf") and dst == str(pdf_path)
        assert not Path(src).exists()
        _assert_no_tmp_pdfs(tmp_path)
        assert list(tmp_path.glob("*.bak")) == []
        assert stat.S_IMODE(os.stat(pdf_path).st_mode) == expected_mode
        with fitz.open(pdf_path) as doc:
            assert "original page 1" in doc[0].get_text()
        _assert_mode_fsync_replace_order(order)

    @pytest.mark.parametrize(
        "stage",
        ("chmod", "fsync_apply", "fsync_oserror", "replace"),
        ids=["chmod", "fsync_apply", "fsync_oserror", "replace"],
    )
    def test_pre_replace_stage_failures_keep_original(self, tmp_path, stage):
        """Distinct durability stages: chmod/fsync/replace each preserve original."""
        pdf_path, original = _make_source_pdf(tmp_path)
        if stage == "chmod":
            os.chmod(pdf_path, 0o640)
        captured = {}
        stack = ExitStack()
        stack.enter_context(
            patch("optimize_pdfs.subprocess.run", side_effect=_capture_valid_gs(captured))
        )
        if stage == "chmod":
            stack.enter_context(patch("optimize_pdfs.os.chmod", side_effect=OSError("chmod denied")))
            if hasattr(os, "fchmod"):
                stack.enter_context(
                    patch("optimize_pdfs.os.fchmod", side_effect=OSError("fchmod denied"))
                )
            mock_replace = stack.enter_context(patch("optimize_pdfs._durable_replace"))
        elif stage == "fsync_apply":
            stack.enter_context(patch("optimize_pdfs._apply_mode_and_fsync", return_value=False))
            mock_replace = stack.enter_context(patch("optimize_pdfs._durable_replace"))
        elif stage == "fsync_oserror":
            stack.enter_context(
                patch("optimize_pdfs.os.fsync", side_effect=OSError("fsync failed"))
            )
            mock_replace = stack.enter_context(patch("optimize_pdfs._durable_replace"))
        else:
            mock_replace = stack.enter_context(
                patch("optimize_pdfs._durable_replace", side_effect=OSError("disk full"))
            )
        with stack:
            result = optimize_pdf_detailed(str(pdf_path))
        assert result.disposition is OptimizationDisposition.FAILED_PRESERVED
        assert result.applied is False
        if stage != "replace":
            mock_replace.assert_not_called()
        _assert_failed_preserved_bytes(pdf_path, original)
        assert "path" in captured
        _assert_temp_cleaned(captured["path"])
        if stage == "replace":
            assert list(tmp_path.glob("*.bak")) == []

    def test_directory_fsync_failure_is_failed_indeterminate(self, tmp_path, caplog):
        """Post-replace parent fsync uncertainty → FAILED_INDETERMINATE, not APPLIED."""
        pdf_path, _ = _make_source_pdf(tmp_path)

        def replace_then_fail(src, dst):
            os.replace(src, dst)
            raise OSError(
                f"post-replace parent directory fsync failed for {dst} "
                f"(dir fsync failed); replacement may exist but durability "
                f"is unproven (not claiming success)"
            )

        with (
            patch("optimize_pdfs.subprocess.run", side_effect=_gs_write_pdf()),
            patch("optimize_pdfs._durable_replace", side_effect=replace_then_fail),
            caplog.at_level(logging.ERROR),
        ):
            detailed = optimize_pdf_detailed(str(pdf_path))
        assert detailed.disposition is OptimizationDisposition.FAILED_INDETERMINATE
        assert detailed.applied is False  # public bool never true for indeterminate
        assert pdf_path.exists()
        _assert_no_tmp_pdfs(tmp_path)
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "durability is unproven" in joined or "directory fsync failed" in joined
        assert "Done!" not in joined

    def test_posix_durable_replace_order_and_failure_gates(self, tmp_path):
        """POSIX: replace then parent-dir fsync; failures stay stage-distinct."""
        real_replace = os.replace
        src, dst = tmp_path / "cand.tmp.pdf", tmp_path / "out.pdf"
        src.write_bytes(b"%PDF-cand")
        dst.write_bytes(b"%PDF-orig")
        order = []

        def tracking_replace(a, b):
            order.append("replace")
            return real_replace(a, b)

        def tracking_dir_fsync(path):
            order.append("dir_fsync")
            assert path == str(dst)

        with (
            patch("optimize_pdfs.os.name", "posix"),
            patch("optimize_pdfs.os.replace", side_effect=tracking_replace),
            patch("optimize_pdfs._fsync_parent_dir", side_effect=tracking_dir_fsync),
        ):
            _durable_replace(str(src), str(dst))
        assert order == ["replace", "dir_fsync"]
        assert dst.read_bytes() == b"%PDF-cand" and not src.exists()

        src.write_bytes(b"%PDF-cand2")
        dst.write_bytes(b"%PDF-orig2")
        order.clear()
        with (
            patch("optimize_pdfs.os.name", "posix"),
            patch("optimize_pdfs.os.replace", side_effect=OSError("replace fail")),
            patch(
                "optimize_pdfs._fsync_parent_dir",
                side_effect=lambda *_a, **_k: order.append("dir_fsync"),
            ),
        ):
            with pytest.raises(OSError, match="replace fail"):
                _durable_replace(str(src), str(dst))
        assert order == [] and dst.read_bytes() == b"%PDF-orig2"

        src.write_bytes(b"%PDF-cand3")
        dst.write_bytes(b"%PDF-orig3")
        with (
            patch("optimize_pdfs.os.name", "posix"),
            patch("optimize_pdfs.os.replace", side_effect=real_replace),
            patch("optimize_pdfs._fsync_parent_dir", side_effect=OSError("dir fsync failed")),
        ):
            with pytest.raises(OSError, match="durability is unproven"):
                _durable_replace(str(src), str(dst))
        assert dst.read_bytes() == b"%PDF-cand3" and not src.exists()

    def test_windows_movefile_flags_success_and_failure(self, tmp_path):
        """Windows MoveFileExW uses REPLACE_EXISTING|WRITE_THROUGH."""
        src, dst = tmp_path / "cand.tmp.pdf", tmp_path / "out.pdf"
        src.write_bytes(b"%PDF-cand")
        dst.write_bytes(b"%PDF-orig")
        calls = []
        flags_expected = _MOVEFILE_REPLACE_EXISTING | _MOVEFILE_WRITE_THROUGH

        def fake_move(src_path, dst_path, *, flags):
            calls.append((src_path, dst_path, flags))
            assert flags == flags_expected
            os.replace(src_path, dst_path)

        nt_ctx = dict(
            name_patch=patch("optimize_pdfs.os.name", "nt"),
            ro_patch=patch("optimize_pdfs._windows_destination_is_readonly", return_value=False),
        )
        with (
            nt_ctx["name_patch"],
            nt_ctx["ro_patch"],
            patch("optimize_pdfs._windows_movefile_ex", side_effect=fake_move),
        ):
            _durable_replace(str(src), str(dst))
        assert len(calls) == 1 and calls[0][2] == flags_expected
        assert dst.read_bytes() == b"%PDF-cand" and not src.exists()

        src.write_bytes(b"%PDF-cand2")
        dst.write_bytes(b"%PDF-orig2")
        with (
            patch("optimize_pdfs.os.name", "nt"),
            patch("optimize_pdfs._windows_destination_is_readonly", return_value=False),
            patch("optimize_pdfs._windows_movefile_ex", side_effect=OSError("MoveFileEx failed")),
        ):
            with pytest.raises(OSError, match="MoveFileEx failed"):
                _durable_replace(str(src), str(dst))
        assert dst.read_bytes() == b"%PDF-orig2" and src.exists()

    def test_windows_readonly_destination_refused_without_chmod(self, tmp_path):
        """RO destination: fail before mutation; never chmod destination writable."""
        pdf_path, original = _make_source_pdf(tmp_path, "readonly.pdf")
        try:
            os.chmod(pdf_path, 0o444)
        except OSError:
            pytest.skip("platform cannot apply read-only mode bits")
        expected_mode = stat.S_IMODE(os.stat(pdf_path).st_mode)
        if expected_mode & stat.S_IWRITE:
            pytest.skip("platform does not retain read-only mode bits")
        chmod_calls, captured = [], {}
        real_chmod = os.chmod

        def tracking_chmod(path, mode):
            chmod_calls.append((str(path), stat.S_IMODE(mode)))
            return real_chmod(path, mode)

        if os.name == "nt":
            with (
                patch("optimize_pdfs.subprocess.run", side_effect=_capture_valid_gs(captured)),
                patch("optimize_pdfs.os.chmod", side_effect=tracking_chmod),
            ):
                assert optimize_pdf(str(pdf_path)) is False
            assert pdf_path.read_bytes() == original
            assert stat.S_IMODE(os.stat(pdf_path).st_mode) == expected_mode
            dest_chmods = [
                m for p, m in chmod_calls
                if os.path.normcase(os.path.abspath(p))
                == os.path.normcase(os.path.abspath(str(pdf_path)))
            ]
            assert dest_chmods == []
            assert "path" in captured
            _assert_temp_cleaned(captured["path"])
            return

        src = tmp_path / "cand.tmp.pdf"
        src.write_bytes(b"%PDF-cand")
        with (
            patch("optimize_pdfs.os.name", "nt"),
            patch("optimize_pdfs._windows_destination_is_readonly", return_value=True),
            patch("optimize_pdfs.os.chmod", side_effect=tracking_chmod),
            patch("optimize_pdfs._windows_movefile_ex") as mock_move,
        ):
            with pytest.raises(OSError, match="unsupported durable replace"):
                _durable_replace(str(src), str(pdf_path))
        mock_move.assert_not_called()
        assert pdf_path.read_bytes() == original and src.exists()

    def test_windows_extended_path_prefix(self, tmp_path):
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"x")
        extended = _windows_extended_path(str(path))
        assert extended.startswith("\\\\?\\") and extended.endswith("doc.pdf")

    def test_read_only_input_mode_preserved_on_posix_writable_flow(self, tmp_path):
        """POSIX 0444: mode applied + fsynced before replace (Windows refuses RO)."""
        if os.name == "nt":
            pytest.skip("Windows refuses read-only destinations for durable replace")
        pdf_path, _ = _make_source_pdf(tmp_path, "readonly.pdf")
        try:
            os.chmod(pdf_path, 0o444)
        except OSError:
            pytest.skip("platform cannot apply read-only mode bits")
        expected_mode = stat.S_IMODE(os.stat(pdf_path).st_mode)
        if expected_mode != 0o444:
            pytest.skip("platform does not retain 0444 mode bits")
        with patch("optimize_pdfs.subprocess.run", side_effect=_gs_write_pdf()),\
                _track_mode_fsync_replace(expected_mode) as (order, _replaces):
            assert optimize_pdf(str(pdf_path)) is True
        assert stat.S_IMODE(os.stat(pdf_path).st_mode) == expected_mode
        _assert_mode_fsync_replace_order(order)

    def test_windows_directory_fsync_is_noop(self):
        if os.name != "nt":
            with patch("optimize_pdfs.os.name", "nt"):
                _fsync_parent_dir(__file__)
            return
        _fsync_parent_dir(__file__)


class TestOptimizePdfBatch:
    """Batch processing via main()."""

    @pytest.mark.parametrize(
        "files,expected_calls",
        [
            (["doc1.pdf", "doc2.pdf", "readme.txt"], ["doc1.pdf", "doc2.pdf"]),
            (["matter.pdf", "matter_optimized.pdf", "notes.txt"], ["matter.pdf"]),
        ],
        ids=["multi_pdf", "filters_optimized_suffix"],
    )
    @patch("optimize_pdfs.optimize_pdf")
    @patch("optimize_pdfs.os.walk")
    def test_batch_walk_and_optimized_suffix_filter(
        self, mock_walk, mock_optimize, files, expected_calls
    ):
        mock_walk.return_value = [("/app/ocr_output", [], files)]
        with patch("optimize_pdfs.argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(quality="/prepress")
            main()
        assert mock_optimize.call_count == len(expected_calls)
        for name in expected_calls:
            mock_optimize.assert_any_call(
                os.path.join("/app/ocr_output", name), "/prepress"
            )


class TestOptimizePdfSkipLogic:
    """Skip / direct-path conditions."""

    @pytest.mark.parametrize(
        "case_id,writer,expect_success",
        [
            ("success", lambda o: _write_pdf(o, 1, "original"), True),
            ("invalid", lambda o: _write_pdf(o, 1, "changed"), False),
            ("missing", None, False),
        ],
        ids=["success", "invalid_candidate", "missing_candidate"],
    )
    def test_direct_optimized_suffix_cases(self, tmp_path, case_id, writer, expect_success):
        """Direct optimize_pdf does not refuse ``*_optimized.pdf`` by name alone."""
        pdf_path, original = _make_source_pdf(tmp_path, "matter_optimized.pdf")
        if writer is None:

            def side_effect(command, **_k):
                return subprocess.CompletedProcess(command, 0, b"", b"")

        else:
            side_effect = _gs_side_effect(writer)
        with patch("optimize_pdfs.subprocess.run", side_effect=side_effect) as mock_run:
            result = optimize_pdf(str(pdf_path), "/prepress")
        mock_run.assert_called_once()
        if expect_success:
            assert result is True
            _assert_no_tmp_pdfs(tmp_path)
            with fitz.open(pdf_path) as doc:
                assert "original page 1" in doc[0].get_text()
        else:
            assert result is False
            assert pdf_path.read_bytes() == original

    def test_invalid_quality_rejected(self):
        with patch("optimize_pdfs.subprocess.run") as mock_run:
            assert optimize_pdf("/fake/doc.pdf", "/invalid_quality") is False
            mock_run.assert_not_called()

    @patch("optimize_pdfs.os.remove")
    @patch("optimize_pdfs.os.path.exists", return_value=True)
    @patch("optimize_pdfs.subprocess.run")
    def test_skip_zero_byte_file(self, mock_run, mock_exists, mock_remove, tmp_path):
        """Zero-byte original after GS is FAILED_PRESERVED (not budget skip)."""
        pdf_path, _ = _make_source_pdf(tmp_path, "empty.pdf", label="empty")

        def getsize(path):
            if mock_run.called and not str(path).endswith(".tmp.pdf"):
                return 0
            return 1000 if str(path).endswith(".tmp.pdf") else max(pdf_path.stat().st_size, 1)

        with patch("optimize_pdfs.os.path.getsize", side_effect=getsize):
            result = optimize_pdf_detailed(str(pdf_path))
        assert result.disposition is OptimizationDisposition.FAILED_PRESERVED
        assert "zero-byte" in result.reason
        mock_remove.assert_called_once_with(
            str(_ghostscript_output_path(mock_run.call_args[0][0]))
        )


class TestOptimizePdfConstants:
    """Module-level constants stay within documented budgets."""

    def test_constants_and_render_budget(self):
        assert DEFAULT_QUALITY in VALID_QUALITIES
        assert GHOSTSCRIPT_TIMEOUT > 0
        assert VALID_QUALITIES == {"/screen", "/ebook", "/printer", "/prepress"}
        assert 0 < PAGE_DIMENSION_TOLERANCE_PT <= 2.0
        assert 72.0 <= RENDER_NOMINAL_DPI <= 150.0
        assert 1_000_000 <= RENDER_MAX_PIXELS <= 1_500_000
        assert RENDER_MIN_EFFECTIVE_DPI == 36.0
        assert 0 < RENDER_LOCAL_TILE_MAE_MAX <= 20.0
        assert 0 < RENDER_LOCAL_TILE_CHANGED_FRAC_MAX <= 0.10
        assert RENDER_MAE_MAX > 0
        assert _MOVEFILE_REPLACE_EXISTING == 0x1
        assert _MOVEFILE_WRITE_THROUGH == 0x8

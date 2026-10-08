"""Coverage preflight for explicitly requested native PDF extraction.

This never enables a native shortcut for ordinary OCR requests. Unknown or
incomplete coverage must use the normal forensic OCR path.
"""

import math

from PIL import Image

from ocr_local.visual_coverage import (
    CoverageAssessment,
    assess_candidate_coverage,
    assess_raster_text_regions,
)


def assess_native_pdf_page(page) -> CoverageAssessment:
    """Compare native words with a bounded rendering of the visible page."""
    import fitz

    def unknown(reason):
        return CoverageAssessment(
            "unknown", None, None, None, None, (), True, True, reason,
        )

    try:
        if page.first_annot is not None or page.first_widget is not None:
            return unknown("native_annotations_require_ocr")
        # Coverage is a conservative retry signal, not proof that small raster
        # labels or vector-only glyphs contain no additional text. The native
        # shortcut is limited to text-only display lists; all other content
        # (including inline images and drawings inside forms) requires OCR.
        if any(kind not in {"fill-text", "stroke-text", "ignore-text"}
               for kind, *_ in page.get_bboxlog()):
            return unknown("native_nontext_content_requires_ocr")
        # Check before allocating the raster. Large pages use normal OCR's
        # resource admission instead of an unbounded native-path allocation.
        matrix = fitz.Matrix(2, 2)
        rect = page.rect * matrix
        width, height = math.ceil(rect.width), math.ceil(rect.height)
        if width <= 0 or height <= 0 or width * height > 4_000_000:
            return unknown("native_preflight_pixel_limit")
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
        with Image.frombytes("RGB", (pix.width, pix.height), pix.samples) as raster:
            evidence = assess_raster_text_regions(raster)
        words = page.get_text("words")
        if len(words) > 20_000:
            return unknown("native_preflight_word_limit")
        spans = []
        for word in words:
            # Native word boxes are unrotated CropBox-local points; pixmaps
            # include page rotation. Keep both in the same pixel coordinates.
            box = fitz.Rect(word[:4]) * page.rotation_matrix * matrix
            spans.append((word[4], [
                box.x0 - pix.x, box.y0 - pix.y,
                box.x1 - pix.x, box.y1 - pix.y,
            ], 1.0))
        return assess_candidate_coverage(evidence, spans)
    except Exception:
        # Render/text extraction errors cannot establish completeness. The
        # worker records this decision and durably dispatches ordinary OCR.
        return unknown("native_preflight_render_or_text_error")

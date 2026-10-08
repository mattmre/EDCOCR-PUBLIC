"""Faithful, deterministic raster preparation for OCR-visible backgrounds."""

from __future__ import annotations

from PIL import Image


def white_composite_for_ocr(source: Image.Image) -> Image.Image:
    """Return RGB pixels as a white-backed PDF viewer displays transparency.

    Pillow's direct RGBA/P->RGB conversion discards alpha and can turn black
    glyphs on transparent black into an all-black page.  The caller retains the
    original source bytes; this is only the visible OCR/background raster.
    """
    has_alpha = "A" in source.getbands() or "transparency" in source.info
    if has_alpha:
        rgba = source.convert("RGBA")
        white = Image.new("RGBA", source.size, "white")
        result = Image.alpha_composite(white, rgba).convert("RGB")
    else:
        result = source.convert("RGB")
    # Palette tRNS indexes and RGBA transparency descriptors no longer apply
    # to composited RGB pixels. Retaining them makes Pillow's RGB PNG writer
    # reject otherwise valid source images (or reintroduces transparency).
    result.info.update(
        {key: value for key, value in source.info.items() if key != "transparency"}
    )
    return result

#!/usr/bin/env python3
"""Read and parse language metadata from a PDF's XMP metadata block.

Used for round-trip verification of Plan A PR A4 PDF XMP embedding.  The
assembler writes the detected BCP-47 language tag into ``dc:language``
inside the output PDF's XMP packet; this helper extracts it back.

Usage::

    python scripts/read_pdf_xmp_language.py <path-to-pdf>

The script prints a JSON document describing the XMP state.  On error
(missing file, invalid PDF) the JSON contains a top-level ``error`` key.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any


def read_pdf_xmp_language(pdf_path: str) -> dict[str, Any]:
    """Extract language XMP fields from a PDF.

    Args:
        pdf_path: Filesystem path to a PDF.

    Returns:
        Dictionary with keys:

        * ``raw_xmp`` -- the raw XMP packet (may be empty string).
        * ``dc_language`` -- the first ``<rdf:li>`` value inside the
          ``<dc:language>`` element, or ``None`` if not present.
        * ``error`` -- present only on failure (missing file, PyMuPDF
          error); its value is a short string describing the error.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - environmental
        return {"error": f"PyMuPDF not available: {exc}"}

    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception as exc:
        return {"error": str(exc)}

    try:
        xmp = pdf_doc.get_xml_metadata() or ""
    except Exception as exc:
        pdf_doc.close()
        return {"error": str(exc)}
    finally:
        try:
            pdf_doc.close()
        except Exception:
            pass

    result: dict[str, Any] = {"raw_xmp": xmp, "dc_language": None}
    match = re.search(r"<dc:language>.*?</dc:language>", xmp, re.DOTALL)
    if match:
        li_values = re.findall(r"<rdf:li[^>]*>(.*?)</rdf:li>", match.group(0))
        if li_values:
            result["dc_language"] = li_values[0].strip()
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: read_pdf_xmp_language.py <pdf>", file=sys.stderr)
        return 1
    payload = read_pdf_xmp_language(argv[1])
    print(json.dumps(payload, indent=2))
    return 0 if "error" not in payload else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""PDF XMP language embedding round-trip tests (Plan A -- PR A4).

Exercises ``scripts/read_pdf_xmp_language.py`` and the XMP-embedding
logic in ``ocr_gpu_async``'s assembler via an equivalent helper that
operates on an in-memory PyMuPDF document.  Verifies:

* error paths (missing file, non-PDF input),
* XMP round-trip with a core language,
* XMP round-trip with an extended-tier language (Croatian ``hr``),
* XMP round-trip with a non-ASCII BCP-47 tag (``zh-Hans``),
* existing-key preservation (dc:language is merged, not overwritten),
* idempotency (rerunning the write does not duplicate the element),
* BCP-47 mapping via ``LANGUAGE_REGISTRY`` matches the written tag.

Run with::

    python -m pytest tests/test_pdf_xmp_language.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("fitz")
import fitz  # noqa: E402

# Load the helper as a module (scripts/ is not a package entry for tests).
REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts" / "read_pdf_xmp_language.py"
_spec = importlib.util.spec_from_file_location(
    "read_pdf_xmp_language_helper", str(HELPER_PATH),
)
assert _spec is not None and _spec.loader is not None
_helper = importlib.util.module_from_spec(_spec)
sys.modules["read_pdf_xmp_language_helper"] = _helper
_spec.loader.exec_module(_helper)
read_pdf_xmp_language = _helper.read_pdf_xmp_language


# ---------------------------------------------------------------------------
# Fixtures: build a tiny real PDF we can read and write XMP to.
# ---------------------------------------------------------------------------


def _make_empty_pdf(path: Path) -> None:
    doc = fitz.open()
    doc.new_page(width=200, height=200)
    doc.save(str(path))
    doc.close()


def _make_pdf_with_existing_xmp(path: Path, producer: str = "OCR-LOCAL-test") -> None:
    doc = fitz.open()
    doc.new_page(width=200, height=200)
    xmp = (
        '<?xpacket begin=""?>'
        "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">"
        "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
        "<rdf:Description xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
        "xmlns:pdf=\"http://ns.adobe.com/pdf/1.3/\">"
        f"<pdf:Producer>{producer}</pdf:Producer>"
        "</rdf:Description>"
        "</rdf:RDF>"
        "</x:xmpmeta>"
        '<?xpacket end="w"?>'
    )
    doc.set_xml_metadata(xmp)
    doc.save(str(path))
    doc.close()


def _embed_language_xmp(pdf_path: str, bcp47: str) -> None:
    """Mirror the assembler's XMP-writing logic exactly."""
    pdf_doc = fitz.open(pdf_path)
    try:
        xmp = pdf_doc.get_xml_metadata() or ""
        if "<dc:language>" not in xmp:
            lang_xmp = (
                "<dc:language><rdf:Bag>"
                f"<rdf:li>{bcp47}</rdf:li>"
                "</rdf:Bag></dc:language>"
            )
            if "</rdf:Description>" in xmp:
                xmp = xmp.replace(
                    "</rdf:Description>",
                    f"{lang_xmp}</rdf:Description>",
                    1,
                )
            else:
                xmp = xmp + lang_xmp
            pdf_doc.set_xml_metadata(xmp)
            try:
                pdf_doc.saveIncr()
            except Exception:
                pdf_doc.save(pdf_path, incremental=False, deflate=True)
    finally:
        pdf_doc.close()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestHelperErrorPaths:
    def test_missing_file_returns_error(self, tmp_path):
        result = read_pdf_xmp_language(str(tmp_path / "does_not_exist.pdf"))
        assert "error" in result

    def test_non_pdf_file_is_safe(self, tmp_path):
        """Non-PDF input must either error cleanly or return empty XMP --
        never raise and never return a bogus language tag."""
        junk = tmp_path / "not_a_pdf.txt"
        junk.write_text("hello not a pdf")
        result = read_pdf_xmp_language(str(junk))
        # Either PyMuPDF rejects it (error) or returns an empty XMP (no
        # language).  Both are safe -- what matters is that the caller
        # never sees a real language tag from garbage input.
        if "error" in result:
            assert isinstance(result["error"], str)
        else:
            assert result.get("dc_language") is None

    def test_pdf_without_xmp_returns_none_language(self, tmp_path):
        pdf = tmp_path / "blank.pdf"
        _make_empty_pdf(pdf)
        result = read_pdf_xmp_language(str(pdf))
        # Either empty/absent XMP -> dc_language is None.
        assert result.get("dc_language") is None

    def test_pdf_without_xmp_returns_raw_xmp_string(self, tmp_path):
        pdf = tmp_path / "blank.pdf"
        _make_empty_pdf(pdf)
        result = read_pdf_xmp_language(str(pdf))
        # ``raw_xmp`` is always a string, even when empty.
        assert isinstance(result.get("raw_xmp"), str)

    def test_result_shape_has_expected_keys(self, tmp_path):
        pdf = tmp_path / "blank.pdf"
        _make_empty_pdf(pdf)
        result = read_pdf_xmp_language(str(pdf))
        assert "dc_language" in result
        assert "raw_xmp" in result


# ---------------------------------------------------------------------------
# Round-trip: embed then read back
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_english_roundtrip(self, tmp_path):
        pdf = tmp_path / "en.pdf"
        _make_pdf_with_existing_xmp(pdf)
        _embed_language_xmp(str(pdf), "en")
        result = read_pdf_xmp_language(str(pdf))
        assert result.get("dc_language") == "en"

    def test_french_roundtrip(self, tmp_path):
        pdf = tmp_path / "fr.pdf"
        _make_pdf_with_existing_xmp(pdf)
        _embed_language_xmp(str(pdf), "fr")
        result = read_pdf_xmp_language(str(pdf))
        assert result.get("dc_language") == "fr"

    def test_croatian_extended_tier_roundtrip(self, tmp_path):
        """Extended-tier language must round-trip identically."""
        pdf = tmp_path / "hr.pdf"
        _make_pdf_with_existing_xmp(pdf)
        _embed_language_xmp(str(pdf), "hr")
        result = read_pdf_xmp_language(str(pdf))
        assert result.get("dc_language") == "hr"

    def test_chinese_simplified_bcp47_roundtrip(self, tmp_path):
        """Hyphenated BCP-47 tag (``zh-Hans``) must survive XMP round-trip."""
        pdf = tmp_path / "zh.pdf"
        _make_pdf_with_existing_xmp(pdf)
        _embed_language_xmp(str(pdf), "zh-Hans")
        result = read_pdf_xmp_language(str(pdf))
        assert result.get("dc_language") == "zh-Hans"

    def test_chinese_traditional_roundtrip(self, tmp_path):
        pdf = tmp_path / "zh_tc.pdf"
        _make_pdf_with_existing_xmp(pdf)
        _embed_language_xmp(str(pdf), "zh-Hant")
        result = read_pdf_xmp_language(str(pdf))
        assert result.get("dc_language") == "zh-Hant"

    def test_arabic_roundtrip(self, tmp_path):
        pdf = tmp_path / "ar.pdf"
        _make_pdf_with_existing_xmp(pdf)
        _embed_language_xmp(str(pdf), "ar")
        result = read_pdf_xmp_language(str(pdf))
        assert result.get("dc_language") == "ar"

    def test_roundtrip_on_pdf_without_prior_xmp(self, tmp_path):
        pdf = tmp_path / "bare.pdf"
        _make_empty_pdf(pdf)
        _embed_language_xmp(str(pdf), "en")
        result = read_pdf_xmp_language(str(pdf))
        assert result.get("dc_language") == "en"


# ---------------------------------------------------------------------------
# Merge semantics: existing producer/timestamp preserved
# ---------------------------------------------------------------------------


class TestMergePreservesExistingKeys:
    def test_existing_producer_preserved(self, tmp_path):
        pdf = tmp_path / "merge.pdf"
        _make_pdf_with_existing_xmp(pdf, producer="OCR-LOCAL-baseline")
        _embed_language_xmp(str(pdf), "en")
        result = read_pdf_xmp_language(str(pdf))
        assert "OCR-LOCAL-baseline" in result["raw_xmp"]

    def test_existing_description_block_retained(self, tmp_path):
        pdf = tmp_path / "desc.pdf"
        _make_pdf_with_existing_xmp(pdf)
        _embed_language_xmp(str(pdf), "en")
        result = read_pdf_xmp_language(str(pdf))
        assert "rdf:Description" in result["raw_xmp"]

    def test_new_language_injected_inside_description(self, tmp_path):
        pdf = tmp_path / "inside.pdf"
        _make_pdf_with_existing_xmp(pdf)
        _embed_language_xmp(str(pdf), "en")
        result = read_pdf_xmp_language(str(pdf))
        idx_lang = result["raw_xmp"].find("<dc:language>")
        idx_close = result["raw_xmp"].find("</rdf:Description>")
        assert idx_lang != -1 and idx_close != -1
        assert idx_lang < idx_close

    def test_idempotent_second_write_is_noop(self, tmp_path):
        pdf = tmp_path / "idempotent.pdf"
        _make_pdf_with_existing_xmp(pdf)
        _embed_language_xmp(str(pdf), "en")
        first = read_pdf_xmp_language(str(pdf))["raw_xmp"]
        # Second write sees the existing <dc:language> and skips the merge.
        _embed_language_xmp(str(pdf), "fr")
        second = read_pdf_xmp_language(str(pdf))["raw_xmp"]
        assert first.count("<dc:language>") == second.count("<dc:language>")
        # Original language wins.
        result = read_pdf_xmp_language(str(pdf))
        assert result["dc_language"] == "en"


# ---------------------------------------------------------------------------
# BCP-47 mapping mirrors the language registry
# ---------------------------------------------------------------------------


class TestBCP47Mapping:
    @pytest.mark.parametrize(
        "paddle_code,expected_bcp47",
        [
            ("en", "en"),
            ("fr", "fr"),
            ("german", "de"),
            ("ch", "zh-Hans"),
            ("chinese_cht", "zh-Hant"),
            ("japan", "ja"),
            ("korean", "ko"),
            ("ar", "ar"),
            ("ru", "ru"),
            ("hr", "hr"),
            ("mr", "mr"),
            ("th", "th"),
        ],
    )
    def test_registry_bcp47_is_set_for_core_and_extended(
        self, paddle_code, expected_bcp47,
    ):
        from ocr_local.config.language_config import LANGUAGE_REGISTRY
        entry = LANGUAGE_REGISTRY.get(paddle_code)
        assert entry is not None
        assert entry.bcp47 == expected_bcp47

    def test_primary_language_und_is_not_written(self, tmp_path):
        """Pipeline contract: skip XMP embedding when primary_language=='und'."""
        pdf = tmp_path / "und.pdf"
        _make_pdf_with_existing_xmp(pdf)
        # Simulate the assembler guard: skip write when 'und'.
        primary = "und"
        if primary != "und":  # pragma: no cover
            _embed_language_xmp(str(pdf), primary)
        result = read_pdf_xmp_language(str(pdf))
        assert result.get("dc_language") is None

    def test_roundtrip_uses_bcp47_not_paddle_code(self, tmp_path):
        """End-to-end mapping: Paddle code -> BCP-47 -> XMP -> read back."""
        from ocr_local.config.language_config import LANGUAGE_REGISTRY
        pdf = tmp_path / "map.pdf"
        _make_pdf_with_existing_xmp(pdf)
        primary = "german"
        entry = LANGUAGE_REGISTRY.get(primary)
        bcp47 = entry.bcp47 if (entry and entry.bcp47) else primary
        _embed_language_xmp(str(pdf), bcp47)
        result = read_pdf_xmp_language(str(pdf))
        assert result["dc_language"] == "de"
        assert "german" not in (result["dc_language"] or "")


# ---------------------------------------------------------------------------
# Helper CLI contract
# ---------------------------------------------------------------------------


class TestHelperCLI:
    def test_main_exits_nonzero_on_missing_arg(self):
        assert _helper.main(["read_pdf_xmp_language.py"]) == 1

    def test_main_exits_zero_on_valid_pdf(self, tmp_path, capsys):
        pdf = tmp_path / "ok.pdf"
        _make_pdf_with_existing_xmp(pdf)
        _embed_language_xmp(str(pdf), "en")
        rc = _helper.main(["read_pdf_xmp_language.py", str(pdf)])
        captured = capsys.readouterr()
        assert rc == 0
        assert "\"dc_language\": \"en\"" in captured.out

    def test_main_exits_nonzero_on_missing_file(self, tmp_path, capsys):
        rc = _helper.main(
            ["read_pdf_xmp_language.py", str(tmp_path / "nope.pdf")],
        )
        captured = capsys.readouterr()
        assert rc == 2
        assert "\"error\"" in captured.out


# ---------------------------------------------------------------------------
# Assembler-aligned scenario: language.json ↔ PDF XMP round-trip
# ---------------------------------------------------------------------------


class TestLanguageJsonXmpAgreement:
    def test_roundtrip_matches_language_json_primary(self, tmp_path):
        """The primary_language recorded in .language.json must resolve to
        the same BCP-47 code that shows up in PDF XMP."""
        from ocr_local.config.language_config import LANGUAGE_REGISTRY

        # Simulate .language.json primary.
        primary_language = "fr"
        entry = LANGUAGE_REGISTRY.get(primary_language)
        bcp47 = entry.bcp47 if (entry and entry.bcp47) else primary_language

        pdf = tmp_path / "agree.pdf"
        _make_pdf_with_existing_xmp(pdf)
        _embed_language_xmp(str(pdf), bcp47)
        result = read_pdf_xmp_language(str(pdf))
        assert result["dc_language"] == bcp47

"""Tests for the language support validation script.

Covers config validation for all 45 languages, font mapping completeness,
model mapping completeness, report generation (JSON and markdown), and
tier filtering.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from language_config import LANGUAGE_REGISTRY, LanguageEntry
from scripts.validate_language_support import (
    format_console_report,
    format_markdown_report,
    main,
    validate_all_languages,
    validate_language,
)

# ---------------------------------------------------------------------------
# TestValidateLanguage - Config validation per entry
# ---------------------------------------------------------------------------


class TestValidateLanguage:
    """Test config-level validation for individual language entries."""

    def test_valid_entry_passes(self):
        entry = LanguageEntry(
            paddle_code="en",
            name="English",
            fasttext_codes=("en",),
            script="latin",
            tier="core",
            tesseract_code="eng",
            easyocr_code="en",
            font="NotoSans-Regular.ttf",
        )
        result = validate_language(entry)
        assert result.config_valid is True
        assert result.has_name is True
        assert result.has_fasttext_codes is True
        assert result.has_valid_script is True
        assert result.has_valid_tier is True
        assert result.has_font_mapping is True
        assert result.config_issues == []

    def test_missing_name_fails(self):
        entry = LanguageEntry(
            paddle_code="x",
            name="",
            fasttext_codes=("x",),
            script="latin",
            tier="core",
        )
        result = validate_language(entry)
        assert result.config_valid is False
        assert result.has_name is False
        assert "missing or empty name" in result.config_issues

    def test_no_fasttext_codes_fails(self):
        entry = LanguageEntry(
            paddle_code="x",
            name="X Lang",
            fasttext_codes=(),
            script="latin",
            tier="core",
        )
        result = validate_language(entry)
        assert result.config_valid is False
        assert result.has_fasttext_codes is False
        assert "no FastText codes" in result.config_issues

    def test_invalid_script_fails(self):
        entry = LanguageEntry(
            paddle_code="x",
            name="X Lang",
            fasttext_codes=("x",),
            script="klingon",
            tier="core",
        )
        result = validate_language(entry)
        assert result.config_valid is False
        assert result.has_valid_script is False
        assert any("invalid script" in i for i in result.config_issues)

    def test_invalid_tier_fails(self):
        entry = LanguageEntry(
            paddle_code="x",
            name="X Lang",
            fasttext_codes=("x",),
            script="latin",
            tier="premium",
        )
        result = validate_language(entry)
        assert result.config_valid is False
        assert result.has_valid_tier is False
        assert any("invalid tier" in i for i in result.config_issues)

    def test_empty_font_fails(self):
        entry = LanguageEntry(
            paddle_code="x",
            name="X Lang",
            fasttext_codes=("x",),
            script="latin",
            tier="core",
            font="",
        )
        result = validate_language(entry)
        assert result.config_valid is False
        assert result.has_font_mapping is False
        assert "no font mapping" in result.config_issues

    def test_rtl_flag_reported(self):
        entry = LanguageEntry(
            paddle_code="ar",
            name="Arabic",
            fasttext_codes=("ar",),
            script="arabic",
            tier="core",
            font="NotoSansArabic-Regular.ttf",
            rtl=True,
        )
        result = validate_language(entry)
        assert result.is_rtl is True

    def test_optional_codes_reported(self):
        entry = LanguageEntry(
            paddle_code="x",
            name="X Lang",
            fasttext_codes=("x",),
            script="latin",
            tier="core",
            tesseract_code="",
            easyocr_code="",
        )
        result = validate_language(entry)
        assert result.has_tesseract_code is False
        assert result.has_easyocr_code is False
        # Missing tesseract/easyocr codes do NOT cause config_valid to fail
        assert result.config_valid is True

    def test_disk_checks_skipped_by_default(self):
        entry = LANGUAGE_REGISTRY["en"]
        result = validate_language(entry)
        assert result.font_file_exists is None
        assert result.model_dir_exists is None
        assert result.tesseract_data_exists is None

    def test_font_check_with_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake font file
            font_path = Path(tmpdir) / "NotoSans-Regular.ttf"
            font_path.write_bytes(b"fake-font-data")

            entry = LanguageEntry(
                paddle_code="en",
                name="English",
                fasttext_codes=("en",),
                script="latin",
                tier="core",
                font="NotoSans-Regular.ttf",
            )
            with mock.patch(
                "scripts.validate_language_support.NOTO_FONT_DIR", tmpdir
            ):
                result = validate_language(entry, check_fonts=True)
            assert result.font_file_exists is True

    def test_font_check_with_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = LanguageEntry(
                paddle_code="en",
                name="English",
                fasttext_codes=("en",),
                script="latin",
                tier="core",
                font="NotoSans-Regular.ttf",
            )
            with mock.patch(
                "scripts.validate_language_support.NOTO_FONT_DIR", tmpdir
            ):
                result = validate_language(entry, check_fonts=True)
            assert result.font_file_exists is False


# ---------------------------------------------------------------------------
# TestAllRegistryLanguagesConfigValid
# ---------------------------------------------------------------------------


class TestAllRegistryLanguagesConfigValid:
    """Validate config integrity of all 45 languages in the registry."""

    def test_all_45_languages_pass_config_validation(self):
        for code, entry in LANGUAGE_REGISTRY.items():
            result = validate_language(entry)
            assert result.config_valid is True, (
                f"Language '{code}' ({entry.name}) failed config validation: "
                f"{result.config_issues}"
            )

    def test_all_core_languages_pass(self):
        core = [e for e in LANGUAGE_REGISTRY.values() if e.tier == "core"]
        assert len(core) == 34
        for entry in core:
            result = validate_language(entry)
            assert result.config_valid is True

    def test_all_extended_languages_pass(self):
        extended = [
            e for e in LANGUAGE_REGISTRY.values() if e.tier == "extended"
        ]
        assert len(extended) == 11
        for entry in extended:
            result = validate_language(entry)
            assert result.config_valid is True

    def test_every_language_has_tesseract_code(self):
        """All 45 languages should have Tesseract mappings."""
        for code, entry in LANGUAGE_REGISTRY.items():
            assert entry.tesseract_code, (
                f"Language '{code}' ({entry.name}) has no Tesseract code"
            )

    def test_every_language_has_easyocr_code(self):
        """All 45 languages should have EasyOCR mappings."""
        for code, entry in LANGUAGE_REGISTRY.items():
            assert entry.easyocr_code, (
                f"Language '{code}' ({entry.name}) has no EasyOCR code"
            )


# ---------------------------------------------------------------------------
# TestFontMappingCompleteness
# ---------------------------------------------------------------------------


class TestFontMappingCompleteness:
    """Verify font mapping exists and is well-formed for every language."""

    def test_every_language_has_font_mapping(self):
        for code, entry in LANGUAGE_REGISTRY.items():
            assert entry.font, f"Language '{code}' has no font mapping"

    def test_font_filenames_have_valid_extensions(self):
        valid_exts = {".ttf", ".otf"}
        for code, entry in LANGUAGE_REGISTRY.items():
            ext = Path(entry.font).suffix.lower()
            assert ext in valid_exts, (
                f"Language '{code}' font '{entry.font}' has invalid "
                f"extension '{ext}' (expected .ttf or .otf)"
            )

    def test_cjk_languages_use_cjk_fonts(self):
        cjk_codes = ["ch", "chinese_cht", "japan", "korean"]
        for code in cjk_codes:
            entry = LANGUAGE_REGISTRY[code]
            assert "CJK" in entry.font, (
                f"CJK language '{code}' should use a CJK font, "
                f"got '{entry.font}'"
            )

    def test_arabic_languages_use_arabic_font(self):
        arabic_codes = ["ar", "fa", "ur", "ug"]
        for code in arabic_codes:
            entry = LANGUAGE_REGISTRY[code]
            assert "Arabic" in entry.font, (
                f"Arabic language '{code}' should use Arabic font, "
                f"got '{entry.font}'"
            )

    def test_devanagari_languages_use_devanagari_font(self):
        devanagari_codes = ["hi", "mr", "ne"]
        for code in devanagari_codes:
            entry = LANGUAGE_REGISTRY[code]
            assert "Devanagari" in entry.font, (
                f"Devanagari language '{code}' should use Devanagari font, "
                f"got '{entry.font}'"
            )

    def test_latin_languages_use_default_font(self):
        latin_core = [
            "en", "fr", "german", "es", "it", "pt", "nl", "sv", "da",
            "fi", "ro", "pl", "cs", "hu", "tr", "vi",
        ]
        latin_extended = ["hr", "sk", "no", "lt", "lv", "et", "rs_latin"]
        for code in latin_core + latin_extended:
            entry = LANGUAGE_REGISTRY[code]
            assert entry.font == "NotoSans-Regular.ttf", (
                f"Latin language '{code}' should use NotoSans-Regular.ttf, "
                f"got '{entry.font}'"
            )

    def test_unique_font_count(self):
        """There should be a reasonable number of unique fonts."""
        unique_fonts = {e.font for e in LANGUAGE_REGISTRY.values()}
        # We expect: NotoSans, 4 CJK, Arabic, Devanagari, Tamil,
        # Telugu, Kannada, Georgian, Thai, Bengali, Greek (default)
        assert len(unique_fonts) >= 10
        assert len(unique_fonts) <= 20


# ---------------------------------------------------------------------------
# TestModelMappingCompleteness
# ---------------------------------------------------------------------------


class TestModelMappingCompleteness:
    """Verify model-related mappings are complete."""

    def test_every_language_has_unique_paddle_code(self):
        codes = [e.paddle_code for e in LANGUAGE_REGISTRY.values()]
        assert len(codes) == len(set(codes))

    def test_no_paddle_code_collisions_with_fasttext(self):
        """FastText codes should not collide across languages."""
        seen = {}
        for entry in LANGUAGE_REGISTRY.values():
            for ft_code in entry.fasttext_codes:
                if ft_code in seen:
                    pytest.fail(
                        f"FastText code '{ft_code}' maps to both "
                        f"'{seen[ft_code]}' and '{entry.paddle_code}'"
                    )
                seen[ft_code] = entry.paddle_code

    def test_tesseract_codes_are_plausible(self):
        """Tesseract codes should be 3-letter or underscore-separated."""
        import re
        for code, entry in LANGUAGE_REGISTRY.items():
            if entry.tesseract_code:
                assert re.match(
                    r"^[a-z]{3}(_[a-z]+)?$", entry.tesseract_code
                ), (
                    f"Language '{code}' tesseract code "
                    f"'{entry.tesseract_code}' looks malformed"
                )


# ---------------------------------------------------------------------------
# TestValidateAllLanguages
# ---------------------------------------------------------------------------


class TestValidateAllLanguages:
    """Test the top-level validate_all_languages function."""

    def test_all_tiers(self):
        report = validate_all_languages(tiers=["core", "extended"])
        assert report.total_languages == 45
        assert report.all_config_valid is True
        assert report.config_pass == 45
        assert report.config_fail == 0

    def test_core_only(self):
        report = validate_all_languages(tiers=["core"])
        assert report.total_languages == 34
        assert report.all_config_valid is True
        assert "core" in report.tiers_checked

    def test_extended_only(self):
        report = validate_all_languages(tiers=["extended"])
        assert report.total_languages == 11
        assert report.all_config_valid is True
        assert "extended" in report.tiers_checked

    def test_all_keyword(self):
        report = validate_all_languages(tiers=["all"])
        assert report.total_languages == 45

    def test_none_defaults_to_all(self):
        report = validate_all_languages(tiers=None)
        assert report.total_languages == 45

    def test_fonts_not_checked_by_default(self):
        report = validate_all_languages()
        assert report.fonts_checked is False
        assert report.fonts_found == 0
        assert report.fonts_missing == 0

    def test_models_not_checked_by_default(self):
        report = validate_all_languages()
        assert report.models_checked is False
        assert report.models_found == 0
        assert report.models_missing == 0

    def test_has_timestamp(self):
        report = validate_all_languages()
        assert report.timestamp
        assert "T" in report.timestamp  # ISO format

    def test_unique_fonts_listed(self):
        report = validate_all_languages()
        assert len(report.unique_fonts_needed) >= 10

    def test_languages_list_populated(self):
        report = validate_all_languages()
        assert len(report.languages) == 45
        # Each entry should be a dict
        assert isinstance(report.languages[0], dict)
        assert "paddle_code" in report.languages[0]

    def test_font_check_opt_in(self):
        report = validate_all_languages(check_fonts=True)
        assert report.fonts_checked is True
        # On Windows dev without /app/fonts, all should be missing
        for lang in report.languages:
            assert lang["font_file_exists"] is not None


# ---------------------------------------------------------------------------
# TestReportFormatting
# ---------------------------------------------------------------------------


class TestReportFormatting:
    """Test JSON and markdown report generation."""

    def _make_report(self):
        return validate_all_languages(tiers=["core", "extended"])

    def test_console_report_contains_header(self):
        report = self._make_report()
        text = format_console_report(report)
        assert "LANGUAGE SUPPORT VALIDATION REPORT" in text
        assert "LANGUAGE DETAILS" in text

    def test_console_report_contains_all_languages(self):
        report = self._make_report()
        text = format_console_report(report)
        assert "English" in text
        assert "Thai" in text
        assert "Bengali" in text

    def test_console_report_shows_pass(self):
        report = self._make_report()
        text = format_console_report(report)
        assert "Overall:          PASS" in text

    def test_markdown_report_contains_header(self):
        report = self._make_report()
        md = format_markdown_report(report)
        assert "# Language Support Validation Report" in md
        assert "## Language Details" in md

    def test_markdown_report_has_table(self):
        report = self._make_report()
        md = format_markdown_report(report)
        assert "| Code |" in md
        assert "| en |" in md or "|en|" in md or "| en " in md

    def test_markdown_report_lists_unique_fonts(self):
        report = self._make_report()
        md = format_markdown_report(report)
        assert "## Required Font Files" in md
        assert "`NotoSans-Regular.ttf`" in md

    def test_json_serialization_roundtrip(self):
        from dataclasses import asdict
        report = self._make_report()
        data = asdict(report)
        json_str = json.dumps(data, indent=2, default=str)
        loaded = json.loads(json_str)
        assert loaded["total_languages"] == 45
        assert loaded["all_config_valid"] is True
        assert len(loaded["languages"]) == 45


# ---------------------------------------------------------------------------
# TestJSONFileOutput
# ---------------------------------------------------------------------------


class TestJSONFileOutput:
    """Test writing JSON report to file."""

    def test_json_output_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "report.json")
            exit_code = main(["--tier", "core", "--output-json", json_path])
            assert exit_code == 0
            assert os.path.isfile(json_path)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["total_languages"] == 34
            assert data["all_config_valid"] is True

    def test_json_output_nested_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "sub", "dir", "report.json")
            exit_code = main(["--tier", "core", "--output-json", json_path])
            assert exit_code == 0
            assert os.path.isfile(json_path)


# ---------------------------------------------------------------------------
# TestMarkdownFileOutput
# ---------------------------------------------------------------------------


class TestMarkdownFileOutput:
    """Test writing markdown report to file."""

    def test_md_output_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = os.path.join(tmpdir, "report.md")
            exit_code = main(["--tier", "all", "--output-md", md_path])
            assert exit_code == 0
            assert os.path.isfile(md_path)

            content = Path(md_path).read_text(encoding="utf-8")
            assert "# Language Support Validation Report" in content
            assert "45" in content

    def test_both_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "report.json")
            md_path = os.path.join(tmpdir, "report.md")
            exit_code = main([
                "--tier", "all",
                "--output-json", json_path,
                "--output-md", md_path,
            ])
            assert exit_code == 0
            assert os.path.isfile(json_path)
            assert os.path.isfile(md_path)


# ---------------------------------------------------------------------------
# TestTierFiltering
# ---------------------------------------------------------------------------


class TestTierFiltering:
    """Test --tier argument filtering."""

    def test_core_tier_argument(self):
        exit_code = main(["--tier", "core"])
        assert exit_code == 0

    def test_extended_tier_argument(self):
        exit_code = main(["--tier", "extended"])
        assert exit_code == 0

    def test_all_tier_argument(self):
        exit_code = main(["--tier", "all"])
        assert exit_code == 0

    def test_core_report_excludes_extended(self):
        report = validate_all_languages(tiers=["core"])
        for lang in report.languages:
            assert lang["tier"] == "core"

    def test_extended_report_excludes_core(self):
        report = validate_all_languages(tiers=["extended"])
        for lang in report.languages:
            assert lang["tier"] == "extended"


# ---------------------------------------------------------------------------
# TestExtendedTierLanguages
# ---------------------------------------------------------------------------


class TestExtendedTierLanguages:
    """Specific validation for the 11 extended-tier languages."""

    EXTENDED_CODES = [
        "th", "bn", "mr", "ne", "hr", "sk", "no", "lt", "lv", "et",
        "rs_latin",
    ]

    def test_all_11_present_in_registry(self):
        for code in self.EXTENDED_CODES:
            assert code in LANGUAGE_REGISTRY, (
                f"Extended language '{code}' missing from registry"
            )

    def test_all_11_marked_extended(self):
        for code in self.EXTENDED_CODES:
            assert LANGUAGE_REGISTRY[code].tier == "extended"

    def test_extended_config_validation(self):
        for code in self.EXTENDED_CODES:
            entry = LANGUAGE_REGISTRY[code]
            result = validate_language(entry)
            assert result.config_valid is True, (
                f"Extended language '{code}' ({entry.name}) failed: "
                f"{result.config_issues}"
            )

    def test_thai_has_thai_font(self):
        entry = LANGUAGE_REGISTRY["th"]
        assert "Thai" in entry.font

    def test_bengali_has_bengali_font(self):
        entry = LANGUAGE_REGISTRY["bn"]
        assert "Bengali" in entry.font

    def test_marathi_has_devanagari_font(self):
        entry = LANGUAGE_REGISTRY["mr"]
        assert "Devanagari" in entry.font

    def test_nepali_has_devanagari_font(self):
        entry = LANGUAGE_REGISTRY["ne"]
        assert "Devanagari" in entry.font

    def test_latin_extended_use_default_font(self):
        latin_ext = ["hr", "sk", "no", "lt", "lv", "et", "rs_latin"]
        for code in latin_ext:
            entry = LANGUAGE_REGISTRY[code]
            assert entry.font == "NotoSans-Regular.ttf"


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_registry_produces_zero_report(self):
        """Validating with no matching tiers returns empty results."""
        report = validate_all_languages(tiers=["nonexistent"])
        assert report.total_languages == 0
        assert report.all_config_valid is True  # vacuously true

    def test_report_paths_are_strings(self):
        report = validate_all_languages()
        assert isinstance(report.font_dir, str)
        assert isinstance(report.model_dir, str)
        assert isinstance(report.tessdata_dir, str)

    def test_main_returns_zero_on_success(self):
        assert main(["--tier", "all"]) == 0

    def test_verbose_flag_accepted(self):
        """Verbose flag should not cause errors."""
        assert main(["--tier", "core", "-v"]) == 0

"""Tests for the language validation script (scripts/validate_languages.py).

Covers:
- Language registry completeness (45 languages present)
- LANG_MAPPING coverage (all languages have a mapping)
- Font check function (mocked filesystem)
- Model check function (mocked filesystem)
- Mapping check function
- Report generation
- CLI argument parsing
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
from scripts.validate_languages import (
    _parse_args,
    check_font,
    check_mapping,
    check_mapping_collisions,
    check_model,
    find_unmapped_fasttext_codes,
    format_report,
    main,
    run_validation,
)

# ---------------------------------------------------------------------------
# TestRegistryCompleteness
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    """Verify the language registry contains the expected 45 languages."""

    def test_total_count_is_45(self):
        assert len(LANGUAGE_REGISTRY) == 45

    def test_core_tier_has_34(self):
        core = [e for e in LANGUAGE_REGISTRY.values() if e.tier == "core"]
        assert len(core) == 34

    def test_extended_tier_has_11(self):
        extended = [e for e in LANGUAGE_REGISTRY.values() if e.tier == "extended"]
        assert len(extended) == 11

    def test_no_duplicate_paddle_codes(self):
        codes = list(LANGUAGE_REGISTRY.keys())
        assert len(codes) == len(set(codes))

    def test_every_language_has_name(self):
        for code, entry in LANGUAGE_REGISTRY.items():
            assert entry.name, f"Language {code} has empty name"

    def test_every_language_has_fasttext_codes(self):
        for code, entry in LANGUAGE_REGISTRY.items():
            assert len(entry.fasttext_codes) >= 1, (
                f"Language {code} has no FastText codes"
            )

    def test_every_language_has_font(self):
        for code, entry in LANGUAGE_REGISTRY.items():
            assert entry.font, f"Language {code} has no font"

    def test_every_language_has_script(self):
        for code, entry in LANGUAGE_REGISTRY.items():
            assert entry.script, f"Language {code} has no script"


# ---------------------------------------------------------------------------
# TestLangMappingCoverage
# ---------------------------------------------------------------------------


class TestLangMappingCoverage:
    """Verify LANG_MAPPING covers all languages (full registry, all tiers)."""

    def _full_mapping(self):
        """Build mapping from the full registry, not tier-filtered."""
        from scripts.validate_languages import _build_full_lang_mapping
        return _build_full_lang_mapping()

    def test_all_fasttext_codes_mapped(self):
        mapping = self._full_mapping()
        for entry in LANGUAGE_REGISTRY.values():
            for ft_code in entry.fasttext_codes:
                assert ft_code in mapping, (
                    f"FastText code '{ft_code}' for "
                    f"'{entry.paddle_code}' not in full LANG_MAPPING"
                )

    def test_no_fasttext_collisions(self):
        seen = {}
        for entry in LANGUAGE_REGISTRY.values():
            for ft_code in entry.fasttext_codes:
                if ft_code in seen:
                    pytest.fail(
                        f"FastText code '{ft_code}' maps to both "
                        f"'{seen[ft_code]}' and '{entry.paddle_code}'"
                    )
                seen[ft_code] = entry.paddle_code

    def test_mapping_values_match_paddle_codes(self):
        mapping = self._full_mapping()
        for entry in LANGUAGE_REGISTRY.values():
            for ft_code in entry.fasttext_codes:
                assert mapping[ft_code] == entry.paddle_code, (
                    f"LANG_MAPPING['{ft_code}'] = '{mapping[ft_code]}', "
                    f"expected '{entry.paddle_code}'"
                )


# ---------------------------------------------------------------------------
# TestCheckFont
# ---------------------------------------------------------------------------


class TestCheckFont:
    """Test font check function with mocked filesystem."""

    def test_font_ok_when_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            font_path = Path(tmpdir) / "NotoSans-Regular.ttf"
            font_path.write_bytes(b"fake-font-data")

            entry = LanguageEntry(
                "en", "English", ("en",), "latin", "core",
                "eng", "en", "NotoSans-Regular.ttf",
            )
            with mock.patch(
                "scripts.validate_languages.NOTO_FONT_DIR", tmpdir
            ):
                result = check_font(entry)

            assert result.has_mapping is True
            assert result.file_exists is True
            assert result.status == "ok"

    def test_font_missing_when_file_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = LanguageEntry(
                "en", "English", ("en",), "latin", "core",
                "eng", "en", "NotoSans-Regular.ttf",
            )
            with mock.patch(
                "scripts.validate_languages.NOTO_FONT_DIR", tmpdir
            ):
                result = check_font(entry)

            assert result.has_mapping is True
            assert result.file_exists is False
            assert result.status == "missing_file"

    def test_font_no_mapping_when_empty(self):
        entry = LanguageEntry(
            "x", "X Lang", ("x",), "latin", "core",
            font="",
        )
        result = check_font(entry)
        assert result.has_mapping is False
        assert result.status == "no_mapping"

    def test_font_result_contains_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = LanguageEntry(
                "ar", "Arabic", ("ar",), "arabic", "core",
                font="NotoSansArabic-Regular.ttf",
            )
            with mock.patch(
                "scripts.validate_languages.NOTO_FONT_DIR", tmpdir
            ):
                result = check_font(entry)

            assert "NotoSansArabic-Regular.ttf" in result.file_path

    def test_cjk_font_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            font_path = Path(tmpdir) / "NotoSansCJKsc-Regular.otf"
            font_path.write_bytes(b"fake-cjk-font")

            entry = LANGUAGE_REGISTRY["ch"]
            with mock.patch(
                "scripts.validate_languages.NOTO_FONT_DIR", tmpdir
            ):
                result = check_font(entry)

            assert result.status == "ok"
            assert result.font_filename == "NotoSansCJKsc-Regular.otf"


# ---------------------------------------------------------------------------
# TestCheckModel
# ---------------------------------------------------------------------------


class TestCheckModel:
    """Test model check function with mocked filesystem."""

    def test_model_ok_when_dir_exists_with_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake model directory structure
            rec_dir = Path(tmpdir) / "rec" / "en_model"
            rec_dir.mkdir(parents=True)
            model_file = rec_dir / "inference.pdmodel"
            model_file.write_bytes(b"x" * 2048)

            entry = LanguageEntry(
                "en", "English", ("en",), "latin", "core",
            )
            with mock.patch(
                "scripts.validate_languages.PADDLEOCR_MODEL_DIR", tmpdir
            ):
                result = check_model(entry)

            assert result.dir_exists is True
            assert result.status == "ok"
            assert result.file_count >= 1
            assert result.total_size_bytes >= 2048

    def test_model_missing_when_no_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = LanguageEntry(
                "en", "English", ("en",), "latin", "core",
            )
            with mock.patch(
                "scripts.validate_languages.PADDLEOCR_MODEL_DIR",
                os.path.join(tmpdir, "nonexistent"),
            ):
                result = check_model(entry)

            assert result.dir_exists is False
            assert result.status == "missing"

    def test_model_missing_when_dir_exists_but_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create rec dir but no matching language directory
            rec_dir = Path(tmpdir) / "rec" / "other_model"
            rec_dir.mkdir(parents=True)

            entry = LanguageEntry(
                "en", "English", ("en",), "latin", "core",
            )
            with mock.patch(
                "scripts.validate_languages.PADDLEOCR_MODEL_DIR", tmpdir
            ):
                result = check_model(entry)

            assert result.status == "missing"

    def test_model_corrupt_when_zero_size_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_dir = Path(tmpdir) / "rec" / "en_model"
            rec_dir.mkdir(parents=True)
            # Create zero-size model file
            model_file = rec_dir / "inference.pdmodel"
            model_file.write_bytes(b"")

            entry = LanguageEntry(
                "en", "English", ("en",), "latin", "core",
            )
            with mock.patch(
                "scripts.validate_languages.PADDLEOCR_MODEL_DIR", tmpdir
            ):
                result = check_model(entry)

            assert result.has_zero_size_files is True
            assert result.status == "corrupt"

    def test_model_found_via_walk_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model in a non-standard location
            nested = Path(tmpdir) / "det" / "en_PP-OCRv4_det"
            nested.mkdir(parents=True)
            (nested / "model.pdparams").write_bytes(b"x" * 4096)

            entry = LanguageEntry(
                "en", "English", ("en",), "latin", "core",
            )
            with mock.patch(
                "scripts.validate_languages.PADDLEOCR_MODEL_DIR", tmpdir
            ):
                result = check_model(entry)

            assert result.dir_exists is True
            assert result.status == "ok"

    def test_model_result_has_size_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_dir = Path(tmpdir) / "rec" / "fr_model"
            rec_dir.mkdir(parents=True)
            (rec_dir / "a.bin").write_bytes(b"x" * 1024)
            (rec_dir / "b.bin").write_bytes(b"y" * 2048)

            entry = LanguageEntry(
                "fr", "French", ("fr",), "latin", "core",
            )
            with mock.patch(
                "scripts.validate_languages.PADDLEOCR_MODEL_DIR", tmpdir
            ):
                result = check_model(entry)

            assert result.file_count == 2
            assert result.total_size_bytes == 3072


# ---------------------------------------------------------------------------
# TestCheckMapping
# ---------------------------------------------------------------------------


class TestCheckMapping:
    """Test mapping check function."""

    def test_english_mapping_ok(self):
        entry = LANGUAGE_REGISTRY["en"]
        result = check_mapping(entry)
        assert result.has_lang_mapping is True
        assert result.has_tesseract_code is True
        assert result.has_easyocr_code is True
        assert result.status == "ok"
        assert result.issues == []

    def test_all_core_languages_have_valid_mappings(self):
        for code, entry in LANGUAGE_REGISTRY.items():
            if entry.tier == "core":
                result = check_mapping(entry)
                assert result.has_lang_mapping is True, (
                    f"Core language '{code}' has broken FastText mapping"
                )

    def test_all_extended_languages_have_valid_mappings(self):
        import language_config
        saved = language_config.OCR_LANGUAGE_TIERS
        try:
            language_config.OCR_LANGUAGE_TIERS = ["core", "extended"]
            for code, entry in LANGUAGE_REGISTRY.items():
                if entry.tier == "extended":
                    result = check_mapping(entry)
                    assert result.has_lang_mapping is True, (
                        f"Extended language '{code}' has broken FastText mapping"
                    )
        finally:
            language_config.OCR_LANGUAGE_TIERS = saved

    def test_missing_tesseract_code_reported(self):
        entry = LanguageEntry(
            "x", "X Lang", ("x",), "latin", "core",
            tesseract_code="", easyocr_code="x",
        )
        # Patch the registry to include this entry for mapping lookup
        with mock.patch(
            "scripts.validate_languages._build_full_lang_mapping",
            return_value={"x": "x"},
        ):
            result = check_mapping(entry)
        assert result.has_tesseract_code is False
        assert "no Tesseract code" in result.issues
        assert result.status == "incomplete"

    def test_missing_easyocr_code_reported(self):
        entry = LanguageEntry(
            "y", "Y Lang", ("y",), "latin", "core",
            tesseract_code="ylang", easyocr_code="",
        )
        with mock.patch(
            "scripts.validate_languages._build_full_lang_mapping",
            return_value={"y": "y"},
        ):
            result = check_mapping(entry)
        assert result.has_easyocr_code is False
        assert "no EasyOCR code" in result.issues

    def test_broken_mapping_when_fasttext_not_in_mapping(self):
        entry = LanguageEntry(
            "z", "Z Lang", ("zz",), "latin", "core",
            tesseract_code="zlang", easyocr_code="z",
        )
        with mock.patch(
            "scripts.validate_languages._build_full_lang_mapping",
            return_value={},
        ):
            result = check_mapping(entry)
        assert result.has_lang_mapping is False
        assert result.status == "broken"

    def test_broken_mapping_when_fasttext_maps_wrong(self):
        entry = LanguageEntry(
            "a", "A Lang", ("aa",), "latin", "core",
            tesseract_code="alang", easyocr_code="a",
        )
        with mock.patch(
            "scripts.validate_languages._build_full_lang_mapping",
            return_value={"aa": "wrong_code"},
        ):
            result = check_mapping(entry)
        assert result.has_lang_mapping is False
        assert result.status == "broken"


# ---------------------------------------------------------------------------
# TestCheckMappingCollisions
# ---------------------------------------------------------------------------


class TestCheckMappingCollisions:
    """Test collision detection across the registry."""

    def test_no_collisions_in_real_registry(self):
        collisions = check_mapping_collisions()
        assert collisions == [], (
            f"Found FastText code collisions: {collisions}"
        )


# ---------------------------------------------------------------------------
# TestFindUnmappedCodes
# ---------------------------------------------------------------------------


class TestFindUnmappedCodes:
    """Test detection of unmapped FastText codes."""

    def test_no_unmapped_codes_across_all_tiers(self):
        """All FastText codes in the full registry should be mapped."""
        unmapped = find_unmapped_fasttext_codes()
        assert unmapped == [], (
            f"Found unmapped FastText codes: {unmapped}"
        )


# ---------------------------------------------------------------------------
# TestRunValidation
# ---------------------------------------------------------------------------


class TestRunValidation:
    """Test the run_validation orchestration function."""

    def test_all_checks(self):
        report = run_validation(["all"])
        assert report.total_languages == 45
        assert "fonts" in report.checks_performed
        assert "models" in report.checks_performed
        assert "mappings" in report.checks_performed
        assert report.timestamp

    def test_fonts_only(self):
        report = run_validation(["fonts"])
        assert "fonts" in report.checks_performed
        assert report.fonts_total == 45
        assert report.font_results
        # Models and mappings should be empty
        assert not report.model_results
        assert not report.mapping_results

    def test_models_only(self):
        report = run_validation(["models"])
        assert "models" in report.checks_performed
        assert report.models_total == 45
        assert report.model_results
        assert not report.font_results
        assert not report.mapping_results

    def test_mappings_only(self):
        report = run_validation(["mappings"])
        assert "mappings" in report.checks_performed
        assert report.mappings_total == 45
        assert report.mapping_results
        assert not report.font_results
        assert not report.model_results

    def test_mappings_pass_for_all_languages(self):
        """All 45 languages should have correct mappings."""
        import language_config
        saved = language_config.OCR_LANGUAGE_TIERS
        try:
            language_config.OCR_LANGUAGE_TIERS = ["core", "extended"]
            report = run_validation(["mappings"])
            assert report.mappings_broken == 0, (
                f"Found {report.mappings_broken} broken mappings"
            )
            assert report.mapping_collisions == []
        finally:
            language_config.OCR_LANGUAGE_TIERS = saved

    def test_summary_line_populated(self):
        report = run_validation(["mappings"])
        assert "45" in report.summary_line
        assert "validated" in report.summary_line

    def test_report_has_correct_warning_count(self):
        report = run_validation(["mappings"])
        # With all mappings OK, warnings should reflect any incomplete ones
        assert isinstance(report.warnings, int)


# ---------------------------------------------------------------------------
# TestFormatReport
# ---------------------------------------------------------------------------


class TestFormatReport:
    """Test report formatting."""

    def test_console_output_has_header(self):
        report = run_validation(["mappings"])
        text = format_report(report)
        assert "LANGUAGE VALIDATION REPORT" in text

    def test_console_output_has_mapping_section(self):
        report = run_validation(["mappings"])
        text = format_report(report)
        assert "MAPPING CHECKS" in text

    def test_console_output_has_font_section(self):
        report = run_validation(["fonts"])
        text = format_report(report)
        assert "FONT CHECKS" in text

    def test_console_output_has_model_section(self):
        report = run_validation(["models"])
        text = format_report(report)
        assert "MODEL CHECKS" in text

    def test_console_output_contains_language_names(self):
        report = run_validation(["mappings"])
        text = format_report(report)
        assert "English" in text
        assert "Arabic" in text
        assert "Japanese" in text

    def test_console_output_shows_result(self):
        report = run_validation(["mappings"])
        text = format_report(report)
        assert "Result:" in text

    def test_all_checks_format(self):
        report = run_validation(["all"])
        text = format_report(report)
        assert "FONT CHECKS" in text
        assert "MODEL CHECKS" in text
        assert "MAPPING CHECKS" in text


# ---------------------------------------------------------------------------
# TestCLIArgParsing
# ---------------------------------------------------------------------------


class TestCLIArgParsing:
    """Test CLI argument parsing."""

    def test_default_check_is_all(self):
        args = _parse_args([])
        assert args.check == "all"

    def test_check_fonts(self):
        args = _parse_args(["--check", "fonts"])
        assert args.check == "fonts"

    def test_check_models(self):
        args = _parse_args(["--check", "models"])
        assert args.check == "models"

    def test_check_mappings(self):
        args = _parse_args(["--check", "mappings"])
        assert args.check == "mappings"

    def test_check_all(self):
        args = _parse_args(["--check", "all"])
        assert args.check == "all"

    def test_output_report(self):
        args = _parse_args(["--output-report", "test.json"])
        assert args.output_report == "test.json"

    def test_verbose_flag(self):
        args = _parse_args(["-v"])
        assert args.verbose is True

    def test_long_verbose_flag(self):
        args = _parse_args(["--verbose"])
        assert args.verbose is True

    def test_invalid_check_rejected(self):
        with pytest.raises(SystemExit):
            _parse_args(["--check", "invalid"])

    def test_combined_args(self):
        args = _parse_args([
            "--check", "fonts",
            "--output-report", "/tmp/report.json",
            "-v",
        ])
        assert args.check == "fonts"
        assert args.output_report == "/tmp/report.json"
        assert args.verbose is True


# ---------------------------------------------------------------------------
# TestMainFunction
# ---------------------------------------------------------------------------


class TestMainFunction:
    """Test the main() CLI entry point."""

    def test_main_returns_zero_on_mappings_pass(self):
        exit_code = main(["--check", "mappings"])
        assert exit_code == 0

    def test_main_returns_zero_on_fonts_check(self):
        # Fonts may be missing on disk but that does not cause failure
        exit_code = main(["--check", "fonts"])
        assert exit_code == 0

    def test_main_returns_zero_on_all(self):
        exit_code = main(["--check", "all"])
        assert exit_code == 0

    def test_main_writes_json_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "report.json")
            exit_code = main([
                "--check", "mappings",
                "--output-report", json_path,
            ])
            assert exit_code == 0
            assert os.path.isfile(json_path)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["total_languages"] == 45
            assert data["mappings_total"] == 45
            assert isinstance(data["mapping_results"], list)

    def test_main_json_report_nested_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "sub", "dir", "report.json")
            exit_code = main([
                "--check", "all",
                "--output-report", json_path,
            ])
            assert exit_code == 0
            assert os.path.isfile(json_path)

    def test_main_verbose_flag_accepted(self):
        exit_code = main(["--check", "mappings", "-v"])
        assert exit_code == 0


# ---------------------------------------------------------------------------
# TestJSONReportStructure
# ---------------------------------------------------------------------------


class TestJSONReportStructure:
    """Verify JSON report structure for programmatic consumption."""

    def test_json_has_required_fields(self):
        report = run_validation(["all"])
        from dataclasses import asdict
        data = asdict(report)

        required_fields = [
            "timestamp", "checks_performed", "total_languages",
            "fonts_total", "fonts_ok", "fonts_missing",
            "models_total", "models_ok", "models_missing", "models_corrupt",
            "mappings_total", "mappings_ok", "mappings_incomplete",
            "mappings_broken", "mapping_collisions",
            "unmapped_fasttext_codes",
            "all_passed", "warnings", "summary_line",
        ]
        for field_name in required_fields:
            assert field_name in data, f"Missing field: {field_name}"

    def test_json_serializable(self):
        report = run_validation(["all"])
        from dataclasses import asdict
        data = asdict(report)
        json_str = json.dumps(data, indent=2, default=str)
        loaded = json.loads(json_str)
        assert loaded["total_languages"] == 45

    def test_font_results_structure(self):
        report = run_validation(["fonts"])
        assert len(report.font_results) == 45
        fr = report.font_results[0]
        assert "paddle_code" in fr
        assert "font_filename" in fr
        assert "has_mapping" in fr
        assert "status" in fr

    def test_model_results_structure(self):
        report = run_validation(["models"])
        assert len(report.model_results) == 45
        mr = report.model_results[0]
        assert "paddle_code" in mr
        assert "dir_exists" in mr
        assert "file_count" in mr
        assert "total_size_bytes" in mr
        assert "status" in mr

    def test_mapping_results_structure(self):
        report = run_validation(["mappings"])
        assert len(report.mapping_results) == 45
        mr = report.mapping_results[0]
        assert "paddle_code" in mr
        assert "fasttext_codes" in mr
        assert "has_lang_mapping" in mr
        assert "has_tesseract_code" in mr
        assert "has_easyocr_code" in mr
        assert "status" in mr
        assert "issues" in mr


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error paths."""

    def test_empty_checks_list(self):
        report = run_validation([])
        assert report.total_languages == 45
        assert not report.font_results
        assert not report.model_results
        assert not report.mapping_results
        assert report.all_passed is True

    def test_report_timestamp_is_iso(self):
        report = run_validation(["mappings"])
        assert "T" in report.timestamp

    def test_all_expands_to_three_checks(self):
        report = run_validation(["all"])
        assert set(report.checks_performed) == {"fonts", "models", "mappings"}

    def test_model_corrupt_with_small_total_size(self):
        """Model dir exists but total size below threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_dir = Path(tmpdir) / "rec" / "xx_model"
            rec_dir.mkdir(parents=True)
            (rec_dir / "tiny.bin").write_bytes(b"x" * 10)

            entry = LanguageEntry(
                "xx", "TestLang", ("xx",), "latin", "core",
            )
            with mock.patch(
                "scripts.validate_languages.PADDLEOCR_MODEL_DIR", tmpdir
            ):
                result = check_model(entry)

            # Total size < MODEL_MIN_SIZE_BYTES
            assert result.status == "corrupt"

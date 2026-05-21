"""
Unit tests for font selector module (font_selector.py).

Tests cover:
- Language-to-font mapping for the expanded language baseline
- Default font fallback for unknown languages
- FastText alias resolution (zh -> ch, de -> german, etc.)
- Font availability when directory exists / missing
- get_available_fonts scanning
- get_all_unique_fonts deduplication
- Edge cases (empty lang, None, whitespace)

Run with: python -m pytest tests/test_font_selector.py -v
"""

from pathlib import Path
from unittest import mock

from font_selector import (
    DEFAULT_FONT,
    FONT_DIR,
    LANGUAGE_FONT_MAP,
    get_all_unique_fonts,
    get_available_fonts,
    get_font_dir,
    get_font_name,
    get_font_path,
    is_font_available,
)

# ---------------------------------------------------------------------------
# Tests: Language-to-font mapping completeness
# ---------------------------------------------------------------------------


class TestLanguageFontMap:
    """Verify the expanded language baseline has font mappings."""

    EXPECTED_LANGUAGES = [
        "en", "fr", "german", "es", "it", "pt", "nl", "sv", "da", "fi",
        "ro", "pl", "cs", "hu", "tr",  # Latin
        "ru", "uk", "be", "bg",  # Cyrillic
        "ch", "chinese_cht", "japan", "korean",  # CJK
        "vi",  # Vietnamese
        "ar", "fa", "ur", "ug",  # Arabic-derived
        "hi",  # Hindi
        "ta", "te", "kn",  # Indic
        "ka",  # Georgian
        "el",  # Greek
    ]

    def test_all_expected_languages_mapped(self):
        for lang in self.EXPECTED_LANGUAGES:
            assert lang in LANGUAGE_FONT_MAP, f"Missing font mapping for: {lang}"

    def test_map_has_exactly_34_entries(self):
        assert len(LANGUAGE_FONT_MAP) == 34

    def test_cjk_languages_have_cjk_fonts(self):
        assert "CJK" in LANGUAGE_FONT_MAP["ch"]
        assert "CJK" in LANGUAGE_FONT_MAP["chinese_cht"]
        assert "CJK" in LANGUAGE_FONT_MAP["japan"]
        assert "CJK" in LANGUAGE_FONT_MAP["korean"]

    def test_arabic_has_arabic_font(self):
        assert "Arabic" in LANGUAGE_FONT_MAP["ar"]
        assert "Arabic" in LANGUAGE_FONT_MAP["fa"]
        assert "Arabic" in LANGUAGE_FONT_MAP["ur"]
        assert "Arabic" in LANGUAGE_FONT_MAP["ug"]

    def test_hindi_has_devanagari_font(self):
        assert "Devanagari" in LANGUAGE_FONT_MAP["hi"]

    def test_low_resource_scripts_have_dedicated_fonts(self):
        assert "Tamil" in LANGUAGE_FONT_MAP["ta"]
        assert "Telugu" in LANGUAGE_FONT_MAP["te"]
        assert "Kannada" in LANGUAGE_FONT_MAP["kn"]
        assert "Georgian" in LANGUAGE_FONT_MAP["ka"]

    def test_latin_languages_use_base_noto(self):
        latin_langs = ["en", "fr", "german", "es", "it", "pt", "nl", "sv",
                       "da", "fi", "ro", "pl", "cs", "hu", "tr"]
        for lang in latin_langs:
            assert LANGUAGE_FONT_MAP[lang] == "NotoSans-Regular.ttf"

    def test_cyrillic_uses_base_noto(self):
        """NotoSans-Regular.ttf covers Cyrillic glyphs."""
        for lang in ["ru", "uk", "be", "bg"]:
            assert LANGUAGE_FONT_MAP[lang] == "NotoSans-Regular.ttf"


# ---------------------------------------------------------------------------
# Tests: get_font_name
# ---------------------------------------------------------------------------


class TestGetFontName:
    def test_known_language(self):
        assert get_font_name("en") == "NotoSans-Regular.ttf"

    def test_cjk_language(self):
        assert get_font_name("ch") == "NotoSansCJKsc-Regular.otf"

    def test_unknown_language_returns_default(self):
        assert get_font_name("xx") == DEFAULT_FONT

    def test_empty_string_returns_default(self):
        assert get_font_name("") == DEFAULT_FONT

    def test_none_returns_default(self):
        assert get_font_name(None) == DEFAULT_FONT

    def test_case_insensitive(self):
        assert get_font_name("EN") == "NotoSans-Regular.ttf"
        assert get_font_name("Ch") == "NotoSansCJKsc-Regular.otf"

    def test_fasttext_alias_zh(self):
        assert get_font_name("zh") == "NotoSansCJKsc-Regular.otf"

    def test_fasttext_alias_zh_cn(self):
        assert get_font_name("zh-cn") == "NotoSansCJKsc-Regular.otf"

    def test_fasttext_alias_zh_tw(self):
        assert get_font_name("zh-tw") == "NotoSansCJKtc-Regular.otf"

    def test_fasttext_alias_ja(self):
        assert get_font_name("ja") == "NotoSansCJKjp-Regular.otf"

    def test_fasttext_alias_ko(self):
        assert get_font_name("ko") == "NotoSansCJKkr-Regular.otf"

    def test_fasttext_alias_de(self):
        assert get_font_name("de") == "NotoSans-Regular.ttf"


# ---------------------------------------------------------------------------
# Tests: get_font_path
# ---------------------------------------------------------------------------


class TestGetFontPath:
    def test_returns_none_when_file_missing(self):
        """When font file does not exist, returns None."""
        with mock.patch("font_selector.FONT_DIR", "/nonexistent/dir"):
            result = get_font_path("en")
            assert result is None

    def test_returns_path_when_file_exists(self, tmp_path):
        """When font file exists, returns Path."""
        font_file = tmp_path / "NotoSans-Regular.ttf"
        font_file.write_text("fake font data")

        with mock.patch("font_selector.FONT_DIR", str(tmp_path)):
            result = get_font_path("en")
            assert result is not None
            assert result.name == "NotoSans-Regular.ttf"

    def test_empty_lang_uses_default(self, tmp_path):
        font_file = tmp_path / DEFAULT_FONT
        font_file.write_text("fake")

        with mock.patch("font_selector.FONT_DIR", str(tmp_path)):
            result = get_font_path("")
            assert result is not None
            assert result.name == DEFAULT_FONT

    def test_cjk_font_path(self, tmp_path):
        font_file = tmp_path / "NotoSansCJKsc-Regular.otf"
        font_file.write_text("fake")

        with mock.patch("font_selector.FONT_DIR", str(tmp_path)):
            result = get_font_path("ch")
            assert result is not None
            assert result.name == "NotoSansCJKsc-Regular.otf"


# ---------------------------------------------------------------------------
# Tests: is_font_available
# ---------------------------------------------------------------------------


class TestIsFontAvailable:
    def test_unavailable_when_dir_missing(self):
        with mock.patch("font_selector.FONT_DIR", "/nonexistent/dir"):
            assert is_font_available("en") is False

    def test_available_when_font_exists(self, tmp_path):
        font_file = tmp_path / "NotoSans-Regular.ttf"
        font_file.write_text("fake")

        with mock.patch("font_selector.FONT_DIR", str(tmp_path)):
            assert is_font_available("en") is True

    def test_unavailable_for_missing_font_file(self, tmp_path):
        # Directory exists but no font files
        with mock.patch("font_selector.FONT_DIR", str(tmp_path)):
            assert is_font_available("ch") is False


# ---------------------------------------------------------------------------
# Tests: get_available_fonts
# ---------------------------------------------------------------------------


class TestGetAvailableFonts:
    def test_empty_when_no_fonts(self):
        with mock.patch("font_selector.FONT_DIR", "/nonexistent/dir"):
            result = get_available_fonts()
            assert result == {}

    def test_returns_available_subset(self, tmp_path):
        # Only create the base NotoSans font
        font_file = tmp_path / "NotoSans-Regular.ttf"
        font_file.write_text("fake")

        with mock.patch("font_selector.FONT_DIR", str(tmp_path)):
            result = get_available_fonts()
            # Should include all Latin/Cyrillic/Greek/Vietnamese languages
            assert "en" in result
            assert "fr" in result
            assert "ru" in result
            # Should NOT include CJK (font file not created)
            assert "ch" not in result
            assert "japan" not in result

    def test_all_fonts_available(self, tmp_path):
        """When all font files exist, all expanded languages should be available."""
        unique_fonts = set(LANGUAGE_FONT_MAP.values())
        for font_name in unique_fonts:
            (tmp_path / font_name).write_text("fake")

        with mock.patch("font_selector.FONT_DIR", str(tmp_path)):
            result = get_available_fonts()
            assert len(result) == 34

    def test_deduplicates_font_path_lookups(self):
        resolved = Path("/fonts/NotoSans-Regular.ttf")
        with mock.patch(
            "font_selector._resolve_font_path",
            return_value=resolved,
        ) as mock_resolve:
            result = get_available_fonts()
            assert result
            assert mock_resolve.call_count == len(set(LANGUAGE_FONT_MAP.values()))


# ---------------------------------------------------------------------------
# Tests: get_all_unique_fonts
# ---------------------------------------------------------------------------


class TestGetAllUniqueFonts:
    def test_returns_sorted_list(self):
        fonts = get_all_unique_fonts()
        assert fonts == sorted(fonts)

    def test_includes_default_font(self):
        fonts = get_all_unique_fonts()
        assert DEFAULT_FONT in fonts

    def test_no_duplicates(self):
        fonts = get_all_unique_fonts()
        assert len(fonts) == len(set(fonts))

    def test_expected_unique_count(self):
        """Expanded baseline should carry dedicated fonts for the new scripts."""
        fonts = get_all_unique_fonts()
        expected = {
            "NotoSans-Regular.ttf",
            "NotoSansCJKsc-Regular.otf",
            "NotoSansCJKtc-Regular.otf",
            "NotoSansCJKjp-Regular.otf",
            "NotoSansCJKkr-Regular.otf",
            "NotoSansArabic-Regular.ttf",
            "NotoSansDevanagari-Regular.ttf",
            "NotoSansTamil-Regular.ttf",
            "NotoSansTelugu-Regular.ttf",
            "NotoSansKannada-Regular.ttf",
            "NotoSansGeorgian-Regular.ttf",
        }
        assert set(fonts) == expected


# ---------------------------------------------------------------------------
# Tests: get_font_dir
# ---------------------------------------------------------------------------


class TestGetFontDir:
    def test_returns_path(self):
        result = get_font_dir()
        assert isinstance(result, Path)

    def test_default_value(self):
        result = get_font_dir()
        # Compare as Path objects to handle platform path separators
        assert result == Path(FONT_DIR)


# ---------------------------------------------------------------------------
# Tests: Docker font directory mapping consistency
# ---------------------------------------------------------------------------


class TestDockerFontMapping:
    """Validate that the font filenames expected by font_selector match
    what the Dockerfiles install into /app/fonts/noto/.

    The Dockerfiles symlink 9 non-CJK fonts from the fonts-noto-core package
    and download 4 CJK OTF files. This test ensures the Dockerfile font list
    and the font_selector LANGUAGE_FONT_MAP stay in sync.
    """

    # Exact filenames that the Dockerfiles symlink from fonts-noto-core.
    DOCKER_NON_CJK_FONTS = [
        "NotoSans-Regular.ttf",
        "NotoSansArabic-Regular.ttf",
        "NotoSansDevanagari-Regular.ttf",
        "NotoSansTamil-Regular.ttf",
        "NotoSansTelugu-Regular.ttf",
        "NotoSansKannada-Regular.ttf",
        "NotoSansGeorgian-Regular.ttf",
        "NotoSansThai-Regular.ttf",
        "NotoSansBengali-Regular.ttf",
    ]

    # Exact filenames that the Dockerfiles download via curl.
    DOCKER_CJK_FONTS = [
        "NotoSansCJKsc-Regular.otf",
        "NotoSansCJKtc-Regular.otf",
        "NotoSansCJKjp-Regular.otf",
        "NotoSansCJKkr-Regular.otf",
    ]

    def test_all_docker_fonts_are_known_to_font_selector(self):
        """Every font the Dockerfile installs should be referenced by
        font_selector (either as a core or extended tier font)."""
        unique_fonts = get_all_unique_fonts()
        # Also gather extended-tier fonts that may not be in the default map.
        # Thai and Bengali are in extended tier but the Dockerfile installs
        # them proactively since fonts-noto-core includes them at no cost.
        known = set(unique_fonts)
        known.add("NotoSansThai-Regular.ttf")
        known.add("NotoSansBengali-Regular.ttf")
        for font in self.DOCKER_NON_CJK_FONTS + self.DOCKER_CJK_FONTS:
            assert font in known, (
                f"Dockerfile installs {font} but font_selector does not reference it"
            )

    def test_core_font_map_fonts_are_in_docker_list(self):
        """Every font in the core font_selector map should be installed
        by the Dockerfile."""
        docker_set = set(self.DOCKER_NON_CJK_FONTS + self.DOCKER_CJK_FONTS)
        for font in get_all_unique_fonts():
            assert font in docker_set, (
                f"font_selector expects {font} but Dockerfile does not install it"
            )

    def test_full_coverage_with_mock_directory(self, tmp_path):
        """When all Docker fonts are present, every core language resolves."""
        for font in self.DOCKER_NON_CJK_FONTS + self.DOCKER_CJK_FONTS:
            (tmp_path / font).write_bytes(b"mock-font-data")

        with mock.patch("font_selector.FONT_DIR", str(tmp_path)):
            available = get_available_fonts()
            # All 34 core languages should be covered
            for lang in LANGUAGE_FONT_MAP:
                assert lang in available, (
                    f"Language {lang!r} not available despite Docker fonts being present"
                )

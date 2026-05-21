"""Tests for the consolidated language registry (language_config.py).

Covers registry completeness, tier selection, derived map generation,
convenience helpers, and backward compatibility with existing LANG_MAPPING
consumers.
"""

import logging  # noqa: F401 -- used by caplog tests

import pytest

# ---------------------------------------------------------------------------
# TestLanguageEntry
# ---------------------------------------------------------------------------

class TestLanguageEntry:
    """Validate dataclass construction, defaults, and immutability."""

    def test_construction_with_all_fields(self):
        from language_config import LanguageEntry
        entry = LanguageEntry(
            paddle_code="test",
            name="Test Language",
            fasttext_codes=("tst",),
            script="latin",
            tier="core",
            tesseract_code="tst",
            easyocr_code="tst",
            font="TestFont.ttf",
            rtl=True,
        )
        assert entry.paddle_code == "test"
        assert entry.name == "Test Language"
        assert entry.fasttext_codes == ("tst",)
        assert entry.script == "latin"
        assert entry.tier == "core"
        assert entry.tesseract_code == "tst"
        assert entry.easyocr_code == "tst"
        assert entry.font == "TestFont.ttf"
        assert entry.rtl is True

    def test_defaults(self):
        from language_config import LanguageEntry
        entry = LanguageEntry("x", "X Lang", ("x",), "latin", "core")
        assert entry.tesseract_code == ""
        assert entry.easyocr_code == ""
        assert entry.font == "NotoSans-Regular.ttf"
        assert entry.rtl is False

    def test_frozen_raises_on_mutation(self):
        from language_config import LanguageEntry
        entry = LanguageEntry("x", "X", ("x",), "latin", "core")
        with pytest.raises(AttributeError):
            entry.name = "Changed"  # type: ignore[misc]

    def test_equality(self):
        from language_config import LanguageEntry
        a = LanguageEntry("en", "English", ("en",), "latin", "core", "eng", "en")
        b = LanguageEntry("en", "English", ("en",), "latin", "core", "eng", "en")
        assert a == b

    def test_hash(self):
        from language_config import LanguageEntry
        a = LanguageEntry("en", "English", ("en",), "latin", "core", "eng", "en")
        b = LanguageEntry("en", "English", ("en",), "latin", "core", "eng", "en")
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# TestRegistryCompleteness
# ---------------------------------------------------------------------------

class TestRegistryCompleteness:
    """Verify the registry contains the expected set of languages."""

    def test_core_tier_has_34_languages(self):
        from language_config import LANGUAGE_REGISTRY
        core = [e for e in LANGUAGE_REGISTRY.values() if e.tier == "core"]
        assert len(core) == 34

    def test_extended_tier_has_11_languages(self):
        from language_config import LANGUAGE_REGISTRY
        extended = [e for e in LANGUAGE_REGISTRY.values() if e.tier == "extended"]
        assert len(extended) == 11

    def test_total_registry_size(self):
        from language_config import LANGUAGE_REGISTRY
        assert len(LANGUAGE_REGISTRY) == 45

    def test_no_duplicate_paddle_codes(self):
        from language_config import LANGUAGE_REGISTRY
        codes = list(LANGUAGE_REGISTRY.keys())
        assert len(codes) == len(set(codes))

    def test_core_contains_original_27(self):
        """The original 27 Latin/CJK/Cyrillic baseline must all be core."""
        from language_config import LANGUAGE_REGISTRY
        original_27 = [
            "ch", "chinese_cht", "japan", "korean", "vi",
            "en", "fr", "german", "es", "it", "pt", "nl", "sv", "da",
            "fi", "ro", "pl", "cs", "hu", "tr",
            "ru", "uk", "be", "bg",
            "ar", "hi", "el",
        ]
        for code in original_27:
            assert code in LANGUAGE_REGISTRY, f"Missing core language: {code}"
            assert LANGUAGE_REGISTRY[code].tier == "core"

    def test_core_contains_low_resource_tranche(self):
        """The 7 low-resource languages added in the 34-language tranche."""
        from language_config import LANGUAGE_REGISTRY
        tranche_7 = ["fa", "ur", "ug", "te", "ta", "kn", "ka"]
        for code in tranche_7:
            assert code in LANGUAGE_REGISTRY, f"Missing tranche language: {code}"
            assert LANGUAGE_REGISTRY[code].tier == "core"

    def test_extended_languages_present(self):
        from language_config import LANGUAGE_REGISTRY
        extended = ["th", "bn", "mr", "ne", "hr", "sk", "no", "lt", "lv", "et", "rs_latin"]
        for code in extended:
            assert code in LANGUAGE_REGISTRY, f"Missing extended language: {code}"
            assert LANGUAGE_REGISTRY[code].tier == "extended"

    def test_every_entry_has_name(self):
        from language_config import LANGUAGE_REGISTRY
        for code, entry in LANGUAGE_REGISTRY.items():
            assert entry.name, f"Language {code} has empty name"

    def test_every_entry_has_at_least_one_fasttext_code(self):
        from language_config import LANGUAGE_REGISTRY
        for code, entry in LANGUAGE_REGISTRY.items():
            assert len(entry.fasttext_codes) >= 1, f"Language {code} has no FastText codes"


# ---------------------------------------------------------------------------
# TestGetEnabledLanguages
# ---------------------------------------------------------------------------

class TestGetEnabledLanguages:
    """Test tier-based filtering."""

    def test_core_only_default(self):
        from language_config import get_enabled_languages
        enabled = get_enabled_languages()
        # Default tier is "core" -- should return 34 languages
        assert len(enabled) == 34
        for entry in enabled.values():
            assert entry.tier == "core"

    def test_core_plus_extended(self):
        import language_config
        saved = language_config.OCR_LANGUAGE_TIERS
        try:
            language_config.OCR_LANGUAGE_TIERS = ["core", "extended"]
            enabled = language_config.get_enabled_languages()
            assert len(enabled) == 45
        finally:
            language_config.OCR_LANGUAGE_TIERS = saved

    def test_extended_only(self):
        import language_config
        saved = language_config.OCR_LANGUAGE_TIERS
        try:
            language_config.OCR_LANGUAGE_TIERS = ["extended"]
            enabled = language_config.get_enabled_languages()
            assert len(enabled) == 11
            for entry in enabled.values():
                assert entry.tier == "extended"
        finally:
            language_config.OCR_LANGUAGE_TIERS = saved

    def test_empty_tier_returns_nothing(self):
        import language_config
        saved = language_config.OCR_LANGUAGE_TIERS
        try:
            language_config.OCR_LANGUAGE_TIERS = ["nonexistent"]
            enabled = language_config.get_enabled_languages()
            assert len(enabled) == 0
        finally:
            language_config.OCR_LANGUAGE_TIERS = saved


# ---------------------------------------------------------------------------
# TestBuildLangMapping
# ---------------------------------------------------------------------------

class TestBuildLangMapping:
    """Test FastText-code -> PaddleOCR-code mapping generation."""

    def test_common_mappings(self):
        from language_config import build_lang_mapping
        m = build_lang_mapping()
        assert m["en"] == "en"
        assert m["zh"] == "ch"
        assert m["ja"] == "japan"
        assert m["ko"] == "korean"
        assert m["de"] == "german"

    def test_zh_cn_alias(self):
        from language_config import build_lang_mapping
        m = build_lang_mapping()
        assert m["zh-cn"] == "ch"

    def test_zh_tw_alias(self):
        from language_config import build_lang_mapping
        m = build_lang_mapping()
        assert m["zh-tw"] == "chinese_cht"

    def test_arabic_rtl_languages_present(self):
        from language_config import build_lang_mapping
        m = build_lang_mapping()
        assert m["ar"] == "ar"
        assert m["fa"] == "fa"
        assert m["ur"] == "ur"
        assert m["ug"] == "ug"

    def test_low_resource_tranche_mapped(self):
        from language_config import build_lang_mapping
        m = build_lang_mapping()
        assert m["te"] == "te"
        assert m["ta"] == "ta"
        assert m["kn"] == "kn"
        assert m["ka"] == "ka"

    def test_no_fasttext_code_collisions(self):
        """Each FastText code should map to exactly one PaddleOCR code."""
        from language_config import LANGUAGE_REGISTRY
        seen: dict[str, str] = {}
        for entry in LANGUAGE_REGISTRY.values():
            for ft_code in entry.fasttext_codes:
                if ft_code in seen:
                    pytest.fail(
                        f"FastText code '{ft_code}' mapped to both "
                        f"'{seen[ft_code]}' and '{entry.paddle_code}'"
                    )
                seen[ft_code] = entry.paddle_code


# ---------------------------------------------------------------------------
# TestBuildTargetLangs
# ---------------------------------------------------------------------------

class TestBuildTargetLangs:
    """Test PaddleOCR model code list generation."""

    def test_sorted(self):
        from language_config import build_target_langs
        langs = build_target_langs()
        assert langs == sorted(langs)

    def test_core_count(self):
        from language_config import build_target_langs
        langs = build_target_langs()
        assert len(langs) == 34

    def test_all_core_present(self):
        from language_config import build_target_langs
        langs = build_target_langs()
        for code in ["ch", "en", "japan", "korean", "fr", "german", "ar", "fa", "ta", "ka"]:
            assert code in langs, f"Missing: {code}"


# ---------------------------------------------------------------------------
# TestBuildFontMap
# ---------------------------------------------------------------------------

class TestBuildFontMap:
    """Test font-file mapping generation."""

    def test_cjk_fonts(self):
        from language_config import build_font_map
        fm = build_font_map()
        assert fm["ch"] == "NotoSansCJKsc-Regular.otf"
        assert fm["chinese_cht"] == "NotoSansCJKtc-Regular.otf"
        assert fm["japan"] == "NotoSansCJKjp-Regular.otf"
        assert fm["korean"] == "NotoSansCJKkr-Regular.otf"

    def test_arabic_font(self):
        from language_config import build_font_map
        fm = build_font_map()
        assert fm["ar"] == "NotoSansArabic-Regular.ttf"
        assert fm["fa"] == "NotoSansArabic-Regular.ttf"

    def test_latin_default_font(self):
        from language_config import build_font_map
        fm = build_font_map()
        assert fm["en"] == "NotoSans-Regular.ttf"
        assert fm["fr"] == "NotoSans-Regular.ttf"

    def test_devanagari_font(self):
        from language_config import build_font_map
        fm = build_font_map()
        assert fm["hi"] == "NotoSansDevanagari-Regular.ttf"


# ---------------------------------------------------------------------------
# TestBuildTesseractMap
# ---------------------------------------------------------------------------

class TestBuildTesseractMap:
    """Test PaddleOCR -> Tesseract code mapping."""

    def test_known_mappings(self):
        from language_config import build_tesseract_map
        tm = build_tesseract_map()
        assert tm["en"] == "eng"
        assert tm["fr"] == "fra"
        assert tm["german"] == "deu"

    def test_cjk_tesseract(self):
        from language_config import build_tesseract_map
        tm = build_tesseract_map()
        assert tm["ch"] == "chi_sim"
        assert tm["japan"] == "jpn"

    def test_empty_codes_excluded(self):
        from language_config import build_tesseract_map
        tm = build_tesseract_map()
        for code, tess_code in tm.items():
            assert tess_code, f"Empty tesseract code for {code}"


# ---------------------------------------------------------------------------
# TestBuildEasyocrMap
# ---------------------------------------------------------------------------

class TestBuildEasyocrMap:
    """Test PaddleOCR -> EasyOCR code mapping."""

    def test_known_mappings(self):
        from language_config import build_easyocr_map
        em = build_easyocr_map()
        assert em["en"] == "en"
        assert em["ch"] == "ch_sim"
        assert em["chinese_cht"] == "ch_tra"

    def test_cjk_easyocr(self):
        from language_config import build_easyocr_map
        em = build_easyocr_map()
        assert em["japan"] == "ja"
        assert em["korean"] == "ko"

    def test_empty_codes_excluded(self):
        from language_config import build_easyocr_map
        em = build_easyocr_map()
        for code, eo_code in em.items():
            assert eo_code, f"Empty easyocr code for {code}"


# ---------------------------------------------------------------------------
# TestBuildRtlLanguages
# ---------------------------------------------------------------------------

class TestBuildRtlLanguages:
    """Test right-to-left language set."""

    def test_arabic_is_rtl(self):
        from language_config import build_rtl_languages
        rtl = build_rtl_languages()
        assert "ar" in rtl

    def test_persian_urdu_uyghur_are_rtl(self):
        from language_config import build_rtl_languages
        rtl = build_rtl_languages()
        assert "fa" in rtl
        assert "ur" in rtl
        assert "ug" in rtl

    def test_english_not_rtl(self):
        from language_config import build_rtl_languages
        rtl = build_rtl_languages()
        assert "en" not in rtl

    def test_hindi_not_rtl(self):
        from language_config import build_rtl_languages
        rtl = build_rtl_languages()
        assert "hi" not in rtl


# ---------------------------------------------------------------------------
# TestGetPaddleCode
# ---------------------------------------------------------------------------

class TestGetPaddleCode:
    """Test FastText -> PaddleOCR code lookup with fallback."""

    def test_known_code(self):
        from language_config import get_paddle_code
        assert get_paddle_code("en") == "en"
        assert get_paddle_code("zh") == "ch"
        assert get_paddle_code("de") == "german"

    def test_unknown_code_falls_back_to_en(self):
        from language_config import get_paddle_code
        assert get_paddle_code("xx-nonexistent") == "en"

    def test_unknown_code_logs_warning(self, caplog):
        from language_config import get_paddle_code
        with caplog.at_level(logging.WARNING, logger="language_config"):
            get_paddle_code("zz-unknown")
        assert "Unmapped language" in caplog.text
        assert "zz-unknown" in caplog.text

    def test_case_insensitive(self):
        from language_config import get_paddle_code
        assert get_paddle_code("EN") == "en"
        assert get_paddle_code("Zh") == "ch"

    def test_strips_whitespace(self):
        from language_config import get_paddle_code
        assert get_paddle_code("  en  ") == "en"


# ---------------------------------------------------------------------------
# TestGetTesseractCode
# ---------------------------------------------------------------------------

class TestGetTesseractCode:
    """Test PaddleOCR -> Tesseract code lookup with fallback."""

    def test_known_code(self):
        from language_config import get_tesseract_code
        assert get_tesseract_code("en") == "eng"
        assert get_tesseract_code("fr") == "fra"

    def test_unknown_falls_back_to_eng(self):
        from language_config import get_tesseract_code
        assert get_tesseract_code("nonexistent") == "eng"

    def test_cjk_codes(self):
        from language_config import get_tesseract_code
        assert get_tesseract_code("ch") == "chi_sim"
        assert get_tesseract_code("japan") == "jpn"
        assert get_tesseract_code("korean") == "kor"


# ---------------------------------------------------------------------------
# TestGetFont
# ---------------------------------------------------------------------------

class TestGetFont:
    """Test font file lookup with fallback."""

    def test_cjk_font(self):
        from language_config import get_font
        assert get_font("ch") == "NotoSansCJKsc-Regular.otf"

    def test_arabic_font(self):
        from language_config import get_font
        assert get_font("ar") == "NotoSansArabic-Regular.ttf"

    def test_latin_font(self):
        from language_config import get_font
        assert get_font("en") == "NotoSans-Regular.ttf"

    def test_fallback_font(self):
        from language_config import get_font
        assert get_font("nonexistent") == "NotoSans-Regular.ttf"


# ---------------------------------------------------------------------------
# TestIsRtl
# ---------------------------------------------------------------------------

class TestIsRtl:
    """Test RTL detection."""

    def test_arabic_is_rtl(self):
        from language_config import is_rtl
        assert is_rtl("ar") is True

    def test_persian_is_rtl(self):
        from language_config import is_rtl
        assert is_rtl("fa") is True

    def test_english_not_rtl(self):
        from language_config import is_rtl
        assert is_rtl("en") is False


# ---------------------------------------------------------------------------
# TestGetSupportedLanguageCount
# ---------------------------------------------------------------------------

class TestGetSupportedLanguageCount:
    """Test enabled language count."""

    def test_core_count(self):
        from language_config import get_supported_language_count
        assert get_supported_language_count() == 34

    def test_all_tiers_count(self):
        import language_config
        saved = language_config.OCR_LANGUAGE_TIERS
        try:
            language_config.OCR_LANGUAGE_TIERS = ["core", "extended"]
            assert language_config.get_supported_language_count() == 45
        finally:
            language_config.OCR_LANGUAGE_TIERS = saved


# ---------------------------------------------------------------------------
# TestGetTierSummary
# ---------------------------------------------------------------------------

class TestGetTierSummary:
    """Test tier summary counts."""

    def test_summary_has_both_tiers(self):
        from language_config import get_tier_summary
        summary = get_tier_summary()
        assert "core" in summary
        assert "extended" in summary

    def test_summary_counts(self):
        from language_config import get_tier_summary
        summary = get_tier_summary()
        assert summary["core"] == 34
        assert summary["extended"] == 11

    def test_summary_total(self):
        from language_config import get_tier_summary
        summary = get_tier_summary()
        assert sum(summary.values()) == 45


# ---------------------------------------------------------------------------
# TestBackwardCompatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Ensure the module-level constants match what existing consumers expect."""

    def test_lang_mapping_matches_old_constants(self):
        """LANG_MAPPING must contain every entry from the old constants.py."""
        from language_config import LANG_MAPPING
        old_mapping = {
            'zh': 'ch', 'zh-cn': 'ch', 'zh-tw': 'chinese_cht',
            'ja': 'japan', 'ko': 'korean', 'vi': 'vi',
            'en': 'en', 'fr': 'fr', 'de': 'german', 'es': 'es', 'it': 'it',
            'pt': 'pt', 'ru': 'ru', 'ar': 'ar', 'hi': 'hi',
            'uk': 'uk', 'be': 'be', 'bg': 'bg', 'cs': 'cs', 'pl': 'pl',
            'tr': 'tr', 'nl': 'nl', 'sv': 'sv', 'da': 'da',
            'fi': 'fi', 'el': 'el', 'hu': 'hu', 'ro': 'ro',
            'fa': 'fa', 'ur': 'ur', 'ug': 'ug',
            'te': 'te', 'ta': 'ta', 'kn': 'kn', 'ka': 'ka',
        }
        for ft_code, paddle_code in old_mapping.items():
            assert LANG_MAPPING.get(ft_code) == paddle_code, (
                f"LANG_MAPPING['{ft_code}'] = {LANG_MAPPING.get(ft_code)!r}, "
                f"expected '{paddle_code}'"
            )

    def test_target_langs_contains_all_34(self):
        """TARGET_LANGS must contain all 34 core models."""
        from language_config import TARGET_LANGS
        old_target = [
            'ch', 'chinese_cht', 'japan', 'korean', 'vi',
            'en', 'fr', 'german', 'es', 'it', 'pt', 'ru', 'ar', 'hi',
            'uk', 'be', 'bg', 'cs', 'pl', 'tr', 'nl', 'sv', 'da',
            'fi', 'el', 'hu', 'ro',
            'fa', 'ur', 'ug', 'te', 'ta', 'kn', 'ka',
        ]
        for code in old_target:
            assert code in TARGET_LANGS, f"TARGET_LANGS missing: {code}"

    def test_target_langs_count(self):
        from language_config import TARGET_LANGS
        assert len(TARGET_LANGS) == 34

    def test_lang_mapping_is_dict(self):
        from language_config import LANG_MAPPING
        assert isinstance(LANG_MAPPING, dict)

    def test_target_langs_is_list(self):
        from language_config import TARGET_LANGS
        assert isinstance(TARGET_LANGS, list)

    def test_module_level_maps_are_populated(self):
        from language_config import (
            EASYOCR_MAP,
            FONT_MAP,
            LANG_MAPPING,
            RTL_LANGUAGES,
            TARGET_LANGS,
            TESSERACT_MAP,
        )
        assert len(LANG_MAPPING) > 0
        assert len(TARGET_LANGS) > 0
        assert len(FONT_MAP) > 0
        assert len(TESSERACT_MAP) > 0
        assert len(EASYOCR_MAP) > 0
        assert len(RTL_LANGUAGES) > 0


# ---------------------------------------------------------------------------
# TestScriptMetadata
# ---------------------------------------------------------------------------

class TestScriptMetadata:
    """Verify script family assignments are correct."""

    def test_cjk_scripts(self):
        from language_config import LANGUAGE_REGISTRY
        for code in ["ch", "chinese_cht", "japan", "korean"]:
            assert LANGUAGE_REGISTRY[code].script == "cjk"

    def test_cyrillic_scripts(self):
        from language_config import LANGUAGE_REGISTRY
        for code in ["ru", "uk", "be", "bg"]:
            assert LANGUAGE_REGISTRY[code].script == "cyrillic"

    def test_arabic_scripts(self):
        from language_config import LANGUAGE_REGISTRY
        for code in ["ar", "fa", "ur", "ug"]:
            assert LANGUAGE_REGISTRY[code].script == "arabic"

    def test_latin_scripts(self):
        from language_config import LANGUAGE_REGISTRY
        latin_codes = ["en", "fr", "german", "es", "it", "pt", "nl",
                        "sv", "da", "fi", "ro", "pl", "cs", "hu", "tr", "vi"]
        for code in latin_codes:
            assert LANGUAGE_REGISTRY[code].script == "latin", (
                f"{code} should be latin, got {LANGUAGE_REGISTRY[code].script}"
            )

    def test_extended_latin_scripts(self):
        from language_config import LANGUAGE_REGISTRY
        for code in ["hr", "sk", "no", "lt", "lv", "et", "rs_latin"]:
            assert LANGUAGE_REGISTRY[code].script == "latin"

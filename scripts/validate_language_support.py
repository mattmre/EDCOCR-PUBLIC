"""Validate language support stack for all 45 EDCOCR languages.

Checks language config completeness, font mapping, font file availability,
PaddleOCR model directory presence, and Tesseract language data presence
for each registered language.

Usage:
    python scripts/validate_language_support.py
    python scripts/validate_language_support.py --tier core
    python scripts/validate_language_support.py --tier extended
    python scripts/validate_language_support.py --tier all --check-fonts --check-models
    python scripts/validate_language_support.py --output-json report.json
    python scripts/validate_language_support.py --output-md report.md

Exit codes:
    0 — All config-level checks pass for the selected tier(s)
    1 — At least one config-level check failed (missing entry, mapping, etc.)
"""

import argparse
import datetime
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ocr_local.config.language_config import (
    LANGUAGE_REGISTRY,
    LanguageEntry,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default PaddleOCR model cache location
PADDLEOCR_HOME = os.environ.get(
    "PADDLEOCR_HOME",
    os.path.join(os.path.expanduser("~"), ".paddleocr"),
)
PADDLEOCR_MODEL_DIR = os.path.join(PADDLEOCR_HOME, "whl")

# Font directory (matches font_selector.py)
NOTO_FONT_DIR = os.environ.get("NOTO_FONT_DIR", "/app/fonts/noto")

# Tesseract data directory
TESSDATA_DIR = os.environ.get(
    "TESSDATA_PREFIX",
    "/usr/share/tesseract-ocr/4.00/tessdata",
)

# Valid tiers
VALID_TIERS = {"core", "extended"}

# Valid scripts
VALID_SCRIPTS = {
    "latin", "cyrillic", "cjk", "arabic", "devanagari",
    "tamil", "telugu", "kannada", "georgian", "greek",
    "thai", "bengali",
}


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class LanguageCheckResult:
    """Validation result for a single language."""

    paddle_code: str = ""
    name: str = ""
    tier: str = ""
    script: str = ""

    # Config-level checks (always run)
    config_valid: bool = False
    has_name: bool = False
    has_fasttext_codes: bool = False
    has_valid_script: bool = False
    has_valid_tier: bool = False
    has_font_mapping: bool = False
    font_filename: str = ""
    has_tesseract_code: bool = False
    tesseract_code: str = ""
    has_easyocr_code: bool = False
    easyocr_code: str = ""
    is_rtl: bool = False

    # Disk-level checks (opt-in)
    font_file_exists: bool | None = None
    font_file_path: str = ""
    model_dir_exists: bool | None = None
    model_dir_path: str = ""
    tesseract_data_exists: bool | None = None
    tesseract_data_path: str = ""

    config_issues: list = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report for all languages."""

    timestamp: str = ""
    tiers_checked: list = field(default_factory=list)
    total_languages: int = 0
    config_pass: int = 0
    config_fail: int = 0

    fonts_checked: bool = False
    fonts_found: int = 0
    fonts_missing: int = 0
    unique_fonts_needed: list = field(default_factory=list)

    models_checked: bool = False
    models_found: int = 0
    models_missing: int = 0

    tesseract_checked: bool = False
    tesseract_found: int = 0
    tesseract_missing: int = 0

    all_config_valid: bool = False
    languages: list = field(default_factory=list)

    font_dir: str = ""
    model_dir: str = ""
    tessdata_dir: str = ""


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


def _check_model_dir(paddle_code: str) -> tuple[bool, str]:
    """Check if PaddleOCR model directory exists for a language.

    PaddleOCR stores models under PADDLEOCR_HOME/whl/ with subdirectories
    for det (detection), rec (recognition), and cls (classification).
    The recognition model directory name includes the language code.

    Returns:
        Tuple of (exists, path_checked).
    """
    # PaddleOCR uses a directory structure like:
    # ~/.paddleocr/whl/rec/<lang>/  or  ~/.paddleocr/whl/det/<lang>/
    # The exact structure varies by PaddleOCR version.
    # We check for any directory containing the language code under whl/.
    model_base = Path(PADDLEOCR_MODEL_DIR)
    if not model_base.is_dir():
        return False, str(model_base)

    # Check rec (recognition) directory — the language-specific one
    rec_dir = model_base / "rec"
    if rec_dir.is_dir():
        for child in rec_dir.iterdir():
            if child.is_dir() and paddle_code in child.name:
                return True, str(child)

    # Fallback: scan the entire whl directory for any match
    for dirpath, dirnames, _filenames in os.walk(str(model_base)):
        for dirname in dirnames:
            if paddle_code in dirname:
                return True, os.path.join(dirpath, dirname)

    return False, str(model_base)


def _check_tesseract_data(tesseract_code: str) -> tuple[bool, str]:
    """Check if Tesseract language data file exists.

    Tesseract stores trained data as .traineddata files in TESSDATA_PREFIX.

    Returns:
        Tuple of (exists, path_checked).
    """
    if not tesseract_code:
        return False, ""

    # Also check if tesseract is installed at all
    tessdata = Path(TESSDATA_DIR)
    traineddata = tessdata / f"{tesseract_code}.traineddata"
    path_str = str(traineddata)

    if traineddata.is_file():
        return True, path_str

    # Check alternate common locations
    alt_dirs = [
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tessdata",
        "/usr/local/share/tessdata",
    ]
    for alt in alt_dirs:
        alt_path = Path(alt) / f"{tesseract_code}.traineddata"
        if alt_path.is_file():
            return True, str(alt_path)

    return False, path_str


def validate_language(
    entry: LanguageEntry,
    check_fonts: bool = False,
    check_models: bool = False,
) -> LanguageCheckResult:
    """Validate a single language entry.

    Parameters
    ----------
    entry : LanguageEntry
        The language registry entry to validate.
    check_fonts : bool
        Whether to check font file existence on disk.
    check_models : bool
        Whether to check model directory and tesseract data on disk.

    Returns
    -------
    LanguageCheckResult
        Detailed validation result.
    """
    result = LanguageCheckResult(
        paddle_code=entry.paddle_code,
        name=entry.name,
        tier=entry.tier,
        script=entry.script,
        is_rtl=entry.rtl,
    )

    issues = []

    # Config checks
    result.has_name = bool(entry.name and entry.name.strip())
    if not result.has_name:
        issues.append("missing or empty name")

    result.has_fasttext_codes = len(entry.fasttext_codes) >= 1
    if not result.has_fasttext_codes:
        issues.append("no FastText codes")

    result.has_valid_script = entry.script in VALID_SCRIPTS
    if not result.has_valid_script:
        issues.append(f"invalid script '{entry.script}'")

    result.has_valid_tier = entry.tier in VALID_TIERS
    if not result.has_valid_tier:
        issues.append(f"invalid tier '{entry.tier}'")

    result.has_font_mapping = bool(entry.font)
    result.font_filename = entry.font
    if not result.has_font_mapping:
        issues.append("no font mapping")

    result.has_tesseract_code = bool(entry.tesseract_code)
    result.tesseract_code = entry.tesseract_code

    result.has_easyocr_code = bool(entry.easyocr_code)
    result.easyocr_code = entry.easyocr_code

    result.config_valid = (
        result.has_name
        and result.has_fasttext_codes
        and result.has_valid_script
        and result.has_valid_tier
        and result.has_font_mapping
    )

    # Disk checks (opt-in)
    if check_fonts:
        font_path = Path(NOTO_FONT_DIR) / entry.font
        result.font_file_exists = font_path.is_file()
        result.font_file_path = str(font_path)

    if check_models:
        model_exists, model_path = _check_model_dir(entry.paddle_code)
        result.model_dir_exists = model_exists
        result.model_dir_path = model_path

        if entry.tesseract_code:
            tess_exists, tess_path = _check_tesseract_data(entry.tesseract_code)
            result.tesseract_data_exists = tess_exists
            result.tesseract_data_path = tess_path

    result.config_issues = issues
    return result


def validate_all_languages(
    tiers: list[str] | None = None,
    check_fonts: bool = False,
    check_models: bool = False,
) -> ValidationReport:
    """Validate all languages in the selected tiers.

    Parameters
    ----------
    tiers : list[str] or None
        Which tiers to validate. None or ["all"] means all tiers.
    check_fonts : bool
        Whether to verify font files on disk.
    check_models : bool
        Whether to verify model directories and tesseract data on disk.

    Returns
    -------
    ValidationReport
        Complete validation report.
    """
    # Determine which tiers to check
    if tiers is None or "all" in tiers:
        selected_tiers = list(VALID_TIERS)
    else:
        selected_tiers = [t for t in tiers if t in VALID_TIERS]

    # Filter languages by tier
    entries = [
        entry for entry in LANGUAGE_REGISTRY.values()
        if entry.tier in selected_tiers
    ]

    report = ValidationReport(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        tiers_checked=sorted(selected_tiers),
        total_languages=len(entries),
        fonts_checked=check_fonts,
        models_checked=check_models,
        tesseract_checked=check_models,
        font_dir=NOTO_FONT_DIR,
        model_dir=PADDLEOCR_MODEL_DIR,
        tessdata_dir=TESSDATA_DIR,
    )

    # Collect unique fonts needed
    unique_fonts = sorted({e.font for e in entries if e.font})
    report.unique_fonts_needed = unique_fonts

    results = []
    for entry in sorted(entries, key=lambda e: (e.tier, e.script, e.paddle_code)):
        result = validate_language(entry, check_fonts, check_models)
        results.append(result)

    report.languages = [asdict(r) for r in results]

    # Aggregate counts
    report.config_pass = sum(1 for r in results if r.config_valid)
    report.config_fail = sum(1 for r in results if not r.config_valid)
    report.all_config_valid = report.config_fail == 0

    if check_fonts:
        report.fonts_found = sum(
            1 for r in results if r.font_file_exists is True
        )
        report.fonts_missing = sum(
            1 for r in results if r.font_file_exists is False
        )

    if check_models:
        report.models_found = sum(
            1 for r in results if r.model_dir_exists is True
        )
        report.models_missing = sum(
            1 for r in results if r.model_dir_exists is False
        )
        report.tesseract_found = sum(
            1 for r in results if r.tesseract_data_exists is True
        )
        report.tesseract_missing = sum(
            1 for r in results
            if r.tesseract_data_exists is False and r.has_tesseract_code
        )

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_console_report(report: ValidationReport) -> str:
    """Format validation report for console output.

    Parameters
    ----------
    report : ValidationReport
        The validation report to format.

    Returns
    -------
    str
        Human-readable console report.
    """
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("LANGUAGE SUPPORT VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Timestamp:        {report.timestamp}")
    lines.append(f"  Tiers checked:    {', '.join(report.tiers_checked)}")
    lines.append(f"  Total languages:  {report.total_languages}")
    lines.append(f"  Config PASS:      {report.config_pass}")
    lines.append(f"  Config FAIL:      {report.config_fail}")
    lines.append(
        f"  Overall:          {'PASS' if report.all_config_valid else 'FAIL'}"
    )
    lines.append("")

    if report.fonts_checked:
        lines.append(f"  Font directory:   {report.font_dir}")
        lines.append(f"  Fonts found:      {report.fonts_found}")
        lines.append(f"  Fonts missing:    {report.fonts_missing}")
        lines.append(
            f"  Unique fonts:     {len(report.unique_fonts_needed)}"
        )
        lines.append("")

    if report.models_checked:
        lines.append(f"  Model directory:  {report.model_dir}")
        lines.append(f"  Models found:     {report.models_found}")
        lines.append(f"  Models missing:   {report.models_missing}")
        lines.append(f"  Tessdata dir:     {report.tessdata_dir}")
        lines.append(f"  Tesseract found:  {report.tesseract_found}")
        lines.append(f"  Tesseract miss:   {report.tesseract_missing}")
        lines.append("")

    # Per-language detail table
    lines.append("LANGUAGE DETAILS")
    lines.append("-" * 80)

    header = f"{'Code':<14} {'Name':<20} {'Tier':<9} {'Script':<12} {'Config':<7}"
    if report.fonts_checked:
        header += f" {'Font':<6}"
    if report.models_checked:
        header += f" {'Model':<6} {'Tess':<6}"
    lines.append(header)
    lines.append("-" * 80)

    for lang in report.languages:
        config_status = "OK" if lang["config_valid"] else "FAIL"
        row = (
            f"{lang['paddle_code']:<14} "
            f"{lang['name']:<20} "
            f"{lang['tier']:<9} "
            f"{lang['script']:<12} "
            f"{config_status:<7}"
        )
        if report.fonts_checked:
            if lang["font_file_exists"] is True:
                row += " YES   "
            elif lang["font_file_exists"] is False:
                row += " NO    "
            else:
                row += " -     "
        if report.models_checked:
            if lang["model_dir_exists"] is True:
                row += " YES   "
            elif lang["model_dir_exists"] is False:
                row += " NO    "
            else:
                row += " -     "
            if lang["tesseract_data_exists"] is True:
                row += " YES   "
            elif lang["tesseract_data_exists"] is False:
                row += " NO    "
            else:
                row += " -     "

        lines.append(row)

    lines.append("-" * 80)

    # Show any config issues
    issues_found = False
    for lang in report.languages:
        if lang["config_issues"]:
            if not issues_found:
                lines.append("")
                lines.append("CONFIG ISSUES")
                lines.append("-" * 80)
                issues_found = True
            lines.append(
                f"  {lang['paddle_code']}: {', '.join(lang['config_issues'])}"
            )

    if report.fonts_checked:
        missing_fonts = [
            lang for lang in report.languages
            if lang["font_file_exists"] is False
        ]
        if missing_fonts:
            lines.append("")
            lines.append("MISSING FONT FILES")
            lines.append("-" * 80)
            for lang in missing_fonts:
                lines.append(
                    f"  {lang['paddle_code']} ({lang['name']}): "
                    f"{lang['font_filename']} -> {lang['font_file_path']}"
                )

    lines.append("")
    return "\n".join(lines)


def format_markdown_report(report: ValidationReport) -> str:
    """Format validation report as markdown.

    Parameters
    ----------
    report : ValidationReport
        The validation report to format.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = []
    lines.append("# Language Support Validation Report")
    lines.append("")
    lines.append(f"**Generated**: {report.timestamp}")
    lines.append(f"**Tiers**: {', '.join(report.tiers_checked)}")
    lines.append(f"**Total languages**: {report.total_languages}")
    lines.append(
        f"**Result**: {'PASS' if report.all_config_valid else 'FAIL'}"
    )
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Config PASS | {report.config_pass} |")
    lines.append(f"| Config FAIL | {report.config_fail} |")
    if report.fonts_checked:
        lines.append(f"| Fonts found | {report.fonts_found} |")
        lines.append(f"| Fonts missing | {report.fonts_missing} |")
        lines.append(
            f"| Unique fonts needed | {len(report.unique_fonts_needed)} |"
        )
    if report.models_checked:
        lines.append(f"| Models found | {report.models_found} |")
        lines.append(f"| Models missing | {report.models_missing} |")
        lines.append(f"| Tesseract found | {report.tesseract_found} |")
        lines.append(f"| Tesseract missing | {report.tesseract_missing} |")
    lines.append("")

    # Per-language table
    lines.append("## Language Details")
    lines.append("")

    header = "| Code | Name | Tier | Script | Config |"
    separator = "|------|------|------|--------|--------|"
    if report.fonts_checked:
        header += " Font |"
        separator += "------|"
    if report.models_checked:
        header += " Model | Tesseract |"
        separator += "-------|-----------|"

    lines.append(header)
    lines.append(separator)

    for lang in report.languages:
        config_mark = "PASS" if lang["config_valid"] else "FAIL"
        row = (
            f"| {lang['paddle_code']} "
            f"| {lang['name']} "
            f"| {lang['tier']} "
            f"| {lang['script']} "
            f"| {config_mark} |"
        )
        if report.fonts_checked:
            if lang["font_file_exists"] is True:
                row += " Yes |"
            elif lang["font_file_exists"] is False:
                row += " No |"
            else:
                row += " - |"
        if report.models_checked:
            if lang["model_dir_exists"] is True:
                row += " Yes |"
            elif lang["model_dir_exists"] is False:
                row += " No |"
            else:
                row += " - |"
            if lang["tesseract_data_exists"] is True:
                row += " Yes |"
            elif lang["tesseract_data_exists"] is False:
                row += " No |"
            else:
                row += " - |"
        lines.append(row)

    lines.append("")

    # Unique fonts section
    if report.unique_fonts_needed:
        lines.append("## Required Font Files")
        lines.append("")
        for font in report.unique_fonts_needed:
            lines.append(f"- `{font}`")
        lines.append("")

    # Issues section
    issue_langs = [
        lang for lang in report.languages if lang["config_issues"]
    ]
    if issue_langs:
        lines.append("## Config Issues")
        lines.append("")
        for lang in issue_langs:
            lines.append(
                f"- **{lang['paddle_code']}** ({lang['name']}): "
                f"{', '.join(lang['config_issues'])}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] or None
        Argument list for testing. Defaults to sys.argv[1:].

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate language support stack for EDCOCR "
            "(config, fonts, models, tesseract)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_language_support.py
  python scripts/validate_language_support.py --tier core
  python scripts/validate_language_support.py --tier extended
  python scripts/validate_language_support.py --tier all --check-fonts --check-models
  python scripts/validate_language_support.py --output-json report.json --output-md report.md
        """,
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["core", "extended", "all"],
        help="Which tier(s) to validate (default: all)",
    )
    parser.add_argument(
        "--check-fonts",
        action="store_true",
        default=False,
        help="Verify font files exist on disk at NOTO_FONT_DIR",
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        default=False,
        help="Verify PaddleOCR model directories and Tesseract data on disk",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write JSON report to this path",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Write markdown report to this path",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """CLI entry point for language support validation.

    Parameters
    ----------
    argv : list[str] or None
        Argument list for testing.

    Returns
    -------
    int
        Exit code: 0 if all config checks pass, 1 otherwise.
    """
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Determine tiers
    if args.tier == "all":
        tiers = ["core", "extended"]
    else:
        tiers = [args.tier]

    logger.info(
        "Validating language support for tier(s): %s", ", ".join(tiers)
    )

    report = validate_all_languages(
        tiers=tiers,
        check_fonts=args.check_fonts,
        check_models=args.check_models,
    )

    # Console output
    console_text = format_console_report(report)
    print(console_text)

    # JSON output
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info("JSON report written to %s", args.output_json)

    # Markdown output
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        md_text = format_markdown_report(report)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        logger.info("Markdown report written to %s", args.output_md)

    return 0 if report.all_config_valid else 1


if __name__ == "__main__":
    sys.exit(main())

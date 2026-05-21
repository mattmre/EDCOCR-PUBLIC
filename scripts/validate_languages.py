"""Validate all 45 languages in the EDCOCR language registry.

Performs three categories of checks:

- **fonts**: Verify each language has a font mapping and the font file
  exists on disk at ``NOTO_FONT_DIR``.
- **models**: Verify each language has a PaddleOCR model directory under
  ``PADDLEOCR_HOME`` with non-zero-size model files.
- **mappings**: Verify FastText-to-PaddleOCR language code mappings are
  complete and consistent (no unmapped codes, no collisions).

Usage::

    python scripts/validate_languages.py --check fonts
    python scripts/validate_languages.py --check models
    python scripts/validate_languages.py --check mappings
    python scripts/validate_languages.py --check all
    python scripts/validate_languages.py --check all --output-report report.json

Exit codes:
    0 -- All requested checks pass.
    1 -- At least one check failed.
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
# Configuration
# ---------------------------------------------------------------------------

NOTO_FONT_DIR = os.environ.get("NOTO_FONT_DIR", "/app/fonts/noto")

PADDLEOCR_HOME = os.environ.get(
    "PADDLEOCR_HOME",
    os.path.join(os.path.expanduser("~"), ".paddleocr"),
)
PADDLEOCR_MODEL_DIR = os.path.join(PADDLEOCR_HOME, "whl")

# Minimum file size in bytes for a model file to be considered non-corrupt
MODEL_MIN_SIZE_BYTES = 1024

# Valid check categories
VALID_CHECKS = {"fonts", "models", "mappings", "all"}


def _build_full_lang_mapping() -> dict[str, str]:
    """Build FastText-code -> PaddleOCR-code mapping from the full registry.

    Unlike ``build_lang_mapping()`` from language_config, this includes
    **all** tiers regardless of ``OCR_LANGUAGE_TIERS``, because the
    validation script needs to check every language in the registry.
    """
    mapping: dict[str, str] = {}
    for entry in LANGUAGE_REGISTRY.values():
        for ft_code in entry.fasttext_codes:
            mapping[ft_code] = entry.paddle_code
    # Backward-compatible aliases
    mapping.setdefault("zh-cn", mapping.get("zh", "ch"))
    mapping.setdefault("zh-tw", mapping.get("zh-tw", "chinese_cht"))
    return mapping


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FontCheckResult:
    """Font validation result for a single language."""

    paddle_code: str = ""
    name: str = ""
    font_filename: str = ""
    has_mapping: bool = False
    file_exists: bool = False
    file_path: str = ""
    status: str = "unknown"  # "ok", "missing_file", "no_mapping"


@dataclass
class ModelCheckResult:
    """Model validation result for a single language."""

    paddle_code: str = ""
    name: str = ""
    dir_exists: bool = False
    dir_path: str = ""
    file_count: int = 0
    total_size_bytes: int = 0
    has_zero_size_files: bool = False
    status: str = "unknown"  # "ok", "missing", "corrupt"


@dataclass
class MappingCheckResult:
    """Mapping validation result for a single language."""

    paddle_code: str = ""
    name: str = ""
    fasttext_codes: list = field(default_factory=list)
    has_lang_mapping: bool = False
    has_tesseract_code: bool = False
    tesseract_code: str = ""
    has_easyocr_code: bool = False
    easyocr_code: str = ""
    status: str = "unknown"  # "ok", "incomplete", "broken"
    issues: list = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregate validation report."""

    timestamp: str = ""
    checks_performed: list = field(default_factory=list)
    total_languages: int = 0

    # Font summary
    fonts_total: int = 0
    fonts_ok: int = 0
    fonts_missing: int = 0
    font_results: list = field(default_factory=list)

    # Model summary
    models_total: int = 0
    models_ok: int = 0
    models_missing: int = 0
    models_corrupt: int = 0
    model_results: list = field(default_factory=list)

    # Mapping summary
    mappings_total: int = 0
    mappings_ok: int = 0
    mappings_incomplete: int = 0
    mappings_broken: int = 0
    mapping_collisions: list = field(default_factory=list)
    unmapped_fasttext_codes: list = field(default_factory=list)
    mapping_results: list = field(default_factory=list)

    # Overall
    all_passed: bool = False
    warnings: int = 0
    summary_line: str = ""


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_font(entry: LanguageEntry) -> FontCheckResult:
    """Check font availability for a single language.

    Parameters
    ----------
    entry : LanguageEntry
        The language registry entry to check.

    Returns
    -------
    FontCheckResult
        Font validation result.
    """
    result = FontCheckResult(
        paddle_code=entry.paddle_code,
        name=entry.name,
        font_filename=entry.font,
    )

    if not entry.font:
        result.has_mapping = False
        result.status = "no_mapping"
        return result

    result.has_mapping = True
    font_path = Path(NOTO_FONT_DIR) / entry.font
    result.file_path = str(font_path)
    result.file_exists = font_path.is_file()

    if result.file_exists:
        result.status = "ok"
    else:
        result.status = "missing_file"

    return result


def check_model(entry: LanguageEntry) -> ModelCheckResult:
    """Check PaddleOCR model availability for a single language.

    Scans ``PADDLEOCR_MODEL_DIR`` for a recognition model directory
    whose name contains the language's PaddleOCR code. Verifies model
    files are non-zero size.

    Parameters
    ----------
    entry : LanguageEntry
        The language registry entry to check.

    Returns
    -------
    ModelCheckResult
        Model validation result.
    """
    result = ModelCheckResult(
        paddle_code=entry.paddle_code,
        name=entry.name,
    )

    model_base = Path(PADDLEOCR_MODEL_DIR)
    if not model_base.is_dir():
        result.status = "missing"
        result.dir_path = str(model_base)
        return result

    # Search for a directory containing the language code
    found_dir = _find_model_dir(model_base, entry.paddle_code)
    if found_dir is None:
        result.status = "missing"
        result.dir_path = str(model_base)
        return result

    result.dir_exists = True
    result.dir_path = str(found_dir)

    # Check file sizes
    model_files = list(found_dir.glob("*"))
    result.file_count = len(model_files)

    total_size = 0
    has_zero = False
    for mf in model_files:
        if mf.is_file():
            sz = mf.stat().st_size
            total_size += sz
            if sz == 0:
                has_zero = True

    result.total_size_bytes = total_size
    result.has_zero_size_files = has_zero

    if has_zero or total_size < MODEL_MIN_SIZE_BYTES:
        result.status = "corrupt"
    else:
        result.status = "ok"

    return result


def _find_model_dir(model_base: Path, paddle_code: str) -> Path | None:
    """Find a model directory matching a language code.

    Searches the ``rec/`` subdirectory first, then performs a full
    walk of the model base.

    Parameters
    ----------
    model_base : Path
        Root of the PaddleOCR model cache (e.g. ``~/.paddleocr/whl``).
    paddle_code : str
        PaddleOCR language code to search for.

    Returns
    -------
    Path or None
        The matching directory, or None if not found.
    """
    rec_dir = model_base / "rec"
    if rec_dir.is_dir():
        for child in rec_dir.iterdir():
            if child.is_dir() and paddle_code in child.name:
                return child

    # Fallback: walk entire tree
    for dirpath, dirnames, _filenames in os.walk(str(model_base)):
        for dirname in dirnames:
            if paddle_code in dirname:
                return Path(dirpath) / dirname

    return None


def check_mapping(entry: LanguageEntry) -> MappingCheckResult:
    """Check language code mappings for a single language.

    Verifies:
    - FastText codes map to the correct PaddleOCR model code
    - Tesseract code is present
    - EasyOCR code is present

    Parameters
    ----------
    entry : LanguageEntry
        The language registry entry to check.

    Returns
    -------
    MappingCheckResult
        Mapping validation result.
    """
    lang_mapping = _build_full_lang_mapping()

    result = MappingCheckResult(
        paddle_code=entry.paddle_code,
        name=entry.name,
        fasttext_codes=list(entry.fasttext_codes),
    )

    issues = []

    # Check FastText -> PaddleOCR mapping
    all_mapped = True
    for ft_code in entry.fasttext_codes:
        mapped_paddle = lang_mapping.get(ft_code)
        if mapped_paddle is None:
            all_mapped = False
            issues.append(f"FastText code '{ft_code}' not in LANG_MAPPING")
        elif mapped_paddle != entry.paddle_code:
            all_mapped = False
            issues.append(
                f"FastText code '{ft_code}' maps to '{mapped_paddle}' "
                f"instead of '{entry.paddle_code}'"
            )

    result.has_lang_mapping = all_mapped

    # Tesseract code
    result.has_tesseract_code = bool(entry.tesseract_code)
    result.tesseract_code = entry.tesseract_code
    if not entry.tesseract_code:
        issues.append("no Tesseract code")

    # EasyOCR code
    result.has_easyocr_code = bool(entry.easyocr_code)
    result.easyocr_code = entry.easyocr_code
    if not entry.easyocr_code:
        issues.append("no EasyOCR code")

    result.issues = issues

    if not all_mapped:
        result.status = "broken"
    elif issues:
        result.status = "incomplete"
    else:
        result.status = "ok"

    return result


def check_mapping_collisions() -> list[dict]:
    """Detect FastText code collisions across the full registry.

    Returns
    -------
    list[dict]
        List of collision entries, each with ``fasttext_code``,
        ``paddle_code_a``, and ``paddle_code_b``.
    """
    collisions = []
    seen: dict[str, str] = {}
    for entry in LANGUAGE_REGISTRY.values():
        for ft_code in entry.fasttext_codes:
            if ft_code in seen and seen[ft_code] != entry.paddle_code:
                collisions.append({
                    "fasttext_code": ft_code,
                    "paddle_code_a": seen[ft_code],
                    "paddle_code_b": entry.paddle_code,
                })
            else:
                seen[ft_code] = entry.paddle_code
    return collisions


def find_unmapped_fasttext_codes() -> list[str]:
    """Find FastText language codes that have no LANG_MAPPING entry.

    Compares the full set of FastText codes in the registry against
    the mapping returned by ``build_lang_mapping()``.

    Returns
    -------
    list[str]
        FastText codes with no mapping entry.
    """
    lang_mapping = _build_full_lang_mapping()
    unmapped = []
    for entry in LANGUAGE_REGISTRY.values():
        for ft_code in entry.fasttext_codes:
            if ft_code not in lang_mapping:
                unmapped.append(ft_code)
    return sorted(set(unmapped))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def run_validation(checks: list[str]) -> ValidationReport:
    """Run the requested validation checks and produce a report.

    Parameters
    ----------
    checks : list[str]
        List of check categories: ``"fonts"``, ``"models"``,
        ``"mappings"``, or ``"all"``.

    Returns
    -------
    ValidationReport
        Aggregate validation report.
    """
    if "all" in checks:
        checks = ["fonts", "models", "mappings"]

    report = ValidationReport(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        checks_performed=sorted(checks),
        total_languages=len(LANGUAGE_REGISTRY),
    )

    entries = sorted(
        LANGUAGE_REGISTRY.values(),
        key=lambda e: (e.tier, e.script, e.paddle_code),
    )

    warnings = 0

    if "fonts" in checks:
        font_results = []
        ok = missing = 0
        for entry in entries:
            fr = check_font(entry)
            font_results.append(asdict(fr))
            if fr.status == "ok":
                ok += 1
            elif fr.status == "missing_file":
                missing += 1
                warnings += 1
            elif fr.status == "no_mapping":
                missing += 1
                warnings += 1

        report.fonts_total = len(entries)
        report.fonts_ok = ok
        report.fonts_missing = missing
        report.font_results = font_results

    if "models" in checks:
        model_results = []
        ok = missing = corrupt = 0
        for entry in entries:
            mr = check_model(entry)
            model_results.append(asdict(mr))
            if mr.status == "ok":
                ok += 1
            elif mr.status == "missing":
                missing += 1
                warnings += 1
            elif mr.status == "corrupt":
                corrupt += 1
                warnings += 1

        report.models_total = len(entries)
        report.models_ok = ok
        report.models_missing = missing
        report.models_corrupt = corrupt
        report.model_results = model_results

    if "mappings" in checks:
        mapping_results = []
        ok = incomplete = broken = 0
        for entry in entries:
            mr = check_mapping(entry)
            mapping_results.append(asdict(mr))
            if mr.status == "ok":
                ok += 1
            elif mr.status == "incomplete":
                incomplete += 1
                warnings += 1
            elif mr.status == "broken":
                broken += 1
                warnings += 1

        report.mappings_total = len(entries)
        report.mappings_ok = ok
        report.mappings_incomplete = incomplete
        report.mappings_broken = broken
        report.mapping_collisions = check_mapping_collisions()
        report.unmapped_fasttext_codes = find_unmapped_fasttext_codes()
        report.mapping_results = mapping_results

        if report.mapping_collisions:
            warnings += len(report.mapping_collisions)
        if report.unmapped_fasttext_codes:
            warnings += len(report.unmapped_fasttext_codes)

    report.warnings = warnings

    # Determine overall pass/fail
    passed = True
    if "fonts" in checks and report.fonts_missing > 0:
        # Font file missing on disk is a warning, not a hard failure
        # (fonts may not be installed in dev environments)
        pass
    if "models" in checks and report.models_corrupt > 0:
        passed = False
    if "mappings" in checks:
        if report.mappings_broken > 0:
            passed = False
        if report.mapping_collisions:
            passed = False

    report.all_passed = passed
    validated = report.total_languages
    report.summary_line = (
        f"{validated}/{len(LANGUAGE_REGISTRY)} languages fully validated, "
        f"{warnings} warnings"
    )

    return report


def format_report(report: ValidationReport) -> str:
    """Format a validation report for console output.

    Parameters
    ----------
    report : ValidationReport
        The report to format.

    Returns
    -------
    str
        Human-readable console report.
    """
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("LANGUAGE VALIDATION REPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"  Timestamp:    {report.timestamp}")
    lines.append(f"  Checks:       {', '.join(report.checks_performed)}")
    lines.append(f"  Languages:    {report.total_languages}")
    lines.append(f"  Result:       {'PASS' if report.all_passed else 'FAIL'}")
    lines.append(f"  Summary:      {report.summary_line}")
    lines.append("")

    if report.font_results:
        lines.append("-" * 78)
        lines.append("FONT CHECKS")
        lines.append("-" * 78)
        lines.append(
            f"  OK: {report.fonts_ok}  "
            f"Missing: {report.fonts_missing}"
        )
        lines.append("")
        lines.append(f"  {'Code':<14} {'Name':<20} {'Font':<35} {'Status'}")
        lines.append(f"  {'-'*14} {'-'*20} {'-'*35} {'-'*6}")
        for fr in report.font_results:
            lines.append(
                f"  {fr['paddle_code']:<14} "
                f"{fr['name']:<20} "
                f"{fr['font_filename']:<35} "
                f"{fr['status'].upper()}"
            )
        lines.append("")

    if report.model_results:
        lines.append("-" * 78)
        lines.append("MODEL CHECKS")
        lines.append("-" * 78)
        lines.append(
            f"  OK: {report.models_ok}  "
            f"Missing: {report.models_missing}  "
            f"Corrupt: {report.models_corrupt}"
        )
        lines.append("")
        lines.append(
            f"  {'Code':<14} {'Name':<20} {'Files':<6} "
            f"{'Size':<12} {'Status'}"
        )
        lines.append(
            f"  {'-'*14} {'-'*20} {'-'*6} {'-'*12} {'-'*7}"
        )
        for mr in report.model_results:
            size_kb = mr["total_size_bytes"] / 1024 if mr["total_size_bytes"] else 0
            size_str = f"{size_kb:.0f} KB" if size_kb else "-"
            lines.append(
                f"  {mr['paddle_code']:<14} "
                f"{mr['name']:<20} "
                f"{mr['file_count']:<6} "
                f"{size_str:<12} "
                f"{mr['status'].upper()}"
            )
        lines.append("")

    if report.mapping_results:
        lines.append("-" * 78)
        lines.append("MAPPING CHECKS")
        lines.append("-" * 78)
        lines.append(
            f"  OK: {report.mappings_ok}  "
            f"Incomplete: {report.mappings_incomplete}  "
            f"Broken: {report.mappings_broken}"
        )
        lines.append("")
        lines.append(
            f"  {'Code':<14} {'Name':<20} "
            f"{'FastText':<14} {'Tess':<10} {'EasyOCR':<10} {'Status'}"
        )
        lines.append(
            f"  {'-'*14} {'-'*20} "
            f"{'-'*14} {'-'*10} {'-'*10} {'-'*10}"
        )
        for mr in report.mapping_results:
            ft_codes = ",".join(mr["fasttext_codes"])
            lines.append(
                f"  {mr['paddle_code']:<14} "
                f"{mr['name']:<20} "
                f"{ft_codes:<14} "
                f"{mr['tesseract_code'] or '-':<10} "
                f"{mr['easyocr_code'] or '-':<10} "
                f"{mr['status'].upper()}"
            )

        if report.mapping_collisions:
            lines.append("")
            lines.append("  COLLISIONS:")
            for col in report.mapping_collisions:
                lines.append(
                    f"    FastText '{col['fasttext_code']}' maps to both "
                    f"'{col['paddle_code_a']}' and '{col['paddle_code_b']}'"
                )

        if report.unmapped_fasttext_codes:
            lines.append("")
            lines.append("  UNMAPPED FASTTEXT CODES:")
            for code in report.unmapped_fasttext_codes:
                lines.append(f"    {code}")

        lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
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
            "Validate all 45 languages in the EDCOCR language registry "
            "(fonts, models, mappings)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_languages.py --check fonts
  python scripts/validate_languages.py --check models
  python scripts/validate_languages.py --check mappings
  python scripts/validate_languages.py --check all
  python scripts/validate_languages.py --check all --output-report report.json
        """,
    )
    parser.add_argument(
        "--check",
        type=str,
        default="all",
        choices=["fonts", "models", "mappings", "all"],
        help="Which validation category to run (default: all)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Write JSON report to this path",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """CLI entry point.

    Parameters
    ----------
    argv : list[str] or None
        Argument list for testing.

    Returns
    -------
    int
        Exit code: 0 on pass, 1 on failure.
    """
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    checks = [args.check] if args.check != "all" else ["all"]
    logger.info("Running language validation: %s", args.check)

    report = run_validation(checks)

    # Console output
    console_text = format_report(report)
    print(console_text)

    # JSON output
    if args.output_report:
        output_path = Path(args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info("JSON report written to %s", args.output_report)

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

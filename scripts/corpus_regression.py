#!/usr/bin/env python3
"""Corpus regression testing framework for OCR accuracy.

Compares OCR output against ground-truth text files, computes CER/WER/
line-accuracy metrics, and detects regressions against a saved baseline.

Supports per-language and per-category stratification via optional JSON
mapping files.

Usage:
    # Basic comparison (text summary):
    python scripts/corpus_regression.py \\
        --corpus-dir tests/fixtures/corpus/output/ \\
        --ground-truth-dir tests/fixtures/corpus/ground-truth/

    # Compare against a saved baseline:
    python scripts/corpus_regression.py \\
        --corpus-dir tests/fixtures/corpus/output/ \\
        --ground-truth-dir tests/fixtures/corpus/ground-truth/ \\
        --compare-baseline

    # Save current results as the new baseline:
    python scripts/corpus_regression.py \\
        --corpus-dir tests/fixtures/corpus/output/ \\
        --ground-truth-dir tests/fixtures/corpus/ground-truth/ \\
        --save-baseline

    # JSON output with language and category stratification:
    python scripts/corpus_regression.py \\
        --corpus-dir tests/fixtures/corpus/output/ \\
        --ground-truth-dir tests/fixtures/corpus/ground-truth/ \\
        --language-map tests/fixtures/corpus/language-map.json \\
        --category-map tests/fixtures/corpus/category-map.json \\
        --json

    # Markdown regression report:
    python scripts/corpus_regression.py \\
        --corpus-dir tests/fixtures/corpus/output/ \\
        --ground-truth-dir tests/fixtures/corpus/ground-truth/ \\
        --compare-baseline \\
        --report docs/reports/corpus-regression-report.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import statistics
import sys
from pathlib import Path

# Ensure project root is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ocr_distributed.text_metrics import accuracy_summary

logger = logging.getLogger("corpus_regression")

# Default baseline location
_DEFAULT_BASELINE_PATH = _PROJECT_ROOT / "docs" / "reports" / "corpus-baseline.json"

# Valid document categories
VALID_CATEGORIES = frozenset({
    "printed_text",
    "degraded_scan",
    "handwritten",
    "multilingual",
    "table_heavy",
    "mixed_layout",
})

# Ground-truth file extensions we recognise
_GT_EXTENSIONS = {".gt.txt"}
_TEXT_EXTENSION = ".txt"


# ---------------------------------------------------------------------------
# Document matching
# ---------------------------------------------------------------------------


def _stem_from_gt(path: Path) -> str:
    """Extract the document stem from a ground-truth filename.

    For example ``sample-clean.gt.txt`` yields ``sample-clean``.
    """
    name = path.name
    if name.endswith(".gt.txt"):
        return name[: -len(".gt.txt")]
    return path.stem


def match_documents(
    corpus_dir: Path,
    ground_truth_dir: Path,
) -> list[tuple[str, Path, Path]]:
    """Find matching output/ground-truth pairs by document stem.

    Returns a list of ``(doc_id, output_path, gt_path)`` tuples sorted
    by *doc_id*.
    """
    gt_files: dict[str, Path] = {}
    for gt_path in ground_truth_dir.iterdir():
        if gt_path.is_file() and (
            gt_path.name.endswith(".gt.txt") or gt_path.suffix == _TEXT_EXTENSION
        ):
            stem = _stem_from_gt(gt_path)
            gt_files[stem] = gt_path

    pairs: list[tuple[str, Path, Path]] = []
    for out_path in corpus_dir.iterdir():
        if out_path.is_file() and out_path.suffix == _TEXT_EXTENSION:
            stem = out_path.stem
            if stem in gt_files:
                pairs.append((stem, out_path, gt_files[stem]))

    return sorted(pairs, key=lambda t: t[0])


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_document_metrics(
    output_text: str,
    gt_text: str,
) -> dict:
    """Compute CER, WER, and line accuracy for a single document.

    Returns a dict with ``cer``, ``wer``, ``line_accuracy``, ``char_count``,
    and ``word_count``.
    """
    return accuracy_summary(gt_text, output_text)


def compute_corpus_metrics(
    corpus_dir: Path,
    ground_truth_dir: Path,
    language_map: dict[str, str] | None = None,
    category_map: dict[str, str] | None = None,
) -> dict:
    """Run full corpus comparison and return structured results.

    Parameters
    ----------
    corpus_dir : Path
        Directory containing OCR output text files.
    ground_truth_dir : Path
        Directory containing ground-truth text files (``*.gt.txt``).
    language_map : dict, optional
        Mapping of ``doc_id`` to language code for stratification.
    category_map : dict, optional
        Mapping of ``doc_id`` to category for stratification.

    Returns
    -------
    dict
        Full result structure suitable for JSON serialisation or
        baseline storage.
    """
    pairs = match_documents(corpus_dir, ground_truth_dir)

    if not pairs:
        logger.warning("No matching document pairs found.")
        return {
            "version": _get_version(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "documents": {},
            "aggregate": {},
            "by_language": {},
            "by_category": {},
        }

    documents: dict[str, dict] = {}
    for doc_id, out_path, gt_path in pairs:
        output_text = out_path.read_text(encoding="utf-8", errors="replace")
        gt_text = gt_path.read_text(encoding="utf-8", errors="replace")

        metrics = compute_document_metrics(output_text, gt_text)
        entry: dict = {
            "cer": round(metrics["cer"], 6),
            "wer": round(metrics["wer"], 6),
            "line_accuracy": round(metrics["line_accuracy"], 6),
        }
        if language_map and doc_id in language_map:
            entry["language"] = language_map[doc_id]
        if category_map and doc_id in category_map:
            entry["category"] = category_map[doc_id]

        documents[doc_id] = entry

    aggregate = _compute_aggregate(documents)
    by_language = _stratify(documents, "language")
    by_category = _stratify(documents, "category")

    return {
        "version": _get_version(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "documents": documents,
        "aggregate": aggregate,
        "by_language": by_language,
        "by_category": by_category,
    }


def _compute_aggregate(documents: dict[str, dict]) -> dict:
    """Compute mean CER, WER, line_accuracy across all documents."""
    if not documents:
        return {}
    cers = [d["cer"] for d in documents.values()]
    wers = [d["wer"] for d in documents.values()]
    las = [d["line_accuracy"] for d in documents.values()]
    return {
        "mean_cer": round(statistics.mean(cers), 6),
        "mean_wer": round(statistics.mean(wers), 6),
        "mean_line_accuracy": round(statistics.mean(las), 6),
        "document_count": len(documents),
    }


def _stratify(documents: dict[str, dict], key: str) -> dict:
    """Group documents by *key* and compute per-group aggregates."""
    groups: dict[str, list[dict]] = {}
    for doc in documents.values():
        group_val = doc.get(key)
        if group_val is not None:
            groups.setdefault(group_val, []).append(doc)

    result: dict[str, dict] = {}
    for group_name, docs in sorted(groups.items()):
        cers = [d["cer"] for d in docs]
        wers = [d["wer"] for d in docs]
        las = [d["line_accuracy"] for d in docs]
        result[group_name] = {
            "mean_cer": round(statistics.mean(cers), 6),
            "mean_wer": round(statistics.mean(wers), 6),
            "mean_line_accuracy": round(statistics.mean(las), 6),
            "document_count": len(docs),
        }
    return result


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------


def load_baseline(path: Path) -> dict | None:
    """Load a saved baseline from JSON, or return None if not found."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load baseline from %s: %s", path, exc)
        return None


def save_baseline(results: dict, path: Path) -> None:
    """Save corpus results as a baseline JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Baseline saved to %s", path)


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


def detect_regressions(
    current: dict,
    baseline: dict,
    threshold: float = 0.02,
) -> list[dict]:
    """Compare current results against a baseline and return regressions.

    A regression is detected when any metric for a document degrades
    (increases for CER/WER, decreases for line_accuracy) by more than
    *threshold*.

    Parameters
    ----------
    current : dict
        Current corpus results (from :func:`compute_corpus_metrics`).
    baseline : dict
        Previously saved baseline.
    threshold : float
        Allowed degradation before flagging a regression (default 0.02).

    Returns
    -------
    list[dict]
        List of regression entries, each with ``doc_id``, ``metric``,
        ``baseline_value``, ``current_value``, ``delta``.
    """
    regressions: list[dict] = []
    baseline_docs = baseline.get("documents", {})
    current_docs = current.get("documents", {})

    for doc_id, cur in current_docs.items():
        base = baseline_docs.get(doc_id)
        if base is None:
            continue  # new document, skip comparison

        # CER / WER: regression if current is higher (worse)
        for metric in ("cer", "wer"):
            delta = cur[metric] - base[metric]
            if delta > threshold:
                regressions.append({
                    "doc_id": doc_id,
                    "metric": metric,
                    "baseline_value": base[metric],
                    "current_value": cur[metric],
                    "delta": round(delta, 6),
                })

        # Line accuracy: regression if current is lower (worse)
        la_delta = base["line_accuracy"] - cur["line_accuracy"]
        if la_delta > threshold:
            regressions.append({
                "doc_id": doc_id,
                "metric": "line_accuracy",
                "baseline_value": base["line_accuracy"],
                "current_value": cur["line_accuracy"],
                "delta": round(-la_delta, 6),
            })

    return regressions


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def format_text_summary(results: dict, regressions: list[dict] | None = None) -> str:
    """Render a text table summarising the corpus results."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Corpus Regression Report")
    lines.append("=" * 70)
    lines.append(f"  Version:   {results.get('version', 'unknown')}")
    lines.append(f"  Timestamp: {results.get('timestamp', 'unknown')}")

    agg = results.get("aggregate", {})
    if agg:
        lines.append(f"  Documents: {agg.get('document_count', 0)}")
        lines.append(f"  Mean CER:  {agg.get('mean_cer', 0):.4f}")
        lines.append(f"  Mean WER:  {agg.get('mean_wer', 0):.4f}")
        lines.append(f"  Mean LA:   {agg.get('mean_line_accuracy', 0):.4f}")

    lines.append("")
    lines.append("Per-document metrics:")
    lines.append(f"  {'Document':<30s} {'CER':>8s} {'WER':>8s} {'Line Acc':>10s}")
    lines.append(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 10}")
    for doc_id, m in sorted(results.get("documents", {}).items()):
        lines.append(
            f"  {doc_id:<30s} {m['cer']:>8.4f} {m['wer']:>8.4f} {m['line_accuracy']:>10.4f}"
        )

    by_lang = results.get("by_language", {})
    if by_lang:
        lines.append("")
        lines.append("Per-language aggregates:")
        lines.append(f"  {'Language':<15s} {'CER':>8s} {'WER':>8s} {'Line Acc':>10s} {'Count':>6s}")
        lines.append(f"  {'-' * 15} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 6}")
        for lang, m in sorted(by_lang.items()):
            lines.append(
                f"  {lang:<15s} {m['mean_cer']:>8.4f} {m['mean_wer']:>8.4f} "
                f"{m['mean_line_accuracy']:>10.4f} {m['document_count']:>6d}"
            )

    by_cat = results.get("by_category", {})
    if by_cat:
        lines.append("")
        lines.append("Per-category aggregates:")
        lines.append(f"  {'Category':<20s} {'CER':>8s} {'WER':>8s} {'Line Acc':>10s} {'Count':>6s}")
        lines.append(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 6}")
        for cat, m in sorted(by_cat.items()):
            lines.append(
                f"  {cat:<20s} {m['mean_cer']:>8.4f} {m['mean_wer']:>8.4f} "
                f"{m['mean_line_accuracy']:>10.4f} {m['document_count']:>6d}"
            )

    if regressions is not None:
        lines.append("")
        if regressions:
            lines.append(f"REGRESSIONS DETECTED ({len(regressions)}):")
            for r in regressions:
                lines.append(
                    f"  {r['doc_id']}: {r['metric']} "
                    f"baseline={r['baseline_value']:.4f} "
                    f"current={r['current_value']:.4f} "
                    f"delta={r['delta']:+.4f}"
                )
        else:
            lines.append("No regressions detected.")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_markdown_report(
    results: dict, regressions: list[dict] | None = None
) -> str:
    """Render a Markdown regression report."""
    lines: list[str] = []
    lines.append("# Corpus Regression Report")
    lines.append("")
    lines.append(f"- **Version**: {results.get('version', 'unknown')}")
    lines.append(f"- **Timestamp**: {results.get('timestamp', 'unknown')}")

    agg = results.get("aggregate", {})
    if agg:
        lines.append(f"- **Documents**: {agg.get('document_count', 0)}")
        lines.append(f"- **Mean CER**: {agg.get('mean_cer', 0):.4f}")
        lines.append(f"- **Mean WER**: {agg.get('mean_wer', 0):.4f}")
        lines.append(f"- **Mean Line Accuracy**: {agg.get('mean_line_accuracy', 0):.4f}")

    lines.append("")
    lines.append("## Per-Document Metrics")
    lines.append("")
    lines.append("| Document | CER | WER | Line Accuracy |")
    lines.append("|----------|-----|-----|---------------|")
    for doc_id, m in sorted(results.get("documents", {}).items()):
        lines.append(
            f"| {doc_id} | {m['cer']:.4f} | {m['wer']:.4f} | {m['line_accuracy']:.4f} |"
        )

    by_lang = results.get("by_language", {})
    if by_lang:
        lines.append("")
        lines.append("## Per-Language Aggregates")
        lines.append("")
        lines.append("| Language | Mean CER | Mean WER | Mean Line Accuracy | Count |")
        lines.append("|----------|----------|----------|-------------------|-------|")
        for lang, m in sorted(by_lang.items()):
            lines.append(
                f"| {lang} | {m['mean_cer']:.4f} | {m['mean_wer']:.4f} "
                f"| {m['mean_line_accuracy']:.4f} | {m['document_count']} |"
            )

    by_cat = results.get("by_category", {})
    if by_cat:
        lines.append("")
        lines.append("## Per-Category Aggregates")
        lines.append("")
        lines.append("| Category | Mean CER | Mean WER | Mean Line Accuracy | Count |")
        lines.append("|----------|----------|----------|-------------------|-------|")
        for cat, m in sorted(by_cat.items()):
            lines.append(
                f"| {cat} | {m['mean_cer']:.4f} | {m['mean_wer']:.4f} "
                f"| {m['mean_line_accuracy']:.4f} | {m['document_count']} |"
            )

    if regressions is not None:
        lines.append("")
        if regressions:
            lines.append(f"## Regressions ({len(regressions)})")
            lines.append("")
            lines.append("| Document | Metric | Baseline | Current | Delta |")
            lines.append("|----------|--------|----------|---------|-------|")
            for r in regressions:
                lines.append(
                    f"| {r['doc_id']} | {r['metric']} | {r['baseline_value']:.4f} "
                    f"| {r['current_value']:.4f} | {r['delta']:+.4f} |"
                )
        else:
            lines.append("## Regressions")
            lines.append("")
            lines.append("No regressions detected.")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_version() -> str:
    """Return the project version string, falling back to 'unknown'."""
    try:
        from ocr_local.config.version import __version__

        return __version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Corpus regression testing for OCR accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus-dir",
        required=True,
        type=Path,
        help="Directory containing OCR output text files.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        required=True,
        type=Path,
        help="Directory containing ground-truth text files (*.gt.txt).",
    )
    parser.add_argument(
        "--language-map",
        type=Path,
        default=None,
        help="JSON mapping doc_id -> language code.",
    )
    parser.add_argument(
        "--category-map",
        type=Path,
        default=None,
        help="JSON mapping doc_id -> document category.",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=_DEFAULT_BASELINE_PATH,
        help="Path to the baseline JSON file (default: docs/reports/corpus-baseline.json).",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current results as the new baseline.",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare current results against a saved baseline.",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.02,
        help="Degradation threshold for regression detection (default: 0.02).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Print results as JSON instead of a text table.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write a Markdown regression report to this path.",
    )
    return parser


def _load_json_map(path: Path | None) -> dict[str, str] | None:
    """Load an optional JSON mapping file."""
    if path is None:
        return None
    if not path.exists():
        logger.warning("Mapping file not found: %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def main(argv: list[str] | None = None) -> int:
    """Run the corpus regression CLI.

    Returns 0 on success, 1 if regressions are detected.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.corpus_dir.is_dir():
        logger.error("Corpus directory does not exist: %s", args.corpus_dir)
        return 1
    if not args.ground_truth_dir.is_dir():
        logger.error("Ground-truth directory does not exist: %s", args.ground_truth_dir)
        return 1

    language_map = _load_json_map(args.language_map)
    category_map = _load_json_map(args.category_map)

    results = compute_corpus_metrics(
        corpus_dir=args.corpus_dir,
        ground_truth_dir=args.ground_truth_dir,
        language_map=language_map,
        category_map=category_map,
    )

    regressions: list[dict] | None = None
    if args.compare_baseline:
        baseline = load_baseline(args.baseline_path)
        if baseline is None:
            logger.warning(
                "No baseline found at %s; skipping regression comparison.",
                args.baseline_path,
            )
        else:
            regressions = detect_regressions(
                results, baseline, threshold=args.regression_threshold
            )

    if args.save_baseline:
        save_baseline(results, args.baseline_path)

    # Output
    if args.output_json:
        output = dict(results)
        if regressions is not None:
            output["regressions"] = regressions
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(format_text_summary(results, regressions))

    if args.report:
        md = format_markdown_report(results, regressions)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(md, encoding="utf-8")
        logger.info("Markdown report written to %s", args.report)

    # Exit code: 1 if regressions detected
    if regressions:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Corpus validation framework for OCR output quality.

Validates OCR output against ground-truth annotated document corpora by
computing character error rate (CER), word error rate (WER), and line
accuracy metrics.  Supports multiple ground-truth formats and configurable
quality thresholds.

Usage:
    # Validate entire corpus (plain text ground truth):
    python scripts/corpus_validator.py --corpus-dir ocr_output/ \
        --ground-truth-dir ground_truth/

    # Validate single document:
    python scripts/corpus_validator.py --corpus-dir ocr_output/ \
        --ground-truth-dir ground_truth/ --doc-id invoice_001

    # Custom thresholds and JSON output:
    python scripts/corpus_validator.py --corpus-dir ocr_output/ \
        --ground-truth-dir ground_truth/ --json \
        --thresholds '{"cer": 0.03, "wer": 0.08, "line_accuracy": 0.95}'

    # ALTO XML ground truth format:
    python scripts/corpus_validator.py --corpus-dir ocr_output/ \
        --ground-truth-dir ground_truth/ --format ALTO_XML
"""

from __future__ import annotations

import argparse
import enum
import json
import logging
import os
import sys
import time
import unicodedata
from dataclasses import dataclass, field

logger = logging.getLogger("corpus_validator")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CorpusFormat(enum.Enum):
    """Supported ground-truth corpus formats."""

    PLAIN_TEXT = "PLAIN_TEXT"
    HOCR = "HOCR"
    ALTO_XML = "ALTO_XML"
    PAGE_XML = "PAGE_XML"
    CUSTOM = "CUSTOM"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ValidationMetric:
    """A single validation metric with pass/fail threshold."""

    name: str
    value: float
    threshold: float
    passed: bool


@dataclass
class DocumentResult:
    """Validation result for a single document."""

    doc_id: str
    source_path: str
    metrics: list[ValidationMetric] = field(default_factory=list)
    overall_pass: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class CorpusReport:
    """Aggregate report for an entire corpus validation run."""

    corpus_name: str
    total_documents: int
    passed: int
    failed: int
    metrics_summary: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    results: list[DocumentResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "corpus_name": self.corpus_name,
            "total_documents": self.total_documents,
            "passed": self.passed,
            "failed": self.failed,
            "metrics_summary": self.metrics_summary,
            "duration_seconds": round(self.duration_seconds, 3),
            "results": [
                {
                    "doc_id": r.doc_id,
                    "source_path": r.source_path,
                    "metrics": [
                        {
                            "name": m.name,
                            "value": round(m.value, 6),
                            "threshold": m.threshold,
                            "passed": m.passed,
                        }
                        for m in r.metrics
                    ],
                    "overall_pass": r.overall_pass,
                    "errors": r.errors,
                }
                for r in self.results
            ],
        }

    def summary_text(self) -> str:
        """Human-readable summary string."""
        lines = [
            "=" * 60,
            "Corpus Validation Report",
            "=" * 60,
            f"  Corpus:            {self.corpus_name}",
            f"  Total documents:   {self.total_documents}",
            f"  Passed:            {self.passed}",
            f"  Failed:            {self.failed}",
            f"  Duration:          {self.duration_seconds:.3f}s",
        ]
        if self.metrics_summary:
            lines.append("  Metrics (avg):")
            for name, val in sorted(self.metrics_summary.items()):
                lines.append(f"    {name}: {val:.6f}")
        if self.results:
            failed_results = [r for r in self.results if not r.overall_pass]
            if failed_results:
                lines.append(f"  Failed documents ({len(failed_results)}):")
                for r in failed_results:
                    lines.append(f"    - {r.doc_id}")
                    for err in r.errors:
                        lines.append(f"        {err}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Text comparator
# ---------------------------------------------------------------------------


class TextComparator:
    """Utility for computing text-level OCR quality metrics."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison.

        Strips leading/trailing whitespace, collapses interior whitespace
        runs to single spaces, and applies NFC unicode normalization.
        """
        text = unicodedata.normalize("NFC", text)
        text = text.strip()
        return " ".join(text.split())

    @staticmethod
    def _levenshtein(seq1: list, seq2: list) -> int:
        """Compute Levenshtein edit distance between two sequences."""
        m, n = len(seq1), len(seq2)
        # Use single-row DP for space efficiency
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
            prev, curr = curr, prev
        return prev[n]

    @classmethod
    def character_error_rate(cls, predicted: str, ground_truth: str) -> float:
        """Compute Character Error Rate (CER).

        CER = levenshtein(predicted, ground_truth) / len(ground_truth).
        Returns 0.0 when both strings are empty.
        Returns 1.0 when ground_truth is empty but predicted is not.
        """
        if len(ground_truth) == 0:
            return 0.0 if len(predicted) == 0 else 1.0
        dist = cls._levenshtein(list(predicted), list(ground_truth))
        return dist / len(ground_truth)

    @classmethod
    def word_error_rate(cls, predicted: str, ground_truth: str) -> float:
        """Compute Word Error Rate (WER).

        WER = word_level_edit_distance / num_words_in_ground_truth.
        Returns 0.0 when both are empty.
        Returns 1.0 when ground_truth is empty but predicted is not.
        """
        gt_words = ground_truth.split()
        pred_words = predicted.split()
        if len(gt_words) == 0:
            return 0.0 if len(pred_words) == 0 else 1.0
        dist = cls._levenshtein(pred_words, gt_words)
        return dist / len(gt_words)

    @staticmethod
    def line_accuracy(predicted: str, ground_truth: str) -> float:
        """Compute fraction of lines that match exactly.

        Splits on newlines and compares positionally.  Extra or missing
        lines count as mismatches.  Returns 1.0 when both are empty.
        """
        gt_lines = ground_truth.splitlines()
        pred_lines = predicted.splitlines()
        if len(gt_lines) == 0 and len(pred_lines) == 0:
            return 1.0
        total = max(len(gt_lines), len(pred_lines))
        matches = 0
        for i in range(total):
            gt_line = gt_lines[i] if i < len(gt_lines) else None
            pred_line = pred_lines[i] if i < len(pred_lines) else None
            if gt_line == pred_line:
                matches += 1
        return matches / total


# ---------------------------------------------------------------------------
# Format extensions
# ---------------------------------------------------------------------------

_FORMAT_EXTENSIONS: dict[CorpusFormat, str] = {
    CorpusFormat.PLAIN_TEXT: ".txt",
    CorpusFormat.HOCR: ".hocr",
    CorpusFormat.ALTO_XML: ".xml",
    CorpusFormat.PAGE_XML: ".xml",
    CorpusFormat.CUSTOM: ".txt",
}


# ---------------------------------------------------------------------------
# Corpus validator
# ---------------------------------------------------------------------------


class CorpusValidator:
    """Validates OCR output quality against ground-truth document corpora.

    Compares OCR output files in *corpus_dir* against corresponding
    ground-truth files in *ground_truth_dir*.  Documents are matched by
    stem name (filename without extension).
    """

    DEFAULT_THRESHOLDS: dict[str, float] = {
        "cer": 0.05,
        "wer": 0.10,
        "line_accuracy": 0.90,
    }

    def __init__(
        self,
        corpus_dir: str,
        ground_truth_dir: str,
        format: CorpusFormat = CorpusFormat.PLAIN_TEXT,
    ) -> None:
        self.corpus_dir = corpus_dir
        self.ground_truth_dir = ground_truth_dir
        self.format = format
        self._comparator = TextComparator()

    # -- File I/O -----------------------------------------------------------

    def load_ground_truth(self, doc_id: str) -> str:
        """Load ground-truth text for *doc_id*.

        Searches *ground_truth_dir* for a file whose stem matches *doc_id*
        and whose extension corresponds to the configured format.
        """
        ext = _FORMAT_EXTENSIONS.get(self.format, ".txt")
        path = os.path.join(self.ground_truth_dir, doc_id + ext)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Ground truth not found for '{doc_id}': {path}"
            )
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def load_ocr_output(self, doc_id: str) -> str:
        """Load OCR output text for *doc_id*.

        Searches *corpus_dir* for a file whose stem matches *doc_id*.
        OCR output is always expected as ``.txt``.
        """
        path = os.path.join(self.corpus_dir, doc_id + ".txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"OCR output not found for '{doc_id}': {path}"
            )
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # -- Discovery ----------------------------------------------------------

    def discover_documents(self) -> list[str]:
        """Find document IDs by matching stems across corpus and ground-truth dirs.

        A document ID is the filename stem (without extension).  Only IDs
        that have corresponding files in *both* directories are returned,
        sorted alphabetically.
        """
        corpus_stems: set[str] = set()
        gt_ext = _FORMAT_EXTENSIONS.get(self.format, ".txt")

        if os.path.isdir(self.corpus_dir):
            for fname in os.listdir(self.corpus_dir):
                stem, ext = os.path.splitext(fname)
                if ext == ".txt":
                    corpus_stems.add(stem)

        gt_stems: set[str] = set()
        if os.path.isdir(self.ground_truth_dir):
            for fname in os.listdir(self.ground_truth_dir):
                stem, ext = os.path.splitext(fname)
                if ext == gt_ext:
                    gt_stems.add(stem)

        matched = corpus_stems & gt_stems
        return sorted(matched)

    # -- Single-document validation -----------------------------------------

    def validate_document(
        self,
        doc_id: str,
        thresholds: dict[str, float] | None = None,
    ) -> DocumentResult:
        """Validate a single document against its ground truth.

        Computes CER, WER, and line_accuracy.  Each metric is checked
        against the corresponding threshold.

        Parameters
        ----------
        doc_id:
            Document identifier (filename stem).
        thresholds:
            Optional override for :attr:`DEFAULT_THRESHOLDS`.
        """
        effective = dict(self.DEFAULT_THRESHOLDS)
        if thresholds:
            effective.update(thresholds)

        source_path = os.path.join(self.corpus_dir, doc_id + ".txt")
        result = DocumentResult(doc_id=doc_id, source_path=source_path)

        try:
            gt_text = self.load_ground_truth(doc_id)
            ocr_text = self.load_ocr_output(doc_id)
        except FileNotFoundError as exc:
            result.errors.append(str(exc))
            result.overall_pass = False
            return result

        # Normalize
        gt_norm = self._comparator.normalize_text(gt_text)
        ocr_norm = self._comparator.normalize_text(ocr_text)

        # CER
        cer = self._comparator.character_error_rate(ocr_norm, gt_norm)
        cer_thresh = effective.get("cer", 0.05)
        cer_passed = cer <= cer_thresh
        result.metrics.append(
            ValidationMetric(name="cer", value=cer, threshold=cer_thresh, passed=cer_passed)
        )

        # WER
        wer = self._comparator.word_error_rate(ocr_norm, gt_norm)
        wer_thresh = effective.get("wer", 0.10)
        wer_passed = wer <= wer_thresh
        result.metrics.append(
            ValidationMetric(name="wer", value=wer, threshold=wer_thresh, passed=wer_passed)
        )

        # Line accuracy (computed on un-collapsed text to preserve line structure)
        gt_stripped = gt_text.strip()
        ocr_stripped = ocr_text.strip()
        la = self._comparator.line_accuracy(ocr_stripped, gt_stripped)
        la_thresh = effective.get("line_accuracy", 0.90)
        la_passed = la >= la_thresh
        result.metrics.append(
            ValidationMetric(name="line_accuracy", value=la, threshold=la_thresh, passed=la_passed)
        )

        result.overall_pass = all(m.passed for m in result.metrics)

        if not result.overall_pass:
            for m in result.metrics:
                if not m.passed:
                    if m.name == "line_accuracy":
                        result.errors.append(
                            f"{m.name}={m.value:.4f} below threshold {m.threshold}"
                        )
                    else:
                        result.errors.append(
                            f"{m.name}={m.value:.4f} exceeds threshold {m.threshold}"
                        )

        return result

    # -- Full-corpus validation ---------------------------------------------

    def validate_corpus(
        self,
        thresholds: dict[str, float] | None = None,
    ) -> CorpusReport:
        """Validate all discovered documents and produce a corpus report.

        Parameters
        ----------
        thresholds:
            Optional override for :attr:`DEFAULT_THRESHOLDS`.
        """
        start = time.monotonic()

        doc_ids = self.discover_documents()
        results: list[DocumentResult] = []

        for doc_id in doc_ids:
            result = self.validate_document(doc_id, thresholds)
            results.append(result)

        elapsed = time.monotonic() - start

        passed = sum(1 for r in results if r.overall_pass)
        failed = len(results) - passed

        # Compute average per metric
        metrics_summary: dict[str, float] = {}
        if results:
            metric_names: set[str] = set()
            for r in results:
                for m in r.metrics:
                    metric_names.add(m.name)
            for name in sorted(metric_names):
                values = [m.value for r in results for m in r.metrics if m.name == name]
                if values:
                    metrics_summary[name] = sum(values) / len(values)

        corpus_name = os.path.basename(os.path.normpath(self.corpus_dir)) if self.corpus_dir else "unknown"

        return CorpusReport(
            corpus_name=corpus_name,
            total_documents=len(results),
            passed=passed,
            failed=failed,
            metrics_summary=metrics_summary,
            duration_seconds=elapsed,
            results=results,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the corpus validator CLI."""
    parser = argparse.ArgumentParser(
        description="Validate OCR output quality against ground-truth corpora",
    )
    parser.add_argument(
        "--corpus-dir",
        required=True,
        help="Directory containing OCR output text files",
    )
    parser.add_argument(
        "--ground-truth-dir",
        required=True,
        help="Directory containing ground-truth files",
    )
    parser.add_argument(
        "--format",
        default="PLAIN_TEXT",
        choices=[f.value for f in CorpusFormat],
        help="Ground-truth format (default: PLAIN_TEXT)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output report as JSON instead of human-readable text",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help='JSON string of thresholds, e.g. \'{"cer": 0.03, "wer": 0.08}\'',
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Validate a single document by ID instead of the whole corpus",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns 0 on success, 1 on failure."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    corpus_format = CorpusFormat(args.format)

    thresholds: dict[str, float] | None = None
    if args.thresholds:
        try:
            thresholds = json.loads(args.thresholds)
        except json.JSONDecodeError as exc:
            print(f"ERROR: Invalid --thresholds JSON: {exc}", file=sys.stderr)
            return 1

    validator = CorpusValidator(
        corpus_dir=args.corpus_dir,
        ground_truth_dir=args.ground_truth_dir,
        format=corpus_format,
    )

    # -- Single document ----------------------------------------------------
    if args.doc_id:
        result = validator.validate_document(args.doc_id, thresholds)
        if args.json:
            report = CorpusReport(
                corpus_name=os.path.basename(os.path.normpath(args.corpus_dir)),
                total_documents=1,
                passed=1 if result.overall_pass else 0,
                failed=0 if result.overall_pass else 1,
                results=[result],
            )
            print(json.dumps(report.to_dict(), indent=2))
        else:
            status = "PASS" if result.overall_pass else "FAIL"
            print(f"Document: {result.doc_id}  [{status}]")
            for m in result.metrics:
                flag = "✓" if m.passed else "✗"
                print(f"  {flag} {m.name}: {m.value:.6f} (threshold: {m.threshold})")
            for err in result.errors:
                print(f"  ERROR: {err}")
        return 0 if result.overall_pass else 1

    # -- Full corpus --------------------------------------------------------
    report = validator.validate_corpus(thresholds)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary_text())

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

# Signature Verification Corpus Validation

## Overview

The signature verification module (`signature_verification.py`) provides **experimental, advisory-only** signature detection. It detects the *presence* of ink in signature-designated areas and flags suspicious patterns for human review. It never asserts that a signature is authentic.

This document describes how to validate the module's false-positive (FP) and false-negative (FN) rates using a labeled corpus of images.

## Preparing a Validation Corpus

### Directory Structure

The benchmark script expects images organized into two directories:

```
corpus/
  signatures/       # Images known to contain handwritten signatures
  no_signatures/    # Images known to NOT contain signatures (blank fields, typed text)
```

### Image Requirements

- **Supported formats**: PNG, JPEG, TIFF, BMP, GIF, WebP
- **Recommended resolution**: 200-400 DPI (matching pipeline operating range)
- **Content guidance**:
  - `signatures/` should contain scanned pages or crops where a genuine handwritten signature is present
  - `no_signatures/` should contain blank signature lines, typed-only pages, or pages without any signature field

### Corpus Size Recommendations

| Corpus Size | Use Case |
|-------------|----------|
| 20-50 images | Quick sanity check, development iteration |
| 100-200 images | Baseline validation, CI gate |
| 500+ images | Production validation, FP/FN bound establishment |

For statistically meaningful FP/FN bounds, aim for at least 100 images per class (200 total).

### Labeling Guidelines

1. Each image goes into exactly one directory based on ground truth
2. If an image contains **any** handwritten signature, it belongs in `signatures/`
3. Typed or machine-printed "signatures" (e.g., "/s/ John Doe") belong in `no_signatures/` since the module specifically targets handwritten ink
4. Ambiguous cases should be documented separately and excluded from the corpus

## Running the Benchmark

### Synthetic Mode (No Corpus Needed)

For CI testing and development, use synthetic mode which generates images programmatically:

```bash
# Default run (50 synthetic images)
python scripts/benchmark_signature_verification.py --synthetic

# Larger sample
python scripts/benchmark_signature_verification.py --synthetic --sample-size 200

# With JSON output
python scripts/benchmark_signature_verification.py --synthetic --output results.json

# With markdown report
python scripts/benchmark_signature_verification.py --synthetic --output-md report.md
```

### Corpus Mode

```bash
# Run against a labeled corpus
python scripts/benchmark_signature_verification.py --corpus-dir /path/to/corpus

# Limit sample size per class
python scripts/benchmark_signature_verification.py --corpus-dir /path/to/corpus --sample-size 100

# Custom targets
python scripts/benchmark_signature_verification.py --corpus-dir /path/to/corpus \
  --target-precision 80.0 --target-recall 75.0
```

### Dry-Run Mode

Validates corpus structure without running detection:

```bash
python scripts/benchmark_signature_verification.py --corpus-dir /path/to/corpus --dry-run
```

This outputs a JSON report showing:
- Whether required directories exist
- Image counts per directory
- Any structural issues found

### CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--corpus-dir PATH` | Path to labeled corpus directory | (none) |
| `--synthetic` | Use synthetic images instead of corpus | false |
| `--dry-run` | Validate corpus structure only | false |
| `--sample-size N` | Images to benchmark | 50 |
| `--output PATH` | JSON output path | (none) |
| `--output-md PATH` | Markdown output path | (none) |
| `--target-precision N` | Target precision % for pass/fail | 70.0 |
| `--target-recall N` | Target recall % for pass/fail | 70.0 |
| `-v, --verbose` | Enable debug logging | false |

## Understanding the Metrics

### Core Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Precision** | TP / (TP + FP) | Of all detections, how many are real signatures |
| **Recall** | TP / (TP + FN) | Of all real signatures, how many are detected |
| **F1** | Harmonic mean of precision and recall | Balanced detection quality |
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **FP Rate** | FP / (FP + TN) | Rate of falsely flagging clean images |
| **FN Rate** | FN / (FN + TP) | Rate of missing real signatures |

### Confusion Matrix

```
                    Predicted Positive    Predicted Negative
Actual Positive     True Positive (TP)    False Negative (FN)
Actual Negative     False Positive (FP)   True Negative (TN)
```

## Acceptable FP/FN Bounds

### Context: Advisory-Only Detection

The signature verification module is **experimental and advisory-only**. Its outputs are analyst triage signals, not forensic conclusions. This fundamentally changes what "acceptable" error rates mean:

- **False positives** (flagging a clean page as having a signature) create extra review work but do not produce incorrect forensic conclusions
- **False negatives** (missing a real signature) are more concerning in forensic contexts but the module is a supplementary signal, not the sole detection mechanism

### Recommended Bounds

| Metric | Advisory-Only Target | Notes |
|--------|---------------------|-------|
| Precision | >= 70% | Acceptable FP overhead for triage workflow |
| Recall | >= 70% | Catches majority of signatures for review |
| FP Rate | <= 30% | At most 30% of clean pages falsely flagged |
| FN Rate | <= 30% | At most 30% of real signatures missed |

These targets are intentionally conservative. For a production forensic pipeline where signature detection is the primary gate, tighter bounds (precision >= 90%, recall >= 85%) would be appropriate. The current module is supplementary, so broader bounds are acceptable.

### Factors Affecting Accuracy

1. **Image quality**: Low-DPI scans or heavy compression degrade ink detection
2. **Signature style**: Very light or very small signatures may be missed
3. **OCR keyword availability**: Detection relies on OCR lines containing "signature", "sign here", etc. Without these keywords, detection falls back to form field metadata
4. **Form field metadata**: When structure data includes `field_type: "signature"`, detection accuracy improves significantly
5. **Typed vs. handwritten**: The module distinguishes typed text from ink strokes; typed names in signature fields are correctly flagged as `review_required`

### Interpreting Synthetic Results

Synthetic benchmarks use programmatically generated images and injected OCR keywords. They establish a **lower bound** on detection capability under controlled conditions. Real-world accuracy depends on corpus diversity and document quality.

## Current Experimental Status

### What the Module Does

1. **Signature presence detection**: Analyzes ink density, stroke complexity, and spatial spread in signature-designated regions
2. **Authenticity review signals**: Flags suspicious patterns (typed text where ink expected, low stroke complexity) as `review_required`
3. **Keyword-based region projection**: Uses OCR text containing signature keywords to project candidate signature regions

### What the Module Does NOT Do

1. Does not verify that a signature is authentic
2. Does not compare signatures against reference exemplars for matching
3. Does not provide forensic-grade signature authentication
4. Does not replace human analyst review

### Limitations

- Detection depends on OCR keyword presence or form field metadata; images without contextual text may produce false negatives
- Very faint signatures (light pencil, low-contrast ink) may fall below ink density thresholds
- The module uses heuristic thresholds that may need tuning for specific document types
- Synthetic benchmark results do not directly predict real-world performance

### Roadmap

- Broader corpus validation with diverse document types
- Threshold tuning based on production corpus data
- Optional reference-based signature similarity scoring (separate from presence detection)
- Integration with document classification for context-aware detection

## Output Files

### JSON Report

```json
{
  "total_images": 100,
  "true_positives": 40,
  "false_positives": 5,
  "true_negatives": 45,
  "false_negatives": 10,
  "precision": 0.8889,
  "recall": 0.8,
  "f1": 0.8421,
  "accuracy": 0.85,
  "fp_rate": 0.1,
  "fn_rate": 0.2,
  "dataset": "/path/to/corpus",
  "passed": true,
  "timestamp": "2026-03-12T10:30:00.000+00:00"
}
```

### Markdown Report

Generated via `--output-md`, includes:
- Summary table with all metrics
- Confusion matrix
- Inference timing statistics
- Notes and experimental status disclaimer

# CJK Vertical Text Validation

This document describes how to prepare a validation corpus for the CJK
vertical text detection module (`vertical_text.py`) and how to run the
benchmark tool.

## Background

The vertical text module detects CJK text direction (horizontal, vertical,
mixed) from OCR bounding box geometry. It groups vertical text into columns,
sorts them in right-to-left reading order, and interleaves vertical and
horizontal regions for mixed-layout pages.

Validation requires a labeled corpus where the expected text direction,
reading order, and column count are specified for each page.

## Corpus Structure

Organize test data into three subdirectories:

```
corpus/
  vertical_only/     # Pages with pure vertical CJK text
  mixed_layout/      # Pages with mixed vertical + horizontal text
  horizontal_only/   # Pages with only horizontal text (negative cases)
```

Each test case consists of:
- An image file (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp`)
- A sidecar JSON file with the same stem and `.expected.json` suffix

Example:
```
corpus/
  vertical_only/
    newspaper_col3.png
    newspaper_col3.expected.json
    poetry_vertical.tiff
    poetry_vertical.expected.json
  mixed_layout/
    brochure_jp.png
    brochure_jp.expected.json
  horizontal_only/
    english_letter.png
    english_letter.expected.json
```

## Sidecar JSON Format

Each `.expected.json` file provides the ground truth for one test case:

```json
{
  "direction": "vertical",
  "expected_order": [
    "right column line 1",
    "right column line 2",
    "middle column line 1",
    "middle column line 2",
    "left column line 1",
    "left column line 2"
  ],
  "expected_columns": 3,
  "ocr_lines": [
    {
      "text": "right column line 1",
      "box": [[770, 50], [800, 50], [800, 250], [770, 250]],
      "confidence": 0.95
    },
    {
      "text": "right column line 2",
      "box": [[770, 270], [800, 270], [800, 470], [770, 470]],
      "confidence": 0.93
    }
  ],
  "page_width": 1000
}
```

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `direction` | string | Yes | Expected page direction: `"vertical"`, `"horizontal"`, or `"mixed"` |
| `expected_order` | list[string] | Yes | Text items in correct reading order |
| `expected_columns` | int | No | Expected number of vertical columns (0 for horizontal pages) |
| `ocr_lines` | list[object] | Yes | Simulated OCR output lines |
| `ocr_lines[].text` | string | Yes | Text content of the line |
| `ocr_lines[].box` | list[list[int]] | Yes | 4-point bounding box polygon `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]` |
| `ocr_lines[].confidence` | float | No | OCR confidence score (default 0.9) |
| `page_width` | float | Yes | Page width in pixels |

### Bounding Box Format

Bounding boxes use the PaddleOCR 4-point polygon format:

```
[top-left, top-right, bottom-right, bottom-left]
```

For a vertical text box with width 30 and height 200 at position (100, 50):
```json
[[100, 50], [130, 50], [130, 250], [100, 250]]
```

Vertical text boxes have a height-to-width aspect ratio >= 2.0. Horizontal
text boxes have a lower aspect ratio.

### Direction Categories

- **vertical_only**: Pages where >= 60% of text boxes are vertical. Expected
  direction is `"vertical"`.
- **mixed_layout**: Pages where 20-59% of text boxes are vertical. Expected
  direction is `"mixed"`.
- **horizontal_only**: Pages where < 20% of text boxes are vertical. Expected
  direction is `"horizontal"`.

## Running the Benchmark

### Dry-Run Mode (Validate Corpus)

Validate corpus structure without running detection:

```bash
python scripts/benchmark_vertical_text.py --corpus-dir /path/to/corpus --dry-run
```

This checks that:
- The corpus directory exists
- At least one recognized subdirectory is present
- Image files have matching `.expected.json` sidecars
- Sidecar JSON is valid and contains required fields

### Corpus Benchmark

Run the full benchmark against a labeled corpus:

```bash
python scripts/benchmark_vertical_text.py --corpus-dir /path/to/corpus
```

Options:
- `--target-direction-accuracy 90.0` -- Target direction accuracy percentage
- `--target-ordering-tau 0.80` -- Target Kendall's tau for reading order
- `--output results.json` -- Save structured JSON results
- `--output-md report.md` -- Save markdown report
- `-v` -- Enable verbose logging

### Synthetic Benchmark (No Corpus Required)

Run with generated test data for CI/testing:

```bash
python scripts/benchmark_vertical_text.py --synthetic
python scripts/benchmark_vertical_text.py --synthetic --sample-size 50
python scripts/benchmark_vertical_text.py --synthetic --output results.json
```

The synthetic mode generates pages with known layouts and orderings:
- 40% vertical-only pages (2-5 columns, 2-5 lines per column)
- 30% mixed-layout pages (1-3 vertical columns + 2-4 horizontal lines)
- 30% horizontal-only pages (3-8 lines, negative cases)

## Metrics

### Direction Accuracy

Percentage of pages where the predicted direction (`horizontal`, `vertical`,
`mixed`) matches the expected direction.

### Kendall's Tau

A rank correlation coefficient measuring reading order accuracy. Values range
from -1.0 (perfectly inverted) to 1.0 (perfect agreement). Only computed for
items common to both predicted and expected orderings.

Interpretation:
- 1.0: Perfect reading order
- 0.8+: Strong agreement (target)
- 0.6-0.8: Moderate agreement
- < 0.6: Poor agreement

### Column Grouping Precision

Measures how closely the detected column count matches the expected count.
Returns 1.0 for exact match, with proportional reduction for mismatches.

## Current Limitations

1. **Mixed-layout interleaving**: Pages with overlapping vertical and
   horizontal regions are merged by Y-position. This can produce suboptimal
   ordering when vertical columns span the same Y-range as horizontal text.

2. **Column grouping tolerance**: The tolerance is a fixed fraction of page
   width (`COLUMN_GROUPING_TOLERANCE=0.05`). Pages with unusual column
   spacing may be over- or under-segmented.

3. **Single-character exclusion**: Boxes containing a single character are
   excluded from vertical detection by default (`min_chars=2`) because they
   are geometrically ambiguous.

4. **No content-based detection**: The module uses only bounding box geometry,
   not character content. CJK Unicode detection (`contains_cjk`) is available
   but not used as a primary detection signal.

5. **Aspect ratio threshold**: The vertical classification threshold
   (`VERTICAL_ASPECT_RATIO_THRESHOLD=2.0`) is fixed. Documents with compact
   vertical text may not reach this threshold.

## Preparing a Real Corpus

To create a validation corpus from real CJK documents:

1. **Collect representative pages**: Gather scanned pages from newspapers,
   books, legal documents, and mixed-layout materials in Japanese, Chinese,
   and Korean.

2. **Run OCR**: Process each page through PaddleOCR to obtain bounding box
   coordinates and text content.

3. **Label ground truth**: Manually determine the correct reading order and
   expected direction for each page. Record column counts for vertical pages.

4. **Create sidecar files**: Write `.expected.json` files with the OCR output
   lines and ground truth labels.

5. **Validate**: Run `--dry-run` to verify the corpus structure before
   benchmarking.

Recommended corpus size:
- Minimum: 10 pages per category (30 total)
- Recommended: 50+ pages per category (150+ total)
- Target distribution: diverse sources, varying column counts, mixed scripts

## Example Output

### JSON Report

```json
{
  "total_cases": 30,
  "direction_accuracy": 93.33,
  "avg_kendall_tau": 0.8721,
  "avg_column_precision": 0.9500,
  "avg_inference_time_ms": 0.15,
  "per_category": {
    "vertical_only": {
      "category": "vertical_only",
      "total_cases": 12,
      "direction_correct": 12,
      "direction_accuracy": 100.0,
      "avg_kendall_tau": 0.9200,
      "avg_column_precision": 1.0
    },
    "mixed_layout": {
      "category": "mixed_layout",
      "total_cases": 9,
      "direction_correct": 7,
      "direction_accuracy": 77.78,
      "avg_kendall_tau": 0.7800,
      "avg_column_precision": 0.8500
    },
    "horizontal_only": {
      "category": "horizontal_only",
      "total_cases": 9,
      "direction_correct": 9,
      "direction_accuracy": 100.0,
      "avg_kendall_tau": 0.0,
      "avg_column_precision": 0.0
    }
  },
  "passed": true
}
```

### Console Report

```
================================================================================
CJK VERTICAL TEXT BENCHMARK
================================================================================

  Dataset:              synthetic
  Total Cases:          30
  Direction Accuracy:   93.33%
  Avg Kendall's Tau:    0.8721
  Avg Column Precision: 0.9500
  Target Dir. Acc:      90.0%
  Target Ordering Tau:  0.80
  Result:               PASS
  Avg Inference:        0.15 ms
  P95 Inference:        0.42 ms

PER-CATEGORY METRICS
--------------------------------------------------------------------------------
Category             Cases    Dir.Acc        Tau    ColPrec   Errors
--------------------------------------------------------------------------------
vertical_only           12     100.00     0.9200     1.0000        0
mixed_layout             9      77.78     0.7800     0.8500        0
horizontal_only          9     100.00     0.0000     0.0000        0
--------------------------------------------------------------------------------
```

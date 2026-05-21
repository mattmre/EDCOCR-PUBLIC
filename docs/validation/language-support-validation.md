# Language Support Validation

Runtime validation framework for verifying the complete language stack across all 45 supported OCR languages.

## Overview

The `scripts/validate_language_support.py` script validates configuration integrity, font resolution, PaddleOCR model availability, and Tesseract language data for every language in the EDCOCR registry. It operates in two modes:

1. **Config validation** (always runs) -- checks that every language entry is well-formed and complete in the `language_config.py` registry. This works in any environment without external dependencies.

2. **Disk checks** (opt-in) -- verifies that font files, PaddleOCR model directories, and Tesseract trained data files actually exist on the local filesystem. Use `--check-fonts` and `--check-models` to enable these.

## Quick Start

```bash
# Validate all 45 languages (config only, works anywhere)
python scripts/validate_language_support.py

# Validate only the 11 extended-tier languages
python scripts/validate_language_support.py --tier extended

# Full validation with disk checks (Docker or production environment)
python scripts/validate_language_support.py --tier all --check-fonts --check-models

# Generate JSON and markdown reports
python scripts/validate_language_support.py --output-json report.json --output-md report.md
```

## What Each Check Validates

### Config Validation (always runs)

| Check | Description | Failure Condition |
|-------|-------------|-------------------|
| Name | Human-readable language name exists | Empty or missing name |
| FastText codes | At least one FastText language code mapped | Empty fasttext_codes tuple |
| Script | Script family is a recognized value | Script not in valid set (latin, cyrillic, cjk, arabic, etc.) |
| Tier | Tier is "core" or "extended" | Tier not in valid set |
| Font mapping | A font filename is assigned | Empty font string |
| Tesseract code | Tesseract language code exists (reported, not required) | Empty string (advisory) |
| EasyOCR code | EasyOCR language code exists (reported, not required) | Empty string (advisory) |

Config validation determines the exit code: exit 0 if all pass, exit 1 if any fail.

### Font File Check (`--check-fonts`)

Verifies that the Noto Sans font file assigned to each language actually exists at the configured `NOTO_FONT_DIR` path (default: `/app/fonts/noto`). Reports which fonts are present and which are missing.

Override the font directory with the `NOTO_FONT_DIR` environment variable:

```bash
NOTO_FONT_DIR=/usr/share/fonts/noto python scripts/validate_language_support.py --check-fonts
```

### Model Directory Check (`--check-models`)

Verifies two things:

1. **PaddleOCR models**: Scans `PADDLEOCR_HOME/whl/` (default: `~/.paddleocr/whl/`) for directories matching each language's paddle code.

2. **Tesseract trained data**: Checks `TESSDATA_PREFIX` (default: `/usr/share/tesseract-ocr/4.00/tessdata/`) for `<tesseract_code>.traineddata` files.

Override paths with environment variables:

```bash
PADDLEOCR_HOME=/custom/models TESSDATA_PREFIX=/custom/tessdata \
    python scripts/validate_language_support.py --check-models
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tier` | `all` | Which tier(s) to validate: `core`, `extended`, or `all` |
| `--check-fonts` | off | Verify font files exist on disk |
| `--check-models` | off | Verify PaddleOCR model dirs and Tesseract data on disk |
| `--output-json <path>` | none | Write structured JSON report |
| `--output-md <path>` | none | Write markdown report |
| `-v`, `--verbose` | off | Enable debug-level logging |

## Tier System

EDCOCR organizes its 45 supported languages into two tiers, controlled by the `OCR_LANGUAGE_TIERS` environment variable:

### Core Tier (34 languages, default)

The production baseline. These models are pre-baked into the Docker image and available for air-gapped deployment.

- **Latin**: en, fr, german, es, it, pt, nl, sv, da, fi, ro, pl, cs, hu, tr, vi
- **Cyrillic**: ru, uk, be, bg
- **CJK**: ch, chinese_cht, japan, korean
- **Arabic/RTL**: ar, fa, ur, ug
- **Indic**: hi, ta, te, kn
- **Other**: el (Greek), ka (Georgian)

### Extended Tier (+11 languages, opt-in)

Enabled by setting `OCR_LANGUAGE_TIERS=core,extended`:

- **Latin**: hr, sk, no, lt, lv, et, rs_latin
- **Indic**: bn (Bengali), mr (Marathi), ne (Nepali)
- **Southeast Asian**: th (Thai)

## Adding a New Language

To add a new language to the registry:

1. **Add the entry in `language_config.py`**:
   ```python
   _reg(LanguageEntry(
       "new_code",           # PaddleOCR model code
       "New Language",       # Human-readable name
       ("xx"),              # FastText lid.176.bin codes
       "latin",              # Script family
       "extended",           # Tier (usually "extended" for new additions)
       "xxx",                # Tesseract language code
       "xx",                 # EasyOCR language code
       "NotoSans-Regular.ttf",  # Font file (use script-appropriate Noto font)
   ))
   ```

2. **Run the validation script** to confirm the entry is well-formed:
   ```bash
   python scripts/validate_language_support.py --tier extended
   ```

3. **Add the font** to the Docker image if a new font file is needed (update `Dockerfile`).

4. **Rebuild the Docker image** to download the PaddleOCR model:
   ```bash
   docker-compose build
   ```

5. **Run full validation** in the Docker container:
   ```bash
   docker exec -it ocr_gpu_processor python scripts/validate_language_support.py \
       --tier all --check-fonts --check-models
   ```

## Report Formats

### Console Output

The default console output shows a summary table followed by per-language details:

```
================================================================================
LANGUAGE SUPPORT VALIDATION REPORT
================================================================================

  Timestamp:        2026-03-12T14:30:00+00:00
  Tiers checked:    core, extended
  Total languages:  45
  Config PASS:      45
  Config FAIL:      0
  Overall:          PASS

LANGUAGE DETAILS
--------------------------------------------------------------------------------
Code           Name                 Tier      Script       Config
--------------------------------------------------------------------------------
ar             Arabic               core      arabic       OK
...
```

### JSON Report (`--output-json`)

Structured JSON with full details for each language, suitable for CI integration:

```json
{
  "timestamp": "2026-03-12T14:30:00+00:00",
  "tiers_checked": ["core", "extended"],
  "total_languages": 45,
  "config_pass": 45,
  "config_fail": 0,
  "all_config_valid": true,
  "languages": [
    {
      "paddle_code": "ar",
      "name": "Arabic",
      "tier": "core",
      "script": "arabic",
      "config_valid": true,
      "font_filename": "NotoSansArabic-Regular.ttf",
      "font_file_exists": null,
      "model_dir_exists": null,
      ...
    }
  ]
}
```

### Markdown Report (`--output-md`)

Formatted markdown table suitable for documentation or PR descriptions.

## CI Integration

Add to your CI pipeline to catch language config regressions:

```yaml
- name: Validate language support
  run: python scripts/validate_language_support.py --tier all
```

For Docker-based validation with full disk checks:

```yaml
- name: Validate language support (full)
  run: |
    docker exec ocr_gpu_processor python scripts/validate_language_support.py \
        --tier all --check-fonts --check-models \
        --output-json /app/ocr_output/language-validation.json
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All config-level checks pass for the selected tier(s) |
| 1 | At least one config-level check failed |

Disk checks (fonts, models) are informational and do not affect the exit code. They report what is available for operational awareness.

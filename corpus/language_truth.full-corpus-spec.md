#  E-A-010 Full Language Corpus Specification

`corpus/language_truth.json` currently freezes the repo-local bootstrap OCR
fixtures. It is intentionally not the full E-A-010 corpus.

## Full Gate Requirement

The full release-gate gate requires at least 50 hand-curated documents with:

- document-level expected primary language
- per-page expected primary language
- source fixture path
- fixture SHA-256 captured in `corpus/language_truth.sha256`
- explicit coverage of:
  - monolingual Latin-script documents
  - mixed CJK page
  - RTL + LTR mixed document
  - short/noisy low-confidence document that should produce `und`
  - extended-tier language lockout case
  - bilingual/multilingual page transitions
  - degraded/low-resolution scans
  - handwriting/degraded text where available

## Current Local State

The repo contains 5 frozen bootstrap entries. The manifest gate is useful for
tamper-evidence and reproducibility, but the full corpus-data requirement is
blocked until the missing curated fixtures are added.

## Full Gate Command

```powershell
python scripts/check_language_corpus_manifest.py --require-full-gate
```

Expected current result: fail with a message stating that at least 50 documents
are required.

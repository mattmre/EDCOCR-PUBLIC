# Plan A E-A-010 Full Language Corpus Specification

`corpus/language_truth.json` currently has 50 rows, but it does not satisfy the
full E-A-010 corpus gate. A row count alone is not sufficient evidence for
language coverage or fixture truth.

## Full Gate Requirement

The full maintenance review E-A-010 gate requires at least 50 hand-curated documents with:

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

The 50 rows currently contain five repo-local OCR fixtures and 45
FLORES-derived single-sentence text fixtures, primarily English, French, and
Spanish, plus one mixed-CJK case. Several required categories are absent:

- no RTL + LTR mixed document
- no short/noisy low-confidence document labeled `und`
- no extended-tier lockout case
- no bilingual page-to-page transition

The full-gate checker also finds five per-document fixture hashes that do not
match their source files. `sample-lowres` is labeled English in the truth file,
while its text contains German, Greek, and Turkish. The FLORES fixture files
include English attribution metadata before the target sentence; that metadata
must not be included in a language-detector accuracy sample.

The checker validates coverage classes and each document's `source_sha256`.
The checked-in `corpus/language_truth.sha256` has 56 entries; the current audit
finds that all 56 entry hashes and the composite do not match the truth file
and the 55 included OCR/FLORES fixture files. The `--require-full-gate` path
now reports the five stale document hashes and four missing coverage classes
before manifest comparison; an ordinary manifest check separately reports
the stale 56-entry manifest. Evidence is retained in
`docs/release-candidate.md`
and `language-corpus-full-gate-content-check-20260923.log` in that evidence
directory, with the hardened full-gate output in
`language-corpus-full-gate-post-hardening-20260923.log`.

The canonical corpus remains unchanged until its curated truth is reviewed and
explicitly approved for re-freezing. The status field inside
`language_truth.json` is not gate evidence.

## Full Gate Command

```powershell
python scripts/check_language_corpus_manifest.py --require-full-gate
```

Expected current result: fail on five stale document fixture hashes and four
missing coverage classes, even though the file has 50 rows. A failed
`--update --require-full-gate` check does not write a manifest. After curation,
obtain approval by setting
`CORPUS_GOLDEN_SET_UPDATE_APPROVED=true` for `--update`, review the new truth and
fixture hashes, then rerun `--require-full-gate`.

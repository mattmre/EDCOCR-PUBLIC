# ADR: fasttext Dependency Assessment

**Status**: Accepted
**Date**: 2026-04-07
**Finding**: (Expert review)

## Context

The EDCOCR pipeline uses Meta's `fasttext` library (v0.9.3) for language
identification via the `lid.176.bin` model. This model powers the adaptive
language detection loop: FastText identifies the document language, then the
pipeline selects the correct PaddleOCR recognition model for that language.

Language detection accuracy is critical to forensic-grade OCR because selecting
the wrong recognition model produces garbled text with no error signal -- the
CTC decoder simply emits wrong characters confidently.

## Current State

- **Package**: `fasttext==0.9.3` (PyPI)
- **Model**: `lid.176.bin` (176-language identification, ~126 MB)
- **Upstream**: Meta (formerly Facebook Research) ended active development of
  fasttext in 2023. The GitHub repository receives no new commits.
- **Build system**: C++ extension compiled via `pybind11`. Requires a working
  C++ compiler at install time. Pre-built wheels exist for Python 3.8-3.11
  on major platforms but are absent for Python 3.12+.
- **Integration points**: `load_fasttext` in `ocr_gpu_async.py` (line ~1169),
  `LanguageDetector` in `ocr_distributed/language.py`

## Risk Assessment

| Risk | Severity | Likelihood | Notes |
|------|----------|------------|-------|
| Build failure on Python 3.12+ | High | Confirmed | No official wheels; C++ build requires manual intervention |
| No security patches | Medium | Ongoing | Unmaintained upstream; any CVE stays unpatched |
| Wheel availability | Medium | Likely | PyPI wheels lag behind Python releases |
| Model accuracy regression | Low | Unlikely | Model is frozen; accuracy cannot regress |
| License risk | Low | Unlikely | MIT license; no known encumbrances |

### Mitigating Factors

1. **Docker-first deployment**: Production runs inside Docker containers with
   Python 3.10, where pre-built wheels are available. The build fragility only
   affects local development on newer Python versions.
2. **Air-gapped model baking**: The `lid.176.bin` model is baked into Docker
   images at build time via `Dockerfile ADD`. No runtime download required.
3. **Pinned version**: `fasttext==0.9.3` is pinned in `requirements.txt`,
   preventing accidental upgrades that might change behavior.

## Alternatives Assessed

### 1. fasttext-langdetect

- **What**: Unofficial community fork that ships pre-built wheels for more
  Python versions and wraps the same `lid.176.bin` model.
- **Pros**: Easier installation; same model quality; drop-in replacement.
- **Cons**: Unofficial maintainer; no guarantee of long-term support;
  introduces a new trust boundary (different PyPI publisher).
- **API compatibility**: Different API surface (`detect` vs `predict`);
  requires wrapper changes in `load_fasttext` and `LanguageDetector`.
- **Verdict**: Strong candidate for migration. Same model accuracy,
  better wheel coverage.

### 2. langdetect

- **What**: Pure Python port of Google's language-detection library.
- **Pros**: No C++ build; works on any Python version; well-maintained.
- **Cons**: Significantly less accurate on short text (< 50 characters);
  non-deterministic by default (uses random seed internally); slower
  inference.
- **Verdict**: Not suitable for forensic use. Accuracy gap on short OCR
  fragments is unacceptable.

### 3. lingua-py

- **What**: Modern language detector backed by Rust via PyO3.
- **Pros**: Good accuracy; active maintenance; pre-built wheels; supports
  140+ languages; deterministic results.
- **Cons**: Different accuracy profile than fasttext on CJK short text;
  model is embedded in the wheel (larger install size); would require
  comprehensive regression testing against the 34-language corpus.
- **Verdict**: Viable long-term candidate. Requires accuracy benchmarking
  before adoption.

### 4. pycld3 (Google CLD3)

- **What**: Python bindings for Google's Compact Language Detector v3.
- **Pros**: Neural network model; good accuracy; Google-maintained.
- **Cons**: C++ build dependency (protobuf); sparse PyPI wheel coverage;
  last significant update was 2021; limited language set compared to
  fasttext's 176.
- **Verdict**: Similar maintenance risk to fasttext. No clear advantage.

## Decision

**Retain `fasttext==0.9.3` for now.** The combination of frozen model quality,
Docker-based deployment (bypassing build issues), and the forensic accuracy
requirement makes a migration non-urgent.

### Actions Taken

1. Pinned `fasttext==0.9.3` explicitly in `requirements.txt` with a
   `TECH-DEBT` comment linking to this ADR.
2. Documented this assessment for future reference.

### Migration Track

- **Target**: Q3 2026 (next review: 2026-09-01)
- **Trigger**: Any of the following accelerates the timeline:
  - Python 3.12+ becomes the Docker base image
  - A CVE is discovered in fasttext's C++ layer
  - `fasttext-langdetect` demonstrates equivalent accuracy in testing
- **Migration plan**: Replace `fasttext` with `fasttext-langdetect` (same
  model, better wheels). Requires updating `load_fasttext` and
  `LanguageDetector` API calls, plus regression testing on the 34-language
  corpus.

## References

- fasttext GitHub: https://github.com/facebookresearch/fastText (archived)
- lid.176.bin model: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
- fasttext-langdetect: https://pypi.org/project/fasttext-langdetect/
- lingua-py: https://github.com/pemistahl/lingua-py
- Integration: `ocr_gpu_async.py` line ~1169, `ocr_distributed/language.py`

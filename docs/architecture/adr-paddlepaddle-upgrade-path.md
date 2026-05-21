# ADR: PaddlePaddle Upgrade Path

**Status**: Accepted
**Date**: 2026-04-07
**Finding**: (Expert review)

## Context

The EDCOCR pipeline is built on PaddleOCR 2.9.1, which depends on
PaddlePaddle 2.6.2 as its deep learning framework. PaddlePaddle is currently
at version 3.x upstream, placing the project two major versions behind.

This document explains why the gap exists, what blocks closing it, and the
planned timeline for qualification.

## Current State

| Component | Pinned Version | Latest Upstream | Gap |
|-----------|---------------|-----------------|-----|
| PaddlePaddle | 2.6.2 | 3.x | 1 major |
| PaddleOCR | 2.9.1 | 2.10+ / 3.x | 1 minor+ |
| PaddleNLP | 2.8.1 | 3.x | 1 major |
| NumPy | 1.26.4 | 2.x | 1 major |
| OpenCV | 4.11.0.86 | 4.12+ / 5.x | 1 minor+ |

## Why Two Major Versions Behind

The gap is caused by a **transitive dependency chain** that requires
coordinated migration across multiple packages:

```
PaddlePaddle 3.x
  -> requires numpy >= 2.0
     -> OpenCV 4.12+ ships numpy 2.x bindings on Python 3.9+
        -> OpenCV 4.12 is blocked in Dependabot (numpy 2 conflict)
     -> PaddleNLP 3.x requires PaddlePaddle 3.x
        -> PaddleNLP 2.8.1 (current) is the last 2.x-compatible release
```

### Specific Blockers

1. **NumPy 1.x to 2.x**: PaddlePaddle 2.6.2 is ABI-locked to numpy 1.x.
   Upgrading numpy to 2.x without upgrading PaddlePaddle breaks at the C
   extension level (segfault, not an import error). NumPy 2.x is already
   blocked in Dependabot.

2. **PaddleOCR 2.9.1 to 2.10+/3.x**: PaddleOCR 2.10 bundles PaddleX, which
   changes the model loading API. The `create_paddle_engine` factory in
   `ocr_gpu_async.py` uses PaddleOCR 2.9.1's `PaddleOCR(use_onnx=True)`
   API. PaddleOCR 3.x uses `enable_hpi` instead.

3. **PaddleNLP UIE**: The structured extraction module (`extraction.py`) uses
   PaddleNLP 2.8.1's `Taskflow` API for Universal Information Extraction.
   PaddleNLP 3.x may change or deprecate this API.

4. **OpenCV pinning**: `opencv-python-headless==4.11.0.86` is the highest
   version that resolves cleanly with numpy 1.26.4. OpenCV 4.12+ pulls in
   numpy 2.x on Python 3.9+.

### Why This Is Acceptable

- **PaddlePaddle 2.6.x is still maintained** by Baidu for security patches.
  It receives bugfix releases (the 2.6 branch is not EOL).
- **PaddleOCR 2.9.1 uses CTC decoding** (zero hallucination), which is the
  forensic-grade requirement. PaddleOCR 3.x introduces PP-OCRv5 with
  attention-based decoding that may hallucinate on degraded inputs.
- **The Docker image is self-contained**. All dependencies are pinned in
  `requirements.txt` and pre-installed at build time. There is no risk of
  version drift in production.
- **ONNX Runtime backend** (`use_onnx=True`) provides CPU inference
  acceleration without requiring PaddlePaddle 3.x features.

## Upgrade Path

The upgrade must be executed as a single coordinated sprint, not incremental
package bumps:

### Step 1: NumPy 2.x Qualification

- Upgrade `numpy` from 1.26.4 to 2.x
- Verify all C extension packages (PaddlePaddle, OpenCV, Pillow) are
  compatible with numpy 2.x ABI
- Run full test suite + OCR accuracy regression

### Step 2: PaddlePaddle 3.x Migration

- Replace `paddlepaddle==2.6.2` with `paddlepaddle==3.x`
- Update `create_paddle_engine` API calls
- Verify GPU inference parity (same model, same output)
- Verify ONNX Runtime backend still works

### Step 3: PaddleOCR 2.10+/3.x Qualification

- Replace `paddleocr==2.9.1` with target version
- Adapt to new model loading API (PaddleX wrapper vs direct)
- **Critical**: Verify CTC-only decoding is still available and default.
  If PaddleOCR 3.x defaults to attention decoding, the upgrade must either
  force CTC mode or be deferred.
- Run 34-language OCR accuracy regression

### Step 4: PaddleNLP 3.x Migration

- Replace `paddlenlp==2.8.1` with 3.x
- Update `extraction.py` UIE Taskflow API calls
- Verify structured extraction accuracy

### Step 5: OpenCV Unpin

- Remove the `opencv-python-headless==4.11.0.86` pin
- Allow resolution to latest 4.x or 5.x
- Verify preprocessing pipeline (deskew, denoise, binarize)

## Timeline

| Milestone | Target Date | Status |
|-----------|------------|--------|
| Qualification sprint planning | Q2 2026 | Planned |
| NumPy 2.x + PaddlePaddle 3.x testing | Q2 2026 | Planned |
| PaddleOCR CTC verification | Q2 2026 | Planned |
| Full regression + Docker rebuild | Q2 2026 | Planned |
| Next review checkpoint | 2026-07-01 | Scheduled |

**Note**: The timeline is deliberately conservative. The upgrade is low-urgency
because the current stack is functional, secure (2.6.x receives patches), and
forensically validated. Rushing the upgrade risks introducing attention-based
decoding regressions.

## Dependabot Configuration

The following packages are blocked in `.github/dependabot.yml` to prevent
premature automated upgrade PRs:

- `paddlepaddle` (major versions)
- `paddleocr` (>= 2.10)
- `numpy` (major versions, 2.x)
- `opencv-python-headless` (>= 4.12)

These blocks remain in place until the qualification sprint completes.

## Risk If Deferred Beyond Q3 2026

- PaddlePaddle 2.6.x security patch cadence may slow
- Newer PaddleOCR models (PP-OCRv5+) will only be available on 3.x
- NumPy 1.x will eventually reach EOL (estimated late 2026)
- Growing incompatibility with other packages that require numpy 2.x

None of these risks are immediate. The Docker-first deployment model and
pinned dependencies provide a stable execution environment regardless of
upstream movement.

## References

- PaddlePaddle releases: https://github.com/PaddlePaddle/Paddle/releases
- PaddleOCR 2.9.1 docs: https://github.com/PaddlePaddle/PaddleOCR
- NumPy 2.0 migration guide: https://numpy.org/devdocs/numpy_2_0_migration_guide.html
- Existing migration plan: `docs/architecture/paddleocr-3.4.0-migration-execution-plan-20260224.md`
- Existing wave analysis: `docs/architecture/wave-a-paddleocr-2x-migration-20260301.md`
(PaddleOCR 2.9.1 `use_onnx` vs 3.x `enable_hpi`)

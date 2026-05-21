# Transform and Stamping Support Architecture

## Scope
Implement transform and stamping support as reusable API and scripting capabilities. This repository remains a support agent and does not become a full eDiscovery platform.

## Goals
- Add reusable transform operations for PDF and images.
- Add Bates and confidentiality designation stamping.
- Expose parity across API and script surfaces.
- Preserve backwards compatibility with existing OCR job behavior.
- Add custody and validation hooks for transform and stamping steps.

## Non-Goals
- No full review/production platform features.
- No built-in Relativity/Nuix packaging workflow in core runtime.
- No case-management or privilege-review orchestration.

## Architecture

### Phase A: Core contracts and operation registry
- Create a transform domain layer with typed operation models, validators, and result schema.
- Create a stamping domain layer with assignment, placement policy, and result schema.
- Provide a central operation registry for discovery/introspection.

### Phase B: Transform operations
- PDF page operations: extract, delete, rotate, reorder, split, merge, insert.
- Format conversion: PDF to images, image to image, images to PDF.
- Cleaning operations: deskew/denoise/binarize/contrast as standalone transforms.

### Phase C: Stamping engine
- Bates assignment with configurable prefix, start, width, suffix.
- Confidentiality designation templates (CONFIDENTIAL, HIGHLY CONFIDENTIAL, ATTORNEYS' EYES ONLY).
- Placement/collision strategy that avoids overlap between content, Bates, and designation overlays.

### Phase D: API surface
- Add transform and stamp routers with Pydantic request/response models.
- Keep job submit backward compatible, add optional transform/stamp config.
- Return operation artifacts and diagnostics in API responses.

### Phase E: Script surface
- Add script-first commands for transform and stamping workflows.
- Support single operation and chained operations.
- Add machine-readable output mode for orchestration.

### Phase F: Validation/custody/docs
- Add validation gates for Bates continuity and stamp overlap conflicts.
- Emit custody events for transform/stamp lifecycle.
- Publish integration docs and examples for external platform attachment.

## Key Integration Points
- Existing API app wiring: `api/main.py`, `api/models.py`, `api/deps.py`, `api/routers/`.
- Existing OCR and utility modules: `preprocessing.py`, `optimize_pdfs.py`, `ocr_gpu_async.py`.
- Existing custody chain: `custody.py`.

## Feature Flags
- `ENABLE_TRANSFORMS` (default false)
- `ENABLE_STAMPING` (default false)

## Testing Strategy
- Unit tests per transform and stamping operation.
- API tests for new routers and request validation.
- Script tests for command behavior and JSON output.
- Integration tests for transform+stamp chains and backward compatibility.

## Rollout Plan
- PR 1: Phase A
- PR 2: Phase B
- PR 3: Phase C
- PR 4: Phase D
- PR 5: Phase E
- PR 6: Phase F

Each PR should include targeted tests and docs updates and be independently reviewable.

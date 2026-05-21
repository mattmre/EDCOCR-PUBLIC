# 07: Transforms and Stamping

## Overview
This guide covers the transform and stamping surfaces added for support-agent workflows. These operations are designed for API callers and scripting automation that need deterministic document transforms without turning EDCOCR into a full eDiscovery platform.

## Scope Boundary
- In scope: reusable transform/stamp operations exposed via API and script.
- In scope: forensic safeguards (validation gates and custody diagnostics).
- Out of scope: full production management platform behavior (case management, review UI, production packaging lifecycle).

## Feature Flags (API)
Transform/stamp endpoints are opt-in.

| Variable | Default | Description |
|---|---|---|
| `ENABLE_TRANSFORMS` | `false` | Enables `/api/v1/transforms*` routes |
| `ENABLE_STAMPING` | `false` | Enables `/api/v1/stamps*` routes |

When disabled, routes return HTTP 403 with `feature_disabled`.

## API Surface

### Transform Routes
- `GET /api/v1/transforms` - list registered transform operations.
- `GET /api/v1/transforms/{operation_id}` - operation metadata.
- `POST /api/v1/transforms/execute` - execute one transform synchronously.

### Stamp Routes
- `GET /api/v1/stamps` - list registered stamp operations.
- `GET /api/v1/stamps/{operation_id}` - operation metadata.
- `POST /api/v1/stamps/execute` - execute one stamp operation synchronously.

## Script Surface
Script entrypoint: `scripts/transform_stamp_cli.py`

Supported commands:
- `transform` - run one transform operation.
- `stamp` - run one stamp operation.
- `chain` - run ordered transform/stamp operations.
- `list` - list available operations.

Machine-readable options:
- `--json-output <path>` writes JSON summary.
- `--json-stdout` prints JSON summary/metadata.

## Built-in Transform Operations
Registered via `ocr_distributed/transforms/builtin.py`.

| Operation ID | Purpose |
|---|---|
| `pdf_extract` | Extract selected pages |
| `pdf_delete` | Delete selected pages |
| `pdf_rotate` | Rotate pages (90/180/270) |
| `pdf_reorder` | Reorder pages |
| `pdf_split` | Split PDF into parts |
| `pdf_merge` | Merge multiple PDFs |
| `pdf_insert` | Insert pages from another PDF |
| `pdf_to_images` | Convert PDF pages to images |
| `image_convert` | Convert image format |
| `images_to_pdf` | Combine images into a PDF |
| `preprocessing` | Run image preprocessing pipeline |

## Built-in Stamp Operations
Registered via `ocr_distributed/stamps/builtin.py`.

| Operation ID | Purpose |
|---|---|
| `bates` | Apply sequential Bates values |
| `designation` | Apply confidentiality designation text |

Supported placements:
`top_left`, `top_center`, `top_right`, `bottom_left`, `bottom_center`, `bottom_right`, `center`

## API Examples

### Example: Extract pages 1 and 3
```json
POST /api/v1/transforms/execute
{
  "operation_id": "pdf_extract",
  "input_path": "C:\\OCR\\input.pdf",
  "output_path": "C:\\OCR\\output_extract.pdf",
  "params": {"pages": [1, 3]},
  "validate_input": true,
  "preserve_metadata": true
}
```

### Example: Bates stamp
```json
POST /api/v1/stamps/execute
{
  "operation_id": "bates",
  "input_path": "C:\\OCR\\input.pdf",
  "output_path": "C:\\OCR\\output_bates.pdf",
  "placement": "bottom_right",
  "params": {"prefix": "PROD", "start": 1000, "width": 6},
  "validate_input": true,
  "check_overlap": true
}
```

### Example: Designation stamp
```json
POST /api/v1/stamps/execute
{
  "operation_id": "designation",
  "input_path": "C:\\OCR\\input.pdf",
  "output_path": "C:\\OCR\\output_confidential.pdf",
  "placement": "top_center",
  "params": {"text": "CONFIDENTIAL"},
  "validate_input": true,
  "check_overlap": true
}
```

## CLI Examples

### Transform
```bash
python scripts/transform_stamp_cli.py transform pdf_extract ^
  --input C:\OCR\input.pdf ^
  --output C:\OCR\extract.pdf ^
  --params "{\"pages\": [1, 2]}"
```

### Stamp (Bates)
```bash
python scripts/transform_stamp_cli.py stamp bates ^
  --input C:\OCR\input.pdf ^
  --output C:\OCR\bates.pdf ^
  --placement bottom_right ^
  --params "{\"prefix\": \"PROD\", \"start\": 1000, \"width\": 6}"
```

### Chained operations
```bash
python scripts/transform_stamp_cli.py chain ^
  --input C:\OCR\input.pdf ^
  --output C:\OCR\final.pdf ^
  --operations "[{\"type\":\"transform\",\"id\":\"pdf_extract\",\"params\":{\"pages\":[1,2]}},{\"type\":\"stamp\",\"id\":\"bates\",\"placement\":\"bottom_right\",\"params\":{\"prefix\":\"PROD\",\"start\":1000}}]"
```

### List operations
```bash
python scripts/transform_stamp_cli.py list
python scripts/transform_stamp_cli.py list --json-stdout
```

## Validation Gates and Custody Diagnostics
Phase F integrates post-operation validation and custody eventing:

- `validate_transform_output`:
  - output exists
  - output is non-empty
  - output is hashable
- `validate_stamp_output`:
  - output integrity checks
  - Bates continuity checks (Bates only)
  - overlap-conflict checks when overlap warnings are present

Custody hooks (`custody_hooks.py`) record:
- operation start/complete/failure events
- input/output hashes
- custody chain hash

API responses include custody diagnostics in `metadata.custody` on success and include custody detail in error payloads when execution fails.

## Troubleshooting
- `operation_not_found`: check operation IDs via API list endpoint or `list` CLI command.
- `validation_failed` / `validation_gate_failed`: inspect returned validation details.
- `feature_disabled`: set `ENABLE_TRANSFORMS=true` or `ENABLE_STAMPING=true` for API usage.
- `input_not_found`: verify server-side input path exists.

## Related Files
- API routes: `api/routers/transforms.py`, `api/routers/stamps.py`
- Script: `scripts/transform_stamp_cli.py`
- Validation: `validation_gates.py`
- Custody hooks: `custody_hooks.py`
- Tests: `tests/test_api_transforms_stamps.py`, `tests/test_transform_stamp_cli.py`, `tests/test_phase_f_validation_custody.py`, `tests/test_phase_f_cli_integration.py`

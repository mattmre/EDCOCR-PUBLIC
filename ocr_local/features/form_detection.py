"""Deterministic form-field / component-piece detection.

Identifies form fields -- key->value pairs (with geometry), checkboxes, and their
states -- from the pipeline-standard ``(text, box, confidence)`` OCR lines. Pure,
heuristic, and DETERMINISTIC (no model, no generation), so it is forensic-safe
and emitted as an additive ``EXPORT/FORM/*.form.json`` sidecar
(``deterministic: true``, ``advisory: false``).

This is the "forms and component pieces" capability: every detected field is a
component piece with key text, value text, a field type, and bounding boxes.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"
EXPORT_SUBDIR = "FORM"
SIDECAR_SUFFIX = ".form.json"

# Leading glyphs / tokens that mark a checkbox and its state.
_CHECKED_MARKERS = ("☑", "☒", "[x]", "[X]", "[✓]", "■")  # ☑ ☒ ■
_UNCHECKED_MARKERS = ("☐", "[ ]", "□", "❑")  # ☐ □


@dataclass
class FormField:
    key_text: str
    field_type: str = "text"  # "text" | "checkbox"
    value_text: str = ""
    checked: bool | None = None
    key_bbox: list[float] | None = None
    value_bbox: list[float] | None = None
    confidence: float = 0.0
    page_num: int = 1


@dataclass
class FormPage:
    page_num: int
    fields: list[FormField] = field(default_factory=list)


@dataclass
class FormDocument:
    pages: list[FormPage] = field(default_factory=list)
    field_count: int = 0
    checkbox_count: int = 0


def _to_xyxy(box: Any) -> list[float] | None:
    """Normalize a polygon or flat box (incl. numpy arrays) to ``[x1, y1, x2, y2]``."""
    if box is None:
        return None
    # OCR engines (e.g. PaddleOCR) return boxes as numpy arrays or lists of numpy
    # rows; normalize to plain Python lists before any truthiness/length check
    # (``if not box`` raises "ambiguous truth value" on a numpy array).
    try:
        import numpy as np

        if isinstance(box, np.ndarray):
            box = box.tolist()
        elif isinstance(box, (list, tuple)):
            box = [p.tolist() if isinstance(p, np.ndarray) else p for p in box]
    except Exception:
        pass
    if not box:
        return None
    first = box[0] if isinstance(box, (list, tuple)) and box else None
    if isinstance(first, (list, tuple)):
        xs = [float(p[0]) for p in box if len(p) >= 2]
        ys = [float(p[1]) for p in box if len(p) >= 2]
        if not xs or not ys:
            return None
        return [min(xs), min(ys), max(xs), max(ys)]
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        try:
            return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        except (TypeError, ValueError):
            return None
    return None


def _checkbox_state(text: str) -> tuple[bool | None, str]:
    """Return ``(checked, label)`` if *text* starts with a checkbox marker."""
    stripped = text.strip()
    for marker in _CHECKED_MARKERS:
        if stripped.startswith(marker):
            return True, stripped[len(marker):].strip()
    for marker in _UNCHECKED_MARKERS:
        if stripped.startswith(marker):
            return False, stripped[len(marker):].strip()
    return None, ""


def detect_form_fields(lines: list[tuple[str, Any, float]], page_num: int = 1) -> list[FormField]:
    """Detect form fields from OCR ``(text, box, confidence)`` lines (deterministic)."""
    entries = []
    for text, box, conf in lines:
        if not text or not str(text).strip():
            continue
        entries.append(
            {"text": str(text).strip(), "xyxy": _to_xyxy(box), "conf": float(conf or 0.0)}
        )

    fields: list[FormField] = []
    consumed: set[int] = set()

    for i, entry in enumerate(entries):
        if i in consumed:
            continue
        text = entry["text"]

        # 1) Checkbox
        checked, label = _checkbox_state(text)
        if checked is not None:
            fields.append(
                FormField(
                    key_text=label or text,
                    field_type="checkbox",
                    checked=checked,
                    key_bbox=entry["xyxy"],
                    confidence=entry["conf"],
                    page_num=page_num,
                )
            )
            consumed.add(i)
            continue

        # 2) Inline "key: value"
        if ":" in text:
            left, right = text.split(":", 1)
            left, right = left.strip(), right.strip()
            if left and right:
                fields.append(
                    FormField(
                        key_text=left,
                        value_text=right,
                        key_bbox=entry["xyxy"],
                        confidence=entry["conf"],
                        page_num=page_num,
                    )
                )
                consumed.add(i)
                continue
            if left and not right:
                # 3) Label awaiting a value to its right on the same row.
                value_entry, value_idx = _find_value_to_right(entries, i, consumed)
                fields.append(
                    FormField(
                        key_text=left,
                        value_text=value_entry["text"] if value_entry else "",
                        key_bbox=entry["xyxy"],
                        value_bbox=value_entry["xyxy"] if value_entry else None,
                        confidence=entry["conf"],
                        page_num=page_num,
                    )
                )
                consumed.add(i)
                if value_idx is not None:
                    consumed.add(value_idx)
                continue

    return fields


def _is_field_like(text: str) -> bool:
    """Return True if *text* is itself a field — a label/inline ``"k: v"`` or a checkbox.

    Such an entry marks a *column boundary*: the value of a field to its left ends
    here, so a bare label must never reach across it to pair with content beyond.
    Any colon counts (bare ``"Name:"`` and inline ``"Foo: bar"`` are both their own
    fields); a checkbox marker counts too.
    """
    if ":" in text:
        return True
    return _checkbox_state(text)[0] is not None


def _find_value_to_right(
    entries: list[dict], key_idx: int, consumed: set[int]
) -> tuple[dict | None, int | None]:
    """Find the value entry immediately to the right of the key on the same row.

    The *nearest* same-row entry to the right defines the key's value column. If that
    nearest entry is already consumed, or is itself a field (label / inline ``k: v`` /
    checkbox), the key's value column is empty and we return ``None`` — we never reach
    *across* an intervening field to steal another column's value. So a single-row
    multi-column layout like ``["Name:", "DOB:", "1990"]`` pairs ``DOB`` with ``1990``
    and leaves ``Name`` empty, instead of ``Name`` stealing ``1990`` across ``DOB:``.
    """
    key = entries[key_idx]
    kbox = key["xyxy"]
    if kbox is None:
        return None, None
    k_mid_y = (kbox[1] + kbox[3]) / 2.0
    k_height = max(kbox[3] - kbox[1], 1.0)
    tol = 0.6 * k_height

    best = None
    best_idx = None
    best_dx = None
    for j, other in enumerate(entries):
        if j == key_idx:
            continue
        obox = other["xyxy"]
        if obox is None:
            continue
        o_mid_y = (obox[1] + obox[3]) / 2.0
        if abs(o_mid_y - k_mid_y) > tol:
            continue
        if obox[0] < kbox[2]:  # must start to the right of the key
            continue
        dx = obox[0] - kbox[2]
        if best_dx is None or dx < best_dx:
            best, best_idx, best_dx = other, j, dx

    # The nearest right-hand entry is the column boundary. If it is already consumed or
    # is itself a field, this key has no value of its own — do not cross the boundary.
    if best is None:
        return None, None
    if best_idx in consumed or _is_field_like(best["text"]):
        return None, None
    return best, best_idx


def finalize_form(doc: FormDocument) -> FormDocument:
    """Populate document-level counts from the per-page fields."""
    doc.field_count = sum(len(p.fields) for p in doc.pages)
    doc.checkbox_count = sum(
        1 for p in doc.pages for f in p.fields if f.field_type == "checkbox"
    )
    return doc


def build_form_payload(
    doc: FormDocument,
    *,
    source_file: str,
    document_id: str,
    pipeline_version: str,
    timestamp: str,
) -> dict[str, Any]:
    """Build the schema-conformant form sidecar payload (pure)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "source_file": source_file,
        "deterministic": True,
        "advisory": False,
        "processing": {"pipeline_version": pipeline_version, "timestamp": timestamp},
        "document_summary": {
            "field_count": doc.field_count,
            "checkbox_count": doc.checkbox_count,
        },
        "pages": [
            {
                "page_num": page.page_num,
                "fields": [asdict(f) for f in page.fields],
            }
            for page in doc.pages
        ],
    }


def write_form_json(
    doc: FormDocument,
    output_folder: str,
    *,
    source_file: str,
    document_id: str,
    pipeline_version: str,
    timestamp: str,
    subfolder: str = "",
) -> str:
    """Write the form sidecar to ``EXPORT/FORM/<subfolder>/<base>.form.json``."""
    payload = build_form_payload(
        doc,
        source_file=source_file,
        document_id=document_id,
        pipeline_version=pipeline_version,
        timestamp=timestamp,
    )
    from ocr_distributed.ocr_utils import (
        build_sidecar_base_name,
        resolve_safe_output_subfolder,
        write_json_atomic,
    )

    base = build_sidecar_base_name(source_file)
    root_dir = os.path.join(output_folder, "EXPORT", EXPORT_SUBDIR)
    out_dir = resolve_safe_output_subfolder(root_dir, subfolder)
    out_path = os.path.join(out_dir, f"{base}{SIDECAR_SUFFIX}")

    os.makedirs(out_dir, exist_ok=True)
    write_json_atomic(
        out_path,
        payload,
        allowed_root=root_dir,
        default=None,
    )
    return out_path

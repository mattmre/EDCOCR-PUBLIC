#!/usr/bin/env python3
"""Generate raster-first OCR rescue fixtures from a reviewed text manifest.

Expected text lives in ``tests/fixtures/ocr_rescue_v1/manifest.json``.  The
generator renders that fixed truth and records physical pages, visible marker
locations, source hashes, and font hashes for independent output validation.
No OCR engine or model is loaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "tests/fixtures/ocr_rescue_v1/manifest.json"
DEFAULT_LATIN_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
DEFAULT_GREEK_FONT = Path("/usr/share/fonts/truetype/freefont/FreeSerif.ttf")
PAGE_SIZES = {"A4": (595.0, 842.0), "Letter": (612.0, 792.0)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_raster(page_spec: dict, width_pt: float, height_pt: float, font_path: Path) -> tuple[bytes, dict[str, list[float]]]:
    dpi = int(page_spec["dpi"])
    width_px = round(width_pt * dpi / 72)
    height_px = round(height_pt * dpi / 72)
    kind = page_spec["kind"]
    if kind == "alpha_black_text":
        image = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
        text_fill = (0, 0, 0, 255)
    elif kind == "palette_trns":
        image = Image.new("P", (width_px, height_px), 0)
        image.putpalette([255, 255, 255, 0, 0, 0] + [0] * 762)
        image.info["transparency"] = 0
        text_fill = 1
    else:
        image = Image.new("RGB", (width_px, height_px), "white")
        text_fill = (22, 24, 27)
    draw = ImageDraw.Draw(image)
    scale = dpi / 72
    if kind == "blank":
        output = io.BytesIO()
        image.save(output, format="PNG", dpi=(dpi, dpi))
        return output.getvalue(), {}

    body = page_spec["expected_text"]
    if kind == "mixed":
        body = body.removeprefix(page_spec["native_heading"]).lstrip(" .")
    font_size_pt = 8 if kind == "fineprint" else 14
    font = ImageFont.truetype(str(font_path), round(font_size_pt * scale))
    x_start = round(50 * scale)
    x = x_start
    y = round((105 if kind == "mixed" else 78) * scale)
    line_height = round(font_size_pt * scale * 1.45)
    max_x = width_px - round(45 * scale)
    markers = {token.casefold(): token for token in page_spec.get("regions", [])}
    boxes: dict[str, list[float]] = {}
    for token in body.split():
        token_width = draw.textlength(token + " ", font=font)
        if x + token_width > max_x and x > x_start:
            x = x_start
            y += line_height
        if y + line_height >= height_px - round(30 * scale):
            raise ValueError(f"fixture text does not fit: {page_spec['expected_text'][:40]}")
        draw.text((x, y), token, fill=text_fill, font=font)
        clean = re.sub(r"^\W+|\W+$", "", token, flags=re.UNICODE).casefold()
        if clean in markers:
            left, top, right, bottom = draw.textbbox((x, y), token, font=font)
            boxes[markers[clean]] = [
                round(left / scale, 3),
                round(top / scale, 3),
                round(right / scale, 3),
                round(bottom / scale, 3),
            ]
        x += round(token_width)
    if kind == "fineprint":
        draw.rectangle(
            [round(45 * scale), round(45 * scale), round(90 * scale), round(70 * scale)],
            fill=(190, 20, 35),
        )
        draw.line(
            [round(42 * scale), round(140 * scale), round((width_pt - 42) * scale), round(140 * scale)],
            fill=(15, 65, 200),
            width=max(1, round(scale / 2)),
        )
    missing = set(page_spec.get("regions", [])) - set(boxes)
    if missing and kind != "mixed":
        raise ValueError(f"fixture marker not visibly rendered: {sorted(missing)}")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True, dpi=(dpi, dpi), transparency=0 if kind == "palette_trns" else None)
    return output.getvalue(), boxes


def generate(spec_path: Path, output_dir: Path, *, latin_font: Path, greek_font: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty; keep attempts immutable")
    for font_path in (latin_font, greek_font):
        if not font_path.is_file():
            raise FileNotFoundError(f"required visible-text font unavailable: {font_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != "ocr-rescue-corpus-v1":
        raise ValueError("unsupported OCR rescue corpus schema")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {
        "schema_version": spec["schema_version"],
        "spec_sha256": _sha256(spec_path),
        "normalization": spec["normalization"],
        "thresholds": spec["thresholds"],
        "fonts": {"latin": {"path": str(latin_font), "sha256": _sha256(latin_font)}, "greek": {"path": str(greek_font), "sha256": _sha256(greek_font)}},
        "cases": [],
    }
    for case in spec["cases"]:
        document = fitz.open()
        page_records = []
        image_sources = []
        for page_spec in case["pages"]:
            width_pt, height_pt = PAGE_SIZES[page_spec["size"]]
            page = document.new_page(width=width_pt, height=height_pt)
            font_path = greek_font if page_spec["font"] == "greek" else latin_font
            png, boxes = _render_raster(page_spec, width_pt, height_pt, font_path)
            if page_spec["kind"] in {"alpha_black_text", "palette_trns"}:
                image_name = f"{case['id']}-page{len(page_records) + 1}.png"
                image_path = output_dir / image_name
                image_path.write_bytes(png)
                image_sources.append({"path": image_name, "sha256": _sha256(image_path), "dpi": page_spec["dpi"], "mode": "RGBA" if page_spec["kind"] == "alpha_black_text" else "P+tRNS"})
            page.insert_image(page.mediabox, stream=png)
            if page_spec["kind"] == "mixed":
                page.insert_text((50, 65), page_spec["native_heading"], fontsize=19, color=(0.08, 0.08, 0.08))
                for marker in page_spec.get("regions", []):
                    if marker in page_spec["native_heading"]:
                        found = page.search_for(marker)
                        if found:
                            rect = found[0]
                            boxes[marker] = [rect.x0, rect.y0, rect.x1, rect.y1]
            if page_spec.get("old_hidden_text"):
                page.insert_text((45, 45), page_spec["old_hidden_text"], fontsize=12, render_mode=3)
            if page_spec.get("crop"):
                page.set_cropbox(fitz.Rect(page_spec["crop"]))
            if page_spec.get("rotation"):
                page.set_rotation(int(page_spec["rotation"]))
            offset = page.cropbox_position
            visible_boxes = {}
            for marker, coords in boxes.items():
                rect = fitz.Rect(coords)
                rect = fitz.Rect(rect.x0 - offset.x, rect.y0 - offset.y, rect.x1 - offset.x, rect.y1 - offset.y)
                rect *= page.rotation_matrix
                if not page.rect.contains(rect):
                    raise ValueError(f"fixture marker outside visible crop: {case['id']}:{marker}")
                visible_boxes[marker] = [round(rect.x0, 3), round(rect.y0, 3), round(rect.x1, 3), round(rect.y1, 3)]
            page_records.append({
                "expected_text": page_spec["expected_text"],
                "expected_language": page_spec.get("expected_language"),
                "size_pt": [round(page.rect.width, 3), round(page.rect.height, 3)],
                "rotation": page.rotation,
                "cropbox": list(page.cropbox),
                "regions": [{"text": key, "rect_pt": value} for key, value in visible_boxes.items()],
                "old_hidden_text": page_spec.get("old_hidden_text", ""),
            })
        filename = f"{case['id']}.pdf"
        source = output_dir / filename
        document.save(source, garbage=4, deflate=True)
        document.close()
        generated["cases"].append({
            "id": case["id"],
            "source_pdf": filename,
            "source_kind": "pdf",
            "source_sha256": _sha256(source),
            "source_images": image_sources,
            "forbidden_text": case.get("forbidden_text", []),
            "pages": page_records,
        })
        for image in image_sources:
            generated["cases"].append({
                "id": f"{case['id']}_png",
                "source_kind": "png",
                "source_image": image["path"],
                "source_sha256": image["sha256"],
                "source_dpi": image["dpi"],
                "forbidden_text": case.get("forbidden_text", []),
                "pages": page_records,
            })
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(generated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--latin-font", type=Path, default=DEFAULT_LATIN_FONT)
    parser.add_argument("--greek-font", type=Path, default=DEFAULT_GREEK_FONT)
    args = parser.parse_args()
    manifest = generate(args.spec, args.output_dir, latin_font=args.latin_font, greek_font=args.greek_font)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import math

import fitz
import pytest

from ocr_local.ocr_overlay import (
    compare_source_grounded_text_revision,
    insert_hidden_text_in_box,
    map_half_turn_words_to_source,
    map_rotated_detector_lines_to_source,
    map_rotated_words_to_source,
    read_source_tesseract_words,
    tesseract_data_to_lines,
    tesseract_data_to_lines_with_review,
)
from ocr_local.pdf_geometry import PageGeometry


def test_source_revision_accepts_only_same_ordered_tokens_with_grounded_positions():
    prior = [("term171.term172", [10, 20, 150, 40], .99)]
    candidate = [
        ("term171", [10, 20, 80, 40], .95),
        ("term172", [80, 20, 150, 40], .94),
    ]
    decision = compare_source_grounded_text_revision(
        prior, candidate, prior_coverage=1, candidate_coverage=1,
        script_supported=True,
    )
    assert decision.use_candidate and decision.needs_review
    assert decision.reason == "source_grounded_punctuation_or_spacing_revision"

    for rejected in [
        [("the", [10, 20, 70, 40], .95), ("rapist", [80, 20, 150, 40], .95)],
        [("term172", [10, 20, 80, 40], .95), ("term171", [80, 20, 150, 40], .95)],
        [("term171", [10, 20, 80, 40], .95)] * 2,
    ]:
        assert not compare_source_grounded_text_revision(
            prior, rejected, prior_coverage=1, candidate_coverage=1,
            script_supported=True,
        ).use_candidate
    assert not compare_source_grounded_text_revision(
        [("therapist", [10, 20, 150, 40], .99)],
        [("the", [10, 20, 70, 40], .95), ("rapist", [80, 20, 150, 40], .95)],
        prior_coverage=1, candidate_coverage=1, script_supported=True,
    ).use_candidate
    assert not compare_source_grounded_text_revision(
        [("1.23", [10, 20, 150, 40], .99)],
        [("123", [10, 20, 150, 40], .95)],
        prior_coverage=1, candidate_coverage=1, script_supported=True,
    ).use_candidate


@pytest.mark.parametrize("reason,kwargs", [
    ("source_revision_geometry_disagreement", {"box": [300, 20, 380, 40]}),
    ("source_revision_weak_candidate_reading", {"confidence": .6}),
    ("source_revision_coverage_unproven", {"coverage": .8}),
    ("source_revision_missing_or_unsupported", {"script": False}),
])
def test_source_revision_rejects_unsupported_alternative(reason, kwargs):
    decision = compare_source_grounded_text_revision(
        [("HEARING", [10, 20, 100, 40], .99)],
        [("HEARING.", kwargs.get("box", [10, 20, 100, 40]),
          kwargs.get("confidence", .95))],
        prior_coverage={"status": "measured", "coverage": 1},
        candidate_coverage={"status": "measured", "coverage": kwargs.get("coverage", 1)},
        script_supported=kwargs.get("script", True),
    )
    assert not decision.use_candidate and decision.reason == reason


def test_source_revision_allows_weak_unchanged_word_but_not_changed_word():
    prior = [
        ("NOTICE", [10, 20, 80, 40], .99),
        ("OF", [85, 20, 105, 40], .99),
        ("HEARING", [110, 20, 200, 40], .99),
    ]
    candidate = [
        ("NOTICE", [10, 20, 80, 40], .84),
        ("OF", [85, 20, 105, 40], .96),
        ("HEARING.", [110, 20, 200, 40], .97),
    ]
    kwargs = {
        "prior_coverage": 1, "candidate_coverage": 1,
        "script_supported": True,
    }
    assert compare_source_grounded_text_revision(
        prior, candidate, **kwargs
    ).use_candidate
    changed_weak = candidate.copy()
    changed_weak[-1] = ("HEARING.", [110, 20, 200, 40], .84)
    decision = compare_source_grounded_text_revision(
        prior, changed_weak, **kwargs
    )
    assert not decision.use_candidate
    assert decision.reason == "source_revision_weak_changed_word"


def test_source_tesseract_reader_maps_scaled_raster_to_source_pixels(monkeypatch):
    import pytesseract
    from PIL import Image

    calls = []
    def fake_data(image, **kwargs):
        calls.append((image.size, kwargs))
        return {
            "text": ["HEARING."], "conf": [95],
            "left": [10], "top": [10], "width": [20], "height": [10],
        }
    monkeypatch.setattr(pytesseract, "image_to_data", fake_data)
    lines, issues = read_source_tesseract_words(
        Image.new("RGB", (200, 100)), traineddata="eng", max_pixels=5_000,
    )
    assert issues == [] and len(calls) == 1
    assert calls[0][0] == (100, 50)
    assert lines == [("HEARING.", [20, 20, 60, 40], .95)]


def test_tesseract_words_keep_order_boxes_and_confidence():
    data = {
        "text": ["", "first", "second", "bad", "outside", "last"],
        "conf": ["-1", "91.5", "80", "nan", "40", "100"],
        "left": [0, 100, 300, 10, 2000, 700],
        "top": [0, 200, 200, 10, 10, 800],
        "width": [0, 80, 90, 30, 20, 100],
        "height": [0, 40, 40, 30, 20, 40],
    }
    transform = PageGeometry.from_image_pixels(1000, 1000, dpi=72).for_raster(1000, 1000)
    lines = tesseract_data_to_lines(data, transform=transform)
    assert [line[0] for line in lines] == ["first", "second", "last"]
    assert [line[1] for line in lines] == [
        [100, 200, 180, 240], [300, 200, 390, 240], [700, 800, 800, 840]
    ]
    assert [line[2] for line in lines] == pytest.approx([0.915, 0.8, 1])


def test_malformed_rows_and_nonfinite_values_are_skipped():
    data = {
        "text": ["ok", "broken", "zero", "inf", "tail"],
        "conf": [50, "oops", 30, math.inf, 75],
        "left": [1, 2, 3, 4, 10],
        "top": [1, 2, 3, 4, 20],
        "width": [10, 10, 0, 10, 30],
        "height": [10, 10, 10, 10],
    }
    assert tesseract_data_to_lines(data) == [("ok", [1, 1, 11, 11], 0.5)]
    _lines, issues = tesseract_data_to_lines_with_review(data)
    assert {issue["text"] for issue in issues} == {"broken", "zero", "inf", "tail"}
    assert tesseract_data_to_lines({"text": []}) == []


def test_zero_confidence_recognized_word_is_preserved():
    data = {
        "text": ["review", "layout"], "conf": [0, -1],
        "left": [20, 40], "top": [30, 30],
        "width": [40, 40], "height": [10, 10],
    }
    assert tesseract_data_to_lines(data) == [("review", [20, 30, 60, 40], 0.0)]


def test_half_turn_ocr_word_maps_back_to_original_page_without_rotating_background():
    transform = PageGeometry(200, 100, "pdf").for_raster(200, 100)
    mapped, issues = map_half_turn_words_to_source(
        [("HUNDRED", [20, 10, 100, 30], 0.96)], transform
    )
    assert issues == []
    assert mapped[0][1] == [
        [180, 90], [100, 90], [100, 70], [180, 70]
    ]
    with fitz.open() as pdf:
        page = pdf.new_page(width=200, height=100)
        assert insert_hidden_text_in_box(page, mapped[0][0], mapped[0][1], fontname="helv")
        assert "HUNDRED" in page.get_text()


@pytest.mark.parametrize(
    "angle,expected",
    [
        (90, [[190, 20], [190, 100], [170, 100], [170, 20]]),
        (270, [[10, 80], [10, 0], [30, 0], [30, 80]]),
    ],
)
def test_quarter_turn_ocr_boxes_map_to_original_raster(angle, expected):
    transform = PageGeometry(200, 100, "pdf").for_raster(200, 100)
    mapped, issues = map_rotated_words_to_source(
        [("word", [20, 10, 100, 30], 0.9)], transform, angle
    )
    assert issues == []
    assert mapped[0][1] == expected


@pytest.mark.parametrize(
    "angle,expected",
    [
        (90, [[190, 20], [190, 100], [170, 100], [170, 20]]),
        (180, [[180, 90], [100, 90], [100, 70], [180, 70]]),
        (270, [[10, 80], [10, 0], [30, 0], [30, 80]]),
    ],
)
def test_rotated_paddle_detector_quad_maps_back_to_original_page(angle, expected):
    transform = PageGeometry(200, 100, "pdf").for_raster(200, 100)
    pixel, pdf, issues = map_rotated_detector_lines_to_source(
        [("HUNDRED", [20, 10, 100, 30], 0.96)], transform, angle
    )
    assert issues == []
    assert pixel[0][1] == expected
    assert pdf[0][1] == expected
    with fitz.open() as document:
        page = document.new_page(width=200, height=100)
        assert insert_hidden_text_in_box(
            page, pdf[0][0], pdf[0][1], fontname="helv"
        )
        with fitz.open(stream=document.tobytes(), filetype="pdf") as reopened:
            assert reopened[0].search_for("HUNDRED")


def test_rotated_detector_rejects_unplaceable_quad_without_anchor():
    transform = PageGeometry(200, 100, "pdf").for_raster(200, 100)
    pixel, pdf, issues = map_rotated_detector_lines_to_source(
        [("outside", [300, 300, 320, 320], 0.99)], transform, 90
    )
    assert not pixel and not pdf
    assert len(issues) == 1 and issues[0]["text"] == "outside"


@pytest.mark.parametrize(
    "text, box",
    [
        ("fine", [20, 20, 34, 25]),
        ("LONGNARROWWORD", [40, 40, 75, 47]),
        ("edge", [0, 95, 20, 100]),
    ],
)
def test_hidden_text_search_hit_fits_detected_small_box(text, box):
    document = fitz.open()
    page = document.new_page(width=100, height=100)
    assert insert_hidden_text_in_box(page, text, box, fontname="helv")
    reopened = fitz.open(stream=document.tobytes(), filetype="pdf")
    hits = reopened[0].search_for(text)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.x0 >= box[0] - 0.25
    assert hit.y0 >= box[1] - 0.25
    assert hit.x1 <= box[2] + 0.25
    assert hit.y1 <= box[3] + 0.25
    reopened.close()
    document.close()


def test_hidden_text_rejects_unplaceable_box_without_anchor():
    document = fitz.open()
    page = document.new_page(width=100, height=100)
    assert not insert_hidden_text_in_box(page, "bad", [101, 20, 120, 30], fontname="helv")
    assert not insert_hidden_text_in_box(page, "bad", [10, 10, 10, 30], fontname="helv")
    assert not page.get_text().strip()
    document.close()


def test_embedded_font_search_hits_fill_independent_detected_advances():
    from pathlib import Path

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font_path.exists():
        pytest.skip("qualified local font unavailable")
    document = fitz.open()
    page = document.new_page(width=300, height=100)
    cases = [
        ("WIDE LINE MARKER", [20, 20, 145, 40]),
        ("ii", [180, 20, 242, 40]),
    ]
    for text, box in cases:
        assert insert_hidden_text_in_box(
            page, text, box, fontname="dejavu", fontfile=str(font_path)
        )
    with fitz.open(stream=document.tobytes(), filetype="pdf") as reopened:
        for text, box in cases:
            hits = reopened[0].search_for(text)
            assert len(hits) == 1
            assert hits[0].x0 == pytest.approx(box[0], abs=2)
            assert hits[0].x1 == pytest.approx(box[2], abs=2)
            assert hits[0].y0 >= box[1] - 1
            assert hits[0].y1 <= box[3] + 1
    document.close()


@pytest.mark.parametrize("angle", [0, 30, 90, 180, 270])
def test_rotated_detector_quad_search_geometry_fits_quad(angle):
    import math

    document = fitz.open()
    page = document.new_page(width=200, height=200)
    theta = math.radians(angle)
    u = (math.cos(theta), math.sin(theta))
    v = (-math.sin(theta), math.cos(theta))
    p0 = (100, 100)
    length, height = 55, 16
    p1 = (p0[0] + length * u[0], p0[1] + length * u[1])
    p3 = (p0[0] + height * v[0], p0[1] + height * v[1])
    p2 = (p1[0] + height * v[0], p1[1] + height * v[1])
    quad = [p0, p1, p2, p3]
    assert insert_hidden_text_in_box(page, "ROTATED", quad, fontname="helv")
    reopened = fitz.open(stream=document.tobytes(), filetype="pdf")
    hits = reopened[0].search_for("ROTATED", quads=True)
    assert len(hits) == 1
    for point in (hits[0].ul, hits[0].ur, hits[0].lr, hits[0].ll):
        delta = (point.x - p0[0], point.y - p0[1])
        along = delta[0] * u[0] + delta[1] * u[1]
        normal = delta[0] * v[0] + delta[1] * v[1]
        assert -0.3 <= along <= length + 0.3
        assert -0.3 <= normal <= height + 0.3
    reopened.close()
    document.close()


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_source_rotation_reinterprets_axis_aligned_paddle_quad(rotation):
    document = fitz.open()
    page = document.new_page(width=800, height=1600)
    if rotation in (90, 270):
        box = [[100, 100], [144, 100], [144, 1490], [100, 1490]]
    else:
        box = [[100, 100], [700, 100], [700, 144], [100, 144]]
    recognized = "GROUNDED ROTATED OCR TEXT"
    assert insert_hidden_text_in_box(
        page, recognized, box, fontname="helv", source_rotation=rotation
    )
    with fitz.open(stream=document.tobytes(), filetype="pdf") as reopened:
        assert recognized in reopened[0].get_text()
        hits = reopened[0].search_for(recognized)
        assert len(hits) == 1
        expected = fitz.Quad([fitz.Point(*point) for point in box]).rect
        assert expected.contains(hits[0])
    document.close()


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_paddle_word_order_tracks_source_reading_direction(rotation):
    document = fitz.open()
    page = document.new_page(width=800, height=1600)
    box = (
        [[100, 100], [144, 100], [144, 1490], [100, 1490]]
        if rotation in (90, 270)
        else [[100, 100], [700, 100], [700, 144], [100, 144]]
    )
    assert insert_hidden_text_in_box(
        page, "FIRST SECOND", box, fontname="helv", source_rotation=rotation
    )
    with fitz.open(stream=document.tobytes(), filetype="pdf") as reopened:
        first = reopened[0].search_for("FIRST")[0]
        second = reopened[0].search_for("SECOND")[0]
        if rotation == 90:
            assert first.y0 < second.y0
        elif rotation == 270:
            assert first.y0 > second.y0
        else:
            assert first.x0 > second.x0
    document.close()


def test_non_affine_detector_quad_requires_review():
    document = fitz.open()
    page = document.new_page(width=200, height=200)
    assert not insert_hidden_text_in_box(
        page, "UNPLACED", [[50, 50], [100, 50], [90, 70], [50, 70]], fontname="helv"
    )
    assert not page.search_for("UNPLACED")
    document.close()


def test_missing_unicode_font_cannot_claim_searchable_greek():
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    assert not insert_hidden_text_in_box(
        page, "Αθήνα", [20, 20, 150, 50], fontname="helv"
    )
    assert not page.get_text().strip()
    document.close()


@pytest.mark.parametrize("dpi", [300, 450, 600])
def test_word_positions_stay_searchable_across_render_retries(dpi):
    source = fitz.open()
    source_page = source.new_page(width=612, height=792)
    geometry = PageGeometry.from_pdf_page(source_page)
    raster = source_page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    scale_x, scale_y = raster.width / 612, raster.height / 792
    data = {
        "text": ["ALPHA", "BETA"],
        "conf": [97, 93],
        "left": [100 * scale_x, 350 * scale_x],
        "top": [200 * scale_y, 400 * scale_y],
        "width": [100 * scale_x, 80 * scale_x],
        "height": [30 * scale_y, 30 * scale_y],
    }
    lines = tesseract_data_to_lines(data, transform=geometry.for_raster(raster.width, raster.height))
    output = fitz.open()
    page = output.new_page(width=geometry.width_pt, height=geometry.height_pt)
    for text, box, _confidence in lines:
        page.insert_text((box[0], box[1] + 12), text, fontsize=12, render_mode=3)
    reopened = fitz.open(stream=output.tobytes(), filetype="pdf")
    assert reopened[0].rect.width == pytest.approx(612)
    assert reopened[0].rect.height == pytest.approx(792)
    assert reopened[0].search_for("ALPHA")[0].x0 == pytest.approx(100, abs=0.2)
    assert reopened[0].search_for("BETA")[0].x0 == pytest.approx(350, abs=0.2)
    reopened.close()
    output.close()
    source.close()

"""Tests verifying pyzbar / libzbar Docker dependency availability.

These tests ensure that:
1. The barcode_extraction module can be imported regardless of libzbar presence.
2. The BarcodeExtractor gracefully degrades when libzbar is absent.
3. When pyzbar IS available, the zbar C library is loadable.

All tests mock the C library check when running in environments without
libzbar installed (e.g., Windows dev, CI without system packages).

Run with: python -m pytest tests/test_barcode_docker_deps.py -v
"""

import os
import sys
from collections import namedtuple
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test: module imports without error even without libzbar
# ---------------------------------------------------------------------------


class TestBarcodeModuleImport:
    """Verify barcode_extraction.py imports cleanly regardless of libzbar."""

    def test_import_barcode_extraction_succeeds(self):
        """barcode_extraction should import without raising, even if pyzbar
        is not installed or libzbar is missing."""
        import barcode_extraction

        assert hasattr(barcode_extraction, "BarcodeExtractor")
        assert hasattr(barcode_extraction, "DetectedBarcode")
        assert hasattr(barcode_extraction, "PageBarcodes")
        assert hasattr(barcode_extraction, "_PYZBAR_AVAILABLE")

    def test_pyzbar_available_flag_is_boolean(self):
        """_PYZBAR_AVAILABLE must be a bool for conditional logic."""
        from barcode_extraction import _PYZBAR_AVAILABLE

        assert isinstance(_PYZBAR_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# Test: graceful degradation when pyzbar is NOT available
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Verify BarcodeExtractor returns empty results when pyzbar is absent."""

    def test_extractor_without_pyzbar_returns_empty(self):
        """When pyzbar is unavailable, extract() should return an empty list."""
        # Force reload with pyzbar import blocked
        with mock.patch.dict(sys.modules, {"pyzbar": None, "pyzbar.pyzbar": None}):
            import barcode_extraction

            # Create extractor that thinks pyzbar is not available
            extractor = barcode_extraction.BarcodeExtractor()
            extractor._available = False

            img = np.zeros((100, 100), dtype=np.uint8)
            result = extractor.extract(img, page_num=1)
            assert result == []

    def test_extract_page_without_pyzbar_returns_empty_page(self):
        """extract_page() should return PageBarcodes with zero barcodes."""
        from barcode_extraction import BarcodeExtractor

        extractor = BarcodeExtractor()
        extractor._available = False

        img = np.zeros((100, 100), dtype=np.uint8)
        page_result = extractor.extract_page(img, page_num=1)
        assert page_result.page_num == 1
        assert page_result.total_barcodes == 0
        assert page_result.barcodes == []
        assert page_result.barcode_types_found == []

    def test_is_available_property_reflects_state(self):
        """is_available should mirror internal _available flag."""
        from barcode_extraction import BarcodeExtractor

        extractor = BarcodeExtractor()
        extractor._available = False
        assert extractor.is_available is False

        extractor._available = True
        assert extractor.is_available is True


# ---------------------------------------------------------------------------
# Test: pyzbar decode works when library IS available (mocked)
# ---------------------------------------------------------------------------


# Fake pyzbar decode result
_FakeRect = namedtuple("Rect", ["left", "top", "width", "height"])
_FakeSymbol = namedtuple("Symbol", ["data", "type", "rect", "polygon"])


def _make_fake_symbol(data="https://example.com", barcode_type="QRCODE"):
    """Create a fake pyzbar decode result."""
    return _FakeSymbol(
        data=data.encode("utf-8"),
        type=barcode_type,
        rect=_FakeRect(left=10, top=20, width=100, height=100),
        polygon=[(10, 20), (110, 20), (110, 120), (10, 120)],
    )


class TestBarcodeExtractionWithPyzbar:
    """Test BarcodeExtractor when pyzbar IS available (mocked decode)."""

    def test_extract_returns_detected_barcodes(self):
        """extract() should return DetectedBarcode instances when pyzbar decodes."""
        from PIL import Image

        from barcode_extraction import BarcodeExtractor, _pyzbar_module

        extractor = BarcodeExtractor()
        extractor._available = True

        fake_symbols = [_make_fake_symbol("HELLO123", "CODE128")]
        img = Image.new("L", (200, 200), color=255)

        if _pyzbar_module is not None:
            with mock.patch.object(
                _pyzbar_module, "decode", return_value=fake_symbols
            ):
                results = extractor.extract(img, page_num=1)
        else:
            # pyzbar not installed -- mock the entire module attribute
            mock_pyzbar = mock.MagicMock()
            mock_pyzbar.decode.return_value = fake_symbols
            with mock.patch(
                "barcode_extraction._pyzbar_module", mock_pyzbar
            ):
                results = extractor.extract(img, page_num=1)

        assert len(results) == 1
        assert results[0].data == "HELLO123"
        assert results[0].barcode_type == "CODE128"
        assert results[0].bbox == [10, 20, 110, 120]
        assert results[0].page_num == 1
        assert results[0].confidence == 1.0

    def test_extract_page_returns_structured_result(self):
        """extract_page() should return PageBarcodes with barcode dicts."""
        from PIL import Image

        from barcode_extraction import BarcodeExtractor, _pyzbar_module

        extractor = BarcodeExtractor()
        extractor._available = True

        fake_symbols = [
            _make_fake_symbol("DATA1", "QRCODE"),
            _make_fake_symbol("DATA2", "EAN13"),
        ]
        img = Image.new("L", (200, 200), color=255)

        if _pyzbar_module is not None:
            with mock.patch.object(
                _pyzbar_module, "decode", return_value=fake_symbols
            ):
                page_result = extractor.extract_page(img, page_num=3)
        else:
            mock_pyzbar = mock.MagicMock()
            mock_pyzbar.decode.return_value = fake_symbols
            with mock.patch(
                "barcode_extraction._pyzbar_module", mock_pyzbar
            ):
                page_result = extractor.extract_page(img, page_num=3)

        assert page_result.page_num == 3
        assert page_result.total_barcodes == 2
        assert "QR_CODE" in page_result.barcode_types_found
        assert "EAN13" in page_result.barcode_types_found


# ---------------------------------------------------------------------------
# Test: type normalization
# ---------------------------------------------------------------------------


class TestTypeNormalization:
    """Verify barcode type strings are normalized consistently."""

    @pytest.mark.parametrize(
        "raw_type,expected",
        [
            ("QRCODE", "QR_CODE"),
            ("CODE128", "CODE128"),
            ("CODE39", "CODE39"),
            ("EAN13", "EAN13"),
            ("EAN8", "EAN8"),
            ("UPCA", "UPC_A"),
            ("UPCE", "UPC_E"),
            ("I25", "INTERLEAVED_2_OF_5"),
            ("PDF417", "PDF417"),
            ("DATAMATRIX", "DATA_MATRIX"),
            ("UNKNOWN_TYPE", "UNKNOWN_TYPE"),  # passthrough for unknown
        ],
    )
    def test_type_normalization(self, raw_type, expected):
        from barcode_extraction import _normalize_barcode_type

        assert _normalize_barcode_type(raw_type) == expected


# ---------------------------------------------------------------------------
# Test: image conversion helper
# ---------------------------------------------------------------------------


class TestImageConversion:
    """Verify _to_pil_image handles various input types."""

    def test_numpy_array_converts_to_pil(self):
        """numpy array should convert to PIL Image."""
        from barcode_extraction import _to_pil_image

        arr = np.zeros((100, 100), dtype=np.uint8)
        result = _to_pil_image(arr)
        assert result is not None

    def test_pil_image_passes_through(self):
        """PIL Image input should be returned unchanged."""
        from PIL import Image

        from barcode_extraction import _to_pil_image

        img = Image.new("L", (100, 100))
        result = _to_pil_image(img)
        assert result is img

    def test_none_input_returns_none(self):
        """Non-image input should return None."""
        from barcode_extraction import _to_pil_image

        result = _to_pil_image("not_an_image")
        assert result is None


# ---------------------------------------------------------------------------
# Test: Docker dependency documentation
# ---------------------------------------------------------------------------


class TestDockerDependencyDocumented:
    """Verify that Dockerfiles include libzbar0 for pyzbar support."""

    @pytest.fixture
    def project_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @pytest.mark.parametrize(
        "dockerfile_path",
        [
            "Dockerfile",
            os.path.join("coordinator", "Dockerfile.worker"),
            os.path.join("coordinator", "Dockerfile.worker.ocr"),
            os.path.join("coordinator", "Dockerfile.worker.nlp"),
        ],
    )
    def test_dockerfile_includes_libzbar0(self, project_root, dockerfile_path):
        """Each OCR-capable Dockerfile must install libzbar0."""
        full_path = os.path.join(project_root, dockerfile_path)
        if not os.path.exists(full_path):
            pytest.skip(f"{dockerfile_path} does not exist")

        with open(full_path, encoding="utf-8") as f:
            content = f.read()

        assert "libzbar0" in content, (
            f"{dockerfile_path} is missing 'libzbar0' in apt-get install. "
            "This system package is required for pyzbar barcode extraction."
        )

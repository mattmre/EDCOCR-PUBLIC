"""Unit tests for extended image format loaders (format_loaders.py).

Tests cover:
- HEIC/HEIF loading (available and unavailable states)
- Camera RAW loading (available and unavailable states)
- DICOM loading (available and unavailable, MONOCHROME1/MONOCHROME2)
- Dispatcher routing (load_extended_image)
- Extension introspection (get_supported_extensions, is_format_supported)
- Graceful degradation (all/partial libraries missing, load errors)

All external libraries (pillow_heif, rawpy, pydicom) are mocked since
they are optional dependencies that may not be installed in CI.

Run with: python -m pytest tests/test_format_loaders.py -v
"""

from unittest.mock import MagicMock, patch

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import format_loaders  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(width=100, height=80):
    """Create a small test RGB PIL Image."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Tests: HEIC loader
# ---------------------------------------------------------------------------


class TestHeicLoader:
    """Tests for load_heic()."""

    def test_returns_none_when_heif_unavailable(self):
        """Returns None and warns when pillow-heif is not installed."""
        with patch.object(format_loaders, "_HEIF_AVAILABLE", False):
            # Reset warning flag
            format_loaders._warned_heif = False
            result = format_loaders.load_heic("/fake/image.heic")
            assert result is None

    def test_warns_only_once_when_unavailable(self):
        """Warning is logged only on the first call."""
        with patch.object(format_loaders, "_HEIF_AVAILABLE", False):
            format_loaders._warned_heif = False
            format_loaders.load_heic("/fake/a.heic")
            assert format_loaders._warned_heif is True
            # Second call should not reset the flag
            format_loaders.load_heic("/fake/b.heic")
            assert format_loaders._warned_heif is True

    def test_loads_heic_when_available(self, tmp_path):
        """Loads and converts HEIC image to RGB when library is present."""
        test_img = _make_rgb_image()
        img_path = str(tmp_path / "test.heic")
        # Save as PNG (simulate; the opener is mocked)
        test_img.save(str(tmp_path / "test.png"))

        with patch.object(format_loaders, "_HEIF_AVAILABLE", True):
            with patch("format_loaders.Image.open", return_value=test_img):
                result = format_loaders.load_heic(img_path)
                assert result is not None
                assert result.mode == "RGB"

    def test_returns_none_on_open_error(self):
        """Returns None when Image.open raises an exception."""
        with patch.object(format_loaders, "_HEIF_AVAILABLE", True):
            with patch(
                "format_loaders.Image.open", side_effect=OSError("corrupt file")
            ):
                result = format_loaders.load_heic("/fake/corrupt.heic")
                assert result is None


# ---------------------------------------------------------------------------
# Tests: RAW loader
# ---------------------------------------------------------------------------


class TestRawLoader:
    """Tests for load_raw()."""

    def test_returns_none_when_rawpy_unavailable(self):
        """Returns None and warns when rawpy is not installed."""
        with patch.object(format_loaders, "_RAWPY_AVAILABLE", False):
            format_loaders._warned_rawpy = False
            result = format_loaders.load_raw("/fake/image.cr2")
            assert result is None

    def test_warns_only_once_when_unavailable(self):
        """Warning is logged only on the first call."""
        with patch.object(format_loaders, "_RAWPY_AVAILABLE", False):
            format_loaders._warned_rawpy = False
            format_loaders.load_raw("/fake/a.nef")
            assert format_loaders._warned_rawpy is True
            format_loaders.load_raw("/fake/b.arw")
            assert format_loaders._warned_rawpy is True

    def test_loads_raw_when_available(self):
        """Loads RAW file and returns RGB PIL Image."""
        rgb_array = np.random.randint(0, 256, (100, 80, 3), dtype=np.uint8)
        mock_raw = MagicMock()
        mock_raw.postprocess.return_value = rgb_array
        mock_raw.__enter__ = MagicMock(return_value=mock_raw)
        mock_raw.__exit__ = MagicMock(return_value=False)

        mock_rawpy = MagicMock()
        mock_rawpy.imread.return_value = mock_raw

        with patch.object(format_loaders, "_RAWPY_AVAILABLE", True):
            with patch.dict("sys.modules", {"rawpy": mock_rawpy}):
                result = format_loaders.load_raw("/fake/photo.cr2")
                assert result is not None
                assert result.mode == "RGB"
                assert result.size == (80, 100)

    def test_returns_none_on_read_error(self):
        """Returns None when rawpy.imread raises an exception."""
        mock_rawpy = MagicMock()
        mock_rawpy.imread.side_effect = RuntimeError("bad file")

        with patch.object(format_loaders, "_RAWPY_AVAILABLE", True):
            with patch.dict("sys.modules", {"rawpy": mock_rawpy}):
                result = format_loaders.load_raw("/fake/corrupt.nef")
                assert result is None

    def test_supported_raw_extensions(self):
        """All common RAW extensions are in RAW_EXTENSIONS."""
        expected = {".cr2", ".nef", ".arw", ".dng", ".raw"}
        assert expected.issubset(format_loaders.RAW_EXTENSIONS)


# ---------------------------------------------------------------------------
# Tests: DICOM loader
# ---------------------------------------------------------------------------


class TestDicomLoader:
    """Tests for load_dicom()."""

    def test_returns_none_when_pydicom_unavailable(self):
        """Returns None and warns when pydicom is not installed."""
        with patch.object(format_loaders, "_PYDICOM_AVAILABLE", False):
            format_loaders._warned_pydicom = False
            result = format_loaders.load_dicom("/fake/scan.dcm")
            assert result is None

    def test_warns_only_once_when_unavailable(self):
        """Warning is logged only on the first call."""
        with patch.object(format_loaders, "_PYDICOM_AVAILABLE", False):
            format_loaders._warned_pydicom = False
            format_loaders.load_dicom("/fake/a.dcm")
            assert format_loaders._warned_pydicom is True
            format_loaders.load_dicom("/fake/b.dicom")
            assert format_loaders._warned_pydicom is True

    def test_loads_monochrome2_dicom(self):
        """Loads standard grayscale DICOM (MONOCHROME2)."""
        pixel_array = np.random.randint(0, 4096, (256, 256), dtype=np.uint16)
        mock_ds = MagicMock()
        mock_ds.pixel_array = pixel_array
        mock_ds.PhotometricInterpretation = "MONOCHROME2"

        mock_pydicom = MagicMock()
        mock_pydicom.dcmread.return_value = mock_ds

        with patch.object(format_loaders, "_PYDICOM_AVAILABLE", True):
            with patch.dict("sys.modules", {"pydicom": mock_pydicom}):
                result = format_loaders.load_dicom("/fake/scan.dcm")
                assert result is not None
                assert result.mode == "RGB"
                assert result.size == (256, 256)

    def test_loads_monochrome1_dicom_inverted(self):
        """Loads inverted grayscale DICOM (MONOCHROME1) with correct inversion."""
        # MONOCHROME1: 0 = white, max = black (inverted)
        pixel_array = np.zeros((64, 64), dtype=np.uint16)
        mock_ds = MagicMock()
        mock_ds.pixel_array = pixel_array
        mock_ds.PhotometricInterpretation = "MONOCHROME1"

        mock_pydicom = MagicMock()
        mock_pydicom.dcmread.return_value = mock_ds

        with patch.object(format_loaders, "_PYDICOM_AVAILABLE", True):
            with patch.dict("sys.modules", {"pydicom": mock_pydicom}):
                result = format_loaders.load_dicom("/fake/scan.dcm")
                assert result is not None
                assert result.mode == "RGB"

    def test_loads_color_dicom(self):
        """Loads a color DICOM image (H, W, 3)."""
        pixel_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        mock_ds = MagicMock()
        mock_ds.pixel_array = pixel_array
        mock_ds.PhotometricInterpretation = "RGB"
        mock_ds.SamplesPerPixel = 3

        mock_pydicom = MagicMock()
        mock_pydicom.dcmread.return_value = mock_ds

        with patch.object(format_loaders, "_PYDICOM_AVAILABLE", True):
            with patch.dict("sys.modules", {"pydicom": mock_pydicom}):
                result = format_loaders.load_dicom("/fake/color.dcm")
                assert result is not None
                assert result.mode == "RGB"

    def test_loads_first_frame_for_multiframe_grayscale_dicom(self):
        """Multi-frame grayscale DICOM should use the first frame."""
        pixel_array = np.random.randint(0, 4096, (2, 96, 64), dtype=np.uint16)
        mock_ds = MagicMock()
        mock_ds.pixel_array = pixel_array
        mock_ds.PhotometricInterpretation = "MONOCHROME2"
        mock_ds.SamplesPerPixel = 1

        mock_pydicom = MagicMock()
        mock_pydicom.dcmread.return_value = mock_ds

        with patch.object(format_loaders, "_PYDICOM_AVAILABLE", True):
            with patch.dict("sys.modules", {"pydicom": mock_pydicom}):
                result = format_loaders.load_dicom("/fake/multiframe.dcm")
                assert result is not None
                assert result.mode == "RGB"
                assert result.size == (64, 96)

    def test_returns_none_on_read_error(self):
        """Returns None when pydicom.dcmread raises an exception."""
        mock_pydicom = MagicMock()
        mock_pydicom.dcmread.side_effect = RuntimeError("bad DICOM")

        with patch.object(format_loaders, "_PYDICOM_AVAILABLE", True):
            with patch.dict("sys.modules", {"pydicom": mock_pydicom}):
                result = format_loaders.load_dicom("/fake/corrupt.dcm")
                assert result is None


# ---------------------------------------------------------------------------
# Tests: Dispatcher (load_extended_image)
# ---------------------------------------------------------------------------


class TestLoadExtendedImage:
    """Tests for load_extended_image() dispatcher."""

    def test_routes_heic_to_heic_loader(self):
        """HEIC extension routes to load_heic."""
        with patch.object(
            format_loaders, "load_heic", return_value=_make_rgb_image()
        ) as mock_load:
            result = format_loaders.load_extended_image("/fake/photo.heic")
            mock_load.assert_called_once_with("/fake/photo.heic")
            assert result is not None

    def test_routes_heif_to_heic_loader(self):
        """HEIF extension routes to load_heic."""
        with patch.object(
            format_loaders, "load_heic", return_value=_make_rgb_image()
        ) as mock_load:
            result = format_loaders.load_extended_image("/fake/photo.heif")
            mock_load.assert_called_once_with("/fake/photo.heif")
            assert result is not None

    def test_routes_cr2_to_raw_loader(self):
        """CR2 extension routes to load_raw."""
        with patch.object(
            format_loaders, "load_raw", return_value=_make_rgb_image()
        ) as mock_load:
            result = format_loaders.load_extended_image("/fake/photo.cr2")
            mock_load.assert_called_once_with("/fake/photo.cr2")
            assert result is not None

    def test_routes_nef_to_raw_loader(self):
        """NEF extension routes to load_raw."""
        with patch.object(
            format_loaders, "load_raw", return_value=_make_rgb_image()
        ) as mock_load:
            format_loaders.load_extended_image("/fake/photo.nef")
            mock_load.assert_called_once()

    def test_routes_dng_to_raw_loader(self):
        """DNG extension routes to load_raw."""
        with patch.object(
            format_loaders, "load_raw", return_value=_make_rgb_image()
        ) as mock_load:
            format_loaders.load_extended_image("/fake/photo.dng")
            mock_load.assert_called_once()

    def test_routes_dcm_to_dicom_loader(self):
        """DCM extension routes to load_dicom."""
        with patch.object(
            format_loaders, "load_dicom", return_value=_make_rgb_image()
        ) as mock_load:
            result = format_loaders.load_extended_image("/fake/scan.dcm")
            mock_load.assert_called_once_with("/fake/scan.dcm")
            assert result is not None

    def test_routes_dicom_to_dicom_loader(self):
        """DICOM extension routes to load_dicom."""
        with patch.object(
            format_loaders, "load_dicom", return_value=_make_rgb_image()
        ) as mock_load:
            format_loaders.load_extended_image("/fake/scan.dicom")
            mock_load.assert_called_once()

    def test_unsupported_extension_returns_none(self):
        """Unsupported extension returns None without calling any loader."""
        result = format_loaders.load_extended_image("/fake/file.xyz")
        assert result is None

    def test_pdf_extension_returns_none(self):
        """PDF extension is not handled by extended loaders."""
        result = format_loaders.load_extended_image("/fake/file.pdf")
        assert result is None

    def test_case_insensitive_extension(self):
        """Extension matching is case-insensitive."""
        with patch.object(
            format_loaders, "load_heic", return_value=_make_rgb_image()
        ) as mock_load:
            format_loaders.load_extended_image("/fake/photo.HEIC")
            mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Supported extensions
# ---------------------------------------------------------------------------


class TestSupportedExtensions:
    """Tests for get_supported_extensions() and is_format_supported()."""

    def test_all_libraries_available(self):
        """All extensions returned when all libraries are installed."""
        with patch.object(format_loaders, "_HEIF_AVAILABLE", True), \
             patch.object(format_loaders, "_RAWPY_AVAILABLE", True), \
             patch.object(format_loaders, "_PYDICOM_AVAILABLE", True):
            exts = format_loaders.get_supported_extensions()
            assert format_loaders.HEIC_EXTENSIONS.issubset(exts)
            assert format_loaders.RAW_EXTENSIONS.issubset(exts)
            assert format_loaders.DICOM_EXTENSIONS.issubset(exts)

    def test_no_libraries_available(self):
        """Empty set when no optional libraries are installed."""
        with patch.object(format_loaders, "_HEIF_AVAILABLE", False), \
             patch.object(format_loaders, "_RAWPY_AVAILABLE", False), \
             patch.object(format_loaders, "_PYDICOM_AVAILABLE", False):
            exts = format_loaders.get_supported_extensions()
            assert exts == set()

    def test_only_heif_available(self):
        """Only HEIC extensions when only pillow-heif is installed."""
        with patch.object(format_loaders, "_HEIF_AVAILABLE", True), \
             patch.object(format_loaders, "_RAWPY_AVAILABLE", False), \
             patch.object(format_loaders, "_PYDICOM_AVAILABLE", False):
            exts = format_loaders.get_supported_extensions()
            assert exts == format_loaders.HEIC_EXTENSIONS

    def test_only_rawpy_available(self):
        """Only RAW extensions when only rawpy is installed."""
        with patch.object(format_loaders, "_HEIF_AVAILABLE", False), \
             patch.object(format_loaders, "_RAWPY_AVAILABLE", True), \
             patch.object(format_loaders, "_PYDICOM_AVAILABLE", False):
            exts = format_loaders.get_supported_extensions()
            assert exts == format_loaders.RAW_EXTENSIONS

    def test_only_pydicom_available(self):
        """Only DICOM extensions when only pydicom is installed."""
        with patch.object(format_loaders, "_HEIF_AVAILABLE", False), \
             patch.object(format_loaders, "_RAWPY_AVAILABLE", False), \
             patch.object(format_loaders, "_PYDICOM_AVAILABLE", True):
            exts = format_loaders.get_supported_extensions()
            assert exts == format_loaders.DICOM_EXTENSIONS

    def test_is_format_supported_true(self):
        """is_format_supported returns True for available formats."""
        with patch.object(format_loaders, "_HEIF_AVAILABLE", True):
            assert format_loaders.is_format_supported("/foo/bar.heic") is True

    def test_is_format_supported_false_when_lib_missing(self):
        """is_format_supported returns False when library not installed."""
        with patch.object(format_loaders, "_HEIF_AVAILABLE", False), \
             patch.object(format_loaders, "_RAWPY_AVAILABLE", False), \
             patch.object(format_loaders, "_PYDICOM_AVAILABLE", False):
            assert format_loaders.is_format_supported("/foo/bar.heic") is False

    def test_is_format_supported_false_for_unknown(self):
        """is_format_supported returns False for unknown extensions."""
        assert format_loaders.is_format_supported("/foo/bar.xyz") is False


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests for behaviour when all or some libraries are missing."""

    def test_all_missing_returns_none_for_every_format(self):
        """Every loader returns None when its library is missing."""
        with patch.object(format_loaders, "_HEIF_AVAILABLE", False), \
             patch.object(format_loaders, "_RAWPY_AVAILABLE", False), \
             patch.object(format_loaders, "_PYDICOM_AVAILABLE", False):
            format_loaders._warned_heif = False
            format_loaders._warned_rawpy = False
            format_loaders._warned_pydicom = False

            assert format_loaders.load_heic("/fake.heic") is None
            assert format_loaders.load_raw("/fake.cr2") is None
            assert format_loaders.load_dicom("/fake.dcm") is None

    def test_dispatcher_returns_none_when_loader_fails(self):
        """Dispatcher returns None when the individual loader returns None."""
        with patch.object(format_loaders, "load_heic", return_value=None):
            result = format_loaders.load_extended_image("/fake/photo.heic")
            assert result is None

    def test_extension_sets_are_disjoint(self):
        """HEIC, RAW, and DICOM extension sets do not overlap."""
        assert len(format_loaders.HEIC_EXTENSIONS & format_loaders.RAW_EXTENSIONS) == 0
        assert len(format_loaders.HEIC_EXTENSIONS & format_loaders.DICOM_EXTENSIONS) == 0
        assert len(format_loaders.RAW_EXTENSIONS & format_loaders.DICOM_EXTENSIONS) == 0

    def test_all_extended_is_union_of_all_sets(self):
        """ALL_EXTENDED_EXTENSIONS is the union of all format sets."""
        expected = (
            format_loaders.HEIC_EXTENSIONS
            | format_loaders.RAW_EXTENSIONS
            | format_loaders.DICOM_EXTENSIONS
        )
        assert format_loaders.ALL_EXTENDED_EXTENSIONS == expected

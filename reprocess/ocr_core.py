"""Standalone OCR engine with 3-tier cascade."""

import logging
from typing import Optional

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


logger = logging.getLogger(__name__)


class OCREngine:
    """OCR engine with fallback cascade: PaddleOCR → Tesseract → image-only."""

    def __init__(self, use_paddle: bool = True, lang: str = "en"):
        """Initialize OCR engine.

        Args:
            use_paddle: Whether to attempt PaddleOCR
            lang: Language code for OCR
        """
        self.use_paddle = use_paddle
        self.lang = lang
        self._paddle_ocr: Optional[object] = None
        self._paddle_available = PaddleOCR is not None and use_paddle
        self._tesseract_available = pytesseract is not None

    def run_ocr(self, image) -> tuple[str, str]:
        """Run OCR with 3-tier cascade.

        Args:
            image: PIL Image object

        Returns:
            Tuple of (extracted_text, method_used)
        """
        # Tier 1: PaddleOCR
        if self.use_paddle and self._paddle_available:
            try:
                text = self._run_paddle(image)
                if text and text.strip():
                    return text, "paddle"
                logger.debug("PaddleOCR returned empty text")
            except Exception as e:
                logger.debug(f"PaddleOCR failed: {e}")

        # Tier 2: Tesseract
        if self._tesseract_available:
            try:
                text = self._run_tesseract(image)
                if text and text.strip():
                    return text, "tesseract"
                logger.debug("Tesseract returned empty text")
            except Exception as e:
                logger.debug(f"Tesseract failed: {e}")

        # Tier 3: Image-only fallback
        return "", "image_only"

    def _run_paddle(self, image) -> str:
        """Run PaddleOCR on image.

        Args:
            image: PIL Image object

        Returns:
            Extracted text
        """
        import numpy as np

        # Initialize once
        if self._paddle_ocr is None:
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                show_log=False,
            )

        # Convert PIL to format PaddleOCR expects (numpy array)

        img_array = np.array(image)

        # Run OCR
        result = self._paddle_ocr.ocr(img_array, cls=True)

        # Extract text from result structure
        if not result or not result[0]:
            return ""

        text_lines = []
        for line in result[0]:
            if line and len(line) >= 2:
                # line[1] is (text, confidence)
                text_lines.append(line[1][0])

        return "\n".join(text_lines)

    def _run_tesseract(self, image) -> str:
        """Run Tesseract on image.

        Args:
            image: PIL Image object

        Returns:
            Extracted text
        """
        text = pytesseract.image_to_string(image, lang=self.lang)
        return text

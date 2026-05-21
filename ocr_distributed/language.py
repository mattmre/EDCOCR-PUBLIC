"""Language detection utilities using FastText.

Encapsulates FastText language detection as a class with no global state,
suitable for both the monolithic pipeline and distributed Celery tasks.
"""

import logging
import os

import fitz  # PyMuPDF

from .constants import LANG_MAPPING

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Encapsulates FastText language detection (no global state).

    Args:
        model_path: Path to the FastText lid.176.bin model file.
            If None or the file does not exist, detection falls back to 'en'.
    """

    def __init__(self, model_path=None):
        self._model = None
        if model_path and os.path.exists(model_path):
            try:
                import fasttext
                self._model = fasttext.load_model(model_path)
                logger.info("FastText Language Model Loaded.")
            except Exception as e:
                logger.warning(f"FastText Load Failed: {e}")

    @property
    def is_loaded(self):
        """Returns True if the FastText model is available."""
        return self._model is not None

    def detect_from_pdf(self, doc_path, default='en'):
        """Detect language from the first few pages of a PDF.

        Extracts embedded text from the first 3 pages and runs FastText
        prediction. Falls back to *default* if text is too short or
        prediction fails.

        Args:
            doc_path: Path to the PDF file.
            default: Fallback language code (default 'en').

        Returns:
            PaddleOCR language code string.
        """
        if not self._model:
            return default
        try:
            with fitz.open(doc_path) as doc:
                text_sample = ""
                for i in range(min(3, doc.page_count)):
                    text_sample += doc[i].get_text()

                if len(text_sample) > 50:
                    text_sample = text_sample.replace("\n", " ")[:200]
                    prediction = self._model.predict(text_sample)
                    lang_code = prediction[0][0].replace('__label__', '')
                    return LANG_MAPPING.get(lang_code, default)
        except Exception:
            pass

        return default

    def detect_from_text(self, text, confidence_threshold=0.4):
        """Detect language from an OCR output string.

        Args:
            text: OCR text to analyze (at least 20 chars for reliable detection).
            confidence_threshold: Minimum confidence to accept a prediction.

        Returns:
            (paddle_lang_code or None, confidence_float) tuple.
        """
        if not self._model or len(text) < 20:
            return None, 0.0
        try:
            sample = text.replace("\n", " ")[:300]
            prediction = self._model.predict(sample)
            lang_code = prediction[0][0].replace('__label__', '')
            conf = prediction[1][0]
            mapped = LANG_MAPPING.get(lang_code)

            if mapped and conf > confidence_threshold:
                return mapped, conf
        except Exception:
            pass
        return None, 0.0

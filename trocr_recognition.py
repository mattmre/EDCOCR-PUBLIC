"""TrOCR-based handwriting recognition with confidence gating.

Extends the existing handwriting *detection* module (handwriting.py) with actual
text *recognition* of handwritten regions using Microsoft TrOCR. PaddleOCR text
is always preserved alongside TrOCR output for forensic integrity.

Three agreement modes control how PaddleOCR and TrOCR results are merged:
  - "verify"  (default, safest) -- TrOCR used only when it agrees with PaddleOCR
  - "trust"   -- always use TrOCR output for handwriting regions
  - "reject"  -- never use TrOCR, keep PaddleOCR text

Opt-in via ENABLE_TROCR=true environment variable.

Output: HandwritingRecognitionResult dataclasses consumed by the assembler.

Graceful degradation: if torch or transformers are not installed, the module
logs a warning and returns PaddleOCR text unchanged.
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class _TorchPlaceholder:
    """Stable patch target when torch is not installed."""

    softmax = None
    no_grad = None

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False


torch = _TorchPlaceholder()
TrOCRProcessor = None
VisionEncoderDecoderModel = None

# ---------------------------------------------------------------------------
# Configuration (env vars)
# ---------------------------------------------------------------------------

ENABLE_TROCR = os.environ.get("ENABLE_TROCR", "").lower() in ("1", "true", "yes")
TROCR_MODEL_PATH = os.environ.get("TROCR_MODEL_PATH", "microsoft/trocr-base-handwritten")
TROCR_CONFIDENCE_THRESHOLD = float(os.environ.get("TROCR_CONFIDENCE_THRESHOLD", "0.85"))
TROCR_AGREEMENT_MODE = os.environ.get("TROCR_AGREEMENT_MODE", "verify")  # verify, trust, reject
TROCR_MAX_LENGTH = int(os.environ.get("TROCR_MAX_LENGTH", "256"))

# Minimum edit-distance similarity for "verify" mode agreement
TROCR_AGREEMENT_THRESHOLD = float(os.environ.get("TROCR_AGREEMENT_THRESHOLD", "0.40"))

# ---------------------------------------------------------------------------
# Guarded imports
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False

try:
    import torch as _torch

    torch = _torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import (
        TrOCRProcessor as _TrOCRProcessor,
    )
    from transformers import (
        VisionEncoderDecoderModel as _VisionEncoderDecoderModel,
    )

    TrOCRProcessor = _TrOCRProcessor
    VisionEncoderDecoderModel = _VisionEncoderDecoderModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

_TROCR_DEPS_AVAILABLE = _TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HandwritingRecognitionResult:
    """Recognition result for a single handwriting region."""

    trocr_text: str = ""
    trocr_confidence: float = 0.0
    paddle_text: str = ""
    paddle_confidence: float = 0.0
    agreement_score: float = 0.0
    selected_text: str = ""
    selection_reason: str = ""
    bbox: list = field(default_factory=list)  # [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Agreement scoring
# ---------------------------------------------------------------------------


def compute_agreement_score(text_a: str, text_b: str) -> float:
    """Normalized edit-distance similarity between two strings.

    Returns a float in [0.0, 1.0] where 1.0 means identical.
    Uses a simple Levenshtein distance via dynamic programming.
    """
    if text_a == text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    a = text_a.lower().strip()
    b = text_b.lower().strip()

    if a == b:
        return 1.0

    len_a, len_b = len(a), len(b)
    max_len = max(len_a, len_b)
    if max_len == 0:
        return 1.0

    # Levenshtein distance via two-row DP
    prev = list(range(len_b + 1))
    curr = [0] * (len_b + 1)

    for i in range(1, len_a + 1):
        curr[0] = i
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, curr = curr, prev

    distance = prev[len_b]
    return round(1.0 - (distance / max_len), 4)


# ---------------------------------------------------------------------------
# TrOCR Recognizer
# ---------------------------------------------------------------------------


class TrOCRRecognizer:
    """Lazy-loaded TrOCR model for handwriting recognition.

    Thread-safe: model loading uses a lock. Once loaded, inference is
    safe for concurrent reads (PyTorch model.eval mode).
    """

    def __init__(self, model_path: str = TROCR_MODEL_PATH, device: Optional[str] = None):
        self._model_path = model_path
        self._device = device
        self._processor = None
        self._model = None
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()

    def _load_model(self) -> bool:
        """Load TrOCR processor and model. Returns True on success."""
        if self._loaded:
            return True
        if self._load_failed:
            return False

        with self._lock:
            # Double-check after acquiring lock
            if self._loaded:
                return True
            if self._load_failed:
                return False

            if not _TROCR_DEPS_AVAILABLE:
                logger.warning(
                    "TrOCR dependencies not available (torch=%s, transformers=%s). "
                    "Handwriting recognition will use PaddleOCR text only.",
                    _TORCH_AVAILABLE,
                    _TRANSFORMERS_AVAILABLE,
                )
                self._load_failed = True
                return False

            try:
                logger.info("Loading TrOCR model from: %s", self._model_path)
                self._processor = TrOCRProcessor.from_pretrained(self._model_path)
                self._model = VisionEncoderDecoderModel.from_pretrained(self._model_path)

                # Determine device
                if self._device is None:
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"

                self._model.to(self._device)
                self._model.eval()
                self._loaded = True
                logger.info("TrOCR model loaded on device: %s", self._device)
                return True
            except Exception as e:
                logger.error("Failed to load TrOCR model from %s: %s", self._model_path, e)
                self._load_failed = True
                return False

    @property
    def is_available(self) -> bool:
        """Check if the model can be loaded (or is already loaded)."""
        if self._loaded:
            return True
        if self._load_failed:
            return False
        return _TROCR_DEPS_AVAILABLE

    def recognize(self, image_crop) -> tuple:
        """Run TrOCR on a cropped handwriting region image.

        Args:
            image_crop: PIL Image of the handwriting region.

        Returns:
            Tuple of (text: str, confidence: float). Returns ("", 0.0) on failure.
        """
        if not self._load_model():
            return ("", 0.0)

        try:
            pixel_values = self._processor(
                images=image_crop, return_tensors="pt"
            ).pixel_values.to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    pixel_values,
                    max_length=TROCR_MAX_LENGTH,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Decode text
            generated_ids = outputs.sequences
            text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Compute confidence from scores
            confidence = _compute_sequence_confidence(outputs)

            return (text.strip(), round(confidence, 4))
        except Exception as e:
            logger.warning("TrOCR recognition failed: %s", e)
            return ("", 0.0)

    def recognize_with_gating(
        self,
        image_crop,
        paddle_text: str,
        paddle_confidence: float,
        agreement_mode: Optional[str] = None,
    ) -> HandwritingRecognitionResult:
        """Recognize handwriting with confidence gating against PaddleOCR.

        Args:
            image_crop: PIL Image of the handwriting region.
            paddle_text: Text from PaddleOCR for this region.
            paddle_confidence: PaddleOCR confidence for this region.
            agreement_mode: Override global TROCR_AGREEMENT_MODE for this call.

        Returns:
            HandwritingRecognitionResult with both texts and selection rationale.
        """
        mode = agreement_mode or TROCR_AGREEMENT_MODE

        result = HandwritingRecognitionResult(
            paddle_text=paddle_text,
            paddle_confidence=round(paddle_confidence, 4),
        )

        # Reject mode: never use TrOCR
        if mode == "reject":
            result.selected_text = paddle_text
            result.selection_reason = "reject_mode"
            return result

        # Attempt TrOCR recognition
        trocr_text, trocr_confidence = self.recognize(image_crop)
        result.trocr_text = trocr_text
        result.trocr_confidence = trocr_confidence

        # If TrOCR failed (empty result), fall back to PaddleOCR
        if not trocr_text:
            result.selected_text = paddle_text
            result.selection_reason = "trocr_empty_fallback"
            return result

        # Trust mode: always use TrOCR
        if mode == "trust":
            result.selected_text = trocr_text
            result.selection_reason = "trust_mode"
            result.agreement_score = compute_agreement_score(trocr_text, paddle_text)
            return result

        # Verify mode (default): compare TrOCR vs PaddleOCR
        agreement = compute_agreement_score(trocr_text, paddle_text)
        result.agreement_score = agreement

        if trocr_confidence >= TROCR_CONFIDENCE_THRESHOLD:
            if agreement >= TROCR_AGREEMENT_THRESHOLD:
                # High confidence + agreement: use TrOCR
                result.selected_text = trocr_text
                result.selection_reason = "verify_agreed"
            else:
                # High confidence but disagreement: TrOCR if much more confident
                if trocr_confidence > paddle_confidence + 0.2:
                    result.selected_text = trocr_text
                    result.selection_reason = "verify_trocr_higher_confidence"
                else:
                    result.selected_text = paddle_text
                    result.selection_reason = "verify_disagreed_keep_paddle"
        else:
            # Low TrOCR confidence: keep PaddleOCR
            result.selected_text = paddle_text
            result.selection_reason = "verify_low_trocr_confidence"

        return result


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_recognizer_instance: Optional[TrOCRRecognizer] = None
_recognizer_lock = threading.Lock()


def get_trocr_recognizer() -> TrOCRRecognizer:
    """Get or create the singleton TrOCR recognizer (thread-safe, lazy)."""
    global _recognizer_instance  # noqa: PLW0603
    if _recognizer_instance is not None:
        return _recognizer_instance

    with _recognizer_lock:
        if _recognizer_instance is not None:
            return _recognizer_instance
        _recognizer_instance = TrOCRRecognizer()
        return _recognizer_instance


def reset_trocr_recognizer() -> None:
    """Reset the singleton recognizer (for testing)."""
    global _recognizer_instance  # noqa: PLW0603
    with _recognizer_lock:
        _recognizer_instance = None


# ---------------------------------------------------------------------------
# Confidence computation helper
# ---------------------------------------------------------------------------


def _compute_sequence_confidence(outputs) -> float:
    """Compute average token-level confidence from generation output scores.

    Args:
        outputs: GenerateOutput with .scores attribute (tuple of logit tensors).

    Returns:
        Float confidence in [0.0, 1.0].
    """
    try:
        if not hasattr(outputs, "scores") or not outputs.scores:
            return 0.0

        probs_per_step = []
        for step_logits in outputs.scores:
            # step_logits shape: (batch_size, vocab_size)
            probs = torch.softmax(step_logits, dim=-1)
            max_prob = probs.max(dim=-1).values.item()
            probs_per_step.append(max_prob)

        if not probs_per_step:
            return 0.0

        return sum(probs_per_step) / len(probs_per_step)
    except Exception as exc:
        logger.warning("Failed to compute TrOCR sequence confidence: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# High-level region recognition
# ---------------------------------------------------------------------------


def recognize_handwriting_regions(
    page_image,
    handwriting_result,
    paddle_lines: list,
    agreement_mode: Optional[str] = None,
) -> list:
    """Recognize text in detected handwriting regions using TrOCR.

    Takes the handwriting detection result (from handwriting.py), the full page
    image, and PaddleOCR line data. Crops each handwriting region and runs TrOCR.

    Args:
        page_image: PIL Image of the full page.
        handwriting_result: PageHandwriting or dict from handwriting detection.
        paddle_lines: List of (text, confidence, [x1, y1, x2, y2]) tuples.
        agreement_mode: Override agreement mode for all regions.

    Returns:
        List of HandwritingRecognitionResult for each handwriting region.
    """
    if not ENABLE_TROCR:
        logger.debug("TrOCR disabled (ENABLE_TROCR not set). Skipping recognition.")
        return []

    # Extract regions from handwriting result
    if hasattr(handwriting_result, "handwriting_regions"):
        regions = handwriting_result.handwriting_regions
        has_hw = handwriting_result.has_handwriting
    elif isinstance(handwriting_result, dict):
        regions = handwriting_result.get("handwriting_regions", [])
        has_hw = handwriting_result.get("has_handwriting", False)
    else:
        return []

    if not has_hw or not regions:
        return []

    recognizer = get_trocr_recognizer()
    if not recognizer.is_available:
        logger.warning("TrOCR recognizer not available. Returning empty results.")
        return []

    results = []
    img_width, img_height = page_image.size if hasattr(page_image, "size") else (1, 1)

    # Build a lookup from bbox to paddle line data
    paddle_lookup = {}
    for text, confidence, bbox in paddle_lines:
        key = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        paddle_lookup[key] = (text, confidence)

    for region in regions:
        # Extract bbox from region (dict or dataclass)
        if isinstance(region, dict):
            bbox = region.get("bbox", [])
            region_text = region.get("text", "")
            region_conf = region.get("ocr_confidence", 0.0)
        else:
            bbox = getattr(region, "bbox", [])
            region_text = getattr(region, "text", "")
            region_conf = getattr(region, "ocr_confidence", 0.0)

        if len(bbox) < 4:
            continue

        # Clamp bbox to image boundaries
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(img_width, int(bbox[2]))
        y2 = min(img_height, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            continue

        # Try matching to paddle line for text/confidence
        paddle_key = (x1, y1, x2, y2)
        if paddle_key in paddle_lookup:
            paddle_text, paddle_conf = paddle_lookup[paddle_key]
        else:
            paddle_text = region_text
            paddle_conf = region_conf

        try:
            crop = page_image.crop((x1, y1, x2, y2))
            rec_result = recognizer.recognize_with_gating(
                crop, paddle_text, paddle_conf, agreement_mode
            )
            rec_result.bbox = [x1, y1, x2, y2]
            results.append(rec_result)
        except Exception as e:
            logger.warning("Failed to recognize handwriting region at %s: %s", bbox, e)
            # Preserve PaddleOCR text on failure
            fallback = HandwritingRecognitionResult(
                paddle_text=paddle_text,
                paddle_confidence=round(paddle_conf, 4),
                selected_text=paddle_text,
                selection_reason="trocr_exception_fallback",
                bbox=[x1, y1, x2, y2],
            )
            results.append(fallback)

    return results

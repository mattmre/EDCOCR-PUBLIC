"""Configuration for LayoutLMv3 Celery worker queue.

Controls model selection, device placement, and enable/disable toggle for
the ``ocr_layoutlm`` queue.  All values are read from environment variables
with sensible defaults so the worker can be deployed without a config file.

Environment Variables:
    ENABLE_LAYOUTLM (bool):
        Master toggle for LayoutLMv3 extraction tasks.  Default: ``false``.
    LAYOUTLM_MODEL_PATH (str):
        HuggingFace model ID or local path.  Default: ``microsoft/layoutlmv3-base``.
    LAYOUTLM_REGISTRY_DIR (str):
        Registry directory used when resolving ``LAYOUTLM_ACTIVE_MODEL``.
        Default: ``"./models/registry"``.
    LAYOUTLM_ACTIVE_MODEL (str):
        Active model spec as ``name:version`` or ``name``. Default: empty.
    LAYOUTLM_DEVICE (str):
        PyTorch device string (``cuda``, ``cpu``, ``cuda:1``, ...).
        Default: ``auto`` (CUDA if available, else CPU).
    LAYOUTLM_BATCH_SIZE (int):
        Maximum batch size for inference.  Default: ``1``.
    LAYOUTLM_CONFIDENCE_THRESHOLD (float):
        Minimum confidence for emitted entities.  Default: ``0.5``.
    LAYOUTLM_MAX_LENGTH (int):
        Maximum token sequence length (LayoutLMv3 limit is 512).  Default: ``512``.
    LAYOUTLM_TASK_TIMEOUT (int):
        Per-page hard timeout in seconds.  Default: ``120``.
"""

import os

# ---------------------------------------------------------------------------
# Master toggle
# ---------------------------------------------------------------------------

ENABLE_LAYOUTLM: bool = os.environ.get(
    "ENABLE_LAYOUTLM", "false"
).lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

LAYOUTLM_MODEL_PATH: str = os.environ.get(
    "LAYOUTLM_MODEL_PATH", "microsoft/layoutlmv3-base"
)

LAYOUTLM_REGISTRY_DIR: str = os.environ.get(
    "LAYOUTLM_REGISTRY_DIR", "./models/registry"
)

LAYOUTLM_ACTIVE_MODEL: str = os.environ.get("LAYOUTLM_ACTIVE_MODEL", "")

LAYOUTLM_DEVICE: str = os.environ.get("LAYOUTLM_DEVICE", "auto")

LAYOUTLM_BATCH_SIZE: int = max(
    1, int(os.environ.get("LAYOUTLM_BATCH_SIZE", "1"))
)

LAYOUTLM_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("LAYOUTLM_CONFIDENCE_THRESHOLD", "0.5")
)

LAYOUTLM_MAX_LENGTH: int = min(
    512, max(1, int(os.environ.get("LAYOUTLM_MAX_LENGTH", "512")))
)

LAYOUTLM_TASK_TIMEOUT: int = max(
    10, int(os.environ.get("LAYOUTLM_TASK_TIMEOUT", "120"))
)

# ---------------------------------------------------------------------------
# Queue name
# ---------------------------------------------------------------------------

LAYOUTLM_QUEUE: str = "ocr_layoutlm"

# ---------------------------------------------------------------------------
# BIO entity label mapping (mirrors semantic_extraction.py)
# ---------------------------------------------------------------------------

LAYOUTLM_ENTITY_LABELS = [
    "O",
    "B-INVOICE_NUMBER", "I-INVOICE_NUMBER",
    "B-DATE", "I-DATE",
    "B-AMOUNT", "I-AMOUNT",
    "B-PERSON_NAME", "I-PERSON_NAME",
    "B-ORGANIZATION", "I-ORGANIZATION",
    "B-ADDRESS", "I-ADDRESS",
    "B-REFERENCE_NUMBER", "I-REFERENCE_NUMBER",
    "B-PHONE_NUMBER", "I-PHONE_NUMBER",
    "B-EMAIL", "I-EMAIL",
]

LAYOUTLM_LABEL2ID = {label: idx for idx, label in enumerate(LAYOUTLM_ENTITY_LABELS)}
LAYOUTLM_ID2LABEL = {idx: label for idx, label in enumerate(LAYOUTLM_ENTITY_LABELS)}

"""Fail-closed runtime controls for optional OCR training/video surfaces.

The OCR worker admits images and PDFs, not video streams or augmentation
training jobs. OpenCV wheels nevertheless bundle FFmpeg and PaddleOCR 2.9.1
depends on imgaug 0.4.0. Disable those unadmitted entry points before worker
application imports so scanner dispositions are backed by executable controls.
"""

from __future__ import annotations

import os


class DisabledOcrWorkerCapability(RuntimeError):
    """Raised when code attempts an explicitly unadmitted worker capability."""


def _blocked_capability(*_args: object, **_kwargs: object) -> None:
    raise DisabledOcrWorkerCapability(
        "video decoding and imgaug background multiprocessing are disabled "
        "in the OCR processing worker"
    )


if os.environ.get("OCR_HARDEN_MEDIA_RUNTIME") == "1":
    import cv2
    import imgaug.multicore

    cv2.VideoCapture = _blocked_capability
    cv2.VideoWriter = _blocked_capability
    cv2.VideoWriter_fourcc = _blocked_capability

    class _DisabledBackgroundAugmenter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            _blocked_capability()

    imgaug.multicore.BackgroundAugmenter = _DisabledBackgroundAugmenter

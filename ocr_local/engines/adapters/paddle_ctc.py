"""PaddleOCR CTC adapter -- the forensic default engine.

Wraps the production `create_paddle_engine` + `extract_paddle_lines`
(`ocr_distributed.ocr_utils`) behind the :class:`OCREngine` ABC so the existing
zero-generative-surface path is reachable through the registry **without
changing its behavior**: `run()` returns exactly `extract_paddle_lines(engine.ocr(img))`,
the same lines the production worker feeds to `insert_text_line`.

Heavy deps (paddleocr, fitz via ocr_utils) are imported lazily inside methods so
`import ocr_local.engines.adapters` stays cheap and dependency-free.
"""
from __future__ import annotations

import threading
from typing import Any

from ocr_local.engines import catalog
from ocr_local.engines.base import OCREngine, OcrLine
from ocr_local.engines.capability import EngineCapability


class PaddleCTCEngine(OCREngine):
    """Adapter over the production PaddleOCR CTC engine (forensic default)."""

    def __init__(self) -> None:
        # Per-(lang, device) engine cache with an inference lock, mirroring the
        # semantics of get_or_create_engine in ocr_gpu_async.py.
        self._cache: dict[tuple[str, str], tuple[Any, Any]] = {}
        self._cache_lock = threading.RLock()

    @property
    def capability(self) -> EngineCapability:
        return catalog.PADDLE_CTC

    def is_available(self) -> bool:
        try:
            import paddleocr  # noqa: F401

            return True
        except Exception:
            return False

    def get_handle(self, lang_code: str, device: str = "gpu:0") -> tuple[Any, Any]:
        # create_paddle_engine takes device "gpu"/"cpu"; normalize "gpu:0" etc.
        dev = "cpu" if str(device).lower().startswith("cpu") else "gpu"
        key = (lang_code or "en", dev)
        with self._cache_lock:
            handle = self._cache.get(key)
            if handle is None:
                from ocr_distributed.ocr_utils import create_paddle_engine

                engine = create_paddle_engine(key[0], device=dev)
                handle = (engine, threading.Lock())
                self._cache[key] = handle
            return handle

    def run(self, handle: Any, image: Any, lang_code: str) -> list[OcrLine]:
        from ocr_distributed.ocr_utils import extract_paddle_lines

        engine, lock = handle if isinstance(handle, tuple) else (handle, None)
        if lock is not None:
            with lock:
                result = engine.ocr(image)
        else:
            result = engine.ocr(image)
        return extract_paddle_lines(result)

    def evict(self, lang_code: str, device: str = "gpu:0") -> None:
        """Drop a cached engine (e.g. after an OOM), mirroring _ENGINE_CACHE eviction."""
        dev = "cpu" if str(device).lower().startswith("cpu") else "gpu"
        with self._cache_lock:
            self._cache.pop((lang_code or "en", dev), None)

"""Pre-download PaddleOCR models for air-gapped deployment.

Downloads and caches PaddleOCR models for all target languages so the
Docker image can operate without internet access.

Supports three modes:
- **GPU build** (default): ``python download_models.py``
- **CPU-only build**: ``python download_models.py --cpu-only``
  Forces ``use_gpu=False`` and ``use_onnx=True`` to avoid CUDA probe
  segfault (exit 139) on environments without GPU drivers.
- **Skip entirely**: ``SKIP_MODEL_PRELOAD=1 python download_models.py``
  Exits immediately without downloading anything.

The ``CPU_ONLY_BUILD=1`` environment variable is equivalent to ``--cpu-only``.
"""

import argparse
import logging
import os
import sys

# Force CPU mode for downloading to avoid libcuda.so errors during build.
# MUST be set before importing paddle/paddleocr.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelDownloader")

# Environment variable controls
SKIP_MODEL_PRELOAD = os.environ.get("SKIP_MODEL_PRELOAD", "0") == "1"
CPU_ONLY_BUILD = os.environ.get("CPU_ONLY_BUILD", "0") == "1"

# List of languages to pre-download
# Comprehensive list based on PaddleOCR support
# Step 1: CJK & Common
# Step 2: European & Others
LANGUAGES = [
    'ch', 'en', 'french', 'german', 'korean', 'japan', 'chinese_cht', 'te', 'ka', 'ta', 'kn',
    'ru', 'ar', 'hi', 'ug', 'fa', 'ur', 'rs_latin', 'oc', 'russia', 'sh',
    'uk', 'be', 'bg', 'cs', 'pl', 'tr', 'nl', 'sv', 'da', 'fi', 'el', 'hu', 'ro',
    'it', 'es', 'pt', 'vi'
]
# Note: 'russia' might be 'ru'. Paddle uses specific codes. Standardizing on the ones we mapped.

# Refined list matching LANG_MAPPING.
# Prefer the consolidated registry; keep inline fallback for air-gapped builds
# where language_config.py may not be on the Python path.
try:
    from ocr_local.config.language_config import TARGET_LANGS
except ImportError:
    TARGET_LANGS = [
        'ch', 'chinese_cht', 'japan', 'korean', 'vi',
        'en', 'fr', 'german', 'es', 'it', 'pt', 'ru', 'ar', 'hi',
        'uk', 'be', 'bg', 'cs', 'pl', 'tr', 'nl', 'sv', 'da',
        'fi', 'el', 'hu', 'ro',
        'fa', 'ur', 'ug', 'te', 'ta', 'kn', 'ka',
    ]


def _import_paddleocr():
    """Lazily import PaddleOCR to avoid CUDA probe at module load time.

    Returns
    -------
    type or None
        The PaddleOCR class, or None if paddleocr is not installed.
    """
    try:
        from paddleocr import PaddleOCR

        return PaddleOCR
    except ImportError:
        logger.warning("paddleocr is not installed -- skipping model download")
        return None


def download_models(cpu_only=False):
    """Download and cache PaddleOCR models for all target languages.

    Parameters
    ----------
    cpu_only : bool
        When True, forces ``use_gpu=False`` and ``use_onnx=True`` to
        prevent CUDA probe segfault on CPU-only build environments.
        Default behaviour (False) uses ``use_gpu=False`` during the
        download stage (CUDA_VISIBLE_DEVICES is already blanked) but
        does **not** set ``use_onnx``.
    """
    if SKIP_MODEL_PRELOAD:
        logger.info("SKIP_MODEL_PRELOAD=1 -- skipping model download")
        return

    PaddleOCR = _import_paddleocr()
    if PaddleOCR is None:
        return

    mode_label = "CPU-only (use_onnx=True)" if cpu_only else "standard"
    logger.info(
        "--- Downloading OCR Models for %d Languages [mode: %s] ---",
        len(TARGET_LANGS),
        mode_label,
    )
    failures = []

    for lang in TARGET_LANGS:
        logger.info("Downloading: %s ...", lang)
        try:
            kwargs = {
                "use_angle_cls": True,
                "lang": lang,
                "use_gpu": False,
                "show_log": False,
            }
            if cpu_only:
                kwargs["use_onnx"] = True

            ocr_engine = PaddleOCR(**kwargs)
            del ocr_engine
            logger.info("Successfully downloaded: %s", lang)
        except Exception as e:
            logger.warning("Failed to download model for %s: %s", lang, e)
            failures.append((lang, str(e)))

    if failures:
        logger.error("--- Model download failures detected ---")
        for lang, err in failures:
            logger.error("  - %s: %s", lang, err)
        raise RuntimeError(
            f"Model preload failed for {len(failures)} language(s)"
        )

    logger.info("--- Model preload completed successfully ---")


def download_uie_model():
    """Pre-download PaddleNLP UIE model for structured extraction (Phase 6C).

    The model name is controlled by the UIE_MODEL env var (default: uie-base).
    Falls back gracefully if paddlenlp is not installed.
    """
    if SKIP_MODEL_PRELOAD:
        logger.info("SKIP_MODEL_PRELOAD=1 -- skipping UIE model download")
        return

    model_name = os.environ.get("UIE_MODEL", "uie-base")
    try:
        from paddlenlp import Taskflow  # noqa: F811

        logger.info("--- Downloading UIE model: %s ---", model_name)
        engine = Taskflow(
            "information_extraction",
            schema=["Date"],
            model=model_name,
            device="cpu",
        )
        del engine
        logger.info("--- UIE model '%s' downloaded successfully ---", model_name)
    except ImportError:
        logger.info("paddlenlp not installed -- skipping UIE model download")
    except Exception as e:
        logger.warning("UIE model download failed: %s", e)


def _parse_args(argv=None):
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] or None
        Argument list for testing; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Download and cache PaddleOCR models for air-gapped deployment",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help=(
            "Force CPU mode: use_gpu=False and use_onnx=True. "
            "Prevents CUDA probe segfault (exit 139) on CPU-only builds. "
            "Equivalent to setting CPU_ONLY_BUILD=1."
        ),
    )
    parser.add_argument(
        "--skip-uie",
        action="store_true",
        default=False,
        help="Skip UIE model download (PaddleNLP Taskflow).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    cpu_only = args.cpu_only or CPU_ONLY_BUILD

    try:
        download_models(cpu_only=cpu_only)
    except RuntimeError:
        # download_models already logged the failures; exit non-zero
        # but do not re-print the traceback.
        sys.exit(1)

    if not args.skip_uie:
        download_uie_model()

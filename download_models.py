"""Pre-download PaddleOCR models for air-gapped deployment.

Downloads and caches PaddleOCR models for all target languages so the
Docker image can operate without internet access.

Supports three modes:
- **GPU build** (default): ``python download_models.py``
- **CPU-only build**: ``python download_models.py --cpu-only``
  Forces ``use_gpu=False`` while leaving PaddleOCR in its native model format.
  ``CUDA_VISIBLE_DEVICES`` is blanked before Paddle imports, so no CUDA device
  is selected. ONNX mode cannot preload freshly downloaded Paddle models.
- **Skip entirely**: ``SKIP_MODEL_PRELOAD=1 python download_models.py``
  Exits immediately without downloading anything.

The ``CPU_ONLY_BUILD=1`` environment variable is equivalent to ``--cpu-only``.
"""

import argparse
import hashlib
import importlib.util
import logging
import os
import sys
from pathlib import Path

# Force CPU mode for downloading to avoid libcuda.so errors during build.
# MUST be set before importing paddle/paddleocr.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelDownloader")

# Environment variable controls
SKIP_MODEL_PRELOAD = os.environ.get("SKIP_MODEL_PRELOAD", "0") == "1"
CPU_ONLY_BUILD = os.environ.get("CPU_ONLY_BUILD", "0") == "1"
# S2-02: optional SHA-256 pin file (one "sha256  relative/or/abs/path" per line,
# or "sha256 *filename" GNU style). When set, verify after download; missing
# file under REQUIRE_MODEL_CHECKSUMS=true fails closed.
MODEL_CHECKSUM_MANIFEST = os.environ.get("MODEL_CHECKSUM_MANIFEST", "").strip()
REQUIRE_MODEL_CHECKSUMS = os.environ.get("REQUIRE_MODEL_CHECKSUMS", "").lower() in (
    "1",
    "true",
    "yes",
)

# PaddleOCR 2.9.1 maps most language codes to model families internally.  It
# does not ship recognition models for these scripts; the qualified runtime
# routes them to language-specific Tesseract models instead.
PADDLEOCR_29_UNSUPPORTED_LANGS = frozenset({"bn", "el", "kn", "th"})
PADDLEOCR_29_RUNTIME_ALIASES = {"fi": "latin"}

# List of languages to pre-download
# Comprehensive list based on PaddleOCR support
# Step 1: CJK & Common
# Step 2: European & Others
LANGUAGES = [
    "ch",
    "en",
    "french",
    "german",
    "korean",
    "japan",
    "chinese_cht",
    "te",
    "ka",
    "ta",
    "kn",
    "ru",
    "ar",
    "hi",
    "ug",
    "fa",
    "ur",
    "rs_latin",
    "oc",
    "russia",
    "sh",
    "uk",
    "be",
    "bg",
    "cs",
    "pl",
    "tr",
    "nl",
    "sv",
    "da",
    "fi",
    "el",
    "hu",
    "ro",
    "it",
    "es",
    "pt",
    "vi",
]
# Note: 'russia' might be 'ru'. Paddle uses specific codes. Standardizing on the ones we mapped.

# Refined list matching LANG_MAPPING.
# Prefer the consolidated registry; keep inline fallback for air-gapped builds
# where language_config.py may not be on the Python path.
try:
    from ocr_local.config.language_config import TARGET_LANGS
except ImportError:
    TARGET_LANGS = [
        "ch",
        "chinese_cht",
        "japan",
        "korean",
        "vi",
        "en",
        "fr",
        "german",
        "es",
        "it",
        "pt",
        "ru",
        "ar",
        "hi",
        "uk",
        "be",
        "bg",
        "cs",
        "pl",
        "tr",
        "nl",
        "sv",
        "da",
        "fi",
        "el",
        "hu",
        "ro",
        "fa",
        "ur",
        "ug",
        "te",
        "ta",
        "kn",
        "ka",
    ]


def _preload_target_langs(cpu_only=False, requested_langs=None):
    """Return selected language codes safe for the selected preload mode."""
    if requested_langs is None:
        candidates = list(TARGET_LANGS)
    else:
        candidates = list(dict.fromkeys(requested_langs))
        if not candidates:
            raise ValueError("language selection must not be empty")
        unknown = [lang for lang in candidates if lang not in TARGET_LANGS]
        if unknown:
            raise ValueError(
                "language selection contains codes outside TARGET_LANGS: "
                + ", ".join(unknown)
            )
    unsupported = [
        lang for lang in candidates if lang in PADDLEOCR_29_UNSUPPORTED_LANGS
    ]
    if unsupported:
        logger.warning(
            "Skipping %d PaddleOCR 2.9.1 preload language(s) "
            "unsupported by the current preload contract: %s",
            len(unsupported),
            ", ".join(unsupported),
        )

    return [lang for lang in candidates if lang not in PADDLEOCR_29_UNSUPPORTED_LANGS]


def _parse_language_selection(value):
    """Parse and validate the CLI's comma-separated language subset."""
    candidates = list(
        dict.fromkeys(lang.strip() for lang in value.split(",") if lang.strip())
    )
    if not candidates:
        raise argparse.ArgumentTypeError("language selection must not be empty")
    unknown = [lang for lang in candidates if lang not in TARGET_LANGS]
    if unknown:
        raise argparse.ArgumentTypeError(
            "language selection contains codes outside TARGET_LANGS: "
            + ", ".join(unknown)
        )
    return candidates


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
    except ImportError as exc:
        try:
            paddleocr_spec = importlib.util.find_spec("paddleocr")
        except (ImportError, ValueError):
            paddleocr_spec = None

        if paddleocr_spec is None:
            logger.warning("paddleocr is not installed -- skipping model download")
            return None

        origin = getattr(paddleocr_spec, "origin", None) or (
            next(iter(paddleocr_spec.submodule_search_locations), None)
            if getattr(paddleocr_spec, "submodule_search_locations", None)
            else None
        )
        origin_str = f" at {origin}" if origin else ""
        logger.error(
            "paddleocr is installed%s but failed to import: %s",
            origin_str,
            exc,
        )
        raise RuntimeError(
            f"paddleocr import failed during model preload{origin_str}"
        ) from exc


def _parse_checksum_manifest(path: Path) -> list[tuple[str, Path]]:
    """Parse GNU-style sha256sum lines: ``<hex>  <path>`` or ``<hex> *<path>``."""
    entries: list[tuple[str, Path]] = []
    text = path.read_text(encoding="utf-8")
    base = path.parent
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_no}: expected 'sha256  path'")
        digest, rel = parts[0].lower(), parts[1]
        if rel.startswith("*"):
            rel = rel[1:]
        target = Path(rel)
        if not target.is_absolute():
            target = base / rel
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{path}:{line_no}: invalid sha256 digest")
        entries.append((digest, target))
    return entries


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_model_checksums(manifest_path: str | None = None) -> int:
    """Verify files listed in *manifest_path* (S2-02 supply-chain pin).

    Returns number of files verified. Raises RuntimeError on mismatch or
    missing required manifest.
    """
    path_s = (manifest_path or MODEL_CHECKSUM_MANIFEST or "").strip()
    if not path_s:
        if REQUIRE_MODEL_CHECKSUMS:
            raise RuntimeError(
                "S2-02: REQUIRE_MODEL_CHECKSUMS=true but MODEL_CHECKSUM_MANIFEST "
                "is unset"
            )
        logger.info(
            "S2-02: no MODEL_CHECKSUM_MANIFEST — model SHA-256 pin skipped "
            "(set MODEL_CHECKSUM_MANIFEST + REQUIRE_MODEL_CHECKSUMS for fail-closed)"
        )
        return 0
    path = Path(path_s)
    if not path.is_file():
        raise RuntimeError(f"S2-02: MODEL_CHECKSUM_MANIFEST not found: {path}")
    entries = _parse_checksum_manifest(path)
    if not entries and REQUIRE_MODEL_CHECKSUMS:
        raise RuntimeError(f"S2-02: empty checksum manifest: {path}")
    for digest, target in entries:
        if not target.is_file():
            raise RuntimeError(f"S2-02: checksum target missing: {target}")
        actual = _sha256_file(target)
        if actual != digest:
            raise RuntimeError(
                f"S2-02: checksum mismatch for {target}: "
                f"expected {digest[:12]}… got {actual[:12]}…"
            )
        logger.info("S2-02 checksum OK: %s", target)
    print(f"MODEL_CHECKSUM_OK verified={len(entries)}", flush=True)
    return len(entries)


def download_models(cpu_only=False, languages=None):
    """Download and cache PaddleOCR models for all target languages.

    Parameters
    ----------
    cpu_only : bool
        Labels the build as CPU-only. Unsupported PaddleOCR 2.9.1 scripts are
        filtered in every mode. ``use_gpu=False`` is forced and
        ``CUDA_VISIBLE_DEVICES`` is blanked before Paddle imports during the
        download stage.
    languages : list[str] or None
        Optional explicit subset of ``TARGET_LANGS``. ``None`` retains the
        complete configured preload. A subset never changes the default.
    """
    if SKIP_MODEL_PRELOAD:
        logger.info("SKIP_MODEL_PRELOAD=1 -- skipping model download")
        print("MODEL_PRELOAD_SKIPPED attempted=0 succeeded=0 skipped=0", flush=True)
        # S2-02: REQUIRE still fail-closed even when preload skipped
        if REQUIRE_MODEL_CHECKSUMS:
            verify_model_checksums()
        return

    PaddleOCR = _import_paddleocr()
    if PaddleOCR is None:
        print("MODEL_PRELOAD_UNAVAILABLE attempted=0 succeeded=0 skipped=0", flush=True)
        if REQUIRE_MODEL_CHECKSUMS:
            raise RuntimeError(
                "S2-02: REQUIRE_MODEL_CHECKSUMS=true but paddleocr unavailable "
                "and models cannot be verified"
            )
        return

    mode_label = "CPU-only native Paddle" if cpu_only else "standard"
    requested_langs = (
        list(TARGET_LANGS) if languages is None else list(dict.fromkeys(languages))
    )
    preload_langs = _preload_target_langs(
        cpu_only=cpu_only, requested_langs=requested_langs
    )
    logger.info(
        "--- Downloading OCR Models for %d Languages [mode: %s] ---",
        len(preload_langs),
        mode_label,
    )
    failures = []

    for lang in preload_langs:
        logger.info("Downloading: %s ...", lang)
        try:
            kwargs = {
                "use_angle_cls": True,
                "lang": PADDLEOCR_29_RUNTIME_ALIASES.get(lang, lang),
                "use_gpu": False,
                "show_log": False,
            }
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
        raise RuntimeError(f"Model preload failed for {len(failures)} language(s)")

    # S2-02: optional pin after successful download
    verify_model_checksums()

    logger.info("--- Model preload completed successfully ---")
    skipped_count = len(requested_langs) - len(preload_langs)
    print(
        f"MODEL_PRELOAD_OK attempted={len(preload_langs)} "
        f"succeeded={len(preload_langs)} skipped={skipped_count}",
        flush=True,
    )


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
            "Force CPU mode: use_gpu=False with CUDA visibility disabled. "
            "Uses native Paddle model format so fresh preloads are valid. "
            "Equivalent to setting CPU_ONLY_BUILD=1."
        ),
    )
    parser.add_argument(
        "--skip-uie",
        action="store_true",
        default=False,
        help="Skip UIE model download (PaddleNLP Taskflow).",
    )
    parser.add_argument(
        "--languages",
        type=_parse_language_selection,
        help=(
            "optional comma-separated subset of TARGET_LANGS to preload; "
            "omitting this argument downloads the full configured set"
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    cpu_only = args.cpu_only or CPU_ONLY_BUILD

    try:
        download_models(cpu_only=cpu_only, languages=args.languages)
    except RuntimeError:
        # download_models already logged the failures; exit non-zero
        # but do not re-print the traceback.
        sys.exit(1)

    if not args.skip_uie:
        download_uie_model()

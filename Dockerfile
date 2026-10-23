# PaddlePaddle 2.6.2 aborts on Debian 13 and its predictor SIGILLs on qualified
# AMD/virtualized CPU hosts. Keep Debian 12 and use Python 3.12, required by
# Django 6.0.x in coordinator/requirements.txt.
ARG BASE_IMAGE=python:3.12-slim-bookworm

FROM ${BASE_IMAGE} AS runtime-base

ENV HOME=/home/ocr
ENV PADDLE_HOME=/home/ocr/.paddle
ENV PADDLEOCR_HOME=/home/ocr/.paddleocr
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    fonts-noto-core \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libzbar0 \
    poppler-utils \
    tesseract-ocr-all \
    tini \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Noto Sans fonts for searchable-PDF text layer embedding (font_selector.py)
# ---------------------------------------------------------------------------
# Symlink non-CJK fonts installed by fonts-noto-core into the app font dir.
RUN mkdir -p /app/fonts/noto && \
    for f in \
        NotoSans-Regular.ttf \
        NotoSansArabic-Regular.ttf \
        NotoSansDevanagari-Regular.ttf \
        NotoSansTamil-Regular.ttf \
        NotoSansTelugu-Regular.ttf \
        NotoSansKannada-Regular.ttf \
        NotoSansGeorgian-Regular.ttf \
        NotoSansThai-Regular.ttf \
        NotoSansBengali-Regular.ttf; \
    do \
        [ -f "/usr/share/fonts/truetype/noto/$f" ] && \
            ln -sf "/usr/share/fonts/truetype/noto/$f" "/app/fonts/noto/$f"; \
    done

# Download individual CJK OTF files (the Debian fonts-noto-cjk package only
# provides a .ttc collection, not individual .otf files needed by font_selector).
# Pinned to commit f8d1575 for reproducible builds (SEC-017 / maintenance review).
# Air-gapped deployment: pre-stage these files in a local fonts/ directory and
# replace the curl commands below with:
#   COPY fonts/NotoSansCJK*.otf /app/fonts/noto/
RUN curl -fSL "https://github.com/notofonts/noto-cjk/raw/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf" \
         -o /app/fonts/noto/NotoSansCJKsc-Regular.otf && \
    curl -fSL "https://github.com/notofonts/noto-cjk/raw/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf" \
         -o /app/fonts/noto/NotoSansCJKtc-Regular.otf && \
    curl -fSL "https://github.com/notofonts/noto-cjk/raw/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf" \
         -o /app/fonts/noto/NotoSansCJKjp-Regular.otf && \
    curl -fSL "https://github.com/notofonts/noto-cjk/raw/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/OTF/Korean/NotoSansCJKkr-Regular.otf" \
         -o /app/fonts/noto/NotoSansCJKkr-Regular.otf

RUN mkdir -p \
    /app/models \
    /app/ocr_source \
    /app/ocr_output \
    /app/ocr_output/logs \
    /app/ocr_temp \
    /home/ocr/.paddle \
    /home/ocr/.paddleocr

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade "pip==26.2.1"

FROM runtime-base AS model-preload

RUN pip install --no-cache-dir --upgrade "setuptools>=83,<84" "wheel>=0.46.3,<0.47"

RUN pip install --no-cache-dir \
    "protobuf==7.35.1" \
    "paddlepaddle==3.0.0" \
    "paddleocr==2.9.1"

COPY download_models.py /tmp/download_models.py
# Use --cpu-only to force use_gpu=False during the build stage. Native Paddle
# model format is required because freshly downloaded caches are not ONNX.
# SKIP_MODEL_PRELOAD=1 can be set to skip download entirely.
# MODEL_PRELOAD_LANGUAGES may select a validated comma-separated subset for
# bounded lab images; empty retains every configured supported target language.
ARG SKIP_MODEL_PRELOAD=0
ARG MODEL_PRELOAD_LANGUAGES=""
RUN if [ -n "${MODEL_PRELOAD_LANGUAGES}" ]; then \
        SKIP_MODEL_PRELOAD=${SKIP_MODEL_PRELOAD} \
          python3 /tmp/download_models.py --cpu-only \
            --languages "${MODEL_PRELOAD_LANGUAGES}"; \
    else \
        SKIP_MODEL_PRELOAD=${SKIP_MODEL_PRELOAD} \
          python3 /tmp/download_models.py --cpu-only; \
    fi && rm /tmp/download_models.py

FROM runtime-base


# Bake refreshed model caches into the final runtime image.
COPY --from=model-preload /home/ocr/.paddle /home/ocr/.paddle
COPY --from=model-preload /home/ocr/.paddleocr /home/ocr/.paddleocr

# Download FastText language identification model (126MB)
# Used by detect_language() for multi-language OCR routing
ADD https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin /app/models/lid.176.bin

COPY requirements.txt /app/requirements.txt
COPY requirements-language-id.txt /app/requirements-language-id.txt
COPY coordinator/requirements.txt /app/coordinator-requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    -r /app/requirements-language-id.txt \
    -r /app/coordinator-requirements.txt
RUN pip install --no-cache-dir --upgrade "setuptools>=83,<84" "wheel>=0.46.3,<0.47"

# ---------------------------------------------------------------------------
# Remove build-essential from runtime image (maintenance review / INFRA-002)
# gcc/g++/make are only needed for compiling C extensions during pip install;
# the compiled .so files remain but the toolchain is stripped to reduce
# image size (~200MB) and CVE surface area.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY scripts/harden_imgaug_runtime.py /tmp/harden_imgaug_runtime.py
RUN python /tmp/harden_imgaug_runtime.py && rm /tmp/harden_imgaug_runtime.py

COPY *.py /app/
COPY api/ /app/api/
COPY coordinator/ /app/coordinator/
COPY ocr_distributed/ /app/ocr_distributed/
COPY ocr_local/ /app/ocr_local/
COPY reprocess/ /app/reprocess/
COPY schemas/ /app/schemas/
COPY --chmod=755 healthcheck.sh /app/healthcheck.sh

ENV PYTHONPATH=/app/coordinator:/app

# ---------------------------------------------------------------------------
# Non-root runtime user (SEC-003)
# ---------------------------------------------------------------------------
RUN groupadd -r -g 1000 ocr && \
    useradd -r -u 1000 -g ocr -d /home/ocr -s /sbin/nologin ocr && \
    mkdir -p /home/ocr && \
    touch /app/ocr_healthcheck && \
    chown -R ocr:ocr /app/ocr_output /app/ocr_temp /app/ocr_source \
                      /app/models /app/ocr_healthcheck /home/ocr

ARG SOURCE_REVISION=unknown
ARG SOURCE_TREE_SHA256=unknown
RUN if [ -z "$SOURCE_REVISION" ] || [ "$SOURCE_REVISION" = "unknown" ] || \
       [ -z "$SOURCE_TREE_SHA256" ] || [ "$SOURCE_TREE_SHA256" = "unknown" ]; then \
      printf '%s\n' 'ERROR: pass SOURCE_REVISION and SOURCE_TREE_SHA256 (root-cpu profile); see docs/setup/AIRGAP-STANDUP.md' >&2; \
      exit 1; \
    fi

LABEL org.opencontainers.image.revision="${SOURCE_REVISION}" \
      io.ocr_local.runtime-source-sha256="${SOURCE_TREE_SHA256}" \
      io.ocr_local.runtime-profile="root-cpu"

USER ocr

# Keep OCR Python out of PID 1: a same-namespace watchdog cannot SIGKILL the
# namespace init when a native predictor holds the GIL past its deadline.
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD /app/healthcheck.sh

CMD ["python3", "-u", "/app/ocr_gpu_async.py"]

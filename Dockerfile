ARG BASE_IMAGE=python:3.11-slim

FROM ${BASE_IMAGE} AS runtime-base

ENV PADDLE_HOME=/app/.paddle
ENV PADDLEOCR_HOME=/app/.paddleocr
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
    tesseract-ocr \
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
# Pinned to commit f8d1575 for reproducible builds (SEC-017 / ).
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
    /app/.paddle \
    /app/.paddleocr

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

FROM runtime-base AS model-preload

RUN pip install --no-cache-dir \
    paddlepaddle \
    paddleocr

COPY download_models.py /tmp/download_models.py
# Use --cpu-only to force use_gpu=False and avoid CUDA probe segfault
# during the build stage (no GPU drivers available in build containers).
# SKIP_MODEL_PRELOAD=1 can be set to skip download entirely.
ARG SKIP_MODEL_PRELOAD=0
RUN SKIP_MODEL_PRELOAD=${SKIP_MODEL_PRELOAD} \
    python3 /tmp/download_models.py --cpu-only && rm /tmp/download_models.py

FROM runtime-base

# Bake refreshed model caches into the final runtime image.
COPY --from=model-preload /app/.paddle /app/.paddle
COPY --from=model-preload /app/.paddleocr /app/.paddleocr

# Download FastText language identification model (126MB)
# Used by detect_language for multi-language OCR routing
ADD https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin /app/models/lid.176.bin

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir --upgrade "setuptools>=82,<83" "wheel>=0.46.3,<0.47"

# ---------------------------------------------------------------------------
# Remove build-essential from runtime image ( / INFRA-002)
# gcc/g++/make are only needed for compiling C extensions during pip install;
# the compiled .so files remain but the toolchain is stripped to reduce
# image size (~200MB) and CVE surface area.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY *.py /app/
COPY api/ /app/api/
COPY ocr_distributed/ /app/ocr_distributed/
COPY ocr_local/ /app/ocr_local/
COPY reprocess/ /app/reprocess/
COPY schemas/ /app/schemas/
COPY --chmod=755 healthcheck.sh /app/healthcheck.sh

# ---------------------------------------------------------------------------
# Non-root runtime user (SEC-003)
# ---------------------------------------------------------------------------
RUN groupadd -r -g 1000 ocr && \
    useradd -r -u 1000 -g ocr -d /home/ocr -s /sbin/nologin ocr && \
    mkdir -p /home/ocr && \
    touch /app/ocr_healthcheck && \
    chown -R ocr:ocr /app/ocr_output /app/ocr_temp /app/ocr_source \
                      /app/.paddle /app/.paddleocr /app/models \
                      /app/ocr_healthcheck /home/ocr

USER ocr

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD /app/healthcheck.sh

CMD ["python3", "-u", "/app/ocr_gpu_async.py"]


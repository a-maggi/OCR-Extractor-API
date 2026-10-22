# The llama.cpp image to take llama-server from: server-b11669 is the v0.7.0 release (README, Running it, to bump)
ARG LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-b11669

# Python deps and app code. Buildable on its own with `--target app` (no llama-server).
FROM python:3.12-slim AS app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# tini reaps the shared model servers once the worker that started them exits.
# libgomp1 is llama.cpp's OpenMP runtime; curl is used by llama.cpp's model loader; procps gives `ps` for
# checking each process's memory (see README, Sizing).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini libgomp1 libglib2.0-0 curl procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.12.17

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Marker downloads a font into site-packages the first time a converter is built. The app user can't write
# there, so fetch it now, as root, and fail the build if it's missing.
RUN /app/.venv/bin/python -c "from marker.util import download_font; download_font()" \
    && /app/.venv/bin/python -c "import os; from marker.settings import settings; assert os.path.getsize(settings.FONT_PATH) > 0"

# Non-root. Model caches live under its home; mount volumes on the two model dirs (see docker-compose.yml)
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.cache/huggingface /home/app/.cache/datalab/models \
    && chown -R app:app /home/app/.cache

ENV PATH="/app/.venv/bin:/opt/llama:$PATH" \
    LD_LIBRARY_PATH=/opt/llama \
    HF_HOME=/home/app/.cache/huggingface

USER app
EXPOSE 8000
ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "ocr_extractor.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]


# llama-server serves the Surya OCR model on CPU
FROM ${LLAMA_CPP_IMAGE} AS llama


FROM app AS runtime
USER root
# The official image keeps llama-server and its shared libraries together in /app
COPY --from=llama /app/ /opt/llama/
RUN llama-server --version
USER app

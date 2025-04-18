FROM ghcr.io/astral-sh/uv:python3.9-bookworm-slim

WORKDIR /app

# Install build dependencies and PDF-related dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libfreetype6-dev \
    libxslt-dev \
    libxml2-dev \
    libmupdf-dev \
    git

ENV UV_COMPILE_BYTECODE=1

ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Install the project in development mode
RUN uv pip install -e .

# Explicitly install PyMuPDF and related packages
RUN uv pip install PyMuPDF==1.25.4 pymupdf-fonts

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT []

CMD ["python", "-m", "src.core.main"]
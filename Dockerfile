# syntax=docker/dockerfile:1.18

FROM vastai/base-image:stock-ubuntu24.04-py312-2026-08-26@sha256:98778a39bdf31d0160b50b1339d815a703ee04c0bb433bc8da575581d67de589

ARG PROJECT_REVISION

LABEL org.opencontainers.image.source="https://github.com/Fierynx/ternary-pretrain" \
      org.opencontainers.image.revision="${PROJECT_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PATH="/opt/ternary-pretrain/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/ternary-pretrain/.venv \
    UV_PYTHON=/venv/main/bin/python \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.12.6@sha256:21fedfa1151bb363a6cb32e4a34a032e26f02b3e9e288a96f126ac27d3b472d6 \
    /uv /uvx /usr/local/bin/

WORKDIR /workspace/ternary-pretrain

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra transformers --no-install-project

COPY src ./src
COPY configs ./configs

RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    printf '%s\n' "${PROJECT_REVISION}" | grep -Eq '^[0-9a-f]{40}$'; \
    command -v zstd >/dev/null; \
    printf '%s\n' "${PROJECT_REVISION}" > .build-revision; \
    uv sync --locked --no-dev --extra transformers; \
    python -c 'import torch; assert torch.__version__.startswith("2.13.0"); assert torch.version.cuda == "13.0"'; \
    ternary-pretrain inspect --config configs/models/25m.toml >/dev/null; \
    env-hash > /.env_hash

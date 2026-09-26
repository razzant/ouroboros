# Ouroboros — Docker image for web UI runtime
# Usage:
#   docker build -t ouroboros-web .
#   docker run --rm -p 8765:8765 ouroboros-web
# The RUN --mount caches need BuildKit, Docker's default builder.

FROM ghcr.io/astral-sh/uv:0.12.1 AS uv
FROM python:3.10-slim

COPY --from=uv /uv /uvx /bin/

# Browsers first, dependencies second, sources last: every release rewrites
# pyproject.toml/uv.lock, so anything below the lock copy is rebuilt per
# release while the apt packages and the Chromium/WebKit downloads above it
# are reused. The Playwright pin must equal the locked version so the
# downloaded browser revisions match the venv's driver
# (tests/test_build_scripts.py::TestDockerfile). The installer runs from an
# ephemeral uvx tool environment, so the image carries one Playwright: the
# venv's. Browsers live in a shared path the runtime honors as-is
# (ouroboros/tools/browser.py) — not inside the package tree, which the
# per-release dependency layer would rebuild.
ARG PLAYWRIGHT_VERSION=1.62.0
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# System dependencies: git for the agent's own history and updates, plus
# every Chromium/WebKit native library from Playwright's authoritative list.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends git \
    && uvx --from "playwright==${PLAYWRIGHT_VERSION}" playwright install-deps chromium webkit \
    && uvx --from "playwright==${PLAYWRIGHT_VERSION}" playwright install chromium webkit

# Working directory
ENV APP_HOME=/app \
    PATH="/app/.venv/bin:$PATH"
WORKDIR ${APP_HOME}

# Resolve only from the reviewed lock; the project itself is installed after
# the source copy so source edits reuse this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra browser --no-install-project

# Copy application
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra browser --no-editable

# Default environment
ENV OUROBOROS_SERVER_HOST=0.0.0.0 \
    OUROBOROS_SERVER_PORT=8765 \
    OUROBOROS_FILE_BROWSER_DEFAULT=${APP_HOME}

EXPOSE 8765

ENTRYPOINT ["python", "server.py"]

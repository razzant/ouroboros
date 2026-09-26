# Derived Cowork Bench image: the upstream task image plus a pinned, clean Ouroboros seed,
# its own interpreter prefix, a persistent MCP gateway and the engine entrypoint.
#
# Build context (assembled by run_cowork_bench.py --build-image, never the repo itself):
#   seed/                  full `git clone` of the seed checkout at the measured commit
#   main_ouroboros.py      the engine entrypoint from this directory
#   eval_attempt.py        its stdlib eval-attempt helper and
#   official_receipt.py    the evaluator-receipt reader it reuses (from the parent directory)
ARG BASE_IMAGE=cowork-pack:latest
FROM ${BASE_IMAGE}

ARG SEED_SHA=""
ARG BENCH_SHA=""
ARG MCP_PROXY_VERSION=0.12.0
ARG MCP_SDK_VERSION=1.30.0
ARG REQUIRE_CHROMIUM=1
ARG VENDORED_MCP_SDK_VERSION=1.26.0
LABEL org.ouroboros.cowork.seed_sha="${SEED_SHA}" \
      org.ouroboros.cowork.bench_sha="${BENCH_SHA}" \
      org.ouroboros.cowork.mcp_proxy="${MCP_PROXY_VERSION}"

# The benchmark tasks never see this prefix on PATH: agent dependencies stay separate from
# task/runtime dependencies (the base image keeps /opt/venv and /workspace/.venv untouched).
COPY seed/ /opt/ouroboros-src/
RUN set -eux; \
    test "$(git -C /opt/ouroboros-src rev-parse HEAD)" = "${SEED_SHA}"; \
    test -z "$(git -C /opt/ouroboros-src status --porcelain)"; \
    uv venv /opt/ouroboros-venv --python /usr/bin/python3; \
    uv pip install --python /opt/ouroboros-venv/bin/python -r /opt/ouroboros-src/requirements-runtime.lock; \
    uv pip install --python /opt/ouroboros-venv/bin/python --no-deps -e /opt/ouroboros-src; \
    /opt/ouroboros-venv/bin/python -c "import importlib.metadata as m, mcp, ouroboros; print('ouroboros', m.version('ouroboros'), 'mcp', m.version('mcp'))"; \
    test -z "$(git -C /opt/ouroboros-src status --porcelain)"

# mcp-proxy keeps ONE stdio session per benchmark MCP server for the whole task. The SDK is
# pinned below 2.x: mcp-proxy 0.12 imports a symbol the 2.x line removed.
RUN set -eux; \
    uv venv /opt/mcp-proxy-venv --python /usr/bin/python3; \
    uv pip install --python /opt/mcp-proxy-venv/bin/python "mcp-proxy==${MCP_PROXY_VERSION}" "mcp==${MCP_SDK_VERSION}"; \
    /opt/mcp-proxy-venv/bin/mcp-proxy --help >/dev/null

# Two vendored servers (hr1c, clickhouse) ship WITHOUT a uv.lock and declare a bare `mcp`
# floor, so a fresh build resolves the MCP SDK 2.x line, whose removed decorator API makes
# them crash at startup ("'Server' object has no attribute 'list_resources'") - 148 tasks
# become structural zeros for ANY engine. Re-pin every vendored venv that drifted onto 2.x to
# the 1.x release the locked servers resolve to, and refuse to build if one is still on 2.x.
RUN set -eu; \
    for venv in /opt/local_servers/*/.venv; do \
      [ -x "$venv/bin/python" ] || continue; \
      version="$("$venv/bin/python" -c "import importlib.metadata as m; print(m.version('mcp'))" 2>/dev/null || true)"; \
      case "$version" in \
        2.*) echo "re-pinning $venv from mcp $version"; \
             uv pip install --python "$venv/bin/python" "mcp==${VENDORED_MCP_SDK_VERSION}";; \
      esac; \
      version="$("$venv/bin/python" -c "import importlib.metadata as m; print(m.version('mcp'))" 2>/dev/null || true)"; \
      case "$version" in 2.*) echo "$venv still on mcp $version" >&2; exit 1;; esac; \
    done

# The finance servers use local PostgreSQL-backed yfinance shims, but their vendored
# project dependencies still describe the original public-API servers and omit psycopg2.
# Install the benchmark task-runtime's pinned driver into both actual uv environments.
ARG POSTGRES_CLIENT_VERSION=2.9.10
RUN set -eu; \
    for server in moex-finance-mcp yahoo-finance-mcp; do \
      uv pip install --python "/opt/local_servers/$server/.venv/bin/python" \
        "psycopg2-binary==${POSTGRES_CLIENT_VERSION}"; \
    done

# The upstream Dockerfile runs `playwright install chromium || true` with no `playwright` CLI
# on PATH, so the image it produces carries NO browser and every playwright_with_chunk task
# is a structural zero. Install the exact revision the vendored playwright-mcp pins, then
# refuse to build without it unless explicitly waived.
RUN set -eu; \
    if ! ls -d /root/.cache/ms-playwright/chromium-* >/dev/null 2>&1; then \
      cd /opt/local_servers/playwright-mcp && npx playwright install chromium; \
    fi; \
    if [ "${REQUIRE_CHROMIUM}" = "1" ]; then \
      ls -d /root/.cache/ms-playwright/chromium* >/dev/null 2>&1 \
        || { echo "Chromium is missing from the base image (playwright tasks would be structural zeros)" >&2; exit 1; }; \
    fi

COPY main_ouroboros.py eval_attempt.py official_receipt.py /workspace/
RUN mkdir -p /opt/ouroboros-data /opt/ouroboros-app /opt/ouroboros-runtime \
    && /opt/venv/bin/python3 -m py_compile /workspace/main_ouroboros.py \
       /workspace/eval_attempt.py /workspace/official_receipt.py \
    && cd /workspace && /opt/venv/bin/python3 -c "import eval_attempt"

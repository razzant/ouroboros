# syntax=docker/dockerfile:1
FROM ubuntu:24.04

ENV PATH=/opt/venv/bin:$PATH \
    PLAYWRIGHT_BROWSERS_PATH=/opt/browsers \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8
COPY requirements-runtime.lock /opt/requirements-runtime.lock
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates git python3-venv \
    && python3 -m venv /opt/venv \
    && python -m pip install --no-cache-dir -r /opt/requirements-runtime.lock \
    && python -m pip check \
    && python -m playwright install --with-deps chromium \
    && useradd --create-home --uid 10001 tester \
    && install -d -o tester -g tester /case /case/repo /evidence

# Only the selected commit and its files enter the runtime. No remote, runner
# checkout metadata or QA workflow is part of the product tree. The candidate's
# runtime lock also supplies pytest for imported fixture modules (not a test run).
USER tester
WORKDIR /case/repo
ARG CANDIDATE_SHA
COPY candidate.pack /opt/candidate.pack
RUN --network=none git init -q --template= --initial-branch=ouroboros . \
    && git index-pack --stdin < /opt/candidate.pack \
    && printf '%s\n' "$CANDIDATE_SHA" > .git/shallow \
    && git update-ref refs/heads/ouroboros "$CANDIDATE_SHA" \
    && git read-tree --reset -u HEAD \
    && test "$(git rev-parse HEAD)" = "$CANDIDATE_SHA" \
    && git diff --quiet HEAD
COPY scenario.py loopback.py /opt/qa/
ENV OUROBOROS_PYTEST_ACTIVE=1 PIP_NO_INDEX=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "-IB", "/opt/qa/scenario.py"]

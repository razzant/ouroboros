#!/bin/bash
set -euo pipefail

# Stages the official Node.js runtime that ouroboros/claudexor_runtime_pin.json names, as
# node-standalone/bin/node (macOS/Linux), after checking the archive's exact size and SHA-256.
# Run from the repo root: bash scripts/download_node_standalone.sh
# The pin and its verification live in scripts/download_node_standalone.py, shared with Windows and
# Docker. Why bundle node: skill payloads can declare runtime=node and the preflight runs `node --check`;
# macOS code-signing enforcement kills a Homebrew node launched from the packaged app, while the official
# notarized build runs once the signing pass re-signs it (entitlements.plist).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/python-standalone/bin/python3"
[ -x "$PYTHON" ] || PYTHON=python3
exec "$PYTHON" "$ROOT/scripts/download_node_standalone.py" "$@"

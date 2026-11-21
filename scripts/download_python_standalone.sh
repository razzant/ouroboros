#!/bin/bash
set -euo pipefail

# Downloads python-build-standalone for macOS (arm64 + x86_64) and Linux (x86_64)
# Run from repo root: bash scripts/download_python_standalone.sh
# CI test jobs take the same pinned interpreter without the packaged dependencies:
# PYTHON_STANDALONE_DEST=<dir> PYTHON_STANDALONE_INTERPRETER_ONLY=1 bash scripts/download_python_standalone.sh

RELEASE="20261009"
PY_VERSION="3.12.15"
DEST="${PYTHON_STANDALONE_DEST:-python-standalone}"

OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        PLATFORM="aarch64-apple-darwin"
        SHA256="233f8e15255b3e1d1fd47ea5dd5230dfed803f371138e896e54d98c60ae19f6f"
    elif [ "$ARCH" = "x86_64" ]; then
        PLATFORM="x86_64-apple-darwin"
        SHA256="fb3187d95334813b88940db0f1ca57b84cedfd552739c3a5771c3ea0e49988c9"
    else
        echo "Unsupported macOS architecture: $ARCH"
        exit 1
    fi
elif [ "$OS" = "Linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        PLATFORM="x86_64-unknown-linux-gnu"
        SHA256="ffa8f85f1b56e88b687f08d743d817015524b86a9365fd921243132b80f1a36d"
    elif [ "$ARCH" = "aarch64" ]; then
        PLATFORM="aarch64-unknown-linux-gnu"
        SHA256="2e1ba0e7f22e01c534caebf7f09e7820322bc0d055271d3434af671e8400f96f"
    else
        echo "Unsupported Linux architecture: $ARCH"
        exit 1
    fi
else
    echo "Unsupported OS: $OS"
    exit 1
fi

FILENAME="cpython-${PY_VERSION}+${RELEASE}-${PLATFORM}-install_only_stripped.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${RELEASE}/${FILENAME}"

echo "=== Downloading Python ${PY_VERSION} for ${PLATFORM} ==="
echo "URL: ${URL}"

rm -rf "$DEST" _python_tmp
mkdir -p _python_tmp

# Pinned SHA256 (from the release's SHA256SUMS): a swapped/truncated archive
# fails here instead of becoming the packaged runtime. Update the pins when
# bumping RELEASE/PY_VERSION.
curl -L --fail --retry 5 --retry-all-errors --retry-delay 5 --progress-bar "$URL" -o _python_tmp/"$FILENAME"
if command -v shasum >/dev/null 2>&1; then
    ACTUAL="$(shasum -a 256 "_python_tmp/${FILENAME}" | awk '{print $1}')"
else
    ACTUAL="$(sha256sum "_python_tmp/${FILENAME}" | awk '{print $1}')"
fi
if [ "$ACTUAL" != "$SHA256" ]; then
    echo "SHA256 mismatch for ${FILENAME}: expected ${SHA256}, got ${ACTUAL} — refusing to install."
    exit 1
fi
tar xz -C _python_tmp -f _python_tmp/"$FILENAME"
rm -f _python_tmp/"$FILENAME"

# Archive extracts to python/ — rename to python-standalone/
mv _python_tmp/python "$DEST"
rm -rf _python_tmp

if [ "${PYTHON_STANDALONE_INTERPRETER_ONLY:-0}" = "1" ]; then
    echo "Python: ${DEST}/bin/python3 (interpreter only)"
    exit 0
fi

echo ""
echo "=== Installing agent dependencies ==="
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required for locked dependency installation."
    exit 1
fi
uv pip install --python "${DEST}/bin/python3" --quiet -r requirements-runtime.lock

echo ""
echo "=== Done ==="
echo "Python: ${DEST}/bin/python3"
"${DEST}/bin/python3" --version

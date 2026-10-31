#!/bin/bash
set -euo pipefail

# Downloads python-build-standalone for macOS (arm64 + x86_64) and Linux (x86_64)
# Run from repo root: bash scripts/download_python_standalone.sh
# CI test jobs take the same pinned interpreter without the packaged dependencies:
# PYTHON_STANDALONE_DEST=<dir> PYTHON_STANDALONE_INTERPRETER_ONLY=1 bash scripts/download_python_standalone.sh

RELEASE="20261009"
PY_VERSION="3.10.22"
DEST="${PYTHON_STANDALONE_DEST:-python-standalone}"

OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        PLATFORM="aarch64-apple-darwin"
        SHA256="056952e8ee266df0dc4e595a88a0d691efe0d6af7cd9e772b89e5efd276ead79"
    elif [ "$ARCH" = "x86_64" ]; then
        PLATFORM="x86_64-apple-darwin"
        SHA256="7c579c3d9913a36b86a63772704cc08960de9b819b8915455218a536dbd4f3f2"
    else
        echo "Unsupported macOS architecture: $ARCH"
        exit 1
    fi
elif [ "$OS" = "Linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        PLATFORM="x86_64-unknown-linux-gnu"
        SHA256="caa66b308e10f35b8114054df7d927903f971f631cfa3237b84c40d5d5f35549"
    elif [ "$ARCH" = "aarch64" ]; then
        PLATFORM="aarch64-unknown-linux-gnu"
        SHA256="ecbe978a7ecbeb3a9aed67bb5d161180701513b4fa8101037b2a3f1ac11e96be"
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

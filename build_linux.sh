#!/bin/bash
set -e

VERSION=$(tr -d '[:space:]' < VERSION)
ARCHIVE_NAME="Ouroboros-${VERSION}-linux-$(uname -m).tar.gz"
MANAGED_SOURCE_BRANCH="${OUROBOROS_MANAGED_SOURCE_BRANCH:-ouroboros}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/ouroboros-build-pycache}"
mkdir -p "$PYTHONPYCACHEPREFIX"

HOST_PYTHON_CMD="${PYTHON_CMD:-python3}"
if ! command -v "$HOST_PYTHON_CMD" >/dev/null 2>&1; then
    HOST_PYTHON_CMD=python
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required for locked dependency installation."
    echo "Install uv 0.12.1: curl -LsSf https://astral.sh/uv/0.12.1/install.sh | sh"
    exit 1
fi

echo "=== Building Ouroboros for Linux (v${VERSION}) ==="

# Preserve the pre-#688 host-backend build on other Linux builders/architectures.
# Release CI explicitly requires 1; auto never silently skips a broken Jammy build.
GTK_MODE=${OUROBOROS_BUNDLE_LINUX_GTK:-auto}
if [ "$GTK_MODE" = auto ]; then
    GTK_MODE=0
    if [ "$(uname -m)" = x86_64 ] && [ -r /etc/os-release ]; then
        if ( . /etc/os-release; [ "$ID" = ubuntu ] && [ "$VERSION_ID" = 22.04 ] ); then
            GTK_MODE=1
        fi
    fi
fi
case "$GTK_MODE" in 0|1) ;; *) echo "Invalid OUROBOROS_BUNDLE_LINUX_GTK: $GTK_MODE" >&2; exit 1 ;; esac
export OUROBOROS_BUNDLE_LINUX_GTK=$GTK_MODE
GTK_EXTRAS=()
if [ "$GTK_MODE" = 1 ]; then
    GTK_EXTRAS=(--extra linux-desktop)
    # Compile GI/Cairo with this builder, not the portable Python's build machine.
    export CC="${CC:-gcc}" CXX="${CXX:-g++}" LDSHARED="${LDSHARED:-gcc -shared}"
    if ! pkg-config --exists gtk+-3.0 webkit2gtk-4.1 gobject-introspection-1.0 cairo; then
        echo "ERROR: GTK/WebKit build inputs are missing. See packaging/appimage/README.md." >&2
        exit 1
    fi
else
    echo "Host-backend Linux build: no bundled native-window claim (see packaging/appimage/README.md)."
fi

if [ ! -f "python-standalone/bin/python3" ]; then
    echo "ERROR: python-standalone/ not found."
    echo "Run first: bash scripts/download_python_standalone.sh"
    exit 1
fi
PORTABLE_PYTHON="python-standalone/bin/python3"

# Bundle the official Node.js runtime so node-runtime skills work in the
# packaged app out of the box.
if [ ! -f "node-standalone/bin/node" ]; then
    echo "--- Downloading bundled Node.js runtime ---"
    bash scripts/download_node_standalone.sh
fi

if [ ! -f "ripgrep-standalone/bin/rg" ]; then
    echo "--- Downloading bundled ripgrep runtime ---"
    bash scripts/download_ripgrep_standalone.sh
fi

echo "--- Verifying bundled Betterleaks runtime ---"
"$PORTABLE_PYTHON" -m ouroboros.betterleaks_runtime install \
    --build-output betterleaks-standalone

rm -rf build dist

echo "--- Creating portable-Python launcher build environment ---"
# PyInstaller inherits the build interpreter's libpython/glibc floor. Derive a
# build-only venv from the portable interpreter the payload ships so a newer
# runner cannot make the desktop launcher less portable or add build tooling to
# the packaged Python tree.
BUILD_VENV="build/linux-pyinstaller-venv"
"$PORTABLE_PYTHON" -m venv "$BUILD_VENV"
BUILD_PYTHON="$BUILD_VENV/bin/python"
BUILD_REQUIREMENTS="$BUILD_VENV/build-requirements.txt"
uv export --locked --no-dev --extra browser --extra desktop --extra build \
    "${GTK_EXTRAS[@]}" \
    --no-emit-project --no-hashes --no-annotate --output-file "$BUILD_REQUIREMENTS"
uv pip install --python "$BUILD_PYTHON" -q -r "$BUILD_REQUIREMENTS"

echo "--- Installing agent dependencies into python-standalone ---"
uv pip install --python "$PORTABLE_PYTHON" -q -r requirements-runtime.lock
# The source-launcher handoff cannot import the frozen launcher's private PYZ.
DESKTOP_REQUIREMENTS="build/linux-desktop-requirements.txt"
uv export --locked --no-dev --extra desktop "${GTK_EXTRAS[@]}" \
    --no-emit-project --no-hashes --no-annotate --output-file "$DESKTOP_REQUIREMENTS"
uv pip install --python "$PORTABLE_PYTHON" -q -r "$DESKTOP_REQUIREMENTS"

echo "--- Fetching exact Claudexor runtime seed ---"
"$PORTABLE_PYTHON" scripts/fetch_claudexor_runtime.py --output-dir claudexor-runtime

export PYINSTALLER_CONFIG_DIR="$PWD/.pyinstaller-cache"
mkdir -p "$PYINSTALLER_CONFIG_DIR"

echo "--- Installing Chromium/WebKit for browser tools (bundled into python-standalone) ---"
if [ "${OUROBOROS_SKIP_PLAYWRIGHT_INSTALL_DEPS:-0}" = "1" ]; then
    echo "Skipping Playwright host-library installation by request."
else
    "$PORTABLE_PYTHON" -m playwright install-deps chromium webkit
fi
PLAYWRIGHT_BROWSERS_PATH=0 "$PORTABLE_PYTHON" -m playwright install chromium webkit

echo "--- Building embedded managed repo bundle ---"
"$HOST_PYTHON_CMD" scripts/build_repo_bundle.py --source-branch "$MANAGED_SOURCE_BRANCH"

echo "--- Running PyInstaller ---"
"$BUILD_PYTHON" -m PyInstaller Ouroboros.spec --clean --noconfirm

# The native shell is independent of Playwright's WebKit browser-tool runtime.
if [ "$GTK_MODE" = 1 ]; then
    "$BUILD_PYTHON" scripts/bundle_linux_gtk.py dist/Ouroboros/_internal
    "$BUILD_PYTHON" scripts/capture_linux_sources.py dist/Ouroboros/_internal/linux-gtk \
        "dist/native-sources-linux-${VERSION}.tar.gz"
fi

echo "--- Installing packaged CLI wrappers ---"
mkdir -p dist/Ouroboros/bin
cp packaging/cli/ouroboros dist/Ouroboros/bin/ouroboros
cp packaging/cli/install-ouroboros-cli dist/Ouroboros/bin/install-ouroboros-cli
chmod +x dist/Ouroboros/bin/ouroboros dist/Ouroboros/bin/install-ouroboros-cli

# WA6 parity: precompile bytecode instead of deleting it. Linux has no codesign
# seal, so this is purely for start-speed + consistency with the macOS build (where
# precompiled+sealed .pyc keep the signature valid). --invalidation-mode
# unchecked-hash means a read-only payload never rewrites the .pyc at import.
echo "--- Precompiling Python bytecode in archive payload (start-speed parity) ---"
APP_EMBEDDED_PY="$(find dist/Ouroboros -type f -path '*/python-standalone/bin/python3' 2>/dev/null | head -1)"
if [ -z "$APP_EMBEDDED_PY" ]; then
    APP_EMBEDDED_PY="$PWD/python-standalone/bin/python3"
fi
echo "Using embedded interpreter for compileall: $APP_EMBEDDED_PY"
COMPILE_TARGETS=()
while IFS= read -r d; do
    [ -n "$d" ] && COMPILE_TARGETS+=("$d")
done < <(find dist/Ouroboros -type d \( -path '*/python-standalone' -o -name ouroboros \) 2>/dev/null)
if [ "${#COMPILE_TARGETS[@]}" -gt 0 ]; then
    # Neutralize the build-time PYTHONDONTWRITEBYTECODE=1 + PYTHONPYCACHEPREFIX for
    # THIS command only, else compileall writes no in-tree .pyc (start-speed parity).
    env -u PYTHONDONTWRITEBYTECODE -u PYTHONPYCACHEPREFIX \
        "$APP_EMBEDDED_PY" -m compileall -q -f --invalidation-mode unchecked-hash "${COMPILE_TARGETS[@]}" || true
else
    echo "WARNING: no compileall targets found in dist/Ouroboros (python-standalone / ouroboros)."
fi

echo ""
echo "=== Creating archive ==="
cd dist
tar -czf "$ARCHIVE_NAME" Ouroboros/
cd ..

echo ""
echo "=== Creating AppImage ==="
bash scripts/build_appimage.sh

echo ""
echo "=== Done ==="
echo "Archive: dist/$ARCHIVE_NAME"
echo "AppImage: dist/Ouroboros-${VERSION}-linux-$(uname -m).AppImage"
echo ""
echo "To run: extract and execute ./Ouroboros/Ouroboros"
echo "To install CLI: ./Ouroboros/bin/install-ouroboros-cli"

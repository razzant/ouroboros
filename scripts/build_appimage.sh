#!/bin/bash
set -euo pipefail

VERSION=$(tr -d '[:space:]' < VERSION)
DIST_DIR=${OUROBOROS_DIST_DIR:-dist}
PAYLOAD_DIR="$DIST_DIR/Ouroboros"
APPDIR=${OUROBOROS_APPDIR:-$DIST_DIR/Ouroboros.AppDir}
TOOL_VERSION=1.9.1
RUNTIME_VERSION=20251108

if [ ! -x "$PAYLOAD_DIR/Ouroboros" ]; then
    echo "ERROR: PyInstaller payload not found at $PAYLOAD_DIR/Ouroboros" >&2
    echo "Run build_linux.sh first, or point OUROBOROS_DIST_DIR at its dist directory." >&2
    exit 1
fi

if [ "${OUROBOROS_BUNDLE_LINUX_GTK:-1}" != 0 ] && [ ! -f "$PAYLOAD_DIR/_internal/linux-gtk/manifest.json" ]; then
    echo "ERROR: bundled GTK/WebKit closure is missing; run scripts/bundle_linux_gtk.py first." >&2
    exit 1
fi

rm -rf "$APPDIR"
mkdir -p \
    "$APPDIR/usr/lib" \
    "$APPDIR/usr/share/applications" \
    "$APPDIR/usr/share/icons/hicolor/1024x1024/apps"
cp -a "$PAYLOAD_DIR" "$APPDIR/usr/lib/ouroboros"
install -m 0755 packaging/appimage/AppRun "$APPDIR/AppRun"
install -m 0644 packaging/appimage/ouroboros.desktop "$APPDIR/ouroboros.desktop"
install -m 0644 packaging/appimage/ouroboros.desktop \
    "$APPDIR/usr/share/applications/ouroboros.desktop"
install -m 0644 assets/icon_1024.png "$APPDIR/ouroboros.png"
install -m 0644 assets/icon_1024.png \
    "$APPDIR/usr/share/icons/hicolor/1024x1024/apps/ouroboros.png"

if [ "${1:-}" = "--appdir-only" ]; then
    echo "Prepared AppDir: $APPDIR"
    exit 0
fi

ARCH=$(uname -m)
OUTPUT="$DIST_DIR/Ouroboros-${VERSION}-linux-${ARCH}.AppImage"
case "$ARCH" in
    x86_64)
        TOOL_ARCH=x86_64
        TOOL_SHA256=ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0
        RUNTIME_SHA256=2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d
        ;;
    aarch64)
        TOOL_ARCH=aarch64
        TOOL_SHA256=f0837e7448a0c1e4e650a93bb3e85802546e60654ef287576f46c71c126a9158
        RUNTIME_SHA256=00cbdfcf917cc6c0ff6d3347d59e0ca1f7f45a6df1a428a0d6d8a78664d87444
        ;;
    *)
        echo "Unsupported AppImage architecture: $ARCH" >&2
        exit 1
        ;;
esac

TOOL_CACHE=${APPIMAGE_TOOL_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}/ouroboros/appimage}
TOOL="$TOOL_CACHE/appimagetool-${TOOL_VERSION}-${TOOL_ARCH}.AppImage"
TOOL_URL="https://github.com/AppImage/appimagetool/releases/download/${TOOL_VERSION}/appimagetool-${TOOL_ARCH}.AppImage"
RUNTIME="$TOOL_CACHE/type2-runtime-${RUNTIME_VERSION}-${TOOL_ARCH}"
RUNTIME_URL="https://github.com/AppImage/type2-runtime/releases/download/${RUNTIME_VERSION}/runtime-${TOOL_ARCH}"
mkdir -p "$TOOL_CACHE"

fetch_verified() {
    local destination="$1" url="$2" expected="$3" actual tmp
    if [ -f "$destination" ]; then
        actual=$(sha256sum "$destination" | awk '{print $1}')
        if [ "$actual" = "$expected" ]; then
            return 0
        fi
    fi
    tmp="${destination}.tmp.$$"
    rm -f "$tmp"
    curl --fail --location --silent --show-error "$url" --output "$tmp"
    actual=$(sha256sum "$tmp" | awk '{print $1}')
    if [ "$actual" != "$expected" ]; then
        echo "SHA256 mismatch for $(basename "$destination"): expected $expected, got $actual" >&2
        rm -f "$tmp"
        return 1
    fi
    mv -f "$tmp" "$destination"
}

fetch_verified "$TOOL" "$TOOL_URL" "$TOOL_SHA256"
fetch_verified "$RUNTIME" "$RUNTIME_URL" "$RUNTIME_SHA256"
chmod +x "$TOOL"

rm -f "$OUTPUT"
ARCH="$TOOL_ARCH" APPIMAGE_EXTRACT_AND_RUN=1 \
    "$TOOL" --runtime-file "$RUNTIME" "$APPDIR" "$OUTPUT"
chmod +x "$OUTPUT"
rm -rf "$APPDIR"

echo "AppImage: $OUTPUT"

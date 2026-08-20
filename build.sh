#!/bin/bash
set -e

# Signing identity: explicit SIGN_IDENTITY, then Developer ID auto-detect, else fail later.
if [ -z "${SIGN_IDENTITY:-}" ]; then
    SIGN_IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
        | grep -E '\"Developer ID Application' \
        | head -1 \
        | sed -E 's/^.*\"([^\"]+)\".*$/\1/')"
    if [ -n "${SIGN_IDENTITY:-}" ]; then
        echo "Auto-detected SIGN_IDENTITY from keychain: $SIGN_IDENTITY"
    fi
fi
ENTITLEMENTS="entitlements.plist"
SIGN_MODE="${OUROBOROS_SIGN:-1}"
MANAGED_SOURCE_BRANCH="${OUROBOROS_MANAGED_SOURCE_BRANCH:-ouroboros}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/ouroboros-build-pycache}"
mkdir -p "$PYTHONPYCACHEPREFIX"

APP_PATH="dist/Ouroboros.app"
DMG_NAME="Ouroboros-$(cat VERSION | tr -d '[:space:]').dmg"
DMG_PATH="dist/$DMG_NAME"

echo "=== Building Ouroboros.app ==="

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required for locked dependency installation."
    echo "Install uv 0.12.1: curl -LsSf https://astral.sh/uv/0.12.1/install.sh | sh"
    exit 1
fi

if [ ! -f "python-standalone/bin/python3" ]; then
    echo "ERROR: python-standalone/ not found."
    echo "Run first: bash scripts/download_python_standalone.sh"
    exit 1
fi

# Bundle the official notarized Node.js runtime so node-runtime skills work in
# the packaged app (Homebrew node is code-signing-killed by macOS). The signing
# pass below re-signs node-standalone/bin/node under the hardened runtime.
if [ ! -f "node-standalone/bin/node" ]; then
    echo "--- Downloading bundled Node.js runtime ---"
    bash scripts/download_node_standalone.sh
fi

if [ ! -f "ripgrep-standalone/bin/rg" ]; then
    echo "--- Downloading bundled ripgrep runtime ---"
    bash scripts/download_ripgrep_standalone.sh
fi

echo "--- Verifying bundled Betterleaks runtime ---"
python-standalone/bin/python3 -m ouroboros.betterleaks_runtime install \
    --build-output betterleaks-standalone

echo "--- Installing launcher dependencies ---"
BUILD_REQUIREMENTS="$(mktemp "${TMPDIR:-/tmp}/ouroboros-build-requirements.XXXXXX")"
trap 'rm -f "$BUILD_REQUIREMENTS"' EXIT
uv export --locked --no-dev --extra browser --extra desktop --extra build \
    --no-emit-project --no-hashes --no-annotate --output-file "$BUILD_REQUIREMENTS"
uv pip install --python python3 -q -r "$BUILD_REQUIREMENTS"

echo "--- Installing agent dependencies into python-standalone ---"
uv pip install --python python-standalone/bin/python3 -q -r requirements-runtime.lock

echo "--- Fetching exact Claudexor runtime seed ---"
python-standalone/bin/python3 scripts/fetch_claudexor_runtime.py --output-dir claudexor-runtime

echo "--- Installing Chromium for browser tools (bundled into python-standalone) ---"
# Full Chromium app bundle breaks nested-bundle codesign on arm64 runners.
PLAYWRIGHT_BROWSERS_PATH=0 python-standalone/bin/python3 -m playwright install --only-shell chromium

echo "--- Skipping bundled WebKit on macOS ---"
# Playwright WebKit contains nested .framework/.xpc bundles and .tbd stubs that
# do not survive PyInstaller's app layout plus hardened-runtime codesigning as a
# simple embedded payload. WebKit remains available through browser.py's managed
# Playwright cache on first engine=webkit use; Chromium stays bundled.

echo "--- Removing stale bundled WebKit payloads from macOS package tree ---"
python3 - <<'PY'
import pathlib
import shutil

removed = 0
for local_browsers in pathlib.Path("python-standalone").rglob(".local-browsers"):
    for webkit_payload in local_browsers.glob("webkit-*"):
        if webkit_payload.is_dir():
            shutil.rmtree(webkit_payload)
        else:
            webkit_payload.unlink(missing_ok=True)
        removed += 1
print(f"Removed {removed} stale WebKit browser payload(s) from python-standalone")
PY

echo "--- Normalizing python-standalone symlinks for PyInstaller ---"
python3 - <<'PY'
import pathlib
import shutil

root = pathlib.Path("python-standalone")
replaced = 0
skipped = 0


def _should_skip_symlink(path: pathlib.Path) -> bool:
    # Preserve any nested app/framework symlinks that third-party payloads may
    # carry. Bundled macOS Playwright WebKit is intentionally excluded earlier.
    parts = path.parts
    return (
        ".local-browsers" in parts
        or any(part.endswith(".app") or part.endswith(".framework") for part in parts)
    )

for path in sorted(root.rglob("*")):
    if not path.is_symlink():
        continue
    if _should_skip_symlink(path):
        skipped += 1
        continue
    target = path.resolve()
    path.unlink()
    if target.is_dir():
        shutil.copytree(target, path)
    else:
        shutil.copy2(target, path)
    replaced += 1

print(
    f"Replaced {replaced} symlinks in python-standalone "
    f"(skipped {skipped} inside bundled browser bundles)"
)
PY

echo "--- Building embedded managed repo bundle ---"
python3 scripts/build_repo_bundle.py --source-branch "$MANAGED_SOURCE_BRANCH"

rm -rf build dist

echo "--- Running PyInstaller ---"
python3 -m PyInstaller Ouroboros.spec --clean --noconfirm

echo "--- Installing packaged CLI wrappers ---"
CLI_BIN_DIR="$APP_PATH/Contents/Resources/bin"
mkdir -p "$CLI_BIN_DIR"
cp packaging/cli/ouroboros "$CLI_BIN_DIR/ouroboros"
cp packaging/cli/install-ouroboros-cli "$CLI_BIN_DIR/install-ouroboros-cli"
chmod +x "$CLI_BIN_DIR/ouroboros" "$CLI_BIN_DIR/install-ouroboros-cli"

# WA6 (macOS codesign integrity): precompile + seal .pyc instead of deleting them.
# A signed+notarized .app must not write __pycache__/*.pyc into its own bundle at
# runtime — that mutation breaks the codesign seal and triggers AppTranslocation
# ("Reconnecting"). Previously we deleted all bytecode before signing, leaving the
# runtime to regenerate it inside the sealed bundle. Instead, precompile now so the
# .pyc EXIST and get SEALED inside the signature; nothing is left for the runtime to
# write. --invalidation-mode unchecked-hash means a read-only bundle never rewrites
# them (no source-mtime check at import). The env-level PYTHONDONTWRITEBYTECODE guards
# (launcher.py / embedded_python_env) remain as defense-in-depth.
echo "--- Precompiling Python bytecode inside app bundle (sealed before signing) ---"
# Find the bundled embedded interpreter inside the .app so the .pyc magic matches the
# interpreter that runs at launch; fall back to the build-host copy (same standalone build).
APP_EMBEDDED_PY="$(find "$APP_PATH" -type f -path '*/python-standalone/bin/python3' 2>/dev/null | head -1)"
if [ -z "$APP_EMBEDDED_PY" ]; then
    APP_EMBEDDED_PY="$PWD/python-standalone/bin/python3"
fi
echo "Using embedded interpreter for compileall: $APP_EMBEDDED_PY"
# Compile both the bundled stdlib/site-packages (python-standalone) and the ouroboros
# payload trees that PyInstaller copies under Contents/Resources.
COMPILE_TARGETS=()
while IFS= read -r d; do
    [ -n "$d" ] && COMPILE_TARGETS+=("$d")
done < <(find "$APP_PATH" -type d \( -path '*/python-standalone' -o -name ouroboros \) 2>/dev/null)
if [ "${#COMPILE_TARGETS[@]}" -gt 0 ]; then
    # CRITICAL: neutralize the build-time PYTHONDONTWRITEBYTECODE=1 + PYTHONPYCACHEPREFIX
    # (set at the top of this script) for THIS command only — otherwise compileall
    # writes ZERO in-tree .pyc (dont-write-bytecode) or redirects them to the temp
    # prefix, and the seal would seal nothing.
    env -u PYTHONDONTWRITEBYTECODE -u PYTHONPYCACHEPREFIX \
        "$APP_EMBEDDED_PY" -m compileall -q -f --invalidation-mode unchecked-hash "${COMPILE_TARGETS[@]}" || true
    # Python console-script sources under python-standalone/bin are launchers,
    # not importable runtime modules. compileall still descends into that folder
    # (for example bottle.py -> bin/__pycache__/bottle.*.pyc), but macOS treats a
    # nested `bin` directory as code-bearing and codesign then refuses the data
    # bytecode as an unsigned code object. Keep sealed bytecode in stdlib and
    # site-packages, while removing only these non-imported launcher caches.
    find "$APP_PATH" -type d -path '*/python-standalone/bin/__pycache__' \
        -prune -exec rm -rf {} + 2>/dev/null || true
    # Post-condition: the seal is only meaningful if .pyc actually landed in-tree.
    if [ -z "$(find "$APP_PATH" -name '*.pyc' -type f -print -quit 2>/dev/null)" ]; then
        echo "ERROR: precompile produced no in-bundle .pyc — the codesign seal would not cover bytecode." >&2
        exit 1
    fi
else
    echo "WARNING: no compileall targets found inside $APP_PATH (python-standalone / ouroboros)."
fi

echo "--- Pruning stray detritus from app bundle (keeping freshly-built .pyc) ---"
# Only remove dotfile/resource-fork detritus; do NOT delete the sealed .pyc/__pycache__.
find "$APP_PATH" -name '._*' -type f -delete 2>/dev/null || true
find "$APP_PATH" -name '.DS_Store' -type f -delete 2>/dev/null || true
# Strip extended attributes / FinderInfo / resource forks so codesign --strict passes.
xattr -cr "$APP_PATH" 2>/dev/null || true

if [ "$SIGN_MODE" != "0" ]; then
    echo ""
    echo "=== Signing Ouroboros.app ==="

    echo "--- Finding and signing all Mach-O binaries ---"
    find "$APP_PATH" -type f | while read -r f; do
        if file "$f" | grep -q "Mach-O"; then
            codesign -s "$SIGN_IDENTITY" --timestamp --force --options runtime \
                --entitlements "$ENTITLEMENTS" "$f" 2>&1 || true
        fi
    done
    # A failed best-effort nested sign can leave an atomic-replace scratch file.
    # `--deep` treats that stale .cstemp as a missing code object and refuses the
    # bundle, so prune only codesign's own temporary artifacts before the seal.
    find "$APP_PATH" -name '*.cstemp' -type f -delete 2>/dev/null || true
    echo "Signed embedded binaries"

    echo "--- Signing the app bundle ---"
    # PyInstaller places the standalone Python tree under Contents/Frameworks.
    # macOS therefore classifies its sealed .pyc files as nested code objects;
    # outer signing without --deep refuses them as unsigned. Mach-O children
    # already carry their explicit signatures above; deep signing seals the
    # remaining bytecode objects into the final app hierarchy.
    codesign -s "$SIGN_IDENTITY" --timestamp --force --options runtime --deep \
        --entitlements "$ENTITLEMENTS" "$APP_PATH"

    echo "--- Verifying signature ---"
    codesign -dvv "$APP_PATH"
    codesign --verify --strict --deep "$APP_PATH"
    echo "Signature OK"
else
    echo ""
    echo "=== Skipping signing (OUROBOROS_SIGN=0) ==="
fi

echo ""
echo "=== Creating DMG ==="
rm -f "$DMG_PATH"
DMG_STAGE_DIR="dist/dmg-stage"
rm -rf "$DMG_STAGE_DIR"
mkdir -p "$DMG_STAGE_DIR"
cp -R "$APP_PATH" "$DMG_STAGE_DIR/Ouroboros.app"
ln -s /Applications "$DMG_STAGE_DIR/Applications"
cp packaging/cli/install-ouroboros-cli-macos.command "$DMG_STAGE_DIR/Install CLI.command"
chmod +x "$DMG_STAGE_DIR/Install CLI.command"
for attempt in 1 2 3; do
    if hdiutil create -volname Ouroboros -srcfolder "$DMG_STAGE_DIR" -ov -format UDZO "$DMG_PATH"; then
        break
    fi
    if [ "$attempt" -eq 3 ]; then
        echo "ERROR: hdiutil create failed after $attempt attempts."
        exit 1
    fi
    echo "WARNING: hdiutil create failed (attempt $attempt/3); waiting for disk image helpers to settle."
    hdiutil detach "/Volumes/Ouroboros" -force >/dev/null 2>&1 || true
    pkill -f diskimages-helper >/dev/null 2>&1 || true
    rm -f "$DMG_PATH"
    sleep 5
done

if [ "$SIGN_MODE" != "0" ]; then
    codesign -s "$SIGN_IDENTITY" --timestamp "$DMG_PATH"
fi

# Optional notarization only after signing and complete Apple credentials.
# Outcome enum keeps final summary honest: success/staple_failed/submit_failed/unconfigured.
NOTARIZE_OUTCOME="unconfigured"
if [ "$SIGN_MODE" != "0" ] \
        && [ -n "${APPLE_ID:-}" ] \
        && [ -n "${APPLE_TEAM_ID:-}" ] \
        && [ -n "${APPLE_APP_SPECIFIC_PASSWORD:-}" ]; then
    echo ""
    echo "=== Notarizing DMG (Apple ID: $APPLE_ID) ==="
    # Submit failures warn, not abort; signed DMG still ships with clear logs.
    if xcrun notarytool submit "$DMG_PATH" \
            --apple-id "$APPLE_ID" \
            --team-id "$APPLE_TEAM_ID" \
            --password "$APPLE_APP_SPECIFIC_PASSWORD" \
            --wait; then
        echo "--- Stapling notarization ticket ---"
        # Stapler can fail after successful notarization; Gatekeeper can fetch online.
        if xcrun stapler staple "$DMG_PATH"; then
            NOTARIZE_OUTCOME="success"
        else
            NOTARIZE_OUTCOME="staple_failed"
            echo "WARNING: stapler staple failed — DMG is notarized but ticket not embedded; receivers may briefly need right-click → Open until Apple's ticket propagates."
        fi
    else
        NOTARIZE_OUTCOME="submit_failed"
        echo "WARNING: notarytool submit failed — DMG is signed but not notarized; verify APPLE_ID / APPLE_TEAM_ID / APPLE_APP_SPECIFIC_PASSWORD are correct or check the notarytool log above."
    fi
fi

echo ""
echo "=== Done ==="
if [ "$SIGN_MODE" != "0" ]; then
    echo "Signed app: $APP_PATH"
    echo "Signed DMG: $DMG_PATH"
else
    echo "Unsigned app: $APP_PATH"
    echo "Unsigned DMG: $DMG_PATH"
fi
case "$NOTARIZE_OUTCOME" in
    success)
        echo "(Notarized + stapled — no right-click → Open required on first launch)"
        ;;
    staple_failed)
        echo "(Notarized but ticket not stapled — Gatekeeper will fetch the ticket online; receivers need internet on first launch)"
        ;;
    submit_failed)
        echo "(Signed but notarytool submit failed — DMG was not accepted by Apple; check the WARNING above for details)"
        ;;
    unconfigured)
        if [ "$SIGN_MODE" != "0" ]; then
            echo "(Signed but not notarized — set APPLE_ID / APPLE_TEAM_ID / APPLE_APP_SPECIFIC_PASSWORD to enable notarization)"
        else
            echo "(Not notarized — users need right-click → Open on first launch)"
        fi
        ;;
    *)
        # Surface future enum drift instead of omitting the summary.
        echo "(Unknown notarization outcome: '$NOTARIZE_OUTCOME' — please report; likely a missing case arm in build.sh)"
        ;;
esac

#!/bin/bash
# Tagless, secretless diagnostic AppImage, using the production spec/closure/packer.
# Run under safe_test in native-qualification.Dockerfile's builder, never on a host.
set -euo pipefail
EVIDENCE=$(realpath "$1")
: "${OUROBOROS_TEST_TEMP_ROOT:?run through scripts/safe_test.py}"
test "$(uname -sm)" = 'Linux x86_64'
test -z "$(git status --porcelain --untracked-files=no)"
export CC=gcc CXX=g++ LDSHARED='gcc -shared' PYTHONDONTWRITEBYTECODE=1
export UV_PYTHON_DOWNLOADS=never OUROBOROS_BUNDLE_LINUX_GTK=1
mkdir -p "$EVIDENCE/apt-release"
# The ordinary archive is not a snapshot: keep the signed index state used.
cp /etc/apt/sources.list /etc/apt/sources.list.d/native-sources.list \
    /var/lib/apt/lists/*_InRelease "$EVIDENCE/apt-release/"
# Fail before the long build when a declared closure package is not installed.
python3 -I -c 'import runpy, subprocess; subprocess.run(["dpkg-query", "-W",
    *runpy.run_path("scripts/bundle_linux_gtk.py")["PACKAGES"]], check=True)' > "$EVIDENCE/closure-packages.tsv"
git rev-parse HEAD > "$EVIDENCE/source-sha.txt"
git rev-parse HEAD^{tree} > "$EVIDENCE/source-tree.txt"
sha256sum pyproject.toml uv.lock requirements-runtime.lock Ouroboros.spec \
    scripts/bundle_linux_gtk.py scripts/build_appimage.sh > "$EVIDENCE/build-inputs.sha256"
bash scripts/download_python_standalone.sh
bash scripts/download_node_standalone.sh
bash scripts/download_ripgrep_standalone.sh
PY="$PWD/python-standalone/bin/python3"
BUILD_ENV="$OUROBOROS_TEST_TEMP_ROOT/build-venv"
"$PY" -m venv "$BUILD_ENV"
uv export --locked --no-dev --extra browser --extra desktop --extra linux-desktop --extra build \
    --no-emit-project --no-hashes --no-annotate -o "$EVIDENCE/build-requirements.txt"
uv pip install --python "$BUILD_ENV/bin/python" -r "$EVIDENCE/build-requirements.txt"
uv export --locked --no-dev --extra browser --extra desktop --extra linux-desktop \
    --no-emit-project --no-hashes --no-annotate -o "$EVIDENCE/payload-requirements.txt"
uv pip install --python "$PY" -r "$EVIDENCE/payload-requirements.txt"
"$PY" -m ouroboros.betterleaks_runtime install --build-output betterleaks-standalone
"$PY" scripts/fetch_claudexor_runtime.py --output-dir claudexor-runtime
"$PY" -m playwright install-deps chromium webkit
PLAYWRIGHT_BROWSERS_PATH=0 "$PY" -m playwright install chromium webkit
# No release tag is created or validation weakened. This diagnostic seed is
# intentionally tagless; the normal build_linux.sh release guard is unchanged.
git bundle create repo.bundle HEAD
"$PY" -I -S - <<'PY'
import hashlib, json, pathlib, subprocess
root=pathlib.Path.cwd()
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
manifest={'schema_version':1, 'bundle_file':'repo.bundle', 'source_sha':sha,
          'app_version':(root/'VERSION').read_text().strip(), 'release_tag':'',
          'bundle_sha256':hashlib.sha256((root/'repo.bundle').read_bytes()).hexdigest(),
          'source_branch':'native-qualification', 'managed_remote_name':'managed',
          'managed_remote_url':'https://example.invalid/qualification-only.git',
          'managed_remote_branch':'native-qualification', 'managed_local_branch':'ouroboros',
          'managed_local_stable_branch':'ouroboros-stable', 'managed_remote_stable_branch':'ouroboros-stable'}
(root/'repo_bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
"$BUILD_ENV/bin/python" -m PyInstaller Ouroboros.spec --clean --noconfirm
"$BUILD_ENV/bin/python" scripts/bundle_linux_gtk.py dist/Ouroboros/_internal
mkdir -p dist/Ouroboros/bin
cp packaging/cli/ouroboros packaging/cli/install-ouroboros-cli dist/Ouroboros/bin/
chmod +x dist/Ouroboros/bin/*
bash scripts/build_appimage.sh
cp dist/*.AppImage "$EVIDENCE/"
cp dist/Ouroboros/_internal/linux-gtk/manifest.json "$EVIDENCE/gtk-manifest.json"
# Production and diagnostic builds retain the same exact source closure.
"$PY" -I -S scripts/capture_linux_sources.py dist/Ouroboros/_internal/linux-gtk \
    "$EVIDENCE/native-sources-linux.tar.gz"
cp repo_bundle_manifest.json "$EVIDENCE/"
sha256sum "$EVIDENCE/"*.AppImage > "$EVIDENCE/appimage.sha256"
dpkg-query -W > "$EVIDENCE/builder-packages.tsv"
cp /etc/apt/sources.list "$EVIDENCE/apt-sources.list"
# Dependency-only environment; every launcher test retains the network guard.
"$BUILD_ENV/bin/python" -m pytest tests/test_appimage_packaging.py tests/test_desktop_media.py \
    tests/test_launcher_source_handoff.py tests/test_launcher_headless_fallback.py \
    tests/test_build_scripts.py tests/test_linux_desktop_environment.py \
    tests/test_linux_native_source_capture.py tests/test_x11_window_identity.py -o addopts='' --junitxml="$EVIDENCE/focused-tests.xml"

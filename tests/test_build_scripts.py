"""Regression tests for platform build scripts.

These tests ensure each build script contains the critical Playwright install
steps for its platform policy with the correct env-var flag and that bundled
install steps appear BEFORE the actual PyInstaller command-line invocation.
Linux/Windows/Docker bundle Chromium and WebKit; macOS bundles the Chromium
headless shell and leaves WebKit to the managed runtime cache because the
Playwright WebKit payload does not survive the signed PyInstaller app layout.

In v5.15.x this module also absorbed packaging-asset completeness checks
(formerly tests/test_packaging_assets.py) and the CI release-workflow
contract checks (formerly tests/test_release_workflow.py) so packaging
contracts evolve in one place.
"""
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent
REPO = REPO_ROOT


def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def _find_pyinstaller_cmd_pos(src: str) -> int:
    """Return the character position of the first line that actually *runs*
    PyInstaller (i.e. contains 'PyInstaller' outside a comment/echo line).

    Build scripts have comment lines and echo lines mentioning 'PyInstaller'
    before the actual invocation; we need the real command line.
    """
    for match in re.finditer(r"PyInstaller", src):
        # Find the start of the line containing this match.
        line_start = src.rfind("\n", 0, match.start()) + 1
        line = src[line_start: src.find("\n", match.start())]
        stripped = line.strip()
        # Skip comment lines (bash: '#', PowerShell: '#') and echo/Write-Host.
        if stripped.startswith("#") or stripped.lower().startswith("echo") or stripped.lower().startswith("write-host"):
            continue
        return match.start()
    return -1


# ---------------------------------------------------------------------------
# build.sh  (macOS)
# ---------------------------------------------------------------------------

class TestBuildSh:
    """build.sh must bundle Chromium headless shell without bundled macOS WebKit."""

    def test_playwright_install_chromium_present(self):
        src = _read("build.sh")
        assert "playwright install --only-shell chromium" in src, (
            "build.sh must call 'playwright install --only-shell chromium' on macOS"
        )
        assert "Skipping bundled WebKit on macOS" in src, (
            "build.sh must document the macOS WebKit packaging policy"
        )
        assert "Removing stale bundled WebKit payloads from macOS package tree" in src
        assert 'rglob(".local-browsers")' in src
        assert 'glob("webkit-*")' in src
        assert "shutil.rmtree" in src
        assert "managed" in src and "Playwright cache" in src, (
            "build.sh must leave WebKit available through the managed runtime cache"
        )
        assert "playwright install webkit" not in src
        assert "pw.webkit.launch" not in src

    def test_playwright_browsers_path_zero_set(self):
        src = _read("build.sh")
        assert "PLAYWRIGHT_BROWSERS_PATH=0" in src, (
            "build.sh must set PLAYWRIGHT_BROWSERS_PATH=0 for the playwright install step"
        )

    def test_playwright_install_before_pyinstaller(self):
        src = _read("build.sh")
        pw_pos = src.find("playwright install --only-shell chromium")
        pi_pos = _find_pyinstaller_cmd_pos(src)
        assert pw_pos != -1, "playwright install --only-shell chromium not found in build.sh"
        assert pi_pos != -1, "PyInstaller command not found in build.sh"
        assert pw_pos < pi_pos, (
            "playwright install --only-shell chromium must appear BEFORE PyInstaller in build.sh "
            f"(found at char {pw_pos}, PyInstaller cmd at {pi_pos})"
        )

    def test_repo_bundle_generation_before_pyinstaller(self):
        src = _read("build.sh")
        bundle_pos = src.find("scripts/build_repo_bundle.py")
        pi_pos = _find_pyinstaller_cmd_pos(src)
        assert bundle_pos != -1, "build.sh must generate repo.bundle before packaging"
        assert "--source-branch" in src, "build.sh must pass an explicit source branch for detached-head builds"
        assert pi_pos != -1, "PyInstaller command not found in build.sh"
        assert bundle_pos < pi_pos, "repo bundle generation must happen before PyInstaller in build.sh"

    def test_ripgrep_download_before_pyinstaller(self):
        src = _read("build.sh")
        rg_pos = src.find("download_ripgrep_standalone.sh")
        pi_pos = _find_pyinstaller_cmd_pos(src)
        assert rg_pos != -1
        assert pi_pos != -1
        assert rg_pos < pi_pos

    def test_ripgrep_download_script_verifies_checksum(self):
        src = _read("scripts/download_ripgrep_standalone.sh")
        assert ".sha256" in src
        assert "hashlib.sha256" in src
        assert "SHA256 mismatch" in src

    def test_repo_bundle_delegates_release_tag_validation_to_python_ssot(self):
        src = _read("build.sh")
        assert 'scripts/build_repo_bundle.py' in src
        assert 'refs/tags/$RELEASE_TAG' not in src
        assert 'git tag --points-at HEAD' not in src
        assert 'OUROBOROS_RELEASE_TAG="$RELEASE_TAG"' not in src

    def test_repo_bundle_script_has_no_duplicate_annotated_tag_guard(self):
        src = _read("build.sh")
        assert 'git cat-file -t "refs/tags/$RELEASE_TAG"' not in src
        assert 'requires annotated git tag' not in src

    def test_packaged_cli_wrappers_are_copied_before_signing_and_dmg(self):
        src = _read("build.sh")
        wrapper_pos = src.find("Installing packaged CLI wrappers")
        precompile_pos = src.find("Precompiling Python bytecode inside app bundle")
        sign_pos = src.find("=== Signing Ouroboros.app ===")
        dmg_pos = src.find("=== Creating DMG ===")
        assert wrapper_pos != -1
        assert "Contents/Resources/bin" in src
        assert "packaging/cli/ouroboros" in src
        assert "packaging/cli/install-ouroboros-cli" in src
        assert "PYTHONDONTWRITEBYTECODE=1" in src
        assert "PYTHONPYCACHEPREFIX" in src
        # WA6: precompile + seal .pyc (not delete) so the signature stays valid.
        assert "compileall" in src and "--invalidation-mode unchecked-hash" in src
        assert sign_pos != -1 and wrapper_pos < sign_pos
        assert precompile_pos != -1 and precompile_pos < sign_pos  # sealed before signing
        assert dmg_pos != -1 and wrapper_pos < dmg_pos

    def test_precompiles_and_seals_bytecode_before_signing(self):
        """WA6: precompile .pyc so they are SEALED inside the signature (rather than
        deleted, letting the runtime regenerate them inside the bundle and break the
        codesign seal). xattr hygiene + the --strict verify gate remain."""
        src = _read("build.sh")
        precompile_pos = src.find("compileall")
        sign_pos = src.find("=== Signing Ouroboros.app ===")
        assert precompile_pos != -1, "build.sh must precompile bytecode before signing"
        assert "--invalidation-mode unchecked-hash" in src
        assert precompile_pos < sign_pos
        assert 'xattr -cr "$APP_PATH"' in src
        assert "codesign --verify --strict" in src
        # CRITICAL-1: the build-time PYTHONDONTWRITEBYTECODE=1 + PYTHONPYCACHEPREFIX
        # (exported at the top of build.sh) MUST be neutralized for the compileall
        # command, else it writes ZERO in-tree .pyc and the seal seals nothing.
        assert "env -u PYTHONDONTWRITEBYTECODE -u PYTHONPYCACHEPREFIX" in src, (
            "build.sh must neutralize PYTHONDONTWRITEBYTECODE/PYTHONPYCACHEPREFIX "
            "around compileall (else the seal covers no bytecode — WA6 no-op)"
        )
        # Post-condition guard: fail the build if no in-bundle .pyc actually landed.
        assert "precompile produced no in-bundle .pyc" in src, (
            "build.sh must verify .pyc actually landed in-bundle after compileall "
            "(a silent no-op would otherwise ship a self-resealing bundle)"
        )
        # Console-script sources are launchers, not imported runtime modules.
        # Their pyc files under a nested bin/ directory are treated by macOS as
        # unsigned code objects and make the outer app signature fail closed.
        assert "python-standalone/bin/__pycache__" in src
        assert "-prune -exec rm -rf {} +" in src
        assert "-name '*.cstemp' -type f -delete" in src, (
            "best-effort nested signing may leave scratch files that make the "
            "final deep-sign refuse a missing code object"
        )
        assert "--options runtime --deep" in src, (
            "the outer app signature must seal Python bytecode nested beneath "
            "Contents/Frameworks"
        )
        assert "codesign --verify --strict --deep" in src

    def test_macos_dmg_includes_finder_install_layout(self):
        src = _read("build.sh")
        assert "dmg-stage" in src
        assert 'ln -s /Applications "$DMG_STAGE_DIR/Applications"' in src
        assert "Install CLI.command" in src
        assert "install-ouroboros-cli-macos.command" in src

    def test_macos_dmg_cli_command_delegates_installed_app_before_refusing_dmg(self):
        src = _read("packaging/cli/install-ouroboros-cli-macos.command")
        installed_check = src.find('if [ -x "$INSTALLED_CLI" ]; then')
        refusal = src.find('Install Ouroboros.app to /Applications')
        assert installed_check != -1
        assert refusal != -1
        assert installed_check < refusal

    def test_symlink_normalizer_skips_playwright_browser_bundles(self):
        src = _read("build.sh")
        assert "_should_skip_symlink" in src, (
            "build.sh should centralize the macOS symlink-skip guard for bundled "
            "browser bundles"
        )
        assert ".local-browsers" in src, (
            "build.sh must skip symlink normalization inside Playwright's bundled "
            "browser tree on macOS"
        )
        assert ".app" in src and ".framework" in src, (
            "build.sh must preserve nested macOS app/framework bundles during "
            "symlink normalization"
        )


def test_compileall_env_neutralization_actually_seals_bytecode(tmp_path):
    """BEHAVIORAL guard for WA6 / CRITICAL-1 (the string checks above can't catch
    an env re-inheritance regression). Every build script exports
    PYTHONDONTWRITEBYTECODE=1 + PYTHONPYCACHEPREFIX at the top; that SUPPRESSES
    in-tree .pyc. The codesign-seal compileall step therefore MUST run with those
    vars neutralized — otherwise it seals ZERO bytecode and the macOS .app
    re-generates .pyc into its own signed bundle at runtime (the original bug).

    This asserts the underlying mechanism end-to-end with the real interpreter:
      - WITH the suppression env  -> compileall writes NO in-tree .pyc (the bug),
      - WITHOUT it (neutralized)  -> .pyc land in-tree (the fix the seal relies on).
    """
    import os
    import subprocess
    import sys

    def _compile(extra_env: dict) -> list:
        pkg = tmp_path / ("sup" if extra_env else "clean")
        pkg.mkdir()
        (pkg / "mod.py").write_text("x = 1\n", encoding="utf-8")
        env = dict(os.environ)
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        env.pop("PYTHONPYCACHEPREFIX", None)
        env.update(extra_env)
        subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "-f",
             "--invalidation-mode", "unchecked-hash", str(pkg)],
            env=env, check=False, capture_output=True,
        )
        return list(pkg.rglob("*.pyc"))

    # The suppression env every build script exports -> no in-tree .pyc.
    suppressed = _compile({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(tmp_path / "prefix"),
    })
    assert suppressed == [], (
        "PYTHONDONTWRITEBYTECODE/PYTHONPYCACHEPREFIX must suppress in-tree .pyc — "
        "this is WHY the build scripts must neutralize them around compileall"
    )
    # Neutralized env (what the build scripts do around compileall) -> .pyc sealed.
    sealed = _compile({})
    assert sealed, (
        "with the suppression env neutralized, compileall MUST write in-tree .pyc "
        "(the macOS codesign seal depends on it); if this fails the WA6 fix is a no-op"
    )


# ---------------------------------------------------------------------------
# build_linux.sh  (Linux)
# ---------------------------------------------------------------------------

class TestBuildLinuxSh:
    """build_linux.sh must install Chromium/WebKit with PLAYWRIGHT_BROWSERS_PATH=0 before PyInstaller."""

    def test_playwright_install_chromium_present(self):
        src = _read("build_linux.sh")
        assert "playwright install chromium webkit" in src
        assert "playwright install-deps chromium webkit" in src

    def test_playwright_browsers_path_zero_set(self):
        src = _read("build_linux.sh")
        assert "PLAYWRIGHT_BROWSERS_PATH=0" in src

    def test_playwright_install_before_pyinstaller(self):
        src = _read("build_linux.sh")
        deps_pos = src.find("playwright install-deps chromium webkit")
        pw_pos = src.find("playwright install chromium webkit")
        pi_pos = _find_pyinstaller_cmd_pos(src)
        assert deps_pos != -1
        assert pw_pos != -1
        assert pi_pos != -1
        assert deps_pos < pw_pos
        assert pw_pos < pi_pos, (
            "playwright install chromium webkit must appear BEFORE PyInstaller in build_linux.sh"
        )

    def test_repo_bundle_generation_before_pyinstaller(self):
        src = _read("build_linux.sh")
        bundle_pos = src.find("scripts/build_repo_bundle.py")
        pi_pos = _find_pyinstaller_cmd_pos(src)
        assert bundle_pos != -1
        assert "--source-branch" in src
        assert pi_pos != -1
        assert bundle_pos < pi_pos, "repo bundle generation must happen before PyInstaller in build_linux.sh"

    def test_pyinstaller_uses_the_packaged_portable_python(self):
        src = _read("build_linux.sh")
        assert 'PORTABLE_PYTHON="python-standalone/bin/python3"' in src
        assert '"$PORTABLE_PYTHON" -m venv "$BUILD_VENV"' in src
        assert "uv export --locked --no-dev --extra browser --extra desktop --extra build" in src
        assert 'uv pip install --python "$BUILD_PYTHON" -q -r "$BUILD_REQUIREMENTS"' in src
        assert '"$BUILD_PYTHON" -m PyInstaller Ouroboros.spec' in src
        assert 'uv pip install --python "$PORTABLE_PYTHON" -q -r requirements-runtime.lock' in src
        assert '"$HOST_PYTHON_CMD" -m PyInstaller' not in src

    def test_ripgrep_download_before_pyinstaller(self):
        src = _read("build_linux.sh")
        rg_pos = src.find("download_ripgrep_standalone.sh")
        pi_pos = _find_pyinstaller_cmd_pos(src)
        assert rg_pos != -1
        assert pi_pos != -1
        assert rg_pos < pi_pos

    def test_ripgrep_download_script_verifies_checksum(self):
        src = _read("scripts/download_ripgrep_standalone.sh")
        assert ".sha256" in src
        assert "hashlib.sha256" in src
        assert "SHA256 mismatch" in src

    def test_repo_bundle_delegates_release_tag_validation_to_python_ssot(self):
        src = _read("build_linux.sh")
        assert 'scripts/build_repo_bundle.py' in src
        assert 'refs/tags/$RELEASE_TAG' not in src
        assert 'git tag --points-at HEAD' not in src
        assert 'OUROBOROS_RELEASE_TAG="$RELEASE_TAG"' not in src

    def test_repo_bundle_script_has_no_duplicate_annotated_tag_guard(self):
        src = _read("build_linux.sh")
        assert 'git cat-file -t "refs/tags/$RELEASE_TAG"' not in src
        assert 'requires annotated git tag' not in src

    def test_linux_archive_includes_packaged_cli_bin(self):
        src = _read("build_linux.sh")
        wrapper_pos = src.find("Installing packaged CLI wrappers")
        precompile_pos = src.find("Precompiling Python bytecode in archive payload")
        archive_pos = src.find("=== Creating archive ===")
        assert wrapper_pos != -1
        assert "dist/Ouroboros/bin" in src
        assert "packaging/cli/ouroboros" in src
        assert "packaging/cli/install-ouroboros-cli" in src
        assert "PYTHONDONTWRITEBYTECODE=1" in src
        assert "PYTHONPYCACHEPREFIX" in src
        # WA6 parity: precompile instead of deleting .pyc.
        assert "compileall" in src and "--invalidation-mode unchecked-hash" in src
        # CRITICAL-1 parity: neutralize the suppression env around compileall so the
        # in-tree .pyc actually get written (start-speed parity with the macOS seal).
        assert "env -u PYTHONDONTWRITEBYTECODE -u PYTHONPYCACHEPREFIX" in src, (
            "build_linux.sh must neutralize PYTHONDONTWRITEBYTECODE/PYTHONPYCACHEPREFIX "
            "around compileall (else compileall writes no in-tree .pyc)"
        )
        assert precompile_pos != -1 and precompile_pos < archive_pos
        assert archive_pos != -1 and wrapper_pos < archive_pos

    def test_posix_cli_wrappers_search_pyinstaller_internal_root_without_env_trust(self):
        for name in ("packaging/cli/ouroboros", "packaging/cli/install-ouroboros-cli"):
            src = _read(name)
            assert 'OUROBOROS_PACKAGED_BUNDLE_ROOT:-' not in src
            assert '$SCRIPT_DIR/../_internal' in src


# ---------------------------------------------------------------------------
# build_windows.ps1  (Windows / PowerShell)
# ---------------------------------------------------------------------------

class TestBuildWindowsPs1:
    """build_windows.ps1 must install Chromium/WebKit with PLAYWRIGHT_BROWSERS_PATH=0 before PyInstaller."""

    def test_critical_native_commands_fail_fast_on_windows_powershell_5(self):
        src = _read("build_windows.ps1")
        assert "function Invoke-NativeChecked" in src
        assert "$ExitCode = $LASTEXITCODE" in src
        assert "if ($ExitCode -ne 0)" in src
        for description in (
            "Node.js runtime download",
            "Launcher dependency installation",
            "ripgrep runtime download",
            "Agent dependency installation",
            "Claudexor runtime seed fetch",
            "Chromium installation",
            "WebKit installation",
            "Managed repo bundle build",
            "PyInstaller build",
        ):
            assert f'Invoke-NativeChecked "{description}"' in src

    def test_playwright_install_chromium_present(self):
        src = _read("build_windows.ps1")
        assert "playwright install --only-shell chromium" in src
        assert "playwright install webkit" in src

    def test_playwright_browsers_path_zero_set(self):
        src = _read("build_windows.ps1")
        # PowerShell syntax: $env:PLAYWRIGHT_BROWSERS_PATH = "0"
        assert 'PLAYWRIGHT_BROWSERS_PATH' in src and '"0"' in src, (
            "build_windows.ps1 must set PLAYWRIGHT_BROWSERS_PATH to '0'"
        )

    def test_playwright_install_before_pyinstaller(self):
        src = _read("build_windows.ps1")
        pw_pos = src.find("playwright install --only-shell chromium")
        webkit_pos = src.find("playwright install webkit")
        pi_pos = _find_pyinstaller_cmd_pos(src)
        assert pw_pos != -1
        assert webkit_pos != -1
        assert pi_pos != -1
        assert pw_pos < webkit_pos
        assert pw_pos < pi_pos, (
            "playwright install --only-shell chromium must appear BEFORE PyInstaller in build_windows.ps1"
        )
        assert webkit_pos < pi_pos, (
            "playwright install webkit must appear BEFORE PyInstaller in build_windows.ps1"
        )

    def test_windows_build_has_path_length_guard(self):
        src = _read("build_windows.ps1")
        assert "Checking Windows archive path lengths" in src
        assert "Length -gt 200" in src
        assert "paths longer than 200 chars" in src

    def test_windows_build_prunes_optional_long_chromium_paths(self):
        src = _read("build_windows.ps1")
        assert "Pruning optional Chromium resources with long Windows paths" in src
        assert "PrivacySandboxAttestationsPreloaded" in src
        assert "reading_mode_gdocs_helper" in src

    def test_repo_bundle_generation_before_pyinstaller(self):
        src = _read("build_windows.ps1")
        bundle_pos = src.find("scripts/build_repo_bundle.py")
        pi_pos = _find_pyinstaller_cmd_pos(src)
        assert bundle_pos != -1
        assert "--source-branch" in src
        assert pi_pos != -1
        assert bundle_pos < pi_pos, "repo bundle generation must happen before PyInstaller in build_windows.ps1"

    def test_ripgrep_download_before_pyinstaller(self):
        src = _read("build_windows.ps1")
        rg_pos = src.find("download_ripgrep_standalone.ps1")
        pi_pos = _find_pyinstaller_cmd_pos(src)
        assert rg_pos != -1
        assert pi_pos != -1
        assert rg_pos < pi_pos

    def test_ripgrep_download_script_verifies_checksum(self):
        src = _read("scripts/download_ripgrep_standalone.ps1")
        assert ".sha256" in src
        assert "Get-FileHash" in src
        assert "SHA256 mismatch" in src

    def test_repo_bundle_delegates_release_tag_validation_to_python_ssot(self):
        src = _read("build_windows.ps1")
        assert 'scripts/build_repo_bundle.py' in src
        assert 'refs/tags/$ReleaseTag' not in src
        assert 'git tag --points-at HEAD' not in src
        assert '$env:OUROBOROS_RELEASE_TAG' not in src

    def test_repo_bundle_script_has_no_duplicate_annotated_tag_guard(self):
        src = _read("build_windows.ps1")
        assert 'git cat-file -t "refs/tags/$ReleaseTag"' not in src
        assert 'annotated git tag' not in src

    def test_windows_archive_includes_packaged_cli_bin(self):
        src = _read("build_windows.ps1")
        wrapper_pos = src.find("Installing packaged CLI wrappers")
        precompile_pos = src.find("Precompiling Python bytecode in archive payload")
        length_pos = src.find("Checking Windows archive path lengths")
        archive_pos = src.find("=== Creating archive ===")
        assert wrapper_pos != -1
        assert "dist\\Ouroboros\\bin" in src
        assert "packaging\\cli\\ouroboros.cmd" in src
        assert "packaging\\cli\\install-ouroboros-cli.cmd" in src
        assert "PYTHONDONTWRITEBYTECODE" in src
        assert "PYTHONPYCACHEPREFIX" in src
        # WA6 parity: precompile instead of deleting .pyc.
        assert "compileall" in src and "--invalidation-mode unchecked-hash" in src
        # CRITICAL-1 parity (PowerShell form): neutralize the suppression env around
        # compileall via Remove-Item Env: so in-tree .pyc actually get written.
        assert "Remove-Item Env:PYTHONDONTWRITEBYTECODE" in src, (
            "build_windows.ps1 must neutralize PYTHONDONTWRITEBYTECODE around "
            "compileall (else compileall writes no in-tree .pyc)"
        )
        # v6.36.1: compileall must TOLERATE per-file failures (a known broken Tcl/Tix
        # WmDefault.py) — parity with the POSIX `|| true`. Under pwsh 7.4+ a native
        # non-zero exit aborts the build unless Stop-on-native-error is neutralized
        # and $LASTEXITCODE is reset.
        assert '$ErrorActionPreference = "Continue"' in src, (
            "build_windows.ps1 must neutralize Stop-on-native-error around compileall "
            "so a single broken bundled file does not fail the Windows build"
        )
        assert "$global:LASTEXITCODE = 0" in src, (
            "build_windows.ps1 must reset $LASTEXITCODE after the tolerant compileall"
        )
        assert length_pos != -1 and wrapper_pos < length_pos
        assert precompile_pos != -1 and archive_pos != -1 and precompile_pos < archive_pos

    def test_windows_cli_wrappers_search_pyinstaller_internal_root(self):
        for name in ("packaging/cli/ouroboros.cmd", "packaging/cli/install-ouroboros-cli.cmd"):
            src = _read(name)
            assert r"%ROOT%\_internal\repo.bundle" in src
            assert 'set "ROOT=%ROOT%\\_internal"' in src


# ---------------------------------------------------------------------------
# Dockerfile  (Docker / web runtime)
# ---------------------------------------------------------------------------

class TestDockerfile:
    """Dockerfile must install Playwright Chromium/WebKit binaries so browser tools work
    out of the box in the container without additional setup."""

    def test_playwright_install_chromium_present(self):
        src = _read("Dockerfile")
        assert "playwright install chromium webkit" in src, (
            "Dockerfile must call 'playwright install chromium webkit' to bundle the browsers"
        )

    def test_playwright_browsers_path_zero_set(self):
        src = _read("Dockerfile")
        assert "PLAYWRIGHT_BROWSERS_PATH=0" in src, (
            "Dockerfile must set PLAYWRIGHT_BROWSERS_PATH=0 so Chromium installs "
            "inside the pip package tree (not into a user cache that won't survive "
            "image layer boundaries)"
        )

    def test_playwright_install_deps_present(self):
        """Dockerfile must use 'playwright install-deps chromium webkit' (the authoritative
        Playwright dependency resolver) rather than a hand-curated apt library list.
        This ensures all runtime native libs required by Chromium/WebKit are present."""
        src = _read("Dockerfile")
        assert "playwright install-deps chromium webkit" in src, (
            "Dockerfile must call 'playwright install-deps chromium webkit' to install all "
            "native system libraries required by Chromium/WebKit via Playwright's authoritative "
            "dependency resolver"
        )

    def test_install_deps_before_install_chromium(self):
        """Native system dependencies must be installed BEFORE the Chromium binary
        is downloaded, so the binary can find its runtime libraries on first launch."""
        src = _read("Dockerfile")
        deps_pos = src.find("playwright install-deps chromium webkit")
        src.find("playwright install chromium webkit")
        # binary_pos must not match the install-deps line itself
        # find the standalone 'playwright install chromium webkit' (not install-deps)
        import re as _re
        binary_match = _re.search(r"(?<!install-deps )playwright install chromium webkit", src)
        assert deps_pos != -1, "playwright install-deps chromium webkit not found in Dockerfile"
        assert binary_match is not None, "standalone playwright install chromium webkit not found in Dockerfile"
        assert deps_pos < binary_match.start(), (
            "playwright install-deps must appear BEFORE playwright install chromium webkit in Dockerfile"
        )

    def test_uv_sync_before_playwright_install_deps(self):
        """uv sync must appear BEFORE playwright install-deps chromium webkit — the
        playwright Python package must be importable when install-deps runs."""
        src = _read("Dockerfile")
        sync_pos = src.find("uv sync")
        deps_pos = src.find("playwright install-deps chromium webkit")
        assert sync_pos != -1, "uv sync step not found in Dockerfile"
        assert deps_pos != -1, "playwright install-deps chromium webkit not found in Dockerfile"
        assert sync_pos < deps_pos, (
            "uv sync must appear BEFORE playwright install-deps chromium webkit in Dockerfile "
            f"(sync at char {sync_pos}, install-deps at {deps_pos})"
        )

    def test_uv_sync_before_all_playwright_invocations(self):
        """uv sync must appear BEFORE every ``python3 -m playwright ...`` invocation
        in the Dockerfile — both ``install-deps`` and ``install chromium webkit``.
        If *any* playwright invocation precedes dependency sync, ModuleNotFoundError occurs."""
        src = _read("Dockerfile")
        sync_pos = src.find("uv sync")
        assert sync_pos != -1, "uv sync step not found in Dockerfile"

        import re as _re
        playwright_invocations = [
            m.start() for m in _re.finditer(r"python3 -m playwright", src)
        ]
        assert playwright_invocations, "No 'python3 -m playwright' invocations found in Dockerfile"

        earliest_playwright = min(playwright_invocations)
        assert sync_pos < earliest_playwright, (
            "uv sync must appear BEFORE the earliest 'python3 -m playwright' invocation "
            f"in the Dockerfile (sync at char {sync_pos}, earliest playwright at {earliest_playwright}). "
            f"Found {len(playwright_invocations)} playwright invocation(s) at positions: "
            f"{playwright_invocations}"
        )


@pytest.mark.parametrize(
    "path",
    [
        ".github/actions/setup-python-env/action.yml",
        "Dockerfile",
        "Makefile",
        "build.sh",
        "build_linux.sh",
        "build_windows.ps1",
    ],
)
def test_uv_project_commands_validate_lock_freshness(path):
    source = _read(path)
    commands = re.findall(r"uv (?:sync|run|export)[^\n]*", source)
    assert commands, f"no uv project command found in {path}"
    assert all("--locked" in command for command in commands), commands
    assert all("--frozen" not in command for command in commands), commands


def test_generated_runtime_lock_keeps_lf_on_windows():
    attributes = _read(".gitattributes").splitlines()
    assert "requirements-runtime.lock text eol=lf" in attributes


# ---------------------------------------------------------------------------
# scripts/build_repo_bundle.py  (release tag SSOT)
# ---------------------------------------------------------------------------

class TestRepoBundleReleaseTagGuard:
    """The platform scripts delegate release-tag integrity to the Python
    bundler so the invariant has one fail-closed implementation."""

    def test_bundler_requires_release_tag_at_head(self):
        src = _read("scripts/build_repo_bundle.py")
        assert "def _resolve_release_tag" in src
        assert '"tag", "--points-at", "HEAD"' in src
        assert "does not match VERSION" in src

    def test_bundler_requires_annotated_tag(self):
        src = _read("scripts/build_repo_bundle.py")
        assert "def _verify_release_tag_in_repo" in src
        assert '"cat-file", "-t"' in src
        assert "is not an annotated tag" in src

    def test_bundler_checks_tag_points_at_head_sha(self):
        src = _read("scripts/build_repo_bundle.py")
        assert '"rev-parse", "HEAD"' in src
        assert '"rev-list", "-1"' in src
        assert "does not point at HEAD" in src

    def test_dirty_tree_refusal_reports_the_offending_paths(self):
        src = _read("scripts/build_repo_bundle.py")
        assert '"status", "--porcelain"' in src
        assert '"Commit or stash changes first so the embedded managed repo matches the packaged code.\\n"' in src
        assert "+ status" in src


# ---------------------------------------------------------------------------
# .github/workflows/ci.yml + build.sh — macOS code signing & notarization
# ---------------------------------------------------------------------------

class TestMacOSSigning:
    """The CI build job and build.sh together implement optional macOS code
    signing and notarization. The contracts here prevent
    regression of the GitHub Actions `secrets.*`-in-step-`if:` pitfall, the
    build-script env override / optional-notarytool gate, the keychain
    cleanup guard, and the stapler-failure-as-soft-warning behaviour.

    See docs/DEVELOPMENT.md::"GitHub Actions: secrets in step-level if
    conditions" for the rationale.
    """

    _CI_PATH = ".github/workflows/ci.yml"
    _SIGNING_SECRETS = (
        "BUILD_CERTIFICATE_BASE64",
        "P12_PASSWORD",
        "KEYCHAIN_PASSWORD",
        "APPLE_TEAM_ID",
    )
    _NOTARIZE_SECRETS = (
        "APPLE_ID",
        "APPLE_APP_SPECIFIC_PASSWORD",
    )

    @staticmethod
    def _build_job_header(src: str) -> str:
        """Slice the build job header (everything between `  build:` and the
        first `    steps:` underneath it) so signing-secret env mappings can
        be located without false positives from later step-level env blocks."""
        build_idx = src.find("\n  build:\n")
        assert build_idx != -1, "build job not found in ci.yml"
        steps_idx = src.find("\n    steps:", build_idx)
        assert steps_idx != -1, "build.steps: not found in ci.yml"
        return src[build_idx:steps_idx]

    @staticmethod
    def _iter_step_if_blocks(src: str):
        """Yield every `if:` expression in the workflow as a flat string.

        Catches BOTH step-level and job-level `if:` blocks (the
        `Unrecognized named-value: 'secrets'` rejection applies at every
        level, so checking job-level too is strictly more conservative).

        Heuristic: collect lines starting from `if:` until the next YAML
        key starts (a line whose first non-space char is `-` or whose
        stripped form contains a `:`). Known limitation: a future `if:`
        whose continuation lines legitimately contain `:` (string literals,
        nested expressions) would be split prematurely; the current ci.yml
        has no such case. If that pattern is added, switch to a real YAML
        parser walking each step's `if` field.
        """
        lines = src.splitlines()
        in_if = False
        block: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("if:"):
                if in_if and block:
                    yield " ".join(block)
                in_if = True
                block = [stripped]
                continue
            if in_if:
                # Continuation: indented, not a new YAML key, not a step start.
                if stripped and not stripped.startswith("- ") and ":" not in stripped:
                    block.append(stripped)
                else:
                    yield " ".join(block)
                    in_if = False
                    block = []
        if in_if and block:
            yield " ".join(block)

    def test_ci_keeps_signing_values_step_scoped(self):
        """The build job exposes one non-secret gate globally and gives
        credential values only to the first-party steps that need them."""
        src = _read(self._CI_PATH)
        header = self._build_job_header(src)
        assert "HAS_APPLE_SIGNING:" in header
        assert "matrix.os == 'macos-latest'" in header
        for secret in self._SIGNING_SECRETS:
            assert f"secrets.{secret} != ''" in header
            assert not re.search(rf"^\s+{secret}:\s", header, re.MULTILINE)

        import_idx = src.index("- name: Import Apple signing certificate")
        import_region = src[import_idx : import_idx + 900]
        for secret in ("BUILD_CERTIFICATE_BASE64", "P12_PASSWORD", "KEYCHAIN_PASSWORD"):
            assert f"{secret}: ${{{{ secrets.{secret} }}}}" in import_region

        build_idx = src.index("- name: Build macOS app")
        build_region = src[build_idx : build_idx + 1000]
        for secret in ("APPLE_TEAM_ID", *self._NOTARIZE_SECRETS):
            assert f"{secret}: ${{{{ secrets.{secret} }}}}" in build_region

    def test_release_waits_for_non_provider_smoke_jobs(self):
        src = _read(self._CI_PATH)
        release_idx = src.find("\n  release:\n")
        assert release_idx != -1, "release job not found"
        release_block = src[release_idx:]
        needs_line = next(
            line.strip()
            for line in release_block.splitlines()
            if line.strip().startswith("needs:")
        )
        for job in ("marker-guards", "ui-smoke", "docker-ui-smoke", "docker-portable-test", "skill-smoke"):
            assert job in needs_line, f"release job must wait for {job}"
        assert "OUROBOROS_EXPECT_BROWSER_ENGINES: chromium,webkit" in src
        assert "Run Docker browser tools Chromium/WebKit smoke" in src
        assert "tests/test_browser_tools_smoke.py -m browser" in src
        assert "-e OUROBOROS_EXPECT_BROWSER_ENGINES=chromium,webkit" in src

    def test_marker_guard_uses_pipefail(self):
        src = _read(self._CI_PATH)
        guard_idx = src.find("Guard non-empty browser marker lanes")
        assert guard_idx != -1, "marker guard step not found"
        guard_block = src[guard_idx:guard_idx + 700]
        assert "set -euo pipefail" in guard_block

    def test_ci_uses_env_context_for_condition(self):
        """No `if:` expression in ci.yml (step-level OR job-level) may
        reference `secrets.*`.

        GitHub Actions rejects `secrets.*` in `if:` with
        `Unrecognized named-value: 'secrets'`. Always use `env.*` instead
        (see the non-secret job-level gate test above). The parser used here
        catches both step-level and job-level `if:` blocks deliberately —
        the rejection applies at every level, so a job-level violation
        would also break the workflow.
        """
        src = _read(self._CI_PATH)
        offending = [
            block for block in self._iter_step_if_blocks(src)
            if "secrets." in block
        ]
        assert not offending, (
            "secrets.* must not appear in any step-level if-condition "
            "(derive a non-secret job-level env gate and reference env.* instead). "
            f"Offenders: {offending}"
        )

    def test_ci_import_gates_on_full_secret_set(self):
        """The import step uses the non-secret gate derived from the full
        required signing set."""
        src = _read(self._CI_PATH)
        import_idx = src.find("Import Apple signing certificate")
        assert import_idx != -1, (
            "Apple signing-certificate Import step not found in ci.yml — "
            "the macOS signing path is missing"
        )
        region = src[import_idx:import_idx + 800]
        assert "env.HAS_APPLE_SIGNING == 'true'" in region

    def test_ci_cleanup_keychain_step_present(self):
        """A `Cleanup keychain` step must run with `if: always() &&
        matrix.os == 'macos-latest' && env.HAS_APPLE_SIGNING == 'true'`
        so signing material never persists across runs even when the build
        itself fails, and the bash-only `security` invocation never fires
        on Linux/Windows shards."""
        src = _read(self._CI_PATH)
        # Match the actual STEP definition (`- name: Cleanup keychain`), not
        # any prose mentioning the step elsewhere in the workflow file (e.g.
        # an explanatory comment in the Import step that references the later
        # Cleanup step would match a bare substring search). The `- name:`
        # anchor pins the assertion to the real step header.
        cleanup_anchor = "- name: Cleanup keychain"
        assert cleanup_anchor in src, (
            "ci.yml must include a `- name: Cleanup keychain` step that "
            "deletes the temporary signing keychain after every macOS build"
        )
        cleanup_idx = src.find(cleanup_anchor)
        cleanup_region = src[cleanup_idx:cleanup_idx + 500]
        assert "always()" in cleanup_region, (
            "Cleanup keychain must run with `if: always()` so it fires on "
            "build failures too"
        )
        assert "matrix.os == 'macos-latest'" in cleanup_region, (
            "Cleanup keychain must gate on matrix.os == 'macos-latest' so "
            "the bash-only `security delete-keychain` invocation does not "
            "fire on Linux/Windows shards"
        )
        assert "env.HAS_APPLE_SIGNING == 'true'" in cleanup_region, (
            "Cleanup keychain must gate on env.HAS_APPLE_SIGNING so "
            "it does not try to delete a keychain that was never created"
        )

    def test_build_sh_signing_identity_env_override(self):
        """build.sh must allow the signing identity to be overridden via env
        AND auto-detect from the keychain when env is unset/empty.

        The previous hardcoded `Developer ID Application: <Maintainer>
        (<TeamID>)` default broke any fork whose imported cert had a
        different CN (`codesign: no identity found`). The current contract:
        a non-empty `SIGN_IDENTITY` env wins; otherwise auto-detect via
        `security find-identity -v -p codesigning`.
        """
        src = _read("build.sh")
        # The empty-env auto-detect block must check `${SIGN_IDENTITY:-}`
        # explicitly (not `$SIGN_IDENTITY` alone, which would be unbound
        # under `set -u`).
        assert re.search(
            r'\[\s*-z\s*"\$\{SIGN_IDENTITY:-\}"\s*\]',
            src,
        ), (
            "build.sh must guard the auto-detect block with "
            "`[ -z \"${SIGN_IDENTITY:-}\" ]` so the env var wins when set "
            "and auto-detect runs only when env is unset/empty"
        )
        assert "security find-identity" in src and "-p codesigning" in src, (
            "build.sh must call `security find-identity -v -p codesigning` "
            "to auto-detect the signing identity from the keychain when "
            "SIGN_IDENTITY is not set externally"
        )
        # The hardcoded maintainer-specific default must be GONE (it caused
        # `codesign: no identity found` on forks; replaced by auto-detect).
        # We pin a substring that any future re-introduction would trip on.
        assert 'SIGN_IDENTITY="${SIGN_IDENTITY:-Developer ID Application:' not in src, (
            "build.sh must NOT carry a hardcoded maintainer-specific "
            "Developer ID Application default — it breaks forks whose "
            "imported cert has a different CN. Use auto-detect via "
            "`security find-identity` instead."
        )

    def test_build_sh_notarization_optional(self):
        """build.sh must include an optional notarization block guarded on
        APPLE_ID + APPLE_TEAM_ID + APPLE_APP_SPECIFIC_PASSWORD, calling
        `xcrun notarytool submit` followed by `xcrun stapler staple`."""
        src = _read("build.sh")
        assert "xcrun notarytool submit" in src, (
            "build.sh must call `xcrun notarytool submit` to upload the "
            "DMG for Apple notarization when credentials are configured"
        )
        assert "xcrun stapler staple" in src, (
            "build.sh must call `xcrun stapler staple` after a successful "
            "notarytool submission so the ticket is attached to the DMG"
        )
        # The notarization block must be guarded on the three notarytool
        # credential env vars, otherwise builds without an Apple ID hard-fail.
        for var in ("APPLE_ID", "APPLE_TEAM_ID", "APPLE_APP_SPECIFIC_PASSWORD"):
            assert var in src, (
                f"build.sh notarization block must reference {var} so it "
                f"is gated on the full credential set"
            )

    def test_build_sh_stapler_failure_is_soft(self):
        """`xcrun stapler staple` must be wrapped in an `if/then/else`
        (or paired with `||`) so a transient stapler failure becomes a
        warning instead of aborting the build under `set -e`.

        Apple's stapler service can fail intermittently after a successful
        `notarytool submit` (CDN propagation lag, transient 5xx). A
        signed-and-notarized-but-unstapled DMG is still functional —
        Gatekeeper fetches the ticket online on first launch — so a
        stapler hiccup must not delete the macOS artifact from the
        release.
        """
        src = _read("build.sh")
        # Find the stapler invocation and check it is inside an `if` head
        # (i.e. `if xcrun stapler staple ...; then`) OR followed by `||`.
        # The simplest robust check: locate the line, then verify either
        # (a) it begins with `if ` after stripping leading whitespace, or
        # (b) it ends with ` || ...` style continuation.
        # Only inspect actual code lines — strip both whole-line bash comments
        # (`# …`) and inline trailing comments (`code # …`) before testing.
        stapler_lines = []
        for raw in src.splitlines():
            code = raw.split("#", 1)[0]
            if "xcrun stapler staple" in code:
                stapler_lines.append(code)
        assert stapler_lines, (
            "build.sh must call `xcrun stapler staple` (notarization step)"
        )
        for line in stapler_lines:
            stripped = line.strip()
            wrapped_in_if = stripped.startswith("if ") and stripped.endswith("; then")
            soft_or = "||" in stripped
            assert wrapped_in_if or soft_or, (
                "build.sh `xcrun stapler staple` invocation must be guarded "
                "(`if xcrun stapler staple ...; then ... else WARN ... fi` "
                "or `xcrun stapler staple ... || echo WARN`) so a transient "
                "stapler failure does not abort the build under `set -e` and "
                f"silently drop the macOS DMG. Offending line: {stripped!r}"
            )


# ===========================================================================
# Packaging asset completeness (merged from former test_packaging_assets.py)
# and CI release workflow checks (merged from former test_release_workflow.py).
# All three originally exercised the same /repo packaging surface; the merge
# keeps them under one module so packaging contracts evolve in one place.
# ===========================================================================


# ----- shared helpers for packaging asset tests -----

_REPO_PATH = REPO_ROOT
_BUNDLE_FILES_PRESENT = (_REPO_PATH / "Ouroboros.spec").exists() and (_REPO_PATH / "launcher.py").exists()
_PACKAGING_SKIP_REASON = "Bundle-only files (Ouroboros.spec, launcher.py) not present in repo"


def _packaging_launcher_has_bootstrap() -> bool:
    launcher = _REPO_PATH / "launcher.py"
    bootstrap = _REPO_PATH / "ouroboros" / "launcher_bootstrap.py"
    if not launcher.exists() or not bootstrap.exists():
        return False
    launcher_src = launcher.read_text(encoding="utf-8")
    bootstrap_src = bootstrap.read_text(encoding="utf-8")
    return (
        "from ouroboros.launcher_bootstrap import" in launcher_src
        and 'BUNDLE_REPO_NAME = "repo.bundle"' in bootstrap_src
        and 'BUNDLE_MANIFEST_NAME = "repo_bundle_manifest.json"' in bootstrap_src
        and "ensure_managed_repo" in bootstrap_src
    )


_LAUNCHER_HAS_BOOTSTRAP = _packaging_launcher_has_bootstrap()


def _pkg_read(rel: str) -> str:
    return (_REPO_PATH / rel).read_text(encoding="utf-8")


# ----- spec/launcher/bundle invariants -----


@pytest.mark.skipif(not _BUNDLE_FILES_PRESENT, reason=_PACKAGING_SKIP_REASON)
def test_spec_bundles_assets_and_icon():
    source = _pkg_read("Ouroboros.spec")
    gitignore = _pkg_read(".gitignore")
    assert "('repo.bundle', '.')" in source
    assert "('repo_bundle_manifest.json', '.')" in source
    assert "('assets', 'assets')" in source
    assert "icon='assets/icon.icns'" in source
    # (RWS v2) The execd payload is a CI-verified artifact downloaded into the
    # bundle at build time, never a checked-in binary.
    assert "/assets/execd/" in gitignore


@pytest.mark.skipif(not _BUNDLE_FILES_PRESENT, reason=_PACKAGING_SKIP_REASON)
def test_spec_bundles_ouroboros_package_for_packaged_cli_bridge():
    source = _pkg_read("Ouroboros.spec")
    assert "('ouroboros', 'ouroboros')" in source
    assert "('python-standalone', 'python-standalone')" in source
    assert "('ripgrep-standalone', 'ripgrep-standalone')" in source


@pytest.mark.skipif(
    not _LAUNCHER_HAS_BOOTSTRAP,
    reason="launcher.py does not import launcher_bootstrap (may be a newer version without bootstrap bridge)",
)
def test_launcher_does_not_exclude_assets_on_bootstrap():
    launcher_source = _pkg_read("launcher.py")
    bootstrap_source = _pkg_read("ouroboros/launcher_bootstrap.py")
    assert '"python-standalone", "assets"' not in launcher_source
    assert "from ouroboros.launcher_bootstrap import" in launcher_source
    assert 'BUNDLE_REPO_NAME = "repo.bundle"' in bootstrap_source
    assert 'BUNDLE_MANIFEST_NAME = "repo_bundle_manifest.json"' in bootstrap_source
    assert "ensure_managed_repo(" in bootstrap_source


@pytest.mark.skipif(not _BUNDLE_FILES_PRESENT, reason=_PACKAGING_SKIP_REASON)
def test_spec_retains_cross_platform_packaging_hooks():
    source = _pkg_read("Ouroboros.spec")
    assert "assets/icon.ico" in source
    assert "collect_all as _collect_all" in source
    assert "scripts/pyi_rth_pythonnet.py" in source
    assert "pythonnet" in source
    assert "clr_loader" in source


@pytest.mark.skipif(not _BUNDLE_FILES_PRESENT, reason=_PACKAGING_SKIP_REASON)
def test_launcher_retains_cross_platform_runtime_hooks():
    launcher_source = _pkg_read("launcher.py")
    assert "embedded_python_candidates" in launcher_source
    assert "_prepare_windows_webview_runtime" in launcher_source
    assert "git_install_hint()" in launcher_source
    assert "create_kill_on_close_job" in launcher_source
    assert "kill_process_on_port(port)" in launcher_source
    assert "force_kill_pid(child.pid)" in launcher_source


@pytest.mark.skipif(not _BUNDLE_FILES_PRESENT, reason=_PACKAGING_SKIP_REASON)
def test_launcher_preserves_macos_git_setup_path():
    launcher_source = _pkg_read("launcher.py")
    assert 'subprocess.Popen(["xcode-select", "--install"])' in launcher_source
    assert "Install Git (Xcode CLI Tools)" in launcher_source
    assert "Installing... A system dialog may appear." in launcher_source
    assert '["lsof", "-ti", f"tcp:{port}"]' in launcher_source


def test_cross_platform_build_scripts_are_present():
    assert (_REPO_PATH / "build_linux.sh").exists()
    assert (_REPO_PATH / "build_windows.ps1").exists()
    assert (_REPO_PATH / "scripts" / "download_python_standalone.ps1").exists()
    assert (_REPO_PATH / "scripts" / "pyi_rth_pythonnet.py").exists()


def test_release_builds_embed_the_exact_claudexor_runtime_seed():
    for name in ("build.sh", "build_linux.sh", "build_windows.ps1"):
        source = _pkg_read(name)
        fetch_pos = source.find("scripts/fetch_claudexor_runtime.py")
        pyinstaller_pos = _find_pyinstaller_cmd_pos(source)
        assert fetch_pos != -1, f"{name} must fetch the reviewed Claudexor runtime seed"
        assert pyinstaller_pos != -1
        assert fetch_pos < pyinstaller_pos, f"{name} must fetch Claudexor before PyInstaller"

    spec = _pkg_read("Ouroboros.spec")
    assert "('claudexor-runtime', 'claudexor-runtime')" in spec
    assert "/claudexor-runtime/" in _pkg_read(".gitignore")


def test_execd_build_scripts_and_dependency_lock_are_present():
    assert (_REPO_PATH / "scripts" / "assemble_execd_stage.py").exists()
    assert (_REPO_PATH / "scripts" / "build_execd_bundle.py").exists()
    assert (_REPO_PATH / "scripts" / "smoke_execd_stage.py").exists()
    assert (_REPO_PATH / "scripts" / "execd_dependency_lock.json").exists()


def test_execd_stage_smoke_asserts_the_attested_release_identity():
    """The smoke must ASSERT the attested identity, not merely hand it in.

    A stage that starts but misreports which release and which artifact bytes it
    is would pass a naive smoke and then fail admission against Home's manifest.
    This test used to check only that the values were PASSED to `ExecdService`,
    while `docs/ARCHITECTURE.md` said the smoke "asserts the ATTESTED identity
    (continuity host id, release id, artifact SHA-256)" — three facts, none of
    which the smoke compared to anything. So the assertion is on the comparison.
    """

    source = _pkg_read("scripts/smoke_execd_stage.py")
    assert 'host_id = initialize_continuity_host_id(base / "state")' in source
    for fact, expected in (
        ("host_id", "host_id"),
        ("release_id", "release_id"),
        ("artifact_sha256", "artifact_sha256"),
    ):
        assert f'handshake["{fact}"] != {expected}' in source, (
            f"the stage smoke does not assert the handshake's {fact}"
        )
        assert f'prepared["{fact}"] != {expected}' in source, (
            f"the stage smoke does not assert a prepared object's {fact}"
        )
    # And it must refuse to grade an artifact from outside it: run under a host
    # Python, the stage's `lib/` is imported into the host runtime and everything
    # passes while proving nothing about the interpreter that reaches the target.
    assert "must run under the staged interpreter" in source


def test_ci_release_publishes_only_named_execd_assets():
    workflow = _ci_workflow()

    release_job = workflow.split("\n  release:\n", 1)[1]
    assert "pattern: ouroboros-*" in release_job
    files = release_job.split("files: |", 1)[1].split(
        "generate_release_notes:", 1
    )[0]
    assert "release-artifacts/Ouroboros-*" in files
    assert "release-artifacts/ouroboros-execd-*-linux-gnu-x86_64.tar.gz" in files
    assert "release-artifacts/ouroboros-execd-*-linux-gnu-aarch64.tar.gz" in files
    assert "release-artifacts/ouroboros-execd-*-manifest.json" in files
    # A wildcard would publish whatever happened to land in the directory.
    assert "release-artifacts/*" not in files


def test_ci_execd_jobs_run_for_target_pr_manual_or_tag_builds():
    workflow = _ci_workflow()

    stage_header = workflow.split("\n  execd-stage:\n", 1)[1].split(
        "\n    strategy:", 1
    )[0]
    bundle_header = workflow.split("\n  execd-bundle:\n", 1)[1].split(
        "\n    needs:", 1
    )[0]
    for header in (stage_header, bundle_header):
        assert "github.event_name == 'pull_request'" in header
        assert "github.base_ref == 'ouroboros'" in header
        assert "github.event_name == 'workflow_dispatch'" in header
        assert "startsWith(github.ref, 'refs/tags/v')" in header


def test_ci_ssh_smoke_covers_target_and_stable_pushes():
    workflow = _ci_workflow()
    smoke_header = workflow.split("\n  ssh-executor-smoke:\n", 1)[1].split(
        "\n    runs-on:", 1
    )[0]

    assert "refs/heads/ouroboros')" in smoke_header
    assert "refs/heads/ouroboros-stable')" in smoke_header
    assert "startsWith(github.ref, 'refs/tags/v')" in smoke_header


def test_ci_execd_release_gating_requires_the_verified_bundle():
    workflow = _ci_workflow()

    build_needs = workflow.split("\n  build:\n", 1)[1].split("needs:", 1)[1].splitlines()[0]
    release_needs = workflow.split("\n  release:\n", 1)[1].split("needs:", 1)[1].splitlines()[0]
    # A packaged app must never ship an unverified execd payload, and a release
    # must never ship without the real Docker/OpenSSH lane having run.
    assert "execd-bundle" in build_needs
    assert "execd-bundle" in release_needs
    assert "ssh-executor-smoke" in release_needs
    # `needs:` alone bought NOTHING here for a year: `build` declared the
    # dependency and then never downloaded the artifact, so `Ouroboros.spec`
    # packed an `assets/` tree with no execd in it and every platform archive
    # shipped without the remote executor. Declaring a dependency is not
    # consuming it — require the download AND the refusal that makes an empty
    # payload fatal instead of silent.
    build_job = workflow.split("\n  build:\n", 1)[1].split("\n  release:\n", 1)[0]
    assert "name: ouroboros-execd-linux" in build_job, (
        "the build job needs execd-bundle but never downloads its artifact"
    )
    assert "path: assets/execd" in build_job
    assert "Refuse to package a release with no remote executor" in build_job


# ── CI reachability: a step that cannot run is not a gate ────────────────────
#
# Two halves of ONE defect were found by an external review, and both are the
# same shape as the vacuous guards the Guard Proof Rule is about: a step that is
# structurally unable to execute looks identical, from the outside, to a step
# that passes. `quick-test` downloaded `ouroboros-execd-linux` with no `needs:`
# on the job that produces it, so on a push to `ouroboros` (where `execd-bundle`
# does not run at all) and on a pull request (where it races a ~40-minute build)
# the step failed FIRST — taking the Pages reproducibility check, the
# `ruff --select F` gate, both pytest lanes and the transport import guard down
# with it. Four gates executing nowhere, on the branch they exist to protect.
#
# So this is enforced mechanically rather than by reading: the trigger conditions
# are evaluated, and every artifact a job downloads by name must be produced by a
# job that BOTH runs on the same trigger and is in the downloader's `needs`
# closure. The evaluator refuses an `if:` expression it does not fully
# understand, so a future exotic condition fails loudly instead of being read as
# "always runs".

_CI_TRIGGERS: dict[str, dict[str, str]] = {
    "push:main": {
        "event_name": "push", "ref": "refs/heads/main", "base_ref": "",
    },
    "push:ouroboros": {
        "event_name": "push", "ref": "refs/heads/ouroboros", "base_ref": "",
    },
    "push:ouroboros-stable": {
        "event_name": "push", "ref": "refs/heads/ouroboros-stable", "base_ref": "",
    },
    "pull_request:ouroboros": {
        "event_name": "pull_request", "ref": "refs/pull/1/merge",
        "base_ref": "ouroboros",
    },
    "tag:v": {
        "event_name": "push", "ref": "refs/tags/v9.9.9", "base_ref": "",
    },
    "workflow_dispatch": {
        "event_name": "workflow_dispatch", "ref": "refs/heads/ouroboros",
        "base_ref": "",
    },
}


def _ci_job_runs(condition, trigger: dict[str, str]) -> bool:
    """Evaluate a job-level `if:` for one trigger, or refuse to guess."""

    if condition is None:
        return True
    expression = " ".join(str(condition).split())
    for name, value in (
        ("github.event_name", trigger["event_name"]),
        ("github.ref", trigger["ref"]),
        ("github.base_ref", trigger["base_ref"]),
    ):
        expression = expression.replace(name, repr(value))
    expression = re.sub(
        r"startsWith\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)",
        r"(\1).startswith(\2)",
        expression,
    )
    expression = expression.replace("||", " or ").replace("&&", " and ")
    # Everything left must be quoted text, parentheses, whitespace, the two
    # comparison operators, the boolean words, and `.startswith`. Anything else
    # is a construct this evaluator has not been taught.
    residue = re.sub(r"'[^']*'", "", expression)
    residue = residue.replace(".startswith", "")
    for token in ("==", "!=", " or ", " and ", " not ", "(", ")"):
        residue = residue.replace(token, " ")
    assert not residue.strip(), (
        f"ci.yml job condition {condition!r} contains {residue.strip()!r}, which "
        "this reachability gate cannot evaluate — teach it the construct rather "
        "than letting an unreadable condition count as 'always runs'"
    )
    return bool(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307


def _ci_artifact_names(job: dict, template: str) -> set[str]:
    """Expand `${{ matrix.key }}` in an artifact name over the job's matrix.

    `execd-stage` uploads `execd-stage-${{ matrix.architecture }}`; without the
    expansion the producer of `execd-stage-x86_64` looks like nobody.
    """

    if "${{" not in template:
        return {template}
    matrix = ((job.get("strategy") or {}).get("matrix") or {})
    rows = list(matrix.get("include") or [])
    for key, values in matrix.items():
        if key == "include" or not isinstance(values, list):
            continue
        rows.extend({key: value} for value in values)
    names = set()
    for row in rows:
        expanded = template
        for key, value in row.items():
            expanded = re.sub(
                r"\$\{\{\s*matrix\." + re.escape(str(key)) + r"\s*\}\}",
                str(value),
                expanded,
            )
        if "${{" not in expanded:
            names.add(expanded)
    return names


def _ci_needs_closure(jobs: dict, job: str) -> set[str]:
    declared = jobs[job].get("needs") or []
    if isinstance(declared, str):
        declared = [declared]
    closure: set[str] = set()
    for parent in declared:
        closure.add(parent)
        closure |= _ci_needs_closure(jobs, parent)
    return closure


def test_ci_every_downloaded_artifact_is_reachable_on_every_trigger():
    """A `download-artifact` step must be able to find its artifact."""

    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(_ci_workflow())
    jobs = workflow["jobs"]

    producers: dict[str, set[str]] = {}
    for name, job in jobs.items():
        for step in job.get("steps") or []:
            uses = str(step.get("uses") or "")
            with_block = step.get("with") or {}
            if "actions/upload-artifact" in uses and with_block.get("name"):
                for artifact in _ci_artifact_names(job, str(with_block["name"])):
                    producers.setdefault(artifact, set()).add(name)

    consumers: list[tuple[str, str]] = []
    for name, job in jobs.items():
        for step in job.get("steps") or []:
            uses = str(step.get("uses") or "")
            with_block = step.get("with") or {}
            if "actions/download-artifact" in uses and with_block.get("name"):
                for artifact in _ci_artifact_names(job, str(with_block["name"])):
                    consumers.append((name, artifact))
    assert consumers, "no named artifact download found — this gate has gone stale"

    unreachable: list[str] = []
    for job_name, artifact in consumers:
        closure = _ci_needs_closure(jobs, job_name)
        upstream = producers.get(artifact, set())
        if not upstream:
            unreachable.append(f"{job_name} downloads {artifact}, which nothing uploads")
            continue
        ordered = upstream & closure
        if not ordered:
            unreachable.append(
                f"{job_name} downloads {artifact} without any of {sorted(upstream)} "
                "in its needs closure, so the download races the build"
            )
            continue
        for label, trigger in _CI_TRIGGERS.items():
            if not _ci_job_runs(jobs[job_name].get("if"), trigger):
                continue
            if not any(_ci_job_runs(jobs[producer].get("if"), trigger) for producer in ordered):
                unreachable.append(
                    f"{job_name} runs on {label} and downloads {artifact}, which "
                    f"none of {sorted(ordered)} produces on that trigger"
                )
    assert not unreachable, "\n".join(unreachable)


def test_ci_marker_lane_canaries_really_carry_their_marker():
    """The marker guards pin canary FILENAMES; the markers must agree.

    Every `! grep -q "no tests collected"` in that job was doubly unreachable —
    an empty lane exits 5 through `pipefail` and kills the step before grep, and
    a non-empty lane never prints that text under `-q`. The replacement pins a
    known member of each lane, which is a restated fact, so it gets the liveness
    check a restated fact needs: the file must really be selected by the marker.
    """

    workflow = _ci_workflow()
    guard = workflow.split("Guard non-empty browser marker lanes", 1)[1].split(
        "\n      - name: Guard non-empty serial marker lane", 1
    )[0]
    lanes = re.findall(r"--collect-only -m (\w+) -q \| tee (\S+)", guard)
    anchors = dict(re.findall(r'grep -q "(tests/\S+?\.py)" (\S+)', guard))
    assert len(lanes) == 4, f"expected four browser-family lanes, found {lanes}"
    assert "no tests collected" not in guard, (
        "the vacuous `no tests collected` check is back: pytest never prints it "
        "under -q, and an empty lane dies at exit 5 before grep runs"
    )
    by_file = {path: marker for marker, path in lanes}
    for anchor, path in anchors.items():
        marker = by_file[path]
        # `pytestmark = pytest.mark.<name>` and a bare decorator both spell the
        # marker the same way, so one pattern covers module- and test-level use.
        declared = set(re.findall(r"pytest\.mark\.(\w+)", _pkg_read(anchor)))
        assert marker in declared, (
            f"ci.yml pins {anchor} as the {marker} lane canary, but that file "
            f"declares {sorted(declared)} and not {marker}"
        )


def test_build_sh_supports_unsigned_macos_release():
    build_source = _pkg_read("build.sh")
    assert 'OUROBOROS_SIGN' in build_source
    assert 'Skipping signing' in build_source
    assert 'Unsigned DMG:' in build_source


# ----- CI release workflow checks (from test_release_workflow.py) -----


def _ci_workflow() -> str:
    return (_REPO_PATH / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_ci_release_preflight_validates_tag_matches_version():
    workflow = _ci_workflow()

    assert "release-preflight:" in workflow
    assert "Validate tag matches VERSION" in workflow
    assert 'expected_tag = f"v{version}"' in workflow
    assert 'tag != expected_tag' in workflow


def test_ci_branch_filters_include_packaging_assets():
    workflow = _ci_workflow()

    assert "- 'packaging/**'" in workflow
    assert "- 'devtools/**'" in workflow


def test_ci_release_prerelease_flag_uses_preflight_output():
    workflow = _ci_workflow()

    assert "needs.release-preflight.outputs.is_prerelease" in workflow
    assert "prerelease: ${{ needs.release-preflight.outputs.is_prerelease == 'true' }}" in workflow
    assert "re.search(r'(?:rc|alpha|beta|a|b)\\.?\\d+$'" in workflow
    assert "fh.write(f\"is_prerelease={'true' if is_prerelease else 'false'}\\n\")" in workflow


def test_ci_build_job_exports_release_tag_and_fetches_full_history():
    workflow = _ci_workflow()

    assert "OUROBOROS_RELEASE_TAG: ${{ github.ref_name }}" in workflow
    assert "OUROBOROS_MANAGED_SOURCE_BRANCH: ouroboros" in workflow
    assert "fetch-depth: 0" in workflow


def test_ci_release_smokes_the_exact_embedded_claudexor_archive_in_all_assets():
    workflow = _ci_workflow()
    # DMG, Linux tarball, Linux AppImage, and Windows ZIP.
    assert workflow.count("fetch_claudexor_runtime.py --verify-only") == 4
    assert workflow.count("scripts/claudexor_platform_smoke.py") == 4
    assert workflow.count("--managed-runtime --lane fixture") == 4
    assert "embedded_claudexor_runtime" in workflow

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
import os
import pathlib
import re
import shutil
import subprocess
import sys

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


def _python_standalone_extras(src: str, interpreter: str) -> tuple[set, list]:
    """Extras of the locked exports a build script installs into ``interpreter``."""
    exported = {
        match.group(2): set(re.findall(r"--extra (\w+)", match.group(1)))
        for match in re.finditer(r'^ *uv export (.*)--output-file "?(\$\w+)"?$', src, re.M)
    }
    lock_header = _read("requirements-runtime.lock").splitlines()[1]
    exported["requirements-runtime.lock"] = set(re.findall(r"--extra (\w+)", lock_header))
    installs = list(re.finditer(rf"^ *uv pip install --python {re.escape(interpreter)} .*-r (\S+)", src, re.M))
    return set().union(*(exported[match.group(1).strip('"')] for match in installs)), installs


# ---------------------------------------------------------------------------
# build.sh  (macOS)
# ---------------------------------------------------------------------------

class TestBuildSh:
    """build.sh must bundle Chromium headless shell without bundled macOS WebKit."""

    def test_macos_source_entry_is_built_before_headers_are_pruned(self):
        src = _read("build.sh")
        compile_pos = src.index("xcrun clang")
        install_pos = src.index('cp build/macos/ouroboros-shell')
        assert src.index("rm -rf build dist") < compile_pos < _find_pyinstaller_cmd_pos(src) < install_pos
        assert '-mmacosx-version-min=12.0' in src
        assert '-Wl,-rpath,@executable_path/../Frameworks/python-standalone/lib' in src
        assert 'python-standalone/include/python${SHELL_PY_VERSION}' in src
        assert '"python-standalone/lib/$SHELL_PY_LIB"' in src
        stub = _read("packaging/macos/ouroboros_shell.c")
        assert "Py_InitializeFromConfig" in stub and "Py_RunMain" in stub
        assert "Py_SetPythonHome" not in stub and "execv" not in stub

    @pytest.mark.parametrize("matching", [True, False])
    def test_entry_install_requires_its_own_runtime_library_not_build_pythons(self, tmp_path, matching):
        """Run the real install block against the static-python layout (no artifact is executed)."""
        # The PATH's bash, as the notarization tests below: a bare "bash" on Windows starts System32's WSL launcher.
        bash = shutil.which("bash")
        if not bash:
            pytest.skip("build.sh's install block needs bash")
        src = _read("build.sh")
        start = src.index('cp build/macos/ouroboros-shell')
        end = src.index('echo "--- Installing packaged CLI wrappers', start)
        app = tmp_path / "New app.app"
        frameworks = app / "Contents/Frameworks"
        (app / "Contents/MacOS").mkdir(parents=True)
        (frameworks / "python-standalone/lib").mkdir(parents=True)
        library = frameworks / ("python-standalone/lib/libpython3.10.dylib" if matching else "libpython3.10.dylib")
        library.write_bytes(b"library fixture, never executed")
        compiled = tmp_path / "build/macos/ouroboros-shell"
        compiled.parent.mkdir(parents=True)
        compiled.write_bytes(b"entry fixture, never executed")
        result = subprocess.run([bash, "-ec", src[start:end]], cwd=tmp_path, capture_output=True, text=True,
                                env={**os.environ, "APP_PATH": str(app), "SHELL_PY_LIB": "libpython3.10.dylib"})
        assert (result.returncode == 0) is matching, result.stderr
        assert (app / "Contents/MacOS/ouroboros-shell").read_bytes() == compiled.read_bytes()

    def test_source_entry_identity_is_restored_after_deep_sign_then_outer_seal_verified(self):
        src = _read("build.sh").replace("\\\n", " ")
        commands = [line.strip() for line in src.splitlines() if line.strip().startswith("codesign -s")]
        deep = next(i for i, line in enumerate(commands) if "--deep" in line)
        assert '--identifier com.ouroboros.agent' in commands[deep + 1]
        assert 'Contents/MacOS/ouroboros-shell' in commands[deep + 1]
        assert '--deep' not in commands[deep + 2] and '"$APP_PATH"' in commands[deep + 2]
        assert '"$SHELL_SIGN_ID" != "com.ouroboros.agent"' in src
        assert src.index('"$SHELL_SIGN_ID" !=') < src.index('codesign --verify --strict --deep')

    def test_playwright_install_chromium_present(self):
        src = _read("build.sh")
        assert "playwright install --only-shell chromium" in src, (
            "build.sh must call 'playwright install --only-shell chromium' on macOS"
        )
        assert 'rglob(".local-browsers")' in src
        assert 'glob("webkit-*")' in src
        assert "shutil.rmtree" in src
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


    def test_source_handoff_interpreter_receives_the_desktop_extra(self):
        """The source launcher needs desktop modules outside PyInstaller's archive."""
        src = _read("build.sh").replace("\\\n", " ")
        extras, installs = _python_standalone_extras(src, "python-standalone/bin/python3")
        assert "desktop" in extras, f"python-standalone receives only extras {sorted(extras)}"
        assert "build" not in extras, "build tooling must stay out of the packaged interpreter"
        symlink_pos = src.find("Normalizing python-standalone symlinks")
        assert all(match.start() < symlink_pos < _find_pyinstaller_cmd_pos(src) for match in installs)


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
        # Explicit empty values disable the controls; a scrubbed child otherwise
        # receives the test suite's safe cache defaults again.
        env["PYTHONDONTWRITEBYTECODE"] = ""
        env["PYTHONPYCACHEPREFIX"] = ""
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

    def test_source_handoff_interpreter_receives_the_desktop_extra(self):
        """Headless detection after the source handoff imports webview from python-standalone."""
        assert "from webview.guilib import initialize" in _read("launcher.py")
        src = _read("build_linux.sh").replace("\\\n", " ")
        extras, installs = _python_standalone_extras(src, '"$PORTABLE_PYTHON"')
        assert "desktop" in extras, f"python-standalone receives only extras {sorted(extras)}"
        assert "build" not in extras, "build tooling must stay out of the packaged interpreter"
        assert all(match.start() < _find_pyinstaller_cmd_pos(src) for match in installs)


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
            "Betterleaks runtime installation",
            "Agent dependency installation",
            "Desktop dependency export",
            "Desktop dependency installation",
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

    def test_source_handoff_interpreter_receives_the_desktop_extra(self):
        """The source launcher prepares pythonnet from python-standalone before importing webview."""
        runtime = _read("ouroboros/launcher_windows_runtime.py")
        assert "from clr_loader import get_netfx" in runtime and "from pythonnet import set_runtime" in runtime
        src = _read("build_windows.ps1").replace("`\n", " ")
        extras, installs = _python_standalone_extras(src, '"python-standalone\\python.exe"')
        assert "desktop" in extras, f"python-standalone receives only extras {sorted(extras)}"
        assert "build" not in extras, "build tooling must stay out of the packaged interpreter"
        assert all(match.start() < _find_pyinstaller_cmd_pos(src) for match in installs)


def test_python_standalone_pin_is_the_fixed_sqlite_release():
    """Desktop packages take CPython 3.10.22 from python-build-standalone 20261009.

    That release is the last one that still ships Python 3.10, and its SQLite
    is 3.53.1 (the WAL-reset fix). The two download scripts are the pin.
    """
    shell = _read("scripts/download_python_standalone.sh")
    windows = _read("scripts/download_python_standalone.ps1")
    assert 'RELEASE="20261009"' in shell
    assert 'PY_VERSION="3.10.22"' in shell
    assert '$Release = "20261009"' in windows
    assert '$PyVersion = "3.10.22"' in windows
    assert "20260211" not in shell and "20260211" not in windows
    assert "3.10.19" not in shell and "3.10.19" not in windows
    for platform in (
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
    ):
        assert platform in shell
    assert "x86_64-pc-windows-msvc" in windows
    pinned = dict(re.findall(r'PLATFORM="([^"]+)"\n\s+SHA256="([0-9a-f]{64})"', shell))
    assert set(pinned) == {
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
    }
    windows_sha = re.search(r'\$Sha256 = "([0-9a-f]{64})"', windows)
    assert windows_sha is not None
    docker = _read("Dockerfile")
    docker_sha = dict(re.findall(r'PLATFORM="([^"]+)"; SHA256="([0-9a-f]{64})"', docker))
    assert docker_sha == {
        "x86_64-unknown-linux-gnu": pinned["x86_64-unknown-linux-gnu"],
        "aarch64-unknown-linux-gnu": pinned["aarch64-unknown-linux-gnu"],
    }
    assert windows_sha.group(1) not in docker_sha.values()


# ---------------------------------------------------------------------------
# Dockerfile  (Docker / web runtime)
# ---------------------------------------------------------------------------

class TestDockerignore:
    """The root .dockerignore owns the build context of Dockerfile's COPY . .

    Both halves matter: private local state must stay out of image layers,
    and the paths CI needs inside the image (Git history, tests, sources)
    must stay in."""

    def _patterns(self):
        lines = _read(".dockerignore").splitlines()
        return {ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")}

    def test_private_state_is_excluded(self):
        patterns = self._patterns()
        required = {
            "**/.env", "**/.env.*", "**/*.key", "**/*.pem",
            ".venv/", "venv/", "env/", "/data/",
            "/.review-drive/", "/.claudexor/", "/.adversarial-review/",
        }
        missing = sorted(required - patterns)
        assert not missing, f".dockerignore must exclude private local state: {missing}"

    def test_ci_needed_paths_stay_in_context(self):
        patterns = self._patterns()
        for kept in (".git", "tests", "ouroboros", "web", "prompts", "docs",
                     "supervisor", "pyproject.toml", "uv.lock", "server.py"):
            for spelling in (kept, kept + "/", "/" + kept, "/" + kept + "/", "**/" + kept):
                assert spelling not in patterns, (
                    f".dockerignore must not exclude {kept}: CI runs pytest inside the image"
                )
        assert "*" not in patterns and "**" not in patterns


class TestDockerfile:
    """One self-contained Dockerfile ships Chromium/WebKit for the locked Playwright.

    The pins below are the contract the tag-only ``docker-ui-smoke`` and
    ``docker-portable-test`` lanes rely on: browsers land in the shared
    ``/ms-playwright`` the runtime honors as-is, they are downloaded ABOVE the
    lock copy (every release rewrites ``pyproject.toml``/``uv.lock``), and CI
    exercises the image's own browsers instead of re-downloading them.
    """

    def test_build_is_self_contained(self):
        """``docker build -t ouroboros-web .`` must not depend on a locally built base tag.

        The interpreter is the pinned python-build-standalone build, checked in
        the image build for Python 3.10.22 and SQLite >= 3.51.3.
        """
        src = _read("Dockerfile")
        assert "FROM python:3.10-slim" not in src
        assert "FROM debian:trixie-slim" in src
        assert "ghcr.io/astral-sh/uv:" in src
        assert "FROM ${" not in src, "the image must not start from an unpublished local tag"
        assert "sha256sum -c -" in src
        assert "assert sqlite>=(3, 51, 3)" in src
        assert "--python /opt/python-standalone/bin/python3" in src

    def test_runtime_stage_keeps_the_python_base_packages(self):
        """debian:trixie-slim lacks what python:3.10-slim installed: without CA
        certificates git and OpenSSL cannot verify HTTPS (Update fetches), and
        tzdata/netbase back zoneinfo and service lookups."""
        src = _read("Dockerfile")
        runtime = src[src.index("FROM debian:trixie-slim AS runtime"):]
        install = re.search(r"apt-get install -y --no-install-recommends ([^\\\n]+)", runtime)
        assert install is not None
        assert {"ca-certificates", "git", "netbase", "tzdata"} <= set(install.group(1).split())
        assert "LANG=C.UTF-8" in runtime

    def test_playwright_pin_matches_lock(self):
        src = _read("Dockerfile")
        match = re.search(r'\[\[package\]\]\nname = "playwright"\nversion = "([^"]+)"', _read("uv.lock"))
        assert match is not None, "uv.lock must lock playwright"
        assert f"PLAYWRIGHT_VERSION={match.group(1)}" in src, (
            "Dockerfile must pin the browser installer to the locked Playwright version"
        )

    def test_shared_browser_path_is_the_image_environment(self):
        src = _read("Dockerfile")
        assert "PLAYWRIGHT_BROWSERS_PATH=/ms-playwright" in src
        assert "PLAYWRIGHT_BROWSERS_PATH=0" not in src, (
            "browsers must not live in the package tree: the dependency layer is rebuilt on every release"
        )

    def test_browsers_install_before_the_lock_copy(self):
        src = _read("Dockerfile")
        deps_pos = src.find("playwright install-deps chromium webkit")
        install_pos = src.find("playwright install chromium webkit")
        lock_pos = src.find("COPY pyproject.toml uv.lock")
        assert deps_pos != -1 and install_pos != -1 and lock_pos != -1
        assert deps_pos < install_pos < lock_pos, (
            "install-deps, then the browser download, then the lock copy — otherwise a release "
            f"bump re-downloads the browsers (deps {deps_pos}, install {install_pos}, lock {lock_pos})"
        )

    def test_no_runtime_uv_project_environment(self):
        """A baked ``UV_PROJECT_ENVIRONMENT`` makes an agent's ``uv sync`` in a task
        workspace rewrite Ouroboros's own venv."""
        assert "UV_PROJECT_ENVIRONMENT" not in _read("Dockerfile")

    def test_ci_docker_lanes_use_the_shipped_browsers(self):
        """The release-gate container commands must not redirect Playwright away from the image."""
        ci = _read(".github/workflows/ci.yml")
        assert "docker build -t ouroboros-web:test ." in ci
        assert "PLAYWRIGHT_BROWSERS_PATH=0" not in ci, (
            "a PLAYWRIGHT_BROWSERS_PATH=0 prefix hides /ms-playwright and makes the lane download at test time"
        )
        assert "playwright install --only-shell" not in ci, "the portable lane measures the shipped image"

    def test_ci_docker_builds_never_bake_the_checkout_token(self):
        """`COPY . .` keeps `.git`, so a job building the image from its checkout must not
        persist GITHUB_TOKEN as `.git/config`'s extraheader into a layer (#1263)."""
        import yaml

        jobs = yaml.safe_load(_read(".github/workflows/ci.yml"))["jobs"]
        building = sorted(name for name, job in jobs.items()
                          if any("docker build" in str(step.get("run", "")) for step in job.get("steps", [])))
        assert {"docker-portable-test", "docker-ui-smoke"} <= set(building), building
        for name in building:
            checkouts = [step for step in jobs[name]["steps"]
                         if str(step.get("uses", "")).startswith("actions/checkout@")]
            assert checkouts, name
            assert all((step.get("with") or {}).get("persist-credentials") is False for step in checkouts), name


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
    """The regenerated lock keeps LF in a Windows checkout, so the clean-tree bundle gate sees no false edit."""
    attributes = subprocess.run(["git", "check-attr", "eol", "--", "requirements-runtime.lock"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout
    assert attributes.strip() == "requirements-runtime.lock: eol: lf", attributes


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
        assert "+ status" in src


# ---------------------------------------------------------------------------
# .github/workflows/ci.yml + build.sh — macOS code signing & notarization
# ---------------------------------------------------------------------------

class TestMacOSSigning:
    """The CI build job and build.sh together implement optional macOS code
    signing and notarization. The contracts here prevent
    regression of the GitHub Actions `secrets.*`-in-step-`if:` pitfall, the
    build-script env override, the keychain cleanup guard, and the
    notarization contract: off without Apple credentials, otherwise the app
    and the DMG each get a validated stapled ticket or the build stops.

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
        # Host browser-tool coverage lives in the shared lane both triggers call;
        # the Docker lane keeps its own container spelling here.
        shared = _read(".github/workflows/ui-browser.yml")
        assert "OUROBOROS_EXPECT_BROWSER_ENGINES: chromium,webkit" in shared
        assert "tests/test_browser_tools_smoke.py -m browser" in shared
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

    @staticmethod
    def _shell_function(src: str, name: str) -> str:
        """Return one top-level shell function of build.sh, verbatim."""
        start = src.index(f"\n{name}() {{\n") + 1
        return src[start:src.index("\n}\n", start) + 3]

    def _run_notarization(self, tmp_path, *, status: str, staple_rc: int = 0):
        """Execute build.sh's own notarize/staple functions against a fake xcrun."""
        bash = shutil.which("bash")
        if not bash:
            pytest.skip("build.sh functions need bash")
        src = _read("build.sh")
        fake = tmp_path / "bin"
        fake.mkdir()
        (fake / "xcrun").write_text(
            "#!/bin/sh\n"
            'echo "$*" >> "$XCRUN_LOG"\n'
            'case "$1 $2" in\n'
            '  "notarytool submit") printf \'{"id": "sub-1", "status": "%s"}\' "$FAKE_STATUS" ;;\n'
            '  "stapler staple"|"stapler validate") exit "$FAKE_STAPLE_RC" ;;\n'
            "esac\n",
            encoding="utf-8", newline="\n",  # bash rejects CRLF on the Windows shards
        )
        (fake / "xcrun").chmod(0o755)
        script = (
            "set -e\nsleep() { :; }\n"
            + self._shell_function(src, "notarize")
            + self._shell_function(src, "staple")
            + 'notarize dist/app.zip\nstaple dist/Ouroboros.app\necho DONE\n'
        )
        log = tmp_path / "xcrun.log"
        env = dict(os.environ, PATH=f"{fake}{os.pathsep}{os.environ['PATH']}", XCRUN_LOG=str(log),
                   FAKE_STATUS=status, FAKE_STAPLE_RC=str(staple_rc), APPLE_ID="dev@example.test",
                   APPLE_TEAM_ID="TEAM", APPLE_APP_SPECIFIC_PASSWORD="secret")
        result = subprocess.run([bash, "--noprofile", "--norc", "-c", script], env=env,
                                capture_output=True, text=True, check=False)
        return result, log.read_text(encoding="utf-8") if log.exists() else ""

    def test_build_sh_notarization_accepts_only_apples_accepted_status(self, tmp_path):
        """notarytool can exit 0 on "Invalid": build.sh reads the JSON status,
        prints the developer log, staples and validates, and stops otherwise."""
        accepted, calls = self._run_notarization(tmp_path, status="Accepted")
        assert accepted.returncode == 0 and "DONE" in accepted.stdout, accepted.stderr
        assert "notarytool log sub-1" in calls
        assert "stapler staple dist/Ouroboros.app" in calls
        assert "stapler validate dist/Ouroboros.app" in calls

        (tmp_path / "invalid").mkdir()
        invalid, calls = self._run_notarization(tmp_path / "invalid", status="Invalid")
        assert invalid.returncode != 0 and "DONE" not in invalid.stdout
        assert "status 'Invalid'" in invalid.stderr
        assert "notarytool log sub-1" in calls and "stapler" not in calls

    def test_build_sh_staple_failure_stops_the_build(self, tmp_path):
        """A notarized build that cannot staple a validated ticket is not shipped:
        v7.7.0 published a DMG whose app Gatekeeper rejected."""
        failed, calls = self._run_notarization(tmp_path, status="Accepted", staple_rc=1)
        assert failed.returncode != 0 and "DONE" not in failed.stdout
        assert "no validated notarization ticket" in failed.stderr
        assert calls.count("stapler staple dist/Ouroboros.app") == 3

    def test_build_sh_notarizes_the_app_before_the_dmg_and_only_with_credentials(self):
        src = _read("build.sh")
        gate = src[src.index("NOTARIZE=0"):src.index("notarize() {")]
        for var in ("SIGN_MODE", "APPLE_ID", "APPLE_TEAM_ID", "APPLE_APP_SPECIFIC_PASSWORD"):
            assert var in gate, f"notarization must stay off without {var}"
        order = [
            src.index("=== Signing Ouroboros.app ==="),
            src.index('ditto -c -k --keepParent "$APP_PATH" "$APP_ZIP"'),
            src.index('notarize "$APP_ZIP"'),
            src.index('staple "$APP_PATH"'),
            src.index('ditto "$APP_PATH" "$DMG_STAGE_DIR/Ouroboros.app"'),
            src.index("hdiutil create"),
            src.index('codesign -s "$SIGN_IDENTITY" --timestamp "$DMG_PATH"'),
            src.index('notarize "$DMG_PATH"'),
            src.index('staple "$DMG_PATH"'),
        ]
        assert order == sorted(order), "the stapled app must be what the DMG contains"

    def test_build_sh_drops_debug_symbols_and_pyobjc_tests_before_pyinstaller(self, tmp_path):
        """Executes build.sh's own payload prune: debug-symbol bundles and pyobjc's
        test suite go, runtime extensions stay, and it happens before PyInstaller."""
        src = _read("build.sh")
        marker = "--- Removing stale WebKit, debug-symbol and test payloads"
        assert src.index(marker) < src.index("python3 -m PyInstaller")
        body = src[src.index("python3 - <<'PY'\n", src.index(marker)) + len("python3 - <<'PY'\n"):]
        body = body[:body.index("\nPY\n")]
        site = tmp_path / "python-standalone" / "lib" / "python3.10" / "site-packages"
        dsym = site / "objc" / "_objc.cpython-310-darwin.so.dSYM" / "Contents" / "Resources" / "DWARF"
        dsym.mkdir(parents=True)
        (dsym / "_objc.cpython-310-darwin.so").write_bytes(b"dwarf")
        (site / "objc" / "_objc.cpython-310-darwin.so").write_bytes(b"runtime")
        (site / "PyObjCTest" / "NULL.cpython-310-darwin.so.dSYM").mkdir(parents=True)
        (site / "PyObjCTest" / "test_null.py").write_text("", encoding="utf-8")
        (site / "AppKit").mkdir()
        (site / "AppKit" / "__init__.py").write_text("", encoding="utf-8")

        result = subprocess.run([sys.executable, "-c", body], cwd=tmp_path,
                                capture_output=True, text=True, check=False)

        assert result.returncode == 0, result.stderr
        assert not (site / "PyObjCTest").exists()
        assert not (site / "objc" / "_objc.cpython-310-darwin.so.dSYM").exists()
        assert (site / "objc" / "_objc.cpython-310-darwin.so").read_bytes() == b"runtime"
        assert (site / "AppKit" / "__init__.py").exists()
        assert "Removed 1 test package(s) and 2 debug-symbol bundle(s)" in result.stdout

    def test_macos_smoke_proves_gatekeeper_on_the_copied_out_app(self):
        """codesign passed v7.7.0; the release proof must ask Gatekeeper about the
        app a user drags out (the decision itself is executed in
        tests/test_release_proof_macos.py)."""
        src = _read(self._CI_PATH)
        smoke = src[src.index("- name: Smoke final macOS DMG"):src.index("- name: Smoke final Linux archive")]
        assert 'xcrun stapler validate "$DMG"' in smoke
        assert 'xcrun stapler validate "$SBOM_ROOT/Ouroboros.app"' in smoke
        assert 'spctl --assess --type exec -vv "$SBOM_ROOT/Ouroboros.app"' in smoke
        assert "--context context:primary-signature" in smoke
        assert smoke.count("grep -q 'source=Notarized Developer ID'") == 2
        assert smoke.index('ditto "$MOUNT" "$SBOM_ROOT"') < smoke.index("spctl --assess")


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
    assert "('repo.bundle', '.')" in source
    assert "('repo_bundle_manifest.json', '.')" in source
    assert "('assets', 'assets')" in source
    assert "icon='assets/icon.icns'" in source


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
    assert "find_embedded_python" in launcher_source
    assert "embedded_python_candidates" in _pkg_read("ouroboros/launcher_bootstrap.py")
    assert "_prepare_windows_webview_runtime" in launcher_source
    assert "git_install_hint()" in launcher_source
    assert "create_kill_on_close_job" in launcher_source
    assert "kill_runtime_orphans(DATA_DIR" in launcher_source
    reaper = _pkg_read("ouroboros/launcher_server_reaper.py")
    assert "kill_process_on_port(int(companion_port))" in reaper
    assert "force_kill_pid(child.pid)" in reaper


@pytest.mark.skipif(not _BUNDLE_FILES_PRESENT, reason=_PACKAGING_SKIP_REASON)
def test_launcher_preserves_macos_git_setup_path():
    launcher_source = _pkg_read("launcher.py")
    assert '_hidden_popen(["xcode-select", "--install"], bootstrap=True)' in launcher_source  # owned, no window flag off Windows
    assert "Install Git (Xcode CLI Tools)" in launcher_source
    assert "Installing... A system dialog may appear." in launcher_source


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


def test_release_builds_stage_one_pinned_betterleaks_runtime_before_bundle_and_pyinstaller():
    module_command = "-m ouroboros.betterleaks_runtime install"
    for name in ("build.sh", "build_linux.sh", "build_windows.ps1"):
        source = _pkg_read(name)
        install_pos = source.find(module_command)
        bundle_pos = source.find("scripts/build_repo_bundle.py")
        pyinstaller_pos = _find_pyinstaller_cmd_pos(source)
        assert install_pos != -1, f"{name} must invoke the package-local installer"
        assert "--build-output betterleaks-standalone" in source
        assert bundle_pos != -1 and install_pos < bundle_pos, (
            f"{name} must stage Betterleaks before the clean-tree repo bundle gate"
        )
        assert pyinstaller_pos != -1 and install_pos < pyinstaller_pos

    windows = _pkg_read("build_windows.ps1")
    assert 'Invoke-NativeChecked "Betterleaks runtime installation"' in windows
    macos = _pkg_read("build.sh")
    assert macos.find(module_command) < macos.find("=== Signing Ouroboros.app ===")

    spec = _pkg_read("Ouroboros.spec")
    assert "('betterleaks-standalone', 'betterleaks-standalone')" in spec
    assert "ouroboros.betterleaks_runtime install --build-output" in spec
    assert "/betterleaks-standalone/" in _pkg_read(".gitignore")


def test_build_sh_supports_unsigned_macos_release():
    build_source = _pkg_read("build.sh")
    assert 'OUROBOROS_SIGN' in build_source
    assert 'Skipping signing' in build_source
    assert 'Unsigned DMG:' in build_source


# ----- CI release workflow checks (from test_release_workflow.py) -----


def _ui_browser_jobs() -> dict:
    """The shared browser lane ci.yml and ui-browser-push.yml both call."""
    import yaml

    return yaml.safe_load(
        (_REPO_PATH / ".github/workflows/ui-browser.yml").read_text(encoding="utf-8"))["jobs"]


def _ci_workflow() -> str:
    return (_REPO_PATH / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_ci_release_preflight_validates_tag_matches_version():
    workflow = _ci_workflow()

    assert "release-preflight:" in workflow
    assert "Validate tag matches VERSION" in workflow
    assert 'expected_tag = f"v{version}"' in workflow
    assert 'tag != expected_tag' in workflow


_RELEASE_REQUIRED = ("build", "windows-proof", "release-preflight", "lint", "full-test", "system-e2e-mock",
                     "ui-smoke", "integration-test", "marker-guards", "docker-ui-smoke", "docker-portable-test",
                     "skill-smoke", "test-verdict")


def test_ci_collects_independent_failures_without_relaxing_release_gates():
    import yaml

    workflow = _ci_workflow()
    jobs = yaml.safe_load(workflow)["jobs"]
    assert "tag_valid: ${{ steps.release_meta.outputs.tag_valid }}" in workflow
    # Builds start once the tag is valid, beside the tests, instead of waiting for them.
    assert "needs" not in jobs["release-preflight"]
    assert jobs["build"]["needs"] == ["release-preflight"]
    assert "needs.release-preflight.outputs.tag_valid == 'true'" in jobs["build"]["if"]
    # ...and the release job, not the preflight, requires every lane: nothing is waived by tag.
    release = jobs["release"]["if"]
    for name in _RELEASE_REQUIRED:
        assert f"needs.{name}.result == 'success'" in release, name
        assert name in jobs["release"]["needs"], name
    assert "refs/tags/v7" not in workflow
    assert "if: startsWith(github.ref, 'refs/tags/v')" in workflow.split("  android-build:", 1)[1].split("  release-preflight:", 1)[0]
    assert "continue-on-error: true" in workflow.split("  vendor-package-smoke:", 1)[1].split("  release:", 1)[0]


def test_ci_setup_aware_failure_collection_guards_each_independent_lane():
    import yaml
    jobs = yaml.safe_load(_ci_workflow())["jobs"]
    for job_name in ("quick-test", "full-test-runs", "marker-guards", "docker-ui-smoke", "system-e2e-mock", "android-test"):
        steps = jobs[job_name]["steps"]
        assert any("!cancelled()" in str(step.get("if", "")) for step in steps if "run" in step), job_name
    ui = {step.get("name"): step for step in _ui_browser_jobs()["ui-shard"]["steps"]}
    host = ui["Run complete host UI lane with collection and availability guards"]
    assert "github.event_name" not in host["if"]
    assert "--require-ui-browser" in host["run"]
    assert "workflow_dispatch" in ui["Run browser tools Chromium/WebKit smoke"]["if"]
    assert "refs/tags/v" in ui["Run browser tools Chromium/WebKit smoke"]["if"]


def test_ci_release_requires_success_and_never_a_skipped_or_red_lane():
    """The release `if` asks every required job for success, so a skipped lane (no result at all)
    or a red one stops publication on every tag alike; FLAKY and ACCEPTED tests are green lanes."""
    import yaml

    release = yaml.safe_load(_ci_workflow())["jobs"]["release"]["if"]
    assert "github.ref == 'refs/tags/" not in release and "github.repository ==" not in release
    assert "!= 'skipped'" not in release
    assert release.count("== 'success'") == len(_RELEASE_REQUIRED)


def test_ci_failure_collection_guards_every_independent_step_and_rerun_uploads():
    import yaml

    jobs = yaml.safe_load(_ci_workflow())["jobs"]
    expected = {
        "lint": ["Verify generated Pages output", "Lint (deterministic F-rule gate — catches the NameError-under-except class)", "Lint browser modules (ESLint no-undef — CI-only second layer of the acorn gate)"],
        "quick-test": ["Run tests (parallel — excludes the costly marker lanes AND the serial real-process suites)", "Run tests (serial — real subprocess/port/global-state suites that flake under -n)", "Run tests (size-ratchet lane — blocking here, warning-only locally)", "Guard extracted transport imports stay out of core"],
        "full-test-runs": ["Run browser-module tests (node --test — mirrored by the hermetic commit gate)", "Run tests (parallel — excludes the costly marker lanes AND the serial real-process suites)", "Run tests (serial — real subprocess/port/global-state suites that flake under -n)", "Run tests (size-ratchet lane — blocking here, warning-only locally)", "Guard extracted transport imports stay out of core"],
        "marker-guards": ["Guard non-empty browser marker lanes", "Guard non-empty serial marker lane", "Guard non-empty skill_smoke marker lane", "Guard non-empty size_ratchet marker lane"],
        "docker-ui-smoke": ["Install UI smoke browser binaries", "Run Docker UI smoke", "Run Docker browser tools Chromium/WebKit smoke"],
        "system-e2e-mock": ["Run the keyless system E2E scenario lane (real isolated servers)", "Run the cancellation E-suite mock lane"],
        "android-test": ["Run Android source and release contract tests", "Compile and verify explicitly test-signed Android host"],
    }
    expected["ui-shard"] = ["Install UI smoke Chromium and WebKit",
                           "Run complete host UI lane with collection and availability guards",
                           "Run browser tools Chromium/WebKit smoke"]
    jobs = {**jobs, "ui-shard": _ui_browser_jobs()["ui-shard"]}
    for job, names in expected.items():
        steps = {step.get("name"): step for step in jobs[job]["steps"]}
        for name in names:
            assert "!cancelled()" in str(steps[name].get("if", "")), (job, name)
    assert "overwrite: true" in _ci_workflow().split("name: Upload build artifact", 1)[1].split("name: Vendor", 1)[0]
    assert "overwrite: true" in _ci_workflow().split("name: Upload Android release artifacts", 1)[1].split("  release-preflight:", 1)[0]

def test_ci_step_outcome_references_resolve_to_prior_step_ids():
    import yaml

    jobs = yaml.safe_load(_ci_workflow())["jobs"]
    pattern = re.compile(r"steps\.([A-Za-z_][A-Za-z0-9_]*)\.outcome")
    for job_name, job in jobs.items():
        declared = set()
        for step in job.get("steps", []):
            for step_id in pattern.findall(str(step.get("if", ""))):
                assert step_id in declared, (job_name, step_id)
            if step.get("id"):
                declared.add(step["id"])




def test_release_tags_and_manual_browser_runs_keep_unfiltered_entrypoints():
    import yaml

    # Event matrix: test_platform_ci_events; exact triggers: test_ci_diagnostic_workflows.
    push = yaml.safe_load((_REPO_PATH / ".github/workflows/ui-browser-push.yml").read_text(encoding="utf-8"))
    trigger = push.get("on", push.get(True))
    assert list(trigger) == ["workflow_dispatch"], "the UI-only entry is manual and adds no paid lane"
    assert trigger["workflow_dispatch"]["inputs"]["diagnostic"]["options"] == ["full", "viewport", "inflight"]
    assert push["jobs"]["ui-smoke"]["uses"] == "./.github/workflows/ui-browser.yml"

    shared = yaml.safe_load((_REPO_PATH / ".github/workflows/ui-browser.yml").read_text(encoding="utf-8"))
    assert list(shared.get("on", shared.get(True))) == ["workflow_call"]
    assert "secrets" not in str(shared["jobs"])


def test_ci_release_prerelease_flag_uses_preflight_output():
    workflow = _ci_workflow()

    assert "needs.release-preflight.outputs.is_prerelease" in workflow
    assert "prerelease: ${{ needs.release-preflight.outputs.is_prerelease == 'true' }}" in workflow
    assert "re.search(r'(?:rc|alpha|beta|a|b)\\.?\\d+$'" in workflow
    assert "fh.write(f\"is_prerelease={'true' if is_prerelease else 'false'}\\n\")" in workflow


def test_ci_build_job_exports_release_tag_and_fetches_full_history():
    workflow = _ci_workflow()

    assert "OUROBOROS_RELEASE_TAG: ${{ github.ref_name }}" in workflow
    # The managed source branch is RESOLVED per tag (the live line → `ouroboros`;
    # a pre-release on another branch → the one remote branch containing HEAD),
    # never hardcoded: a hardcoded `ouroboros` refused every side-branch pre-release
    # at the bundle step (v7.0.0-rc.1, run 33678261200).
    assert "OUROBOROS_MANAGED_SOURCE_BRANCH: ouroboros" not in workflow
    assert "Resolve the managed source branch of this tag" in workflow
    assert "git merge-base --is-ancestor HEAD origin/ouroboros" in workflow
    assert 'echo "OUROBOROS_MANAGED_SOURCE_BRANCH=$branch" >> "$GITHUB_ENV"' in workflow
    # macOS runners execute `shell: bash` steps under bash 3.2: bash-4 builtins are
    # a red build there and nowhere else (v7.0.0-rc.4, run 33730918681).
    assert "mapfile" not in workflow and "readarray" not in workflow
    assert "fetch-depth: 0" in workflow


def test_ci_job_level_env_avoids_step_only_runner_context():
    workflow = _ci_workflow()
    job_env_blocks = re.findall(r"(?m)^    env:\n((?:      .*\n)*)", workflow)
    assert job_env_blocks
    assert not [block for block in job_env_blocks if re.search(r"\brunner(?:\.|\[)", block)]


def test_ci_release_smokes_the_exact_embedded_claudexor_archive_in_all_assets():
    workflow = _ci_workflow()
    # DMG, Linux tarball, Linux AppImage, and Windows ZIP.
    assert workflow.count("fetch_claudexor_runtime.py --verify-only") == 4
    assert workflow.count("scripts/claudexor_platform_smoke.py") == 4
    assert workflow.count("--managed-runtime --lane fixture") == 4
    assert "embedded_claudexor_runtime" in workflow


def test_ci_has_fork_safe_three_os_betterleaks_candidate_matrix():
    workflow = _ci_workflow()
    quick_start = workflow.index("  quick-test:")
    quick_job = workflow[quick_start : workflow.index("\n  # ─", quick_start)]
    start = workflow.index("  betterleaks-platform-smoke:")
    job = workflow[start : workflow.index("\n  # ─", start)]
    assert "github.event_name == 'workflow_dispatch'" in quick_job
    # Manual runs only: ci.yml has no pull_request trigger, so no dormant PR branch is kept.
    assert "if: github.event_name == 'workflow_dispatch'" in job
    assert "pull_request" not in job
    assert "os: [ubuntu-latest, windows-latest, macos-latest]" in job
    assert "uses: ./.github/actions/setup-python-env" in job
    assert "python -m ouroboros.betterleaks_runtime install" in job
    assert "python scripts/betterleaks_platform_smoke.py --managed-runtime" in job
    assert "python -m pytest tests/test_skill_publish_scanner_real.py -q" in job
    assert 'OUROBOROS_BETTERLEAKS_REQUIRE_REAL: "1"' in job
    job_env, steps = job.split("    steps:", 1)
    assert "runner.temp" not in job_env
    assert steps.count("OUROBOROS_DATA_DIR: ${{ runner.temp }}/ouroboros-betterleaks-data") == 3
    assert "secrets." not in job

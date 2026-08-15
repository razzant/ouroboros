from __future__ import annotations

import os
import pathlib
import subprocess

import pytest


REPO = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="AppImage packaging is POSIX-only"
)


def test_appdir_layout_wraps_the_existing_pyinstaller_payload(tmp_path: pathlib.Path):
    dist = tmp_path / "dist"
    payload = dist / "Ouroboros"
    payload.mkdir(parents=True)
    launcher = payload / "Ouroboros"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    internal = payload / "_internal"
    internal.mkdir()
    (internal / "repo.bundle").write_bytes(b"bundle")
    (internal / "VERSION").write_text("6.96.2\n", encoding="utf-8")
    embedded_python = internal / "python-standalone/bin/python3"
    embedded_python.parent.mkdir(parents=True)
    embedded_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    appdir = tmp_path / "Ouroboros.AppDir"
    env = os.environ.copy()
    env["OUROBOROS_DIST_DIR"] = str(dist)
    env["OUROBOROS_APPDIR"] = str(appdir)
    subprocess.run(
        ["bash", "scripts/build_appimage.sh", "--appdir-only"],
        cwd=REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (appdir / "AppRun").stat().st_mode & 0o111
    assert (appdir / "ouroboros.desktop").is_file()
    assert (appdir / "ouroboros.png").is_file()
    assert (appdir / "usr/lib/ouroboros/Ouroboros").is_file()
    assert (appdir / "usr/lib/ouroboros/_internal/repo.bundle").is_file()
    assert (
        appdir / "usr/lib/ouroboros/_internal/python-standalone/bin/python3"
    ).is_file()
    desktop = (appdir / "ouroboros.desktop").read_text(encoding="utf-8")
    assert "Exec=Ouroboros" in desktop
    assert "Icon=ouroboros" in desktop

    version = subprocess.run(
        [str(appdir / "AppRun"), "--version"],
        env={**os.environ, "APPDIR": str(appdir)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert version.stdout.strip() == "Ouroboros 6.96.2"


def test_apprun_exposes_cli_without_writing_to_the_mount(tmp_path: pathlib.Path):
    appdir = tmp_path / "AppDir"
    cli = appdir / "usr/lib/ouroboros/bin/ouroboros"
    cli.parent.mkdir(parents=True)
    cli.write_text('#!/bin/sh\nprintf "%s\\n" "$OUROBOROS_BUNDLE_DIR"\n', encoding="utf-8")
    cli.chmod(0o755)
    launcher = appdir / "usr/lib/ouroboros/Ouroboros"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    apprun = appdir / "AppRun"
    apprun.write_bytes((REPO / "packaging/appimage/AppRun").read_bytes())
    apprun.chmod(0o755)

    result = subprocess.run(
        [str(apprun), "--cli", "--help"],
        env={**os.environ, "APPDIR": str(appdir)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == str(appdir / "usr/lib/ouroboros/_internal")


@pytest.mark.parametrize(
    ("original_tmpdir", "payload_status"),
    [(None, 0), ("/caller/tmp", 37)],
)
def test_apprun_custodian_restores_environment_and_removes_private_runtime(
    tmp_path: pathlib.Path,
    original_tmpdir: str | None,
    payload_status: int,
):
    private_base = tmp_path / "ouroboros-appimage-runtime-test"
    appdir = private_base / "appimage_extracted_test"
    observed = tmp_path / "payload-environment.txt"
    launcher = appdir / "usr/lib/ouroboros/Ouroboros"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "#!/bin/sh\n"
        "if [ \"${TMPDIR+x}\" = x ]; then value=set:$TMPDIR; else value=unset; fi\n"
        "if env | grep -q '^OUROBOROS_APPIMAGE_'; then exit 9; fi\n"
        "printf '%s\\n' \"$value\" > \"$OBSERVED_PATH\"\n"
        "exit \"$TEST_PAYLOAD_STATUS\"\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    apprun = appdir / "AppRun"
    apprun.write_bytes((REPO / "packaging/appimage/AppRun").read_bytes())
    apprun.chmod(0o755)
    env = {
        **os.environ,
        "APPDIR": str(appdir),
        "TMPDIR": str(private_base),
        "OUROBOROS_APPIMAGE_RESTORE_TMPDIR": "1",
        "OUROBOROS_APPIMAGE_ORIGINAL_TMPDIR_SET": "1" if original_tmpdir is not None else "0",
        "OUROBOROS_APPIMAGE_ORIGINAL_TMPDIR": original_tmpdir or "",
        "OBSERVED_PATH": str(observed),
        "TEST_PAYLOAD_STATUS": str(payload_status),
    }

    result = subprocess.run(
        [str(apprun)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == payload_status
    assert observed.read_text(encoding="utf-8").strip() == (
        f"set:{original_tmpdir}" if original_tmpdir is not None else "unset"
    )
    assert not appdir.exists()
    assert not private_base.exists()


def test_apprun_custodian_refuses_unrelated_private_runtime(tmp_path: pathlib.Path):
    private_base = tmp_path / "ouroboros-appimage-runtime-test"
    private_base.mkdir()
    sentinel = private_base / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    appdir = tmp_path / "elsewhere" / "appimage_extracted_test"
    launcher = appdir / "usr/lib/ouroboros/Ouroboros"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    apprun = appdir / "AppRun"
    apprun.write_bytes((REPO / "packaging/appimage/AppRun").read_bytes())
    apprun.chmod(0o755)

    result = subprocess.run(
        [str(apprun)],
        env={
            **os.environ,
            "APPDIR": str(appdir),
            "TMPDIR": str(private_base),
            "OUROBOROS_APPIMAGE_RESTORE_TMPDIR": "1",
            "OUROBOROS_APPIMAGE_ORIGINAL_TMPDIR_SET": "0",
            "OUROBOROS_APPIMAGE_ORIGINAL_TMPDIR": "",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "outside its private runtime root" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert appdir.is_dir()


@pytest.mark.serial
def test_apprun_unmarked_desktop_path_keeps_exec_pid_and_appdir(tmp_path: pathlib.Path):
    appdir = tmp_path / "AppDir"
    pid_output = tmp_path / "launcher.pid"
    launcher = appdir / "usr/lib/ouroboros/Ouroboros"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$$\" > \"$PID_OUTPUT\"\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    apprun = appdir / "AppRun"
    apprun.write_bytes((REPO / "packaging/appimage/AppRun").read_bytes())
    apprun.chmod(0o755)

    process = subprocess.Popen(
        [str(apprun)],
        env={**os.environ, "APPDIR": str(appdir), "PID_OUTPUT": str(pid_output)},
    )
    process_pid = process.pid
    assert process.wait(timeout=10) == 0

    assert int(pid_output.read_text(encoding="utf-8")) == process_pid
    assert appdir.is_dir()


def test_appimage_builder_pins_tool_and_embedded_runtime():
    script = (REPO / "scripts/build_appimage.sh").read_text(encoding="utf-8")

    assert "RUNTIME_VERSION=20251108" in script
    assert "2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d" in script
    assert "00cbdfcf917cc6c0ff6d3347d59e0ca1f7f45a6df1a428a0d6d8a78664d87444" in script
    assert "releases/download/${RUNTIME_VERSION}/runtime-${TOOL_ARCH}" in script
    assert 'fetch_verified "$RUNTIME" "$RUNTIME_URL" "$RUNTIME_SHA256"' in script
    assert '"$TOOL" --runtime-file "$RUNTIME" "$APPDIR" "$OUTPUT"' in script

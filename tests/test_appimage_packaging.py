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
    (internal / "linux-gtk").mkdir()
    (internal / "linux-gtk/manifest.json").write_text("{}\n", encoding="utf-8")
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


def test_appdir_refuses_payload_without_native_closure(tmp_path):
    payload = tmp_path / 'Ouroboros'
    payload.mkdir()
    (payload / 'Ouroboros').write_text('#!/bin/sh\nexit 0\n')
    (payload / 'Ouroboros').chmod(0o755)
    result = subprocess.run(['bash', 'scripts/build_appimage.sh', '--appdir-only'],
                            cwd=REPO, env={**os.environ, 'OUROBOROS_DIST_DIR': str(tmp_path)},
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'GTK/WebKit closure is missing' in result.stderr


def test_webkit_relocation_is_bounded_and_preserves_elf_offsets():
    import runpy

    relocate = runpy.run_path(str(REPO / 'scripts/bundle_linux_gtk.py'))['relocate_webkit']
    original = (b'\x7fELF\0/usr/lib/x86_64-linux-gnu/webkit2gtk-4.1\0'
                b'/usr/share/webkitgtk-4.1\0/usr/bin/bwrap\0/usr/bin/xdg-dbus-proxy\0'
                b'/usr/libexec/webkit2gtk-4.1/WebKitWebProcess\0/usr/unrelated\0'
                b'/usr/lib/x86_64-linux-gnu/webkit2gtk-4.10\0'
                b'https://example.invalid/usr/bin/bwrap\0')
    changed, edits = relocate(original)
    assert len(changed) == len(original)
    assert b'././/lib/x86_64-linux-gnu/webkit2gtk-4.1\0' in changed
    assert b'/usr/unrelated\0' in changed
    assert b'/usr/lib/x86_64-linux-gnu/webkit2gtk-4.10\0' in changed
    assert b'https://example.invalid/usr/bin/bwrap\0' in changed
    assert b'././/libexec/webkit2gtk-4.1/WebKitWebProcess\0' in changed
    assert b'././/bin/xdg-dbus-proxy\0' in changed
    assert len(edits) == 5
    with pytest.raises(RuntimeError, match='sandbox helper paths changed'):
        relocate(original.replace(b'/usr/bin/bwrap', b'/unknown/bwrap'))
    with pytest.raises(RuntimeError, match='prefix changed'):
        relocate(b'\x7fELF\0/a/new/upstream/layout\0')


def test_desktop_bundle_environment_survives_exec_and_preserves_cli_cwd(tmp_path):
    appdir = tmp_path / 'AppDir'
    payload = appdir / 'usr/lib/ouroboros'
    root = payload / '_internal/linux-gtk'
    root.mkdir(parents=True)
    (root / 'manifest.json').write_text('{}')
    (root / 'desktop-env.sh').write_bytes((REPO / 'packaging/appimage/desktop-env.sh').read_bytes())
    apprun = appdir / 'AppRun'
    apprun.write_bytes((REPO / 'packaging/appimage/AppRun').read_bytes())
    apprun.chmod(0o755)
    successor = payload / 'successor'
    successor.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$PYWEBVIEW_GUI" "$LD_LIBRARY_PATH"\n')
    successor.chmod(0o755)
    launcher = payload / 'Ouroboros'
    launcher.write_text('#!/bin/sh\nexec "$OUROBOROS_BUNDLE_DIR/../successor"\n')
    import runpy
    runpy.run_path(str(REPO / 'scripts/bundle_linux_gtk.py'))['install_desktop_entry'](payload)
    (payload / 'Ouroboros.bin').chmod(0o755)
    launcher.chmod(0o755)
    env = {**os.environ, 'APPDIR': str(appdir), 'PYWEBVIEW_GUI': 'qt', 'LD_LIBRARY_PATH': '/caller/lib'}
    result = subprocess.run([str(apprun)], env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(root), 'gtk', f'{payload}/_internal:{root}/lib/x86_64-linux-gnu:/caller/lib']
    cli = payload / 'bin/ouroboros'
    cli.parent.mkdir()
    cli.write_bytes(successor.read_bytes())
    cli.chmod(0o755)
    result = subprocess.run([str(apprun), '--cli'], env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(tmp_path), 'qt', '/caller/lib']


def _typelib(dependencies=b'', libraries=b''):
    """A minimal format-4 typelib whose header fields 36/52 point at C strings."""
    import struct

    data = bytearray(b'GOBJ\nMETADATA\r\n\x1a' + bytes([4, 0]) + bytes(46))
    for field, value in ((36, dependencies), (52, libraries)):
        if value:
            struct.pack_into('<I', data, field, len(data))
            data += value + b'\0'
    return bytes(data)


def test_typelib_closure_refuses_dependencies_the_host_would_supply(tmp_path):
    import runpy

    bundle = runpy.run_path(str(REPO / 'scripts/bundle_linux_gtk.py'))
    requirements = bundle['typelib_requirements']
    assert requirements(_typelib(b'cairo-1.0|HarfBuzz-0.0|GObject-2.0', b'libpango-1.0.so.0')) == (
        ['cairo-1.0', 'HarfBuzz-0.0', 'GObject-2.0'], ['libpango-1.0.so.0'])
    assert requirements(_typelib()) == ([], [])
    with pytest.raises(RuntimeError, match='Unknown typelib format'):
        requirements(b'not a typelib' + bytes(64))
    (tmp_path / 'Pango-1.0.typelib').write_bytes(_typelib(b'HarfBuzz-0.0|GObject-2.0', b'libpango-1.0.so.0'))
    (tmp_path / 'GObject-2.0.typelib').write_bytes(_typelib(b'', b'libgobject-2.0.so.0'))
    with pytest.raises(RuntimeError, match=r"Pango-1\.0\.typelib requires typelibs outside the closure: \['HarfBuzz-0\.0'\]"):
        bundle['typelib_libraries'](tmp_path)
    (tmp_path / 'HarfBuzz-0.0.typelib').write_bytes(
        _typelib(b'GObject-2.0', b'libharfbuzz.so.0,libharfbuzz-gobject.so.0'))
    assert bundle['typelib_libraries'](tmp_path) == [
        ('GObject-2.0.typelib', 'libgobject-2.0.so.0'), ('HarfBuzz-0.0.typelib', 'libharfbuzz.so.0'),
        ('HarfBuzz-0.0.typelib', 'libharfbuzz-gobject.so.0'), ('Pango-1.0.typelib', 'libpango-1.0.so.0')]
    assert 'gir1.2-harfbuzz-0.0' in bundle['PACKAGES']


def test_gi_extension_takes_libpython_only_from_its_own_interpreter(tmp_path):
    import runpy

    linked = runpy.run_path(str(REPO / 'scripts/bundle_linux_gtk.py'))['linked_libraries']
    interpreter = tmp_path / 'python-standalone/lib'
    interpreter.mkdir(parents=True)
    ldd = ('\tlinux-vdso.so.1 (0x00007ffc)\n'
           '\tlibgirepository-1.0.so.1 => /lib/x86_64-linux-gnu/libgirepository-1.0.so.1 (0x00007f01)\n'
           '\tlibpython3.10.so.1.0 => {}\n'
           '\t/lib64/ld-linux-x86-64.so.2 (0x00007f02)\n')
    targets = ('not found', '/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 (0x00007f03)')
    for target in targets:
        for supplier in (interpreter, None):
            with pytest.raises(RuntimeError, match='its interpreter does not ship'):
                linked('_gi.so', ldd.format(target), supplier)
    (interpreter / 'libpython3.10.so.1.0').write_bytes(b'')
    for target in targets:
        # The interpreter's own copy satisfies it; a builder libpython is never shipped.
        assert linked('_gi.so', ldd.format(target), interpreter) == [
            ('libgirepository-1.0.so.1', '/lib/x86_64-linux-gnu/libgirepository-1.0.so.1')]
    with pytest.raises(RuntimeError, match='Unresolved dependency'):
        linked('_gi.so', '\tlibffi.so.8 => not found\n', interpreter)
    with pytest.raises(RuntimeError, match='Unresolved dependency'):
        linked('_gi.so', "/x/_gi.so: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found\n", interpreter)


def test_native_recipe_uses_only_the_signed_ordinary_archive():
    import re

    recipe = (REPO / 'packaging/appimage/native-qualification.Dockerfile').read_text()
    keyring = '[signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg]'
    assert re.findall(r"'(deb [^']*)'", recipe) == [
        f'deb {keyring} http://archive.ubuntu.com/ubuntu jammy main universe',
        f'deb {keyring} http://archive.ubuntu.com/ubuntu jammy-updates main universe',
        f'deb {keyring} http://security.ubuntu.com/ubuntu jammy-security main universe']
    for bypass in ('trusted=yes', 'allow-insecure', 'check-valid-until', 'AllowUnauthenticated', 'snapshot.ubuntu.com'):
        assert bypass not in recipe
    # apt-get source cannot fetch the image's superseded versions.
    assert recipe.index('dist-upgrade') < recipe.index('ca-certificates python3 git curl')
    builder = recipe.split('FROM common AS builder')[1].split('FROM common AS desktop')[0]
    release = (REPO / '.github/workflows/ci.yml').read_text().split('Install Linux GTK closure build inputs')[1]
    for install in (builder, release.split('- name:')[0]):
        assert 'libwebkit2gtk-4.1-dev gir1.2-webkit2-4.1' in install


def test_native_qualifier_records_webkit_sandbox_without_requiring_it():
    import ast
    import runpy

    script = REPO / 'scripts/qualify_linux_appimage.py'
    observe = runpy.run_path(str(script))['sandbox_observation']

    def process(pid, exe, namespace, **environment):
        return {'pid': pid, 'exe': exe, 'mount_namespace': namespace, 'environment': environment}

    helper = '/x/_internal/linux-gtk/lib/x86_64-linux-gnu/webkit2gtk-4.1/WebKitWebProcess'
    records = [process(1, '/x/python3', 'mnt:[1]'), process(2, helper, 'mnt:[1]'),
               process(3, '/x/_internal/linux-gtk/bin/bwrap', 'mnt:[2]', WEBKIT_FORCE_SANDBOX='1'),
               process(4, helper, 'mnt:[2]'),
               {'pid': 5, 'unreadable': 'denied', 'exe': '', 'maps': '', 'environment': {}}]
    assert observe(records, 'mnt:[1]') == {
        'launcher_mount_namespace': 'mnt:[1]',
        'web_processes': [{'pid': 2, 'mount_namespace': 'mnt:[1]', 'separate_mount_namespace': False},
                          {'pid': 4, 'mount_namespace': 'mnt:[2]', 'separate_mount_namespace': True}],
        'sandbox_helpers': ['bwrap'], 'webkit_sandbox_variables': ['WEBKIT_FORCE_SANDBOX']}
    asserts = [ast.unparse(node) for node in ast.walk(ast.parse(script.read_text())) if isinstance(node, ast.Assert)]
    assert asserts and not any('mount_namespace' in text or 'ns/mnt' in text for text in asserts)


def test_native_fixture_declares_the_gles_front_end_libepoxy_dlopens():
    import re
    import runpy

    recipe = (REPO / 'packaging/appimage/native-qualification.Dockerfile').read_text()
    desktop = recipe.split('FROM common AS desktop')[1].split('RUN useradd')[0]
    packages = set(re.findall(r'[\w.+-]+', desktop.split('--no-install-recommends')[1].split('&&')[0]))
    # Run 39636484833: "Couldn't open libGLESv2.so.2" and no Setup window.
    assert {'libgl1', 'libegl1', 'libgles2', 'libgbm1', 'libdrm2', 'libgl1-mesa-dri'} <= packages
    script = runpy.run_path(str(REPO / 'scripts/qualify_linux_appimage.py'))
    host = runpy.run_path(str(REPO / 'scripts/bundle_linux_gtk.py'))['HOST_LIBRARIES']
    assert 'libGLESv2.so.2' in script['HOST_GRAPHICS']
    # Everything but the dlopened GLES front-end is already a declared host exception.
    assert [name for name in script['HOST_GRAPHICS'] if not host.fullmatch(name)] == ['libGLESv2.so.2']


def test_native_qualifier_does_not_leak_the_test_pip_guard(tmp_path, monkeypatch):
    import json
    import runpy
    import types

    qualify = runpy.run_path(str(REPO / 'scripts/qualify_linux_appimage.py'))
    seen = {}

    class Aborted:
        """An AppImage that exits before any window, as the launcher did in run 39636484833."""
        returncode = -6

        def __init__(self, args, env, **kwargs):
            seen.update(env)

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setenv('PIP_REQUIRE_VIRTUALENV', 'true')
    monkeypatch.setitem(qualify['one_case'].__globals__, 'subprocess', types.SimpleNamespace(
        Popen=Aborted, CalledProcessError=subprocess.CalledProcessError, STDOUT=subprocess.STDOUT))
    # A dead process is reported at once, with its exit status, not after 180 s.
    with pytest.raises(RuntimeError, match='launcher pid: AppImage exited first with -6'):
        qualify['one_case'](tmp_path / 'x.AppImage', tmp_path / 'state', tmp_path / 'evidence')
    assert 'PIP_REQUIRE_VIRTUALENV' not in seen
    assert seen['OUROBOROS_DATA_DIR'] == str(tmp_path / 'state/data')
    assert json.loads((tmp_path / 'evidence/result.json').read_text()) == {
        'status': 'incomplete_or_failed', 'returncode': -6}


def test_native_qualifier_wait_reports_an_early_exit_not_a_timeout():
    import runpy

    wait_for = runpy.run_path(str(REPO / 'scripts/qualify_linux_appimage.py'))['wait_for']

    class Process:
        def __init__(self, returncode):
            self.returncode = returncode

        def poll(self):
            return self.returncode

    with pytest.raises(RuntimeError, match=r'^Setup: AppImage exited first with 134$'):
        wait_for(lambda: None, 'Setup', timeout=1, process=Process(134))
    # A live process still gets the ordinary deadline.
    with pytest.raises(RuntimeError, match=r'^Timed out: Setup$'):
        wait_for(lambda: None, 'Setup', timeout=0.3, process=Process(None))


@pytest.fixture
def renamed_linux_payload(tmp_path):
    """Real entries around a recording stand-in for the frozen ELF, never a GUI."""
    import runpy
    import sys

    payload = tmp_path / 'payload/Ouroboros'
    native = payload / '_internal/linux-gtk'
    native.mkdir(parents=True)
    (native / 'manifest.json').write_text('{}')
    (native / 'desktop-env.sh').write_bytes((REPO / 'packaging/appimage/desktop-env.sh').read_bytes())
    entry = payload / 'Ouroboros'
    entry.write_text(f'#!{sys.executable}\nimport json, os, pathlib, sys\n'
                     'pathlib.Path(os.environ["TEST_ENTRY_OUTPUT"]).write_text(json.dumps('
                     '{"argv":sys.argv,"bundle":os.environ.get("OUROBOROS_BUNDLE_DIR"),'
                     '"saved":os.environ.get("OUROBOROS_DESKTOP_ENV_SAVED")}))\n')
    entry.chmod(0o755)
    runpy.run_path(str(REPO / 'scripts/bundle_linux_gtk.py'))['install_desktop_entry'](payload)
    return payload


@pytest.mark.serial
def test_release_archive_digest_and_ci_extraction_keep_the_shared_entry(tmp_path, renamed_linux_payload, monkeypatch):
    import argparse
    import json
    import tarfile
    import yaml
    from scripts import release_proof

    payload = renamed_linux_payload
    dist, extracted = tmp_path / 'dist', tmp_path / 'extracted'
    dist.mkdir()
    extracted.mkdir()
    archive = dist / 'Ouroboros-7.7.1-linux-x86_64.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(payload, arcname='Ouroboros')
    digest = release_proof.sha256_file(archive)
    args = argparse.Namespace(directory=dist, expect_sha256=digest, github_output=None)
    release_proof.command_locate(args)
    workflow = yaml.safe_load((REPO / '.github/workflows/ci.yml').read_text())
    steps = workflow['jobs']['build']['steps']
    smoke = next(s['run'] for s in steps if s.get('id') == 'smoke_linux')
    extract = next(line.strip() for line in smoke.splitlines() if line.strip().startswith('tar -xzf '))
    extract = extract.replace('${{ steps.release_asset.outputs.path }}', str(archive))
    subprocess.run(['bash', '-eu', '-c', extract], env={**os.environ, 'SMOKE_ROOT': str(extracted)}, check=True)
    output = tmp_path / 'entry.json'
    monkeypatch.setenv('TEST_ENTRY_OUTPUT', str(output))
    entry = extracted / 'Ouroboros/Ouroboros'
    subprocess.run([str(entry), '--launch-intent', 'automatic'], check=True)
    observed = json.loads(output.read_text())
    assert observed['argv'] == [str(entry.with_name('Ouroboros.bin')), '--launch-intent', 'automatic']
    assert observed['bundle'] == str(entry.parent / '_internal') and observed['saved'] == '1'
    archive.write_bytes(archive.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='not the .* digest'):
        release_proof.command_locate(args)
    entry.with_name('Ouroboros.bin').unlink()
    output.unlink()
    refused = subprocess.run([str(entry)], capture_output=True)
    assert refused.returncode != 0 and not output.exists()


@pytest.mark.serial
def test_ci_runtime_root_uses_the_live_renamed_executable(tmp_path):
    import yaml

    steps = yaml.safe_load((REPO / '.github/workflows/ci.yml').read_text())['jobs']['build']['steps']
    smoke = next(s['run'] for s in steps if s.get('name') == 'Smoke final Linux AppImage')
    lines = [line.strip() for line in smoke.splitlines()]
    # Execute the exact production parameter expansions; no /proc, ldd or AppImage runs on this Mac.
    assignments = '\n'.join(next(line for line in lines if line.startswith(name + '='))
                            for name in ('APPIMAGE_RUNTIME_ROOT', 'APPIMAGE_PRIVATE_BASE'))
    extraction = tmp_path / 'private/appimage_extracted_fixture'
    binary = extraction / 'usr/lib/ouroboros/Ouroboros.bin'
    script = assignments + '\nprintf "%s\\n%s\\n" "$APPIMAGE_RUNTIME_ROOT" "$APPIMAGE_PRIVATE_BASE"'
    actual = subprocess.check_output(['bash', '-eu', '-c', script],
                                    env={**os.environ, 'LAUNCHER_EXE': str(binary)}, text=True).splitlines()
    assert actual == [str(extraction), str(extraction.parent)]
    wrong = subprocess.check_output(['bash', '-eu', '-c', script],
                                   env={**os.environ, 'LAUNCHER_EXE': str(binary.with_name('Ouroboros'))}, text=True).splitlines()
    assert wrong != actual  # the old basename must not masquerade as the renamed ELF
    assert 'ldd "$LAUNCHER_EXE" >"$LDD_LOG"' in lines
    assert 'kill -TERM "$LAUNCHER_PID"' in lines
    assert 'if [ "$APPIMAGE_RUNTIME_EXE" != "$APPIMAGE" ]; then' in lines


@pytest.mark.serial
def test_cli_and_signin_resolve_public_entry_and_refuse_only_internal_binary(tmp_path, renamed_linux_payload, monkeypatch):
    import json
    from types import SimpleNamespace
    from ouroboros import desktop_autostart, packaged_cli

    payload = renamed_linux_payload
    monkeypatch.setattr(packaged_cli, 'IS_MACOS', False)
    monkeypatch.setattr(packaged_cli, 'IS_WINDOWS', False)
    monkeypatch.delenv('APPIMAGE', raising=False)
    monkeypatch.delenv('APPDIR', raising=False)
    output = tmp_path / 'entry.json'
    monkeypatch.setenv('TEST_ENTRY_OUTPUT', str(output))
    original_popen, children = subprocess.Popen, []

    def capture(*args, **kwargs):
        proc = original_popen(*args, **kwargs)
        children.append(proc)
        return proc

    monkeypatch.setattr(packaged_cli.subprocess, 'Popen', capture)
    runtime = SimpleNamespace(bundle_root=payload / '_internal')
    packaged_cli._launch_desktop_app(runtime)
    assert len(children) == 1
    assert children[0].wait(timeout=10) == 0
    assert json.loads(output.read_text())['argv'] == [str(payload / 'Ouroboros.bin')]
    import launcher
    from ouroboros import launcher_bootstrap, platform_layer
    plain_python = payload / '_internal/python-standalone/bin/python3'
    plain_python.parent.mkdir(parents=True)
    plain_python.touch()
    monkeypatch.setattr(launcher, 'sys', SimpleNamespace(
        frozen=True, _MEIPASS=str(runtime.bundle_root), executable=str(payload / 'Ouroboros.bin')))
    monkeypatch.setattr(launcher_bootstrap, 'sys', SimpleNamespace(executable=str(payload / 'Ouroboros.bin')))
    assert launcher._bundle_dir() == runtime.bundle_root
    assert launcher_bootstrap.find_embedded_python(runtime.bundle_root, macos=False) == str(plain_python)
    assert runtime.bundle_root in platform_layer.bundled_resource_ancestor_bases(payload / 'Ouroboros.bin')
    monkeypatch.setattr(desktop_autostart.sys, 'platform', 'linux')
    for key, value in {'OUROBOROS_BUNDLE_DIR': str(runtime.bundle_root), 'OUROBOROS_PRESENTATION': 'desktop_window',
                       'OUROBOROS_MANAGED_BY_LAUNCHER': '1', 'OUROBOROS_APP_VERSION': '7.7.1'}.items():
        monkeypatch.setenv(key, value)
    assert desktop_autostart.launcher_target() == (payload / 'Ouroboros', '')
    (payload / 'Ouroboros').unlink()
    assert packaged_cli._desktop_app_path(runtime.bundle_root) is None
    with pytest.raises(packaged_cli.PackagedCLIError, match='could not locate packaged desktop launcher'):
        packaged_cli._launch_desktop_app(runtime)
    assert desktop_autostart.launcher_target()[0] is None
    assert len(children) == 1  # no fallback to the internal ELF, and no registration was changed


@pytest.mark.serial
def test_package_builder_stages_renamed_payload_and_public_desktop_unit(tmp_path, renamed_linux_payload):
    """Actual builder staging/spec; package encoders are recording fixtures, not native installs."""
    import json
    import sys

    tools, evidence, dist = (tmp_path / name for name in ('tools', 'package-records', 'dist'))
    tools.mkdir()
    evidence.mkdir()
    # dpkg-deb snapshots staging; rpmbuild executes its generated %install in a
    # temporary buildroot. Actual package encoding/installing remains native QA.
    encoder = '''import json, os, pathlib, shutil, subprocess, sys
args = sys.argv[1:]
evidence = pathlib.Path(os.environ['TEST_PACKAGE_RECORDS'])
if pathlib.Path(sys.argv[0]).name == 'dpkg-deb':
    shutil.copytree(args[-2], evidence / 'deb-root', symlinks=True)
    pathlib.Path(args[-1]).write_bytes(b'fixture deb, not a package')
else:
    defines = dict(args[i+1].split(' ', 1) for i, arg in enumerate(args) if arg == '--define')
    suffix = defines.get('rel_suffix', '')
    spec = pathlib.Path(args[-1]).read_text()
    (evidence / ('rpm' + suffix + '.spec')).write_text(spec)
    buildroot = evidence / ('rpm' + suffix + '-root')
    install = spec.split('%install\\n', 1)[1].split('\\n%files', 1)[0]
    subprocess.run(['bash', '-eu', '-c', install.replace('%{buildroot}', str(buildroot))], check=True)
    version = next(line.split()[1] for line in spec.splitlines() if line.startswith('Version:'))
    target = pathlib.Path(defines['_topdir']) / 'RPMS/x86_64' / ('ouroboros-' + version + '-1' + suffix + '.x86_64.rpm')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'fixture rpm, not a package')
'''
    for tool in ('dpkg-deb', 'rpmbuild'):
        path = tools / tool
        path.write_text(f'#!{sys.executable}\n' + encoder)
        path.chmod(0o755)
    env = {**os.environ, 'PATH': str(tools) + os.pathsep + os.environ['PATH'],
           'TEST_PACKAGE_RECORDS': str(evidence)}
    subprocess.run(['bash', 'scripts/build_linux_packages.sh', str(renamed_linux_payload), str(dist)],
                   cwd=REPO, env=env, check=True, capture_output=True, text=True)
    stage = evidence / 'deb-root'
    installed = stage / 'opt/ouroboros'
    for name in ('Ouroboros', 'Ouroboros.bin', '_internal/linux-gtk/desktop-env.sh'):
        assert (installed / name).read_bytes() == (renamed_linux_payload / name).read_bytes()
    assert os.access(installed / 'Ouroboros', os.X_OK) and os.access(installed / 'Ouroboros.bin', os.X_OK)
    assert (stage / 'usr/bin/ouroboros').readlink() == pathlib.Path('/opt/ouroboros/bin/ouroboros')
    desktop = (stage / 'usr/share/applications/ouroboros.desktop').read_text().splitlines()
    assert 'Exec=/opt/ouroboros/Ouroboros' in desktop
    unit = (stage / 'usr/lib/systemd/user/ouroboros.service').read_text().splitlines()
    command = next(line.removeprefix('ExecStart=') for line in unit if line.startswith('ExecStart='))
    assert command == '/opt/ouroboros/Ouroboros --launch-intent automatic'
    assert 'KillMode=control-group' in unit and not any(line.startswith('Restart=') for line in unit)
    output = tmp_path / 'unit-entry.json'
    subprocess.run([str(installed / 'Ouroboros'), *command.split()[1:]],
                   env={**env, 'TEST_ENTRY_OUTPUT': str(output)}, check=True)
    assert json.loads(output.read_text())['argv'] == [str(installed / 'Ouroboros.bin'), '--launch-intent', 'automatic']
    for name in ('rpm.spec', 'rpm.red80.spec'):
        spec = (evidence / name).read_text()
        rpm_root = evidence / name.replace('.spec', '-root')
        assert (rpm_root / 'opt/ouroboros/Ouroboros.bin').read_bytes() == (installed / 'Ouroboros.bin').read_bytes()
        assert (rpm_root / 'opt/ouroboros/Ouroboros').read_bytes() == (installed / 'Ouroboros').read_bytes()
        assert (rpm_root / 'usr/lib/systemd/user/ouroboros.service').read_bytes() == (stage / 'usr/lib/systemd/user/ouroboros.service').read_bytes()
        assert 'cp -al ' in spec and '"%{buildroot}/opt/ouroboros"' in spec
        assert '/opt/ouroboros\n' in spec.split('%files\n')[1]
        assert '"%{buildroot}/usr/lib/systemd/user/ouroboros.service"' in spec
    assert not (dist / '.package-stage').exists()  # builder cleanup still ran

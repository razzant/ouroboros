"""Actual entry/exec and child seams; no GTK display or installed user state."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from ouroboros import desktop_media as media

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name == 'nt', reason='POSIX packaged entry')


def native_root(tmp_path):
    root = tmp_path / '_internal/linux-gtk'
    root.mkdir(parents=True)
    (root / 'manifest.json').write_text('{}')
    (root / 'desktop-env.sh').write_bytes((ROOT / 'packaging/appimage/desktop-env.sh').read_bytes())
    return root


@pytest.mark.parametrize('entry', ['direct', 'appimage', 'symlink'])
@pytest.mark.parametrize('value', [None, '', '/caller/lib:literal $() `code` "quoted"\nКириллица:'])
def test_packaged_entry_preserves_exact_caller_environment_across_hooks_and_exec(tmp_path, entry, value):
    payload = tmp_path / 'app/usr/lib/ouroboros'
    root = native_root(payload)
    launcher = payload / 'Ouroboros'
    # Exercise the real entry, simulated bootloader/runtime-hook writes, source
    # exec, resource setup and a real helper process. Hook writes are destructive
    # on purpose, including the empty LD_LIBRARY_PATH loss in PyInstaller 6.22.
    code = f'''#!{sys.executable}
import json, os, runpy, subprocess, sys
from pathlib import Path
media = runpy.run_path({str(ROOT / 'ouroboros/desktop_media.py')!r})
if os.environ.get('AFTER_SOURCE_EXEC') != '1':
    for key in media['LINUX_DESKTOP_KEYS']:
        os.environ[key] = '/frozen-hook/' + key
    os.environ['AFTER_SOURCE_EXEC'] = '1'
    os.execve(sys.executable, [sys.executable, __file__], dict(os.environ))
media['prepare_linux_bundle']({str(payload / '_internal')!r})
media['prepare_linux_bundle']({str(payload / '_internal')!r})
gui = dict(os.environ)
host = media['host_environment']()
helper = subprocess.check_output([sys.executable, '-I', '-S', '-c',
    'import json, os; print(json.dumps(dict(os.environ)))'], text=True)
print(json.dumps({{'gui': gui, 'host': host, 'cwd': str(Path.cwd()), 'helper': json.loads(helper)}}))
'''
    launcher.write_text(code)
    launcher.chmod(0o755)
    runpy.run_path(str(ROOT / 'scripts/bundle_linux_gtk.py'))['install_desktop_entry'](payload)
    command = launcher
    if entry == 'appimage':
        command = tmp_path / 'app/AppRun'
        command.write_bytes((ROOT / 'packaging/appimage/AppRun').read_bytes())
        command.chmod(0o755)
    elif entry == 'symlink':
        command = tmp_path / 'desktop-link'
        command.symlink_to(launcher)
    env = {k: v for k, v in os.environ.items() if k not in media.LINUX_DESKTOP_KEYS
           and not k.startswith('OUROBOROS_DESKTOP_') and k != 'APPDIR'}
    if value is not None:
        env.update(dict.fromkeys(media.LINUX_DESKTOP_KEYS, value))
    observed = json.loads(subprocess.check_output([str(command)], env=env, cwd=tmp_path, text=True))
    for key in media.LINUX_DESKTOP_KEYS:
        assert (key in observed['host'], observed['host'].get(key)) == (value is not None, value), key
    assert not any(k.startswith('OUROBOROS_DESKTOP_') for k in observed['host'])
    assert observed['cwd'] == str(root)
    for snapshot in (observed['gui'], observed['helper']):
        assert snapshot['GI_TYPELIB_PATH'] == str(root / 'lib/x86_64-linux-gnu/girepository-1.0')
        assert snapshot['PYWEBVIEW_GUI'] == 'gtk'
        assert snapshot['OUROBOROS_DESKTOP_ENV_SAVED'] == '1'


@pytest.fixture
def gui_environment(tmp_path, monkeypatch):
    root = native_root(tmp_path)
    monkeypatch.chdir(tmp_path)
    env = {k: v for k, v in os.environ.items() if k not in media.LINUX_DESKTOP_KEYS
           and not k.startswith('OUROBOROS_DESKTOP_')}
    env.update(LD_LIBRARY_PATH='', XDG_DATA_DIRS='/caller/share', GIO_MODULE_DIR='/caller/gio',
               GST_PLUGIN_PATH='', PYWEBVIEW_GUI='qt')
    monkeypatch.setattr(os, 'environ', env)
    caller = dict(env)
    media.prepare_linux_bundle(root.parent)
    return caller, dict(os.environ)


def assert_host(actual, caller):
    for key in media.LINUX_DESKTOP_KEYS:
        assert (key in actual, actual.get(key)) == (key in caller, caller.get(key)), key
    assert not any(k.startswith('OUROBOROS_DESKTOP_HOST_') or k == 'OUROBOROS_DESKTOP_ENV_SAVED' for k in actual)


def assert_gui_unchanged(gui):
    # pytest moves its own setup/call marker after the fixture takes the snapshot.
    assert {k: v for k, v in os.environ.items() if k != 'PYTEST_CURRENT_TEST'} == {
        k: v for k, v in gui.items() if k != 'PYTEST_CURRENT_TEST'}


def test_start_agent_passes_host_environment_to_actual_server_child(tmp_path, monkeypatch, gui_environment):
    import launcher

    caller, gui = gui_environment
    repo = tmp_path / 'repo'
    repo.mkdir()
    output = tmp_path / 'server-environment.json'
    (repo / 'server.py').write_text(
        'import json, os, pathlib\n' + f'pathlib.Path({str(output)!r}).write_text(json.dumps(dict(os.environ)))\n')
    monkeypatch.setattr(launcher, 'IS_LINUX', True)
    monkeypatch.setattr(launcher, 'IS_WINDOWS', False)
    monkeypatch.setattr(launcher, 'DATA_DIR', tmp_path / 'data')
    monkeypatch.setattr(launcher, 'REPO_DIR', repo)
    monkeypatch.setattr(launcher, 'EMBEDDED_PYTHON', sys.executable)
    monkeypatch.setattr(launcher, '_load_settings', lambda: {})
    monkeypatch.setattr(launcher, '_apply_settings_to_env', lambda _: None)
    monkeypatch.setattr(launcher, '_agent_proc', None)
    gui = dict(os.environ)
    process = launcher.start_agent(12345)
    try:
        assert process.wait(timeout=10) == 0
        observed = json.loads(output.read_text())
        assert_host(observed, caller)
        # conftest deliberately removes managed-process authority markers at
        # real Popen; the loader/resource environment still reaches the child.
        assert observed['PYTHONPATH'] == str(repo)
        assert_gui_unchanged(gui)
    finally:
        process.stdin.close()


@pytest.mark.parametrize('external', [False, True])
@pytest.mark.parametrize('status', [0, 7])
def test_browser_uses_real_clean_child_without_mutating_gui(tmp_path, monkeypatch, gui_environment, external, status):
    import launcher

    caller, gui = gui_environment
    output = tmp_path / 'browser-environment.json'
    browser = tmp_path / 'browser'
    browser.write_text(f'#!{sys.executable}\nimport json, os, pathlib, sys\n'
                       f'pathlib.Path({str(output)!r}).write_text(json.dumps(dict(os.environ)))\n'
                       f'sys.exit({status})\n')
    browser.chmod(0o755)
    monkeypatch.setenv('BROWSER', str(browser))
    gui['BROWSER'] = str(browser)
    # No system default opener may run on this Mac, even on the negative path.
    # The child still executes the production browser command, with a registry
    # containing only our real GenericBrowser fixture.
    python = tmp_path / 'browser-python'
    python.write_text(f'#!{sys.executable}\nimport os, sys, webbrowser\n'
                      'webbrowser._browsers = {}; webbrowser._tryorder = []\n'
                      "webbrowser.register('fixture', None, webbrowser.GenericBrowser(os.environ['BROWSER']), preferred=True)\n"
                      "code = sys.argv[4]; sys.argv = ['-c', *sys.argv[5:]]; exec(code)\n")
    python.chmod(0o755)
    monkeypatch.setattr(launcher, 'IS_LINUX', True)
    monkeypatch.setattr(launcher, 'EMBEDDED_PYTHON', str(python))
    if external:
        assert launcher._open_external_url('https://example.invalid')['ok'] is (status == 0)
    else:
        outcome = []
        thread = launcher._open_browser_detached('https://example.invalid', outcome)
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert outcome == [status == 0]
    assert_host(json.loads(output.read_text()), caller)
    assert_gui_unchanged(gui)


def test_source_probe_and_exec_bases_keep_gui_while_bootstrap_defaults_are_clean(tmp_path, gui_environment, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import launcher_bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, 'sys', SimpleNamespace(platform='linux'))
    embedded_python_env = bootstrap.embedded_python_env

    caller, gui = gui_environment
    assert_host(embedded_python_env(tmp_path), caller)
    explicit = embedded_python_env(tmp_path, gui)
    for key in (*media.LINUX_DESKTOP_KEYS, 'OUROBOROS_DESKTOP_ENV_SAVED'):
        assert explicit.get(key) == gui.get(key)
    assert_gui_unchanged(gui)


def test_external_file_opener_passes_clean_environment(monkeypatch, gui_environment):
    from ouroboros import platform_layer

    caller, gui = gui_environment
    seen = {}
    monkeypatch.setattr(platform_layer, 'IS_MACOS', False)
    monkeypatch.setattr(platform_layer, 'IS_WINDOWS', False)
    monkeypatch.setattr(platform_layer.subprocess, 'Popen', lambda cmd, **kw: seen.update(cmd=cmd, **kw))
    platform_layer.open_path_external('/tmp/qualification-only-file')
    assert seen['cmd'] == ['xdg-open', '/tmp/qualification-only-file']
    assert_host(seen['env'], caller)
    assert_gui_unchanged(gui)


def test_broken_snapshot_is_an_error_not_a_partially_clean_host_environment(gui_environment, monkeypatch):
    monkeypatch.delenv('OUROBOROS_DESKTOP_HOST_LD_LIBRARY_PATH')
    with pytest.raises(KeyError):
        media.host_environment()


def test_gtk_blank_navigation_uses_same_detached_opener_after_backend_selection(gui_environment, monkeypatch):
    import launcher
    import types
    import webbrowser

    from tests.test_launcher_headless_fallback import _install_fake_webview

    _install_fake_webview(monkeypatch, init_raises=None)
    monkeypatch.setattr(launcher, 'IS_LINUX', True)
    monkeypatch.setenv('DISPLAY', ':qualification-fixture')
    monkeypatch.setattr(launcher, '_display_ready', lambda _: (True, ''))
    called = []
    stdlib_open = webbrowser.open
    monkeypatch.setattr(launcher, '_open_browser_detached', lambda url, **kw: called.append((url, kw)) or object())
    assert launcher._webview_ready() == (True, '')
    from webview.guilib import initialize
    backend = initialize()
    # This is the exact three-argument call in pywebview 5.4 GTK on_navigation.
    assert backend.webbrowser.open('https://example.invalid', 2, True)
    assert called == [('https://example.invalid', {'new': 2, 'autoraise': True})]
    assert webbrowser.open is stdlib_open
    qt = types.SimpleNamespace(__name__='webview.platforms.qt', webbrowser=webbrowser)
    media.route_gtk_browser(qt, lambda *_: None)
    assert qt.webbrowser is webbrowser

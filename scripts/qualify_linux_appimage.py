#!/usr/bin/env python3
"""Native GTK qualification of one real AppImage in the clean X11 CI container.

Run through safe_test, inside a network-disabled, non-root disposable container.
Only an inert loopback provider is configured through the real onboarding API;
every generation request is recorded, refused and fails qualification. The real
MainApi bridge is exercised by a recorded HTML suffix in the disposable repo.
The AppImage bytes are never edited. A local comment-only fixture commit then
exercises the production cold source-launcher handoff. No update feed is used.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import runpy
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import uuid


DESKTOP_KEYS = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'ouroboros/desktop_media.py'))['LINUX_DESKTOP_KEYS']

HOST_GRAPHICS = ('libGL.so.1', 'libEGL.so.1', 'libGLESv2.so.2', 'libGLdispatch.so.0', 'libgbm.so.1', 'libdrm.so.2')


def setup_payload(base_url):
    """The real wizard contract, without credentials, subscriptions or installs."""
    return {'OPENAI_COMPATIBLE_BASE_URL': base_url,
            'OUROBOROS_MODEL': 'openai-compatible::native-qualification-inert',
            'OUROBOROS_REVIEW_ENFORCEMENT': 'advisory',
            'OUROBOROS_RUNTIME_MODE': 'advanced',
            'subscriptionsConnected': False, 'skipSubscriptionPresets': True,
            'starter_skills': []}


def fixture_receiver(evidence, nonce):
    """One loopback listener for bridge evidence and a NON-generating provider."""
    received, unexpected = [], []

    class Receiver(BaseHTTPRequestHandler):
        def answer(self, status, body):
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            self.answer(200 if self.path == '/v1/models' else 404,
                        {'object': 'list', 'data': [{'id': 'native-qualification-inert', 'object': 'model'}]}
                        if self.path == '/v1/models' else {'error': 'fixture route not supported'})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get('Content-Length', 0))).decode()
            if self.path == '/bridge':
                data = json.loads(raw)
                if data.get('nonce') != nonce:
                    self.answer(403, {'error': 'wrong witness nonce'})
                    return
                received.append(data)
                self.answer(200, {'ok': True})
                return
            attempt = {'method': 'POST', 'path': self.path, 'body': raw, 'refused': True}
            unexpected.append(attempt)
            with (evidence / 'unexpected-generation.jsonl').open('a') as stream:
                stream.write(json.dumps(attempt) + '\n')
            self.answer(503, {'error': 'native qualification never generates'})

        def log_message(self, *args):
            pass

    receiver = ThreadingHTTPServer(('127.0.0.1', 0), Receiver)
    threading.Thread(target=receiver.serve_forever, daemon=True).start()
    return receiver, received, unexpected


def complete_setup(port, base_url, evidence):
    payload = setup_payload(base_url)
    (evidence / 'setup-payload.json').write_text(json.dumps(payload, indent=2))
    request = urllib.request.Request(f'http://127.0.0.1:{port}/api/onboarding/complete',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read().decode()
        (evidence / 'setup-response.json').write_text(json.dumps({'status': response.status, 'body': raw}, indent=2))
        assert response.status == 200, raw
        result = json.loads(raw)
        assert result.get('ok') is True and result.get('restart_required') is False, result
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/onboarding', timeout=10) as response:
        (evidence / 'setup-readiness.json').write_text(json.dumps({'status': response.status}))
        assert response.status == 204, 'Completed fixture must be structurally configured'


def install_witness(page, receiver_port, nonce, evidence):
    original = page.read_text()
    # Keep the original UI. In particular, never dismiss its blocking overlay.
    suffix = """<script>
    (() => {
        function visible(node) {
            if (!node || !node.innerText.trim()) return false;
            const box = node.getBoundingClientRect(), style = getComputedStyle(node);
            return box.width > 0 && box.height > 0 && box.bottom > 0 && box.right > 0 &&
                box.top < innerHeight && box.left < innerWidth &&
                style.display !== 'none' && style.visibility === 'visible' && Number(style.opacity) > 0;
        }
        function ready() {
            return document.readyState === 'complete' && !document.querySelector('#onboarding-overlay') &&
                visible(document.querySelector('#primary-sidebar')) &&
                document.querySelector('#primary-sidebar').innerText.includes('Main Chat') &&
                visible(document.querySelector('#content'));
        }
        let busy = false;
        const witness = setInterval(async () => {
            if (busy || !window.pywebview?.api?.shell_info || !ready()) return;
            busy = true;
            try {
                await document.fonts.ready;
                await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                const info = await window.pywebview.api.shell_info();
                if (!ready()) return;
                await fetch('http://127.0.0.1:PORT/bridge', {method:'POST',
                    body:JSON.stringify({nonce:'NONCE', info, title:document.title, href:location.href,
                        overlay_absent:true, main_visible:true, text:document.body.innerText.slice(0,2000)})});
                clearInterval(witness);
            } finally { busy = false; }
        }, 200);
    })();
    </script>""".replace('PORT', str(receiver_port)).replace('NONCE', nonce)
    page.write_text(original + suffix)
    (evidence / 'bridge-instrumentation.html').write_text(suffix)
    return original


def assert_bridge(proof, source, revision, port):
    assert proof['info']['source_launcher'] is source, proof
    assert proof['info']['source_revision'] == revision, proof
    assert proof.get('href') == f'http://127.0.0.1:{port}/', proof
    assert proof.get('overlay_absent') is True and proof.get('main_visible') is True, proof


def refresh_fixture_endpoint(settings_path, base_url, evidence):
    """Only the ephemeral endpoint changes between cold launches; no setup replay."""
    saved = json.loads(settings_path.read_text())
    before = saved['OPENAI_COMPATIBLE_BASE_URL']
    saved['OPENAI_COMPATIBLE_BASE_URL'] = base_url
    settings_path.write_text(json.dumps(saved, indent=2))
    (evidence / 'fixture-endpoint-refresh.json').write_text(json.dumps({'before': before, 'after': base_url}))


def run(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def wait_for(read, description, timeout=180, process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'{description}: AppImage exited first with {process.returncode}')
        try:
            value = read()
            if value:
                return value
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
        time.sleep(0.25)
    raise RuntimeError(f'Timed out: {description}')


def descendants(pid):
    parents = {}
    for status in Path('/proc').glob('[0-9]*/status'):
        try:
            fields = dict(line.split(':', 1) for line in status.read_text().splitlines() if ':' in line)
            parents[int(status.parent.name)] = int(fields['PPid'])
        except (OSError, ValueError, KeyError):
            pass
    found = {pid}
    while more := {child for child, parent in parents.items() if parent in found} - found:
        found |= more
    return found


def find_window(title, evidence, launcher_pid, wm_class='Ouroboros'):
    """Observe actual clients and their raw properties; an error is never absence."""
    reader = Path(__file__).with_name('x11_window_identity.py')
    args = [sys.executable, '-I', '-S', str(reader)]
    observed = subprocess.run(args, capture_output=True, timeout=10)
    report = {'argv': args, 'returncode': observed.returncode,
              'stdout': observed.stdout.decode('ascii', errors='backslashreplace'),
              'stderr': observed.stderr.decode('utf-8', errors='backslashreplace'),
              'requested_title': title, 'launcher_pid': launcher_pid, 'requested_class': wm_class}
    # Keep every poll, including failures, and the last observation for inspection.
    with (evidence / 'window-search.jsonl').open('a') as stream:
        stream.write(json.dumps(report) + '\n')
    (evidence / 'window-search.json').write_text(json.dumps(report, indent=2))
    if observed.returncode != 0 or observed.stderr:
        raise RuntimeError(f'X11 identity reader failed: {report}')
    try:
        observation = json.loads(observed.stdout)
    except ValueError as exc:
        raise RuntimeError(f'Invalid X11 identity response: {report}') from exc
    return runpy.run_path(str(reader))['select_window'](observation, launcher_pid, title, wm_class)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def process_evidence(pid, evidence, filename='processes.json'):
    records = []
    for child in sorted(descendants(pid)):
        proc = Path(f'/proc/{child}')
        try:
            maps = (proc / 'maps').read_text()
            env = dict(part.split('=', 1) for part in (proc / 'environ').read_bytes().decode().split('\0') if '=' in part)
            record = {'pid': child, 'exe': os.readlink(proc / 'exe'), 'cwd': os.readlink(proc / 'cwd'),
                      'mount_namespace': os.readlink(proc / 'ns/mnt'),
                      'argv': (proc / 'cmdline').read_bytes().decode().split('\0'),
                      'environment': {k: v for k, v in env.items() if k in (
                          'OUROBOROS_PRESENTATION', *DESKTOP_KEYS,
                          'WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS', 'WEBKIT_FORCE_SANDBOX')
                          or k.startswith('OUROBOROS_DESKTOP_')},
                      'maps': maps}
            records.append(record)
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError as exc:
            records.append({'pid': child, 'unreadable': str(exc), 'exe': '', 'maps': '', 'environment': {}})
    (evidence / filename).write_text(json.dumps(records, indent=2))
    return records


def assert_server_environment(records, caller):
    servers = [r for r in records if any(arg.endswith('/server.py') for arg in r.get('argv', []))]
    assert servers, 'No actual server process observed'
    for server in servers:
        actual = server['environment']
        for key in DESKTOP_KEYS:
            assert (key in actual, actual.get(key)) == (key in caller, caller.get(key)), (key, actual)
        assert not any(key.startswith('OUROBOROS_DESKTOP_HOST_') or key == 'OUROBOROS_DESKTOP_ENV_SAVED'
                       for key in actual), actual


def sandbox_observation(records, launcher_namespace):
    """Record WebKit's own web-process isolation; neither outcome is required.

    WebKitGTK 4.1 sandboxes only after webkit_web_context_set_sandbox_enabled,
    which pywebview 5.4 never calls (packaging/appimage/README.md).
    """
    return {'launcher_mount_namespace': launcher_namespace,
            'web_processes': [{'pid': r['pid'], 'mount_namespace': r['mount_namespace'],
                               'separate_mount_namespace': r['mount_namespace'] != launcher_namespace}
                              for r in records if r['exe'].endswith('/WebKitWebProcess')],
            'sandbox_helpers': sorted({os.path.basename(r['exe']) for r in records} & {'bwrap', 'xdg-dbus-proxy'}),
            'webkit_sandbox_variables': sorted({k for r in records for k in r['environment'] if k.startswith('WEBKIT_')})}


def one_case(appimage, state, evidence, *, source=False, display=True, native=True, command=None):
    evidence.mkdir(parents=True)
    # Preserve safe_test process ownership markers; replace the fixture's data roots.
    env = dict(os.environ)
    # PIP_REQUIRE_VIRTUALENV is safe_test's host guard. The product's own pip
    # installs --user into the disposable data dir and must not be refused.
    for key in ('APPDIR', 'APPIMAGE', *DESKTOP_KEYS,
                'PIP_REQUIRE_VIRTUALENV'):
        env.pop(key, None)
    for name in ('home', 'data', 'app', 'tmp', 'cache', 'config', 'xdg-data'):
        (state / name).mkdir(parents=True, exist_ok=True)
    env.update(HOME=str(state / 'home'), XDG_CACHE_HOME=str(state / 'cache'),
               XDG_CONFIG_HOME=str(state / 'config'), XDG_DATA_HOME=str(state / 'xdg-data'),
               TMPDIR=str(state / 'tmp'), OUROBOROS_DATA_DIR=str(state / 'data'),
               OUROBOROS_APP_ROOT=str(state / 'app'), OUROBOROS_REPO_DIR=str(state / 'app/repo'),
               OUROBOROS_PID_FILE=str(state / 'app/ouroboros.pid'),
               OUROBOROS_PORT_FILE=str(state / 'data/state/server_port'),
               OUROBOROS_SETTINGS_PATH=str(state / 'data/settings.json'),
               OUROBOROS_SERVER_PORT='0', BROWSER='/bin/true', APPIMAGE_EXTRACT_AND_RUN='1')
    if not display:
        env.pop('DISPLAY', None)
        env.pop('WAYLAND_DISPLAY', None)
    port_file = state / 'data/state/server_port'
    pid_file = state / 'app/ouroboros.pid'
    port_file.unlink(missing_ok=True)
    nonce = uuid.uuid4().hex
    receiver, received, unexpected = fixture_receiver(evidence, nonce)
    base_url = f'http://127.0.0.1:{receiver.server_port}/v1'
    page = state / 'app/repo/web/index.html'
    settings_path = state / 'data/settings.json'
    configured = settings_path.exists()
    original = None
    process = None
    tracked = set()
    try:
        if configured:
            refresh_fixture_endpoint(settings_path, base_url, evidence)
            if native:
                # Configured launches skip Setup; Main can load before health is read.
                original = install_witness(page, receiver.server_port, nonce, evidence)
        with (evidence / 'stdout.log').open('w') as log:
            process = subprocess.Popen(command or [str(appimage)], env=env, stdout=log, stderr=subprocess.STDOUT)
            launcher_pid = wait_for(lambda: int(pid_file.read_text()), 'launcher pid', process=process)
            port = wait_for(lambda: int(port_file.read_text()), 'server port', process=process)
            def health():
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=2) as response:
                    return json.load(response)
            (evidence / 'health.json').write_text(json.dumps(wait_for(health, 'server health', process=process), indent=2))
            if native:
                wm_class = 'Launcher.py' if source else 'Ouroboros.bin'
                if not configured:
                    setup = wait_for(lambda: find_window('Ouroboros — Setup', evidence, launcher_pid, wm_class),
                                     'Setup', process=process)
                    (evidence / 'setup-identity.json').write_text(json.dumps(setup, indent=2))
                    (evidence / 'setup-title.txt').write_text(setup['title'], encoding='utf-8')
                    run('import', '-window', str(setup['id']), str(evidence / 'setup-window.png'))
                    original = install_witness(page, receiver.server_port, nonce, evidence)
                    complete_setup(port, base_url, evidence)
                    # Ask Openbox for an ordinary WM close; XDestroyWindow bypasses GTK cleanup.
                    run('wmctrl', '-ic', hex(setup['id']))
                version = (state / 'app/repo/VERSION').read_text().strip()
                identity = wait_for(lambda: find_window(f'Ouroboros v{version}', evidence, launcher_pid, wm_class),
                                    'Main window', process=process)
                (evidence / 'main-identity.json').write_text(json.dumps(identity, indent=2))
                window = str(identity['id'])
                proof = wait_for(lambda: received[0] if received else None, 'real MainApi shell_info roundtrip', process=process)
                (evidence / 'bridge.json').write_text(json.dumps(proof, indent=2))
                assert_bridge(proof, source, run('git', '-C', str(state / 'app/repo'), 'rev-parse', 'HEAD'), port)
                geometry = run('xwininfo', '-id', window)
                assert 'IsViewable' in geometry
                (evidence / 'window.txt').write_text(geometry + '\n' + run('xprop', '-id', window))
                run('import', '-window', window, str(evidence / 'window.png'))
                records = process_evidence(process.pid, evidence)
                assert_server_environment(records, env)
                (evidence / 'webkit-sandbox.json').write_text(json.dumps(
                    sandbox_observation(records, os.readlink(f'/proc/{launcher_pid}/ns/mnt')), indent=2))
                for library in ('libgtk-3', 'libwebkit2gtk-4.1', 'libjavascriptcoregtk-4.1', 'libgirepository-1.0', 'libglib-2.0', 'libsoup-3.0'):
                    mapped = [line for r in records for line in r['maps'].splitlines() if library in line]
                    assert mapped and all('/_internal/linux-gtk/' in line for line in mapped), (library, mapped)
                for helper in ('WebKitWebProcess', 'WebKitNetworkProcess'):
                    matches = [r for r in records if r['exe'].endswith('/' + helper)]
                    assert matches and all('/_internal/linux-gtk/' in r['exe'] for r in matches), helper
                assert any(r['environment'].get('OUROBOROS_PRESENTATION') == 'desktop_window' for r in records)
                if source:
                    assert 'python-standalone' in os.readlink(f'/proc/{launcher_pid}/exe')
                    assert 'Loading managed source launcher through' in (state / 'data/logs/launcher.log').read_text()
                tracked = descendants(process.pid)
                run('wmctrl', '-ic', hex(identity['id']))
            else:
                records = process_evidence(process.pid, evidence)
                assert_server_environment(records, env)
                assert any(r['environment'].get('OUROBOROS_PRESENTATION') == 'browser_fallback' for r in records)
                warning = (state / 'data/logs/launcher.log').read_text()
                assert 'continuing in browser mode' in warning
                if display:
                    assert 'no DISPLAY or WAYLAND_DISPLAY' not in warning
                assert not any('WebKitWebProcess' in r['exe'] for r in records)
                tracked = descendants(process.pid)
                os.kill(launcher_pid, signal.SIGTERM)
            assert process.wait(timeout=45) == 0
            wait_for(lambda: not pid_file.exists(), 'PID lock cleanup', timeout=10)
            wait_for(lambda: not any(alive(pid) for pid in tracked), 'helper/server cleanup', timeout=15)
            assert not unexpected, f'Unexpected generation requests refused: {unexpected}'
            (evidence / 'result.json').write_text(json.dumps({'status': 'mechanical_checks_passed', 'tracked_pids': sorted(tracked)}))
    finally:
        if process is not None and process.poll() is None:
            process_evidence(process.pid, evidence, 'failure-processes.json')
            if native:
                subprocess.run(['import', '-window', 'root', str(evidence / 'failure-desktop.png')], timeout=10)
                for name, args in (
                    ('x11-tree', ['xwininfo', '-root', '-tree']),
                    ('x11-clients', ['xprop', '-root', '_NET_CLIENT_LIST']),
                    ('x11-properties', [sys.executable, '-I', '-S', str(Path(__file__).with_name('x11_window_identity.py'))]),
                ):
                    diagnostic = subprocess.run(args, text=True, capture_output=True, timeout=10)
                    (evidence / (name + '.json')).write_text(json.dumps({
                        'argv': args, 'returncode': diagnostic.returncode,
                        'stdout': diagnostic.stdout, 'stderr': diagnostic.stderr,
                    }, indent=2))
            # Existing safe_test owns final tree cleanup if ordinary teardown fails.
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGTERM)
                except (OSError, ValueError):
                    pass
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                pass
        receiver.shutdown()
        receiver.server_close()
        (evidence / 'generation-summary.json').write_text(json.dumps({'unexpected_count': len(unexpected), 'all_refused': True}))
        if original is not None:
            page.write_text(original)
        logs = state / 'data/logs'
        if logs.exists():
            shutil.copytree(logs, evidence / 'logs', dirs_exist_ok=True)
        if not (evidence / 'result.json').exists():
            (evidence / 'result.json').write_text(json.dumps({
                'status': 'incomplete_or_failed',
                'returncode': None if process is None else process.poll()}) + '\n')


def main():
    assert sys.platform == 'linux' and os.uname().machine == 'x86_64'
    assert os.getuid() != 0
    assert os.environ.get('OUROBOROS_TEST_TEMP_ROOT'), 'run through safe_test'
    appimage, evidence = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'appimage.sha256').write_text(run('sha256sum', str(appimage)) + '\n')
    (evidence / 'desktop-packages.tsv').write_text(run('dpkg-query', '-W'))
    ldconfig = run('/sbin/ldconfig', '-p')
    (evidence / 'desktop-ldconfig.txt').write_text(ldconfig)
    assert not any(name in ldconfig for name in ('libwebkit', 'libjavascriptcoregtk', 'libgtk-3', 'libgirepository'))
    # The declared host graphics ABI (bundle_linux_gtk.HOST_LIBRARIES plus the
    # GLES front-end the shipped libepoxy dlopens) must be the host's own.
    for soname in HOST_GRAPHICS:
        assert f'\t{soname} (' in ldconfig, f'host graphics ABI missing: {soname}'
    assert not any(key in os.environ for key in ('WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS', 'WEBKIT_DISABLE_SANDBOX'))
    for package in ('libwebkit2gtk-4.1-0', 'libgtk-3-0', 'gir1.2-webkit2-4.1', 'python3-gi'):
        assert subprocess.run(['dpkg-query', '-W', package], capture_output=True).returncode != 0, package
    state = Path(os.environ['OUROBOROS_TEST_TEMP_ROOT']) / 'native-state'
    one_case(appimage, state, evidence / 'frozen')
    repo = state / 'app/repo'
    with (repo / 'launcher.py').open('a') as stream:
        stream.write('\n# issue688: disposable source-handoff qualification fixture\n')
    run('git', '-C', str(repo), '-c', 'core.hooksPath=/dev/null',
        '-c', 'user.name=Native qualification', '-c', 'user.email=qa@example.invalid',
        'commit', '--only', 'launcher.py', '-m', 'Disposable handoff fixture')
    (evidence / 'handoff-source.txt').write_text(run('git', '-C', str(repo), 'rev-parse', 'HEAD') + '\n')
    (evidence / 'handoff-fixture.patch').write_text(run('git', '-C', str(repo), 'show', '--format=fuller', 'HEAD') + '\n')
    one_case(appimage, state, evidence / 'source-handoff', source=True)
    one_case(appimage, state, evidence / 'no-display', source=True, display=False, native=False)
    # Damage a separate extraction, never the measured AppImage or first two cases.
    damaged = state / 'damaged'
    damaged.mkdir()
    run(str(appimage), '--appimage-extract', cwd=damaged)
    extracted = damaged / 'squashfs-root'
    # deb/rpm/tarball execute this exact public payload entry, bypassing AppRun.
    one_case(appimage, state / 'direct-entry', evidence / 'direct-entry',
             command=[str(extracted / 'usr/lib/ouroboros/Ouroboros')])
    for typelib in extracted.rglob('WebKit2-4*.typelib'):
        typelib.unlink()
    one_case(appimage, state / 'missing-backend', evidence / 'missing-backend', native=False,
             command=[str(extracted / 'AppRun')])
    # Visual review is an independent required step; this script cannot approve its own screenshots.
    (evidence / 'result.json').write_text(json.dumps({'status': 'mechanical_checks_passed_visual_review_pending',
        'scope': 'Ubuntu 22.04 x86_64; Xvfb/Openbox virtual X11; no Wayland or physical GPU claim'}, indent=2))


if __name__ == '__main__':
    main()

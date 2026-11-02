"""An ESM widget's declared files download in the desktop app and in a browser.

The desktop URL bridge admits only its own loopback API prefixes, so the page
reads a declared file with the mount's lease and hands the captured bytes to
the existing byte owner; skill routes stay on the streamed URL bridge. The
served page's ``window.pywebview.api`` relays each call through the test server
to launcher.py's MainApi, run unchanged (the AST harness of
tests/test_widget_stream_download_ui.py), in place of pywebview's own
marshalling. No native window is opened. The same widget also mounts behind a
reverse proxy that rewrites Host: the lease controls take the browser's Fetch
Metadata, not the URL this server sees.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.test_evidence import diagnostic_directory
from tests.widget_esm_fixture import BINARY, NAME, install

pytestmark = pytest.mark.ui_browser
REPO = Path(__file__).resolve().parents[1]
JSON_FILE = b'{"value":"declared"}'
_PAGE = '''<!doctype html><html><head><meta charset="utf-8"></head><body>
<section data-widget-key="esm_read_files:main"><div class="widgets-card-status"></div><div id="mount"></div></section>
<script>SHELL</script>
<script type="module">
import {mountModuleWidget} from '/web/modules/widget_module.js';
const tab = (await (await fetch('/api/widgets')).json()).ui_tabs.find(row => row.tab_id === 'main');
window.disposeWidget = await mountModuleWidget(document.getElementById('mount'), tab, tab.render);
</script></body></html>'''
# pywebview's js_api shape: every public MainApi method answers a Promise.
_SHELL = '''window.__bridgeCalls = [];
window.pywebview = {api: Object.fromEntries(METHODS.map(name => [name, async (...args) => {
  const reply = await fetch('/__launcher', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                           body: JSON.stringify({name, args})});
  const result = await reply.json();
  window.__bridgeCalls.push({name, file: String(args[name === 'save_bytes_to_downloads' ? 0 : 1] || '')});
  return result;
}]))};'''
_DOWNLOADS = '''async () => {
  // evaluate() has no timeout: an unanswered download reports instead of hanging the test.
  const late = () => new Promise((_, reject) => setTimeout(() => reject(new Error('no answer in 20 s')), 20000));
  const settle = (pending) => Promise.race([pending, late()]).then(result => ({settled: 'resolved', ...result}),
                                                                   error => ({settled: 'rejected', error: error.message}));
  return {asset: await settle(OuroborosWidget.download('esm-data.bin', './data.bin')),
          route: await settle(OuroborosWidget.download('route-ping.json', '/api/extensions/esm_read_files/ping'))};
}'''
_DOWNLOAD_ONE = '''([name, source]) => Promise.race([OuroborosWidget.download(name, source),
  new Promise((_, reject) => setTimeout(() => reject(new Error('no answer in 20 s')), 20000))])'''
_ANCHOR = '''() => { const a = document.createElement('a'); a.id = 'save-json'; a.href = 'data.json';
  a.download = 'esm-anchor.json'; a.textContent = 'Save JSON'; document.body.append(a); }'''


@pytest.fixture
def esm_server(tmp_path, monkeypatch):
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles

    from ouroboros import extension_loader, server_auth
    from ouroboros.gateway.extensions import api_extension_dispatch
    from ouroboros.gateway.widget_assets import api_widget_asset_lease
    from ouroboros.gateway.widgets import api_widgets
    from ouroboros.skill_loader import find_skill
    from tests._shared import clean_extension_runtime_state

    def page(request):
        api = request.app.state.launcher
        shell = _SHELL.replace('METHODS', json.dumps(sorted(
            name for name in dir(api) if not name.startswith('_') and callable(getattr(api, name))))) if api else ''
        return HTMLResponse(_PAGE.replace('SHELL', shell))

    async def launcher_call(request):
        # Off the loop, so the server still answers the launcher's own loopback download.
        call, state = await request.json(), request.app.state
        result = await asyncio.to_thread(getattr(state.launcher, call['name']), *call['args'])
        state.launcher_calls.append({'method': call['name'], 'url_or_name': call['args'][0], 'result': result})
        return JSONResponse(result)

    clean_extension_runtime_state()
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    monkeypatch.delenv('OUROBOROS_SKILLS_REPO_PATH', raising=False)
    # The desktop window is a loopback owner: the real gate serves leases and files without a password.
    monkeypatch.setattr(server_auth, 'get_configured_network_password', lambda: '')
    root = tmp_path / 'drive'
    install(root)
    assert extension_loader.load_extension(find_skill(root, NAME), lambda: {}, drive_root=root) is None
    app = Starlette(routes=[Route('/', page), Route('/__launcher', launcher_call, methods=['POST']),
        Route('/api/widgets', api_widgets),
        Route('/api/widgets/asset-leases', api_widget_asset_lease, methods=['POST']),
        Route('/api/widgets/asset-leases/{op}', api_widget_asset_lease, methods=['POST']),
        Route('/api/extensions/{skill}/{rest:path}', api_extension_dispatch, methods=['GET', 'HEAD', 'POST']),
        Mount('/web', app=StaticFiles(directory=REPO / 'web'))])
    app.state.drive_root, app.state.repo_dir = root, REPO
    app.state.launcher, app.state.launcher_calls, app.state.proxy_host, app.state.lease_headers = None, [], '', []
    gate = server_auth.NetworkAuthGate(app)

    async def proxy(scope, receive, send):
        # With proxy_host set: a reverse proxy rewriting Host (nginx's default $proxy_host), so this server sees
        # another URL than the page's origin.
        if scope['type'] == 'http' and app.state.proxy_host:
            scope = {**scope, 'headers': [(name, app.state.proxy_host.encode() if name == b'host' else value)
                                          for name, value in scope['headers']]}
        if scope['type'] == 'http' and scope['path'] == '/api/widgets/asset-leases':
            sent = dict(scope['headers'])
            app.state.lease_headers.append({key: sent.get(key.encode(), b'').decode()
                                            for key in ('host', 'origin', 'sec-fetch-site')})
        await gate(scope, receive, send)

    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    server = uvicorn.Server(uvicorn.Config(proxy, log_level='warning'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(.01)
    assert server.started
    try:
        yield SimpleNamespace(port=sock.getsockname()[1], state=app.state)
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        clean_extension_runtime_state()
        assert not thread.is_alive()


def _mounted(page, port):
    page.goto(f'http://127.0.0.1:{port}/')
    page.frame_locator('iframe').locator('#root[data-ready="1"]').wait_for(timeout=30000)
    return page.locator('iframe').element_handle().content_frame()


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_desktop_saves_declared_files_as_bytes_and_streams_routes(esm_server, engine, tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright

    from tests.test_widget_stream_download_ui import native_file_api

    home = tmp_path / 'home'
    reads, calls = [], esm_server.state.launcher_calls
    api, launcher_sha = native_file_api(esm_server.port, reads)
    esm_server.state.launcher = api
    evidence = diagnostic_directory(tmp_path / 'evidence', f'widget-esm-download-desktop-{engine}')
    downloads = home / 'Downloads'
    outcome, base = {}, ''
    with sync_playwright() as playwright:
        browser = getattr(playwright, engine).launch()
        # Only after launch: WebKit stalls on a missing HOME; MainApi resolves ~/Downloads per call.
        monkeypatch.setenv('HOME', str(home))
        monkeypatch.setenv('USERPROFILE', str(home))
        try:
            page = browser.new_page()
            widget = _mounted(page, esm_server.port)
            base = widget.evaluate('document.baseURI')
            outcome = widget.evaluate(_DOWNLOADS)
            widget.evaluate(_ANCHOR)
            page.frame_locator('iframe').locator('#save-json').click()
            page.wait_for_function("() => window.__bridgeCalls.some(call => call.file === 'esm-anchor.json')")
        finally:
            browser.close()
            saved = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(downloads.glob('*'))} if downloads.is_dir() else {}
            (evidence / 'receipt.json').write_text(json.dumps({'engine': engine, 'launcher_sha256': launcher_sha,
                'outcome': outcome, 'calls': calls, 'reads': reads, 'saved': saved}, indent=2) + '\n')
    assert outcome['asset']['settled'] == 'resolved' and outcome['asset']['native'] is True, outcome
    assert outcome['route']['settled'] == 'resolved' and outcome['route']['native'] is True, outcome
    assert (downloads / 'esm-data.bin').read_bytes() == BINARY
    assert (downloads / 'esm-anchor.json').read_bytes() == JSON_FILE
    assert json.loads((downloads / 'route-ping.json').read_bytes()) == {'ok': True}
    assert {call['url_or_name'] for call in calls if call['method'] == 'save_bytes_to_downloads'} == {
        'esm-data.bin', 'esm-anchor.json'}
    assert [call['url_or_name'] for call in calls if call['method'] == 'download_file_to_downloads'] == [
        f'/api/extensions/{NAME}/ping'], calls
    assert reads and max(reads) <= 1024 * 1024  # the route download kept the launcher's streamed reads
    # The desktop URL bridge itself stays closed to capability URLs.
    refused = api.download_file_to_downloads(base.split(f':{esm_server.port}', 1)[1] + 'data.bin', 'direct.bin')
    assert refused['ok'] is False and 'file URL path must be' in refused['error']


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_browser_downloads_declared_files_and_routes(esm_server, engine, tmp_path):
    from playwright.sync_api import sync_playwright

    saved = {}
    with sync_playwright() as playwright:
        browser = getattr(playwright, engine).launch()
        try:
            page = browser.new_page(accept_downloads=True)
            widget = _mounted(page, esm_server.port)
            for name, source in (('esm-data.bin', './data.bin'),
                                 ('route-ping.json', f'/api/extensions/{NAME}/ping')):
                with page.expect_download() as pending:
                    result = widget.evaluate(_DOWNLOAD_ONE, [name, source])
                assert result['ok'] is True and result['native'] is False, result
                assert pending.value.suggested_filename == name
                pending.value.save_as(tmp_path / name)
                saved[name] = (tmp_path / name).read_bytes()
        finally:
            browser.close()
    assert saved['esm-data.bin'] == BINARY
    assert json.loads(saved['route-ping.json']) == {'ok': True}


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_widget_mounts_behind_a_host_rewriting_proxy(esm_server, engine):
    from playwright.sync_api import sync_playwright

    esm_server.state.proxy_host = 'core.internal:8765'
    with sync_playwright() as playwright:
        browser = getattr(playwright, engine).launch()
        try:
            widget = _mounted(browser.new_page(), esm_server.port)
            assert widget.evaluate('window.proof.value') == 'Module ready'
        finally:
            browser.close()
    # What the browser sent: its own origin and verdict, while this server saw the proxy's Host.
    assert esm_server.state.lease_headers[0] == {
        'host': 'core.internal:8765', 'origin': f'http://127.0.0.1:{esm_server.port}', 'sec-fetch-site': 'same-origin'}

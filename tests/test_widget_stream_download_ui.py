"""Optional real-browser/native-window widget exports through installed skill routes."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
import urllib.parse
import urllib.request

import pytest

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
REPO = Path(__file__).resolve().parents[1]

_WIDGET = r'''
const root = document.getElementById('root');
root.innerHTML = `<style>body{font:16px sans-serif;background:#18212d;color:#edf4ff;padding:20px}button{display:block;margin:14px 0;padding:12px 20px;min-width:280px}#status{padding-top:12px}</style>
<h2>Export from an isolated skill</h2>
<button id="blob">Export Blob</button><button id="data">Export data URL</button>
<button id="explicit">Save with widget API</button><button id="route">Download large file</button><button id="stream">Read stream</button><p id="status">Ready</p>`;
const anchor = (href, filename) => {
  const a = document.createElement('a'); a.href=href; a.download=filename;
  document.body.append(a); a.click(); a.remove();
};
document.getElementById('blob').onclick = () => {
  const url=URL.createObjectURL(new Blob(['blob export'], {type:'text/plain'}));
  anchor(url,'widget-blob.txt'); URL.revokeObjectURL(url);
};
document.getElementById('data').onclick = () => anchor('data:text/plain;base64,ZGF0YSBleHBvcnQ=','widget-data.txt');
document.getElementById('explicit').onclick = async () => {
  const result=await OuroborosWidget.download('widget-explicit.txt',new Blob(['explicit export']));
  document.getElementById('status').textContent=result.native?'Saved by the desktop app':'Browser download started';
};
document.getElementById('route').onclick = () => anchor('/api/extensions/export_widget/export','widget-large.bin');
document.getElementById('stream').onclick = async () => {
  const reader=(await OuroborosWidget.fetch('/api/extensions/export_widget/stream')).body.getReader();
  let text=new TextDecoder().decode((await reader.read()).value);
  window.parent.postMessage({type:'fixture-first'},'*');
  await new Promise(resolve=>{const next=e=>{if(e.data?.fixtureContinue){window.removeEventListener('message',next);resolve()}};window.addEventListener('message',next)});
  for(;;){const chunk=await reader.read();if(chunk.done)break;text+=new TextDecoder().decode(chunk.value)}
  document.getElementById('status').textContent='Stream: '+text;
  window.parent.postMessage({type:'fixture-stream-done',text},'*');
};
requestAnimationFrame(() => window.parent.postMessage({type:'fixture-layout', buttons:
  Object.fromEntries(['blob','data','explicit','route','stream'].map(id=>{const r=document.getElementById(id).getBoundingClientRect();return [id,{x:r.x+r.width/2,y:r.y+r.height/2}]}))},'*'));
'''
_PLUGIN = '''import asyncio
from starlette.responses import FileResponse, StreamingResponse
def export(request):
    return FileResponse(request.app.state.drive_root / 'large.bin', filename='widget-large.bin')
def stream(request):
    async def chunks():
        yield b'first'
        while not (request.app.state.drive_root / 'release-stream').exists():
            await asyncio.sleep(.01)
        yield b'last'
    return StreamingResponse(chunks())
def register(api):
    api.register_route('export', export)
    api.register_route('stream', stream)
    api.register_ui_tab('main', 'Exports', render={'kind':'module','entry':'widget.js'})
'''
_HTML = '''<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="/web/style.css"></head>
<body style="padding:30px"><h1>Skill export check</h1><section data-widget-key="export_widget:main"><div class="widgets-card-status"></div><div id="mount"></div></section>
<script type="module">
import {mountModuleWidget} from '/web/modules/widget_module.js';
window.addEventListener('message',e=>{
 if(e.data?.type==='fixture-layout')window.fixtureButtons=e.data.buttons;
 if(e.data?.type==='fixture-first')window.fixtureFirst=true;
 if(e.data?.type==='fixture-stream-done')window.fixtureStreamText=e.data.text;
});
const nativeFetch=window.fetch.bind(window);
window.fixtureReads=0;
window.fetch=async (...args)=>{
 const response=await nativeFetch(...args);
 if(String(args[0]).endsWith('/stream')){
   const get=response.body.getReader.bind(response.body);
   response.body.getReader=(...options)=>{const reader=get(...options),read=reader.read.bind(reader);reader.read=(...values)=>{window.fixtureReads++;return read(...values)};return reader};
 }
 return response;
};
// A classic mount mints its route lease for the live tab's publication.
const live=(await (await fetch('/api/widgets')).json()).ui_tabs.find(row=>row.skill==='export_widget'&&row.tab_id==='main');
window.disposeWidget=await mountModuleWidget(document.getElementById('mount'),{skill:'export_widget',ws_prefix:'',tab_id:live.tab_id,generation:live.generation},{entry:'widget.js',height:520});
window.fixtureReady=true;
</script></body></html>'''


@pytest.fixture
def widget_server(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles
    import uvicorn
    from ouroboros import extension_loader, server_auth
    from ouroboros.gateway.extensions import api_extension_module, api_extension_dispatch
    from ouroboros.gateway.widget_assets import api_widget_asset_lease
    from ouroboros.gateway.widgets import api_widgets
    from ouroboros.skill_loader import find_skill, save_enabled, save_review_state, SkillReviewState
    from tests._extension_loader_shared import _write_ext_skill, _add_fake_native_dep, _mark_isolated_deps_installed
    from tests._shared import clean_extension_runtime_state

    root, skills = tmp_path / 'drive', tmp_path / 'skills'
    root.mkdir()
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    monkeypatch.setattr('ouroboros.config.get_skills_repo_path', lambda: str(skills))
    skill_dir = _write_ext_skill(skills, 'export_widget', plugin_body=_PLUGIN, permissions=['route','widget'],
                               extra_frontmatter='dependencies:\n  - dummy_pkg\n')
    (skill_dir / 'widget.js').write_text(_WIDGET)
    loaded = find_skill(root, 'export_widget', repo_path=str(skills))
    save_enabled(root, loaded.name, True)
    save_review_state(root, loaded.name, SkillReviewState(status='pass', content_hash=loaded.content_hash))
    loaded = find_skill(root, loaded.name, repo_path=str(skills))
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(root, loaded)
    clean_extension_runtime_state()
    assert extension_loader.load_extension(loaded, lambda: {}, drive_root=root, repo_path=str(skills)) is None
    (root / 'large.bin').write_bytes(b'large export\n' * (128 * 1024))
    expected = {'widget-blob.txt': b'blob export', 'widget-data.txt': b'data export',
                'widget-explicit.txt': b'explicit export', 'widget-large.bin': (root / 'large.bin').read_bytes()}
    app = Starlette(routes=[Route('/', lambda _r: HTMLResponse(_HTML)),
        Route('/api/widgets', api_widgets),
        Route('/api/widgets/asset-leases', api_widget_asset_lease, methods=['POST']),
        Route('/api/widgets/asset-leases/{op}', api_widget_asset_lease, methods=['POST']),
        Route('/api/extensions/{skill}/module/{entry:path}', api_extension_module),
        Route('/api/extensions/{skill}/{rest:path}', api_extension_dispatch, methods=['GET','HEAD']),
        Mount('/web', app=StaticFiles(directory=REPO / 'web'))])
    app.state.drive_root, app.state.repo_dir = root, REPO
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    # The real gate owns the lease controls and the route-lease reads, as in the core.
    server = uvicorn.Server(uvicorn.Config(server_auth.NetworkAuthGate(app), log_level='warning'))
    thread = threading.Thread(target=server.run, kwargs={'sockets':[sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(.01)
    assert server.started
    try:
        yield {'url':f'http://127.0.0.1:{sock.getsockname()[1]}', 'port':sock.getsockname()[1], 'root':root, 'expected':expected}
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        clean_extension_runtime_state()
        assert not thread.is_alive()


def native_file_api(port, read_sizes):
    """Reuse the sibling lane's AST harness for this checkout's real launcher owner."""
    source = REPO / 'launcher.py'
    names = {'_resolve_bridge_file_url','_unique_bridge_target','_fetch_bridge_url_to','MainApi'}
    selected = [node for node in ast.walk(ast.parse(source.read_text()))
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected} == names
    def open_recorded(*args, **kwargs):
        response = urllib.request.urlopen(*args, **kwargs)
        original = response.read
        def read(size=-1):
            assert 0 < size <= 1024 * 1024
            read_sizes.append(size)
            return original(size)
        response.read = read
        return response
    from ouroboros.launcher_background import DesktopApi
    # The names MainApi's class body reads, as the sibling harness supplies them (headless, no facts).
    namespace = {'actual_port':port,'pathlib':__import__('pathlib'),'shutil':shutil,'tempfile':tempfile,
                 'base64':base64,'log':logging.getLogger(__name__),
                 'DesktopApi':DesktopApi,'background':None,'_shell_facts':lambda: {},
                 'urllib':SimpleNamespace(parse=urllib.parse,request=SimpleNamespace(urlopen=open_recorded))}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['MainApi'](), hashlib.sha256(source.read_bytes()).hexdigest()


def _evidence(tmp_path):
    root = Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', str(tmp_path / 'evidence')))
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_browser_widget_exports(widget_server, tmp_path):
    from playwright.sync_api import sync_playwright
    evidence = _evidence(tmp_path)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=os.environ.get('OUROBOROS_TEST_CHROME') or None)
        try:
            page = browser.new_page(viewport={'width':1100,'height':850}, accept_downloads=True)
            page.goto(widget_server['url'])
            frame = page.frame_locator('iframe')
            for control, name in zip(['blob','data','explicit','route'], widget_server['expected']):
                with page.expect_download() as pending:
                    frame.locator('#'+control).click()
                path = evidence / ('browser-' + name)
                pending.value.save_as(path)
                assert path.read_bytes() == widget_server['expected'][name]
            frame.locator('#stream').click()
            page.wait_for_function('window.fixtureFirst === true')
            assert page.evaluate('window.fixtureReads') == 1
            (widget_server['root'] / 'release-stream').touch()
            page.evaluate('document.querySelector("iframe").contentWindow.postMessage({fixtureContinue:true},"*")')
            page.wait_for_function('window.fixtureStreamText === "firstlast"')
            page.screenshot(path=str(evidence / 'widget-browser.png'))
        finally:
            browser.close()


def _native_controller(window, actions, expected, root, evidence, failures):
    """Drive real Qt mouse events into the opaque iframe and verify saved files."""
    from qtpy.QtCore import QPoint, Qt
    from qtpy.QtTest import QTest
    def gui(function):
        done, result = threading.Event(), []
        def execute():
            try:
                result.append(function())
            except BaseException as exc:
                result.append(exc)
            finally:
                done.set()
        actions.call.emit(execute)
        assert done.wait(10), 'GUI action did not complete'
        if isinstance(result[0], BaseException):
            raise result[0]
        return result[0]
    try:
        deadline = time.monotonic()+30
        while time.monotonic()<deadline:
            if window.evaluate_js('Boolean(window.fixtureButtons && window.pywebview?.api?.save_bytes_to_downloads)'):
                break
            time.sleep(.05)
        else:
            raise AssertionError('native widget and bridge did not become ready')
        coords = window.evaluate_js('({buttons:window.fixtureButtons,frame:(()=>{const r=document.querySelector("iframe").getBoundingClientRect();return {x:r.x,y:r.y}})()})')
        for control, name in zip(['blob','data','explicit','route'], expected):
            point = coords['buttons'][control]
            x, y = round(coords['frame']['x']+point['x']+1), round(coords['frame']['y']+point['y']+1)
            gui(lambda: QTest.mouseClick(window.native.webview.focusProxy() or window.native.webview,
                                       Qt.MouseButton.LeftButton, pos=QPoint(x,y)))
            path = Path.home() / 'Downloads' / name
            deadline = time.monotonic()+20
            while time.monotonic()<deadline:
                if path.exists() and path.read_bytes()==expected[name]:
                    break
                time.sleep(.05)
            else:
                raise AssertionError(f'native export did not save {name}')
            shutil.copy2(path,evidence / ('native-'+name))
        point = coords['buttons']['stream']
        x, y = round(coords['frame']['x']+point['x']+1), round(coords['frame']['y']+point['y']+1)
        gui(lambda: QTest.mouseClick(window.native.webview.focusProxy() or window.native.webview,
                                   Qt.MouseButton.LeftButton, pos=QPoint(x,y)))
        deadline = time.monotonic()+20
        while not window.evaluate_js('window.fixtureFirst === true') and time.monotonic()<deadline:
            time.sleep(.02)
        assert window.evaluate_js('window.fixtureFirst === true')
        assert window.evaluate_js('window.fixtureReads') == 1
        (root / 'release-stream').touch()
        window.evaluate_js('document.querySelector("iframe").contentWindow.postMessage({fixtureContinue:true},"*")')
        deadline = time.monotonic()+20
        while window.evaluate_js('window.fixtureStreamText') != 'firstlast' and time.monotonic()<deadline:
            time.sleep(.02)
        assert window.evaluate_js('window.fixtureStreamText') == 'firstlast'
        from qtpy.QtWidgets import QApplication
        gui(lambda: QTest.qWait(100))
        gui(lambda: QApplication.primaryScreen().grabWindow(int(window.native.winId())).save(str(evidence / 'widget-native-screen.png')))
    except BaseException as exc:
        failures.append(exc)
    finally:
        window.destroy()


def test_native_widget_exports(widget_server, tmp_path, monkeypatch):
    # The opt-in comes FIRST: the browser lane installs no desktop extra, and its
    # registered skip is this reason, not a missing optional import. A run that DID
    # select native Qt still stops on absent webview/qtpy below, a skip reason the
    # required lane does not register, so it fails there instead of passing.
    if os.environ.get('PYWEBVIEW_GUI') != 'qt':
        pytest.skip('native Qt probe is explicitly selected by its isolated launcher')
    runtime = tmp_path / 'qt-runtime'
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(runtime))
    webview = pytest.importorskip('webview')
    pytest.importorskip('qtpy')
    from qtpy.QtCore import QObject, Signal, Slot
    class Actions(QObject):
        call = Signal(object)
        @Slot(object)
        def execute(self, function):
            function()
    actions = Actions()
    actions.call.connect(actions.execute)
    evidence = _evidence(tmp_path)
    read_sizes, failures = [], []
    api, source_hash = native_file_api(widget_server['port'], read_sizes)
    window = webview.create_window('Widget export test', widget_server['url'], js_api=api, width=1100,height=850)
    webview.start(_native_controller, (window,actions,widget_server['expected'],widget_server['root'],evidence,failures), gui='qt')
    assert not failures, failures
    assert read_sizes and all(size > 0 for size in read_sizes)
    (evidence / 'native-receipt.json').write_text(json.dumps({'launcher_sha256':source_hash,
        'reads':read_sizes,'files':{name:{'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
                                  for name,data in widget_server['expected'].items()}},indent=2)+'\n')

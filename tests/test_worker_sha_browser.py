"""Worker checkout observations in the real Logs renderer, desktop and phone."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import os

import pytest

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
WEB = Path(__file__).resolve().parents[1] / 'web'
BOOT = '''<script type="module">
import {initDashboard} from '/static/modules/dashboard.js';
import {initLogs} from '/static/modules/logs.js';
document.getElementById('reconnect-overlay').remove();
document.querySelectorAll('[data-nav-page]').forEach(n=>n.classList.toggle('active',n.dataset.navPage==='dashboard'));
const state={activePage:'dashboard',dashboardActiveSubtab:'logs'};
const dashboard=initDashboard({state}); dashboard.page.classList.add('active');
const handlers=new Map();
initLogs({mount:document.getElementById('dashboard-panel-logs'), state,
 ws:{on(type,fn){handlers.set(type,fn);return()=>handlers.delete(type);}}});
dashboard.activateTab('logs');
window.emitWorker=(relation,observed,pid)=>handlers.get('log')({data:{
 type:'worker_sha_verify', ts:`2026-10-04T10:00:00.${pid}Z`, expected_sha:'a'.repeat(40),
 observed_sha:observed.repeat(40), worker_pid:pid, ok:false, relation}});
</script>'''


@pytest.mark.parametrize('width', [1360, 390])
def test_descendant_is_informational_and_problem_stays_warning(width, tmp_path):
    if os.environ.get('OUROBOROS_RUN_UI_SMOKE') != '1':
        pytest.skip('Set OUROBOROS_RUN_UI_SMOKE=1')
    playwright = pytest.importorskip('playwright.sync_api')
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args): pass
        def do_GET(self):
            if self.path == '/fixture':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                html = (WEB / 'index.html').read_text(encoding='utf-8').replace(
                    '<script type="module" src="/static/app.js"></script>', BOOT)
                self.wfile.write(html.encode('utf-8'))
            else:
                self.path = self.path.removeprefix('/static')
                super().do_GET()
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={'width': width, 'height': 900}, has_touch=width < 980)
                errors = []
                page.on('pageerror', lambda exc: errors.append(str(exc)))
                page.route('**/api/**', lambda route: route.fulfill(json={'entries': []}))
                page.goto(f'http://127.0.0.1:{server.server_port}/fixture')
                page.wait_for_function('typeof window.emitWorker === "function"')
                page.evaluate("emitWorker('descendant', 'b', 101)")
                page.evaluate("emitWorker('non_descendant', 'c', 102)")
                page.evaluate("emitWorker('unavailable', 'd', 103)")
                descendant = page.locator('.log-entry').filter(has_text='Worker checkout descends from baseline')
                assert descendant.locator('.log-phase').inner_text() == 'info'
                problem = page.locator('.log-entry').filter(has_text='Worker checkout is not a descendant of baseline')
                assert problem.locator('.log-phase').inner_text() == 'warn'
                assert page.locator('.log-entry').filter(has_text='comparison unavailable').locator('.log-phase').inner_text() == 'warn'
                descendant.get_by_role('button', name='Raw', exact=True).click()
                assert 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' in descendant.locator('.log-raw').inner_text()
                assert 'Worker SHA mismatch' not in page.locator('#page-logs').inner_text()
                assert not errors, errors
                page.screenshot(path=str(tmp_path / f'worker-sha-{width}.png'), full_page=True)
                print(f'WORKER_SHA_SCREENSHOT={tmp_path / f"worker-sha-{width}.png"}')
            finally:
                browser.close()
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)

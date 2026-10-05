"""The Updates consumer must not offer Restart while local-work recovery holds it."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from threading import Thread

import pytest
from tests.test_update_letter_browser import BOOT, WEB

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.mark.parametrize('width', [1360, 390])
def test_pending_restore_explains_real_server_restart(width, tmp_path):
    if os.environ.get('OUROBOROS_RUN_UI_SMOKE') != '1':
        pytest.skip('Set OUROBOROS_RUN_UI_SMOKE=1')
    playwright = pytest.importorskip('playwright.sync_api')
    status = {'managed': True, 'check_ok': True, 'available': False,
              'current_version': '7.5.1', 'current_sha': 'a' * 40, 'warnings': [],
              'update_tx': {'active': True, 'phase': 'pending_boot_smoke', 'local_work_recovery': True}}
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
    requests = []
    def respond(route):
        requests.append((route.request.method, route.request.url))
        if '/api/update/' in route.request.url:
            body = status
        else:
            body = {'commits': [], 'tags': []}
        route.fulfill(content_type='application/json', body=json.dumps(body))
    try:
        with playwright.sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={'width': width, 'height': 900}, has_touch=width < 980)
                page.route('**/api/**', respond)
                page.goto(f'http://127.0.0.1:{server.server_port}/fixture')
                page.get_by_text('Local changes still need recovery confirmation.', exact=True).wait_for()
                assert page.locator('#btn-update-primary').inner_text() == 'Check again'
                assert 'restart the server process' in page.locator('#dashboard-panel-updates').inner_text()
                page.locator('#btn-update-primary').click()
                page.wait_for_function("document.querySelector('#btn-update-primary').textContent.trim() === 'Check again'")
                assert not any('/api/restart' in url or '/api/update/apply' in url for _method, url in requests)
                image = tmp_path / f'update-recovery-{width}.png'
                page.screenshot(path=str(image), full_page=True)
                print(f'UPDATE_RECOVERY_SCREENSHOT={image}')
            finally:
                browser.close()
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)

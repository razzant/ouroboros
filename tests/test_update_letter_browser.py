"""Real Dashboard/Updates UI with synthetic API replies; no runtime or model starts.

Check is the fetching check: it keeps the shown text and the install action, and a retained old
failure names its own range. Writing a description has its own request
(tests/test_update_description_browser.py)."""
import copy
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import pytest
from ouroboros.test_evidence import diagnostic_directory

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
WEB = Path(__file__).resolve().parents[1] / "web"
BOOT = """<script type="module">
import {initDashboard} from '/static/modules/dashboard.js';
import {initUpdates} from '/static/modules/updates.js';
document.querySelectorAll('[data-nav-page]').forEach(node=>node.classList.toggle('active',node.dataset.navPage==='dashboard'));
const state={activePage:'dashboard',dashboardActiveSubtab:'updates'};
const dashboard=initDashboard({state}); dashboard.page.classList.add('active');
const handlers=new Map();
const ws={on(type,fn){handlers.set(type,fn);return()=>handlers.delete(type);},isConnected:()=>true};
initUpdates({mount:document.getElementById('dashboard-panel-updates'),state,ws});
dashboard.activateTab('updates');
</script>"""


@pytest.mark.parametrize(("engine", "width"), [("chromium", 1360), ("webkit", 390)])
def test_check_keeps_text_and_install_action_then_shows_exact_failed_range(engine, width, tmp_path):
    output = diagnostic_directory(tmp_path, "update-letter")
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    status = {
        "managed": True, "check_ok": True, "available": True, "safe_to_apply": True,
        "current_version": "7.5.1", "latest_version": "7.5.1", "current_sha": "b" * 40,
        "current_short_sha": "bbbbbbbb", "latest_short_sha": "cccccccc",
        "running_sha": "b" * 40, "latest_sha": "c" * 40, "warnings": [], "update_tx": {"active": False},
        "letter": {"state": "ready", "relation": "superseded", "description_current": False, "text": "Предыдущее **описание** обновления.",
                   "author_version": "7.5.1", "target_version": "7.5.1", "written_at": "2026-10-03T13:00:00Z",
                   "key": {"base_sha": "a" * 40, "target_sha": "c" * 40}, "has_last_good": False},
    }
    held = []

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == "/fixture":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                html = (WEB / "index.html").read_text(encoding="utf-8").replace(
                    '<script type="module" src="/static/app.js"></script>', BOOT)
                self.wfile.write(html.encode("utf-8"))
            else:
                self.path = self.path.removeprefix("/static")
                super().do_GET()

    def respond(route):
        path = urlparse(route.request.url).path
        if path == "/api/update/check":
            assert route.request.method == "POST"
            held.append(route)
            return
        body = status if path == "/api/update/status" else {"commits": [], "tags": []}
        route.fulfill(content_type="application/json", body=json.dumps(body))

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width < 980)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.route("**/api/**", respond)
                page.goto(f"http://127.0.0.1:{server.server_port}/fixture")
                page.wait_for_selector("#updates-letter:not([hidden])")
                assert page.locator("#updates-letter-note").inner_text() == "This description covers an earlier update (aaaaaaaa → cccccccc)."
                assert page.locator("#updates-letter-write").inner_text() == "Write description"
                page.evaluate("window.savedBody=document.querySelector('#updates-letter-body').firstChild")
                primary = page.locator("#btn-update-primary").inner_text()
                page.locator("#updates-letter-refresh").click()
                page.wait_for_function("document.querySelector('#updates-letter-refresh').disabled")
                assert page.locator("#updates-letter-refresh-label").inner_text() == "Checking…"
                assert page.locator("#btn-update-primary").inner_text() == primary
                assert page.evaluate("window.savedBody===document.querySelector('#updates-letter-body').firstChild")
                assert len(held) == 1
                failure = copy.deepcopy(status)
                failure["letter"].update(state="failed", relation="applied", has_last_good=True,
                    failed_at="2026-10-03T13:07:00Z", error_text="Configured model capacity unavailable.",
                    latest_failed_key={"base_sha": "b" * 40, "target_sha": "c" * 40},
                    key={"base_sha": "a" * 40, "target_sha": "b" * 40})
                held.pop().fulfill(content_type="application/json", body=json.dumps(failure))
                page.wait_for_function("!document.querySelector('#updates-letter-refresh').disabled")
                assert "Refresh failed on" in page.locator("#updates-letter-note").inner_text()
                assert page.evaluate("window.savedBody===document.querySelector('#updates-letter-body').firstChild")
                page.locator("#updates-letter-details summary").click()
                assert "aaaaaaaa → bbbbbbbb" in page.locator("#updates-letter-provenance").inner_text()
                assert "bbbbbbbb → cccccccc" in page.locator("#updates-letter-error").inner_text()
                page.screenshot(path=str(output / f"update-letter-failed-{engine}-{width}.png"), full_page=True)
                assert "subscription limits, local resources, or API budget" in page.locator("#updates-letter-resources").inner_text()
                assert page.evaluate("document.documentElement.scrollWidth<=document.documentElement.clientWidth")
                page.locator("#updates-letter-refresh").click()
                page.wait_for_function("document.querySelector('#updates-letter-refresh').disabled")
                fresh = copy.deepcopy(status)
                fresh["letter"].update(relation="pending", description_current=True, key={"base_sha": "b" * 40, "target_sha": "c" * 40},
                                       text="Полное описание изменений текущего обновления.")
                held.pop().fulfill(content_type="application/json", body=json.dumps(fresh))
                page.wait_for_function("!document.querySelector('#updates-letter-refresh').disabled")
                assert page.locator("#updates-letter-note").is_hidden()
                assert "Полное описание" in page.locator("#updates-letter-body").inner_text()
                page.locator("#updates-letter-refresh").click()
                page.wait_for_function("document.querySelector('#updates-letter-refresh').disabled")
                held.pop().fulfill(content_type="application/json", body=json.dumps(fresh))
                page.wait_for_selector(".toast")
                assert page.locator(".toast").inner_text() == "Description is up to date."
                page.screenshot(path=str(output / f"update-letter-{engine}-{width}.png"), full_page=True)
                print(f"visual evidence: {output / f'update-letter-{engine}-{width}.png'}")
                assert errors == []
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

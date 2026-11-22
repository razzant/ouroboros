"""Real Dashboard/Updates UI with synthetic API replies; no runtime or model starts.

A retained old failure names its own range and keeps Rewrite of the shown text's own range beside the update
on offer. Check and describe keeps the shown text and the install action while its one request runs; a failed
check is said by the headline, whose action then is the retry, and stays said after a status read the socket
queued, until a later check. The request's lifecycle and the writer are
tests/test_update_description_browser.py's."""
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

pytestmark = [pytest.mark.ui_browser]
WEB = Path(__file__).resolve().parents[1] / "web"
BOOT = """<script type="module">
import {initDashboard} from '/static/modules/dashboard.js';
import {initUpdates} from '/static/modules/updates.js';
document.querySelectorAll('[data-nav-page]').forEach(node=>node.classList.toggle('active',node.dataset.navPage==='dashboard'));
const state={activePage:'dashboard',dashboardActiveSubtab:'updates'};
const dashboard=initDashboard({state}); dashboard.page.classList.add('active');
const handlers=new Map();
window.__wsHandlers=handlers;
const ws={on(type,fn){handlers.set(type,fn);return()=>handlers.delete(type);},isConnected:()=>true};
initUpdates({mount:document.getElementById('dashboard-panel-updates'),state,ws});
dashboard.activateTab('updates');
</script>"""


@pytest.mark.parametrize(("engine", "width"), [("chromium", 1044), ("webkit", 390)])
def test_check_and_describe_keeps_text_and_install_action_then_shows_exact_failed_range(engine, width, tmp_path):
    output = diagnostic_directory(tmp_path, "update-letter", publish=True)
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    # The update b..c is on offer; the record keeps an old failed rewrite of it beside the last good text of a..b.
    status = {
        "managed": True, "check_ok": True, "available": True, "safe_to_apply": True,
        "current_version": "7.5.1", "latest_version": "7.5.1", "current_sha": "b" * 40,
        "current_short_sha": "bbbbbbbb", "latest_short_sha": "cccccccc",
        "running_sha": "b" * 40, "latest_sha": "c" * 40, "warnings": [], "update_tx": {"active": False},
        "letter": {"state": "failed", "relation": "applied", "description_current": False, "text": "Предыдущее **описание** обновления.",
                   "author_version": "7.5.1", "target_version": "7.5.1", "written_at": "2026-10-03T13:00:00Z",
                   "key": {"base_sha": "a" * 40, "target_sha": "b" * 40}, "has_last_good": True,
                   "failed_at": "2026-10-03T13:07:00Z", "error_text": "Configured model capacity unavailable.",
                   "latest_failed_key": {"base_sha": "b" * 40, "target_sha": "c" * 40}},
    }
    current = {"status": status}
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
        if path == "/api/update/description":
            assert route.request.method == "POST" and json.loads(route.request.post_data) == {"mode": "latest"}
            held.append(route)
            return
        body = current["status"] if path == "/api/update/status" else {"commits": [], "tags": []}
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

                def wait_held():  # the press reached the server: its one request is held here
                    for _ in range(100):
                        if held:
                            assert len(held) == 1
                            return
                        page.wait_for_timeout(50)
                    raise AssertionError("the press sent no request")

                page.goto(f"http://127.0.0.1:{server.server_port}/fixture")
                page.wait_for_selector("#updates-letter:not([hidden])")
                # The retained old failure names its own range; Rewrite keeps the shown text's own earlier range.
                assert "Refresh failed on" in page.locator("#updates-letter-note").inner_text()
                page.locator("#updates-letter-details summary").click()
                assert "aaaaaaaa → bbbbbbbb" in page.locator("#updates-letter-provenance").inner_text()
                assert "bbbbbbbb → cccccccc" in page.locator("#updates-letter-error").inner_text()
                rewrite = page.locator("#updates-letter-rewrite")
                assert rewrite.is_visible() and rewrite.inner_text() == "Rewrite"
                assert "subscription limits, local resources, or API budget" in page.locator("#updates-letter-resources").inner_text()
                page.screenshot(path=str(output / f"update-letter-failed-{engine}-{width}.png"), full_page=True)
                page.locator("#updates-letter-details summary").click()
                action = page.locator("#updates-letter-action")
                assert action.inner_text() == "Check and describe"
                page.evaluate("window.savedBody=document.querySelector('#updates-letter-body').firstChild")
                primary = page.locator("#btn-update-primary")
                headline = primary.inner_text()
                action.click()
                page.wait_for_function("document.querySelector('#updates-letter-action').disabled")
                assert action.inner_text() == "Checking…"
                assert primary.inner_text() == headline
                assert page.evaluate("window.savedBody===document.querySelector('#updates-letter-body').firstChild")
                wait_held()
                # A failed POST is an attempt fact: the successful status and old text stay visible.
                failure = copy.deepcopy(status)
                failure.update(check_ok=False, available=False, latest_sha="", warnings=["fetch_error:offline"])
                held.pop().fulfill(content_type="application/json", body=json.dumps(
                    {"status": "check_failed", "update": failure, "checked_key": None}))
                page.wait_for_function("document.querySelector('#updates-hint').textContent === 'The last check failed: fetch_error:offline'")
                assert primary.inner_text() == "Update to 7.5.1"
                assert page.locator("#updates-summary").inner_text() == "Update available: 7.5.1"
                assert action.is_visible() and action.inner_text() == "Check and describe"
                assert page.evaluate("window.savedBody===document.querySelector('#updates-letter-body').firstChild")
                # A later WS read preserves the same status, old text and failed-attempt hint.
                with page.expect_response("**/api/update/status") as refreshed:
                    page.evaluate("window.__wsHandlers.get('update_progress_changed')()")
                refreshed.value.finished()
                page.wait_for_function("document.querySelector('#btn-update-primary').textContent === 'Update to 7.5.1'")
                assert page.locator("#updates-hint").inner_text() == "The last check failed: fetch_error:offline"
                assert action.inner_text() == "Check and describe"
                assert page.evaluate("window.savedBody===document.querySelector('#updates-letter-body').firstChild")
                # A text of the range found was saved meanwhile: the next press reuses it, and its check clears the failure.
                fresh = copy.deepcopy(status)
                fresh["letter"].update(state="ready", relation="pending", description_current=True, has_last_good=False,
                                       key={"base_sha": "b" * 40, "target_sha": "c" * 40}, latest_failed_key=None,
                                       error_text="", text="Полное описание изменений текущего обновления.")
                current["status"] = fresh
                action.click()
                wait_held()
                held.pop().fulfill(content_type="application/json", body=json.dumps(
                    {"status": "current", "update": fresh, "key": fresh["letter"]["key"], "checked_key": fresh["letter"]["key"]}))
                page.wait_for_selector(".toast")
                assert page.locator(".toast").inner_text() == "Description is up to date."
                assert page.locator("#updates-hint").is_hidden()
                assert page.locator("#updates-letter-note").is_hidden()
                assert "Полное описание" in page.locator("#updates-letter-body").inner_text()
                assert page.evaluate("document.documentElement.scrollWidth<=document.documentElement.clientWidth")
                page.screenshot(path=str(output / f"update-letter-{engine}-{width}.png"), full_page=True)
                print(f"visual evidence: {output / f'update-letter-{engine}-{width}.png'}")
                assert errors == []
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

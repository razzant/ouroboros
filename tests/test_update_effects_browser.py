"""Real Updates controls on desktop/mobile engines, backed only by synthetic HTTP replies."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import pytest

from ouroboros.test_evidence import diagnostic_directory
from tests.test_update_letter_browser import BOOT, WEB
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.mark.parametrize("engine,width", [("chromium", 1360), ("chromium", 390), ("webkit", 390)])
def test_apply_discloses_before_post_and_applied_card_restores_locally(engine, width, tmp_path):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    base, target = "a" * 40, "b" * 40
    status = {"managed": True, "check_ok": True, "available": True, "safe_to_apply": True,
              "current_sha": base, "latest_sha": target, "warnings": [], "update_tx": {"active": False}}
    plan = {"base_sha": base, "target_sha": target, "available": True,
            "kind": "clean", "auto_mergeable": True, "local_dirty_count": 0}
    pin = {"action": "replace", "direction": "forward", "serving": {"version": "4.5.0"},
           "target": {"version": "4.6.0"}}
    effects = {"target_sha": target, "engine": pin, "replacement_engine": pin,
               "active_work": {"status": "unknown", "total": None},
               "runtime": {"installed": {"python": "3.10.19", "sqlite": "3.50.4"},
                           "bundle": {"python": "3.10.22", "sqlite": "3.53.1"}}}
    calls = []

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
        calls.append((path, route.request.post_data))
        body = {"commits": [], "tags": []}
        if path == "/api/update/status":
            body = status
        elif path == "/api/update/preflight":
            body = {"merge_plan": plan, "effects": effects}
        elif path in {"/api/update/apply", "/api/git/rollback"}:
            body = {"status": "ok", "restarting": True, "message": "Nothing was pushed."}
        route.fulfill(content_type="application/json", body=json.dumps(body))

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            if engine == "webkit" and not Path(pw.webkit.executable_path).is_file():
                pytest.skip(f"{engine} is not installed in the test interpreter")
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width < 980)
                errors = PageErrors(page, engine)
                page.route("**/api/**", respond)
                url = f"http://127.0.0.1:{server.server_port}/fixture"
                for button, strategy in [("#btn-update-primary", "auto_merge"), ("#btn-update-replace", "replace")]:
                    page.goto(url)
                    if strategy == "replace":
                        page.locator(".updates-recovery summary").click()
                    page.locator(button).click()
                    dialog = page.get_by_role("dialog")
                    dialog.wait_for()
                    text = dialog.inner_text()
                    for fact in ["4.5.0 → 4.6.0", "end all its active runs", "work: unknown",
                                 "forward-only", "3.10.19", "3.50.4", "3.10.22", "3.53.1"]:
                        assert fact in text
                    before = len([path for path, _body in calls if path == "/api/update/apply"])
                    dialog.get_by_role("button", name="Cancel", exact=True).click()
                    assert len([path for path, _body in calls if path == "/api/update/apply"]) == before
                    page.locator(button).click()
                    dialog.wait_for()
                    output = diagnostic_directory(tmp_path, "update-effects") / f"{strategy}-{engine}-{width}.png"
                    page.screenshot(path=str(output), full_page=True)
                    print(f"UPDATE_EFFECTS_SCREENSHOT={output}")
                    dialog.locator("[data-confirm-ok]").click()
                    page.wait_for_function("document.querySelector('#btn-update-primary').textContent.includes('Restart')")
                    posted = [json.loads(body) for path, body in calls if path == "/api/update/apply"]
                    assert len(posted) == before + 1 and posted[-1]["strategy"] == strategy
                    assert posted[-1]["expected_base_sha"] == base and posted[-1]["expected_target_sha"] == target
                status.update(available=False, current_sha=target, update_progress={
                    "operation_id": "applied", "active": False, "outcome": "applied", "base_sha": base})
                page.goto(url)
                page.locator("#updates-attempt-restore").click()
                dialog = page.get_by_role("dialog")
                assert "Nothing will be pushed" in dialog.inner_text()
                assert not any(path == "/api/git/rollback" for path, _body in calls)
                dialog.locator("[data-confirm-ok]").click()
                page.wait_for_function("document.querySelector('#btn-update-primary').disabled")
                assert [json.loads(body) for path, body in calls if path == "/api/git/rollback"] == [{"target": base}]
                assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                assert not errors.unexpected(), errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

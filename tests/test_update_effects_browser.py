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
CANDIDATE_ID = "update_bbbbbbbbbbbb_000001"


class Handler(SimpleHTTPRequestHandler):
    """The real Updates panel at ``/fixture`` and the real modules under ``/static``; every API reply is routed."""

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


@pytest.mark.parametrize("engine,width", [("chromium", 1360), ("chromium", 390), ("webkit", 390)])
def test_restart_to_update_discloses_its_candidate_effects_before_post(engine, width, tmp_path):
    """Restart to update first reads the effects of the candidate it shows; a read refused for another one sends
    nothing, an unreadable one stays unknown and keeps the action."""
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    base, target = "a" * 40, "b" * 40
    candidate = {"candidate_id": CANDIDATE_ID, "state": "ready", "target_sha": target, "target_version": "7.9.1",
                 "base_sha": base, "base_version": "7.9.0", "checks": {"ok": True, "tests": {"file_count": 3}},
                 "actions": ["activate", "discard_candidate"]}
    status = {"managed": True, "check_ok": True, "available": True, "safe_to_apply": True, "current_sha": base,
              "latest_sha": target, "latest_version": "7.9.1", "warnings": [], "update_tx": {"active": False},
              "candidate": candidate}
    # Its composition pins 4.6.0 while the official target keeps 4.5.0, and its new app brings another Python.
    effects = {"candidate_id": CANDIDATE_ID, "target_sha": target, "provisional": False,
               "engine": {"action": "replace", "direction": "forward", "serving": {"version": "4.5.0"},
                          "target": {"version": "4.6.0"}},
               "replacement_engine": {"action": "keep", "direction": "unchanged", "serving": {"version": "4.5.0"},
                                      "target": {"version": "4.5.0"}},
               "active_work": {"status": "known", "model": 2, "non_model": 1, "total": 3},
               "runtime": {"installed": {"python": "3.10.19", "sqlite": "3.50.4"},
                           "selected": {"python": "3.10.22", "sqlite": "3.53.1"}, "changed_by_apply": True,
                           "bundle": {"python": "3.10.22", "sqlite": "3.53.1"}}}
    preflight = {"status": 200, "body": {"merge_plan": {"base_sha": base, "target_sha": target}, "effects": effects}}
    calls = []

    def respond(route):
        path = urlparse(route.request.url).path
        calls.append((path, route.request.post_data))
        reply, code = {"commits": [], "tags": []}, 200
        if path == "/api/update/status":
            reply = status
        elif path == "/api/update/preflight":
            reply, code = preflight["body"], preflight["status"]
        elif path == "/api/update/apply":
            reply = {"status": "ok", "restarting": True, "strategy": "activate",
                     "candidate": {**candidate, "state": "activating"}}
        route.fulfill(status=code, content_type="application/json", body=json.dumps(reply))

    def sent(name):
        return [json.loads(body or "{}") for path, body in calls if path == name]

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
                primary, dialog = page.locator("#btn-update-primary"), page.get_by_role("dialog")
                ready = "document.querySelector('#btn-update-primary').textContent === 'Restart to update'"
                page.goto(url)
                page.wait_for_function(ready)
                primary.click()
                dialog.wait_for()
                text = dialog.inner_text()
                output = diagnostic_directory(tmp_path, "update-effects") / f"activate-{engine}-{width}.png"
                page.screenshot(path=str(output), full_page=True)
                output.with_suffix(".txt").write_text(text, encoding="utf-8")  # the words beside the picture
                print(f"UPDATE_EFFECTS_SCREENSHOT={output}")
                assert sent("/api/update/preflight") == [{"candidate_id": CANDIDATE_ID}], calls
                for fact in ["4.5.0 → 4.6.0", "end all its active runs", "2 model call(s), 1 other job(s)",
                             "forward-only", "Python 3.10.19, SQLite 3.50.4", "runs Python 3.10.22, SQLite 3.53.1"]:
                    assert fact in text, (fact, text)
                assert "keeps the interpreter" not in text, text
                dialog.get_by_role("button", name="Cancel", exact=True).click()
                assert sent("/api/update/apply") == []
                primary.click()
                dialog.wait_for()
                dialog.locator("[data-confirm-ok]").click()
                page.wait_for_function("document.querySelector('#btn-update-primary').textContent === 'Restarting…'")
                assert sent("/api/update/apply") == [{"strategy": "activate", "candidate_id": CANDIDATE_ID}]
                # The server named another live candidate: no dialog, nothing sent, the panel reads again.
                preflight.update(status=409, body={"error": "the prepared update changed; check again",
                                                   "reason_code": "candidate_changed"})
                page.goto(url)
                page.wait_for_function(ready)
                primary.click()
                page.locator("#toast-stack").get_by_text("The prepared update changed").wait_for()
                assert dialog.count() == 0 and len(sent("/api/update/apply")) == 1
                # An unreadable observation is unknown, never zero, and the owner can still restart.
                preflight.update(status=500, body={"error": "engine status unreadable"})
                page.goto(url)
                page.wait_for_function(ready)
                primary.click()
                dialog.wait_for()
                text = dialog.inner_text()
                page.screenshot(path=str(output.with_name(f"activate-unknown-{engine}-{width}.png")), full_page=True)
                assert "Currently active engine work: unknown" in text and "could not be confirmed" in text, text
                assert "0 model call" not in text and "keeps the interpreter" not in text, text
                dialog.locator("[data-confirm-ok]").click()
                page.wait_for_function("document.querySelector('#btn-update-primary').textContent === 'Restarting…'")
                assert len(sent("/api/update/apply")) == 2
                assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                assert not errors.unexpected(), errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

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
from supervisor import update_prepare
from tests.test_update_activation_endpoint import _answered, _apply, _undecided_work_root
from tests.test_update_activation_endpoint import scene as _scene
from tests.test_update_letter_browser import BOOT, WEB
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
scene = _scene  # the real Update over a miniature body (``test_update_activation_endpoint``)
CANDIDATE_ID = "update_bbbbbbbbbbbb_000001"
# The real Main Update indicator beside the panel, reading the same routed status (``update_status.initUpdateStatus``).
INDICATOR = "import('/static/modules/update_status.js').then((m) => m.initUpdateStatus({ws}));\n"
ON_SCREEN = ("(node) => { const box = node.getBoundingClientRect(); return box.width > 0 && box.left >= 0 "
             "&& box.right <= innerWidth && box.top >= 0 && box.bottom <= innerHeight; }")


class Handler(SimpleHTTPRequestHandler):
    """The real Updates panel at ``/fixture`` and the real modules under ``/static``; every API reply is routed."""

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path in ("/fixture", "/fixture?indicator"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            boot = BOOT if self.path == "/fixture" else BOOT.replace("</script>", INDICATOR + "</script>")
            html = (WEB / "index.html").read_text(encoding="utf-8").replace(
                '<script type="module" src="/static/app.js"></script>', boot)
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


@pytest.mark.parametrize("engine,width", [("chromium", 1360), ("chromium", 390), ("webkit", 390)])
def test_a_refused_carried_list_is_said_never_saved_and_restore_keeps_its_commit_in_view(engine, width, tmp_path):
    """The complete list downloads only while the server serves it for the candidate shown: its refusals (409 another
    candidate is live, 404 no list is kept) are said, nothing is saved and the status is read again, the reader keeping
    their place; a served list is saved whole. Restore names its exact commit inside the dialog at every width, and
    Cancel sends nothing."""
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    base, target = "88acfdbe3c1f0a2b4d5e6f708192a3b4c5d6e7f8", "d2f6a91c7e5b4a3928170f6e5d4c3b2a19087f6e"
    later = CANDIDATE_ID[:-1] + "2"  # a later preparation that superseded the one shown
    url, name = f"/api/files/download?carried_list={later}", f"{later}.uncommitted.txt"
    notes = [["A", f"notes/n{index:03d}.md"] for index in range(250)]
    listing = "Staged: 0\n\nIn the files: 250\n" + "".join(f"{status}\t{path}\n" for status, path in notes)
    candidate = {"candidate_id": CANDIDATE_ID, "state": "ready", "target_sha": target, "target_version": "7.9.1",
                 "base_sha": base, "base_version": "7.9.0", "local_changes": True,
                 "carried": {"index": {"count": 0, "paths": []}, "files": {"count": 250, "paths": notes[:200]},
                             "list": f"/o/{CANDIDATE_ID}.uncommitted.txt",
                             "list_url": f"/api/files/download?carried_list={CANDIDATE_ID}"},
                 "checks": {"ok": True, "tests": {"file_count": 3}}, "actions": ["activate", "discard_candidate"]}
    superseding = {**candidate, "candidate_id": later, "carried": {
        "index": {"count": 0, "paths": []}, "files": {"count": 230, "paths": notes[:200]}, "list": f"/o/{name}",
        "list_url": url}}
    status = {"managed": True, "check_ok": True, "available": True, "safe_to_apply": True, "current_sha": base,
              "latest_sha": target, "latest_version": "7.9.1", "warnings": [], "update_tx": {"active": False},
              "candidate": candidate}
    refusals = {409: {"error": "the prepared update changed; check again", "reason_code": "candidate_changed"},
                404: {"error": "this update keeps no list of carried work", "reason_code": "list_unavailable"}}
    answer = {"status": 409}
    calls = []

    def respond(route):
        parts = urlparse(route.request.url)
        calls.append((route.request.method, parts.path))
        if parts.path == "/api/files/download":
            if answer["status"] == 200:
                route.fulfill(status=200, content_type="text/plain; charset=utf-8", body=listing,
                              headers={"Content-Disposition": f'attachment; filename="{name}"'})
            else:
                if answer["status"] == 409:
                    status["candidate"] = superseding
                route.fulfill(status=answer["status"], content_type="application/json",
                              body=json.dumps(refusals[answer["status"]]))
            return
        body = status if parts.path == "/api/update/status" else {"commits": [], "tags": []}
        route.fulfill(content_type="application/json", body=json.dumps(body))

    def asked(path, method=None):
        return [seen for seen in calls if seen[1] == path and (method is None or seen[0] == method)]

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            if engine == "webkit" and not Path(pw.webkit.executable_path).is_file():
                pytest.skip(f"{engine} is not installed in the test interpreter")
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width < 980,
                                        accept_downloads=True)
                errors = PageErrors(page, engine)
                page.route("**/api/**", respond)
                fixture = f"http://127.0.0.1:{server.server_port}/fixture"
                page.goto(fixture)
                page.wait_for_function("document.querySelector('#btn-update-primary').textContent === 'Restart to update'")
                page.locator("#updates-candidate-details > summary").click()
                link = page.locator("#updates-candidate-facts a[download]")
                link.scroll_into_view_if_needed()
                place = "document.querySelector('.updates-scroll').scrollTop"
                for code, words in ((409, "The prepared update changed; its list was not downloaded. Review it again."),
                                    (404, "This update no longer keeps a list of its carried work; nothing was downloaded.")):
                    answer["status"] = code
                    reads, before = len(asked("/api/update/status")), page.evaluate(place)
                    with pytest.raises(playwright.TimeoutError):
                        with page.expect_download(timeout=2_000):
                            link.click()
                    page.locator("#toast-stack").get_by_text(words).wait_for()
                    for _ in range(50):  # the read again follows the words; routes are answered while the page waits
                        if len(asked("/api/update/status")) > reads:
                            break
                        page.wait_for_timeout(100)
                    assert len(asked("/api/update/status")) > reads, "the status is read again after a refusal"
                    assert asked("/api/files/download", "GET") == [], "a refused list is never fetched as the list"
                    page.locator("#updates-candidate-facts").get_by_text("230 change(s): 0 staged").wait_for()  # the live one
                    assert before > 0 and abs(page.evaluate(place) - before) <= 1, "the reader keeps their place"
                answer["status"] = 200
                with page.expect_download() as saved:
                    link.click()
                assert saved.value.suggested_filename == name
                assert Path(saved.value.path()).read_text(encoding="utf-8") == listing
                assert len(asked("/api/files/download", "GET")) == 1
                # The last Restart to update's outcome names the commit Restore goes back to, whole.
                status["candidate"] = {"state": "none", "actions": ["prepare"], "last_activation": {
                    "candidate_id": CANDIDATE_ID, "at": "2026-10-31T09:05:00+00:00", "outcome": "adopted",
                    "sha": target, "previous": base, "target_sha": target, "target_version": "7.9.1",
                    "base_version": "7.9.0", "merge_kind": "fast_forward", "local_commits": 0, "local_changes": False,
                    "without_local_changes": False}}
                page.goto(fixture)
                page.locator("#updates-attempt-restore").click()
                dialog = page.get_by_role("dialog")
                dialog.wait_for()
                assert f"commit {base}," in dialog.inner_text()
                output = diagnostic_directory(tmp_path, "update-effects") / f"restore-{engine}-{width}.png"
                page.screenshot(path=str(output))
                print(f"UPDATE_EFFECTS_SCREENSHOT={output}")
                inside = dialog.locator(".marketplace-modal-body").evaluate(
                    "(body) => [body.scrollWidth, body.clientWidth]")
                assert inside[0] <= inside[1], f"the commit runs past the dialog: {inside}"
                dialog.get_by_role("button", name="Cancel", exact=True).click()
                assert asked("/api/git/rollback") == []
                assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                assert not errors.unexpected(), errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


_DRAIN, _DNS, _SIG = "tests/test_queue.py::test_drain", "tests/test_net.py::test_dns", "assert: AssertionError: expected 2"
_LANES = [{"lane": "parallel", "returncode": 1, "cases": 300, "seconds": 1790.2},
          {"lane": "serial", "returncode": 124, "cases": 0, "seconds": 0.0}]
# Each record is the shape supervisor/update_checks.selected_tests writes for that outcome (reason, then checks.tests),
# with the panel's expected headline and a phrase of its hint.
CHECK_SCENES = {
    "passed": ("ready", "", {"ok": True, "tier": "selected", "file_count": 14, "blocking": [], "already_failing": []},
               "7.9.1 (bbbbbbbb) is ready. Restart to switch to it.", "If it does not start"),
    "failed": ("failed", "checks_failed", {"ok": False, "tier": "selected", "file_count": 14, "failed": {_DRAIN: _SIG},
                                           "blocking": [_DRAIN], "already_failing": [], "target_failing": [],
                                           "baseline": {"complete": True, "failed": {}, "returncode": 0, "excuses": True}},
               "7.9.1 (bbbbbbbb) did not pass its checks. Nothing was changed.", "fail only with the update"),
    "prerequisite": ("failed", "checks_unverified", {
        "ok": False, "tier": "full", "file_count": 412, "budget_seconds": 1800, "failed": {}, "blocking": [],
        "unverified": "a prerequisite of the checks is missing: Node.js for the full tier's 37 browser-module test "
                      "file(s) is absent: none is bundled and none is on PATH",
        "prerequisite": {"prerequisite": "node", "status": "absent"}},
        "7.9.1 (bbbbbbbb) could not be checked. Nothing was changed.", "until that changes"),
    "unfinished": ("failed", "checks_unverified", {
        "ok": False, "tier": "selected", "file_count": 40, "budget_seconds": 1800, "failed": {_DRAIN: _SIG},
        "blocking": [_DRAIN], "new": {"complete": False, "returncode": 1, "lanes": _LANES},
        "unverified": "the selected run did not complete (parallel: exit 1, 300 case(s), serial: exit 124, 0 case(s)); "
                      "124 is its deadline, 125 a refused start, 2 an interrupted collection, 5 nothing collected"},
        "7.9.1 (bbbbbbbb) did not pass its checks. Nothing was changed.", "reached their 30-minute limit"),
    "not_excused": ("failed", "checks_failed", {
        "ok": False, "tier": "selected", "file_count": 14, "failed": {_DRAIN: _SIG, _DNS: _SIG},
        "blocking": [_DNS, _DRAIN], "already_failing": [], "target_failing": [],
        "user_site": {"declared": "/u", "serving": "/u", "harness": {"W'": "opt_in", "W": "predates_opt_in",
                                                                     "T": "opt_in"}},
        "baseline": {"complete": True, "failed": {_DNS: _SIG}, "returncode": 1, "excuses": False, "not_excused": [_DNS],
                     "why": "W's test harness predates the installation user-site opt-in: its tests' own children could "
                            "not import the installation's packages, so a failure W shares proves nothing and blocks, "
                            "attributed through T"}},
        "7.9.1 (bbbbbbbb) did not pass its checks. Nothing was changed.", "a failure there excuses nothing"),
    "differently": ("failed", "checks_failed", {
        "ok": False, "tier": "selected", "file_count": 14, "failed": {_DRAIN: _SIG, _DNS: _SIG},
        "blocking": [_DNS, _DRAIN], "already_failing": [], "target_failing": [],
        "baseline": {"complete": True, "failed": {_DNS: "failure:AssertionError:['AssertionError: /tmp/tmp2b8c']"},
                     "returncode": 1, "excuses": True}},
        "7.9.1 (bbbbbbbb) did not pass its checks. Nothing was changed.", "with another failure"),
    "outside_tests": ("failed", "checks_unverified", {
        "ok": False, "tier": "selected", "file_count": 14, "budget_seconds": 1800, "failed": {}, "blocking": [],
        "new": {"complete": False, "returncode": 1, "lanes": [
            {"lane": "parallel", "returncode": 1, "cases": 300, "seconds": 812.4, "no_failing_case": True},
            {"lane": "serial", "returncode": 5, "cases": 0, "seconds": 2.1}]},
        "unverified": "the selected run did not complete (parallel: exit 1, 300 case(s), serial: exit 5, 0 case(s); "
                      "parallel exited 1 though none of its cases failed); 124 is its deadline, 125 a refused start, 2 "
                      "an interrupted collection, 5 nothing collected"},
        "7.9.1 (bbbbbbbb) could not be verified. Nothing was changed.", "failed though none of its tests did"),
}


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_checks_outcomes_name_failures_and_missing_evidence_apart(engine, tmp_path):
    """Each readiness-check outcome the producer records reads as itself in the real panel at the desk and phone
    windows, light and dark: one headline, its hint, and the specifics in the open facts list; the actions stay the
    failed checks' own (Prepare again, Discard), never a new one."""
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    base, target = "a" * 40, "b" * 40
    status = {"managed": True, "check_ok": True, "available": True, "safe_to_apply": True, "current_sha": base,
              "current_version": "7.9.0", "latest_sha": target, "latest_version": "7.9.1", "warnings": [],
              "update_tx": {"active": False}}

    def respond(route):
        path = urlparse(route.request.url).path
        body = status if path == "/api/update/status" else {"commits": [], "tags": []}
        route.fulfill(content_type="application/json", body=json.dumps(body))

    output = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or diagnostic_directory(tmp_path, "update-checks"))
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            if engine == "webkit" and not Path(pw.webkit.executable_path).is_file():
                pytest.skip(f"{engine} is not installed in the test interpreter")
            browser = getattr(pw, engine).launch(headless=True)
            try:
                for (width, height), theme in [(size, theme) for size in ((1110, 750), (390, 844))
                                               for theme in ("light", "dark")]:
                    page = browser.new_page(viewport={"width": width, "height": height}, color_scheme=theme,
                                            has_touch=width < 980)
                    errors = PageErrors(page, engine)
                    page.route("**/api/**", respond)
                    for scene, (state, reason, tests, headline, phrase) in CHECK_SCENES.items():
                        status["candidate"] = {
                            "candidate_id": CANDIDATE_ID, "state": state, "reason": reason, "target_sha": target,
                            "target_version": "7.9.1", "base_sha": base, "base_version": "7.9.0",
                            "merge_kind": "fast_forward", "environment_same": True,
                            "checks": {"ok": tests["ok"], "tests": tests},
                            "actions": ["activate", "discard_candidate"] if state == "ready"
                            else ["prepare", "discard_candidate"]}
                        page.goto(f"http://127.0.0.1:{server.server_port}/fixture")
                        page.locator("#updates-summary").get_by_text(headline, exact=True).wait_for()
                        hint = page.locator("#updates-hint").inner_text()
                        assert phrase in hint, (scene, hint)
                        details = page.locator("#updates-candidate-details")
                        if not details.evaluate("node => node.open"):
                            details.locator("summary").click()
                        facts = page.locator("#updates-candidate-facts").inner_text()
                        # The recorded reason is one disclosure away whenever the checks reached no verdict.
                        assert ("Why" in facts) == (reason == "checks_unverified"), (scene, facts)
                        if scene in ("unfinished", "not_excused", "differently"):  # a test failed: never "neither"
                            assert "neither passed nor failed" not in f"{hint}\n{facts}", (scene, hint, facts)
                            assert _DRAIN in hint, (scene, hint)
                        if scene == "differently":  # W failed it too, differently: named so, never "only"
                            assert "Failing in 7.9.0 too, differently" in facts, (scene, facts)
                            assert f"only with the update: {_DNS}" not in hint, (scene, hint)
                        labels = [page.locator("#btn-update-primary").inner_text(),
                                  *page.locator("[data-secondary]").all_inner_texts()]
                        assert labels == (["Restart to update", "Discard"] if state == "ready"
                                          else ["Prepare again", "Discard"]), (scene, labels)
                        name = f"checks-{scene}-{engine}-{width}x{height}-{theme}"
                        page.screenshot(path=str(output / f"{name}.png"))
                        page.locator("#updates-status-card").screenshot(path=str(output / f"{name}-card.png"))
                        (output / f"{name}.txt").write_text(
                            f"{page.locator('#updates-summary').inner_text()}\n{hint}\n\n{facts}\n", encoding="utf-8")
                        print(f"UPDATE_CHECKS_SCREENSHOT={output / name}.png")
                        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                    assert not errors.unexpected(), errors
                    page.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_refused_preparation_is_said_in_the_real_panel_from_the_real_projection(engine, scene, monkeypatch, tmp_path):
    """The candidate each scene shows is the projection after a real refused Prepare (only the app's need is stood in):
    no candidate before any check (Astra F1), no candidate with an app needed, and a ready candidate that keeps its
    action. The panel says the refusal first; the Main indicator keeps the failed attempt until it is hidden (F2), and
    the ready candidate's own current state."""
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    from ouroboros.gateway.update_activation import candidate_view

    monkeypatch.setattr(update_prepare, "_last_refusal", {})
    monkeypatch.setattr(update_prepare, "_app_need_now", lambda: {
        "status": "needed", "reason": "signed_entry_missing", "installed_version": "7.7.0"})
    _undecided_work_root(scene, monkeypatch)
    assert _apply({"strategy": "prepare"})[0] == 202 and _answered() is None
    needed = candidate_view()
    monkeypatch.setattr(update_prepare, "_app_need_now", lambda: {})  # the same refusal under an app that hosts it
    bare = candidate_view()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(scene.tmp / "chosen"))
    _apply({"strategy": "prepare"})
    assert _answered()["state"] == "ready"
    monkeypatch.delenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT")
    assert _apply({"strategy": "prepare"})[0] == 202 and _answered()["state"] == "ready"
    ready = candidate_view()
    assert needed["refusal"]["code"] == ready["refusal"]["code"] == "UPDATE_WORK_ROOT_UNDECIDED"
    status = {"managed": True, "check_ok": True, "available": False, "safe_to_apply": True, "current_sha": "a" * 40,
              "current_version": "7.7.0", "latest_sha": "a" * 40, "warnings": [], "update_tx": {"active": False}}
    # The passive status after an API prepare with no checked cache (``official_status_requires_check``, no check).
    unchecked = {"check_ok": None, "warnings": ["official_status_requires_check"]}
    scenes = {"unchecked": (bare, "has not been checked yet", ["Check for updates"], unchecked, "Update not prepared"),
              "app-needed": (needed, "needs the official app", ["Prepare app update"], {}, "Update not prepared"),
              "ready": (ready, "is ready", ["Restart to update", "Discard"], {}, "Update ready")}
    assert "app_needed" not in bare and bare["refusal"] == needed["refusal"]
    shown = {}

    def respond(route):
        path = urlparse(route.request.url).path
        body = {**status, **shown} if path == "/api/update/status" else {"commits": [], "tags": []}
        route.fulfill(content_type="application/json", body=json.dumps(body))

    output = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or diagnostic_directory(tmp_path, "update-refusal"))
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            if engine == "webkit" and not Path(pw.webkit.executable_path).is_file():
                pytest.skip(f"{engine} is not installed in the test interpreter")
            browser = getattr(pw, engine).launch(headless=True)
            try:
                for (width, height), theme in (((1110, 750), "light"), ((390, 844), "dark")):
                    page = browser.new_page(viewport={"width": width, "height": height}, color_scheme=theme,
                                            has_touch=width < 980)
                    errors = PageErrors(page, engine)
                    page.route("**/api/**", respond)
                    for name, (candidate, headline, labels, overrides, pill_text) in scenes.items():
                        shown.clear()
                        shown.update(overrides, candidate=candidate)
                        page.goto(f"http://127.0.0.1:{server.server_port}/fixture?indicator")
                        page.locator("#updates-summary").get_by_text(headline).wait_for()
                        hint = page.locator("#updates-hint").inner_text()
                        assert hint.startswith("The last preparation was refused: the default subagent_worktrees "), hint
                        assert "OUROBOROS_SUBAGENT_WORKTREE_ROOT" in hint, hint
                        assert [page.locator("#btn-update-primary").inner_text(),
                                *page.locator("[data-secondary]").all_inner_texts()] == labels
                        pill = page.locator("#update-pill")
                        page.wait_for_function("(text) => document.getElementById('update-pill')?.textContent === text",
                                               arg=pill_text)
                        indicator = {"text": pill.inner_text(), "hidden": pill.evaluate("node => node.hidden"),
                                     "tone": pill.get_attribute("data-tone"), "title": pill.get_attribute("title"),
                                     "close": not page.locator(".update-notice-close").evaluate("node => node.hidden"),
                                     "on_screen": pill.evaluate(ON_SCREEN)}  # 390 px: in the closed drawer
                        failed = pill_text == "Update not prepared"
                        assert (indicator["hidden"], indicator["tone"], indicator["close"]) == (
                            False, "error" if failed else None, failed), indicator
                        assert not failed or indicator["title"].startswith("The last preparation was refused: ")
                        stem = f"refused-{name}-{engine}-{width}x{height}-{theme}"
                        page.screenshot(path=str(output / f"{stem}.png"))
                        page.locator("#updates-status-card").screenshot(path=str(output / f"{stem}-card.png"))
                        if indicator["on_screen"]:
                            page.locator(".update-notice").screenshot(path=str(output / f"{stem}-indicator.png"))
                        (output / f"{stem}.json").write_text(json.dumps({"candidate": candidate, "hint": hint,
                                                                         "indicator": indicator}, indent=1),
                                                             encoding="utf-8")
                        print(f"UPDATE_REFUSAL_SCREENSHOT={output / stem}.png")
                        assert page.evaluate(
                            "document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                    assert not errors.unexpected(), errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

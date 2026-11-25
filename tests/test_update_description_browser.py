"""Real Dashboard › Updates and the legacy Apply confirmation with the update description.

The page is the shipped SPA modules (dashboard.js, updates.js, update_status.js, confirm_dialog.js,
chat_markdown.js) served from ``web/``; only the API replies are synthetic, derived from the real
``1b5608..fcd6`` and ``#2367`` ranges (``tests/fixtures/update_description_ranges.json``: receipts
parsed by the production collector, post-merge runs from a retained live read). No runtime, task or
model starts. Screens land in the test-evidence directory for inspection.

Covered: an update on offer with nothing written offers one action, Check and describe; it posts one
request naming no range and the request's task shows as writing, with the owner's fixed request time and a
door that opens that task's card in Main (kept through a passive refresh and the check that writes nothing
while it writes, gone once it saved; where it lands is the writing note's own test, through the real gateway
and router, with a status frame before the answer and an answer that never arrives); Apply opened meanwhile
pins only its own prose, so the worker's ``update_progress_changed`` and the task's ``task_done`` behind it
still read the status and Cancel shows the large structured description rendered as Markdown; Check and
describe over a current text keeps the read text's DOM; the Apply confirmation renders a large and a small
description in a bounded region that scrolls to the text's last paragraph, keeps the consequences in its body
at the prose's size and ink (ordinary dialogs keep theirs) and every per-commit receipt inside collapsed
Details, and cancels by Escape and by Cancel without applying; Rewrite waits in Details. The third test loads
the whole SPA (app.js and its sidebar): the panel's Read all follows a description saved while the Dashboard
is away and a window that narrows (#2934).
"""
import copy
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qsl, urlparse

import pytest

from ouroboros.test_evidence import diagnostic_directory
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui  # pytest fixture: the real SPA
from tests.test_update_letter_browser import WEB
from tests.test_update_description import BASE, TARGET, _checked, _key, _offer
from tests.test_update_description import _no_git_for_versions as _no_git_for_versions  # autouse pytest fixture
from tests.test_update_description import door as door  # pytest fixture; the alias re-exports it (no F811)
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser]
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "update_description_ranges.json").read_text(encoding="utf-8"))
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
TASK = "d3adb33f"
INK = "el => [getComputedStyle(el).fontSize, getComputedStyle(el).color]"
# The host's own line for the large range: two of its merges carry a receipt that observed no bound review.
OVER_LINE = "Went in over a review FAIL or no verdict: 2 merges."
PIN = {"action": "keep", "direction": "forward", "serving": {"version": "4.8.0"}, "target": {"version": "4.8.0"}}
EFFECTS = {"engine": PIN, "replacement_engine": PIN, "active_work": {"status": "known", "model": 0, "non_model": 1},
           "runtime": {"installed": {"python": "3.10.19", "sqlite": "3.50.4"},
                       "bundle": {"python": "3.10.19", "sqlite": "3.50.4"}}}


def _letter(case, state):
    """The ``letter`` the backend projects for one case and stage (shapes of update_letter.project_letter)."""
    key = case["key"]
    base = {"state": "none", "relation": "", "text": "", "author_version": "", "target_version": "", "written_at": "",
            "error_kind": "", "error_text": "", "key": {}, "has_last_good": False, "description_current": False,
            "failed_at": "", "latest_failed_key": None, "qualification": case["qualification"], "revision": 0,
            "author": {}, "checked_key": key, "material": {"key": key, "text_bytes": 152000, "commits": case["commits"]},
            "request": None}
    request = {"task_id": TASK, "mode": "write", "key": key, "requested_at": "2026-10-28T11:20:00Z"}
    if state == "writing":
        return {**base, "request": {**request, "status": "running", "saved": False}}
    if state == "ready":
        return {**base, "state": "ready", "relation": "pending", "text": case["text"], "key": key,
                "description_current": True, "revision": 1, "author_version": "7.8.1", "target_version": case["version"],
                "written_at": "2026-10-28T11:31:00Z",
                "author": {"task_id": TASK, "model": "anthropic/claude-opus-5.5", "requested": True},
                "request": {**request, "status": "completed", "saved": True}}
    return base


def _status(case, letter):
    key = case["key"]
    return {"managed": True, "check_ok": None, "available": True, "safe_to_apply": True,
            "current_version": "7.8.1", "latest_version": case["version"], "current_sha": key["base_sha"],
            "current_short_sha": key["base_sha"][:8], "running_sha": key["base_sha"], "latest_sha": key["target_sha"],
            "latest_short_sha": key["target_sha"][:8], "checked_target_sha": key["target_sha"],
            "update_channel": "development", "target_ref": "managed/ouroboros", "behind": case["commits"], "ahead": 0,
            "local_changes": "absent", "warnings": [], "update_tx": {"active": False}, "update_progress": {},
            "checked_at": "2026-10-28T08:10:03Z", "letter": letter}


def _preflight(case):
    key = case["key"]
    plan = {"available": True, "kind": "clean", "auto_mergeable": True, "base_sha": key["base_sha"],
            "target_sha": key["target_sha"], "local_dirty_count": 0, "update_channel": "development",
            "target_ref": "managed/ouroboros", "recommended_strategy": "auto_merge"}
    return {"merge_plan": plan, "effects": {**EFFECTS, "target_sha": key["target_sha"]},
            "letter": _letter(case, "ready"), "qualification": case["qualification"]}


@pytest.mark.parametrize(("engine", "width", "height"), [("webkit", 1044, 653), ("chromium", 1044, 653),
                                                          ("webkit", 390, 844)])
def test_check_and_describe_then_apply_read_the_description_as_prose(engine, width, height, tmp_path):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    output = diagnostic_directory(tmp_path, "update-description", publish=True)
    large, small = FIXTURE["large"], FIXTURE["small"]
    current = {"status": _status(large, _letter(large, "none")), "preflight": _preflight(large)}
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
        calls.append((path, route.request.method, route.request.post_data))
        if path == "/api/update/description":
            asked = json.loads(route.request.post_data or "{}")
            letter = current["status"]["letter"]
            if asked.get("mode") == "latest" and letter.get("text"):  # the server reuses the found range's text
                body = {"status": "current", "key": letter["key"], "checked_key": letter["key"],
                        "update": {**current["status"], "check_ok": True}}
            else:
                case = large if letter["key"] in ({}, large["key"]) else small
                current["status"] = _status(case, _letter(case, "writing"))
                body = {"status": "queued", "task_id": TASK, "key": case["key"]}
                if asked.get("mode") == "latest":
                    body.update(checked_key=case["key"], update={**current["status"], "check_ok": True})
        elif path in ("/api/update/status", "/api/update/check"):
            body = {**current["status"], "check_ok": path == "/api/update/check" or None}
        elif path == "/api/update/preflight":
            body = current["preflight"]
        elif path == "/api/update/apply":
            body = {"status": "ok", "restarting": True}
        else:
            body = {"commits": [], "tags": []}
        route.fulfill(content_type="application/json", body=json.dumps(body))

    def shot(page, name):
        path = output / f"{name}-{engine}-{width}.png"
        page.screenshot(path=str(path), full_page=False)
        print(f"UPDATE_DESCRIPTION_SCREENSHOT={path}")

    def posted(path):
        return [json.loads(body) if body else None for p, method, body in calls if p == path and method == "POST"]

    def status_reads():
        return sum(1 for p, _method, _body in calls if p == "/api/update/status")

    def wait_until(condition, what):
        for _ in range(50):
            if condition():
                return
            page.wait_for_timeout(100)
        raise AssertionError(f"the panel never read the status again after {what}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            if engine == "webkit" and not Path(pw.webkit.executable_path).is_file():
                pytest.skip(f"{engine} is not installed in the test interpreter")
            browser = getattr(pw, engine).launch(headless=True)
            try:
                # The request's time is shown on the owner's clock; UTC makes the expected words exact.
                page = browser.new_page(viewport={"width": width, "height": height}, has_touch=width < 980,
                                        timezone_id="UTC")
                errors = PageErrors(page, engine)
                page.route("**/api/**", respond)
                page.goto(f"http://127.0.0.1:{server.server_port}/fixture")

                # 1. An update on offer with nothing written: the section stays, with its one action.
                page.wait_for_selector("#updates-letter:not([hidden])")
                assert page.locator("#updates-letter-note").inner_text() == "No description of this update yet."
                action = page.locator("#updates-letter-action")
                assert action.is_visible() and action.inner_text() == "Check and describe" and action.is_enabled()
                assert page.locator(".updates-letter-actions button:visible").count() == 1
                assert page.locator("#btn-update-primary").inner_text() == "Update to 7.8.1"
                # The host's line about the range stands with no description too, as a warn status sentence.
                over = page.locator("#updates-letter-over")
                assert over.is_visible() and over.inner_text() == OVER_LINE
                assert over.get_attribute("data-tone") == "warn" and "ui-status" in over.get_attribute("class").split()
                shot(page, "01-empty")

                # 2. Check and describe is one request naming no range (the server checks and asks); the request's
                #    own task shows as writing and the action becomes a check that writes nothing.
                action.click()
                page.wait_for_function("document.querySelector('#updates-letter-action').textContent === 'Check'")
                assert posted("/api/update/description") == [{"mode": "latest"}]
                assert posted("/api/update/check") == [], "the server's own check answers the press"
                # The panel's one time formatter, as this engine writes it ("28 Oct, 11:20" / "28 Oct at 11:20").
                asked = page.evaluate("import('/static/modules/update_status.js')"
                                      ".then(m => m.attemptTime('2026-10-28T11:20:00Z'))")
                assert "28 Oct" in asked and "11:20" in asked, asked
                assert page.locator("#updates-letter-note").inner_text() == (
                    f"Ouroboros is writing the description you asked for on {asked} (task {TASK}).")
                door = page.locator("#updates-letter-task button")
                assert door.inner_text() == "Open in Main ↗" and door.is_visible()
                line = page.locator("#updates-letter-note-line")
                assert line.evaluate("el => el.scrollWidth <= el.clientWidth"), "the line wraps, never widens"
                assert page.evaluate("document.documentElement.scrollWidth<=document.documentElement.clientWidth")
                shot(page, "02-writing")
                line.screenshot(path=str(output / f"02-writing-note-{engine}-{width}.png"))
                # The keyboard reaches it next after the action (WebKit's plain Tab skips buttons as macOS does,
                # Option+Tab reaches every control). A passive refresh while it still writes keeps the very door
                # node, and with it focus.
                action.focus()
                page.keyboard.press("Alt+Tab" if engine == "webkit" else "Tab")
                assert door.evaluate("el => el === document.activeElement && el.matches(':focus-visible')")
                line.screenshot(path=str(output / f"02-writing-note-focus-{engine}-{width}.png"))
                page.evaluate("document.documentElement.dataset.theme='light'")
                line.screenshot(path=str(output / f"02-writing-note-light-{engine}-{width}.png"))
                page.evaluate("delete document.documentElement.dataset.theme")
                page.evaluate("window.savedDoor=document.querySelector('#updates-letter-task button')")
                before = status_reads()
                page.evaluate("window.__wsHandlers.get('update_progress_changed')()")
                wait_until(lambda: status_reads() > before, "a passive refresh")
                page.wait_for_timeout(200)
                assert page.evaluate("window.savedDoor===document.querySelector('#updates-letter-task button')"
                                     " && document.activeElement===window.savedDoor")
                # The check that writes nothing while it writes: one fetching check, no second request, and the
                # writer's note and door stay through it.
                action.click()
                wait_until(lambda: len(posted("/api/update/check")) == 1, "the check while it writes")
                page.wait_for_function("document.querySelector('#updates-letter-action').textContent === 'Check'")
                assert len(posted("/api/update/description")) == 1
                assert page.evaluate("window.savedDoor===document.querySelector('#updates-letter-task button')")
                # The press is the one navigation to that task's card in Main (its landing: the test below).
                page.evaluate("window.addEventListener('ouro:open-task', e => { window.openedTask = e.detail; })")
                door.click()
                assert page.evaluate("window.openedTask") == {"task_id": TASK, "chat_id": 1}

                # 3. Apply opened while it writes: the plan has no text yet and the confirmation says so. The
                #    worker's save (update_progress_changed) and its task's end (task_done) arrive while it is
                #    open; the plan pins only the dialog's prose, so Cancel shows the description now saved.
                current["preflight"] = {**_preflight(large), "letter": _letter(large, "none")}
                page.locator("#btn-update-primary").click()
                dialog = page.get_by_role("dialog")
                dialog.wait_for()
                assert "No description is available for this exact preflight range." in dialog.inner_text()
                assert dialog.locator("[data-confirm-content] .updates-letter-over").inner_text() == OVER_LINE, (
                    "the confirmation carries the line with no description")
                current["status"] = _status(large, _letter(large, "ready"))
                for frame in ("window.__wsHandlers.get('update_progress_changed')()",
                              f"window.__wsHandlers.get('log')({{data: {{type: 'task_done', task_id: '{TASK}'}}}})"):
                    before = status_reads()
                    page.evaluate(frame)
                    wait_until(lambda: status_reads() > before, frame)
                page.wait_for_function("document.querySelector('#updates-letter-body').textContent.includes('Что будет заметно сразу')")
                assert "No description is available for this exact preflight range." in dialog.inner_text()
                dialog.get_by_role("button", name="Cancel", exact=True).click()
                dialog.wait_for(state="detached")
                current["preflight"] = _preflight(large)
                page.wait_for_selector("#updates-letter-body h2, #updates-letter-body .md-h2")
                body = page.locator("#updates-letter-body")
                rendered = body.inner_text()
                assert "Что будет заметно сразу" in rendered and "Telegram." in rendered
                assert "##" not in rendered and "**" not in rendered, "Markdown renders, never shows raw"
                assert body.locator("li").count() >= 12
                assert over.is_visible() and over.inner_text() == OVER_LINE
                assert over.bounding_box()["y"] < body.bounding_box()["y"], "the line stands above the description"
                assert page.locator("#updates-letter-label").inner_text() == "What's new"
                # The range the text covers is its provenance, in the letter's Details (the heading keeps its date).
                covers = page.locator("#updates-letter-provenance").text_content()
                assert f"{large['key']['base_sha'][:8]} → {large['key']['target_sha'][:8]}" in covers
                assert action.inner_text() == "Check and describe" and action.is_enabled()
                assert page.locator("#updates-letter-note").is_hidden()
                assert page.locator("#updates-letter-task button").count() == 0, "an ended request has no door"
                assert page.evaluate("document.documentElement.scrollWidth<=document.documentElement.clientWidth")
                heading, actions = (page.locator(f".updates-letter-{part}").bounding_box() for part in ("heading", "actions"))
                if width < 500:  # a phone: the actions follow the heading instead of squeezing its date and range
                    assert actions["y"] >= heading["y"] + heading["height"] - 1, (heading, actions)
                else:
                    assert actions["y"] < heading["y"] + heading["height"], "one row while both fit"
                shot(page, "03-large")
                page.locator("#updates-letter-details summary").click()
                provenance = page.locator("#updates-letter-provenance").inner_text()
                assert f"in task {TASK}" in provenance and "at your request" in provenance
                # Asking again for this very range waits here, after the sentence on what writing uses.
                rewrite = page.locator("#updates-letter-rewrite")
                assert rewrite.is_visible() and rewrite.inner_text() == "Rewrite" and rewrite.is_enabled()
                shot(page, "03b-details")
                named = page.locator("#updates-letter-over-names").inner_text()  # Details name what the line counts
                assert named.startswith("Went in over a review FAIL or no verdict: ") and named.count("(merge; no review verdict recorded)") == 2
                assert "Made on this computer: Not read for this range" in named, "rows that were not read are unknown, never none"
                page.locator("#updates-letter-details summary").click()

                # 4. Check and describe over the text of the very range it finds reuses it: no task, and a refreshed
                #    run leaves the read text's very DOM node in place.
                page.evaluate("window.savedBody=document.querySelector('#updates-letter-body').firstChild")
                ready = _status(large, _letter(large, "ready"))
                ready["letter"]["qualification"] = copy.deepcopy(large["qualification"])
                ready["letter"]["qualification"]["github"]["runs"][0]["conclusion"] = "failure"
                current["status"] = ready
                action.click()
                page.wait_for_selector(".toast")
                assert page.locator(".toast").first.inner_text() == "Description is up to date."
                assert page.evaluate("window.savedBody===document.querySelector('#updates-letter-body').firstChild")
                assert posted("/api/update/description") == [{"mode": "latest"}] * 2
                assert len(posted("/api/update/check")) == 1

                # 5. Apply: the large description reads as prose in its own region; evidence stays in Details.
                page.locator("#btn-update-primary").click()
                dialog = page.get_by_role("dialog")
                dialog.wait_for()
                region = dialog.locator("[data-confirm-content]")
                assert region.get_attribute("aria-label") == "What's new"
                region_text = region.inner_text()
                assert "Что будет заметно сразу" in region_text and "##" not in region_text and "**" not in region_text
                assert region.locator("li").count() >= 12
                line = region.locator(".updates-letter-over")
                assert line.inner_text() == OVER_LINE and line.get_attribute("data-tone") == "warn"
                assert line.bounding_box()["y"] < region.locator(".updates-letter-body").bounding_box()["y"]
                plain = dialog.locator(".ui-modal-body > p").inner_text()
                for fact in ("Git applies this update directly", "Claudexor: 4.8.0 → 4.8.0", "forward-only"):
                    assert fact in plain, fact
                details = dialog.locator("details.confirm-dialog-details")
                assert details.locator("summary").inner_text() == "Checks for this range"
                assert details.get_attribute("open") is None, "per-commit receipts stay collapsed"
                # Beside the prose, the consequences and the Details summary read at its size and ink.
                prose = region.locator(".updates-letter-body").evaluate(INK)
                assert prose[0] == "14px" and dialog.locator(".ui-modal-body > p").evaluate(INK) == prose
                assert details.locator("summary").evaluate(INK) == prose
                assert "post-merge" not in plain and "full_battery" not in plain and "receipt" not in plain
                box = region.bounding_box()
                assert box["height"] <= min(0.42 * height, 380) + 2, box
                actions = dialog.locator(".ui-modal-actions").bounding_box()
                assert actions["y"] + actions["height"] <= height, "the actions stay in view beside long prose"
                shot(page, "04-apply-large")
                details.locator("summary").click()
                rows = details.locator(".confirm-dialog-detail-row")
                # The pinned range leads the evidence, even when the versions are shown in the title.
                qualification = large["qualification"]
                assert rows.locator("dt").all_text_contents() == [
                    "Commits", "Made on this computer", "Review receipts",
                    *(commit["sha"][:8] for commit in qualification["commits"]),
                    "GitHub checks",
                    *(f"{run['sha'][:8]} post-merge" for run in qualification["github"]["runs"]),
                    "No matching run observed",
                ]
                key = large["key"]
                assert rows.first.locator("dd").inner_text() == f"{key['base_sha'][:8]} → {key['target_sha'][:8]}"
                assert rows.nth(1).locator("dd").inner_text().startswith("Not read for this range")
                assert "7 of the 17 commits listed here carry one" in rows.nth(2).locator("dd").inner_text()
                assert "Merge pull request #2384" in details.inner_text() and "No matching run observed" in details.inner_text()
                details.get_by_text(qualification["commits"][1]["sha"][:8], exact=True).scroll_into_view_if_needed()
                shot(page, "05-apply-large-details")
                # Keyboard: the prose region takes focus and scrolls; Escape cancels without applying.
                region.focus()
                assert page.evaluate("document.activeElement.matches('[data-confirm-content]')")
                page.keyboard.press("End")  # it scrolls the whole description to its last paragraph (animated)
                page.wait_for_function("""el => { const box = el.getBoundingClientRect();
                    const last = el.querySelector('.updates-letter-body').lastElementChild.getBoundingClientRect();
                    return el.scrollTop > 0 && last.top >= box.top && last.bottom <= box.bottom + 1; }""",
                                       arg=region.element_handle(), timeout=5000)
                shot(page, "05b-apply-large-end")
                page.keyboard.press("Escape")
                dialog.wait_for(state="detached")
                assert posted("/api/update/apply") == []

                # 5b. The same screens in the light theme (the product has two).
                page.evaluate("document.documentElement.dataset.theme='light'")
                shot(page, "03-large-light")
                page.locator("#btn-update-primary").click()
                dialog.wait_for()
                shot(page, "04-apply-large-light")
                dialog.get_by_role("button", name="Cancel", exact=True).click()
                dialog.wait_for(state="detached")
                page.evaluate("delete document.documentElement.dataset.theme")

                # The dialog has the newer range B. Cancel's failed status read keeps the panel's text A.
                current["preflight"] = _preflight(small)
                page.locator("#btn-update-primary").click()
                dialog.wait_for()
                assert "Небольшое исправление Dashboard" in dialog.locator("[data-confirm-content]").inner_text()
                assert "Что будет заметно сразу" in body.text_content()
                page.route("**/api/update/status", lambda r: r.fulfill(status=503, json={"error": "offline"}), times=1)
                dialog.get_by_role("button", name="Cancel", exact=True).click()
                page.wait_for_function("document.querySelector('#updates-letter-label').textContent === 'Saved description'")
                assert page.evaluate("window.savedBody===document.querySelector('#updates-letter-body').firstChild")
                assert "Что будет заметно сразу" in body.inner_text()
                assert page.locator("#btn-update-replace").is_disabled()
                assert posted("/api/update/apply") == []

                # 6. A small description reads the same way; Cancel closes without applying.
                current["preflight"] = _preflight(small)
                current["status"] = _status(small, _letter(small, "ready"))
                page.evaluate("window.__wsHandlers.get('update_progress_changed')()")
                page.wait_for_function(f"document.querySelector('#updates-letter-provenance').textContent.includes('{small['key']['target_sha'][:8]}')")
                page.locator("#btn-update-primary").click()
                dialog.wait_for()
                small_text = dialog.locator("[data-confirm-content]").inner_text()
                assert "Небольшое исправление Dashboard" in small_text and "**" not in small_text
                # A range whose receipts all passed has no line, in the panel and in the confirmation alike.
                assert page.locator("#updates-letter-over").is_hidden() and dialog.locator(".updates-letter-over").count() == 0
                shot(page, "06-apply-small")
                dialog.get_by_role("button", name="Cancel", exact=True).click()
                dialog.wait_for(state="detached")
                assert posted("/api/update/apply") == []
                # An ordinary dialog keeps its own text: only a dialog carrying prose steps its facts down.
                page.evaluate("import('/static/modules/confirm_dialog.js')"  # the dialog's own promise stays open
                              ".then((m) => { m.openConfirmDialog({title: 'Plain', body: 'An ordinary question.'}); })")
                dialog.wait_for()
                assert dialog.locator(".ui-modal-body > p").evaluate(
                    "el => getComputedStyle(el).fontSize === getComputedStyle(el.parentElement).fontSize")
                page.keyboard.press("Escape")
                dialog.wait_for(state="detached")

                # 7. Rewrite (Details) is the explicit request for the shown text's own range, with no check first.
                page.locator("#updates-letter-details summary").click()
                page.locator("#updates-letter-rewrite").click()
                page.wait_for_function("document.querySelector('#updates-letter-action').textContent === 'Check'")
                assert posted("/api/update/description")[-1] == {"mode": "rewrite", "base_sha": small["key"]["base_sha"],
                                                                 "target_sha": small["key"]["target_sha"]}
                assert len(posted("/api/update/check")) == 1 and page.locator("#updates-letter-rewrite").is_hidden()
                assert errors.unexpected() == []
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# The door's destination, through the real producers and consumers: the gateway's admission and Main's
# history read by the real router (``door``), the task's tool rows by the real tool-log writer, Main by
# chat.js, and the navigation by app.js's own revealQuotedSource and ``ouro:open-task`` listener, read
# from app.js verbatim. Only the shell stands in: ``showPage`` switches the page and announces it, and the page
# navigation's ``capture`` names the place a jump leaves (app.js revealQuotedSource takes its origin there).
MAIN_BOOT = """<script type="module">
import {initDashboard} from '/static/modules/dashboard.js';
import {initUpdates} from '/static/modules/updates.js';
import {createChatInstance} from '/static/modules/chat.js';
import {apiClient} from '/static/modules/api_client.js';
import {tr} from '/static/modules/i18n.js';
const state={activePage:'dashboard',dashboardActiveSubtab:'updates',projectChatIds:new Set(),unreadCount:0};
const handlers=new Map();
const ws={on(type,fn){(handlers.get(type)||handlers.set(type,new Set()).get(type)).add(fn);return()=>handlers.get(type).delete(fn);},
    isConnected:()=>true,send(){}};
window.emit=(type,payload)=>handlers.get(type)?.forEach(fn=>fn(payload));
let generation=0;
window.mainChat=createChatInstance({ws,state,chatId:1,stateSnapshots:{begin:()=>({generation:++generation,requestedAt:Date.now()}),
    gate(){return Promise.resolve(this.begin());},isCurrent:()=>true,apply(){}}});
mainChat.page.classList.remove('active');
const dashboard=initDashboard({state}); dashboard.page.classList.add('active');
initUpdates({mount:document.getElementById('dashboard-panel-updates'),state,ws});
dashboard.activateTab('updates');
window.appState=state; window.notices=[];
const showPage=async(name)=>{
    document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
    document.getElementById(`page-${name}`).classList.add('active');
    state.activePage=name;
    window.dispatchEvent(new CustomEvent('ouro:page-shown',{detail:{page:name}}));
    return true;
};
const app=await (await fetch('/static/app.js')).text();
const door=app.slice(app.indexOf('async function revealQuotedSource('),app.indexOf('\\nfunction freshProjectRow'));
const noProject=()=>{throw new Error('a Main task never reads Projects');};
const pageNavigation={capture:()=>({key:state.activePage})};
new Function('apiClient','mainChat','state','showPage','pageNavigation','showToast','tr','fetchJson','openProjectPanel',
    'let projectNavigationGeneration = 0;\\n'+door)(apiClient,mainChat,state,showPage,pageNavigation,text=>notices.push(text),tr,noProject,noProject);
window.ready=true;
</script>"""


def _successor(root, handle, task, lineage):
    """The request's current task after a host retry or an admitted Continue, recorded by their own writers
    as the backend tests record them: the first attempt spoke once, its successor has not spoken yet."""
    from ouroboros import dialogue_store
    from ouroboros.owner_continue import binding_sha, claim_on_predecessor, mark_claim_admitted, successor_id
    from ouroboros.task_results import write_task_result

    dialogue_store.append(root, "progress", {"ts": "2026-10-30T09:59:00Z", "chat_id": 1, "task_id": handle,
                                             "content": "Reading the update changes."})
    if lineage == "retry":
        current = f"{handle}-retry"
        write_task_result(root, current, "running", chat_id=1, root_task_id=handle, original_task_id=handle,
                          timeout_retry_from=handle, supersedes_task_id=handle, delegation_role="root")
        write_task_result(root, handle, "interrupted", reason_code="idle_timeout_retry", retry_task_id=current,
                          superseded_by=current)
        return current, {**task, "id": current, "original_task_id": handle, "timeout_retry_from": handle, "_attempt": 2}
    nonce = "door-continue"
    current = successor_id(handle, nonce)
    binding = {"verb": "continue", "binding_version": 1, "predecessor_task_id": handle, "action_nonce": nonce,
               "successor_task_id": current, "cause": "snapshot_restore"}
    write_task_result(root, handle, "cancelled", cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    claim_on_predecessor(root, handle, nonce, binding)
    write_task_result(root, current, "running", chat_id=1, root_task_id=current, delegation_role="root",
                      continuation_admission={"binding": binding, "binding_sha256": binding_sha(binding)})
    mark_claim_admitted(root, handle, nonce)
    return current, {**task, "id": current, "root_task_id": current}


@pytest.mark.parametrize(("engine", "width", "height", "owner_chat", "lineage", "reply"), [
    ("webkit", 1044, 653, None, "", "held"), ("chromium", 1044, 653, None, "", "lost"),
    ("webkit", 390, 844, None, "", "held"), ("chromium", 1044, 653, 1, "", "held"),
    ("chromium", 1044, 653, 1, "retry", "held"), ("chromium", 1044, 653, 1, "continue", "held")])
def test_the_writing_note_opens_main_s_card_before_its_first_message(door, monkeypatch, engine, width, height,
                                                                      owner_chat, lineage, reply, tmp_path):
    """Check and describe pressed before any owner message (the binding known empty) or after the web owner bound
    Main, through the real gateway: the server checks and admits the request as Main's work, and the status frame
    it broadcasts reaches the window before the answer does, so the writer and its door show while the press still
    reads Checking…; an answer that never arrives is said beside that writer, never taken for it. The request's
    current task (the first one, a host retry or an admitted Continue) calls a tool before it says anything, and
    the note's door lands on the card Main draws for that task, focused under the header, without a history read
    (around-history has no row of it yet and answers absent)."""
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    from ouroboros import update_letter as ul
    from ouroboros.task_results import write_task_result
    from ouroboros.tool_call_log import append_call_row, invocation_fields, new_invocation
    from supervisor import queue, state

    state.update_state(lambda live: live.update(owner_chat_id=owner_chat))
    _offer(door.root)
    _checked(monkeypatch)  # the press's own check on the server finds exactly the offered range
    output = diagnostic_directory(tmp_path, "update-description-door", publish=True)
    case = {"key": _key(), "commits": 1, "version": "6.114.0", "qualification": None}
    history_reads, held = [], []

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == "/fixture":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                html = (WEB / "index.html").read_text(encoding="utf-8").replace(
                    '<script type="module" src="/static/app.js"></script>', MAIN_BOOT)
                self.wfile.write(html.encode("utf-8"))
            else:
                self.path = self.path.removeprefix("/static")
                super().do_GET()

    def respond(route):
        url = urlparse(route.request.url)
        if url.path == "/api/update/status":
            letter = ul.project_letter_for_panel({"running_sha": BASE, "checked_target_sha": TARGET}, drive_root=door.root)
            route.fulfill(content_type="application/json", body=json.dumps(_status(case, letter)))
        elif url.path in ("/api/update/description", "/api/chat/history"):
            if url.path == "/api/chat/history":
                history_reads.append(dict(parse_qsl(url.query)))
            answer = door.client.request(route.request.method, url.path, params=parse_qsl(url.query),
                                         content=route.request.post_data_buffer)
            if url.path == "/api/update/description":
                held.append((route, answer))  # the server has answered; the window has not heard it yet
                return
            route.fulfill(status=answer.status_code, content_type="application/json", body=answer.content)
        else:
            route.fulfill(content_type="application/json", body="{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            if engine == "webkit" and not Path(pw.webkit.executable_path).is_file():
                pytest.skip(f"{engine} is not installed in the test interpreter")
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": width, "height": height}, has_touch=width < 980,
                                        timezone_id="UTC")
                errors = PageErrors(page, engine)
                page.route("**/api/**", respond)
                page.goto(f"http://127.0.0.1:{server.server_port}/fixture")
                page.wait_for_function("window.ready")
                action = page.locator("#updates-letter-action")
                assert action.inner_text() == "Check and describe"
                action.click()
                for _ in range(100):  # the gateway has checked and admitted; its answer is held from the window
                    if held:
                        break
                    page.wait_for_timeout(100)
                [task] = door.pending
                assert task["chat_id"] == 1, "the panel's request is Main's work, even before any owner message"
                assert json.loads(held[0][1].content)["status"] == "queued" and door.frames
                task_id = task["id"]
                # The server's status frame arrives first: the writer and its door show while the press reads Checking….
                page.evaluate("window.emit('update_progress_changed')")
                door_button = page.locator("#updates-letter-task button")
                door_button.wait_for()
                assert task_id in page.locator("#updates-letter-note").inner_text()
                assert action.inner_text() == "Checking…" and action.is_disabled()
                page.evaluate("window.heldDoor=document.querySelector('#updates-letter-task button')")
                route, answer = held.pop()
                if reply == "lost":
                    route.abort()
                else:
                    route.fulfill(status=answer.status_code, content_type="application/json", body=answer.content)
                page.wait_for_function("document.querySelector('#updates-letter-action').textContent === 'Check'")
                note = page.locator("#updates-letter-note").inner_text()
                if reply == "lost":  # the writer is a status fact, not this press's answer
                    assert note.startswith("This press received no confirmation (")
                else:
                    assert not note.startswith("This press received no confirmation")
                assert task_id in note and len(door.pending) == 1, "nothing is asked again on its own"
                assert page.evaluate("window.heldDoor===document.querySelector('#updates-letter-task button')"), (
                    "the door stayed the same node from the frame to the answer")

                # The current task calls a tool before its first message (the real tool-log writer).
                door.pending.clear()
                write_task_result(door.root, task_id, "running")
                if lineage:
                    task_id, task = _successor(door.root, task_id, task, lineage)
                queue.RUNNING[task_id] = {"task": task}
                call = {"task_id": task_id, "tool": "query_data", "args": {"kind": "update_material"},
                        **invocation_fields(new_invocation("door-read-1", {"execution_id": "door-exec",
                                                                           "round_id": "door-round", "llm_call_id": "door-llm"}, 1))}
                meta = {"root_task_id": task_id, "delegation_role": "root"}
                for row in ({"type": "tool_call_started", "ts": "2026-10-30T09:59:58Z"},
                            {"type": "tool_call", "status": "ok", "is_error": False, "elapsed_ms": 1,
                             "result_preview": "material", "ts": "2026-10-30T09:59:59Z"}):
                    assert append_call_row(meta, door.root / "logs", {**call, **row})["task_log"] is True
                around = door.client.get("/api/chat/history", params={"chat_id": 1, "around": f"task:{task_id}"}).json()
                assert around["anchor"]["state"] == "absent", "around-history has no row of it before it speaks"
                # Main reads its newest page as on any arrival and draws the task's card.
                page.evaluate("mainChat.refreshHistory({revision: 1})")
                card = page.locator(f'#page-chat .chat-live-card[data-task-id="{task_id}"]')
                card.wait_for(state="attached")
                page.evaluate("window.emit('update_progress_changed')")
                page.wait_for_function(f"document.querySelector('#updates-letter-note').textContent.includes('{task_id}')")
                before = [read for read in history_reads if "around" in read]
                shot = output / f"main-card-{engine}-{width}-{owner_chat}-{lineage or 'first'}.png"

                door_button.click()
                page.wait_for_function("document.querySelector('#page-chat .is-quote-target') || notices.length")
                assert page.evaluate("notices") == [], "the door lands; it never says the task is unavailable"
                landed = page.locator(f'#page-chat .chat-live-card[data-task-id="{task_id}"].is-quote-target')
                landing = landed.evaluate("""el => {
                    const card = el.getBoundingClientRect(), feed = el.closest('#chat-messages').getBoundingClientRect();
                    return { focused: document.activeElement === el, top: card.top, feedTop: feed.top,
                             feedBottom: feed.bottom, page: window.appState.activePage, address: el.dataset.rowAddress };
                }""")
                assert landing["focused"] and landing["page"] == "chat" and landing["address"] == f"task:{task_id}", landing
                assert landing["feedTop"] - 2 <= landing["top"] < min(height, landing["feedBottom"]), landing
                assert [read for read in history_reads if "around" in read] == before == [], "no history read"
                page.screenshot(path=str(shot), full_page=False)
                print(f"UPDATE_DESCRIPTION_SCREENSHOT={shot}")
                assert errors.unexpected() == []
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(("engine", "width", "height", "found"), [
    ("chromium", 1044, 653, True), ("webkit", 390, 844, True), ("chromium", 1044, 653, False)])
def test_the_headline_check_of_a_current_install_describes_what_it_finds(door, monkeypatch, engine, width, height,
                                                                         found, tmp_path):
    """Up to date with nothing recorded and no update known: the headline's one action is Check and describe. One
    press sends the single server-side request through the real gateway; when its check finds an update exactly one
    writer is admitted and shown with its door, and when it finds none nothing is admitted and the headline stays
    up to date. No second press and no second check."""
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    from ouroboros import update_letter as ul

    _checked(monkeypatch, available=found)  # what the server's own check finds
    output = diagnostic_directory(tmp_path, "update-description-headline", publish=True)
    case = {"key": _key(), "commits": 1, "version": "6.114.0", "qualification": None}
    level = {"managed": True, "check_ok": None, "available": False, "current_version": "7.8.1",
             "current_sha": BASE, "current_short_sha": BASE[:8], "running_sha": BASE, "checked_target_sha": BASE,
             "update_channel": "stable", "target_ref": "managed/main", "checked_at": "2026-11-17T08:00:00Z",
             "warnings": [], "update_tx": {"active": False}, "update_progress": {}}
    asked, calls, held = [], [], []
    transaction = {"active": False}

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
        url = urlparse(route.request.url)
        calls.append(url.path)
        if url.path == "/api/update/description":
            asked.append(json.loads(route.request.post_data))
            answer = door.client.post(url.path, json=asked[-1])
            held.append((route, answer))
        elif url.path == "/api/update/status":
            offered = bool(asked) and found  # after the press, the status reads what the server's check recorded
            status = _status(case, None) if offered else dict(level)
            status["update_tx"] = transaction
            status["letter"] = ul.project_letter_for_panel(
                {"running_sha": BASE, "checked_target_sha": TARGET if offered else BASE}, drive_root=door.root)
            if held and offered:  # status sampled before admission, delivered while its reply is pending
                status["letter"] = _letter(FIXTURE["small"], "ready")
            route.fulfill(content_type="application/json", body=json.dumps(status))
        elif url.path == "/api/update/preflight":
            route.fulfill(json=_preflight(FIXTURE["small"]))
        else:
            route.fulfill(content_type="application/json", body=json.dumps({"commits": [], "tags": []}))

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            if engine == "webkit" and not Path(pw.webkit.executable_path).is_file():
                pytest.skip(f"{engine} is not installed in the test interpreter")
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": width, "height": height}, has_touch=width < 980,
                                        timezone_id="UTC")
                errors = PageErrors(page, engine)
                page.route("**/api/**", respond)
                page.goto(f"http://127.0.0.1:{server.server_port}/fixture")
                primary = page.locator("#btn-update-primary")
                page.wait_for_function("document.querySelector('#btn-update-primary').textContent === 'Check and describe'")
                assert page.locator("#updates-summary").inner_text() == "Up to date · 7.8.1"
                assert page.locator("#updates-letter").is_hidden() and door.pending == []
                page.screenshot(path=str(output / f"01-current-{engine}-{width}.png"))
                primary.click()
                page.wait_for_function("document.querySelector('#btn-update-primary').textContent === 'Checking…'")
                assert primary.is_disabled()
                for _ in range(100):
                    if held:
                        break
                    page.wait_for_timeout(50)
                assert len(held) == 1
                if found:
                    # A WS read overtakes the held reply. The visible primary action now belongs to Update.
                    page.evaluate("window.__wsHandlers.get('update_progress_changed')()")
                    page.wait_for_function("document.querySelector('#btn-update-primary').dataset.action === 'update'")
                    assert primary.is_enabled() and primary.inner_text() == "Update to 6.114.0"
                    secondary = page.locator("#updates-letter-action")
                    assert secondary.is_visible() and secondary.is_disabled()
                    page.locator("#updates-letter-details summary").click()
                    rewrite = page.locator("#updates-letter-rewrite")
                    assert rewrite.is_visible() and rewrite.is_disabled()
                    for control in (secondary, rewrite):
                        control.dispatch_event("click")  # Exercise the handler as well as the native disabled gate.
                    primary.click()
                    dialog = page.get_by_role("dialog")
                    dialog.get_by_role("button", name="Cancel", exact=True).click()
                    dialog.wait_for(state="detached")
                    assert calls.count("/api/update/preflight") == 1 and "/api/update/apply" not in calls
                    # Durable recovery may instead require a restart, a plain check or no action at all.
                    for tx_phase, action in (("pending_boot_smoke", "restart"), ("gate_blocked", "check"),
                                             ("assisted_resolution", "")):
                        transaction = {"active": True, "phase": tx_phase}
                        page.evaluate("window.__wsHandlers.get('update_progress_changed')()")
                        page.wait_for_function("action => document.querySelector('#btn-update-primary').dataset.action === action",
                                               arg=action)
                        if action:
                            assert primary.is_enabled()
                        else:
                            assert primary.is_hidden()
                    transaction = {"active": False}
                route, answer = held.pop()
                route.fulfill(status=answer.status_code, content_type="application/json", body=answer.content)
                if found:
                    page.wait_for_function("document.querySelector('#updates-letter-task button')"
                                           " && document.querySelector('#updates-letter-action').textContent === 'Check'")
                    [task] = door.pending
                    assert task["id"] in page.locator("#updates-letter-note").inner_text()
                    assert primary.inner_text() == "Update to 6.114.0"
                    assert page.locator("#updates-letter-action").inner_text() == "Check", "the writer's free check"
                else:
                    page.wait_for_selector(".toast")
                    assert page.locator(".toast").first.inner_text() == "No new update to describe."
                    assert door.pending == [] and page.locator("#updates-letter").is_hidden()
                    page.wait_for_function("document.querySelector('#btn-update-primary').textContent === 'Check and describe'")
                    assert page.locator("#updates-summary").inner_text() == "Up to date · 7.8.1"
                assert asked == [{"mode": "latest"}] and "/api/update/check" not in calls, "one request, one check"
                assert page.evaluate("document.documentElement.scrollWidth<=document.documentElement.clientWidth")
                page.screenshot(path=str(output / f"02-{'writing' if found else 'no-update'}-{engine}-{width}.png"))
                assert errors.unexpected() == []
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# The owner's window (macOS pywebview, WebKit, 100%) and the frame the server broadcasts once a description is saved
# (supervisor/events_worker_reports.py): it carries nothing, the status read brings the text.
OWNER_WINDOW = {"width": 1282, "height": 747}
SAVED = json.dumps({"type": "update_progress_changed"})
# Both status readers (the panel and the sidebar notice) read once per frame. A read counts once the page has parsed
# its body; the render that follows runs in the same microtask chain, before any frame.
COUNT_STATUS_READS = """() => {
    window.statusReads = {started: 0, read: 0};
    const nativeFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
        if (new URL(String(input?.url || input), location.href).pathname !== '/api/update/status') return nativeFetch(input, init);
        statusReads.started += 1;
        return nativeFetch(input, init).then((response) => {
            const json = response.json.bind(response);
            response.json = () => json().finally(() => { statusReads.read += 1; });
            return response;
        });
    };
}"""
QUIET = "statusReads.started === statusReads.read"
FRAMES = "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
LETTER = """() => {
    const body = document.querySelector('#updates-letter-body'), more = document.querySelector('#updates-letter-more');
    const box = body.getBoundingClientRect(), end = body.lastElementChild.getBoundingClientRect();
    return {height: body.clientHeight, content: body.scrollHeight, open: body.classList.contains('is-open'),
            more: more.hidden ? null : more.textContent, last: body.lastElementChild.textContent,
            lastInView: end.top >= 0 && end.bottom <= Math.min(box.bottom + 1, innerHeight)};
}"""
SELECT_LAST = """() => {
    const body = document.querySelector('#updates-letter-body'), range = document.createRange();
    range.selectNodeContents(body.lastElementChild);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    window.kept = {first: body.firstChild, last: body.lastElementChild, text: getSelection().toString()};
}"""
KEPT = """() => {
    const body = document.querySelector('#updates-letter-body'), selection = getSelection();
    return body.firstChild === kept.first && body.lastElementChild === kept.last && selection.toString() === kept.text
        && kept.last.contains(selection.anchorNode);
}"""


@pytest.fixture(params=["webkit", "chromium"])
def letter_ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    ui = request.getfixturevalue("subscription_ui")
    ui["engine"] = request.param
    return ui


def test_read_all_follows_a_description_saved_while_the_dashboard_is_away(letter_ui, tmp_path):
    """#2934 through app.js and its sidebar at the owner's window: the owner leaves Updates for Chat, the description
    saved meanwhile arrives by the server's own frame and renders off-page, and the sidebar brings the Dashboard back
    on Updates (announcing only the page). Read all is offered and opens the whole text to its last paragraph; a
    passive refresh of the same text keeps the open DOM and the owner's selection; Show less folds it; the subtabs
    still switch; a short text, read on entry or saved the same way, offers no button until a narrower window folds
    some of it; on a phone a long text saved behind the Dashboard's list offers Read all once Updates is opened from
    the list."""
    ui, engine = letter_ui, letter_ui["engine"]
    output = diagnostic_directory(tmp_path, "update-description-away", publish=True)
    large = FIXTURE["large"]
    # About four lines in the Updates page's column at the owner's width: with Dashboard's list beside the open page
    # (PR #3369) that column is narrower than the whole content area the first two paragraphs were sized for.
    short = FIXTURE["small"]["text"].split("\n\n")[0]
    current = {"status": _status(large, {**_letter(large, "ready"), "text": short})}
    sockets = []

    def connect(socket):
        sockets.append(socket)
        socket.send(json.dumps({"type": "heartbeat"}))

    page = ui["new_page"](viewport=OWNER_WINDOW, color_scheme="dark")
    page.route_web_socket("**/ws", connect)
    page.route("**/api/update/status", lambda route: route.fulfill(content_type="application/json",
                                                                   body=json.dumps(current["status"])))
    page.add_init_script(f"({COUNT_STATUS_READS})()")

    def shot(name):
        path = output / f"{name}-{engine}.png"
        page.screenshot(path=str(path))
        print(f"UPDATE_DESCRIPTION_SCREENSHOT={path}")

    def show(where):
        page.click(f'[data-nav-page="{where}"]')
        page.wait_for_selector(f"#page-{where}.active")
        page.evaluate(FRAMES)

    def letter(step):
        seen = page.evaluate(LETTER)
        print(f"UPDATE_DESCRIPTION_LETTER {engine} {step} {json.dumps(seen, ensure_ascii=False)}")
        return seen

    def shows(text):
        page.wait_for_function("t => document.querySelector('#updates-letter-body').textContent.startsWith(t)",
                               arg=text[:40])

    def saved_while_away(text):
        show("chat")
        current["status"] = _status(large, {**_letter(large, "ready"), "text": text})
        sockets[-1].send(SAVED)
        shows(text)
        assert page.evaluate("document.querySelector('#updates-letter-body').clientHeight") == 0, "rendered off-page"
        show("dashboard")

    page.goto(ui["url"] + "/")
    page.wait_for_selector("#page-chat")
    show("dashboard")
    page.click('[data-dashboard-tab="updates"]')
    shows(short)
    page.evaluate(FRAMES)
    assert letter("fresh-short")["more"] is None, "a short text read on entry offers no button"

    # 1. The long description is saved while the owner is in Chat; the sidebar brings the Dashboard back on Updates.
    saved_while_away(large["text"])
    shot("01-returned")
    returned = letter("returned")
    assert returned["more"] == "Read all" and returned["content"] > returned["height"] > 0, returned

    # 2. Read all opens the whole text in place, to its last paragraph.
    page.locator("#updates-letter-more").click()
    page.evaluate("document.querySelector('#updates-letter-body').lastElementChild.scrollIntoView({block: 'center'})")
    page.evaluate(FRAMES)
    shot("02-read-all-end")
    opened = letter("read-all")
    assert opened["open"] and opened["more"] == "Show less" and opened["content"] <= opened["height"] + 1, opened
    assert opened["last"] == large["text"].split("\n\n")[-1] and opened["lastInView"], opened

    # 3. A passive refresh of the same text keeps the open text's very nodes and the owner's selection in it.
    page.evaluate(SELECT_LAST)
    page.wait_for_function(QUIET)
    read = page.evaluate("statusReads.read")
    sockets[-1].send(SAVED)
    page.wait_for_function(f"statusReads.read >= {read + 2} && {QUIET}")
    page.evaluate(FRAMES)
    assert page.evaluate(KEPT), "the refresh kept the open DOM and the selection"
    assert letter("refreshed")["more"] == "Show less"

    # 4. Show less folds it; switching subtabs away and back keeps the offer.
    page.locator("#updates-letter-more").click()
    folded = letter("show-less")
    assert not folded["open"] and folded["more"] == "Read all" and folded["content"] > folded["height"], folded
    page.click('[data-dashboard-tab="logs"]')
    assert page.locator("#dashboard-panel-logs").is_visible() and page.locator("#dashboard-panel-updates").is_hidden()
    page.click('[data-dashboard-tab="updates"]')
    page.wait_for_function(QUIET)  # the subtab re-reads the status; the same text keeps its nodes
    page.evaluate(FRAMES)
    assert letter("subtabs")["more"] == "Read all"

    # 5. A short text saved the same way offers no button.
    saved_while_away(short)
    shot("03-short-returned")
    back = letter("short-returned")
    assert back["more"] is None and back["content"] <= back["height"] + 1, back

    # 6. A narrower window (or interface size) folds some of it: Read all comes, and goes at the owner's width again.
    # A phone's column folds it. There a Dashboard brought back by the sidebar may give its page up for its list (each
    # showing resets the open page, dashboard.js onPageShown; only a row pressed in the visit keeps it); the owner then
    # opens Updates from the list. Either way the letter is read on Updates.
    page.set_viewport_size({"width": 320, "height": OWNER_WINDOW["height"]})
    page.evaluate(FRAMES)
    if not page.locator("#dashboard-panel-updates").is_visible():
        page.click('[data-dashboard-tab="updates"]')
        page.wait_for_selector("#dashboard-panel-updates", state="visible")
        page.evaluate(FRAMES)
    shot("04-short-narrow")
    narrow = letter("short-narrow")
    assert narrow["more"] == "Read all" and narrow["content"] > narrow["height"], narrow
    page.set_viewport_size(OWNER_WINDOW)
    page.evaluate(FRAMES)
    assert letter("short-owner-width")["more"] is None

    # 7. On a phone the Dashboard's list stands in place of its page. A long description saved while the list shows
    # is measured once the owner opens Updates from the list: pressing the row that is already chosen changes no tab
    # (dashboard.js openLevel), so the body's own size is the door, and Read all is offered.
    page.set_viewport_size({"width": 320, "height": OWNER_WINDOW["height"]})
    page.evaluate(FRAMES)
    if page.locator("#dashboard-panel-updates").is_visible():
        page.get_by_role("button", name="Back to Dashboard", exact=True).click()
        page.wait_for_selector("#dashboard-panel-updates", state="hidden")
    current["status"] = _status(large, {**_letter(large, "ready"), "text": large["text"]})
    sockets[-1].send(SAVED)
    shows(large["text"])
    page.evaluate(FRAMES)
    assert page.evaluate("document.querySelector('#updates-letter-body').clientHeight") == 0, "rendered behind the list"
    page.click('[data-dashboard-tab="updates"]')
    page.wait_for_selector("#dashboard-panel-updates", state="visible")
    page.evaluate(FRAMES)
    shot("05-long-from-list")
    from_list = letter("long-from-list")
    assert from_list["more"] == "Read all" and from_list["content"] > from_list["height"] > 0, from_list


@pytest.mark.parametrize("channel", ["stable", "qa", "development"])
def test_applied_update_button_admits_retained_range(subscription_ui, door, tmp_path, channel):
    """Real SPA button → gateway → admitted task → its save, on a merged install after a level check: the serving
    merge holds the official target, so the checked pair names nothing incoming and the section's action is Write
    for the retained material. Older prose reads "Earlier update"; the text saved for the material reads neutrally
    until the next check refreshes its ancestry, with Rewrite in Details; with no text, Details name the range and
    no offer."""
    from ouroboros import update_letter as ul
    from tests.test_update_description import TARGET, NEWER, _request, _finish

    ui = subscription_ui
    merge = "d" * 40
    recent = {"base_sha": TARGET, "target_sha": NEWER, "update_channel": channel,
              "target_ref": {"development": "managed/ouroboros", "qa": "managed/ouroboros-stable",
                             "stable": "managed/main"}[channel]}
    # What the level check recorded: the pair M..N and, for the older text it showed, that its target is inside M.
    level = {"checked_key": {**recent, "base_sha": merge}, "material": {"key": recent, "text_bytes": 123},
             "checked_running_sha": merge, "target_in_running": True, "target_in_for": TARGET}
    _offer(door.root, state="ready", text="Older update: widgets and sharing.", revision=23, **level)
    status = {**_status(FIXTURE["small"], {}), "available": False, "check_ok": True,
              "current_sha": merge, "running_sha": merge, "latest_sha": "", "checked_target_sha": NEWER,
              "current_version": "7.11.1-alpha.0", "update_channel": channel, "target_ref": recent["target_ref"]}
    calls, sockets = [], []

    def respond(route):
        path = urlparse(route.request.url).path
        if path == "/api/update/description":
            posted = json.loads(route.request.post_data)
            calls.append(posted)
            response = door.client.post(path, json=posted)
            route.fulfill(status=response.status_code, content_type="application/json", body=response.text)
        else:
            projected = ul.project_letter_for_panel(status, drive_root=door.root)
            route.fulfill(content_type="application/json", body=json.dumps({**status, "letter": projected}))

    def connect(socket):
        sockets.append(socket)
        socket.send(json.dumps({"type": "heartbeat"}))

    def saved_shows(text):  # the server's own frame after a save; the status read brings the text
        sockets[-1].send(json.dumps({"type": "update_progress_changed"}))
        page.wait_for_function("t => document.querySelector('#updates-letter-body').textContent.includes(t)", arg=text)

    page = ui["new_page"](viewport={"width": 1422, "height": 1058}, color_scheme="dark")
    page.route_web_socket("**/ws", connect)
    page.route("**/api/update/status", respond)
    page.route("**/api/update/check", respond)
    page.route("**/api/update/description", respond)
    page.goto(ui["url"] + "/")
    page.click('[data-nav-page="dashboard"]')
    page.click('[data-dashboard-tab="updates"]')
    button, label, note = (page.locator(f"#updates-letter-{part}") for part in ("action", "label", "note"))
    provenance = page.locator("#updates-letter-provenance")
    button.wait_for()
    assert button.inner_text() == "Write description"
    assert label.inner_text() == "Earlier update"
    assert note.inner_text() == f"This description covers an earlier update ({BASE[:8]} → {TARGET[:8]})."
    assert "Older update" in page.locator("#updates-letter-body").inner_text()
    output = diagnostic_directory(tmp_path, "update-range", publish=True)
    page.screenshot(path=str(output / f"{channel}-retained.png"))
    assert door.pending == [] and calls == []
    button.click()
    # The headline's own action checks, so while the request writes the section adds no second check.
    page.wait_for_function("document.querySelector('#updates-letter-action').hidden")
    assert calls == [{"mode": "write", "base_sha": TARGET, "target_sha": NEWER}]
    [task] = door.pending
    assert task["metadata"]["update_description"]["key"] == recent
    assert f"{TARGET}..{NEWER}" in task["text"]
    assert ul.read_record(door.root)["text"] == "Older update: widgets and sharing."
    # The requested task saves the retained range; until the next check its ancestry still describes the older
    # text, so the heading is neutral, no note calls it earlier and Rewrite names its own range.
    saved = ul.save_description("What this update brings: a calmer Updates panel.", TARGET, NEWER, 23,
                                author={"task_id": task["id"], "model": "test"}, drive_root=door.root)
    assert saved["outcome"] == "saved"
    _finish(door.root, task["id"], "completed")
    saved_shows("a calmer Updates panel")
    page.wait_for_function("!document.querySelector('#updates-letter-rewrite').hidden")
    assert label.inner_text() == "What's new" and note.is_hidden() and button.is_hidden()
    assert f"Changes {TARGET[:8]} → {NEWER[:8]}." in provenance.text_content()
    page.screenshot(path=str(output / f"{channel}-saved.png"))
    # The public API still lets an owner intentionally rephrase the saved text.
    rephrase = _request(door, mode="rewrite", base=TARGET, target=NEWER)
    assert rephrase.status_code == 200
    _finish(door.root, rephrase.json()["task_id"], "completed")
    # With no text at all, Write still names the retained range and Details name it without claiming an offer.
    _offer(door.root, **level)
    sockets[-1].send(json.dumps({"type": "update_progress_changed"}))
    page.wait_for_function("document.querySelector('#updates-letter-action').textContent === 'Write description'")
    assert label.inner_text() == "What's new" and note.inner_text() == "No description of this update yet."
    assert provenance.text_content() == f"Changes {TARGET[:8]} → {NEWER[:8]}."
    page.screenshot(path=str(output / f"{channel}-letterless.png"))


@pytest.mark.parametrize("ws_first", [True, False])
@pytest.mark.parametrize("saved", [True, False])
def test_late_failed_check_keeps_the_writer_or_text_read_through_ws(subscription_ui, door, monkeypatch,
                                                                 ws_first, saved):
    """The two HTTP/WS orderings through the real gateway: a failed check's old payload cannot roll back
    another window's admitted writer or saved text. The reconciliation read may itself be unavailable."""
    from ouroboros import update_letter as ul
    from tests.test_update_description import _describe, _request

    _offer(door.root, state="ready", text="The earlier description.", revision=1)
    _checked(monkeypatch)
    ui, held, sockets, posts = subscription_ui, [], [], []
    fail_status = False

    def status(route):
        if fail_status:
            route.fulfill(status=503, content_type="application/json", body='{"error":"status unavailable"}')
        else:
            response = door.client.get("/api/update/status")
            route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

    def describe(route):
        posts.append(route.request.post_data_json)
        _checked(monkeypatch, check_ok=False)
        held.append((route, _describe(door)))
        _checked(monkeypatch)

    def connect(socket):
        sockets.append(socket)
        socket.send(json.dumps({"type": "heartbeat"}))

    page = ui["new_page"](viewport={"width": 1044, "height": 653})
    page.route_web_socket("**/ws", connect)
    page.route("**/api/update/status", status)
    page.route("**/api/update/description", describe)
    page.goto(ui["url"] + "/")
    page.click('[data-nav-page="dashboard"]')
    page.click('[data-dashboard-tab="updates"]')
    page.locator("#updates-letter-action").click()
    page.wait_for_function("document.querySelector('#updates-letter-action').textContent === 'Checking…'")
    for _ in range(100):
        if held:
            break
        page.wait_for_timeout(50)
    assert len(held) == 1
    route, answer = held.pop()
    assert answer.json()["status"] == "check_failed"

    def newer():
        task_id = _request(door, mode="rewrite").json()["task_id"]
        if saved:
            assert ul.save_description("The newer saved description.", BASE, TARGET, 1,
                author={"task_id": task_id, "model": "test"}, drive_root=door.root)["outcome"] == "saved"
        sockets[-1].send(json.dumps({"type": "update_progress_changed"}))
        page.wait_for_selector(f'#updates-letter-task [data-open-task="{task_id}"]')
        if saved:
            page.wait_for_function("document.querySelector('#updates-letter-body').textContent.includes('newer saved')")
        return task_id

    if ws_first:
        task_id = newer()
    fail_status = True
    route.fulfill(status=answer.status_code, content_type="application/json", body=answer.content)
    page.wait_for_function("document.querySelector('#updates-letter-action').textContent !== 'Checking…'")
    if not ws_first:
        fail_status = False
        task_id = newer()
    assert page.locator(f'#updates-letter-task [data-open-task="{task_id}"]').is_visible()
    assert task_id in page.locator("#updates-letter-note").inner_text()
    assert page.locator("#updates-letter-body").inner_text() == (
        "The newer saved description." if saved else "The earlier description.")
    assert "Last check failed" in page.locator("#updates-letter-error").text_content()
    assert len(door.pending) == 1 and posts == [{"mode": "latest"}]


@pytest.mark.parametrize("read_failure", ["http_503", "unreadable_200", "empty_object", "letter_record"])
@pytest.mark.parametrize(("outcome", "newer_first"), [
    ("queued", False), ("writing", False), ("current", False), ("queued", True), ("current", True)])
def test_confirmed_description_reply_survives_a_failed_passive_read(subscription_ui, door, monkeypatch, outcome,
                                                                 newer_first, read_failure):
    """A successful check/admission receipt is still known when its following GET fails. The receipt opens
    the actual task without claiming it is running; task_done can reconcile it even without a prior status."""
    from ouroboros import update_letter as ul
    from tests.test_update_description import NEWER, _describe, _finish, _request

    _offer(door.root, state="ready", text="The saved description.", revision=1,
           request={"task_id": "ended-writer", "mode": "write", "key": _key(), "revision_at_request": 0})
    _finish(door.root, "ended-writer", "completed")
    _checked(monkeypatch, target=TARGET if outcome == "current" else NEWER)
    ui, sockets, receipts, posts, held = subscription_ui, [], [], [], []
    fail_status = False

    def with_unread_letter(call, *args, **kwargs):
        path, read = ul.record_path(door.root), Path.read_text

        def read_file(target, *read_args, **read_kwargs):
            if target == path:
                raise PermissionError("letter unavailable on this status read")
            return read(target, *read_args, **read_kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(Path, "read_text", read_file)
            return call(*args, **kwargs)

    def status(route):
        if fail_status and read_failure != "letter_record":
            bodies = {"http_503": '{"error":"status unavailable"}', "unreadable_200": '{"letter":', "empty_object": '{}'}
            route.fulfill(status=503 if read_failure == "http_503" else 200, content_type="application/json",
                          body=bodies[read_failure])
        else:
            response = (with_unread_letter(door.client.get, "/api/update/status") if fail_status
                        else door.client.get("/api/update/status"))
            if fail_status:
                assert response.status_code == 200 and response.json()["letter_unavailable"] is True
                assert response.json()["available"] and response.json()["safe_to_apply"]
            route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

    def describe(route):
        posts.append(route.request.post_data_json)
        if outcome == "writing":  # another window wins admission while this press checks
            assert _request(door, mode="rewrite").status_code == 200
        if read_failure == "letter_record":
            project = ul.project_letter_for_panel

            with monkeypatch.context() as fault:
                fault.setattr(ul, "project_letter_for_panel", partial(with_unread_letter, project))
                response = _describe(door)
            assert response.json()["update"]["letter_unavailable"]
        else:
            response = _describe(door)
        receipts.append(response.json())
        held.append((route, response))

    def connect(socket):
        sockets.append(socket)
        socket.send(json.dumps({"type": "heartbeat"}))

    page = ui["new_page"](viewport={"width": 1044, "height": 653})
    page.route_web_socket("**/ws", connect)
    page.route("**/api/update/status", status)
    page.route("**/api/update/description", describe)
    page.goto(ui["url"] + "/")
    page.click('[data-nav-page="dashboard"]')
    page.click('[data-dashboard-tab="updates"]')
    page.locator("#updates-letter-action").click()
    for _ in range(100):
        if held:
            break
        page.wait_for_timeout(50)
    assert len(held) == 1
    route, response = held.pop()
    if newer_first:
        key = receipts[0]["key"]
        assert ul.save_description("Newer text already observed through WS.", key["base_sha"], key["target_sha"], 1,
            author={"task_id": receipts[0].get("task_id", "other-writer"), "model": "test"},
            drive_root=door.root)["outcome"] == "saved"
        sockets[-1].send(json.dumps({"type": "update_progress_changed"}))
        page.wait_for_function("document.querySelector('#updates-letter-body').textContent.includes('already observed')")
    fail_status = True
    route.fulfill(status=response.status_code, content_type="application/json", body=response.content)
    page.wait_for_function("document.querySelector('#updates-letter-note').textContent.includes('Current status could not be refreshed')")
    assert receipts[0]["status"] == outcome
    assert "Could not check" not in page.locator("#updates-summary").inner_text()
    assert "unknown" not in page.locator("#updates-letter-note").inner_text()
    assert page.locator("#updates-letter-body").inner_text() == (
        "Newer text already observed through WS." if newer_first else "The saved description.")
    primary = page.locator("#btn-update-primary")
    only_letter_failed = read_failure == "letter_record"
    assert page.locator("#btn-update-replace").is_disabled() == (not only_letter_failed)
    if only_letter_failed:
        assert primary.get_attribute("data-action") == "update" and primary.is_enabled()
    check = page.locator("#updates-letter-action") if only_letter_failed else primary
    if outcome == "current":
        assert check.inner_text() == "Check and describe" and check.is_enabled()
        assert page.locator("#updates-letter-task [data-open-task]").count() == 0
    else:
        task_id = receipts[0]["task_id"]
        assert page.locator(f'#updates-letter-task [data-open-task="{task_id}"]').is_visible()
        assert check.inner_text().startswith("Check") and "describe" not in check.inner_text()
        assert check.is_enabled() and page.locator("#updates-letter-rewrite").is_hidden()
        if outcome == "queued" and not newer_first:
            assert "is writing" not in page.locator("#updates-letter-note").inner_text(), "receipt is not a liveness fact"
            if read_failure == "empty_object":
                page.route("**/api/update/check", lambda r: r.fulfill(json={}), times=1)
                with page.expect_request(lambda req: urlparse(req.url).path == "/api/update/check" and req.method == "POST"):
                    check.click()
                page.wait_for_function("document.querySelector('#btn-update-primary').dataset.action === 'check'"
                                       " && !document.querySelector('#btn-update-primary').disabled")
                assert page.locator(f'#updates-letter-task [data-open-task="{task_id}"]').is_visible()
                assert posts == [{"mode": "latest"}]
        key = receipts[0]["key"]
        assert ul.save_description("The finished description.", key["base_sha"], key["target_sha"], 2 if newer_first else 1,
            author={"task_id": task_id, "model": "test"}, drive_root=door.root)["outcome"] == "saved"
        _finish(door.root, task_id, "completed")
        fail_status = False
        sockets[-1].send(json.dumps({"type": "log", "data": {"type": "task_done", "task_id": task_id}}))
        page.wait_for_function("document.querySelector('#updates-letter-body').textContent.includes('finished description')")
        assert "could not be refreshed" not in page.locator("#updates-letter-note").text_content()
    assert posts == [{"mode": "latest"}] and len(door.pending) == (0 if outcome == "current" else 1)



def test_channel_choice_keeps_the_existing_description_and_writer(subscription_ui, door, tmp_path):
    """Real channel control/save/refresh, with backend-shaped unchecked facts for the new selection:
    known history and the gateway-admitted writer remain visible; the control starts no new work."""
    from ouroboros import update_letter as ul
    from tests.test_update_description import _request

    ui, posts = subscription_ui, []
    channel_key = "OUROBOROS_UPDATE_CHANNEL"
    ui["settings"][channel_key] = "stable"
    _offer(door.root, state="ready", text="The saved description remains readable.", revision=1)
    task_id = _request(door, mode="rewrite").json()["task_id"]

    def settings(route):
        if route.request.method != "POST":
            route.fallback()
            return
        body = route.request.post_data_json
        posts.append(body)
        changes = {key: {"old": ui["settings"].get(key), "new": value, "token": f"{key}@1"}
                   for key, value in body.items()}
        ui["settings"].update(body)
        route.fulfill(content_type="application/json", body=json.dumps({"status": "saved", "changes": changes}))

    def status(route):
        chosen = ui["settings"][channel_key]
        data = {"managed": True, "check_ok": None, "available": chosen == "stable", "safe_to_apply": True,
                "running_sha": BASE, "current_sha": BASE, "current_version": "6.113.5",
                "update_channel": chosen, "target_ref": "managed/main" if chosen == "stable" else "managed/ouroboros",
                "latest_sha": TARGET if chosen == "stable" else "", "latest_version": "6.114.0",
                "warnings": [] if chosen == "stable" else ["official_status_requires_check"]}
        data["letter"] = ul.project_letter_for_panel(data, drive_root=door.root)
        route.fulfill(content_type="application/json", body=json.dumps(data))

    page = ui["new_page"](viewport={"width": 1044, "height": 653})
    page.route("**/api/settings", settings)
    page.route("**/api/update/status", status)
    page.goto(ui["url"] + "/")
    page.click('[data-nav-page="dashboard"]')
    page.click('[data-dashboard-tab="updates"]')
    page.locator(f'#updates-letter-task [data-open-task="{task_id}"]').wait_for()
    page.evaluate("window.keptLetter=document.querySelector('#updates-letter-body').firstChild")
    page.locator(f'[data-setting-row="{channel_key}"] [data-setting-value="development"]').click()
    page.wait_for_function("document.querySelector('#updates-summary').textContent.includes('has not been checked')")
    assert posts == [{channel_key: "development"}]
    assert page.locator("#updates-letter").is_visible()
    assert page.locator("#updates-letter-body").inner_text() == "The saved description remains readable."
    assert page.evaluate("window.keptLetter === document.querySelector('#updates-letter-body').firstChild")
    assert page.locator(f'#updates-letter-task [data-open-task="{task_id}"]').is_visible()
    assert f"{BASE[:8]} → {TARGET[:8]}" in page.locator("#updates-letter-note").inner_text()
    assert "selected channel has not been checked" in page.locator("#updates-letter-note").inner_text()
    assert page.locator("#btn-update-primary").inner_text() == "Check for updates"
    assert page.locator("#updates-letter-action").is_hidden() and len(door.pending) == 1
    output = diagnostic_directory(tmp_path, "update-description-channel", publish=True)
    page.screenshot(path=str(output / "unchecked-channel-keeps-writer.png"))


@pytest.mark.parametrize("mode", ["write", "rewrite"])
@pytest.mark.parametrize("read_failure", ["http_503", "unreadable_200"])
@pytest.mark.parametrize("newer_first", [False, True])
def test_named_description_keeps_admission_and_text_when_status_read_fails(subscription_ui, door, monkeypatch,
                                                                         mode, read_failure, newer_first):
    from ouroboros import update_letter as ul
    from tests.test_update_description import NEWER, _finish

    # After applying B, retained A..B still offers Write; the shown A..C text has its own Rewrite.
    _offer(door.root, state="ready", text="Saved text before the request.", revision=1, key=_key(BASE, NEWER))
    _checked(monkeypatch, base=TARGET, target=TARGET, available=False)
    ui, sockets, posts, held = subscription_ui, [], [], []
    fail_status = False

    def status(route):
        if fail_status:
            route.fulfill(status=503 if read_failure == "http_503" else 200, content_type="application/json",
                          body='{"error":"status unavailable"}' if read_failure == "http_503" else '{"letter":')
        else:
            response = door.client.get("/api/update/status")
            route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

    def request(route):
        posts.append(route.request.post_data_json)
        response = door.client.post("/api/update/description", json=posts[-1])
        held.append((route, response))

    def connect(socket):
        sockets.append(socket)
        socket.send(json.dumps({"type": "heartbeat"}))

    page = ui["new_page"](viewport={"width": 1044, "height": 653})
    page.route_web_socket("**/ws", connect)
    page.route("**/api/update/status", status)
    page.route("**/api/update/description", request)
    page.goto(ui["url"] + "/")
    page.click('[data-nav-page="dashboard"]')
    page.click('[data-dashboard-tab="updates"]')
    if mode == "rewrite":
        page.locator("#updates-letter-details summary").click()
        page.locator("#updates-letter-rewrite").click()
    else:
        page.wait_for_function("document.querySelector('#updates-letter-action')?.textContent === 'Write description'")
        assert page.locator("#updates-letter-action").inner_text() == "Write description"
        page.locator("#updates-letter-action").click()
    for _ in range(100):
        if held:
            break
        page.wait_for_timeout(50)
    assert len(held) == 1
    route, response = held.pop()
    assert response.status_code == 200, response.text
    receipt = response.json()
    task_id, key = receipt["task_id"], receipt["key"]
    assert receipt["status"] == "queued"
    if newer_first:
        assert ul.save_description("New text already seen through WS.", key["base_sha"], key["target_sha"], 1,
            author={"task_id": task_id, "model": "test"}, drive_root=door.root)["outcome"] == "saved"
        sockets[-1].send(json.dumps({"type": "update_progress_changed"}))
        page.wait_for_function("document.querySelector('#updates-letter-body').textContent.includes('already seen')")
    fail_status = True
    route.fulfill(status=200, content_type="application/json", body=response.content)
    page.wait_for_function("document.querySelector('#updates-letter-note').textContent.includes('Current status could not be refreshed')")
    assert page.locator(f'#updates-letter-task [data-open-task="{task_id}"]').is_visible()
    assert page.locator("#updates-letter-body").inner_text() == (
        "New text already seen through WS." if newer_first else "Saved text before the request.")
    assert "Could not check" not in page.locator("#updates-summary").inner_text()
    assert page.locator("#btn-update-primary").get_attribute("data-action") == "check"
    assert page.locator("#btn-update-primary").is_enabled() and page.locator("#updates-letter-rewrite").is_hidden()
    assert page.locator("#btn-update-replace").is_disabled()
    if not newer_first:
        assert "is writing" not in page.locator("#updates-letter-note").inner_text()
    assert posts == [{"mode": mode, "base_sha": BASE, "target_sha": TARGET if mode == "write" else NEWER}]
    assert len(door.pending) == 1
    _finish(door.root, task_id, "completed")
    fail_status = False
    sockets[-1].send(json.dumps({"type": "log", "data": {"type": "task_done", "task_id": task_id}}))
    page.wait_for_function("!document.querySelector('#updates-letter-note').textContent.includes('could not be refreshed')")
    assert len(door.pending) == 1 and len(posts) == 1


@pytest.mark.parametrize("mode", ["write", "rewrite"])
@pytest.mark.parametrize("read_failure", ["http_503", "unreadable_200"])
def test_named_description_says_an_unknown_outcome_when_its_answer_is_unreadable(subscription_ui, door, monkeypatch,
                                                                                    mode, read_failure):
    """A named Write/Rewrite whose answer cannot be read is an unknown outcome: the press says so, the old text
    stays, and the writer the server really admitted is neither doubled nor claimed as a refusal."""
    from tests.test_update_description import NEWER

    _offer(door.root, state="ready", text="Saved text before the request.", revision=1, key=_key(BASE, NEWER))
    _checked(monkeypatch, base=TARGET, target=TARGET, available=False)
    ui, sockets, posts, held = subscription_ui, [], [], []
    fail_status = False

    def status(route):
        if fail_status:
            route.fulfill(status=503 if read_failure == "http_503" else 200, content_type="application/json",
                          body='{"error":"status unavailable"}' if read_failure == "http_503" else '{"letter":')
        else:
            response = door.client.get("/api/update/status")
            route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

    def request(route):
        posts.append(route.request.post_data_json)
        response = door.client.post("/api/update/description", json=posts[-1])
        held.append((route, response))

    def connect(socket):
        sockets.append(socket)
        socket.send(json.dumps({"type": "heartbeat"}))

    page = ui["new_page"](viewport={"width": 1044, "height": 653})
    page.route_web_socket("**/ws", connect)
    page.route("**/api/update/status", status)
    page.route("**/api/update/description", request)
    page.goto(ui["url"] + "/")
    page.click('[data-nav-page="dashboard"]')
    page.click('[data-dashboard-tab="updates"]')
    if mode == "rewrite":
        page.locator("#updates-letter-details summary").click()
        page.locator("#updates-letter-rewrite").click()
    else:
        page.wait_for_function("document.querySelector('#updates-letter-action')?.textContent === 'Write description'")
        page.locator("#updates-letter-action").click()
    # Every read after the press fails, so the press is left with its own answer alone.
    fail_status = True
    for _ in range(100):
        if held:
            break
        page.wait_for_timeout(50)
    assert len(held) == 1
    route, response = held.pop()
    assert response.status_code == 200, response.text
    receipt = response.json()
    assert receipt["status"] == "queued", "the server admitted its writer; only the answer is unreadable"
    # The writer is admitted, the answer arrives cut in half: the press cannot know the outcome.
    body = response.content.decode("utf-8")
    route.fulfill(status=200, content_type="application/json", body=body[: len(body) // 2])
    page.wait_for_function("document.querySelector('#updates-letter-note').textContent.includes('no confirmation')")
    note = page.locator("#updates-letter-note").inner_text()
    assert "whether it requested a description is unknown" in note
    assert "Could not request a description" not in note, "an unreadable answer is not a refusal the server stated"
    assert page.locator("#updates-letter-body").inner_text() == "Saved text before the request."
    assert "Could not check" not in page.locator("#updates-summary").inner_text()
    assert page.locator("#updates-letter-task [data-open-task]").count() == 0, "an unknown outcome names no task"
    assert posts == [{"mode": mode, "base_sha": BASE, "target_sha": TARGET if mode == "write" else NEWER}]
    assert len(door.pending) == 1, "one writer, the one the server admitted; nothing is requested twice"

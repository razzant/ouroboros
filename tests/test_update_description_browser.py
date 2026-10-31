"""Real Dashboard › Updates and the legacy Apply confirmation with the update description.

The page is the shipped SPA modules (dashboard.js, updates.js, update_status.js, confirm_dialog.js,
chat_markdown.js) served from ``web/``; only the API replies are synthetic, derived from the real
``1b5608..fcd6`` and ``#2367`` ranges (``tests/fixtures/update_description_ranges.json``: receipts
parsed by the production collector, post-merge runs from a retained live read). No runtime, task or
model starts. Screens land in the test-evidence directory for inspection.

Covered: an update on offer with nothing written offers Write beside Check; Write posts the exact range
once and the request's task shows as writing, with the owner's fixed request time and a door that opens
that task's card in Main (kept through a passive refresh, gone once it saved; where it lands is the last
test, through app.js and the real router); Apply opened meanwhile pins only its own prose, so the
worker's ``update_progress_changed`` and the task's ``task_done`` behind it still read the status and
Cancel shows the large structured description rendered as Markdown; Check never authors and keeps the
read text's DOM; the Apply confirmation renders a large and a small description in a bounded region,
keeps the consequences in its body at the prose's size and ink (ordinary dialogs keep theirs) and every
per-commit receipt inside collapsed Details, and cancels by Escape and by Cancel without applying.
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
from tests.test_update_letter_browser import WEB
from tests.test_update_description import BASE, TARGET, _key, _no_git_for_versions, _offer, door  # noqa: F401 — fixtures
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
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


@pytest.mark.parametrize(("engine", "width", "height"), [("webkit", 1165, 851), ("chromium", 1165, 851),
                                                          ("webkit", 390, 844)])
def test_write_check_and_apply_read_the_description_as_prose(engine, width, height, tmp_path):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    output = diagnostic_directory(tmp_path, "update-description")
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
            current["status"] = _status(large, _letter(large, "writing"))
            body = {"status": "queued", "task_id": TASK, "key": large["key"]}
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

                # 1. An update on offer with nothing written: the section stays, with Check and Write.
                page.wait_for_selector("#updates-letter:not([hidden])")
                assert page.locator("#updates-letter-note").inner_text() == "No description of this update yet."
                write = page.locator("#updates-letter-write")
                assert write.is_visible() and write.inner_text() == "Write description" and write.is_enabled()
                assert page.locator("#updates-letter-refresh-label").inner_text() == "Check"
                assert page.locator("#btn-update-primary").inner_text() == "Update to 7.8.1"
                shot(page, "01-empty")

                # 2. Write posts the exact offered range once; the request's own task shows as writing.
                write.click()
                page.wait_for_function("document.querySelector('#updates-letter-write').textContent === 'Writing…'")
                assert posted("/api/update/description") == [{"mode": "write", "base_sha": large["key"]["base_sha"],
                                                              "target_sha": large["key"]["target_sha"]}]
                assert page.locator("#updates-letter-write").is_disabled()
                # The panel's one time formatter, as this engine writes it ("28 Oct, 11:20" / "28 Oct at 11:20").
                asked = page.evaluate("import('/static/modules/update_status.js')"
                                      ".then(m => m.attemptTime('2026-10-28T11:20:00Z'))")
                assert "28 Oct" in asked and "11:20" in asked, asked
                assert page.locator("#updates-letter-note").inner_text() == (
                    f"Ouroboros is writing the description you asked for on {asked} (task {TASK}).")
                assert posted("/api/update/check") == [], "Write is never a check"
                door = page.locator("#updates-letter-task button")
                assert door.inner_text() == "Open in Main ↗" and door.is_visible()
                line = page.locator("#updates-letter-note-line")
                assert line.evaluate("el => el.scrollWidth <= el.clientWidth"), "the line wraps, never widens"
                assert page.evaluate("document.documentElement.scrollWidth<=document.documentElement.clientWidth")
                shot(page, "02-writing")
                line.screenshot(path=str(output / f"02-writing-note-{engine}-{width}.png"))
                # The keyboard reaches it next after Check (Writing… is disabled; WebKit's plain Tab skips
                # buttons as macOS does, Option+Tab reaches every control). A passive refresh while it still
                # writes keeps the very door node, and with it focus.
                page.locator("#updates-letter-refresh").focus()
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
                current["status"] = _status(large, _letter(large, "ready"))
                for frame in ("window.__wsHandlers.get('update_progress_changed')()",
                              f"window.__wsHandlers.get('log')({{data: {{type: 'task_done', task_id: '{TASK}'}}}})"):
                    before = status_reads()
                    page.evaluate(frame)
                    wait_until(lambda: status_reads() > before, frame)
                dialog.get_by_role("button", name="Cancel", exact=True).click()
                dialog.wait_for(state="detached")
                current["preflight"] = _preflight(large)
                page.wait_for_selector("#updates-letter-body h2, #updates-letter-body .md-h2")
                body = page.locator("#updates-letter-body")
                rendered = body.inner_text()
                assert "Что будет заметно сразу" in rendered and "Telegram." in rendered
                assert "##" not in rendered and "**" not in rendered, "Markdown renders, never shows raw"
                assert body.locator("li").count() >= 12
                assert page.locator("#updates-letter-label").inner_text() == "What's new"
                covers = page.locator("#updates-letter-covers").inner_text()
                assert f"{large['key']['base_sha'][:8]} → {large['key']['target_sha'][:8]}" in covers
                assert page.locator("#updates-letter-write").inner_text() == "Rewrite"
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
                page.locator("#updates-letter-details summary").click()

                # 4. Check is facts only: a refreshed run leaves the read text's very DOM node in place.
                page.evaluate("window.savedBody=document.querySelector('#updates-letter-body').firstChild")
                ready = _status(large, _letter(large, "ready"))
                ready["letter"]["qualification"] = copy.deepcopy(large["qualification"])
                ready["letter"]["qualification"]["github"]["runs"][0]["conclusion"] = "failure"
                current["status"] = ready
                page.locator("#updates-letter-refresh").click()
                page.wait_for_selector(".toast")
                assert page.locator(".toast").first.inner_text() == "Description is up to date."
                assert page.evaluate("window.savedBody===document.querySelector('#updates-letter-body').firstChild")
                assert len(posted("/api/update/check")) == 1 and len(posted("/api/update/description")) == 1

                # 5. Apply: the large description reads as prose in its own region; evidence stays in Details.
                page.locator("#btn-update-primary").click()
                dialog = page.get_by_role("dialog")
                dialog.wait_for()
                region = dialog.locator("[data-confirm-content]")
                assert region.get_attribute("aria-label") == "What's new"
                region_text = region.inner_text()
                assert "Что будет заметно сразу" in region_text and "##" not in region_text and "**" not in region_text
                assert region.locator("li").count() >= 12
                plain = dialog.locator(".marketplace-modal-body > p").inner_text()
                for fact in ("Git applies this update directly", "Claudexor: 4.8.0 → 4.8.0", "forward-only"):
                    assert fact in plain, fact
                details = dialog.locator("details.confirm-dialog-details")
                assert details.locator("summary").inner_text() == "Checks for this range"
                assert details.get_attribute("open") is None, "per-commit receipts stay collapsed"
                # Beside the prose, the consequences and the Details summary read at its size and ink.
                prose = region.locator(".updates-letter-body").evaluate(INK)
                assert prose[0] == "14px" and dialog.locator(".marketplace-modal-body > p").evaluate(INK) == prose
                assert details.locator("summary").evaluate(INK) == prose
                assert "post-merge" not in plain and "full_battery" not in plain and "receipt" not in plain
                box = region.bounding_box()
                assert box["height"] <= min(0.42 * height, 380) + 2, box
                actions = dialog.locator(".marketplace-modal-actions").bounding_box()
                assert actions["y"] + actions["height"] <= height, "the actions stay in view beside long prose"
                shot(page, "04-apply-large")
                details.locator("summary").click()
                rows = details.locator(".confirm-dialog-detail-row")
                assert rows.count() == len(large["qualification"]["commits"]) + 1 + len(large["qualification"]["github"]["runs"]) + 1
                assert "Merge pull request #2384" in details.inner_text() and "No matching run observed" in details.inner_text()
                rows.nth(3).scroll_into_view_if_needed()
                shot(page, "05-apply-large-details")
                # Keyboard: the prose region takes focus and scrolls; Escape cancels without applying.
                region.focus()
                assert page.evaluate("document.activeElement.matches('[data-confirm-content]')")
                page.keyboard.press("End")
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

                # 6. A small description reads the same way; Cancel closes without applying.
                current["preflight"] = _preflight(small)
                current["status"] = _status(small, _letter(small, "ready"))
                page.evaluate("window.__wsHandlers.get('update_progress_changed')()")
                page.wait_for_function(f"document.querySelector('#updates-letter-covers').textContent.includes('{small['key']['target_sha'][:8]}')")
                page.locator("#btn-update-primary").click()
                dialog.wait_for()
                small_text = dialog.locator("[data-confirm-content]").inner_text()
                assert "Небольшое исправление Dashboard" in small_text and "**" not in small_text
                shot(page, "06-apply-small")
                dialog.get_by_role("button", name="Cancel", exact=True).click()
                dialog.wait_for(state="detached")
                assert posted("/api/update/apply") == []
                # An ordinary dialog keeps its own text: only a dialog carrying prose steps its facts down.
                page.evaluate("import('/static/modules/confirm_dialog.js')"  # the dialog's own promise stays open
                              ".then((m) => { m.openConfirmDialog({title: 'Plain', body: 'An ordinary question.'}); })")
                dialog.wait_for()
                assert dialog.locator(".marketplace-modal-body > p").evaluate(
                    "el => getComputedStyle(el).fontSize === getComputedStyle(el.parentElement).fontSize")
                page.keyboard.press("Escape")
                dialog.wait_for(state="detached")
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
# from app.js verbatim. Only ``showPage`` stands in for the shell: it switches the page and announces it.
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
new Function('apiClient','mainChat','state','showPage','closeProjectPanel','showToast','tr','fetchJson','openProjectPanel',
    'let projectNavigationGeneration = 0;\\n'+door)(apiClient,mainChat,state,showPage,()=>{},text=>notices.push(text),tr,noProject,noProject);
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


@pytest.mark.parametrize(("engine", "width", "height", "owner_chat", "lineage"), [
    ("webkit", 1165, 851, None, ""), ("chromium", 1165, 851, None, ""), ("webkit", 390, 844, None, ""),
    ("chromium", 1165, 851, 1, ""), ("chromium", 1165, 851, 1, "retry"), ("chromium", 1165, 851, 1, "continue")])
def test_the_writing_note_opens_main_s_card_before_its_first_message(door, engine, width, height, owner_chat, lineage,
                                                                      tmp_path):
    """Write pressed before any owner message (the binding known empty) or after the web owner bound Main:
    the request is Main's work, its current task (the first one, a host retry or an admitted Continue) calls
    a tool before it says anything, and the note's door lands on the card Main draws for that task, focused
    under the header, without a history read (around-history has no row of it yet and answers absent)."""
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    from ouroboros import update_letter as ul
    from ouroboros.task_results import write_task_result
    from ouroboros.tool_call_log import append_call_row, invocation_fields, new_invocation
    from supervisor import queue, state

    state.update_state(lambda live: live.update(owner_chat_id=owner_chat))
    _offer(door.root)
    output = diagnostic_directory(tmp_path, "update-description-door")
    case = {"key": _key(), "commits": 1, "version": "6.114.0", "qualification": None}
    history_reads = []

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
                page.locator("#updates-letter-write").click()
                door_button = page.locator("#updates-letter-task button")
                door_button.wait_for()
                [task] = door.pending
                assert task["chat_id"] == 1, "the panel's Write is Main's work, even before any owner message"
                task_id = task["id"]
                assert task_id in page.locator("#updates-letter-note").inner_text()

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

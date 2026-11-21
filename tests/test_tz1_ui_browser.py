"""Exercise TZ-1 owner history/readiness presentation in the real SPA."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.test_subscription_setup_browser import subscription_ui as subscription_ui, capture
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

pytestmark = [pytest.mark.ui_browser]


def test_saved_input_and_starting_survive_reload(subscription_ui):
    ui = subscription_ui
    page = ui["page"]
    rows = [{
        "role": "user", "text": "TZ-1 saved message", "ts": "2026-09-25T03:00:00Z",
        "client_message_id": "tz1-saved-message", "ingress_accepted": True,
    }]
    page.route("**/api/state*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({
            "supervisor_ready": False, "supervisor_error": "still starting",
            "active_chat_activities": [], "active_chat_activities_complete": True,
            "projects": [],
        }),
    ))
    page.route("**/api/chat/history*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({"messages": rows, "progress": []}),
    ))
    page.goto(ui["url"])
    saved = page.locator('.chat-bubble.user[data-client-message-id="tz1-saved-message"] [data-ingress-saved]')
    saved.wait_for()
    assert saved.inner_text() == "Input saved"
    assert page.get_by_text("Starting…", exact=True).count() >= 1
    capture(page, "tz1-saved-and-starting")
    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    saved.wait_for()
    assert saved.count() == 1
    assert page.get_by_text("Starting…", exact=True).count() >= 1
    assert not ui["errors"].unexpected(), ui["errors"].report()


def test_failed_supervisor_init_never_paints_online(subscription_ui):
    """The host's failure rail publishes `supervisor_ready: false` beside the
    error; the real SPA header says Starting… and never Online over it."""
    ui = subscription_ui
    page = ui["page"]
    page.route("**/api/state*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({
            "supervisor_ready": False, "supervisor_error": "Supervisor init failed: boot dependency refused",
            "active_chat_activities": [], "active_chat_activities_complete": True, "projects": [],
        }),
    ))
    page.route("**/api/chat/history*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({"messages": [], "progress": []}),
    ))
    page.goto(ui["url"])
    badge = page.locator("#chat-status")
    badge.wait_for()
    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Starting…'")
    assert "online" not in (badge.get_attribute("class") or "").split()
    assert page.get_by_text("Online", exact=True).count() == 0
    capture(page, "tz1-failed-init-starting")
    assert not ui["errors"], ui["errors"]


def test_recorded_folder_download_from_real_gateway_in_chat(subscription_ui, tmp_path):
    """The served SPA reads real task detail and ZIP bytes from the gateway,
    rather than an authored HTML fragment or a fake detail response."""
    import io
    import zipfile

    from tests.test_task_file_serving import TASK, _client, _split_child
    from ouroboros import headless

    data, child, _store = _split_child(tmp_path)
    headless.copy_child_task_result(data, {"id": TASK, "drive_root": str(child)})
    client = _client(data)
    subscription_ui["backend"]["task_gateway"] = client
    page = subscription_ui["page"]
    url_prefix = f"/api/tasks/{TASK}"
    seen = []

    def gateway(route):
        from urllib.parse import urlparse
        path = urlparse(route.request.url).path
        suffix = route.request.url.split(path, 1)[1]
        response = client.get(path + suffix)
        seen.append((path, response.status_code))
        route.fulfill(status=response.status_code,
                      headers={"content-type": response.headers.get("content-type", "application/json"),
                               "content-disposition": response.headers.get("content-disposition", "")},
                      body=response.content)

    page.route(f"**{url_prefix}*", gateway)
    page.goto(subscription_ui["url"])
    # The event is only an activation fixture. The browser's task-detail read,
    # archive availability and downloaded bytes all come from the real gateway.
    page.evaluate("row => window.__ouroWs.emit('chat', row)", {
        "chat_id": 1, "task_id": TASK, "role": "assistant", "is_progress": True,
        "content": "Building reports", "ts": "2026-09-25T07:00:00Z",
    })
    page.evaluate("row => window.__ouroWs.emit('chat', row)", {
        "chat_id": 1, "task_id": TASK, "role": "system", "system_type": "task_summary",
        "task_terminal_status": "completed", "content": "Done", "ts": "2026-09-25T07:01:00Z",
    })
    card = page.locator(f'#chat-messages .chat-live-card[data-task-id="{TASK}"]')
    card.wait_for()
    assert card.locator('[data-live-phase]').inner_text() == "Done"
    card.locator('[data-live-disclosure]').click()
    page.wait_for_selector(f'#chat-messages .chat-live-card[data-task-id="{TASK}"] [data-result-files]')
    folder = card.locator('[data-result-files] a[href*="?archive=reports"]')
    assert folder.count() == 1 and "reports/" in folder.inner_text()
    with page.expect_download() as item:
        folder.click()
    assert item.value.failure() is None, (item.value.failure(), seen, subscription_ui["errors"])
    payload = item.value.path().read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.namelist() == ["reports/a/summary.txt", "reports/b/summary.txt"]
        assert archive.read("reports/a/summary.txt") == b"alpha"
    assert (url_prefix, 200) in seen, seen  # real detail, not a fixture JSON object
    capture(page, "tz1-real-folder-download")
    assert not subscription_ui["errors"], subscription_ui["errors"]


# The boot documents below keep the step names and states a real candidate server published while it started
# (.review-evidence of #3376: state-capture.json); `test_boot_readiness` binds every named step to its producer.
def _steps(*running, failed=()):
    return [{"name": name, "state": "running"} for name in running] + [
        {"name": name, "state": "failed", "error": error} for name, error in failed]


CHAT_CLOSED = {"phase": "serving", "dialogue_ready": False, "tasks_ready": False,
               "steps": _steps("skills", "dialogue_cutover", "startup:event_queue")}
CONCURRENT = {"phase": "serving", "dialogue_ready": True, "tasks_ready": False,
              "steps": _steps("startup:event_queue", "extensions")}
SOFT_FAILED = {"phase": "serving", "dialogue_ready": True, "tasks_ready": False,
               "steps": _steps("supervisor_outcome", "supervisor", "startup:bootstrap", failed=[
                   ("startup:event_queue", "OSError: the first event-bus Queue() RPC is refused")])}
FAILED = {"phase": "failed", "dialogue_ready": True, "tasks_ready": False, "error":
          "Supervisor init failed: the first event-bus Queue() RPC is refused", "steps": _steps()}
SAVED = "Messages are saved and answered once the start finishes."
SAVED_ROW = {"role": "user", "text": "Answer in one line: are you there?", "ts": "2026-11-15T13:55:38.238874+00:00",
             "client_message_id": "m2-saved", "ingress_accepted": True, "ingress_dispatched": True, "chat_id": 1,
             "history_id": "chat:0", "history_position": {"offset": 0, "source": "chat"}, "source": "web",
             "row_address": "row:1@2026-11-15T13:55:38.238874+00:00#a165f95697f6"}


def _history(rows):
    """The history answer's shape as the server sent it while starting: one complete dialog window."""
    span = {"chain": "dialog", "from": 0, "gaps": []}
    return {"messages": rows, "progress": [], "has_more": False, "next_cursor": None, "page_cursor": "m2",
            "window": {"complete": True, "truncated_by": []},
            "coverage": {"v": 1, "view": "m2", "upper": {"chat": len(rows), "progress": 0},
                         "spans": {"chat": {**span, "to": len(rows)}, "progress": {**span, "to": 0}}}}


def test_completed_card_files_distinguish_byte_names_after_reload(subscription_ui, tmp_path):
    """Replay an ordinary result in the SPA; detail and the regular download use the real gateway."""
    from playwright.sync_api import expect

    from ouroboros.artifacts import task_artifacts_dir
    from ouroboros.task_results import write_task_result
    from tests._task_file_http import task_file_client

    task_id = "result-byte-names"
    data = tmp_path / "data"
    directory = task_artifacts_dir(data, task_id)
    payload = b"Verified browser result.\n"
    (directory / "report.txt").write_bytes(payload)
    # Synthetic records model the wire names even on filesystems that cannot create them.
    names = [b"report-\xff.txt".decode("utf-8", "surrogateescape"),
             b"report-\xfe.txt".decode("utf-8", "surrogateescape"), r"report-\xff.txt", "report.txt"]
    write_task_result(data, task_id, "completed", result="The report is ready.", artifacts=[
        {"kind": "task_artifact", "name": name, "relpath": name, "path": str(directory / name), "status": "ready"}
        for name in names
    ])
    root_row = {"chat_id": 1, "task_id": task_id, "suggested_name": "Saved report"}
    rows = [
        {**root_row, "role": "assistant", "is_progress": True, "narration": True, "text": "Saving the report",
         "ts": "2026-09-25T07:00:00Z", "history_id": f"progress-{task_id}"},
        {**root_row, "role": "system", "system_type": "task_summary", "text": "Done", "status": "completed",
         "task_terminal_status": "completed", "ts": "2026-09-25T07:01:00Z", "history_id": f"end-{task_id}"},
    ]
    ui, page = subscription_ui, subscription_ui["page"]
    ui["backend"]["task_gateway"] = task_file_client(data)
    detail_url = f"{ui['url']}/api/tasks/{task_id}"
    # Let these two reads reach the fixture's loopback HTTP server, which delegates to the real task endpoints.
    for url in (f"{detail_url}*", f"{detail_url}/artifacts/report.txt"):
        page.route(url, lambda route: route.continue_())
    page.route("**/api/chat/history*", lambda route: route.fulfill(json=_history(rows)))
    card = page.locator(f'#chat-messages .chat-live-card[data-task-id="{task_id}"]')
    files = card.locator('[data-result-files]')
    link = files.get_by_role("link", name="report.txt", exact=True)

    for visit in ("first", "reload"):
        with page.expect_response(lambda response: response.url.split("?", 1)[0] == detail_url) as detail:
            if visit == "first":
                page.goto(ui["url"])
            else:
                ui["errors"].watch(page)
                page.reload()
            expect(card).to_have_attribute("data-finished", "1")
            expect(card.locator('[data-live-phase]')).to_have_text("Done")
            door = card.locator(':scope > [data-live-summary-button] [data-live-disclosure]')
            if door.get_attribute("aria-expanded") != "true":
                door.click()
            expect(files).to_be_visible()
        assert detail.value.status == 200
        expect(files).to_contain_text("4 files")
        for label in (r"report-\xff.txt", r"report-\xfe.txt", r"report-\\xff.txt"):
            row = files.locator('.md-li').filter(has_text=label)
            expect(row).to_be_visible()
            expect(row).to_contain_text("download unavailable")
            expect(row.locator("a")).to_have_count(0)
        expect(files.get_by_role("link")).to_have_count(1)
        expect(link).to_be_visible()

    with page.expect_download() as download:
        link.click()
    assert download.value.suggested_filename == "report.txt"
    assert download.value.path().read_bytes() == payload
    capture(page, "tz1-byte-names-reloaded")
    if evidence_dir := os.environ.get("OUROBOROS_UI_EVIDENCE_DIR"):
        download.value.save_as(Path(evidence_dir) / "tz1-byte-names-report.txt")


def _start_world(ui, theme, width):
    """The production SPA over controlled boot snapshots; ``current`` is the /api/state answer, ``rows`` Main's history."""
    page = ui["page"]
    current = {"supervisor_ready": False, "works_complete": True, "works": [], "rooms": {}, "active_chat_activities": [],
               "active_chat_activities_complete": True, "projects": [], "boot": CONCURRENT}
    rows = [SAVED_ROW]
    page.route("**/api/state*", lambda route: route.fulfill(json=current))
    page.route("**/api/chat/history*", lambda route: route.fulfill(json=_history(rows if "chat_id=7" not in route.request.url else [])))
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.set_viewport_size({"width": width, "height": 844 if width < 600 else 864})
    return page, current


def _said(page, trigger, word, step, *, shown=True):
    """The bar says ``word`` with ``step`` beside it; a short bar drops the step first and keeps the word."""
    page.wait_for_function("([t, w, s]) => t.querySelector('.status-badge')?.textContent === w"
                           " && (t.querySelector('.status-badge').dataset.step || '') === s",
                           arg=[trigger.element_handle(), word, step])
    if shown:
        assert trigger.get_attribute("data-word") != "off", f"{word!r} stays in the bar"


def _open_by_keyboard(page, trigger):
    """The list opens from the keyboard, as from a press: no detail of the start needs a pointer over the title."""
    from playwright.sync_api import expect

    expect(trigger).to_be_visible()
    expect(trigger).to_be_enabled()
    trigger.focus()
    expect(trigger).to_be_focused()
    page.keyboard.press("Enter")
    page.wait_for_function("t => t.getAttribute('aria-expanded') === 'true'", arg=trigger.element_handle())


@pytest.mark.parametrize("width", [1536, 390])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_main_says_the_start_without_a_hover(subscription_ui, theme, width):
    """Main: whether a message is saved and what runs, said by the bar and by its list; then a failed step and start."""
    ui = subscription_ui
    page, current = _start_world(ui, theme, width)
    tag = f"{theme}-{width}"
    status, trigger, note = page.locator("#chat-status"), page.locator(".chat-page-header .rw-trigger"), page.locator(".chat-page-header .rw-note")
    current["boot"] = CHAT_CLOSED
    ui["errors"].watch(page)
    page.goto(ui["url"])
    _said(page, trigger, "Starting…", "preparing skills")
    _open_by_keyboard(page, trigger)
    assert note.inner_text() == ("The chat opens once the data store is ready; a message sent before that is not saved. "
                                 "Now: preparing skills, preparing task transport.")
    capture(page, f"m2-main-chat-closed-{tag}")
    current["boot"] = CONCURRENT
    ui["errors"].watch(page)
    page.goto(ui["url"])
    _said(page, trigger, "Chat ready", "loading extensions")
    saved = page.locator('.chat-bubble.user[data-client-message-id="m2-saved"] [data-ingress-saved]')
    saved.wait_for()
    assert saved.inner_text() == "Input saved"
    assert status.get_attribute("title") == "loading extensions · preparing task transport"
    _open_by_keyboard(page, trigger)
    assert note.inner_text() == f"{SAVED} Now: loading extensions, preparing task transport."
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    capture(page, f"m2-main-saved-concurrent-{tag}")
    current["boot"] = SOFT_FAILED
    ui["errors"].watch(page)
    page.goto(ui["url"])
    _said(page, trigger, "Chat ready", "checking program files")
    _open_by_keyboard(page, trigger)
    assert note.inner_text() == (f"{SAVED} Now: checking program files. A start step failed and the start goes on: "
                                 "startup:event_queue: OSError: the first event-bus Queue() RPC is refused")
    capture(page, f"m2-main-soft-failed-{tag}")
    current["boot"] = FAILED
    ui["errors"].watch(page)
    page.goto(ui["url"])
    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Start failed'")
    _open_by_keyboard(page, trigger)
    failure = page.locator(".chat-page-header .rw-failure")
    failure.wait_for()
    assert failure.locator(".rw-failure-lead").inner_text() == "Ouroboros could not finish starting."
    assert failure.locator("button").inner_text() == "Restart"
    failure.locator("summary").click()
    assert failure.locator(".ui-result-details").inner_text() == FAILED["error"]
    capture(page, f"m2-main-failed-{tag}")
    current.update(supervisor_ready=True, boot={**CONCURRENT, "phase": "ready", "tasks_ready": True, "steps": []})
    ui["errors"].watch(page)
    page.goto(ui["url"])
    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Online'")
    assert trigger.is_disabled() or width < 600, "a ready, idle desk Main has nothing to list (a phone's keeps its commands)"
    assert not ui["errors"].unexpected(), ui["errors"].report()


@pytest.mark.parametrize("width", [1536, 390])
def test_a_project_and_settings_say_the_start_without_a_hover(subscription_ui, width):
    """A Project's list says the start under the room's name; Settings' card says the word and, when the start failed,
    its cause and Restart on its first line."""
    from tests.settings_browser_helpers import open_settings

    ui = subscription_ui
    page, current = _start_world(ui, "light", width)
    current["projects"] = [{"id": "alpha", "name": "A long Project name that the bar has to cut", "chat_id": 7}]
    ui["errors"].watch(page)
    page.goto(ui["url"])
    page.locator('.nav-project-row[data-project-id="alpha"]').evaluate("el => el.click()")
    panel = page.locator('[id="panel-pchat-alpha"]:not([hidden])')
    panel.wait_for()
    bar = page.locator(".project-panel-bar:visible")
    trigger = bar.locator(".rw-trigger")
    _said(page, trigger, "Chat ready", "loading extensions", shown=False)
    capture(page, f"m2-project-bar-light-{width}")
    _open_by_keyboard(page, trigger)
    assert bar.locator(".rw-note").inner_text() == f"{SAVED} Now: loading extensions, preparing task transport."
    assert bar.locator(".rw-title").inner_text() == "A long Project name that the bar has to cut"
    capture(page, f"m2-project-saved-concurrent-light-{width}")
    open_settings(page)
    page.wait_for_function("() => [...document.querySelectorAll('[data-being-state], [data-overview-answer]')]"
                           ".some((el) => el.textContent === 'Chat ready · loading extensions')")
    capture(page, f"m2-settings-starting-light-{width}")
    current["boot"] = FAILED
    ui["errors"].watch(page)
    page.goto(ui["url"])
    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Start failed'")
    open_settings(page)
    overview = page.locator("[data-settings-overview-row]:visible")
    if overview.count():  # two panes: the list's Ouroboros row says the state and opens the card beside it
        overview.locator("[data-overview-answer]").filter(has_text="Start failed").wait_for()
        assert overview.locator("[data-overview-answer]").inner_text() == "Start failed · One thing needs you"
        overview.click()
    page.wait_for_function("() => [...document.querySelectorAll('[data-being-state]')].some((el) => el.textContent === 'Start failed')")
    row = page.locator('[data-needs="start"]:visible').first
    row.wait_for()
    assert row.locator(".ui-row-title").inner_text() == "Ouroboros could not finish starting."
    assert row.locator(".ui-row-meta").inner_text() == FAILED["error"]
    assert row.locator("[data-need-action]").inner_text() == "Restart"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    capture(page, f"m2-settings-failed-light-{width}")
    assert not ui["errors"].unexpected(), ui["errors"].report()


READY = {**CONCURRENT, "phase": "ready", "tasks_ready": True, "steps": []}
# A head start that failed and was replaced (#3376): its step stays failed in a boot that became ready.
RECOVERED = {**READY, "steps": _steps(failed=[("startup:event_queue", "OSError: the first event-bus Queue() RPC is refused")])}
# Every distinct reading of Settings' card, in order (a MutationObserver: a change the page made, never a sample).
_RECORD_CARD = """() => {
    window.__m2SameDocument = true;
    window.__m2Card = [];
    const record = () => {
        const now = document.querySelector('[data-being-state]')?.textContent || '';
        if (now && now !== window.__m2Card.at(-1)) window.__m2Card.push(now);
    };
    new MutationObserver(record).observe(document, { subtree: true, childList: true, characterData: true });
    record();
}"""


def _card_says(page, word, timeout=15_000):
    page.wait_for_function("(w) => [...document.querySelectorAll('[data-being-state]')].some((el) => el.textContent === w)",
                           arg=word, timeout=timeout)


def _outage(page):
    """The page's own socket closes; its manager reconnects as after any outage (ws.js)."""
    page.evaluate("() => window.__ouroWs.ws.close()")


def test_an_open_settings_follows_the_start_through_outages_without_a_reload(subscription_ui):
    """Synthetic and adversarial (#3376 M2): Settings stays open while the host's start moves and the socket drops.
    Every snapshot the page reads reaches the card without a reload; a reconnect whose read is held says the start's
    word, never the Online of the server before the outage; a failed start is the card's first line with Restart; and a
    recovered head-start step in a ready boot says nothing failed."""
    from playwright.sync_api import Error as PlaywrightError
    from tests.settings_browser_helpers import open_settings

    ui = subscription_ui
    page, current = _start_world(ui, "light", 1536)
    # A stable served SHA: a reconnect then keeps this document (ws.js decide), so any reload is a finding.
    current.update(sha="m2-fixture", supervisor_ready=True, boot=READY)
    holding, held, dropped = [False], [], []

    def state(route):
        if holding[0]:
            held.append(route)
        else:
            route.fulfill(json=current)

    page.route("**/api/state*", state)  # registered after _start_world's route, so this one answers
    ui["errors"].watch(page)
    page.goto(ui["url"])
    open_settings(page)
    page.evaluate(_RECORD_CARD)
    _card_says(page, "Online")
    holding[0] = True
    _outage(page)
    _card_says(page, "Starting…")  # reconnected, its reads held: the snapshot from before the outage is forgotten
    page.wait_for_function("() => window.__ouroWs?.isConnected?.() === true")
    current.update(supervisor_ready=False, boot=CONCURRENT)
    holding[0] = False
    for route in held:
        try:
            route.fulfill(json=current)
        except PlaywrightError as exc:
            dropped.append(str(exc).splitlines()[0])  # a read the page itself let go at the outage
    _card_says(page, "Chat ready · loading extensions")
    current["boot"] = FAILED
    _outage(page)
    _card_says(page, "Start failed")
    start = page.locator('[data-needs="start"]').first
    start.wait_for(state="attached")
    assert start.locator("[data-need-action]").text_content().strip() == "Restart"
    assert start.locator(".ui-row-meta").text_content().strip() == FAILED["error"]
    capture(page, "m2-settings-outage-failed")
    current.update(supervisor_ready=True, boot=RECOVERED)
    _outage(page)
    _card_says(page, "Online")
    assert page.locator('[data-needs="start"]').count() == 0, "a recovered step of a ready start is no need"
    seen = page.evaluate("() => window.__m2Card")
    held_span = seen[seen.index("Reconnecting…"):seen.index("Chat ready · loading extensions")]
    assert "Online" not in held_span and "Starting…" in held_span, seen
    assert page.evaluate("() => window.__m2SameDocument === true"), f"the page reloaded: {seen}"
    capture(page, "m2-settings-outage-recovered")
    assert not ui["errors"].unexpected(), (ui["errors"].report(), dropped)


def test_a_reconnect_after_a_restart_never_says_the_old_servers_online(subscription_ui):
    """Synthetic and adversarial (#3376, the reconnect contract): the page polls while its socket is down, and the
    server from before a restart answers ready; the socket then reopens on a new server that is still starting. Main's
    bar and Settings' card (mounted at start beside Main) both forget readiness at that reconnect and say the start's
    word until a body read after it, never the old server's Online; that body then speaks at once, without a reload."""
    from playwright.sync_api import Error as PlaywrightError

    ui = subscription_ui
    page, current = _start_world(ui, "light", 1536)
    # A stable served SHA: a reconnect then keeps this document (ws.js decide), so any reload is a finding.
    current.update(sha="m2-fixture", supervisor_ready=True, boot=READY)
    holding, held, dropped = [False], [], []

    def state(route):
        if holding[0]:
            held.append(route)
        else:
            route.fulfill(json=current)

    page.route("**/api/state*", state)  # registered after _start_world's route, so this one answers
    ui["errors"].watch(page)
    page.goto(ui["url"])
    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Online'")
    _card_says(page, "Online")  # the socket's first open keeps the boot read's readiness
    page.evaluate(_RECORD_CARD)
    # The socket drops and stays down: its manager's next attempt waits for this test (ws.js connect).
    page.evaluate("() => { const ws = window.__ouroWs; ws.connect = () => { window.__m2Attempted = true; }; ws.ws.close(); }")
    page.wait_for_function("() => window.__m2Attempted === true")
    # Main's 3 s poll still reaches the server from before the restart, which answers ready; the Projects row its body
    # carries proves the page applied that body.
    current["projects"] = [{"id": "m2-outage", "name": "Answered while the socket was down", "chat_id": 9}]
    page.locator('.nav-project-row[data-project-id="m2-outage"]').wait_for(state="attached")
    assert page.locator("#chat-status").text_content() == "Reconnecting…"
    holding[0] = True
    current.update(supervisor_ready=False, boot=CONCURRENT, projects=[])
    page.evaluate("() => { const ws = window.__ouroWs; delete ws.connect; ws.connect(); }")
    page.wait_for_function("() => window.__ouroWs?.isConnected?.() === true")
    # The new server's reads are held: both say the start's word, neither the Online of the server before the restart.
    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Starting…'")
    _card_says(page, "Starting…")
    holding[0] = False
    for route in held:
        try:
            route.fulfill(json=current)
        except PlaywrightError as exc:
            dropped.append(str(exc).splitlines()[0])  # a read the page itself let go at the reconnect
    _said(page, page.locator(".chat-page-header .rw-trigger"), "Chat ready", "loading extensions")
    _card_says(page, "Chat ready · loading extensions")
    current.update(supervisor_ready=True, boot=READY)  # Main's next poll reads the new server ready
    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Online'", timeout=15_000)
    _card_says(page, "Online")
    seen = page.evaluate("() => window.__m2Card")
    held_span = seen[seen.index("Reconnecting…"):seen.index("Chat ready · loading extensions")]
    assert "Online" not in held_span and "Starting…" in held_span, seen
    assert page.evaluate("() => window.__m2SameDocument === true"), f"the page reloaded: {seen}"
    capture(page, "m2-reconnect-after-restart")
    assert not ui["errors"].unexpected(), (ui["errors"].report(), dropped)


def test_an_open_question_keeps_mains_amber_while_the_host_starts(subscription_ui):
    """Synthetic (#3376 M2): Main at rest with a question still unanswered: while the host starts, its bar says the
    start's word and step beside the owner's amber dot, and says the question in words; readiness keeps the amber."""
    ui = subscription_ui
    page, current = _start_world(ui, "light", 1536)
    asked = {"activity": "idle", "transition": "none", "call": "", "amber": True, "mark": "none", "confirmed": True,
             "works": [], "turn": [{"kind": "open_question", "task_id": "m2-asked", "quiz_id": "m2-quiz"}]}
    current.update(rooms={"1": asked}, works_gaps=[], work_summary={"counts": {}, "turn": 1, "confirmed": True})
    # A stable served SHA from the first read keeps this document across the reconnect (ws.js decide:
    # a missing sha after a connection reloads as unknown, and a fresh page repaints the mark ink
    # before the state reads restore the owner's amber). The outage tests above pin it the same way.
    current["sha"] = "m2-question-amber"
    ui["errors"].watch(page)
    page.goto(ui["url"])
    trigger = page.locator(".chat-page-header .rw-trigger")
    _said(page, trigger, "Chat ready", "loading extensions")
    page.wait_for_function("t => t.querySelector('#chat-status')?.dataset.owner === '1'"
                           " && t.querySelector('.status-mark')?.dataset.hue === 'owner'", arg=trigger.element_handle())
    mark = trigger.locator(".status-mark")
    assert mark.get_attribute("data-shape") == "dot"
    assert page.locator("#chat-status").get_attribute("data-amber") == "unanswered"
    assert trigger.locator(".rw-sr").text_content().strip() == "An unanswered question is here", "said in words too"
    capture(page, "m2-main-question-while-starting")
    current.update(supervisor_ready=True, boot=READY)
    page.evaluate("() => window.__ouroWs.ws.close()")  # the reconnect's read carries readiness
    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Online'", timeout=15_000)
    assert mark.get_attribute("data-hue") == "owner", "readiness keeps the question's amber"
    assert not ui["errors"].unexpected(), ui["errors"].report()



# --- one genuine consumer flow: the real server restarts under open pages ------------------------------------------

_SLOW_START_SECONDS = 8  # how long the reviewed extension below holds the boot's `extensions` step


def _install_slow_start_extension(data_dir):
    """A reviewed extension whose registration takes `_SLOW_START_SECONDS`: a boot then stays at `extensions` (Chat
    ready, the supervisor waiting for the subscribers) long enough for a page to see it and for a message to be sent."""
    import textwrap
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "m2_slow_start"
    skill = data_dir / "skills" / "external" / name
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: Holds the boot's extension step so a page can watch a real start.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: ["route"]
        ---
        # Slow start
        """), encoding="utf-8")
    (skill / "plugin.py").write_text(textwrap.dedent(f"""\
        import time


        async def ping(_request):
            return {{"ok": True}}


        def register(api):
            time.sleep({_SLOW_START_SECONDS})
            api.register_route("ping", ping, methods=("GET",))
        """), encoding="utf-8")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=compute_content_hash(skill, manifest_entry="plugin.py")))
    return name


_RECORD_STATUS = """() => {
    window.__m2SameDocument = true;
    window.__m2Seen = [];
    const record = () => {
        const badge = document.querySelector('#chat-status');
        const card = document.querySelector('[data-being-state]');
        const now = [badge ? [badge.textContent, badge.dataset.step || ''].filter(Boolean).join(' · ') : '',
                     card?.textContent || ''].join(' || ');
        if (now !== window.__m2Seen.at(-1)?.now) window.__m2Seen.push({ now, at: Date.now() });
    };
    new MutationObserver(record).observe(document, { subtree: true, childList: true, characterData: true, attributes: true,
                                                     attributeFilter: ['data-step'] });
    record();
}"""


def _health_boot(url):
    import urllib.request

    with urllib.request.urlopen(f"{url}/api/health", timeout=5) as resp:  # noqa: S310 - the fixture's own loopback server
        return json.loads(resp.read().decode("utf-8"))["boot"]


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_a_real_restart_is_said_as_it_happens_and_a_message_is_accepted_before_readiness(direct_server_with_data,
                                                                                        browser_name):
    """Genuine consumer flow (#3376 M1 and M2): the real SPA stays open on Main and on Settings while the real server
    restarts on the candidate (its event bus prepared beside the extensions, a reviewed extension holding that step).
    Without a reload, Main's bar and Settings' card say the outage, then the start's own words, then Online, never
    Online over a start and never Start failed; a message sent while the host still starts is accepted durably and
    answered by the loopback model once it is ready; the restarted process has one bus, made by the boot's head start
    and closed by nobody yet."""
    import threading
    from datetime import datetime

    from playwright.sync_api import sync_playwright
    from tests.settings_browser_helpers import open_settings
    from tests.ui_console_errors import PageErrors

    server, url, data_dir = direct_server_with_data, direct_server_with_data["url"], direct_server_with_data["data_dir"]
    name = _install_slow_start_extension(data_dir)
    with sync_playwright() as playwright:
        browser = getattr(playwright, browser_name).launch(headless=True)
        try:
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            main, settings = context.new_page(), context.new_page()
            errors = [PageErrors(main, browser_name), PageErrors(settings, browser_name)]
            for page in (main, settings):
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_function("() => window.__ouroWs?.isConnected?.() === true", timeout=30_000)
            toggled = main.evaluate("""async (skill) => {
                const response = await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: true }) });
                return { status: response.status, body: await response.json().catch(() => ({})) };
            }""", name)
            assert toggled["status"] == 200 and toggled["body"].get("enabled") is True, toggled
            open_settings(settings)
            for page in (main, settings):
                page.evaluate(_RECORD_STATUS)
            main.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Online'", timeout=30_000)
            journal = data_dir / "logs" / "supervisor.jsonl"
            before = len(journal.read_text(encoding="utf-8").splitlines())
            failure = []

            def restart():
                try:
                    server["restart_server"]()
                except BaseException as exc:  # re-raised on this test's thread below
                    failure.append(exc)

            restarting = threading.Thread(target=restart, name="m2-restart", daemon=True)
            restarting.start()
            # While the extension holds its step: the chat accepts and the start is said with its step.
            main.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Chat ready'"
                                   " && document.querySelector('#chat-status')?.dataset.step === 'loading extensions'",
                                   timeout=90_000)
            text = f"m2 genuine flow {browser_name}: are you there?"
            main.fill("#chat-input", text)
            main.click("#chat-send")
            saved = main.locator(".chat-bubble.user", has_text=text).locator("[data-ingress-saved]")
            saved.wait_for(timeout=30_000)
            boot_at_acceptance = _health_boot(url)
            assert saved.inner_text() == "Input saved"
            assert boot_at_acceptance["tasks_ready"] is False, ("accepted only after readiness", boot_at_acceptance)
            capture(main, f"m2-genuine-accepted-while-starting-{browser_name}")
            restarting.join(180)
            assert not restarting.is_alive(), "the restart did not reach readiness"
            if failure:
                raise failure[0]
            main.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Online'", timeout=30_000)
            main.locator(".chat-bubble.assistant", has_text="OK").first.wait_for(timeout=90_000)
            settings.wait_for_function("() => document.querySelector('[data-being-state]')?.textContent === 'Online'",
                                       timeout=60_000)  # Settings reads the state every 20 s and at a reconnect
            capture(settings, f"m2-genuine-settings-online-{browser_name}")
            for page, label in ((main, "main"), (settings, "settings")):
                seen = [entry["now"] for entry in page.evaluate("() => window.__m2Seen")]
                assert page.evaluate("() => window.__m2SameDocument === true"), (label, "reloaded", seen)
                bars = [entry.split(" || ")[0] for entry in seen]
                cards = [entry.split(" || ")[1] for entry in seen]
                assert not any(word.startswith("Start failed") for word in bars + cards), (label, seen)
                said = bars if label == "main" else cards  # a room away draws no status: Settings' page says its card
                outage = next((i for i, word in enumerate(said) if word.startswith("Reconnecting…")), None)
                assert outage is not None, (label, "the outage was never said", seen)
                after = [word for word in said[outage:] if word and not word.startswith("Reconnecting…")]
                assert after and after[0] != "Online" and after[-1] == "Online", (label, seen)
            boot = _health_boot(url)
            steps = {row["name"]: row for row in boot["steps"]}
            ended = {key: datetime.fromisoformat(steps[key]["ended_at"]) for key in ("startup:event_queue", "extensions")}
            assert steps["startup:event_queue"]["state"] == "done", steps
            assert ended["startup:event_queue"] < ended["extensions"], ("the bus was made while the extensions loaded", steps)
            rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()[before:]]
            started = [row for row in rows if row.get("type") == "event_queue_generation_started"]
            assert len(started) == 1, started  # the restarted process made one bus
            spawned = {row.get("event_queue_generation") for row in rows[rows.index(started[0]):] if row.get("type") == "worker_spawn_start"}
            assert spawned == {started[0]["generation"]}, (started, spawned)
            assert all(row.get("generation") != started[0]["generation"] for row in rows
                       if row.get("type") == "event_queue_generation_stopped"), "the running bus is not ended"
            evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or data_dir.parent)  # beside capture()'s screenshots
            evidence.mkdir(parents=True, exist_ok=True)
            (evidence / f"m2-genuine-{browser_name}.json").write_text(json.dumps({
                "seen": {label: page.evaluate("() => window.__m2Seen") for label, page in (("main", main), ("settings", settings))},
                "boot_at_acceptance": boot_at_acceptance, "boot_after": boot, "journal_rows": rows,
                "page_errors": [str(err) for err in errors]}, indent=2), encoding="utf-8")
            for err in errors:
                assert err.unexpected() == [], err
        finally:
            browser.close()

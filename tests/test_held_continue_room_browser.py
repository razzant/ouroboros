"""A held Continue in its Project room, read from the server's work projection in the real SPA.

The Continue waits for three helpers of its failed earlier attempt. The census rows, queue facts and stored
result are the ones ``tests/test_work_projection_web_fixture.py`` projects (its ``held_continue`` case), and the
page reads the server's own ``works``/``rooms`` for them over the controlled transport of
``tests/test_room_work_browser.py``: one work, one row, and every surface says the same thing.
"""
from __future__ import annotations

import time

import pytest

from tests import test_room_work_browser as rw
from tests.test_work_projection_web_fixture import CASES, FENCES

room_ui = rw.room_ui
subscription_ui = rw.subscription_ui
pytestmark = [pytest.mark.ui_browser]

REASON = "Continue starts when 3 subagents of the earlier attempt finish"
NAME = "Ouroboros 7.7.0 — finish the saved release"
MARKER = f".nav-project-row[data-project-id='{rw.PROJECT['id']}'] .status-mark"


def test_held_continue_with_three_helpers_is_one_working_row_with_stop_earlier_and_no_resume(room_ui):
    owners, raw, stored, _held = CASES["held_continue"]
    history = {1: rw.reading(4), 42: [*rw.reading(6, 0, "Project"),
                                       rw.progress("c", "Waiting for the earlier attempt's helpers.", 30)]}
    transport = rw.Transport(room_ui["page"], history=history, projects=[rw.PROJECT], activities=owners)
    transport.raw, transport.results, transport.fences = raw, stored, FENCES
    page = rw.open_app(room_ui, transport, viewport={"width": 1100, "height": 722})
    rw.open_project(page, rw.PROJECT)
    transport.refresh()

    # The room works: its word in the bar beside its moving mark, never amber, and the sidebar's three dots move.
    closed = rw.wait_facts(page, "project", lambda f: f["badge"] == "Working" and not f["disabled"], "the room working")
    assert (closed["tone"], closed["mark"]) == ("live", "moving ink"), closed
    page.wait_for_function("(selector) => document.querySelector(selector)?.dataset.shape === 'moving'", arg=MARKER)
    marker = page.locator(MARKER)
    assert (marker.get_attribute("data-shape"), marker.get_attribute("data-hue")) == ("moving", "ink")

    # One work, one row: the Continue and its earlier attempt's helpers are one row, which says why it waits.
    rw.trigger(page, "project").click()
    shade = rw.wait_facts(page, "project", lambda f: len(f["rows"]) == 1 and f["rows"][0]["secondary"] == REASON,
                          "one row with its reason")
    rw.evidence("held-continue-room", shade)
    rw.capture(page, "held-continue-room-open")
    row = shade["rows"][0]
    assert (row["name"], row["word"], row["tone"]) == (NAME, "Working", "live"), row
    assert not row["resume"] and not row["question"] and row["more"], row

    # Its card is the Continue's: the same word and reason, and no Resume.
    card = page.locator("#project-panel .chat-live-card[data-task-id='c']")
    assert card.locator("[data-live-phase]").inner_text() == "Working"
    assert card.locator("[data-live-phase-secondary]").inner_text() == REASON
    assert card.locator("[data-resume-run]").count() == 0

    # Its controls: Stop for the whole work and Stop earlier attempt for what still runs of the failed attempt.
    page.locator("#project-panel .rw-row[data-id='p'] .rw-more").click()
    menu = page.locator("body > .task-control-menu")
    menu.wait_for(state="visible")
    assert menu.evaluate("n => [...n.querySelectorAll('.task-control-item')].map(i => i.dataset.taskControl)") == [
        "stop_earlier", "stop_now"]
    assert menu.locator('[data-task-control="stop_earlier"]').inner_text() == "Stop earlier attempt"
    menu.locator('[data-task-control="stop_earlier"]').click()
    deadline = time.monotonic() + 5
    while not transport.posts and time.monotonic() < deadline:
        page.wait_for_timeout(20)
    # Only the earlier attempt is stopped, as one cascade; the held Continue itself stays.
    assert [path for path, _ in transport.posts] == ["/api/tasks/p/cancel"], transport.posts
    assert transport.posts[0][1].get("cascade") is True, transport.posts
    assert transport.decision_posts() == []


def test_a_continue_waiting_over_one_unconfirmed_start_offers_its_card_a_resume_whose_choice_asks_first(room_ui):
    """No helper runs: the Continue waits only over one start of its earlier attempt that has no recorded run. Its
    real card in the Project room carries the work's own Resume. The door answers the press with the choice; Cancel
    sends nothing more, the next press asks again, and Resume anyway sends the id with the owner's reason."""
    import copy

    from playwright.sync_api import expect

    owners, raw, stored, _held = copy.deepcopy(CASES["held_continue"])
    owners = [row for row in owners if not row.get("parent_task_id")]
    start = {"root_task_id": "p", "task_id": "p", "invocation_id": "lost", "choice_id": "delegate_start:p/p/lost"}
    unconfirmed = {"kind": "delegated_run", "task_id": "p", "run_id": "", "invocation_id": "lost",
                   "state": "stop_unknown", "stop_outcome": "pending_invocation_unbound", "ordinary_start": start}
    choice = "delegate_start:p/p/lost/c///hold"
    raw["c"]["_budget_pause_hold"] = {"reason": "continuation_writer_unsettled", "predecessor_task_id": "p",
                                      "root_task_id": "c", "blockers": [unconfirmed]}
    history = {1: rw.reading(4), 42: [*rw.reading(6, 0, "Project"),
                                       rw.progress("c", "Waiting for the earlier attempt's start.", 30)]}
    page = room_ui["page"]
    transport = rw.Transport(page, history=history, projects=[rw.PROJECT], activities=owners)
    transport.raw, transport.results, transport.fences = raw, stored, FENCES
    sent = []

    def resume(route):  # the door's answers: the choice to a plain press, success once the id comes back
        body = route.request.post_data_json or {}
        sent.append(body)
        if body.get("accept_unknown"):
            route.fulfill(json={"ok": True, "task_id": "c", "state": "resumed"})
        else:
            route.fulfill(status=409, json={"ok": False, "error": "predecessor_writers_unsettled",
                                            "blockers": [{**unconfirmed, "choice_id": choice, "read_only": True}],
                                            "acceptable_unknown": [choice]})

    page.route("**/api/tasks/c/resume", resume)
    page = rw.open_app(room_ui, transport, viewport={"width": 1100, "height": 722})
    rw.open_project(page, rw.PROJECT)
    transport.refresh()
    rw.trigger(page, "project").click()
    shade = rw.wait_facts(page, "project", lambda f: len(f["rows"]) == 1 and f["rows"][0]["resume"], "its row with Resume")
    assert shade["rows"][0]["word"] == "Waiting", shade
    card = page.locator("#project-panel .chat-live-card[data-task-id='c']")
    door = card.locator("[data-resume-run]")
    expect(door).to_be_enabled()
    dialog = page.locator(".confirm-dialog")

    door.click()  # the press: the door answers with the choice, and only now is the owner asked
    dialog.wait_for(state="visible")
    assert dialog.locator("#confirm-dialog-title").inner_text() == "Resume without this result?"
    assert "The earlier request was read-only" in dialog.locator(".ui-modal-body p").inner_text()
    rw.capture(page, "held-continue-resume-choice")
    dialog.locator("[data-confirm-cancel]").click()
    dialog.wait_for(state="detached")
    assert sent == [{}], "Cancel sends nothing beyond the press that asked"
    expect(door).to_be_enabled()

    door.click()  # the same press asks again
    dialog.wait_for(state="visible")
    dialog.locator("[data-confirm-ok]").click()
    deadline = time.monotonic() + 5
    while len(sent) < 3 and time.monotonic() < deadline:
        page.wait_for_timeout(20)
    assert sent == [{}, {}, {"accept_unknown": [choice],
                              "reason": "The owner accepted that the listed start may still be working or spending money."}], sent
    expect(page.locator("#toast-stack .toast")).to_contain_text("Resuming: the task returns to the queue.")


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_activity_offers_the_held_work_its_own_stop_targets(engine, tmp_path):
    """Dashboard → Activity reads the same work: Stop ends the Continue and its earlier attempt, each at its own
    task, and Stop earlier attempt only the earlier one; the held Continue offers no Resume or Pause."""
    import json
    import os
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path
    from threading import Thread

    from playwright.sync_api import sync_playwright

    from tests._work_projection_shared import work_fields
    from tests.test_activity_outcomes_browser import HTML, WEB

    owners, raw, stored, _held = CASES["held_continue"]
    projection = work_fields(owners, raw, results=stored, fences=FENCES)
    helpers = [row for row in owners if row.get("parent_task_id")]
    queue = {"running": [{"id": row["activity_id"], "task": {"id": row["activity_id"], "parent_task_id": "p",
                                                             "root_task_id": "p"}} for row in helpers],
             "pending": [{"id": "c", "task": {"id": "c", "chat_id": 42, "title": "Continue: release", **raw["c"]}}]}
    posts = []

    def api(route):
        request = route.request
        path = request.url.split("/api/", 1)[1]
        if request.method == "POST":
            posts.append(path)
            body = {"ok": True, "status": "accepted"}
        elif path.startswith("tasks?"):
            body = {"tasks": [], "queue": queue}
        elif path == "state":
            body = {"bg_consciousness_enabled": False, "supervisor_ready": True, "active_chat_activities": owners[:1],
                    "active_chat_activities_complete": True, **projection}
        elif path == "schedules":
            body = {"tasks": []}
        else:
            body = {}
        route.fulfill(status=200, content_type="application/json", body=json.dumps(body))

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path.startswith("/fixture"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML.encode())
            else:
                super().do_GET()

    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = getattr(playwright, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1100, "height": 760})
                page.route("**/api/**", api)
                page.goto(f"http://127.0.0.1:{server.server_port}/fixture")
                page.wait_for_function("window.ready === true")
                for action, expected in (("stop_now", ["tasks/c/cancel", "tasks/p/cancel"]),
                                         ("stop_earlier", ["tasks/p/cancel"])):
                    posts.clear()
                    page.locator('[data-act="task-control"][data-id="c"]').click()
                    menu = page.locator("body > .task-control-menu")
                    menu.wait_for(state="visible")
                    items = menu.locator("[data-task-control]").evaluate_all("els => els.map(e => e.dataset.taskControl)")
                    page.screenshot(path=str(evidence / f"activity-held-work-{engine}-{action}.png"), animations="disabled")
                    assert items == ["stop_earlier", "stop_now"], items
                    menu.locator(f'[data-task-control="{action}"]').click()
                    deadline = time.monotonic() + 5
                    while len(posts) < len(expected) and time.monotonic() < deadline:
                        page.wait_for_timeout(20)
                    assert posts == expected, posts
                    page.wait_for_timeout(200)
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

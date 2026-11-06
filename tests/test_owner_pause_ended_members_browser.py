"""An ended root's requested Pause in the real Activity tab and the Main room's work list (#2669).

Each case builds its tree offline through the real producers (``tests/test_owner_pause_ended_members``: the accept
step and the off-loop settle pass with its stand-in engine; no daemon, credential or live installation) and answers
the page's reads from the real gateway readers over that data root: the census and the server's own work projection
of it (``state_fields``), the queue publication (``gateway.tasks._queue_snapshot``) and the task detail. The page runs
the production modules: the Activity tab (``initActivity``) and the whole SPA's Main room work list
(``room_work.js``). Screenshots, the rendered rows' HTML and the facts read in the page go to
``OUROBOROS_UI_EVIDENCE_DIR`` before anything is asserted.

Activity says the confirmed work's truth, never the census row's parked guess, and a work that lists no control
offers none, never the generic Stop. The room reads the same work. The settled Pause and the review it lets finish
are positive baselines, and so is the run still seen running, whose Pause is still settling.
"""
from __future__ import annotations

import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests._work_projection_shared import state_fields
from tests.test_activity_outcomes_browser import HTML, WEB
from tests.test_owner_pause_ended_members import _park_root, _run, _start, _tail_tree, _tick
from tests.test_room_work_browser import Transport, open_app, trigger, wait_facts
from tests.test_room_work_browser import room_ui as room_ui  # noqa: F401 — fixture

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

CASES = ("failed_owner", "gateway_unknown", "unbound", "known_live", "failed_owner_late", "known_live_late",
         "settled_pause", "review_finishing")
_BY = {"actor": "owner", "text": "", "at": "2026-11-04T01:00:00+00:00"}
_UNKNOWN = ("Activity unconfirmed", "You paused it", "1 external run may still be running")
# What each surface must say for the root's work, and the controls its doors take (``None``: no control at all).
ACTIVITY = {
    "failed_owner": (_UNKNOWN, ("Paused", "Pausing"), ["stop_now"]),
    "gateway_unknown": (_UNKNOWN, ("Paused", "Pausing"), ["stop_now"]),
    "unbound": (_UNKNOWN, ("Paused", "Pausing"), None),
    "known_live": (("Pausing…", "You paused it", "1 external run still working"), ("Paused",), ["pause", "stop_now"]),
    # An answered root whose post-task synthesis completed names its late work: its state comes from the work too.
    "failed_owner_late": (("Remaining work status unknown", *_UNKNOWN[1:]), ("work paused", "Paused"), ["stop_now"]),
    "known_live_late": (("Remaining work pausing", "You paused it", "1 external run still working"),
                        ("work paused", "unconfirmed"), ["pause", "stop_now"]),
    "settled_pause": (("Paused", "You paused it"), ("Pausing", "unconfirmed"), ["resume", "stop_now"]),
    "review_finishing": (("Paused", "You paused it", "Review work finishing"), ("Pausing", "unconfirmed"),
                         ["resume", "stop_now"]),
}
ROOM = {
    "failed_owner": ("Activity unconfirmed", _UNKNOWN[1:], ["stop_now"]),
    "gateway_unknown": ("Activity unconfirmed", _UNKNOWN[1:], ["stop_now"]),
    "unbound": ("Activity unconfirmed", _UNKNOWN[1:], None),
    "known_live": ("Pausing…", ("You paused it", "1 external run still working"), ["pause", "stop_now"]),
    "failed_owner_late": ("Activity unconfirmed", _UNKNOWN[1:], ["stop_now"]),
    "known_live_late": ("Pausing…", ("You paused it", "1 external run still working"), ["pause", "stop_now"]),
    "settled_pause": ("Paused", ("You paused it",), ["stop_now"]),
    "review_finishing": ("Paused", ("You paused it", "Review work finishing"), ["stop_now"]),
}

ACTIVITY_FACTS = """() => {
    const text = (el) => (el?.innerText || el?.textContent || '').replace(/\\s+/g, ' ').trim();
    const queue = document.querySelector('[data-activity-section="queue"]');
    const rows = [...(queue?.querySelectorAll('.activity-row') || [])].map((row) => {
        const button = row.querySelector('[data-act="task-control"]');
        return {name: text(row.querySelector('.activity-name')), sub: text(row.querySelector('.activity-sub')),
                chip: text(row.querySelector('[data-activity-phase]')),
                trigger: button ? {id: button.dataset.id, root: button.dataset.root || '', label: text(button),
                                   budgetPaused: button.dataset.budgetPaused || '', resume: button.dataset.resume || ''}
                    : null,
                html: row.outerHTML};
    });
    return {rows, queue: text(queue)};
}"""
MENU = """() => { const menu = document.querySelector('body > .task-control-menu');
    return menu ? [...menu.querySelectorAll('[data-task-control]')].map((item) => item.dataset.taskControl) : null; }"""
# An Activity control with one offered action is that action (its label names it); its press opens the shared
# confirmation instead of a menu of one item.
DIRECT = """() => document.querySelector('.confirm-dialog')
    ? document.querySelector('[data-act="task-control"][data-task-control-direct="1"]')?.textContent.trim() || null : null"""
DIRECT_ACTIONS = {"Stop now": "stop_now", "Stop earlier attempt": "stop_earlier", "Pause": "pause", "Resume": "resume",
                  "Wrap up": "finalize", "Hurry up": "hurry"}


def _world(tmp_path, monkeypatch, case):
    """One tree in the state its case names, built by the real accept step and settle pass; Main (1) is its room.
    A ``_late`` case's root answered with its post-task synthesis completed: its census row names its late work."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    ended = case not in {"settled_pause", "review_finishing"}
    owner = "failed" if case in {"failed_owner", "failed_owner_late", "unbound", "settled_pause"} else "completed"
    queue, workers, engine = _tail_tree(tmp_path, monkeypatch, root_status="completed" if ended else "running",
                                        owner_status=owner, tail=_start if case == "unbound" else _run, chat_id=1)
    if case.endswith("_late"):
        write_task_result(tmp_path, "root-1", "completed", root_phase_checkpoint={"post_task_synthesis": "completed"})
    if case == "gateway_unknown":
        engine.cancel_error = engine.read_error = ClaudexorUnavailable("daemon_unreachable", "no answer")
    if case == "review_finishing":
        _run(tmp_path, run_id="run-review", source="review_substrate")
    assert request_owner_pause("root-1", request_id="press", by=_BY)["ok"]
    if not ended:
        _park_root(tmp_path, monkeypatch, queue, workers)
    _tick(queue)
    if case == "settled_pause":
        engine.state = "cancelled"
    elif case == "review_finishing":
        engine.states["run-leaf"] = "cancelled"
    if not ended:
        assert _tick(queue) == ["root-1"], "a positive baseline is a settled Pause"
    assert queue.persist_queue_snapshot(reason="c2 browser fixture")


def _task_detail(root, task_id):
    from ouroboros.outcomes import public_task_result
    from ouroboros.task_results import load_task_result

    row = load_task_result(root, task_id)
    return None if row is None else json.loads(json.dumps(public_task_result(row)))


def _state(root):
    """``/api/state`` as the server sends it for this data root: its census and its own work projection."""
    return {"supervisor_ready": True, "bg_consciousness_enabled": False, "projects": [], "project_chat_ids": [],
            **state_fields(root, direct_turns=[]), "active_chat_activities_complete": True}


def _evidence(*parts):
    target = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or Path.cwd() / "c2-evidence", *parts)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _says(text, says, never):
    missing = [word for word in says if word not in text]
    present = [word for word in never if word in text]
    return missing, present


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("case", CASES)
def test_activity_says_the_works_truth_and_offers_only_its_controls(tmp_path, monkeypatch, case, engine):
    from playwright.sync_api import sync_playwright

    from ouroboros.gateway.tasks import _queue_snapshot

    _world(tmp_path, monkeypatch, case)
    evidence = _evidence("activity", engine)
    state = _state(tmp_path)
    (evidence / f"{case}-state.json").write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    posts, errors = [], []

    def api(route):
        request = route.request
        path = request.url.split("/api/", 1)[1]
        if request.method == "POST":  # never expected: a menu is only opened here, no action is taken
            posts.append({"path": path, "body": request.post_data_json})
            body = {"ok": True, "status": "accepted"}
        elif path.startswith("tasks?"):
            body = {"tasks": [], "queue": _queue_snapshot(tmp_path)}
        elif path == "state":
            body = _state(tmp_path)
        elif path == "schedules":
            body = {"tasks": []}
        elif path.startswith("tasks/"):
            body = _task_detail(tmp_path, path.split("/", 1)[1].split("?", 1)[0])
            if body is None:
                return route.fulfill(status=404, content_type="application/json", body='{"error": "task not found"}')
        else:
            return route.fulfill(status=404, body="{}")
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

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1200, "height": 420}, color_scheme="light")
                page.on("pageerror", lambda exc: errors.append(str(exc)))
                page.route("**/api/**", api)
                page.goto(f"http://127.0.0.1:{server.server_port}/fixture?theme=light")
                page.wait_for_function("window.ready === true")
                page.wait_for_timeout(200)
                page.screenshot(path=str(evidence / f"{case}.png"))
                shown = page.evaluate(ACTIVITY_FACTS)
                menu = None
                if page.locator('[data-act="task-control"]').count():
                    page.locator('[data-act="task-control"]').first.click()
                    page.wait_for_timeout(600)
                    menu = page.evaluate(MENU)
                    if menu is None and (label := page.evaluate(DIRECT)):
                        menu = [DIRECT_ACTIONS[label]]
                    page.screenshot(path=str(evidence / f"{case}-menu.png"))
                facts = {"case": case, "engine": engine, **shown, "menu": menu, "posts": posts, "errors": errors}
                (evidence / f"{case}-facts.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False),
                                                             encoding="utf-8")
                (evidence / f"{case}-rows.html").write_text("\n".join(row["html"] for row in shown["rows"]),
                                                            encoding="utf-8")
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()

    assert errors == [] and posts == [], (errors, posts)
    [row] = shown["rows"]  # the tree's one live root: a census row once it ended, its queue row while parked
    says, never, controls = ACTIVITY[case]
    assert _says(" · ".join([row["chip"], row["sub"]]), says, never) == ([], []), row
    assert (row["trigger"] is not None) == (controls is not None), row
    assert menu == controls, (menu, row)


@pytest.mark.parametrize("case", CASES)
def test_the_room_list_says_the_same_work(room_ui, tmp_path, monkeypatch, case):
    _world(tmp_path, monkeypatch, case)

    class Backend(Transport):
        """The real census, work projection and task detail of this data root, never fields written by hand."""

        def body(self, *, ready=True):
            return {**_state(tmp_path), "sha": "c2-fixture", "supervisor_ready": ready}

        def _detail(self, route, task_id):
            detail = _task_detail(tmp_path, task_id)
            if detail is None:
                route.fulfill(status=404, json={"error": "task not found"})
            else:
                route.fulfill(json=detail)

    page = open_app(room_ui, Backend(room_ui["page"]), theme="light")
    trigger(page).click()
    opened = wait_facts(page, "main", lambda f: any(item["id"] == "root-1" for item in f["rows"]), "the root's row")
    evidence = _evidence()
    page.screenshot(path=str(evidence / f"room-{case}.png"), animations="disabled")
    [item] = [item for item in opened["rows"] if item["id"] == "root-1"]
    menu = None
    if item["more"]:
        page.locator("#page-chat .rw-row[data-id='root-1'] .rw-more").click()
        page.locator("body > .task-control-menu").wait_for(state="visible", timeout=10000)
        menu = page.evaluate(MENU)
        page.screenshot(path=str(evidence / f"room-{case}-menu.png"), animations="disabled")
    html = page.evaluate("() => document.querySelector(\"#page-chat .rw-row[data-id='root-1']\")?.outerHTML || ''")
    (evidence / f"room-{case}-facts.json").write_text(json.dumps({"case": case, "row": item, "menu": menu},
                                                                 indent=1, ensure_ascii=False), encoding="utf-8")
    (evidence / f"room-{case}-row.html").write_text(html, encoding="utf-8")

    word, line, controls = ROOM[case]
    assert item["word"] == word, item
    assert [part for part in line if part not in item["secondary"]] == [], item
    assert item["more"] == (controls is not None) and menu == controls, (item, menu)

"""A work-list row reaches its work's message wherever it lies (DESIGN "Room work status", "History edges").

The real server, history endpoint, archive reader and SPA of ``direct_server_with_data``. The room's saved
log holds a task's answer (or, for a task that only narrated, its narration) far back -- newer saved lines
follow it, so a fresh session's newest page does not hold it -- and the census lists the task with its
post-task work paused by the owner (D10), which is no current work, so no card or row of it is loaded. Only
that census row is added to the server's own ``/api/state`` answer. Before a press the row claims nothing about
where its message is; the press reads the page around the task (``around=task:<id>``) in one history read,
never walking the pages between, closes the list and lands on the task's own rows; Back to reading then reads
the present the reader was following again.
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess

import pytest

from tests import test_ui_smoke_playwright as smoke
from tests._work_projection_shared import with_work_fields
from tests.test_chat_history_paging_browser import _idle
from tests.test_room_work_backend_browser import Evidence, _decision_posts, _http, _launch, _open_chat, _wait_facts
from tests.test_room_work_pager_return_browser import FILLER, LATER, _FRAMES, _seed_filler

direct_server_with_data = smoke.direct_server_with_data

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

TASK = "jump-far-task"
ANSWER = "JUMP The release audit found two missing citations."
NARRATION = "JUMP Comparing the cited pages with the release notes."

# Every history read the page makes, with the place it asked for (cursor or around).
_OBSERVE = """() => {
    const fetch = window.fetch.bind(window);
    window.__historyReads = [];
    window.fetch = async (input, init) => {
        const url = new URL(typeof input === 'string' ? input : input.url, location.href);
        if (url.pathname !== '/api/chat/history') return fetch(input, init);
        const read = {cursor: url.searchParams.get('cursor'), around: url.searchParams.get('around'), done: false};
        window.__historyReads.push(read);
        try {
            const response = await fetch(input, init);
            read.status = response.status;
            read.body = await response.clone().json();
            read.done = true;
            return response;
        } catch (error) { read.error = String(error); read.done = true; throw error; }
    };
}"""
# A node of the feed on screen: one of the task's own (its card, its row) or the row the read landed on.
_ON_SCREEN = """([feed, id, historyId]) => {
    const root = document.querySelector(feed), box = root.getBoundingClientRect();
    return [...root.querySelectorAll('[data-task-id], [data-history-id]')].filter(node => node.dataset.taskId === id
        || (historyId && node.dataset.historyId === historyId)).some(node => {
        const r = node.getBoundingClientRect();
        return r.height > 0 && r.bottom > box.top && r.top < box.bottom; });
}"""
_NEWEST_ON_SCREEN = """([feed, text]) => {
    const root = document.querySelector(feed), box = root.getBoundingClientRect();
    return [...root.querySelectorAll('[data-history-id]')].some(node => node.textContent.includes(text)
        && node.getBoundingClientRect().bottom > box.top && node.getBoundingClientRect().top < box.bottom);
}"""


def _seed_task(data_dir, chat_id, work):
    """The task's own rows, written after the older filler and before the newer one: its exchange, or (a
    task that only narrated) its narration followed by more newer narration than a newest page holds."""
    now = _dt.datetime.now(_dt.timezone.utc)

    def stamp(seconds):
        return (now - _dt.timedelta(seconds=seconds)).isoformat()

    if work == "answered":
        rows, log = [{"ts": stamp(2), "chat_id": chat_id, "direction": "in", "client_message_id": "jump-ask",
                      "text": "JUMP Please audit the release notes."},
                     {"ts": stamp(1), "chat_id": chat_id, "direction": "out", "task_id": TASK, "text": ANSWER}], "chat"
    else:
        rows, log = [{"ts": stamp(3), "chat_id": chat_id, "task_id": TASK, "content": NARRATION},
                     *({"ts": stamp(2), "chat_id": chat_id, "task_id": "jump-later-task", "content": f"JUMP later {n}"}
                       for n in range(120))], "progress"
    with (data_dir / "logs" / f"{log}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in rows))


def _row(facts, task_id):
    return next((row for row in facts["rows"] if row["id"] == task_id), None)


@pytest.mark.parametrize(("scope", "engine", "work"), [
    ("main", "chromium", "answered"), ("main", "webkit", "answered"), ("project", "chromium", "answered"),
    ("project", "webkit", "answered"), ("main", "chromium", "narrated")])
def test_a_work_row_far_back_lands_on_its_work_and_back_returns(direct_server_with_data, tmp_path, scope, engine, work):
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    evidence = Evidence(data_dir, f"jump-{scope}-{engine}-{work}")
    shown = ANSWER if work == "answered" else NARRATION
    project = None
    if scope == "project":
        folder = tmp_path / "jump-room"
        folder.mkdir()
        subprocess.run(["git", "init", "-q", str(folder)], check=True, capture_output=True)
        project = _http(url, "POST", "/api/projects", {"name": "Jump reading room", "path": str(folder)})["project"]
    chat_id = int(project["chat_id"]) if project else 1
    room = ({"scope": "main", "root": "#page-chat", "feed": "#chat-messages"} if not project else
            {"scope": "project", "root": "#project-panel", "feed": f"#pchat-{project['id']}-messages"})
    tag = "P" if project else "M"
    server["stop_server"]()
    _seed_filler(data_dir, chat_id, tag, FILLER)
    _seed_task(data_dir, chat_id, work)
    _seed_filler(data_dir, chat_id, tag, LATER, later=True)
    server["start_server"]()

    def with_paused_answer(route):
        """The server's census, plus the task whose answer is far back and whose post-task work is paused, with the
        server's work projection of that census."""
        response = route.fetch()
        body = response.json()
        body["active_chat_activities"] = [*(body.get("active_chat_activities") or []), {
            "activity_id": TASK, "kind": "managed_task", "phase": "budget_paused", "pause_cause": "owner",
            "chat_id": chat_id, "task_attempt": 1, "started_at": 1792000000}]
        route.fulfill(response=response, json=with_work_fields(body))

    with sync_playwright() as pw:
        browser, page, log = _launch(pw, engine)
        page.add_init_script(f"({_OBSERVE})()")
        page.route("**/api/state", with_paused_answer)
        try:
            _open_chat(page, url)
            if project:
                page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))",
                              project)
                page.wait_for_selector(f"{room['root']} .project-panel-bar .rw-trigger", state="attached",
                                       timeout=30_000)
            _idle(page, room["feed"])
            # Nothing of the task is loaded: its rows lie pages back.
            assert page.locator(f'{room["feed"]} [data-task-id="{TASK}"]').count() == 0
            assert shown not in page.locator(room["feed"]).inner_text()
            page.locator(f"{room['root']} .rw-trigger").click()
            opened = _wait_facts(page, room["scope"], lambda f: _row(f, TASK) is not None, "the paused task's row")
            evidence.record("opened", opened)
            row = _row(opened, TASK)
            # Before it asks, the row claims nothing about where its message is.
            assert row["target"] == "history" and row["label"].startswith("Open: "), row
            assert "loaded" not in json.dumps(opened), opened
            evidence.shot(page, "01-list-before-press")

            before = page.evaluate("() => window.__historyReads.length")
            page.locator(f"{room['root']} .rw-row[data-id='{TASK}'] .rw-main").click()
            landed = _wait_facts(page, room["scope"], lambda f: f["surfaceHidden"] and f["back"] == "Back to reading",
                                 "landed on the task, Back offered")
            page.evaluate(_FRAMES)
            reads = page.evaluate("n => window.__historyReads.slice(n)", before)
            evidence.record("landed", {"facts": landed, "reads": [{k: r.get(k) for k in ("cursor", "around", "status")}
                                                                   for r in reads]})
            # One history read around the task, never a walk of the pages between.
            assert [(read["cursor"], read["around"]) for read in reads] == [(None, f"task:{TASK}")], reads
            anchor = reads[0]["body"]["anchor"]
            assert anchor["state"] == "found" and anchor["task_id"] == TASK, anchor
            assert "after_cursor" not in reads[0]["body"], "a page around a task is not the present"
            page.wait_for_function(_ON_SCREEN, arg=[room["feed"], TASK, anchor["history_id"]], timeout=30_000)
            assert shown in page.locator(room["feed"]).inner_text()
            evidence.shot(page, "02-landed-on-the-task")

            # Back returns to the present the reader was following: it reads the newest page again, whatever
            # stream the jumped page held.
            before = page.evaluate("() => window.__historyReads.length")
            page.locator(f"{room['root']} .rw-back").click()
            newest = f"PG-{tag}L-{LATER - 1:04d}"
            page.wait_for_function(_NEWEST_ON_SCREEN, arg=[room["feed"], newest], timeout=30_000)
            back = _wait_facts(page, room["scope"], lambda f: not f["back"], "Back leaves once it has returned")
            page.wait_for_function("n => window.__historyReads.slice(n).some(r => !r.cursor && !r.around && r.done)",
                                   arg=before, timeout=30_000)
            evidence.record("back", back)
            evidence.shot(page, "03-back-to-the-present")
            assert _decision_posts(log) == [], "navigation answers nothing"
            assert not log["pageerrors"], log["pageerrors"]
        finally:
            browser.close()

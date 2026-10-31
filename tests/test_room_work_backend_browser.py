"""The room's work status (DESIGN "Room work status") through the REAL server, gateway, worker
pool and SPA.

Each test runs the shared ``direct_server_with_data`` fixture (a byte-faithful candidate checkout
with origin proof, an isolated data root, ``server.py`` with ONE worker and the in-process mock
model). Only model judgment is scripted, at the ``fixtures_mock_llm`` handler seam; queue, census,
Pause/Resume, owner waits, task detail and the WebSocket are the product's own.

``test_main_sheet_follows_real_roots_through_pause_resume_and_finish``
  Three managed roots are restored the way a Quit leaves them and resumed by the owner: one runs
  (its first model call held), two queue behind it. Main's status control lists exactly those
  roots; the ⋯ of the running row opens the shared task controls for that exact id; Pause from
  the row is the real Pause; Resume is the row's own shared button. While the sheet stays open its
  order holds, words change in place, and finished work keeps its name and outcome without
  controls; after close Main keeps no finished list.

``test_main_view_question_goes_to_the_exact_waited_question``
  A direct turn asks a waited question (``escalate``) and waits. The row offers View question;
  it lands on the question itself (no answer field focus, no decision POST), and Back to reading
  returns. With the question outside the loaded history (the history read is filtered in the
  browser) the reveal reads the real task detail and must prove Main's room by its chat id.

``test_project_shade_lists_only_its_room_and_survives_disconnect``
  A Project direct turn runs held. Its Shade lists it and Main does not; an outside press on the
  dimmed Main only dismisses the Shade; a socket interruption keeps the last known row as
  unconfirmed without controls and says so; reconnect restores controls; switching to another
  Project takes the control along; after Finish the Project keeps its latest represented root.

Evidence (screenshots at 1100x722 and 390x844, facts JSON) goes to ``OUROBOROS_UI_EVIDENCE_DIR``.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import threading
import time

import pytest

from ouroboros import dialogue_store
from tests import test_ui_smoke_playwright as smoke
from tests.test_room_work_browser import ROOM_FACTS
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET
from tests.ui_console_errors import PageErrors

direct_server_with_data = smoke.direct_server_with_data

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

ALPHA, BRAVO, CHARLIE = "rw-alpha", "rw-bravo", "rw-charlie"
TEXT = {
    ALPHA: "RW-ALPHA-9K: inventory the available tools, then report back.",
    BRAVO: "RW-BRAVO-9K: draft the weekly summary of the reading notes.",
    CHARLIE: "RW-CHARLIE-9K: tidy the archived notes.",
}
TITLE = {ALPHA: "Inventory of the available tools", BRAVO: "Weekly summary draft", CHARLIE: "Archived notes tidy-up"}
ASK = "RW-ASK-9K"
QUESTION = "Which evidence should the weekly summary cite first?"
PROJECT_HOLD = "RW-PROJECT-9K"


# --------------------------------------------------------------------------------------------
# The scripted model: answers per marker at the mock handler seam (the one the fixture serves).

def _user_text(message):
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(block.get("text") or "") for block in content if isinstance(block, dict))
    return str(content or "")


def _reply(handler, payload, message, finish):
    streaming = payload.get("stream") is True
    if streaming and message.get("tool_calls"):
        message = {**message, "tool_calls": [{**call, "index": index}
                                             for index, call in enumerate(message["tool_calls"])]}
    choice = {"index": 0, "finish_reason": finish, ("delta" if streaming else "message"): message}
    answer = {"id": "rw-completion", "object": "chat.completion", "choices": [choice],
              "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    body = (f"data: {json.dumps(answer)}\n\ndata: [DONE]\n\n" if streaming else json.dumps(answer)).encode()
    try:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except OSError:
        pass  # the calling worker was stopped while this call was held


class ScriptedModel:
    """``holds`` maps a marker to the Event its FIRST main call waits on; ``ask`` makes a marker's
    first main call an ``escalate`` that waits for the owner. Everything else answers ``OK``."""

    def __init__(self, monkeypatch, *, holds=(), ask=()):
        from tests import fixtures_mock_llm

        self.lock = threading.Lock()
        self.calls: list[tuple[str, bool]] = []
        self.release = {marker: threading.Event() for marker in holds}
        self.entered = {marker: threading.Event() for marker in [*holds, *ask]}
        self.ask = set(ask)
        self.teardown = threading.Event()
        monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", lambda handler: self.handle(handler))

    def markers(self):
        return [*self.release, *self.ask]

    def marker(self, payload):
        text = "\n".join(_user_text(m) for m in payload.get("messages") or []
                         if isinstance(m, dict) and m.get("role") == "user")
        return next((name for name in self.markers() if name in text), "OTHER")

    def main_calls(self, marker):
        with self.lock:
            return self.calls.count((marker, True))

    def handle(self, handler):
        length = int(handler.headers.get("Content-Length", 0) or 0)
        payload = json.loads(handler.rfile.read(length) or b"{}")
        marker, main = self.marker(payload), bool(payload.get("tools"))
        with self.lock:
            self.calls.append((marker, main))
            ordinal = self.calls.count((marker, True))
        message, finish = {"role": "assistant", "content": "OK"}, "stop"
        if main and ordinal == 1 and marker in self.entered:
            self.entered[marker].set()
        if main and ordinal == 1 and marker in self.release:
            self.release[marker].wait(600)
        elif main and ordinal == 1 and marker in self.ask:
            message, finish = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_rw_question", "type": "function", "function": {"name": "escalate", "arguments": json.dumps({
                    "question": QUESTION, "wait_for_answer": True,
                    "options": [{"label": "The primary notes"}, {"label": "The replication"}]})}}]}, "tool_calls"
        elif main and ordinal == 2 and marker in self.ask:
            # The wait keeps the turn parked; a call here means it resumed (an answer or a stop).
            self.teardown.wait(600)
        _reply(handler, payload, message, finish)

    def finish(self):
        for event in self.release.values():
            event.set()
        self.teardown.set()


# --------------------------------------------------------------------------------------------
# Seeding (only while the server is stopped) and HTTP helpers.

def _seed_roots(data_dir, task_ids, filler=0):
    """Accepted, never-dispatched roots in the queue snapshot, each with the owner's message, its
    scheduled result and a progress row; ``filler`` earlier exchanges give Main real scrolling."""
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result

    now = _dt.datetime.now(_dt.timezone.utc)
    (data_dir / "state").mkdir(parents=True, exist_ok=True)
    chat, progress, pending = [], [], []
    for index in range(filler):
        ts = (now - _dt.timedelta(minutes=filler - index)).isoformat()
        chat.append({"ts": ts, "direction": "in", "chat_id": 1, "text": f"Earlier question {index}: what changed in section {index}?",
                     "client_message_id": f"rw-filler-{index}"})
        chat.append({"ts": ts, "direction": "out", "chat_id": 1,
                     "text": f"Earlier answer {index}. " + "The reading notes continue with ordinary context. " * 5})
    for seq, task_id in enumerate(task_ids, start=1):
        text, title = TEXT[task_id], TITLE[task_id]
        queued_at = (now + _dt.timedelta(seconds=seq)).isoformat()
        ref = build_owner_message_ref(chat_id=1, client_message_id=f"seed-{task_id}", ts=queued_at, text=text)
        chat.append({"ts": queued_at, "direction": "in", "chat_id": 1, "text": text, "client_message_id": f"seed-{task_id}"})
        task = {"id": task_id, "type": "task", "chat_id": 1, "priority": 0, "text": text, "description": text,
                "objective": text, "title": title, "root_task_id": task_id, "delegation_role": "root",
                "origin_message_text": text, "origin_message_ref": ref, "admitted_dispatch": "none"}
        write_task_result(data_dir, task_id, "scheduled", result="Task is queued.", description=text, objective=text,
                          chat_id=1, title=title, delegation_role="root", root_task_id=task_id,
                          origin_message_text=text, origin_message_ref=ref)
        pending.append({"id": task_id, "type": "task", "priority": 0, "attempt": 1, "queued_at": queued_at,
                        "queue_seq": seq, "task": task})
        progress.append({"ts": queued_at, "chat_id": 1, "task_id": task_id, "content": f"Working on {title}",
                         "cancelable": True})
    (data_dir / "state" / "queue_snapshot.json").write_text(json.dumps({
        "_schema_version": 1, "ts": now.isoformat(), "reason": "room_work_browser_seed",
        "pending_count": len(pending), "running_count": 0, "reaping_count": 0,
        "acceptance_fences": [], "budget_root_fences": [], "pending": pending, "running": [],
    }), encoding="utf-8")
    for stream, rows in (("chat", chat), ("progress", progress)):
        for row in rows:
            dialogue_store.append(data_dir, stream, row)


def _http(url, method, path, body=None):
    import urllib.request

    data = json.dumps(body or {}).encode() if method == "POST" else None
    request = urllib.request.Request(url + path, method=method, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - the local candidate server
        return json.loads(response.read() or b"{}")


def _wait(predicate, timeout, what, interval=0.5):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = predicate()
        except Exception as exc:  # a server between generations answers nothing
            last = exc
        else:
            if last:
                return last
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}; last={last!r}")


def _resume_after_app_stop(url, *task_ids):
    """The seeded snapshot is what a Quit leaves: each root waits under ``saved_work_hold`` for the
    owner's explicit Resume (owner S1), which admits it as ordinary queued work, in order."""
    for task_id in task_ids:
        _wait(lambda: _http(url, "GET", f"/api/tasks/{task_id}").get("reason_code") == "saved_work_hold",
              30, f"{task_id} held after the application stop")
        answer = _http(url, "POST", f"/api/tasks/{task_id}/resume")
        assert answer.get("ok"), answer


def _census(url):
    return {row["activity_id"]: row for row in _http(url, "GET", "/api/state").get("active_chat_activities") or []}


def _phase(url, task_id):
    return str((_census(url).get(task_id) or {}).get("phase") or "")


def _fence(data_dir, task_id):
    from ouroboros.owner_pause import read_fence

    return read_fence(data_dir, task_id)


# --------------------------------------------------------------------------------------------
# Browser helpers.

class Evidence:
    def __init__(self, data_dir, name):
        root = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
        self.dir = root / "room-work-backend" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.facts: dict = {}

    def shot(self, page, name):
        page.screenshot(path=str(self.dir / f"{name}.png"), animations="disabled")

    def record(self, label, value):
        self.facts[label] = value
        (self.dir / "facts.json").write_text(json.dumps(self.facts, indent=2, default=str), encoding="utf-8")


def _launch(pw, engine, *, width=1100, height=722, theme="dark"):
    browser = getattr(pw, engine).launch(headless=True)
    page = browser.new_page(viewport={"width": width, "height": height})
    log = {"pageerrors": PageErrors(page, engine), "console": [], "requests": []}
    page.on("console", lambda msg: log["console"].append(f"{msg.type}: {msg.text}")
            if msg.type in {"error", "warning"} else None)
    page.on("request", lambda request: log["requests"].append((request.method, request.url))
            if "/api/" in request.url else None)
    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    return browser, page, log


def _open_chat(page, url):
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("#page-chat .rw-trigger", state="attached", timeout=30_000)
    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)", timeout=30_000)


def _facts(page, scope="main"):
    return page.evaluate(ROOM_FACTS, scope)


def _wait_facts(page, scope, predicate, what, timeout=45):
    deadline = time.monotonic() + timeout
    facts = None
    while time.monotonic() < deadline:
        facts = _facts(page, scope)
        if predicate(facts):
            return facts
        page.wait_for_timeout(250)
    raise AssertionError(f"timed out waiting for {what}; facts={json.dumps(facts, indent=1)}")


def _row(facts, task_id):
    return next((row for row in facts["rows"] if row["id"] == task_id), None)


def _decision_posts(log):
    return [(method, url) for method, url in log["requests"] if method == "POST" and "/api/decisions" in url]


# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_main_sheet_follows_real_roots_through_pause_resume_and_finish(direct_server_with_data, monkeypatch, engine):
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    evidence = Evidence(data_dir, f"main-roots-{engine}")
    marks = {ALPHA: "RW-ALPHA-9K", BRAVO: "RW-BRAVO-9K"}
    model = ScriptedModel(monkeypatch, holds=marks.values())
    server["stop_server"]()
    _seed_roots(data_dir, (ALPHA, BRAVO, CHARLIE), filler=14)
    server["start_server"]()
    _resume_after_app_stop(url, ALPHA, BRAVO, CHARLIE)
    try:
        with sync_playwright() as pw:
            browser, page, log = _launch(pw, engine)
            try:
                assert model.entered[marks[ALPHA]].wait(90), "ALPHA never reached its first model call"
                _wait(lambda: _phase(url, ALPHA) == "working" and _phase(url, BRAVO) == "queued"
                      and _phase(url, CHARLIE) == "queued", 60, "ALPHA working, BRAVO and CHARLIE queued")
                _open_chat(page, url)
                closed = _wait_facts(page, "main", lambda f: f["tone"] == "live" and not f["disabled"],
                                     "Main's control to show live work")
                assert closed["surfaceHidden"] is True and closed["expanded"] == "false", closed
                assert closed["badge"], "the room's own status word stays inside the control"
                trigger = page.locator("#page-chat .rw-trigger")
                assert trigger.locator("#chat-status").count() == 1, "the real status badge is the control's word"
                evidence.record("closed", closed)
                evidence.shot(page, "01-main-closed-1100")

                trigger.click()
                opened = _wait_facts(page, "main", lambda f: f["surfaceHidden"] is False and len(f["rows"]) == 3
                                     and all(r["name"] and r["target"] == "card" for r in f["rows"]),
                                     "the sheet with three named roots, each on its card")
                evidence.record("opened", opened)
                ids = [row["id"] for row in opened["rows"]]
                assert sorted(ids) == sorted([ALPHA, BRAVO, CHARLIE]), opened
                assert ids[0] == ALPHA, "moving work leads; queued work follows"
                alpha, bravo, charlie = (_row(opened, t) for t in (ALPHA, BRAVO, CHARLIE))
                assert alpha["tone"] == "live" and alpha["word"] == "Working", alpha
                assert bravo["tone"] == charlie["tone"] == "queued", (bravo, charlie)
                assert bravo["word"] == charlie["word"] == "Queued", (bravo, charlie)
                cards = page.evaluate("""ids => Object.fromEntries(ids.map(id => [id,
                    document.querySelector(`#chat-messages .chat-live-card[data-task-id="${id}"] .chat-live-title`)
                        ?.textContent.replace(/\\s+/g, ' ').trim() || '']))""", ids)
                evidence.record("card_titles", cards)
                for task_id in ids:
                    row = _row(opened, task_id)
                    assert row["target"] == "card", row
                    # A row names its work by the request's title; its card shows the same work.
                    assert row["name"] == TITLE[task_id] and TITLE[task_id] in cards[task_id], ("a row names its work", row, cards)
                assert alpha["more"] and alpha["moreId"] == ALPHA, alpha
                # One Shade unfolds under Main's bar, which marks itself open (DESIGN "Room work status").
                assert page.locator("#page-chat .chat-page-header[data-rw-open] .rw-shade:not([hidden])").count() == 1
                assert opened["active"]["inSurface"], "a pointer open lands in the list"
                evidence.shot(page, "02-main-open-1100")

                # The ⋯ is the shared task controls for that exact id; Pause is the real Pause.
                more = page.locator(f"#page-chat .rw-row[data-id='{ALPHA}'] .rw-more")
                more.click()
                menu = page.locator("body > .task-control-menu")
                menu.wait_for(state="visible", timeout=10_000)
                actions = menu.evaluate("n => [...n.querySelectorAll('.task-control-item')].map(i => i.dataset.taskControl)")
                assert actions == ["finalize", "hurry", "pause", "stop_now"], actions
                assert more.get_attribute("aria-expanded") == "true"
                evidence.shot(page, "03-main-row-menu")
                with page.expect_response(lambda r: r.url.endswith(f"/api/tasks/{ALPHA}/pause")
                                          and r.request.method == "POST", timeout=30_000) as paused:
                    menu.locator('[data-task-control="pause"]').click()
                assert paused.value.ok, paused.value.text()
                evidence.record("pause_response", paused.value.json())
                assert not _facts(page, "main")["surfaceHidden"], "choosing from the row's menu keeps the list open"
                _wait(lambda: _fence(data_dir, ALPHA).get("state") == "paused", 60, "ALPHA's tree saved")
                _wait(lambda: _phase(url, ALPHA) == "budget_paused" and _phase(url, BRAVO) == "working",
                      60, "ALPHA paused and BRAVO running")
                paused_facts = _wait_facts(page, "main", lambda f: (_row(f, ALPHA) or {}).get("resume")
                                           and (_row(f, BRAVO) or {}).get("word") == "Working",
                                           "ALPHA's Resume and BRAVO working, in place")
                evidence.record("paused", paused_facts)
                assert [row["id"] for row in paused_facts["rows"]] == ids, "order holds while open"
                alpha = _row(paused_facts, ALPHA)
                # The owner's own Pause: neutral ink (amber is only what waits for the owner), the word alone, and
                # its line says who paused and when (the owner's client declares the press).
                assert (alpha["tone"], alpha["word"]) == ("held", "Paused"), alpha
                assert alpha["secondary"].startswith("You paused it · "), alpha
                assert alpha["more"], "a paused root keeps its shared controls"
                evidence.shot(page, "04-main-paused-open")

                # Resume is the row's own shared button for that exact id.
                with page.expect_response(lambda r: r.url.endswith(f"/api/tasks/{ALPHA}/resume")
                                          and r.request.method == "POST", timeout=30_000) as resumed:
                    page.locator(f"#page-chat .rw-row[data-id='{ALPHA}'] .rw-pill-resume").click()
                assert resumed.value.ok, resumed.value.text()
                evidence.record("resume_response", resumed.value.json())
                requeued = _wait_facts(page, "main", lambda f: not (_row(f, ALPHA) or {}).get("resume")
                                       and (_row(f, ALPHA) or {}).get("word") in {"Queued", "Working"},
                                       "ALPHA admitted again without its Resume")
                evidence.record("requeued", requeued)
                assert [row["id"] for row in requeued["rows"]] == ids

                # Finish while open: rows stay with names and outcomes, controls are withdrawn.
                model.finish()
                _wait(lambda: all(_http(url, "GET", f"/api/tasks/{t}").get("status") == "completed"
                                  for t in (ALPHA, BRAVO, CHARLIE)), 180, "all three roots completed")
                finished = _wait_facts(page, "main", lambda f: all(r["settled"] for r in f["rows"])
                                       and all(r["word"] == "Done" for r in f["rows"]),
                                       "three settled rows saying Done", timeout=60)
                evidence.record("finished_open", finished)
                assert [row["id"] for row in finished["rows"]] == ids
                for row in finished["rows"]:
                    assert row["name"] == TITLE[row["id"]], row
                    assert not (row["more"] or row["resume"] or row["question"]), row
                    assert row["tone"] == "done" and row["target"] == "card", row
                assert not finished["surfaceHidden"], "work that ends never closes the list"
                evidence.shot(page, "05-main-finished-open")

                # Escape closes from inside; Main keeps no finished list.
                page.locator(f"#page-chat .rw-row[data-id='{ALPHA}'] .rw-main").focus()
                page.keyboard.press("Escape")
                _wait_facts(page, "main", lambda f: f["surfaceHidden"] is True, "Escape to close")
                # Closed rows are not repainted; a disabled control can never show them again.
                gone = _wait_facts(page, "main", lambda f: f["disabled"] and f["surfaceHidden"], "no finished list in Main")
                evidence.record("closed_after_finish", gone)
                # The keyboard reader is not dropped to the document when the closed control disables itself.
                focus_after_close = page.evaluate("""() => { const a = document.activeElement;
                    return a === document.body || !a ? 'body' : `${a.tagName}.${a.className}`; }""")
                evidence.record("focus_after_escape_on_finished_list", focus_after_close)
                trigger.click(force=True)
                assert _facts(page, "main")["surfaceHidden"] is True, "a disabled control opens nothing"
                evidence.shot(page, "06-main-closed-after-finish")
                assert _decision_posts(log) == []
                assert log["pageerrors"] == [], log["pageerrors"]
                assert focus_after_close != "body", "Escape on a list of finished work dropped focus to the document"
            except Exception:
                evidence.shot(page, "failure")
                raise
            finally:
                evidence.record("log", log)
                browser.close()
    finally:
        model.finish()
        evidence.record("model_calls", list(model.calls))


# --------------------------------------------------------------------------------------------

def _send_from_composer(page, text, scope="main", project_id=""):
    root = "#page-chat" if scope == "main" else f"#panel-pchat-{project_id}"
    page.locator(f"{root} textarea").first.fill(text)
    page.locator(f"{root} .chat-send-inline").first.click()


def _waiting_turn(url, chat_id=1):
    for activity in _census(url).values():
        wait = activity.get("owner_wait") or {}
        if int(activity.get("chat_id") or 1) == chat_id and wait.get("owner_wait_state") == "waiting" and wait.get("quiz_id"):
            return activity
    return None


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_main_view_question_goes_to_the_exact_waited_question(direct_server_with_data, monkeypatch, engine):
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    evidence = Evidence(data_dir, f"main-question-{engine}")
    model = ScriptedModel(monkeypatch, ask=[ASK])
    server["stop_server"]()
    _seed_roots(data_dir, (), filler=16)
    server["start_server"]()
    try:
        with sync_playwright() as pw:
            browser, page, log = _launch(pw, engine)
            try:
                _open_chat(page, url)
                _send_from_composer(page, f"{ASK}: check which evidence the weekly summary should cite, and ask me first.")
                turn = _wait(lambda: _waiting_turn(url), 90, "a direct turn waiting on its own question")
                task_id, quiz_id = turn["activity_id"], turn["owner_wait"]["quiz_id"]
                detail = _http(url, "GET", f"/api/tasks/{task_id}")
                evidence.record("census_turn", turn)
                evidence.record("detail_stamp", {k: detail.get(k) for k in ("task_id", "chat_id", "project_id", "status")})
                assert detail.get("chat_id") == 1 and not detail.get("project_id"), "Main's question proves Main"
                waiting = _wait_facts(page, "main", lambda f: (_row(f, task_id) or {}).get("question")
                                      or f["tone"] == "wait", "the control in the owner's turn")
                assert waiting["tone"] in {"wait", "live"}, waiting
                card = page.locator(f'#chat-messages .chat-quiz-card[data-task-id="{task_id}"][data-quiz-id="{quiz_id}"]')
                card.wait_for(state="attached", timeout=30_000)

                def view_question(label):
                    feed = page.locator("#chat-messages")
                    feed.evaluate("el => { el.scrollTop = Math.round(el.scrollHeight * 0.3); }")
                    page.wait_for_timeout(300)
                    reading_at = feed.evaluate("el => el.scrollTop")
                    page.locator("#page-chat .rw-trigger").click()
                    opened = _wait_facts(page, "main", lambda f: (_row(f, task_id) or {}).get("question"), "View question")
                    row = _row(opened, task_id)
                    evidence.record(f"{label}_row", row)
                    assert row["word"] == "Waiting for your answer" and row["tone"] == "wait", row
                    title = page.evaluate("""id => document.querySelector(`#chat-messages .chat-live-card[data-task-id="${id}"]
                        .chat-live-title`)?.textContent.replace(/\\s+/g, ' ').trim() || ''""", task_id)
                    evidence.record(f"{label}_card_title", title)
                    # The card's coined name (the mock namer answers "OK"), else the owner's own message.
                    assert row["name"] == title if title else row["name"].startswith(f"Reply to “{ASK}"), (row, title)
                    evidence.shot(page, f"{label}-open")
                    page.locator(f"#page-chat .rw-row[data-id='{task_id}'] .rw-pill-question").click()
                    card.wait_for(state="visible", timeout=30_000)
                    page.wait_for_function("() => document.activeElement?.classList.contains('chat-quiz-question')",
                                           timeout=30_000)
                    landed = card.evaluate("""card => ({ focus: document.activeElement.className,
                        comment: card.querySelector('.chat-quiz-comment')?.value ?? null,
                        enabled: [...card.querySelectorAll('.chat-quiz-option')].filter(b => !b.disabled).length })""")
                    evidence.record(f"{label}_landed", landed)
                    assert landed["comment"] in ("", None) and landed["enabled"] == 2, landed
                    assert _facts(page, "main")["back"] == "Back to reading"
                    evidence.shot(page, f"{label}-landed")
                    page.locator("#page-chat .rw-back").click()
                    page.wait_for_timeout(400)
                    assert abs(feed.evaluate("el => el.scrollTop") - reading_at) <= 40, "Back to reading returns"

                view_question("loaded")
                # The question outside the loaded history: the history read is filtered in the browser, so the
                # reveal reads the real task detail and must prove Main's room by its chat id.
                def drop_question(route):
                    response = route.fetch()
                    body = response.json()
                    for key in ("messages", "progress"):
                        if isinstance(body.get(key), list):
                            body[key] = [m for m in body[key] if quiz_id not in json.dumps(m)]
                    route.fulfill(response=response, json=body)

                page.route("**/api/chat/history*", drop_question)
                _open_chat(page, url)
                page.wait_for_timeout(1500)
                assert card.count() == 0, "the question is outside the loaded history"
                reads = len([1 for method, u in log["requests"] if method == "GET" and u.endswith(f"/api/tasks/{task_id}")])
                view_question("evicted")
                assert len([1 for method, u in log["requests"] if method == "GET"
                            and u.endswith(f"/api/tasks/{task_id}")]) > reads, "the reveal read the real detail"
                assert _decision_posts(log) == [], "View question answers nothing"
                assert _waiting_turn(url)["activity_id"] == task_id, "the turn still waits for the owner"
                assert log["pageerrors"].unexpected() == [], log["pageerrors"]
            except Exception:
                evidence.shot(page, "failure")
                raise
            finally:
                evidence.record("log", log)
                browser.close()
    finally:
        model.finish()
        evidence.record("model_calls", list(model.calls))


# Every frame for a while: the Shade's note and rows (a socket gap lasts about a second).
_SHADE_RECORDER = """(ms) => {
    window.__rwShade = [];
    const start = performance.now();
    const tick = () => {
        const shade = document.querySelector('#project-panel .rw-shade:not(.cx-shade)');
        const note = shade?.querySelector('.rw-note');
        window.__rwShade.push({ t: Math.round(performance.now() - start),
            note: note && !note.hidden ? note.textContent : '',
            rows: [...(shade?.querySelectorAll('.rw-row') || [])].map((row) => ({ id: row.dataset.id,
                tone: row.dataset.tone, more: Boolean(row.querySelector('.rw-more')) })) });
        if (performance.now() - start < ms) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}"""


def test_project_shade_lists_only_its_room_and_survives_disconnect(direct_server_with_data, monkeypatch, tmp_path):
    import subprocess

    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    evidence = Evidence(data_dir, "project-chromium")
    model = ScriptedModel(monkeypatch, holds=[PROJECT_HOLD])
    projects = []
    for name in ("Archive evidence review", "Release notes"):
        folder = tmp_path / name.replace(" ", "-")
        folder.mkdir()
        subprocess.run(["git", "init", "-q", str(folder)], check=True, capture_output=True)
        projects.append(_http(url, "POST", "/api/projects", {"name": name, "path": str(folder)})["project"])
    first, second = projects
    try:
        with sync_playwright() as pw:
            browser, page, log = _launch(pw, "chromium")
            try:
                _open_chat(page, url)
                page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))", first)
                page.wait_for_selector("#project-panel .project-panel-bar .rw-trigger", state="attached", timeout=30_000)
                _send_from_composer(page, f"{PROJECT_HOLD}: read the archive folder and summarise it.", "project", first["id"])
                assert model.entered[PROJECT_HOLD].wait(90), "the Project turn never reached the model"
                turn = _wait(lambda: next((a for a in _census(url).values() if a.get("project_id") == first["id"]), None),
                             60, "the Project turn in the census")
                task_id = turn["activity_id"]
                live = _wait_facts(page, "project", lambda f: f["tone"] == "live" and not f["disabled"], "a live Shade control")
                evidence.record("closed", live)
                evidence.shot(page, "01-project-closed")
                main = _facts(page, "main")
                assert main["disabled"] and not main["rows"], ("Main never lists a Project's work", main)
                page.locator("#project-panel .project-panel-bar .rw-trigger").click()
                shade = _wait_facts(page, "project", lambda f: (_row(f, task_id) or {}).get("tone") == "live",
                                    "the confirmed Project turn")
                # A loaded card answers with its own Stop predicate (chat.js stopEligible: blockHasWork), so a fresh
                # turn's row may offer no ⋯ yet; recorded, not assumed.
                evidence.record("project_row_controls", {"more": _row(shade, task_id)["more"]})
                evidence.record("open", shade)
                row = _row(shade, task_id)
                assert row["name"].startswith(f"Reply to “{PROJECT_HOLD}"), row
                assert [r["id"] for r in shade["rows"]] == [task_id], shade
                evidence.shot(page, "02-project-open")

                # A socket interruption: the last known row stays, unconfirmed, without controls, and says so.
                page.evaluate(_SHADE_RECORDER, 6000)
                count = page.evaluate("() => window.__testSockets.length")
                page.evaluate("() => window.__testSockets.at(-1).close()")
                page.wait_for_function("n => window.__testSockets.length > n && window.__testSockets.at(-1).readyState === 1",
                                       arg=count, timeout=60_000)
                page.wait_for_timeout(3000)
                frames = page.evaluate("() => window.__rwShade")
                offline = [f for f in frames if f["note"].startswith("Reconnecting")]
                evidence.record("disconnect_frames", {"offline": offline[:3], "last": frames[-1], "count": len(frames)})
                assert offline, "the Shade said it was reconnecting"
                assert all(r["tone"] == "unknown" and not r["more"] for f in offline for r in f["rows"]), offline[:2]
                assert [r["id"] for r in offline[0]["rows"]] == [task_id], "the last known row stays"
                back = _wait_facts(page, "project", lambda f: not f["note"] and (_row(f, task_id) or {}).get("tone") == "live",
                                   "confirmed again after the reconnect", timeout=30)
                assert _row(back, task_id)["more"] == _row(shade, task_id)["more"], "controls return as they were"
                assert _row(back, task_id)["tone"] == "live", back

                # An outside press on the dimmed Main only dismisses the Shade, never the room.
                backdrop = page.locator("#project-panel-backdrop")
                if backdrop.is_visible():
                    backdrop.click(position={"x": 20, "y": 300})
                    _wait_facts(page, "project", lambda f: f["surfaceHidden"], "the Shade dismissed")
                    assert page.evaluate("() => !document.getElementById('project-panel').hidden"), "the room stayed open"

                # Another Project takes the control along; its own list is its own.
                page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))", second)
                page.wait_for_selector(f"#panel-pchat-{second['id']}", state="attached", timeout=30_000)
                other = _wait_facts(page, "project", lambda f: f["exists"], "the other Project's control")
                evidence.record("switched", other)
                assert other["disabled"] and not other["rows"], other
                assert page.locator(f"#panel-pchat-{first['id']}").count() == 0
                evidence.shot(page, "03-other-project")

                # Finish, then back in the first Project: nothing current, no controls.
                model.finish()
                _wait(lambda: task_id not in _census(url), 90, "the Project turn finished")
                page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))", first)
                page.wait_for_selector(f"#panel-pchat-{first['id']}", state="attached", timeout=30_000)
                page.wait_for_timeout(1500)
                after = _facts(page, "project")
                evidence.record("after_finish", after)
                if not after["disabled"]:
                    page.locator("#project-panel .project-panel-bar .rw-trigger").click()
                    after = _wait_facts(page, "project", lambda f: not f["surfaceHidden"], "the Shade after finish")
                    evidence.record("after_finish_open", after)
                    assert all(r["slot"] == "latest" or r["settled"] for r in after["rows"]), after
                    assert not any(r["more"] or r["resume"] or r["question"] for r in after["rows"]), after
                evidence.shot(page, "04-project-after-finish")
                assert _decision_posts(log) == []
                assert log["pageerrors"] == [], log["pageerrors"]
            except Exception:
                evidence.shot(page, "failure")
                raise
            finally:
                evidence.record("log", log)
                browser.close()
    finally:
        model.finish()
        evidence.record("model_calls", list(model.calls))

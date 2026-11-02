"""Opening an unread Project room shows the message that arrived last however far below the
first history page it lies (owner decision 2026-10-11; DESIGN "Unread").

Real server, real producer seam (``message_bus.send_with_budget``), real history endpoint and
read cursor. A reply is buried by its children's words — card content, so the reply stays the
room's newest message — deeper than the recent read and than the pages the pager keeps loaded,
or past the recent read's bounded search. The opening reads the older pages itself and lands on
the reply once; the reader's own move, a box scrolled, a question opened from Main, a closed or
hidden room or a failed page ends that reach, and a later read never resumes it.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from ouroboros.gateway.history_paging import _ARRIVAL_SEARCH_ROWS, _RECENT_WINDOW_ROWS
from tests.test_project_unread_read_band_browser import PANEL, _client, _unread, _wait_for
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _SETTLE_RESTORE_FRAMES, _emit_ws_frame

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
REPLY = "STANDALONE REPLY: show this on opening"
_OLDER = "/api/chat/history"
# The reply's box against the band the reader sees: the feed less the composer drawn over it.
_SHOWN = """(messages, text) => {
    const node = [...messages.querySelectorAll('[data-history-id]')].find(n => n.textContent.includes(text));
    if (!node) return null;
    const r = node.getBoundingClientRect(), feed = messages.getBoundingClientRect();
    const composer = messages.parentElement.querySelector('.chat-input-area').getBoundingClientRect();
    return {top: r.top, bottom: r.bottom, feedTop: feed.top, bandBottom: Math.min(feed.bottom, composer.top),
            nodes: messages.querySelectorAll('[data-history-id]').length};
}"""


def _buried_room(data, monkeypatch, project_id, children, chat_id=None):
    """A reply, then ``children`` words of 40 child tasks through the real producer, in a new room or
    after what ``chat_id`` holds."""
    from ouroboros.projects_registry import create_project, get_project
    from supervisor import message_bus

    chat_id = chat_id or create_project(data, project_id, name=project_id.replace("-", " ").title())["chat_id"]
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = lambda _frame: None
    monkeypatch.setattr(message_bus, "DATA_DIR", data)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7, "session_id": project_id})
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    message_bus.send_with_budget(chat_id, REPLY)
    revision = get_project(data, project_id)["visible_revision"]
    for index in range(children):
        child = f"child-{index % 40}"
        message_bus.send_with_budget(chat_id, f"Child word {index}", task_id=child, progress_meta={
            "delegation_role": "subagent", "subagent_task_id": child, "parent_task_id": "root-1", "root_task_id": "root-1"})
    assert get_project(data, project_id)["visible_revision"] == revision, "a child's words are card content"
    return chat_id, revision


def _cursor_kind(url):
    """The read a history request's cursor names: ``older`` walks a chain's pages, ``page`` reads one
    again, ``after`` takes what arrived since the window's newest read; None reads the newest page."""
    cursor = parse_qs(urlparse(url).query).get("cursor")
    if not cursor:
        return None
    return json.loads(base64.urlsafe_b64decode(cursor[0] + "=" * (-len(cursor[0]) % 4)))["kind"]


class _Room:
    """One client opening the unread room; older page requests are recorded, and one may be held or failed.
    Every other history read with a cursor (an ``after`` read of new arrivals) is recorded in ``reads``."""

    def __init__(self, browser, url, project_id, hold=None, fail=None, init=()):
        self.acks, self.older, self.held, self.reads = [], [], [], []
        self.page, self.row = _client(browser, url, project_id, self.acks, init)
        self.project_id = project_id

        def route(route):
            self.older.append(route.request.url)
            if len(self.older) == hold:
                self.held.append(route)
            elif len(self.older) == fail:
                route.fulfill(status=503, content_type="application/json", body='{"error": "unavailable"}')
            else:
                route.continue_()
        self.page.route(lambda url: _OLDER in url and _cursor_kind(url) == "older", route)
        self.page.on("request", lambda request: self.reads.append((_cursor_kind(request.url), request.url))
                     if _OLDER in request.url and _cursor_kind(request.url) not in (None, "older") else None)
        self.messages = self.page.locator(PANEL)

    def open(self):
        self.row.evaluate("el => el.click()")
        self.messages.locator(".chat-bubble").first.wait_for(state="attached")

    def seen(self):
        return [ack["project_seen_revision"][self.project_id] for ack in self.acks
                if self.project_id in (ack.get("project_seen_revision") or {})]

    def release(self):
        for route in self.held:
            try:
                route.continue_()
            except Exception:  # noqa: BLE001 - a closed room aborted the request it was waiting for
                pass
        self.held.clear()

    def settled(self, wait=1500):
        self.page.evaluate(_SETTLE_RESTORE_FRAMES)
        self.page.wait_for_timeout(wait)

    def shown(self):
        return self.messages.evaluate(_SHOWN, REPLY)


def _texts(response):
    """The texts of the rows a history read answered with (none for a read that failed)."""
    try:
        return [row.get("text") for row in response.json().get("messages", [])]
    except Exception:  # noqa: BLE001 - a failed or non-JSON answer brought no rows
        return []


def _evidence(data, name, **facts):
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / f"{name}.json").write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    return evidence


def _on_screen(box):
    return box is not None and box["top"] >= box["feedTop"] - 1 and box["top"] < box["bandBottom"]


def _in_band(box):
    """Some of the reply inside the band: what a read needs; a landing (`_on_screen`) puts its top there."""
    return box is not None and box["bottom"] > box["feedTop"] and box["top"] < box["bandBottom"]


@pytest.mark.parametrize("children", [160, 900, 1310], ids=["buried", "past-kept-pages", "past-bounded-search"])
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_an_unread_room_opens_on_its_reply_however_deep_its_childrens_words_bury_it(
        direct_server_with_data, monkeypatch, engine, children):
    """160 child words fill the recent read (the retained regression); 900 put the reply past the
    three pages the pager keeps; 1310 outrun the recent read's bounded search (its window of the room's
    newest rows and the rows it searches past it), so its quiet chain carries the search on, eight pages
    down. The opening reads the older pages itself, lands on the reply and reads it; a later read never
    pages again, and the room reopened once read opens at its bottom."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    project_id = f"buried-{children}"
    searched = children > _RECENT_WINDOW_ROWS + _ARRIVAL_SEARCH_ROWS
    chat_id, revision = _buried_room(data, monkeypatch, project_id, children)
    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            room = _Room(browser, url, project_id)
            recent = room.page.request.get(f"{url}/api/chat/history?chat_id={chat_id}").json()
            window = recent.get("window") or {}
            assert REPLY not in [row.get("text") for row in recent.get("messages") or []], "below the recent read"
            assert (window.get("latest_message") is None and isinstance(window.get("latest_before"), int)
                    if searched else window["latest_message"]["history_id"]), ("searched, or named", window)
            started = time.monotonic()
            room.open()
            read = _wait_for(room.page, lambda: room.seen() == [revision], attempts=200)
            seconds = round(time.monotonic() - started, 2)  # opening to acknowledgement, every page included
            room.settled(300)
            landed, pages = room.shown(), len(room.older)
            name = f"unread-opening-{engine}-{project_id}"
            evidence = _evidence(data, name, landed=landed, older=room.older, acks=room.acks, window=window,
                                 seconds=seconds)
            room.page.screenshot(path=str(evidence / f"{name}.png"))
            assert read and _on_screen(landed), ("the opening lands on the reply and reads it", landed, room.acks)
            assert pages >= {160: 1, 900: 5, 1310: 8}[children], ("it read the older pages itself", room.older)
            _unread(room.row).wait_for(state="detached")

            _emit_ws_frame(room.page, {"type": "projects_changed"})
            room.settled(800)
            assert len(room.older) == pages, "a later read finds no opening owed: it never pages"
            room.page.locator("#project-panel-close").evaluate("el => el.click()")
            room.open()
            room.settled()
            gap = room.messages.evaluate("n => n.scrollHeight - n.scrollTop - n.clientHeight")
            assert len(room.older) == pages and gap <= 1, ("a read room reopens at its bottom", gap, room.older)
        finally:
            browser.close()


_BOX = """messages => {
    const box = document.createElement('div');
    box.style.cssText = 'height: 120px; overflow-y: auto';
    box.innerHTML = '<div style="height: 1200px">A bounded box</div>';
    messages.insertBefore(box, messages.lastElementChild);
    box.scrollIntoView({block: 'center'});
    const r = box.getBoundingClientRect();
    return {x: r.left + r.width / 2, y: r.top + r.height / 2};
}"""


def _wheel(room, box):
    feed = room.messages.bounding_box()
    point = room.messages.evaluate(_BOX) if box else {"x": feed["x"] + feed["width"] / 2, "y": feed["y"] + 80}
    room.page.mouse.move(point["x"], point["y"])
    room.page.mouse.wheel(0, 120 if box else -400)


# Another tab shown: the document reports itself hidden and says so; shown again, the same in reverse.
_WINDOW = """hidden => {
    if (hidden) for (const [key, value] of [['hidden', true], ['visibilityState', 'hidden']]) {
        Object.defineProperty(document, key, {configurable: true, get: () => value});
    } else { delete document.hidden; delete document.visibilityState; }
    document.dispatchEvent(new Event('visibilitychange'));
}"""
# Leaving the room: its own close, or another page (showPage closes the open Project).
_LEAVE = {"close": "#project-panel-close", "settings": '[data-nav-page="settings"]'}


@pytest.mark.parametrize("move", ["wheel", "box", "close", "settings", "window"])
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_the_readers_own_move_ends_the_reach(direct_server_with_data, monkeypatch, engine, move):
    """While the second older page is in flight the reader wheels the feed, scrolls a bounded box,
    closes the room, goes to Settings or switches to another tab: that page may still be drawn, but
    nothing more is asked or landed, even once Chat or the tab is shown again and read, and the dot
    stays until the reply is shown."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    _buried_room(data, monkeypatch, "moved-room", 480)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            room = _Room(browser, direct_server_with_data["url"], "moved-room", hold=2)
            room.open()
            assert _wait_for(room.page, lambda: len(room.held) == 1, attempts=200), room.older
            room.settled(200)
            if move in _LEAVE:
                room.page.locator(_LEAVE[move]).evaluate("el => el.click()")
            elif move == "window":
                room.page.evaluate(_WINDOW, True)
            else:
                _wheel(room, move == "box")
                room.page.evaluate(_SETTLE_RESTORE_FRAMES)
            before = None if move in _LEAVE else room.messages.evaluate("n => n.scrollTop")
            room.release()
            room.settled()
            if move in ("settings", "window"):
                if move == "window":
                    room.page.evaluate(_WINDOW, False)
                else:
                    room.page.locator('[data-nav-page="chat"]').evaluate("el => el.click()")
                _emit_ws_frame(room.page, {"type": "projects_changed"})
                room.settled(800)
            landed = None if move in _LEAVE else room.shown()
            name = f"unread-opening-{engine}-moved-{move}"
            evidence = _evidence(data, name, landed=landed, older=room.older, acks=room.acks, before=before)
            room.page.screenshot(path=str(evidence / f"{name}.png"))
            assert len(room.older) == 2, ("nothing more is asked once the reader moved", room.older)
            assert room.seen() == [] and _unread(room.row).count() == 1, room.acks
            assert not _on_screen(landed), ("nothing lands over the reader's move", landed)
        finally:
            browser.close()


_ON_FEED = """n => {
    const r = n.getBoundingClientRect(), feed = n.closest('.chat-messages').getBoundingClientRect();
    return {top: r.top, bottom: r.bottom, feedTop: feed.top, feedBottom: feed.bottom,
            focused: n.contains(document.activeElement)};
}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_question_opened_from_main_during_the_reach_owns_the_room(direct_server_with_data, monkeypatch, engine):
    """While the second older page is in flight, Main's mirror opens a question of this room: the question
    has priority. That page may still be drawn, but nothing more is asked, the reply is not landed on, and
    showing the question reads nothing newer: the dot stays, and a later read does not resume the reach."""
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import bind_task_to_project, get_project
    from ouroboros.task_results import write_task_result

    data = direct_server_with_data["data_dir"]
    chat_id, _revision = _buried_room(data, monkeypatch, "asked-room", 480)
    project = get_project(data, "asked-room")
    quiz = {"quiz_id": "q-reach", "question": "Merge the deep-room fix now?", "options": ["Yes", "No"],
            "state": "open", "asked_at": "2026-09-28T10:05:00Z"}
    bind_task_to_project(data, "reach-task", project["id"], chat_id, origin={"absent": "system"})
    write_task_result(data, "reach-task", "running", project_id=project["id"], chat_id=chat_id,
                      owner_quiz={"q-reach": quiz})
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            room = _Room(browser, direct_server_with_data["url"], "asked-room", hold=2)
            room.open()
            assert _wait_for(room.page, lambda: len(room.held) == 1, attempts=200), room.older
            room.page.evaluate("""project => window.dispatchEvent(new CustomEvent('ouro:open-project', {
                detail: {project, task_id: 'reach-task', quiz_id: 'q-reach'}}))""", project)
            card = room.messages.locator('.chat-quiz-card[data-quiz-id="q-reach"]')
            card.wait_for(state="visible")
            room.release()
            room.settled()
            _emit_ws_frame(room.page, {"type": "projects_changed"})
            room.settled(800)
            landed, asked = room.shown(), card.evaluate(_ON_FEED)
            name = f"unread-opening-{engine}-question-mid-reach"
            evidence = _evidence(data, name, landed=landed, asked=asked, older=room.older, acks=room.acks)
            room.page.screenshot(path=str(evidence / f"{name}.png"))
            assert len(room.older) == 2, ("nothing more is asked once the question owns the room", room.older)
            assert asked["focused"] and asked["feedTop"] <= asked["top"] < asked["feedBottom"], asked
            assert not _on_screen(landed), ("the reply is not landed on", landed)
            assert room.seen() == [] and _unread(room.row).count() == 1, room.acks
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_room_kept_hidden_during_the_reach_asks_for_nothing_until_it_is_opened_again(
        direct_server_with_data, monkeypatch, engine):
    """A staged file keeps the room when it is closed mid-reach: hidden, not destroyed. The page in flight
    may be drawn, but the hidden room asks for nothing more and lands nowhere; opened again, it reads on
    to the reply, lands there once and reads it."""
    from playwright.sync_api import sync_playwright

    from tests.test_history_bookmark_browser import _stage_file

    data = direct_server_with_data["data_dir"]
    _chat_id, revision = _buried_room(data, monkeypatch, "kept-room", 480)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            room = _Room(browser, direct_server_with_data["url"], "kept-room", hold=2)
            room.open()
            assert _wait_for(room.page, lambda: len(room.held) == 1, attempts=200), room.older
            _stage_file(room.page)
            room.page.locator("#project-panel-close").evaluate("el => el.click()")
            room.page.locator('.chat-instance-panel[data-pending-work="1"]').wait_for(state='attached')
            assert room.page.locator('.chat-instance-panel[data-pending-work="1"]').count() == 1
            room.release()
            room.settled()
            hidden = list(room.older)
            assert len(hidden) == 2 and room.seen() == [], ("a hidden room asks for nothing more", hidden, room.acks)
            assert _unread(room.row).count() == 1
            room.open()
            read = _wait_for(room.page, lambda: room.seen() == [revision], attempts=200)
            room.settled(300)
            landed = room.shown()
            name = f"unread-opening-{engine}-hidden-reopened"
            evidence = _evidence(data, name, landed=landed, older=room.older, hidden=hidden, acks=room.acks)
            room.page.screenshot(path=str(evidence / f"{name}.png"))
            assert read and _on_screen(landed) and len(room.older) > len(hidden), (landed, room.older, room.acks)
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_reopened_room_lands_once_after_its_superseded_reach(direct_server_with_data, monkeypatch, engine):
    """Closed and opened again while its reach awaits a page: the old request spends nothing, and the
    new opening reads on to the reply, lands there once and reads it."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    _chat_id, revision = _buried_room(data, monkeypatch, "reopened-room", 480)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            room = _Room(browser, direct_server_with_data["url"], "reopened-room", hold=2)
            room.open()
            assert _wait_for(room.page, lambda: len(room.held) == 1, attempts=200), room.older
            room.page.locator("#project-panel-close").evaluate("el => el.click()")
            # After the exit, the replacement reads independently of the destroyed room's held page.
            room.messages.wait_for(state='detached')
            room.open()
            read = _wait_for(room.page, lambda: room.seen() == [revision], attempts=200)
            room.release()
            room.settled()
            landed = room.shown()
            name = f"unread-opening-{engine}-reopened"
            evidence = _evidence(data, name, landed=landed, older=room.older, acks=room.acks)
            room.page.screenshot(path=str(evidence / f"{name}.png"))
            assert read and _on_screen(landed) and room.seen() == [revision], (landed, room.acks)
        finally:
            browser.close()


# Each read receipt as it is posted, with the reply's box at that moment.
def _read_receipts(text):
    return f"""() => {{
    const shown = {_SHOWN}, post = window.fetch;
    window.__receipts = [];
    window.fetch = function (url, init) {{
        if (init?.method === 'POST' && String(url).endsWith('/api/ui/preferences')) {{
            const messages = document.querySelector({json.dumps(PANEL)});
            window.__receipts.push({{seen: JSON.parse(init.body).project_seen_revision,
                                     box: messages && shown(messages, {json.dumps(text)})}});
        }}
        return post.apply(this, arguments);
    }};
}}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_reply_arriving_while_the_opening_proves_a_quiet_start_is_reached_after_its_return(
        direct_server_with_data, monkeypatch, engine):
    """The read-band quiet room past the pages kept: only the owner's own notes, more than the recent
    read's bounded search covers, the start eight older pages down. While its opening walks there, a reply
    arrives and 160 child words bury it below the next recent read. The clean start proven, the opening returns to the newest page once, reads on to the reply that
    fresh read names, lands there and reads it; a state refresh that reads the arrivals first brings the reply
    itself, and the opening lands on it with no return. Its receipt names the revision the client's state then
    holds: the opened one, or the reply's if a state refresh came first; the next refresh reads the reply's
    where the reader already is. Every receipt is posted with the reply shown, and the cursor ends at its revision."""
    from playwright.sync_api import sync_playwright

    from tests.test_project_unread_read_band_browser import _seed_room

    data = direct_server_with_data["data_dir"]
    note = lambda index: {"ts": f"2026-09-28T10:{(10 + index) // 60:02d}:{(10 + index) % 60:02d}Z",  # noqa: E731
                          "direction": "in", "text": f"Owner note {index}: " + "a long thought " * 270}
    early = [{"ts": f"2026-09-28T09:{i // 60:02d}:{i % 60:02d}Z", "direction": "in", "text": f"Early note {i}"}
             for i in range(1000)]
    assert len(early) + 330 > _RECENT_WINDOW_ROWS + _ARRIVAL_SEARCH_ROWS, "the notes outrun the recent read's search"
    chat_id, revision = _seed_room(data, "late-room", "Late room", [note(i) for i in range(170, 330)], archives=[
        ("chat_20260928T095900.jsonl", early), ("chat_20260928T101000.jsonl", [note(i) for i in range(170)])])
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            url = direct_server_with_data["url"]
            room = _Room(browser, url, "late-room", hold=3, init=(_read_receipts(REPLY),))
            # How far the walk had asked when a refresh's read of what arrived brought the reply (#1954).
            brought = []
            room.page.on("response", lambda response: brought.append(len(room.older))
                         if _cursor_kind(response.url) == "after" and _OLDER in response.url
                         and REPLY in _texts(response) else None)
            room.open()
            assert _wait_for(room.page, lambda: len(room.held) == 1, attempts=200), room.older
            _chat_id, newer = _buried_room(data, monkeypatch, "late-room", 160, chat_id=chat_id)
            room.release()
            read = _wait_for(room.page, lambda: room.seen() and _on_screen(room.shown()), attempts=200)
            room.settled(300)
            landed = room.shown()
            _emit_ws_frame(room.page, {"type": "projects_changed"})
            newest = _wait_for(room.page, lambda: room.seen()[-1:] == [newer], attempts=100)
            receipts = [(posted["seen"]["late-room"], posted["box"]) for posted in
                        room.page.evaluate("() => window.__receipts") if "late-room" in (posted.get("seen") or {})]
            cursor = room.page.request.get(f"{url}/api/ui/preferences").json()["project_seen_revision"]
            name = f"unread-opening-{engine}-reply-after-return"
            evidence = _evidence(data, name, landed=landed, older=room.older, reads=room.reads, acks=room.acks,
                                 receipts=receipts, cursor=cursor, revisions=[revision, newer])
            room.page.screenshot(path=str(evidence / f"{name}.png"))
            assert read and _on_screen(landed), ("the opening reaches the reply the return named", landed, room.acks)
            # The walk reaches the start, eight pages down, then reads the page holding the reply; unless a state
            # refresh's read of what arrived brings the reply first: the opening lands on it with no return and asks
            # for no page past the one it had asked for.
            assert len(room.older) == 9 or (brought and len(room.older) <= brought[0] + 1), \
                ("eight pages to the start, then the page holding the reply, or a refresh brought it first",
                 room.older, brought)
            assert receipts and all(_in_band(box) for _seen, box in receipts), ("read only while shown", receipts)
            assert newest and room.seen() == sorted(room.seen()) and set(room.seen()) <= {revision, newer}, \
                ("the newer revision is read where the reader already is", room.acks)
            assert cursor.get("late-room") == newer, ("the shared cursor covers the reply's revision", cursor)
            _unread(room.row).wait_for(state="detached")
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_failed_page_ends_the_reach_and_its_repair_moves_no_one(direct_server_with_data, monkeypatch, engine):
    """The second older page fails: the opening stays at the bottom with the dot, the Retry control is
    the reader's; the next state refresh reads nothing older and moves no one."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    _buried_room(data, monkeypatch, "failed-room", 480)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            room = _Room(browser, direct_server_with_data["url"], "failed-room", fail=2)
            room.open()
            assert _wait_for(room.page, lambda: len(room.older) == 2, attempts=200), room.older
            room.settled()
            top = room.messages.evaluate("n => n.scrollTop")
            _emit_ws_frame(room.page, {"type": "projects_changed"})
            room.settled()
            name = f"unread-opening-{engine}-failed"
            evidence = _evidence(data, name, older=room.older, acks=room.acks, top=top)
            room.page.screenshot(path=str(evidence / f"{name}.png"))
            assert len(room.older) == 2, ("no automatic retry and no later reach", room.older)
            assert room.messages.evaluate("n => n.scrollTop") == top, "the repair moves no one"
            assert room.seen() == [] and _unread(room.row).count() == 1, room.acks
            assert room.page.locator(f"{PANEL} .chat-load-older button").count() == 1, "Retry stays the reader's"
        finally:
            browser.close()

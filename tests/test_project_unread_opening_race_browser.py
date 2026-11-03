"""A still-owned unread opening follows fresh arrival evidence across its frozen pager chain.

The retained counterexample holds the fifth older read after a kept-page hole forms, then a
real producer adds a reply and 160 child words. A reconnect reads the newest page whole (frames
sent while the window was away are lost): that fresh recent read names the reply without
containing it. The opening must rebase through the existing pager before continuing its reach;
a reader cancelling either before or during that return must never inherit the new chain. A state
refresh instead reads only what arrived since the newest read (``after_cursor``), which brings the
reply itself: the opening lands on it with no return.
"""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import tests.test_project_unread_opening_browser as opening
from tests.test_project_unread_read_band_browser import _seed_room, _unread, _wait_for
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _SETTLE_RESTORE_FRAMES, _emit_ws_frame

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
LATEST = "NEW ARRIVAL: opening must reach this reply"
# The socket drops; the client reconnects on its own and reads the newest page whole.
_RECONNECT = """() => {
    const socket = window.__testSockets.find(s => s.readyState === WebSocket.OPEN);
    if (!socket) throw new Error('No open socket to reconnect');
    socket.close();
}"""


def _backlog(data, project_id):
    base = datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc)
    rows = [{"ts": base.isoformat(), "direction": "out", "text": opening.REPLY}]
    rows += [{"ts": (base + timedelta(seconds=index + 1)).isoformat(), "direction": "in",
              "text": f"Owner backlog note {index}: preserve this row"} for index in range(1100)]
    return _seed_room(data, project_id, "New arrival race", rows)


def _question(data, project_id, chat_id):
    from ouroboros.projects_registry import bind_task_to_project, get_project
    from ouroboros.task_results import write_task_result

    project = get_project(data, project_id)
    quiz = {"quiz_id": "q-rebase", "question": "Merge the opening fix now?", "options": ["Yes", "No"],
            "state": "open", "asked_at": "2026-09-28T10:05:00Z"}
    bind_task_to_project(data, "rebase-task", project_id, chat_id, origin={"absent": "system"})
    write_task_result(data, "rebase-task", "running", project_id=project_id, chat_id=chat_id,
                      owner_quiz={"q-rebase": quiz})
    return project


class _Race:
    """Observe real request chains and hold their return; all room mechanics use the shared helper."""

    def __init__(self, browser, server, monkeypatch, project_id, question=False):
        self.data, self.monkeypatch, self.project_id = server["data_dir"], monkeypatch, project_id
        self.candidate = server["candidate_identity"]
        self.chat_id, self.initial = _backlog(self.data, project_id)
        self.question = _question(self.data, project_id, self.chat_id) if question else None
        self.room = opening._Room(browser, server["url"], project_id, hold=5,
                                  init=(opening._read_receipts(LATEST),))
        self.recent, self.arrivals, self.history, self.returning = [], [], [], []
        self.hold_return = False

        def recent_route(route):
            if self.hold_return:
                self.hold_return = False
                self.returning.append(route)
            else:
                route.continue_()

        def record(response):
            query = parse_qs(urlparse(response.url).query)
            if "/api/chat/history?" not in response.url or query.get("chat_id") != [str(self.chat_id)]:
                return
            body, kind = response.json(), opening._cursor_kind(response.url)
            self.history.append({"url": response.url, "kind": kind, "window": body.get("window"),
                                 "coverage": body.get("coverage"),
                                 "ids": [row.get("history_id") for row in body.get("messages", [])]})
            if kind is None:
                self.recent.append(body)
            elif kind == "after":
                self.arrivals.append(body)

        self.room.page.route(lambda url: "/api/chat/history?" in url and "cursor=" not in url, recent_route)
        self.room.page.on("response", record)

    def admit(self, kind="named"):
        """``named``: 160 child words; the reconnect's fresh read names the reply. ``searched``: 1310, more
        than the recent read's window and its search past it cover: its bounded search runs out first.
        ``arrived``: 160 child words read by a state refresh, whose read of the arrivals holds the reply."""
        room = self.room
        room.open()
        assert _wait_for(room.page, lambda: len(room.held) == 1, attempts=250), room.older
        self.original_window = self.recent[-1]["window"]
        # Page zero remains protected by the viewport after page one has been evicted: canNewer.
        assert room.messages.locator(".chat-bubble").filter(has_text="Owner backlog note 900:").count() == 0
        assert room.messages.locator(".chat-bubble").filter(has_text="Owner backlog note 1099:").count() == 1
        self.monkeypatch.setattr(opening, "REPLY", LATEST)
        _chat, self.revision = opening._buried_room(
            self.data, self.monkeypatch, self.project_id, 1310 if kind == "searched" else 160, chat_id=self.chat_id)
        before, arrived = len(self.recent), len(self.arrivals)
        if kind == "arrived":
            # The refresh's read of the arrivals; one that found nothing yet names no message.
            named = lambda: next((body for body in self.arrivals[arrived:]  # noqa: E731
                                  if "latest_message" in (body.get("window") or {})), None)
            _emit_ws_frame(room.page, {"type": "projects_changed"})
            assert _wait_for(room.page, lambda: named() is not None, attempts=150), self.history
            self.fresh = named()
        else:
            sockets = room.page.evaluate("() => window.__testSockets.length")
            room.page.evaluate(_RECONNECT)
            # The socket reopens on its own, and only then the room reads its newest page whole: on a loaded host
            # both take their time, the server's search of the long room above all, so each is waited for itself.
            room.page.wait_for_function("n => window.__testSockets.length > n && window.__testSockets.at(-1).readyState === 1",
                                        arg=sockets, timeout=60_000)
            assert _wait_for(room.page, lambda: len(self.recent) > before, attempts=600), self.history
            self.fresh = self.recent[-1]
        latest = self.fresh["window"]["latest_message"]
        if kind == "searched":
            assert latest is None and isinstance(self.fresh["window"].get("latest_before"), int), self.fresh["window"]
        else:
            assert latest["history_id"] != self.original_window["latest_message"]["history_id"]
        texts = [row.get("text") for row in self.fresh.get("messages", [])]
        assert (LATEST in texts) == (kind == "arrived"), ("only a read of the arrivals holds the reply", kind)
        room.page.evaluate(_SETTLE_RESTORE_FRAMES)
        assert room.seen() == [] and _unread(room.row).count() == 1, room.acks

    def evidence(self, engine, suffix, **facts):
        name = f"unread-race-{engine}-{suffix}"
        root = Path(__file__).resolve().parents[1]
        sources = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in (
            "web/modules/project_read_state.js", "web/modules/chat.js",
            "tests/test_project_unread_opening_race_browser.py", "tests/test_project_unread_opening_browser.py")}
        evidence = opening._evidence(self.data, name, initial=self.initial, revision=self.revision,
                                     initial_window=self.original_window, fresh_window=self.fresh["window"],
                                     fresh_rows=len(self.fresh.get("messages", [])), older=self.room.older,
                                     history=self.history, acks=self.room.acks, landed=self.room.shown(),
                                     sources=sources, candidate=self.candidate, receipts=self.receipts(), **facts)
        self.room.page.screenshot(path=str(evidence / f"{name}.png"))

    def chains(self):
        """The frozen chain each older page continued: its cursor's ``upper``. A rebase starts a new one."""
        def upper(url):
            cursor = parse_qs(urlparse(url).query)["cursor"][0]
            return json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))["upper"]["chat"]
        return [upper(url) for url in self.room.older]

    def receipts(self):
        return [(posted["seen"][self.project_id], posted["box"]) for posted in
                self.room.page.evaluate("() => window.__receipts") if self.project_id in (posted.get("seen") or {})]


@pytest.mark.parametrize("kind", ["named", "searched", "arrived"])
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_new_offpage_reply_replaces_the_openings_frozen_target(direct_server_with_data, monkeypatch, engine, kind):
    """Latest admission wins while this opening still owns navigation, whether the fresh read names the
    reply, only searched past the frozen chain's bytes, or brought the reply with the arrivals; then
    later reads never jump."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            race = _Race(browser, direct_server_with_data, monkeypatch, "arrival-race")
            race.admit(kind)
            room = race.room
            before_return = len(race.recent)
            room.release()
            read = _wait_for(room.page, lambda: race.revision in room.seen(), attempts=200)
            room.settled(300)
            landed = room.shown()
            race.evidence(engine, f"new-arrival-{kind}", read=read, before_return=before_return,
                          recent_reads=len(race.recent))
            assert read and opening._on_screen(landed), ("latest admitted reply is reached and read", landed, room.acks)
            receipts = race.receipts()
            assert receipts and all(opening._in_band(box) for _seen, box in receipts), ("read only while shown", receipts)
            assert room.seen() and all(value == race.revision for value in room.seen()), room.acks
            chains = race.chains()
            frozen, fresh = chains[0], race.fresh["coverage"]["upper"]["chat"]
            if kind == "arrived":
                assert chains == [frozen] * 5 and len(race.recent) == before_return, \
                    ("the reply is in the window: no return, and no page past the one held", chains, race.history)
            else:
                assert chains == [frozen] * chains.index(fresh) + [fresh] * (len(chains) - chains.index(fresh)), \
                    ("one return to the fresh snapshot, no old-chain page after it", chains)
            if kind == "named":
                assert len(race.recent) == before_return + 1, ("one pager return to the fresh snapshot", race.history)
            _unread(room.row).wait_for(state="detached")
            pages, recent = len(room.older), len(race.recent)
            # A completed opening owns no later refresh: no repeated return, old-chain walk or jump.
            opening._wheel(room, False)
            room.settled(300)
            top = room.messages.evaluate("n => n.scrollTop")
            _emit_ws_frame(room.page, {"type": "projects_changed"})
            room.settled(800)
            after = room.messages.evaluate("n => n.scrollTop")
            race.evidence(engine, f"later-read-{kind}", pages=pages, before=top, after=after,
                          before_return=before_return, recent_reads=len(race.recent))
            assert len(room.older) == pages and len(race.recent) == recent, race.history
            assert abs(after - top) <= 1, ("a later read preserves the reader's position", top, after)
        finally:
            browser.close()


@pytest.mark.parametrize(("stage", "move", "kind"), [
    ("fresh-admission", "wheel", "named"), ("pager-return", "wheel", "named"), ("pager-return", "question", "named"),
    ("fresh-admission", "wheel", "searched"), ("pager-return", "wheel", "searched"),
    ("fresh-admission", "wheel", "arrived")])
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_cancelling_a_rebased_opening_never_resumes_it(direct_server_with_data, monkeypatch, engine, stage, move, kind):
    """A wheel or explicit question ends ownership before an old page or the fresh return resolves."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            race = _Race(browser, direct_server_with_data, monkeypatch, "cancel-race", question=move == "question")
            race.admit(kind)
            room = race.room
            before_return = len(race.recent)
            if stage == "pager-return":
                race.hold_return = True
                room.release()
                assert _wait_for(room.page, lambda: len(race.returning) == 1, attempts=200), race.history
            if move == "question":
                room.page.evaluate("""project => window.dispatchEvent(new CustomEvent('ouro:open-project', {
                    detail: {project, task_id: 'rebase-task', quiz_id: 'q-rebase'}}))""", race.question)
                card = room.messages.locator('.chat-quiz-card[data-quiz-id="q-rebase"]')
                card.wait_for(state="visible")
            else:
                opening._wheel(room, False)
            room.page.evaluate(_SETTLE_RESTORE_FRAMES)
            pages = len(room.older)
            room.release()
            for route in race.returning:
                route.continue_()
            room.settled()
            top = room.messages.evaluate("n => n.scrollTop")
            latest = room.shown()
            assert not opening._on_screen(latest), latest
            assert room.seen() == [] and len(room.older) == pages, (room.acks, room.older)
            assert set(race.chains()) == {race.chains()[0]}, ("no page of a new chain after cancellation", race.chains())
            # A searched read names no newest message, so state refreshes re-read the unread room
            # (DESIGN "Unread"); a named or arrived one is covered and re-reads nothing.
            if kind != "searched":
                assert len(race.recent) == before_return + (stage == "pager-return"), race.history
            _emit_ws_frame(room.page, {"type": "projects_changed"})
            room.settled(800)
            after = room.messages.evaluate("n => n.scrollTop")
            asked = card.evaluate(opening._ON_FEED) if move == "question" else None
            race.evidence(engine, f"cancel-{stage}-{move}-{kind}", pages=pages, before=top, after=after,
                          before_return=before_return, recent_reads=len(race.recent), asked=asked)
            assert len(room.older) == pages and not opening._on_screen(room.shown()), race.history
            assert room.seen() == [] and _unread(room.row).count() == 1, room.acks
            assert abs(after - top) <= 1, ("a later read cannot revive a cancelled opening", top, after)
            if asked:
                assert asked["focused"] and asked["feedTop"] <= asked["top"] < asked["feedBottom"], asked
        finally:
            browser.close()

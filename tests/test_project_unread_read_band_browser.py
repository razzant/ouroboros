"""A Project room is read where the reader can see the newest message (DESIGN "Unread").

Real server, real retained chat history, real history endpoint and read cursor:
a late answer keeps the time it was written, so it sorts into the middle of the
room while it is the message that arrived last. Opening the unread room lands on
it (owner decision 2026-10-11); one arriving while the room is open moves no one,
and beneath the composer drawn over the feed's bottom it is not read until it
clears it. A newest reply above the owner's own later messages and a task card is
landed on the same way: they are not conversation messages, and the bottom past
them is not reading it. An ordinary room is read on landing; a
question delivered live is read once the next read names it and its card is on
screen; a late answer written before the whole recent window is landed on through
the older pages the opening reads itself. Clients share one read cursor: a room read on one clears
the dot on another at its next state refresh.
"""
from __future__ import annotations
from tests._dialogue import advance_room_revision

import json
import os
from pathlib import Path

import pytest

from ouroboros import dialogue_store
from ouroboros.gateway.history_paging import _ARRIVAL_SEARCH_ROWS, _RECENT_WINDOW_ROWS
from tests import _dialogue as dialogue_test
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _SETTLE_RESTORE_FRAMES, _emit_ws_frame
from tests._store_shared import seed_result

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
PANEL = "#project-panel .chat-messages"
LATE = "Late final answer"
LATER = "Later late answer"
NEWEST = "Newest reply in its place"
QUESTION = "Merge the release branch now?"
DRAFT = "Yes, after the tag"
_FOCUSED_DRAFT = """() => document.activeElement?.classList.contains('chat-quiz-comment')
    ? document.activeElement.value : null"""
# The browser's page visibility as a background/foreground switch reports it.
_SET_HIDDEN = """hidden => {
    Object.defineProperty(document, 'hidden', {configurable: true, get: () => hidden});
    Object.defineProperty(document, 'visibilityState', {configurable: true, get: () => hidden ? 'hidden' : 'visible'});
    document.dispatchEvent(new Event('visibilitychange'));
}"""
# Place the message's top `offset` px below the composer's top edge, and report
# the geometry the reader actually has.
_PLACE = """(messages, [text, offset]) => {
    const node = [...messages.querySelectorAll('[data-history-id]')].find(n => n.textContent.includes(text));
    const composer = messages.parentElement.querySelector('.chat-input-area');
    messages.scrollTop += node.getBoundingClientRect().top - (composer.getBoundingClientRect().top + offset);
    messages.dispatchEvent(new Event('scroll'));
    const box = (el) => { const r = el.getBoundingClientRect(); return {top: r.top, bottom: r.bottom}; };
    return {node: box(node), feed: box(messages), composer: box(composer), id: node.dataset.historyId};
}"""
_AT_BOTTOM = """(messages) => {
    messages.scrollTop = messages.scrollHeight;
    messages.dispatchEvent(new Event('scroll'));
    return messages.scrollHeight - messages.scrollTop - messages.clientHeight;
}"""
_BOX_OF = """(messages, text) => {
    const node = [...messages.querySelectorAll('[data-history-id]')].find(n => n.textContent.includes(text));
    const r = node.getBoundingClientRect(), feed = messages.getBoundingClientRect();
    return {top: r.top, bottom: r.bottom, feedTop: feed.top};
}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_an_opened_room_lands_on_its_late_answer_and_a_later_one_is_read_only_clear_of_the_composer(
        direct_server_with_data, engine):
    """Opening the unread room lands on the answer that arrived last and reads it (owner decision
    2026-10-11). A second late answer arriving while the room is open moves no one: beneath the
    composer it is not read, clear of it it is."""
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project

    data = direct_server_with_data["data_dir"]
    project = create_project(data, "late-room", name="Late room")
    chat_id = project["chat_id"]
    rows = [{"ts": f"2026-09-28T10:{minute:02d}:00Z", "direction": "out", "chat_id": chat_id,
             "text": f"Reply {minute}\nline two\nline three"} for minute in range(30)]
    # Arrived last, written at 10:05:30: it sorts between Reply 5 and Reply 6.
    rows.append({"ts": "2026-09-28T10:05:30Z", "direction": "out", "chat_id": chat_id,
                 "text": f"{LATE}\nwith its details"})
    for row in rows:
        dialogue_store.append(data, "chat", row)
    revision = advance_room_revision(data, chat_id=chat_id)["visible_revision"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    acks = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1187, "height": 734})
            page.on("request", lambda request: acks.append(json.loads(request.post_data or "{}"))
                    if request.method == "POST" and request.url.endswith("/api/ui/preferences") else None)
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
            row = page.locator('.nav-project-row[data-project-id="late-room"]')
            _unread(row).wait_for(state="attached")
            row.evaluate("el => el.click()")
            messages = page.locator(PANEL)
            messages.locator(".chat-bubble").filter(has_text=LATE).wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            seen = lambda: [ack["project_seen_revision"]["late-room"] for ack in acks  # noqa: E731
                            if "late-room" in (ack.get("project_seen_revision") or {})]
            landed = messages.evaluate(_BOX_OF, LATE)
            page.screenshot(path=str(evidence / f"read-band-{engine}-late-landing.png"))
            assert _wait_for(page, lambda: seen() == [revision]), ("the opening lands on the late answer", landed, acks)
            assert landed["top"] >= landed["feedTop"], ("it is on screen", landed)
            _unread(row).wait_for(state="detached")

            # Arrived last again, written at 10:15:30: it sorts between Reply 15 and Reply 16.
            dialogue_store.append(data, "chat", {"ts": "2026-09-28T10:15:30Z", "direction": "out", "chat_id": chat_id,
                                                 "text": f"{LATER}\nwith its details"})
            second = advance_room_revision(data, chat_id=chat_id)["visible_revision"]
            _emit_ws_frame(page, {"type": "projects_changed"})
            messages.locator(".chat-bubble").filter(has_text=LATER).wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.wait_for_timeout(300)
            kept = messages.evaluate(_BOX_OF, LATE)
            assert abs(kept["top"] - landed["top"]) <= 2, ("a later arrival moves no one", landed, kept)
            assert seen() == [revision], "the later answer is not on screen yet"

            under = messages.evaluate(_PLACE, [LATER, 12])
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.screenshot(path=str(evidence / f"read-band-{engine}-under-composer.png"))
            assert under["node"]["top"] < under["feed"]["bottom"], under  # it intersects the feed's viewport
            assert under["node"]["top"] >= under["composer"]["top"], under  # ...but only beneath the composer
            page.wait_for_timeout(300)
            assert seen() == [revision], ("an answer beneath the composer is not read", under)
            assert _unread(row).count() == 1

            clear = messages.evaluate(_PLACE, [LATER, -160])
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            assert clear["node"]["bottom"] <= clear["composer"]["top"], clear
            read = _wait_for(page, lambda: seen() == [revision, second])
            page.screenshot(path=str(evidence / f"read-band-{engine}-clear.png"))
            (evidence / f"read-band-{engine}.json").write_text(
                json.dumps({"landed": landed, "kept": kept, "under": under, "clear": clear, "acks": acks}, indent=2),
                encoding="utf-8")
            assert read, ("the answer clear of the composer is read", clear, acks)
            _unread(row).wait_for(state="detached")
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_an_opened_room_lands_on_its_newest_reply_above_later_rows_and_a_read_room_reopens_at_the_bottom(
        direct_server_with_data, engine):
    """The owner's later notes and a card below the newest reply push it above the bottom: the unread
    room opens on it and reads it (owner decision 2026-10-11); reopened once read, it opens at its bottom."""
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project

    data = direct_server_with_data["data_dir"]
    project = create_project(data, "tail-room", name="Tail room")
    chat_id = project["chat_id"]
    at = lambda minute: f"2026-09-28T10:{minute:02d}:00Z"  # noqa: E731
    rows = [{"ts": at(minute), "direction": "out", "chat_id": chat_id,
             "text": f"Reply {minute}\nline two\nline three"} for minute in range(20)]
    # The newest reply arrived last and sorts last among the conversation messages.
    rows.append({"ts": at(30), "direction": "out", "chat_id": chat_id, "text": f"{NEWEST}\nwith its details"})
    # After it: the owner's own follow-ups and a row the host placed in a task card.
    rows += [{"ts": at(31 + index), "direction": "in", "chat_id": chat_id,
              "text": "\n".join(f"Owner follow-up {index}, line {line}" for line in range(8))} for index in range(6)]
    rows.append({"ts": at(41), "direction": "system", "chat_id": chat_id, "task_id": "tail-task",
                 "type": "custody_notice", "card_row": "timeline", "card_row_id": "final:tail-task:custody",
                 "text": "Custody settled"})
    for row in rows:
        dialogue_store.append(data, "chat", row)
    dialogue_store.append(data, "progress",
                          {"ts": at(40), "chat_id": chat_id, "task_id": "tail-task", "content": "Working on the follow-up"})
    revision = advance_room_revision(data, chat_id=chat_id)["visible_revision"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    acks = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1187, "height": 734})
            page.on("request", lambda request: acks.append(json.loads(request.post_data or "{}"))
                    if request.method == "POST" and request.url.endswith("/api/ui/preferences") else None)
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
            row = page.locator('.nav-project-row[data-project-id="tail-room"]')
            _unread(row).wait_for(state="attached")
            row.evaluate("el => el.click()")
            messages = page.locator(PANEL)
            messages.locator(".chat-bubble").filter(has_text="Owner follow-up 5").wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            seen = lambda: [ack for ack in acks if "tail-room" in (ack.get("project_seen_revision") or {})]  # noqa: E731
            landed = messages.evaluate(_BOX_OF, NEWEST)
            gap = messages.evaluate("n => n.scrollHeight - n.scrollTop - n.clientHeight")
            page.screenshot(path=str(evidence / f"read-band-{engine}-tail-landing.png"))
            assert _wait_for(page, lambda: seen() == [{"project_seen_revision": {"tail-room": revision}}]), \
                ("the opening lands on the newest reply and reads it", landed, gap, acks)
            assert landed["top"] >= landed["feedTop"] and gap > 1, ("on screen, above the bottom", landed, gap)
            _unread(row).wait_for(state="detached")

            page.locator("#project-panel-close").evaluate("el => el.click()")
            row.evaluate("el => el.click()")
            messages.locator(".chat-bubble").filter(has_text="Owner follow-up 5").wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.wait_for_timeout(300)
            reopened = messages.evaluate("n => n.scrollHeight - n.scrollTop - n.clientHeight")
            page.screenshot(path=str(evidence / f"read-band-{engine}-tail-reopened.png"))
            assert reopened <= 1, ("a read room reopens at its bottom, as before", reopened)
            assert len(seen()) == 1, "and reads nothing again"
        finally:
            browser.close()


def _seed_room(data, project_id, name, rows, archives=()):
    """A Project room holding ``rows`` as the server stores them, after ``archives``: an older
    install's chain segments ``(name, rows)`` (a raw string is a line as written), imported first."""
    from ouroboros.projects_registry import create_project

    chat_id = create_project(data, project_id, name=name)["chat_id"]
    if archives:
        dialogue_test.seed(data, archives={segment: [line if isinstance(line, str) else {**line, "chat_id": chat_id}
                                                     for line in lines] for segment, lines in archives})
        dialogue_test.import_all(data)
    for row in rows:
        dialogue_store.append(data, "chat", {**row, "chat_id": chat_id})
    return chat_id, advance_room_revision(data, chat_id=chat_id)["visible_revision"]


def _unread(row):
    """A Project row's unread fact: its own attribute, never its mark's colour, which the owner's amber outranks."""
    return row.and_(row.page.locator("[data-unread]"))


def _client(browser, url, project_id, acks, init=()):
    """One client showing the room's row unread; ``acks`` collects its read receipts."""
    page = browser.new_page(viewport={"width": 1187, "height": 734})
    page.on("request", lambda request: acks.append(json.loads(request.post_data or "{}"))
            if request.method == "POST" and request.url.endswith("/api/ui/preferences") else None)
    for script in (_CAPTURE_TEST_SOCKET, *init):
        page.add_init_script(f"({script})()")
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
    row = page.locator(f'.nav-project-row[data-project-id="{project_id}"]')
    _unread(row).wait_for(state="attached")
    return page, row


def _open_room(pw, engine, url, project_id, acks, init=()):
    browser = getattr(pw, engine).launch()
    page, row = _client(browser, url, project_id, acks, init)
    row.evaluate("el => el.click()")
    return browser, page, row


def _wait_for(page, check, attempts=100):
    for _ in range(attempts):
        if check():
            return True
        page.wait_for_timeout(100)
    return check()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_question_delivered_live_is_read_where_its_card_is_shown(direct_server_with_data, engine):
    """The live card is the node the next read names: it gains that row, so the reader at it has read."""
    from playwright.sync_api import sync_playwright


    data = direct_server_with_data["data_dir"]
    chat_id, first = _seed_room(data, "ask-room", "Ask room", [
        {"ts": f"2026-09-28T10:{minute:02d}:00Z", "direction": "out", "text": f"Reply {minute}"} for minute in range(4)])
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    acks = []
    seen = lambda: [ack["project_seen_revision"]["ask-room"] for ack in acks  # noqa: E731
                    if "ask-room" in (ack.get("project_seen_revision") or {})]
    with sync_playwright() as pw:
        browser, page, row = _open_room(pw, engine, direct_server_with_data["url"], "ask-room", acks)
        try:
            messages = page.locator(PANEL)
            messages.locator(".chat-bubble").filter(has_text="Reply 3").wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            assert _wait_for(page, lambda: seen() == [first]), ("an ordinary room is read on landing", acks)
            _unread(row).wait_for(state="detached")

            # The bridge broadcasts the question, then persists it and advances the revision.
            ts = "2026-09-28T10:30:00Z"
            quiz = {"quiz_id": "q-live", "wait_for_answer": False, "options": [{"label": "Yes"}, {"label": "No"}],
                    "stake": "", "assumption": "", "state": "open"}
            _emit_ws_frame(page, {"type": "quiz", "role": "assistant", "question": QUESTION, "ts": ts,
                                  "chat_id": chat_id, "task_id": "ask-task", **quiz})
            card = messages.locator(".chat-quiz-card").filter(has_text=QUESTION)
            card.wait_for(state="visible")
            assert card.evaluate("n => n.closest('[data-history-id]')") is None, "live, the card names no row yet"
            # The owner starts answering before any read names the question.
            card.evaluate("n => { n.__liveCard = true; }")
            card.locator(".chat-quiz-comment").fill(DRAFT)
            dialogue_store.append(data, "chat", {"ts": ts, "direction": "out", "chat_id": chat_id, "text": QUESTION,
                                                 "type": "quiz", "task_id": "ask-task", "quiz": quiz})
            second = advance_room_revision(data, chat_id=chat_id)["visible_revision"]
            _emit_ws_frame(page, {"type": "projects_changed"})
            read = _wait_for(page, lambda: seen() == [first, second])
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.screenshot(path=str(evidence / f"read-band-{engine}-live-question.png"))
            stamped = card.evaluate("n => n.closest('[data-history-id]')?.dataset.historyId")
            geometry = card.evaluate("""n => {
                const box = (el) => { const r = el.getBoundingClientRect(); return {top: r.top, bottom: r.bottom}; };
                const feed = n.closest('.chat-messages');
                return {card: box(n), feed: box(feed), composer: box(feed.parentElement.querySelector('.chat-input-area'))};
            }""")
            (evidence / f"read-band-{engine}-live-question.json").write_text(
                json.dumps({"acks": acks, "stamped": stamped, "box": geometry}, indent=2), encoding="utf-8")
            assert geometry["card"]["top"] < geometry["composer"]["top"], ("the card is on screen", geometry)
            assert read, ("the question on screen is read", acks, stamped, geometry)
            assert str(stamped).startswith("chat:"), stamped
            assert messages.locator(".chat-quiz-card").filter(has_text=QUESTION).count() == 1
            assert card.evaluate("n => n.__liveCard === true"), "the read adopted the live card, not a copy"
            assert page.evaluate(_FOCUSED_DRAFT) == DRAFT, "the draft and its focus survive the adoption"
            _unread(row).wait_for(state="detached")
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_an_opening_reads_the_older_page_holding_a_late_answer_below_the_recent_window(direct_server_with_data, engine):
    """Written before every reply of the 150-row recent window but arriving last, the answer is
    delivered only by the first older page: the opening reads it, lands there and reads it, no click."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    rows = [{"ts": f"2026-09-28T{10 + minute // 60:02d}:{minute % 60:02d}:00Z", "direction": "out",
             "text": f"Reply {minute}"} for minute in range(160)]
    rows.append({"ts": "2026-09-28T09:00:00Z", "direction": "out", "text": f"{LATE}\nwith its details"})
    _chat_id, revision = _seed_room(data, "deep-room", "Deep room", rows)
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    acks = []
    seen = lambda: [ack["project_seen_revision"]["deep-room"] for ack in acks  # noqa: E731
                    if "deep-room" in (ack.get("project_seen_revision") or {})]
    with sync_playwright() as pw:
        browser, page, row = _open_room(pw, engine, direct_server_with_data["url"], "deep-room", acks)
        try:
            messages = page.locator(PANEL)
            messages.locator(".chat-bubble").filter(has_text=LATE).wait_for(state="attached")
            read = _wait_for(page, lambda: seen() == [revision])
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            landed = messages.evaluate(_BOX_OF, LATE)
            composer = page.locator("#project-panel .chat-input-area").evaluate("n => n.getBoundingClientRect().top")
            page.screenshot(path=str(evidence / f"read-band-{engine}-deep-landing.png"))
            (evidence / f"read-band-{engine}-deep.json").write_text(
                json.dumps({"landed": landed, "composer": composer, "acks": acks}, indent=2), encoding="utf-8")
            assert read, ("the opening reaches the answer below the recent window and reads it", landed, acks)
            assert landed["feedTop"] <= landed["top"] < composer, ("on screen, no Load more history", landed)
            _unread(row).wait_for(state="detached")
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_room_read_on_one_client_clears_the_dot_on_another_at_its_next_state_refresh(direct_server_with_data, engine):
    """Two clients share one forward-only read cursor: the reader's receipt clears the other
    client's dot at its next state refresh, and the other client posts no receipt of its own."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    _chat_id, revision = _seed_room(data, "shared-room", "Shared room", [
        {"ts": f"2026-09-28T10:{minute:02d}:00Z", "direction": "out", "text": f"Reply {minute}"} for minute in range(4)])
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    reader_acks, other_acks = [], []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            # Control cursor reads from navigation onward, then settle startup
            # reads before the ACK so only a later refresh can clear the dot.
            hold_preferences = """() => {
                const fetch = window.fetch.bind(window);
                window.__holdPreferences = true;
                window.__pendingPreferences = [];
                window.__settledPreferences = 0;
                window.fetch = async (input, init) => {
                    const url = new URL(typeof input === 'string' ? input : input.url, location.href);
                    if (url.pathname === '/api/ui/preferences' && (!init?.method || init.method === 'GET')
                            && window.__holdPreferences) {
                        await new Promise(resolve => window.__pendingPreferences.push(resolve));
                        try { return await fetch(input, init); } finally { window.__settledPreferences += 1; }
                    }
                    return fetch(input, init);
                };
            }"""
            other, other_row = _client(browser, direct_server_with_data["url"], "shared-room", other_acks,
                                       init=(hold_preferences,))
            # Keep the barrier armed after draining the initial page-load reads.
            released = other.evaluate("""() => {
                const held = window.__pendingPreferences.splice(0);
                held.forEach(resolve => resolve());
                return held.length;
            }""")
            assert released >= 1, "the initial preferences read must settle before the room is read"
            other.wait_for_function("n => window.__settledPreferences >= n", arg=released)
            reader, row = _client(browser, direct_server_with_data["url"], "shared-room", reader_acks)
            row.evaluate("el => el.click()")
            reader.locator(PANEL).locator(".chat-bubble").filter(has_text="Reply 3").wait_for(state="attached")
            reader.evaluate(_SETTLE_RESTORE_FRAMES)
            assert _wait_for(reader, lambda: [ack.get("project_seen_revision") for ack in reader_acks]
                             == [{"shared-room": revision}]), reader_acks
            _unread(row).wait_for(state="detached")
            assert _unread(other_row).count() == 1, "the other client has not refreshed yet"
            _emit_ws_frame(other, {"type": "projects_changed"})
            other.wait_for_function("() => window.__pendingPreferences.length > 0")
            other.evaluate("""() => {
                window.__holdPreferences = false;
                window.__pendingPreferences.splice(0).forEach(resolve => resolve());
            }""")
            _unread(other_row).wait_for(state="detached")
            other.screenshot(path=str(evidence / f"read-band-{engine}-other-client.png"))
            assert [ack for ack in other_acks if ack.get("project_seen_revision")] == [], \
                "the other client never read the room and acknowledges nothing"
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_question_revealed_from_main_is_not_reading_the_newer_messages_below_it(direct_server_with_data, engine):
    """Main's question mirror opens the room at the question; the newer replies below it are read only
    once the reader reaches them (DESIGN "Unread")."""
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import bind_task_to_project, get_project
    from ouroboros.task_results import write_task_result

    data = direct_server_with_data["data_dir"]
    at = lambda minute: f"2026-09-28T10:{minute:02d}:00Z"  # noqa: E731
    quiz = {"quiz_id": "q-mirror", "question": QUESTION, "options": ["Yes", "No"], "state": "open", "asked_at": at(5)}
    rows = [{"ts": at(minute), "direction": "out", "text": f"Reply {minute}"} for minute in range(3)]
    rows.append({"ts": at(5), "direction": "out", "text": QUESTION, "type": "quiz", "task_id": "mirror-task",
                 "quiz": {**quiz, "options": [{"label": "Yes"}, {"label": "No"}]}})
    rows += [{"ts": at(10 + index), "direction": "out", "text": f"Later reply {index}\nline two\nline three"}
             for index in range(25)]
    chat_id, revision = _seed_room(data, "mirror-room", "Mirror room", rows)
    project = get_project(data, "mirror-room")
    bind_task_to_project(data, "mirror-task", project["id"], chat_id, origin={"absent": "system"})
    write_task_result(data, "mirror-task", "running", project_id=project["id"], chat_id=chat_id,
                      owner_quiz={"q-mirror": quiz})
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    acks = []
    seen = lambda: [ack["project_seen_revision"]["mirror-room"] for ack in acks  # noqa: E731
                    if "mirror-room" in (ack.get("project_seen_revision") or {})]
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, row = _client(browser, direct_server_with_data["url"], "mirror-room", acks)
            page.evaluate("""project => window.dispatchEvent(new CustomEvent('ouro:open-project', {
                detail: {project, task_id: 'mirror-task', quiz_id: 'q-mirror'}}))""", project)
            messages = page.locator(PANEL)
            card = messages.locator('.chat-quiz-card[data-quiz-id="q-mirror"]')
            card.wait_for(state="visible")
            page.wait_for_function("() => Boolean(document.activeElement?.closest('[data-quiz-id=\"q-mirror\"]'))")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.wait_for_timeout(500)
            newest = messages.evaluate(_BOX_OF, "Later reply 24")
            page.screenshot(path=str(evidence / f"read-band-{engine}-mirror-question.png"))
            assert newest["top"] > messages.evaluate("n => n.getBoundingClientRect().bottom"), newest
            assert seen() == [], "landing on the question is not reading the newer replies below it"
            assert _unread(row).count() == 1

            messages.evaluate(_AT_BOTTOM)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            assert _wait_for(page, lambda: seen() == [revision]), acks
            page.screenshot(path=str(evidence / f"read-band-{engine}-mirror-latest.png"))
            _unread(row).wait_for(state="detached")
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_reply_arriving_while_the_window_is_hidden_is_read_when_it_is_shown_again(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright


    data = direct_server_with_data["data_dir"]
    chat_id, first = _seed_room(data, "away-room", "Away room", [
        {"ts": f"2026-09-28T10:{minute:02d}:00Z", "direction": "out", "text": f"Reply {minute}"} for minute in range(4)])
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    acks = []
    seen = lambda: [ack["project_seen_revision"]["away-room"] for ack in acks  # noqa: E731
                    if "away-room" in (ack.get("project_seen_revision") or {})]
    with sync_playwright() as pw:
        browser, page, row = _open_room(pw, engine, direct_server_with_data["url"], "away-room", acks)
        try:
            messages = page.locator(PANEL)
            messages.locator(".chat-bubble").filter(has_text="Reply 3").wait_for(state="attached")
            assert _wait_for(page, lambda: seen() == [first]), acks
            _unread(row).wait_for(state="detached")

            page.evaluate(_SET_HIDDEN, True)
            dialogue_store.append(data, "chat", {"ts": "2026-09-28T10:30:00Z", "direction": "out", "chat_id": chat_id,
                                                 "text": "Reply while away"})
            second = advance_room_revision(data, chat_id=chat_id)["visible_revision"]
            _emit_ws_frame(page, {"type": "projects_changed"})
            _unread(row).wait_for(state="attached")
            messages.locator(".chat-bubble").filter(has_text="Reply while away").wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.wait_for_timeout(500)
            assert seen() == [first], "nobody can see a hidden window"

            page.evaluate(_SET_HIDDEN, False)
            assert _wait_for(page, lambda: seen() == [first, second]), acks
            _unread(row).wait_for(state="detached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.screenshot(path=str(evidence / f"read-band-{engine}-shown-again.png"))
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_failed_read_receipt_is_retried_at_the_next_state_refresh(direct_server_with_data, engine):
    """The reader stays at the newest message while every POST fails; once the server accepts it
    again, the next state refresh, which changes nothing in the Projects list, retries it."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    _chat_id, revision = _seed_room(data, "retry-room", "Retry room", [
        {"ts": f"2026-09-28T10:{minute:02d}:00Z", "direction": "out", "text": f"Reply {minute}"} for minute in range(4)])
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    attempts, failing = [], [True]

    def preferences(route):
        body = route.request.post_data or ""
        if route.request.method == "POST" and "retry-room" in body:
            attempts.append((json.loads(body)["project_seen_revision"], failing[0]))
            if failing[0]:
                return route.abort()
        return route.continue_()

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1187, "height": 734})
            page.route("**/api/ui/preferences", preferences)
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
            row = page.locator('.nav-project-row[data-project-id="retry-room"]')
            _unread(row).wait_for(state="attached")
            row.evaluate("el => el.click()")
            page.locator(PANEL).locator(".chat-bubble").filter(has_text="Reply 3").wait_for(state="attached")
            assert _wait_for(page, lambda: len(attempts) >= 1), attempts
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.wait_for_timeout(1500)  # the app's own start-up snapshots settle
            assert _unread(row).count() == 1, ("a failed POST keeps the dot", attempts)
            page.screenshot(path=str(evidence / f"read-band-{engine}-ack-failed.png"))

            failed, failing[0] = len(attempts), False
            _emit_ws_frame(page, {"type": "projects_changed"})
            assert _wait_for(page, lambda: len(attempts) > failed), attempts
            _unread(row).wait_for(state="detached")
            cursors = page.evaluate("() => fetch('/api/ui/preferences').then(r => r.json())")["project_seen_revision"]
            assert {json.dumps(seen) for seen, _ in attempts} == {json.dumps({"retry-room": revision})}, attempts
            assert attempts[-1][1] is False and cursors["retry-room"] == revision, (attempts, cursors)
            page.screenshot(path=str(evidence / f"read-band-{engine}-ack-retried.png"))
        finally:
            browser.close()


# The question detail of a reveal from Main is held until the test releases it.
_HOLD_QUESTION = """() => {
    const original = window.fetch.bind(window);
    window.fetch = async (input, init) => {
        const url = new URL(typeof input === 'string' ? input : input.url, location.href);
        if (window.__holdQuestion && url.pathname === '/api/tasks/held-task') {
            await new Promise(resolve => { window.__releaseQuestion = resolve; });
        }
        return original(input, init);
    };
}"""
_COMPOSER = "#project-panel .chat-input-area textarea"


@pytest.mark.parametrize("retained", [True, False], ids=["retained", "rebuilt"])
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_an_ordinary_reopen_reads_its_room_while_an_old_question_reveal_is_held(direct_server_with_data, engine,
                                                                              retained):
    """A question reveal belongs to the showing that started it. Closed while its question detail is
    held and reopened ordinarily, the room is read at its newest message at once, whether the reopen
    reuses the instance its staged file kept or builds a new one; the old reveal, released later,
    moves nothing and takes neither the draft nor the focus."""
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import bind_task_to_project, get_project
    from ouroboros.task_results import write_task_result

    data = direct_server_with_data["data_dir"]
    slug = "held-kept" if retained else "held-new"
    chat_id, revision = _seed_room(data, slug, "Held room", [
        {"ts": f"2026-09-28T10:{minute:02d}:00Z", "direction": "out", "text": f"Reply {minute}"} for minute in range(4)])
    project = get_project(data, slug)
    bind_task_to_project(data, "held-task", project["id"], chat_id, origin={"absent": "system"})
    write_task_result(data, "held-task", "running", project_id=project["id"], chat_id=chat_id, owner_quiz={"q-held": {
        "quiz_id": "q-held", "question": QUESTION, "options": ["Yes", "No"], "state": "open",
        "asked_at": "2026-09-28T10:02:30Z"}})
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    acks = []
    seen = lambda: [ack["project_seen_revision"][slug] for ack in acks  # noqa: E731
                    if slug in (ack.get("project_seen_revision") or {})]
    ask = """project => window.dispatchEvent(new CustomEvent('ouro:open-project', {
        detail: {project, task_id: 'held-task', quiz_id: 'q-held'}}))"""
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1187, "height": 734})
            page.add_init_script(f"({_HOLD_QUESTION})()")
            page.on("request", lambda request: acks.append(json.loads(request.post_data or "{}"))
                    if request.method == "POST" and request.url.endswith("/api/ui/preferences") else None)
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
            row = page.locator(f'.nav-project-row[data-project-id="{slug}"]')
            _unread(row).wait_for(state="attached")
            messages = page.locator(PANEL)
            if retained:
                row.evaluate("el => el.click()")
                messages.locator(".chat-bubble").filter(has_text="Reply 3").wait_for(state="attached")
                assert _wait_for(page, lambda: seen() == [revision]), acks
                page.locator("#project-panel .chat-file-input-hidden").set_input_files(
                    [{"name": "kept.txt", "mimeType": "text/plain", "buffer": b"kept"}])
                page.locator("#project-panel .attach-name").filter(has_text="kept.txt").wait_for()
                page.locator(_COMPOSER).fill(DRAFT)
                page.locator("#project-panel-close").click()
                page.locator('.chat-instance-panel[data-pending-work="1"]').wait_for(state='attached')
                assert page.locator('.chat-instance-panel[data-pending-work="1"]').count() == 1
                dialogue_store.append(data, "chat", {"ts": "2026-09-28T10:30:00Z", "direction": "out", "chat_id": chat_id,
                                                     "text": "Reply while closed"})
                revision = advance_room_revision(data, chat_id=chat_id)["visible_revision"]
                _emit_ws_frame(page, {"type": "projects_changed"})
                _unread(row).wait_for(state="attached")
            before = seen()
            page.evaluate("() => { window.__holdQuestion = true; }")
            page.evaluate(ask, project)
            page.wait_for_function("() => typeof window.__releaseQuestion === 'function'")
            page.wait_for_timeout(500)
            assert seen() == before, "landing on the question is not reading"
            page.locator("#project-panel-close").click()
            parked = page.locator('.chat-instance-panel[data-pending-work="1"]') if retained else messages
            parked.wait_for(state="attached" if retained else "detached")
            row.evaluate("el => el.click()")
            if not retained:
                page.locator(_COMPOSER).fill(DRAFT)
            read = _wait_for(page, lambda: seen() == [*before, revision], attempts=50)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.screenshot(path=str(evidence / f"read-band-{engine}-reopen-{slug}.png"))
            held = page.evaluate("() => typeof window.__releaseQuestion === 'function'")
            assert read and held, ("the reopened room is read while the old reveal is still held", acks, held)
            _unread(row).wait_for(state="detached")

            page.locator(_COMPOSER).focus()
            page.evaluate("() => { window.__holdQuestion = false; window.__releaseQuestion(); }")
            page.wait_for_timeout(500)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.screenshot(path=str(evidence / f"read-band-{engine}-released-{slug}.png"))
            assert page.locator(_COMPOSER).input_value() == DRAFT, "the draft is kept"
            assert page.evaluate("() => document.activeElement?.matches('.chat-input-area textarea')"), \
                "the released reveal takes no focus"
            assert messages.locator('[data-quiz-id="q-held"]').count() == 0, "nor reveals its question"
            if retained:
                assert page.locator("#project-panel .attach-name").filter(has_text="kept.txt").count() == 1
            assert seen() == [*before, revision], ("the retired reveal decides nothing", acks)
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("lineage_source", ["result", "progress", "pre-upgrade-result"])
def test_a_legacy_child_final_after_the_root_answer_does_not_hold_the_room_unread(
        direct_server_with_data, engine, lineage_source):
    """A child's final written before chat rows carried lineage is shown in the child's card by the
    result or progress lineage; the root's answer before it is the newest message, and it is read."""
    from playwright.sync_api import sync_playwright

    from ouroboros.task_results import write_task_result

    data = direct_server_with_data["data_dir"]
    at = lambda minute: f"2026-09-28T10:{minute:02d}:00Z"  # noqa: E731
    rows = [{"ts": at(minute), "direction": "out", "text": f"Reply {minute}"} for minute in range(3)]
    rows += [{"ts": at(5), "direction": "out", "text": "Root answer", "task_id": "root-legacy"},
             {"ts": at(6), "direction": "out", "text": "Legacy child final", "task_id": "kid-legacy"}]
    _chat_id, revision = _seed_room(data, "legacy-room", "Legacy room", rows)
    write_task_result(data, "root-legacy", "completed")
    lineage = {"delegation_role": "subagent", "parent_task_id": "root-legacy",
               "root_task_id": "root-legacy", "subagent_task_id": "kid-legacy", "subagent_role": "researcher"}
    if lineage_source == "result":
        write_task_result(data, "kid-legacy", "completed", **lineage)
    else:
        dialogue_store.append(data, "progress", {
            "ts": at(4), "chat_id": _chat_id, "task_id": "root-legacy", "content": "Child scheduled",
            "subagent_event": "scheduled", **lineage})
        if lineage_source == "pre-upgrade-result":
            seed_result(data, "kid-legacy", {"id": "kid-legacy", "status": "completed", **lineage})
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    acks = []
    seen = lambda: [ack["project_seen_revision"]["legacy-room"] for ack in acks  # noqa: E731
                    if "legacy-room" in (ack.get("project_seen_revision") or {})]
    with sync_playwright() as pw:
        browser, page, row = _open_room(pw, engine, direct_server_with_data["url"], "legacy-room", acks)
        try:
            messages = page.locator(PANEL)
            messages.locator(".chat-bubble").filter(has_text="Root answer").wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            read = _wait_for(page, lambda: seen() == [revision], attempts=50)
            page.screenshot(path=str(evidence / f"read-band-{engine}-legacy-child-{lineage_source}.png"))
            assert read, ("the root's answer on screen is read", acks)
            assert messages.locator(".chat-bubble").filter(has_text="Legacy child final").count() == 0
            assert messages.locator(".chat-live-card").filter(has_text="Legacy child final").count() > 0
            _unread(row).wait_for(state="detached")
        finally:
            browser.close()


# What each Project history read the page received says about the newest message. A read's
# cursor names it (``kind``): ``older`` walks the chain, ``after`` takes what arrived since the
# newest read; a read without one (null) is the newest page.
_RECORD_HISTORY = """() => {
    const original = window.fetch.bind(window);
    window.__historyReads = [];
    window.fetch = async (input, init) => {
        const response = await original(input, init);
        const url = new URL(typeof input === 'string' ? input : input.url, location.href);
        if (url.pathname === '/api/chat/history' && url.searchParams.get('chat_id')) {
            const cursor = url.searchParams.get('cursor');
            const kind = cursor ? JSON.parse(atob(cursor.replace(/-/g, '+').replace(/_/g, '/'))).kind : null;
            response.clone().json().then(data => window.__historyReads.push({
                kind, older: kind === 'older', named: 'latest_message' in (data.window || {}),
                latest: data.window?.latest_message ?? null, before: data.window?.latest_before ?? null,
                absent: data.window?.latest_absent === true,
                upper: data.coverage?.upper?.chat ?? null, from: data.coverage?.spans?.chat?.from ?? null,
                texts: (data.messages || []).map(row => String(row.text || '').slice(0, 24)),
            }), () => {});
        }
        return response;
    };
}"""


@pytest.mark.parametrize("interrupted", [False, True], ids=["continuous", "interrupted"])
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_newest_answer_past_the_bounded_search_is_reached_by_the_opening_through_its_quiet_chain(
        direct_server_with_data, engine, interrupted):
    """Owner notes after the newest answer outrun the recent read's bounded search (its window of the
    room's newest rows and the rows it searches past it), so its arrival is unknown. The opening reads
    the older pages through them itself, carrying the search on to the answer, lands there and reads
    it; an unreadable row between them leaves it unknown and ends that reach. A newer answer is read
    only where it is."""
    from playwright.sync_api import sync_playwright


    data = direct_server_with_data["data_dir"]
    at = lambda second: f"2026-09-28T10:{second // 60:02d}:{second % 60:02d}Z"  # noqa: E731
    note = lambda index: {"ts": at(10 + index), "direction": "in",  # noqa: E731
                          "text": f"Owner note {index}: " + "a long thought " * 8}
    notes = 1230
    assert notes > _RECENT_WINDOW_ROWS + _ARRIVAL_SEARCH_ROWS, "the notes outrun the recent read's bounded search"
    # The answer and the older notes are an older install's archive, imported: an unreadable line is a
    # gap only the import keeps; the newest 160 notes are stored as the server stores them.
    archived = [{"ts": at(second), "direction": "out", "text": f"Reply {second}"} for second in range(3)]
    archived += [{"ts": at(5), "direction": "out", "text": f"{LATE}\nwith its details"}, *map(note, range(notes - 160))]
    if interrupted:
        archived.insert(4, '{"direction": "out", "text": "torn\n')
    chat_id, revision = _seed_room(data, "searched-room", "Searched room", [note(i) for i in range(notes - 160, notes)],
                                   archives=[("chat_20260928T101000.jsonl", archived)])
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    acks = []
    seen = lambda: [ack["project_seen_revision"]["searched-room"] for ack in acks  # noqa: E731
                    if "searched-room" in (ack.get("project_seen_revision") or {})]
    name = f"read-band-{engine}-searched-{'interrupted' if interrupted else 'continuous'}"
    with sync_playwright() as pw:
        browser, page, row = _open_room(pw, engine, direct_server_with_data["url"], "searched-room", acks,
                                        (_RECORD_HISTORY,))
        try:
            messages = page.locator(PANEL)
            messages.locator(".chat-bubble").filter(has_text=LATE).wait_for(state="attached")
            read = _wait_for(page, lambda: seen() == [revision], attempts=40 if interrupted else 100)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            (landing, *_) = page.evaluate("() => window.__historyReads")
            assert not landing["older"] and landing["latest"] is None and 0 < landing["before"] < landing["upper"], \
                ("the recent read's search ran out above the answer", landing)
            older = [item for item in page.evaluate("() => window.__historyReads") if item["older"]]
            assert [item["named"] for item in older] == [False] * (len(older) - 1) + [True], older
            assert (older[-1]["latest"] is None) == interrupted and any(LATE in text for text in older[-1]["texts"]), \
                ("the opening read on to the quiet page holding the answer, which names it unless a row after it is unreadable", older[-1])
            clear = messages.evaluate(_BOX_OF, LATE)
            page.screenshot(path=str(evidence / f"{name}.png"))
            reads = [{key: value for key, value in item.items() if key != "texts"}
                     for item in page.evaluate("() => window.__historyReads")]
            (evidence / f"{name}.json").write_text(json.dumps({"clear": clear, "acks": acks, "reads": reads}, indent=2),
                                                   encoding="utf-8")
            if interrupted:
                assert not read and seen() == [], ("past an unreadable row the answer may not be the newest", acks)
                assert _unread(row).count() == 1
                return
            assert read and clear["top"] >= clear["feedTop"], ("the opening lands on the answer and reads it", acks, clear)
            # A refresh reads what arrived since the newest read once the newest message is known
            # (project_read_state.js `refresh`): those arrivals start at the chain's frozen upper.
            since = reads[max(index for index, item in enumerate(reads) if item["older"]) + 1:]
            assert [item["kind"] for item in reads].count(None) == 1 and since \
                and all(item["kind"] == "after" and not item["named"] for item in since) \
                and since[0]["from"] == older[-1]["upper"], \
                ("the uncovered revision was read again, and what arrived since the chain found nothing newer", reads)
            _unread(row).wait_for(state="detached")

            dialogue_store.append(data, "chat", {"ts": at(2000), "direction": "out", "chat_id": chat_id,
                                                 "text": "Newer answer"})
            newer = advance_room_revision(data, chat_id=chat_id)["visible_revision"]
            _emit_ws_frame(page, {"type": "projects_changed"})
            _unread(row).wait_for(state="attached")
            messages.evaluate(_PLACE, [LATE, -160])
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.wait_for_timeout(800)
            page.screenshot(path=str(evidence / f"{name}-newer-unread.png"))
            assert seen() == [revision], ("the older answer on screen does not read a newer one", acks)
            page.locator("#project-panel-body .chat-scroll-bottom-btn").click()
            messages.locator(".chat-bubble").filter(has_text="Newer answer").wait_for(state="attached")
            assert _wait_for(page, lambda: seen() == [revision, newer]), acks
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.screenshot(path=str(evidence / f"{name}-newer-read.png"))
            _unread(row).wait_for(state="detached")
        finally:
            browser.close()


def _png(width=240, height=120):
    """A real image, so the photo bubble has its ordinary size."""
    import struct
    import zlib

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x40\x80\xc0" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_photo_a_child_delivers_is_read_where_it_is_not_at_the_root_answer_above_it(direct_server_with_data, engine):
    """A child's words are card content, but a photo it delivers is its own bubble and the message
    that arrived last: the root answer the reader keeps on screen is not reading it."""
    from urllib.parse import quote

    from playwright.sync_api import sync_playwright

    from ouroboros.artifacts import store_chat_media_bytes
    from ouroboros.task_results import write_task_result

    data = direct_server_with_data["data_dir"]
    at = lambda minute: f"2026-09-28T10:{minute:02d}:00Z"  # noqa: E731
    rows = [{"ts": at(minute), "direction": "out", "text": f"Reply {minute}"} for minute in range(3)]
    rows.append({"ts": at(5), "direction": "out", "text": "Root answer\nwith its details", "task_id": "root-photo"})
    # The owner's own follow-ups keep the bottom of the room below the fold while the answer is on screen.
    rows += [{"ts": at(6 + index), "direction": "in",
              "text": "\n".join(f"Owner follow-up {index}, line {line}" for line in range(8))} for index in range(6)]
    chat_id, first = _seed_room(data, "photo-room", "Photo room", rows)
    write_task_result(data, "root-photo", "completed")
    write_task_result(data, "kid-photo", "completed", delegation_role="subagent", parent_task_id="root-photo",
                      root_task_id="root-photo", role="researcher")
    stored = store_chat_media_bytes(data, "kid-photo", _png(), "image/png")
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    acks = []
    seen = lambda: [ack["project_seen_revision"]["photo-room"] for ack in acks  # noqa: E731
                    if "photo-room" in (ack.get("project_seen_revision") or {})]
    with sync_playwright() as pw:
        browser, page, row = _open_room(pw, engine, direct_server_with_data["url"], "photo-room", acks)
        try:
            messages = page.locator(PANEL)
            messages.locator(".chat-bubble").filter(has_text="Owner follow-up 5").wait_for(state="attached")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            answer = messages.evaluate(_PLACE, ["Root answer", -160])
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            assert answer["node"]["bottom"] <= answer["composer"]["top"], answer
            assert _wait_for(page, lambda: seen() == [first]), ("the root answer on screen is read", acks)
            _unread(row).wait_for(state="detached")

            # The child's photo arrives while the reader stays at the answer (message_bus.send_photo's row).
            dialogue_store.append(data, "chat", {
                "ts": at(30), "direction": "out", "chat_id": chat_id, "task_id": "kid-photo", "type": "photo",
                "text": "Child chart", "caption": "Child chart", "mime": "image/png",
                "download_url": f"/api/tasks/kid-photo/artifacts/{quote(stored['name'])}"})
            second = advance_room_revision(data, chat_id=chat_id)["visible_revision"]
            _emit_ws_frame(page, {"type": "projects_changed"})
            photo = messages.locator("figure.chat-gallery-item").filter(has_text="Child chart")
            photo.wait_for(state="attached")
            page.wait_for_function("() => [...document.querySelectorAll('#project-panel img.chat-photo')]"
                                   ".every(img => img.complete && img.naturalWidth > 0)")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.wait_for_timeout(800)
            held = messages.evaluate(_BOX_OF, "Root answer")
            below = photo.evaluate("""n => ({top: n.getBoundingClientRect().top, card: Boolean(n.closest('.chat-live-card')),
                feedBottom: n.closest('.chat-messages').getBoundingClientRect().bottom})""")
            page.screenshot(path=str(evidence / f"read-band-{engine}-child-photo-below.png"))
            assert not below["card"], ("the child's photo is shown alone, not in a card", below)
            assert held["top"] >= held["feedTop"] and below["top"] >= below["feedBottom"], (held, below)
            assert seen() == [first], ("the answer on screen is not the photo that arrived after it", acks)
            assert _unread(row).count() == 1

            messages.evaluate(_AT_BOTTOM)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            read = _wait_for(page, lambda: seen() == [first, second])
            page.screenshot(path=str(evidence / f"read-band-{engine}-child-photo-read.png"))
            (evidence / f"read-band-{engine}-child-photo.json").write_text(
                json.dumps({"answer": answer, "held": held, "below": below, "acks": acks}, indent=2), encoding="utf-8")
            assert read, ("the photo on screen is read", acks)
            _unread(row).wait_for(state="detached")
        finally:
            browser.close()


@pytest.mark.parametrize("early", [0, 1000], ids=["start-in-the-recent-read", "start-past-kept-pages"])
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_long_room_with_no_standalone_message_is_read_at_the_bottom_once_the_opening_reaches_its_start(
        direct_server_with_data, engine, early):
    """Only the owner's own notes (the room's revision once counted a child's words): their arrival is
    known once a read reaches the start of the chat, which proves no standalone message is there, and
    the bottom is then read. The recent read takes the room's newest rows whole up to its window, so 330
    notes are proven by that read itself, without an older page. With ``early`` short notes older still,
    the room outruns that read's bounded search: its arrival is unknown until the older pages the opening
    reads reach the start, whose page lies past the three the pager keeps; the newest page went on the
    way: the opening returns there, as ↓ does, and reads the room's own bottom."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    at = lambda second: f"2026-09-28T10:{second // 60:02d}:{second % 60:02d}Z"  # noqa: E731
    note = lambda index: {"ts": at(10 + index), "direction": "in",  # noqa: E731
                          "text": f"Owner note {index}: " + "a long thought " * 270}
    # An older install's archives (the 170 oldest notes, and with ``early`` an earlier segment of short
    # notes), imported; the newest 160 notes are stored as the server stores them.
    archives = [("chat_20260928T101000.jsonl", [note(i) for i in range(170)])]
    if early:
        archives.insert(0, ("chat_20260928T095900.jsonl", [{
            "ts": f"2026-09-28T09:{i // 60:02d}:{i % 60:02d}Z", "direction": "in", "text": f"Early note {i}"}
            for i in range(early)]))
    assert (330 + early > _RECENT_WINDOW_ROWS + _ARRIVAL_SEARCH_ROWS) == bool(early) and 330 <= _RECENT_WINDOW_ROWS
    chat_id, revision = _seed_room(data, "quiet-room", "Quiet room", [note(i) for i in range(170, 330)],
                                   archives=archives)
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    acks = []
    seen = lambda: [ack["project_seen_revision"]["quiet-room"] for ack in acks  # noqa: E731
                    if "quiet-room" in (ack.get("project_seen_revision") or {})]
    name = f"read-band-{engine}-quiet-room{'-past-kept-pages' if early else ''}"
    with sync_playwright() as pw:
        browser, page, row = _open_room(pw, engine, direct_server_with_data["url"], "quiet-room", acks,
                                        (_RECORD_HISTORY,))
        try:
            messages = page.locator(PANEL)
            if not early:
                messages.locator(".chat-bubble").filter(has_text="Owner note 329:").wait_for(state="attached")
            read = _wait_for(page, lambda: seen() == [revision], attempts=200)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            (landing, *_) = page.evaluate("() => window.__historyReads")
            older = [item for item in page.evaluate("() => window.__historyReads") if item["older"]]
            if not early:
                assert not landing["older"] and not landing["named"] and not older, \
                    ("the recent read reached the start itself: it names no standalone message, no older page is read",
                     landing, older)
            else:
                assert not landing["older"] and landing["latest"] is None and 0 < landing["before"] < landing["upper"], \
                    ("the recent read's search ran out", landing)
                assert not any(item["named"] for item in older), older
                assert [item["absent"] for item in older] == [False] * (len(older) - 1) + [True], \
                    ("only the page reaching the start proves no standalone message is there", older)
            assert messages.evaluate("n => n.scrollHeight - n.scrollTop - n.clientHeight") <= 1, "at the bottom"
            newest = messages.evaluate(_BOX_OF, "Owner note 329:")
            page.screenshot(path=str(evidence / f"{name}.png"))
            reads = [{key: value for key, value in item.items() if key != "texts"}
                     for item in page.evaluate("() => window.__historyReads")]
            (evidence / f"{name}.json").write_text(
                json.dumps({"acks": acks, "reads": reads, "newest": newest}, indent=2), encoding="utf-8")
            assert read, ("no standalone message: the bottom is read", acks, reads[-3:])
            assert newest["bottom"] > newest["feedTop"], ("the room's own newest note is on screen", newest)
            if early:
                start = max(index for index, item in enumerate(reads) if item["absent"])
                # A state refresh may read what arrived meanwhile (an `after` read, #1954): not the opening's own.
                opening = [item["kind"] for item in reads[start + 1:] if item["kind"] != "after"]
                assert len(older) > 3 and opening[:1] == [None], \
                    ("past the pages kept, the opening went on to read the newest page again", reads)
                assert messages.locator(".chat-bubble").filter(has_text="Early note 0").count() == 0, \
                    "the pages walked past went with their chain"
            _unread(row).wait_for(state="detached")
        finally:
            browser.close()

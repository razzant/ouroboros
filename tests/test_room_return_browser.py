"""A Project room left within this page session is found where it was left (owner 2026-11-03; DESIGN §10
"Moving between places", "Unread").

Real server, real retained chat history, real history endpoint, read cursor and socket. A room left reading an
older passage returns to that passage, its own selection, draft, answer, disclosure, living card (the tab chosen and
what was typed in its form) and node, through ten other rooms (A→B→…→A) while its history answer is held: every visited
room stays (parent decision 2026-11-04). Messages that
arrived while it was away are read again on return and stay unread below the passage; a room left at its newest message
follows the fresh end and reads it. A history read still in flight when the room leaves is not drawn while it is away.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from ouroboros import dialogue_store
from tests._dialogue import advance_room_revision
from tests._room_browser_shared import (
    _ROW_AT_TOP,
    _SELECT,
    _client,
    _open,
    _read_older,
    _rows,
    _seed_room,
    _unread,
    _wait_for,
)
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _SETTLE_RESTORE_FRAMES, _emit_ws_frame

pytestmark = [pytest.mark.ui_browser]

_AT_BOTTOM = """pid => {
    const feed = document.querySelector(`[id="panel-pchat-${pid}"] .chat-messages`);
    return feed.scrollHeight - feed.scrollTop - feed.clientHeight;
}"""


def _read_newest(page, pid):
    """The owner's own wheel takes the room down to its newest message."""
    box = page.locator(f'[id="panel-pchat-{pid}"] .chat-messages').bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.wheel(0, 20_000)
    page.evaluate(_SETTLE_RESTORE_FRAMES)
    page.wait_for_timeout(300)
    return page.evaluate(_AT_BOTTOM, pid)


def _seen(acks, pid):
    return [ack["project_seen_revision"][pid] for ack in acks if pid in (ack.get("project_seen_revision") or {})]


_LOADING = "Loading saved history"
_CARD_TABS = [{"label": "Monthly", "components": [{"type": "markdown", "text": "Monthly totals."}]},
              {"label": "Yearly", "components": [{"type": "form", "id": "note", "title": "Note", "submit_label": "Send note",
                                                  "fields": [{"name": "text", "label": "Text"}], "submit": {"id": "note"}}]}]


def _seed_card(data, chat_id):
    """An ordinary living card at the room's fresh end, as the server stores it: two tabs, a form on the second."""
    from ouroboros.card_store import write_card

    record = write_card(data, chat_id=chat_id, task_id="", title="Plan note", fallback_text="A plan and a note",
                        kind="components", body={"components": [{"type": "tabs", "tabs": _CARD_TABS}]})
    dialogue_store.append(data, "chat", {"ts": "2026-09-28T12:00:00Z", "direction": "out", "chat_id": chat_id, "user_id": 0,
                                         "task_id": "", "text": record["fallback_text"], "type": "card", "card": {
                                             "card_id": record["card_id"], "kind": record["kind"], "title": record["title"], "revision": 1}})
    advance_room_revision(data, chat_id=chat_id)


def seed_visited_rooms(data):
    """Rooms A and C with a living card, room B, and eight more to go through. Returns their chats and the eight."""
    chats = {pid: _seed_room(data, pid, name, _rows(word, count))[0]
             for pid, name, word, count in (("room-a", "Room A", "Alpha", 60), ("room-b", "Room B", "Bravo", 24),
                                            ("room-c", "Room C", "Charlie", 30))}
    for pid in ("room-a", "room-c"):
        _seed_card(data, chats[pid])
    fillers = [f"room-f{index}" for index in range(1, 9)]
    for pid in fillers:
        _seed_room(data, pid, pid.replace("-", " ").title(), _rows(pid, 3))
    return chats, fillers


def _room_state(page, pid):
    """What the owner left in a room: its node, a typed question answer, its draft, a card disclosure, a living card's
    chosen tab and form text, the pill."""
    return page.evaluate("""pid => {
        const room = document.getElementById(`panel-pchat-${pid}`);
        const card = room.querySelector('.chat-live-card[data-task-id="work-' + pid + '"]');
        const living = room.querySelector('.chat-card[data-card-id]');
        return {kept: room.__kept || null, answer: room.querySelector('.chat-quiz-comment')?.value ?? null,
            draft: room.querySelector('textarea[id$="-input"]')?.value ?? null, expanded: card?.dataset.expanded ?? null,
            tab: living?.querySelector('[data-widget-tab-key][aria-selected="true"]')?.textContent ?? null,
            note: living?.querySelector('form.widget-form input[name="text"], form.widget-form textarea[name="text"]')?.value ?? null,
            loading: [...room.querySelectorAll('*')].some(n => n.childElementCount === 0 && n.offsetParent !== null
                && n.textContent.includes('""" + _LOADING + """'))};
    }""", pid)


def _leave_with_state(page, pid, chat_id, *, reading_older):
    """The owner answers a question in the room (not sent), chooses a living card's tab and types in its form (not sent),
    opens a card's details, types a draft; optionally reads older history and selects a passage there. Returns the place
    and the selection."""
    _open(page, pid)
    _emit_ws_frame(page, {"type": "chat", "role": "system", "is_progress": True, "chat_id": chat_id, "task_id": f"work-{pid}",
                          "content": "Reading the build logs", "ts": "2026-11-04T09:30:00Z"})
    _emit_ws_frame(page, {"type": "quiz", "role": "assistant", "question": f"Which build ships from {pid}?", "chat_id": chat_id,
                          "ts": "2026-11-04T09:30:01Z", "task_id": f"ask-{pid}", "quiz_id": f"q-{pid}", "wait_for_answer": False,
                          "options": [{"label": "Yes"}, {"label": "No"}]})
    room = page.locator(f'[id="panel-pchat-{pid}"]')
    room.locator(".chat-quiz-comment").fill(f"Typed answer kept in {pid}")
    living = room.locator(".chat-card[data-card-id]")
    living.locator("[data-widget-tab-key]", has_text="Yearly").click()
    living.locator('form.widget-form [name="text"]').fill(f"Typed note kept in {pid}")
    if reading_older:
        room.locator(f'.chat-live-card[data-task-id="work-{pid}"] [data-live-disclosure]').click()
    room.locator('textarea[id$="-input"]').fill(f"Draft kept in {pid}")
    page.evaluate("() => document.activeElement?.blur()")
    page.evaluate("pid => { document.getElementById(`panel-pchat-${pid}`).__kept = pid; }", pid)
    if not reading_older:
        return _read_newest(page, pid), None
    place = _read_older(page, pid)
    return place, page.evaluate(_SELECT, [pid, "Alpha row"])


# Retained rooms keep hidden controls in the shared bar; inspect only the shown room.
_BAR_WORD = "() => document.querySelector('#project-panel .rw-trigger:not([hidden]) .status-badge')?.textContent || ''"


def _bar_word_becomes(page, test, timeout=5_000):
    """The shown room's status word once ``test`` (a JS predicate on it) holds, else what it said at the deadline."""
    try:
        page.wait_for_function(f"() => ({test})(({_BAR_WORD})())", timeout=timeout)
    except Exception:  # the assertion reports the word it found
        pass
    return page.evaluate(_BAR_WORD)


def visited_rooms_scenario(page, chats, fillers):
    """A (reading older, selection, answer, living card, disclosure, draft) and C (at its newest, answer, living card and
    draft), then B and the fillers, then A and C again while their history answers are held, and once those land."""
    c_below_newest, _ = _leave_with_state(page, "room-c", chats["room-c"], reading_older=False)
    place, chosen = _leave_with_state(page, "room-a", chats["room-a"], reading_older=True)
    before = {pid: _room_state(page, pid) for pid in ("room-a", "room-c")}
    _open(page, "room-b")
    page.evaluate(_SELECT, ["room-b", "Bravo row 2"])
    for pid in fillers:
        _open(page, pid)
    mounted = page.evaluate("() => document.querySelectorAll('.chat-instance-panel').length")
    held = []
    held_chats = {str(chats["room-a"]), str(chats["room-c"])}

    def history(route):
        query = parse_qs(urlparse(route.request.url).query)
        if (query.get("chat_id") or [""])[0] in held_chats:
            held.append(route)
        else:
            route.continue_()
    page.route("**/api/chat/history*", history)
    facts = {"mounted_rooms_before_return": mounted, "before": before, "c_px_below_newest_when_left": c_below_newest}
    for pid in ("room-a", "room-c"):
        page.locator(f'.nav-project-row[data-project-id="{pid}"]').evaluate("el => el.click()")
        page.locator(f'[id="panel-pchat-{pid}"]:not([hidden])').wait_for(timeout=5_000)
        page.evaluate(_SETTLE_RESTORE_FRAMES)
        facts[pid] = {**_room_state(page, pid), "selection": page.evaluate("() => getSelection().toString()"),
                      "held_history_reads": len(held)}
        # Still reading what it missed after the 300 ms bound: the bar says so over the last known state.
        facts[pid]["word_while_held"] = _bar_word_becomes(page, "word => word === 'Updating…'")
        if evidence_root := os.environ.get("OUROBOROS_UI_EVIDENCE_DIR"):
            engine = page.context.browser.browser_type.name
            page.screenshot(path=str(Path(evidence_root) / f"visited-rooms-updating-{engine}-{pid}.png"))
        if pid == "room-a":
            facts[pid]["place"] = page.evaluate(_ROW_AT_TOP, pid)
    facts.update(place=place, chosen=chosen)
    for route in held:
        route.continue_()
    page.unroute("**/api/chat/history*")
    # The held answers land: nothing the owner left is redrawn away, in the room shown or on going back to the other.
    page.evaluate(_SETTLE_RESTORE_FRAMES)
    page.wait_for_timeout(500)
    facts["word_once_answered"] = _bar_word_becomes(page, "word => word && word !== 'Updating…'")
    facts["room-c-answered"] = _room_state(page, "room-c")
    _open(page, "room-a")
    facts["room-a-answered"] = {**_room_state(page, "room-a"), "selection": page.evaluate("() => getSelection().toString()"),
                                "place": page.evaluate(_ROW_AT_TOP, "room-a")}
    return facts


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_every_visited_room_returns_at_once_with_selection_answer_draft_and_disclosure_while_its_history_is_held(direct_server_with_data, engine):
    """Parent decision 2026-11-04: every room visited in this page session returns at once, as left, by its own owners;
    nothing waits on the server and nothing is evicted, however many rooms were visited."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    chats, fillers = seed_visited_rooms(data)
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _row = _client(browser, direct_server_with_data["url"], "room-a", [])
            facts = visited_rooms_scenario(page, chats, fillers)
            (evidence / f"visited-rooms-{engine}.json").write_text(json.dumps(facts, indent=2))
            page.screenshot(path=str(evidence / f"visited-rooms-{engine}.png"))
            assert facts["mounted_rooms_before_return"] == 11, "no visited room was let go"
            assert facts["c_px_below_newest_when_left"] <= 2, "C was left at its newest message"
            a, c = facts["room-a"], facts["room-c"]
            assert a["kept"] == "room-a" and c["kept"] == "room-c", "the same rooms, never rebuilt"
            assert a["held_history_reads"] >= 1, "the return's catch-up read was asked and is still unanswered"
            assert a["word_while_held"] == c["word_while_held"] == "Updating…", "a return still catching up says so"
            assert facts["word_once_answered"] not in ("", "Updating…"), facts["word_once_answered"]
            for pid, now in (("room-a", a), ("room-c", c), ("room-a", facts["room-a-answered"]), ("room-c", facts["room-c-answered"])):
                assert now["answer"] == f"Typed answer kept in {pid}" and now["draft"] == f"Draft kept in {pid}", now
                assert now["tab"] == "Yearly" and now["note"] == f"Typed note kept in {pid}", now
                assert now["loading"] is False, f"{pid} shows no loading while its history answer is held"
            for now in (a, facts["room-a-answered"]):
                assert now["expanded"] == "1", "the card A's owner opened is still open"
                assert now["selection"] == facts["chosen"], "A's own selection, not B's"
                assert now["place"]["id"] == facts["place"]["id"] and abs(now["place"]["offset"] - facts["place"]["offset"]) <= 2
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_messages_arriving_while_a_room_is_away_stay_unread_below_its_passage_until_shown(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    chat_a, first = _seed_room(data, "away-a", "Away A", _rows("Alpha", 60))
    _seed_room(data, "away-b", "Away B", _rows("Bravo", 4))
    acks = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, row = _client(browser, direct_server_with_data["url"], "away-a", acks)
            _open(page, "away-a")
            assert _wait_for(page, lambda: _seen(acks, "away-a") == [first]), acks
            place = _read_older(page, "away-a")
            chosen = page.evaluate(_SELECT, ["away-a", "Alpha row"])
            _open(page, "away-b")
            # While A is away the host saves two new messages and says so.
            for minute, text in ((40, "News while away one"), (41, "News while away two")):
                dialogue_store.append(data, "chat", {"ts": f"2026-09-28T11:{minute}:00Z", "direction": "out",
                                                     "chat_id": chat_a, "text": text})
            second = advance_room_revision(data, chat_id=chat_a)["visible_revision"]
            _emit_ws_frame(page, {"type": "projects_changed"})
            _unread(row).wait_for(state="attached")
            away = page.locator('[id="panel-pchat-away-a"] [data-history-id]').filter(has_text="News while away")
            page.wait_for_timeout(600)
            assert away.count() == 0, "a room away draws nothing it was not there for"

            _open(page, "away-a")
            away.last.wait_for(state="attached")  # the return reads what arrived meanwhile
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            back = page.evaluate(_ROW_AT_TOP, "away-a")
            assert back["id"] == place["id"] and abs(back["offset"] - place["offset"]) <= 2, (place, back)
            assert page.evaluate("() => getSelection().toString()") == chosen, "the catch-up keeps the selection"
            arrow = page.locator('[id="panel-pchat-away-a"] .chat-scroll-bottom-btn')
            assert "visible" in (arrow.get_attribute("class") or "")
            assert page.locator('[id="panel-pchat-away-a"] .chat-scroll-activity-dot').is_hidden() is False
            page.wait_for_timeout(800)
            assert _seen(acks, "away-a") == [first], "the newer messages were not shown: not read"
            _unread(row).wait_for(state="attached")
            arrow.click()
            assert _wait_for(page, lambda: _seen(acks, "away-a") == [first, second]), acks
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_room_left_at_its_newest_message_follows_the_fresh_end_and_reads_it(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    chat_a, first = _seed_room(data, "end-a", "End A", _rows("Alpha", 30))
    _seed_room(data, "end-b", "End B", _rows("Bravo", 4))
    acks = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, row = _client(browser, direct_server_with_data["url"], "end-a", acks)
            _open(page, "end-a")
            assert _wait_for(page, lambda: _seen(acks, "end-a") == [first]), acks
            _open(page, "end-b")
            dialogue_store.append(data, "chat", {"ts": "2026-09-28T11:50:00Z", "direction": "out", "chat_id": chat_a,
                                                 "text": "Fresh end while away"})
            second = advance_room_revision(data, chat_id=chat_a)["visible_revision"]
            _emit_ws_frame(page, {"type": "projects_changed"})
            _unread(row).wait_for(state="attached")
            _open(page, "end-a")
            page.locator('[id="panel-pchat-end-a"] [data-history-id]').filter(has_text="Fresh end while away").wait_for()
            assert _wait_for(page, lambda: page.evaluate(_AT_BOTTOM, "end-a") <= 8), "it follows the fresh end"
            assert _wait_for(page, lambda: _seen(acks, "end-a") == [first, second]), acks
            _unread(row).wait_for(state="detached")
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_history_read_still_in_flight_when_a_room_leaves_is_not_drawn_while_it_is_away(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    chat_a, _first = _seed_room(data, "late-a", "Late A", _rows("Alpha", 12))
    _seed_room(data, "late-b", "Late B", _rows("Bravo", 4))
    held = []
    acks = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _row = _client(browser, direct_server_with_data["url"], "late-a", acks)

            def hold_first_read(route):
                query = route.request.url.split("?", 1)[-1]
                if f"chat_id={chat_a}" in query and not held:
                    held.append(route)
                    return
                route.continue_()
            page.route("**/api/chat/history*", hold_first_read)
            page.locator('.nav-project-row[data-project-id="late-a"]').evaluate("el => el.click()")
            assert _wait_for(page, lambda: bool(held)), "A's first read is in flight"
            _open(page, "late-b")
            held[0].continue_()  # A's original answer arrives while A is away
            page.wait_for_timeout(800)
            rows = page.locator('[id="panel-pchat-late-a"] [data-history-id]')
            assert rows.count() == 0, "the late answer is not drawn into a room away"
            _open(page, "late-a")
            rows.filter(has_text="Alpha row 11").wait_for(state="attached")
            assert json.dumps(page.evaluate("() => document.getElementById('panel-pchat-late-a').hidden")) == "false"
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_an_upload_in_flight_finishes_while_its_room_is_away_and_the_saved_row_settles_it_once(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright

    data = direct_server_with_data["data_dir"]
    _seed_room(data, "send-a", "Send A", _rows("Alpha", 6))
    _seed_room(data, "send-b", "Send B", _rows("Bravo", 4))
    acks, held = [], []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _row = _client(browser, direct_server_with_data["url"], "send-a", acks)
            _open(page, "send-a")
            room = '[id="panel-pchat-send-a"]'

            def hold_post(route):
                if route.request.method == "POST":
                    held.append(route)
                else:
                    route.continue_()
            page.route("**/api/chat/upload", hold_post)
            page.locator(f"{room} .chat-file-input-hidden").set_input_files(
                [{"name": "inflight.txt", "mimeType": "text/plain", "buffer": b"kept bytes"}])
            page.locator(f"{room} .attach-name").filter(has_text="inflight.txt").wait_for()
            field = page.locator(f"{room} textarea")
            field.fill("Words sent with the file")
            field.press("Enter")
            assert _wait_for(page, lambda: bool(held)), "the upload is in flight"

            _open(page, "send-b")
            assert page.locator(f'{room}[data-pending-work="1"]').count() == 1, "kept for its upload"
            held[0].continue_()  # it finishes while the room is away; its own send and saved echo still reach it
            bubble = page.locator(f"{room} .chat-bubble.user").filter(has_text="Words sent with the file")
            bubble.wait_for(state="attached", timeout=30_000)
            bubble.locator("[data-ingress-saved]").wait_for(state="attached", timeout=30_000)

            _open(page, "send-a")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            expect(bubble).to_have_count(1)
            assert bubble.get_by_role("button", name="Send again", exact=True).count() == 0, "a saved send is not offered again"
            assert page.locator(f"{room} .attach-name").count() == 0, "the staged file left with the send"
            assert page.locator(f"{room}[data-pending-work]").count() == 0
            page.wait_for_timeout(600)
            expect(bubble).to_have_count(1)
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_on_a_phone_a_room_left_through_the_open_drawer_returns_to_its_passage_and_selection(direct_server_with_data, engine):
    """The narrow layout's own path: the room leaves inside the card as the drawer closes (retired, then away). A phone
    is pressed by touch, which leaves its selection native (selection_lifecycle.js), so the selection comes back."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    _seed_room(data, "phone-a", "Phone A", _rows("Alpha", 40))
    _seed_room(data, "phone-b", "Phone B", _rows("Bravo", 6))
    acks, selections = [], []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _row = _client(browser, direct_server_with_data["url"], "phone-a", acks, has_touch=True)
            page.set_viewport_size({"width": 390, "height": 780})
            page.wait_for_function("() => document.documentElement.dataset.layout === 'stack'")
            page.evaluate("() => { window.__presses = []; document.addEventListener('pointerdown', e => __presses.push(e.pointerType), true); }")

            def snapshot(pid, phase):
                selections.append(page.evaluate("""([pid, phase]) => {
                    const f=document.querySelector('[id="panel-pchat-phone-a"] .chat-messages'), r=f?.getBoundingClientRect();
                    const row=f && [...f.querySelectorAll('[data-history-id]')].find(n=>n.getBoundingClientRect().bottom>r.top+1);
                    return {pid,phase,selection:getSelection().toString(),top:f?.scrollTop,height:f?.scrollHeight,
                        rect:r?.toJSON(),width:f?.offsetWidth,row:row?.dataset.historyId,offset:row && row.getBoundingClientRect().top-r.top};
                }""", [pid, phase]))

            def through_drawer(pid):
                snapshot(pid, "before-drawer")
                room = "#project-panel" if page.locator("#project-panel.open").count() else "#page-chat"
                toggle = page.locator(f"{room} [data-mobile-nav-toggle]:visible").first
                toggle.tap()
                page.wait_for_function("() => document.body.classList.contains('nav-drawer-open')")
                snapshot(pid, "drawer-open")
                page.locator(f'.nav-project-row[data-project-id="{pid}"]').tap()
                page.locator(f'[id="panel-pchat-{pid}"]:not([hidden])').wait_for()
                assert "mouse" not in page.evaluate("() => window.__presses"), "the phone path is pressed by touch only"
                page.wait_for_function("() => !document.body.classList.contains('nav-drawer-open')")
                page.wait_for_function("() => !document.querySelector('.chat-instance-panel[data-leaving]')")
                page.locator(f'[id="panel-pchat-{pid}"] [data-history-id]').first.wait_for(state="attached")
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                page.wait_for_timeout(300)
                snapshot(pid, "room-shown")

            through_drawer("phone-a")
            place = _read_older(page, "phone-a")
            snapshot("phone-a", "after-read")
            chosen = page.evaluate(_SELECT, ["phone-a", "Alpha row"])
            page.evaluate("() => { document.getElementById('panel-pchat-phone-a').__kept = 'A'; }")
            through_drawer("phone-b")
            assert page.evaluate("() => document.getElementById('panel-pchat-phone-a').hidden") is True
            page.evaluate(_SELECT, ["phone-b", "Bravo row"])
            through_drawer("phone-a")
            assert page.evaluate("() => document.getElementById('panel-pchat-phone-a').__kept") == "A"
            back = page.evaluate(_ROW_AT_TOP, "phone-a")
            assert back["id"] == place["id"] and abs(back["offset"] - place["offset"]) <= 2, (place, back)
            assert page.evaluate("() => getSelection().toString()") == chosen
        finally:
            evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent))
            evidence.mkdir(parents=True, exist_ok=True)
            (evidence / f"phone-selection-{engine}.json").write_text(json.dumps(selections))
            browser.close()


_STORAGE_REFUSES = """() => {
    // A full or blocked session store: every write this tab tries is refused, and counted.
    window.__storageRefusals = 0;
    const set = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
        if (this === window.sessionStorage) { window.__storageRefusals += 1; throw new DOMException('full', 'QuotaExceededError'); }
        return set.call(this, key, value);
    };
}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_room_keeps_its_living_card_form_and_draft_when_the_tab_cannot_store_them(direct_server_with_data, engine):
    """A retained room keeps input even if every sessionStorage write is refused."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    chat_a, _first = _seed_room(data, "store-a", "Store A", _rows("Alpha", 12))
    _seed_room(data, "store-b", "Store B", _rows("Bravo", 4))
    _seed_card(data, chat_a)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _row = _client(browser, direct_server_with_data["url"], "store-a", [], init=(_STORAGE_REFUSES,))
            _open(page, "store-a")
            room = page.locator('[id="panel-pchat-store-a"]')
            living = room.locator(".chat-card[data-card-id]")
            living.locator("[data-widget-tab-key]", has_text="Yearly").click()
            living.locator('form.widget-form [name="text"]').fill("Typed note kept without a store")
            room.locator('textarea[id$="-input"]').fill("Draft kept without a store")
            page.evaluate("() => document.activeElement?.blur()")
            assert page.evaluate("window.__storageRefusals") > 0, "a write was actually refused"
            _open(page, "store-b")
            _open(page, "store-a")
            state = _room_state(page, "store-a")
            assert (state["tab"], state["note"], state["draft"]) == (
                "Yearly", "Typed note kept without a store", "Draft kept without a store"), state
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_selection_a_click_dismissed_does_not_come_back_on_return(direct_server_with_data, engine):
    """Owner 2026-11-07: any click anywhere clears the selection, so a room left by that click gives none back."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    _seed_room(data, "sel-a", "Sel A", _rows("Alpha", 30))
    _seed_room(data, "sel-b", "Sel B", _rows("Bravo", 4))
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _row = _client(browser, direct_server_with_data["url"], "sel-a", [])
            _open(page, "sel-a")
            assert page.evaluate(_SELECT, ["sel-a", "Alpha row"]), "a passage of A is selected"
            for pid in ("sel-b", "sel-a"):  # real mouse clicks: the press that dismisses also leaves the room
                page.locator(f'.nav-project-row[data-project-id="{pid}"]').click()
                page.locator(f'[id="panel-pchat-{pid}"]:not([hidden])').wait_for(timeout=5_000)
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                page.wait_for_timeout(600)
                assert page.evaluate("() => getSelection().toString()") == "", f"no selection comes back in {pid}"
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_click_that_ends_one_rooms_selection_leaves_another_rooms_kept_selection(direct_server_with_data, engine):
    """A room left without a press (keyboard, script) keeps its selection; a click that ends another room's is not a
    press that dismissed it."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    _seed_room(data, "keep-a", "Keep A", _rows("Alpha", 30))
    _seed_room(data, "keep-b", "Keep B", _rows("Bravo", 6))
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _row = _client(browser, direct_server_with_data["url"], "keep-a", [])
            _open(page, "keep-a")
            chosen = page.evaluate(_SELECT, ["keep-a", "Alpha row 26"])
            _open(page, "keep-b")
            assert page.evaluate(_SELECT, ["keep-b", "Bravo row 01"])
            page.locator('[id="panel-pchat-keep-b"] [data-history-id] .message', has_text="Bravo row 04").first.click()
            page.wait_for_function("() => getSelection().isCollapsed || !getSelection().rangeCount")
            _open(page, "keep-a")
            page.wait_for_timeout(600)
            assert page.evaluate("() => getSelection().toString()") == chosen, "A's own selection comes back"
        finally:
            browser.close()


_SET_HIDDEN = """hidden => {
    Object.defineProperty(document, 'hidden', {configurable: true, get: () => hidden});
    Object.defineProperty(document, 'visibilityState', {configurable: true, get: () => (hidden ? 'hidden' : 'visible')});
    document.dispatchEvent(new Event('visibilitychange'));
}"""
_IN_SELECTION = """pid => {
    const rect = [...getSelection().getRangeAt(0).getClientRects()].find(r => r.width > 4 && r.height > 4);
    const x = rect.left + rect.width / 2, y = rect.top + rect.height / 2;
    return {x, y, inside: Boolean(document.elementFromPoint(x, y)?.closest(`[id="panel-pchat-${pid}"]`))};
}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_selection_ended_after_the_window_hid_does_not_come_back_on_return(direct_server_with_data, engine):
    """Hiding the window keeps a copy of the selection. Shown again, the live selection is the room's own: its end, by
    the browser's own collapse of a press inside it or by script, is not undone by that copy on a later return."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    _seed_room(data, "hide-a", "Hide A", _rows("Alpha", 30))
    _seed_room(data, "hide-b", "Hide B", _rows("Bravo", 4))
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _row = _client(browser, direct_server_with_data["url"], "hide-a", [])
            _open(page, "hide-a")
            page.evaluate(_SET_HIDDEN, True)  # the first return repaints the history: each round starts on its rows
            page.evaluate(_SET_HIDDEN, False)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.wait_for_timeout(600)

            def round_trip(end, go):
                assert page.evaluate(_SELECT, ["hide-a", "Alpha row 26"])
                page.evaluate(_SET_HIDDEN, True)
                page.evaluate(_SET_HIDDEN, False)
                end()
                page.wait_for_function("() => getSelection().isCollapsed || !getSelection().rangeCount")
                go("hide-b")
                go("hide-a")
                page.wait_for_timeout(600)
                return page.evaluate("() => getSelection().toString()")

            def press_inside():
                point = page.evaluate(_IN_SELECTION, "hide-a")
                assert point["inside"], point
                page.mouse.click(point["x"], point["y"])

            move = lambda pid: _open(page, pid)  # no press after the end: only the end itself can drop the copy
            assert round_trip(press_inside, move) == "", "a press inside the selection ended it"
            assert round_trip(lambda: page.evaluate("() => getSelection().collapseToStart()"), move) == "", \
                "a script ended it"
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_on_a_phone_a_selection_under_the_open_drawer_survives_the_window_hiding(direct_server_with_data, engine):
    """The drawer keeps the room's selection while its inert holds none (WebKit drops it there). Hiding and showing
    the window meanwhile leaves that copy alone, so closing the drawer gives the selection back."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    _seed_room(data, "cover-a", "Cover A", _rows("Alpha", 30))
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _row = _client(browser, direct_server_with_data["url"], "cover-a", [], has_touch=True)
            page.set_viewport_size({"width": 390, "height": 780})
            page.wait_for_function("() => document.documentElement.dataset.layout === 'stack'")
            _open(page, "cover-a")
            chosen = page.evaluate(_SELECT, ["cover-a", "Alpha row 26"])
            assert chosen
            page.locator("#project-panel [data-mobile-nav-toggle]:visible").first.tap()
            page.wait_for_function("() => document.body.classList.contains('nav-drawer-open')")
            covered = "() => !!document.querySelector('#panel-pchat-cover-a')?.closest('[inert]')"
            page.wait_for_function(covered, timeout=5_000)  # the open drawer holds the room under an inert layer
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            # A phone may drop the live selection under the drawer; the room's kept copy is what brings it back.
            page.evaluate("() => getSelection().removeAllRanges()")
            page.evaluate(_SET_HIDDEN, True)
            page.evaluate(_SET_HIDDEN, False)
            page.keyboard.press("Escape")
            page.wait_for_function(f"() => !({covered})()", timeout=5_000)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            try:
                page.wait_for_function("chosen => getSelection().toString() === chosen", arg=chosen, timeout=3_000)
            except Exception:  # the assertion reports what came back
                pass
            assert page.evaluate("() => getSelection().toString()") == chosen, "closing the drawer gives it back"
        finally:
            browser.close()

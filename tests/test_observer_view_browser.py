"""Observer views of the real SPA on a real server, in both engines (DESIGN "Unread").

The builtin browser opens this installation's UI as an observer (``tools/browser.py`` adds ``observer=1`` under the
existing identity proof of the fixture server's own binding) and a script asks for the same with the parameter. The
three automatic readers are crossed against what the server keeps: Main's cursor (one no client has set yet, then a
numeric one while a fresh message arrives between the builtin's calls), an unread Project room opened by a press, and
one message in two rooms (the room's root's notice to Main, through the real producer). An observer draws, lands on
and counts what it shows, records none of it and reads no history again for it while it stays unread; an intentional
write from such a page still lands, and a view without the parameter (the control) records all three. What a window
opened from an observer view inherits is measured and reported, not promised: one opened with an opener gets a copy
of the tab's session storage, a noopener one starts without it and is a human view unless its address asks. The
observer tab opening ``observer=0`` reads for the owner again: the next message is read there.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import time
import types
from collections import deque
from pathlib import Path

import pytest

from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._browser_fakes import close_generations
from tests.test_project_unread_opening_browser import _main_bus
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data  # noqa: F401
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

pytestmark = [pytest.mark.ui_browser]
QUIET_SEC = 8.0  # an observer is watched this long; the human control reads within it (its time is reported)
DRAWN = "#chat-messages [data-history-id]"
PROJECT = "observer-room"
ROW = f'.nav-project-row[data-project-id="{PROJECT}"]'
REPLY = "Observer case: the Project's own reply, unread"
NOTICE = "Observer case: the Project's notice, one message in two rooms"
FRESH = "Observer case: a fresh reply while the builtin idles"
STATE = "JSON.stringify({href: location.href, key: sessionStorage.getItem('ouroboros.observer')})"
WRITE = ("import('/static/modules/api_client.js').then((m) => m.fetchJson('/api/ui/preferences', {method: 'POST', "
         "headers: {'Content-Type': 'application/json'}, body: JSON.stringify({main_seen_revision: %d})}))"
         ".then((answer) => String(answer.main_seen_revision))")
# A quiet window's own count of history reads starts here (the page's earlier loads may have filled the buffer).
TIMING = "() => { performance.setResourceTimingBufferSize(5000); performance.clearResourceTimings(); return 'cleared'; }"
# What a view shows: the words each room drew, the open room's newest message on screen, Main's badge, the row's fact.
SHOWN = """() => {
    const feed = document.querySelector('#project-panel:not([hidden]) .chat-messages');
    const main = document.querySelector('#chat-messages')?.innerText || '', room = feed?.innerText || '';
    const notice = feed && [...feed.querySelectorAll('[data-history-id]')].find((n) => n.textContent.includes(%(notice)s));
    const box = notice?.getBoundingClientRect(), band = feed?.getBoundingClientRect();
    return JSON.stringify({fresh_in_main: main.includes(%(fresh)s), notice_in_main: main.includes(%(notice)s),
        reply_in_room: room.includes(%(reply)s), notice_on_screen: Boolean(box && box.bottom > band.top && box.top < band.bottom),
        main_badge: document.querySelector('[data-nav-page="chat"] .unread-badge')?.textContent.trim() || '',
        row_unread: Boolean(document.querySelector(%(row)s + '[data-unread]')),
        history_reads: performance.getEntriesByType('resource').filter((e) => e.name.includes('/api/chat/history')).length});
}""" % {key: json.dumps(value) for key, value in {"notice": NOTICE, "fresh": FRESH, "reply": REPLY, "row": ROW}.items()}


def _writes(page, into):
    """Every read write a page posts: Main's cursor, a Project's, or a message read whole (`shared_read`)."""
    def note(request):
        if request.method == "POST" and request.url.endswith("/api/ui/preferences"):
            body = json.loads(request.post_data or "{}")
            into.append(body.get("main_seen_revision", body.get("project_seen_revision", body.get("shared_read", "other"))))
    page.on("request", note)


def _history_reads(page, into):
    page.on("request", lambda r: into.append(r.url) if "/api/chat/history" in r.url else None)


def _drawn(page):
    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)", timeout=30_000)
    page.locator(DRAWN).first.wait_for(state="attached", timeout=30_000)


def _rooms(data, bus, monkeypatch):
    """Main's first reply; a Project room with its own reply; then its root's notice to Main through the real producer
    and delivery seam, one row counted unread in both rooms (``room_revisions``). Answers the room's chat and revision."""
    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_runtime import _send_user_message
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as delivery

    bus.send_with_budget(1, "Observer case: the first reply, unread")
    chat_id = int(create_project(data, PROJECT, name="Observer room")["chat_id"])
    bus.send_with_budget(chat_id, REPLY)
    write_task_result(data, "observer-root", "running", chat_id=chat_id, title="observer-root")  # a root born there
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    (data / "logs").mkdir(parents=True, exist_ok=True)
    events: queue.Queue = queue.Queue()
    turn = types.SimpleNamespace(current_chat_id=chat_id, task_id="observer-root", task_metadata={}, event_queue=events,
                                 pending_events=[], drive_root=data, drive_logs=lambda: data / "logs")
    assert _send_user_message(turn, NOTICE, destination="main").startswith("OK:")
    delivery._handle_send_message(events.get_nowait(), types.SimpleNamespace(
        DRIVE_ROOT=data, RUNNING={}, append_jsonl=append_jsonl, send_with_budget=bus.send_with_budget))
    return chat_id, get_project(data, PROJECT)["visible_revision"]


def _seen(text):
    """A builtin `evaluate` observation's own value (its first line; the session's conditions follow)."""
    return json.loads(text.split("\n", 1)[0])


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_observer_views_read_nothing_a_human_view_reads_and_an_intentional_write_lands(
        direct_server_with_data, monkeypatch, tmp_path, engine):
    from playwright.sync_api import sync_playwright

    data, url = direct_server_with_data["data_dir"], direct_server_with_data["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", tmp_path))
    evidence.mkdir(parents=True, exist_ok=True)
    bus = _main_bus(data, monkeypatch)
    facts: dict = {"engine": engine}

    def revision():
        return int(json.loads(urllib_get(f"{url}/api/state"))["main_visible_revision"])

    def cursor():
        return json.loads(urllib_get(f"{url}/api/ui/preferences")).get("main_seen_revision")

    def kept():
        """What the server keeps of the owner's reads: Main's cursor, the room's, and messages read whole elsewhere."""
        prefs = json.loads(urllib_get(f"{url}/api/ui/preferences"))
        return {"main": prefs.get("main_seen_revision"), "room": (prefs.get("project_seen_revision") or {}).get(PROJECT),
                "read_ahead": prefs.get("read_ahead_revisions") or {}}

    def until(page, check, seconds=30):
        """Waits inside the page's own calls, so its request events are delivered meanwhile."""
        deadline = time.monotonic() + seconds
        while not check() and time.monotonic() < deadline:
            page.wait_for_timeout(100)
        return check()

    # 1. The builtin browser on this installation's UI: an observer by the existing identity proof.
    chat_id, room_revision = _rooms(data, bus, monkeypatch)
    first = revision()
    facts["seeded"] = {"main": first, "room_chat": chat_id, "room": room_revision}
    ctx = ToolContext(repo_dir=tmp_path, drive_root=data)
    tools = ToolRegistry(tmp_path, data)
    tools.set_context(ctx)
    page = {"url": f"{url}/", "engine": engine, "wait_for": DRAWN, "state": "attached", "timeout": 30000}
    try:
        facts["builtin_opened"] = tools.execute("browse_page", page)[:300]
        tools.execute("browser_action", {"action": "evaluate", "value": TIMING})
        time.sleep(QUIET_SEC)  # the page keeps running between calls: its reads would have been sent by now
        facts["builtin_view"] = tools.execute("browser_action", {"action": "evaluate", "value": STATE})
        facts["builtin_shown"] = _seen(tools.execute("browser_action", {"action": "evaluate", "value": SHOWN}))
        facts["builtin_cursor_after_quiet"] = cursor()
        facts["builtin_kept_after_quiet"] = kept()
        written = tools.execute("browser_action", {"action": "evaluate", "value": WRITE % first})
        facts["builtin_intentional_write"] = written[:200]
        facts["cursor_after_intentional_write"] = cursor()
        # 1b. Opened again over that numeric cursor, the builtin idles between calls while a fresh message arrives.
        tools.execute("browse_page", page)
        tools.execute("browser_action", {"action": "evaluate", "value": TIMING})
        bus.send_with_budget(1, FRESH)
        fresh = revision()
        time.sleep(QUIET_SEC)
        facts["builtin_fresh"] = {"revision": fresh, **_seen(tools.execute(
            "browser_action", {"action": "evaluate", "value": SHOWN}))}
        facts["builtin_kept_after_fresh"] = kept()
        # 1c. The unread Project room, opened by the builtin's own press: drawn and landed on, not read.
        facts["builtin_room_press"] = tools.execute("browser_action", {"action": "click", "selector": ROW})[:80]
        tools.execute("browser_action", {"action": "evaluate", "value": TIMING})
        time.sleep(QUIET_SEC)
        facts["builtin_room"] = _seen(tools.execute("browser_action", {"action": "evaluate", "value": SHOWN}))
        facts["builtin_kept_after_room"] = kept()
        tools.execute("browser_action", {"action": "screenshot"})
        (evidence / f"observer-{engine}-builtin-room.png").write_bytes(
            base64.b64decode(ctx.browser_state.last_screenshot_b64))
    finally:
        close_generations(ctx.browser_state, *getattr(ctx, "_retired_browser_generations", []))

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            context = browser.new_context(viewport={"width": 1187, "height": 734})
            context.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            # 2. A script asks for the mode with the same parameter; the others and the hash stay.
            bus.send_with_budget(1, "Observer case: the second reply, unread")
            second = revision()
            script_writes: list = []
            script_reads: list = []
            observer = context.new_page()
            _writes(observer, script_writes)
            observer.goto(f"{url}/?keep=a%20b&observer=1#chat", wait_until="domcontentloaded")
            _drawn(observer)
            _history_reads(observer, script_reads)
            observer.wait_for_timeout(QUIET_SEC * 1000)
            facts["script_view"] = json.loads(observer.evaluate(STATE))
            facts["script_writes"] = list(script_writes)
            facts["script_history_reads_quiet"] = len(script_reads)
            facts["cursor_after_script"] = cursor()
            observer.reload(wait_until="domcontentloaded")  # the tab keeps the mode across a reload
            _drawn(observer)
            observer.wait_for_timeout(2000)
            facts["script_writes_after_reload"] = list(script_writes)
            # 3. A window opened from it with an opener: a copy of the tab's session storage (measured).
            with context.expect_page() as popup_info:
                observer.evaluate("window.open(location.origin + '/', '_blank')")
            popup, popup_writes = popup_info.value, []
            _writes(popup, popup_writes)
            popup.wait_for_load_state("domcontentloaded")
            _drawn(popup)
            popup.wait_for_timeout(2000)
            facts["opener_popup"] = {"view": json.loads(popup.evaluate(STATE)), "writes": list(popup_writes)}
            popup.close()
            # 4. The script presses the unread Project row: the room is drawn and landed on, nothing is recorded.
            observer.locator(ROW).click()
            observer.wait_for_timeout(QUIET_SEC * 1000)
            facts["script_room"] = {**json.loads(observer.evaluate(SHOWN)), "writes": list(script_writes)}
            facts["script_kept_after_room"] = kept()
            observer.screenshot(path=str(evidence / f"observer-{engine}-script-room.png"))
            # 5. The control: a view without the parameter reads Main's newest message, the message in two rooms
            # as Main shows it whole, then the room it presses.
            human_writes: list = []
            human = context.new_page()
            _writes(human, human_writes)
            started = time.monotonic()
            human.goto(f"{url}/", wait_until="domcontentloaded")
            _drawn(human)
            deadline = started + 30
            while second not in human_writes and time.monotonic() < deadline:
                human.wait_for_timeout(100)
            facts["human_read_after_sec"] = round(time.monotonic() - started, 2) if second in human_writes else None
            facts["cursor_after_human"] = cursor()
            until(human, lambda: str(chat_id) in kept()["read_ahead"])
            facts["human_kept_after_main"] = kept()
            human.locator(ROW).click()
            until(human, lambda: kept()["room"] == room_revision and {PROJECT: room_revision} in human_writes)
            facts["human_kept_after_room"] = kept()
            facts["human_writes"] = list(human_writes)
            human.wait_for_timeout(1000)  # the room's panel has finished arriving in the picture
            human.screenshot(path=str(evidence / f"observer-{engine}-human-room.png"))
            human.close()
            # 6. A noopener window opened from the observer starts without its storage: a human view (disclosed).
            bus.send_with_budget(1, "Observer case: the third reply, unread")
            third = revision()
            observer.wait_for_timeout(1500)
            with context.expect_page() as noopener_info:
                observer.evaluate("window.open(location.origin + '/', '_blank', 'noopener')")
            loose, loose_writes = noopener_info.value, []
            _writes(loose, loose_writes)
            loose.wait_for_load_state("domcontentloaded")
            _drawn(loose)
            deadline = time.monotonic() + 20
            while third not in loose_writes and time.monotonic() < deadline:
                loose.wait_for_timeout(100)
            facts["noopener_popup"] = {"view": json.loads(loose.evaluate(STATE)), "writes": list(loose_writes)}
            facts["observer_writes_at_end"] = list(script_writes)
            loose.close()
            # 7. The observer tab opens observer=0: it reads for the owner again, its other parameters and hash kept.
            observer.goto(f"{url}/?keep=a%20b&observer=0#chat", wait_until="domcontentloaded")
            _drawn(observer)
            facts["exit_view"] = json.loads(observer.evaluate(STATE))
            exited = len(script_writes)
            bus.send_with_budget(1, "Observer case: the fourth reply, read after the exit")
            fourth = revision()
            deadline = time.monotonic() + 30
            while fourth not in script_writes[exited:] and time.monotonic() < deadline:
                observer.wait_for_timeout(100)
            facts["exit"] = {"revision": fourth, "writes": script_writes[exited:], "cursor": cursor()}
            observer.screenshot(path=str(evidence / f"observer-{engine}-after-exit.png"))
        finally:
            browser.close()
    (evidence / f"observer-facts-{engine}.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print("OBSERVER_FACTS " + json.dumps(facts))

    nothing = {"main": None, "room": None, "read_ahead": {}}
    # The builtin: drew Main (the message in two rooms whole), set no first cursor and recorded no read.
    assert facts["builtin_cursor_after_quiet"] is None and facts["builtin_kept_after_quiet"] == nothing, facts
    assert facts["builtin_shown"]["notice_in_main"], facts
    # An unread view that records nothing asks for no history again: a quiet window reads none, a fresh arrival its own.
    assert facts["builtin_shown"]["history_reads"] == facts["builtin_room"]["history_reads"] == 0, facts
    assert facts["script_history_reads_quiet"] == 0 and facts["builtin_fresh"]["history_reads"] <= 2, facts
    assert '"key":"1"' in facts["builtin_view"] and "observer=" not in facts["builtin_view"], facts
    assert facts["cursor_after_intentional_write"] == first and str(first) in facts["builtin_intentional_write"], facts
    # Over a numeric cursor the fresh message is drawn between the builtin's calls and stays unread.
    assert facts["builtin_fresh"]["fresh_in_main"] and facts["builtin_fresh"]["revision"] > first, facts
    assert facts["builtin_kept_after_fresh"] == {**nothing, "main": first}, facts
    # The room it pressed is drawn, landed on its newest message and still unread; Main counts its fresh message.
    room = facts["builtin_room"]
    assert room["reply_in_room"] and room["notice_on_screen"] and room["row_unread"], facts
    assert room["main_badge"] == "1" and facts["builtin_kept_after_room"] == {**nothing, "main": first}, facts
    # The script: nothing in Main, after a reload, from an opener window, or in the room it pressed.
    assert facts["script_writes"] == [] and facts["script_writes_after_reload"] == [], facts
    assert facts["script_view"] == {"href": f"{url}/?keep=a%20b#chat", "key": "1"}, facts
    assert facts["cursor_after_script"] == first, facts
    assert facts["script_room"]["reply_in_room"] and facts["script_room"]["notice_on_screen"], facts
    assert facts["script_room"]["row_unread"] and facts["script_room"]["writes"] == [], facts
    assert facts["script_room"]["main_badge"] == "2", facts  # Main's fresh and second replies, still counted
    assert facts["script_kept_after_room"] == {**nothing, "main": first}, facts
    # The control records each: Main's newest message, the message in two rooms for the room, the room it pressed.
    assert facts["human_read_after_sec"] is not None and facts["human_read_after_sec"] < QUIET_SEC, facts
    assert facts["cursor_after_human"] == second, facts
    assert facts["human_kept_after_main"]["read_ahead"].get(str(chat_id)) == [[room_revision, room_revision]], facts
    assert facts["human_kept_after_room"]["room"] == room_revision, facts
    assert {PROJECT: room_revision} in facts["human_writes"] and second in facts["human_writes"], facts
    assert any(isinstance(write, list) for write in facts["human_writes"]), facts  # its `shared_read`
    assert facts["noopener_popup"]["view"]["key"] is None and third in facts["noopener_popup"]["writes"], facts
    assert facts["observer_writes_at_end"] == [], facts
    # observer=0 ends the tab's mode: the address keeps its other bytes, and the next message is read there.
    assert facts["exit_view"] == {"href": f"{url}/?keep=a%20b#chat", "key": None}, facts
    assert facts["exit"]["revision"] in facts["exit"]["writes"] and facts["exit"]["cursor"] == facts["exit"]["revision"], facts


def urllib_get(address: str) -> str:
    import urllib.request

    with urllib.request.urlopen(address, timeout=10) as answer:  # noqa: S310 - the fixture's loopback server
        return answer.read().decode("utf-8")

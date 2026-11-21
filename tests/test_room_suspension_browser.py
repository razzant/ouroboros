"""Main suspension and measured retention of twenty distinct stateful rooms, on real browser consumers."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

from ouroboros import dialogue_store
from tests._dialogue import advance_room_revision
from tests._room_browser_shared import (
    _ROW_AT_TOP,
    _SELECT,
    _client,
    _nav,
    _open,
    _read_older,
    _rows,
    _seed_room,
    _wait_for,
)
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _SETTLE_RESTORE_FRAMES, _emit_ws_frame

pytestmark = [pytest.mark.ui_browser]


def _evidence(data):
    """Screenshots and readouts: the run's OUROBOROS_UI_EVIDENCE_DIR, else beside the test data."""
    path = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or data.parent)
    path.mkdir(parents=True, exist_ok=True)
    return path


_HANDLES = """() => {
    const intervals = new Map(), rafs = new Set(), observers = new Map();
    const si = setInterval, ci = clearInterval, raf = requestAnimationFrame, caf = cancelAnimationFrame;
    window.setInterval = (fn, ms, ...args) => {const id=si(fn, ms, ...args); intervals.set(id, {ms, name:fn.name}); return id;};
    window.clearInterval = id => {intervals.delete(id); ci(id);};
    window.requestAnimationFrame = fn => {const id=raf(t => {rafs.delete(id); fn(t);}); rafs.add(id); return id;};
    window.cancelAnimationFrame = id => {rafs.delete(id); caf(id);};
    for (const name of ['ResizeObserver', 'IntersectionObserver', 'MutationObserver']) {
        const Native = window[name];
        window[name] = class extends Native {
            observe(target, options) {if (!observers.has(this)) observers.set(this, {name, targets:new Set()});
                observers.get(this).targets.add(target); return super.observe(target, options);}
            unobserve(target) {const entry=observers.get(this); entry?.targets.delete(target);
                if (!entry?.targets.size) observers.delete(this); return super.unobserve(target);}
            disconnect() {observers.delete(this); return super.disconnect();}
        };
    }
    window.roomMeasures = () => ({
        panels: [...document.querySelectorAll('.chat-instance-panel')].map(p => ({id:p.id, hidden:p.hidden, nodes:p.querySelectorAll('*').length})),
        dom: document.querySelectorAll('*').length,
        subscriptions: Object.fromEntries(Object.entries(window.__ouroWs.listeners).map(([name,set])=>[name,set.size])),
        intervals: [...intervals.values()].map(v=>v.ms).sort((a,b)=>a-b), intervalOwners:[...intervals.values()], rafs:rafs.size,
        roomObservers: [...observers.values()].flatMap(o => [...o.targets].map(t => ({kind:o.name, room:t.closest?.('.chat-instance-panel')?.id,
            target:t.tagName+'.'+String(t.className), connected:t.isConnected}))).filter(o=>o.room),
    });
}"""


_WATCH_MAIN = """() => {window.mainWrites=0; window.mainWatch=new MutationObserver(r=>mainWrites+=r.length);
    mainWatch.observe(document.getElementById('chat-messages'), {subtree:true,childList:true,characterData:true});}"""


def _away_traffic(page, data, tag):
    """What reaches Main while it is not on screen: work frames, another client's words, a saved reply."""
    for i in range(30):
        _emit_ws_frame(page, {"type": "log", "data": {"type": "task_heartbeat", "task_id": f"{tag}-{i}", "chat_id": 1,
                                                      "ts": f"2026-11-04T02:00:{i:02d}Z"}})
    _emit_ws_frame(page, {"type": "chat", "role": "user", "chat_id": 1, "content": f"Other client {tag}",
                          "sender_session_id": "other"})
    dialogue_store.append(data, "chat", {"chat_id": 1, "direction": "out", "text": f"Main news {tag}",
                                         "ts": "2026-11-04T02:01:00Z"})
    advance_room_revision(data, chat_id=1)
    _emit_ws_frame(page, {"type": "projects_changed"})
    page.wait_for_timeout(700)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_main_stays_live_beside_a_docked_project_pauses_behind_pages_and_restores_its_own_selection(
        direct_server_with_data, engine):
    """Main is live while any of it shows: docked beside a Project it keeps drawing; behind another page it is away."""
    from playwright.sync_api import expect, sync_playwright

    data = direct_server_with_data["data_dir"]
    for row in _rows("Main", 60):
        dialogue_store.append(data, "chat", {**row, "chat_id": 1})
    advance_room_revision(data, chat_id=1)
    _seed_room(data, "main-cover", "Main cover", _rows("Project", 20))
    acks = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _ = _client(browser, direct_server_with_data["url"], "main-cover", acks, init=(_HANDLES,))
            assert page.evaluate("document.documentElement.dataset.layout") == "docked"
            page.locator("#chat-messages[data-history-hydrated=true]").wait_for()
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            main = page.locator("#chat-messages")
            box = main.bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.wheel(0, -1400); page.wait_for_timeout(450)
            page.locator("#chat-input").fill("Main draft kept")
            page.evaluate("() => document.activeElement.blur()")
            chosen = main.locator(".message").first.evaluate("""n => {const r=document.createRange();
                r.selectNodeContents(n); getSelection().removeAllRanges(); getSelection().addRange(r); return getSelection().toString();}""")
            place = main.evaluate("f => f.scrollTop")
            _open(page, "main-cover")
            # Navigation itself must preserve Main's range, before any new frame can change its feed.
            assert page.evaluate("getSelection().toString()") == chosen, "opening the docked room lost Main's selection"
            # Docked, the Project is a column over Main's right edge: Main still shows, so it stays live and draws.
            assert page.evaluate("""() => {const r=document.getElementById('chat-messages').getBoundingClientRect(),
                p=document.getElementById('project-panel').getBoundingClientRect(); return r.width > 0 && r.left < p.left;}""")
            page.evaluate(_WATCH_MAIN)
            _emit_ws_frame(page, {"type": "chat", "role": "user", "chat_id": 1, "content": "Other client beside the room",
                                  "sender_session_id": "other"})
            expect(main).to_contain_text("Other client beside the room")
            assert page.evaluate("mainWrites") > 0
            assert abs(main.evaluate("f=>f.scrollTop") - place) <= 2, "a reply arriving below keeps the passage"
            page.evaluate("mainWatch.disconnect()")
            # Opening the room beside Main took nothing from it: its selection is the page's selection still. (A
            # selection the owner then makes in the room would replace it: one page, one selection.)
            assert page.evaluate("getSelection().toString()") == chosen
            # The page owns the one state poll (app.js); no room keeps a timer of its own.
            owners = page.evaluate("roomMeasures().intervalOwners.map(i=>i.name)")
            assert "refreshHeaderControlState" not in owners and owners.count("pollShownRoomState") == 1, owners
            _nav(page, "settings")
            page.evaluate(_WATCH_MAIN)
            _away_traffic(page, data, "while away")
            assert page.evaluate("mainWrites") == 0, "behind another page nothing of Main shows: it draws nothing"
            page.evaluate("mainWatch.disconnect()")
            assert "refreshHeaderControlState" not in page.evaluate("roomMeasures().intervalOwners.map(i=>i.name)")
            _nav(page, "chat")
            expect(main).to_contain_text("Main news while away")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            assert page.evaluate("getSelection().toString()") == chosen
            assert abs(main.evaluate("f=>f.scrollTop") - place) <= 2
            expect(page.locator("#chat-input")).to_have_value("Main draft kept")
            assert page.locator("#chat-scroll-bottom").is_visible()
            _nav(page, "settings")
            _nav(page, "chat")
            assert page.evaluate("getSelection().toString()") == chosen
            owners = page.evaluate("roomMeasures().intervalOwners.map(i=>i.name)")
            assert "refreshHeaderControlState" not in owners and owners.count("pollShownRoomState") == 1, owners
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_main_covered_by_a_project_on_a_phone_is_away_until_the_project_closes(direct_server_with_data, engine):
    """The stack layout's Project covers Main whole: Main draws nothing beneath it and catches up when it shows."""
    from playwright.sync_api import expect, sync_playwright

    data = direct_server_with_data["data_dir"]
    for row in _rows("Main", 30):
        dialogue_store.append(data, "chat", {**row, "chat_id": 1})
    advance_room_revision(data, chat_id=1)
    _seed_room(data, "phone-cover", "Phone cover", _rows("Project", 12))
    acks = []
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _ = _client(browser, direct_server_with_data["url"], "phone-cover", acks, init=(_HANDLES,))
            page.set_viewport_size({"width": 390, "height": 780})
            page.wait_for_function("() => document.documentElement.dataset.layout === 'stack'")
            page.locator("#chat-messages[data-history-hydrated=true]").wait_for()
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            main = page.locator("#chat-messages")
            page.locator("#chat-input").fill("Phone draft kept")
            page.evaluate("() => document.activeElement.blur()")
            _open(page, "phone-cover")
            page.locator("#project-panel").evaluate("p => Promise.all(p.getAnimations().map(a => a.finished.catch(() => {})))")
            # Quiet is required only where nothing of Main shows: the room covers the whole window, Main's feed included.
            covered = page.evaluate("""() => {
                const panel = document.getElementById('project-panel').getBoundingClientRect();
                const feed = document.getElementById('chat-messages').getBoundingClientRect();
                const x = Math.min(Math.max(feed.left + feed.width / 2, 1), innerWidth - 1);
                const y = Math.min(Math.max(feed.top + feed.height / 2, 1), innerHeight - 1);
                return {whole: panel.left <= 0 && panel.top <= 0 && panel.right >= innerWidth && panel.bottom >= innerHeight,
                        onTop: Boolean(document.elementFromPoint(x, y)?.closest('#project-panel'))};
            }""")
            assert covered == {"whole": True, "onTop": True}, covered
            page.evaluate(_WATCH_MAIN)
            _away_traffic(page, data, "under the room")
            assert page.evaluate("mainWrites") == 0, "a Project covering Main whole leaves Main drawing nothing"
            page.evaluate("mainWatch.disconnect()")
            page.locator("#project-panel-close").evaluate("el => el.click()")
            page.wait_for_function("() => !document.getElementById('project-panel').classList.contains('open')")
            expect(main).to_contain_text("Main news under the room")
            expect(page.locator("#chat-input")).to_have_value("Phone draft kept")
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_twenty_visited_rooms_keep_state_and_stay_quiet_away(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    ids = [f"stateful-{i:02d}" for i in range(20)]
    for pid in ids:
        _seed_room(data, pid, pid, _rows("Same content", 60))
    evidence = _evidence(data)
    samples, places, choices = [], {}, {}
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _ = _client(browser, direct_server_with_data["url"], ids[0], [], init=(_HANDLES,))
            cdp = page.context.new_cdp_session(page) if engine == "chromium" else None

            def sample(label):
                page.wait_for_timeout(180)
                result = page.evaluate("roomMeasures()")
                result["label"] = label
                if cdp:
                    cdp.send("HeapProfiler.collectGarbage")
                    result["heap"] = cdp.send("Runtime.getHeapUsage")
                    result["native_dom"] = cdp.send("Memory.getDOMCounters")
                else:
                    result["heap"] = {"unavailable": "WebKit Playwright has no comparable heap measurement"}
                samples.append(result)

            sample("baseline")
            for index, pid in enumerate(ids):
                _open(page, pid)
                places[pid] = _read_older(page, pid)
                choices[pid] = page.evaluate(_SELECT, [pid, "Same content row"])
                page.locator(f'[id="panel-pchat-{pid}"]').evaluate("(p,id)=>p.dataset.retainedIdentity=id", pid)
                _nav(page, "settings")
                if index in (0, 4, 9, 19):
                    sample(f"retained-{index + 1}")
            assert len(samples[-1]["panels"]) == 20
            assert samples[-1]["intervals"] == samples[1]["intervals"], "no per-away-room polling timer"
            assert samples[-1]["subscriptions"].get("log") == samples[1]["subscriptions"].get("log"), "no hidden heavy socket listener growth"
            assert samples[-1]["subscriptions"].get("outbound_sent", 0) - samples[1]["subscriptions"].get("outbound_sent", 0) == 19
            for pid in ids:
                _open(page, pid)
                assert page.locator(f'[id="panel-pchat-{pid}"]').get_attribute("data-retained-identity") == pid
                assert page.evaluate("getSelection().toString()") == choices[pid]
                back = page.evaluate(_ROW_AT_TOP, pid)
                assert back["id"] == places[pid]["id"] and abs(back["offset"] - places[pid]["offset"]) <= 2
            page.screenshot(path=str(evidence / f"twenty-stateful-{engine}.png"))
            _nav(page, "settings")
            sample("retained-after-all-returns")
            # Every visited room stays (parent decision 2026-11-04), also once the owner left it following its newest
            # message with nothing selected: memory grows with the rooms visited, and away they stay quiet.
            for pid in ids:
                _open(page, pid)
                page.evaluate("() => getSelection().removeAllRanges()")
                page.locator(f'[id="panel-pchat-{pid}"] .chat-scroll-bottom-btn').click()
                page.wait_for_function("pid => { const f=document.querySelector(`[id=\"panel-pchat-${pid}\"] .chat-messages`); return f.scrollHeight-f.scrollTop-f.clientHeight<8;}", arg=pid)
                _nav(page, "settings")
            sample("all-following-newest")
            assert len(samples[-1]["panels"]) == 20, "no visited room is let go"
            assert samples[-1]["intervals"] == samples[1]["intervals"] and samples[-1]["rafs"] == 0, "away, no timer or frame"
            assert samples[-1]["subscriptions"].get("log") == samples[1]["subscriptions"].get("log")
            page.wait_for_timeout(1500)
            sample("settled")
            if cdp:
                with (evidence / "retained.heapsnapshot").open("w") as snapshot:
                    cdp.on("HeapProfiler.addHeapSnapshotChunk", lambda event: snapshot.write(event["chunk"]))
                    cdp.send("HeapProfiler.takeHeapSnapshot")
        finally:
            (evidence / f"stateful-retention-{engine}.json").write_text(json.dumps({"same_rows_per_room": 60, "rooms": ids, "samples": samples}, indent=2))
            browser.close()


# The host never receives a send carrying this marker, so its saved echo never comes (a frame lost on a dying socket).
_DROP_MARKED_SENDS = """() => {
    const send = WebSocket.prototype.send;
    WebSocket.prototype.send = function (data) {
        if (String(data).includes('never-reaches-the-host')) return;
        return send.call(this, data);
    };
}"""
# Counts writes into one room's feed while it is away (attributes excluded: the shell hides and shows the room).
_WATCH_AWAY = """pid => { window.awayWrites = 0; window.awayWatch = new MutationObserver(r => awayWrites += r.length);
    awayWatch.observe(document.querySelector(`[id="panel-pchat-${pid}"] .chat-messages`),
        {subtree: true, childList: true, characterData: true}); }"""
_LATE_REVIEW = {"task_id": "late-review", "status": "running", "review_projection": {"panels": [], "acceptance_incident": {
    "incident_id": "acceptance-preparation:late", "status": "failed", "stage": "preparation", "attempts": 1,
    "source_known": True, "failure_kind": "UnicodeDecodeError", "failure_detail": "bad byte", "feedback_delivered": True}}}


def custody_scenario(page, pid):
    """A send the host never saved; its room away while the socket closes and comes back; then the room again."""
    _open(page, pid)
    room = page.locator(f'[id="panel-pchat-{pid}"]')
    room.locator("textarea").fill("Send never-reaches-the-host")
    room.locator(f'[id="pchat-{pid}-send"]').click()
    bubble = room.locator(".chat-bubble.user", has_text="never-reaches-the-host")
    bubble.wait_for()
    before = bubble.locator("[data-unconfirmed-action]").count()
    _nav(page, "settings")
    page.evaluate(_WATCH_AWAY, pid)
    sockets = page.evaluate("() => window.__testSockets.length")
    page.evaluate("() => window.__testSockets.find(s => s.readyState === 1).close()")
    page.wait_for_function("n => window.__testSockets.length > n && window.__testSockets.at(-1).readyState === 1",
                           arg=sockets, timeout=30_000)
    page.wait_for_timeout(600)
    away = page.evaluate("() => { awayWatch.disconnect(); return awayWrites; }")
    _open(page, pid)
    page.wait_for_timeout(1500)  # the return's catch-up read has landed
    return {"doubt_before": before, "away_writes": away,
            "actions_after_return": bubble.locator("[data-unconfirmed-action]").all_inner_texts()}


def late_review_scenario(page, pid, chat_id):
    """A card's review-detail read still in flight when its room leaves, answered while away; then the room again."""
    held = []
    page.route("**/api/tasks/late-review*", lambda route: held.append(route))
    _open(page, pid)
    for frame in ({"content": "assembling", "ts": "2026-11-04T09:00:00Z"},
                  {"system_type": "review_reference", "surface": "task_acceptance", "presentation_owner_task_id": "late-review",
                   "state_revision": "a" * 64, "content": "", "ts": "2026-11-04T09:00:01Z"}):
        _emit_ws_frame(page, {"type": "chat", "role": "system", "is_progress": True, "chat_id": chat_id, "task_id": "late-review", **frame})
    host = page.locator(f'[id="panel-pchat-{pid}"] .chat-live-card[data-task-id="late-review"] [data-live-reviews-host]')
    host.wait_for(state="attached")
    _wait_for(page, lambda: held, attempts=100)
    requested = len(held)
    _nav(page, "settings")
    page.evaluate(_WATCH_AWAY, pid)
    for route in held[:]:
        route.fulfill(json=_LATE_REVIEW)
    held.clear()
    page.unroute("**/api/tasks/late-review*")
    page.wait_for_timeout(1000)
    away = page.evaluate("() => { awayWatch.disconnect(); return awayWrites; }")
    away_text = host.evaluate("h => h.textContent")
    _open(page, pid)
    page.wait_for_timeout(1000)
    return {"detail_reads_held": requested, "away_writes": away, "drawn_while_away": "acceptance evidence" in away_text,
            "drawn_after_return": "acceptance evidence" in host.evaluate("h => h.textContent")}


def state_poll_scenario(page, pid):
    """/api/state reads in 7 s with a Project on screen, then in 7 s on Settings; the interval owners."""
    reads = []
    page.on("request", lambda request: reads.append(request.url) if request.url.split("?")[0].endswith("/api/state") else None)
    _open(page, pid)
    page.wait_for_timeout(1200)
    before = len(reads)
    page.wait_for_timeout(7000)
    on_screen = len(reads) - before
    owners = page.evaluate("roomMeasures().intervalOwners.map(i=>i.name)")
    _nav(page, "settings")
    page.wait_for_timeout(1200)
    before = len(reads)
    page.wait_for_timeout(7000)
    return {"project_on_screen_reads_in_7s": on_screen, "settings_reads_in_7s": len(reads) - before, "interval_owners": owners}


def _record(name, engine, facts, data):
    (_evidence(data) / f"{name}-{engine}.json").write_text(json.dumps(facts, indent=2))


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_send_a_disconnect_left_unconfirmed_while_its_room_was_away_offers_send_again_on_return(direct_server_with_data, engine):
    """R1: a socket close while the room is away is still the send's custody; nothing is drawn into the hidden room."""
    from playwright.sync_api import sync_playwright

    _seed_room(direct_server_with_data["data_dir"], "custody-room", "Custody room", _rows("Custody", 12))
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _ = _client(browser, direct_server_with_data["url"], "custody-room", [], init=(_DROP_MARKED_SENDS,))
            facts = custody_scenario(page, "custody-room")
            _record("custody-away", engine, facts, direct_server_with_data["data_dir"])
            page.screenshot(path=str(_evidence(direct_server_with_data["data_dir"]) / f"send-again-after-away-{engine}.png"))
            assert facts["doubt_before"] == 0, "no doubt before a fact"
            assert facts["away_writes"] == 0, "the room away is not drawn"
            assert facts["actions_after_return"] == ["Send again", "Discard"], facts
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_review_detail_answered_while_its_room_is_away_is_drawn_on_return_not_while_hidden(direct_server_with_data, engine):
    """R1: a card's delayed review-detail read lands while its room is away; it is held, then drawn on the return."""
    from playwright.sync_api import sync_playwright

    chat_id, _revision = _seed_room(direct_server_with_data["data_dir"], "review-room", "Review room", _rows("Review", 12))
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _ = _client(browser, direct_server_with_data["url"], "review-room", [])
            facts = late_review_scenario(page, "review-room", chat_id)
            _record("late-review", engine, facts, direct_server_with_data["data_dir"])
            page.screenshot(path=str(_evidence(direct_server_with_data["data_dir"]) / f"late-review-return-{engine}.png"))
            assert facts["detail_reads_held"] >= 1, facts
            assert facts["away_writes"] == 0 and not facts["drawn_while_away"], facts
            assert facts["drawn_after_return"], facts
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_project_on_screen_keeps_the_three_second_state_read_and_another_page_drops_it(direct_server_with_data, engine):
    """R1: the page's /api/state poll keeps the parent's 3 s while a room (a Project included) is on screen."""
    from playwright.sync_api import sync_playwright

    _seed_room(direct_server_with_data["data_dir"], "fresh-room", "Fresh room", _rows("Fresh", 8))
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _ = _client(browser, direct_server_with_data["url"], "fresh-room", [], init=(_HANDLES,))
            facts = state_poll_scenario(page, "fresh-room")
            _record("state-poll", engine, facts, direct_server_with_data["data_dir"])
            assert facts["project_on_screen_reads_in_7s"] >= 2, facts
            owners = facts["interval_owners"]
            assert owners.count("pollShownRoomState") == 1 and "refreshHeaderControlState" not in owners, owners
            assert facts["settings_reads_in_7s"] <= 1, facts
        finally:
            browser.close()


def _cover_main(page, data, how):
    """Main away: behind Settings, or under a Project covering it whole on a phone."""
    if how == "settings":
        _nav(page, "settings")
        return lambda: _nav(page, "chat")
    page.set_viewport_size({"width": 390, "height": 780})
    page.wait_for_function("() => document.documentElement.dataset.layout === 'stack'")
    _open(page, "cover")
    page.locator("#project-panel").evaluate("p => Promise.all(p.getAnimations().map(a => a.finished.catch(() => {})))")

    def back():
        page.locator("#project-panel-close").evaluate("el => el.click()")
        page.wait_for_function("() => !document.getElementById('project-panel').classList.contains('open')")
    return back


def _incident(task_id, key, text="⏳ Task {task}: reached its deadline. Finalize artifacts/results now."):
    """The reaper's grace warning as the supervisor sends it (supervisor/task_reaper.py): a progress frame of Main's task."""
    return {"type": "chat", "role": "system", "system_type": "finalization_notice", "is_progress": True, "chat_id": 1,
            "task_id": task_id, "content": text.format(task=task_id), "task_incident": "deadline",
            "toast_tone": "warning", "toast_once": key, "ts": "2026-11-06T03:00:00Z"}


@pytest.mark.parametrize("how", ["settings", "phone"])
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_main_task_incident_toasts_while_main_is_away_and_draws_nothing_into_it(direct_server_with_data, engine, how):
    """Away, Main draws nothing, but its task's incident still toasts once page-wide (DESIGN §4: incidents raised in a
    task's progress still toast); an ordinary progress frame neither toasts nor draws, and Main reads it on return."""
    from playwright.sync_api import expect, sync_playwright

    data = direct_server_with_data["data_dir"]
    for row in _rows("Main", 20):
        dialogue_store.append(data, "chat", {**row, "chat_id": 1})
    advance_room_revision(data, chat_id=1)
    _seed_room(data, "cover", "Cover", _rows("Project", 8))
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _ = _client(browser, direct_server_with_data["url"], "cover", [])
            page.locator("#chat-messages[data-history-hydrated=true]").wait_for()
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            toasts = page.locator("#toast-stack .toast")
            back = _cover_main(page, data, how)
            before = toasts.count()
            page.evaluate(_WATCH_MAIN)
            _emit_ws_frame(page, _incident("inc-away", f"inc-away:{how}"))
            expect(toasts).to_have_count(before + 1)
            expect(toasts.last).to_contain_text("Finalize artifacts/results now")
            _emit_ws_frame(page, _incident("inc-away", f"inc-away:{how}"))  # the same incident again: one toast
            _emit_ws_frame(page, {"type": "chat", "role": "assistant", "is_progress": True, "chat_id": 1,
                                  "task_id": "plain-away", "content": "Reading the plan", "narration": True,
                                  "ts": "2026-11-06T03:00:01Z"})
            page.wait_for_timeout(400)
            assert toasts.count() == before + 1, "one incident toasts once; plain progress never toasts"
            assert page.evaluate("mainWrites") == 0, "away, Main draws neither the incident nor the progress"
            page.evaluate("mainWatch.disconnect()")
            evidence = _evidence(data)
            page.screenshot(path=str(evidence / f"incident-away-{how}-{engine}.png"))
            back()
            page.locator("#chat-messages").wait_for()
            # Shown again, Main is live: a new incident of its task toasts as before and the room is drawn.
            _emit_ws_frame(page, _incident("inc-back", f"inc-back:{how}"))
            expect(toasts).to_have_count(before + 2)
        finally:
            browser.close()


def _main_cards(page, phase):
    return page.evaluate("""phase => [...document.querySelectorAll('#chat-messages .chat-live-card')]
        .map(card => card.dataset.taskId).filter(id => id.endsWith('-' + phase)).sort()""", phase)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_main_work_that_ran_while_covered_is_restored_from_saved_history_on_return(direct_server_with_data, engine):
    """Covered by a phone Project, Main hears no work frames. Its return reads the server's saved history: a task whose
    progress the supervisor saved has its card again, exactly as on a fresh page load. A task heard only through live
    log frames (never saved) has none until its next frame, also as on a fresh load; Main left live draws both."""
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    for row in _rows("Main", 20):
        dialogue_store.append(data, "chat", {**row, "chat_id": 1})
    advance_room_revision(data, chat_id=1)
    _seed_room(data, "cover", "Cover", _rows("Project", 8))

    def work(page, phase):
        """One saved-progress task and one log-only task, as the supervisor stores and relays them."""
        ts, saved = "2026-11-06T03:01:00Z", f"saved-{phase}"
        dialogue_store.append(data, "progress", {
            "ts": ts, "type": "send_message", "task_id": saved, "is_progress": True, "direction": "out", "chat_id": 1,
            "user_id": 0, "text": "💬 Checking the ledger", "content": "Checking the ledger", "format": "",
            "role": "assistant", "system_type": "", "narration": True})  # supervisor/message_bus.py, then the relay
        _emit_ws_frame(page, {"type": "chat", "role": "assistant", "is_progress": True, "chat_id": 1, "task_id": saved,
                              "content": "Checking the ledger", "narration": True, "ts": ts})
        for i, (kind, extra) in enumerate((("task_checkpoint", {"checkpoint": "started"}),
                                           ("tool_call_started", {"tool": "repo_read"}))):
            _emit_ws_frame(page, {"type": "log", "data": {"type": kind, "task_id": f"logged-{phase}", "chat_id": 1,
                                                          "ts": f"2026-11-06T03:01:0{i + 1}Z", **extra}})
        page.wait_for_timeout(500)

    def hydrated(page):
        page.locator("#chat-messages[data-history-hydrated=true]").wait_for()
        page.evaluate(_SETTLE_RESTORE_FRAMES)

    readouts = {}
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            # Negative control: Main on screen draws both cards from their frames.
            page, _ = _client(browser, direct_server_with_data["url"], "cover", [])
            hydrated(page)
            work(page, "live")
            readouts["live"] = _main_cards(page, "live")
            page.close()
            # Covered on a phone: nothing is drawn; then the return's catch-up read.
            page, _ = _client(browser, direct_server_with_data["url"], "cover", [])
            hydrated(page)
            readouts["fresh_after_live"] = _main_cards(page, "live")
            back = _cover_main(page, data, "phone")
            page.evaluate(_WATCH_MAIN)
            work(page, "covered")
            readouts["covered"] = _main_cards(page, "covered")
            readouts["covered_writes"] = page.evaluate("mainWrites")
            page.evaluate("mainWatch.disconnect()")
            back()
            _wait_for(page, lambda: "saved-covered" in _main_cards(page, "covered"))
            page.wait_for_timeout(800)
            readouts["returned"] = _main_cards(page, "covered")
            # A fresh page load reads the same saved sources.
            page.reload(wait_until="domcontentloaded")
            hydrated(page)
            page.wait_for_timeout(800)
            readouts["fresh_load"] = _main_cards(page, "covered")
        finally:
            evidence = _evidence(data)
            (evidence / f"main-cards-while-covered-{engine}.json").write_text(json.dumps(readouts, indent=2))
            browser.close()
    assert readouts["live"] == ["logged-live", "saved-live"], readouts
    assert readouts["fresh_after_live"] == ["saved-live"], ("a fresh load has no source for a log-only task", readouts)
    assert readouts["covered"] == [] and readouts["covered_writes"] == 0, readouts
    assert readouts["returned"] == readouts["fresh_load"] == ["saved-covered"], readouts


# A hidden window as the page sees it (the tab or app in the background, a locked phone): `document.hidden` and the
# event, without the engine's own throttling, which is not what is tested here.
_HIDE = """hidden => {
    for (const [name, value] of [['hidden', hidden], ['visibilityState', hidden ? 'hidden' : 'visible']]) {
        if (hidden) Object.defineProperty(document, name, { configurable: true, get: () => value });
        else delete document[name];
    }
    document.dispatchEvent(new Event('visibilitychange'));
}"""


def _clip(page, feed, chat_id, tag):
    """A clip delivered live into a room (a 'video' frame, chat_media.js) and played through its own Play button."""
    from tests.test_chat_attachments_browser import _VP8_CLIP

    _emit_ws_frame(page, {"type": "video", "chat_id": chat_id, "mime": "video/webm", "caption": f"Clip {tag}",
                          "video_base64": base64.b64encode(_VP8_CLIP).decode("ascii"), "ts": "2026-11-06T04:00:00Z"})
    bubble = page.locator(f"{feed} .chat-bubble", has_text=f"Clip {tag}")
    video = bubble.locator("video")
    video.wait_for(state="attached")
    video.evaluate("v => { v.loop = true; }")  # three seconds long: it keeps playing for the whole scenario
    bubble.locator('[data-media-action="play"]').click()
    page.wait_for_function("v => !v.paused && v.currentTime > 0.05", arg=video.element_handle(), timeout=10_000)
    return video


def _plays(page, video):
    """Still playing: not paused, and its time moved over a few hundred milliseconds."""
    before = video.evaluate("v => v.currentTime")
    page.wait_for_timeout(450)
    return video.evaluate("(v, before) => !v.paused && v.currentTime !== before", before)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_main_plays_on_behind_pages_and_a_hidden_window_while_a_replaced_room_pauses(direct_server_with_data, engine):
    """Suspension stops drawing and reading, not what the owner listens to (Opus review, 6 Nov): Main's clip plays on
    behind Settings and in a hidden window, whose reader also stays open; another page closes that reader. A Project
    room another place replaces pauses its own clip, as when such rooms were destroyed."""
    from playwright.sync_api import expect, sync_playwright
    from tests.test_chat_attachments_browser import _png

    data = direct_server_with_data["data_dir"]
    for row in _rows("Main", 12):
        dialogue_store.append(data, "chat", {**row, "chat_id": 1})
    advance_room_revision(data, chat_id=1)
    chat_id, _ = _seed_room(data, "media-room", "Media room", _rows("Project", 6))
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            page, _ = _client(browser, direct_server_with_data["url"], "media-room", [])
            page.locator("#chat-messages[data-history-hydrated=true]").wait_for()
            main_clip = _clip(page, "#chat-messages", 1, "main")
            _nav(page, "settings")
            assert _plays(page, main_clip), "behind another page Main draws nothing, but its clip plays on"
            _nav(page, "chat")
            _emit_ws_frame(page, {"type": "photo", "chat_id": 1, "mime": "image/png", "caption": "Photo reader",
                                  "image_base64": base64.b64encode(_png(48, 32, (200, 120, 60))).decode("ascii"),
                                  "ts": "2026-11-06T04:00:05Z"})
            page.locator("#chat-messages .chat-bubble", has_text="Photo reader").locator(".chat-photo-open").click()
            reader = page.locator("dialog.document-reader[open]")
            reader.wait_for(state="visible")
            page.evaluate(_HIDE, True)
            page.wait_for_timeout(300)
            assert reader.count() == 1, "a hidden window closes nothing the owner opened"
            assert _plays(page, main_clip), "and pauses nothing"
            page.evaluate(_HIDE, False)
            _nav(page, "settings")
            expect(reader).to_have_count(0)  # another page covers the room: the reader over it closes
            assert _plays(page, main_clip)
            _nav(page, "chat")
            _open(page, "media-room")
            room_clip = _clip(page, '[id="panel-pchat-media-room"]', chat_id, "room")
            assert _plays(page, main_clip), "docked beside the room, Main plays on"
            page.locator("#project-panel-close").evaluate("el => el.click()")
            page.wait_for_function("v => v.paused", arg=room_clip.element_handle(), timeout=5_000)
            assert _plays(page, main_clip), "the room left pauses only its own clip"
            page.screenshot(path=str(_evidence(data) / f"playback-across-suspension-{engine}.png"))
        finally:
            browser.close()

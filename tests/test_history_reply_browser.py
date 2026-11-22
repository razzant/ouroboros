"""Quote navigation in the real app over real dialogue/history/project endpoints.

The fixture is also the reusable inspection stand's seed. Transport holds/failures
delay real responses; no successful lookup or history response is fabricated.

Search in a chat lands on its matches through the same quote landing (the scenes at the end).
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

import pytest

from ouroboros import dialogue_store as ds
from ouroboros.chat_chain import format_address, row_address
from ouroboros.chat_references import reference_from_row
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.projects_registry import bind_task_to_project, create_project
from tests import _room_browser_shared as room_browser
from tests._shared import producer_evidence_dir
from tests.test_chat_history_paging_browser import _FRAMES, _idle, _open, _result, _step
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _SETTLE_RESTORE_FRAMES, _emit_ws_frame

pytestmark = [pytest.mark.ui_browser]


def seed_reply_history(root, *, outside_sidebar=False):
    """S1 quotes triage's owner row, also retained as S1's origin; long history in both rooms."""
    a = create_project(root, "quote-triage", name="Triage source", origin="owner_ui")
    b = create_project(root, "quote-s1", name="S1 quoting work", origin="owner_ui")
    x = create_project(root, "quote-converted", name="Converted source", origin="owner_ui")
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    saved = {}

    def say(chat, index, text, *, stream="chat", name="", **extra):
        row = {"chat_id": chat, "ts": (start + timedelta(seconds=index)).isoformat(),
               "direction": "in", "text": text, **extra}
        ds.append(root, stream, row, feed=stream == "chat")
        if name:
            saved[name] = {"row": row, "address": format_address(row_address(row))}
        return row

    for index in range(40):
        for chat in (1, a["chat_id"]):
            say(chat, index, f"Earlier context {index:03}")
    main = say(1, 40, "Exact old Main source", name="main", client_message_id="old-main")
    owner = say(a["chat_id"], 40, "Exact triage owner source", name="a", client_message_id="owner-a")
    say(a["chat_id"], 40, owner["text"], name="twin", client_message_id="other-same-words-time")
    say(1, 41, "A task's first row", task_id="converted-source", direction="out")
    moved = say(1, 42, "Exact later row of converted work", name="converted", task_id="converted-source", direction="out")
    bind_task_to_project(root, "s1-task", b["id"], origin={"ref": build_owner_message_ref(
        chat_id=a["chat_id"], client_message_id="owner-a", ts=owner["ts"], text=owner["text"]), "text": owner["text"]})
    # A late narration in a collapsed child's long bounded timeline: same words elsewhere cannot land for it.
    for index in range(50):
        say(a["chat_id"], 50 + index, f"Child timeline line {index:02}: " + "retained narration " * 12,
            stream="progress", name="line" if index == 30 else "", task_id="child-source", parent_task_id="parent-source",
            root_task_id="parent-source", direction="out", narration=True, delegation_role="subagent",
            subagent_task_id="child-source", subagent_role="worker")
    for task, extra in (("s1-task", {"chat_id": b["chat_id"]}), ("converted-source", {"chat_id": 1}),
                        ("parent-source", {"chat_id": a["chat_id"]}),
                        ("child-source", {"chat_id": a["chat_id"], "parent_task_id": "parent-source", "root_task_id": "parent-source"})):
        _result(root, task, **extra)
    for index in range(100, 570):
        for chat in (1, a["chat_id"]):
            say(chat, index, f"Recent context {index:03}")
    for name, source in (("a", owner), ("converted", moved), ("line", saved["line"]["row"])):
        say(1, 600 + len(saved), f"Reply quoting {name}", direction="out", task_id=f"answer-{name}",
            reply_to=reference_from_row(source))
    say(a["chat_id"], 620, "Newest Project message remains unread", direction="out")
    say(a["chat_id"], 621, "Project answer quoting Main", direction="out", reply_to=reference_from_row(main))
    if outside_sidebar:
        for index in range(205):
            create_project(root, f"newer-than-source-{index}", origin="owner_ui")
    return {"a": a, "b": b, "x": x, "sources": saved}


_OBSERVE_QUOTES = """() => {
    const fetch = window.fetch.bind(window);
    window.__quoteReads = []; window.__quoteSends = []; window.__quoteAcks = [];
    window.__quoteToasts = [];
    document.addEventListener('DOMContentLoaded', () => new MutationObserver(() => {
        for (const toast of document.querySelectorAll('#toast-stack .toast'))
            if (!window.__quoteToasts.includes(toast.textContent)) window.__quoteToasts.push(toast.textContent);
    }).observe(document.body, {childList:true, subtree:true}));
    const send = WebSocket.prototype.send;
    WebSocket.prototype.send = function(value) { window.__quoteSends.push(value); return send.call(this, value); };
    window.fetch = async (input, init) => {
        const url = new URL(typeof input === 'string' ? input : input.url, location.href);
        if (init?.body && String(init.body).includes('project_seen_revision')) window.__quoteAcks.push(JSON.parse(init.body));
        const address = url.searchParams.get('around');
        const lookup = url.pathname === '/api/projects' && url.searchParams.has('chat_id');
        if (!address && !lookup) return fetch(input, init);
        const read = {url: url.pathname + url.search, address, lookup, done: false};
        window.__quoteReads.push(read);
        const response = await fetch(input, init);  // always the real endpoint, even the held answer
        read.status = response.status; read.body = await response.clone().json();
        if (window.__quoteHold === (lookup ? 'lookup' : address)) {
            window.__quoteHold = null; window.__quoteHeld = read;
            await new Promise(resolve => { window.__quoteRelease = resolve; });
        }
        read.done = true;
        if (address && window.__quoteFail === address) { window.__quoteFail = null; throw new TypeError('injected offline read'); }
        return response;
    };
}"""


def _quote(page, address, feed="#chat-messages"):
    return page.locator(f'{feed} .chat-quote[data-address={json.dumps(address)}] .chat-quote-main')


def _landed(page, address, feed):
    page.wait_for_function("""([address, feed]) => {
        const target = document.querySelector(`${feed} .is-quote-target`);
        return target && (target.dataset.rowAddress === address || target.dataset.historyId
            === window.__quoteReads.findLast(r => r.address === address && r.body?.anchor?.state === 'found')?.body.anchor.history_id);
    }""", arg=[address, feed], timeout=15_000)
    target = page.locator(f"{feed} .is-quote-target")
    facts = target.evaluate("""(node, selector) => {
        const feed = document.querySelector(selector), page = feed.closest('.page, .chat-instance-panel'), r = node.getBoundingClientRect();
        const bounds = [feed.getBoundingClientRect()];
        for (let p = node.parentElement; p && p !== feed; p = p.parentElement)
            if (getComputedStyle(p).overflowY.match(/auto|scroll/)) bounds.push(p.getBoundingClientRect());
        const header = page?.querySelector('.chat-page-header, .chat-panel-statusbar')?.getBoundingClientRect();
        return {address:node.dataset.rowAddress, id:node.dataset.historyId, focused:document.activeElement === node,
            visible:node.getClientRects().length > 0, top:r.top, bottom:r.bottom,
            clear:Math.max(header?.bottom || 0, ...bounds.map(b => b.top)), edge:Math.min(...bounds.map(b => b.bottom))};
    }""", feed)
    assert facts["focused"] and facts["visible"] and facts["top"] >= facts["clear"] - 2, facts
    assert min(facts["bottom"], facts["edge"]) - facts["top"] >= 12, facts
    return facts


def _arrived(page, address, feed):
    """A reveal's first landing on the exact row address, as it first stands: recorded, never judged here (its place
    is judged once the room settles, `_settled_row`). Passive: the feed's scroll and whether `Scroll to latest message`
    shows, the observable trace of the reader following the newest message."""
    page.wait_for_function("""([address, feed]) => {
        const target = document.querySelector(`${feed} .is-quote-target`);
        return target && target.dataset.rowAddress === address;
    }""", arg=[address, feed], timeout=15_000)
    return page.locator(f"{feed} .is-quote-target").evaluate("""(node, selector) => {
        const feed = document.querySelector(selector), r = node.getBoundingClientRect();
        const latest = feed.closest('.page, .chat-instance-panel')?.querySelector('.chat-scroll-bottom-btn');
        return {address: node.dataset.rowAddress, focused: document.activeElement === node, top: r.top, bottom: r.bottom,
                at: performance.now(), feedTop: feed.scrollTop, feedHeight: feed.scrollHeight, feedView: feed.clientHeight,
                latestShown: Boolean(latest?.classList.contains('visible'))};
    }""", feed)


def _evidence(page, root, name, fixture, facts):
    destination = producer_evidence_dir(root, "quoted-navigation")
    page.screenshot(path=str(destination / f"{name}.png"), animations="disabled")
    (destination / f"{name}.html").write_text(page.content())
    state = page.evaluate("() => ({reads:window.__quoteReads, history:window.__historyReads, sends:window.__quoteSends, acks:window.__quoteAcks, toasts:window.__quoteToasts})")
    (destination / f"{name}.json").write_text(json.dumps({"fixture": fixture, "facts": facts, **state}, indent=2))


def _settled_row(page, target, feed):
    """Observe the actual row beyond panel/highlight motion and pending history, without moving it.

    A first-frame focus is insufficient: WebKit's later resize used to restore the
    opening anchor at scrollTop 0 and put this same focused line behind the composer.
    """
    page.wait_for_function("() => window.__historyReads.every(r => r.done) && window.__quoteReads.every(r => r.done)")
    return target.evaluate("""(node, selector) => new Promise((resolve, reject) => {
        const feed = document.querySelector(selector), panel = feed.closest('.page, .chat-instance-panel');
        const rect = n => { const r = n.getBoundingClientRect(); return {x:r.x, top:r.top, bottom:r.bottom, width:r.width}; };
        const start = performance.now(), samples = [];
        let previous = '', changed = start;
        const sample = () => {
            const now = performance.now(), box = rect(node), clips = [];
            for (let p = node.parentElement; p; p = p.parentElement) {
                if (p === feed || /auto|scroll/.test(getComputedStyle(p).overflowY))
                    clips.push({...rect(p), id:p.id, scrollTop:p.scrollTop});
                if (p === feed) break;
            }
            const blockers = [...panel.querySelectorAll('.chat-scroll-bottom-btn.visible, .chat-input-area .chat-input-wrap')]
                .filter(p => p.getClientRects().length && getComputedStyle(p).visibility !== 'hidden')
                .map(p => ({...rect(p), role:p.className}));
            const header = panel.querySelector('.chat-page-header, .chat-panel-statusbar');
            const facts = {id:node.dataset.historyId, address:node.dataset.rowAddress, focused:document.activeElement === node,
                ...box, feedTop:feed.scrollTop, clips, blockers,
                clear:Math.max(header?.getBoundingClientRect().bottom || 0, ...clips.map(p => p.top)),
                edge:Math.min(...clips.map(p => p.bottom)),
                overlaps:blockers.filter(p => p.x < box.x+box.width && p.x+p.width > box.x && p.top < box.bottom && p.bottom > box.top)};
            const key = JSON.stringify(facts);
            if (key !== previous) { previous = key; changed = now; samples.push({elapsed:now-start, ...facts}); }
            if (now-start >= 2200 && now-changed >= 600) resolve({...facts, elapsed:now-start, quiet:now-changed, samples});
            else if (now-start > 8000) reject(new Error('Quote geometry did not settle: '+JSON.stringify(samples)));
            else requestAnimationFrame(sample);
        };
        requestAnimationFrame(sample);
    })""", feed)


def _promote_second_request(data, mp, scratch, *, gap=False):
    """Real promotion, stopping before the completion: the second of two identical requests starts the work.

    Returns (a, b, words, tid, task, queue, workers) and leaves the caller's ``mp`` patches in force, so a
    caller may open Main in front of the reader first and only then run ``_deliver_second_completion``.
    ``gap`` keeps the neutral row ``_complete_on_second_request`` always inserted; both halves stay real."""
    from types import SimpleNamespace

    from ouroboros.agent import OuroborosAgent
    from ouroboros.owner_relay import owner_row_address
    from ouroboros.queue_observation import load_queue_document, pool_observation_status
    from ouroboros.tools.control import _start_task
    from ouroboros.tools.control_events import _promotion_pool_disabled_from_snapshot
    from tests.test_continue_quiz_origin import _owner_turn

    mp.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(scratch / "projects"))
    queue, workers, turn, (first, second, _), _, _ = _owner_turn(data, mp, "same_words_bare")
    a, b = owner_row_address(first), owner_row_address(second)
    assert first["text"] == second["text"] and a != b

    assert queue.publish_queue_snapshot(reason="prepared_synthetic_pool") is True
    snapshot = load_queue_document(data)
    assert pool_observation_status(snapshot) == "observed" and snapshot["worker_total"] == 1
    assert _promotion_pool_disabled_from_snapshot(turn) == ""
    receipt = _start_task(turn, "Count the seeds", title="Seed count", project_name="Seed inventory",
                          workspace="none", owner_source=b)
    assert receipt.startswith("OK: task"), receipt
    if gap:
        # Ordinary neutral chatter between B and the completion (its ts lands after B's 13:45 and long
        # before the terminal delivery's now): the quote's target is no longer the row directly above.
        ds.append(data, "chat", {"chat_id": 1, "ts": "2024-04-08T15:00:00+00:00", "direction": "out",
                                 "text": "Routine note while the seed work runs."}, feed=True)
    [task] = workers.PENDING
    workers.PENDING.remove(task)
    workers.RUNNING.pop(turn.task_id)
    OuroborosAgent._persist_running_record(SimpleNamespace(env=SimpleNamespace(drive_root=data)), task)
    return a, b, second["text"], task["id"], task, queue, workers


def _deliver_second_completion(data, mp, b, tid, task, queue, workers):
    """Real terminal mirroring and delivery of the promoted work; only model and transport are fixtures.

    The returned frame is the producer's own unmodified bridge broadcast — the payload a running server
    hands its ws broadcaster (test_body_adoption_card_browser feeds such captured frames to an open page,
    filtering them by ``type == "chat"``) — so a caller may dispatch it into an open page's socket as the
    delivery's live arrival. The dispatch, the delivery and the quote stay the real machinery's."""
    import time
    from collections import deque
    from types import SimpleNamespace

    from ouroboros import server_liveness
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.terminal_projection import SETTLEMENT_DEFERRED, main_delivery_recorded, settle_terminal_projection
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery, message_bus
    from supervisor.event_observation import ObservedEventQueue
    from tests.test_dialogue_retry import bridge

    answer = "The seed table is ready to read."
    stored = write_task_result(data, tid, STATUS_COMPLETED, result=answer, terminal_origin="model_final")
    done = {"chat_id": 1, "status": "completed", "outcome_axes": {"execution": {"status": "ok"}}}
    frames, bus = bridge(data, mp), ObservedEventQueue()
    for name, empty in (("_DELIVERED_MESSAGE_IDS", deque(maxlen=256)), ("_ACTIVE_DELIVERIES", set()),
                        ("_DEFERRED_DELIVERIES", set())):
        mp.setattr(events_chat_delivery, name, empty)
    mp.setattr(workers, "get_event_q", lambda **_kw: bus)
    ctx = SimpleNamespace(DRIVE_ROOT=data, RUNNING={}, EVENT_Q=bus, append_jsonl=append_jsonl,
                          send_with_budget=message_bus.send_with_budget)
    assert append_terminal_task_projection(data, tid, task, stored, done)
    assert settle_terminal_projection(data, tid, task=task, event_queue=bus) == SETTLEMENT_DEFERRED
    server_liveness.drain_worker_events(bus, ctx, [time.monotonic(), {}, time.thread_time(), None],
                                       on_restart=lambda *_: pytest.fail("unexpected restart"))
    assert main_delivery_recorded(data, f"project-completion:{tid}", task_id=tid)
    [frame] = [f for f in frames if f.get("system_type") == "project_completion_summary" and f.get("task_id") == tid]
    assert (frame["chat_id"], frame["completion_answer"], frame["reply_to"]["address"]) == (1, answer, b)
    assert queue.persist_queue_snapshot(reason="completed_synthetic_project")
    return answer, frame


def _complete_on_second_request(data, mp, scratch, *, gap=False):
    """Drive real promotion, terminal mirroring and delivery. Only the model and transport are fixtures.

    ``gap``: an ordinary neutral synthetic row in Main between the selected request and the completion, so
    the answer's host-derived quote is not directly-above hidden (DESIGN "Replies"). The dispatch, the
    delivery and the quote stay the real machinery's either way. Both halves share one ``mp``, so the
    promotion's patches live until the delivery returns; the return keeps its original four values."""
    a, b, words, tid, task, queue, workers = _promote_second_request(data, mp, scratch, gap=gap)
    answer, _frame = _deliver_second_completion(data, mp, b, tid, task, queue, workers)
    return a, b, words, answer


@pytest.mark.parametrize("gap", [True, False], ids=["quote-shown", "quote-hidden-directly-above"])
def test_main_quotes_the_request_that_started_the_completed_project(direct_server_with_data, monkeypatch, tmp_path, gap):
    """The real completion arrives while Main stands open, then again from the persisted room after reload.

    Two phases of one delivery, named in the evidence. ``live-arrival`` dispatches the producer's own
    unmodified bridge broadcast into the open page's socket — a live-bridged arrival, not a network-socket
    proof — and the room is proven to hold no quote of the selected row before that emit, so an initial
    replay cannot stand in for the delivery. ``replay`` reloads and lets the same room re-read the
    persisted row through the running server, independent of the bridged hop. Promotion, terminal
    mirroring and delivery are the real machinery, split around the moment Main opens; only the model is a
    mock and only the socket hop is bridged. DESIGN "Replies" holds in both: with an ordinary gap row the
    derived quote shows and its click lands on the exact second of two identical requests, never its
    same-words twin (the address alone tells them apart); without it the selected row is directly above
    the answer, where the quote exists, addresses the exact row and stays hidden, never forced clickable.
    """
    from playwright.sync_api import sync_playwright

    stand = direct_server_with_data
    feed = "#chat-messages"
    with monkeypatch.context() as mp:
        a, b, words, tid, task, queue, workers = _promote_second_request(stand["data_dir"], mp, tmp_path, gap=gap)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 850})
                # Both witnesses go in before navigation: a script added after goto never runs, and
                # _emit_ws_frame dispatches into the socket _CAPTURE_TEST_SOCKET recorded.
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.add_init_script(f"({_OBSERVE_QUOTES})()")
                _open(page, stand["url"])
                assert page.evaluate("() => (window.__testSockets || []).length") >= 1, "no socket to deliver into"
                for address in (a, b):
                    assert page.locator(f'{feed} [data-row-address={json.dumps(address)}]').count() == 1, address
                selected_quote = f'{feed} .chat-quote[data-address={json.dumps(b)}]'
                assert page.locator(selected_quote).count() == 0, "the completion has not been delivered yet"
                answer, frame = _deliver_second_completion(stand["data_dir"], mp, b, tid, task, queue, workers)
                _emit_ws_frame(page, frame)
                page.wait_for_function(f"() => document.querySelectorAll({json.dumps(selected_quote)}).length === 1",
                                       timeout=15_000)
                for phase in ("live-arrival", "replay"):
                    if phase == "replay":
                        page.reload(wait_until="domcontentloaded")
                        _idle(page, feed)
                    bubble = page.locator(f"{feed} .project-answer").filter(has_text=answer)
                    bubble.wait_for(state="visible")
                    quote = bubble.locator(".chat-quote")
                    assert quote.get_attribute("data-address") == b != a
                    assert quote.locator(".chat-quote-text").inner_text() == words
                    if gap:
                        assert quote.is_visible(), "an ordinary row above the answer shows the derived quote"
                        quote.locator(".chat-quote-main").click()
                        facts = _landed(page, b, feed)
                        assert facts["address"] == b != a
                    else:
                        facts = quote.evaluate("""node => {
                            const main = node.querySelector('.chat-quote-main');
                            return {hidden: node.hidden, mainZero: !main || main.getBoundingClientRect().width === 0,
                                    address: node.dataset.address};
                        }""")
                        assert facts == {"hidden": True, "mainZero": True, "address": b}, facts
                    _evidence(page, tmp_path, f"selected-origin-project-{phase}-{'shown' if gap else 'hidden'}-chromium",
                              {"selected": b, "arrival": phase}, facts)
            finally:
                browser.close()


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_main_quote_lands_in_the_project_already_docked_beside_it(direct_server_with_data, browser_engine):
    from playwright.sync_api import sync_playwright

    data = direct_server_with_data["data_dir"]
    project = create_project(data, "docked-source", name="Docked source", origin="owner_ui")
    source = {"chat_id": project["chat_id"], "direction": "out", "ts": "2026-10-01T12:00:00Z",
              "text": "Copper light, then copper light."}
    ds.append(data, "chat", source, feed=True)
    excerpt = {"text": "copper light", "start_utf16": source["text"].index("copper")}
    ref = {**reference_from_row(source), "excerpt": excerpt}
    ds.append(data, "chat", {"chat_id": 1, "direction": "out", "ts": "2026-10-01T12:01:00Z",
                              "text": "Follow the second lamp.", "reply_to": ref}, feed=True)
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 850})
            page.add_init_script(f"({_OBSERVE_QUOTES})()")
            _open(page, direct_server_with_data["url"])
            room_browser._open(page, project["id"])
            feed = "#pchat-docked-source-messages"
            _idle(page, feed)
            page.locator("#pchat-docked-source-input").fill("Keep my project draft.")
            # The docked Project's scrim takes pointer clicks. Main stays keyboard-accessible beside it.
            _quote(page, ref["address"]).press("Enter")
            _landed(page, ref["address"], feed)
            page.wait_for_function("""([feed, text, start]) => {
                const selection = getSelection();
                const r = [...(CSS.highlights?.get('reply-excerpt') || [])][0]
                    || (selection.rangeCount && selection.getRangeAt(0));
                return r?.toString() === text && r.startOffset === start && r.startContainer.parentElement.closest(feed);
            }""", arg=[feed, excerpt["text"], excerpt["start_utf16"]])
            assert page.locator("#pchat-docked-source-input").input_value() == "Keep my project draft."
        finally:
            browser.close()


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_exact_cross_room_quote_and_return_preserve_drafts_unread_and_history(direct_server_with_data, browser_engine, tmp_path):
    from playwright.sync_api import sync_playwright
    fixture = seed_reply_history(direct_server_with_data["data_dir"], outside_sidebar=True)
    source = fixture["sources"]["a"]["address"]
    feed = "#pchat-quote-triage-messages"
    facts = {}
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 850})
        page.add_init_script(f"({_OBSERVE_QUOTES})()")
        try:
            _open(page, direct_server_with_data["url"])
            assert page.locator('.nav-project-row[data-project-id="quote-triage"]').count() == 0
            page.locator('#chat-input').fill("Keep this unsent draft")
            page.locator('#chat-file-input').set_input_files({"name": "draft.txt", "mimeType": "text/plain", "buffer": b"unsent"})
            _quote(page, source).click()
            facts["triage"] = _landed(page, source, feed)
            _evidence(page, tmp_path, f"triage-target-{browser_engine}", fixture, facts)
            assert page.locator('#project-panel-title').inner_text() == "Triage source"
            assert not page.evaluate("() => window.__quoteAcks.some(x => x.project_seen_revision?.['quote-triage'])")
            reads = page.evaluate("() => window.__quoteReads")
            assert any(r["lookup"] and str(fixture["a"]["chat_id"]) in r["url"] for r in reads)
            assert next(r["body"]["projects"][0]["visible_revision"] for r in reads if r["lookup"]) > 0
            _step(page, feed)
            assert page.locator(f'{feed} [data-client-message-id="owner-a"]').count() == 1
            page.locator('#pchat-quote-triage-scroll-bottom').click()
            _idle(page, feed)
            main = fixture["sources"]["main"]["address"]
            _quote(page, main, feed).click()
            facts["main"] = _landed(page, main, "#chat-messages")
            _evidence(page, tmp_path, f"main-target-{browser_engine}", fixture, facts)
            assert page.locator('#chat-input').input_value() == "Keep this unsent draft"
            assert "draft.txt" in page.locator('#chat-attachment-preview').inner_text()
            # Reply in the composer survives reload, and its quote retries from an evicted window.
            # Evidence capture may outlast the landing highlight; Reply must still address that saved row.
            page.locator('#chat-messages .is-quote-target').wait_for(state='detached')
            page.locator(f'#chat-messages [data-row-address={json.dumps(main)}] .chat-message-reply').click()
            assert page.evaluate("() => JSON.parse(sessionStorage.getItem('ouro_chat_reply')).address") == main
            page.locator('#chat-scroll-bottom').click()
            assert not page.evaluate("() => window.__quoteSends.some(x => JSON.parse(x).type === 'chat')")
            page.reload(wait_until="domcontentloaded")
            _idle(page, '#chat-messages')
            page.locator('#page-chat .chat-quote--composer .chat-quote-main').click()
            facts["composer"] = _landed(page, main, "#chat-messages")
            sends = page.evaluate("() => window.__quoteSends.map(x => {try{return JSON.parse(x)}catch{return {}}})")
            assert not any(frame.get("type") == "chat" for frame in sends)
        finally:
            _evidence(page, tmp_path, f"cross-room-{browser_engine}", fixture, facts)
            browser.close()


CARD_QUESTION = "Which colors should the table legend use?"
CARD_CHOICE = "1. Use blue and gold"
CARD_WORDS = "Use stripes for the smallest plots"


def _seed_card_answer_reply(root, answer="option", gap=0, *, block=True, selected=False):
    """Another work asked a card in a Project room; my human chose its first option (``answer="option"``), wrote
    their own words (``"free_text"``) or both (``"option_comment"``), after ``gap`` more messages of the room, recorded
    as a System row whose copy the question's block keeps. Seed an existing Reply to that answer for browser
    rendering and navigation; this fixture does not invoke ``start_task``."""
    from unittest.mock import patch

    from ouroboros.owner_quiz import mark_quiz_history_written, quiz_answer_frame, record_answered, record_asked
    from ouroboros.task_results import load_task_result

    room = create_project(root, "table-room", name="Seed table", origin="owner_ui")
    start = datetime(2024, 4, 9, 10, tzinfo=timezone.utc)
    for index in range(30):
        for chat in (1, room["chat_id"]):
            ds.append(root, "chat", {"chat_id": chat, "ts": (start + timedelta(seconds=index)).isoformat(),
                                     "direction": "in" if index % 2 else "out", "text": f"Context {index:03}"}, feed=True)
    asked_at = (start + timedelta(minutes=5)).isoformat()
    with patch("ouroboros.owner_quiz.utc_now_iso", return_value=asked_at):
        record_asked(root, "palette-asker", quiz_id="q-palette", question=CARD_QUESTION,
                     options=["Use blue and gold", "Use gray and green"], wait_for_answer=True)
    ds.append(root, "chat", {"chat_id": room["chat_id"], "ts": asked_at,
                             "direction": "out", "type": "quiz", "task_id": "palette-asker", "text": CARD_QUESTION,
                             "quiz": {"quiz_id": "q-palette", "options": [{"label": "Use blue and gold"},
                                                                         {"label": "Use gray and green"}],
                                      "state": "open", "wait_for_answer": True}}, feed=True)
    index, comment = {"option": (0, ""), "free_text": (None, CARD_WORDS), "option_comment": (0, CARD_WORDS)}[answer]
    with patch("ouroboros.owner_quiz.utc_now_iso", return_value=(start + timedelta(minutes=14)).isoformat()):
        quiz_block = record_answered(root, "palette-asker", quiz_id="q-palette", option_index=index,
                                     request_id="owner-choice", comment=comment)["block"]
    for later in range(gap):  # the room goes on between the question and my human's answer
        ds.append(root, "chat", {"chat_id": room["chat_id"], "ts": (start + timedelta(minutes=6, seconds=later)).isoformat(),
                                 "direction": "in" if later % 2 else "out", "text": f"Later {later:03}"}, feed=True)
    answer = {"chat_id": room["chat_id"], "ts": str(quiz_block["answered_at"]), "direction": "system", "type": "quiz_answer",
              "source": "owner_quiz_answer", "task_id": "palette-asker", "client_message_id": "quiz_answer:palette-asker:q-palette",
              "text": quiz_answer_frame(quiz_block, index, comment), "quiz": dict(quiz_block)}
    mark_quiz_history_written(root, "palette-asker", "q-palette", source_row=ds.append(root, "chat", answer).row)
    if not block:
        from ouroboros.owner_quiz import mark_discharged, quiz_states
        mark_discharged(root, "palette-asker", "q-palette")
        for later in range(17):
            record_asked(root, "palette-asker", quiz_id=f"later-{later}", question="Another question", options=["Yes"])
        assert "q-palette" not in quiz_states(root, "palette-asker")
    _result(root, "palette-asker", chat_id=room["chat_id"], owner_quiz=load_task_result(root, "palette-asker")["owner_quiz"])
    _result(root, "legend-work", chat_id=1)
    if selected:
        from ouroboros.owner_relay import owner_row_source
        project = create_project(root, "selected-answer", name="Table legend", origin="owner_ui")
        ref, words = owner_row_source(answer)
        bind_task_to_project(root, "legend-work", project["id"], origin={"ref": ref, "text": words})
        ds.append(root, "chat", {"chat_id": project["chat_id"], "ts": (start + timedelta(minutes=21)).isoformat(),
                                 "direction": "out", "task_id": "legend-work", "text": "The legend uses the chosen colors.",
                                 "reply_to": reference_from_row(answer)}, feed=True)
    ds.append(root, "chat", {"chat_id": 1, "ts": (start + timedelta(minutes=20)).isoformat(), "direction": "out",
                             "task_id": "legend-work", "text": "The table legend is ready.",
                             "reply_to": reference_from_row(answer)}, feed=True)
    return room, answer


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_a_reply_to_my_humans_card_answer_reads_their_choice_in_their_voice(direct_server_with_data, browser_engine, tmp_path):
    """An existing Reply to my human's card answer shows their choice under their name, replayed and after reload;
    the quote keeps the answer row's address (its jump lands at the question card). The neighboring
    ``test_main_quotes_the_request_that_started_the_completed_project`` exercises the real start path."""
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data["data_dir"]
    _room, answer = _seed_card_answer_reply(root)
    address = format_address(row_address(answer))
    facts = {}
    with sync_playwright() as pw:
        try:
            browser = getattr(pw, browser_engine).launch(headless=True)
        except Exception as error:  # an engine this host has not installed is named, never passed silently
            if "Executable doesn't exist" in str(error):
                pytest.skip(f"{browser_engine} is not installed for this Playwright")
            raise
        page = browser.new_page(viewport={"width": 1280, "height": 850})
        page.add_init_script(f"({_OBSERVE_QUOTES})()")
        try:
            _open(page, direct_server_with_data["url"])
            for attempt in ("replay", "reload"):
                if attempt == "reload":  # a fresh page over the same saved rows
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === WebSocket.OPEN)")
                    _idle(page, "#chat-messages")
                quote = _quote(page, address)
                facts[attempt] = quote.evaluate("""main => ({
                    author: main.querySelector('.chat-quote-author').textContent,
                    words: main.querySelector('.chat-quote-text').textContent,
                    voice: main.closest('.chat-quote').dataset.author, label: main.getAttribute('aria-label')})""")
                assert facts[attempt]["author"] == "You" and facts[attempt]["words"] == CARD_CHOICE, facts
                assert "Owner quiz answer" not in facts[attempt]["label"] and CARD_QUESTION not in facts[attempt]["label"]
                assert facts[attempt]["voice"] == "user", "the quote of a recorded choice uses the owner's ink"
                quote.scroll_into_view_if_needed()
                _evidence(page, tmp_path, f"card-answer-quote-{attempt}-{browser_engine}", {"address": address}, facts)
        finally:
            browser.close()


@pytest.mark.parametrize("answer,gap,block,selected", [
    ("option", 0, True, False), ("free_text", 0, True, False), ("option_comment", 0, True, False),
    ("option", 200, True, False), ("option_comment", 200, False, False),
    ("option", 200, True, True), ("option_comment", 200, False, True)],
    ids=["option", "free_text", "option_comment", "option_far", "evicted_far", "selected_far", "selected_evicted_far"])
@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_a_reply_to_my_humans_card_answer_lands_on_that_answer(direct_server_with_data, browser_engine, answer, gap, block, selected, tmp_path):
    """Pressed, Main's Reply to work my human's card answer started opens the card's Project room and lands on the
    answer that card already draws for the answer row (their chosen option, else their own words; a choice with words
    lights both), carrying that row's identity: never the question, never the card as a whole, never Main's mirror of
    it; replayed and after a reload, also when the room said much more between the question and the answer."""
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data["data_dir"]
    room, row = _seed_card_answer_reply(root, answer, gap, block=block, selected=selected)
    address, feed = format_address(row_address(row)), f'[id="pchat-{room["id"]}-messages"]'
    words = CARD_WORDS if answer == "free_text" else CARD_CHOICE.split(". ", 1)[1]
    holder = ".chat-quiz-answer" if answer == "free_text" else ".chat-quiz-option.chosen"
    facts = {}
    with sync_playwright() as pw:
        try:
            browser = getattr(pw, browser_engine).launch(headless=True)
        except Exception as error:  # an engine this host has not installed is named, never passed silently
            if "Executable doesn't exist" in str(error):
                pytest.skip(f"{browser_engine} is not installed for this Playwright")
            raise
        page = browser.new_page(viewport={"width": 1280, "height": 850})
        page.add_init_script(f"({_OBSERVE_QUOTES})()")
        try:
            _open(page, direct_server_with_data["url"])
            for attempt in ("replay", "reload"):
                if attempt == "reload":  # a fresh page over the same saved rows
                    page.goto(direct_server_with_data["url"], wait_until="domcontentloaded", timeout=30_000)
                    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === WebSocket.OPEN)")
                    _idle(page, "#chat-messages")
                # Main draws no node of the answer row: its Reply names it, the card in its room shows it.
                assert page.locator(f'#chat-messages [data-row-address={json.dumps(address)}]').count() == 0
                source_feed = "#chat-messages"
                if selected:
                    page.locator('.nav-project-row[data-project-id="selected-answer"]').click()
                    source_feed = "#pchat-selected-answer-messages"
                    _idle(page, source_feed)
                _quote(page, address, source_feed).last.click()
                if answer == "option_comment":  # one answer: the words beside the choice light with it, that moment
                    page.wait_for_function("""feed => {
                        const node = document.querySelector(`${feed} .is-quote-target`);
                        const line = node?.closest('.chat-quiz-card')?.querySelector('.chat-quiz-answer');
                        return Boolean(line && line !== node && getComputedStyle(line).animationName === 'chat-quote-target');
                    }""", arg=feed, timeout=15_000)
                # The exact holder is marked first; where it stands is judged only once the room's motion and its history
                # reads have settled (_settled_row), passively: the test neither scrolls nor focuses anything itself.
                facts[attempt] = _arrived(page, address, feed)
                facts[f"{attempt}-target"] = page.locator(f"{feed} [data-row-address={json.dumps(address)}]").evaluate(
                    """(node, holder) => {
                    const card = node.closest('.chat-quiz-card');
                    const stamped = [...document.querySelectorAll('[data-row-address]')].filter(
                        n => n.dataset.rowAddress === node.dataset.rowAddress);
                    return {inCard: Boolean(card), wholeCard: node.matches('.chat-quiz-card'), holder: node.matches(holder),
                        inMain: Boolean(node.closest('#chat-messages')), text: node.textContent,
                        focused: document.activeElement === node, historyId: node.dataset.historyId || '',
                        pressable: node.matches('button') ? !node.disabled && node.getAttribute('aria-disabled') !== 'true' : null,
                        stamped: stamped.length, retained: stamped.filter(n => n.dataset.originProjected === '1'
                            && n.closest('#pchat-selected-answer-messages')).length}}""", holder)
                target = facts[f"{attempt}-target"]
                assert target["inCard"] and not target["wholeCard"] and not target["inMain"], target
                assert target["holder"] and target["focused"] and not target["historyId"], target
                assert target["stamped"] == 1 + target["retained"] and (selected or not target["retained"]), target
                assert words in target["text"] and CARD_QUESTION not in target["text"], target
                assert target["pressable"] in (None, False), "the recorded choice takes no press"
                # Past the room's motion (its panel slides in, its newest page arrives): the answer stays focused, whole
                # inside the window, below the header and clear of the composer, with at least a line of it in view.
                settled = facts[f"{attempt}-settled"] = _settled_row(
                    page, page.locator(f"{feed} [data-row-address={json.dumps(address)}]"), feed)
                assert settled["focused"] and settled["x"] >= 0 and settled["x"] + settled["width"] <= 1280, settled
                assert settled["top"] >= settled["clear"] - 2 and settled["bottom"] <= settled["edge"] + 2, settled
                assert min(settled["bottom"], settled["edge"]) - settled["top"] >= 12 and not settled["overlaps"], settled
                _evidence(page, tmp_path, f"card-answer-landing-{answer}-{gap}-{attempt}-{browser_engine}", {"address": address}, facts)
        finally:
            _evidence(page, tmp_path, f"card-answer-landing-{answer}-{gap}-{browser_engine}", {"address": address}, facts)
            browser.close()


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_a_landed_card_answer_keeps_its_ring_in_light_reduced_motion(direct_server_with_data, browser_engine, tmp_path):
    """Reduced motion holds a quote's ring still instead of fading it. On the light theme the chosen answer is flat,
    yet its ring shows there as on the dark theme's lifted one: the theme's no-lift is a shadow layer it can join."""
    from playwright.sync_api import sync_playwright

    room, row = _seed_card_answer_reply(direct_server_with_data["data_dir"])
    address, feed = format_address(row_address(row)), f'[id="pchat-{room["id"]}-messages"]'
    ring = {}
    with sync_playwright() as pw:
        try:
            browser = getattr(pw, browser_engine).launch(headless=True)
        except Exception as error:  # an engine this host has not installed is named, never passed silently
            if "Executable doesn't exist" in str(error):
                pytest.skip(f"{browser_engine} is not installed for this Playwright")
            raise
        page = browser.new_page(viewport={"width": 1280, "height": 850}, color_scheme="light", reduced_motion="reduce")
        page.add_init_script(f"({_OBSERVE_QUOTES})()")
        try:
            _open(page, direct_server_with_data["url"])
            _quote(page, address).last.click()
            ring = page.wait_for_function("""feed => {
                const node = document.querySelector(`${feed} .chat-quiz-option.chosen.is-quote-target`);
                if (!node || document.documentElement.dataset.theme !== 'light') return null;
                const style = getComputedStyle(node);
                return {shadow: style.boxShadow, animation: style.animationName, address: node.dataset.rowAddress};
            }""", arg=feed, timeout=15_000).json_value()
            assert ring["address"] == address and ring["animation"] == "none", ring
            assert ring["shadow"] not in ("", "none") and "3px" in ring["shadow"], ring
        finally:
            _evidence(page, tmp_path, f"card-answer-light-reduced-{browser_engine}", {"address": address}, ring)
            browser.close()


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
@pytest.mark.parametrize("viewport", [(1280, 850), (1084, 724)], ids=["wide", "sender"])
def test_exact_nested_line_and_post_quote_conversion(direct_server_with_data, browser_engine, viewport, tmp_path):
    from playwright.sync_api import sync_playwright
    root = direct_server_with_data["data_dir"]
    fixture = seed_reply_history(root)
    facts = {}
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        page.add_init_script(f"({_OBSERVE_QUOTES})()")
        try:
            _open(page, direct_server_with_data["url"])
            address = fixture["sources"]["converted"]["address"]
            # The quote is already painted with its old chat label before the binding changes.
            assert _quote(page, address).count() == 1
            bind_task_to_project(root, "converted-source", fixture["x"]["id"], origin={"absent": "system"})
            _quote(page, address).click()
            facts["converted"] = _landed(page, address, '#pchat-quote-converted-messages')
            assert page.evaluate("window.__ouroLayers.returnPage()?.back.label") == "Main"
            _evidence(page, tmp_path, f"converted-target-{browser_engine}", fixture, facts)
            page.locator('#project-panel-close').click()
            _idle(page, '#chat-messages')
            address = fixture["sources"]["line"]["address"]
            _quote(page, address).click()
            feed = '#pchat-quote-triage-messages'
            facts["line_initial"] = _landed(page, address, feed)
            target = page.locator(f'{feed} .chat-live-line[data-history-id={json.dumps(facts["line_initial"]["id"])}]')
            facts["line"] = _settled_row(page, target, feed)
            _evidence(page, tmp_path, f"line-target-{browser_engine}", fixture, facts)
            assert facts["line"]["id"].startswith('progress:')
            assert facts["line"]["focused"] and "Child timeline line 30" in target.inner_text()
            assert facts["line"]["top"] >= facts["line"]["clear"] - 2, facts["line"]
            assert facts["line"]["bottom"] <= facts["line"]["edge"] + 2, facts["line"]
            assert not facts["line"]["overlaps"], facts["line"]
            timeline_top = "n => n.closest('[data-live-timeline]').scrollTop"
            before = target.evaluate(timeline_top)
            assert before > 0
            # Real input reads further inside the bounded timeline. A subsequent
            # composer resize must not reveal the old quote again or take focus.
            page.keyboard.press('PageDown')
            facts["after_key"] = _settled_row(page, target, feed)
            after = target.evaluate(timeline_top)
            assert after > before + 40, (before, after, facts["after_key"])
            page.locator('#pchat-quote-triage-input').fill('Project draft\n' * 5)
            facts["after_resize"] = _settled_row(page, target, feed)
            assert abs(target.evaluate(timeline_top) - after) <= 2, facts["after_resize"]
            assert not facts["after_resize"]["focused"], 'late layout stole focus from the composer'
            _evidence(page, tmp_path, f"line-after-gesture-{browser_engine}", fixture, facts)
            page.locator('#project-panel-close').click()
            _idle(page, '#chat-messages')
            # Ordinary reopening from the sidebar finds the kept room as left, its reading place and draft (PR #2951,
            # owner 2026-11-03: DESIGN "History edges"), and clears the source jump's return; `Scroll to latest message`
            # then returns it to the present.
            page.locator('.nav-project-row[data-project-id="quote-triage"]').click()
            _idle(page, '#pchat-quote-triage-messages')
            assert page.evaluate("window.__ouroLayers.returnPage() === null")
            assert target.count() == 1, 'the kept room still holds the passage it was left on'
            facts["reopened"] = _settled_row(page, target, feed)
            assert abs(target.evaluate(timeline_top) - after) <= 2, facts["reopened"]
            assert abs(facts["reopened"]["feedTop"] - facts["after_resize"]["feedTop"]) <= 2, facts["reopened"]
            assert page.locator('#pchat-quote-triage-input').input_value() == 'Project draft\n' * 5
            page.locator('#pchat-quote-triage-scroll-bottom').click()
            _idle(page, '#pchat-quote-triage-messages')
            assert page.locator('#pchat-quote-triage-messages').evaluate(
                'n => n.scrollHeight - n.scrollTop - n.clientHeight') < 48
        finally:
            _evidence(page, tmp_path, f"line-conversion-{browser_engine}", fixture, facts)
            browser.close()


def _arrive(page, root, index, text):
    """A reply the server stores and delivers live while the reader is in Main."""
    ts = (datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(seconds=700 + index)).isoformat()
    ds.append(root, "chat", {"chat_id": 1, "ts": ts, "direction": "out", "text": text, "format": "markdown"})
    _emit_ws_frame(page, {"type": "chat", "role": "assistant", "chat_id": 1, "content": text, "ts": ts})
    page.locator("#chat-messages .chat-bubble").filter(has_text=text.split()[0]).wait_for(state="attached")


_LIVE_EDGE = "n => ({gap: n.scrollHeight - n.scrollTop - n.clientHeight, top: n.scrollTop, height: n.scrollHeight})"


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_cancelled_failed_and_overlapping_quotes_keep_newer_intent(direct_server_with_data, browser_engine, tmp_path):
    from playwright.sync_api import sync_playwright
    root = direct_server_with_data["data_dir"]
    fixture = seed_reply_history(root)
    a, main = [fixture["sources"][key]["address"] for key in ("a", "main")]
    facts = {}
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 850})
        page.add_init_script(f"({_OBSERVE_QUOTES})()")
        try:
            _open(page, direct_server_with_data["url"])
            page.locator('#chat-input').fill('Draft through failed lookup')
            # The reader follows the live edge. A reply arrives while the quote's read is held; the read then
            # fails. Nothing moved, so that reply and the next one are followed with no ↓ press.
            page.evaluate('address => { window.__quoteHold = address; window.__quoteFail = address; }', a)
            _quote(page, a).click()
            page.wait_for_function('() => window.__quoteHeld')
            _arrive(page, root, 0, "LIVE_DURING_FAILED_QUOTE " + "while the quoted read is held " * 40)
            facts["held"] = page.locator('#chat-messages').evaluate(_LIVE_EDGE)
            assert facts["held"]["gap"] > 48, facts  # held below while the read was pending
            page.evaluate('() => window.__quoteRelease()')
            page.get_by_text('Could not load the quoted message. Try again.').wait_for()
            page.wait_for_function("() => { const n = document.querySelector('#chat-messages');"
                                   " return n.scrollHeight - n.scrollTop - n.clientHeight < 4; }", timeout=10_000)
            facts["after_failure"] = page.locator('#chat-messages').evaluate(_LIVE_EDGE)
            _arrive(page, root, 1, "LIVE_AFTER_FAILED_QUOTE " + "the next reply is still followed " * 40)
            page.wait_for_function("() => { const n = document.querySelector('#chat-messages');"
                                   " return n.scrollHeight - n.scrollTop - n.clientHeight < 4; }", timeout=10_000)
            facts["next_reply"] = page.locator('#chat-messages').evaluate(_LIVE_EDGE)
            _evidence(page, tmp_path, f"failed-read-follows-{browser_engine}", fixture, facts)
            assert page.locator('#project-panel').get_attribute('class').find('open') == -1
            for interrupt in ('gesture', 'leave', 'newer'):
                page.evaluate("() => { window.__quoteHold = 'lookup'; window.__quoteHeld = null; }")
                _quote(page, a).click()
                page.wait_for_function('() => window.__quoteHeld?.lookup')
                if interrupt == 'gesture':
                    page.locator('#chat-messages').dispatch_event('wheel', {'deltaY': -100})
                elif interrupt == 'leave':
                    page.locator('[data-nav-page="files"]').click()
                else:
                    _quote(page, fixture['sources']['converted']['address']).click()
                    facts['newer'] = _landed(page, fixture['sources']['converted']['address'], '#chat-messages')
                page.evaluate('() => window.__quoteRelease()')
                page.wait_for_function('() => window.__quoteReads.every(r => r.done)')
                page.evaluate(_FRAMES)
                assert page.locator('#pchat-quote-triage-messages').count() == 0, 'stale lookup created a destination'
                if interrupt == 'leave':
                    page.locator('[data-nav-page="chat"]').click()
                page.locator('#chat-scroll-bottom').evaluate('n => n.click()')
                _idle(page, '#chat-messages')
            assert page.locator('#chat-input').input_value() == 'Draft through failed lookup'
            # Same original quote, now healthy, retries without a permanently unavailable label/state.
            _quote(page, a).click()
            facts['retry'] = _landed(page, a, '#pchat-quote-triage-messages')
        finally:
            _evidence(page, tmp_path, f"races-{browser_engine}", fixture, facts)
            browser.close()


# ---- Search in a chat (docs/DESIGN.md "Search in a chat"): a match lands through the same quote landing ----------
# Main holds four hundred saved messages, so most matches lie far older than its newest page, and a Project holds its
# own, with a question Main shows only as its copy. Ctrl+F opens the room in front's search with the keyboard in its
# field and the owner's draft untouched; Enter and Shift+Enter land on the exact rows (reading the page around one that
# is not loaded) below the open sheet; Esc closes it and gives the field back. On a phone, "+" → Search this chat
# opens it for a finger. Screenshots of each scene, in both themes, go to the run's evidence.

SEARCH_START = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
SEARCH_LONG = ("We walked along the river for most of the afternoon and stopped by the weir where the water slows down. "
        "Somewhere past the alders a Kingfisher flashed blue and orange, gone before the camera came up, and we "
        "waited another hour at the bank hoping it would come back to the same branch above the pool.")
# Owner messages are the odd ones: the matches alternate Ouroboros and the owner, newest first.
SEARCH_MAIN_MATCHES = {31: "The kingfisher nested by the old mill.", 100: SEARCH_LONG,
                201: "Saw a KINGFISHER again near the bridge", 390: "Remember the kingfisher photo?"}
SEARCH_QUESTION = "Which kingfisher route should the survey take?"
SEARCH_QUESTION_DETAIL = "Along the alders past the mill"
SEARCH_QUESTION_STAKE = "Nesting season closes the bank"


def _seed_search_rooms(root):
    from ouroboros.owner_quiz import record_asked

    for index in range(400):
        text = SEARCH_MAIN_MATCHES.get(index, f"Routine note {index}: nothing to report about the garden today.")
        ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(minutes=index)).isoformat(),
                                 "direction": "in" if index % 2 else "out", "text": text}, feed=True)
    room = create_project(root, "search-room", name="Bird notes", origin="owner_ui")
    for index, text in enumerate(["Plan the kingfisher survey for the project.", "Weather looks fine on Saturday.",
                                  "Kingfisher survey route is ready."]):
        ds.append(root, "chat", {"chat_id": room["chat_id"], "ts": (SEARCH_START + timedelta(days=1, minutes=index)).isoformat(),
                                 "direction": "out" if index % 2 else "in", "text": text}, feed=True)
    # The Project's open question: Main draws it only as a copy of the Project's, so only the Project finds it.
    # Stored as the host writes a question (supervisor/message_bus.py send_quiz): the question is the row's text, its
    # card's offered answers, details and stake in its quiz block.
    record_asked(root, "survey-asker", quiz_id="route", question=SEARCH_QUESTION, options=["River", "Weir"],
                 option_details=[SEARCH_QUESTION_DETAIL, ""], recommended_index=0, stake=SEARCH_QUESTION_STAKE,
                 wait_for_answer=True)
    ds.append(root, "chat", {"chat_id": room["chat_id"], "ts": (SEARCH_START + timedelta(days=1, minutes=5)).isoformat(),
                             "direction": "out", "type": "quiz", "task_id": "survey-asker", "text": SEARCH_QUESTION,
                             "quiz": {"quiz_id": "route", "options": [
                                 {"label": "River", "detail": SEARCH_QUESTION_DETAIL, "recommended": True},
                                 {"label": "Weir"}], "stake": SEARCH_QUESTION_STAKE, "assumption": "River",
                                 "state": "open", "wait_for_answer": True}}, feed=True)
    return room


_SEARCH_STATE = """page => {
    const root = document.querySelector(page);
    const find = root?.querySelector('.chat-find');
    const input = find?.querySelector('.chat-find-input');
    const active = document.activeElement;
    return {
        open: Boolean(find && !find.hidden),
        focused: active === input,
        activeId: active?.id || '',
        count: find?.querySelector('.chat-find-count')?.textContent || '',
        note: find?.querySelector('.chat-find-note')?.textContent || '',
        results: [...(find?.querySelectorAll('.chat-find-result') || [])].map(node => ({
            selected: node.getAttribute('aria-selected') === 'true',
            snippet: node.querySelector('.chat-find-snippet').textContent,
            marks: [...node.querySelectorAll('mark')].map(mark => mark.textContent),
            meta: node.querySelector('.chat-find-meta').textContent})),
        authored: Boolean(find?.querySelector('.chat-find-results')?.closest('[data-i18n-authored]')),
        bottom: find && !find.hidden ? find.getBoundingClientRect().bottom : 0,
    };
}"""

# The innermost shown row holding ``words``: a message bubble, or the card it is drawn in (a question's).
_SEARCH_ROW = """([page, words]) => {
    const rows = [...document.querySelectorAll(`${page} [data-row-address], ${page} [data-history-id]`)]
        .filter(row => row.textContent.includes(words) && row.getClientRects().length);
    return rows.find(row => !rows.some(other => other !== row && row.contains(other))) || null;
}"""
_SEARCH_LANDED = """([page, words]) => {
    const node = (""" + _SEARCH_ROW + """)([page, words]);
    if (!node) return null;
    const box = node.getBoundingClientRect(), feed = node.closest('.chat-messages, #chat-messages').getBoundingClientRect();
    return {top: box.top, bottom: box.bottom, feedBottom: feed.bottom, target: window.__quoteMarks.includes(node),
            highlighted: CSS.highlights ? CSS.highlights.has('ouro-chat-find') : null};
}"""


def _search_evidence(tmp_path):
    return producer_evidence_dir(tmp_path / "chat-search-evidence", "chat-search", env="CHAT_SEARCH_EVIDENCE_DIR")


# The case server runs no model engine, so its own /api/model-catalog read fails and the page's Circles mount reports
# it as a page error. These scenes supply the model catalog as a documented, empty ModelCatalogResponse (no models
# listed, none failed); history, search and every landing stay the real endpoints.
SEARCH_CATALOG = {"items": [], "errors": [], "model_sources": []}


# A landing marks its row for 1.6 s (chat_reply.js markTarget, TARGET_MS), which a slow runner can outlast before it
# reads the row. Every mark is recorded as it is set: the class arrives on the row and is still on when the change is
# seen (a row marked again counts again). `_land` starts a landing's own record, so a check reads that landing's marks.
_RECORD_QUOTE_TARGET = """() => {
    window.__quoteMarks = [];
    const marked = value => String(value || '').split(/\\s+/).includes('is-quote-target');
    new MutationObserver(changes => {
        const after = new Map();  // a row's class right after the change being read, newest change first
        for (let index = changes.length - 1; index >= 0; index -= 1) {
            const {target, oldValue} = changes[index];
            const value = after.has(target) ? after.get(target) : target.getAttribute('class');
            if (!marked(oldValue) && marked(value) && target.classList.contains('is-quote-target')
                && !window.__quoteMarks.includes(target)) window.__quoteMarks.push(target);
            after.set(target, oldValue);
        }
    }).observe(document, {subtree: true, attributes: true, attributeFilter: ['class'], attributeOldValue: true});
}"""


def _land(page, key="Enter"):
    """Land on the search's current match with ``key``; the mark record holds what this landing marks alone."""
    page.evaluate("() => { window.__quoteMarks = []; }")
    page.keyboard.press(key)


def _open_main_for_search(page, url):
    """Main, as a client opening the app sees it: its socket open and its newest saved page drawn."""
    page.route("**/api/model-catalog*", lambda route: route.fulfill(json=SEARCH_CATALOG))
    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
    page.add_init_script(f"({_RECORD_QUOTE_TARGET})()")
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_function("() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)")
    page.locator("#chat-messages [data-history-id]").first.wait_for(state="attached", timeout=30_000)
    page.evaluate(_SETTLE_RESTORE_FRAMES)


def _search_state(page, scope):
    return page.evaluate(_SEARCH_STATE, scope)


def _wait_search_results(page, scope, count):
    page.wait_for_function("""([scope, n]) => {
        const find = document.querySelector(`${scope} .chat-find`);
        return find && !find.hasAttribute('data-searching') && find.querySelectorAll('.chat-find-result').length === n
            && (n !== 0 || find.querySelector('.chat-find-count').textContent === '0 found');
    }""", arg=[scope, count], timeout=30_000)


def _search_landed(page, scope, words):
    """The row holding ``words`` is shown whole below the open sheet (as it stands after the landing)."""
    page.wait_for_function(f"args => ({_SEARCH_ROW})(args)?.getBoundingClientRect().height > 0", arg=[scope, words],
                           timeout=30_000)
    page.evaluate(_SETTLE_RESTORE_FRAMES)
    landed = page.evaluate(_SEARCH_LANDED, [scope, words])
    below = _search_state(page, scope)["bottom"]
    assert landed and landed["top"] >= below - 1 and landed["bottom"] <= landed["feedBottom"] + 1, (landed, below)
    return landed


def _search_both_themes(page, path):
    """The scene in the dark theme it opened in, then in the light one a new client takes from the system."""
    page.screenshot(path=str(path.with_name(f"{path.stem}-dark.png")))
    page.emulate_media(color_scheme="light")
    page.wait_for_function("() => document.documentElement.dataset.theme === 'light'")
    page.evaluate(_SETTLE_RESTORE_FRAMES)
    page.screenshot(path=str(path.with_name(f"{path.stem}-light.png")))
    page.emulate_media(color_scheme="dark")
    page.wait_for_function("() => document.documentElement.dataset.theme === 'dark'")
    page.evaluate(_SETTLE_RESTORE_FRAMES)


@pytest.mark.parametrize("browser_engine", ["webkit", "chromium"])
def test_ctrl_f_finds_saved_messages_of_the_room_in_front_and_lands_on_them(direct_server_with_data, browser_engine, tmp_path):
    from playwright.sync_api import sync_playwright

    room = _seed_search_rooms(direct_server_with_data["data_dir"])
    shots = _search_evidence(tmp_path)
    errors = []
    with sync_playwright() as pw:
        try:
            browser = getattr(pw, browser_engine).launch(headless=True)
        except Exception as error:  # an engine this host has not installed is named, never passed silently
            if "Executable doesn't exist" in str(error):
                pytest.skip(f"{browser_engine} is not installed for this Playwright")
            raise
        page = browser.new_page(viewport={"width": 1512, "height": 949}, device_scale_factor=2, color_scheme="dark")
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            page.fill("#chat-input", "my unsent draft")
            # A pointer-opened '+' menu leaves focus in the composer, but still owns the key.
            page.locator("#chat-plus").click()
            menu = page.locator(".chat-plus-menu:not([data-leaving])")
            menu.wait_for(state="visible")
            assert page.evaluate("document.activeElement.id") == "chat-input"
            page.keyboard.press("ControlOrMeta+f")
            assert not page.locator(".chat-find:visible").count(), "an open menu keeps Find even without keyboard focus"
            assert menu.is_visible()
            page.keyboard.press("Escape")
            menu.wait_for(state="hidden")
            page.keyboard.press("ControlOrMeta+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            state = _search_state(page, "#page-chat")
            assert state["open"] and state["focused"] and not state["results"], state
            page.keyboard.type("Kingfisher")
            _wait_search_results(page, "#page-chat", 4)
            state = _search_state(page, "#page-chat")
            assert state["count"] == "4 found" and state["authored"], state  # never the Project's question
            assert [result["marks"][0].casefold() for result in state["results"]] == ["kingfisher"] * 4, state
            assert state["results"][0]["snippet"] == "Remember the kingfisher photo?", state
            assert state["results"][2]["snippet"].startswith("…") and state["results"][2]["snippet"].endswith("…"), state
            assert "Ouroboros" in state["results"][0]["meta"] and "You" in state["results"][1]["meta"], state
            _search_both_themes(page, shots / f"{browser_engine}-main-results.png")

            _land(page)  # the newest match, loaded already
            landed = _search_landed(page, "#page-chat", "Remember the kingfisher photo?")
            assert landed["target"], landed
            state = _search_state(page, "#page-chat")
            assert state["focused"] and state["count"] == "1 of 4" and state["results"][0]["selected"], state
            page.keyboard.press("Enter")
            page.keyboard.press("Enter")  # far older than the newest page: the page around it is read
            _search_landed(page, "#page-chat", "flashed blue and orange")
            state = _search_state(page, "#page-chat")
            assert state["focused"] and state["count"] == "3 of 4" and state["results"][2]["selected"], state
            _search_both_themes(page, shots / f"{browser_engine}-main-landed-older.png")
            page.keyboard.press("Shift+Enter")
            _search_landed(page, "#page-chat", "Saw a KINGFISHER again")
            assert _search_state(page, "#page-chat")["count"] == "2 of 4"

            page.keyboard.press("Escape")
            state = _search_state(page, "#page-chat")
            assert not state["open"] and state["activeId"] == "chat-input", state
            assert page.input_value("#chat-input") == "my unsent draft"
            assert page.evaluate("() => CSS.highlights ? !CSS.highlights.has('ouro-chat-find') : true")

            # A Project in front: the key searches the Project's own messages, never Main behind it.
            room_browser._open(page, room["id"])
            panel, feed = f'[id="panel-pchat-{room["id"]}"]', f'[id="pchat-{room["id"]}-messages"]'
            page.locator(feed).click(position={"x": 40, "y": 200})
            page.keyboard.press("ControlOrMeta+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            assert not _search_state(page, "#page-chat")["open"]
            page.keyboard.type("kingfisher survey")
            _wait_search_results(page, panel, 2)
            state = _search_state(page, panel)
            assert [result["snippet"] for result in state["results"]] == [
                "Kingfisher survey route is ready.", "Plan the kingfisher survey for the project."], state
            page.keyboard.press("Enter")
            _search_landed(page, panel, "Kingfisher survey route is ready.")
            _search_both_themes(page, shots / f"{browser_engine}-project-landed.png")
            # Its question is the Project's: found and landed on here, where Main finds nothing of it.
            page.fill(f"{panel} .chat-find-input", "kingfisher route")
            _wait_search_results(page, panel, 1)
            assert _search_state(page, panel)["results"][0]["snippet"] == (
                f"{SEARCH_QUESTION} {SEARCH_QUESTION_STAKE} River {SEARCH_QUESTION_DETAIL} Weir")
            _land(page)
            assert _search_landed(page, panel, SEARCH_QUESTION)["target"]
            # Its stake's label stands in a span of its own (mountQuizStake), the words a search leaves out.
            stake = page.evaluate("""feed => [...document.querySelectorAll(`${feed} .chat-quiz-stake`)].map(stake => ({
                labels: [...stake.querySelectorAll('.chat-quiz-stake-label')].map(label => label.textContent),
                text: stake.textContent.trim()}))""", feed)
            assert stake == [{"labels": ["At stake:"], "text": f"At stake: {SEARCH_QUESTION_STAKE}"}], stake
            # Its card's words are its offered answers, their details and its stake; never the card's labels, status
            # or recommendation. A match among them is marked where the card draws it.
            for words, count in (("along the alders", 1), ("survey take? Nesting season", 1), ("bank River", 1),
                                 ("At stake", 0), ("recommended", 0), ("Unanswered", 0), ("Question", 0)):
                page.fill(f"{panel} .chat-find-input", words)
                _wait_search_results(page, panel, count)
            for words in ("along the alders", "Nesting season", "Weir"):
                page.fill(f"{panel} .chat-find-input", words)
                _wait_search_results(page, panel, 1)
                page.keyboard.press("Enter")
                page.wait_for_function("([feed, words]) => (" + _SEARCH_CARD_MARK + ")(feed).text.toLowerCase() "
                                       "=== words.toLowerCase()", arg=[feed, words], timeout=30_000)
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                marks = page.evaluate(_SEARCH_CARD_MARK, feed)
                assert marks["count"] == 1 and marks["text"].casefold() == words.casefold(), (words, marks)
                assert marks["in"] == {"along the alders": "chat-quiz-option-detail",
                                       "Nesting season": "chat-quiz-stake", "Weir": "chat-quiz-option-label"}[words], marks
            _search_both_themes(page, shots / f"{browser_engine}-project-question-landed.png")
            page.keyboard.press("Escape")
            assert not _search_state(page, panel)["open"]
            assert page.locator(feed).is_visible(), "Esc closed the search, not the Project"
        finally:
            browser.close()
    assert not errors, errors


def test_search_lands_on_a_rendered_project_answer_and_other_pages_keep_find(direct_server_with_data, tmp_path):
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data["data_dir"]
    room = _seed_search_rooms(root)
    answer = "The swallows returned to their nests."
    ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(days=2)).isoformat(),
                              "direction": "system", "type": "project_completion_summary", "project_id": room["id"],
                              "text": "The project has finished.", "completion_answer": answer}, feed=True)
    # Production start/handoff rows carry both IDs: they bind a work portal, not a navigable chat bubble.
    for index, kind in enumerate(("project_started", "project_handoff")):
        ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(days=2, minutes=index + 1)).isoformat(),
                                  "direction": "system", "type": kind, "project_id": room["id"],
                                  "task_id": f"portal-work-{index}", "text": "portal-only falcon",
                                  "completion_answer": "portal-only falcon answer"}, feed=True)
    ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(days=2, minutes=3)).isoformat(),
                              "direction": "in", "source": "web", "text": "Here is the document.\n\n[Attached file: bird.txt]",
                              "attachments": [{"name": "bird.txt", "unavailable": "missing"}]}, feed=True)
    with sync_playwright() as pw:
        browser = pw.webkit.launch(headless=True)
        page = browser.new_page(viewport={"width": 1512, "height": 949}, device_scale_factor=2)
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            page.keyboard.press("ControlOrMeta+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            page.keyboard.type("swallows")
            _wait_search_results(page, "#page-chat", 1)
            result = _search_state(page, "#page-chat")["results"][0]
            assert result["snippet"] == answer and "Ouroboros" in result["meta"]
            page.keyboard.press("Enter")
            _search_landed(page, "#page-chat", answer)
            page.screenshot(path=str(_search_evidence(tmp_path) / "webkit-project-answer-found.png"))
            page.fill("#page-chat .chat-find-input", "portal-only falcon")
            _wait_search_results(page, "#page-chat", 0)
            assert _search_state(page, "#page-chat")["count"] == "0 found"
            page.fill("#page-chat .chat-find-input", "kingfisher route")  # the Project's question Main copies
            _wait_search_results(page, "#page-chat", 0)
            page.fill("#page-chat .chat-find-input", "Attached file")
            _wait_search_results(page, "#page-chat", 0)
            page.fill("#page-chat .chat-find-input", "bird.txt")  # the name its file card shows
            _wait_search_results(page, "#page-chat", 1)
            page.fill("#page-chat .chat-find-input", "Here is the document")
            _wait_search_results(page, "#page-chat", 1)
            assert _search_state(page, "#page-chat")["results"][0]["snippet"] == "bird.txt Here is the document."
            page.keyboard.press("Enter")
            _search_landed(page, "#page-chat", "Here is the document.")
            page.keyboard.press("Escape")
            room_browser._open(page, room["id"])
            room_browser._nav(page, "settings")
            page.keyboard.press("ControlOrMeta+f")
            assert not page.locator(".chat-find:visible").count(), "a departed Project never steals Find on Settings"
        finally:
            browser.close()


SEARCH_OWNER_ANSWER = "Neither: follow the heron upstream at dawn"


@pytest.mark.parametrize("browser_engine", ["webkit", "chromium"])
def test_an_answered_questions_owner_words_are_found_and_marked_where_its_card_draws_them(
        direct_server_with_data, browser_engine, tmp_path):
    """A question card settled as answered draws the owner's own words (``renderOwnerAnswer``): the Project's search
    finds and marks them there; the card's `Owner's answer:` label is not words."""
    from playwright.sync_api import sync_playwright

    from ouroboros.owner_quiz import record_answered

    root = direct_server_with_data["data_dir"]
    room = _seed_search_rooms(root)
    assert record_answered(root, "survey-asker", quiz_id="route", option_index=None, request_id="owner-answer",
                           comment=SEARCH_OWNER_ANSWER)["ok"]
    errors = []
    with sync_playwright() as pw:
        try:
            browser = getattr(pw, browser_engine).launch(headless=True)
        except Exception as error:  # an engine this host has not installed is named, never passed silently
            if "Executable doesn't exist" in str(error):
                pytest.skip(f"{browser_engine} is not installed for this Playwright")
            raise
        page = browser.new_page(viewport={"width": 1512, "height": 949}, device_scale_factor=2, color_scheme="dark")
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            room_browser._open(page, room["id"])
            panel, feed = f'[id="panel-pchat-{room["id"]}"]', f'[id="pchat-{room["id"]}-messages"]'
            page.locator(f"{feed} .chat-quiz-answer").wait_for(timeout=30_000)
            line = page.evaluate("""feed => { const line = document.querySelector(`${feed} .chat-quiz-answer`);
                return {text: line.textContent, labels: [...line.querySelectorAll('.chat-quiz-answer-label')]
                    .map(label => label.textContent)}; }""", feed)
            assert line == {"text": f"Owner's answer: {SEARCH_OWNER_ANSWER}", "labels": ["Owner's answer:"]}, line
            page.locator(feed).click(position={"x": 40, "y": 200})
            page.keyboard.press("ControlOrMeta+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            page.keyboard.type("heron upstream")
            _wait_search_results(page, panel, 1)
            result = _search_state(page, panel)["results"][0]
            assert result["marks"] == ["heron upstream"] and SEARCH_OWNER_ANSWER in result["snippet"], result
            _land(page)
            assert _search_landed(page, panel, SEARCH_QUESTION)["target"]
            page.wait_for_function("""feed => [...(CSS.highlights?.get('ouro-chat-find') || [])]
                .some(range => document.querySelector(feed).contains(range.startContainer))""", arg=feed, timeout=30_000)
            marks = page.evaluate("""feed => [...(CSS.highlights?.get('ouro-chat-find') || [])]
                .filter(range => document.querySelector(feed).contains(range.startContainer))
                .map(range => ({text: range.toString(),
                    in: range.startContainer.parentElement.closest('.chat-quiz-answer-label, .chat-quiz-answer')
                        ?.className || ''}))""", feed)
            assert marks == [{"text": "heron upstream", "in": "chat-quiz-answer"}], marks
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            page.screenshot(path=str(_search_evidence(tmp_path) / f"{browser_engine}-owner-answer-landed.png"))
            for words in ("Owner's answer", "answer: Neither"):
                page.fill(f"{panel} .chat-find-input", words)
                _wait_search_results(page, panel, 0)
            page.fill(f"{panel} .chat-find-input", "Neither: follow")
            _wait_search_results(page, panel, 1)
            page.keyboard.press("Escape")
        finally:
            browser.close()
    assert not errors, errors


# A finger's targets in the sheet (ui.css "Touch reach"): each control's box, what a touch 1px inside each of its edges
# and at its centre hits (the topmost target there, as a tap does), and the field's width.
_SEARCH_TARGETS = """scope => {
    const find = document.querySelector(`${scope} .chat-find`);
    const name = (node) => {
        const target = node?.closest('.chat-find-button, .chat-find-input, .chat-find-more, .chat-find-result');
        return target ? (target.className.match(/chat-find-(newer|older|close|input|more|result)/) || [])[1] : '';
    };
    const probe = (node) => {
        const box = node.getBoundingClientRect(), x = box.left + box.width / 2, y = box.top + box.height / 2;
        const at = (px, py) => name(document.elementFromPoint(px, py));
        return {width: box.width, height: box.height, own: name(node), hits: [at(box.left + 1, y), at(box.right - 1, y),
                at(x, box.top + 1), at(x, box.bottom - 1), at(x, y)]};
    };
    const nodes = [...find.querySelectorAll('.chat-find-input, .chat-find-button, .chat-find-more')]
        .filter((node) => node.getClientRects().length);
    return {coarse: matchMedia('(pointer: coarse)').matches, targets: nodes.map(probe),
            input: find.querySelector('.chat-find-input').getBoundingClientRect().width};
}"""


def _search_targets(page, scope, *, touch=44):
    """Every control of the sheet is a touch target of its own: at least ``touch`` px each way (the field as tall), and
    a tap anywhere inside it reaches it, never a neighbour's reach."""
    found = page.evaluate(_SEARCH_TARGETS, scope)
    assert found["coarse"], found
    for target in found["targets"]:
        assert target["height"] >= touch - 0.5 and (target["own"] == "input" or target["width"] >= touch - 0.5), found
        assert target["hits"] == [target["own"]] * 5, found
    return found


@pytest.mark.parametrize("browser_engine", ["webkit"])
def test_a_finger_opens_search_from_plus_and_lands_on_a_tapped_match(direct_server_with_data, browser_engine, tmp_path):
    from playwright.sync_api import sync_playwright

    _seed_search_rooms(direct_server_with_data["data_dir"])
    shots = _search_evidence(tmp_path)
    errors = []
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2, has_touch=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            page.locator("#page-chat #chat-plus").tap()
            item = page.locator(".chat-plus-menu:not([data-leaving]) [data-plus-item='search']")
            assert page.locator(".chat-plus-menu:not([data-leaving]) [role=menuitem]").all_inner_texts()[:2] == [
                "Attach file", "Search this chat"]
            item.tap()
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            _search_frame(page, "#page-chat", alone=True)  # a capsule around a finger's field too
            page.keyboard.type("kingfisher")
            _wait_search_results(page, "#page-chat", 4)
            targets = _search_targets(page, "#page-chat")
            assert [target["own"] for target in targets["targets"]] == ["input", "newer", "older", "close"], targets
            assert targets["input"] >= 120, targets
            sheet = page.locator("#page-chat .chat-find").bounding_box()
            assert sheet["x"] >= 0 and sheet["x"] + sheet["width"] <= 390.5, sheet
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            page.screenshot(path=str(shots / f"{browser_engine}-phone-results.png"))
            page.locator("#page-chat .chat-find-result").nth(3).tap()
            _search_landed(page, "#page-chat", "The kingfisher nested by the old mill.")
            assert page.locator("#page-chat .chat-find-results").is_hidden(), "the list folds over a phone's landing"
            page.screenshot(path=str(shots / f"{browser_engine}-phone-landed.png"))
            page.locator("#page-chat .chat-find-input").tap()
            assert page.locator("#page-chat .chat-find-results").is_visible(), "touching the field opens the list again"
            # ↓ and ↑ under a finger step through the matches, the keyboard staying in the field.
            page.locator("#page-chat .chat-find-newer").tap()
            _search_landed(page, "#page-chat", "flashed blue and orange")
            assert _search_state(page, "#page-chat")["count"] == "3 of 4"
            page.locator("#page-chat .chat-find-older").tap()
            _search_landed(page, "#page-chat", "The kingfisher nested by the old mill.")
            state = _search_state(page, "#page-chat")
            assert state["focused"] and state["count"] == "4 of 4", state
            # A failed read's Try again is a finger's target of its own in the note.
            page.route("**/api/chat/search?*", lambda route: route.abort())
            page.locator("#page-chat .chat-find-input").fill("heron at dusk")
            page.wait_for_function("() => document.querySelector('#page-chat .chat-find-more')?.textContent === 'Try again'")
            failed = _search_targets(page, "#page-chat")
            assert [target["own"] for target in failed["targets"]][-1] == "more", failed
            _search_both_themes(page, shots / f"{browser_engine}-phone-try-again.png")
            page.unroute("**/api/chat/search?*")
            page.locator("#page-chat .chat-find-more").tap()
            _wait_search_results(page, "#page-chat", 0)
            # A small, short phone: the field keeps room for its words and every control its own reach.
            page.set_viewport_size({"width": 320, "height": 568})
            page.locator("#page-chat .chat-find-input").fill("kingfisher")
            _wait_search_results(page, "#page-chat", 4)
            small = _search_targets(page, "#page-chat")
            assert small["input"] >= 72, small
            sheet = page.locator("#page-chat .chat-find").bounding_box()
            assert sheet["x"] >= 0 and sheet["x"] + sheet["width"] <= 320.5, sheet
            _search_both_themes(page, shots / f"{browser_engine}-phone-small-results.png")
            page.locator("#page-chat .chat-find-close").tap()
            assert not _search_state(page, "#page-chat")["open"]
        finally:
            browser.close()
    assert not errors, errors


# The sheet's frame: its corner, its padding and its field's corner as drawn (DESIGN "Search in a chat").
_SEARCH_FRAME = """page => {
    const find = document.querySelector(`${page} .chat-find`), bar = find.querySelector('.chat-find-bar');
    const sheet = find.getBoundingClientRect(), field = bar.getBoundingClientRect(), style = getComputedStyle(find);
    const note = find.querySelector('.chat-find-note');
    return {height: sheet.height, barHeight: field.height, pad: parseFloat(style.paddingTop),
            radius: parseFloat(style.borderTopLeftRadius),
            barRadius: Math.min(parseFloat(getComputedStyle(bar).borderTopLeftRadius), field.height / 2),
            note: note.getClientRects().length ? note.textContent : null, searching: find.hasAttribute('data-searching')};
}"""


def _search_frame(page, scope, *, alone):
    """The field is concentric with the sheet; the sheet is a capsule while it holds the field alone, else a card with
    the same corner (never an oval, never a box whose corner fights its field's)."""
    frame = page.evaluate(_SEARCH_FRAME, scope)
    assert abs(frame["radius"] - frame["pad"] - frame["barRadius"]) <= 0.5, frame
    if alone:
        assert frame["note"] is None, frame
        assert abs(frame["height"] - frame["barHeight"] - 2 * frame["pad"]) <= 0.5, frame
        assert abs(frame["radius"] - frame["height"] / 2) <= 0.5, frame
    else:
        assert frame["height"] > frame["barHeight"] + 2 * frame["pad"] + 1, frame
    return frame


@pytest.mark.parametrize("viewport", [(1512, 949), (1100, 750), (390, 844)], ids=["owner", "window", "phone"])
def test_the_search_sheet_is_a_capsule_until_it_has_more_to_say(direct_server_with_data, viewport, tmp_path):
    from playwright.sync_api import sync_playwright

    _seed_search_rooms(direct_server_with_data["data_dir"])
    shots, (width, height) = _search_evidence(tmp_path), viewport
    name = f"webkit-{width}x{height}"
    errors, held = [], []
    with sync_playwright() as pw:
        browser = pw.webkit.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2, color_scheme="dark")
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            page.keyboard.press("ControlOrMeta+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            empty = _search_frame(page, "#page-chat", alone=True)
            _search_both_themes(page, shots / f"{name}-empty.png")

            page.keyboard.type("Kingfisher")
            _wait_search_results(page, "#page-chat", 4)
            assert _search_frame(page, "#page-chat", alone=False)["radius"] == empty["radius"]
            _search_both_themes(page, shots / f"{name}-results.png")

            # A read held in transport: the sheet says it is searching (the real answer is released after).
            page.route("**/api/chat/search?*", lambda route: held.append(route))
            page.fill("#page-chat .chat-find-input", "routine note 3")
            page.wait_for_function("() => document.querySelector('#page-chat .chat-find')?.hasAttribute('data-searching')")
            page.wait_for_function("() => document.querySelector('#page-chat .chat-find-note')?.textContent.length > 0")
            assert _search_frame(page, "#page-chat", alone=False)["searching"]
            _search_both_themes(page, shots / f"{name}-loading.png")
            for route in held:
                route.abort()  # an offline read: the sheet says the history could not be searched, with Try again
            page.unroute("**/api/chat/search?*")
            page.wait_for_function("""() => document.querySelector('#page-chat .chat-find-more')?.textContent === 'Try again'""")
            failed = _search_frame(page, "#page-chat", alone=False)
            assert failed["note"].startswith("Saved history could not be searched."), failed
            _search_both_themes(page, shots / f"{name}-error.png")
            page.locator("#page-chat .chat-find-more").click()
            page.wait_for_function("""() => {
                const find = document.querySelector('#page-chat .chat-find');
                return !find.hasAttribute('data-searching') && find.querySelectorAll('.chat-find-result').length > 0;
            }""", timeout=30_000)
            _search_frame(page, "#page-chat", alone=False)
            page.fill("#page-chat .chat-find-input", "")
            page.wait_for_function("() => document.querySelector('#page-chat .chat-find-results')?.hidden")
            _search_frame(page, "#page-chat", alone=True)  # emptied words give the capsule back
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        finally:
            browser.close()
    assert not errors, errors


# A reply longer than the screen: its first matched words, not only its top, land between the sheet and the composer,
# across the line break inside them. A folded Project answer whose words its fold hides says where they read in full.
SEARCH_HERON = "\n\n".join([f"Field log line {n}: the reeds were still and the water was low." for n in range(110)]
                           + ["Then the grey\nheron stood in the shallows, quite still."]
                           + [f"After note {n}: nothing else moved on the pool." for n in range(20)])
SEARCH_OSPREY = "\n\n".join([f"Survey step {n}: counted the nests along the bank." for n in range(30)]
                            + ["An osprey took the last fish of the day."])
# A code line wider than its block, its words at the far end: the block scrolls sideways to show them.
SEARCH_WIDE_CODE = "Wide code:\n\n```\nroute = [" + ", ".join(f"'bank{n}'" for n in range(160)) + ", 'cormorant']\n```"
# The first mark's box inside every clipping box between it and the feed (the feed's own band is checked apart).
_SEARCH_MARK_CLIPPED = """scope => {
    const feed = document.querySelector(scope).querySelector('.chat-messages, #chat-messages');
    const range = [...(CSS.highlights?.get('ouro-chat-find') || [])].find(r => feed.contains(r.startContainer));
    const word = range?.getClientRects()[0];
    const clips = [];
    for (let box = range?.startContainer.parentElement; box && box !== feed; box = box.parentElement) {
        const style = getComputedStyle(box);
        if (style.overflowX === 'visible' && style.overflowY === 'visible') continue;
        const r = box.getBoundingClientRect(), left = r.left + box.clientLeft, top = r.top + box.clientTop;
        if (word.left < left - 1 || word.right > left + box.clientWidth + 1
                || word.top < top - 1 || word.bottom > top + box.clientHeight + 1) {
            clips.push({box: box.className || box.tagName, left, right: left + box.clientWidth, word: [word.left, word.right],
                        scrollLeft: box.scrollLeft});
        }
    }
    const pre = range?.startContainer.parentElement.closest('pre');
    return {text: range?.toString() || '', clips, scrollLeft: pre ? pre.scrollLeft : null,
            pageScrollX: window.scrollX, docLeft: document.scrollingElement.scrollLeft};
}"""
# This room's own marks in the shared highlight, the first of them with its text and box, the sheet's foot and the
# composer's head: what the reader sees of the landing.
_SEARCH_MARKS = """scope => {
    const room = document.querySelector(scope), feed = room.querySelector('.chat-messages, #chat-messages');
    const marks = CSS.highlights?.get('ouro-chat-find');
    const ranges = marks ? [...marks].filter(range => feed.contains(range.startContainer)) : [];
    ranges.sort((a, b) => a.compareBoundaryPoints(Range.START_TO_START, b));
    const first = ranges[0]?.getClientRects()[0];
    const row = ranges[0]?.startContainer.parentElement.closest('[data-row-address]')?.getBoundingClientRect();
    const find = room.querySelector('.chat-find');
    return {count: ranges.length, text: ranges[0] ? ranges[0].toString().replace(/\\s+/g, ' ') : '',
            top: first?.top, bottom: first?.bottom, rowTop: row?.top, rowBottom: row?.bottom,
            sheet: find && !find.hidden ? find.getBoundingClientRect().bottom : null,
            composer: room.querySelector('[id$="input-area"]').getBoundingClientRect().top};
}"""


# The one search mark in a question card: its words and the part of the card it stands in.
_SEARCH_CARD_MARK = """feed => {
    const marks = [...(CSS.highlights?.get('ouro-chat-find') || [])]
        .filter(range => document.querySelector(feed).contains(range.startContainer));
    const part = marks[0]?.startContainer.parentElement.closest('.chat-quiz-option-label, .chat-quiz-option-detail, '
        + '.chat-quiz-stake, .chat-quiz-question');
    return {count: marks.length, text: marks[0]?.toString() || '', in: part ? part.classList[0] : ''};
}"""


def _search_marks(page, scope):
    return page.evaluate(_SEARCH_MARKS, scope)


@pytest.mark.parametrize("browser_engine,viewport", [("webkit", (1512, 949)), ("chromium", (1512, 949)),
                                                     ("webkit", (390, 844))], ids=["webkit", "chromium", "phone"])
def test_a_long_reply_lands_on_its_matched_words_and_a_fold_says_where_they_are(
        direct_server_with_data, browser_engine, viewport, tmp_path):
    import sys

    from playwright.sync_api import sync_playwright

    root = direct_server_with_data["data_dir"]
    room = _seed_search_rooms(root)
    ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(hours=8)).isoformat(), "direction": "out",
                              "text": SEARCH_HERON}, feed=True)
    ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(hours=9)).isoformat(),
                              "direction": "system", "type": "project_completion_summary", "project_id": room["id"],
                              "text": "The project has finished.", "completion_answer": SEARCH_OSPREY}, feed=True)
    ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(hours=9, minutes=1)).isoformat(),
                              "direction": "out", "text": "Read [the guide](https://example.test/needle-only-url)."}, feed=True)
    ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(hours=9, minutes=2)).isoformat(),
                              "direction": "out", "text": "Hangul: \u1100\u1161\u11a8"}, feed=True)
    ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(hours=9, minutes=3)).isoformat(),
                              "direction": "out", "text": SEARCH_WIDE_CODE}, feed=True)
    for index in range(30):  # newer rows, so neither lands already on screen
        ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(hours=10, minutes=index)).isoformat(),
                                  "direction": "in", "text": f"Later chatter {index}."}, feed=True)
    shots = _search_evidence(tmp_path)
    errors = []
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]}, device_scale_factor=2,
                                color_scheme="dark")
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            if sys.platform == "darwin":  # a Mac's Ctrl+F is the field's own: one character forward, no search
                page.fill("#chat-input", "abc")
                page.evaluate("() => { const input = document.getElementById('chat-input'); input.focus(); "
                              "input.setSelectionRange(0, 0); }")
                page.keyboard.press("Control+f")
                assert not _search_state(page, "#page-chat")["open"]
                assert page.evaluate("document.getElementById('chat-input').selectionStart") == 1
                page.fill("#chat-input", "")
            page.locator("#chat-input").focus()
            page.keyboard.press("ControlOrMeta+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            page.keyboard.type("grey heron")
            _wait_search_results(page, "#page-chat", 1)
            page.keyboard.press("Enter")
            page.wait_for_function("scope => (" + _SEARCH_MARKS + ")(scope).count > 0", arg="#page-chat", timeout=30_000)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            marks = _search_marks(page, "#page-chat")
            assert marks["text"] == "greyheron", marks  # "grey", the soft line break drawn as <br>, "heron"
            assert marks["rowTop"] < marks["sheet"] and marks["rowBottom"] > marks["composer"], "a row longer than the screen"
            assert marks["sheet"] - 1 <= marks["top"] and marks["bottom"] <= marks["composer"] + 1, marks
            state = _search_state(page, "#page-chat")
            assert state["focused"] and state["count"] == "1 of 1", state
            page.screenshot(path=str(shots / f"{browser_engine}-{viewport[0]}-long-reply-landed.png"))

            page.fill("#page-chat .chat-find-input", "osprey")
            _wait_search_results(page, "#page-chat", 1)
            _land(page)
            page.wait_for_function("() => document.querySelector('#page-chat .chat-find-note')?.textContent.includes('folded')",
                                   timeout=30_000)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            state = _search_state(page, "#page-chat")
            assert state["note"] == "The words are in the folded part of this answer: its Project shows it in full.", state
            folded = page.evaluate("""() => {
                const row = [...document.querySelectorAll('#page-chat .chat-bubble.project-answer.is-folded')].find(
                    node => window.__quoteMarks.includes(node));
                return row && {top: row.getBoundingClientRect().top, bottom: row.getBoundingClientRect().bottom};
            }""")
            assert folded and folded["top"] >= state["bottom"] - 1, (folded, state)
            page.screenshot(path=str(shots / f"{browser_engine}-{viewport[0]}-folded-answer.png"))

            page.fill("#page-chat .chat-find-input", "needle-only-url")  # a link's address is not a word it shows
            _wait_search_results(page, "#page-chat", 0)
            page.fill("#page-chat .chat-find-input", "read the guide")
            _wait_search_results(page, "#page-chat", 1)
            page.fill("#page-chat .chat-find-input", "각")
            _wait_search_results(page, "#page-chat", 1)
            page.keyboard.press("Enter")
            page.wait_for_function("scope => (" + _SEARCH_MARKS + ")(scope).count > 0", arg="#page-chat")
            assert _search_marks(page, "#page-chat")["text"] == "\u1100\u1161\u11a8"
            assert "not shown here" not in _search_state(page, "#page-chat")["note"]

            # Words at the far end of a wide code line: the code block scrolls sideways to them, the page never does.
            page.fill("#page-chat .chat-find-input", "cormorant")
            _wait_search_results(page, "#page-chat", 1)
            page.keyboard.press("Enter")
            page.wait_for_function("scope => (" + _SEARCH_MARKS + ")(scope).text === 'cormorant'", arg="#page-chat",
                                   timeout=30_000)
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            clipped = page.evaluate(_SEARCH_MARK_CLIPPED, "#page-chat")
            marks = _search_marks(page, "#page-chat")
            assert clipped["text"] == "cormorant" and not clipped["clips"] and clipped["scrollLeft"] > 0, clipped
            assert clipped["pageScrollX"] == 0 and clipped["docLeft"] == 0, clipped
            assert marks["sheet"] - 1 <= marks["top"] and marks["bottom"] <= marks["composer"] + 1, marks
            state = _search_state(page, "#page-chat")
            assert state["focused"] and state["count"] == "1 of 1", state
            page.screenshot(path=str(shots / f"{browser_engine}-{viewport[0]}-wide-code-landed.png"))
            page.keyboard.press("Escape")
            assert page.evaluate("() => CSS.highlights ? !CSS.highlights.has('ouro-chat-find') : true")
        finally:
            browser.close()
    assert not errors, errors


# The words the room shows, not the stored source: Markdown's marks, a link's hidden or an image's address, a diagram's
# source and escapes are not words, an address the text shows is; code keeps its marks; a formula is its characters,
# never its TeX; a file its name and caption, an owner's attachment the name its card shows; a System sentence is
# searched as shown in the chosen language; NFC and
# Unicode 17 case folding. These messages are older than everything Main draws first, so each is read and drawn
# from its saved row, and a landing marks the words where the row then stands.
SEARCH_SHOWN = [
    ("out", "Say hello **world** to the river."),
    ("out", "Read [the river guide](https://example.test/hidden-url) today."),
    ("out", "Code: `a*b*c` and `_leading_name` stay literal."),
    ("out", "Escapes: \\*literal stars\\* and &amp; entity &lt;tag&gt;."),
    ("out", "Look: ![heron photo](https://example.test/heron.png)"),
    ("out", "| Bird | Count |\n| --- | --- |\n| Egret | 4 |"),
    ("out", "Diagram:\n\n```mermaid\ngraph TD; Pelican-->Stork\n```\n\nafter the diagram"),
    ("in", "Le café noir"),
    ("out", "Unicode: Die Straße, ΟΔΟΣ, \ufb01ne, \U00010400\U00010401"),
    ("out", "Hangul **\u1100\u1161**\u11a8 split"),
    ("out", "Formula: $$E=mc^2$$ and \\(\\alpha \\le \\text{velocity}\\) here."),
    ("out", "Roots: \\(\\sqrt{x}\\) and \\(\\sqrt[3]{y}\\) over \\(\\frac{q}{z}\\) end."),
    ("out", "Visible address: https://example.test/visible-url works."),
    ("in", "Owner link https://example.test/owner-url here"),
]
SEARCH_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWQAAAABJRU5ErkJggg=="
SEARCH_FILE = {"type": "document", "filename": "heron-survey.pdf", "caption": "The survey map", "text": "The survey map",
               "mime": "application/pdf"}
SEARCH_MEMORY_NOTICE = "🧠 Memory: the old notes were archived."
SEARCH_MEMORY_SHOWN = "Память: старые заметки перенесены в архив."


def _search_for(page, words, count):
    page.fill("#page-chat .chat-find-input", words)
    _wait_search_results(page, "#page-chat", count)
    return _search_state(page, "#page-chat")


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_search_reads_the_words_the_room_shows(direct_server_with_data, browser_engine, tmp_path):
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data["data_dir"]
    early = SEARCH_START - timedelta(days=1)
    for index, (direction, text) in enumerate(SEARCH_SHOWN):
        ds.append(root, "chat", {"chat_id": 1, "ts": (early + timedelta(minutes=index)).isoformat(),
                                  "direction": direction, "text": text}, feed=True)
    ds.append(root, "chat", {"chat_id": 1, "ts": (early + timedelta(minutes=30)).isoformat(), "direction": "system",
                              "type": "legacy_memory_notice", "text": SEARCH_MEMORY_NOTICE}, feed=True)
    ds.append(root, "chat", {"chat_id": 1, "ts": (early + timedelta(minutes=31)).isoformat(), "direction": "out",
                              **SEARCH_FILE}, feed=True)
    # The owner's attachments: a file card shows its name, an available image only its picture; the host's words for
    # an owner who sent only files are not shown.
    from ouroboros.chat_uploads import store_upload

    _, picture = store_upload(base64.b64decode(SEARCH_PNG), "heron-photo.png", data_dir=root)
    ds.append(root, "chat", {"chat_id": 1, "ts": (early + timedelta(minutes=32)).isoformat(), "direction": "in",
                              "text": "Here are my notes", "attachments": [
                                  picture, {"name": "survey-notes.txt", "unavailable": "missing"}]}, feed=True)
    ds.append(root, "chat", {"chat_id": 1, "ts": (early + timedelta(minutes=33)).isoformat(), "direction": "in",
                              "text": "(file attached)", "text_placeholder": True,
                              "attachments": [{"name": "egret-count.csv", "unavailable": "missing"}]}, feed=True)
    _seed_search_rooms(root)
    shots = _search_evidence(tmp_path)
    errors = []
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1512, "height": 949}, device_scale_factor=2, color_scheme="dark")
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/api/ui/i18n", lambda route: route.fulfill(json={"language": "ru", "english": False, "revision": 1,
                   "entries": {SEARCH_MEMORY_NOTICE: {"text": "🧠 " + SEARCH_MEMORY_SHOWN}}}))
        page.route("**/api/ui/i18n/missing", lambda route: route.fulfill(json={"ok": True}))
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            assert not page.evaluate("() => document.querySelector('#chat-messages').textContent.includes('river guide')"), \
                "the searched messages are not drawn yet"
            page.fill("#chat-input", "my unsent draft")
            page.keyboard.press("ControlOrMeta+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            found = {}
            for words, count, snippet in (
                    ("hello world", 1, "Say hello world to the river."), ("**world**", 0, None),
                    ("guide today", 1, "Read the river guide today."), ("hidden-url", 0, None),
                    ("a*b*c", 1, "Code: a*b*c and _leading_name stay literal."), ("abc", 0, None),
                    ("_leading_name", 1, None),
                    ("*literal stars*", 1, "Escapes: *literal stars* and & entity <tag>."), ("&amp;", 0, None),
                    ("& entity <tag>", 1, None),
                    ("Image: heron photo", 1, "Look: Image: heron photo"), ("heron.png", 0, None),
                    ("Egret 4", 1, None), ("Pelican", 0, None), ("Stork", 0, None),
                    ("CAFÉ", 1, "Le café noir"), ("cafe", 0, None),
                    ("STRASSE", 1, None), ("οδος", 1, None), ("FINE", 1, None), ("\U00010428\U00010429", 1, None),
                    ("\uac01", 1, None),
                    ("старые заметки", 1, SEARCH_MEMORY_SHOWN), ("old notes", 0, None), ("Memory:", 0, None),
                    ("E=mc2", 1, "Formula: E=mc2 and α≤velocity here."), ("mc^2", 0, None), ("\\text", 0, None),
                    ("velocity here", 1, None),
                    # A root shows its sign, its index first; a fraction's parts stand apart (MathML holds neither).
                    ("√x", 1, "Roots: √x and 3√y over q z end."), ("3√y", 1, None), ("y3", 0, None),
                    ("qz", 0, None), ("q z end", 1, None),
                    ("example.test/visible-url", 1, "Visible address: https://example.test/visible-url works."),
                    ("owner-url here", 1, "Owner link https://example.test/owner-url here"),
                    ("heron-survey.pdf", 1, "heron-survey.pdf The survey map"), ("survey map", 1, None),
                    ("survey-notes.txt", 1, "survey-notes.txt Here are my notes"), ("heron-photo", 0, None),
                    ("egret-count.csv", 1, "egret-count.csv"), ("file attached", 0, None)):
                state = _search_for(page, words, count)
                found[words] = state["results"]
                if snippet is not None:
                    assert state["results"][0]["snippet"] == snippet, (words, state)
                if count:
                    assert state["count"] == "1 found" and state["results"][0]["marks"], (words, state)
            assert found["STRASSE"][0]["marks"] == ["Straße"] and found["οδος"][0]["marks"] == ["ΟΔΟΣ"], found
            assert found["FINE"][0]["marks"] == ["\ufb01ne"] and found["Egret 4"][0]["marks"] == ["Egret 4"], found
            _search_for(page, "hello world", 1)
            _search_both_themes(page, shots / f"{browser_engine}-shown-words-results.png")
            for words, landed in (("hello world", "hello world"), ("guide today", "guide today"),
                                  ("\uac01", "\u1100\u1161\u11a8"), ("старые заметки", "старые заметки"),
                                  ("_leading_name", "_leading_name"), ("STRASSE", "Straße"),
                                  ("heron-survey.pdf", "heron-survey.pdf"), ("survey-notes.txt", "survey-notes.txt"),
                                  ("egret-count.csv", "egret-count.csv"),
                                  ("velocity", None), ("√x", None)):  # a formula's match marks it drawn, whole
                _search_for(page, words, 1)
                page.keyboard.press("Enter")
                page.wait_for_function("scope => (" + _SEARCH_MARKS + ")(scope).count > 0", arg="#page-chat",
                                       timeout=30_000)
                page.evaluate(_SETTLE_RESTORE_FRAMES)
                marks = _search_marks(page, "#page-chat")
                assert marks["count"] == 1 and (marks["text"] == landed if landed else (
                    words.strip("√") in marks["text"] and marks["bottom"] - marks["top"] > 4)), (words, marks)
                assert marks["sheet"] - 1 <= marks["top"] and marks["bottom"] <= marks["composer"] + 1, marks
                state = _search_state(page, "#page-chat")
                assert state["focused"] and "not shown here" not in state["note"], state
            page.screenshot(path=str(shots / f"{browser_engine}-shown-words-landed.png"))
            page.keyboard.press("Escape")
            assert page.input_value("#chat-input") == "my unsent draft"
            assert page.evaluate("document.activeElement.id") == "chat-input"
        finally:
            browser.close()
    assert not errors, errors


# A long room, searched as a reader would: words only its oldest message holds (every page read and drawn, a slice at
# a time, to the first match at the beginning), one common letter its newest short messages all hold (a page's worth of
# matches listed at once), a word its Markdown block holds (pages without a match read on to it), and words none holds,
# ended by Esc in the middle of the reading (no read after it; the draft and the focus come back). One message of about
# 200 KB is drawn whole, as its bubble is: its words are one synchronous step, timed here beside the room's own render.
# What it measured (pages, candidates, bytes, seconds, the longest main-thread task or frame gap the engine reports) goes
# to the evidence as observed on this host, never a general performance claim.
SEARCH_LONG_ROWS = 4000
SEARCH_LONG_OLDEST = "An osprey nested on the oldest post."
SEARCH_HUGE_LINES = 1500


def _seed_long_room(root):
    start = SEARCH_START - timedelta(days=30)
    for index in range(SEARCH_LONG_ROWS):
        direction = "out"
        if index == 0:
            text = SEARCH_LONG_OLDEST
        elif index == 1000:  # one message of about 200 KB
            text = "\n\n".join(f"Line {k}: the **reeds** were *still* at `post-{k}`; see [the map](https://example.test/{k})."
                               for k in range(SEARCH_HUGE_LINES))
        elif index < 2000:
            text = (f"## Field log {index}\n\nThe **reeds** were still, see [the map](https://example.test/{index}); "
                    f"water `level-{index}` was low" + (f", \\(v_{{{index}}} = n^2\\)" if index % 10 == 0 else "")
                    + f".\n\n- egrets: {index % 7}\n- note: nothing *else* moved on the pool.")
        else:  # the newest: short words both ways, every one holding an "e"
            direction = "in" if index % 2 else "out"
            text = f"see you at {index}" if index % 2 else f"Sure, here it is {index}."
        ds.append(root, "chat", {"chat_id": 1, "ts": (start + timedelta(minutes=index)).isoformat(),
                                  "direction": direction, "text": text}, feed=True)


_SEARCH_SETTLED = """() => { const find = document.querySelector('#page-chat .chat-find');
    return find && !find.hasAttribute('data-searching') && find.querySelector('.chat-find-count').textContent !== ''; }"""
# Long tasks where the engine reports them (Chromium), and the longest gap between animation frames (both engines).
_LOOP_PROBE = """() => { window.__longTasks = PerformanceObserver.supportedEntryTypes?.includes('longtask') ? [] : null;
    if (window.__longTasks && !window.__observing) {
        window.__observing = true;
        new PerformanceObserver(list => window.__longTasks?.push(...list.getEntries().map(e => e.duration)))
            .observe({type: 'longtask'});
    }
    window.__gap = 0; window.__frames = 0; window.__probe = (window.__probe || 0) + 1;
    const probe = window.__probe; let last = performance.now();
    const tick = (now) => { if (probe !== window.__probe) return;
        window.__gap = Math.max(window.__gap, now - last); window.__frames += 1; last = now; requestAnimationFrame(tick); };
    requestAnimationFrame(tick); }"""


def _measure_search(page, reads, words):
    """Types ``words`` into the open search and waits until its reading settles: what that took."""
    import time

    first = len(reads)
    page.evaluate(_LOOP_PROBE)
    began = time.monotonic()
    page.fill("#page-chat .chat-find-input", words)
    page.wait_for_function(_SEARCH_SETTLED, timeout=300_000)
    ended = time.monotonic()
    taken, state = reads[first:], _search_state(page, "#page-chat")
    loop = page.evaluate("() => { window.__probe += 1; return {tasks: window.__longTasks, gap: window.__gap, "
                         "frames: window.__frames}; }")
    tasks = loop["tasks"]
    return {"words": words, "count": state["count"], "listed": len(state["results"]), "pages": len(taken),
            "candidates": sum(read["candidates"] for read in taken),
            "response_bytes": sum(read["bytes"] for read in taken),
            "largest_page_bytes": max((read["bytes"] for read in taken), default=0),
            "page_arrivals_s": [round(read["at"] - began, 3) for read in taken],
            "seconds": round(ended - began, 3), "longest_task_ms": max(tasks, default=0) if tasks is not None else None,
            "long_tasks": len(tasks) if tasks is not None else None, "longest_frame_gap_ms": round(loop["gap"], 1),
            "frames": loop["frames"], "state": state}


# The one long message's words timed in the page: its bubble's own render, and its candidate drawn and read for words.
_HUGE_ROW = """async () => {
    const answer = await (await fetch('/api/chat/search?chat_id=1&q=reeds&probe=1')).json();
    let page = answer, huge = null;
    while (!huge && page) {
        huge = page.candidates.find((candidate) => candidate.text.length > 100000) || null;
        page = !huge && page.next_cursor ? await (await fetch(`/api/chat/search?chat_id=1&q=reeds&probe=1&cursor=${
            encodeURIComponent(page.next_cursor)}`)).json() : null;
    }
    const words = await import('/static/modules/chat_search_words.js');
    const { renderChatMarkdown } = await import('/static/modules/chat_markdown.js');
    const time = (work) => { const at = performance.now(); work(); return performance.now() - at; };
    const doc = document.implementation.createHTMLDocument('');
    const runs = [0, 1, 2].map(() => ({
        render_ms: time(() => renderChatMarkdown(huge.text)),
        words_ms: time(() => words.findIn(words.wordStream(words.wordPieces(words.candidateBox(huge, doc))),
            words.searchKey('reeds'))),
    }));
    const median = (key) => runs.map((run) => run[key]).sort((a, b) => a - b)[1];
    return { chars: huge.text.length, render_ms: median('render_ms'), words_ms: median('words_ms'), runs };
}"""


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_a_long_room_is_read_to_its_beginning_in_slices_and_esc_ends_the_reading(
        direct_server_with_data, browser_engine, tmp_path):
    import time

    from playwright.sync_api import sync_playwright

    _seed_long_room(direct_server_with_data["data_dir"])
    shots = _search_evidence(tmp_path)
    errors, reads = [], []
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1512, "height": 949}, device_scale_factor=2, color_scheme="dark")
        page.on("pageerror", lambda error: errors.append(str(error)))

        def answered(response):
            if "/api/chat/search?" in response.url and "probe=1" not in response.url:  # the page's own reads
                body = response.body()
                reads.append({"at": time.monotonic(), "bytes": len(body),
                              "candidates": len(json.loads(body).get("candidates", []))})
        page.on("response", answered)
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            page.fill("#chat-input", "my unsent draft")
            page.keyboard.press("ControlOrMeta+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")

            oldest = _measure_search(page, reads, "osprey")
            assert oldest["count"] == "1 found" and oldest["state"]["results"][0]["snippet"] == SEARCH_LONG_OLDEST, oldest
            owner_left_out = sum(1 for index in range(2000, SEARCH_LONG_ROWS) if index % 2)
            assert oldest["candidates"] == SEARCH_LONG_ROWS - owner_left_out and oldest["pages"] >= 3, \
                ("every message read once, in pages; the host leaves out only the owner's ASCII words without it", oldest)

            common = _measure_search(page, reads, "e")
            assert common["pages"] == 1 and common["count"] == f"{common['listed']}+" and common["listed"] >= 1900, common
            assert common["state"]["note"].endswith("Search older messages"), common
            page.screenshot(path=str(shots / f"{browser_engine}-long-room-common-letter.png"))

            dense = _measure_search(page, reads, "reeds")
            assert dense["pages"] >= 2 and dense["listed"] >= 500 and dense["count"].endswith("+"), dense

            huge = page.evaluate(_HUGE_ROW)
            assert huge["chars"] > 100_000, huge

            # Esc in the middle of a reading that would go on to the beginning.
            first = len(reads)
            page.fill("#page-chat .chat-find-input", "kingfisher")
            deadline = time.monotonic() + 60
            while len(reads) == first and time.monotonic() < deadline:
                page.wait_for_timeout(20)
            assert len(reads) > first, "the reading began"
            reading = _search_state(page, "#page-chat")
            assert reading["note"].startswith("Searching saved messages"), reading
            page.screenshot(path=str(shots / f"{browser_engine}-long-room-reading.png"))
            page.keyboard.press("Escape")
            closed, count = time.monotonic(), len(reads)
            page.wait_for_timeout(2000)
            late = [read for read in reads[count:] if read["at"] > closed + 0.3]
            assert not late, ("no read after Esc", late)
            assert sum(read["candidates"] for read in reads[first:]) < oldest["candidates"], "Esc ended it before the beginning"
            state = _search_state(page, "#page-chat")
            assert not state["open"] and state["activeId"] == "chat-input", state
            assert page.input_value("#chat-input") == "my unsent draft"
            strip = lambda measured: {key: value for key, value in measured.items() if key != "state"}  # noqa: E731
            measured = {"engine": browser_engine, "rows": SEARCH_LONG_ROWS, "oldest_match": strip(oldest),
                        "common_letter": strip(common), "dense_markdown": strip(dense), "one_long_message": huge,
                        "esc_mid_reading": {"pages_before_esc": len(reads) - first,
                                            "candidates_before_esc": sum(read["candidates"] for read in reads[first:]),
                                            "note_while_reading": reading["note"], "count_while_reading": reading["count"]}}
            (shots / f"{browser_engine}-long-room-measured.json").write_text(json.dumps(measured, indent=2))
        finally:
            browser.close()
    assert not errors, errors


@pytest.mark.parametrize("browser_engine", ["webkit", "chromium"])
def test_two_rooms_searching_side_by_side_keep_their_own_marks(direct_server_with_data, browser_engine, tmp_path):
    from playwright.sync_api import sync_playwright

    room = _seed_search_rooms(direct_server_with_data["data_dir"])
    errors = []
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1512, "height": 949}, device_scale_factor=2)
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            page.locator("#chat-input").focus()
            page.keyboard.press("ControlOrMeta+f")
            page.keyboard.type("kingfisher")
            _wait_search_results(page, "#page-chat", 4)
            page.keyboard.press("Enter")
            _search_landed(page, "#page-chat", "Remember the kingfisher photo?")
            assert _search_marks(page, "#page-chat")["count"] == 1
            # The Project docked beside Main, which stays live with its search open and its marks shown.
            room_browser._open(page, room["id"])
            panel, feed = f'[id="panel-pchat-{room["id"]}"]', f'[id="pchat-{room["id"]}-messages"]'
            page.locator(feed).click(position={"x": 40, "y": 200})
            page.keyboard.press("ControlOrMeta+f")
            page.keyboard.type("kingfisher survey")
            _wait_search_results(page, panel, 2)
            page.keyboard.press("Enter")
            _search_landed(page, panel, "Kingfisher survey route is ready.")
            assert _search_state(page, "#page-chat")["open"], "Main beside a docked Project keeps its search"
            assert (_search_marks(page, "#page-chat")["count"], _search_marks(page, panel)["count"]) == (1, 1)
            page.fill(f"{panel} .chat-find-input", "kingfisher surveys")  # new words drop only the Project's marks
            _wait_search_results(page, panel, 0)
            assert (_search_marks(page, "#page-chat")["count"], _search_marks(page, panel)["count"]) == (1, 0)
            page.keyboard.press("Escape")
            assert not _search_state(page, panel)["open"]
            assert _search_marks(page, "#page-chat")["count"] == 1, "closing the Project's search keeps Main's marks"
        finally:
            browser.close()
    assert not errors, errors


# The room's first message, and a room of one message, land between the open sheet and the composer: while the search
# is open the feed's top clears the sheet as it clears the room's bar, so the first row can be placed below it; closed,
# that room is gone and the row stands under the bar again. A resize, a close and a reopen keep the landing.
SEARCH_ONLY = "The only heron note in this room."
_SEARCH_TOP = """scope => {
    const room = document.querySelector(scope), feed = room.querySelector('.chat-messages, #chat-messages');
    return {scrollTop: feed.scrollTop, padding: parseFloat(getComputedStyle(feed).paddingTop),
            reserve: feed.style.getPropertyValue('--chat-find-reserve'),
            first: feed.querySelector('[data-history-id]')?.getBoundingClientRect().top
                - feed.getBoundingClientRect().top};
}"""


def _land_and_measure(page, words, marked):
    page.keyboard.press("Enter")
    page.wait_for_function("scope => (" + _SEARCH_MARKS + ")(scope).count > 0", arg="#page-chat", timeout=30_000)
    page.evaluate(_SETTLE_RESTORE_FRAMES)
    marks = _search_marks(page, "#page-chat")
    assert marks["count"] == 1 and marks["text"] == marked, (words, marks)
    assert marks["sheet"] - 1 <= marks["top"] and marks["bottom"] <= marks["composer"] + 1, (words, marks)
    return marks


@pytest.mark.parametrize("viewport", [(1512, 949), (390, 844)], ids=["desktop", "phone"])
@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
@pytest.mark.parametrize("room", ["one", "first"])
def test_search_lands_the_rooms_first_message_below_the_sheet(direct_server_with_data, browser_engine, viewport, room,
                                                              tmp_path):
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data["data_dir"]
    if room == "one":
        ds.append(root, "chat", {"chat_id": 1, "ts": SEARCH_START.isoformat(), "direction": "in", "text": SEARCH_ONLY},
                  feed=True)
        words, marked = "heron note", "heron note"
    else:
        _seed_search_rooms(root)  # four hundred messages: the first, "Routine note 0", is far older than the first page
        words, marked = "Routine note 0:", "Routine note 0:"
    shots, (width, height) = _search_evidence(tmp_path), viewport
    name = f"{browser_engine}-{room}-{width}x{height}"
    errors = []
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2,
                                color_scheme="dark", has_touch=width < 600)
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            closed = page.evaluate(_SEARCH_TOP, "#page-chat")
            page.fill("#chat-input", "my unsent draft")
            page.keyboard.press("ControlOrMeta+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            page.keyboard.type(words)
            _wait_search_results(page, "#page-chat", 1)
            first = _land_and_measure(page, words, marked)
            page.screenshot(path=str(shots / f"{name}-landed.png"))
            state = _search_state(page, "#page-chat")
            assert state["focused"] and state["count"] == "1 of 1", state

            # A narrower or shorter window re-measures the sheet: the landed words stay between it and the composer.
            page.set_viewport_size({"width": max(360, width - 300), "height": height - 120})
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            resized = _search_marks(page, "#page-chat")
            assert resized["count"] == 1 and resized["sheet"] - 1 <= resized["top"], resized
            assert resized["bottom"] <= resized["composer"] + 1, resized
            page.screenshot(path=str(shots / f"{name}-resized.png"))
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate(_SETTLE_RESTORE_FRAMES)

            # Closed, the search leaves no room above the feed: the first row stands under the bar again.
            page.keyboard.press("Escape")
            page.evaluate(_SETTLE_RESTORE_FRAMES)
            after = page.evaluate(_SEARCH_TOP, "#page-chat")
            assert not after["reserve"] and abs(after["padding"] - closed["padding"]) <= 0.5, (closed, after)
            assert page.input_value("#chat-input") == "my unsent draft"
            assert page.evaluate("document.activeElement.id") == "chat-input"
            if room == "one":
                assert abs(after["first"] - closed["first"]) <= 1, (closed, after)
            page.screenshot(path=str(shots / f"{name}-closed.png"))

            # Reopened, the kept words are read again and the same row lands below the sheet again.
            page.keyboard.press("ControlOrMeta+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            _wait_search_results(page, "#page-chat", 1)
            again = _land_and_measure(page, words, marked)
            assert abs(again["top"] - first["top"]) <= 2, (first, again)
            page.keyboard.press("Escape")
        finally:
            browser.close()
    assert not errors, errors


# A gallery is one bubble whose photos are each their own saved message (chat_media.js groups a work's consecutive
# photos, and files, into the first one's bubble). A match lands on its own photo's caption, never the next one's, and
# is never said to be hidden: each caption alone, and a query both captions hold, landing on each in turn.
SEARCH_GALLERY = [("photo", "Grey heron at dawn on the weir."), ("photo", "Grey heron at dusk by the alders."),
                  ("document", "Grey heron count sheet for dawn."), ("document", "Grey heron count sheet for dusk.")]
_SEARCH_MARKED_ITEMS = """scope => {
    const feed = document.querySelector(scope).querySelector('.chat-messages, #chat-messages');
    const marks = CSS.highlights?.get('ouro-chat-find');
    const ranges = marks ? [...marks].filter(range => feed.contains(range.startContainer)) : [];
    return ranges.map(range => {
        const item = range.startContainer.parentElement.closest('.chat-gallery-item, .chat-file-item');
        return {text: range.toString(), item: item?.dataset.historyId || null,
                caption: item?.querySelector('figcaption, .chat-file-caption')?.textContent || null,
                grouped: (item?.closest('.chat-bubble')?.querySelectorAll('.chat-gallery-item, .chat-file-item')
                    .length || 0) > 1};
    });
}"""


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_search_lands_on_the_matching_photo_or_file_of_a_group(direct_server_with_data, browser_engine, tmp_path):
    import hashlib

    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    from ouroboros.artifacts import store_task_artifact_bytes

    root = direct_server_with_data["data_dir"]
    image = base64.b64decode(SEARCH_PNG)
    image_name = f"chat-media-{hashlib.sha256(image).hexdigest()}.png"
    store_task_artifact_bytes(root, "heron-gallery", image_name, image)
    for index in range(30):  # the room's earlier words
        ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(minutes=index)).isoformat(),
                                  "direction": "in" if index % 2 else "out", "text": f"Routine note {index}."},
                  feed=True)
    for index, (kind, caption) in enumerate(SEARCH_GALLERY):
        row = {"chat_id": 1, "ts": (SEARCH_START + timedelta(hours=1, seconds=index)).isoformat(), "direction": "out",
               "type": kind, "task_id": "heron-gallery", "text": caption, "caption": caption}
        if kind == "photo":
            row.update(mime="image/png", download_url=f"/api/tasks/heron-gallery/artifacts/{image_name}")
        else:
            row.update(mime="text/plain", filename=f"heron-count-{index}.txt", size_bytes=12)
        ds.append(root, "chat", row, feed=True)
    ds.append(root, "chat", {"chat_id": 1, "ts": (SEARCH_START + timedelta(hours=2)).isoformat(), "direction": "in",
                              "text": "Newest owner message."}, feed=True)
    shots = _search_evidence(tmp_path)
    errors, landed = [], {}
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1512, "height": 949}, device_scale_factor=2, color_scheme="dark")
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            _open_main_for_search(page, direct_server_with_data["url"])
            grouped = page.evaluate("""() => [...document.querySelectorAll('#chat-messages .chat-bubble')]
                .map(bubble => bubble.querySelectorAll('.chat-gallery-item, .chat-file-item').length)
                .filter(n => n > 1)""")
            assert grouped == [2, 2], ("the photos and the files are each one group", grouped)
            page.keyboard.press("ControlOrMeta+f")
            page.wait_for_function("() => document.activeElement?.classList.contains('chat-find-input')")
            dawn, dusk, dawn_sheet, dusk_sheet = (caption for _, caption in SEARCH_GALLERY)
            for words, captions in (("heron at dawn", [dawn]), ("heron at dusk", [dusk]),
                                    ("sheet for dawn", [dawn_sheet]), ("sheet for dusk", [dusk_sheet]),
                                    # both of a group hold it: newest first, each landing marks only its own caption
                                    ("grey heron at", [dusk, dawn]), ("grey heron count", [dusk_sheet, dawn_sheet])):
                state = _search_for(page, words, len(captions))
                # a photo is its caption; a file its name, then its caption
                assert [result["snippet"].endswith(caption) for result, caption in zip(state["results"], captions)] \
                    == [True] * len(captions), (words, state)
                for at, caption in enumerate(captions):
                    page.keyboard.press("Enter")
                    try:
                        page.wait_for_function("""([scope, caption]) => (""" + _SEARCH_MARKED_ITEMS + """)(scope)
                            .some(mark => mark.caption === caption)""", arg=["#page-chat", caption], timeout=30_000)
                    except PlaywrightTimeout:  # what the landing marked and said instead
                        raise AssertionError((words, at, caption, page.evaluate(_SEARCH_MARKED_ITEMS, "#page-chat"),
                                              _search_state(page, "#page-chat"))) from None
                    page.evaluate(_SETTLE_RESTORE_FRAMES)
                    marks = page.evaluate(_SEARCH_MARKED_ITEMS, "#page-chat")
                    assert [mark["caption"] for mark in marks] == [caption] and marks[0]["grouped"], (words, at, marks)
                    assert marks[0]["text"].casefold() == words.casefold(), (words, marks)
                    state = _search_state(page, "#page-chat")
                    assert "not shown here" not in state["note"], (words, state)
                    assert "cannot be shown" not in state["note"], (words, state)
                    assert state["count"] == f"{at + 1} of {len(captions)}" and state["focused"], (words, state)
                    placed = _search_marks(page, "#page-chat")
                    assert placed["sheet"] - 1 <= placed["top"] and placed["bottom"] <= placed["composer"] + 1, placed
                    landed[f"{words}#{at}"] = marks
                page.screenshot(path=str(shots / f"{browser_engine}-group-{words.replace(' ', '-')}.png"))
            page.keyboard.press("Escape")
        finally:
            (shots / f"{browser_engine}-group-marks.json").write_text(json.dumps(landed, indent=2))
            browser.close()
    assert not errors, errors


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_selected_passage_lands_on_second_group_item_from_project(direct_server_with_data, browser_engine):
    from playwright.sync_api import sync_playwright

    root = direct_server_with_data["data_dir"]
    project = create_project(root, "group-quotes", name="Group quotes", origin="owner_ui")
    sources = [{"chat_id": 1, "ts": f"2026-10-01T12:00:0{i}Z", "direction": "out", "type": "document",
                "task_id": "field-notes", "filename": f"{stem}.txt", "caption": f"The {stem} pier", "text": f"The {stem} pier"}
               for i, stem in enumerate(("south", "north"))]
    for source in sources:
        ds.append(root, "chat", source, feed=True)
    address = format_address(row_address(sources[1]))
    excerpt = {"text": "north", "start_utf16": 4}
    ds.append(root, "chat", {"chat_id": project["chat_id"], "ts": "2026-10-01T12:01:00Z", "direction": "out",
                              "text": "Read the north note.", "reply_to": {**reference_from_row(sources[1]), "excerpt": excerpt}}, feed=True)
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 850})
            page.add_init_script(f"({_OBSERVE_QUOTES})()")
            _open(page, direct_server_with_data["url"])
            room_browser._open(page, project["id"])
            _quote(page, address, "#pchat-group-quotes-messages").click()
            _landed(page, address, "#chat-messages")
            page.wait_for_function("""address => {
                const target = document.querySelector('#chat-messages .chat-file-item.is-quote-target');
                const r = [...CSS.highlights.get('reply-excerpt') || []][0];
                return target?.dataset.rowAddress === address && target.closest('.is-multiple')
                    && target.querySelector('.chat-file-name').textContent === 'north.txt'
                    && target.querySelector('.chat-file-caption').contains(r?.startContainer)
                    && r.toString() === 'north';
            }""", arg=address)
            assert page.evaluate("address => window.__quoteReads.some(r => r.address === address && r.body?.anchor?.state === 'found')", address)
        finally:
            browser.close()


@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_selected_reply_keeps_second_rendered_passage_through_reload(direct_server_with_data, browser_engine):
    from playwright.sync_api import expect, sync_playwright

    phrase = "Silver gate"
    row = {"chat_id": 1, "direction": "out", "ts": "2026-10-01T12:00:00+00:00",
           "text": f"🌙 **{phrase}** then [{phrase}](https://example.invalid/atlas)."}
    ds.append(direct_server_with_data["data_dir"], "chat", row, feed=True)
    with sync_playwright() as pw:
        browser = getattr(pw, browser_engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 750})
        try:
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            page.locator("#chat-input").fill("Keep the night route.")
            # Native Selection spans the second occurrence. The subsequent real button press includes
            # the shell's document-capture pointerdown and the browser's usual mousedown default.
            link = page.locator("#chat-messages .message a", has_text=phrase)
            link.evaluate("""node => {
                const range = document.createRange();
                range.selectNodeContents(node);
                const selection = getSelection();
                selection.removeAllRanges(); selection.addRange(range);
            }""")
            action = page.get_by_role("button", name="Reply to quote", exact=True)
            action.wait_for()
            assert link.evaluate("""link => {
                const r = getSelection().getRangeAt(0).getBoundingClientRect();
                link.dispatchEvent(new PointerEvent('pointerdown', {bubbles:true, button:2,
                    clientX:r.left+r.width/2, clientY:r.top+r.height/2}));
                const api = window.ouroReplyToQuote;
                const offered = api.available('menu-A');
                const refused = !api.activate('other-menu');
                const saved = getSelection().getRangeAt(0).cloneRange();
                getSelection().removeAllRanges();
                const stale = !api.activate('menu-A');
                link.dispatchEvent(new PointerEvent('pointerdown', {bubbles:true, button:0}));
                getSelection().addRange(saved);
                return offered && refused && stale;
            }""")
            assert page.evaluate("() => sessionStorage.getItem('ouro_chat_reply')") is None
            action.click()
            strip = page.locator(".chat-quote--composer[data-excerpt]")
            expect(strip.locator(".chat-quote-text")).to_have_text(phrase, use_inner_text=True)
            kept = page.evaluate("() => JSON.parse(sessionStorage.getItem('ouro_chat_reply'))")
            assert kept["address"] == format_address(row_address(row))
            assert kept["excerpt"] == {"text": phrase, "start_utf16": len("🌙 Silver gate then ".encode("utf-16-le")) // 2}
            page.reload(wait_until="domcontentloaded")
            strip.wait_for()
            assert page.locator("#chat-input").input_value() == "Keep the night route."
            strip.locator(".chat-quote-main").click()
            page.wait_for_function("""() => {
                const range = [...(CSS.highlights?.get('reply-excerpt') || [])][0] || (getSelection()?.rangeCount ? getSelection().getRangeAt(0) : null);
                return !!range?.startContainer.parentElement.closest('a');
            }""")
            strip.get_by_role("button", name="Cancel reply").click()
            assert page.locator("#chat-input").input_value() == "Keep the night route."
        finally:
            browser.close()


@pytest.mark.parametrize("room,direction", [("main", "in"), ("main", "out"), ("project", "in"), ("project", "out")])
def test_selected_passage_round_trip(direct_server_with_data, room, direction):
    """Real mouse, WS acceptance, stored owner row and the reloaded room; only the fixture's LLM is scripted."""
    from playwright.sync_api import sync_playwright

    root, url = direct_server_with_data["data_dir"], direct_server_with_data["url"]
    project = create_project(root, "night-atlas", name="Night atlas", origin="owner_ui") if room == "project" else None
    chat = project["chat_id"] if project else 1
    prefix = f"pchat-{project['id']}" if project else "chat"
    passage = ("The violet beacon above the winter quay guides the little survey boat past the shoals. "
               "A chalk star beside the eastern stair marks the tide gauge that the night crew should read.")
    opening = f"🌙 {passage} Again, "
    row = {"chat_id": chat, "direction": direction, "text": opening + passage + " At dawn.", "ts": "2026-10-07T12:00:00Z"}
    ds.append(root, "chat", row, feed=True)
    address = format_address(row_address(row))
    new_words = "Please explain the second observation."
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 750})
        try:
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function("() => window.__testSockets.some(s => s.readyState === WebSocket.OPEN)")
            if project:
                room_browser._open(page, project["id"])
            feed = page.locator(f"#{prefix}-messages")
            body = feed.locator(f'.chat-bubble[data-row-address={json.dumps(address)}] .message[data-reply-body]')
            body.scroll_into_view_if_needed()
            # Read glyph edges to aim physical pointer events. Never program the browser's Selection.
            points = body.evaluate("""(body, passage) => {
                const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT);
                let node;
                while ((node = walker.nextNode())) {
                    const start = node.data.lastIndexOf(passage);
                    if (start < 0) continue;
                    const glyph = (i, end) => {
                        const r = document.createRange();
                        r.setStart(node, i); r.setEnd(node, i + 1);
                        const box = r.getBoundingClientRect();
                        return [end ? box.right - 0.2 : box.left + 0.2, box.top + box.height / 2];
                    };
                    return [glyph(start, false), glyph(start + passage.length - 1, true)];
                }
            }""", passage)
            assert points
            page.mouse.move(*points[0])
            page.mouse.down()
            page.mouse.move(*points[1], steps=18)
            page.mouse.up()
            assert page.evaluate("getSelection().toString()") == passage
            page.get_by_role("button", name="Reply to quote", exact=True).click()
            page.locator(f"#{prefix}-input").fill(new_words)
            page.locator(f"#{prefix}-send").click()
            saved = feed.locator('.chat-bubble.user[data-row-address]').filter(has_text=new_words)
            saved.wait_for(timeout=30_000)
            cmid = saved.get_attribute("data-client-message-id")
            accepted = ds.accepted_row(root, chat, cmid)
            assert accepted and accepted.row["text"] == new_words
            excerpt = {"text": passage, "start_utf16": len(opening.encode("utf-16-le")) // 2}
            ref = {**reference_from_row(row), "excerpt": excerpt}
            assert accepted.row["reply_to"] == ref
            page.reload(wait_until="domcontentloaded")
            if project and not feed.is_visible():
                room_browser._open(page, project["id"])
            saved = feed.locator(f'.chat-bubble.user[data-row-address={json.dumps(format_address(row_address(accepted.row)))}]')
            quote = saved.locator(".chat-quote[data-excerpt]")
            jump, expand = quote.locator(".chat-quote-main"), quote.get_by_role("button", name="Show all", exact=True)
            expand.wait_for(state="visible")
            target_box, expand_box = jump.bounding_box(), expand.bounding_box()
            assert target_box and expand_box
            assert expand_box["x"] - (target_box["x"] + target_box["width"]) >= 7.5  # --space-2, as in the composer
            expand.click()
            assert quote.locator(".chat-quote-text").text_content() == passage
            jump.click()
            page.wait_for_function("""([address, expected, start]) => {
                const r = [...(CSS.highlights?.get('reply-excerpt') || [])][0]
                    || (getSelection().rangeCount ? getSelection().getRangeAt(0) : null);
                return r?.toString() === expected && r.startOffset === start
                    && r.startContainer.parentElement.closest('.chat-bubble')?.dataset.rowAddress === address;
            }""", arg=[address, passage, excerpt["start_utf16"]])
        finally:
            browser.close()

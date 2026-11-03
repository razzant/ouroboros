"""One engine-keyed utterance in the served SPA, live, after a reconnect and after a reload.

Two runs of one task card stream their engine-keyed words through the real producer
(``delegate_progress.emit`` over the fixture daemon's event stream), the real progress
transport and the real history endpoint (``tests/test_delegated_activity_browser.py``'s
harness). The first run's paragraph streams across three polls, one utterance in its first
row while it streams, and completes; a retried draft is abandoned a poll after it began;
equal words come under two keys; a same-length correction completes a draft and the run's
final answer repeats it; the source link keeps its contrast in both themes (``measure.js``).
The second run says only its final answer. A second case serves the engine's own normalized
journals of four recorded CLI streams (``tests/fixtures/claudexor_speech``) verbatim, each
cut inside its first streamed block and two thirds in, into one card.
Screenshots (1110 light and dark, 390) and the DOM go to ``OUROBOROS_UI_EVIDENCE_DIR``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.history import make_chat_history_endpoint
from tests.test_delegated_activity import Daemon, _gateway, _harness, _run_event
from tests.test_delegated_activity_browser import ROOT, SNAPSHOT, Producer, _evidence
from tests.test_delegated_activity_browser import engine_ui as engine_ui
from tests.test_delegated_activity_speech import (
    K1,
    K2,
    LOST,
    PARAGRAPH,
    SCENARIOS,
    Lossy,
    Verbatim,
    _abandoned,
    _final,
    _pieces,
    _said,
    _speech_facts,
    _stream,
    _tool,
    engine_journal,
)
from tests.test_reconnect_feed_browser import HISTORY_LANDED
from tests.test_subscription_setup_browser import capture
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui
from tests.test_task_file_serving import _client

MEASURE = Path(__file__).resolve().parents[1] / "skills" / "ui_design" / "references" / "measure.js"
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
AGAIN = "Re-running the import tests."
SOLE = "The feed validates: 0 errors and 1 warning (no ttl element), which readers ignore."
K3, K4, K5 = "claude:ses-1:3:0", "claude:ses-1:3:2", "claude:ses-1:4:0"
EXPAND = """id => { const card = document.querySelector(`.chat-live-card[data-task-id="${id}"]`);
    if (card && card.dataset.expanded !== '1') card.querySelector('[data-live-disclosure]').click(); }"""


def _journals():
    pieces = _pieces(PARAGRAPH, 4)
    say = Daemon([
        [_harness(1, "tool_call", tool={"name": "Read", "target": "blog/rss_import.py"}),
         _harness(2, "tool_result", tool={"name": "Read", "status": "ok"}), *_stream(3, pieces[:2], K1, complete=False)],
        [_said(5, pieces[2], K1, "fragment")],
        [_said(6, pieces[3], K1, "fragment"), _said(7, PARAGRAPH, K1, "complete"),
         _harness(8, "tool_call", tool={"name": "Edit"}), _said(9, "Checking", K2, "fragment")],
        [_said(10, " the", K2, "fragment"), _abandoned(11, K2),
         _said(12, AGAIN, K3, "complete"), _said(13, AGAIN, K4, "complete")],
        [*_stream(14, ["teh", " cat."], K5, complete="the cat."), _final(17, "the cat."), _run_event(18, "run.completed")],
    ], run_id="run-say")
    check = Daemon([[_final(1, SOLE), _run_event(2, "run.completed")]], run_id="run-check")
    return say, check


def _look(page, name):
    """The screen, the timeline at 1:1 and what ``measure.js`` reads off it after paint (returned)."""
    capture(page, name)
    timeline = page.locator(f"#chat-live-timeline-{ROOT}")
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        timeline.screenshot(path=str(Path(root) / f"{name}-timeline.png"), animations="disabled")
    measured = page.evaluate(f"({MEASURE.read_text(encoding='utf-8')})({json.dumps({'subject': f'#chat-live-timeline-{ROOT}'})})")
    _evidence(f"{name}-measure.json", measured)
    return measured


def _low_contrast_sources(measured):
    """``measure.js`` findings of a delegated source link below its WCAG floor."""
    return [finding for finding in measured["findings"]
            if finding["kind"] == "contrast-low" and "chat-delegated-source" in finding["where"]]


def _rows(page):
    """Each executor row's reading text (a folded answer's words are not read) and its markup facts."""
    return page.evaluate("""id => Array.from(document.querySelectorAll(`#chat-live-timeline-${id} > .chat-live-line`))
        .filter(line => !line.hidden && line.querySelector('.chat-delegated-activity')).map(line => ({
            text: line.innerText.replace(/\\s+/g, ' ').trim(),
            answers: Array.from(line.querySelectorAll('details.chat-delegated-answer')).map(d => d.open ? 'open' : 'folded'),
            notes: Array.from(line.querySelectorAll('.chat-delegated-message .chat-delegated-note')).map(n => n.textContent.trim()),
        }))""", ROOT)


def test_one_keyed_utterance_live_reconnect_reload_with_updated_abandoned_and_answers(engine_ui, tmp_path, monkeypatch):
    ui = engine_ui
    page = ui["page"]
    page.set_viewport_size({"width": 1110, "height": 750})
    page.emulate_media(color_scheme="light")         # the client's System theme falls back to dark
    data = tmp_path / "data"
    producer = Producer(data, monkeypatch)
    history = TestClient(Starlette(routes=[Route("/api/chat/history", endpoint=make_chat_history_endpoint(data))]))
    ui["backend"]["task_gateway"] = _client(data)
    served, sockets, frames = [], [], []

    def history_route(route):
        url = route.request.url
        response = history.get(url[url.index("/api/chat/history"):])
        served.append(response.json())
        route.fulfill(status=response.status_code, content_type="application/json", body=response.text)

    def socket_route(ws):
        sockets.append(ws)
        ws.send(json.dumps({"type": "heartbeat"}))

    page.route("**/api/state", lambda route: route.fulfill(content_type="application/json", body=json.dumps(
        {"supervisor_ready": True, "active_chat_activities": [], "projects": [], "sha": "cand1698"})))
    page.route("**/api/chat/history*", history_route)
    page.route("**/api/tasks/**", lambda route: route.continue_())
    page.route_web_socket("**/ws", socket_route)
    page.goto(ui["url"])
    page.wait_for_selector("#chat-input")

    def push(payloads):
        for payload in payloads:
            frames.append(payload)
            sockets[-1].send(json.dumps(payload))

    say, check = _journals()
    pieces = _pieces(PARAGRAPH, 4)
    worker = producer.worker(ROOT, None)
    push(producer.say(worker, "Fixing the duplicate posts: Claude runs it, I watch the run."))
    say.revealed = 1
    push(producer.observe(worker, _gateway(say), "run-say", 4, after=0))
    card = page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first
    card.wait_for()
    page.evaluate(EXPAND, ROOT)
    page.wait_for_function(f"() => document.querySelector('#chat-live-timeline-{ROOT} .chat-delegated-message')")
    say.revealed = 2
    push(producer.observe(worker, _gateway(say), "run-say", 5, after=4))
    streamed = "".join(pieces[:3])
    page.wait_for_function(f"""words => Array.from(document.querySelectorAll('#chat-live-timeline-{ROOT} .chat-delegated-message'))
        .some(m => m.textContent.replace(/\\s+/g, ' ').includes(words))""", arg=" ".join(streamed.split()), timeout=5000)
    draft = _rows(page)
    assert len(draft) == 1 and " ".join(streamed.split()) in draft[0]["text"] and PARAGRAPH not in draft[0]["text"], \
        "two polls of one draft read as one utterance in the row where it began"
    capture(page, "1698-draft-streaming")

    for step, (through, after) in enumerate(((9, 5), (13, 9)), start=3):
        say.revealed = step
        push(producer.observe(worker, _gateway(say), "run-say", through, after=after))
    say.revealed = 5
    push(producer.observe(worker, _gateway(say), "run-say", 18, after=13, drain=30))
    check.revealed = 1
    push(producer.observe(worker, _gateway(check), "run-check", 2, after=0, drain=30))
    page.wait_for_function(f"() => document.querySelectorAll('#chat-live-timeline-{ROOT} details.chat-delegated-answer').length === 2")
    rows = _rows(page)
    assert len(rows) == 5, rows
    first, abandoned, again, third, sole = rows
    whole = " ".join(row["text"] for row in rows)
    assert whole.count(PARAGRAPH) == 1 and PARAGRAPH in first["text"], "the completed paragraph replaced its draft in place"
    assert "2 technical events since seq 1" in first["text"]
    assert whole.count("Checking the") == 1 and abandoned["notes"] == ["Abandoned: the executor never completed these words."], \
        "a draft abandoned in a later poll is one utterance, marked once"
    assert again["text"].count(AGAIN) == 2, "equal words under two keys are two utterances"
    assert third["notes"] == ["Updated. The streamed draft is in the full source."], "a same-length correction is visible"
    assert third["answers"] == ["folded"] and third["text"].count("the cat.") == 1, "the answer folds after the run's words"
    assert sole["answers"] == ["open"] and SOLE in sole["text"] and "run-check" in sole["text"]
    assert not _low_contrast_sources(_look(page, "1698-live-1110-light"))
    with page.expect_download() as download:
        card.locator(".chat-delegated-note a.chat-delegated-source", has_text="the full source").first.click()
    body = Path(download.value.path()).read_bytes()
    assert download.value.failure() is None
    retained = [json.loads(line) for line in body.decode().splitlines()]
    assert [(row["seq"], row["payload"].get("text")) for row in retained] == [
        (14, "teh"), (15, " cat."), (16, "the cat."), (17, "the cat."), (18, None)], "the streamed draft stays in the source"
    answer = card.locator("details.chat-delegated-answer").first
    answer.locator("summary").click()
    assert answer.locator(".chat-delegated-message").inner_text().strip() == "the cat."
    capture(page, "1698-answer-opened")
    answer.locator("summary").click()
    live = page.evaluate(SNAPSHOT, [ROOT])
    page.emulate_media(color_scheme="dark")
    link = card.locator(".chat-delegated-note", has_text="Updated").first.locator("a.chat-delegated-source")
    link.scroll_into_view_if_needed()
    box = link.bounding_box()
    assert box and 0 <= box["y"] and box["y"] + box["height"] <= 750, "measure.js reads only what is on screen"
    dark = _look(page, "1698-live-1110-dark")
    assert not _low_contrast_sources(dark), "the Updated note's source link keeps its contrast on the dark Working card"
    page.emulate_media(color_scheme="light")

    fetched = len(served)
    page.evaluate("() => { window.__live1698 = true; }")
    page.evaluate(HISTORY_LANDED)
    sockets[-1].close()
    for _ in range(100):
        if len(sockets) >= 2 and len(served) > fetched:
            break
        page.wait_for_timeout(100)
    assert len(sockets) >= 2 and len(served) > fetched
    page.wait_for_function("() => window.historyLanded > 0")
    page.wait_for_timeout(500)
    assert page.evaluate("() => window.__live1698 === true")
    page.evaluate(EXPAND, ROOT)
    reconnected = page.evaluate(SNAPSHOT, [ROOT])
    assert reconnected == live, "a reconnect adopts the live rows"

    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first.wait_for()
    page.evaluate(EXPAND, ROOT)
    page.wait_for_function(f"() => document.querySelectorAll('#chat-live-timeline-{ROOT} details.chat-delegated-answer').length === 2")
    assert _rows(page) == rows, "history replay presents the same utterances, notes and answers"
    page.set_viewport_size({"width": 390, "height": 844})
    page.locator(f"#chat-live-timeline-{ROOT} .chat-delegated-note", has_text="Updated").first.scroll_into_view_if_needed()
    assert not _low_contrast_sources(_look(page, "1698-reload-390"))
    _evidence("1698-frames.json", frames)
    _evidence("1698-history-pages.json", served)
    _evidence("1698-rows.json", {"draft": draft, "live": rows, "snapshot": live, "reconnected": reconnected})
    assert not ui["errors"].unexpected(), ui["errors"].report()


# What each executor row shows: its run, its readable words (a folded answer's are not read), each
# message's notes, and every Answer disclosure with its words.
SPEECH = """id => Array.from(document.querySelectorAll(`#chat-live-timeline-${id} > .chat-live-line`))
    .filter(line => !line.hidden && line.querySelector('.chat-delegated-activity')).map(line => {
        const words = (el) => { const copy = el.cloneNode(true);
            copy.querySelectorAll('.chat-delegated-note, .chat-delegated-actor').forEach(n => n.remove());
            return copy.textContent.trim(); };
        return {
            title: line.querySelector('.chat-live-line-title')?.textContent.trim() || '',
            messages: Array.from(line.querySelectorAll('.chat-delegated-message'))
                .filter(m => !m.closest('details.chat-delegated-answer'))
                .map(m => ({ text: words(m), notes: Array.from(m.querySelectorAll('.chat-delegated-note')).map(n => n.textContent.trim()) })),
            answers: Array.from(line.querySelectorAll('details.chat-delegated-answer'))
                .map(d => ({ open: d.open, text: words(d.querySelector('.chat-delegated-message')) })),
        };
    })"""


def test_the_engines_four_journals_in_one_served_card_cut_into_polls(engine_ui, tmp_path, monkeypatch):
    ui = engine_ui
    page = ui["page"]
    page.set_viewport_size({"width": 1110, "height": 750})
    page.emulate_media(color_scheme="light")
    data = tmp_path / "data"
    producer = Producer(data, monkeypatch)
    history = TestClient(Starlette(routes=[Route("/api/chat/history", endpoint=make_chat_history_endpoint(data))]))
    ui["backend"]["task_gateway"] = _client(data)
    served, sockets, frames = [], [], []

    def history_route(route):
        url = route.request.url
        response = history.get(url[url.index("/api/chat/history"):])
        served.append(response.json())
        route.fulfill(status=response.status_code, content_type="application/json", body=response.text)

    def socket_route(ws):
        sockets.append(ws)
        ws.send(json.dumps({"type": "heartbeat"}))

    page.route("**/api/state", lambda route: route.fulfill(content_type="application/json", body=json.dumps(
        {"supervisor_ready": True, "active_chat_activities": [], "projects": [], "sha": "cand1698"})))
    page.route("**/api/chat/history*", history_route)
    page.route("**/api/tasks/**", lambda route: route.continue_())
    page.route_web_socket("**/ws", socket_route)
    page.goto(ui["url"])
    page.wait_for_selector("#chat-input")

    def push(payloads):
        for payload in payloads:
            frames.append(payload)
            sockets[-1].send(json.dumps(payload))

    worker = producer.worker(ROOT, None)
    push(producer.say(worker, "Watching four recorded Claude runs."))
    journals = {name: engine_journal(name) for name in SCENARIOS}
    for journal in journals.values():
        count = len(journal.events)
        # A poll ends inside the first streamed block that has two fragments, and another two thirds in.
        split = next(event["seq"] for event, after in zip(journal.events, journal.events[1:])
                     if (event["payload"].get("speech") or {}).get("op") == "fragment"
                     and after["payload"].get("speech") == event["payload"]["speech"])
        daemon = Verbatim(journal, tuple(sorted({split, 2 * count // 3})))
        gateway, after = _gateway(daemon), 0
        for step, bound in enumerate(daemon.bounds, start=1):
            daemon.revealed = step
            last = step == len(daemon.bounds)
            push(producer.observe(worker, gateway, daemon.run_id, bound, after=after, drain=30 if last else None))
            after = bound
    card = page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first
    card.wait_for()
    page.evaluate(EXPAND, ROOT)
    page.wait_for_function(f"() => document.querySelectorAll('#chat-live-timeline-{ROOT} details.chat-delegated-answer').length === 4")
    live = page.evaluate(SPEECH, ROOT)
    for name, journal in journals.items():
        run = journal.events[0]["run_id"]
        rows = [row for row in live if row["title"].endswith(run)]
        messages = [message for row in rows for message in row["messages"]]
        texts = [message["text"] for message in messages]
        keys, abandoned, unkeyed = _speech_facts(journal)
        utterances = [facts["draft"] if facts["complete"] is None else facts["complete"] for facts in keys.values()]
        assert sorted(texts) == sorted(words.strip() for words in [*utterances, *unkeyed]), \
            (name, texts, "one message per key, its complete text or else its whole draft, and each unkeyed one")
        marked = [message["text"] for message in messages if "Abandoned: the executor never completed these words." in message["notes"]]
        assert marked == [keys[key]["draft"].strip() for key in abandoned], (name, messages)
        assert not any(note.startswith("Updated") for message in messages for note in message["notes"])
        answers = [answer for row in rows for answer in row["answers"]]
        assert answers == [{"open": False, "text": journal.answer["text"]}], "folded after the run's words, whole inside"
        said = [frame for frame in frames if (frame.get("delegated_activity") or {}).get("run_id") == run]
        plain = [str(frame.get("content") or frame.get("text") or "") for frame in said]
        assert sum(text.count(f"(answer): {journal.answer['text']}") for text in plain) == 1, (name, plain)
    capture(page, "1698-fixtures-live-1110-light")
    answer = card.locator("details.chat-delegated-answer").first
    answer.locator("summary").click()
    assert answer.locator(".chat-delegated-message").inner_text().strip() == journals["messages"].answer["text"]
    capture(page, "1698-fixtures-answer-opened")
    answer.locator("summary").click()
    snapshot = page.evaluate(SNAPSHOT, [ROOT])

    fetched = len(served)
    page.evaluate(HISTORY_LANDED)
    sockets[-1].close()
    for _ in range(100):
        if len(sockets) >= 2 and len(served) > fetched:
            break
        page.wait_for_timeout(100)
    assert len(sockets) >= 2 and len(served) > fetched
    page.wait_for_function("() => window.historyLanded > 0")
    page.wait_for_timeout(500)
    page.evaluate(EXPAND, ROOT)
    assert page.evaluate(SNAPSHOT, [ROOT]) == snapshot, "a reconnect adopts the live rows"
    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first.wait_for()
    page.evaluate(EXPAND, ROOT)
    page.wait_for_function(f"() => document.querySelectorAll('#chat-live-timeline-{ROOT} details.chat-delegated-answer').length === 4")
    assert page.evaluate(SPEECH, ROOT) == live, "history replay presents the same utterances and answers"
    page.set_viewport_size({"width": 390, "height": 844})
    capture(page, "1698-fixtures-reload-390")
    _evidence("1698-fixtures-frames.json", frames)
    _evidence("1698-fixtures-history-pages.json", served)
    _evidence("1698-fixtures-rows.json", {"live": live, "snapshot": snapshot})
    assert not ui["errors"].unexpected(), ui["errors"].report()


def test_words_lost_between_one_keys_fragments_stay_apart_in_the_served_card(engine_ui, tmp_path, monkeypatch):
    """Seq 2 of a draft never reaches the reader whole (not served; an unreadable line), in one poll and
    across two, before any complete text: the card never joins the words around it and names the loss."""
    ui = engine_ui
    page = ui["page"]
    page.set_viewport_size({"width": 1110, "height": 750})
    page.emulate_media(color_scheme="light")
    data = tmp_path / "data"
    producer = Producer(data, monkeypatch)
    history = TestClient(Starlette(routes=[Route("/api/chat/history", endpoint=make_chat_history_endpoint(data))]))
    ui["backend"]["task_gateway"] = _client(data)
    sockets, frames = [], []

    def socket_route(ws):
        sockets.append(ws)
        ws.send(json.dumps({"type": "heartbeat"}))

    page.route("**/api/state", lambda route: route.fulfill(content_type="application/json", body=json.dumps(
        {"supervisor_ready": True, "active_chat_activities": [], "projects": [], "sha": "cand1698"})))
    page.route("**/api/chat/history*", lambda route: route.fulfill(
        content_type="application/json", body=history.get(route.request.url[route.request.url.index("/api/chat/history"):]).text))
    page.route("**/api/tasks/**", lambda route: route.continue_())
    page.route_web_socket("**/ws", socket_route)
    page.goto(ui["url"])
    page.wait_for_selector("#chat-input")
    worker = producer.worker(ROOT, None)
    runs = (("run-lost-one", "missing", [4]), ("run-lost-two", "unreadable", [1, 4]))
    for payload in producer.say(worker, "Watching two runs that lose an event."):
        sockets[-1].send(json.dumps(payload))
    for run, lose, bounds in runs:
        steps = [[event for event in LOST if low < event["seq"] <= high] for low, high in zip([0, *bounds], bounds)]
        daemon = Lossy(steps, run_id=run, **{lose: [2]})
        gateway = _gateway(daemon)
        for step, (after, through) in enumerate(zip([0, *bounds], bounds), start=1):
            daemon.revealed = step
            for payload in producer.observe(worker, gateway, run, through, after=after):
                frames.append(payload)
                sockets[-1].send(json.dumps(payload))
    page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first.wait_for()
    page.evaluate(EXPAND, ROOT)
    landed = f"""() => Array.from(document.querySelectorAll('#chat-live-timeline-{ROOT} .chat-delegated-technical'))
        .filter(line => line.textContent.includes('Read ×1')).length === 2"""      # each run's last poll is shown
    page.wait_for_function(landed)
    live, rows = page.evaluate(SPEECH, ROOT), _rows(page)
    for run, _lose, _bounds in runs:
        assert [message["text"] for row in live if row["title"].endswith(run) for message in row["messages"]] == ["the", "cat"], \
            (run, live, "the words around the lost seq stay apart, never 'the cat'")
        assert sum("Not read whole: seq 2–2" in row["text"] for row in rows if run in row["text"]) == 1, (run, rows)
    capture(page, "1698-lost-event-1110-light")
    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first.wait_for()
    page.evaluate(EXPAND, ROOT)
    page.wait_for_function(landed)
    assert page.evaluate(SPEECH, ROOT) == live, "history replay keeps them apart too"
    _evidence("1698-lost-event-frames.json", frames)
    _evidence("1698-lost-event-rows.json", {"live": live, "rows": rows})
    assert not ui["errors"].unexpected(), ui["errors"].report()


def test_a_poll_holding_only_the_space_between_words_keeps_it_in_the_served_card(engine_ui, tmp_path, monkeypatch):
    """The space between a draft's words arrives alone in its poll, through the real producer: the card reads
    `the cat` where the draft began, live and after a reload, and the space alone is never an empty row."""
    ui = engine_ui
    page = ui["page"]
    page.set_viewport_size({"width": 1110, "height": 750})
    page.emulate_media(color_scheme="light")
    data = tmp_path / "data"
    producer = Producer(data, monkeypatch)
    history = TestClient(Starlette(routes=[Route("/api/chat/history", endpoint=make_chat_history_endpoint(data))]))
    ui["backend"]["task_gateway"] = _client(data)
    sockets, frames = [], []

    def socket_route(ws):
        sockets.append(ws)
        ws.send(json.dumps({"type": "heartbeat"}))

    page.route("**/api/state", lambda route: route.fulfill(content_type="application/json", body=json.dumps(
        {"supervisor_ready": True, "active_chat_activities": [], "projects": [], "sha": "cand1698"})))
    page.route("**/api/chat/history*", lambda route: route.fulfill(
        content_type="application/json", body=history.get(route.request.url[route.request.url.index("/api/chat/history"):]).text))
    page.route("**/api/tasks/**", lambda route: route.continue_())
    page.route_web_socket("**/ws", socket_route)
    page.goto(ui["url"])
    page.wait_for_selector("#chat-input")
    worker = producer.worker(ROOT, None)
    for payload in producer.say(worker, "Watching a run whose space arrives alone."):
        sockets[-1].send(json.dumps(payload))
    daemon = Daemon([[_said(1, "the", K1, "fragment")], [_said(2, " ", K1, "fragment")], [_said(3, "cat", K1, "fragment"), _tool(4)]],
                    run_id="run-space")
    gateway = _gateway(daemon)
    for step, (after, through) in enumerate(((0, 1), (1, 2), (2, 4)), start=1):
        daemon.revealed = step
        for payload in producer.observe(worker, gateway, "run-space", through, after=after):
            frames.append(payload)
            sockets[-1].send(json.dumps(payload))
    page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first.wait_for()
    page.evaluate(EXPAND, ROOT)
    landed = f"""() => Array.from(document.querySelectorAll('#chat-live-timeline-{ROOT} .chat-delegated-technical'))
        .some(line => line.textContent.includes('Read ×1'))"""
    page.wait_for_function(landed)
    blank = f"""() => Array.from(document.querySelectorAll('#chat-live-timeline-{ROOT} .chat-delegated-message'))
        .filter(m => !m.textContent.trim()).length"""
    live = page.evaluate(SPEECH, ROOT)
    assert [message["text"] for row in live for message in row["messages"]] == ["the cat"], live
    assert page.evaluate(blank) == 0, "the space alone is no empty paragraph"
    capture(page, "1698-space-1110-light")
    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first.wait_for()
    page.evaluate(EXPAND, ROOT)
    page.wait_for_function(landed)
    assert page.evaluate(SPEECH, ROOT) == live and page.evaluate(blank) == 0, "history replay reads the same"
    _evidence("1698-space-frames.json", frames)
    assert not ui["errors"].unexpected(), ui["errors"].report()


def test_a_keyed_piece_a_preview_left_out_is_never_joined_across_in_the_served_card(engine_ui, tmp_path, monkeypatch):
    """The reported case through the real producer: a poll whose 40 other messages fill the preview's part count
    leaves the draft's ' big ' out. The card shows 'the' and 'cat' apart, never 'thecat', live and after a reload, and
    the complete text once it comes. With 39 other messages the piece is shown and the card reads 'the big cat'."""
    ui = engine_ui
    page = ui["page"]
    page.set_viewport_size({"width": 1110, "height": 750})
    page.emulate_media(color_scheme="light")
    data = tmp_path / "data"
    producer = Producer(data, monkeypatch)
    history = TestClient(Starlette(routes=[Route("/api/chat/history", endpoint=make_chat_history_endpoint(data))]))
    ui["backend"]["task_gateway"] = _client(data)
    sockets, frames = [], []

    def socket_route(ws):
        sockets.append(ws)
        ws.send(json.dumps({"type": "heartbeat"}))

    page.route("**/api/state", lambda route: route.fulfill(content_type="application/json", body=json.dumps(
        {"supervisor_ready": True, "active_chat_activities": [], "projects": [], "sha": "cand1698"})))
    page.route("**/api/chat/history*", lambda route: route.fulfill(
        content_type="application/json", body=history.get(route.request.url[route.request.url.index("/api/chat/history"):]).text))
    page.route("**/api/tasks/**", lambda route: route.continue_())
    page.route_web_socket("**/ws", socket_route)
    page.goto(ui["url"])
    page.wait_for_selector("#chat-input")
    worker = producer.worker(ROOT, None)
    for payload in producer.say(worker, "Watching two runs whose middle poll is long."):
        sockets[-1].send(json.dumps(payload))

    def journal(others):
        return [[_said(1, "the", K1, "fragment")],
                [*(_harness(2 + i, "message", text=f"Note {i}.") for i in range(others)), _said(2 + others, " big ", K1, "fragment")],
                [_said(3 + others, "cat", K1, "fragment"), _tool(4 + others)],
                [_said(5 + others, "the big cat", K1, "complete"), _tool(6 + others)]]

    runs = {"run-left": Daemon(journal(40), run_id="run-left"), "run-shown": Daemon(journal(39), run_id="run-shown")}

    def observe(run, steps):
        daemon, gateway = runs[run], _gateway(runs[run])
        for step in steps:
            daemon.revealed = step
            after = daemon.steps[step - 2][-1]["seq"] if step > 1 else 0
            for payload in producer.observe(worker, gateway, run, daemon.steps[step - 1][-1]["seq"], after=after):
                frames.append(payload)
                sockets[-1].send(json.dumps(payload))

    def words():
        rows = page.evaluate(SPEECH, ROOT)
        return {run: [message["text"] for row in rows if row["title"].endswith(run) for message in row["messages"]
                      if not message["text"].startswith("Note ")] for run in runs}

    shown = f"""texts => texts.every(text => Array.from(document.querySelectorAll('#chat-live-timeline-{ROOT} .chat-delegated-message'))
        .some(m => m.textContent.trim() === text))"""
    # Both runs' last polls are rows of their own (each says `Read ×1`): every frame before them has landed.
    landed = f"""() => Array.from(document.querySelectorAll('#chat-live-timeline-{ROOT} .chat-delegated-technical'))
        .filter(line => line.textContent.includes('Read ×1')).length >= 2"""
    observe("run-left", (1, 2, 3))
    observe("run-shown", (1, 2, 3))
    page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first.wait_for()
    page.evaluate(EXPAND, ROOT)
    page.wait_for_function(landed)
    apart = words()
    assert apart == {"run-left": ["the", "cat"], "run-shown": ["the big cat"]}, apart
    capture(page, "1698-preview-left-out-live")
    ui["errors"].watch(page)  # the reloads may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first.wait_for()
    page.evaluate(EXPAND, ROOT)
    page.wait_for_function(landed)
    assert words() == apart, "history replay keeps them apart too"
    capture(page, "1698-preview-left-out-reload")
    observe("run-left", (4,))
    page.wait_for_function(f"""() => Array.from(document.querySelectorAll('#chat-live-timeline-{ROOT} .chat-delegated-message'))
        .filter(m => m.textContent.trim() === 'the big cat').length === 2""")
    assert words() == {"run-left": ["the big cat"], "run-shown": ["the big cat"]}, "the complete text is authoritative"
    page.reload()
    page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first.wait_for()
    page.evaluate(EXPAND, ROOT)
    page.wait_for_function(shown, arg=["the big cat"])
    assert words() == {"run-left": ["the big cat"], "run-shown": ["the big cat"]}
    _evidence("1698-preview-left-out-frames.json", frames)
    assert not ui["errors"].unexpected(), ui["errors"].report()

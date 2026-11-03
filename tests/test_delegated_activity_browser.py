"""Qualify #1350 in the served SPA without a model call.

A root and a child task each observe a delegated run through the real producer
(``delegate_progress.emit`` → ``delegate_activity.observations`` over the engine's event
stream, served by a fixture daemon), the real progress transport (``agent._emit_progress`` →
``events_chat_delivery`` → ``message_bus``) and the real history endpoint. The page receives
the live frames over its WebSocket, then reconnects and reloads; every retained original is
downloaded through the real task-file route. Screenshots and the raw frames, history and
DOM snapshots go to ``OUROBOROS_UI_EVIDENCE_DIR`` when it is set.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import delegate_activity, delegate_progress
from ouroboros.agent import OuroborosAgent
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.subagent_messages import subagent_message_meta
from ouroboros.task_results import write_task_result
from supervisor import events_chat_delivery, message_bus
from tests.test_delegated_activity import Daemon, _gateway, _harness, _run_event
from tests.test_reconnect_feed_browser import HISTORY_LANDED
from tests.test_subscription_setup_browser import capture, subscription_ui as subscription_ui
from tests.test_task_file_serving import _client

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
ROOT, KID = "root1350q", "kid1350q"
LINEAGE = {"delegation_role": "subagent", "root_task_id": ROOT, "parent_task_id": ROOT, "subagent_role": "UI reviewer",
           "model": "coordinator-model", "executor_route": "claude"}
LONG = "I traced the card renderer end to end: " + "the executor's sentence stays whole in its source. " * 150
# What a reader sees (layout text), and what each row holds (for live/reconnect/reload equality).
VISIBLE, SNAPSHOT = ("""ids => ids.map(id => Array.from(document.querySelectorAll(`#chat-live-timeline-${id} > .chat-live-line`))
    .filter(line => !line.hidden).map(line => line.%s.replace(/\\s+/g, ' ').trim()))""" % key for key in ("innerText", "textContent"))
EXPAND = """ids => { for (const id of ids) {
    const card = document.querySelector(`.chat-live-card[data-task-id="${id}"]`);
    if (card && card.dataset.expanded !== '1') card.querySelector('[data-live-disclosure]').click();
} }"""


@pytest.fixture(params=["chromium", "webkit"])
def engine_ui(request, monkeypatch):
    """``subscription_ui`` in each engine the UI lane must prove; evidence per engine."""
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        monkeypatch.setenv("OUROBOROS_UI_EVIDENCE_DIR", str(Path(root) / request.param))
    return request.getfixturevalue("subscription_ui")


def _evidence(name, value):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class Producer:
    """The worker and supervisor halves of the live progress path, in process."""

    def __init__(self, data, monkeypatch):
        self.data, self.live = data, []
        bridge = message_bus.LocalChatBridge()
        bridge._broadcast_fn = self.live.append
        monkeypatch.setattr(message_bus, "DATA_DIR", data)
        monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
        monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
        monkeypatch.setattr(message_bus, "publish_event", lambda *_: None)
        monkeypatch.setattr(events_chat_delivery, "_bound_project_chat_id", lambda *_: 0)
        kid = {"id": KID, "_attempt": 0, **LINEAGE}
        self.delivery = SimpleNamespace(DRIVE_ROOT=data, RUNNING={ROOT: {"task": {"id": ROOT, "_attempt": 0}},
                                                                  KID: {"task": kid}},
                                        send_with_budget=message_bus.send_with_budget,
                                        append_jsonl=lambda *_: pytest.fail("delivery raised"))
        write_task_result(data, ROOT, "running")
        write_task_result(data, KID, "running", **LINEAGE)

    def worker(self, task_id, lineage):
        events = queue.Queue()
        ctx = SimpleNamespace(task_id=task_id, task_attempt=0, drive_root=self.data, task_metadata={})
        agent = SimpleNamespace(
            _last_progress_ts=None, _event_queue=events, _current_chat_id=1, _current_task_id=task_id,
            tools=SimpleNamespace(_ctx=ctx),
            _subagent_progress_meta=lambda event: subagent_message_meta(lineage, task_id=task_id, event=event)
            if lineage else {})
        ctx.emit_progress_fn = partial(OuroborosAgent._emit_progress, agent)
        return ctx, events

    def _deliver(self, events):
        before = len(self.live)
        while not events.empty():
            frame = events.get_nowait()
            if frame["type"] == "send_message":
                events_chat_delivery._handle_send_message(frame, self.delivery)
        return self.live[before:]

    def say(self, worker, text):
        ctx, events = worker
        ctx.emit_progress_fn(text, narration=True)
        return self._deliver(events)

    def observe(self, worker, gateway, run_id, through, *, after, drain=None):
        """One wait advance through the seam ``delegate_wait`` uses."""
        ctx, events = worker
        entry = SimpleNamespace(run_id=run_id, task_id=ctx.task_id)
        delegate_progress.emit(ctx, run_id, SimpleNamespace(seq=through, events=[], events_omitted=0),
                               entry=entry, gateway=gateway, after_seq=after, drain_sec=drain)
        return self._deliver(events)

    def lifecycle(self, text, event):
        before = len(self.live)
        message_bus.send_with_budget(1, text, is_progress=True, task_id=ROOT,
                                     progress_meta=subagent_message_meta(LINEAGE, task_id=KID, event=event))
        return self.live[before:]


def _journals():
    child = Daemon([
        [_harness(1, "tool_call", tool={"name": "Read", "target": "web/modules/chat.js"}),
         _harness(2, "tool_result", tool={"name": "Read", "status": "ok"}),
         _harness(3, "message", text=LONG),
         _harness(4, "thinking", text="Maybe the key lives in the card renderer, not in the transport."),
         _harness(5, "tool_call", tool={"name": "Bash", "target": "node --test web/tests"}),
         _harness(6, "tool_result", tool={"name": "Bash", "status": "error", "error_summary": "exit 1: 2 failed"})],
        [_harness(seq, "tool_call", tool={"name": "Grep", "target": "delegated_activity"}) for seq in (7, 8, 9)],
        [_harness(10, "tool_call", tool={"name": "Read"}), _harness(11, "tool_result", tool={"name": "Read", "status": "ok"}),
         _harness(12, "tool_call", tool={"name": "Bash"})],
        [_harness(13, "message", text="Reading the consumer."), _harness(14, "message", text="Reading the consumer."),
         _harness(15, "message", text="😀", delta=True)],
        [_harness(16, "message", text="All checks", delta=True), _harness(17, "message", text=" pass.", delta=True),
         _harness(18, "message", text="Summary: the card leads with the executor. " + "detail " * 400),
         _run_event(19, "run.completed")],
    ], run_id="run-kid")
    root = Daemon([[_harness(1, "message", text="Delegated the UI check; watching it."),
                    _run_event(2, "reviewer.failed", error="Reviewer setup failed: model unavailable"),
                    _harness(3, "tool_call", tool={"name": "Read"})]], run_id="run-root")
    return child, root


def test_delegated_rows_live_reconnect_reload_root_child_source_and_final_gap(engine_ui, tmp_path, monkeypatch):
    import ouroboros.gateways.claudexor_run_events as run_events

    ui = engine_ui
    page = ui["page"]
    data = tmp_path / "data"
    producer = Producer(data, monkeypatch)
    history_app = Starlette(routes=[Route("/api/chat/history", endpoint=make_chat_history_endpoint(data))])
    history = TestClient(history_app)
    ui["backend"]["task_gateway"] = _client(data)
    served, sockets = [], []

    def history_route(route):
        url = route.request.url
        response = history.get(url[url.index("/api/chat/history"):])
        served.append(response.json())
        route.fulfill(status=response.status_code, content_type="application/json", body=response.text)

    def task_route(route):
        # Past the page's API mock to the served origin, whose HTTP handler answers from
        # the real task-file route (``subscription_ui`` ``task_gateway``).
        route.continue_()

    def socket_route(ws):
        sockets.append(ws)
        ws.send(json.dumps({"type": "heartbeat"}))

    def state_route(route):
        # The served build is named as the real endpoint names it (``gateway/state.py``), so a
        # reconnect keeps this page and must adopt its live rows. A state without ``sha``
        # makes the client reload the window instead, which is the reload case below.
        route.fulfill(content_type="application/json", body=json.dumps(
            {"supervisor_ready": True, "active_chat_activities": [], "projects": [], "sha": "cand1350"}))

    page.route("**/api/state", state_route)
    page.route("**/api/chat/history*", history_route)
    page.route("**/api/tasks/**", task_route)
    page.route_web_socket("**/ws", socket_route)
    page.goto(ui["url"])
    page.wait_for_selector("#chat-input")

    frames = []

    def push(payloads):
        for payload in payloads:
            frames.append(payload)
            sockets[-1].send(json.dumps(payload))

    child_daemon, root_daemon = _journals()
    child_gateway, root_gateway = _gateway(child_daemon), _gateway(root_daemon)
    root_worker, kid_worker = producer.worker(ROOT, None), producer.worker(KID, LINEAGE)
    push(producer.say(root_worker, "Coordinating the UI check."))
    push(producer.lifecycle("Child queued", "scheduled"))
    push(producer.say(kid_worker, "Watching the delegated run."))
    cursor = 0
    for step, through in enumerate((6, 9, 12, 15), start=1):
        child_daemon.revealed = step
        push(producer.observe(kid_worker, child_gateway, "run-kid", through, after=cursor))
        cursor = through
    root_daemon.revealed = 1
    push(producer.observe(root_worker, root_gateway, "run-root", 3, after=0))
    # The worker restarts and the run ends: its drain resumes at the last retained range,
    # and its read allowance is spent after a byte bound cuts the first tail read.
    delegate_activity.reset_process_memo()
    child_daemon.revealed = 5
    real_read = run_events.read_run_events
    clock = SimpleNamespace(now=0.0)

    def bounded_read(*args, **kwargs):
        read = real_read(*args, **{**kwargs, "max_bytes": 1800})
        clock.now += 1.0
        return read

    with monkeypatch.context() as bounded:
        bounded.setattr(delegate_activity.time, "monotonic", lambda: clock.now)
        bounded.setattr(run_events, "read_run_events", bounded_read)
        push(producer.observe(kid_worker, child_gateway, "run-kid", 19, after=19, drain=1))

    root_card = page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first
    root_card.wait_for()
    root_card.locator("[data-live-disclosure]").first.click()
    kid_card = page.locator(f'.chat-live-card[data-task-id="{KID}"]').first
    kid_card.wait_for()
    kid_card.locator("[data-live-disclosure]").first.click()
    page.wait_for_function(f"() => document.querySelector('#chat-live-timeline-{KID} .chat-delegated-final')")

    texts = page.evaluate(VISIBLE, [ROOT, KID])
    root_rows, child_rows = texts
    assert any("Watching the delegated run." in row for row in child_rows), "the child's own narration keeps its row"
    executor = [row for row in child_rows if "run-kid" in row]
    assert len(executor) == 4, child_rows
    first, stretch, repeated, final = executor
    assert "I traced the card renderer end to end" in first and "Thinking" in first
    assert "Maybe the key lives" not in first, "thinking stays behind its labelled disclosure"
    assert kid_card.locator(".chat-delegated-problem").first.inner_text() == \
        "Problem: claude · a01 · Bash: exit 1: 2 failed"
    assert "technical activity" in stretch and "6 technical events since seq 7 · Grep ×3" in stretch, \
        "two silent observations read as one row"
    assert repeated.count("Reading the consumer.") == 2, "equal words at distinct seqs are two utterances"
    assert repeated.count("😀") == 1 and "😀" not in final and "\ude00" not in final, \
        "overlap after an astral fragment uses the producer's code point offset"
    assert "All checks pass." in final and "Reading the consumer." not in final, "the re-read shows only new seqs"
    assert "Not shown: seq 18–19 (read_bound:time)" in final, final
    assert "Bash · tool_result" not in " ".join(child_rows), "the flattened label text is not what the row shows"
    assert any("Reviewer setup failed: model unavailable" in row for row in root_rows)
    assert kid_card.locator("[data-live-title]").first.inner_text() == "UI reviewer"
    assert kid_card.locator("[data-live-activity]").first.inner_text() == "Watching the delegated run."
    assert root_card.get_attribute("data-finished") != "1" and kid_card.get_attribute("data-finished") != "1", \
        "a failed step or reviewer is a warning row, not a finished or failed task"
    capture(page, "1350-live-child-and-root")
    thinking = kid_card.locator("details.chat-delegated-thinking > summary").first
    assert thinking.inner_text() == "Thinking"
    thinking.click()
    assert "Maybe the key lives in the card renderer" in kid_card.locator(".chat-delegated-thought").first.inner_text()
    capture(page, "1350-live-thinking-open")
    with page.expect_download() as download:
        kid_card.locator("a.chat-delegated-source", has_text="open the full source").first.click()
    body = Path(download.value.path()).read_bytes()
    assert download.value.failure() is None
    lines = [json.loads(line) for line in body.decode().splitlines()]
    assert [line["seq"] for line in lines] == [1, 2, 3, 4, 5, 6] and lines[2]["payload"]["text"] == LONG
    assert hashlib.sha256(body).hexdigest() in download.value.suggested_filename
    capture(page, "1350-live-final-gap")
    live_snapshot = page.evaluate(SNAPSHOT, [ROOT, KID])
    stretch_row = page.locator(f"#chat-live-timeline-{KID} > .chat-live-line", has_text="technical activity").first
    stretch_row.locator("[data-live-line-toggle]").click()
    detail = stretch_row.inner_text()
    assert "#7 Grep — delegated_activity" in detail and "#12 Bash" in detail, detail
    assert stretch_row.locator("a.chat-delegated-source").count() == 2, "each folded observation's source is one click away"
    stretch_row.scroll_into_view_if_needed()
    capture(page, "1350-live-stretch-expanded")
    stretch_row.locator("[data-live-line-toggle]").click()

    # A later wait on the ended run catches the tail up: no new run, only the journal read.
    push(producer.observe(kid_worker, child_gateway, "run-kid", 19, after=19, drain=30))
    page.wait_for_function(f"() => !document.querySelector('#chat-live-timeline-{KID} .chat-delegated-final')")
    caught_up = page.evaluate(SNAPSHOT, [ROOT, KID])
    assert any("Summary: the card leads with the executor." in row for row in caught_up[1])
    capture(page, "1350-live-caught-up")

    fetched = len(served)
    page.evaluate("() => { window.__live1350 = true; }")
    page.evaluate(HISTORY_LANDED)
    sockets[-1].close()                               # reconnect: the client refetches history
    for _ in range(100):
        if len(sockets) >= 2 and len(served) > fetched:
            break
        page.wait_for_timeout(100)
    assert len(sockets) >= 2 and len(served) > fetched, (len(sockets), len(served), fetched)
    page.wait_for_function("() => window.historyLanded > 0")  # response received; assert the rebuilt rows below
    assert page.evaluate("() => window.__live1350 === true") and "_ouro_refresh" not in page.url, \
        "the reconnect refetched history into the live page instead of reloading it"
    page.wait_for_timeout(500)
    assert page.locator('[data-system-type="reconnect"]').count() == 0, "a reconnect adds no row"
    page.evaluate(EXPAND, [ROOT, KID])
    page.wait_for_function(f"() => document.querySelectorAll('#chat-live-timeline-{KID} > .chat-live-line').length > 3")
    reconnected = page.evaluate(SNAPSHOT, [ROOT, KID])
    capture(page, "1350-reconnected")
    assert reconnected == caught_up, "a reconnect adopts the live rows instead of drawing them again"

    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.locator(f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]').first.wait_for()
    page.evaluate(EXPAND, [ROOT])
    page.locator(f'.chat-live-card[data-task-id="{KID}"]').first.wait_for()
    page.evaluate(EXPAND, [KID])
    page.wait_for_function(f"() => document.querySelectorAll('#chat-live-timeline-{KID} > .chat-live-line').length > 3")
    reloaded = page.evaluate(SNAPSHOT, [ROOT, KID])
    assert [row for row in reloaded[1] if "run-kid" in row] == [row for row in caught_up[1] if "run-kid" in row], \
        "history replay presents the same executor rows as live"
    capture(page, "1350-reload-desktop")
    page.set_viewport_size({"width": 390, "height": 844})
    capture(page, "1350-reload-mobile")
    with page.expect_download() as replayed:
        page.locator(f'.chat-live-card[data-task-id="{KID}"]').first.locator(
            "a.chat-delegated-source", has_text="open the full source").first.click()
    assert Path(replayed.value.path()).read_bytes() == body, "the replayed row opens the same retained original"
    _evidence("1350-frames.json", frames)
    _evidence("1350-history-pages.json", served)
    _evidence("1350-dom.json", {"live_with_gap": live_snapshot, "caught_up": caught_up, "reconnected": reconnected,
                                "reloaded": reloaded})
    assert not ui["errors"].unexpected(), ui["errors"].report()

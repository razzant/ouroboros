"""Finalization observations must preserve the answer's existing delivery path."""

from tests import _dialogue as dialogue_test

import json
import os
import queue
import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline, delegate_custody, delegate_terminal, loop, server_liveness
from ouroboros.tools.registry import ToolRegistry
from ouroboros.utils import append_jsonl
from supervisor import events_chat_delivery as delivery


@pytest.fixture(params=["direct", "queued", "child", "wake"])
def completed_turn(tmp_path, monkeypatch, request):
    """Use the real loop, finalizer, result writer and sender; replace only I/O."""
    kind = request.param
    root = tmp_path / "canonical"
    drive = tmp_path / "child" if kind == "child" else root
    logs = drive / "logs"
    logs.mkdir(parents=True)
    task = {"id": "turn", "type": "task", "chat_id": 1, "text": "Say hello.",
            "_attempt": 1, "budget_drive_root": str(root),
            "_is_direct_chat": kind in {"direct", "wake"}}
    if kind == "child":
        task.update(parent_task_id="parent", root_task_id="parent", delegation_role="subagent")
    if kind == "wake":
        task["metadata"] = {"initiator": "consciousness"}
    order, sent, queued = [], [], []
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(pipeline, "in_worker_process", lambda: kind in {"queued", "child"})
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))

    def forbid_gateway(*args, **kwargs):
        pytest.fail("An empty custody release must not contact a daemon")

    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", forbid_gateway)

    def observe(module, name, label):
        original = getattr(module, name)

        def call(*args, **kwargs):
            order.append(label + "_start")
            result = original(*args, **kwargs)
            order.append(label + "_end")
            return result

        monkeypatch.setattr(module, name, call)

    observe(loop, "_cleanup_loop_resources", "cleanup")
    observe(delegate_custody, "release_task_runs", "custody")
    observe(delegate_terminal, "record_terminal_reconciliation", "annotation")
    observe(pipeline, "register_final_answer_owed", "outbox")
    observe(pipeline, "_store_task_result", "store")
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async",
                        lambda *args, **kwargs: order.append("post_task"))

    class LLM:
        def default_model(self):
            return "openai-compatible::test"

        def chat(self, **kwargs):
            order.append("model_return")
            return {"content": "Hello.", "tool_calls": []}, {"cost": 0.0}

    class Events:
        def put(self, event):
            # Cross-process delivery owns a value snapshot, not the worker's dict.
            queued.append(json.loads(json.dumps(event)))
            order.append("enqueue")

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=drive)
    registry._ctx.task_metadata = dict(task)
    registry._ctx.task_attempt = 1
    registry._ctx.owner_message_admission_lock = threading.RLock()
    registry._ctx.owner_message_admission_agent = SimpleNamespace(_accepting_owner_messages=True)
    text, usage, trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": task["text"]}], tools=registry, llm=LLM(),
        drive_logs=logs, emit_progress=lambda *args, **kwargs: None,
        incoming_messages=queue.Queue(), task_id=task["id"], drive_root=drive,
    )
    pending = []
    pipeline.emit_task_results(
        SimpleNamespace(drive_root=drive, repo_dir=tmp_path), None, None, pending,
        task, text, usage, trace, 0.0, logs, ctx=registry._ctx, event_queue=Events(),
    )

    def send(chat_id, body, **kwargs):
        assert pipeline.load_task_result(drive, task["id"])["result"] == "Hello."
        order.append("sender_return")
        sent.append((chat_id, body, kwargs))

    host = SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, append_jsonl=append_jsonl, send_with_budget=send)
    # The live copies leave the bus through the supervisor's real drain (which stamps the
    # dequeue); a buffered copy reaches the handler directly.
    bus = queue.Queue()
    for event in queued:
        if event["type"] == "send_message":
            bus.put(event)
    assert server_liveness.drain_worker_events(bus, host, [time.monotonic(), {}, time.thread_time(), None],
                                               on_restart=lambda *_args: None) is False
    for event in pending:
        if event["type"] == "send_message":
            delivery._handle_send_message(event, host)
    return SimpleNamespace(kind=kind, root=root, drive=drive, task=task, usage=usage,
                           trace=trace, text=text, pending=pending, queued=queued,
                           sent=sent, order=order, host=host)


def test_finalization_keeps_delivery_and_step_order(completed_turn):
    turn = completed_turn
    assert turn.text == "Hello."
    assert [(chat, text) for chat, text, _ in turn.sent] == [(1, "Hello.")]
    assert [event["type"] for event in turn.pending][:3] == ["send_message", "task_metrics", "task_done"]
    order = turn.order
    assert order.index("model_return") < order.index("cleanup_start")
    assert order.index("cleanup_start") < order.index("custody_start") < order.index("custody_end")
    assert order.index("custody_end") < order.index("cleanup_end") < order.index("store_start")
    assert order.index("store_start") < order.index("store_end") < order.index("sender_return")
    if turn.kind != "child":
        # Answer first (owner 1A(1)): owed in the outbox, on the live bus, THEN the
        # result store; the buffered task_done still trails the store and post-task.
        assert order.index("outbox_end") < order.index("enqueue") < order.index("annotation_start")
        assert order.index("annotation_end") < order.index("store_start")
        assert order.index("store_end") < order.index("post_task")
        assert len([event for event in turn.queued if event["type"] == "send_message"]) == 1
    else:
        assert "post_task" not in order


def timing_rows(root, basis=None):
    path = root / "logs" / "events.jsonl"
    rows = [row for line in path.read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line)).get("type") == "task_finalization_timing"]
    return rows if basis is None else [row for row in rows if row.get("basis") == basis]


def test_finalization_records_one_ordered_tail_on_every_task_surface(completed_turn):
    turn = completed_turn
    row, = timing_rows(turn.root, "send_handler_returned")
    assert row["task_id"] == "turn" and row["task_attempt"] == 1
    assert row["is_direct_chat"] is (turn.kind in {"direct", "wake"})
    assert row["initiator"] == ("consciousness" if turn.kind == "wake" else "")
    assert row["delegation_role"] == ("subagent" if turn.kind == "child" else "")
    phases = row["phases"]
    assert {"source_ack", "child_lookup", "admission_wait", "admission_close", "cleanup",
            "release_task_runs", "custody_reconcile", "custody_audit", "sender"} <= phases.keys()
    assert "acceptance" not in phases, "review mode off buys no panel"
    assert "custody_daemon_request" not in phases and "custody_run" not in phases
    assert phases["child_lookup"]["count"] >= 8
    assert phases["source_ack"]["finished_sec"] <= phases["admission_wait"]["started_sec"]
    assert phases["admission_wait"]["finished_sec"] <= phases["admission_close"]["started_sec"]
    assert phases["admission_close"]["finished_sec"] <= phases["cleanup"]["started_sec"]
    assert phases["custody_reconcile"]["finished_sec"] <= phases["custody_audit"]["started_sec"]
    for phase in phases.values():
        assert phase["count"] >= 1 and phase["errors"] == 0
        assert phase["seconds"] >= 0
        assert 0 <= phase["started_sec"] <= phase["finished_sec"] <= row["total_sec"]
        assert row["last_answer_at"] <= phase["started_at"] <= phase["finished_at"] <= row["ts"]
    if turn.kind != "child":
        # The live copy's snapshot is frozen at its enqueue, BEFORE the store: the
        # sender's row cannot hold a store span that had not started yet, and the
        # answer's own path ends at the sender.
        assert "result_store" not in phases and "custody_annotation" not in phases
        assert phases["cleanup"]["finished_sec"] <= row["enqueued_sec"] <= phases["sender"]["started_sec"]
        assert row["enqueue_to_sender_sec"] >= 0
        # The consumer stamped its take: the bus wait and the handler's pre-send work split.
        assert row["enqueued_sec"] <= row["dequeued_sec"] <= phases["sender"]["started_sec"]
        assert row["enqueued_at"] <= row["dequeued_at"] <= phases["sender"]["started_at"]
        assert row["enqueue_to_dequeue_sec"] == pytest.approx(row["dequeued_sec"] - row["enqueued_sec"])
        assert row["dequeue_to_sender_sec"] == pytest.approx(phases["sender"]["started_sec"] - row["dequeued_sec"])
        assert row["enqueue_to_sender_sec"] == pytest.approx(row["enqueue_to_dequeue_sec"] + row["dequeue_to_sender_sec"])
        # The packaging that ran after the answer left is journaled as its own
        # correlated row (same task, same delivery id), internally ordered.
        store_row, = timing_rows(turn.root, "post_answer_store")
        assert store_row["task_id"] == "turn" and store_row["delivery_id"] == row["delivery_id"]
        store = store_row["phases"]["result_store"]
        assert row["enqueued_sec"] <= store["started_sec"] <= store["finished_sec"] == store_row["total_sec"]
        assert store_row["phases"]["cleanup"]["finished_sec"] <= store["started_sec"]
        annotation = store_row["phases"]["custody_annotation"]
        assert row["enqueued_sec"] <= annotation["started_sec"] <= annotation["finished_sec"] <= store["started_sec"]
        assert "enqueued_sec" not in store_row, "the buffered dict never borrows the live copy's enqueue"
    else:
        # The buffered copy carries the live dict, so the handler sees the store span.
        assert phases["cleanup"]["finished_sec"] <= phases["result_store"]["started_sec"]
        assert phases["result_store"]["finished_sec"] <= phases["sender"]["started_sec"]
        assert row["enqueue_to_sender_sec"] is None  # this fixture calls the buffered handler directly
        assert "dequeued_sec" not in row and row["enqueue_to_dequeue_sec"] is row["dequeue_to_sender_sec"] is None
        assert timing_rows(turn.root, "post_answer_store") == []


@pytest.mark.parametrize("kind", ["direct", "queued", "stopped_direct"])
def test_answer_reaches_the_chat_while_the_store_is_blocked_and_the_card_follows(tmp_path, monkeypatch, kind):
    """Owner 1A(1)/Fork 2: the answer is in the chat while ``_store_task_result`` is
    still running, no Done card exists yet, and after the store the card follows the
    answer (the frame's pre-store status is the one the row settles to, W10)."""
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.terminal_projection import SETTLEMENT_NONE, settle_terminal_projection
    from supervisor import terminal_delivery as td

    root = tmp_path / "data"
    logs = root / "logs"
    logs.mkdir(parents=True)
    chat_path = logs / "chat.jsonl"
    task = {"id": "barrier", "type": "task", "chat_id": 1, "text": "Say hello.", "_attempt": 1,
            "budget_drive_root": str(root), "_is_direct_chat": kind != "queued"}
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(pipeline, "in_worker_process", lambda: kind == "queued")
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *args, **kwargs: None)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    store_gate, store_entered, enqueued = threading.Event(), threading.Event(), threading.Event()
    original_store = pipeline._store_task_result

    def blocked_store(*args, **kwargs):
        store_entered.set()
        assert store_gate.wait(timeout=10), "the test never released the store"
        return original_store(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_store_task_result", blocked_store)
    live = []

    class Events:
        def put(self, event):
            live.append(json.loads(json.dumps(event)))
            enqueued.set()

    def send(chat_id, body, **kwargs):
        append_jsonl(chat_path, {"ts": "2026-10-15T00:00:00Z", "chat_id": chat_id, "direction": "out",
                                 "text": body, "task_id": kwargs.get("task_id"),
                                 **({"progress_meta": kwargs["progress_meta"]} if kwargs.get("progress_meta") else {})})

    host = SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, append_jsonl=append_jsonl, send_with_budget=send)
    ctx = SimpleNamespace(pending_restart_reason=None, _skip_post_task_synthesis=kind == "stopped_direct")
    pending, usage = [], {"cost": 0.0, "rounds": 1, "prompt_tokens": 1, "completion_tokens": 1}
    worker = threading.Thread(target=pipeline.emit_task_results, kwargs=dict(
        env=SimpleNamespace(drive_root=root, repo_dir=tmp_path), memory=None, llm=None,
        pending_events=pending, task=task, text="Hello.", usage=usage,
        llm_trace={"tool_calls": [], "reasoning_notes": []}, start_time=0.0, drive_logs=logs,
        ctx=ctx, event_queue=Events()), daemon=True)
    worker.start()
    try:
        assert enqueued.wait(timeout=10), "the answer never reached the live bus"
        assert store_entered.wait(timeout=10)
        # The store is blocked; the real supervisor handler consumes the live copy.
        (answer,) = [event for event in live if event["type"] == "send_message"]
        assert answer["text"] == "Hello." and answer["delivery_id"].startswith("final:barrier:")
        assert td.terminal_answer_receipts(root, "barrier")["state"] == "owed"
        delivery._handle_send_message(answer, host)
        rows = dialogue_test.log_rows(chat_path)
        assert [row["text"] for row in rows] == ["Hello."], "the answer is in the chat before the store"
        assert td.terminal_answer_receipts(root, "barrier")["state"] == "delivered"
        assert not any(row.get("type") == "task_summary" for row in rows), "no Done card before the row"
        stored = pipeline.load_task_result(root, "barrier") or {}
        assert stored.get("status") not in {"completed", "failed"}
        assert settle_terminal_projection(root, "barrier") == SETTLEMENT_NONE
    finally:
        store_gate.set()
        worker.join(timeout=20)
    assert not worker.is_alive()
    stored = pipeline.load_task_result(root, "barrier")
    assert stored["status"] == "completed" and stored["result"] == "Hello."
    # The frame's pre-store conclusion (a finalizing root's typed status, a stopped
    # direct turn's own final word) is the status the row settled to: the early
    # live card never contradicts the store, so no extra "finalizing" phase is owed.
    assert answer["progress_meta"]["task_terminal_status"] == "completed"
    assert answer["progress_meta"].get("task_phase") == (None if kind == "stopped_direct" else "finalizing")
    done, = [event for event in pending if event["type"] == "task_done"]
    assert done["status"] == "completed"
    if kind != "stopped_direct":
        # Post-task synthesis (stubbed here) is still open: the card stays deferred
        # behind "Finalizing" until the post-task callback closes the checkpoint.
        assert not append_terminal_task_projection(root, "barrier", task, stored, {**done, "chat_id": 1})
        stored = pipeline.write_task_result(root, "barrier", "completed",
                                            root_phase_checkpoint={"post_task_synthesis": "completed"})
    assert append_terminal_task_projection(root, "barrier", task, stored, {**done, "chat_id": 1})
    rows = dialogue_test.log_rows(chat_path)
    assert [row.get("text") for row in rows][0] == "Hello."
    card = rows[-1]
    assert card["summary_kind"] == "terminal_root_projection" and card["outcome_phase"] == "done"
    assert rows.index(card) > 0, "the Done card strictly follows the answer"


def test_last_answer_replaces_prior_rounds_without_cross_thread_leaks(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from ouroboros import observability as obs
    from ouroboros.loop_llm_call import call_llm_with_retry
    from ouroboros.task_status import find_child_tasks

    class LLM:
        def chat(self, **kwargs):
            return {"content": "answer", "tool_calls": []}, {"cost": 0.0}

    barrier = threading.Barrier(2)

    def run(task_id):
        usage = {}
        with obs.task_timing_scope():
            for round_index in (1, 2):
                call_llm_with_retry(LLM(), [{"role": "user", "content": "hello"}],
                                    "openai-compatible::test", None, "medium", 1,
                                    tmp_path / task_id / "logs", task_id, round_index, queue.Queue(), usage)
                find_child_tasks(tmp_path / task_id, parent_task_id=task_id)
                barrier.wait(timeout=5)
            timing = usage["_finalization_timing"]
            assert timing["phases"]["child_lookup"]["count"] == 1
            assert usage["first_answer_at"] < timing["last_answer_at"]
        assert obs.task_timing() is None
        return timing

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(run, ["one", "two"]))
    assert first is not second and first["phases"] is not second["phases"]
    assert obs.task_timing() is None


def test_phase_seconds_ignore_wall_clock_reversal_and_preserve_exceptions(monkeypatch):
    from ouroboros import finalization_timing, observability as obs

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(finalization_timing, "time", SimpleNamespace(monotonic=lambda: clock.now))
    stamps = iter(["2026-10-06T00:00:03Z", "2026-10-06T00:00:02Z", "2026-10-06T00:00:01Z"])
    monkeypatch.setattr(finalization_timing, "utc_now_iso", lambda: next(stamps))
    with obs.task_timing_scope():
        usage = {}
        obs.mark_last_answer(usage)
        with pytest.raises(ValueError, match="original failure"):
            with obs.timed_phase("child_lookup"):
                clock.now += 7
                raise ValueError("original failure")
        phase = usage["_finalization_timing"]["phases"]["child_lookup"]
    assert phase["seconds"] == phase["finished_sec"] == 7
    assert phase["errors"] == 1 and phase["count"] == 1
    assert obs.task_timing() is None


def test_failed_sender_keeps_retry_and_timing_write_cannot_block_receipt(tmp_path, monkeypatch):
    from ouroboros import observability as obs
    from supervisor import terminal_delivery as td

    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    with obs.task_timing_scope():
        usage = {}
        obs.mark_last_answer(usage)
    event = {"type": "send_message", "task_id": "send", "chat_id": 1, "text": "Answer.",
             "delivery_id": td.delivery_id_for("send", "Answer."),
             "_finalization_timing": usage["_finalization_timing"]}
    event = obs.stamp_finalization_enqueue(event)
    calls = []

    def send(*args, **kwargs):
        calls.append("send")
        if len(calls) == 1:
            raise RuntimeError("sender unavailable")

    host = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=append_jsonl, send_with_budget=send)
    delivery._handle_send_message(event, host)
    assert not td.already_delivered(tmp_path, event["delivery_id"])
    assert not (tmp_path / "logs/events.jsonl").exists()

    def unavailable_log(*args, **kwargs):
        raise OSError("journal unavailable")

    monkeypatch.setattr("ouroboros.utils.append_jsonl", unavailable_log)
    delivery._handle_send_message(event, host)
    assert td.already_delivered(tmp_path, event["delivery_id"])
    delivery._handle_send_message(event, host)
    assert calls == ["send", "send"]


def test_the_dequeue_stamp_is_taken_only_from_a_live_final_and_never_raises():
    from ouroboros.finalization_timing import stamp_finalization_dequeue

    live = {"type": "send_message", "_finalization_timing": {"_origin": time.monotonic() - 2.0, "phases": {}}}
    sidecar = live["_finalization_timing"]
    stamp_finalization_dequeue(live)
    assert live["_finalization_timing"]["dequeued_sec"] >= 2.0 and live["_finalization_timing"]["dequeued_at"]
    assert "dequeued_sec" not in sidecar, "the event's sidecar is replaced, not mutated"
    replayed = {"type": "send_message", "_finalization_timing": {"phases": {}}}  # no live origin
    malformed = {"type": "send_message", "_finalization_timing": {"_origin": "yesterday", "phases": {}}}
    for event in (replayed, malformed, {"type": "task_heartbeat"}, "not an event"):
        before = json.dumps(event)
        stamp_finalization_dequeue(event)
        assert json.dumps(event) == before


def test_custody_notice_progress_and_durable_replay_do_not_add_timing_rows(tmp_path, monkeypatch):
    from ouroboros import observability as obs
    from supervisor import terminal_delivery as td

    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    with obs.task_timing_scope():
        usage = {}
        obs.mark_last_answer(usage)
    event = {"type": "send_message", "task_id": "answer", "chat_id": 1, "text": "Answer.",
             "terminal_custody_notice": "A delegated run remains open.",
             "delivery_id": td.delivery_id_for("answer", "Answer."),
             "_finalization_timing": usage["_finalization_timing"]}
    assert td.register_pending_delivery(tmp_path, event)
    owed, = td.pending_deliveries(tmp_path)
    assert "_finalization_timing" not in owed, "monotonic origins cannot survive a reboot"
    sent = []
    host = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=append_jsonl,
                           send_with_budget=lambda *args, **kwargs: sent.append(args))
    delivery._handle_send_message(obs.stamp_finalization_enqueue(event), host)
    delivery._handle_send_message(event, host)
    delivery._handle_send_message(owed, host)
    delivery._handle_send_message({"type": "send_message", "task_id": "answer", "chat_id": 1,
                                  "text": "Working.", "is_progress": True}, host)
    assert len(timing_rows(tmp_path)) == 1
    assert [args[1] for args in sent] == ["Answer.", "A delegated run remains open.", "Working."]
    # Replay on a fresh sender has no prior-boot clock, even if never delivered.
    owed["delivery_id"] = td.delivery_id_for("different", "Answer.")
    delivery._handle_send_message(owed, host)
    assert len(timing_rows(tmp_path)) == 1


@pytest.mark.parametrize("failed_read", [False, True])
def test_release_counts_actual_runs_and_daemon_requests(tmp_path, failed_read):
    import httpx
    from ouroboros import observability as obs
    from ouroboros.config import CLAUDEXOR_MIN_VERSION, CLAUDEXOR_PROTOCOL_MAJOR
    from ouroboros.gateways.claudexor import ClaudexorGateway

    requests = []

    def respond(request):
        requests.append(request.url.path)
        if request.url.path == "/v2/handshake":
            return httpx.Response(200, json={"compatible": True, "protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR,
                                             "engine": {"version": CLAUDEXOR_MIN_VERSION}})
        if failed_read and request.url.path.endswith("run-two"):
            raise httpx.ReadTimeout("fixture transport failure", request=request)
        return httpx.Response(200, json={"summary": {"state": "running", "effectiveAccess": "readonly"}})

    # Exercise the real HTTP method without discovering credentials or opening a socket.
    gateway = ClaudexorGateway.__new__(ClaudexorGateway)
    gateway._client = httpx.Client(base_url="http://daemon.test", transport=httpx.MockTransport(respond))
    gateway._endpoint = None  # no daemon descriptor: the handshake caches nothing for this bare gateway
    for run_id in ("run-one", "run-two"):
        delegate_custody.record_started(tmp_path, delegate_custody.RunCustody(
            run_id=run_id, task_id="task", root_task_id="task", route_id="route", model="model",
            project_id="project", project_owned=False, ledger_root=str(tmp_path),
        ))
    with obs.task_timing_scope():
        usage = {}
        obs.mark_last_answer(usage)
        outcomes = delegate_custody.release_task_runs(tmp_path, "task", gateway_factory=lambda: gateway)
    assert requests == ["/v2/handshake", "/v2/runs/run-one", "/v2/runs/run-two"]
    assert [row["action"] for row in outcomes] == ["left_live", "unreadable" if failed_read else "left_live"]
    phases = usage["_finalization_timing"]["phases"]
    assert phases["custody_run"]["count"] == 2
    assert phases["custody_daemon_request"]["count"] == 3
    assert phases["custody_daemon_request"]["errors"] == int(failed_read)
    assert phases["custody_reconcile"]["finished_sec"] <= phases["custody_audit"]["started_sec"]
    assert phases["custody_audit"]["finished_sec"] <= phases["release_task_runs"]["finished_sec"]
    assert timing_rows(tmp_path) == [], "a custody sweep is not another task completion"


@pytest.mark.parametrize("pooled", [False, True])
def test_buffered_worker_handoff_stamps_a_value_snapshot(tmp_path, monkeypatch, pooled):
    import sys
    from ouroboros import observability as obs
    from supervisor import worker_chat_lane, worker_process, workers

    with obs.task_timing_scope():
        usage = {}
        obs.mark_last_answer(usage)
    event = {"type": "send_message", "task_id": "buffered", "chat_id": 1, "text": "Done.",
             "_finalization_timing": usage["_finalization_timing"]}
    task = {"id": "buffered", "type": "task", "chat_id": 1, "project_id": "room", "text": "Work."}
    agent = SimpleNamespace(handle_task=lambda task: [event])
    outgoing = queue.Queue()
    if pooled:
        # Run the actual pooled handback without process/bootstrap side effects.
        for name in ("_bind_worker_repo_root", "_prepare_worker_task_runtime", "_adopt_published_extensions",
                     "_configure_worker_logging"):
            monkeypatch.setattr(worker_process, name, lambda *args: None)
        for path in ("ouroboros.platform_layer.create_new_session", "ouroboros.process_custody.start_parent_lifeline",
                     "ouroboros.extension_loader.reload_all", "ouroboros.config.initialize_runtime_mode_baseline",
                     "ouroboros.utils.set_log_sink"):
            monkeypatch.setattr(path, lambda *args, **kwargs: None)
        monkeypatch.setattr("ouroboros.agent.make_agent", lambda **kwargs: agent)
        monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: "")
        monkeypatch.setattr("ouroboros.utils.get_git_info", lambda *args: ("fixture", "fixture"))
        monkeypatch.setattr(sys, "path", list(sys.path))
        incoming = queue.Queue()
        incoming.put(task)
        incoming.put(None)
        worker_process.worker_main(1, incoming, outgoing, str(tmp_path), str(tmp_path))
    else:
        monkeypatch.setattr(workers, "get_event_q", lambda: outgoing)
        released = []
        assert worker_chat_lane._execute_chat_task({"task": task, "agent": agent, "chat_id": 1,
                                                   "registry": SimpleNamespace(unregister=released.append)})
        assert released == ["buffered"]
    emitted, = [row for row in list(outgoing.queue) if row["type"] == "send_message"]
    assert emitted["text"] == event["text"]
    assert emitted["_finalization_timing"]["enqueued_sec"] >= 0
    assert "enqueued_sec" not in event["_finalization_timing"]
    assert emitted["_finalization_timing"] is not event["_finalization_timing"]


def test_cold_continuation_preserves_usage_without_reusing_live_clocks():
    from ouroboros import observability as obs, owner_wait

    ctx = SimpleNamespace(task_id="sleeping", task_attempt=1)
    with obs.task_timing_scope():
        usage = {"cost": 0.2, "rounds": 2, "first_answer_at": "2026-10-06T00:00:00Z"}
        obs.mark_last_answer(usage)
        state = owner_wait.continuation_state(ctx, [{"role": "user", "content": "Continue."}],
                                              {"task_id": ctx.task_id}, usage, 2, [], {"message"})
        saved = json.loads(json.dumps(state))
        assert "_finalization_timing" in usage, "the warm continuation retains its valid live clock"
    resumed_usage, messages, trace, seen = {}, [], {}, set()
    with obs.task_timing_scope():
        owner_wait.restore_continuation_state(SimpleNamespace(_ctx=ctx), saved,
                                              messages, trace, resumed_usage, seen)
        assert obs.task_timing() == {}
    assert resumed_usage == {"cost": 0.2, "rounds": 2, "first_answer_at": "2026-10-06T00:00:00Z"}
    assert messages == state["messages"] and trace == state["trace"] and seen == {"message"}


@pytest.mark.parametrize("is_root", [False, True])
def test_synthesis_snapshot_preserves_usage_without_live_clocks(tmp_path, is_root):
    from ouroboros import observability as obs

    env = SimpleNamespace(drive_root=tmp_path)
    task = {"id": "snapshot", "root_task_id": "snapshot" if is_root else "parent"}
    if not is_root:
        task["parent_task_id"] = "parent"
    fields = {"rounds": 2, "prompt_tokens": 3, "first_answer_at": "2026-10-06T00:00:00Z",
              "llm_call_refs": [{"call_id": "last-answer"}]}
    usage = dict(fields)
    with obs.task_timing_scope():
        obs.mark_last_answer(usage)
        timing = usage["_finalization_timing"]
        snapshot = pipeline._pre_synthesis_usage_snapshot(env, task, usage)
    assert "_finalization_timing" not in snapshot
    assert all(snapshot[key] == value for key, value in fields.items())
    assert snapshot["llm_call_refs"] is not usage["llm_call_refs"]
    assert usage == {**fields, "_finalization_timing": timing}
    assert usage["_finalization_timing"] is timing


def test_agent_scope_keeps_post_loop_child_lookups_and_resets_reused_workers(tmp_path, monkeypatch):
    from ouroboros import agent as agent_module, observability as obs
    from ouroboros.settings_integrity import task_settings_snapshot
    from ouroboros.task_status import find_child_tasks

    class LLM:
        def default_model(self):
            return "openai-compatible::test"

        def chat(self, **kwargs):
            return {"content": "Done.", "tool_calls": []}, {"cost": 0.0}

    def task_body(task):
        registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
        text, usage, _ = loop.run_llm_loop(
            messages=[{"role": "user", "content": "Work."}], tools=registry, llm=LLM(),
            drive_logs=tmp_path / "logs", emit_progress=lambda *args, **kwargs: None,
            incoming_messages=queue.Queue(), task_id=task["id"], drive_root=tmp_path,
        )
        timing = usage["_finalization_timing"]
        count = timing["phases"]["child_lookup"]["count"]
        find_child_tasks(tmp_path, parent_task_id=task["id"])
        assert timing["phases"]["child_lookup"]["count"] == count + 1
        assert text == "Done."
        return timing

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(agent_module.subagent_runtime, "apply_task_start_settings_or_disclose",
                        lambda *args: task_settings_snapshot({}, os.environ))
    from tests._agent_entry_helpers import task_entry_agent

    host = task_entry_agent(tmp_path, task_body)
    first = agent_module.OuroborosAgent.handle_task(host, {"id": "one", "type": "task"})
    assert obs.task_timing() is None
    second = agent_module.OuroborosAgent.handle_task(host, {"id": "two", "type": "task"})
    assert obs.task_timing() is None and first is not second
    assert first["phases"] is not second["phases"]


def test_a_delivery_context_without_a_drive_root_still_delivers_once_and_records_nothing(tmp_path):
    """The timing row is best effort: a sender context that carries no DRIVE_ROOT must deliver
    exactly as before and simply record no timing (the call site may not read a missing root)."""
    from ouroboros.finalization_timing import emit_finalization_timing

    timed = {"_finalization_timing": {"phases": {"sender": {"finished_at": "t", "finished_sec": 1.0, "started_sec": 0.5}}},
             "task_id": "t1", "delivery_id": "d1"}
    emit_finalization_timing(timed, None)
    assert not (tmp_path / "logs" / "events.jsonl").exists()
    emit_finalization_timing(timed, tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows] == ["task_finalization_timing"]

    sent = []

    class RootlessCtx:
        RUNNING = {}

        @staticmethod
        def send_with_budget(chat_id, text, **kwargs):
            sent.append(text)

        @staticmethod
        def append_jsonl(path, data):
            raise AssertionError(f"no error row expected, got {data!r}")

    delivery._handle_send_message({"type": "send_message", "chat_id": 1, "text": "Done.", **timed}, RootlessCtx())
    assert sent == ["Done."]


@pytest.mark.parametrize("own_run", [False, True])
@pytest.mark.parametrize("put_fails", [False, True])
def test_final_answer_discloses_own_custody_before_annotation_and_keeps_failed_put_debt(
    tmp_path, monkeypatch, own_run, put_fails,
):
    from ouroboros import delegate_custody_current as current, store

    root = tmp_path / "data"
    task = {"id": "own", "type": "task", "chat_id": 1, "_is_direct_chat": True,
            "budget_drive_root": str(root)}
    pipeline.write_task_result(root, "own", "running")
    runs = [("review", "review_substrate")]
    if own_run:
        runs.append(("leaf", "delegated_subagent"))
    with store.open_store(root, write=True) as txn:
        for run_id, source in runs:
            txn.put_custody(run_id, {"run_id": run_id, "task_id": "own", "source": source}, state=current.HELD)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "in_worker_process", lambda: False)
    order, events = [], []
    record = delegate_terminal.record_terminal_reconciliation

    def annotate(*args, **kwargs):
        order.append("annotation")
        assert order[0] == "enqueue"
        return record(*args, **kwargs)

    monkeypatch.setattr(delegate_terminal, "record_terminal_reconciliation", annotate)

    class Events:
        def put(self, event):
            if put_fails:
                raise RuntimeError("queue unavailable")
            order.append("enqueue")
            events.append(event)
            notice = event.get("terminal_custody_notice", "")
            assert ("leaf" in notice) is own_run
            assert "review" not in notice
            assert "delegate_terminal_reconciliation" not in pipeline.load_task_result(root, "own")

    pending = []
    pipeline.emit_task_results(
        SimpleNamespace(drive_root=root, repo_dir=tmp_path), None, None, pending, task,
        "Answer.", {"cost": 0.0, "rounds": 1}, {"tool_calls": [], "reasoning_notes": []},
        0.0, root / "logs", ctx=SimpleNamespace(pending_restart_reason=None), event_queue=Events(),
    )
    assert order == ([] if put_fails else ["enqueue", "annotation"])
    stored = pipeline.load_task_result(root, "own")
    assert stored["result"] == "Answer."
    assert stored.get("delegate_terminal_reconciliation", {}).get("open_run_ids", []) == (["leaf"] if own_run else [])
    assert stored.get("delegated_runs_unreconciled", []) == (["leaf"] if own_run else [])
    assert ("delegate_terminal_reconciliation" in stored) is own_run
    assert stored["reason_code"] == "final_message", "an open debt rides beside the cause, never as it (#1583)"
    assert pending[0]["text"] == "Answer."

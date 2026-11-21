"""The body is the one observer of held runs (pool-capacity D3): their speech, their rights floor, the engine's reach.

Driven through the real body (``delegate_collection.BodyRuntime``, its driver parked), the real custody rows, the
supervisor queue's rows as the live tasks, and engine doubles that speak the wire the gateway adapters read. No
nanny runs in these tests: what reaches the owner, and what stops a run wider than its grant, happen with every
author asleep.
"""
from __future__ import annotations

import contextlib
import json
import threading
import time

import pytest

from ouroboros import delegate_activity, delegate_feed
from ouroboros import delegate_collection as body
from ouroboros import delegate_custody_current as current
from ouroboros.gateways import claudexor_observation as obs
from ouroboros.gateways import claudexor_run_events as run_events
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests.test_delegate_body_collection import started

WAIT_MANY = {"method": "POST", "path": "/v2/runs/wait-many"}
EVENTS = {"method": "GET", "path": "/v2/runs/:id/events"}
TOKEN = "t" * 43


def _speech(run_id, seq, text):
    return seq, {"type": "harness.event", "seq": seq, "run_id": run_id,
                 "payload": {"type": "message", "text": text, "harness_id": "codex", "attempt_id": "a1"}}


def _frames(events, *, end=False):
    text = "".join(f"id: {seq}\nevent: {value['type']}\ndata: {json.dumps(value)}\n\n" for seq, value in events)
    return text + ("event: end\ndata: {}\n\n" if end else "")


class _Stream:
    """One open journal read: the frames it serves, or a quiet read held until the journal is released."""
    def __init__(self, journal, text):
        self.journal, self.text, self.status_code = journal, text, 200

    def iter_text(self):
        with self.journal.lock:
            self.journal.open_now += 1
            self.journal.peak = max(self.journal.peak, self.journal.open_now)
        try:
            if self.text:
                yield self.text
            else:
                self.journal.release.wait(30)
        finally:
            with self.journal.lock:
                self.journal.open_now -= 1


class Journal:
    """Every run's journal on one engine: per run, the batches its successive reads serve (then quiet reads)."""
    def __init__(self, batches=None):
        self.batches, self.opens, self.lock = dict(batches or {}), [], threading.Lock()
        self.open_now = self.peak = 0
        self.release = threading.Event()

    def open(self, run_id, after_seq):
        with self.lock:
            self.opens.append((run_id, after_seq))
            queue = self.batches.get(run_id) or []
            text = queue.pop(0) if queue else ""
        return contextlib.nullcontext(_Stream(self, text))


def _engine(journal=None, *, details=None, routes=(WAIT_MANY, EVENTS), attention=None, made=None):
    """An engine double class, one instance per transport the body opens (``made`` counts them)."""
    class Engine:
        engine_version, engine_build_sha, store_identity = "5.1.0", "sha-body-feed", ""
        cancels: list = []

        def __init__(self):
            if made is not None:
                made.append(self)

        def operations(self, *, timeout_sec=None):
            return list(routes)

        def wait_run_attention(self, targets, *, wait_ms, timeout_sec, include_blocking=False):
            if targets == [{"runId": obs.PROBE_RUN_ID, "seen": None}]:
                return {"wakeReason": "timeout", "attention": [], "summaries": [{"problem": "not_found"}]}
            answer = attention.pop(0) if attention else None
            if isinstance(answer, BaseException):
                raise answer
            return {"wakeReason": "timeout", "attention": [], "summaries": [
                {"status": "running", "pending": False, "token": TOKEN, "problem": None} for _ in targets]}

        def get_run(self, run_id, **_kw):
            script = details[run_id]
            return script.pop(0) if len(script) > 1 else script[0]

        def cancel_run(self, run_id, reason=""):
            Engine.cancels.append((run_id, reason))
            details[run_id][:] = [{**details[run_id][0], "summary": {
                **details[run_id][0]["summary"], "state": "cancelled"}}]
            return {"accepted": True, "status": "cancelling"}

        def close(self):
            pass
    if journal is not None:
        Engine.open_run_events = lambda self, run_id, after_seq, timeout_sec: journal.open(run_id, after_seq)
    if WAIT_MANY not in routes:
        del Engine.wait_run_attention
    return Engine


class _Bus:
    def __init__(self):
        self.events, self.lock = [], threading.Lock()

    def put(self, event):
        with self.lock:
            self.events.append(event)

    def frames(self, task_id=None):
        with self.lock:
            return [event for event in self.events if task_id in (None, event["task_id"])]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Live tasks in the supervisor queue (asleep: pending, no worker), a bus, fresh process memos."""
    from supervisor import queue as q
    from supervisor.active_activity import get_direct_activity_registry

    monkeypatch.setattr(obs, "_SUPPORT", {})
    monkeypatch.setattr(run_events, "_SUPPORT", {})
    delegate_activity.reset_process_memo()
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "RUNNING", {})
    get_direct_activity_registry().clear()
    bus = _Bus()
    monkeypatch.setattr(delegate_feed, "_bus", lambda: bus)
    yield type("World", (), {"root": tmp_path, "queue": q, "bus": bus})
    delegate_activity.reset_process_memo()


def _asleep(world, task_id, chat_id=7, **row):
    world.queue.PENDING.append({"id": task_id, "type": "task", "chat_id": chat_id, **row})


@contextlib.contextmanager
def _body(world, factory):
    """The real body with its driver parked in its first tick; the test drives the real tick itself."""
    release, entered, tick = threading.Event(), threading.Event(), body.BodyRuntime.tick
    body.BodyRuntime.tick = lambda _runtime: entered.set() or release.wait(30)
    try:
        runtime = body.start_body_runtime(world.root, gateway_factory=factory)
        assert entered.wait(5)
    finally:
        body.BodyRuntime.tick = tick
    try:
        yield runtime, (lambda: tick(runtime))
    finally:
        runtime.close(wait=False)
        release.set()
        runtime.close(wait=True)


def _until(check, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.01)
    return check()


def _knows_the_journal(runtime, factory):
    """The observation pass read this engine's catalog (the first pass of any body does): the stream is listed."""
    runtime._observing = factory()
    assert run_events.route_listed(runtime._observing, run_events.RUN_EVENTS_OPERATION, timeout_sec=1)
    runtime._compact_wait = True


@pytest.mark.parametrize("outcome", ["prepared", "delivery_unknown"])
def test_fresh_body_reconstructs_pending_owner_frame_retry(world, monkeypatch, outcome):
    import hashlib

    from ouroboros import delegate_message_store as frames
    from tests.test_delegate_message import _Stub, _install, _receipts
    from tests.test_owner_words_at_receipt import _open_run

    entry = _open_run(world.root, "run-live", task_id="t-nanny", root_task_id="t-nanny")
    _asleep(world, "t-nanny")
    ctx = frames.starter(world.root, entry)
    row, _prior = frames.place_frame(ctx, entry, "Keep the original scope", owner_words={
        "address": "chat 7 / owner-m1", "chat_id": 7, "ts": "2026-11-09T10:00:00+00:00"}, host=True)
    message_id, wire = row["frame"]["message_id"], row["wire_text"]
    if outcome != "prepared":
        frames.record_outcome(ctx, entry.run_id, message_id, outcome, "daemon_unreachable")
    stub = _install(monkeypatch, _Stub(result={"outcome": "delivered", "reason": ""}))
    details = {entry.run_id: [{"summary": {"runId": entry.run_id, "state": "running"}, "lastSeq": 1}]}
    with _body(world, _engine(details=details, routes=())) as (runtime, tick):
        assert frames.pending_frames(ctx, entry.run_id)
        tick()
        assert _until(lambda: runtime.observation(entry.run_id).get("answered"))
        tick()
        assert _until(lambda: bool(stub.posts)), "the durable frame needs no process-local hint"
        runtime.feed.placing[entry.run_id].result(timeout=10)
        tick()
        assert [(post["key"], post["text"]) for post in stub.posts] == [(message_id, wire)]
        assert frames.pending_frames(ctx, entry.run_id) == []
        receipts = [r for r in _receipts(world.root) if r["message_id"] == message_id]  # digest and size, no read
        digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        assert receipts and {r.get("text_sha256") for r in receipts} == {digest} and stub.reads == []


def test_an_answered_run_without_pending_frames_costs_only_one_followed_set_read(world, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import delegate_message_store as frames, store

    entries = [started(world.root, rid, "task") for rid in ("run-empty", "run-delivered", "run-elsewhere")]
    ctx = frames.starter(world.root, entries[1])
    for entry in entries[1:]:
        row, _prior = frames.place_frame(ctx, entry, "scope", {"address": "owner:1"}, host=True)
        if entry.run_id == "run-delivered":
            frames.record_outcome(ctx, entry.run_id, row["frame"]["message_id"], "delivered", "")
    opened, queries, real_open = [], [], store.open_store

    @contextlib.contextmanager
    def counted(*args, **kwargs):
        opened.append(kwargs["write"])
        with real_open(*args, **kwargs) as txn:
            txn.conn.set_trace_callback(queries.append)
            try:
                yield txn
            finally:
                txn.conn.set_trace_callback(None)

    monkeypatch.setattr(store, "open_store", counted)
    monkeypatch.setattr(frames, "starter", lambda *_a: pytest.fail("no starter read without a pending frame"))
    runtime = SimpleNamespace(observation=lambda _rid: {"answered": True}, observers=SimpleNamespace(
        submit=lambda *_a: pytest.fail("no job without a pending frame in a followed run")))
    feed = delegate_feed.Feed(world.root)
    held = {entry.run_id: SimpleNamespace(entry=entry) for entry in entries[:2]}
    for _ in range(2):
        feed.place(runtime, held)
    assert opened == [False, False]
    assert len([query for query in queries if "FROM obligations" in query]) == 2
    assert not frames.pending_runs(world.root, held)


def test_the_feed_reaches_the_owner_once_per_new_journal_span_with_the_nanny_asleep(world):
    entry = started(world.root, "run-s", "task-s")
    _asleep(world, "task-s")
    journal = Journal({"run-s": [_frames([_speech("run-s", 1, "hello"), _speech("run-s", 2, "world")]),
                                 _frames([_speech("run-s", 3, "again")])]})
    factory = _engine(journal)
    with _body(world, factory) as (runtime, tick):
        _knows_the_journal(runtime, factory)
        tick()
        assert _until(lambda: len(world.bus.frames()) == 2 and ("run-s", 3) in journal.opens)
        assert _until(lambda: journal.open_now == 1), "the follow holds its next read: the run is quiet now"
        tick()  # a second tick starts no second follow of the same run
        frames = world.bus.frames()
        assert [frame["progress_meta"]["delegated_activity"]["through_seq"] for frame in frames] == [2, 3]
        assert [frame["progress_meta"]["delegated_activity"]["after_seq"] for frame in frames] == [0, 2]
        assert "hello" in frames[0]["text"] and "again" in frames[1]["text"]
        assert journal.opens == [("run-s", 0), ("run-s", 2), ("run-s", 3)], "each read resumes after the last shown"
        assert {(frame["type"], frame["chat_id"], frame["task_id"], frame["is_progress"], frame["role"],
                 frame["system_type"], frame["progress_meta"]["narration"]) for frame in frames} == {
            ("send_message", 7, entry.task_id, True, "system", "host_progress", False)}
        journal.release.set()
    assert len(world.bus.frames()) == 2, "a quiet read publishes nothing"


def test_a_new_body_resumes_at_the_last_retained_range_and_a_task_nobody_holds_is_not_read(world):
    started(world.root, "run-r", "task-r")
    _asleep(world, "task-r")
    journal = Journal({"run-r": [_frames([_speech("run-r", 1, "one"), _speech("run-r", 2, "two")])]})
    factory = _engine(journal)
    with _body(world, factory) as (runtime, tick):
        _knows_the_journal(runtime, factory)
        tick()
        assert _until(lambda: len(world.bus.frames()) == 1 and len(journal.opens) == 2)
        journal.release.set()
    delegate_activity.reset_process_memo()  # a restarted server: nothing of what this process showed is known
    journal.release.clear()
    journal.batches["run-r"] = [_frames([_speech("run-r", 1, "one"), _speech("run-r", 2, "two"),
                                         _speech("run-r", 3, "three")])]
    with _body(world, factory) as (runtime, tick):
        _knows_the_journal(runtime, factory)
        tick()
        assert _until(lambda: len(world.bus.frames()) == 2)
        assert journal.opens[2] == ("run-r", 0), "the retained range (1..2) is read again once"
        assert world.bus.frames()[1]["progress_meta"]["delegated_activity"]["through_seq"] == 3
        journal.release.set()
    world.queue.PENDING.clear()  # the task is over: nobody has a card for the run's words
    opened = len(journal.opens)
    with _body(world, factory) as (runtime, tick):
        _knows_the_journal(runtime, factory)
        tick()
        time.sleep(0.2)
        assert runtime.feeds == {} and len(journal.opens) == opened


def test_the_frame_has_the_shape_its_worker_gave_it(world, monkeypatch):
    """The body's frame for a delegated record is the event the task's own worker emitted for it."""
    import queue as queue_module
    from types import SimpleNamespace

    from ouroboros.agent import OuroborosAgent
    from ouroboros.subagent_messages import initiator_meta, subagent_message_meta

    task = {"id": "child-1", "chat_id": 9, "_attempt": 2, "delegation_role": "subagent", "parent_task_id": "root-1",
            "root_task_id": "root-1", "metadata": {"initiator": "consciousness", "subagent_role": "researcher"}}
    _asleep(world, "child-1", chat_id=9, **{k: v for k, v in task.items() if k not in ("id", "chat_id")})
    record = {"v": 1, "task_id": "child-1", "run_id": "run-c", "after_seq": 0, "through_seq": 1,
              "source": {"kind": "run_events", "read_through": 1, "events": 1},
              "parts": [{"kind": "message", "actor": "codex/a1", "text": "hi", "first_seq": 1, "last_seq": 1}]}
    target = delegate_feed.live_tasks(["child-1"])["child-1"]
    delegate_feed.publisher("child-1", target)("speech", delegated_activity=record)
    worker = SimpleNamespace(_event_queue=queue_module.Queue(), _last_progress_ts=0.0,
                             _current_chat_id=None, _current_task_id=None)
    bound = {**task["metadata"], **{key: task[key] for key in ("parent_task_id", "root_task_id", "delegation_role")}}
    meta = subagent_message_meta(bound, task_id="child-1", event="progress")  # the worker's own projection
    meta.update(initiator_meta(bound))
    emit = OuroborosAgent._bind_task_progress(worker, "child-1", 9, meta, task_attempt=2)
    worker._emit_progress = lambda text, **kw: OuroborosAgent._emit_progress(worker, text, **kw)
    emit("speech", delegated_activity=record)
    ours, theirs = world.bus.frames()[0], worker._event_queue.get_nowait()
    assert {**ours, "ts": ""} == {**theirs, "ts": ""}
    assert ours["progress_meta"]["delegated_activity"] == record


def test_an_engine_without_the_run_journal_keeps_the_window_view(world):
    started(world.root, "run-w", "task-w")
    _asleep(world, "task-w")

    def detail(seq, rows):
        return {"summary": {"runId": "run-w", "state": "running", "effectiveAccess": "readonly"}, "lastSeq": seq,
                "timeline": [{"type": "message", "title": text, "textKind": "message", "detail": text,
                              "harnessId": "codex", "attemptId": "a1"} for text in rows], "pendingInteractions": []}
    details = {"run-w": [detail(2, ["one", "two"]), detail(2, ["one", "two"]), detail(3, ["one", "two", "three"])]}
    factory = _engine(details=details, routes=())
    with _body(world, factory) as (runtime, _tick):
        runtime.feed.refresh(runtime._held())
        entry = current.run(world.root, "run-w")
        for _ in range(3):
            runtime._observe(entry)
    frames = world.bus.frames()
    assert [frame["progress_meta"]["delegated_activity"]["source"]["kind"] for frame in frames] == ["timeline_window"] * 2
    assert [frame["progress_meta"]["delegated_activity"]["through_seq"] for frame in frames] == [2, 3], \
        "an unchanged detail publishes nothing"
    observation = frames[0]["progress_meta"]["executor_observation"]
    assert (observation["run_id"], observation["harness_id"], observation["revision"]) == ("run-w", "codex", 2)


def _engine_refusal(code, retryable):
    """The engine's own refusal inside an open event stream (``run-events-stream.ts``)."""
    from tests.test_delegated_activity import _refusal

    return _refusal(code, retryable)


def _reclaimed_detail(run_id, seq, texts):
    return {"summary": {"runId": run_id, "state": "running", "effectiveAccess": "readonly"}, "lastSeq": seq,
            "timeline": [{"type": "message", "title": text, "textKind": "message", "detail": text,
                          "harnessId": "codex", "attemptId": "a1"} for text in texts], "pendingInteractions": []}


@pytest.mark.parametrize(("served", "retryable"), [([(1, "hello"), (2, "world")], False), ([], True)],
                         ids=["partial-then-reclaimed", "refused-at-once"])
def test_a_refused_journal_shows_what_it_served_and_a_refusal_for_good_is_not_read_again(world, served, retryable):
    """#3051 on the body's follow: the words served before the engine's refusal reach the owner, a retryable refusal
    is a failed read asked again next beat, and a refusal for good is never read again."""
    entry = started(world.root, "run-s", "task-s")
    _asleep(world, "task-s")
    code = "run_event_not_found" if retryable else "run_expired_by_retention"
    journal = Journal({"run-s": [_frames([_speech("run-s", seq, text) for seq, text in served])
                                 + _engine_refusal(code, retryable)]})
    factory = _engine(journal)
    with _body(world, factory) as (runtime, tick):
        _knows_the_journal(runtime, factory)
        tick()
        if retryable:
            assert _until(lambda: journal.opens == [("run-s", 0), ("run-s", 0)]), journal.opens
            assert world.bus.frames() == [] and not delegate_feed.journal_refused(entry)
            journal.release.set()
        else:
            assert _until(lambda: len(world.bus.frames()) == 1)
            activity = world.bus.frames()[0]["progress_meta"]["delegated_activity"]
            assert activity["through_seq"] == 2 and "hello" in world.bus.frames()[0]["text"]
            assert _until(lambda: runtime.feeds["run-s"].done()) and delegate_feed.journal_refused(entry)
            tick()  # refused for good: no second follow, no second read
            time.sleep(0.1)
            assert journal.opens == [("run-s", 0)]
    assert len(world.bus.frames()) == (0 if retryable else 1)


def test_after_a_refusal_for_good_the_window_view_shows_first_the_words_a_failed_frame_left(world, monkeypatch):
    """#3051: the frame of the words read just before the refusal could not be handed over. The journal is not read
    again; the body's detail read shows those words first (its window view), then the range it cannot show."""
    entry = started(world.root, "run-s", "task-s")
    _asleep(world, "task-s")
    journal = Journal({"run-s": [_frames([_speech("run-s", 1, "hello"), _speech("run-s", 2, "world")])
                                 + _engine_refusal("run_expired_by_retention", False)]})
    factory = _engine(journal, details={"run-s": [_reclaimed_detail("run-s", 5, ["later words"])]})
    real_put, failed = world.bus.put, []

    def put_after_a_failure(event):
        if not failed:
            failed.append(event)
            raise RuntimeError("progress channel closed")
        real_put(event)

    monkeypatch.setattr(world.bus, "put", put_after_a_failure)
    with _body(world, factory) as (runtime, tick):
        _knows_the_journal(runtime, factory)
        tick()
        assert _until(lambda: failed and runtime.feeds["run-s"].done()) and delegate_feed.journal_refused(entry)
        assert world.bus.frames() == [] and journal.opens == [("run-s", 0)]
        runtime._observe(current.run(world.root, "run-s"))
    shown = [frame["progress_meta"]["delegated_activity"] for frame in world.bus.frames()]
    assert [part["text"] for part in shown[0]["parts"]] == ["hello", "world"], "the undelivered words come first"
    assert shown[1]["source"]["kind"] == "timeline_window" and shown[1]["source"]["retryable"] is False
    assert shown[1]["gaps"][0]["reason"] == "stream_error:run_expired_by_retention"
    assert shown[1]["gaps"][0]["retryable"] is False
    assert journal.opens == [("run-s", 0)], "a journal refused for good is not read again"


def test_a_refusal_with_nothing_new_publishes_the_shown_ranges_availability_once(world, monkeypatch):
    """#3051: an earlier range was shown but its full source could not be kept here; the engine then refuses the
    journal for good with no new event. Live readers learn the range is not recoverable from a reread, once."""
    from ouroboros.subagent_messages import delegated_activity_meta

    started(world.root, "run-s", "task-s")
    _asleep(world, "task-s")
    monkeypatch.setattr(delegate_activity, "_persist", lambda *_a: {})
    journal = Journal({"run-s": [_frames([_speech("run-s", 1, "earlier words " * 200)]),
                                 _engine_refusal("run_expired_by_retention", False)]})
    factory = _engine(journal)
    with _body(world, factory) as (runtime, tick):
        _knows_the_journal(runtime, factory)
        tick()
        assert _until(lambda: len(world.bus.frames()) == 2 and runtime.feeds["run-s"].done())
    first, second = [frame["progress_meta"]["delegated_activity"] for frame in world.bus.frames()]
    assert first["source"].get("retained") is False
    assert (second["after_seq"], second["through_seq"]) == (0, 1) and not second["parts"] and not second.get("gaps")
    assert second["source"]["kind"] == "timeline_window" and second["source"]["retryable"] is False
    assert delegated_activity_meta(second, task_id="task-s") == second
    assert journal.opens == [("run-s", 0), ("run-s", 1)]


def test_a_run_on_a_work_engine_is_followed_through_that_engine(world, monkeypatch):
    """#3054: a run started on a work engine is read through that engine; the owned engine's transport never reads it."""
    from ouroboros import execution_domain

    _asleep(world, "task-w")
    bound = {"descriptor": "/work/descriptor", "store": "work-store"}
    entry = type("Bound", (), {"run_id": "run-w", "task_id": "task-w", "engine_ref": bound})()
    journal = Journal({"run-w": [_frames([_speech("run-w", 1, "from the work engine")])]})
    made: list = []
    owned = _engine(None, routes=())  # the owned engine has no journal route and must not be asked
    work = _engine(journal, made=made)
    seen: list = []

    def run_gateway(record, default, **_kw):
        seen.append(record.run_id)
        return work()

    monkeypatch.setattr(execution_domain, "run_gateway", run_gateway)
    feed = delegate_feed.Feed(world.root)
    feed.live = {"task-w": {"chat_id": 7, "meta": {}, "attempt": 1, "direct": False}}
    runtime = type("Runtime", (), {"_gateway": staticmethod(owned)})()
    cursor, transport = delegate_feed._Cursor(), []
    assert feed._beat(runtime, entry, feed.live["task-w"], cursor, transport) is True
    assert seen == ["run-w"] and journal.opens == [("run-w", 0)] and cursor.seq == 1
    assert "from the work engine" in world.bus.frames()[0]["text"] and transport == made
    for gateway in transport:
        gateway.close()


def test_the_body_records_an_async_questions_whole_set_fact_for_its_parent(world):
    """#3066: the parent's view of a run never reads a question that holds nothing as a blocked agent."""
    from ouroboros.tools.control_task_results import _delegated_run_facts

    entry = started(world.root, "run-q", "task-q")
    feed = delegate_feed.Feed(world.root)
    fact = {"run_state": "running", "waiting_on_user": True, "async_questions_only": True, "answered": True}
    feed.record(entry, fact)
    record, why = delegate_feed.read_observation(world.root, "run-q")
    assert not why and (record["waiting_on_user"], record["async_questions_only"]) == (True, True)
    [row] = _delegated_run_facts(world.root, "task-q")
    assert row["observation"]["observed_waiting_on_user"] is True
    assert row["observation"]["observed_async_questions_only"] is True
    feed.record(entry, {**fact, "async_questions_only": False})  # a blocking question arrived: the fact changes
    [row] = _delegated_run_facts(world.root, "task-q")
    assert row["observation"]["observed_waiting_on_user"] is True and "observed_async_questions_only" not in row["observation"]


def test_the_body_holds_every_observed_run_to_its_recorded_grant(world, monkeypatch):
    from ouroboros import usage_accounting

    monkeypatch.setattr(usage_accounting, "record_subscription_session", lambda *a, **kw: "attempt")
    wide = started(world.root, "run-wide", "task-a")  # its start row records the readonly grant
    fine = started(world.root, "run-fine", "task-a")

    def running(run_id, access):
        return [{"summary": {"runId": run_id, "state": "running", "effectiveAccess": access}, "lastSeq": 3,
                 "pendingInteractions": []}]
    factory = _engine(details={"run-wide": running("run-wide", "full"), "run-fine": running("run-fine", "readonly")},
                      routes=())
    with _body(world, factory) as (runtime, _tick):
        runtime._observe(wide)
        runtime._observe(fine)
        runtime._observe(current.run(world.root, "run-wide"))  # the next beat reads the stopped run again
    assert factory.cancels == [("run-wide", "access_profile_widened")], "one stop, the floor's own reason"
    assert current.run(world.root, "run-wide").containment_fault.get("reason") == "access_profile_widened"
    assert not current.run(world.root, "run-fine").containment_fault, "a run within its grant is left alone"


def test_the_wake_read_leaves_a_recorded_grant_to_the_body_and_holds_a_row_without_one(tmp_path, monkeypatch):
    """A wake's one read (``tools/delegate._read_run``) does not cancel a run whose start row recorded its grant: the
    body holds that run's floor. A row with no recorded grant has only the read's live derivation, which halts it."""
    from tests._delegated_transport_shared import _delegating_ctx, _isolation_stub
    from ouroboros.tools import delegate

    ctx = _delegating_ctx(tmp_path, acting=False)
    for access, status in (("readonly", "progress"), ("", "refused")):
        cancelled = _isolation_stub(monkeypatch, run_dir=tmp_path / "run-1", effective_access="full")
        delegate._CUSTODY.clear()
        delegate._CUSTODY["run-1"] = delegate._RunCustody(task_id=ctx.task_id, route_id="some-route", model="m",
                                                          project_id="prj", access=access, mode="ask")
        out = json.loads(delegate._read_run(ctx, "run-1", 0))
        assert out["status"] == status, (access, out)
        assert bool(cancelled) is (not access)
    delegate._CUSTODY.clear()


def _hole(reason="daemon_unreachable"):
    return ClaudexorUnavailable("daemon_unreachable", "nothing answered", observation_timeout=True,
                                observation_reason=reason)


def test_an_unreachable_engine_is_one_owner_line_each_way_per_episode(world):
    """N passes nothing answered tell each live task once; the first answered pass tells it once more and closes
    the episode. A slow read and a refusal that never reached the daemon say nothing, and the second episode is a
    second pair of lines on both client surfaces (its own stamp, named in the text)."""
    a1, a2, b = (started(world.root, rid, tid) for rid, tid in (("run-a1", "task-a"), ("run-a2", "task-a"),
                                                               ("run-b", "task-b")))
    _asleep(world, "task-a")
    _asleep(world, "task-b", chat_id=8)
    passes = [_hole(), _hole(), _hole("observation_read_timeout"), None, _hole("observation_read_timeout"),
              None, _hole(), None]
    factory = _engine(attention=passes)
    with _body(world, factory) as (runtime, _tick):
        runtime.feed.refresh(runtime._held())
        entries = [current.run(world.root, entry.run_id) for entry in (a1, a2, b)]
        for index in range(len(passes)):
            if index == 6:
                time.sleep(0.002)  # a second episode in another millisecond
                runtime._drop_observer_gateway()
                runtime._gateway = lambda: (_ for _ in ()).throw(ClaudexorUnavailable("daemon_not_discovered", "gone"))
                with pytest.raises(ClaudexorUnavailable):
                    runtime._watch(entries)  # never reached the daemon: opens nothing, closes nothing
                del runtime._gateway
            runtime._watch(entries)
    lines = [(frame["task_id"], frame["text"], frame["progress_meta"]) for frame in world.bus.frames()]
    for task in ("task-a", "task-b"):
        mine = [(text, meta) for task_id, text, meta in lines if task_id == task]
        assert [text.startswith("💬 Delegation daemon unreachable since ") for text, _ in mine] == [
            True, False, True, False], mine
        assert [text.startswith("💬 Delegation daemon reachable again; the outage that began at ")
                for text, _ in mine] == [False, True, False, True], mine
        keys = [meta["toast_once"] for _, meta in mine]
        assert len(set(keys)) == 4 and all(key.startswith(f"{task}:delegation_daemon_") for key in keys)
        assert keys[0].rsplit(":", 1)[1] == keys[1].rsplit(":", 1)[1] != keys[2].rsplit(":", 1)[1]
        assert [meta["toast_tone"] for _, meta in mine] == ["warn", "ok", "warn", "ok"]
        assert {meta["task_incident"] for _, meta in mine} == {"delegation_daemon_unreachable"}


def test_the_body_records_a_runs_state_for_its_parent_when_it_changes_and_removes_it_once_settled(world, monkeypatch):
    """The parent's view of a held run (``delegate_feed.read_observation``) has one writer, the body: a record when
    the run's state or question changes or the engine stops or resumes answering (the owner line's outage class),
    dated by when that began, never once per read; a slow read changes nothing. A settled run's record goes, and a
    new body removes the records an earlier one left for runs it no longer holds."""
    from types import SimpleNamespace

    from ouroboros import delegate_custody as custody

    clock = [1_800_000_000.0]
    monkeypatch.setattr(delegate_feed, "time", SimpleNamespace(
        time=lambda: clock[0], strftime=time.strftime, gmtime=time.gmtime, monotonic=time.monotonic))
    writes, real = [], delegate_feed.atomic_write_json
    monkeypatch.setattr(delegate_feed, "atomic_write_json", lambda path, value: writes.append(value) or real(path, value))
    entry = started(world.root, "run-p", "task-p")

    def detail(run_id="run-p", asks=False):
        return {"summary": {"runId": run_id, "state": "running", "effectiveAccess": "readonly", "waitingOnUser": asks},
                "lastSeq": 3, "pendingInteractions": []}
    details = {"run-p": [detail()]}
    leftover = delegate_feed.observation_path(world.root, "run-gone")
    with _body(world, _engine(details=details, routes=())) as (runtime, _tick):
        for _ in range(3):
            runtime._observe(entry)
            clock[0] += 5
        assert [(w["run_state"], w["answered"], w["since"]) for w in writes] == [
            ("running", True, "2027-01-15T08:00:00Z")], "three reads of one state: one record"
        details["run-p"][:] = [detail("run-p", asks=True)]
        runtime._observe(entry)
        clock[0] += 5
        runtime._unanswered([entry], _hole(), None)
        clock[0] += 5
        runtime._unanswered([entry], _hole(), None)
        runtime._unanswered([entry], _hole("observation_read_timeout"), None)
        assert len(writes) == 3
        record, why = delegate_feed.read_observation(world.root, "run-p")
        assert why == "" and (record["run_state"], record["waiting_on_user"], record["reason"]) == (
            "running", True, "daemon_unreachable"), "the last answered state stands beside why it is not current"
        assert (record["since"], record["last_answered_at"]) == ("2027-01-15T08:00:20Z", "2027-01-15T08:00:15Z")
        runtime._observe(entry)
        assert len(writes) == 4 and delegate_feed.read_observation(world.root, "run-p")[0]["answered"] is True
        assert custody.emit(world.root, custody.SETTLED, {"run_id": "run-p", "task_id": "task-p", "state": "succeeded"})
        runtime.feed.refresh(runtime._held())
        assert delegate_feed.read_observation(world.root, "run-p") == (None, "no_observation_record")
    leftover.write_text("{}", encoding="utf-8")  # an earlier body's record of a run that settled while none ran
    kept = started(world.root, "run-k", "task-k")
    with _body(world, _engine(details={"run-k": [detail("run-k")]}, routes=())) as (runtime, _tick):
        runtime._observe(kept)
        runtime.feed.refresh(runtime._held())
        assert not leftover.exists() and delegate_feed.read_observation(world.root, "run-k")[1] == ""


@pytest.mark.parametrize("runs", [20, 60, 200])
def test_one_observer_reads_every_run_at_once_and_a_quiet_run_blocks_no_speech(world, runs):
    """The cost of one observer at 20/60/200 held runs: one journal read in flight per run, each on the run's own
    transport and thread, no pool width in between. With every other run's read held open (a quiet run), one run's
    two spans still reach the owner at once, each once, and a second tick submits nothing new."""
    for index in range(runs):
        started(world.root, f"run-{index:03d}", f"task-{index:03d}")
        _asleep(world, f"task-{index:03d}")
    journal = Journal({"run-000": [_frames([_speech("run-000", 1, "first")]),
                                   _frames([_speech("run-000", 2, "second")])]})
    made = []
    factory = _engine(journal, made=made)
    with _body(world, factory) as (runtime, tick):
        _knows_the_journal(runtime, factory)
        made.clear()
        tick()
        assert _until(lambda: journal.open_now == runs and len(world.bus.frames()) == 2), (journal.open_now,
                                                                                        len(world.bus.frames()))
        assert journal.peak == runs, "every run's read is in flight at once"
        assert [frame["task_id"] for frame in world.bus.frames()] == ["task-000", "task-000"]
        assert len(made) == runs, "one transport per followed run (the compact pass keeps the observer's own)"
        opened = sorted(journal.opens)
        assert opened.count(("run-000", 0)) == 1 and len(opened) == runs + 2, "one read per quiet run, three for the speaker"
        with runtime.lock:
            feeds = dict(runtime.feeds)
        tick()
        with runtime.lock:
            assert runtime.feeds == feeds, "a run already followed is not followed twice"
        assert len(runtime.observers.threads) >= runs
        journal.release.set()
    assert runtime.observers.threads == set(), "the body's close joins every job it started"


@pytest.mark.parametrize("runs", [20, 60, 200])
def test_an_engine_outage_costs_each_followed_run_one_attempt_per_beat(world, monkeypatch, runs):
    """Every follow's transport refuses: each run retries once per observation beat, never in a busy loop."""
    monkeypatch.setattr(delegate_feed, "_TICK_SEC", 0.2)
    for index in range(runs):
        started(world.root, f"run-{index:03d}", f"task-{index:03d}")
        _asleep(world, f"task-{index:03d}")
    attempts = []
    factory = _engine(Journal())

    def refusing():
        attempts.append(time.monotonic())
        raise _hole()
    with _body(world, factory) as (runtime, tick):
        _knows_the_journal(runtime, factory)
        runtime._gateway = refusing
        began = time.monotonic()
        tick()
        time.sleep(0.7)
        elapsed = time.monotonic() - began
    beats = elapsed / 0.2 + 2
    assert runs <= len(attempts) <= runs * beats, (len(attempts), runs, beats)

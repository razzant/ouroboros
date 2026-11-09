"""A Host-admitted Presence turn's body runs in a conversation-lane carrier (A7).

The Host keeps authentication, capture, dedup and its reservation; ``dispatch_presence_turn``
admits the body as a lane RUNNING row (origin ``presence``, carrier ``conversation``); the
carrier re-admits from durable facts, takes the gate itself, runs ``run_presence_turn`` and
reports its settlement as ONE frame the Host resolves its waiter from; errors travel by
value. An arrival for a carried author is accepted through one transaction under the
author's mailbox lock against its durable intake fact.
"""

import json
import pathlib
import queue as stdqueue
from types import SimpleNamespace

import pytest

from ouroboros import presence_carrier, presence_runner
from ouroboros.presence_carrier import (
    PRESENCE_RUN_KEY, carry_presence_turn, dispatch_presence_turn, handle_presence_turn_settled, presence_seed_task,
)
from ouroboros.presence_events import PresenceTurnError
from ouroboros.presence_runner import PresenceTurnEvent, PresenceTurnExecutions, PresenceTurnResult
from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
from supervisor import conversation_lane, workers
from tests.test_owner_wait_pool import InputQueue, Process, pool as pool  # noqa: F401 -- fixture
from tests.test_presence_runner import _admission, _event
from tests.worker_pool_support import affirm_ready


_REAL_LANE_RECONCILE = conversation_lane.reconcile_conversation_lane


def _payload(event: PresenceTurnEvent) -> dict:
    return {"source_event_id": event.source_event_id, "provider": event.provider, "account_id": event.account_id,
            "conversation_id": event.conversation_id, "thread_id": event.thread_id,
            "conversation_key": event.conversation_key, "actor": dict(event.actor),
            "conversation": dict(event.conversation), "message": dict(event.message), "text": event.text}


@pytest.fixture
def lane(pool, monkeypatch):
    monkeypatch.setattr(conversation_lane, "reconcile_conversation_lane", _REAL_LANE_RECONCILE)
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: pytest.fail("the server constructed an agent"))
    spare = workers.Worker(5, Process(5005), InputQueue(), lane=LANE_CONVERSATION)
    affirm_ready(spare)
    workers.WORKERS[5] = spare
    pool.spare = spare
    return pool


def test_the_host_dispatches_the_body_as_a_presence_lane_row(lane):
    admission, event = _admission(), _event()
    registry = PresenceTurnExecutions()
    turn_id = presence_runner.presence_turn_task_id(admission.binding_id, event.source_event_id)  # the Host's id
    execution, started = registry.start_or_dispatch(
        turn_id, identity="id-1", reserve=lambda: True, release=lambda: None,
        dispatch=lambda execution: dispatch_presence_turn(
            execution, drive_root=lane.root, skill_name="telegram-bot", admission=admission, event=event,
            event_payload=_payload(event), staged_files=[], gate_wait=True))
    assert started and execution is registry.find(turn_id)
    rows = {tid: meta for tid, meta in workers.RUNNING.items() if meta.get("lane") == LANE_CONVERSATION}
    assert len(rows) == 1
    tid, meta = next(iter(rows.items()))
    task = meta["task"]
    assert task["_presence_turn"] and task["_execution_owner"]["kind"] == "presence"
    assert task["_execution_owner"]["lane"] == LANE_CONVERSATION and task["_execution_owner"]["executor"] == EXECUTOR_WORKER
    assert task["metadata"]["presence_request_waits"] is True
    dispatched = lane.spare.in_q.get_nowait()
    assert dispatched["id"] == tid and dispatched[PRESENCE_RUN_KEY]["skill_name"] == "telegram-bot"
    assert dispatched[PRESENCE_RUN_KEY]["event"]["source_event_id"] == event.source_event_id
    assert not workers.PENDING
    # The settlement frame resolves the waiter and releases the row and the slot.
    handle_presence_turn_settled({"turn_id": tid, "physical_id": tid, "error": None,
                                  "result": presence_carrier._result_payload(
                                      PresenceTurnResult(outcome="message", text="Hi", task_id=tid))},
                                 SimpleNamespace(RUNNING=workers.RUNNING, WORKERS=workers.WORKERS))
    assert execution.result.result(timeout=1).text == "Hi"
    assert tid not in workers.RUNNING and lane.spare.busy_task_id is None and registry.find(tid) is None


def test_the_carrier_runs_the_turn_under_the_gate_and_settles_by_frame(tmp_path, monkeypatch):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    admission, event = _admission(), _event()
    from ouroboros import presence_admission
    monkeypatch.setattr(presence_admission, "admit_presence_turn", lambda **kw: admission)
    seen = {}

    class Agent:
        def __init__(self, **kwargs):
            self.owner_wait_callback = None

        def handle_task(self, task):
            seen.update(task)
            from ouroboros.task_results import write_task_result
            write_task_result(pathlib.Path(data), task["id"], "completed", metadata=task["metadata"], result="Hi")
            return [{"type": "presence_result", "outcome": "message", "text": "Hi", "work_ref": ""}]

    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **kwargs: Agent(**kwargs))
    task = presence_seed_task(admission=admission, event=event, skill_name="telegram-bot",
                              event_payload=_payload(event), staged_files=[], gate_wait=True)
    task["_execution_owner"] = {"kind": "presence", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}
    out_q = stdqueue.Queue()
    carry_presence_turn(task, out_q, repo, data, owner_wait_callback="pooled-wait")
    frames = []
    while not out_q.empty():
        frames.append(out_q.get_nowait())
    settled = [f for f in frames if f["type"] == "presence_turn_settled"]
    assert len(settled) == 1 and settled[0]["error"] is None and settled[0]["result"]["text"] == "Hi"
    assert seen["_execution_owner"]["lane"] == LANE_CONVERSATION  # the carrier facts rode the presence record
    assert seen["metadata"]["presence_request_waits"] is True  # the Host request waits: the post-task rule's fact
    # The gate was taken and released by the carrier: a second turn in the conversation is admissible.
    gate = presence_runner._configured_gate(data)
    lease, busy = gate.try_admit(event.conversation_key)
    assert lease is not None and busy == ""
    lease.release()


def test_a_refusal_travels_by_value_to_the_waiter(lane):
    registry = PresenceTurnExecutions()
    execution, _started = registry.start_or_dispatch("presence-2", identity="id-2", reserve=lambda: True,
                                                      release=lambda: None, dispatch=lambda execution: None)
    error = PresenceTurnError("presence_capacity_full", "conversation", disposition="retry")
    error.retry_after = 1
    handle_presence_turn_settled({"turn_id": "presence-2", "physical_id": "presence-2", "result": None,
                                  "error": presence_carrier._error_payload(error)},
                                 SimpleNamespace(RUNNING=workers.RUNNING, WORKERS=workers.WORKERS))
    with pytest.raises(PresenceTurnError) as raised:
        execution.result.result(timeout=1)
    assert raised.value.code == "presence_capacity_full" and raised.value.retry_after == 1
    assert registry.find("presence-2") is None


def test_a_dispatch_that_raises_settles_the_waiter_and_reraises(lane):
    registry = PresenceTurnExecutions()

    def boom(execution):
        raise RuntimeError("no lane")

    with pytest.raises(RuntimeError):
        registry.start_or_dispatch("presence-3", identity="", reserve=lambda: True, release=lambda: None, dispatch=boom)
    assert registry.find("presence-3") is None


def test_a_remote_author_accepts_mail_only_while_its_durable_intake_is_open(tmp_path):
    from ouroboros.conversation_lease import (
        accept_author_mail, author_intake_accepting, close_author_intake, open_author_intake,
    )

    root = tmp_path
    (root / "memory" / "owner_mailbox").mkdir(parents=True)
    appended = []

    def append(admit=None):
        if admit is not None and not admit():
            return False
        appended.append(1)
        return True

    assert accept_author_mail(root, "author-1", "key-1", append) is False  # no fact: nobody reads
    open_author_intake(root, "author-1", "key-1")
    assert author_intake_accepting(root, "author-1", "key-1") and not author_intake_accepting(root, "author-1", "other")
    assert accept_author_mail(root, "author-1", "key-1", append) is True and appended == [1]
    assert accept_author_mail(root, "author-1", "other-key", append) is False
    close_author_intake(root, "author-1")
    assert accept_author_mail(root, "author-1", "key-1", append) is False and appended == [1]
    # The runner opens the fact before the body and closes it in its own finally once the body returned
    # (``presence_runner.run_presence_turn``), after the wait scope closed: never inside the agent's release.
    close_author_intake(root, "author-1")  # closing twice is quiet
    assert not author_intake_accepting(root, "author-1", "key-1")


_REAL_THREAD = __import__("threading").Thread  # the pool fixture stubs the lifecycle's threads; the inline carrier needs one


@pytest.mark.parametrize("server", [False, True], ids=["host_alone", "server_whose_boot_spawn_failed"])
def test_a_host_serving_alone_runs_the_turn_on_the_disclosed_degraded_executor(pool, tmp_path, monkeypatch, server):
    """A Host serving without a supervisor queue of its own (no ``queue.init`` in this process) has no pool
    generation to wait for: that physical fact (2A) runs the body on this process's thread as
    ``inline_degraded``, disclosed in the row and the vitals, and its own frames resolve the Host's waiter.
    A SERVER before its pool arms is the typed refusal below, never this executor; a server whose boot spawn
    failed is that same physical fact (no supervisor loop will retry it), disclosed with its own reason."""
    import os

    from supervisor import inline_carrier, message_bus, queue

    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", None)  # no generation is armed here
    monkeypatch.setattr(queue, "INITIALIZED", server)  # a Host serving alone owns no live queue
    monkeypatch.setattr(queue, "OWNER_PID", os.getpid())
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", ["carrier_spawn_refused:OSError" if server else ""])
    assert queue.owns_live_queue(pool.root) is server
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: pytest.fail("a Presence body needs no native agent"))
    monkeypatch.setattr(inline_carrier, "threading", SimpleNamespace(Thread=_REAL_THREAD))
    monkeypatch.setattr(inline_carrier, "_INLINE", type(inline_carrier._INLINE)(maxlen=50))
    monkeypatch.setattr(inline_carrier, "_DEGRADED_TOTAL", [0])
    monkeypatch.setattr(message_bus, "_BRIDGE", SimpleNamespace(send_chat_action=lambda *a, **k: None,
                                                                 broadcast=lambda m: None, push_log=lambda *a, **k: None))
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    (pool.root / "logs").mkdir(parents=True, exist_ok=True)
    admission, event = _admission(), _event()
    from ouroboros import presence_admission
    monkeypatch.setattr(presence_admission, "admit_presence_turn", lambda **kw: admission)
    ran = []

    def runner(**kwargs):
        ran.append(kwargs)
        return PresenceTurnResult(outcome="message", text="Hi", task_id=kwargs["admitted"].conversation_key and "x")

    presence_carrier.set_presence_runner(pool.root, runner)
    try:
        registry = PresenceTurnExecutions()
        turn_id = presence_runner.presence_turn_task_id(admission.binding_id, event.source_event_id)
        execution, started = registry.start_or_dispatch(
            turn_id, identity="id-u", reserve=lambda: True, release=lambda: None,
            dispatch=lambda execution: dispatch_presence_turn(
                execution, drive_root=pool.root, skill_name="telegram-bot", admission=admission, event=event,
                event_payload=_payload(event), staged_files=[], gate_wait=True))
        assert started
        assert execution.result.result(timeout=10).text == "Hi"  # settled by the executor's own frame
        # The frame resolves the waiter before releasing its lane row; join the actual carrier before
        # asserting teardown, rather than racing the rest of handle_presence_turn_settled.
        for slot in list(workers.WORKERS.values()):
            if slot.proc.pid is None:
                slot.proc.join(5)
                assert not slot.proc.is_alive()
        assert ran and ran[0]["carrier_owner"]["executor"] == "inline_degraded"
        assert turn_id not in workers.RUNNING and registry.find(turn_id) is None
        vitals = inline_carrier.inline_vitals()
        assert vitals["degraded_total"] == 1 and vitals["inline_executions"][-1]["reason"] == (
            "carrier_spawn_refused:OSError" if server else "pool_not_started")
        assert vitals["inline_executions"][-1]["task_id"] == turn_id
    finally:
        presence_carrier.set_presence_runner(pool.root, None)
        for slot in list(workers.WORKERS.values()):
            if getattr(slot.proc, "join", None) and slot.proc.pid is None:
                slot.proc.join(5)


@pytest.mark.parametrize("saved", [False, True], ids=["empty_saved_queue", "saved_queue_not_restored"])
def test_a_server_before_its_pool_arms_refuses_the_turn_typed_and_touches_no_saved_queue(pool, monkeypatch, saved):
    """S9: in the server process (it owns the live queue of this root) a Presence event that arrives before the
    pool arms (the boot's window between ``queue.init`` and ``spawn_workers``, or after a whole-table stop) is not
    started: ``PresenceTurnNotStarted``, the typed 503 ``presence_turn_not_started`` retry the connector already
    handles (CREATING_SKILLS), never the degraded executor on a server thread. Nothing is admitted, so no one-row
    document is published over a saved queue this generation has not restored yet (#1952): the saved
    snapshot's bytes stay as they were."""
    import os

    from supervisor import queue
    from supervisor.inline_carrier import inline_vitals

    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", None)
    monkeypatch.setattr(queue, "INITIALIZED", True)
    monkeypatch.setattr(queue, "OWNER_PID", os.getpid())
    assert queue.owns_live_queue(pool.root)
    path = queue.QUEUE_SNAPSHOT_PATH
    if saved:  # the previous generation's queue, not restored by this one yet
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ts": "2026-11-02T17:54:00+00:00", "pending": [{"id": "saved", "task": {"id": "saved"}}],
                                    "running": []}), encoding="utf-8")
    before = path.read_bytes() if saved else None
    degraded_before = inline_vitals()["degraded_total"]
    admission, event = _admission(), _event()
    from ouroboros import presence_admission
    monkeypatch.setattr(presence_admission, "admit_presence_turn", lambda **kw: admission)
    registry = PresenceTurnExecutions()
    turn_id = presence_runner.presence_turn_task_id(admission.binding_id, event.source_event_id)
    with pytest.raises(presence_runner.PresenceTurnNotStarted):
        dispatch_presence_turn(SimpleNamespace(turn_id=turn_id), drive_root=pool.root, skill_name="telegram-bot",
                               admission=admission, event=event, event_payload=_payload(event), staged_files=[],
                               gate_wait=True)
    assert turn_id not in workers.RUNNING and registry.find(turn_id) is None
    assert inline_vitals()["degraded_total"] == degraded_before, "nothing ran on a server thread"
    if saved:
        assert path.read_bytes() == before, "the saved queue is untouched"
    else:
        assert not path.exists()


def test_an_owner_turns_intake_follows_its_own_loop_and_never_another_writers(tmp_path):
    """An owner turn's fact (``owner:<chat>``, its carrier's) moves with its loop's admission whatever the loop's
    metadata says; a fact another attempt or another process wrote stays as its writer left it."""
    import json

    from ouroboros.conversation_lease import (
        _intake_path,
        author_intake_accepting,
        follow_loop_admission,
        open_author_intake,
    )

    (tmp_path / "memory" / "owner_mailbox").mkdir(parents=True)
    loop = SimpleNamespace(task_id="main-turn", drive_root=tmp_path, task_metadata={}, task_attempt=None)
    assert open_author_intake(tmp_path, "main-turn", "owner:1", attempt=1)
    follow_loop_admission(loop, False)  # the final drain is about to read
    assert author_intake_accepting(tmp_path, "main-turn", "owner:1", attempt=1) is False
    follow_loop_admission(loop, True)  # a follow-up reopened its loop
    assert author_intake_accepting(tmp_path, "main-turn", "owner:1", attempt=1) is True
    follow_loop_admission(SimpleNamespace(**{**vars(loop), "task_attempt": 2}), False)  # not this attempt's loop
    assert author_intake_accepting(tmp_path, "main-turn", "owner:1", attempt=1) is True
    path = _intake_path(tmp_path, "main-turn")
    path.write_text(json.dumps({**json.loads(path.read_text()), "pid": 1}))
    follow_loop_admission(loop, False)  # another process's fact is its own
    assert json.loads(path.read_text())["accepting"] is True


def test_a_dead_carriers_intake_accepts_nothing_and_the_rows_end_closes_it(tmp_path):
    """A carrier that died before its release leaves ``accepting: true`` behind: the fact is its writer's, so a
    dead writer accepts nothing (an arrival would be swallowed), and the row's end in the server closes it."""
    import json
    import subprocess
    import sys

    from ouroboros.conversation_lease import author_intake_accepting, open_author_intake

    root = tmp_path
    (root / "memory" / "owner_mailbox").mkdir(parents=True)
    open_author_intake(root, "author-dead", "key")
    assert author_intake_accepting(root, "author-dead", "key")
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()  # reaped: its pid names no process now
    path = root / "memory" / "owner_mailbox" / "author-dead.intake.json"
    row = json.loads(path.read_text())
    row["pid"] = gone.pid
    path.write_text(json.dumps(row))
    assert not author_intake_accepting(root, "author-dead", "key"), "a dead writer reads no mail"
    open_author_intake(root, "author-live", "key")
    assert author_intake_accepting(root, "author-live", "key")
    handle_presence_turn_settled({"turn_id": "author-live", "physical_id": "author-live", "error": None, "result": None},
                                 SimpleNamespace(RUNNING={}, WORKERS={}, DRIVE_ROOT=root))
    assert not author_intake_accepting(root, "author-live", "key")
    assert not (root / "memory" / "owner_mailbox" / "author-live.intake.json").exists()


def test_the_intake_writers_liveness_is_the_platform_layers_probe(tmp_path, monkeypatch):
    """Review C3: the writer's liveness is ``platform_layer.pid_is_alive`` (OpenProcess on Windows), never a raw
    ``os.kill(pid, 0)``, which on Windows delivers CTRL_C_EVENT to the pid's console group and proves nothing."""
    import os

    from ouroboros import platform_layer
    from ouroboros.conversation_lease import author_intake_accepting, open_author_intake

    (tmp_path / "memory" / "owner_mailbox").mkdir(parents=True)
    open_author_intake(tmp_path, "author", "key")  # written by this (live) process
    probed = []
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: probed.append(pid) or False)
    assert not author_intake_accepting(tmp_path, "author", "key")
    assert probed == [os.getpid()]
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: True)
    assert author_intake_accepting(tmp_path, "author", "key")


def test_a_presence_turn_whose_carrier_ended_unreported_settles_at_its_published_terminal(tmp_path):
    """Review C2: a Stop's custody kill (or a crash, or a pool stop) ends a Presence carrier before its
    ``presence_turn_settled`` frame. The server's published terminal (the ``task_done`` dispatch, once the row left
    RUNNING) settles the Host waiter with a typed retry: its in-flight reservation returns, and a retry of the same
    event starts a fresh execution instead of joining the dead one forever. A turn its carrier already settled
    keeps its own answer: the terminal finds no live waiter to overwrite."""
    from ouroboros.presence_executions import PresenceTurnInterrupted
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor import events

    (tmp_path / "memory" / "owner_mailbox").mkdir(parents=True)
    metadata = {"source": "presence", "presence": {"binding_id": "b"}}
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, WORKERS={}, append_jsonl=append_jsonl,
                          persist_queue_snapshot=lambda reason="": None)
    registry, released = PresenceTurnExecutions(), []

    def start(turn_id):
        return registry.start_or_dispatch(turn_id, identity="i", reserve=lambda: True,
                                          release=lambda: released.append(turn_id), dispatch=lambda execution: None)

    killed, started = start("pres-killed")
    assert started and registry.live() == ["pres-killed"]
    write_task_result(tmp_path, "pres-killed", "cancelled", result="Running task cancelled and worker terminated.",
                      metadata=metadata)  # the custody's terminal; the row already left RUNNING
    events._handle_task_done({"task_id": "pres-killed", "status": "cancelled", "task_type": "presence",
                              "chat_id": 5}, ctx)
    with pytest.raises(PresenceTurnInterrupted):
        killed.result.result(timeout=1)
    assert released == ["pres-killed"] and registry.live() == []
    retry, started_again = start("pres-killed")
    assert started_again and retry is not killed  # the resend is admitted afresh; its durable row answers it

    answered, _ = start("pres-answered")
    handle_presence_turn_settled({"turn_id": "pres-answered", "physical_id": "pres-answered", "error": None,
                                  "result": {"outcome": "message", "text": "hello", "task_id": "pres-answered"}}, ctx)
    write_task_result(tmp_path, "pres-answered", "completed", result="hello", metadata=metadata)
    events._handle_task_done({"task_id": "pres-answered", "status": "completed", "task_type": "presence",
                              "chat_id": 5}, ctx)
    assert answered.result.result(timeout=1).text == "hello" and released.count("pres-answered") == 1


def test_the_carrier_takes_the_hosts_admitted_signal_facts_by_value():
    """Review K2: the Host admits ``WAKE_FACTS`` plus the authenticated connector's declared signals (its live
    publication, in the server). The seed carries the facts the Host admitted, and the carrier's re-admission
    honours them by value, never its own process's registry; a fact nobody vouched for is still refused."""
    from dataclasses import replace

    from ouroboros.presence_events import presence_turn_event

    event = replace(_event(), facts=("mention", "signal.words"))
    payload = {"source_event_id": event.source_event_id, "provider": event.provider, "account_id": event.account_id,
               "conversation_id": event.conversation_id, "thread_id": event.thread_id,
               "conversation_key": event.conversation_key, "actor": dict(event.actor),
               "conversation": dict(event.conversation), "message": dict(event.message), "text": event.text,
               "facts": list(event.facts)}
    seed = presence_seed_task(admission=_admission(), event=event, skill_name="telegram-bot", event_payload=payload,
                              staged_files=[], gate_wait=False)
    vouched = seed[PRESENCE_RUN_KEY]["vouched_facts"]
    assert vouched == ["mention", "signal.words"]
    readmitted = presence_turn_event(seed[PRESENCE_RUN_KEY]["event"], _admission(), vouched_facts=vouched)
    assert readmitted.facts == ("mention", "signal.words")
    with pytest.raises(PresenceTurnError) as refused:
        presence_turn_event(seed[PRESENCE_RUN_KEY]["event"], _admission())
    assert refused.value.code == "presence_payload_invalid"


def test_a_version_2_event_finding_the_gate_busy_is_refused_before_any_lane_admission(lane, monkeypatch):
    """Review M11: version 2 never waits in the gate. A busy gate is refused before the lane admits anything (no
    row, no barrier, no spare claimed, no carrier spawned) with base's answer: the busy resource named, retry
    after 1 s for the conversation and 5 s for the active-turn cap; a free gate admits the turn as usual."""
    from ouroboros.conversation_lease import PresenceTurnGate

    monkeypatch.setenv("OUROBOROS_PRESENCE_MAX_ACTIVE", "1")
    admission, event = _admission(), _event()
    other_process = PresenceTurnGate(1, state_root=lane.root / "state")  # another process's view of the same files

    def dispatch_v2():
        registry = PresenceTurnExecutions()
        turn_id = presence_runner.presence_turn_task_id(admission.binding_id, event.source_event_id)
        return registry.start_or_dispatch(
            turn_id, identity="id-v2", reserve=lambda: True, release=lambda: None,
            dispatch=lambda execution: dispatch_presence_turn(
                execution, drive_root=lane.root, skill_name="telegram-bot", admission=admission, event=event,
                event_payload=_payload(event), staged_files=[], gate_wait=False))

    for held_key, busy, retry_after in ((event.conversation_key, "conversation", 1), ("chat:a:other:", "capacity", 5)):
        holder = other_process.acquire(held_key)
        try:
            with pytest.raises(PresenceTurnError) as refused:
                dispatch_v2()
        finally:
            holder.release()
        assert (refused.value.code, refused.value.field, refused.value.retry_after) == (
            "presence_capacity_full", busy, retry_after)
        assert not _lane_rows() and lane.spare.busy_task_id is None and lane.spare.in_q.empty()
    execution, started = dispatch_v2()
    assert started and len(_lane_rows()) == 1


def _lane_rows():
    return [meta for meta in workers.RUNNING.values() if meta.get("lane") == LANE_CONVERSATION]


def test_a_carried_presence_author_gets_no_generic_owner_wait_and_parks_its_review_through_the_carrier(
        tmp_path, monkeypatch):
    """Review M9: a Presence agent never holds the generic owner-wait capability (no sleep, detach, owner Pause or
    budget pause: those rails gate on it). The carrier's pooled wait reaches only the qualified review/private
    question park, through the turn's lease."""
    from ouroboros import agent as agent_module, presence_admission

    seen = {}
    monkeypatch.setattr(presence_admission, "admit_presence_turn", lambda **_kw: _admission())
    monkeypatch.setattr(agent_module, "make_agent", lambda **_kw: SimpleNamespace(owner_wait_callback=None))

    def module_runner(*, agent_factory, admitted, **_kw):  # the carrier's own runner path, with its factory
        agent = agent_factory(repo_dir=str(tmp_path), drive_root=str(tmp_path), event_queue=None)
        seen.update(generic=agent.owner_wait_callback, review=getattr(admitted, "review_wait", None))
        return PresenceTurnResult(outcome="silent", text="", task_id="t")

    monkeypatch.setattr(presence_carrier, "run_presence_turn", module_runner)
    pooled_wait = object()
    task = presence_seed_task(admission=_admission(), event=_event(), skill_name="telegram-bot",
                              event_payload=_payload(_event()), staged_files=[], gate_wait=False)
    frames = stdqueue.Queue()
    presence_carrier.carry_presence_turn(task, frames, tmp_path, tmp_path, owner_wait_callback=pooled_wait)
    assert seen == {"generic": None, "review": pooled_wait}
    settled = [frame for frame in list(frames.queue) if frame.get("type") == "presence_turn_settled"]
    assert len(settled) == 1 and settled[0]["error"] is None


def test_a_durable_intake_follows_its_loops_admission_under_the_mail_lock(tmp_path, monkeypatch):
    """Review M10: the loop's admission transitions move a Presence author's durable intake (closed before the final
    drain, reopened for a follow-up); only the writer's own process moves its fact, a non-Presence turn has none,
    and open/close honour the mail lock (an unordered open is refused, a close never leaves the fact standing)."""
    import contextlib
    import json

    from ouroboros import task_custody
    from ouroboros.conversation_lease import (
        _intake_path, accept_author_mail, close_author_intake, follow_loop_admission, open_author_intake,
    )

    (tmp_path / "memory" / "owner_mailbox").mkdir(parents=True)
    appended = []

    def append(admit=None):
        if admit is not None and not admit():
            return False
        appended.append(1)
        return True

    author = SimpleNamespace(task_id="author-m10", drive_root=tmp_path, task_metadata={"presence": {"binding_id": "b"}})
    assert open_author_intake(tmp_path, "author-m10", "key") is True
    follow_loop_admission(author, False)  # the final drain is about to read
    assert accept_author_mail(tmp_path, "author-m10", "key", append) is False and appended == []
    follow_loop_admission(author, True)  # an owner follow-up reopened the loop
    assert accept_author_mail(tmp_path, "author-m10", "key", append) is True and appended == [1]
    follow_loop_admission(SimpleNamespace(task_id="author-m10", drive_root=tmp_path, task_metadata={}), False)
    assert accept_author_mail(tmp_path, "author-m10", "key", append) is True  # a non-Presence turn moves nothing
    path = _intake_path(tmp_path, "author-m10")
    path.write_text(json.dumps({**json.loads(path.read_text()), "pid": 1}))
    follow_loop_admission(author, False)
    assert json.loads(path.read_text())["accepting"] is True  # another process's fact is its own
    close_author_intake(tmp_path, "author-m10")

    monkeypatch.setattr(task_custody, "task_mail_lock", lambda *_a, **_k: contextlib.nullcontext(False))
    assert open_author_intake(tmp_path, "author-unordered", "key") is False
    assert not _intake_path(tmp_path, "author-unordered").exists()
    _intake_path(tmp_path, "author-held").write_text('{"accepting": true, "conversation_key": "key", "pid": 1}')
    assert close_author_intake(tmp_path, "author-held") is False and not _intake_path(tmp_path, "author-held").exists()

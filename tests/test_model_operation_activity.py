"""Model-operation observations reach existing Main/Project activity without new work."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ouroboros import model_wait
from ouroboros.gateway import state as gateway_state
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.physical_attempt_context import attempt_fact
from ouroboros.task_results import write_task_result
from supervisor import events, queue as task_queue
from supervisor.active_activity import get_direct_activity_registry
from tests.test_llm_claudexor import MODEL, REF, ledger, result, retained, setup as setup


def _sequence(gateway, *, outage=False):
    terminal = gateway.detail(0)
    queued = {**terminal, "state": "queued", "startedAt": None,
              "dispatch": {"state": "not_started", "route": {"private": "account"}},
              "response": {"state": "absent"},
              "admission": {"class": "model", "phase": "queued",
                            "blockers": [{"kind": "global_limit", "limit": 24, "secret": "private"}]}}
    preparing = {**queued, "state": "running", "startedAt": "2026-10-11T00:00:00Z"}
    sending = {**preparing, "dispatch": {"state": "started"}}
    observations = [queued, deepcopy(queued), preparing, sending]
    if outage:
        observations.extend([ClaudexorUnavailable("daemon_unreachable", "control connection lost"), sending])
    observations.append(terminal)

    def detail(index):
        assert index == 0
        value = observations.pop(0)
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)

    gateway.detail = detail


@pytest.mark.parametrize("method", ["chat", "chat_async", "vision_query"])
def test_observed_phases_deduplicate_and_rejoin_one_generation(setup, method):
    root, gateway, client = setup
    _sequence(gateway, outage=True)
    observations = []
    kwargs = {"model": MODEL, "model_operation_observer": lambda value: observations.append(value)}
    if method == "vision_query":
        client.vision_query("describe", [{"url": "data:image/png;base64,AAAA"}], **kwargs)
    else:
        call = getattr(client, method)([], **kwargs)
        if method == "chat_async":
            asyncio.run(call)
    activity = [row["model_activity"] for row in observations]
    assert [row["phase"] for row in activity] == [
        "preparing", "queued", "dispatching", "awaiting_response", "rejoining",
        "awaiting_response", "response_ready", "response_received"]
    assert activity[1]["blockers"] == [{"kind": "global_limit"}]
    assert all(set(row) <= {"phase", "operation_id", "invocation_id", "blockers"} for row in activity)
    assert {row["operation_id"] for row in activity[1:]} == {"op-0"}
    assert len({row["invocation_id"] for row in activity}) == 1
    assert len(gateway.creates) == len(gateway.accepted_operations) == len(gateway.uploads) == 1
    assert gateway.acks == [("op-0", REF["sha256"])]
    assert retained(root) == result() and retained(root, "request") == gateway.uploads[0][0]
    assert len(ledger(root)) == 1 and ledger(root)[0]["state"] == "settled"


@pytest.mark.parametrize("detail", [
    {"state": "queued"},  # no startedAt evidence
    {"state": "running", "dispatch": {"state": "unknown"}},
    {"state": "running", "dispatch": {}},
])
def test_missing_phase_facts_stay_unknown_and_do_not_send(detail):
    from ouroboros.llm_claudexor import _ModelInvocation

    rows = []
    invocation = _ModelInvocation({}, {}, {"model_operation_observer": rows.append})
    invocation.operation_id, invocation.invocation_id, invocation.detail = "existing", "attempt", detail
    invocation.observe_operation()
    assert rows[0]["model_activity"]["phase"] == "unknown"
    assert invocation.gateway is None and not invocation.create_attempted


def test_definite_create_refusal_does_not_claim_a_rejoin(setup, monkeypatch):
    from ouroboros.llm_claudexor import ClaudexorModelNotDispatched

    root, gateway, client = setup
    observed = []

    def refuse(*_args, **_kwargs):
        raise ClaudexorUnavailable("model_request_invalid", "request rejected", status_code=400)

    monkeypatch.setattr(gateway, "create_model_operation", refuse)
    with pytest.raises(ClaudexorModelNotDispatched):
        client.chat([], MODEL, model_operation_observer=observed.append)
    assert [row["model_activity"]["phase"] for row in observed] == ["preparing"]
    assert ledger(root)[0]["state"] == "released" and gateway.reads == []


def test_old_engine_queue_does_not_invent_a_wait_reason():
    from ouroboros.llm_claudexor import _ModelInvocation

    observed = []
    invocation = _ModelInvocation({}, {}, {"model_operation_observer": observed.append})
    invocation.operation_id, invocation.invocation_id = "existing", "attempt"
    invocation.detail = {"state": "queued", "startedAt": None, "dispatch": {"state": "not_started"}}
    invocation.observe_operation()
    assert observed[0]["model_activity"] == {"phase": "queued", "operation_id": "existing", "invocation_id": "attempt"}


@pytest.mark.parametrize("kind,chat_id", [("direct_chat", 1), ("direct_chat", 7), ("managed_task", 7)])
def test_main_forwarder_reaches_current_census_without_quota_clock(setup, monkeypatch, kind, chat_id):
    root, gateway, client = setup
    _sequence(gateway)
    task = {"id": "task-one", "chat_id": chat_id, "root_task_id": "task-one"}
    if kind == "managed_task":
        task["_attempt"] = 3
    write_task_result(root, task["id"], "running", root_task_id=task["id"], chat_id=chat_id,
                      task_attempt=int(task.get("_attempt") or 1), _is_direct_chat=kind == "direct_chat")
    meta = {"task": task, "attempt": task.get("_attempt"), "started_at": 1.0}
    running = {task["id"]: meta} if kind == "managed_task" else {}
    monkeypatch.setattr(task_queue, "RUNNING", running)
    monkeypatch.setattr(task_queue, "PENDING", [])
    monkeypatch.setattr(task_queue, "BUDGET_ROOT_FENCES", {})
    registry = get_direct_activity_registry()
    registry.clear()
    recorded, snapshots = [], []

    class EventQueue:
        def put(self, event):
            if event.get("type") == "main_llm_call_state":
                recorded.append(deepcopy(event))
                events._handle_main_llm_call_state(event, SimpleNamespace(RUNNING=running))
                if event["phase"] == "observed":
                    snapshots.append(gateway_state._chat_activities_snapshot_safe(root))

        put_nowait = put

    event_queue = EventQueue()
    try:
        with model_wait.task_model_wait_scope(task=task, drive_root=root, event_queue=event_queue,
                                              worker_slot_held=kind == "managed_task") as owner:
            if kind == "direct_chat":
                actor = SimpleNamespace(tools=SimpleNamespace(_ctx=SimpleNamespace(model_wait_context=owner)))
                registry.register(task["id"], chat_id, actor=actor)
                assert registry.snapshot()[0]["phase"] == "preparing"
            answer, _ = call_llm_with_retry(client, [], MODEL, None, "medium", 1, root / "logs",
                task["id"], 1, event_queue, {}, task_attempt=task.get("_attempt"), initial_messages=[])
            assert answer == result()["message"]
            assert [rows[0]["model_activity"]["phase"] for rows in snapshots] == [
                "preparing", "queued", "dispatching", "awaiting_response", "response_ready", "response_received"]
            assert all(rows[0]["kind"] == kind and rows[0]["chat_id"] == chat_id for rows in snapshots)
            assert all(rows[0]["task_attempt"] == owner.attempt for rows in snapshots)
            assert len({rows[0]["model_activity"]["llm_call_id"] for rows in snapshots}) == 1
            assert all(not rows[0].get("model_waits") for rows in snapshots)
            assert owner.paused_seconds() == 0 and owner.snapshot()["model_waits"] == {}
            terminal = gateway_state._chat_activities_snapshot_safe(root)[0]
            assert "model_activity" not in terminal
            # A delayed observation of a settled call cannot reconstruct it.
            event_queue.put(recorded[-2])
            assert "model_activity" not in gateway_state._chat_activities_snapshot_safe(root)[0]
            assert len(gateway.creates) == 1
    finally:
        registry.clear()


@pytest.mark.parametrize("phase", ["queued", "awaiting_response", "response_received"])
def test_pooled_old_call_or_attempt_cannot_overwrite_successor(phase):
    meta = {"task": {"id": "t", "_attempt": 2}, "attempt": 2}
    ctx = SimpleNamespace(RUNNING={"t": meta})
    old = {"task_id": "t", "task_attempt": 2, "llm_call_id": "old", "execution_id": "exec",
           "round_id": "round", "call_attempt": 1, "phase": "started"}
    new = {**old, "llm_call_id": "new", "call_attempt": 2}
    events._handle_main_llm_call_state(old, ctx)
    events._handle_main_llm_call_state(new, ctx)
    before = deepcopy(meta)
    events._handle_main_llm_call_state({**old, "phase": "observed", "model_activity": {"phase": phase}}, ctx)
    events._handle_main_llm_call_state({**new, "task_attempt": 1, "phase": "observed",
                                      "model_activity": {"phase": phase}}, ctx)
    assert meta == before
    events._handle_main_llm_call_state({**new, "phase": "observed", "model_activity": {"phase": phase}}, ctx)
    assert meta["active_llm_call"]["model_activity"]["phase"] == phase
    events._handle_main_llm_call_state({**new, "phase": "finished"}, ctx)
    events._handle_main_llm_call_state({**new, "phase": "observed", "model_activity": {"phase": phase}}, ctx)
    assert "active_llm_call" not in meta


def test_census_drops_a_prior_attempts_model_observation(tmp_path, monkeypatch):
    task = {"id": "new-attempt", "chat_id": 1, "root_task_id": "new-attempt", "_attempt": 2}
    monkeypatch.setattr(task_queue, "RUNNING", {task["id"]: {"task": task, "attempt": 2,
        "active_llm_call": {"task_attempt": 1, "model_activity": {"task_attempt": 1, "phase": "queued"}}}})
    monkeypatch.setattr(task_queue, "PENDING", [])
    monkeypatch.setattr(task_queue, "BUDGET_ROOT_FENCES", {})
    write_task_result(tmp_path, task["id"], "running", root_task_id=task["id"], task_attempt=2, chat_id=1)
    rows = gateway_state._chat_activities_snapshot_safe(tmp_path, direct_turns=[])
    assert rows[0]["task_attempt"] == 2 and "model_activity" not in rows[0]


def test_a_sent_attempts_context_reaches_its_activity_record_bound_to_its_ledger_row(setup, monkeypatch):
    """The real Main round sends through the usage ledger with its Main context; the terminal frame carries that
    attempt's context (``physical_attempt_context.attempt_fact``): the ledger row's own ``physical_context`` (the
    SSOT, the owner's mode at send and the cause of the lower one included), its attempt id and how far it went.
    The supervisor keeps it on the task's activity record past the call; an older call's late frame never replaces
    a newer one, and a projection that fails changes neither the send nor its accounting."""
    from dataclasses import asdict

    from ouroboros import loop_llm_call
    from ouroboros.physical_attempt_context import PhysicalAttemptContext

    root, gateway, client = setup
    _sequence(gateway)
    task = {"id": "task-one", "chat_id": 7, "root_task_id": "task-one", "_attempt": 3}
    write_task_result(root, task["id"], "running", root_task_id=task["id"], chat_id=7, task_attempt=3)
    meta = {"task": task, "attempt": 3, "started_at": 1.0}
    running = {task["id"]: meta}
    monkeypatch.setattr(task_queue, "RUNNING", running)
    monkeypatch.setattr(task_queue, "PENDING", [])
    monkeypatch.setattr(task_queue, "BUDGET_ROOT_FENCES", {})
    get_direct_activity_registry().clear()
    recorded, rows = [], []

    class EventQueue:
        def put(self, event):
            if event.get("type") == "main_llm_call_state":
                recorded.append(deepcopy(event))
                events._handle_main_llm_call_state(event, SimpleNamespace(RUNNING=running))
                if event["phase"] == "dispatched":  # the public census row while ``llm.chat`` still waits
                    rows.append(gateway_state._chat_activities_snapshot_safe(root)[0])

        put_nowait = put

    context = PhysicalAttemptContext(
        profile="task_local_low", rendered_mode="low", measurement_basis="cold_estimate", route_fp="",
        round_id="exec-1:round:4", target_total_tokens=None, capacity_total_tokens=None, context_target_miss=False,
        automatic_pass_used=False, preferred_mode="max", mode_cause="memory")
    with model_wait.task_model_wait_scope(task=task, drive_root=root, event_queue=EventQueue(), worker_slot_held=True):
        call_llm_with_retry(client, [], MODEL, None, "medium", 1, root / "logs", task["id"], 4, EventQueue(),
                            {"execution_id": "exec-1"}, task_attempt=3, initial_messages=[], physical_context=context)
    sent = [row for row in ledger(root) if row.get("state") == "settled"]
    assert len(sent) == 1 and sent[0]["physical_context"] == asdict(context)  # the SSOT holds the new facts
    # A focus with no activity row (a wake of consciousness) is read where every task's is: its exact task id in
    # the money ledger (``query_data(op="sql", db="money")``), through the store's own reader.
    from ouroboros import usage_store
    with usage_store.read(root) as txn:
        own = [row["physical_context"] for row in txn.attempts("task_id=?", (task["id"],)) if row["state"] == "settled"]
    assert [(ctx["preferred_mode"], ctx["rendered_mode"], ctx["mode_cause"]) for ctx in own] == [("max", "low", "memory")]
    fact = meta["context_attempt"]
    assert {key: fact[key] for key in ("attempt_id", "ledger_state", "preferred_at_send", "applied", "profile", "cause",
                                       "round_id", "task_attempt", "call_attempt")} == {
        "attempt_id": sent[0]["attempt_id"], "ledger_state": "settled", "preferred_at_send": "max", "applied": "low",
        "profile": "task_local_low", "cause": "memory", "round_id": "exec-1:round:4", "task_attempt": 3,
        "call_attempt": 1}
    # Live: the ledger's dispatch is heard while ``llm.chat`` still waits (``physical_attempt_context.dispatched``):
    # one ``dispatched`` frame right after ``started``, before the first provider observation; then every frame of
    # the active call carries it; the provider's own progress stays ``model_activity`` beside it.
    carried = [(e["phase"], e["context_attempt"]["ledger_state"]) for e in recorded if "context_attempt" in e]
    assert [e["phase"] for e in recorded][:2] == ["started", "dispatched"]
    assert carried[0] == ("dispatched", "dispatched") and carried[-1] == ("finished", "settled")
    assert set(carried[1:-1]) == {("observed", "dispatched")}
    assert len(carried) - 2 == sum(e["phase"] == "observed" for e in recorded)
    live = rows[0]["context_attempt"]  # what the public row held while the call waited
    assert (live["attempt_id"], live["ledger_state"], live["applied"], live["cause"]) == (
        sent[0]["attempt_id"], "dispatched", "low", "memory")
    after = gateway_state._chat_activities_snapshot_safe(root)[0]
    assert after["context_attempt"] == fact and "model_activity" not in after  # kept past the call; typed shape
    from ouroboros.physical_attempt_context import ContextAttemptFact
    assert set(after["context_attempt"]) == set(ContextAttemptFact.__annotations__) - {"previous"}
    assert fact["observed_at"] == live["observed_at"], "settlement keeps the first observation's time"
    # Only the active call of this task attempt is kept: a late frame of an earlier call, of another execution of
    # the same task attempt, or of an earlier task attempt changes nothing; the next call's frames replace it.
    handle = lambda frame: events._handle_main_llm_call_state(frame, SimpleNamespace(RUNNING=running))  # noqa: E731
    base = {k: recorded[-1][k] for k in ("type", "task_id", "task_attempt", "llm_call_id", "execution_id", "round_id",
                                          "call_attempt")}
    nxt = {**base, "llm_call_id": "next", "execution_id": "exec-2", "round_id": "exec-2:round:1"}
    handle({**nxt, "phase": "started"})
    for late in ({**base, "phase": "finished"}, {**base, "phase": "failed", "call_attempt": 2},
                 {**base, "task_attempt": 2, "phase": "finished"}, {**nxt, "task_attempt": 2, "phase": "failed"}):
        handle({**late, "context_attempt": {**fact, "attempt_id": "late"}})
        assert meta["context_attempt"]["attempt_id"] == sent[0]["attempt_id"], late
    handle({**nxt, "phase": "observed", "model_activity": {"phase": "dispatching"},
            "context_attempt": {**fact, "attempt_id": "next-attempt"}})
    assert meta["context_attempt"]["attempt_id"] == "next-attempt" and meta["context_attempt"]["execution_id"] == "exec-2"
    # A capture that is no send, another round's capture, and a broken one project nothing.
    assert attempt_fact(SimpleNamespace(state="released", physical_context=context, attempt_id="x"))["ledger_state"] == "released"
    for capture in (SimpleNamespace(state="reserved", physical_context=context, attempt_id="x"),  # queued, prepared
                    SimpleNamespace(state="settled", physical_context=context, attempt_id="x"),
                    SimpleNamespace(state="settled", physical_context=object(), attempt_id="x")):
        assert attempt_fact(capture, "exec-1:round:9" if capture.state == "settled" else "") is None
        frames = []
        loop_llm_call._emit_main_llm_call_state(SimpleNamespace(put=frames.append), ("task-one", 3, "c", "exec-1",
                                                "exec-1:round:9", 1), "dispatched", capture=capture)
        assert frames == []  # no dispatched frame without this round's dispatched attempt
    # A projection that fails leaves the send, its lifecycle frame and its ledger row whole.
    class Broken:
        state, attempt_id = "settled", "broken"

        @property
        def physical_context(self):
            raise RuntimeError("an unreadable capture")

    assert attempt_fact(Broken()) is None
    frames = []
    loop_llm_call._emit_main_llm_call_state(SimpleNamespace(put=frames.append), ("task-one", 3, "c", "exec-1",
                                            "exec-1:round:5", 1), "finished", capture=Broken())
    assert [frame["phase"] for frame in frames] == ["finished"] and "context_attempt" not in frames[0]
    # The fact is read only after ``llm.chat`` returned (or raised): the settled row above was written before it.


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("prior_send", [False, True])
@pytest.mark.parametrize("internal_repair", [False, True], ids=["create-refusal", "in-call-repair"])
def test_a_disproven_dispatch_reaches_the_census_with_its_identity_and_one_genuine_previous_send(
        setup, monkeypatch, direct, prior_send, internal_repair):
    """The real ledger releases a definite transport refusal after its dispatch notification. Both activity
    owners carry that retraction; successive refusals never consume the prior send or grow a history chain."""
    from dataclasses import replace

    from ouroboros.physical_attempt_context import PhysicalAttemptContext

    root, gateway, client = setup
    task = {"id": "release-work", "chat_id": 1, "root_task_id": "release-work", "_attempt": 1}
    write_task_result(root, task["id"], "running", root_task_id=task["id"], chat_id=1, task_attempt=1,
                      _is_direct_chat=direct)
    running = {} if direct else {task["id"]: {"task": task, "attempt": 1, "started_at": 1.0}}
    for name, value in (("RUNNING", running), ("PENDING", []), ("BUDGET_ROOT_FENCES", {})):
        monkeypatch.setattr(task_queue, name, value)
    registry, frames, live = get_direct_activity_registry(), [], []
    registry.clear()

    class EventQueue:
        def put(self, event):
            if event.get("type") == "main_llm_call_state":
                frames.append(deepcopy(event))
                events._handle_main_llm_call_state(event, SimpleNamespace(RUNNING=running))
                if event["phase"] == "dispatched":
                    live.append(deepcopy(gateway_state._chat_activities_snapshot_safe(root)[0]["context_attempt"]))
        put_nowait = put

    event_queue = EventQueue()
    context = PhysicalAttemptContext(
        profile="owner_max", rendered_mode="max", measurement_basis="cold_estimate", route_fp="",
        round_id="release-exec:round:1", target_total_tokens=None, capacity_total_tokens=None,
        context_target_miss=False, automatic_pass_used=False, preferred_mode="max")

    def call(round_idx, physical):
        # The fixture's empty input remains identical when auth repair re-prepares this same logical call.
        with owner.register_reprepare("main", lambda values: values):
            return call_llm_with_retry(client, [], MODEL, None, "medium", 1, root / "logs", task["id"], round_idx,
                                       event_queue, {"execution_id": "release-exec"}, task_attempt=1,
                                       initial_messages=[], physical_context=physical)

    def refuse(*_args, **_kwargs):
        raise ClaudexorUnavailable("model_request_invalid", "rejected before acceptance", status_code=400)

    try:
        with model_wait.task_model_wait_scope(task=task, drive_root=root, event_queue=event_queue,
                                              worker_slot_held=not direct) as owner:
            if direct:
                registry.register(task["id"], 1, actor=SimpleNamespace(tools=SimpleNamespace(
                    _ctx=SimpleNamespace(model_wait_context=owner))))
            earlier = None
            if prior_send:
                assert call(1, context)[0] == result()["message"]
                earlier = deepcopy(gateway_state._chat_activities_snapshot_safe(root)[0]["context_attempt"])
                assert earlier["ledger_state"] == "settled" and earlier["applied"] == "max"
            if internal_repair:
                from tests.test_auth_changed_reissue import auth_refusal

                gateway.results = ([result()] if prior_send else []) + [auth_refusal() for _ in range(4)]
                gateway.dispatch = (["response_received"] if prior_send else []) + ["not_started"] * 4
            else:
                monkeypatch.setattr(gateway, "create_model_operation", refuse)
            for round_idx in (2, 3):
                assert call(round_idx, replace(context, profile="task_local_low", rendered_mode="low",
                            mode_cause="memory", round_id=f"release-exec:round:{round_idx}"))[0] is None
                fact = gateway_state._chat_activities_snapshot_safe(root)[0]["context_attempt"]
                assert (fact["ledger_state"], fact["applied"], fact["attempt_id"], fact["observed_at"]) == (
                    "released", "low", live[-1]["attempt_id"], live[-1]["observed_at"])
                assert fact.get("previous") == earlier
                assert not (fact.get("previous") or {}).get("previous")
                assert fact["llm_call_id"] != (earlier or {}).get("llm_call_id")
                # A late dispatch frame of the now-ended call cannot revive it in either activity owner.
                late = next(frame for frame in reversed(frames) if frame["phase"] == "dispatched"
                            and frame.get("context_attempt", {}).get("ledger_state") == "dispatched")
                event_queue.put(late)
                assert gateway_state._chat_activities_snapshot_safe(root)[0]["context_attempt"] == fact
        attempts = ledger(root)
        refused = 4 if internal_repair else 2
        assert len(attempts) == refused + int(prior_send)
        assert sum(row["state"] == "released" for row in attempts) == refused
        assert sum(row["state"] == "settled" for row in attempts) == int(prior_send)
    finally:
        registry.clear()


def test_a_call_that_dispatched_nothing_never_carries_the_preceding_attempt(setup, monkeypatch):
    """One physical attempt settles an empty reply; the next calls of the same round fail before any dispatch (a
    configured fallback whose named connection is unavailable). The preceding attempt's capture is still the context's
    last one, yet only its own call's frames carry it: the activity record keeps it under that call's identity, and a
    call that dispatched nothing carries no attempt fact (``loop_llm_call._own_capture``)."""
    from ouroboros import loop_llm_call
    from ouroboros.physical_attempt_context import PhysicalAttemptContext

    root, gateway, client = setup
    empty = result()
    empty["message"].update(content="", tool_calls=[])  # a settled empty reply: the round retries it
    gateway.results, gateway.dispatch = [empty], ["response_received"]
    _sequence(gateway)
    task = {"id": "task-one", "chat_id": 7, "root_task_id": "task-one", "_attempt": 3}
    write_task_result(root, task["id"], "running", root_task_id=task["id"], chat_id=7, task_attempt=3)
    meta = {"task": task, "attempt": 3, "started_at": 1.0}
    running = {task["id"]: meta}
    for name, value in (("RUNNING", running), ("PENDING", []), ("BUDGET_ROOT_FENCES", {})):
        monkeypatch.setattr(task_queue, name, value)
    get_direct_activity_registry().clear()
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *a, **kw: True)
    real, sends = loop_llm_call._send_main_candidate, []

    def send(*args, **kwargs):
        sends.append(1)
        if len(sends) == 1:
            return real(*args, **kwargs)
        raise ValueError("configured fallback: named connection unavailable")

    monkeypatch.setattr(loop_llm_call, "_send_main_candidate", send)
    recorded = []

    class EventQueue:
        def put(self, event):
            if event.get("type") == "main_llm_call_state":
                recorded.append(deepcopy(event))
                events._handle_main_llm_call_state(event, SimpleNamespace(RUNNING=running))

        put_nowait = put

    context = PhysicalAttemptContext(
        profile="owner_max", rendered_mode="max", measurement_basis="cold_estimate", route_fp="",
        round_id="exec-1:round:4", target_total_tokens=None, capacity_total_tokens=None, context_target_miss=False,
        automatic_pass_used=False, preferred_mode="max")
    with model_wait.task_model_wait_scope(task=task, drive_root=root, event_queue=EventQueue(), worker_slot_held=True):
        assert call_llm_with_retry(client, [], MODEL, None, "medium", 3, root / "logs", task["id"], 4, EventQueue(),
                                   {"execution_id": "exec-1"}, task_attempt=3, initial_messages=[],
                                   physical_context=context) == (None, 0.0)  # the round ends unanswered
    sent = [row for row in ledger(root) if row.get("state") in {"dispatched", "settled", "unresolved"}]
    calls = list(dict.fromkeys(event["llm_call_id"] for event in recorded))
    assert len(sent) == 1 and len(calls) >= 2 and len(sends) == len(calls)
    assert {event["context_attempt"]["attempt_id"] for event in recorded if "context_attempt" in event} == {
        sent[0]["attempt_id"]}
    assert not [event for event in recorded if event["llm_call_id"] in calls[1:] and "context_attempt" in event]
    kept = meta["context_attempt"]
    assert (kept["attempt_id"], kept["llm_call_id"], kept["call_attempt"]) == (sent[0]["attempt_id"], calls[0], 1)
    assert loop_llm_call._own_capture(SimpleNamespace(attempt_id="a"), "a") is None
    assert loop_llm_call._own_capture(SimpleNamespace(attempt_id="b"), "a").attempt_id == "b"


def test_a_direct_turns_row_carries_its_sent_attempts_context_while_it_waits_and_after(setup, monkeypatch):
    """The same fact on an owner turn's registry row (``DirectActivityEntry``): published live at the ledger's
    dispatch and kept past the call; a fact of another task attempt is never published on this row."""
    from ouroboros.physical_attempt_context import PhysicalAttemptContext

    root, gateway, client = setup
    _sequence(gateway)
    task = {"id": "task-one", "chat_id": 1, "root_task_id": "task-one"}
    write_task_result(root, task["id"], "running", root_task_id=task["id"], chat_id=1, task_attempt=1,
                      _is_direct_chat=True)
    for name, value in (("RUNNING", {}), ("PENDING", []), ("BUDGET_ROOT_FENCES", {})):
        monkeypatch.setattr(task_queue, name, value)
    registry, live = get_direct_activity_registry(), []
    registry.clear()

    class EventQueue:
        def put(self, event):
            if event.get("type") == "main_llm_call_state":
                events._handle_main_llm_call_state(event, SimpleNamespace(RUNNING={}))
                if event["phase"] == "dispatched":
                    live.append(gateway_state._chat_activities_snapshot_safe(root)[0])

        put_nowait = put

    context = PhysicalAttemptContext(
        profile="owner_max", rendered_mode="max", measurement_basis="cold_estimate", route_fp="",
        round_id="exec-1:round:1", target_total_tokens=None, capacity_total_tokens=None, context_target_miss=False,
        automatic_pass_used=False, preferred_mode="max")
    try:
        with model_wait.task_model_wait_scope(task=task, drive_root=root, event_queue=EventQueue(),
                                              worker_slot_held=False) as owner:
            registry.register(task["id"], 1, actor=SimpleNamespace(tools=SimpleNamespace(
                _ctx=SimpleNamespace(model_wait_context=owner))))
            call_llm_with_retry(client, [], MODEL, None, "medium", 1, root / "logs", task["id"], 1, EventQueue(),
                                {"execution_id": "exec-1"}, initial_messages=[], physical_context=context)
            after = gateway_state._chat_activities_snapshot_safe(root)[0]
            assert after["kind"] == "direct_chat" and after["context_attempt"]["ledger_state"] == "settled"
            assert live[0]["context_attempt"]["ledger_state"] == "dispatched"
            assert (live[0]["context_attempt"]["applied"], live[0]["context_attempt"]["cause"]) == ("max", "")
            entry = registry.get(task["id"])
            entry.context_attempt = {**entry.context_attempt, "task_attempt": owner.attempt + 1}
            assert "context_attempt" not in gateway_state._chat_activities_snapshot_safe(root)[0]
    finally:
        registry.clear()

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

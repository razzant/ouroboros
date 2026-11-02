"""A historical chat origin never substitutes for the current attempt's owner."""

from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue


@pytest.mark.parametrize("kind", ["direct", "pooled"])
def test_completion_uses_attempt_owner_without_erasing_chat_origin(tmp_path, monkeypatch, kind):
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import execution_owner_record
    from supervisor.events_task_done import _handle_task_done
    from tests.test_task_done_durable_vs_transport import _mk_ctx

    _install_queue(tmp_path, monkeypatch)
    pushed, sent = [], []
    ctx = _mk_ctx(tmp_path, sent, pushed)
    task = {"id": "detached", "type": "task", "_is_direct_chat": True, "_attempt": 2, "chat_id": 7}
    owner = execution_owner_record(tmp_path, task, kind)
    task["_execution_owner"] = owner
    write_task_result(tmp_path, "detached", "completed", _is_direct_chat=True,
                      execution_owner=owner, task_attempt=2, result="done", chat_id=7)
    ctx.RUNNING["detached"] = {"task": task, "attempt": 2}
    from supervisor import task_reaper
    prepared = []
    monkeypatch.setattr(task_reaper, "enqueue_terminal_file_recovery",
                        lambda _ctx, _event, _task: prepared.append(_task["id"]) or True)
    event = {"type": "task_done", "task_id": "detached", "task_type": "task",
             "status": "completed", "chat_id": 7, "_is_direct_chat": True}
    _handle_task_done(event, ctx)
    if kind == "pooled":
        assert prepared == ["detached"] and not pushed
        _handle_task_done({**event, "_files_prepared_attempt": 2}, ctx)
    else:
        assert not prepared
    event = next(row for row in pushed if row.get("type") == "task_done")
    assert event["_is_direct_chat"] is True and event["execution_owner"] == owner


def test_policy_uses_attempt_carrier_and_preserves_historical_source():
    from ouroboros.task_carrier import inline_execution
    from ouroboros.tool_access_roots import active_tool_profile

    task = {"_is_direct_chat": True, "_execution_owner": {"kind": "pooled"}}
    assert not inline_execution(task)
    ctx = SimpleNamespace(is_direct_chat=True, task_metadata={
        "_execution_owner": task["_execution_owner"], "tool_profile": "operator_control"})
    assert not inline_execution(ctx) and active_tool_profile(ctx) == "operator_control"
    task["_execution_owner"] = {"kind": "direct"}
    assert inline_execution(task) and task["_is_direct_chat"]


def test_consciousness_blocks_on_wake_across_queue_carriers_but_not_detached_owner(tmp_path, monkeypatch):
    from ouroboros.consciousness import BackgroundConsciousness
    from supervisor.active_activity import get_direct_activity_registry

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    mind = object.__new__(BackgroundConsciousness)
    mind._last_wake_task_id = ""
    mind._own_wake, mind._in_flight = {}, ""
    workers.PENDING.append({"id": "owner-work", "_is_direct_chat": True, "detached_at": 1.0})
    assert mind.live_wake() == ""
    wake = {"id": "wake", "_is_direct_chat": True, "detached_at": 1.0,
            "metadata": {"initiator": "consciousness", "usage_category": "consciousness", "wake_cause": "own_clock"}}
    workers.PENDING.append(wake)
    assert mind.live_wake() == "wake"
    workers.PENDING.remove(wake)
    workers.RUNNING["wake"] = {"task": wake}
    assert mind.live_wake() == "wake"
    workers.RUNNING.clear()
    registry = get_direct_activity_registry()
    registry.register("conversing", 7)
    try:
        assert mind.live_wake() == ""
    finally:
        registry.unregister("conversing")


def test_stop_detached_wake_records_existing_graceful_cancel_intent(tmp_path, monkeypatch):
    from ouroboros.consciousness import BackgroundConsciousness
    from ouroboros.task_results import write_task_result
    from ouroboros.cancel_intents import cancel_pending

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "wake", "type": "task", "_is_direct_chat": True, "admitted_dispatch": "none"}
    workers.PENDING.append(task)
    write_task_result(tmp_path, "wake", "scheduled", root_task_id="wake")
    BackgroundConsciousness._stop_live_wake("wake")
    assert cancel_pending(tmp_path, "wake")


def test_assignment_refuses_another_actor_until_inline_registration_ends(tmp_path, monkeypatch):
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor.worker_assignment import _choose_candidate

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    workers.PENDING.append({"id": "head", "type": "task", "depth": 0, "admitted_dispatch": "none"})
    registry = get_direct_activity_registry()
    registry.register("head", 7)
    try:
        assert _choose_candidate(q, 5.0, set(), set(), {}, set()) == (None, False)
    finally:
        registry.unregister("head")
    assert _choose_candidate(q, 5.0, set(), set(), {}, set()) == (0, False)

"""Saved supervision reaches the real restore and first-input seams before admission.

Adapted from the retained startup boundary tests; lifecycle creation is stubbed
for the entire fixture. No engine process or provider request belongs here.
"""
import json
from types import SimpleNamespace

import pytest

from tests.test_available_subagents_runtime import _session_row, _settings, _snapshot
from tests._stub_delegate_engines import stub_delegate_engines as stub_delegate_engines

pytestmark = pytest.mark.serial


class InputReached(BaseException):
    """Stop at the first consumer, before unrelated finalization can run."""


def _restored_registry(tmp_path, monkeypatch, carrier):
    from ouroboros import owner_wait
    from tests.test_context_fit_v664 import _plan
    from tests.test_working_checkpoint import _registry, _saved_working_state

    task_id = "nanny"
    if carrier == "working":
        handoff = _saved_working_state(tmp_path, monkeypatch, task_id)
        registry = _registry(tmp_path, monkeypatch, task_id, 2)
        registry._ctx.working_recovery = handoff
    elif carrier == "owner_wait":
        from ouroboros.task_results import write_task_result

        registry = _registry(tmp_path, monkeypatch, task_id, 1)
        ctx = registry._ctx
        write_task_result(tmp_path, task_id, "running")
        ctx._owner_wait_requested = "question-1"
        messages = _plan().messages_for("max") + [
            {"role": "assistant", "tool_calls": [
                {"id": "ask", "type": "function", "function": {"name": "escalate", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "ask", "content": "Question saved"}]
        wait = owner_wait.checkpoint_owner_wait(ctx, messages, {}, {}, 1, [], set())
        owner_wait.set_owner_wait(tmp_path, task_id, {**wait, "state": "waiting"})
        ctx.owner_wait_resume = {**wait, "restart_transaction_id": "confirmed"}
        ctx.owner_wait_callback = lambda *_: "owner_input"
    else:
        from tests._budget_pause_exact_helpers import _install_queue, _parked

        q, state, _workers = _install_queue(tmp_path, monkeypatch)
        monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
        task, _row = _parked(tmp_path, monkeypatch, task_id=task_id)
        assert q.resume_budget_paused_task(task_id)["ok"]
        registry = _registry(tmp_path, monkeypatch, task_id, 1)
        registry._ctx.budget_pause_resume = task["_budget_pause_resume"]
    return registry


def _task(ctx, carrier):
    key = {"working": "_working_recovery", "owner_wait": "_owner_wait_resume",
           "exact_pause": "_budget_pause_resume"}[carrier]
    value = getattr(ctx, key[1:])
    return {"id": ctx.task_id, "_attempt": ctx.task_attempt, "delegation_role": "subagent",
            "type": "task", "drive_root": str(ctx.drive_root), "budget_drive_root": str(ctx.drive_root),
            "configured_subagent": _snapshot(_settings(_session_row()), "session-builder"),
            "parent_cognitive_route": {"model": "same-model", "effort": "high", "use_local_model": False},
            "task_constraint": {}, "task_contract": {"objective": "Build"}, key: value}


def _known_custody(root, task, fate):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_recovery import authority_fingerprint_from_task
    from ouroboros.subagent_work_order import work_order_fingerprint

    fields = {"task_id": task["id"], "invocation_id": "invoke-one", "route": "codex",
              "selected_subagent_id": task["configured_subagent"]["selected_subagent_id"],
              "config_fingerprint": task["configured_subagent"]["config_fingerprint"],
              "authority_fingerprint": authority_fingerprint_from_task(task),
              "work_order_fingerprint": work_order_fingerprint(task)}
    if fate == "none":
        return
    assert custody.record_start_requested(root, run_id="", idempotency_key="invoke-one", max_seconds=60,
                                         request={"prompt": "canonical"}, **fields)
    if fate != "pending":
        assert custody.emit(root, custody.STARTED, {**fields, "run_id": "run-one"})
    if fate == "terminal":
        assert custody.emit(root, custody.SETTLED, {**fields, "run_id": "run-one", "state": "succeeded",
                                                  "model": "stub-model"})


@pytest.mark.parametrize("carrier", ["working", "exact_pause", "owner_wait"])
@pytest.mark.parametrize("fate", ["none", "live", "terminal", "pending"])
def test_saved_dispatch_and_first_input_restore_before_leaf_admission(
        tmp_path, monkeypatch, stub_delegate_engines, carrier, fate):
    from ouroboros import loop, subagent_bootstrap, subagent_runtime
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from tests.test_loop_transport_wait import _loop_kwargs

    registry = _restored_registry(tmp_path, monkeypatch, carrier)
    ctx = registry._ctx
    task = _task(ctx, carrier)
    _known_custody(tmp_path, task, fate)
    monkeypatch.setattr(subagent_runtime, "exact_start", lambda *_a, **_k: pytest.fail("early POST"))
    dispatch = resolve_dispatch_axes(task)
    assert dispatch.executor == "harness" and dispatch.route == task["configured_subagent"]["route"]["target_id"]
    assert dispatch.executor_resolution.reason == "saved_supervision_unobserved"
    assert dispatch.availability["status"] == "unobserved" and not dispatch.delta.reduced
    ctx._configured_startup_wake = subagent_bootstrap.bootstrap_before_context(ctx, task, dispatch)
    receipt = json.loads(ctx._configured_startup_wake)
    assert ctx._configured_actor_bootstrap["route_available"] is None
    assert receipt["continuation"]["physical_start"] == "not_repeated"
    facts = receipt["continuation"]
    assert len(facts["runs"]) == (fate in {"live", "terminal"})
    assert len(facts["pending_invocations"]) == (fate == "pending")
    if fate == "pending":
        assert facts["pending_invocations"] == [{"pending_invocation_id": "invoke-one", "retry_of": "invoke-one"}]
    assert not stub_delegate_engines
    callbacks = []
    if carrier == "owner_wait":
        def owner_input(actual, _checkpoint):
            assert any(m.get("content") == "Question saved" for m in actual.messages)
            assert not stub_delegate_engines
            callbacks.append(True)
            return "owner_input"
        ctx.owner_wait_callback = owner_input

    def dispatch_model(call, *_args, **_kwargs):
        contents = [str(row.get("content") or "") for row in call.messages]
        assert any("CONFIGURED SESSION STARTUP / WAKE RECEIPT" in text for text in contents)
        assert any("unobserved" in text for text in contents)
        if carrier == "working":
            assert "Saved object 42" in contents
            assert any("UNKNOWN" in text for text in contents)
        if carrier == "owner_wait":
            assert callbacks == [True] and "Question saved" in contents
        assert not stub_delegate_engines
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch_model)
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs["task_id"] = task["id"]
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)


@pytest.mark.parametrize("locator", [None, {}, "unreadable", {"source_ref": {"path": "missing"}}])
def test_present_invalid_working_carrier_is_a_gap_never_a_fresh_start(tmp_path, monkeypatch, locator):
    from ouroboros import loop, subagent_bootstrap, subagent_runtime
    from ouroboros.agent import _load_start_continuation
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    registry = _registry(tmp_path, monkeypatch, "nanny", 2)
    ctx = registry._ctx
    ctx.working_recovery = locator
    task = _task(ctx, "working")
    assert _load_start_continuation(ctx, task) == {}
    monkeypatch.setattr(subagent_runtime, "exact_start", lambda *_a, **_k: pytest.fail("fresh POST from bad carrier"))
    ctx._configured_startup_wake = subagent_bootstrap.bootstrap_before_context(ctx, task, resolve_dispatch_axes(task))

    def reached(call, *_args, **_kwargs):
        text = str(call.messages)
        assert "could not be restored" in text and "configured_session_continuation" in text
        raise InputReached

    monkeypatch.setattr(loop, "_dispatch_round_model", reached)
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs["task_id"] = "nanny"
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)


def test_owner_wait_repark_does_not_deliver_or_ack_startup_receipt(tmp_path, monkeypatch):
    from ouroboros import delegate_supervision as ds, loop
    from tests.test_loop_transport_wait import _loop_kwargs

    registry = _restored_registry(tmp_path, monkeypatch, "owner_wait")
    ctx = registry._ctx
    wake = json.dumps({"wake": {"supervision_wake_id": "unshown"}})
    ctx._configured_startup_wake = wake
    monkeypatch.setattr(ds, "acknowledge_pending_wake", lambda *_a: pytest.fail("unshown ACK"))
    def park(actual, _checkpoint):
        assert any(row.get("content") == "Question saved" for row in actual.messages)
        raise InputReached
    ctx.owner_wait_callback = park
    with pytest.raises(InputReached):
        loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    assert ctx._configured_startup_wake == wake
    assert not getattr(ctx, "_pending_content_acks", [])


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("ready", [False, True])
def test_fresh_start_and_retry_keep_bootstrap_once_rules(tmp_path, monkeypatch, retry, ready):
    from ouroboros import claudexor_daemon, subagents, subagent_bootstrap, subagent_runtime
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.delegate_shared import delegate_result
    from tests.test_working_checkpoint import _registry

    ctx = _registry(tmp_path, monkeypatch, "nanny", 2 if retry else 1)._ctx
    ctx.working_recovery = None
    task = _task(ctx, "working")
    task.pop("_working_recovery")
    if retry:
        _known_custody(tmp_path, task, "terminal")
    calls = []
    if ready:
        monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: SimpleNamespace(close=lambda: None))
        monkeypatch.setattr(subagents, "route_health", lambda *_a, **_k: ("", ""))
    monkeypatch.setattr(subagent_runtime, "exact_start", lambda *_a, **_k:
                        calls.append(True) or delegate_result({"status": "started", "run_id": "fresh-run"}))
    raw = subagent_bootstrap.bootstrap_before_context(ctx, task, resolve_dispatch_axes(task))
    if ready and retry:
        assert not calls
        payload = json.loads(raw)
        assert payload["status"] == "configured_session_continuation"
        assert payload["startup"]["physical_start"] == "not_repeated"
    elif ready:
        assert calls == [True] and json.loads(raw)["status"] == "configured_session_started"
    else:
        assert not calls and raw == ""
        assert ctx._configured_startup_refusal["reason"] == "daemon_recovery_only"


@pytest.mark.parametrize("save_fails", [False, True])
@pytest.mark.parametrize("delivery", ["startup", "tool"])
def test_shown_interactions_save_before_ack_and_survive_cold_restore(
        tmp_path, monkeypatch, save_fails, delivery):
    from ouroboros import delegate_supervision as ds, loop, working_checkpoint as wc
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message
    from ouroboros.subagent_bootstrap import append_startup_receipt
    from tests.test_working_checkpoint import _registry, _saved_working_state
    from tests.test_sleep_questions import _deliver

    task_id, run_id = "nanny", "run-one"
    handoff = _saved_working_state(tmp_path, monkeypatch, task_id)
    registry = _registry(tmp_path, monkeypatch, task_id, 2)
    ctx = registry._ctx
    ctx.working_recovery = handoff
    messages, trace, usage, seen = [], {"tool_calls": []}, {}, set()
    ctx._loop_mailbox_seen_ids = seen
    state = loop._load_working_recovery(ctx, messages)
    loop._resume_continuation(registry, ({}, {}, state), messages, trace, usage, seen,
                              (ctx.active_model, ctx.active_effort, False, "max", 0, ctx.context_fit_plan),
                              ctx._cost_ceiling, 100.0)
    assert write_owner_message(tmp_path, "Retained correction", task_id, msg_id="wake-mail")
    payload = {"status": "waiting_on_user", "run_id": run_id, "supervision_wake_id": "wake-q1",
               "pendingInteractions": [{"interactionId": "q1", "question": "Which path?"}],
               "wake_events": [{"kind": "owner_text", "msg_id": "wake-mail", "text": "Retained correction"}]}
    ds._save_state(ctx, {"schema": 1, "run_id": run_id, "journal_cursor": 9, "pending_wake": {
        "wake_id": "wake-q1", "attempt_key": "2", "payload": payload, "mailbox_ids": ["wake-mail"],
        "interaction_ids": ["q1", "q-unshown"], "coordination_cursor": {"child": "v1"}}})
    if delivery == "startup":
        raw = json.dumps({"status": "configured_session_continuation", "wake": payload})
        append_startup_receipt(ctx, messages, raw, seen)
    else:
        raw = json.dumps(payload)
        messages.append({"role": "assistant", "tool_calls": [{"id": "wait-call", "type": "function",
                         "function": {"name": "delegate_wait", "arguments": json.dumps({"run_id": run_id})}}]})
        messages.extend(_deliver(ctx, raw, monkeypatch, tool="delegate_wait", meta={"supervision_wake_id": "wake-q1"}))
    limit = SimpleNamespace(tools=registry, messages=messages, llm_trace=trace, accumulated_usage=usage,
                            owner_msg_seen=seen, round_idx=8, tool_schemas=[], budget_tail="tool")
    assert ctx._sleep_interaction_seen == {run_id: ["q1"]}
    assert "wake-mail" in seen
    assert not drain_owner_entries(tmp_path, task_id, seen, 2), "live transcript already contains this mail"
    assert ds._load_state(ctx, "")["pending_wake"]
    assert drain_owner_entries(tmp_path, task_id, set(), 2)
    if save_fails:
        def fail(*_args, **_kwargs):
            raise OSError("controlled failed incorporated save")
        monkeypatch.setattr(wc, "_write", fail)
    wc.save_ready(limit)
    if save_fails:
        assert ds._load_state(ctx, "")["pending_wake"]
        assert drain_owner_entries(tmp_path, task_id, set(), 2)
        assert not wc.checkpoint_path(tmp_path, task_id, 2).exists()
        assert ctx._sleep_interaction_seen == {run_id: ["q1"]}, "shown bytes remain in live transcript"
        crashed = SimpleNamespace(task_id=task_id, task_attempt=2, drive_root=tmp_path,
                                  budget_drive_root=tmp_path, _loop_mailbox_seen_ids=set())
        replay = ds._pending_payload(crashed, ds._load_state(crashed, run_id))
        assert replay["wake_events"][0]["msg_id"] == "wake-mail", "unsaved crash replays at least once"
        return
    saved = wc.read_working(tmp_path, task_id, 2)[1]
    assert saved["sleep_interaction_seen"] == {run_id: ["q1"]}
    assert any(raw in str(row.get("content")) for row in saved["messages"])
    assert not ds._load_state(ctx, "")["pending_wake"]
    assert not drain_owner_entries(tmp_path, task_id, set(), 2)
    next_task = {"id": task_id}
    assert wc.attach_recovery(tmp_path, next_task, source_task_id=task_id, from_attempt=2, cause="worker_crash")
    cold = _registry(tmp_path, monkeypatch, task_id, 3)
    cold._ctx.working_recovery = next_task["_working_recovery"]
    cold_messages = []
    state = loop._load_working_recovery(cold._ctx, cold_messages)
    loop._resume_continuation(cold, ({}, {}, state), cold_messages, {}, {}, set(),
                             (ctx.active_model, ctx.active_effort, False, "max", 0, ctx.context_fit_plan),
                             ctx._cost_ceiling, 100.0)
    assert cold._ctx._sleep_interaction_seen == {run_id: ["q1"]}
    wait_calls = []
    def observe(_ctx, rid, *_args, **_kwargs):
        assert rid == run_id
        current = ds._load_state(_ctx, rid)
        assert current["journal_cursor"] == 9 and current["coordination_cursor"] == {"child": "v1"}
        wait_calls.append(True)
        return json.dumps({"status": "terminal", "run_id": rid, "state": "succeeded"})
    raw_wait = ds.supervised_wait(cold._ctx, run_id, wait_once=observe)
    assert wait_calls == [True], "the acknowledged old wake is not replayed by the later wait"
    assert "terminal" in raw_wait.text


def test_restored_run_terminal_keeps_child_and_parent_execution_evidence(tmp_path, monkeypatch, stub_delegate_engines):
    from ouroboros import delegate_custody as custody, delegate_supervision as ds, subagent_history
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.subagents import envelope_from_task
    from ouroboros.tools.control_task_results import _compact_child_projection
    from tests.test_working_checkpoint import _registry

    ctx = _registry(tmp_path, monkeypatch, "nanny", 2)._ctx
    ctx.working_recovery = {"source_ref": {"path": "saved"}}
    task = _task(ctx, "working")
    _known_custody(tmp_path, task, "live")
    resolve_dispatch_axes(task)
    assert not stub_delegate_engines
    def succeeded(_ctx, rid, *_args, **_kw):
        assert custody.emit(tmp_path, custody.SETTLED, {"task_id": "nanny", "run_id": rid,
                                                       "state": "succeeded", "model": "stub-model"})
        return json.dumps({"status": "terminal", "run_id": rid, "state": "succeeded"})
    assert "terminal" in ds.supervised_wait(ctx, "run-one", wait_once=succeeded).text
    envelope = envelope_from_task(task, status="completed")
    assert envelope["actual_substrate"] == "harness_used"
    assert envelope["execution_evidence"]["delegated_runs_succeeded"] == 1
    projection = _compact_child_projection("nanny", {**task, "status": "completed", "subagent_envelope": envelope}, None)
    assert projection["execution_evidence"]["actual_substrate"] == "harness_used"
    assert projection["execution_evidence"]["delegated_runs_started"] == 1
    history = []
    monkeypatch.setattr(subagent_history, "record_last_delegation", lambda **kw: history.append(kw))
    subagent_history.record_task_execution(task, {}, drive_root=tmp_path)
    assert not history, "unobserved availability cannot invent a not_started history row"


def test_saved_dispatch_keeps_dated_failure_separate_from_unobserved_availability(tmp_path, monkeypatch):
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from tests.test_working_checkpoint import _registry

    ctx = _registry(tmp_path, monkeypatch, "nanny", 2)._ctx
    ctx.working_recovery = {}
    task = _task(ctx, "working")
    previous = {"status": "unavailable", "reason": "daemon_recovery_only", "observed_at": "2026-10-20T00:00:00Z"}
    task["subagent_availability"] = previous
    dispatch = resolve_dispatch_axes(task)
    assert dispatch.availability["status"] == "unobserved"
    assert dispatch.availability["previous_observation"] == previous
    assert not dispatch.delta.reduced
    task.pop("parent_cognitive_route")
    refused = resolve_dispatch_axes(task)
    assert refused.blocked and refused.executor_resolution.reason == "parent_cognitive_route_missing"


def test_governing_carrier_priority_does_not_fall_back_from_an_invalid_owner_wait(tmp_path, monkeypatch):
    from ouroboros.agent import _load_start_continuation
    from ouroboros.owner_wait import governing_continuation
    from supervisor.task_admission import governing_continuation as admission_selector
    from tests.test_working_checkpoint import _registry

    assert admission_selector is governing_continuation
    task = {"_owner_wait_resume": {"wait_id": "missing", "restart_transaction_id": "old"},
            "_budget_pause_resume": {"pause_id": "older"}, "_working_recovery": {}}
    assert governing_continuation(task) == "owner_wait"
    ctx = _registry(tmp_path, monkeypatch, "nanny", 2)._ctx
    monkeypatch.setattr("ouroboros.budget_pause.load_budget_pause", lambda *_a: pytest.fail("selected source cannot fall back"))
    monkeypatch.setattr("ouroboros.working_checkpoint.load_recovery", lambda *_a: pytest.fail("selected source cannot fall back"))
    with pytest.raises(ValueError, match="active planned-restart"):
        _load_start_continuation(ctx, task)


def test_saved_zero_run_episode_can_choose_its_first_start_after_restoration(tmp_path, monkeypatch):
    from ouroboros import loop, subagent_bootstrap, subagent_runtime
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.delegate_shared import delegate_result
    from tests.test_loop_transport_wait import _loop_kwargs

    registry = _restored_registry(tmp_path, monkeypatch, "working")
    ctx = registry._ctx
    task = _task(ctx, "working")
    starts = []
    monkeypatch.setattr(subagent_runtime, "exact_start", lambda _ctx, prompt, spec:
                        starts.append((prompt, spec)) or delegate_result({"status": "started", "run_id": "chosen-run"}))
    ctx._configured_startup_wake = subagent_bootstrap.bootstrap_before_context(ctx, task, resolve_dispatch_axes(task))
    assert not starts
    def choose(call, *_a, **_kw):
        assert any(row.get("content") == "Saved object 42" for row in call.messages)
        result = subagent_runtime.delegate_start_entry(ctx, "")
        assert json.loads(result.text)["run_id"] == "chosen-run"
        assert len(starts) == 1
        raise InputReached
    monkeypatch.setattr(loop, "_dispatch_round_model", choose)
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs["task_id"] = "nanny"
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)

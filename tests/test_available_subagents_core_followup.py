"""Focused regressions for the core Available-subagents runtime follow-up."""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest


def _settings(*rows):
    return {
        "OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": list(rows)}),
    }


def _api_row():
    return {
        "subagent_id": "api-builder",
        "name": "API builder",
        "recommended_use": "Exact recursive API actor.",
        "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-sol"},
        "effort": "high",
    }


def _session_row():
    return {
        "subagent_id": "session-builder",
        "name": "Session builder",
        "recommended_use": "Subscription-backed implementation.",
        "route": {
            "kind": "agent_session",
            "target_id": "codex=gpt-5.6-sol",
            "credential_profile_id": "profile-1",
        },
        "effort": "high",
    }


def _snapshot(settings, row_id):
    from ouroboros.subagent_runtime import select_subagent_snapshot

    return select_subagent_snapshot(settings, subagent_id=row_id)[0]


def test_configured_api_actor_is_requested_native_without_fallback_note(monkeypatch):
    from ouroboros.subagent_dispatch_notes import dispatch_executor_note
    from ouroboros.subagent_runtime import resolve_configured_actor_dispatch

    monkeypatch.setattr(
        "ouroboros.provider_models.model_has_credentials",
        lambda _model: True,
    )
    snapshot = _snapshot(_settings(_api_row()), "api-builder")
    dispatch = resolve_configured_actor_dispatch({
        "id": "child1",
        "type": "task",
        "configured_subagent": snapshot,
        "task_constraint": {},
    }, task_type="task")
    assert dispatch.executor_resolution.reason == "requested_native"
    assert dispatch_executor_note(dispatch.executor_resolution, dispatch.lane) == ""


def test_configured_api_actor_without_credentials_refuses_with_alternatives(monkeypatch):
    from ouroboros.subagent_dispatch_notes import executor_blocked_outcome
    from ouroboros.subagent_runtime import resolve_configured_actor_dispatch

    monkeypatch.setattr(
        "ouroboros.provider_models.model_has_credentials",
        lambda _model: False,
    )
    monkeypatch.setattr(
        "ouroboros.subagent_runtime.current_subagent_alternatives",
        lambda excluded: [{
            "subagent_id": "session-builder",
            "name": "Session builder",
            "route_kind": "agent_session",
            "availability": "check_at_dispatch",
        }] if excluded == "api-builder" else [],
    )
    snapshot = _snapshot(_settings(_api_row(), _session_row()), "api-builder")
    dispatch = resolve_configured_actor_dispatch({
        "id": "child1",
        "type": "task",
        "configured_subagent": snapshot,
        "task_constraint": {},
    }, task_type="task")

    assert dispatch.blocked is True
    assert dispatch.executor == "blocked"
    assert dispatch.route == ""
    assert dispatch.executor_resolution.reason == "credentials_unavailable"
    assert dispatch.availability == {
        "observed_at": dispatch.availability["observed_at"],
        "status": "credentials_unavailable",
        "reason": "credentials_unavailable",
        "route_kind": "api_model",
        "selected_subagent_id": "api-builder",
        "alternatives": [{
            "subagent_id": "session-builder",
            "name": "Session builder",
            "route_kind": "agent_session",
            "availability": "check_at_dispatch",
        }],
        "host_fallback": False,
    }
    text, usage = executor_blocked_outcome(
        dispatch.executor_resolution,
        availability=dispatch.availability,
    )
    assert "selected API-model actor has no usable credentials" in text
    assert "session-builder" in text
    assert "delegated substrate" not in text
    assert "executor='harness'" not in text
    assert usage["reason_code"] == "subagent_executor_unavailable"
    assert usage["unavailable_reason"] == "credentials_unavailable"
    assert usage["host_fallback"] is False


def test_uncredentialed_api_actor_stops_before_the_llm_loop(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module
    from ouroboros.agent import Env, OuroborosAgent

    repo, drive = tmp_path / "repo", tmp_path / "drive"
    repo.mkdir()
    drive.mkdir()
    monkeypatch.setattr(
        "ouroboros.provider_models.model_has_credentials", lambda _model: False,
    )
    monkeypatch.setattr(
        "ouroboros.subagent_runtime.current_subagent_alternatives",
        lambda excluded: [{"subagent_id": "session-builder"}]
        if excluded == "api-builder" else [],
    )
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr(agent_module, "build_llm_messages", lambda **_kwargs: ([], {}))
    calls = []
    monkeypatch.setattr(
        agent_module,
        "run_llm_loop",
        lambda **kwargs: calls.append(kwargs) or ("unexpected", {}, {}),
    )

    snapshot = _snapshot(_settings(_api_row(), _session_row()), "api-builder")
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    events = agent._handle_task_scoped({
        "id": "api-child",
        "type": "task",
        "chat_id": 1,
        "text": "Use the exact API actor",
        "delegation_role": "subagent",
        "configured_subagent": snapshot,
        "task_constraint": {},
        "drive_root": str(drive),
        "budget_drive_root": str(drive),
    })

    assert calls == []
    result = json.loads(
        (drive / "task_results" / "api-child.json").read_text(encoding="utf-8")
    )
    assert result["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert result["reason_code"] == "subagent_executor_unavailable"
    assert result["subagent_availability"]["host_fallback"] is False
    assert "session-builder" in result["result"]
    assert any(event.get("type") == "task_done" for event in events)


def test_unreadable_named_owner_source_refuses_before_model_or_tool_work(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.project_dialogue import build_owner_message_ref

    repo, drive = tmp_path / "repo", tmp_path / "drive"
    repo.mkdir()
    drive.mkdir()
    missing_text = "owner directive that no surviving source can resolve"
    ref = build_owner_message_ref(
        chat_id=1, client_message_id="missing-owner", ts="2026-08-21T00:00:00Z",
        text=missing_text,
    )
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    model_calls, tool_context_calls = [], []
    monkeypatch.setattr(
        agent_module, "run_llm_loop",
        lambda **kwargs: model_calls.append(kwargs) or ("unexpected", {}, {}),
    )
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    monkeypatch.setattr(
        agent.tools, "set_context", lambda ctx: tool_context_calls.append(ctx),
    )

    events = agent._handle_task_scoped({
        "id": "missing-authority",
        "type": "task",
        "chat_id": 1,
        "text": "continue",
        "origin_message_ref": ref,
    })

    assert model_calls == []
    assert tool_context_calls == []
    result = json.loads(
        (drive / "task_results" / "missing-authority.json").read_text(encoding="utf-8")
    )
    assert result["reason_code"] == "authority_source_unavailable"
    assert result["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert result["trace_summary"].count("authority_source_unavailable") == 1
    assert any(event.get("type") == "task_done" for event in events)


def test_malformed_named_authority_shapes_refuse_before_model_or_tool_work(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.project_dialogue import _text_sha256

    repo, drive = tmp_path / "repo", tmp_path / "drive"
    repo.mkdir()
    (drive / "task_results").mkdir(parents=True)
    (drive / "task_results" / "old-root.json").write_text(json.dumps({
        "task_id": "old-root", "status": "completed", "objective": "old authority",
    }), encoding="utf-8")
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    model_calls, tool_context_calls = [], []
    monkeypatch.setattr(
        agent_module, "run_llm_loop",
        lambda **kwargs: model_calls.append(kwargs) or ("unexpected", {}, {}),
    )
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    monkeypatch.setattr(agent.tools, "set_context", lambda ctx: tool_context_calls.append(ctx))
    tasks = [{
        "id": "malformed-origin", "type": "task", "chat_id": 1, "text": "continue",
        "origin_message_ref": {
            "chat_id": 1, "client_message_id": "owner-1",
            "text_sha256": _text_sha256("exact retained text"),
        },
        "origin_message_text": "exact retained text",
    }, {
        "id": "malformed-predecessor", "type": "task", "chat_id": 1, "text": "continue",
        "predecessor_authority_source": {
            "kind": "task_result", "task_id": "old-root", "tool": "get_task_result",
            "arguments": {"task_id": "old-root", "include_authority": False},
        },
    }]

    for task in tasks:
        events = agent._handle_task_scoped(task)
        result = json.loads(
            (drive / "task_results" / f"{task['id']}.json").read_text(encoding="utf-8")
        )
        assert result["reason_code"] == "authority_source_unavailable"
        assert any(event.get("type") == "task_done" for event in events)
    assert model_calls == []
    assert tool_context_calls == []


def test_context_build_exception_after_pre_start_still_propagates(monkeypatch, tmp_path):
    # Charter (owner 2026-08-28): the leaf pre-starts BEFORE the context build,
    # so a context-assembly failure now happens with a LIVE run behind it.
    # The failure must still propagate loudly (the durable custody rows let the
    # retry adopt the running leaf instead of starting a duplicate).
    from ouroboros import agent as agent_module
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.delegate_supervision as supervision
    import ouroboros.subagent_runtime as runtime
    import ouroboros.subagents as subagents
    from ouroboros.agent import Env, OuroborosAgent

    repo, drive = tmp_path / "repo", tmp_path / "drive"
    repo.mkdir()
    drive.mkdir()
    order = []
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(subagents, "route_health", lambda *_a, **_k: ("", ""))
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr(runtime, "exact_start", lambda _ctx, _prompt, _spec: (
        order.append("physical_start")
        or json.dumps({"status": "started", "run_id": "run-pre"})
    ))
    monkeypatch.setattr(
        supervision, "supervised_wait",
        lambda *_a, **_kw: pytest.fail("the host must not wait inside bootstrap (owner 1=A)"),
    )
    def fail_context(**_kwargs):
        order.append("context_build_failed")
        raise RuntimeError("context assembly failed after the exact leaf start")

    monkeypatch.setattr(agent_module, "build_llm_messages", fail_context)
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    agent.tools.available_tools = lambda: ["delegate_start", "delegate_wait", "delegate_cancel"]
    with pytest.raises(RuntimeError, match="after the exact leaf start"):
        agent._prepare_task_context({
            "id": "child1", "type": "task", "chat_id": 1, "text": "Build",
            "delegation_role": "subagent", "configured_subagent": snapshot,
            "parent_cognitive_route": {
                "model": "openai/parent", "effort": "high", "use_local_model": False,
            },
            "task_constraint": {},
            "task_contract": {"objective": "Build", "expected_output": "Patch"},
            "drive_root": str(drive), "budget_drive_root": str(drive),
        })
    assert order == ["physical_start", "context_build_failed"]


@pytest.mark.parametrize("message_kind", ["owner", "task"])
def test_awake_loop_durably_acks_injected_mailbox_before_next_sleep(tmp_path, message_kind):
    import ouroboros.delegate_supervision as supervision
    from ouroboros.loop import _drain_incoming_messages
    from ouroboros.owner_mailbox import (
        acknowledged_task_message_ids,
        write_owner_message,
        write_task_message,
    )

    if message_kind == "owner":
        assert write_owner_message(tmp_path, "owner direction", "child1", msg_id="m1")
    else:
        assert write_task_message(
            tmp_path, "ancestor direction", "child1",
            source_task_id="parent1", provenance="ancestor_task", msg_id="m1",
        )
    ctx = SimpleNamespace(
        task_id="child1", drive_root=tmp_path, budget_drive_root=str(tmp_path),
        task_metadata={},
    )
    messages = []
    assert _drain_incoming_messages(
        messages, queue.Queue(), tmp_path, "child1", None, set(), owner_ctx=ctx,
    ) == {}
    assert acknowledged_task_message_ids(tmp_path, "child1") == {"m1"}
    assert supervision._addressed_wakes(ctx, {}) == []
    rendered = json.dumps(messages, ensure_ascii=False)
    expected = "Message from my human" if message_kind == "owner" else "ancestor task parent1"
    assert expected in rendered


def test_delegate_owner_wake_ack_replays_on_fresh_physical_attempt(tmp_path):
    import ouroboros.delegate_supervision as supervision
    from ouroboros.owner_mailbox import write_owner_message

    exact = "delegate owner bytes  \n"
    assert write_owner_message(tmp_path, exact, "child1", msg_id="owner-1")
    first_ctx = SimpleNamespace(
        task_id="child1", task_attempt=1, drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_metadata={},
    )
    wake = json.loads(supervision.supervised_wait(
        first_ctx, "run-1",
        wait_once=lambda *_a, **_k: json.dumps({
            "status": "no_progress", "run_id": "run-1", "last_seq": 0,
        }),
    ))
    assert wake["wake_events"][0]["text"] == exact
    assert supervision.acknowledge_pending_wake(first_ctx, wake)
    assert supervision._addressed_wakes(first_ctx, supervision.supervision_checkpoint(first_ctx)) == []

    successor = SimpleNamespace(**{**first_ctx.__dict__, "task_attempt": 2})
    replay = supervision._addressed_wakes(
        successor, supervision.supervision_checkpoint(successor),
    )
    assert [(row["msg_id"], row["text"]) for row in replay] == [
        ("owner-1", exact),
    ]


def test_unacknowledged_delegate_wake_replays_before_successor_poll(tmp_path):
    import ouroboros.delegate_supervision as supervision
    from ouroboros.owner_mailbox import write_owner_message

    assert write_owner_message(tmp_path, "pending exact", "child1", msg_id="owner-pending")
    first_ctx = SimpleNamespace(
        task_id="child1", task_attempt=1, drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_metadata={},
    )
    first = json.loads(supervision.supervised_wait(
        first_ctx, "run-1",
        wait_once=lambda *_a, **_k: json.dumps({
            "status": "no_progress", "run_id": "run-1", "last_seq": 0,
        }),
    ))
    successor = SimpleNamespace(**{**first_ctx.__dict__, "task_attempt": 2})
    assert supervision.acknowledge_pending_wake(successor) is False
    assert supervision.supervision_checkpoint(successor)["pending_wake"]
    replay = json.loads(supervision.supervised_wait(
        successor, "run-1",
        wait_once=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("successor must replay before polling the harness")
        ),
    ))
    assert replay["supervision_wake_id"] == first["supervision_wake_id"]
    assert replay["wake_events"][0]["text"] == "pending exact"


@pytest.mark.parametrize("control_kind", ["finalize_now", "hurry"])
def test_sleeping_control_wakes_then_loop_routes_without_supervision_ack(tmp_path, control_kind):
    import ouroboros.delegate_supervision as supervision
    from ouroboros.loop import _drain_incoming_messages
    from ouroboros.owner_mailbox import (
        KIND_FINALIZE_NOW,
        KIND_HURRY,
        acknowledged_task_message_ids,
        write_owner_message,
    )

    kind = KIND_FINALIZE_NOW if control_kind == "finalize_now" else KIND_HURRY
    text = "owner_requested_finalization" if kind == KIND_FINALIZE_NOW else "owner_hurry"
    control_msg_id = "control-1"
    if kind == KIND_FINALIZE_NOW:
        from ouroboros.cancel_intents import (
            STOP_POLICY_FINALIZE,
            request_cancel,
        )
        from supervisor.owner_stop import owner_stop_control_id

        intent = request_cancel(
            tmp_path,
            "child1",
            requested_stop_policy=STOP_POLICY_FINALIZE,
        )
        control_msg_id = owner_stop_control_id(intent)
    assert write_owner_message(
        tmp_path,
        text,
        "child1",
        msg_id=control_msg_id,
        kind=kind,
    )
    ctx = SimpleNamespace(
        task_id="child1", drive_root=tmp_path, budget_drive_root=str(tmp_path),
        task_metadata={}, task_attempt=1,
    )
    wake = json.loads(supervision.supervised_wait(
        ctx, "run-1",
        wait_once=lambda *_a, **_k: json.dumps({
            "status": "no_progress", "run_id": "run-1", "last_seq": 0,
        }),
    ))
    assert wake["wake_events"][0]["kind"] == kind
    assert supervision.supervision_checkpoint(ctx)["pending_wake"]["mailbox_ids"] == []
    assert supervision.acknowledge_pending_wake(ctx, wake)
    assert acknowledged_task_message_ids(tmp_path, "child1") == set()
    assert supervision._addressed_wakes(
        ctx, supervision.supervision_checkpoint(ctx),
    ) == []

    controls = _drain_incoming_messages(
        [], queue.Queue(), tmp_path, "child1", None, set(), owner_ctx=ctx,
    )
    assert control_kind in controls
    assert acknowledged_task_message_ids(tmp_path, "child1") == set()
    assert supervision._addressed_wakes(ctx, {}) == []


def test_refused_bootstrap_receipt_never_claims_a_live_leaf(monkeypatch):
    import ouroboros.delegate_supervision as supervision
    from ouroboros.subagent_bootstrap import append_startup_receipt

    monkeypatch.setattr(supervision, "acknowledge_pending_wake", lambda *_a, **_k: True)
    messages = []
    append_startup_receipt(
        SimpleNamespace(), messages,
        json.dumps({
            "status": "configured_session_start_wake",
            "startup": {
                "status": "refused",
                "reason": "work_order_source_channel_unavailable",
            },
        }),
    )
    receipt = messages[0]["content"]
    assert "Physical custody is unresolved" in receipt
    assert "proves a physical run" not in receipt
    assert "never authorizes native/API fallback" in receipt


def test_crash_handoff_does_not_replay_attempt_local_loop_controls(tmp_path):
    from ouroboros import delegate_custody as custody
    import ouroboros.delegate_recovery as recovery
    from ouroboros.owner_mailbox import KIND_HURRY
    from ouroboros.subagent_work_order import work_order_fingerprint
    from ouroboros.utils import atomic_write_json

    custody._CUSTODY.clear()
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    task = {
        "id": "child1", "_attempt": 2, "configured_subagent": snapshot,
        "drive_root": str(tmp_path), "task_constraint": {}, "task_contract": {},
    }
    custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-1", task_id="child1", route_id="codex",
        selected_subagent_id="session-builder",
        config_fingerprint=snapshot["config_fingerprint"],
        authority_fingerprint=recovery.authority_fingerprint_from_task(task),
        work_order_fingerprint=work_order_fingerprint(task),
    ))
    state_path = tmp_path / "state" / "delegate_supervision" / "child1.json"
    state_path.parent.mkdir(parents=True)
    atomic_write_json(state_path, {
        "schema": 1, "run_id": "run-1", "status": "awake",
        "last_acknowledged_wake": {
            "wake_id": "old-wake", "acknowledged_at": "now",
            "payload": {
                "status": "no_progress", "run_id": "run-1",
                "wake_events": [
                    {"type": "owner_message", "kind": KIND_HURRY, "msg_id": "h1"},
                    {"type": "task_message", "kind": "task_message", "msg_id": "m1"},
                ],
            },
        },
    })
    handoff = recovery.prepare_handoff(
        tmp_path, task, cause=recovery.CAUSE_WORKER_CRASH,
        old_attempt=1, new_attempt=2, worker_id=1, exitcode=1,
    )
    assert handoff["pending_wake"]["payload"]["wake_events"] == [
        {"type": "task_message", "kind": "task_message", "msg_id": "m1"},
    ]
    assert recovery._successor_pending_wake({
        "payload": {"status": "no_progress", "wake_events": [{"kind": KIND_HURRY}]},
    }) == {}


def test_planned_restart_retry_reset_preserves_owner_not_controls(monkeypatch, tmp_path):
    from ouroboros.owner_mailbox import (
        KIND_HURRY,
        drain_owner_entries,
        write_owner_message,
    )
    from supervisor import queue as task_queue
    from supervisor import workers

    repo = tmp_path / "repo"
    repo.mkdir()
    workers.init(repo, tmp_path, 1)
    workers.WORKERS.clear()
    workers.PENDING.clear()
    workers.RUNNING.clear()
    task = {"id": "child", "parent_task_id": "parent", "root_task_id": "parent"}
    workers.RUNNING["child"] = {"task": task, "attempt": 1}
    assert write_owner_message(tmp_path, "exact owner", "child", msg_id="owner-1")
    assert write_owner_message(
        tmp_path, "owner_hurry", "child", msg_id="hurry-1", kind=KIND_HURRY,
    )
    monkeypatch.setattr(task_queue, "persist_queue_snapshot", lambda *_a, **_k: True)

    workers.kill_workers(
        preserve_pending=True, preserve_running_task_ids={"child"},
    )

    assert [row["_attempt"] for row in workers.PENDING] == [2]
    assert [(row["msg_id"], row["text"]) for row in drain_owner_entries(
        tmp_path, "child", attempt_key=2,
    )] == [("owner-1", "exact owner")]
    workers.PENDING.clear()
    workers.RUNNING.clear()

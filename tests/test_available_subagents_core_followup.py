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


@pytest.mark.parametrize(
    ("startup_status", "reason"),
    [("refused", "work_order_budget_exceeded"), ("temporarily_unavailable", "subscription_window_exhausted")],
)
def test_definite_configured_session_no_start_terminalizes_before_llm(
    monkeypatch, tmp_path, startup_status, reason,
):
    """A selected session refusal is not permission to do the assignment natively."""
    from ouroboros import agent as agent_module
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.subagent_bootstrap as bootstrap
    import ouroboros.subagents as subagents
    from ouroboros.agent import Env, OuroborosAgent

    repo, drive = tmp_path / "repo", tmp_path / "drive"
    repo.mkdir()
    drive.mkdir()
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(subagents, "route_health", lambda *_a, **_k: ("", ""))
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr(agent_module, "build_llm_messages", lambda **_kwargs: ([], {}))
    monkeypatch.setattr(
        bootstrap,
        "bootstrap_session_leaf",
        lambda *_a, **_k: json.dumps({
            "status": "configured_session_start_wake",
            "startup": {"status": startup_status, "reason": reason},
        }),
    )
    calls = []
    monkeypatch.setattr(
        agent_module,
        "run_llm_loop",
        lambda **kwargs: calls.append(kwargs) or ("unexpected", {}, {}),
    )

    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    agent.tools.available_tools = lambda: ["delegate_start", "delegate_wait", "delegate_cancel"]
    events = agent._handle_task_scoped({
        "id": "session-child",
        "type": "task",
        "chat_id": 1,
        "text": "Use the exact session actor",
        "delegation_role": "subagent",
        "configured_subagent": snapshot,
        "parent_cognitive_route": {
            "model": "openai/parent", "effort": "high", "use_local_model": False,
        },
        "task_constraint": {},
        "task_contract": {"objective": "Build", "expected_output": "Patch"},
        "drive_root": str(drive),
        "budget_drive_root": str(drive),
    })

    assert calls == []
    result = json.loads(
        (drive / "task_results" / "session-child.json").read_text(encoding="utf-8")
    )
    assert result["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert result["subagent_availability"]["host_fallback"] is False
    assert reason in result["result"]
    assert any(event.get("type") == "task_done" for event in events)


def test_startup_refusal_classifier_preserves_ambiguous_wakes():
    from ouroboros.subagent_bootstrap import startup_refusal_outcome

    def receipt(status, *, outer="configured_session_start_wake"):
        return json.dumps({"status": outer, "startup": {"status": status, "reason": "x"}})

    assert startup_refusal_outcome(receipt("refused"))["usage"]["reason_code"] == "configured_subagent_start_refused"
    assert startup_refusal_outcome(receipt("temporarily_unavailable"))["usage"]["reason_code"] == "subagent_executor_unavailable"
    assert startup_refusal_outcome(receipt("started_uncustodied")) is None
    assert startup_refusal_outcome(receipt("pending")) is None
    assert startup_refusal_outcome(receipt("refused", outer="configured_session_recovery_wake")) is None
    for key, value in (
        ("pending_invocation_id", "inv-1"),
        ("run_id", "run-1"),
        ("queued_handle", {"jobId": "job-1"}),
    ):
        payload = json.loads(receipt("refused"))
        payload["startup"][key] = value
        assert startup_refusal_outcome(json.dumps(payload)) is None

    # A malformed exact-start response is also ambiguous: the host cannot prove
    # that the POST did not reach the daemon, so it must not synthesize a clean
    # terminal no-start.
    assert startup_refusal_outcome("not-json") is None


def test_context_build_exception_propagates_after_exact_leaf_bootstrap(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.subagent_bootstrap as bootstrap
    import ouroboros.subagents as subagents
    from ouroboros.agent import Env, OuroborosAgent

    repo, drive = tmp_path / "repo", tmp_path / "drive"
    repo.mkdir()
    drive.mkdir()
    order = []
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(subagents, "route_health", lambda *_a, **_k: ("", ""))
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr(
        bootstrap,
        "bootstrap_session_leaf",
        lambda *_a, **_k: order.append("exact_leaf_started") or json.dumps({
            "status": "configured_session_wake", "wake": {"status": "completed"},
        }),
    )

    def fail_context(**_kwargs):
        order.append("context_build_failed")
        raise RuntimeError("context assembly failed after exact leaf start")

    monkeypatch.setattr(agent_module, "build_llm_messages", fail_context)
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    agent.tools.available_tools = lambda: ["delegate_start", "delegate_wait", "delegate_cancel"]
    with pytest.raises(RuntimeError, match="after exact leaf start"):
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
    assert order == ["exact_leaf_started", "context_build_failed"]


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
    assert write_owner_message(tmp_path, text, "child1", msg_id="control-1", kind=kind)
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
            "startup": {"status": "refused", "reason": "work_order_budget_exceeded"},
        }),
    )
    receipt = messages[0]["content"]
    assert "does not imply a live leaf" in receipt
    assert "external start already happened" not in receipt
    assert "typed receipt alone" in receipt


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

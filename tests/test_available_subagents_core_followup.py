"""Focused regressions for the core Available-subagents runtime follow-up."""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from tests._store_shared import read_result, seed_result


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
    result = read_result(drive, "api-child")
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
    result = read_result(drive, "missing-authority")
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
    seed_result(drive, "old-root", {
        "task_id": "old-root", "status": "completed", "objective": "old authority",
    })
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
        result = read_result(drive, task['id'])
        assert result["reason_code"] == "authority_source_unavailable"
        assert any(event.get("type") == "task_done" for event in events)
    assert model_calls == []
    assert tool_context_calls == []


def _child_attempt(tmp_path, attempt):
    """One physical attempt of ``child1`` in a context of its own, as
    ``OuroborosAgent._prepare_task_context`` builds one per attempt: the ids of the
    mail a transcript holds (``_loop_mailbox_seen_ids``) return only with that
    transcript's restore."""
    return SimpleNamespace(
        task_id="child1", task_attempt=attempt, drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_metadata={},
    )


def test_failed_direct_wake_ack_records_no_exposure(tmp_path, monkeypatch):
    import ouroboros.delegate_supervision as supervision
    from tests._run_wait_shared import wake_now
    from ouroboros.owner_mailbox import write_owner_message

    assert write_owner_message(tmp_path, "held owner words", "child1", msg_id="owner-held")
    ctx = _child_attempt(tmp_path, 1)
    wake = json.loads(wake_now(
        ctx, "run-1",
        wait_once=lambda *_a, **_k: json.dumps({
            "status": "no_progress", "run_id": "run-1", "last_seq": 0,
        }),
    ).text)
    save = supervision._save_state

    def refuse_ack_save(actual_ctx, state):
        if state.get("last_acknowledged_wake") and not state.get("pending_wake"):
            raise OSError(28, "No space left on device")
        return save(actual_ctx, state)

    monkeypatch.setattr(supervision, "_save_state", refuse_ack_save)
    with pytest.raises(OSError):
        supervision.acknowledge_pending_wake(ctx, wake)
    # The UNKNOWN hold drops this receipt (#2142): a checkpoint saved now must not mark its mail seen.
    assert not getattr(ctx, "_loop_mailbox_seen_ids", None)
    restored = _child_attempt(tmp_path, 2)
    restored._loop_mailbox_seen_ids = set(getattr(ctx, "_loop_mailbox_seen_ids", None) or ())
    replay = supervision._addressed_wakes(restored, supervision.supervision_checkpoint(restored))
    assert [(row["msg_id"], row["text"]) for row in replay] == [("owner-held", "held owner words")]

    monkeypatch.setattr(supervision, "_save_state", save)
    assert supervision.acknowledge_pending_wake(ctx, wake)
    assert ctx._loop_mailbox_seen_ids == {"owner-held"}


@pytest.mark.parametrize("control_kind", ["finalize_now", "hurry"])
def test_sleeping_control_wakes_then_loop_routes_without_supervision_ack(tmp_path, control_kind):
    import ouroboros.delegate_supervision as supervision
    from tests._run_wait_shared import wake_now
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
    wake = json.loads(wake_now(
        ctx, "run-1",
        wait_once=lambda *_a, **_k: json.dumps({
            "status": "no_progress", "run_id": "run-1", "last_seq": 0,
        }),
    ).text)
    assert wake["wake_events"][0]["kind"] == kind
    assert supervision.supervision_checkpoint(ctx)["runs"]["run-1"]["pending_wake"]["mailbox_ids"] == []
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

"""A carried conversation turn keeps every behaviour that reads the turn's ORIGIN (review M2, K4).

``task_carrier.inline_execution`` is the CARRIER fact (custody, liveness: a thread of the admitting process)
and ``conversation_origin`` the ORIGIN fact (an owner's or a correspondent's turn, whatever carries it). An
owner turn carried by a conversation-lane worker (``executor="worker"``) must keep what the owner's
foreground turn always had: the interactive outage wait, the chat answer when the budget is exhausted, no
queued-task resume contract or root fence, no terminal-file detour, no workspace capture and no mutation
baseline. A pooled task keeps the queued-task behaviour in each case.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION, LANE_PROJECT

CARRIED = {"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}
POOLED = {"kind": "pooled", "lane": LANE_PROJECT, "executor": EXECUTOR_WORKER}


def _tool_ctx(root, owner, *, direct):
    return SimpleNamespace(task_id="t", drive_root=root, budget_drive_root=root, owner_wait_callback=None,
                           task_metadata={"_execution_owner": dict(owner)}, task_attempt=None,
                           is_direct_chat=direct, task_contract={})


def test_a_carried_owner_turn_keeps_the_interactive_outage_wait(tmp_path):
    from ouroboros import loop_transport

    def enter(ctx, task_id):
        return loop_transport.reconcile_transport_wait(
            None, ctx, msg_present=False, error_kind="transport_unavailable", drive_logs=tmp_path,
            task_id=task_id, model="m", emit_progress=lambda *_a, **_k: None)

    carried = enter(_tool_ctx(tmp_path, CARRIED, direct=True), "carried")
    assert carried.interactive is True and carried.wait_bound_sec is not None  # the idle bound, interactive notes
    pooled = enter(_tool_ctx(tmp_path, POOLED, direct=False), "pooled")
    assert pooled.interactive is False and pooled.wait_bound_sec is None


def test_the_loops_budget_rail_answers_a_carried_owner_turn_and_pauses_a_pooled_task(tmp_path, monkeypatch):
    from ouroboros import loop

    monkeypatch.setattr(loop, "_finalize_task_services", lambda _ctx: None)

    def rail(owner, direct):
        ctx = loop._LoopExitContext(tools=SimpleNamespace(_ctx=_tool_ctx(tmp_path, owner, direct=direct)),
                                    drive_root=tmp_path, task_id="t", event_queue=None, drive_logs=tmp_path,
                                    accumulated_usage={}, llm_trace={})
        return loop._handle_budget_exceeded(ua.BudgetExceeded("global model budget exhausted"), ctx)

    _text, terminal, trace = rail(CARRIED, True)  # no physical call yet: still the owner's chat answer
    assert terminal["reason_code"] == "budget_exhausted" and trace["resource_limit"]["replay_safe"] is False
    assert trace["resource_limit"]["resume_policy"] == "increase_or_reset_budget_then_retry"
    with pytest.raises(ua.BudgetExceeded):  # a pooled task with no physical call: the exact replay-safe pause
        rail(POOLED, False)


def _agent(root, monkeypatch):
    from ouroboros import agent as agent_module
    from ouroboros.settings_integrity import task_settings_snapshot

    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda *_a: None)
    monkeypatch.setattr(agent_module.OuroborosAgent, "_start_task_heartbeat_loop", lambda *_a: None)
    monkeypatch.setattr(agent_module, "validate_task_authority_sources", lambda *_a: None)
    monkeypatch.setattr(agent_module.subagent_runtime, "apply_task_start_settings_or_disclose",
                        lambda *_a: task_settings_snapshot({}, os.environ))
    agent = agent_module.OuroborosAgent(agent_module.Env(repo_dir=root, drive_root=root))
    agent.llm = SimpleNamespace(default_model=lambda: "test-model")

    def exhausted(_task, _refusal):
        raise ua.BudgetExceeded("global model budget exhausted")

    agent._prepare_task_context = exhausted
    return agent


@pytest.mark.parametrize("owner, direct, pauses", [(CARRIED, True, False), (POOLED, False, True)])
def test_the_agents_budget_branch_answers_a_carried_owner_turn_in_chat(tmp_path, monkeypatch, owner, direct, pauses):
    """Base answered a direct turn that met an exhausted budget with its chat reply; a carried owner turn is the
    same turn. A pooled task with no physical call becomes the queued-task budget pause."""
    task = {"id": "turn-b", "type": "task", "chat_id": 1, "text": "hello", "_is_direct_chat": direct,
            "_execution_owner": dict(owner), "metadata": {}}
    events = _agent(tmp_path, monkeypatch).handle_task(task)
    kinds = [event.get("type") for event in events]
    assert ("budget_pause" in kinds) is pauses
    assert ("task_done" in kinds) is not pauses


def test_a_carried_turns_completion_settles_without_the_terminal_file_detour(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor import events, task_reaper

    detours = []
    monkeypatch.setattr(task_reaper, "enqueue_terminal_file_recovery", lambda *a, **_k: detours.append(a) or True)
    task = {"id": "turn-c", "type": "task", "chat_id": 1, "_is_direct_chat": True, "_execution_owner": dict(CARRIED),
            "metadata": {}}
    write_task_result(tmp_path, "turn-c", "completed", result="answer", _is_direct_chat=True,
                      execution_owner=dict(CARRIED))
    slot = SimpleNamespace(busy_task_id="turn-c", reaping=False)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, WORKERS={3: slot}, append_jsonl=append_jsonl,
                          RUNNING={"turn-c": {"task": task, "worker_id": 3, "attempt": 1, "lane": LANE_CONVERSATION}},
                          persist_queue_snapshot=lambda reason="": None)
    events._handle_task_done({"task_id": "turn-c", "status": "completed", "worker_id": 3, "_is_direct_chat": True}, ctx)
    assert detours == [] and "turn-c" not in ctx.RUNNING and slot.busy_task_id is None


def test_an_owner_turns_workspace_is_never_captured_whatever_carries_it(tmp_path):
    from ouroboros.headless import task_requires_workspace_capture

    carried = {"id": "c", "workspace_root": str(tmp_path), "_is_direct_chat": True, "_execution_owner": dict(CARRIED)}
    pooled = {"id": "p", "workspace_root": str(tmp_path), "_execution_owner": dict(POOLED)}
    assert task_requires_workspace_capture(carried) is False
    assert task_requires_workspace_capture(pooled) is True


def test_only_a_queued_root_captures_a_mutation_baseline(tmp_path, monkeypatch):
    from ouroboros import mutation_attribution

    captured = []
    monkeypatch.setattr(mutation_attribution, "capture_mutation_baseline", lambda *a, **k: captured.append(a))
    agent = _agent(tmp_path, monkeypatch)
    agent._capture_mutation_baseline({"id": "c", "_is_direct_chat": True, "_execution_owner": dict(CARRIED)}, {})
    assert captured == []
    agent._capture_mutation_baseline({"id": "p", "_execution_owner": dict(POOLED)}, {})
    assert len(captured) == 1

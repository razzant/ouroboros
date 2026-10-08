"""A cold grant finishes the saved round before another ordinary model/tool call."""

import json
from dataclasses import replace

import pytest

from ouroboros import budget_pause, loop, owner_wait, pricing, task_pacing, usage_accounting as accounting
from ouroboros.contracts.task_contract import normalize_budget_profile
from ouroboros.owner_mailbox import write_owner_message
from tests.test_owner_wait_cold_loop import EXPLICIT, cold_registry
from tests.test_loop_transport_wait import _loop_kwargs


@pytest.mark.parametrize("queued_override", [False, True])
@pytest.mark.parametrize("cold_restart", [False, True])
def test_cold_grant_checks_saved_budget_before_ordinary_dispatch(tmp_path, monkeypatch, queued_override, cold_restart):
    (tmp_path / "logs").mkdir()
    (tmp_path / "fixture.txt").write_text("A real extra tool read")
    from ouroboros.task_results import write_task_result
    write_task_result(tmp_path, 't-wait', 'running', billing_group={
        'billing_group_id': 't-wait', 'billing_group_limit_usd': 50.0,
        'billing_group_limit_source': 'initial_task_admission', 'billing_group_limit_revision': 'fixture'})
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="t-wait", root_task_id="t-wait",
                                  global_limit_usd=200.0, root_limit_usd=50.0)
    calls, checkpoints, network = [], [], []
    phase = "warm"
    owner_answer = "Use the existing prepared result and explain the owner choice."

    def settle(amount):
        held = accounting.reserve_attempt(accounting.AttemptRequest(
            model="fixture", provider="fixture", reservation_usd=amount))
        accounting.mark_dispatched(held)
        accounting.settle_attempt(held, {"prompt_tokens": 1, "completion_tokens": 1},
                                  cost_usd=amount, cost_final=True)

    def refuse_network(*args, **kwargs):
        network.append((args, kwargs))
        raise AssertionError("fixture attempted a provider call")

    def forced(call, **kwargs):
        calls.append((phase, "forced", call.round_idx, call.active_model))
        return ("Current owner choice considered." if owner_answer in json.dumps(call.messages)
                else "Stale pre-answer draft.")

    def ordinary(call, disposition, **kwargs):
        calls.append((phase, "ordinary", call.round_idx, call.active_model))
        settle(.2)
        call.accumulated_usage["cost"] = float(call.accumulated_usage.get("cost") or 0) + .2
        if phase == "warm":
            if queued_override:
                # The switch_model tool leaves this pending until the NEXT round.
                call.tools._ctx.active_model_override = "after-budget"
            name, args = "escalate", {"question": "Continue the prepared result?",
                "options": [{"label": "Continue"}, {"label": "Stop"}],
                "stake": "Fixture action", "wait_for_answer": True}
        else:
            name, args = "read_file", {"root": "active_workspace", "path": "fixture.txt"}
        return {"role": "assistant", "content": "", "tool_calls": [{"id": phase, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}, .2

    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", refuse_network)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat_async", refuse_network)
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda provider: {})
    monkeypatch.setattr(loop, "_dispatch_round_model", ordinary)
    monkeypatch.setattr(loop, "_call_forced_model_once", forced)
    with accounting.usage_scope(scope):
        # An explicit experiment profile's authored stop ($50 cap minus its margin): the
        # saved threshold a cold grant must re-check before ordinary dispatch.
        ceiling = task_pacing.resolve_cost_ceiling(200, normalize_budget_profile(EXPLICIT["budget_profile"]),
                                                   root_cap_usd=50)
        with accounting.usage_scope(replace(scope, task_id="prior-child", parent_task_id="t-wait")):
            settle(46.9)

        def registry():
            tools = cold_registry(tmp_path, monkeypatch, ceiling, EXPLICIT)
            tools._ctx._owner_wait_requested = ""
            tools._ctx.current_chat_id, tools._ctx.current_task_type = 1, "task"
            tools._ctx.task_model_override = "same-model"
            return tools

        def park(ctx, checkpoint):
            checkpoints.append(checkpoint)
            owner_wait.set_owner_wait(tmp_path, "t-wait", {**checkpoint, "state": "waiting"})
            if cold_restart:
                # End the old actor at its owner wait, before either monetary
                # tail runs. The cold actor consumes that saved tail once.
                raise InterruptedError("planned restart")
            write_owner_message(tmp_path, owner_answer, task_id="t-wait", msg_id="warm-answer")

        warm = registry()
        warm._ctx.owner_wait_resume, warm._ctx.owner_wait_callback = None, park
        selected, remaining = warm, 200
        if cold_restart:
            with pytest.raises(InterruptedError, match="planned restart"):
                loop.run_llm_loop(**{**_loop_kwargs(tmp_path, warm, []),
                                    "budget_remaining_usd": remaining, "drive_logs": tmp_path / "logs"})
            phase = "cold"
            selected, remaining = registry(), 152.9
            owner_wait.set_owner_wait(tmp_path, "t-wait", {**checkpoints[0], "state": "waiting"})
            selected._ctx.owner_wait_resume = {**checkpoints[0], "restart_transaction_id": "observed-restart"}
            selected._ctx.owner_wait_callback = lambda *_: write_owner_message(
                tmp_path, owner_answer, task_id="t-wait", msg_id="cold-answer")
        try:
            with pytest.raises(budget_pause.BudgetPauseRequested) as paused:
                loop.run_llm_loop(**{**_loop_kwargs(tmp_path, selected, []),
                                    "budget_remaining_usd": remaining, "drive_logs": tmp_path / "logs"})
            state = json.loads(owner_wait.read_actor_source_bytes(tmp_path, "t-wait", paused.value.pause["source_ref"]))
            assert owner_answer in json.dumps(state["messages"])
            assert state["round_idx"] == 1 and state["usage"]["cost"] == pytest.approx(.2)
            assert state["route"]["active_model"] == "same-model"
            assert state["route"]["active_model_override"] == ("after-budget" if queued_override else None)
            assert [row["tool"] for row in state["trace"]["tool_calls"]] == ["escalate"]
        finally:
            budget_pause.end_dispatch_fence("t-wait")
        held = accounting.reserve_attempt(accounting.AttemptRequest(
            model="fixture", provider="fixture", reservation_usd=.2))
        accounting.release_attempt(held, "test did not send")  # hard50 still has room
    assert network == []
    assert len(checkpoints) == 1
    assert calls == [("warm", "ordinary", 1, "same-model")]

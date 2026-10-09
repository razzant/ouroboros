"""#1223 — a round that spends is a round that is measured, tools or not.

Only the TOOL tail ever reached `_check_budget_limits`, so a task that kept
re-answering under acceptance feedback (no tool calls at all) could run past its
cost ceiling untouched. These pin the unified tail: the SAME existing
comparison, none of the tool-only bookkeeping beside it, an unchanged disabled
ceiling, and a ready answer that still buys no extra wrap-up.
"""

from __future__ import annotations

import ast
import copy
import json
import pathlib
import queue
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ouroboros import task_pacing
from ouroboros.contracts.task_contract import normalize_budget_profile
from ouroboros.loop import _RoundLimitContext, _finish_no_tool_round_budget
from tests.test_acceptance_async_loop import full_loop as full_loop


def _ctx(**overrides):
    values = dict(
        messages=[], llm=MagicMock(), active_model="anthropic/claude-test",
        active_effort="high", max_retries=1, drive_logs=None, task_id="task1",
        round_idx=3, event_queue=queue.Queue(),
        accumulated_usage={"cost": 1.0, "_context_prompt_estimate": 0},
        task_type="task", active_use_local=False, max_rounds=100, llm_trace={},
    )
    values.update(overrides)
    return _RoundLimitContext(**values)


def _ceiling(root_cap):
    return task_pacing.resolve_cost_ceiling(
        None, normalize_budget_profile({"cost_hard_stop_pct": 50}), root_cap_usd=root_cap,
    )


def test_an_unfinished_no_tool_round_takes_the_same_over_ceiling_exit(monkeypatch):
    """Parity with the tool tail: same comparison, same typed reason."""
    from ouroboros import loop as loop_mod

    ctx = _ctx()
    monkeypatch.setattr(loop_mod, "_loop_tree_accounting", lambda **_k: {"settled_usd": 99.0, "accounted_usd": 99.0})
    monkeypatch.setattr(
        loop_mod, "_forced_final_answer",
        lambda ctx_, **kwargs: ("wrapped up", ctx_.accumulated_usage, {"kwargs": kwargs}),
    )
    result = _finish_no_tool_round_budget(ctx, None, _ceiling(50.0))
    assert result is not None
    text, _usage, trace = result
    assert text == "wrapped up"
    assert trace is ctx.llm_trace          # the trace the loop keeps returning
    assert ctx.accumulated_usage["cost_stop_spend_basis"]


def test_it_never_reaches_for_the_tool_only_bookkeeping_or_delivery_arming(monkeypatch):
    """No tools ran, so the metered nanny baseline and the post-tool
    delivery-control arming have nothing to say about this round."""
    from ouroboros import loop as loop_mod

    touched = []
    monkeypatch.setattr(loop_mod, "_loop_tree_accounting", lambda **_k: {"settled_usd": 99.0, "accounted_usd": 99.0})
    monkeypatch.setattr(loop_mod, "_prepare_post_tool_budget_context",
                        lambda *a, **k: touched.append("prepare"))
    monkeypatch.setattr(loop_mod, "_note_nanny_delegate_activity",
                        lambda *a, **k: touched.append("nanny"))
    monkeypatch.setattr(loop_mod, "_arm_delivery_control", lambda *a, **k: touched.append("arm"))
    monkeypatch.setattr(
        loop_mod, "_forced_final_answer",
        lambda ctx_, **kwargs: ("wrapped up", ctx_.accumulated_usage, {}),
    )
    _finish_no_tool_round_budget(_ctx(), None, _ceiling(50.0))
    assert touched == []


def test_below_the_ceiling_and_a_disabled_ceiling_both_continue_unchanged(monkeypatch):
    from ouroboros import loop as loop_mod

    monkeypatch.setattr(loop_mod, "_loop_tree_accounting", lambda **_k: {"settled_usd": 1.0, "accounted_usd": 1.0})
    assert _finish_no_tool_round_budget(_ctx(), None, _ceiling(50.0)) is None
    # An explicitly disabled ceiling is untouched by this unification.
    disabled = task_pacing.resolve_cost_ceiling(
        None, normalize_budget_profile({"cost_hard_stop_pct": 0}), root_cap_usd=50.0,
    )
    monkeypatch.setattr(
        task_pacing, "wrapup_reservation_fits",
        lambda **_k: (_ for _ in ()).throw(AssertionError("armed on a disabled ceiling")),
    )
    assert _finish_no_tool_round_budget(_ctx(), None, disabled) is None
    assert _finish_no_tool_round_budget(_ctx(), None, None) is None


def test_the_current_candidate_survives_the_budget_exit(monkeypatch):
    """A budget stop wraps up the answer the round produced; it does not discard it."""
    from ouroboros import loop as loop_mod

    candidate = SimpleNamespace(full_text="the answer so far")
    ctx = _ctx(delivery_candidate=candidate)
    monkeypatch.setattr(loop_mod, "_loop_tree_accounting", lambda **_k: {"settled_usd": 99.0, "accounted_usd": 99.0})
    monkeypatch.setattr(
        loop_mod, "_forced_final_answer",
        lambda ctx_, **kwargs: ("wrapped up", ctx_.accumulated_usage, {}),
    )
    assert _finish_no_tool_round_budget(ctx, None, _ceiling(50.0)) is not None
    assert ctx.delivery_candidate is candidate


def test_only_the_unfinished_no_tool_branch_arms_the_budget_tail():
    """A READY answer returns from the loop before any budget wrap-up: the flag
    is armed only where the round continues (owner: no extra paid wrap-up for a
    perfectly ordinary final answer)."""
    source = (pathlib.Path(__file__).resolve().parents[1] / "ouroboros" / "loop.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    armings = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pending_no_tool_budget" for t in node.targets)
        and isinstance(node.value, ast.Constant) and node.value.value is True
    ]
    # Two rounds continue without a final answer: an unfinished no-tool round
    # (`if final_result is None:`) and a reply that ended on its output limit before any
    # visible output (the next ordinary round decides; the same money rail bounds it).
    assert len(armings) == 2, "only the two continuing rounds may arm the no-tool budget tail"

    def unfinished(test: ast.AST) -> bool:
        return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "final_result"
                and any(isinstance(op, ast.Is) for op in test.ops))

    def exhausted(test: ast.AST) -> bool:
        return "llm_output_exhausted" in ast.unparse(test)

    for arming in armings:
        guards = [node for node in ast.walk(tree) if isinstance(node, ast.If) and arming in ast.walk(node)
                  and (unfinished(node.test) or exhausted(node.test))]
        assert guards, "an arming must sit in a continuing round: unfinished no-tool, or output exhaustion"


@pytest.mark.serial
def test_preparation_failure_pauses_and_resumes_the_no_tool_tail(full_loop, monkeypatch):
    """The merged acceptance preparation owner survives a real loop/queue pause.

    Resume neither buys a final nor mistakes the saved no-tool answer for a
    tool batch: its preparation incident and opaque identities survive intact.
    """
    from ouroboros import budget_pause, loop, loop_budget, loop_acceptance_review, owner_wait, usage_accounting
    from ouroboros.artifacts import read_actor_source_bytes
    from supervisor.events import _handle_budget_pause
    from tests._budget_pause_exact_helpers import _install_queue, _supervisor_ctx

    f = full_loop
    root, task_id = f.ctx.drive_root, f.ctx.task_id
    q, state, workers = _install_queue(root, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 100.0)
    f.ctx.task_started_at = time.time() - 5
    f.ctx.context_fit_plan = None
    # The shared scripted-model fixture has no immutable ContextFit core;
    # route rebuilding is exercised by the owner-wait/ContextFit suites.
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    f.ctx._cost_ceiling = _ceiling(50.0)
    tree = {"settled_usd": 49.0, "accounted_usd": 49.0, "root_limit_usd": 50.0, "age_sec": 0.0}
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_k: dict(tree))
    monkeypatch.setattr(loop_budget, "_loop_tree_accounting", lambda **_k: dict(tree))
    monkeypatch.setattr(loop_budget, "_wrapup_global_remaining", lambda: 100.0)
    monkeypatch.setattr(usage_accounting, "refresh_root_accounting", lambda *_a, **_k: dict(tree))
    monkeypatch.setattr(loop, "_forced_final_answer", lambda *_a, **_k: pytest.fail("paid budget final"))
    prepared_tool_rounds = []
    original_prepare = loop._prepare_post_tool_budget_context
    def prepare_fresh_completion(*args, **kwargs):
        assert f.model_step == 2, "the saved no-tool continuation armed tool controls"
        assert f.ctx._execution_trace["tool_calls"][-1]["tool"] == "finish_task"
        prepared_tool_rounds.append(f.model_step)
        return original_prepare(*args, **kwargs)
    monkeypatch.setattr(loop, "_prepare_post_tool_budget_context", prepare_fresh_completion)
    builders = []

    def broken(*_a, **_k):
        builders.append(1)
        raise RuntimeError("local preparation unavailable")

    monkeypatch.setattr(loop_acceptance_review, "_build_host_acceptance_evidence", broken)
    saved = {}

    def main(_llm, messages, *_a, **kw):
        from tests.test_acceptance_async_loop import keep

        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        usage = f.ctx._accumulated_usage
        if f.model_step == 2:
            for key in ("incident_id", "source_identity", "attempts", "failure_kind"):
                assert f.ctx._execution_trace["acceptance_preparation"][key] == saved["trace"]["acceptance_preparation"][key]
            assert usage["cost"] == 1.25 and usage["rounds"] == 1
            assert f.ctx._delivery_evidence_fingerprint == saved["delivery"]["_delivery_evidence_fingerprint"]
        assert f.model_step <= 2, "Resume reset the preparation incident or round state"
        usage.update(cost=1.25 * f.model_step, rounds=f.model_step)
        if f.model_step == 2:
            return keep(f), 1.25  # The resumed author selects the retained no-tool answer.
        return {"content": "The complete report includes the requested budget."}, 1.25

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    try:
        with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
            f.run()
        pause = raised.value.pause
        saved.update(json.loads(read_actor_source_bytes(root, task_id, pause["source_ref"])))
        assert pause["rail"] == budget_pause.RAIL_GRACEFUL_CEILING
        assert saved["resume_point"]["budget_tail"] == "no_tool" and saved["round_idx"] == 2
        assert saved["trace"]["acceptance_preparation"]["attempts"] == 1
        assert f.model_step == 1 and builders == [1] and not f.review_sends
        assert prepared_tool_rounds == [] and saved["trace"]["tool_calls"] == []
        task = {"id": task_id, "type": "task", "root_task_id": task_id, "chat_id": 1, "_attempt": 1}
        workers.RUNNING[task_id] = {"task": task, "worker_id": 0, "attempt": 1}
        workers.WORKERS[0] = SimpleNamespace(busy_task_id=task_id)
        _handle_budget_pause({**budget_pause.pause_event(task, pause), "worker_id": 0},
                            _supervisor_ctx(root, workers, q, [], []))
        tree["root_limit_usd"] = 100.0  # owner-authorized headroom, never reset spend
        assert q.resume_budget_paused_task(task_id)["ok"] is True
        f.ctx.budget_pause_resume = copy.deepcopy(workers.PENDING[0]["_budget_pause_resume"])
        f.run_args["messages"] = [{"role": "user", "content": "fresh shell must restore saved cognition"}]
        result, usage, trace = f.run()
        assert result == "The complete report includes the requested budget."
        assert f.model_step == 2 and builders == [1] and not f.review_sends
        assert [row["tool"] for row in trace["tool_calls"]] == ["finish_task"]
        assert usage["rounds"] == 2 and usage["cost"] == 2.5
        assert trace["acceptance_decision"]["status"] == "finalized_unaccepted"
        assert trace["acceptance_decision"]["reason"] == "acceptance_preparation_failed"
        assert budget_pause.budget_pause_row(root, task_id)["state"] == budget_pause.STATE_RESUMED
    finally:
        budget_pause.end_dispatch_fence(task_id)


@pytest.mark.serial
@pytest.mark.parametrize("rail,completed_rounds", [("global", 1), ("soft_land", 1), ("soft_land", 0)])
def test_eligible_monetary_stops_preserve_exact_continuation(tmp_path, monkeypatch, rail, completed_rounds):
    """A first batch is work; a soft threshold also pauses before any work."""
    from ouroboros import budget_pause, loop
    from tests._budget_pause_exact_helpers import _loop_ctx, _running_row

    _running_row(tmp_path, "first-work")
    ctx, limit = _loop_ctx(tmp_path, "first-work")
    limit.round_idx = 1
    limit.accumulated_usage["rounds"] = completed_rounds
    limit.llm_trace = {}
    monkeypatch.setattr(loop, "_forced_final_answer", lambda *_a, **_k: pytest.fail("paid budget final"))
    try:
        with pytest.raises(budget_pause.BudgetPauseRequested):
            if rail == "global":
                loop._check_budget_limits(limit, 0.0)
            else:
                loop._soft_land_exhausted_ceiling(limit, _ceiling(0.01))
        saved = budget_pause.budget_pause_row(tmp_path, ctx.task_id)
        assert saved["exact_continuation"] and saved["resume_point"]["round_idx"] == 1
        assert limit.accumulated_usage["execution_status"] == "paused"
    finally:
        budget_pause.end_dispatch_fence(ctx.task_id)

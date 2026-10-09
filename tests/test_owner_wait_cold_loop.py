"""Cold continuation reaches the real loop with its original route and ceiling."""

import asyncio
from dataclasses import replace
import json
import math
import socket
import threading
from types import SimpleNamespace

import pytest

from ouroboros import context, loop, task_pacing, usage_accounting as accounting
from ouroboros.context_budget import RECLAIM_LOW_WATER_DIVISOR
from ouroboros.contracts.task_contract import normalize_budget_profile
from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait
from ouroboros.task_results import write_task_result
from ouroboros.tools.registry import ToolRegistry
from tests._budget_limits_helpers import _make_args
from tests.test_context_fit_v664 import _plan
from tests.test_loop_compaction import _candidate_request, _failed_capture
from tests.test_loop_transport_wait import _loop_kwargs


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Block provider/catalog calls, leaving asyncio's local IPC available."""
    attempted = []

    def refuse_provider(*args, **kwargs):
        attempted.append((args, kwargs))
        raise AssertionError("cold-loop fixture attempted a provider/catalog call")

    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", refuse_provider)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat_async", refuse_provider)
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", refuse_provider)
    yield
    assert attempted == []


def test_provider_isolation_preserves_event_loop_self_pipe(monkeypatch):
    # Windows uses this TCP fallback for asyncio's internal wakeup channel.
    monkeypatch.setattr(socket, "socketpair",
                        getattr(socket, "_fallback_socketpair", socket.socketpair))
    event_loop = asyncio.new_event_loop()
    event_loop.close()


EXPLICIT = {"budget_profile": {"cost_hard_stop_pct": 50}}  # an authored experiment profile


def _settle(amount, *, provider="openai"):
    """KNOWN spend under the bound scope: the only money a limit decides on (#1487)."""
    held = accounting.reserve_attempt(accounting.AttemptRequest(
        model="fixture", provider=provider, reservation_usd=amount))
    accounting.mark_dispatched(held)
    accounting.settle_attempt(held, {}, cost_usd=amount, cost_final=True)


def cold_registry(tmp_path, monkeypatch, ceiling=None, contract=None):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = "t-wait", 1
    if contract is not None:
        ctx.task_contract = dict(contract)
    ctx.active_model, ctx.active_effort = "same-model", "high"
    ctx.active_use_local, ctx.active_context_mode = False, "max"
    ctx._owner_wait_requested = "quiz"
    ctx._cost_ceiling = ceiling or task_pacing.CostCeiling(state="disabled")
    messages = _plan().messages_for("max") + [
        {"role": "assistant", "tool_calls": [{"id": "saved", "type": "function",
          "function": {"name": "save", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "saved", "content": "Saved object 42"},
    ]
    write_task_result(tmp_path, ctx.task_id, "running")
    wait = checkpoint_owner_wait(ctx, messages, {"tool_calls": []},
                                 {"cost": 10.0, "execution_id": "exec"}, 0, [], set())
    set_owner_wait(tmp_path, ctx.task_id, {**wait, "state": "waiting"})
    ctx.owner_wait_resume = {**wait, "restart_transaction_id": "observed-restart"}
    ctx.owner_wait_callback = lambda *_: None  # the pool grant itself has separate tests
    ctx.context_fit_plan = _plan(window=500_000, known=True)  # startup route A
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_: (
        {"model": task["model"], "provider": "openai", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=500_000, route_fp="route-a"),
    ))
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    return registry


def test_cold_switched_model_keeps_actual_overflow_reclaim_and_retry(tmp_path, monkeypatch):
    registry = cold_registry(tmp_path, monkeypatch)
    sends, reclaimed = [], []

    def reclaim(call, disposition, **kwargs):
        # An actual overflow requests a low-water-sized pass (an eighth of the
        # 500K route), never a token-sized one, before its single strict-shrink retry.
        assert kwargs["minimum_goal_tokens"] == math.ceil(500_000 / RECLAIM_LOW_WATER_DIVISOR)
        reclaimed.append(call.active_model)
        loop._context_reclaim_passes(call.tools._ctx).add(
            (disposition.measurement.route_fp, disposition.measurement.round_id))

    def dispatch(call, disposition, *, candidate_predicate=None, **kwargs):
        assert call.active_model == call.context_fit_plan.model == "same-model"
        assert disposition is not None
        assert any(message.get("content") == "Saved object 42" for message in call.messages)
        if not sends:
            sends.append("overflow")
            call.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        assert candidate_predicate(_candidate_request(disposition, size=800))
        sends.append("smaller retry")
        return {"role": "assistant", "content": "Continued saved work", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_run_main_reclaim", reclaim)
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture())
    result, usage, _ = loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    assert result == "Continued saved work"
    assert sends == ["overflow", "smaller retry"] and reclaimed == ["same-model"]
    assert registry._ctx.active_context_mode == "low"
    assert registry._ctx.owner_wait_resume is None
    assert usage["cost"] == 10.0  # recorded spend is retained, never re-emitted


def test_cold_loop_keeps_original_ceiling_while_live_wallet_still_binds(tmp_path, monkeypatch):
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="t-wait", root_task_id="t-wait",
                                  global_limit_usd=200.0, root_limit_usd=50.0)
    explicit = normalize_budget_profile(EXPLICIT["budget_profile"])
    with accounting.usage_scope(scope):
        original = task_pacing.resolve_cost_ceiling(200.0, explicit, root_cap_usd=50.0)
        assert original.ceiling_usd == 47.0
        _settle(10.0)
        registry = cold_registry(tmp_path, monkeypatch, original, EXPLICIT)
        for index in range(4):
            with accounting.usage_scope(replace(scope, task_id=f"other-{index}", root_task_id=f"other-{index}")):
                _settle(40.0)
        fresh = task_pacing.resolve_cost_ceiling(30.0, explicit, root_cap_usd=50.0)
        assert fresh.ceiling_usd == 15.0
        registry._ctx._cost_ceiling = fresh  # new context-builder disclosure before loading saved messages
        _settle(6.0)
        captured = []
        soft_land = loop._soft_land_exhausted_ceiling
        forced = []

        def record_wrapup(call, *, prompt, fallback_text, reason_code):
            forced.append(reason_code)
            call.accumulated_usage["reason_code"] = reason_code
            return fallback_text, call.accumulated_usage, call.llm_trace

        # The real budget decision may request a final answer; its transport is
        # irrelevant here and must not probe or price the synthetic model.
        monkeypatch.setattr(loop, "_forced_final_answer", record_wrapup)

        def inspect_ceiling(call, ceiling):
            captured.append(ceiling)
            assert ceiling == original
            assert soft_land(call, ceiling) is None
            args = _make_args(task_id="t-wait", accumulated_usage={"cost": 10.0},
                              cost_ceiling=ceiling, budget_remaining_usd=30.0, drive_logs=tmp_path)
            assert loop._check_budget_limits(**args) is None
            assert forced == []
            assert loop._check_budget_limits(**{**args, "cost_ceiling": fresh})[1]["reason_code"] == "budget_exhausted"
            assert forced == ["budget_exhausted"]
            return "ceiling retained", call.accumulated_usage, {}

        monkeypatch.setattr(loop, "_soft_land_exhausted_ceiling", inspect_ceiling)
        result, _, _ = loop.run_llm_loop(**{**_loop_kwargs(tmp_path, registry, []), "budget_remaining_usd": 30.0})
        assert result == "ceiling retained" and captured == [original]
        _settle(24.0)  # the live wallet's known spend reaches its $200
        with pytest.raises(accounting.BudgetExceeded) as global_refusal:
            accounting.reserve_attempt(accounting.AttemptRequest(model="fixture", provider="openai", reservation_usd=0.01))
        assert global_refusal.value.limit_scope == "global"
    with accounting.usage_scope(replace(scope, global_limit_usd=1000.0)):
        _settle(10.0)  # the tree's known spend reaches its $50 cap ($10 + $6 + $24 + $10)
        with pytest.raises(accounting.BudgetExceeded) as root_refusal:
            accounting.reserve_attempt(accounting.AttemptRequest(model="fixture", provider="openai", reservation_usd=0.01))
        assert root_refusal.value.limit_scope == "root"


def test_an_ordinary_cold_restore_drops_a_saved_default_ceiling(tmp_path, monkeypatch):
    """Owner 2026-10-07 for existing installs: an owner wait saved before the upgrade
    carries the removed default ($47 = cap minus margin). With no explicit profile and no
    producer, the planned-restart restore resumes without it and says so."""
    from ouroboros.owner_wait import load_owner_wait

    scope = accounting.UsageScope(drive_root=tmp_path, task_id="t-wait", root_task_id="t-wait",
                                  global_limit_usd=200.0, root_limit_usd=50.0)
    legacy = task_pacing.CostCeiling(state="active", ceiling_usd=47.0, root_cap_usd=50.0,
                                     planning_margin_usd=3.0, basis="min(global_pct, root_cap_minus_margin)")
    with accounting.usage_scope(scope):
        registry = cold_registry(tmp_path, monkeypatch, legacy)
        load_owner_wait(registry._ctx)
    restored = registry._ctx._cost_ceiling
    assert restored.state == task_pacing.COST_CEILING_DISABLED and restored.ceiling_usd is None
    assert restored.basis == "no_default_cost_stop(saved default stop $47.00 removed)"


def test_agent_cold_context_shows_saved_ceiling_and_current_wallet(tmp_path, monkeypatch):
    from ouroboros import agent as agent_module
    from tests._context_shared import _make_health_env

    monkeypatch.setenv("TOTAL_BUDGET", "200")
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "50")
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="t-wait", root_task_id="t-wait",
                                  global_limit_usd=200.0, root_limit_usd=50.0)
    with accounting.usage_scope(scope):
        original = task_pacing.resolve_task_cost_ceiling(SimpleNamespace(task_contract=EXPLICIT), 200.0)
        assert original.ceiling_usd == 47.0
        registry = cold_registry(tmp_path, monkeypatch, original, EXPLICIT)
        handoff = registry._ctx.owner_wait_resume
        _settle(10.0)
        for index in range(4):
            with accounting.usage_scope(replace(scope, task_id=f"other-{index}", root_task_id=f"other-{index}")):
                _settle(40.0)

        env = _make_health_env(tmp_path)
        env.branch_dev, env.budget_drive_root = "ouroboros", tmp_path
        agent = object.__new__(agent_module.OuroborosAgent)
        agent.__dict__.update(
            env=env, tools=registry, memory=None, _event_queue=None,
            _current_chat_id=1, _current_task_type="task", _pending_events=[],
            _task_started_ts=0.0, _owner_message_admission_lock=threading.Lock(),
            owner_wait_callback=lambda *_: None,
        )
        for name in ("_emit_live_log", "_emit_typing_start", "_emit_progress", "_capture_mutation_baseline"):
            monkeypatch.setattr(agent, name, lambda *_a, **_kw: None)
        monkeypatch.setattr(agent, "_run_delegate_preflight", lambda _logs, _task, dispatch: (dispatch, False))

        def runtime_messages(env, task, ctx, **_kwargs):
            # Omit unrelated governance assembly, retain the real Runtime builder
            # and immutable projection that cold-loop rebind sends to the model.
            runtime = context.build_runtime_section(env, task, ctx=ctx)
            plan = _plan(window=500_000, known=True)
            projection = replace(plan.max_projection, system_content_json=json.dumps(runtime))
            ctx.context_fit_plan = replace(plan, max_projection=projection, low_projection=projection)
            return ctx.context_fit_plan.messages_for("max"), {}

        monkeypatch.setattr(agent_module, "build_llm_messages", runtime_messages)
        task = {"id": "t-wait", "root_task_id": "t-wait", "delegation_role": "root",
                "type": "task", "text": "Continue saved work", "_attempt": 1,
                "budget_profile": dict(EXPLICIT["budget_profile"]),
                "_owner_wait_resume": handoff, "budget_drive_root": str(tmp_path)}
        ctx, messages, caps = agent._prepare_task_context(task)
        assert caps["budget_remaining"] == 30.0
        captured = []

        def dispatch(call, _disposition, **_kwargs):
            assert call.messages[0]["role"] == "system"
            shown = json.loads(call.messages[0]["content"].split("\n\n", 1)[1])["budget"]
            captured.append(shown)
            assert ctx._cost_ceiling == original
            assert shown["in_task_cost_ceiling"] == task_pacing.cost_ceiling_disclosure(original)
            assert shown["remaining_usd"] == 30.0 and shown["per_task_tree_cap_usd"] == 50.0
            return {"role": "assistant", "content": "Continued saved work", "tool_calls": []}, 0.0

        monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
        result, _, _ = loop.run_llm_loop(**{
            **_loop_kwargs(tmp_path, registry, []), "messages": messages,
            "budget_remaining_usd": caps["budget_remaining"],
        })
        assert result == "Continued saved work" and len(captured) == 1
        _settle(30.0)  # known spend reaches the $200 wallet
        with pytest.raises(accounting.BudgetExceeded) as refused:
            accounting.reserve_attempt(accounting.AttemptRequest(
                model="fixture", provider="openai", reservation_usd=0.01))
        assert refused.value.limit_scope == "global"

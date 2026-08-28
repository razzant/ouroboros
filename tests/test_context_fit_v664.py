"""Focused v6.64 ordinary-task context-fit regressions."""

from __future__ import annotations

import inspect
import json
import queue
from types import SimpleNamespace

import pytest


def _env_memory(tmp_path):
    from ouroboros.agent import Env
    from ouroboros.memory import Memory

    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    for path in (
        repo / "prompts",
        repo / "docs",
        drive / "state",
        drive / "memory" / "knowledge",
        drive / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)
    (repo / "prompts" / "SYSTEM.md").write_text("stable policy", encoding="utf-8")
    (repo / "BIBLE.md").write_text("# BIBLE\nconstitution", encoding="utf-8")
    (repo / "docs" / "ARCHITECTURE.md").write_text(
        "# Architecture\n\n## One\nFULL_ARCH_SENTINEL\n\n## Two\nbody", encoding="utf-8",
    )
    (repo / "docs" / "DEVELOPMENT.md").write_text("# Development\nfull dev", encoding="utf-8")
    (drive / "state" / "state.json").write_text("{}", encoding="utf-8")
    (drive / "memory" / "identity.md").write_text("identity", encoding="utf-8")
    (drive / "memory" / "scratchpad.md").write_text("scratch", encoding="utf-8")
    env = Env(repo_dir=repo, drive_root=drive)
    return env, Memory(drive_root=drive, repo_dir=repo)


def _evidence(*, status="unprobeable", window=0, stale=False, route_fp="route-a"):
    return SimpleNamespace(
        status=status,
        window_tokens=window,
        stale=stale,
        route_fp=route_fp,
    )


def test_unknown_route_tries_max_and_both_views_share_one_core(tmp_path, monkeypatch):
    import ouroboros.context as context

    env, memory = _env_memory(tmp_path)
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_a, **_kw: (
            {"model": "openai/gpt-test", "provider": "openai", "base_url": "", "use_local": False},
            _evidence(),
        ),
    )

    plan = context.build_context_fit_plan(
        env, memory, {"id": "fit-1", "type": "task", "text": "solve"}, preferred_mode="max",
    )

    assert plan.initial_mode == "max"
    assert plan.max_projection.fits_known_window is None
    max_blocks = plan.messages_for("max")[0]["content"]
    low_blocks = plan.messages_for("low")[0]["content"]
    assert "FULL_ARCH_SENTINEL" in max_blocks[0]["text"]
    assert "FULL_ARCH_SENTINEL" not in low_blocks[0]["text"]
    assert "navigation map" in low_blocks[0]["text"]
    # Stable memory, dynamic evidence and owner intent were captured once.
    assert max_blocks[1:] == low_blocks[1:]
    assert plan.messages_for("max")[1] == plan.messages_for("low")[1]
    assert len(plan.core_sha256) == 64


def test_known_window_uses_family_and_exact_route_calibration(tmp_path, monkeypatch):
    import ouroboros.context as context
    from ouroboros.utils import append_jsonl

    env, memory = _env_memory(tmp_path)
    append_jsonl(
        env.drive_root / "logs" / "events.jsonl",
        {
            "type": "llm_round",
            "context_route_fp": "route-observed",
            "estimated_prompt_tokens": 100,
            "prompt_tokens": 180,
        },
    )
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_a, **_kw: (
            {"model": "openai/gpt-test", "provider": "openai", "base_url": "", "use_local": False},
            _evidence(status="confirmed", window=1_000_000, route_fp="route-observed"),
        ),
    )

    plan = context.build_context_fit_plan(
        env, memory, {"id": "fit-2", "type": "task", "text": "solve"}, preferred_mode="max",
    )
    assert plan.max_projection.calibration_ratio == 1.8
    assert plan.max_projection.calibrated_tokens == int(plan.max_projection.estimated_tokens * 1.8)

    # v6.80.0 (Д7 decoupling): WITHOUT a route sample and WITHOUT a measured density
    # for that model, the main-loop baseline stays the neutral 1.0 — "unknown routes
    # deliberately try Max". The conservative cold-start density used for REVIEW packs
    # is never applied here, because an empty observation store (every fresh install and
    # every isolated benchmark server) would otherwise demote Max to Low silently.
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_a, **_kw: (
            {"model": "anthropic/claude-fable-5", "provider": "anthropic", "base_url": "", "use_local": False},
            _evidence(status="confirmed", window=1_000_000, route_fp="route-claude"),
        ),
    )
    claude = context.build_context_fit_plan(
        env, memory, {"id": "fit-3", "type": "task", "text": "solve"}, preferred_mode="max",
    )
    assert claude.max_projection.calibration_ratio == 1.0
    assert claude.initial_mode == "max"

    # A MEASURED density for that exact model DOES raise the baseline.
    from ouroboros.capability_evidence import _DENSITY_MEMO, record_token_density

    _DENSITY_MEMO.clear()
    record_token_density(
        env.drive_root, "anthropic/claude-fable-5",
        prompt_chars=4_000_000, prompt_tokens=1_650_000,
    )
    measured = context.build_context_fit_plan(
        env, memory, {"id": "fit-4", "type": "task", "text": "solve"}, preferred_mode="max",
    )
    assert measured.max_projection.calibration_ratio >= 1.65


def test_positive_no_fit_evidence_selects_low_before_dispatch(tmp_path, monkeypatch):
    import ouroboros.context as context

    env, memory = _env_memory(tmp_path)
    (env.repo_dir / "docs" / "ARCHITECTURE.md").write_text(
        "# Architecture\n\n## Huge\n" + ("dense code tokens\n" * 30_000), encoding="utf-8",
    )
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_a, **_kw: (
            {"model": "openai/gpt-test", "provider": "openai", "base_url": "", "use_local": False},
            _evidence(status="confirmed", window=100_000),
        ),
    )

    plan = context.build_context_fit_plan(
        env, memory, {"id": "fit-4", "type": "task", "text": "solve"}, preferred_mode="max",
    )
    assert plan.max_projection.fits_known_window is False
    assert plan.low_projection.fits_known_window is True
    assert plan.initial_mode == "low"


def test_tool_schemas_join_known_route_projection_before_dispatch(tmp_path, monkeypatch):
    import ouroboros.context as context

    env, memory = _env_memory(tmp_path)
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_a, **_kw: (
            {"model": "openai/gpt-test", "provider": "openai", "base_url": "", "use_local": False},
            _evidence(status="confirmed", window=100_000),
        ),
    )
    plan = context.build_context_fit_plan(
        env, memory, {"id": "fit-tools", "type": "task", "text": "solve"}, preferred_mode="max",
    )
    assert plan.initial_mode == "max"
    giant_tools = [{
        "type": "function",
        "function": {
            "name": "large_schema",
            "description": "x" * 200_000,
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    assert plan.initial_mode_with_tools(giant_tools) == "low"
    assert plan.projected_tokens_with_tools("max", giant_tools) > plan.max_projection.calibrated_tokens

    # A stale retained window is UNKNOWN and must not manufacture a Low decision.
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_a, **_kw: (
            {"model": "openai/gpt-test", "provider": "openai", "base_url": "", "use_local": False},
            _evidence(status="confirmed", window=100_000, stale=True),
        ),
    )
    stale = context.build_context_fit_plan(
        env, memory, {"id": "fit-tools-stale", "type": "task", "text": "solve"}, preferred_mode="max",
    )
    assert stale.initial_mode_with_tools(giant_tools) == "max"


def test_confirmed_overflow_gets_one_same_model_low_retry(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.tools.registry import ToolRegistry

    class _Plan:
        preferred_mode = "max"
        initial_mode = "max"
        model = "same-model"
        route_fp = "route-a"
        core_sha256 = "a" * 64
        window_tokens = 0

        @staticmethod
        def reproject_transcript(messages, mode):
            assert mode == "low"
            rebuilt = list(messages)
            rebuilt[0] = {"role": "system", "content": "LOW_PROJECTION"}
            return rebuilt

    class _LLM:
        @staticmethod
        def default_model():
            return "same-model"

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.context_fit_plan = _Plan()
    calls = []

    def fake_call(
        _llm, messages, model, _tools, _effort, _max_retries, _logs,
        _task_id, _round_idx, _events, accumulated_usage, _task_type="", **kwargs,
    ):
        calls.append({"model": model, "messages": json.loads(json.dumps(messages)), **kwargs})
        if len(calls) == 1:
            accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            accumulated_usage["context_overflow_suggest_low"] = True
            return None, 0.0
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop, "_persist_compaction_checkpoint", lambda *_a, **_kw: True)
    monkeypatch.setattr(loop, "seal_task_transcript", lambda *_a, **_kw: None)
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")

    result, usage, _trace = loop.run_llm_loop(
        messages=[{"role": "system", "content": "MAX_PROJECTION"}, {"role": "user", "content": "go"}],
        tools=registry,
        llm=_LLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_type="task",
        task_id="fit-retry",
        drive_root=tmp_path,
    )

    assert result == "fits"
    assert len(calls) == 2
    assert calls[0]["model"] == calls[1]["model"] == "same-model"
    assert calls[1]["attempt_cap"] == 1
    assert calls[1]["messages"][0]["content"] == "LOW_PROJECTION"
    assert usage["_context_fit_low_retry_used"] is True
    assert __import__("os").environ["OUROBOROS_CONTEXT_MODE"] == "max"


def test_low_retry_preserves_max_dispatch_count_when_low_is_budget_blocked(
    tmp_path, monkeypatch,
):
    import ouroboros.loop as loop
    from ouroboros.usage_accounting import BudgetExceeded

    messages = [{"role": "user", "content": "solve"}]
    plan = SimpleNamespace(
        model="primary-model",
        preferred_mode="max",
        route_fp="route-primary",
        core_sha256="core-primary",
        reproject_transcript=lambda current, _mode: list(current),
    )
    tool_ctx = SimpleNamespace(
        task_metadata={},
        task_contract={},
        messages=messages,
        active_context_mode="max",
    )
    tools = SimpleNamespace(_ctx=tool_ctx)
    calls = 0

    def fake_call(
        _llm, _messages, _model, _tools, _effort, _max_retries, _logs,
        _task_id, _round_idx, _events, accumulated_usage, _task_type="", **_kwargs,
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            accumulated_usage["_llm_dispatched_attempts"] = 2
            accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        accumulated_usage["_llm_dispatched_attempts"] = 0
        raise BudgetExceeded("Low retry blocked before dispatch")

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop, "_persist_compaction_checkpoint", lambda *_a, **_kw: True)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: None)
    usage = {}

    with pytest.raises(BudgetExceeded):
        loop._call_round_model(loop._RoundModelCallContext(
            llm=SimpleNamespace(), messages=messages, tools=tools,
            context_fit_plan=plan, active_model="primary-model", tool_schemas=[],
            active_effort="high", max_retries=3, drive_logs=tmp_path / "logs",
            task_id="budget-blocked-low", round_idx=1, event_queue=None,
            accumulated_usage=usage, task_type="task", active_use_local=False,
            active_context_mode="max", drive_root=tmp_path,
        ))

    assert calls == 2
    assert usage["_llm_dispatched_attempts"] == 2


def test_low_retry_precedes_fallback_and_accepted_route_rebinds(tmp_path, monkeypatch):
    from ouroboros import config, context, fallback_cooldown, loop
    from ouroboros.tools.registry import ToolRegistry

    env, memory = _env_memory(tmp_path)
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_args, **_kwargs: (
            {
                "model": "primary-model",
                "provider": "openai",
                "base_url": "",
                "use_local": False,
            },
            _evidence(status="unprobeable", window=0, route_fp="route-primary"),
        ),
    )
    plan = context.build_context_fit_plan(
        env,
        memory,
        {"id": "fallback-fit", "type": "task", "text": "solve"},
        preferred_mode="max",
    )
    messages = plan.messages_for("max")
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    registry._ctx.context_fit_plan = plan
    registry._ctx.task_id = "fallback-fit"
    registry._ctx.event_queue = None
    calls = []

    def fake_call(
        _llm, sent_messages, model, _tools, _effort, _max_retries, _logs,
        _task_id, _round_idx, _events, accumulated_usage, _task_type="", **kwargs,
    ):
        calls.append({
            "model": model,
            "system": json.loads(json.dumps(sent_messages[0])),
            "route_fp": accumulated_usage.get("_context_route_fp"),
            **kwargs,
        })
        if len(calls) == 1:
            accumulated_usage["_llm_dispatched_attempts"] = 2
            accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        if len(calls) == 2:
            accumulated_usage["_llm_dispatched_attempts"] = 1
            accumulated_usage["_last_llm_error_kind"] = "provider_transient"
            return None, 0.0
        accumulated_usage["_llm_dispatched_attempts"] = 1
        return {"role": "assistant", "content": "fallback fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop, "_persist_compaction_checkpoint", lambda *_a, **_kw: True)
    accumulated_usage = {}
    msg, _cost, mode = loop._call_round_model(
        loop._RoundModelCallContext(
            llm=SimpleNamespace(),
            messages=messages,
            tools=registry,
            context_fit_plan=plan,
            active_model="primary-model",
            tool_schemas=[],
            active_effort="high",
            max_retries=3,
            drive_logs=env.drive_root / "logs",
            task_id="fallback-fit",
            round_idx=1,
            event_queue=None,
            accumulated_usage=accumulated_usage,
            task_type="task",
            active_use_local=False,
            active_context_mode="max",
            drive_root=env.drive_root,
        )
    )
    assert msg is None
    assert mode == "low"
    assert [call["model"] for call in calls] == ["primary-model", "primary-model"]
    assert calls[1]["attempt_cap"] == 1
    assert calls[1]["system"] == plan.low_projection.system_message()
    assert accumulated_usage["_llm_dispatched_attempts"] == 3

    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_args, **_kwargs: (
            {
                "model": "fallback-model",
                "provider": "openrouter",
                "base_url": "",
                "use_local": False,
            },
            _evidence(status="confirmed", window=1_000_000, route_fp="route-fallback"),
        ),
    )
    monkeypatch.setattr(config, "get_fallback_models", lambda _model: ["fallback-model"])
    monkeypatch.setattr(fallback_cooldown, "attempts_per_model", lambda: 1)
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *_a: None)
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)

    msg, model, use_local, rebound, mode = loop._run_cross_model_fallback_chain(
        llm=SimpleNamespace(),
        ctx=registry._ctx,
        tools=registry,
        messages=messages,
        active_model="primary-model",
        active_use_local=False,
        tool_schemas=[],
        active_effort="high",
        max_retries=3,
        drive_logs=env.drive_root / "logs",
        task_id="fallback-fit",
        round_idx=1,
        event_queue=None,
        accumulated_usage=accumulated_usage,
        task_type="task",
        emit_progress=lambda _text: None,
        context_fit_plan=plan,
        active_context_mode=mode,
    )

    assert msg["content"] == "fallback fits"
    assert [call["model"] for call in calls] == [
        "primary-model", "primary-model", "fallback-model",
    ]
    # The fallback is exact-route rebound before its first physical dispatch;
    # it never inherits the failed primary fingerprint/projection.
    assert calls[2]["route_fp"] == "route-fallback"
    assert calls[2]["system"] == rebound.max_projection.system_message()
    assert model == "fallback-model"
    assert use_local is False
    assert rebound.core_sha256 == plan.core_sha256
    assert rebound.route_fp == "route-fallback"
    assert rebound.model == "fallback-model"
    assert mode == "max"
    assert registry._ctx.context_fit_plan is rebound
    assert registry._ctx.active_context_mode == "max"
    assert messages[0] == rebound.max_projection.system_message()
    assert accumulated_usage["_context_route_fp"] == "route-fallback"
    assert accumulated_usage["_context_fit_mode"] == "max"


def test_unknown_fallback_overflow_gets_one_same_model_low_retry(tmp_path, monkeypatch):
    from ouroboros import config, context, fallback_cooldown, loop
    from ouroboros.tools.registry import ToolRegistry

    env, memory = _env_memory(tmp_path)
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_args, **_kwargs: (
            {"model": "primary", "provider": "openai", "base_url": "", "use_local": False},
            _evidence(status="unprobeable", window=0, route_fp="route-primary"),
        ),
    )
    plan = context.build_context_fit_plan(
        env, memory, {"id": "fallback-low", "type": "task", "text": "solve"},
        preferred_mode="max",
    )
    messages = plan.messages_for("max")
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    registry._ctx.context_fit_plan = plan
    registry._ctx.task_id = "fallback-low"
    registry._ctx.event_queue = None
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_args, **_kwargs: (
            {"model": "fallback", "provider": "openrouter", "base_url": "", "use_local": False},
            _evidence(status="unprobeable", window=0, route_fp="route-fallback"),
        ),
    )
    monkeypatch.setattr(config, "get_fallback_models", lambda _model: ["fallback"])
    monkeypatch.setattr(fallback_cooldown, "attempts_per_model", lambda: 1)
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *_a: None)
    monkeypatch.setattr(loop, "_persist_compaction_checkpoint", lambda *_a, **_kw: True)
    calls = []

    def fake_call(
        _llm, sent_messages, model, _tools, _effort, _max_retries, _logs,
        _task_id, _round_idx, _events, usage, _task_type="", **kwargs,
    ):
        calls.append((model, json.loads(json.dumps(sent_messages[0])), kwargs.get("attempt_cap")))
        if len(calls) == 1:
            usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        return {"role": "assistant", "content": "fits low", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    usage = {"_last_llm_error_kind": "provider_transient"}

    msg, model, _local, rebound, mode = loop._run_cross_model_fallback_chain(
        llm=SimpleNamespace(), ctx=registry._ctx, tools=registry, messages=messages,
        active_model="primary", active_use_local=False, tool_schemas=[], active_effort="high",
        max_retries=3, drive_logs=env.drive_root / "logs", task_id="fallback-low", round_idx=1,
        event_queue=None, accumulated_usage=usage, task_type="task",
        emit_progress=lambda _text: None, context_fit_plan=plan, active_context_mode="max",
    )

    assert msg["content"] == "fits low"
    assert model == "fallback"
    assert [row[0] for row in calls] == ["fallback", "fallback"]
    assert calls[0][1] == rebound.max_projection.system_message()
    assert calls[0][2] == 1 and calls[1][2] == 1
    assert calls[1][1] == rebound.low_projection.system_message()
    assert mode == "low"
    assert usage["_context_fit_low_retry_used"] is True


def test_route_switch_rebinds_same_core_to_new_exact_route(tmp_path, monkeypatch):
    from ouroboros import context, loop
    from ouroboros.tools.registry import ToolRegistry

    env, memory = _env_memory(tmp_path)
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_args, **_kwargs: (
            {"model": "initial-model", "provider": "openai", "base_url": "", "use_local": False},
            _evidence(status="confirmed", window=1_000_000, route_fp="route-initial"),
        ),
    )
    plan = context.build_context_fit_plan(
        env, memory, {"id": "switch", "type": "task", "text": "solve"},
        preferred_mode="max",
    )
    core_sha = plan.core_sha256
    messages = plan.messages_for("max")
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    registry._ctx.task_id = "switch"
    registry._ctx.event_queue = None
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_args, **_kwargs: (
            {"model": "new-model", "provider": "openrouter", "base_url": "", "use_local": False},
            _evidence(status="unprobeable", window=0, route_fp="route-new"),
        ),
    )

    rebound, mode = loop._rebind_context_fit_plan(
        plan,
        registry,
        messages,
        model="new-model",
        use_local=False,
        preferred_mode="max",
        tool_schemas=[],
    )

    assert rebound.core_sha256 == core_sha
    assert rebound.model == "new-model"
    assert rebound.route_fp == "route-new"
    assert rebound.evidence_status == "unprobeable"
    assert mode == "max"
    assert messages[0] == rebound.max_projection.system_message()
    assert registry._ctx.context_fit_plan is rebound

    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_args, **_kwargs: (
            {"model": "small-model", "provider": "openrouter", "base_url": "", "use_local": False},
            _evidence(status="confirmed", window=100_000, route_fp="route-small"),
        ),
    )
    # The immutable core itself fits, but accumulated dialogue must also join the
    # exact-route projection before dispatch.
    messages.append({"role": "user", "content": "x" * 300_000})
    small, small_mode = loop._rebind_context_fit_plan(
        rebound,
        registry,
        messages,
        model="small-model",
        use_local=False,
        preferred_mode="max",
        tool_schemas=[],
    )
    assert small.core_sha256 == core_sha
    assert small.route_fp == "route-small"
    assert small.max_projection.fits_known_window is True
    assert small_mode == "low"
    assert messages[0] == small.low_projection.system_message()


def test_route_switch_without_immutable_core_fails_loudly(tmp_path):
    from ouroboros import loop
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    with pytest.raises(RuntimeError, match="CONTEXT_FIT_REBUILD_FAILED"):
        loop._rebind_context_fit_plan(
            None,
            registry,
            [{"role": "user", "content": "go"}],
            model="new-model",
            use_local=False,
            preferred_mode="max",
            tool_schemas=[],
        )


def test_p3_commit_and_scope_review_do_not_use_task_context_fit():
    """P3 stays on its one-pass ReviewCoordinator path, outside run_llm_loop."""
    from ouroboros import review_substrate
    from ouroboros.tools import review, scope_review

    substrate_source = inspect.getsource(review_substrate.ReviewCoordinator._run_slot)
    commit_source = inspect.getsource(review._run_unified_review)
    scope_source = inspect.getsource(scope_review._call_scope_llm)
    assert "run_llm_loop" not in substrate_source + commit_source + scope_source
    assert "ContextFitPlan" not in substrate_source + commit_source + scope_source
    assert "run_review_request" in inspect.getsource(review)
    assert "run_review_request" in scope_source

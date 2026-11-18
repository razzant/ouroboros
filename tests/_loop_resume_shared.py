"""Shared loop invocation, context-fit plan, and restore-ready registry for tests."""
from __future__ import annotations

import json
import queue


class _FakeLoopLLM:
    def default_model(self):
        return "test-model"


def _loop_kwargs(tmp_path, registry, notes, llm=None):
    return dict(
        messages=[{"role": "user", "content": "go"}],
        tools=registry,
        llm=llm or _FakeLoopLLM(),
        drive_logs=tmp_path,
        # emit_progress honors the ``incident=`` keyword (OuroborosAgent._emit_progress).
        emit_progress=lambda text, *, incident=None: notes.append(text),
        incoming_messages=queue.Queue(),
        task_id="t-wait",
        drive_root=tmp_path,
    )


def _projection(mode: str):
    from ouroboros.context_fit import ContextFitProjection

    return ContextFitProjection(
        mode=mode,
        system_content_json=json.dumps(f"{mode.upper()}_SYSTEM"),
        estimated_tokens=10,
        calibrated_tokens=10,
        calibration_ratio=1.0,
    )


def _plan(*, preferred: str = "max", window: int = 0, known: bool = False):
    from ouroboros.context_fit import ContextFitPlan

    return ContextFitPlan(
        core_sha256="a" * 64,
        preferred_mode=preferred,
        initial_mode=preferred,
        model="openai/test-model",
        provider="openai",
        route_fp="route-a",
        status="confirmed" if known else "unprobeable",
        stale=False,
        window_tokens=window,
        output_reserve_tokens=65_536,
        user_content_json=json.dumps("go"),
        max_projection=_projection("max"),
        low_projection=_projection("low"),
    )


def _registry(tmp_path, monkeypatch, task_id, attempt):
    from ouroboros import context, task_pacing
    from ouroboros.tools.registry import ToolRegistry
    from types import SimpleNamespace

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = task_id, attempt
    ctx.active_model, ctx.active_effort = "same-model", "high"
    ctx.active_use_local, ctx.active_context_mode = False, "max"
    ctx._cost_ceiling = task_pacing.CostCeiling(state="disabled")
    ctx.owner_wait_callback = lambda *_: None
    ctx.context_fit_plan = _plan(window=500_000, known=True)
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_: (
        {"model": task["model"], "provider": "openai", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=500_000, route_fp="route-a"),
    ))
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    return registry

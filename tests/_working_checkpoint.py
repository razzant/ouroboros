"""Build a fresh tool registry for each attempt of a cold checkpoint restore."""
from types import SimpleNamespace


def checkpoint_registry(tmp_path, monkeypatch, task_id, attempt):
    from ouroboros import context, task_pacing
    from ouroboros.tools.registry import ToolRegistry
    from tests._context_fit_plan import fit_plan

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = task_id, attempt
    ctx.active_model, ctx.active_effort = "same-model", "high"
    ctx.active_use_local, ctx.active_context_mode = False, "max"
    ctx._cost_ceiling = task_pacing.CostCeiling(state="disabled")
    ctx.owner_wait_callback = lambda *_: None
    ctx.context_fit_plan = fit_plan(window=500_000, known=True)
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_: (
        {"model": task["model"], "provider": "openai", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=500_000, route_fp="route-a"),
    ))
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    return registry

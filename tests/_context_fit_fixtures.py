"""Synthetic Main view, route rebind and refusal context for fit consumers."""
import dataclasses
import json
from types import SimpleNamespace

from ouroboros.capability_evidence import utc_now_iso
from ouroboros.memory_inventory import VIEW_TRACE_KEY

_TASK = {"type": "task", "text": "hi", "id": "tmain", "chat_id": 1}


def _built(tmp_path, monkeypatch, *, books=80_000, ratio=None, page_chars=0):
    """``(core, plan(window, **kw))`` on the shared installation; Max's books grown by ``books`` characters
    so that Max's fixed part exceeds Low's (the test repository's own books are smaller than its nav map)."""
    from types import SimpleNamespace

    from ouroboros import context, context_fit
    from tests._memory_view_context import world

    env, memory, _rooms = world(tmp_path)
    if page_chars:
        from ouroboros.chronicle_store import ChronicleStore
        from tests._chronicle_pages import _page

        ChronicleStore(memory.drive_root).ensure_activated()
        _page(memory.drive_root, "1", 10, 11, text="Story. " * (page_chars // 7))
    core = context._capture_context_core(env, memory, dict(_TASK), None, None)
    core = dataclasses.replace(core, architecture_md=core.architecture_md + "\n\n" + "word " * (books // 5))
    if ratio is not None:
        monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: ratio)

    def plan(window, *, status="asserted", preferred="max", tool_schemas=None):
        evidence = SimpleNamespace(route_fp="r", status=status, stale=status != "asserted", window_tokens=window)
        return context._build_context_fit_plan(env, core, dict(_TASK), preferred_mode=preferred, tool_schemas=tool_schemas,
                                               route_resolver=lambda *_a, **_kw: ({"model": "m", "provider": "p"}, evidence))
    return core, plan


def _rebind(monkeypatch, root, plan, messages, *, window, ratio=1.0, status="asserted", usage=None, facts=None,
            holds_reply=False, start_mode="low"):
    import ouroboros.context
    from ouroboros import context_fit
    from ouroboros.loop_model_call import _rebind_context_fit_plan
    from ouroboros.tools.registry import ToolRegistry

    root.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry._ctx._accumulated_usage = usage
    # A declared window's RouteFacts (its units). ``holds_reply``: a total known to hold the reply too (a local
    # server's one buffer), read less the whole reserve; any other stated total is read less the reply floor.
    evidence = SimpleNamespace(route_fp=f"r-{window}", status=status, stale=status != "asserted", window_tokens=window,
                               route_facts=facts or {}, source="local_health" if holds_reply else "metadata",
                               ts=utc_now_iso())  # a serving statement is dated, as its read is
    monkeypatch.setattr(ouroboros.context, "_context_fit_route", lambda *_a, **_kw: ({"model": "m2", "provider": "p"}, evidence))
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: ratio)
    rebound, mode = _rebind_context_fit_plan(plan, registry, messages, model="m2", use_local=False, start_mode=start_mode,
                                             tool_schemas=[])
    events = [json.loads(line) for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return rebound, mode, [e for e in events if e.get("checkpoint_kind") == "context_fit_route_rebound"][-1]


def _refusal_ctx(tmp_path, plan):
    from ouroboros import memory_floor as mf

    shown = mf.trace_facts(plan.projection("max").memory_facts)  # as the task's start records the view it sends
    tool_ctx = SimpleNamespace(context_fit_plan=plan, messages=None, memory_view_facts=shown,
                               _execution_trace={VIEW_TRACE_KEY: dict(shown)})
    ctx = SimpleNamespace(context_fit_plan=plan, active_context_mode="max", messages=plan.messages_for("max"),
                          tool_schemas=[], tools=SimpleNamespace(_ctx=tool_ctx), task_id="t", round_idx=1,
                          event_queue=None, drive_logs=tmp_path)
    tool_ctx.messages = ctx.messages
    return ctx

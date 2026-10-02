"""Owner context intent and an adopted view survive actual route changes."""
from __future__ import annotations

from dataclasses import replace
import json
import queue
from types import SimpleNamespace

import pytest

from tests.test_context_fit_v664 import _plan, _projection


@pytest.fixture
def route_tools(monkeypatch, tmp_path):
    from ouroboros import capability_evidence, context, context_fit

    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_: (
        {"model": task["model"], "provider": "openai", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=400_000,
                        route_fp="new-route"),
    ))
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_: 1.0)
    monkeypatch.setattr(capability_evidence, "resolve_main_token_density", lambda *_: (1.0, "cold_estimate"))
    inner = SimpleNamespace(task_metadata={}, task_id="context-route", event_queue=None,
                            drive_logs=lambda: tmp_path / "logs", active_context_mode="max",
                            active_model="openai/test-model", active_use_local=False)
    return SimpleNamespace(_ctx=inner)


@pytest.mark.parametrize("owner,active,requested,expected", [
    ("max", "max", "max", "max"),
    ("max", "low", "low", "low"),  # observed account rotation after Max overflow
    ("max", "low", "max", "low"),  # cold restore still supplies owner Max
    ("low", "low", "max", "low"),
    ("nano", "nano", "max", "nano"),
])
def test_rebind_keeps_owner_intent_and_never_widens(
    route_tools, owner, active, requested, expected,
):
    from ouroboros.loop_model_call import _main_context_profile, _rebind_context_fit_plan

    plan = replace(_plan(preferred=owner), nano_projection=_projection("nano"))
    route_tools._ctx.active_context_mode = active
    messages = plan.messages_for(active) + [{"role": "tool", "tool_call_id": "call", "content": "exact result"}]
    tail = messages[1:]
    rebound, mode = _rebind_context_fit_plan(
        plan, route_tools, messages, model="openai/new", use_local=False,
        preferred_mode=requested, tool_schemas=[],
    )
    assert rebound.preferred_mode == owner
    assert mode == rebound.rendered_mode == expected
    assert messages[0] == rebound.projection(expected).system_message()
    assert messages[1:] == tail
    assert rebound.window_tokens == 400_000
    if owner == "max" and expected == "low":
        assert _main_context_profile(rebound, mode) == "task_local_low"


def test_overflow_view_stays_on_plan_without_mutable_context_state(route_tools, tmp_path):
    from ouroboros.loop_model_call import _rebind_context_fit_plan, _reproject_actual_overflow_low

    plan = _plan()
    ctx = SimpleNamespace(active_context_mode="max", context_fit_plan=plan,
                          messages=plan.messages_for("max"), task_id="context-route", tools=route_tools,
                          event_queue=None, drive_logs=tmp_path / "logs", round_idx=2)
    _reproject_actual_overflow_low(ctx)
    assert ctx.context_fit_plan.preferred_mode == "max"
    assert ctx.context_fit_plan.rendered_mode == "low"
    assert route_tools._ctx.context_fit_plan is ctx.context_fit_plan
    # A fresh holder still takes the adopted view from the captured plan.
    route_tools._ctx.active_context_mode = "max"
    rebound, mode = _rebind_context_fit_plan(
        ctx.context_fit_plan, route_tools, ctx.messages, model="openai/new",
        use_local=False, preferred_mode="max", tool_schemas=[],
    )
    assert mode == "low" and rebound.preferred_mode == "max"


def test_cold_rebind_uses_restored_effective_mode(route_tools, monkeypatch):
    from ouroboros import loop, owner_wait

    plan = _plan()
    route_tools._ctx.context_fit_plan = plan
    route_tools._ctx.active_context_mode = "low"
    monkeypatch.setattr(loop, "get_context_mode", lambda: "max")
    messages = plan.messages_for("low")
    rebound, mode = owner_wait.rebind_restored_route(route_tools, {"tool_schemas": []}, messages)
    assert rebound.preferred_mode == "max" and mode == "low"
    assert messages[0] == rebound.low_projection.system_message()


def test_actual_overflow_does_not_widen_nano(route_tools, tmp_path):
    from ouroboros.loop_model_call import _reproject_actual_overflow_low

    plan = replace(_plan(preferred="nano"), nano_projection=_projection("nano"))
    messages = plan.messages_for("nano")
    ctx = SimpleNamespace(active_context_mode="nano", context_fit_plan=plan,
                          messages=messages, task_id="context-route", tools=route_tools,
                          event_queue=None, drive_logs=tmp_path / "logs", round_idx=2)
    _reproject_actual_overflow_low(ctx)
    assert ctx.active_context_mode == "nano" and ctx.context_fit_plan is plan
    assert messages == plan.messages_for("nano")


def test_route_rebind_renders_the_frozen_chronicle_again_with_distinct_mode_budgets(route_tools, monkeypatch, tmp_path):
    from ouroboros import chronicle_view, context_fit

    calls = []

    def render(blocks, snapshot_json, **facts):
        calls.append((snapshot_json, facts))
        assert blocks[-1]["text"] == "\n[CHRONICLE_VIEW]\n"
        return [{"type": "text", "text": f"{snapshot_json}:{facts['mode']}:{facts['window_tokens']}"}]

    monkeypatch.setattr(chronicle_view, "render_system_view", render)
    monkeypatch.setattr(context_fit, "_render_context_system_content", lambda *_a, **_k: [
        {"type": "text", "text": "\n[CHRONICLE_VIEW]\n"}])
    core = context_fit.ContextCore("system", "bible", "architecture", "development", "identity", "dynamic",
                                  json.dumps("task"), False, chronicle_state_json='{"frozen":1}')
    route = lambda *_a, **_k: ({"model": "openai/test-model", "provider": "openai"},
        SimpleNamespace(status="confirmed", stale=False, route_fp="start", window_tokens=900_000))
    plan = context_fit.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core,
        {"id": "context-route", "type": "task"}, preferred_mode="max", route_resolver=route)
    assert [facts["mode"] for _, facts in calls] == ["max", "low", "nano"]
    assert all(facts["task"]["context_user_tokens"] > 0 for _, facts in calls)
    assert plan.low_projection.system_content_json != plan.nano_projection.system_content_json
    changed = context_fit.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path),
        replace(core, chronicle_state_json='{"frozen":2}'), {"id": "context-route", "type": "task"},
        preferred_mode="max", route_resolver=route)
    assert changed.core_sha256 != plan.core_sha256
    calls.clear()
    messages = plan.messages_for("max")
    from ouroboros.loop_model_call import _rebind_context_fit_plan

    rebound, _ = _rebind_context_fit_plan(plan, route_tools, messages, model="openai/new",
                                        use_local=False, preferred_mode="max", tool_schemas=[])
    assert len(calls) == 3
    assert all(snapshot == '{"frozen":1}' and facts["window_tokens"] == 400_000 for snapshot, facts in calls)
    assert rebound.chronicle_state_json == plan.chronicle_state_json
    assert rebound.system_templates_json == plan.system_templates_json


@pytest.mark.parametrize("exact,unknown,error,allowed", [
    (False, False, "context_overflow", True),
    (True, False, "context_overflow", False),
    (False, True, "context_overflow", False),
    (False, False, "provider_outcome_unknown", False),
    (False, False, "deadline_exhausted", False),
    (False, False, "bad_request", True),
])
def test_overflow_fallback_preserves_exact_route_and_unresolved_fences(exact, unknown, error, allowed):
    from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY
    from ouroboros.loop_transport import fallback_chain_allowed

    usage = {TRANSPORT_DEATHS_KEY: {"count": 1}} if unknown else {}
    assert fallback_chain_allowed(SimpleNamespace(exact_model_route=exact), error, None, usage) is allowed


def test_overflow_fallback_dispatches_rebound_view_before_adoption(route_tools, monkeypatch, tmp_path):
    from ouroboros import fallback_cooldown, loop
    from ouroboros.loop_model_call import _run_cross_model_fallback_chain

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai/backup")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "false")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    plan = replace(_plan(), rendered_mode="low")
    route_tools._ctx.context_fit_plan = plan
    route_tools._ctx.active_context_mode = "low"
    messages = plan.messages_for("low") + [{"role": "user", "content": "preserve this work"}]
    seen = []

    def dispatch(ctx, fit, **_):
        seen.append((ctx.active_model, ctx.active_context_mode, ctx.context_fit_plan, fit.measurement))
        assert ctx.messages[0] == ctx.context_fit_plan.low_projection.system_message()
        assert ctx.messages[-1]["content"] == "preserve this work"
        return {"role": "assistant", "content": "continued"}, 0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    result = _run_cross_model_fallback_chain(
        llm=None, ctx=route_tools._ctx, tools=route_tools, messages=messages,
        active_model=plan.model, active_use_local=False, tool_schemas=[], active_effort="high",
        max_retries=1, drive_logs=tmp_path / "logs", task_id="context-route", round_idx=2,
        event_queue=None, accumulated_usage={"_last_llm_error_kind": "context_overflow"},
        task_type="task", emit_progress=lambda *_a, **_k: None,
        context_fit_plan=plan, active_context_mode="low",
    )
    assert result[0]["content"] == "continued"
    assert result[1] == "openai/backup" and result[4] == "low"
    assert len(seen) == 1
    assert seen[0][2].route_fp == "new-route"
    assert seen[0][2].preferred_mode == "max"
    assert seen[0][3].profile == "task_local_low" and seen[0][3].target_total_tokens is None


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("data,kind", [
    ({"type": "context_overflow_retry_skipped", "reason": "unchanged"}, "context_overflow_retry_skipped"),
    ({"type": "context_reclaim", "checkpoint_kind": "context_reclaim_automatic"}, "context_reclaim_automatic"),
    ({"checkpoint_kind": "owner_wait"}, "owner_wait"),
])
def test_checkpoint_keeps_registered_type_and_subtype(tmp_path, queued, data, kind):
    from ouroboros.loop_messages import _emit_checkpoint_event

    events = queue.Queue() if queued else None
    _emit_checkpoint_event(events, "context-route", tmp_path, data)
    if queued:
        from supervisor import events_worker_reports
        from ouroboros.utils import append_jsonl

        supervisor = SimpleNamespace(
            DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], TASKS={},
            bridge=SimpleNamespace(push_log=lambda _: None), append_jsonl=append_jsonl,
        )
        events_worker_reports._handle_log_event(events.get_nowait(), supervisor)
        source = tmp_path / "logs" / "events.jsonl"
    else:
        source = tmp_path / "events.jsonl"
    row = json.loads(source.read_text(encoding="utf-8"))
    assert row["type"] == "task_checkpoint" and row["task_id"] == "context-route"
    assert row["checkpoint_kind"] == kind

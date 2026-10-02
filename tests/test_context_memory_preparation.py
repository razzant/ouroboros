"""Prepared first Main memory and attempt-bound accounting use the real source paths."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence, consolidator, loop_memory, loop_model_call, loop, usage_accounting as ua
from ouroboros.chronicle_view import CHRONICLE_MARKER, MEMORY_BEGIN, MEMORY_END, MEMORY_FACTS_PREFIX
from ouroboros.llm_attempt import memory_view_measurement
from tests.test_chronicle_consolidation import setup, Helper
from tests.test_context_fit_v664 import _plan
from tests.test_loop_compaction import _ctx
from tests.test_processing_transport import transport  # noqa: F401
from tests.test_main_authored_context import main_loop  # noqa: F401


@pytest.fixture
def prepared_memory(tmp_path, monkeypatch):
    store, tools, chat, _, _ = setup(tmp_path)
    records = [store.append_episode("2", f"Decision {index}: original cause remains. " * 350,
        [], {"kind": "mind"}) for index in range(2)]
    snapshot = {"focus": "1", "rooms": [{"id": "2", "label": "Older work", "records": store.room_records("2")}],
                "marks": [], "open_focus": [], "raw_focus": [], "other_open_rooms": []}
    template = json.dumps([{"type": "text", "text": "Governance"}, {"type": "text", "text": "Shared identity"},
                           {"type": "text", "text": "Task health" + CHRONICLE_MARKER}])
    plan = replace(_plan(window=100_000, known=True), chronicle_state_json=json.dumps(snapshot),
        system_templates_json={mode: template for mode in ("max", "low", "nano")}, context_task={"type": "task"})
    monkeypatch.setattr(capability_evidence, "resolve_main_token_density", lambda *_a, **_k: (1.0, "cold_estimate"))
    context = _ctx(tmp_path)
    context.tools._ctx = tools
    context.active_model = plan.model
    context.context_fit_plan = plan
    context.messages = plan.messages_for("max") + [{"role": "user", "content": "Executor facts " * 1000}]
    context.tool_schemas = [{"type": "function", "function": {"name": "read_file", "description": "Schema " * 1000}}]
    tools.context_fit_plan = plan
    tools.messages = context.messages
    helper, events = Helper(), []
    monkeypatch.setattr(consolidator, "_light_call", lambda *_a: helper)
    monkeypatch.setattr(consolidator, "_light_route", lambda: {"model": "test-light", "use_local": False})
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda _q, _t, _d, event: events.append(event))
    monkeypatch.setattr(loop, "_account_compaction_usage", lambda *_a, **_k: None)
    return SimpleNamespace(store=store, records=records, context=context, helper=helper, events=events, chat=chat)


def test_first_prepared_main_keeps_existing_meaning_and_counts_tools_and_executor(prepared_memory, monkeypatch):
    run = prepared_memory
    # This comparison isolates the prepared memory allowance from send-time clock text.
    monkeypatch.setattr("ouroboros.send_clock.main_clock_policy", lambda *_a, **_k: None)
    plan = run.context.context_fit_plan
    nominal = plan.fit_prepared_memory(plan.messages_for("max"), [], "max")
    complete = plan.fit_prepared_memory(run.context.messages, run.context.tool_schemas, "max")
    assert complete.context_task["context_non_memory_tokens"] > nominal.context_task["context_non_memory_tokens"]
    assert complete.max_projection.memory_facts["requested_memory_tokens"] < nominal.max_projection.memory_facts["requested_memory_tokens"]
    assert nominal.max_projection.memory_facts["target_miss"] is False
    assert complete.max_projection.memory_facts["target_miss"] is True
    assert run.helper.calls == []  # Capture and rendering do not start a paid task.
    run.chat.write_text(json.dumps({"chat_id": 2, "text": "UNCONSUMED_NEW_RAW"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(consolidator, "_capture_generation_window", lambda *_a, **_k: pytest.fail("bootstrap must not scan raw history"))
    prepared, physical = loop_model_call._prepare_first_main_memory(run.context, run.context.messages)
    assert run.helper.calls == []
    assert all(record["text"] in str(prepared) for record in run.records)
    assert run.context.context_fit_plan.max_projection.memory_facts["target_miss"]
    assert all(run.store.get(record["id"])["text"] == record["text"] for record in run.records)
    assert physical.rendered_mode == "max" and physical.automatic_pass_used is False
    assert run.context.context_fit_plan.chronicle_state_json == plan.chronicle_state_json
    assert run.context.accumulated_usage["_context_prompt_estimate"] > nominal.max_projection.estimated_tokens
    loop_model_call._prepare_first_main_memory(run.context, prepared)
    assert run.helper.calls == []


@pytest.mark.parametrize("binding", ["metadata_only", "matched_fields"])
def test_first_main_uses_canonical_memory_in_its_actual_physical_request(prepared_memory, main_loop, binding):  # noqa: F811
    from ouroboros.chronicle_view import refresh_chronicle_snapshot
    from ouroboros.observability import read_call_payload

    run, main = prepared_memory, main_loop
    canonical = run.store.data_root
    for index in range(2):
        run.store.append_episode("2", f"Canonical decision {index} with its source reason. " * 2500,
                                 [], {"kind": "mind"})
    main.ctx.task_metadata = {"id": "authored-main", "type": "task", "workspace_mode": "external",
                              "budget_drive_root": str(canonical)}
    main.ctx.budget_drive_root = str(canonical) if binding == "matched_fields" else ""
    main.ctx.context_fit_plan = replace(run.context.context_fit_plan, window_tokens=200_000,
        context_task=main.ctx.task_metadata, chronicle_state_json=refresh_chronicle_snapshot(
            run.context.context_fit_plan.chronicle_state_json, canonical))
    main.messages = main.ctx.context_fit_plan.messages_for("max")
    assert main.ctx.drive_root != canonical
    assert not (main.ctx.drive_root / "memory/chronicle").exists()

    answer, _, _ = main.run([{"content": "done"}])

    assert answer == "done" and len(main.inputs) == 1
    assert run.helper.calls == []
    digests = run.store.records("2", kinds=["digest"])
    assert digests == []
    _, physical, _ = read_call_payload(main.ctx.drive_root, task_id="authored-main", call_id="authored-send-1")
    assert "Canonical decision 0 with its source reason." in str(physical["messages"][0])
    assert "Canonical decision 1 with its source reason." in str(physical["messages"][0])
    assert json.loads(main.ctx.context_fit_plan.chronicle_state_json)["rooms"]
    assert not (main.ctx.drive_root / "memory/chronicle").exists()


@pytest.mark.parametrize("case", ["fitting", "held", "child", "direct", "consciousness"])
def test_preparation_respects_main_authority_and_unresolved_attempts(prepared_memory, case):
    run = prepared_memory
    if case == "fitting":
        run.context.context_fit_plan = replace(run.context.context_fit_plan, window_tokens=1_000_000)
    elif case == "held":
        run.context.accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
    else:
        run.store.append_episode("2", "Further meaning remains exact. " * 3000, [], {"kind": "mind"})
        snapshot = json.loads(run.context.context_fit_plan.chronicle_state_json)
        snapshot["rooms"][0]["records"] = run.store.room_records("2")
        task = {"type": "task", "_is_direct_chat": case == "direct"}
        if case == "child":
            task["delegation_role"] = "subagent"
            run.context.tools._ctx.task_metadata = {"delegation_role": "subagent"}
            snapshot["is_child"] = True
        elif case == "consciousness":
            task["type"] = run.context.task_type = "consciousness"
        run.context.context_fit_plan = replace(run.context.context_fit_plan,
            context_task=task, chronicle_state_json=json.dumps(snapshot))
    before = run.context.context_fit_plan.chronicle_state_json
    loop_model_call._prepare_first_main_memory(run.context, run.context.messages)
    assert run.helper.calls == [] and run.context.context_fit_plan.chronicle_state_json == before


def test_account_rebind_retains_meaning_and_owner_max_low_horizon_without_preparation(prepared_memory, monkeypatch):
    from ouroboros import context
    run = prepared_memory
    prepared, _ = loop_model_call._prepare_first_main_memory(run.context, run.context.messages)
    assert run.helper.calls == []
    monkeypatch.setattr(context, "_context_fit_route", lambda *_a, **_k: (
        {"model": run.context.active_model, "provider": "openai"},
        SimpleNamespace(route_fp="new-account", status="confirmed", stale=False, window_tokens=500_000)))
    rebound, mode = loop_model_call._rebind_context_fit_plan(run.context.context_fit_plan,
        run.context.tools, prepared, model=run.context.active_model, use_local=False,
        preferred_mode="low", tool_schemas=run.context.tool_schemas)
    run.context.context_fit_plan, run.context.active_context_mode = rebound, mode
    run.context.messages[:] = prepared
    prepared, physical = loop_model_call._prepare_first_main_memory(run.context, prepared)
    max_owner_budget = run.context.context_fit_plan.low_projection.memory_facts["requested_memory_tokens"]
    low_owner = replace(run.context.context_fit_plan, preferred_mode="low").fit_prepared_memory(
        prepared, run.context.tool_schemas, "low")
    assert run.helper.calls == []
    assert all(record["text"] in str(prepared) for record in run.records)
    assert physical.profile == "task_local_low" and physical.target_total_tokens is None
    assert run.context.context_fit_plan.preferred_mode == "max"
    assert run.context.context_fit_plan.context_task["owner_context_mode"] == "max"
    assert max_owner_budget > low_owner.low_projection.memory_facts["requested_memory_tokens"]


def test_real_round_dispatch_sends_published_memory_without_paid_preparation(prepared_memory):
    run = prepared_memory
    sent = []
    run.context.tools._ctx._owner_directives = [{"source": "owner", "content": "Original instruction"}]
    def chat(**kwargs):
        assert run.helper.calls == []
        assert run.context.tools._ctx._completion_observation["owner_directives"] == 1
        run.context.tools._ctx._owner_directives.append({"source": "owner", "content": "Not observed in this response"})
        sent.append(kwargs)
        return {"role": "assistant", "content": "Done", "tool_calls": []}, {
            "provider": "openai", "cost": 0, "prompt_tokens": 100, "completion_tokens": 1}
    run.context.llm = SimpleNamespace(chat=chat)
    ua.adopt_physical_attempt_capture(None)
    message, _, _ = loop_model_call._call_round_model(run.context)
    assert message["content"] == "Done" and len(sent) == 1
    assert all(record["text"] in str(sent[0]["messages"]) for record in run.records)
    assert sent[0]["tools"] == run.context.tool_schemas
    assert "Executor facts" in str(sent[0]["messages"])
    assert run.context.tools._ctx._completion_observation["owner_directives"] == 1
    assert not any(event["checkpoint_kind"] == "context_reclaim_automatic" for event in run.events)


@pytest.mark.parametrize("fallback,recovery_failure", [
    ("none", ""), ("success", ""), ("quota_exhausted", ""), ("context_overflow", ""),
    ("provider_outcome_unknown", ""), ("transport_unavailable", ""),
    ("context_overflow", "context_overflow"), ("context_overflow", "provider_outcome_unknown"),
    ("context_overflow", "helper_unknown"),
])
def test_actual_refusal_tries_configured_route_before_source_bound_memory_repair(
        prepared_memory, monkeypatch, fallback, recovery_failure):
    from ouroboros import fallback_cooldown
    from tests.test_loop_compaction import _failed_capture

    run, sent, captures = prepared_memory, [], []
    ctx = run.context
    primary, backup = ctx.active_model, "openai/backup"
    ctx.active_context_mode = "low"
    ctx.context_fit_plan = replace(ctx.context_fit_plan, preferred_mode="low", rendered_mode="low")
    ctx.messages[:] = ctx.context_fit_plan.messages_for("low")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "" if fallback == "none" else backup)
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "false")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *_a: None)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda *_a: None)
    monkeypatch.setattr(loop, "_reconcile_transport_wait", lambda current, *_a, **_k: current)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: captures[-1])
    monkeypatch.setattr(loop, "_run_main_reclaim", lambda call, fit, **_k:
        loop._context_reclaim_passes(call.tools._ctx).add(loop_model_call._fit_key(fit)))
    if fallback == "quota_exhausted":
        from ouroboros import model_wait
        monkeypatch.setattr(model_wait, "current_model_wait", lambda: SimpleNamespace(waits_allowed=True, overrides={}))

    def rebind(plan, tools, messages, *, model, **_kwargs):
        plan = replace(plan, model=model, route_fp="backup-route")
        tools._ctx.context_fit_plan = plan
        messages[:] = plan.reproject_transcript(messages, "low")
        return plan, "low"

    monkeypatch.setattr(loop, "_rebind_context_fit_plan", rebind)
    if recovery_failure == "helper_unknown":
        def unknown_helper(prompt, label, **options):
            run.helper.calls.append((prompt, label, options))
            return "", {"_consolidation_errors": [{"kind": "provider_outcome_unknown",
                                                    "physical_attempt_id": "held-helper"}]}, None
        monkeypatch.setattr(consolidator, "_light_call", lambda *_a: unknown_helper)

    def dispatch(call, fit, *, candidate_predicate=None, **_kwargs):
        prepared, physical = loop_model_call._prepare_first_main_memory(call, call.messages)
        call.messages[:] = prepared
        size = loop_model_call._main_frame_bytes(call)
        sent.append((call.active_model, len(run.helper.calls), candidate_predicate is not None))
        if candidate_predicate is not None:
            request = ua.AttemptRequest(model=call.active_model, provider="openai", max_completion_tokens=65536,
                candidate_raw_sha256="changed", candidate_context_size_bytes=size,
                candidate_measurement_kind="canonical_json_v1", physical_context=physical)
            assert candidate_predicate(request), "the actual retry must be smaller with the same route and reserve"
            assert "Room history retains the open question and its cause." in str(prepared)
            if recovery_failure:
                call.accumulated_usage["_last_llm_error_kind"] = recovery_failure
                captures.append(replace(_failed_capture(mode="low", size=size), model=call.active_model,
                                        candidate_raw_sha256="changed", physical_context=physical))
                return None, 0.0
            return {"role": "assistant", "content": "recovered"}, 0.0
        if call.active_model == backup and fallback == "success":
            assert run.helper.calls == []
            return {"role": "assistant", "content": "fallback accepted old meanings"}, 0.0
        kind = fallback if call.active_model == backup else "context_overflow"
        call.accumulated_usage["_last_llm_error_kind"] = kind
        if call.active_model == backup and fallback == "quota_exhausted":
            call.tools._ctx._deferred_resource_refusal = SimpleNamespace(
                ask_owner=lambda *_a: pytest.fail("a useful primary memory repair precedes the fallback quota wait"))
        captures.append(replace(_failed_capture(mode="low", size=size), model=call.active_model,
                                physical_context=physical))
        return None, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    message, _, _ = loop_model_call._call_round_model(ctx)
    assert run.helper.calls == []
    assert message is None
    primary_fit_usage = loop_model_call._snapshot_context_fit_usage(ctx.accumulated_usage)
    ctx.accumulated_usage["cost"] = 0.37
    result = loop_model_call._recover_failed_round(ctx, ctx.tools, message, None,
        context_fit_plan=ctx.context_fit_plan, active_context_mode=ctx.active_context_mode,
        emit_progress=lambda *_a, **_k: None)
    message = result[0]
    if recovery_failure:
        shared = ctx.tools._ctx
        assert message is None and len(run.helper.calls) == (2 if recovery_failure == "context_overflow" else 1)
        assert result[1] == primary and shared.context_fit_plan is result[3]
        assert shared.context_fit_plan.model == primary and shared.context_fit_plan.route_fp == "route-a"
        assert shared.messages is ctx.messages and shared.active_context_mode == result[4]
        assert shared.active_model == primary and shared.active_use_local is False
        assert loop_model_call._snapshot_context_fit_usage(ctx.accumulated_usage) == primary_fit_usage
        assert ctx.accumulated_usage["cost"] == 0.37
        expected_error = "provider_outcome_unknown" if recovery_failure == "helper_unknown" else recovery_failure
        assert ctx.accumulated_usage["_last_llm_error_kind"] == expected_error
        assert loop.last_physical_attempt_capture() is captures[-1]
        assert len(sent) == (2 if recovery_failure == "helper_unknown" else
                             4 if recovery_failure == "context_overflow" else 3)
        if recovery_failure == "helper_unknown":
            assert run.store.scan_state()["pending_consolidation_outcomes"][0]["physical_attempt_id"] == "held-helper"
        assert loop_model_call._recover_deferred_memory_refusal(ctx.tools) == (None, None, 0.0)
    elif fallback in {"success", "provider_outcome_unknown"}:
        assert run.helper.calls == [] and len(sent) == 2
        assert bool(message) is (fallback == "success")
    else:
        assert message["content"] == "recovered" and len(run.helper.calls) == 1
        assert sent[-1][0] == primary  # an alternative never replaces the acting route's paid need
        assert sent[-1][1:] == (1, True)
        assert all(count == 0 for _model, count, _retry in sent[:-1])
        event = next(e for e in run.events if e["checkpoint_kind"] == "context_memory_prepared")
        assert event["fitting_demand"]["purpose"] == "actual_context_refusal"
        assert "requirement_tokens" not in event["fitting_demand"]  # catalogue estimates are not refusal evidence
        assert all(run.store.get(r["id"])["text"] == r["text"] for r in run.records)
        assert loop_model_call._recover_deferred_memory_refusal(ctx.tools) == (None, None, 0.0)
        assert len(run.helper.calls) == 1


@pytest.mark.parametrize("error", ["budget_exhausted", "provider_outcome_unknown"])
def test_refused_memory_repair_preserves_paid_interruptions_and_originals(prepared_memory, monkeypatch, error):
    from tests.test_loop_compaction import _failed_capture

    run, calls = prepared_memory, []
    loop_model_call._prepare_first_main_memory(run.context, run.context.messages)

    def interrupted(*_args, **_kwargs):
        calls.append(error)
        return "", {"prompt_tokens": 1, "_consolidation_errors": [{"kind": error}]}, None

    monkeypatch.setattr(consolidator, "_light_call", lambda *_a: interrupted)
    assert not loop_memory._repair_refused_main_memory(run.context, _failed_capture())
    assert calls == [error]
    assert run.context.accumulated_usage["_last_llm_error_kind"] == error
    assert not run.store.records(kinds=["digest"])
    assert all(record["text"] in str(run.context.messages) for record in run.records)


@pytest.mark.parametrize("extra_room,backup_failure,repairable", [
    (False, "", True), (False, "context_overflow", True), (False, "transport_unavailable", True),
    (True, "", True), (False, "transport_unavailable", False),
])
def test_books_remain_max_until_source_bound_recovery_exhausted(
        prepared_memory, monkeypatch, extra_room, backup_failure, repairable):
    from ouroboros import fallback_cooldown
    from ouroboros.chronicle_view import refresh_chronicle_snapshot
    from tests.test_loop_compaction import _failed_capture

    run, ctx = prepared_memory, prepared_memory.context
    primary, backup, sent, captures, notes = ctx.active_model, "openai/backup", [], [], []
    if extra_room:
        for index in range(2):
            run.store.append_episode("3", f"Independent room {index}: original cause. " * 350, [], {"kind": "mind"})
    templates = dict(ctx.context_fit_plan.system_templates_json)
    content = json.loads(templates["max"])
    content[0]["text"] += "Complete resident book text. " * 5000
    templates["max"] = json.dumps(content)
    ctx.context_fit_plan = replace(ctx.context_fit_plan, system_templates_json=templates,
        chronicle_state_json=refresh_chronicle_snapshot(ctx.context_fit_plan.chronicle_state_json, run.store.data_root))
    ctx.messages[:] = ctx.context_fit_plan.messages_for("max")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", backup if backup_failure else "")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "false")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *_a: None)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda *_a: None)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: captures[-1])
    monkeypatch.setattr(loop, "_run_main_reclaim", lambda call, fit, **_k:
        loop._context_reclaim_passes(call.tools._ctx).add(loop_model_call._fit_key(fit)))
    if not repairable:
        def not_shorter(prompt, label, **options):
            run.helper.calls.append((prompt, label, options))
            return "Retained source detail. " * 3000, {"prompt_tokens": 1, "cost": 0.01}, None
        monkeypatch.setattr(consolidator, "_light_call", lambda *_a: not_shorter)

    def rebind(plan, tools, messages, *, model, **_kwargs):
        plan = replace(plan, model=model, route_fp="backup-route")
        messages[:] = plan.reproject_transcript(messages, "max")
        tools._ctx.context_fit_plan = plan
        return plan, "max"

    monkeypatch.setattr(loop, "_rebind_context_fit_plan", rebind)

    def dispatch(call, fit, *, candidate_predicate=None, **_kwargs):
        prepared, physical = loop_memory._prepare_first_main_memory(call, call.messages)
        call.messages[:] = prepared
        size = loop_memory._main_frame_bytes(call)
        capture = replace(_failed_capture(mode=call.active_context_mode, size=size),
            candidate_raw_sha256=f"request-{len(sent)}", model=call.active_model, physical_context=physical)
        if candidate_predicate is not None:
            assert candidate_predicate(capture)
        sent.append((call.active_model, call.active_context_mode, size, len(run.helper.calls)))
        if call.active_model == primary and size < 12000:
            return {"role": "assistant", "content": "source-bound recovery"}, 0.0
        call.accumulated_usage["_last_llm_error_kind"] = backup_failure if call.active_model == backup else "context_overflow"
        captures.append(capture)
        return None, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    message, _, _ = loop_model_call._call_round_model(ctx)
    assert message is None and [row[:2] for row in sent] == [(primary, "max")]
    assert run.helper.calls == []
    result = loop_model_call._recover_failed_round(ctx, ctx.tools, message, None,
        context_fit_plan=ctx.context_fit_plan, active_context_mode=ctx.active_context_mode,
        emit_progress=lambda text, **_kw: notes.append(text))
    assert result[5] is None and not any("Could not establish" in note for note in notes)
    assert len(run.helper.calls) == ((4 if extra_room else 2) if repairable else 1)
    if not repairable:
        assert result[0] is None and ctx.accumulated_usage["_last_llm_error_kind"] == "context_overflow"
        assert not run.store.records(kinds=["digest"])
    else:
        assert result[0]["content"] == "source-bound recovery" and result[1] == primary
    primary_sizes = [row[2] for row in sent if row[0] == primary]
    assert all(later < earlier for earlier, later in zip(primary_sizes, primary_sizes[1:]))
    if backup_failure:
        assert sent[1][0] == backup and sent[1][1] == "max" and sent[1][3] == 0
    assert ctx.tools._ctx.context_fit_plan is result[3] and ctx.tools._ctx.messages is ctx.messages


def test_memory_refusal_keeps_unknown_custody_when_last_error_changes(prepared_memory, monkeypatch):
    from tests.test_loop_compaction import _failed_capture

    run, ctx = prepared_memory, prepared_memory.context
    loop_model_call._prepare_first_main_memory(ctx, ctx.messages)
    held = {"physical_attempt_id": "unknown-old-route", "same_operation_recoverable": False}
    ctx.accumulated_usage.update(_pending_transport_outcome=held, _last_llm_error_kind="context_overflow")
    ctx.tools._ctx._deferred_memory_refusal = loop_memory._DeferredMemoryRefusal(
        ctx, _failed_capture())
    monkeypatch.setattr(loop, "_dispatch_round_model", lambda *_a, **_k: pytest.fail("held memory retry"))
    assert loop_memory._recover_deferred_memory_refusal(ctx.tools) == (None, None, 0.0)
    assert not run.helper.calls and ctx.accumulated_usage["_pending_transport_outcome"] is held


@pytest.mark.parametrize("reason", ["cancelled", "deadline"])
def test_source_bound_recovery_propagates_existing_owner_controls(prepared_memory, monkeypatch, reason):
    from ouroboros.model_wait import ModelWaitInterrupted
    from tests.test_loop_compaction import _failed_capture

    run, ctx = prepared_memory, prepared_memory.context
    _, physical = loop_memory._prepare_first_main_memory(ctx, ctx.messages)
    capture = replace(_failed_capture(size=loop_memory._main_frame_bytes(ctx)),
                      model=ctx.active_model, physical_context=physical)
    ctx.tools._ctx._deferred_memory_refusal = loop_memory._DeferredMemoryRefusal(ctx, capture)
    control = ModelWaitInterrupted(reason, role="light")
    def interrupted(*_a, **_kw):
        raise control
    monkeypatch.setattr(consolidator, "_light_call", lambda *_a: interrupted)
    monkeypatch.setattr(loop, "_dispatch_round_model", lambda *_a, **_kw: pytest.fail("no send after owner control"))
    with pytest.raises(ModelWaitInterrupted) as caught:
        loop_memory._recover_deferred_memory_refusal(ctx.tools)
    assert caught.value is control
    assert not run.store.records(kinds=["digest"])
    assert all(run.store.get(record["id"])["text"] == record["text"] for record in run.records)


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_memory_rebind_uses_new_local_output_reserve(prepared_memory, monkeypatch, mode):
    from ouroboros import context, context_fit, chronicle_view

    run, observed = prepared_memory, []
    plan = replace(run.context.context_fit_plan, preferred_mode=mode)
    monkeypatch.setattr(context_fit, "main_output_reserve_tokens", lambda *, use_local: 4096 if use_local else 65536)

    def route(task, **_kwargs):
        local = task["use_local_model"]
        return {"model": task["model"], "provider": "local" if local else "openai", "use_local": local}, SimpleNamespace(
            route_fp="local" if local else "remote", status="confirmed", stale=False, window_tokens=200000)

    original = chronicle_view.render_system_view
    def render(*args, **kwargs):
        observed.append(kwargs["output_reserve_tokens"])
        return original(*args, **kwargs)

    monkeypatch.setattr(context, "_context_fit_route", route)
    monkeypatch.setattr(chronicle_view, "render_system_view", render)
    messages = plan.messages_for(mode)
    for local, reserve in [(True, 4096), (False, 65536)]:
        observed.clear()
        plan, active = loop_model_call._rebind_context_fit_plan(plan, run.context.tools, messages,
            model="local-memory" if local else "openai/memory", use_local=local,
            preferred_mode=mode, tool_schemas=run.context.tool_schemas)
        assert active == mode and plan.output_reserve_tokens == reserve
        assert observed and set(observed) == {reserve}
        assert all(record["text"] in str(messages) for record in run.records)


@pytest.mark.parametrize("shape", ["system", "split", "anthropic", "two_sections", "ambiguous", "quoted_only"])
def test_actual_memory_measurement_distinguishes_host_projection_from_quoted_text(shape):
    from ouroboros.llm_messages import project_declared_system_prefix
    body = "Сохранённый смысл 🐍\n"
    rendered = MEMORY_BEGIN + body + MEMORY_END + MEMORY_FACTS_PREFIX + '{"target_miss":true}'
    messages = [{"role": "system", "content": [{"type": "text", "text": "governance"},
        {"type": "text", "text": rendered}], "_stable_prefix_blocks": 1}]
    expected = 1
    if shape == "split":
        messages = project_declared_system_prefix({"provider": "claudexor"}, messages)
    elif shape == "two_sections":
        messages[0]["content"].append({"type": "text", "text": rendered})
        expected = 2
    elif shape == "ambiguous":
        messages[0]["content"][1]["text"] = MEMORY_BEGIN + body
    elif shape == "quoted_only":
        messages = [{"role": "tool", "content": rendered}, {"role": "user", "content": rendered}]
    payload = {"system": messages[0]["content"]} if shape == "anthropic" else {"messages": messages}
    facts = memory_view_measurement(payload)
    if shape in {"ambiguous", "quoted_only"}:
        assert facts["status"] == ("ambiguous" if shape == "ambiguous" else "unobserved")
        assert facts["utf8_bytes"] is None and facts["projection_target_miss"] is None
    else:
        assert facts["status"] == "observed" and facts["sections"] == expected
        assert facts["chars"] == len(body) * expected
        assert facts["utf8_bytes"] == len(body.encode("utf-8")) * expected
        assert facts["projection_target_miss"] is True and facts["token_estimate_basis"] == "chars_div_4"


@pytest.mark.parametrize("failed", [False, True])
def test_memory_sizes_follow_real_attempt_ledger_and_success_or_error_event(transport, monkeypatch, failed):  # noqa: F811
    from ouroboros.loop_llm_call import call_llm_with_retry
    monkeypatch.setattr("ouroboros.loop_llm_call.main_loop_wire_options", lambda *_a, **_k: {"stream": False})
    root, client, sent = transport
    body = "Exact shared memory including Unicode: память."
    messages = [{"role": "system", "content": MEMORY_BEGIN + body + MEMORY_END
        + MEMORY_FACTS_PREFIX + '{"target_miss":true}'}, {"role": "user", "content": "Act"}]
    context = ua.PhysicalAttemptContext("owner_max", "max", "cold_estimate", "route", "exec:round:1", None, 1_000_000, False, False)
    if failed:
        class Rejected(Exception):
            status_code = 400
        def create(**candidate):
            sent.append(candidate)
            raise Rejected("invalid request")
        monkeypatch.setattr(client, "_get_remote_client", lambda _target: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    usage = {"execution_id": "exec"}
    result, _ = call_llm_with_retry(client, messages, "openai::test-model", [], "none", 1,
        root / "logs", "processing", 1, None, usage, physical_context=context, attempt_cap=1)
    assert bool(result) is not failed
    rows = [json.loads(line) for line in (root / ua.LEDGER_REL).read_text(encoding="utf-8").splitlines()]
    assert rows and all(row["physical_context"]["memory_view"]["chars"] == len(body) for row in rows)
    events = [json.loads(line) for line in (root / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()]
    event = next(event for event in events if event["type"] == ("llm_api_error" if failed else "llm_round"))
    assert event["context_memory_view"]["utf8_bytes"] == len(body.encode("utf-8"))
    assert event["context_memory_view"]["projection_target_miss"] is True
    assert event["context_target_miss"] is False


@pytest.mark.parametrize("path,backup_failure", [
    ("max_repair", ""),                      # Max meaningful repair send hits connect failure
    ("repair_resend", ""),                   # owner Low, no fallback; repaired re-send hits connect failure
    ("repair_resend", "transport_unavailable"),
    ("repair_resend", "context_overflow"),   # owner Low, backup refuses; repaired re-send hits connect failure
])
def test_acting_route_connect_failure_after_memory_repair_keeps_real_outage(prepared_memory, monkeypatch, path, backup_failure):
    from ouroboros import fallback_cooldown
    from tests.test_loop_compaction import _failed_capture

    run, ctx, sent, captures, notes = prepared_memory, prepared_memory.context, [], [], []
    primary, backup = ctx.active_model, "openai/backup"
    if path == "max_repair":
        templates = dict(ctx.context_fit_plan.system_templates_json)
        content = json.loads(templates["max"])
        content[0]["text"] += "Complete resident book text. " * 5000
        templates["max"] = json.dumps(content)
        ctx.active_context_mode = "max"
        ctx.context_fit_plan = replace(ctx.context_fit_plan, preferred_mode="max", rendered_mode="max",
                                       system_templates_json=templates)
        ctx.messages[:] = ctx.context_fit_plan.messages_for("max")
    else:
        ctx.active_context_mode = "low"
        ctx.context_fit_plan = replace(ctx.context_fit_plan, preferred_mode="low", rendered_mode="low")
        ctx.messages[:] = ctx.context_fit_plan.messages_for("low")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", backup if backup_failure else "")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "false")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *_a: None)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda *_a: None)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: captures[-1])
    monkeypatch.setattr(loop, "_run_main_reclaim", lambda call, fit, **_k:
        loop._context_reclaim_passes(call.tools._ctx).add(loop_model_call._fit_key(fit)))

    def rebind(plan, tools, messages, *, model, **_kwargs):
        plan = replace(plan, model=model, route_fp="backup-route")
        messages[:] = plan.reproject_transcript(messages, "max")
        tools._ctx.context_fit_plan = plan
        return plan, "max"

    monkeypatch.setattr(loop, "_rebind_context_fit_plan", rebind)

    def dispatch(call, fit, *, candidate_predicate=None, **_kwargs):
        prepared, physical = loop_memory._prepare_first_main_memory(call, call.messages)
        call.messages[:] = prepared
        size = loop_memory._main_frame_bytes(call)
        capture = replace(_failed_capture(mode=call.active_context_mode, size=size),
            candidate_raw_sha256=f"request-{len(sent)}", model=call.active_model, physical_context=physical)
        sent.append((call.active_model, call.active_context_mode, size, candidate_predicate is not None,
                     len(run.helper.calls)))
        captures.append(capture)
        if candidate_predicate is not None:
            assert candidate_predicate(capture)
            # The acting route itself cannot be reached for this smaller send.
            call.accumulated_usage["_last_llm_error_kind"] = "transport_unavailable"
            return None, 0.0
        call.accumulated_usage["_last_llm_error_kind"] = (backup_failure if call.active_model == backup
                                                          else "context_overflow")
        return None, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    message, _, _ = loop_model_call._call_round_model(ctx)
    result = loop_model_call._recover_failed_round(ctx, ctx.tools, message, None,
        context_fit_plan=ctx.context_fit_plan, active_context_mode=ctx.active_context_mode,
        emit_progress=lambda text, **_kw: notes.append(text))
    assert result[0] is None
    assert result[5] is not None and result[5].wait_cause == "transport_unavailable"
    assert ctx.accumulated_usage["_last_llm_error_kind"] == "transport_unavailable"
    assert sent[-1][0] == primary and sent[-1][3] is True
    assert len(run.helper.calls) == 1
    assert sum(row[0] == backup for row in sent) == bool(backup_failure)
    assert any("Could not establish" in note for note in notes)


@pytest.mark.parametrize("outcome", ["max_progress", "truncated_group", "irreducible", "fallback_success", "helper_unknown", "helper_budget", "ready_view", "fallback_low_success", "owner_low_fallback"])
def test_physical_recovery_keeps_books_and_options_until_useful_paths_exhausted(
        prepared_memory, transport, monkeypatch, outcome):  # noqa: F811
    """Actual dispatch/capture gate, source writer and configured fallback walk."""
    from copy import deepcopy
    from ouroboros import fallback_cooldown
    run, ctx = prepared_memory, prepared_memory.context
    _root, client, sent = transport
    ctx.llm = client
    primary, backup = "openai::test-model", "openai::backup"
    ctx.active_model = primary
    templates = dict(ctx.context_fit_plan.system_templates_json)
    blocks = json.loads(templates["max"])
    book = "COMPLETE_RESIDENT_BOOK " * 2000
    blocks[0]["text"] += book
    templates["max"] = json.dumps(blocks)
    if outcome == "ready_view":
        from ouroboros.chronicle_view import refresh_chronicle_snapshot
        run.store.append_episode("2", "Ready meaningful history retains its cause.", [], {"kind": "helper"},
            kind="digest", metadata={"covers_record_ids": [r["id"] for r in run.records]})
        ctx.context_fit_plan = replace(ctx.context_fit_plan, chronicle_state_json=refresh_chronicle_snapshot(
            ctx.context_fit_plan.chronicle_state_json, run.store.data_root))
    ctx.context_fit_plan = replace(ctx.context_fit_plan, model=primary, window_tokens=1000000,
                                   system_templates_json=templates)
    if outcome == "owner_low_fallback":
        ctx.active_context_mode = "low"
        ctx.context_fit_plan = replace(ctx.context_fit_plan, preferred_mode="low", rendered_mode="low")
    ctx.messages[:] = ctx.context_fit_plan.messages_for(ctx.active_context_mode)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", backup)
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "false")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *_a: None)
    monkeypatch.setattr("ouroboros.loop_llm_call.main_loop_wire_options", lambda *_a, **_k: {"stream": False})
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda *_a: None)
    monkeypatch.setattr(loop, "_reconcile_transport_wait", lambda current, *_a, **_k: current)
    def rebind(plan, tools, messages, *, model, preferred_mode, **_kwargs):
        assert preferred_mode in {"max", "low"}
        plan = replace(plan, model=model, route_fp="backup-route")
        messages[:] = plan.reproject_transcript(messages, preferred_mode)
        tools._ctx.context_fit_plan = plan
        return plan, preferred_mode
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", rebind)
    helper_calls = []
    def helper(prompt, label, **options):
        helper_calls.append(prompt)
        if outcome == "truncated_group" and len(helper_calls) == 1:
            return "", {"_consolidation_errors": [{"kind": "output_truncated"}], "cost": 0}, None
        if outcome.startswith("helper_"):
            kind = "provider_outcome_unknown" if outcome == "helper_unknown" else "budget_exhausted"
            return "", {"_consolidation_errors": [{"kind": kind}]}, None
        repeats = 80 if len(helper_calls) == 1 else (10 if outcome == "max_progress" else 80)
        return "Meaningful causal account remains open. " * repeats, {"prompt_tokens": 1, "cost": 0}, None
    monkeypatch.setattr(consolidator, "_light_call", lambda *_a: helper)
    observations = []
    class Overflow(Exception):
        status_code = 400
    class Response:
        def model_dump(self):
            return {"choices": [{"message": {"role": "assistant", "content": "Recovered"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "cost": 0}}
    def create(**candidate):
        sent.append(deepcopy(candidate))
        full = book in str(candidate["messages"])
        observations.append((candidate["model"], full, len(helper_calls)))
        is_backup = "backup" in candidate["model"]
        if outcome == "ready_view" and len(sent) == 2:
            return Response()
        if is_backup and (outcome == "fallback_success" or (outcome == "fallback_low_success" and not full)
                          or (outcome == "owner_low_fallback" and len(helper_calls) >= 2)):
            return Response()
        if not is_backup and ((outcome == "max_progress" and len(helper_calls) >= 2)
                              or (outcome == "truncated_group" and len(helper_calls) >= 3)
                              or (outcome == "irreducible" and not full)):
            return Response()
        raise Overflow("maximum context length exceeded")
    monkeypatch.setattr(client, "_get_remote_client", lambda *_a: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    ua.adopt_physical_attempt_capture(None)
    message, _, mode = loop_model_call._call_round_model(ctx)
    if outcome == "ready_view":
        assert message["content"] == "Recovered" and mode == "max" and not helper_calls
        assert len(sent) == 2 and all(row[:2] == ("test-model", True) for row in observations)
        assert all(r["text"] in str(sent[0]["messages"]) for r in run.records)
        assert all(r["text"] not in str(sent[1]["messages"]) for r in run.records)
        return
    assert message is None and mode == ("low" if outcome == "owner_low_fallback" else "max") and not helper_calls
    result = loop_model_call._recover_failed_round(ctx, ctx.tools, message, None,
        context_fit_plan=ctx.context_fit_plan, active_context_mode=mode, emit_progress=lambda *_a, **_k: None)
    assert observations[0][1:] == (outcome != "owner_low_fallback", 0)
    assert observations[1] == ("backup", outcome != "owner_low_fallback", 0)
    if outcome == "fallback_success":
        assert result[0]["content"] == "Recovered" and not helper_calls
    elif outcome in {"fallback_low_success", "owner_low_fallback"}:
        assert result[0]["content"] == "Recovered" and result[1] == backup
        assert result[4] == "low" and len(helper_calls) == 2
        assert observations[-2][0] == "test-model" and observations[-2][1] is False
        assert observations[-1] == ("backup", False, 2)
        backup_sends = [row for row in sent if row["model"] == "backup"]
        assert len(backup_sends) == 2
        assert len(json.dumps(backup_sends[1]["messages"])) < len(json.dumps(backup_sends[0]["messages"]))
    elif outcome.startswith("helper_"):
        assert result[0] is None and len(sent) == 2 and len(helper_calls) == 1
        assert ctx.accumulated_usage["_last_llm_error_kind"] == (
            "provider_outcome_unknown" if outcome == "helper_unknown" else "budget_exhausted")
    else:
        assert result[0]["content"] == "Recovered" and result[1] == primary
        assert len(helper_calls) == (3 if outcome == "truncated_group" else 2)
        if outcome == "truncated_group":
            groups = [[json.loads(row)["record_id"] for row in prompt.split(
                "## Source group: complete text, projected host metadata\n", 1)[1].split("\n\n")]
                      for prompt in helper_calls]
            ids = [record["id"] for record in run.records]
            assert groups == [ids, ids[:1], ids[1:]]  # Failed whole, then its exact immutable halves.
            assert all(full for _model, full, _count in observations)
            assert not any(e.get("checkpoint_kind") == "context_fit_low_retry" for e in run.events)
        assert all(full for _model, full, _count in observations[:-1])
        assert observations[-1][1] is (outcome in {"max_progress", "truncated_group"})
        primary_sends = [row for row in sent if row["model"] != "backup"]
        sizes = [len(json.dumps(row["messages"])) for row in primary_sends]
        assert all(b < a for a, b in zip(sizes, sizes[1:]))
        options = [{key: value for key, value in row.items() if key not in {"messages", "prompt_cache_key"}} for row in primary_sends]
        assert all(options[0] == option for option in options[1:]), options
    assert all(run.store.get(record["id"])["text"] == record["text"] for record in run.records)

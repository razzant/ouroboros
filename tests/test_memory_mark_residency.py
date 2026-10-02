"""Explicit significance remains in the physical Main view after working compaction."""
from copy import deepcopy
from dataclasses import replace
import json
import sqlite3
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence, context_compaction, loop, loop_model_call
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.chronicle_view import CHRONICLE_MARKER
from ouroboros.context_budget import extract_plain_text_from_content
from ouroboros.tools.chronicle import _memory_mark
from tests.test_main_authored_context import main_loop, call  # noqa: F401


@pytest.fixture
def marked_main(main_loop, monkeypatch):  # noqa: F811
    f = main_loop
    monkeypatch.setattr(capability_evidence, "resolve_main_token_density", lambda *_a, **_k: (1.0, "cold_estimate"))
    f.ctx.task_metadata = {"id": "authored-main", "chat_id": 1, "type": "task"}
    f.ctx.current_chat_id = 1
    f.ctx.task_attempt = 1
    f.store = ChronicleStore(f.ctx.drive_root)
    f.source_record = f.store.append_episode("1", "The precise original words.", [], {"kind": "mind"})
    older = f.store.append_episode("2", "Older shared decision. " * 100, [], {"kind": "mind"})
    f.store.append_episode("2", "Stable shared interpretation.", [], {"kind": "helper"}, kind="digest",
                           metadata={"covers_record_ids": [older["id"]]})
    snapshot = {"focus": "1", "rooms": [{"id": "2", "label": "Older work", "records": f.store.room_records("2")}],
                "marks": [], "open_focus": [], "raw_focus": [], "other_open_rooms": []}
    template = json.dumps([{"type": "text", "text": "Governance"}, {"type": "text", "text": "Identity"},
                           {"type": "text", "text": "Task facts" + CHRONICLE_MARKER}])
    plan = replace(f.ctx.context_fit_plan, chronicle_state_json=json.dumps(snapshot),
        system_templates_json={mode: template for mode in ("max", "low", "nano")},
        context_task=f.ctx.task_metadata, nano_projection=replace(f.ctx.context_fit_plan.low_projection, mode="nano"))
    for mode in ("max", "low", "nano"):
        plan = plan.fit_prepared_memory(plan.messages_for(mode), [], mode)
    f.ctx.context_fit_plan = plan
    f.messages = plan.messages_for("max")
    f.mark_args = {"text": "The promise must remain visible.", "node_id": f.source_record["id"],
                   "quote": "The precise original words."}
    return f


@pytest.mark.parametrize("compaction", ["authored", "automatic"])
def test_new_mark_survives_real_main_working_compaction(marked_main, monkeypatch, compaction):
    f, summaries = marked_main, []
    if compaction == "automatic":
        from tests.test_context_reclaim_materializer import _SPEC
        monkeypatch.setattr(context_compaction, "_summarizer_spec", lambda: dict(_SPEC))
        def summarize(parts, **_kwargs):
            summaries.extend(parts)
            return {part.source_id: "The source was inspected." for part in parts}
        monkeypatch.setattr(context_compaction, "_call_summarizer", summarize)
        original = loop_model_call._measure_round_main_fit
        def pressure(ctx, *, automatic_pass_used):
            fit = original(ctx, automatic_pass_used=automatic_pass_used)
            if ctx.round_idx == 4 and not automatic_pass_used:
                return replace(fit, action="reclaim_once", measurement=replace(fit.measurement,
                    reclaim_goal_tokens=1, target_deficit_tokens=1))
            return fit
        monkeypatch.setattr(loop, "_measure_round_main_fit", pressure)
    final_call = (call("compact_context", {"working_note": "The source was inspected.", "keep_unit_ids": []}, "compact")
                  if compaction == "authored" else call("read_file", {"path": "evidence.txt"}, "fresh-read"))
    answer, usage, _ = f.run([call("memory_mark", f.mark_args, "mark"),
        call("read_file", {"path": "evidence.txt"}, "read"), final_call, {"content": "done"}])
    assert answer == "done"
    before, after = f.inputs[0]["messages"], f.inputs[-1]["messages"]
    from ouroboros.observability import read_call_payload
    _, physical, _ = read_call_payload(f.ctx.drive_root, task_id="authored-main", call_id=f"authored-send-{len(f.inputs)}")
    assert physical["messages"][0] == after[0]
    after = physical["messages"]  # custody metadata is stripped at the actual physical seam
    assert f.mark_args["text"] not in extract_plain_text_from_content(before[0]["content"])
    for physical in (f.inputs[1]["messages"], after):
        system = extract_plain_text_from_content(physical[0]["content"])
        assert f.mark_args["text"] in system and f.mark_args["quote"] in system
        assert f.source_record["id"] in system and '"task_id":"authored-main"' in system
    # The mark is resident because of its explicit state, not a pinned tool pair
    # or a helper that happened to repeat its meaning.
    assert not any(row.get("tool_call_id") == "mark" for row in after)
    assert "Stable shared interpretation." in after[0]["content"][1]["text"]
    assert after[0]["content"][1] == before[0]["content"][1]  # shared cache prefix unchanged
    assert len(f.store.active_marks("1")) == 1
    if compaction == "authored":
        assert f.ctx._context_view_receipt["status"] == "applied"
    else:
        assert summaries and any("promise must remain" in part.text for part in summaries)
        assert any(row.get("tool_call_id") == "fresh-read" for row in after)
    assert "prompt_prefix_breaks" not in usage


def test_mark_view_choice_and_release_apply_before_next_physical_send(marked_main):
    f, saved = marked_main, {}
    from ouroboros.owner_wait import continuation_state, restore_continuation_state

    def reduce(kwargs):
        mark = json.loads(next(row["content"] for row in kwargs["messages"] if row.get("tool_call_id") == "mark"))
        saved["id"] = mark["id"]
        return call("memory_mark", {"mark_id": mark["id"], "visibility": "meaning", "reason": "Use my meaning."}, "reduce")
    def restore_words(_kwargs):
        return call("memory_mark", {"mark_id": saved["id"], "visibility": "full", "reason": "Need exact wording."}, "full")
    def release(_kwargs):
        saved["continuation"] = json.loads(json.dumps(continuation_state(f.ctx, f.ctx.messages, {}, {}, 4, [], set())))
        return call("memory_mark", {"release_id": saved["id"], "reason": "The promise is fulfilled."}, "release")
    f.run([call("memory_mark", f.mark_args, "mark"), reduce, restore_words, release, {"content": "done"}])
    systems = [extract_plain_text_from_content(row["messages"][0]["content"]) for row in f.inputs]
    assert f.mark_args["quote"] in systems[1] and f.mark_args["quote"] not in systems[2]
    assert f.mark_args["text"] in systems[2] and f.source_record["id"] in systems[2]
    assert f.mark_args["quote"] in systems[3] and saved["id"] not in systems[4]
    assert f.store.get(saved["id"])["quote"] == f.mark_args["quote"] and f.store.active_marks("1") == []
    restore_continuation_state(f.registry, saved["continuation"], f.messages, {}, {}, set())
    assert saved["id"] in f.ctx.context_fit_plan.chronicle_state_json
    f.run([{"content": "continued"}])
    assert saved["id"] not in extract_plain_text_from_content(f.inputs[-1]["messages"][0]["content"])
    assert json.loads(f.ctx.context_fit_plan.chronicle_state_json)["marks"] == []


@pytest.mark.parametrize("failure", [OSError, sqlite3.DatabaseError])
def test_unavailable_mark_source_keeps_captured_marks_and_recovers(marked_main, monkeypatch, failure):
    f, original = marked_main, ChronicleStore.active_marks
    def unavailable(*_a, **_k):
        raise failure("fixture unavailable")
    def fail_reads(_kwargs):
        monkeypatch.setattr(ChronicleStore, "active_marks", unavailable)
        return call("read_file", {"path": "evidence.txt"}, "first")
    def recover(_kwargs):
        monkeypatch.setattr(ChronicleStore, "active_marks", original)
        return call("read_file", {"path": "evidence.txt"}, "second")
    f.run([call("memory_mark", f.mark_args, "mark"), fail_reads, recover, {"content": "done"}])
    failed = extract_plain_text_from_content(f.inputs[2]["messages"][0]["content"])
    recovered = extract_plain_text_from_content(f.inputs[3]["messages"][0]["content"])
    assert f.mark_args["text"] in failed and f.mark_args["quote"] in failed
    assert "active_marks_unavailable" in failed and failure.__name__ in failed
    assert f.mark_args["text"] in recovered and "active_marks_unavailable" not in recovered


def test_unchanged_marks_preserve_prefix_and_do_not_refresh_raw_sources(marked_main, monkeypatch):
    f = marked_main
    def no_room_read(*_a, **_k):
        pytest.fail("The ongoing mark refresh must not reread room records")
    monkeypatch.setattr(ChronicleStore, "room_records", no_room_read)
    f.run([call("memory_mark", f.mark_args, "mark"),
           call("read_file", {"path": "evidence.txt"}, "read"), {"content": "done"}])
    assert f.inputs[1]["messages"][0] == f.inputs[2]["messages"][0]
    events = [row["data"] for row in list(f.events.queue) if isinstance(row.get("data"), dict)]
    assert len([row for row in events if row.get("checkpoint_kind") == "context_marks_refreshed"]) == 1
    assert [row["sanctioned_by"] for row in events if row.get("checkpoint_kind") == "prompt_prefix_break"] == ["memory_marks"]


def test_mark_survives_real_fallback_that_can_fit_max(marked_main, monkeypatch):
    import httpx
    from ouroboros import context, usage_accounting
    from ouroboros.observability import read_call_payload

    f = marked_main
    primary = f.ctx.context_fit_plan.model
    # Distinct physical book bytes prove the fallback kept Max, independently
    # of the plan's optional rendered_mode override (empty means preferred).
    resident_book = "Complete Max reference book stays resident."
    templates = dict(f.ctx.context_fit_plan.system_templates_json)
    max_blocks = json.loads(templates["max"])
    max_blocks[0]["text"] += "\n" + resident_book
    templates["max"] = json.dumps(max_blocks)
    f.ctx.context_fit_plan = replace(f.ctx.context_fit_plan, system_templates_json=templates)
    f.messages = f.ctx.context_fit_plan.messages_for("max")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai/alternate")
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **kw: (
        {"model": task["model"], "provider": "openai"},
        SimpleNamespace(route_fp="fallback-route", status="confirmed", stale=False, window_tokens=900_000)))
    overflow = httpx.HTTPStatusError("context_length_exceeded", request=httpx.Request("POST", "https://fixture.invalid"),
        response=httpx.Response(400, json={"error": {"message": "context_length_exceeded"}}))
    answer, _, _ = f.run([call("memory_mark", f.mark_args, "mark"), overflow, {"content": "done"}])
    assert answer == "done"
    assert [request["model"] for request in f.inputs] == [primary, primary, "openai/alternate"]
    for request in f.inputs[1:]:
        system = extract_plain_text_from_content(request["messages"][0]["content"])
        assert f.mark_args["text"] in system and f.mark_args["quote"] in system and f.source_record["id"] in system
        assert resident_book in system
    _, physical, _ = read_call_payload(f.ctx.drive_root, task_id="authored-main", call_id="authored-send-3")
    system = extract_plain_text_from_content(physical["messages"][0]["content"])
    assert resident_book in system
    assert f.mark_args["text"] in system and f.mark_args["quote"] in system and f.source_record["id"] in system
    assert f.ctx.context_fit_plan.preferred_mode == "max" and f.ctx.active_context_mode == "max"
    assert usage_accounting.last_physical_attempt_capture().physical_context.rendered_mode == "max"


def test_account_reprepare_refreshes_marks_before_its_physical_candidate(marked_main, monkeypatch):
    from ouroboros import context
    from tests.test_loop_compaction import _ctx

    f = marked_main
    ctx = _ctx(f.ctx.drive_root)
    ctx.context_fit_plan = f.ctx.context_fit_plan
    ctx.active_model = ctx.context_fit_plan.model
    ctx.tools, ctx.messages, ctx.round_idx = f.registry, f.messages, 2
    mark = json.loads(_memory_mark(f.ctx, **f.mark_args))
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **kw: (
        {"model": task["model"], "provider": "openai"},
        SimpleNamespace(route_fp="rebound-account", status="confirmed", stale=False, window_tokens=800_000)))
    prepared = loop_model_call._reprepare_waiting_main(ctx, {"model": ctx.active_model, "model_role": "main",
                                                            "model_account_override": "changed-account"})
    assert mark["id"] in str(prepared.kwargs["messages"])
    f.messages = prepared.kwargs["messages"]
    f.run([{"content": "continued"}])
    from ouroboros.observability import read_call_payload
    _, physical, _ = read_call_payload(f.ctx.drive_root, task_id="authored-main", call_id="authored-send-1")
    system = extract_plain_text_from_content(physical["messages"][0]["content"])
    assert f.mark_args["text"] in system and f.mark_args["quote"] in system
    assert f.ctx.context_fit_plan.route_fp == "rebound-account"


def test_oversized_mark_is_disclosed_and_declared_inputs_remain_isolated(marked_main, monkeypatch):
    from tests.test_loop_compaction import _ctx
    f = marked_main
    mark = json.loads(_memory_mark(f.ctx, **f.mark_args))
    ctx = _ctx(f.ctx.drive_root)
    ctx.context_fit_plan = replace(f.ctx.context_fit_plan, window_tokens=100)
    ctx.active_model = ctx.context_fit_plan.model
    ctx.tools = f.registry
    ctx.messages = ctx.context_fit_plan.messages_for("max")
    fit = loop_model_call._measure_round_main_fit(ctx, automatic_pass_used=False)
    assert fit.predicted_capacity_miss and fit.measurement.capacity_deficit_tokens > 0
    assert f.mark_args["text"] in str(ctx.messages) and f.mark_args["quote"] in str(ctx.messages)
    assert ctx.context_fit_plan.max_projection.memory_facts["target_miss"]
    frozen = deepcopy(ctx.messages)
    ctx.context_fit_plan = replace(ctx.context_fit_plan, chronicle_state_json="")
    monkeypatch.setattr(ChronicleStore, "active_marks", lambda *_: pytest.fail("Declared inputs must not open global marks"))
    loop_model_call._measure_round_main_fit(ctx, automatic_pass_used=False)
    assert ctx.messages == frozen and mark["id"] not in ctx.context_fit_plan.chronicle_state_json


def test_external_task_marks_use_the_tools_canonical_source_authority(marked_main):
    f = marked_main
    canonical = f.ctx.drive_root.parent / "shared-memory"
    accounting = f.ctx.drive_root.parent / "accounting-root"
    store = ChronicleStore(canonical)
    source = store.append_episode("1", f.mark_args["quote"], [], {"kind": "mind"})
    f.ctx.budget_drive_root = str(accounting)
    f.ctx.task_metadata["budget_drive_root"] = str(canonical)
    f.ctx.task_metadata["workspace_mode"] = "external"
    f.mark_args["node_id"] = source["id"]
    f.run([call("memory_mark", f.mark_args, "mark"), {"content": "done"}])
    assert len(store.active_marks("1")) == 1
    assert f.mark_args["text"] in extract_plain_text_from_content(f.inputs[-1]["messages"][0]["content"])
    assert not (accounting / "memory/chronicle").exists()
    assert f.store.active_marks("1") == []


def test_declared_cyber_actor_can_mark_without_automatic_shared_injection(marked_main, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from tests.test_acting_subagents import _enable_cyber_mode_for_test
    from tests.test_context_fit_integration import _plan

    f = marked_main
    _enable_cyber_mode_for_test(monkeypatch)
    f.ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="external_workspace",
                                           write_root=str(f.ctx.repo_dir))
    f.ctx.task_metadata.update(delegation_role="subagent", task_contract={"input_sources": "declared"})
    f.ctx.task_contract = {"input_sources": "declared"}
    # The real declared builder has no chronicle snapshot or render marker;
    # its selected question may still explicitly name an exact memory source.
    f.ctx.context_fit_plan = replace(_plan(preferred="max", window=900_000),
        user_content_json=json.dumps({"source": f.source_record["id"], "quote": f.mark_args["quote"]}),
        context_task=f.ctx.task_metadata)
    f.messages = f.ctx.context_fit_plan.messages_for("max")
    assert f.registry.get_schema_by_name("memory_mark") is not None
    f.run([call("memory_mark", f.mark_args, "mark"),
           call("compact_context", {"working_note": "The declared source was read.", "keep_unit_ids": []}, "compact"),
           {"content": "done"}])
    assert len(f.store.active_marks("1")) == 1  # explicit write remains a real capability
    physical = f.inputs[-1]["messages"]
    assert not any(row.get("tool_call_id") == "mark" for row in physical)
    assert f.mark_args["text"] not in str(physical)
    assert f.ctx.context_fit_plan.chronicle_state_json == ""  # declared inputs stay declared


@pytest.mark.parametrize("fence", ["_last_llm_error_kind", "_transport_deaths"])
def test_unsettled_attempt_keeps_its_frozen_mark_view(marked_main, fence):
    from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY
    from tests.test_loop_compaction import _ctx
    f = marked_main
    mark = json.loads(_memory_mark(f.ctx, **f.mark_args))
    ctx = _ctx(f.ctx.drive_root)
    ctx.context_fit_plan, ctx.tools = f.ctx.context_fit_plan, f.registry
    ctx.active_model, ctx.messages = ctx.context_fit_plan.model, f.messages
    key = TRANSPORT_DEATHS_KEY if fence == "_transport_deaths" else fence
    ctx.accumulated_usage[key] = {"count": 1} if key == TRANSPORT_DEATHS_KEY else "provider_outcome_unknown"
    frozen, plan = deepcopy(ctx.messages), ctx.context_fit_plan
    loop_model_call._measure_round_main_fit(ctx, automatic_pass_used=False)
    assert ctx.messages == frozen and ctx.context_fit_plan is plan
    ctx.accumulated_usage.pop(key)
    loop_model_call._measure_round_main_fit(ctx, automatic_pass_used=False)
    assert mark["id"] in ctx.context_fit_plan.chronicle_state_json
    assert f.mark_args["text"] in str(ctx.messages)

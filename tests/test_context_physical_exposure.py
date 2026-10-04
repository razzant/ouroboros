"""Physical source exposure, restore handles and author attribution survive rewriting."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from ouroboros import context_compaction as cc
from tests.test_context_reclaim_materializer import _SPEC, _request


def source(call_id="read", text="exact evidence " * 500):
    return [{"role": "assistant", "content": "I inspected the source", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "x.py"}'}}]},
        {"role": "tool", "tool_call_id": call_id, "content": text}]


@pytest.mark.parametrize("dialect", ["function", "custom", "native", "gigachat", "claudexor"])
def test_complete_source_survives_real_transport_syntax(dialect):
    from ouroboros.llm import LLMClient

    canonical = source()
    client = LLMClient.__new__(LLMClient)
    if dialect == "custom":
        from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom, project_messages_for_openai_custom
        catalog = project_function_tools_to_openai_custom([{"type": "function", "function": {
            "name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}])
        physical = project_messages_for_openai_custom(canonical, catalog)
    elif dialect == "native":
        _system, physical = client._build_anthropic_messages(canonical)
    elif dialect == "gigachat":
        physical = client._gigachat_messages(canonical)
    elif dialect == "claudexor":
        from ouroboros.llm_messages import project_declared_system_prefix
        physical = project_declared_system_prefix({"provider": "claudexor"}, canonical)
    else:
        physical = canonical
    unit = cc._atomic_units(canonical)[0]
    assert cc.exposed_context_units(canonical, physical) == ({"unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256},)
    altered = copy.deepcopy(canonical)
    altered[1]["content"] = "a new result reusing the same call id"
    assert cc.exposed_context_units(altered, physical) == ()


def test_unconsumed_new_result_is_not_paid_for_or_replaced(tmp_path, monkeypatch):
    observed = source()
    current = observed + source("read", "new result " * 500)
    calls = []
    monkeypatch.setattr(cc, "_summarizer_spec", lambda: dict(_SPEC))
    monkeypatch.setattr(cc, "_call_summarizer", lambda parts, **_: calls.extend(parts) or {
        part.source_id: "The actor inspected the original evidence." for part in parts})
    exposed = cc.exposed_context_units(observed, observed)
    candidate, receipt, _ = cc.compact_tool_history_llm(
        current, request=_request(current, 20_000), drive_root=tmp_path, task_id="exposure",
        exposed_units=exposed,
    )
    assert receipt.status == "applied" and len(calls) == 1
    assert candidate[1:] == current[2:]
    assert candidate[0]["role"] == "user"
    assert cc._capsule_metadata(candidate[0])[1]["authorship"] == "helper"
    assert "Host memory record" in candidate[0]["content"][0]["text"]
    calls.clear()
    unchanged, receipt, usage = cc.compact_tool_history_llm(
        current, request=_request(current, 20_000), drive_root=tmp_path, task_id="exposure",
        exposed_units=[],
    )
    assert unchanged is current and receipt.status == "no_eligible" and usage is None and calls == []


def test_result_bytes_do_not_prove_the_wrong_call_received_them():
    canonical = source("a", "first result") + source("b", "second result")
    canonical[2]["tool_calls"][0]["function"]["arguments"] = '{"path":"other.py"}'
    physical = copy.deepcopy(canonical)
    physical[1]["content"], physical[3]["content"] = physical[3]["content"], physical[1]["content"]
    assert cc.exposed_context_units(canonical, physical) == ()


def test_observation_reads_exact_physical_source_and_inspection_restores_it(tmp_path):
    from ouroboros.model_send_seal import persist_physical_candidate
    from ouroboros.tools.compact_context import record_context_view, _compact_context

    canonical = source() + source("new", "not sent")
    persisted = persist_physical_candidate(tmp_path, task_id="exposure", attempt_id="attempt-one",
        candidate={"messages": canonical[:2]}, candidate_facts={})
    capture = SimpleNamespace(candidate_manifest_ref=persisted["manifest_ref"], attempt_id="attempt-one")
    ctx = SimpleNamespace(task_id="exposure", drive_root=tmp_path, budget_drive_root=tmp_path)
    record_context_view(ctx, canonical, [], physical_capture=capture)
    observed = ctx._last_context_observation
    assert observed["physical_source_status"] == "observed_projection"
    assert len(observed["exposed_units"]) == 1
    inspected = json.loads(_compact_context(ctx, inspect=True))
    assert [unit["physically_exposed"] for unit in inspected["units"]] == [True, False]
    restore = inspected["units"][0]["restore_ref"]
    messages, _ = cc._restored_source_views([restore], drive_root=tmp_path, task_id="exposure",
                                          request=_request([], 0))
    assert messages[0]["role"] == "user" and "exact evidence" in messages[0]["content"][0]["text"]
    assert cc._capsule_metadata(messages[0])[1]["retention"] == "source_view"
    record_context_view(ctx, canonical, [], physical_capture=None)
    assert ctx._last_context_observation["exposed_units"] == []


def test_exposure_restores_through_one_continuation(tmp_path):
    """Adapted from PR #1449: the context-fit plan half belongs to the memory frame and is
    not part of this patch; the exposure observation must survive an owner-wait resume."""
    from ouroboros.owner_wait import continuation_state, restore_continuation_state
    from ouroboros.tools.compact_context import record_context_view
    from tests.test_owner_wait import context

    ctx = context(tmp_path)
    messages = source()
    record_context_view(ctx, messages, [], physical_capture=None)
    ctx._last_context_observation["exposed_units"] = list(cc.exposed_context_units(messages, messages))
    state = json.loads(json.dumps(continuation_state(ctx, messages, {}, {}, 3, [], set())))
    restored = context(tmp_path)
    assert getattr(restored, "_last_context_observation", None) is None
    restore_continuation_state(SimpleNamespace(_ctx=restored), state, [], {}, {}, set())
    assert restored._last_context_observation == ctx._last_context_observation
    assert restored._last_context_observation["exposed_units"]


def test_inspected_child_handle_is_readable_now_and_retained_canonically(tmp_path):
    import shutil
    from dataclasses import asdict
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tool_access_roots import resource_root_path
    from ouroboros.tools.compact_context import _compact_context, record_context_view

    canonical, child = tmp_path / "canonical", tmp_path / "child"
    ctx = SimpleNamespace(task_id="child", drive_root=child, budget_drive_root=canonical)
    record_context_view(ctx, source(), [])
    inspected = json.loads(_compact_context(ctx, inspect=True))
    ref = inspected["units"][0]["restore_ref"]["checkpoint_ref"]
    own = read_actor_source_bytes(child, "child", ref)
    assert read_actor_source_bytes(canonical, "child", ref) == own
    assert (resource_root_path(ctx, ref["root"]) / ref["path"]).read_bytes() == own
    relocated = tmp_path / "relocated-child"
    shutil.move(child, relocated)
    assert read_actor_source_bytes(relocated, "child", ref) == own
    restored, _ = cc._restored_source_views([inspected["units"][0]["restore_ref"]],
        drive_root=relocated, task_id="child", request=_request([], 0))
    assert "exact evidence " * 500 in restored[0]["content"][0]["text"]
    original = source()
    request = replace(_request(original, 0), working_note="I kept the evidence.",
        expected_view_revision=cc.context_reclaim_transcript_sha256(original), keep_unit_ids=(),
        restore_unit_refs=(inspected["units"][0]["restore_ref"],))
    authored, receipt, _ = cc.compact_tool_history_llm(original, request=request,
        drive_root=relocated, task_id="child", observed_messages=original, observed_tool_schemas=[],
        fit_candidate=lambda *_args: {"accepted": True})
    assert receipt.status == "applied" and receipt.restored_unit_refs[0]["checkpoint_ref"]["read"]
    from ouroboros.review_source_closure import retain_review_refs
    retained = tmp_path / "review-custody"
    history = {"messages": authored, "view_receipt": asdict(receipt)}
    retained_history = retain_review_refs(history, relocated, retained, "child", carrier="checkpoint")
    assert retained_history == history  # Source/capsule identity is byte-stable during promotion.
    shutil.rmtree(relocated)
    exact = read_actor_source_bytes(retained, "child", receipt.restored_unit_refs[0]["checkpoint_ref"])
    assert exact == own
    for field, value in (("kind", "unowned"), ("sha256", "0" * 64)):
        bad = copy.deepcopy(inspected["units"][0]["restore_ref"])
        bad["checkpoint_ref"][field] = value
        _, refused, _ = cc.compact_tool_history_llm(original,
            request=replace(request, restore_unit_refs=(bad,)), drive_root=retained, task_id="child",
            observed_messages=original, fit_candidate=lambda *_args: {"accepted": True})
        assert refused.status == "source_unavailable"


def test_legacy_capsule_remains_readable_after_helper_attribution_change(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_summarizer_spec", lambda: dict(_SPEC))
    monkeypatch.setattr(cc, "_call_summarizer", lambda parts, **_: {p.source_id: "A summary" for p in parts})
    current = source()
    candidate, receipt, _ = cc.compact_tool_history_llm(current, request=_request(current, 500),
        drive_root=tmp_path, task_id="legacy")
    assert receipt.status == "applied"
    legacy = copy.deepcopy(candidate[0])
    legacy["role"] = "assistant"
    meta = legacy["content"][0]["_context_capsule"]
    meta.pop("authorship")
    meta["summary_contract_digest"] = cc._LEGACY_SUMMARY_CONTRACT_DIGEST
    assert cc._capsule_metadata(legacy)[1] is not None


@pytest.mark.parametrize("boundary", [None, "Owner changed the goal", "[Control] Preserve the pending decision"])
def test_automatic_range_record_preserves_boundaries_and_each_original_restore_ref(tmp_path, monkeypatch, boundary):
    from ouroboros.tools.compact_context import _compact_context, record_context_view

    messages = source("left", "LEFT_EXACT_SOURCE " * 500)
    if boundary:
        messages.append({"role": "user", "content": boundary})
    messages += source("right", "RIGHT_EXACT_SOURCE " * 500)
    calls = []
    monkeypatch.setattr(cc, "_summarizer_spec", lambda: dict(_SPEC))
    monkeypatch.setattr(cc, "_call_summarizer", lambda parts, **_: calls.extend(parts) or {
        part.source_id: "The actor inspected the complete source." for part in parts})
    candidate, receipt, _ = cc.compact_tool_history_llm(
        messages, request=_request(messages, 100_000), drive_root=tmp_path, task_id="ranges",
        automatic_deficit_tokens=1, exposed_units=cc.exposed_context_units(messages, messages))
    assert receipt.status == "applied" and len(calls) == 2
    capsules = [message for message in candidate if cc._capsule_metadata(message)[1]]
    assert len(capsules) == (2 if boundary else 1)
    assert all(cc._capsule_metadata(message)[1]["generation"] == 1 for message in capsules)
    if boundary:
        assert candidate[1] == {"role": "user", "content": boundary}
        assert all(boundary not in part.text for part in calls)

    ctx = SimpleNamespace(task_id="ranges", drive_root=tmp_path, budget_drive_root=tmp_path)
    record_context_view(ctx, candidate, [])
    inspected = json.loads(_compact_context(ctx, inspect=True))
    refs = [ref for unit in inspected["units"] for ref in unit["source_refs"]
            if all(key in ref for key in ("checkpoint_ref", "unit_id", "raw_sha256"))]
    assert len(refs) == 2 and len({ref["unit_id"] for ref in refs}) == 2
    restored, _ = cc._restored_source_views(refs, drive_root=tmp_path, task_id="ranges",
                                          request=_request(candidate, 0))
    assert "LEFT_EXACT_SOURCE " * 500 in restored[0]["content"][0]["text"]
    assert "RIGHT_EXACT_SOURCE " * 500 in restored[1]["content"][0]["text"]

    calls.clear()
    exposed = cc.exposed_context_units(candidate, candidate)
    unchanged, skipped, usage = cc.compact_tool_history_llm(
        candidate, request=_request(candidate, 100_000), drive_root=tmp_path, task_id="ranges",
        automatic_deficit_tokens=1, exposed_units=exposed)
    assert unchanged is candidate and skipped.status == "no_eligible" and usage is None and calls == []
    authored_request = replace(_request(candidate, 0), working_note="I revised my interpretation.",
        expected_view_revision=cc.context_reclaim_transcript_sha256(candidate),
        keep_unit_ids=(), restore_unit_refs=tuple(refs))
    authored, authored_receipt, _ = cc.compact_tool_history_llm(
        candidate, request=authored_request, drive_root=tmp_path, task_id="ranges",
        observed_messages=candidate, observed_tool_schemas=[], exposed_units=exposed,
        fit_candidate=lambda _messages, _tools: {"accepted": True})
    assert authored_receipt.status == "applied" and calls == []
    assert any(message["role"] == "assistant" and "I revised my interpretation." in str(message["content"])
               for message in authored)
    assert all(any(marker in str(message["content"]) for message in authored)
               for marker in ("LEFT_EXACT_SOURCE " * 500, "RIGHT_EXACT_SOURCE " * 500))

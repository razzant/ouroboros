"""An actor selects a working view without changing source or live tool custody."""

import copy
import dataclasses
import json

import pytest

from ouroboros import context_compaction as cc
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context_budget import ContextReclaimRequest


def _unit(call_id, text="source result", names=("read_file",)):
    calls = [{"id": f"{call_id}-{i}", "type": "function",
              "function": {"name": name, "arguments": json.dumps({"path": f"{call_id}.md"})}}
             for i, name in enumerate(names)]
    return [{"role": "assistant", "content": f"Read {call_id}", "tool_calls": calls},
            *[{"role": "tool", "tool_call_id": call["id"], "content": text} for call in calls]]


def _request(observed, current, *, note="I retained the relevant finding.", keep=(), restore=()):
    return ContextReclaimRequest(
        route_fp="test-route", round_id="round", transcript_sha256=cc.context_reclaim_transcript_sha256(current),
        measurement_basis="cold_estimate", measurement_density=1.0, reclaim_goal_tokens=100,
        working_note=note, expected_view_revision=cc.context_reclaim_transcript_sha256(observed),
        keep_unit_ids=keep, restore_unit_refs=restore,
    )


def _fit(messages, tools):
    return {"accepted": True, "message_count": len(messages), "tools_count": len(tools),
            "strict_bound_proven": False}


def _apply(observed, current, tmp_path, **kwargs):
    request = kwargs.pop("request", None) or _request(observed, current, **kwargs)
    return cc.compact_tool_history_llm(current, request=request, observed_messages=observed,
                                       drive_root=tmp_path, task_id="actor-view", fit_candidate=_fit)


@pytest.fixture(autouse=True)
def no_actor_model_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the authored path must not enter the helper/model path")
    monkeypatch.setattr(cc, "_summarizer_spec", forbidden)
    monkeypatch.setattr(cc, "_call_summarizer", forbidden)


def test_observed_binding_is_separate_from_current_request_pair_and_owner_tail(tmp_path):
    owner = {"role": "user", "content": "Keep the original owner requirement."}
    observed = [{"role": "system", "content": "Identity"}, *_unit("before", names=("read_file", "search_code")),
                owner, *_unit("keep"), *_unit("after")]
    units = cc._atomic_units(observed)
    own_request = _unit("view-request", "The request was accepted.", names=("compact_context",))
    late_owner = {"role": "user", "content": "New owner correction after the observed snapshot."}
    current = copy.deepcopy(observed) + own_request + [late_owner]
    current_owner = current[4]
    before = copy.deepcopy(current)
    request = _request(observed, current, keep=(units[1].unit_id,))
    candidate, receipt, usage = _apply(observed, current, tmp_path, request=request)
    assert receipt.status == "applied" and usage is None
    assert current == before
    assert candidate[-3:] == [*own_request, late_owner]
    assert candidate[-1] is late_owner and candidate[2] is current_owner
    assert candidate[3:5] == current[5:7]  # the selected retained complete tool unit
    assert sum("_context_capsule" in b for m in candidate for b in m.get("content", []) if isinstance(b, dict)) == 1
    assert json.dumps(candidate, ensure_ascii=False).count("I retained the relevant finding.") == 1
    payload = json.loads(read_actor_source_bytes(tmp_path, "actor-view", receipt.checkpoint_ref))
    assert payload["messages"] == observed
    assert payload["observed_view_revision"] == request.expected_view_revision
    assert payload["request"]["transcript_sha256"] == request.transcript_sha256
    assert request.transcript_sha256 != request.expected_view_revision
    meta = candidate[1]["content"][0]["_context_capsule"]
    assert cc._capsule_metadata(candidate[1])[1] is not None
    exact_units = [ref for ref in meta["source_refs"] if "unit_id" in ref]
    assert {ref["unit_id"] for ref in exact_units} == {units[0].unit_id, units[2].unit_id}
    assert {ref["raw_sha256"] for ref in exact_units} == {units[0].raw_sha256, units[2].raw_sha256}
    assert all(ref["checkpoint_ref"] == receipt.checkpoint_ref for ref in exact_units)


@pytest.mark.parametrize("drift", ["expected_view", "current_binding", "changed_source", "truncated_prefix", "unknown_unit"])
def test_real_binding_mismatch_preserves_current_without_a_checkpoint(tmp_path, monkeypatch, drift):
    observed = _unit("evidence")
    current = copy.deepcopy(observed) + _unit("request", names=("compact_context",))
    request = _request(observed, current)
    if drift == "expected_view":
        request = dataclasses.replace(request, expected_view_revision="0" * 64)
    elif drift == "current_binding":
        request = dataclasses.replace(request, transcript_sha256="0" * 64)
    elif drift == "changed_source":
        current[1]["content"] = "different source"
        request = dataclasses.replace(request, transcript_sha256=cc.context_reclaim_transcript_sha256(current))
    elif drift == "truncated_prefix":
        current[:] = current[:1]
        request = dataclasses.replace(request, transcript_sha256=cc.context_reclaim_transcript_sha256(current))
    else:
        request = dataclasses.replace(request, keep_unit_ids=("not-an-observed-unit",))
    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint", lambda *a, **k: pytest.fail("mismatch checkpoint"))
    candidate, receipt, _ = _apply(observed, current, tmp_path, request=request)
    assert candidate is current and receipt.status == "binding_mismatch"


def test_sealing_is_presentation_but_content_fields_remain_source(tmp_path):
    observed = [{"role": "system", "content": [{"type": "text", "text": "Rules", "cache_control": {"type": "ephemeral"}}]},
                *_unit("evidence")]
    current = copy.deepcopy(observed)
    current[0]["content"][0].pop("cache_control")
    current[2]["content"] = [{"type": "text", "text": observed[2]["content"],
                              "cache_control": {"type": "ephemeral"}}]
    candidate, receipt, _ = _apply(observed, current, tmp_path)
    assert receipt.status == "applied"
    assert candidate[0] is current[0]
    # A real field inside tool arguments named cache_control must never be stripped.
    observed[1]["tool_calls"][0]["function"]["arguments"] = {"cache_control": "source meaning"}
    current[1]["tool_calls"][0]["function"]["arguments"] = {"cache_control": "changed meaning"}
    _, receipt, _ = _apply(observed, current, tmp_path)
    assert receipt.status == "binding_mismatch"


def test_real_task_seal_wrapper_preserves_the_observed_source_binding(tmp_path):
    from ouroboros.context_fit import seal_task_transcript

    observed = [{"role": "system", "content": "rules"}, {"role": "user", "content": "Owner task"}, *_unit("source")]
    current = copy.deepcopy(observed)
    seal_task_transcript(current)
    assert current[1]["content"] == [{"type": "text", "text": "Owner task", "cache_control": {"type": "ephemeral"}}]
    candidate, receipt, _ = _apply(observed, current, tmp_path)
    assert receipt.status == "applied" and candidate[1] is current[1]


@pytest.mark.parametrize("position,extra", [(0, {"citation": "keep this meaning"}), (1, {})])
def test_only_plain_first_user_wrapper_can_be_normalized(tmp_path, position, extra):
    observed = [{"role": "user", "content": "first"}, {"role": "user", "content": "later"}, *_unit("source")]
    current = copy.deepcopy(observed)
    current[position]["content"] = [{"type": "text", "text": observed[position]["content"], **extra}]
    candidate, receipt, _ = _apply(observed, current, tmp_path)
    assert candidate is current and receipt.status == "binding_mismatch"


@pytest.mark.parametrize("same_schemas", [True, False, None])
def test_whole_view_noop_requires_unchanged_selected_schemas_without_new_checkpoint(tmp_path, monkeypatch, same_schemas):
    observed = _unit("retained")
    current = copy.deepcopy(observed) + _unit("request", names=("compact_context",))
    schemas = [{"type": "function", "function": {"name": "read_file"}}]
    observed_schemas = schemas if same_schemas else [] if same_schemas is False else None
    request = dataclasses.replace(_request(observed, current, note="", keep=None), schema_names=("read_file",))
    seen = []
    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint", lambda *a, **k: pytest.fail("unchanged messages checkpoint"))

    def fit(messages, tools):
        seen.append((messages, tools))
        return {"accepted": True, "strict_bound_proven": False}

    candidate, receipt, _ = cc.compact_tool_history_llm(
        current, request=request, observed_messages=observed, observed_tool_schemas=observed_schemas,
        tool_schemas=schemas, fit_candidate=fit, drive_root=tmp_path, task_id="actor-view")
    assert candidate is current and receipt.checkpoint_ref is None
    assert receipt.status == ("no_op" if same_schemas is True else "applied")
    assert receipt.before_transcript_sha256 == receipt.after_transcript_sha256
    assert seen == [(current, schemas)]


def test_repeated_authored_view_is_byte_identical_without_checkpoint_or_cache_changes(tmp_path, monkeypatch):
    observed = _unit("source", "detail" * 1000)
    candidate, first, _ = _apply(observed, copy.deepcopy(observed), tmp_path)
    candidate[0]["content"][0]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    observed = copy.deepcopy(candidate)
    own_request = _unit("repeat", names=("compact_context",))
    current = candidate + own_request
    before = cc.context_reclaim_transcript_sha256(current)
    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint", lambda *a, **k: pytest.fail("no-op checkpoint"))
    repeated, receipt, _ = _apply(observed, current, tmp_path, keep=None)
    assert receipt.status == "no_op" and receipt.checkpoint_ref is None
    assert repeated is current
    assert cc.context_reclaim_transcript_sha256(repeated) == before
    assert repeated[0]["content"][0]["_context_capsule"]["checkpoint_ref"] == first.checkpoint_ref
    assert repeated[0]["content"][0]["cache_control"]["ttl"] == "1h"


def test_focus_can_grow_if_the_complete_candidate_and_tools_fit(tmp_path):
    observed = _unit("short")
    current = copy.deepcopy(observed) + _unit("request", names=("compact_context",))
    tools = [{"type": "function", "function": {"name": "read_file", "description": "large schema" * 100}}]
    checked = []

    def known_fit(messages, schemas):
        checked.append((copy.deepcopy(messages), copy.deepcopy(schemas)))
        assert messages[-2:] == current[-2:]
        assert schemas == tools
        # This is a deterministic test witness, not a live route guarantee.
        return {"accepted": len(json.dumps(messages)) + len(json.dumps(schemas)) < 81920,
                "basis": "synthetic_complete_payload", "strict_bound_proven": False}

    request = dataclasses.replace(_request(observed, current, note="More detailed focus. " * 200), schema_names=("read_file",))
    candidate, receipt, usage = cc.compact_tool_history_llm(
        current, request=request, observed_messages=observed, drive_root=tmp_path, task_id="actor-view",
        tool_schemas=tools, fit_candidate=known_fit)
    assert receipt.status == "applied" and usage is None
    assert receipt.reclaimed_tokens == 0 and receipt.goal_reached is False
    assert receipt.fit["basis"] == "synthetic_complete_payload"
    assert receipt.schema_names == ("read_file",)
    assert len(checked) == 1 and checked[0][0] == candidate
    assert len(json.dumps(candidate)) > len(json.dumps(current))


@pytest.mark.parametrize("failure", ["reject", "raise", "missing", "late_owner"])
def test_fit_failure_or_new_owner_turn_keeps_current(tmp_path, failure):
    observed = _unit("source")
    current = copy.deepcopy(observed)
    before = copy.deepcopy(current)

    def fit(messages, tools):
        if failure == "raise":
            raise ValueError("fit unavailable")
        if failure == "late_owner":
            current.append({"role": "user", "content": "A new requirement."})
        return {"accepted": failure == "late_owner"}

    candidate, receipt, _ = cc.compact_tool_history_llm(
        current, request=_request(observed, current), observed_messages=observed, drive_root=tmp_path,
        task_id="actor-view", fit_candidate=None if failure == "missing" else fit)
    assert candidate is current
    assert receipt.status == ("binding_mismatch" if failure == "late_owner" else "fit_rejected")
    assert current[:len(before)] == before
    if failure == "late_owner":
        assert current[-1]["content"] == "A new requirement."


def test_one_note_without_removal_inserts_before_unobserved_tail(tmp_path):
    observed = [{"role": "system", "content": "rules"}, {"role": "user", "content": "task"}]
    current = copy.deepcopy(observed) + _unit("request", names=("compact_context",))
    candidate, receipt, _ = _apply(observed, current, tmp_path, keep=None)
    assert receipt.status == "applied"
    assert candidate[:2] == observed and candidate[3:] == current[2:]
    assert cc._capsule_metadata(candidate[2])[1]["authorship"] == "actor"


def test_restoration_reads_an_exact_checkpoint_unit_as_source_without_protocol_replay(tmp_path):
    observed = _unit("removed", "original source tail")
    candidate, first, _ = _apply(observed, copy.deepcopy(observed), tmp_path)
    unit_ref = next(ref for ref in first.source_refs if "unit_id" in ref)
    observed = copy.deepcopy(candidate)
    current = candidate + _unit("restore", names=("compact_context",))
    restored, receipt, _ = _apply(observed, current, tmp_path, note="Now I inspect the original again.",
                                 keep=None, restore=(unit_ref,))
    assert receipt.status == "applied"
    source_message = next(m for m in restored if (cc._capsule_metadata(m)[1] or {}).get("retention") == "source_view")
    assert source_message.get("tool_calls") is None and source_message["role"] == "user"
    text = source_message["content"][0]["text"]
    assert "read-only projection" in text and "original source tail" in text
    assert "Source reference: " + cc._canonical_json(unit_ref) in text
    assert [m for m in restored if m.get("role") == "tool"] == current[-1:]
    original = json.loads(read_actor_source_bytes(tmp_path, "actor-view", unit_ref["checkpoint_ref"]))
    assert cc._atomic_units(original["messages"])[0].raw_sha256 == unit_ref["raw_sha256"]
    assert receipt.restored_unit_refs == (unit_ref,)


@pytest.mark.parametrize("corruption", ["missing", "wrong_hash", "wrong_unit", "empty_ref", "invalid_checkpoint"])
def test_invalid_restoration_preserves_current(tmp_path, corruption):
    source = _unit("original")
    candidate, first, _ = _apply(source, copy.deepcopy(source), tmp_path)
    ref = copy.deepcopy(next(ref for ref in first.source_refs if "unit_id" in ref))
    if corruption == "missing":
        ref["checkpoint_ref"]["path"] += ".missing"
    elif corruption == "wrong_hash":
        ref["raw_sha256"] = "f" * 64
    elif corruption == "wrong_unit":
        ref["unit_id"] = "unit:0:1:wrong"
    elif corruption == "invalid_checkpoint":
        from ouroboros.artifacts import store_actor_source_bytes
        ref["checkpoint_ref"] = store_actor_source_bytes(
            tmp_path, "actor-view", category="context_checkpoints", source_id="bad-payload", data=b"[]", extension="json")
    else:
        ref = {}
    observed = copy.deepcopy(candidate)
    result, receipt, _ = _apply(observed, candidate, tmp_path, keep=None, restore=(ref,))
    assert result is candidate and receipt.status == "source_unavailable"


def test_current_malformed_and_undescribed_image_units_are_never_removed(tmp_path):
    malformed = _unit("incomplete")[:-1]
    image = _unit("image")
    image[1]["content"] = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    observed = [*malformed, *image, *_unit("ordinary")]
    candidate, receipt, _ = _apply(observed, copy.deepcopy(observed), tmp_path)
    assert receipt.status == "applied"
    assert candidate[:3] == observed[:3]


def test_consumed_anthropic_source_restoration_does_not_reactivate_native_custody(tmp_path):
    from ouroboros.anthropic_native_custody import (
        ANTHROPIC_CONSUMED_RECEIPTS_KEY, ANTHROPIC_NATIVE_RECEIPT_KEY,
        anthropic_tool_unit_active, retain_native_assistant_content,
    )

    turns = _unit("native")
    call = turns[0]["tool_calls"][0]
    turns[0] = retain_native_assistant_content(turns[0], [
        {"type": "thinking", "thinking": "private thought", "signature": "signature"},
        {"type": "tool_use", "id": call["id"], "name": "read_file", "input": {"path": "native.md"}},
    ], {"provider": "anthropic", "base_url": "https://anthropic.example/v1", "resolved_model": "test-model"})
    assert cc._atomic_units(turns) == ()
    # Current active custody stays intact even when another completed unit is removed.
    active = turns + _unit("other")
    kept, receipt, _ = _apply(active, copy.deepcopy(active), tmp_path)
    assert receipt.status == "applied" and kept[:2] == turns
    assert anthropic_tool_unit_active(kept, 0, 1)
    digest = turns[0][ANTHROPIC_NATIVE_RECEIPT_KEY]["content_sha256"]
    consumed = {"role": "assistant", "content": "Already continued.", ANTHROPIC_CONSUMED_RECEIPTS_KEY: [digest]}
    observed = turns + [consumed]
    compacted, first, _ = _apply(observed, copy.deepcopy(observed), tmp_path)
    ref = next(ref for ref in first.source_refs if "unit_id" in ref)
    restored, receipt, _ = _apply(copy.deepcopy(compacted), compacted, tmp_path,
                                 note="Inspect the historical source.", keep=None, restore=(ref,))
    assert receipt.status == "applied"
    assert all(ANTHROPIC_NATIVE_RECEIPT_KEY not in message for message in restored)
    source_view = next(m for m in restored if (cc._capsule_metadata(m)[1] or {}).get("retention") == "source_view")
    assert ANTHROPIC_CONSUMED_RECEIPTS_KEY not in source_view
    assert "private thought" not in json.dumps(source_view)
    assert not any(m.get("tool_calls") or m.get("role") == "tool" for m in restored)
    # Every full private byte is still reachable in the original checkpoint.
    payload = json.loads(read_actor_source_bytes(tmp_path, "actor-view", ref["checkpoint_ref"]))
    assert payload["messages"][0][ANTHROPIC_NATIVE_RECEIPT_KEY] == turns[0][ANTHROPIC_NATIVE_RECEIPT_KEY]


def test_helper_recompaction_keeps_all_actor_source_lineage(tmp_path, monkeypatch):
    original = _unit("first", "a" * 2000) + _unit("second", "b" * 2000)
    authored, receipt, _ = _apply(original, copy.deepcopy(original), tmp_path, note="Detailed understanding. " * 1000)
    old_refs = receipt.source_refs
    monkeypatch.setattr(cc, "_summarizer_spec", lambda: {"model": "fixture", "route_fp": "fixture"})
    monkeypatch.setattr(cc, "_call_summarizer", lambda parts, **kwargs: {p.source_id: "Helper retained the meaning." for p in parts})
    helper_request = ContextReclaimRequest("fixture", "later", cc.context_reclaim_transcript_sha256(authored),
                                          "cold_estimate", 1.0, 100)
    recompressed, helper_receipt, _ = cc.compact_tool_history_llm(
        authored, request=helper_request, drive_root=tmp_path, task_id="actor-view")
    assert helper_receipt.status == "applied"
    meta = cc._capsule_metadata(recompressed[0])[1]
    assert meta is not None and meta["generation"] == 2
    assert all(ref in meta["source_refs"] for ref in old_refs)

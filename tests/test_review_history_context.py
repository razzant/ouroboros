"""Captured review bodies are part of the full request and survive route fitting.

Only synthetic authority, memory and model-route evidence are used. No providers
or current installation data are read; these tests do not qualify publication or
an authored compactor, whose integration is a separate boundary.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from ouroboros import context_fit as cf
from ouroboros.review_history_view import REVIEW_HISTORY_MESSAGE_KEY, REVIEW_CONTEXT_INDEX_KEY, capture_review_history_messages
from tests.test_review_history_view import fixture, source


def authority(*, operative=True):
    history = fixture()
    row = history["rounds"][0]
    review = {"current_attempt": {"fingerprint": "a" * 64, "status": "open"},
              "current_wave": {"wave_artifact": copy.deepcopy(row["source"]["source_ref"]),
                               "aggregate": "REVISE_PLAN", "closed": False},
              "dispute_history": history}
    if operative:
        review["operative_subject"] = {"spec": {"goal": "EXACT CURRENT OPERATIVE SUBJECT"}, "plan_prose": "Current prose"}
    return {"plan_review_authority": review}


def index(messages):
    return json.loads(next(m["content"] for m in messages if REVIEW_CONTEXT_INDEX_KEY in m).partition("\n")[2])["current"]


def bodies(messages):
    return [m for m in messages if REVIEW_HISTORY_MESSAGE_KEY in m]


def core(*, extra=(), memory=False):
    from ouroboros.memory_view import snapshot_json
    from tests import _memory_view_synthetic as syn

    snapshot = syn.snapshot(story=syn.pages(2, 8_000))
    return cf.ContextCore(base_prompt="SYSTEM", bible_md="BIBLE", architecture_md="MAP", development_md="DEV",
        semi_stable_text="IDENTITY", dynamic_text="MANDATORY CURRENT INDEX", user_content_json=json.dumps("ASSIGNMENT"),
        docs_need_development=False, supplementary_messages_json=json.dumps(extra, ensure_ascii=False),
        memory_view_json=snapshot_json(snapshot) if memory else "")


def plan_for(snapshot, monkeypatch, *, preferred="max", window=1_000_000, data_root=None):
    monkeypatch.setattr(cf, "_governance_blocks", lambda _env, _core, *, mode: ("MAX BOOK" if mode == "max" else "BOOK MAP", ""))
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a: 1.0)
    evidence = SimpleNamespace(route_fp="synthetic-route", status="asserted", stale=False, window_tokens=window)
    return cf.build_context_fit_plan(SimpleNamespace(drive_root=data_root), snapshot, {"id": "task", "type": "task"},
        preferred_mode=preferred, tool_schemas=[], route_resolver=lambda *_a, **_k: ({"model": "m", "provider": "p"}, evidence))


def test_capture_splits_before_serialization_with_exact_operatives_and_attachments():
    original = authority()
    saved = copy.deepcopy(original)
    resident, messages = capture_review_history_messages(original, task_id="task")
    assert original == saved
    assert resident["plan_review_authority"]["dispute_history"]["representation"] == "resident_review_index"
    history = index(messages)
    assert history["operative_subject"] == original["plan_review_authority"]["operative_subject"]
    assert history["decision_rows"] == original["plan_review_authority"]["dispute_history"]["decision_rows"]
    assert history["gaps"] == original["plan_review_authority"]["dispute_history"]["gaps"]
    assert history["rounds"][0]["evidence"] == original["plan_review_authority"]["dispute_history"]["rounds"][0]["evidence"]
    fields = {m[REVIEW_HISTORY_MESSAGE_KEY]["binding"]["field"] for m in bodies(messages)}
    assert fields == {"findings", "dispositions", "spec", "plan_prose", "reviewer_outputs[0].text"}
    assert all(m["role"] == "user" and m[REVIEW_HISTORY_MESSAGE_KEY]["task_id"] == "task" for m in bodies(messages))
    assert all(m[REVIEW_HISTORY_MESSAGE_KEY]["visible_sha256"] == hashlib.sha256(m["content"].encode("utf-8")).hexdigest()
               for m in bodies(messages))
    assert "Reviewer rejects the proposed fix" in json.dumps(messages, ensure_ascii=False)


def test_captured_current_author_subject_is_used_instead_of_guessing_latest_wave():
    original = authority(operative=False)
    author = original["plan_review_authority"]["dispute_history"]["current_author_plan"]
    resident, messages = capture_review_history_messages(original, task_id="task")
    assert index(messages)["operative_subject"] == author
    assert any(m[REVIEW_HISTORY_MESSAGE_KEY]["binding"]["field"] == "spec" for m in bodies(messages))
    original["plan_review_authority"]["dispute_history"]["current_author_plan"] = None
    resident, messages = capture_review_history_messages(original, task_id="task")
    assert index(messages)["operative_subject"]["spec"]["goal"] == "Keep the full contract"
    # No exact current reference: retain every old spec/prose rather than pick by cycle.
    original["plan_review_authority"]["current_wave"]["wave_artifact"] = source("unknown", {})
    resident, messages = capture_review_history_messages(original, task_id="task")
    assert "spec" in index(messages)["rounds"][0]
    assert not any(m[REVIEW_HISTORY_MESSAGE_KEY]["binding"]["field"] in {"spec", "plan_prose"} for m in bodies(messages))


def test_no_dispute_capture_is_identical_and_has_no_supplement():
    original = {"task": {"id": "t"}, "plan_review_authority": {"current_attempt": {}}}
    resident, messages = capture_review_history_messages(original, task_id="t")
    assert resident is original and messages == []


@pytest.mark.parametrize("declared", [False, True])
def test_real_capture_threads_the_typed_tail_for_both_input_selections(tmp_path, monkeypatch, declared):
    from ouroboros import context
    from tests.test_doc_context import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    monkeypatch.setattr(context, "_task_authority_projection", lambda *_a: authority())
    task = {"id": "child", "type": "task", "text": "ASSIGNMENT", "delegation_role": "subagent",
            "configured_subagent": {"route": {"kind": "api_model"}},
            "task_contract": {"input_sources": "declared" if declared else "shared"}}
    captured = context._capture_context_core(env, memory, task, None, None)
    tail = json.loads(captured.supplementary_messages_json)
    assert tail and REVIEW_CONTEXT_INDEX_KEY in tail[0] and bodies(tail)
    assert "Reviewer rejects the proposed fix" not in captured.dynamic_text
    assert "EXACT CURRENT OPERATIVE SUBJECT" in json.dumps(tail)
    assert "OWNER DECISION, not disposable transport" in json.dumps(tail)
    assert "All reasons " in json.dumps(tail)
    planned = plan_for(captured, monkeypatch)
    messages = planned.messages_for("max")
    assert messages[1]["role"] == "user" and "ASSIGNMENT" in str(messages[1]["content"])
    assert messages[2:] == tail


def test_direct_runtime_reader_without_a_tail_recipient_stays_full(tmp_path, monkeypatch):
    from ouroboros import context
    from tests.test_doc_context import _make_env_and_memory

    env, _ = _make_env_and_memory(tmp_path)
    monkeypatch.setattr(context, "_task_authority_projection", lambda *_a: authority())
    assert "Reviewer rejects the proposed fix" in context.build_runtime_section(env, {"id": "task"})
    tail = []
    text = context.build_runtime_section(env, {"id": "task"}, supplementary_messages_out=tail)
    assert tail and "Reviewer rejects the proposed fix" not in text
    assert "EXACT CURRENT OPERATIVE SUBJECT" in json.dumps(tail)


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_initial_projection_estimate_counts_full_large_tail_and_frozen_capture_hash(tmp_path, monkeypatch, mode):
    body = {"role": "user", "content": "BIG_REVIEW_BODY " * 20_000,
            REVIEW_HISTORY_MESSAGE_KEY: {"binding": "private receipt " * 1000}}
    empty = plan_for(core(), monkeypatch, preferred=mode, data_root=tmp_path)
    full = plan_for(core(extra=[body]), monkeypatch, preferred=mode, data_root=tmp_path)
    view = full.projection(mode)
    expected = cf._request_tokens(json.loads(view.system_content_json), view.user_content_json or full.user_content_json,
                                  supplementary_messages_json=full.core.supplementary_messages_json)
    assert view.estimated_tokens == expected
    assert view.estimated_tokens > empty.projection(mode).estimated_tokens + 40_000
    assert full.messages_for(mode)[2] == body
    assert full.core_sha256 != empty.core_sha256
    assert len(empty.messages_for(mode)) == 2
    # Transport-only hidden metadata is not counted as text; its binding still
    # changes the captured-source hash without changing the actual input size.
    altered = copy.deepcopy(body)
    altered[REVIEW_HISTORY_MESSAGE_KEY] = {"binding": "different"}
    bound = plan_for(core(extra=[altered]), monkeypatch, preferred=mode, data_root=tmp_path)
    assert bound.core_sha256 != full.core_sha256
    assert bound.projection(mode).estimated_tokens == view.estimated_tokens


def test_mode_projection_and_rebind_preserve_actual_shortened_tail_without_reloading_raw(monkeypatch):
    body = {"role": "user", "content": "BIG_REVIEW_BODY " * 30_000, REVIEW_HISTORY_MESSAGE_KEY: {"binding": "exact"}}
    initial = plan_for(core(extra=[body], memory=True), monkeypatch, window=120_000)
    source_hash = initial.core_sha256
    actual = initial.messages_for("max")[:2] + [{"role": "assistant", "content": "My authored understanding"},
                                               {"role": "user", "content": "Later owner words remain exact"}]
    lowered = initial.reproject_transcript(actual, "low")
    assert lowered[1:] == actual[1:]
    assert "BIG_REVIEW_BODY" not in json.dumps(lowered)
    rebound = initial.reproject_for_route(window_tokens=120_000, known_window=True, ratio=1.0, output_reserve=65_536,
                                          tool_schemas=[], current_messages=actual)
    assert rebound.core_sha256 == source_hash and rebound.core is initial.core
    for mode in ("max", "low", "nano"):
        emitted = rebound.messages_for(mode)
        assert emitted[1:] == actual[1:]
        assert "BIG_REVIEW_BODY" not in json.dumps(emitted)
        assert "MANDATORY CURRENT INDEX" in json.dumps(emitted[0])
        assert rebound.projection(mode).estimated_tokens < initial.projection(mode).estimated_tokens
        assert rebound.reproject_transcript(emitted, mode) == emitted
    # Even a subsequent rebind without a supplied newer transcript keeps the
    # already selected continuation rather than reverting to the initial tail.
    again = rebound.reproject_for_route(window_tokens=130_000, known_window=True, ratio=1.0, output_reserve=65_536, tool_schemas=[])
    assert again.messages_for("max")[1:] == actual[1:]
    assert "F5" in initial.projection("max").memory_facts["floor"]["steps"]
    assert "F5" not in rebound.projection("max").memory_facts["floor"]["steps"]


def test_current_memory_floor_counts_appended_real_turns_not_only_the_initial_assignment(monkeypatch):
    initial = plan_for(core(memory=True), monkeypatch, window=120_000)
    actual = initial.messages_for("max") + [{"role": "tool", "content": "LATER REAL MATERIAL " * 30_000}]
    rebound = initial.reproject_for_route(window_tokens=120_000, known_window=True, ratio=1.0, output_reserve=65_536,
                                          tool_schemas=[], current_messages=actual)
    assert rebound.messages_for("max")[1:] == actual[1:]
    assert rebound.projection("max").estimated_tokens > initial.projection("max").estimated_tokens
    assert "F5" not in initial.projection("max").memory_facts["floor"]["steps"]
    assert "F5" in rebound.projection("max").memory_facts["floor"]["steps"]


def test_estimate_strips_only_top_level_host_receipts_not_nested_same_named_data():
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY

    message = {"role": "user", "content": "Visible words", REVIEW_HISTORY_MESSAGE_KEY: {"receipt": "x" * 80_000},
               TOOL_RESULT_RECORD_KEY: {"receipt": "y" * 80_000}}
    original = copy.deepcopy(message)
    assert cf.estimate_context_prompt_tokens([message]) == cf.estimate_context_prompt_tokens([
        {"role": "user", "content": "Visible words"}])
    nested = {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function",
              "function": {"name": "f", "arguments": {REVIEW_HISTORY_MESSAGE_KEY: "NESTED " * 5000,
                                                          TOOL_RESULT_RECORD_KEY: "ALSO VISIBLE " * 5000}}}]}
    assert cf.estimate_context_prompt_tokens([nested]) > 10_000
    assert message == original


def test_new_capture_changes_supplement_sources_without_mutating_old_snapshot(monkeypatch):
    first = plan_for(core(extra=[{"role": "user", "content": "First full dispute"}]), monkeypatch)
    second = plan_for(dataclasses.replace(first.core, supplementary_messages_json=json.dumps([
        {"role": "user", "content": "Later full dispute"}])), monkeypatch)
    assert first.core_sha256 != second.core_sha256
    assert first.messages_for("max")[-1]["content"] == "First full dispute"
    assert second.messages_for("max")[-1]["content"] == "Later full dispute"

"""Focused contract tests for the unit reader and the mind's authored fold."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import context_compaction as cc
from ouroboros.context_budget import ContextReclaimRequest


def _unit(call_id: str, *, argument: str = "a", result: str = "r") -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": f"reasoning-{call_id}",
            "tool_calls": [{
                "id": call_id,
                "function": {"name": "read_file", "arguments": argument},
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _large_unit(call_id: str, marker: str = "x") -> list[dict]:
    return _unit(
        call_id,
        argument=json.dumps({"content": marker * 4_000 + f"-arg-tail-{call_id}"}),
        result=marker * 4_000 + f"-result-tail-{call_id}",
    )


def _stamp_trace(messages, ref):
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY, make_tool_result_record
    row = messages[-1]
    row[TOOL_RESULT_RECORD_KEY] = make_tool_result_record(
        {"tool_call_id": row["tool_call_id"], "invocation_id": "fixture-invocation", "trace_ref": ref}, row["content"])


def _request(messages: list[dict], *, goal: int = 1_000_000) -> ContextReclaimRequest:
    return ContextReclaimRequest(
        route_fp="main-route",
        round_id="round-7",
        transcript_sha256=cc.context_reclaim_transcript_sha256(messages),
        measurement_basis="fresh_route_usage",
        measurement_density=1.0,
        reclaim_goal_tokens=goal,
        allow_partial_shrink=True,
    )


def _authored(messages: list[dict], *, note: str = "I read the evidence; the conclusion holds.", keep=()) -> ContextReclaimRequest:
    """The mind's own fold of the observed view: a note, the units it keeps raw."""
    return replace(_request(messages), working_note=note,
                   expected_view_revision=cc.context_reclaim_transcript_sha256(messages), keep_unit_ids=keep)


def _fold(messages: list[dict], request: ContextReclaimRequest, tmp_path):
    return cc.compact_tool_history_llm(
        messages, request=request, observed_messages=messages, tool_schemas=[],
        fit_candidate=lambda *_: {"accepted": True}, drive_root=tmp_path, task_id="compaction")


def test_atomic_units_require_all_and_only_contiguous_matching_results():
    complete = _unit("ok", argument="{}", result="done")
    missing = _unit("missing", argument="{}", result="never")[:1]
    delayed = _unit("late", argument="{}", result="late-result")
    duplicate = [{
        "role": "assistant",
        "tool_calls": [
            {"id": "dup", "function": {"name": "x", "arguments": "{}"}},
            {"id": "dup", "function": {"name": "y", "arguments": "{}"}},
        ],
    }, {"role": "tool", "tool_call_id": "dup", "content": "one"}]
    messages = [
        *complete,
        {"role": "user", "content": "hard boundary"},
        *missing,
        {"role": "user", "content": "interrupt"},
        delayed[1],
        *duplicate,
        *_unit("final", argument="{}", result="done"),
    ]

    units = cc._atomic_units(messages)

    assert [(unit.start, unit.end) for unit in units] == [(0, 1), (8, 9)]
    assert all(messages[unit.start]["role"] == "assistant" for unit in units)
    assert messages[2] == {"role": "user", "content": "hard boundary"}


@pytest.mark.parametrize(
    "malformed_call",
    ["not-a-tool-call", {"id": "bad", "function": "not-a-function"}],
)
def test_malformed_tool_call_set_stays_raw_without_crashing(malformed_call):
    messages = [
        {"role": "assistant", "tool_calls": [malformed_call]},
        {"role": "tool", "tool_call_id": "bad", "content": "result"},
    ]

    assert cc._atomic_units(messages) == ()


def test_atomic_source_keeps_complete_arguments_results_and_trace_ref():
    messages = _large_unit("call-complete", "z")
    trace_ref = {"path": "calls/tool.json", "sha256": "d" * 64}
    _stamp_trace(messages, trace_ref)

    unit = cc._atomic_units(
        messages,
        trace_refs_by_tool_call_id={"call-complete": trace_ref},
    )[0]

    assert "-arg-tail-call-complete" in unit.source_text
    assert "-result-tail-call-complete" in unit.source_text
    assert "CONTENT_OMITTED" not in unit.source_text
    assert "LONG_STRING" not in unit.source_text
    assert unit.source_refs == (trace_ref,)


def test_a_request_without_a_note_is_refused_before_any_checkpoint(monkeypatch, tmp_path):
    """Both directions: no helper writes the note, so a note-less request changes nothing and
    writes no checkpoint; the same selection with the mind's note is materialized."""
    messages = _large_unit("note-less")
    before = copy.deepcopy(messages)
    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint",
                        lambda *_a, **_k: pytest.fail("a refused request must not checkpoint"))

    rebuilt, receipt = _fold(messages, _request(messages), tmp_path)

    assert rebuilt is messages and messages == before
    assert receipt.status == "authored_note_required"
    assert receipt.before_transcript_sha256 == receipt.after_transcript_sha256
    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint",
                        lambda *_a, **_k: {"path": "calls/checkpoint.json", "sha256": "c" * 64})
    folded, applied = _fold(messages, _authored(messages), tmp_path)
    assert applied.status == "applied" and len(folded) == 1 and messages == before


def test_an_authored_fold_checkpoints_the_exact_source_before_its_capsule(monkeypatch, tmp_path):
    events: list[str] = []
    messages = _large_unit("ordered")
    _stamp_trace(messages, {"path": "calls/ordered.json", "sha256": "d" * 64})

    def checkpoint(*_args, **_kwargs):
        events.append("checkpoint")
        return {"path": "calls/checkpoint.json", "sha256": "c" * 64}

    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint", checkpoint)
    rebuilt, receipt = _fold(messages, _authored(messages), tmp_path)

    assert events == ["checkpoint"]
    assert receipt.status == "applied"
    assert len(rebuilt) == 1 and rebuilt[0]["role"] == "assistant"
    capsule = rebuilt[0]["content"][0]["_context_capsule"]
    assert rebuilt[0]["content"][0]["text"].startswith("[Actor-authored working view; exact source retained by checkpoint]\n")
    assert capsule["authorship"] == "actor" and capsule["generation"] == 1
    assert capsule["retention"] == "summarized"
    assert capsule["checkpoint_ref"]["sha256"] == "c" * 64
    assert {"path": "calls/ordered.json", "sha256": "d" * 64} in capsule["source_refs"]
    assert capsule["parts"][0]["start_char"] == 0
    assert capsule["parts"][-1]["end_char"] == capsule["source_length_chars"]
    assert receipt.capsule_refs[0]["generation"] == 1


def test_checkpoint_failure_keeps_the_source_raw(monkeypatch, tmp_path):
    messages = _large_unit("checkpoint-fail")
    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint", lambda *_a, **_k: None)

    rebuilt, receipt = _fold(messages, _authored(messages), tmp_path)

    assert rebuilt is messages
    assert receipt.status == "checkpoint_failed"


def test_a_second_fold_over_an_actor_note_keeps_its_lineage_and_a_corrupt_capsule_stays_raw(tmp_path):
    messages = _large_unit("generation", "g")
    first, first_receipt = _fold(messages, _authored(messages, note="First account. " * 200), tmp_path)
    second, second_receipt = _fold(first, _authored(first, note="Merged account."), tmp_path)

    first_meta = first[0]["content"][0]["_context_capsule"]
    second_meta = second[0]["content"][0]["_context_capsule"]
    assert first_receipt.status == second_receipt.status == "applied"
    assert second_meta["generation"] == 2
    assert set(first_meta["source_hashes"]) < set(second_meta["source_hashes"])
    assert all(ref in second_meta["source_refs"] for ref in first_meta["source_refs"])

    corrupt = copy.deepcopy(second)
    corrupt[0]["content"][0]["_context_capsule"]["source_length_chars"] += 1
    assert cc._atomic_units(corrupt) == ()

    corrupt_hash = copy.deepcopy(second)
    corrupt_hash[0]["content"][0]["_context_capsule"]["source_hashes"][0] = "broken"
    assert cc._atomic_units(corrupt_hash) == ()


def test_count_only_compact_context_answers_authored_note_required_and_advertises_no_helper():
    """Both directions: a count without a note queues nothing and names what is missing; the
    same count beside the mind's note queues the authored view."""
    from ouroboros.tools.compact_context import _compact_context, get_tools, record_context_view

    ctx = SimpleNamespace(active_context_mode="max", _active_builtin_tool_result=None)
    record_context_view(ctx, _unit("seen"), [])
    result = _compact_context(ctx, keep_last_n=4)
    description = get_tools()[0].schema["description"]

    assert getattr(ctx, "_pending_compaction", None) is None
    assert "authored_note_required" in result and "working_note" in result
    assert ctx._active_builtin_tool_result.status == "error"
    assert ctx._active_builtin_tool_result.code == "TOOL_ARG_ERROR"
    assert "no helper summarizes for you" in description.lower()
    assert "summaries without a note" not in description.lower()
    assert "raw evidence remains retrievable" in description.lower()
    assert "no information is lost" not in description.lower()
    assert "requested" in _compact_context(ctx, keep_last_n=4, working_note="Four units stay raw.")
    assert isinstance(ctx._pending_compaction, dict) and ctx._pending_compaction["working_note"] == "Four units stay raw."

"""Focused contract tests for complete-input context reclaim."""

from __future__ import annotations

import copy
import json
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


def _install_successful_materializer(monkeypatch, events: list[str] | None = None) -> None:
    monkeypatch.setattr(cc, "_summarizer_spec", lambda: {
        "model": "light-model",
        "resolved_model": "light-model",
        "provider": "test",
        "route_fp": "summary-route",
        "effort": "low",
        "output_budget": 32_768,
        "use_local": False,
    })

    def checkpoint(*_args, **_kwargs):
        if events is not None:
            events.append("checkpoint")
        return {"path": "calls/checkpoint.json", "sha256": "c" * 64}

    def summarize(parts, *, usage_total, **_kwargs):
        if events is not None:
            events.append("summary")
        cc._record_usage(usage_total, {
            "prompt_tokens": len(parts) * 11,
            "completion_tokens": len(parts) * 3,
            "provider": "test",
        })
        return {part.source_id: f"covered {part.sha256}" for part in parts}

    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint", checkpoint)
    monkeypatch.setattr(cc, "_call_summarizer", summarize)


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


@pytest.mark.parametrize(
    ("messages", "status"),
    [
        ([{"role": "user", "content": "only a user turn"}], "no_eligible"),
        (_unit("tiny", argument="{}", result="ok"), "no_positive_reclaim"),
    ],
)
def test_zero_eligible_or_useful_does_no_checkpoint_or_summary(
    monkeypatch, messages, status,
):
    events: list[str] = []
    _install_successful_materializer(monkeypatch, events)
    before = copy.deepcopy(messages)

    rebuilt, receipt, usage = cc.compact_tool_history_llm(
        messages, request=_request(messages, goal=100), negative_memo=set(),
    )

    assert rebuilt is messages
    assert messages == before
    assert receipt.status == status
    assert receipt.before_transcript_sha256 == receipt.after_transcript_sha256
    assert events == []
    assert usage is None


def test_checkpoint_is_after_selection_and_before_summarizer(monkeypatch, tmp_path):
    events: list[str] = []
    _install_successful_materializer(monkeypatch, events)
    messages = _large_unit("ordered")
    _stamp_trace(messages, {"path": "calls/ordered.json", "sha256": "d" * 64})

    rebuilt, receipt, usage = cc.compact_tool_history_llm(
        messages,
        request=_request(messages, goal=100),
        drive_root=tmp_path,
        trace_refs_by_tool_call_id={
            "ordered": {"path": "calls/ordered.json", "sha256": "d" * 64},
        },
        negative_memo=set(),
    )

    assert events == ["checkpoint", "summary"]
    assert receipt.status == "applied"
    assert len(rebuilt) == 1
    capsule = rebuilt[0]["content"][0]["_context_capsule"]
    assert capsule["generation"] == 1
    assert capsule["retention"] == "summarized"
    assert capsule["checkpoint_ref"]["sha256"] == "c" * 64
    assert capsule["source_refs"][0]["sha256"] == "d" * 64
    assert capsule["parts"][0]["start_char"] == 0
    assert capsule["parts"][-1]["end_char"] == capsule["source_length_chars"]
    assert receipt.capsule_refs[0]["generation"] == 1
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 3


def test_checkpoint_failure_keeps_raw_and_makes_no_summarizer_call(monkeypatch):
    messages = _large_unit("checkpoint-fail")
    _install_successful_materializer(monkeypatch)
    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint", lambda *_a, **_k: None)
    monkeypatch.setattr(
        cc, "_call_summarizer",
        lambda *_a, **_k: pytest.fail("summarizer must not run without checkpoint"),
    )

    rebuilt, receipt, usage = cc.compact_tool_history_llm(
        messages, request=_request(messages, goal=100), negative_memo=set(),
    )

    assert rebuilt is messages
    assert receipt.status == "checkpoint_failed"
    assert usage is None


def test_complete_summarizer_payload_has_no_head_or_structural_omission(
    monkeypatch, tmp_path,
):
    from ouroboros import llm, llm_observability

    tail = "TAIL-IS-PRESENT"
    part = cc._part("unit:1:2:abcdef", "head\n" + "middle" * 2_000 + tail)
    seen: dict = {}

    def fake_chat_observed(_client, **kwargs):
        seen.update(kwargs)
        arguments = json.dumps({
            "summaries": [{"source_id": part.source_id, "summary": "complete"}],
        })
        return ({
            "content": "",
            "tool_calls": [{
                "function": {"name": "emit_context_summaries", "arguments": arguments},
            }],
        }, {"prompt_tokens": 17, "completion_tokens": 2, "provider": "test"})

    monkeypatch.setattr(llm, "LLMClient", lambda: object())
    monkeypatch.setattr(llm_observability, "chat_observed", fake_chat_observed)
    usage: dict = {}
    summaries = cc._call_summarizer(
        [part],
        drive_root=tmp_path,
        task_id="task",
        phase="map",
        spec={"model": "m", "effort": "low", "use_local": False},
        summary_budgets={part.root_id: 700},
        usage_total=usage,
    )

    prompt = seen["messages"][0]["content"]
    payload = json.loads(prompt[prompt.index('[{"source_id"'):])
    assert payload[0]["content"] == part.text
    assert payload[0]["summary_budget_tokens"] == 700
    assert tail in prompt
    assert "CONTENT_OMITTED" not in prompt
    assert "LONG_STRING" not in prompt
    assert summaries == {part.source_id: "complete"}
    assert seen["max_tokens"] == 32_768
    assert seen["model"] == "m"
    assert seen["reasoning_effort"] == "low"
    assert usage["prompt_tokens"] == 17


def test_structured_summary_requires_exactly_one_tool_call():
    payload = json.dumps({
        "summaries": [{"source_id": "source", "summary": "complete"}],
    })
    call = {
        "function": {
            "name": "emit_context_summaries",
            "arguments": payload,
        },
    }

    assert cc._parse_structured_summaries({"tool_calls": [call]}) == {
        "source": "complete",
    }
    assert cc._parse_structured_summaries({"tool_calls": [call, call]}) == {}


def test_valid_capsule_recompacts_and_corrupt_capsule_stays_raw(monkeypatch, tmp_path):
    _install_successful_materializer(monkeypatch)
    messages = _large_unit("generation", "g")
    calls = 0

    def summaries(parts, *, usage_total, **_kwargs):
        nonlocal calls
        calls += 1
        text = "S" * 3_000 if calls == 1 else "small second-generation summary"
        return {part.source_id: text for part in parts}

    monkeypatch.setattr(cc, "_call_summarizer", summaries)
    first, first_receipt, _ = cc.compact_tool_history_llm(
        messages, request=_request(messages, goal=100), drive_root=tmp_path,
        negative_memo=set(),
    )
    second, second_receipt, _ = cc.compact_tool_history_llm(
        first, request=_request(first, goal=100), drive_root=tmp_path,
        negative_memo=set(),
    )

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


def test_legacy_omission_api_is_deleted():
    source = cc.pathlib.Path(cc.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "_SUMMARY_INPUT_LIMIT", "_excerpt_for_summary", "_compact_argument_value",
        "LONG_STRING", "CONTENT_OMITTED", "def compact_tool_history(",
        "_round_has_protected_content", "_tool_round_spans",
    ):
        assert forbidden not in source


def test_manual_compact_context_copy_is_truthful_about_summary_and_provenance():
    from ouroboros.tools.compact_context import _compact_context, get_tools

    ctx = SimpleNamespace()
    result = _compact_context(ctx, keep_last_n=4)
    description = get_tools()[0].schema["description"]

    assert ctx._pending_compaction == 4
    assert "active-context" in result
    assert "summaries" in result
    assert "checkpoint/CAS provenance" in result
    assert "raw evidence remains retrievable" in description.lower()
    assert "no information is lost" not in description.lower()

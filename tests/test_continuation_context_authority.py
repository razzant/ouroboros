"""Focused continuity-source, narrative, and Main projection regressions."""

from __future__ import annotations

import copy
import json

import pytest


def _ref(task_id: str) -> dict:
    return {"kind": "task_result", "task_id": task_id, "reader": "get_task_result"}


def _source(task_id: str) -> dict:
    return {
        **_ref(task_id),
        "arguments": {"task_id": task_id, "include_authority": True},
    }


def _summary(root, task_id: str, text: str) -> None:
    from ouroboros.project_dialogue import append_canonical_task_summary

    ref = _ref(task_id)
    assert append_canonical_task_summary(root, {
        "type": "task_summary",
        "summary_kind": "authored_root_summary",
        "summary_id": f"task-narrative:{task_id}",
        "task_id": task_id,
        "result_ref": ref,
        "source_coverage": {"task_result": ref},
        "text": text,
    })


@pytest.mark.parametrize("result", ["Use the existing tower", "r" * 20_000, "\x00" * 6_000])
def test_helper_brief_keeps_the_answer_source_and_conditions_across_rebuilds(result):
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.main_context_authority import project_helper_predecessor_authority

    authority = {
        "task_id": "prior", "source": _source("prior"), "status": "completed", "result": result,
        "origin_message_text": "  Owner: keep the tower.\nUse three levels.  ",
        "verification_receipts": [{"evidence": "old receipt " * 300}],
        "task_contract": {"objective": "Build the tower", "context": "never use native/API fallback",
                          "delegation_budget": {"intent_note": "L1 asks L2 to spawn L3"},
                          "predecessor_authority": {"task_id": "earlier", "result": "earlier body"}},
    }
    envelope = build_task_contract({"predecessor_authority": authority})["predecessor_authority"]
    before = copy.deepcopy(envelope)
    if isinstance(envelope["result"], dict):
        assert len(json.dumps(envelope["result"], ensure_ascii=False)) > 15_000  # wrapper included
    brief = project_helper_predecessor_authority(envelope)
    for key in ("task_id", "source", "status", "result", "task_contract", "origin_message_text",
                "authority_sha256", "authority_chars", "digest_semantics"):
        assert brief[key] == before[key]
    assert "predecessor_authority" not in brief["task_contract"]
    assert "verification_receipts" not in brief
    assert brief["omitted_fields"]["verification_receipts"] == len(json.dumps(
        envelope["verification_receipts"], ensure_ascii=False, sort_keys=True))
    wire = json.dumps(brief, ensure_ascii=False, sort_keys=True)
    assert json.dumps(project_helper_predecessor_authority(brief), ensure_ascii=False, sort_keys=True) == wire
    rebuilt = build_task_contract({"predecessor_authority": brief})["predecessor_authority"]
    assert json.dumps(rebuilt, ensure_ascii=False, sort_keys=True) == wire
    assert project_helper_predecessor_authority(authority) == brief  # legacy normalization is shared
    assert envelope == before
    brief["task_contract"]["delegation_budget"]["intent_note"] = "changed only in the copy"
    assert envelope == before


@pytest.mark.parametrize("selection", ["shared", "declared"])
def test_child_grandchild_and_external_session_keep_the_same_predecessor_brief(selection):
    from types import SimpleNamespace

    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.subagent_work_order import assignment_instructions, compile_external_work_order
    from ouroboros.tools.control_scheduling import _build_child_subagent_contract

    root = build_task_contract({"task_contract": {
        "objective": "Continue the tower", "input_sources": selection,
        "predecessor_authority": {
            "task_id": "prior", "source": _source("prior"), "authority_sha256": "a" * 64,
            "result": "The tower stands", "origin_message_text": "Keep the old tower",
            "task_contract": {"context": "never use native/API fallback",
                              "constraints": "L1 asks L2 to spawn L3"},
            "verification_receipts": [{"evidence": "RECEIPT_BODY"}],
        },
    }})
    before = copy.deepcopy(root)
    child = _build_child_subagent_contract({"tid": "child", "parent_contract": root})
    grandchild = _build_child_subagent_contract({"tid": "grandchild", "parent_contract": child})
    brief = child["predecessor_authority"]
    assert grandchild["predecessor_authority"] == brief
    assert brief["source"] == before["predecessor_authority"]["source"]
    assert brief["authority_sha256"] == "a" * 64
    assert "verification_receipts" not in brief and brief["omitted_fields"]["verification_receipts"] > 0
    for contract in (root, child, grandchild):
        instructions = assignment_instructions(SimpleNamespace(task_contract=contract))
        session, _ = json.JSONDecoder().raw_decode(instructions.split("\n", 1)[1])
        assert session["predecessor_authority"] == brief
        assert "RECEIPT_BODY" not in instructions
    for contract in (child, grandchild):
        leaf = compile_external_work_order({"task_contract": contract})
        assert "get_task_result" in leaf and "RECEIPT_BODY" not in leaf
        for words in ("The tower stands", "Keep the old tower", "never use native/API fallback",
                      "L1 asks L2 to spawn L3"):
            assert (words in leaf) == (selection == "shared")
    assert build_task_contract({"task_contract": grandchild}) == grandchild
    assert root == before


@pytest.mark.parametrize("selection", ["shared", "declared"])
def test_a_task_without_a_predecessor_never_acquires_an_empty_brief(selection):
    from types import SimpleNamespace

    from ouroboros.subagent_work_order import assignment_instructions
    from ouroboros.tools.control_scheduling import _build_child_subagent_contract

    child = _build_child_subagent_contract({"tid": "fresh", "parent_contract": {"input_sources": selection}})
    assert "predecessor_authority" not in child
    instructions = assignment_instructions(SimpleNamespace(task_contract=child))
    session, _ = json.JSONDecoder().raw_decode(instructions.split("\n", 1)[1])
    assert "predecessor_authority" not in session and "full report" not in instructions


def test_legacy_declared_reference_keeps_unknown_omission_sizes_and_cannot_regain_bodies():
    from ouroboros.main_context_authority import project_helper_predecessor_authority

    reference = {"source": _source("prior"), "omitted_fields": ["result", "task_contract"]}
    brief = project_helper_predecessor_authority(reference, declared=True)
    assert brief == {"source": reference["source"], "omitted_fields": {"result": None, "task_contract": None}}
    assert project_helper_predecessor_authority(brief) == brief
    assert project_helper_predecessor_authority(brief, declared=True) == brief


def test_a_stored_authored_narrative_wins_without_chat_access(tmp_path):
    # Nothing writes an authored narrative any more; the readers keep serving the
    # ones earlier versions stored beside a task result and in the chat log.
    from ouroboros.task_results import load_task_result, write_task_result

    task_id = "narrative-root"
    ref = _ref(task_id)
    row = {
        "summary_kind": "authored_root_summary",
        "summary_id": f"task-narrative:{task_id}", "task_id": task_id,
        "result_ref": ref, "source_coverage": {"task_result": ref},
        "text": "The authored account of what actually happened.",
    }
    _summary(tmp_path, task_id, row["text"])
    write_task_result(tmp_path, task_id, "completed", result="raw", continuation_narrative=row)
    stored = load_task_result(tmp_path, task_id)
    assert stored["status"] == "completed"
    assert stored["continuation_narrative"]["text"] == row["text"]

    from ouroboros.main_context_authority import project_main_task_authority

    authority = {
        "task_id": task_id, "source": _source(task_id),
        "result": "R" * 200001, "task_contract": {"objective": "old"},
        **{key: stored[key] for key in ("continuation_narrative",)},
    }
    (tmp_path / "logs" / "chat.jsonl").unlink()
    projected = project_main_task_authority(
        {"id": "next", "predecessor_authority": authority}, drive_root=tmp_path,
    )["predecessor_authority"]
    assert projected["result"]["narrative_status"] == "available"
    assert projected["result"]["narrative"]["text"] == row["text"]
    assert "R" * 200001 not in json.dumps(projected)


@pytest.mark.parametrize("parent_has_narrative", [False, True])
def test_child_copyback_preserves_the_authoritative_narrative(tmp_path, parent_has_narrative):
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_results import load_task_result, write_task_result

    parent = tmp_path / "parent"
    child = tmp_path / "child"
    task_id = "copyback-narrative"
    child_row = {
        "text": "child-born authored account",
        "task_id": task_id,
        "summary_id": f"task-narrative:{task_id}",
        "summary_kind": "authored_root_summary",
        "result_ref": _ref(task_id),
        "source_coverage": {"task_result": _ref(task_id)},
    }
    write_task_result(
        child,
        task_id,
        "completed",
        result="child answer",
        continuation_narrative=child_row,
    )
    parent_row = {
        **child_row,
        "text": "parent-born authoritative account",
    }
    write_task_result(
        parent,
        task_id,
        "completed",
        result="parent placeholder",
        **({"continuation_narrative": parent_row} if parent_has_narrative else {}),
    )

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})
    assert copied is not None
    stored = load_task_result(parent, task_id)
    assert stored["continuation_narrative"]["text"] == (
        "parent-born authoritative account"
        if parent_has_narrative
        else "child-born authored account"
    )


def test_bounded_legacy_lookup_accepts_only_authored_exact_row(tmp_path):
    from ouroboros.main_context_authority import project_main_task_authority

    task_id = "legacy-root"
    narrative = "Legacy authored account"
    _summary(tmp_path, task_id, narrative)
    raw = "X" * 200001
    authority = {
        "task_id": task_id, "source": _source(task_id),
        "result": raw, "task_contract": {"objective": "old"},
    }
    projected = project_main_task_authority(
        {"id": "next", "predecessor_authority": authority}, drive_root=tmp_path,
    )["predecessor_authority"]
    assert projected["result"]["narrative"]["text"] == narrative

    terminal = json.loads(json.dumps(authority))
    terminal["continuation_narrative"] = {
        "text": "terminal must not count", "task_id": task_id,
        "summary_id": f"task-terminal:{task_id}",
        "summary_kind": "terminal_result_projection",
    }
    miss = project_main_task_authority(
        {"id": "next", "predecessor_authority": terminal}, drive_root=tmp_path,
    )["predecessor_authority"]
    assert miss["result"]["narrative_status"] == "available"
    assert miss["result"]["narrative"]["text"] == narrative


def test_legacy_mismatch_and_terminal_only_rows_emit_gap_without_writes(tmp_path):
    from ouroboros.main_context_authority import project_main_task_authority
    from ouroboros.task_results import load_task_result

    task_id = "legacy-gap"
    chat = tmp_path / "logs" / "chat.jsonl"
    chat.parent.mkdir(parents=True)
    rows = [
        {
            "type": "task_summary",
            "summary_kind": "authored_root_summary",
            "summary_id": f"task-narrative:{task_id}",
            "task_id": task_id,
            "result_ref": {**_ref(task_id), "extra": "wrong"},
            "source_coverage": {"task_result": {**_ref(task_id), "extra": "wrong"}},
            "text": "wrong source must not be accepted",
        },
        {
            "type": "task_summary",
            "summary_kind": "terminal_root_projection",
            "summary_id": f"task-terminal:{task_id}",
            "task_id": task_id,
            "result_ref": _ref(task_id),
            "source_coverage": {"task_result": _ref(task_id)},
            "text": "terminal projection is not a narrative",
        },
    ]
    chat.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    authority = {
        "task_id": task_id,
        "source": _source(task_id),
        "result": "Q" * 200001,
        "task_contract": {"objective": "keep"},
    }
    projected = project_main_task_authority(
        {"id": "next", "predecessor_authority": authority}, drive_root=tmp_path,
    )["predecessor_authority"]
    assert projected["result"]["narrative_status"] == "unavailable"
    assert load_task_result(tmp_path, task_id) is None


def test_main_projection_is_deep_and_deduplicates_two_raw_keys_and_current_contract(tmp_path):
    from ouroboros.main_context_authority import project_main_task_authority

    second = {
        "task_id": "older", "source": _source("older"),
        "result": "Y" * 200001, "task_contract": {"objective": "older"},
        "unknown": {"nested": ["keep", {"x": 1}]},
    }
    first = {
        "task_id": "first", "source": _source("first"),
        "result": "A" * 200001, "final_answer": "A" * 200001,
        "task_contract": {"objective": "first", "predecessor_authority": second},
        "predecessor_authority": second,
    }
    contract = {"objective": "next", "predecessor_authority": first}
    task = {"id": "next", "task_contract": contract, "predecessor_authority": first}
    before = copy.deepcopy(task)
    projected = project_main_task_authority(task, drive_root=tmp_path)

    assert task == before
    assert "predecessor_authority" not in projected["task_contract"]
    nested = projected["predecessor_authority"]["task_contract"]["predecessor_authority"]
    assert nested["task_id"] == "older"
    assert nested["unknown"] == second["unknown"]
    first_result = projected["predecessor_authority"]["result"]
    assert first_result["narrative_status"] == "unavailable"
    assert projected["predecessor_authority"]["final_answer"]["narrative_status"] == "unavailable"
    assert nested["result"]["narrative_status"] == "unavailable"


@pytest.mark.parametrize("key", ["result", "final_answer"])
@pytest.mark.parametrize("size,omitted", [(199999, False), (200000, False), (200001, True)])
def test_predecessor_threshold_is_strict_and_key_specific(tmp_path, key, size, omitted):
    from ouroboros.context_budget import PREDECESSOR_RESULT_INLINE_CHARS
    from ouroboros.main_context_authority import project_main_task_authority

    assert PREDECESSOR_RESULT_INLINE_CHARS == 200000
    value = "Z" * size
    authority = {
        "task_id": "threshold", "source": _source("threshold"),
        key: value, "task_contract": {"objective": "keep"},
    }
    projected = project_main_task_authority(
        {"id": "next", "predecessor_authority": authority}, drive_root=tmp_path,
    )["predecessor_authority"][key]
    assert (isinstance(projected, dict) and projected.get("raw_result_resident") is False) is omitted
    if not omitted:
        assert projected == value


def test_numeric_http_code_does_not_hide_response_body_overflow():
    from ouroboros.loop_llm_call import classify_llm_exception

    class Response:
        status_code = 400

        def json(self):
            return {"error": {"code": "400", "message": "maximum context length exceeded"}}

    class ProviderError(Exception):
        status_code = 400
        code = "400"
        response = Response()

    result = classify_llm_exception(ProviderError("bad request"), "bad request")
    assert result.kind == "context_overflow"
    assert result.provider_code == ""


def test_response_body_output_limit_keeps_output_size_precedence():
    from ouroboros.loop_llm_call import classify_llm_exception

    class Response:
        status_code = 400

        def json(self):
            return {"error": {"code": "400", "message": "max_tokens exceeds maximum context length"}}

    class ProviderError(Exception):
        status_code = 400
        code = "400"
        response = Response()

    result = classify_llm_exception(ProviderError("bad request"), "bad request")
    assert result.kind == "request_too_large"


def test_symbolic_rate_code_stays_retryable_even_with_http_400():
    from ouroboros.loop_llm_call import classify_llm_exception

    class ProviderError(Exception):
        status_code = 400
        code = "400"
        body = {"error": {"code": "rate_limit_exceeded", "message": "Bad Request"}}

    result = classify_llm_exception(ProviderError("bad request"), "bad request")
    assert result.kind == "provider_transient"
    assert result.retry_same_request is True
    assert result.provider_code == "rate_limit_exceeded"


@pytest.mark.parametrize("generic_type", ["error", "invalid_request_error"])
def test_symbolic_rate_code_wins_over_generic_provider_type(generic_type):
    from ouroboros.loop_llm_call import classify_llm_exception

    class ProviderError(Exception):
        status_code = 400
        code = "400"

    error = ProviderError("bad request")
    error.type = generic_type
    error.body = {
        "type": generic_type,
        "error": {"code": "rate_limit_exceeded", "message": "Bad Request"},
    }

    result = classify_llm_exception(error, "bad request")
    assert result.kind == "provider_transient"
    assert result.retry_same_request is True
    assert result.provider_code == "rate_limit_exceeded"


def test_exception_overflow_text_survives_a_generic_response_body():
    from ouroboros.loop_llm_call import classify_llm_exception

    class ProviderError(Exception):
        status_code = 400
        code = "400"
        body = {"error": {"code": "400", "message": "Bad Request"}}

    result = classify_llm_exception(
        ProviderError("maximum context length exceeded"),
        "maximum context length exceeded",
    )
    assert result.kind == "context_overflow"
    assert result.retry_same_request is False


def test_router_selector_is_explicit_and_schema_requires_it(tmp_path):
    from ouroboros.tools.control import _promote_chat_to_task, _route_to_project, get_tools

    ctx = type("Ctx", (), {"pending_events": [], "event_queue": None, "drive_root": tmp_path})()
    assert "predecessor_task_id is required" in _promote_chat_to_task(ctx, "work")
    assert "predecessor_task_id is required" in _promote_chat_to_task(ctx, "work", predecessor_task_id=None)
    assert ctx.pending_events == []
    assert "predecessor_task_id is required" in _route_to_project(ctx, "missing", "work")
    assert "predecessor_task_id is required" in _route_to_project(
        ctx, "missing", "work", predecessor_task_id=None,
    )
    entries = {entry.name: entry for entry in get_tools()}
    assert "predecessor_task_id" in entries["promote_chat_to_task"].schema["parameters"]["required"]
    assert "predecessor_task_id" in entries["route_to_project"].schema["parameters"]["required"]


def test_context_overflow_skips_cross_model_and_forced_provider_calls(tmp_path, monkeypatch):
    import queue

    import ouroboros.loop as loop_mod
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.tools.registry import ToolRegistry

    class FakeLLM:
        def default_model(self):
            return "test-model"

    fallback_calls = []

    def fake_round(ctx):
        ctx.accumulated_usage.update(
            _last_llm_error_kind="context_overflow",
            execution_status="infra_failed",
            reason_code="llm_api_error",
        )
        return None, 0.0, ctx.active_context_mode

    def forbidden_fallback(**_kwargs):
        fallback_calls.append("cross-model")
        raise AssertionError("overflow must not enter the cross-model chain")

    monkeypatch.setattr(loop_mod, "_call_round_model", fake_round)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", forbidden_fallback)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    result, usage, trace = loop_mod.run_llm_loop(
        messages=[{"role": "user", "content": "solve"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text, *, incident=None: None,
        incoming_messages=queue.Queue(),
        task_id="overflow-task",
        drive_root=tmp_path,
    )
    assert not fallback_calls
    assert "context exceeded" in result
    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == "llm_api_error"
    assert usage["_last_llm_error_kind"] == "context_overflow"
    outcome = derive_loop_outcome(result, usage, trace)
    assert outcome["failure"]["error_kind"] == "context_overflow"

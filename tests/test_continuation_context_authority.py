"""Focused continuity-source, predecessor-brief and Main projection regressions."""

from __future__ import annotations

import copy
import json

import pytest

from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY


def _ref(task_id: str) -> dict:
    return {"kind": "task_result", "task_id": task_id, "reader": "get_task_result"}


def _source(task_id: str) -> dict:
    return {
        **_ref(task_id),
        "arguments": {"task_id": task_id, "include_authority": True},
    }


@pytest.mark.parametrize("result", ["Use the existing tower", "r" * 20_000, "\x00" * 6_000])
def test_helper_brief_keeps_the_source_and_owner_words_across_rebuilds(result):
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
    for key in ("task_id", "source", "status", "origin_message_text",
                "authority_sha256", "authority_chars", "digest_semantics"):
        assert brief[key] == before[key]
    assert "result" not in brief and "task_contract" not in brief
    assert brief["omitted_fields"]["result"] > 0 and brief["omitted_fields"]["task_contract"] > 0
    assert "verification_receipts" not in brief
    assert brief["omitted_fields"]["verification_receipts"] == len(json.dumps(
        envelope["verification_receipts"], ensure_ascii=False, sort_keys=True))
    wire = json.dumps(brief, ensure_ascii=False, sort_keys=True)
    assert json.dumps(project_helper_predecessor_authority(brief), ensure_ascii=False, sort_keys=True) == wire
    rebuilt = build_task_contract({"predecessor_authority": brief})["predecessor_authority"]
    assert json.dumps(rebuilt, ensure_ascii=False, sort_keys=True) == wire
    assert project_helper_predecessor_authority(authority) == brief  # legacy normalization is shared
    assert envelope == before
    brief["source"]["arguments"]["task_id"] = "changed only in the copy"
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
        assert ("Keep the old tower" in leaf) == (selection == "shared")
        for words in ("The tower stands", "never use native/API fallback", "L1 asks L2 to spawn L3"):
            assert words not in leaf
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


def _narrative(task_id: str, text: str, *, kind: str = "authored_root_summary") -> dict:
    ref = _ref(task_id)
    return {"summary_kind": kind, "summary_id": f"task-narrative:{task_id}", "task_id": task_id,
            "result_ref": ref, "source_coverage": {"task_result": ref}, "text": text}


@pytest.mark.parametrize("kind,served", [("authored_root_summary", True), ("terminal_result_projection", False)])
def test_a_stored_legacy_narrative_is_the_notes_row_and_main_never_substitutes(tmp_path, kind, served):
    """Nothing writes an authored narrative any more; a record an earlier version wrote
    still serves it — as the foldable notes row after the assignment, never as a
    stand-in for the answer inside the frozen prefix. The answer stays behind its
    source at every size; a terminal projection is not a narrative."""
    from types import SimpleNamespace

    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.main_context_authority import (
        PREDECESSOR_NOTES_NAME, predecessor_notes_message, project_main_task_authority,
    )
    from ouroboros.task_results import write_task_result

    task_id = "narrative-root"
    text = "The authored account of what actually happened."
    write_task_result(tmp_path, task_id, "completed", result="R" * 200001,
                      continuation_narrative=_narrative(task_id, text, kind=kind))
    task = {"id": "next", "text": "continue", "budget_drive_root": str(tmp_path),
            "predecessor_authority_source": _pull_source(task_id)}
    assert validate_task_authority_sources(SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path), task) == {}
    brief = task["predecessor_authority"]
    assert "result" not in brief and brief["omitted_fields"]["result"] > 0
    assert brief["source"] == _pull_source(task_id)
    assert "continuation_narrative" in brief["omitted_fields"]

    projected = project_main_task_authority(task)["predecessor_authority"]
    assert projected == brief, "Main's copy is the brief itself: no substitution, no narrative keys"
    assert "narrative_status" not in json.dumps(projected)
    note = predecessor_notes_message(task, drive_root=tmp_path)
    if served:
        assert note["role"] == "user" and note[HOST_CONTEXT_KIND_KEY] == PREDECESSOR_NOTES_NAME
        assert note["content"].endswith(text) and "continuation_narrative" in note["content"]
        assert f"get_task_result(task_id={task_id!r}, include_authority=True)" in note["content"]
    else:
        assert note is None


def test_an_ordinary_fold_names_its_notes_for_the_successor_and_a_noteless_view_clears_them(tmp_path):
    """17.10 review: the notes row read only review-history selections, so a task that folded ordinary
    work handed its successor nothing. Every applied fold now names the surviving notes in the result."""
    from types import SimpleNamespace

    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.main_context_authority import WORKING_NOTES_FIELD, predecessor_notes_message, publish_working_notes
    from ouroboros.task_results import load_task_result, write_task_result
    from tests.test_context_renewal import NOTE, _noted

    task_id = "ordinary-fold-root"
    write_task_result(tmp_path, task_id, "running", result="")
    folded = _noted(tmp_path)
    ref = publish_working_notes(SimpleNamespace(task_id=task_id, drive_root=tmp_path), folded)
    assert load_task_result(tmp_path, task_id)[WORKING_NOTES_FIELD] == ref
    task = {"id": "next", "text": "continue", "budget_drive_root": str(tmp_path),
            "predecessor_authority_source": _pull_source(task_id)}
    assert validate_task_authority_sources(SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path), task) == {}
    note = predecessor_notes_message(task, drive_root=tmp_path)
    assert NOTE in note["content"] and WORKING_NOTES_FIELD in note["content"]
    plain = [row for row in folded if not isinstance(row.get("content"), list)]  # a view whose notes were retired
    assert publish_working_notes(SimpleNamespace(task_id=task_id, drive_root=tmp_path), plain) is None
    assert load_task_result(tmp_path, task_id)[WORKING_NOTES_FIELD] is None
    assert predecessor_notes_message(task, drive_root=tmp_path) is None


def test_a_narrative_only_in_the_chat_log_is_the_notes_row_by_its_exact_task(tmp_path):
    """Records written before narratives were stored beside the result keep theirs in the chat
    log; the notes row reads it by exact task identity from the dialogue store, nothing else."""
    from types import SimpleNamespace

    from ouroboros import dialogue_store as ds
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.main_context_authority import predecessor_notes_message
    from ouroboros.task_results import write_task_result

    task_id = "chat-narrative-root"
    write_task_result(tmp_path, task_id, "completed", result="done")
    task = {"id": "next", "text": "continue", "budget_drive_root": str(tmp_path),
            "predecessor_authority_source": _pull_source(task_id)}
    assert validate_task_authority_sources(SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path), task) == {}
    assert predecessor_notes_message(task, drive_root=tmp_path) is None  # no narrative anywhere: no row
    row = {**_narrative(task_id, "What the earlier version wrote to chat."), "chat_id": 1, "type": "task_summary"}
    ds.append(tmp_path, "chat", {**row, "task_id": "another-task", "text": "Not this task's account."})
    receipt = ds.append(tmp_path, "chat", row)
    note = predecessor_notes_message(task, drive_root=tmp_path)
    assert note[HOST_CONTEXT_KIND_KEY] == "predecessor_notes"
    assert note["content"].endswith("What the earlier version wrote to chat.")
    assert f"chat:{receipt.seq}" in note["content"] and "Not this task's account." not in note["content"]


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


def test_main_projection_shows_the_brief_once_and_the_objective_once():
    """Main's provider copy is the materialized brief at the top level; the equal copy the
    contract spread nests is dropped, a different identity stays; a contract objective that
    IS the assignment becomes a typed reference, any other objective stays a string; and no
    answer body enters the prefix, even when a legacy envelope still carries it."""
    import hashlib

    from ouroboros.main_context_authority import project_main_task_authority

    brief = {"task_id": "first", "source": _source("first"), "kind": "bounded_continuation_envelope",
             "result": "A" * 200001, "task_contract": {"objective": "first"}, "omitted_fields": {"trace_summary": 12}}
    work_order = "[CONTINUE] Continue was requested for task first.\n\nOwner's original instruction (verbatim):\nShip it"
    task = {"id": "next", "text": work_order, "predecessor_authority": brief,
            "task_contract": {"objective": work_order, "context": "keep", "predecessor_authority": copy.deepcopy(brief)}}
    before = copy.deepcopy(task)

    projected = project_main_task_authority(task)

    assert task == before
    assert projected["predecessor_authority"]["source"] == brief["source"]
    assert "result" not in projected["predecessor_authority"]
    assert projected["predecessor_authority"]["omitted_fields"]["result"] > 0
    assert "narrative_status" not in json.dumps(projected)
    assert "predecessor_authority" not in projected["task_contract"], "the brief appears once"
    assert projected["task_contract"]["context"] == "keep"
    assert projected["task_contract"]["objective"] == {
        "representation": "assignment", "chars": len(work_order),
        "sha256": hashlib.sha256(work_order.encode("utf-8")).hexdigest(),
        "rule": "The objective is this task's assignment, the first user turn of this conversation; when the "
                "whole input cannot fit a request, that turn is a pointer to its exact stored source instead. "
                "It is not repeated here."}
    assert work_order not in json.dumps(projected), "the work order is not repeated inside the Runtime view"

    other = {**brief, "task_id": "older", "source": _source("older")}
    kept = project_main_task_authority({"id": "next", "text": "do it", "predecessor_authority": brief,
                                        "task_contract": {"objective": "A shorter title", "predecessor_authority": other}})
    other_brief = kept["task_contract"]["predecessor_authority"]
    assert other_brief["source"] == other["source"], "a different identity keeps its own reader"
    assert "result" not in other_brief and other_brief["omitted_fields"]["result"] > 0
    assert kept["task_contract"]["objective"] == "A shorter title", "only the assignment itself is referenced"


def test_main_projection_shows_a_line_broken_assignment_once():
    """The contract strips its objective; a task text ending in a line break (a wake's rendered
    template does) is still the one assignment, so the Runtime view references it instead of
    delivering it a second time (28.10 19:20Z: 43,221 estimated tokens twice). A wake whose
    observation is bound names that stored source, whichever way its first turn delivers it;
    an objective that differs from the text stays whole."""
    from ouroboros.consciousness_wake import WAKE_OBSERVATION_KEY
    from ouroboros.contracts.task_contract import attach_task_contract
    from ouroboros.main_context_authority import project_main_task_authority

    observation = "[Wake-up · heartbeat]\nObserved since the last accepted wake:\n" + "- task t done\n" * 2_000
    task = attach_task_contract({"id": "wake", "text": observation})
    assert task["task_contract"]["objective"] == observation.strip() != observation
    before = copy.deepcopy(task)
    reference = project_main_task_authority(task)["task_contract"]["objective"]
    assert task == before
    assert reference["representation"] == "assignment" and reference["chars"] == len(observation.strip())
    assert "source" not in reference and "- task t done" not in json.dumps(reference)

    source = {"sha256": "a" * 64, "lines": 2_001, "path": "wake_observation.jsonl",
              "read": {"tool": "read_file", "arguments": {"root": "runtime_data", "path": "w/o.jsonl", "start_line": 1}}}
    bound = attach_task_contract({"id": "wake", "text": observation, "metadata": {
        WAKE_OBSERVATION_KEY: {"source": source, "projection_text": "pointer"}}})
    named = project_main_task_authority(bound)["task_contract"]["objective"]
    assert named["source"] == {key: source[key] for key in ("read", "sha256", "lines")}
    assert named["rule"].startswith(reference["rule"]) and "`source`" in named["rule"]
    assert "- task t done" not in json.dumps(named)

    for text, objective in ((observation, "Summarize the wake"), ("Fix the parser\n", "Fix the parser and the lexer"),
                            ("Fix the parser\n", "fix the parser")):
        kept = project_main_task_authority({"id": "t", "text": text, "task_contract": {"objective": objective}})
        assert kept["task_contract"]["objective"] == objective, "a different objective stays whole"



def test_main_projection_shows_an_equal_task_context_once():
    """A contract context byte-identical to the task block's own ``context`` (an API or recovery root's packet,
    20.10: 2 x 96,893 chars in every round) becomes a typed reference; a different or absent row context, and
    any non-string, stay exactly as the contract holds them."""
    import hashlib

    from ouroboros.main_context_authority import project_main_task_authority

    packet = "FULL RECOVERY PACKET " * 4_000 + " \n"
    task = {"id": "root", "text": "Recover", "context": packet,
            "task_contract": {"objective": "Recover the store", "context": packet}}
    before = copy.deepcopy(task)

    projected = project_main_task_authority(task)

    assert task == before
    assert projected["task_contract"]["context"] == {
        "representation": "task_context", "chars": len(packet),
        "sha256": hashlib.sha256(packet.encode("utf-8")).hexdigest(),
        "rule": "The caller's context is the `context` field of the task block above; it is not repeated here."}
    assert packet not in json.dumps(projected)
    assert projected["task_contract"]["objective"] == "Recover the store"

    child = project_main_task_authority({"id": "c", "text": "x", "context": "delegator orientation",
                                         "task_contract": {"context": "inherited authority"}})
    assert child["task_contract"]["context"] == "inherited authority", "different texts both stay"
    spaced = project_main_task_authority({"id": "s", "text": "x", "context": "keep",
                                          "task_contract": {"context": " keep\n"}})
    assert spaced["task_contract"]["context"] == " keep\n", "equality is byte-exact, no strip"
    for row_context, contract_context in ((None, "keep"), ("", ""), ("x", None), ({"k": 1}, {"k": 1})):
        kept = project_main_task_authority({"id": "n", "text": "x", "context": row_context,
                                            "task_contract": {"context": contract_context}})
        assert kept["task_contract"]["context"] == contract_context

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
    assert "Context-size recovery" in result and "no further provider call" in result
    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == "llm_api_error"
    assert usage["_last_llm_error_kind"] == "context_overflow"
    outcome = derive_loop_outcome(result, usage, trace)
    assert outcome["failure"]["error_kind"] == "context_overflow"


_PLAN_GAP_RULE = "Do not treat this plan/review authority as complete."


def _pull_source(task_id: str) -> dict:
    """The exact host-issued predecessor pointer a Continue startup accepts."""
    return {"kind": "task_result", "task_id": task_id, "tool": "get_task_result",
            "arguments": {"task_id": task_id, "include_authority": True}}


def _plan_predecessor(root, task_id: str, damage: str) -> dict:
    """A failed predecessor whose plan the real writer recorded, then damaged as named."""
    from ouroboros import artifacts
    from ouroboros.task_results import write_task_result
    from tests.test_task_source_carrier_identity import _legacy_damage, _promotion, _record_plan

    spec, plan = _record_plan(root, task_id)
    wave, fields = plan["waves"][-1], {}
    if damage == "spec_missing":
        (artifacts.task_artifact_dir_path(root, task_id) / wave["spec_source_ref"]["path"]).unlink()
    elif damage == "dialogue_missing":  # read after the spec, unwrapped: a raw OSError
        wave["dialogue_source_ref"] = {"kind": "task_source", "root": "artifact_store", "size": 1, "sha256": "0" * 64,
                                       "path": "source_handles/context_checkpoints/plan-dialogue-gone.jsonl"}
        fields["plan_review_state"] = plan
    else:  # the Quotas row: path-less placeholders, each path kept in a fact, the bytes intact
        damaged, facts = _legacy_damage(plan, task_id)
        fields.update(plan_review_state=damaged, child_ref_promotion=_promotion("complete", facts, []))
    write_task_result(root, task_id, "failed", result="Provider unavailable.", reason_code="provider_unavailable",
                      task_contract={"objective": "Build the tower"}, **fields)
    return spec


@pytest.mark.parametrize("damage", ["spec_missing", "dialogue_missing", "legacy_gap"])
def test_continue_carries_an_unreadable_predecessor_plan_as_a_named_gap_through_every_projection(tmp_path, damage):
    """Startup -> bounded envelope -> root Runtime context -> helper briefs -> get_task_result."""
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.context import build_llm_messages
    from ouroboros.main_context_authority import project_helper_predecessor_authority
    from ouroboros.store import open_store
    from ouroboros.task_results import load_plan_review_state
    from ouroboros.task_store import signature
    from ouroboros.tools.control import _get_task_result
    from ouroboros.tools.plan_review_artifacts import PlanReviewSourceUnavailable
    from ouroboros.tools.registry import ToolContext
    from tests.test_doc_context import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    root, predecessor = env.drive_root, "plan-predecessor"
    _plan_predecessor(root, predecessor, damage)
    with open_store(root, write=False) as txn:
        row_text, revision = txn.body_text(predecessor), signature(root, predecessor, txn=txn)
    task = {"id": "plan-successor", "type": "task", "text": "continue", "budget_drive_root": str(root),
            "predecessor_authority_source": _pull_source(predecessor)}

    assert validate_task_authority_sources(env, task) == {}  # the successor starts and decides

    error = {"spec_missing": "PlanReviewSourceUnavailable", "dialogue_missing": "FileNotFoundError"}.get(damage)
    gap = error and {"status": "authority_source_unavailable", "code": "PLAN_REVIEW_SOURCE_UNAVAILABLE",
                     "task_id": predecessor, "error": error, "rule": _PLAN_GAP_RULE}
    envelope = task["predecessor_authority"]
    assert envelope.get("plan_review_source_gap") == gap
    if not gap:  # readable predecessor plans stay behind the same exact reader
        assert "plan_review_state" not in envelope and envelope["omitted_fields"]["plan_review_state"] > 0
    rendered = json.dumps(build_llm_messages(env=env, memory=memory, task=task)[0], ensure_ascii=False)
    assert ("plan_review_source_gap" in rendered and _PLAN_GAP_RULE in rendered) == bool(gap)
    for declared in (False, True):
        brief = project_helper_predecessor_authority(envelope, declared=declared)
        assert brief.get("plan_review_source_gap") == gap
        assert project_helper_predecessor_authority(brief, declared=declared) == brief
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=root)
    for flag, independent in (("include_authority", "authority"), ("include_work_order_source", "work_order_source"),
                              ("include_completion_source", "completion_source"),
                              ("include_focus_source", "focus_source")):
        payload = json.loads(_get_task_result(ctx, predecessor, **{flag: True}))
        assert payload["status"] == "available" and independent in payload
        assert payload["authority"].get("plan_review_source_gap") == gap
    with pytest.raises((PlanReviewSourceUnavailable, OSError)):
        load_plan_review_state(root, predecessor)  # deciders stay strict
    with open_store(root, write=False) as txn:
        assert txn.body_text(predecessor) == row_text
        assert signature(root, predecessor, txn=txn) == revision  # even an identical rewrite is forbidden


def test_a_named_predecessor_whose_row_reads_back_empty_refuses_the_start(tmp_path):
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from tests.test_doc_context import _make_env_and_memory

    env, _memory = _make_env_and_memory(tmp_path)
    task = {"id": "next", "type": "task", "text": "continue", "predecessor_authority_source": _pull_source("never-written")}

    refusal = validate_task_authority_sources(env, task)

    assert refusal["reason_code"] == "authority_source_unavailable"
    assert refusal["detail"] == "named predecessor task result is missing or unreadable"
    assert refusal["source"]["task_id"] == "never-written" and "predecessor_authority" not in task


def test_the_current_tasks_own_unreadable_plan_source_still_refuses_typed(tmp_path):
    """Owner answer 2A: only a predecessor's plan becomes a gap; the task's own plan still refuses."""
    from ouroboros import artifacts
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from tests.test_doc_context import _make_env_and_memory
    from tests.test_task_source_carrier_identity import _record_plan

    env, _memory = _make_env_and_memory(tmp_path)
    _spec, plan = _record_plan(env.drive_root, "own-plan")
    (artifacts.task_artifact_dir_path(env.drive_root, "own-plan") / plan["waves"][-1]["spec_source_ref"]["path"]).unlink()

    refusal = validate_task_authority_sources(env, {"id": "own-plan", "title": "Cat plan", "text": "continue"})

    assert refusal["reason_code"] == "authority_source_unavailable"
    assert refusal["source"]["kind"] == "plan_review_state" and refusal["human_label"] == "Cat plan"


def test_the_runtime_gap_of_the_current_tasks_plan_comes_from_the_same_constructor(tmp_path):
    """context.py and the predecessor projection share one gap: one shape, one rule text."""
    from ouroboros import artifacts
    from ouroboros.context import _task_authority_projection, build_llm_messages
    from tests.test_doc_context import _make_env_and_memory
    from tests.test_task_source_carrier_identity import _record_plan

    env, memory = _make_env_and_memory(tmp_path)
    _spec, plan = _record_plan(env.drive_root, "own-plan")
    (artifacts.task_artifact_dir_path(env.drive_root, "own-plan") / plan["waves"][-1]["spec_source_ref"]["path"]).unlink()
    task = {"id": "own-plan", "type": "task", "text": "continue"}

    assert _task_authority_projection(env, task)["plan_review_authority"] == {
        "status": "authority_source_unavailable", "code": "PLAN_REVIEW_SOURCE_UNAVAILABLE", "task_id": "own-plan",
        "error": "PlanReviewSourceUnavailable", "rule": _PLAN_GAP_RULE,
        "source": {"tool": "get_task_result", "arguments": {"task_id": "own-plan", "include_authority": True}}}
    assert _PLAN_GAP_RULE in json.dumps(build_llm_messages(env=env, memory=memory, task=task)[0], ensure_ascii=False)

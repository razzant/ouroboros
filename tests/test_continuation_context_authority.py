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


def test_authored_summary_is_persisted_and_wins_without_chat_access(tmp_path):
    from ouroboros.project_dialogue import append_authored_task_summary
    from ouroboros.task_results import load_task_result, write_task_result

    task_id = "narrative-root"
    write_task_result(tmp_path, task_id, "completed", result="raw")
    ref = _ref(task_id)
    row = {
        "type": "task_summary", "summary_kind": "authored_root_summary",
        "summary_id": f"task-narrative:{task_id}", "task_id": task_id,
        "result_ref": ref, "source_coverage": {"task_result": ref},
        "text": "The authored account of what actually happened.",
    }
    assert append_authored_task_summary(tmp_path, tmp_path, row)
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


@pytest.mark.parametrize("fallback", ["", "configured/fallback"])
def test_context_overflow_tries_configured_fallback_without_forced_provider_calls(tmp_path, monkeypatch, fallback):
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

    def unavailable_fallback(**kwargs):
        fallback_calls.append("cross-model")
        return (None, kwargs["active_model"], kwargs["active_use_local"],
                kwargs["context_fit_plan"], kwargs["active_context_mode"])

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", fallback)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACK", "")
    monkeypatch.setattr(loop_mod, "_call_round_model", fake_round)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", unavailable_fallback)
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
    assert fallback_calls == (["cross-model"] if fallback else [])
    assert "context exceeded" in result
    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == "llm_api_error"
    assert usage["_last_llm_error_kind"] == "context_overflow"
    outcome = derive_loop_outcome(result, usage, trace)
    assert outcome["failure"]["error_kind"] == "context_overflow"


@pytest.mark.parametrize("old_candidate", [False, True])
def test_cold_continuation_keeps_memory_completion_and_primary_wait_together(tmp_path, old_candidate):
    from dataclasses import replace
    from types import SimpleNamespace
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.loop_delivery import DeliveryCandidate
    from ouroboros.owner_wait import continuation_state, restore_continuation_state, store_continuation_source
    from tests.test_context_fit_v664 import _plan

    memory = {"focus": "7", "rooms": [{"id": "7", "records": [
        {"id": "source", "text": "The owner cancelled publication.", "author": {"kind": "mind"}}]}],
        "marks": [{"id": "mark", "text": "Keep the owner's cancellation."}]}
    plan = replace(_plan(), chronicle_state_json=json.dumps(memory), rendered_mode="low", model_role="fallback:1")
    candidate = DeliveryCandidate("Exact held answer.", "answer-sha", 3, 2, "evidence-sha", {},
                                  finalization_control="acceptance_required")
    ctx = SimpleNamespace(task_id="cold-memory-completion", task_attempt=1, drive_root=tmp_path,
        budget_drive_root=str(tmp_path), context_fit_plan=plan, _delivery_candidate=candidate,
        active_model="fallback-model", active_use_local=False, active_effort="max", active_context_mode="low",
        primary_route={"model": "primary-model", "role": "main", "use_local": False},
        active_role_override="main", route_wait_on_primary=True, _route_facts_pending="Primary temporarily refused.",
        _completion_selected={"source": "held_response", "sha256": "answer-sha"},
        _completion_observation={"owner_source_sha256": "owner-source", "feedback": {"verdict": "advisory"}},
        _completion_held_sha256="answer-sha", _task_acceptance_sealed_fence_token="fence",
        _task_acceptance_sealed_fence_generation=4,
        _last_context_observation={"rendered_mode": "low", "memory_source": "source"},
        _inspected_context_view={"source": "frozen"}, _pending_compaction={"source_sha256": "exposed-only"})
    messages = plan.messages_for("low") + [{"role": "user", "content": "The owner cancelled publication."}]
    state = continuation_state(ctx, messages, {"review_runs": []}, {"cost": 1.0}, 6, [], {"owner-msg"})
    if old_candidate:
        state["delivery_candidate"]["repair_attempted"] = True
    ref = store_continuation_source(ctx, state, "joint-cold-continuation")
    saved = json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, ref))
    fresh = SimpleNamespace()
    restored, trace, usage, seen = [], {}, {}, set()
    restore_continuation_state(SimpleNamespace(_ctx=fresh), saved, restored, trace, usage, seen)
    assert fresh.context_fit_plan == plan and json.loads(fresh.context_fit_plan.chronicle_state_json) == memory
    assert restored == messages and usage == {"cost": 1.0} and seen == {"owner-msg"}
    assert fresh.primary_route == ctx.primary_route and fresh.route_wait_on_primary is True
    assert fresh.active_role_override == "main" and fresh._route_facts_pending == ctx._route_facts_pending
    assert fresh._completion_selected == ctx._completion_selected
    assert fresh._completion_observation == ctx._completion_observation
    assert fresh._completion_held_sha256 == "answer-sha"
    assert fresh._task_acceptance_sealed_fence_token == "fence"
    assert fresh._task_acceptance_sealed_fence_generation == 4
    assert fresh._delivery_candidate.full_text == "Exact held answer."
    assert fresh._delivery_candidate.control_episode_seen and fresh._delivery_control_required
    assert fresh._pending_compaction == ctx._pending_compaction
    assert fresh._inspected_context_view == ctx._inspected_context_view
    assert fresh._last_context_observation == ctx._last_context_observation

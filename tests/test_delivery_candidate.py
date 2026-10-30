from __future__ import annotations

import hashlib
import json
import queue
from types import SimpleNamespace

import pytest

from tests._delivery_candidate_shared import (
    write_child as _write_child,
    write_confirmed_disposition_fixture as _write_confirmed_disposition,
)


def _finish_response(answer, *, by_hash=False):
    """A model-authored selection through the real local completion tool."""
    args = {"action": "finish", "answer_sha256" if by_hash else "answer":
            hashlib.sha256(answer.encode("utf-8")).hexdigest() if by_hash else answer}
    return {"content": None, "tool_calls": [{"id": "finish-selection", "type": "function",
        "function": {"name": "finish_task", "arguments": json.dumps(args)}}]}


def _assert_no_repair_round(calls):
    assert all("DELIVERY_CONTROL_REPAIR" not in str(row) for call in calls for row in call)


def _run_loop(
    tmp_path,
    monkeypatch,
    responses,
    review_holds=None,
    *,
    child=False,
    bind_child_before_second=False,
    progress=None,
):
    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    tmp_path.mkdir(parents=True, exist_ok=True)
    if child:
        _write_child(tmp_path)
    answers = iter(responses)
    calls = []
    progress = progress if progress is not None else []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call(_llm, request_messages, *_args, **_kwargs):
        calls.append([dict(row) for row in request_messages])
        if child and bind_child_before_second and len(calls) == 2:
            _write_confirmed_disposition(
                tmp_path,
                disposition="integrated",
                rationale="test parent consumed the handoff",
            )
        answer = next(answers)
        if isinstance(answer, dict):
            return {"role": "assistant", **answer}, 0.0
        return {"role": "assistant", "content": answer}, 0.0

    holds = iter(review_holds or [])

    def fake_review_at_delivery(_ctx, _content, _llm_trace, hold):
        # The result review's host step at delivery: a hold shows its outcome to the
        # author and retains the first answer, so the next complete answer is selected
        # explicitly through the real completion door.
        held = next(holds, False)
        if held:
            hold("[Result review] The review's findings were shown before delivery.")
        return held

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", fake_review_at_delivery)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {
        "budget_drive_root": str(tmp_path),
        "root_task_id": "parent1",
    }
    result, usage, trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": "do the work"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda text, **facts: progress.append({"text": text, **facts}),
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )
    return result, usage, trace, calls


def test_handoff_service_notices_cannot_erase_full_candidate(tmp_path, monkeypatch):
    from ouroboros.outcomes import derive_loop_outcome

    original = "Complete original answer with all child conclusions."
    result, usage, trace, calls = _run_loop(tmp_path, monkeypatch,
        [original, "service notice: child status refreshed", "service notice again", _finish_response(original, by_hash=True)],
        child=True, bind_child_before_second=True)
    assert result == original and len(calls) == 4
    assert trace["delivery_candidate"]["content_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert trace["delivery_candidate"]["degraded"] is False
    assert trace["delivery_candidate"]["evidence_current"] is True
    assert trace["delivery_candidate"]["acceptance_binding"]["authoritative"] is False
    assert derive_loop_outcome(result, usage, trace)["outcome_axes"]["execution"]["status"] == "ok"
    _assert_no_repair_round(calls)


def test_mutating_tool_acknowledgements_cannot_erase_full_candidate(tmp_path, monkeypatch):
    original = "Complete original answer with all implementation and verification details."
    write = {"content": None, "tool_calls": [{"id": "write-1", "type": "function", "function": {
        "name": "write_file", "arguments": json.dumps({"path": "effect.txt", "content": "durable tool effect"})}}]}
    result, _usage, trace, calls = _run_loop(tmp_path, monkeypatch,
        [original, write, "Review completed.", "Everything is done now.", _finish_response(original, by_hash=True)],
        review_holds=[True, False])
    assert (tmp_path / "effect.txt").read_text(encoding="utf-8") == "durable tool effect"
    assert result == original and len(calls) == 5
    assert trace["delivery_candidate"]["degraded"] is False
    assert trace["delivery_candidate"]["evidence_current"] is True
    assert trace["delivery_candidate"]["acceptance_binding"]["authoritative"] is False
    _assert_no_repair_round(calls)


def test_completion_tool_rejects_non_string_answer_without_erasing_retained_bytes(tmp_path, monkeypatch):
    original = "Complete original answer."
    invalid = _finish_response({"text": "not complete"})
    result, _usage, trace, calls = _run_loop(tmp_path, monkeypatch,
        [original, invalid, _finish_response(original, by_hash=True)], review_holds=[True, False])
    assert result == original and len(calls) == 3
    assert trace["tool_calls"][0]["is_error"] is True
    assert "COMPLETION_ARGUMENT" in trace["tool_calls"][0]["result"]
    assert trace["tool_calls"][-1]["completion_control"] is True
    _assert_no_repair_round(calls)


def test_duplicate_legacy_control_is_held_privately_until_explicit_selection(tmp_path, monkeypatch):
    original = "Complete original answer."
    duplicate = '{"delivery_control":"keep","delivery_control":"replace","full_answer":"Ambiguous replacement."}'
    progress = []
    result, _usage, trace, calls = _run_loop(tmp_path, monkeypatch,
        [original, duplicate, _finish_response(original, by_hash=True)], review_holds=[True, False], progress=progress)
    assert result == original and len(calls) == 3
    assert any(row.get("role") == "assistant" and row.get("content") == duplicate for row in calls[-1])
    assert "latest whole held response" not in str(calls[-1][-1]["content"])
    assert trace["delivery_candidate"]["degraded"] is False
    assert all(duplicate not in row["text"] for row in progress), progress
    _assert_no_repair_round(calls)


def test_repeated_duplicate_legacy_answers_neither_repair_nor_force_termination(tmp_path, monkeypatch):
    original = "Complete original answer."
    duplicate = '{"delivery_control":"replace","full_answer":"First answer.","full_answer":"Second answer."}'
    result, _usage, trace, calls = _run_loop(tmp_path, monkeypatch,
        [original, duplicate, duplicate, duplicate, _finish_response(original, by_hash=True)], review_holds=[True, False])
    assert result == original and len(calls) == 5
    assert trace["delivery_candidate"]["degraded"] is False
    for before, after in zip(calls[2:], calls[3:]):
        assert len(after) > len(before), "each repeated held response is appended with new host input"
    assert sum(row.get("role") == "assistant" and row.get("content") == duplicate for row in calls[-1]) == 3
    _assert_no_repair_round(calls)


def test_service_round_explicitly_selects_retained_or_shorter_complete_answer(tmp_path, monkeypatch):
    original = "Complete original answer with a long explanation and its supporting evidence."
    result, _usage, trace, _calls = _run_loop(tmp_path, monkeypatch,
        [original, _finish_response(original, by_hash=True)], review_holds=[True, False])
    assert result == original and trace["delivery_candidate"]["revision"] == 1
    replacement = "Corrected answer."
    result2, _usage2, trace2, _calls2 = _run_loop(tmp_path / "replace", monkeypatch,
        [original, _finish_response(replacement)], review_holds=[True, False])
    assert result2 == replacement and trace2["delivery_candidate"]["revision"] == 2
    assert trace2["delivery_candidate"]["acceptance_binding"]["authoritative"] is False


def test_delivery_evidence_ignores_service_and_read_only_but_tracks_effects(tmp_path):
    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "parent1"
    registry._ctx._owner_directives = []
    messages = [{"role": "user", "content": "task"}]
    ctx = loop._RoundLimitContext(
        messages,
        SimpleNamespace(),
        "test-model",
        "medium",
        0,
        tmp_path,
        "parent1",
        1,
        None,
        {},
        "",
        False,
        10,
        drive_root=tmp_path,
        status_drive_root=tmp_path,
        root_task_id="parent1",
    )
    trace = {"tool_calls": [], "reasoning_notes": []}
    revision1, fingerprint1 = loop._delivery_evidence_state(registry, ctx, trace)
    messages.append({"role": "user", "content": "[SERVICE] review completed"})
    assert loop._delivery_evidence_state(registry, ctx, trace) == (revision1, fingerprint1)

    trace["tool_calls"].append({
        "tool": "read_file",
        "args": {"path": "README.md"},
        "status": "ok",
        "result": "contents",
        "is_error": False,
    })
    assert loop._delivery_evidence_state(registry, ctx, trace) == (revision1, fingerprint1)

    trace["tool_calls"].append({
        "tool": "write_file",
        "args": {"path": "report.md", "content": "done"},
        "status": "ok",
        "result": "written",
        "is_error": False,
    })
    revision2, fingerprint2 = loop._delivery_evidence_state(registry, ctx, trace)
    assert revision2 == revision1 + 1 and fingerprint2 != fingerprint1

    trace["tool_calls"].append({
        "tool": "stop_service",
        "args": {"name": "preview"},
        "status": "ok",
        "result": "artifact registered",
        "artifact_registered": True,
        "is_error": False,
    })
    revision3, fingerprint3 = loop._delivery_evidence_state(registry, ctx, trace)
    assert revision3 == revision2 + 1 and fingerprint3 != fingerprint2

    trace["verification_events"] = [{
        "kind": "services_stopped",
        "services": [{
            "service_id": "preview",
            "name": "preview",
            "lifecycle": "stopped",
            "artifact_output_failed": True,
            "artifact_outputs": "⚠️ ARTIFACT_OUTPUT_ERROR: report.html is missing",
        }],
    }]
    revision4, fingerprint4 = loop._delivery_evidence_state(registry, ctx, trace)
    assert revision4 == revision3 + 1 and fingerprint4 != fingerprint3


def test_service_outputs_finalize_before_the_result_review_and_require_replacement(tmp_path, monkeypatch):
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.tools import services as services_mod

    calls = 0

    def fake_stop(_ctx):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{
                "service_id": "preview",
                "name": "preview",
                "lifecycle": "stopped",
                "artifact_output_failed": True,
                "artifact_outputs": "⚠️ ARTIFACT_OUTPUT_ERROR: report.html is missing",
            }]
        return []

    monkeypatch.setattr(services_mod, "stop_task_services", fake_stop)
    original = "Complete answer written before the preview service stopped."
    replacement = "Complete answer disclosing that the preview output could not be finalized."
    result, usage, trace, model_calls = _run_loop(
        tmp_path,
        monkeypatch,
        [
            original,
            _finish_response(replacement),
        ],
    )

    assert result == replacement
    assert len(model_calls) == 2
    controls = [str(row.get("content") or "") for row in model_calls[1]
                if "[DELIVERY_FINALIZATION_CONTROL]" in str(row.get("content") or "")]
    assert any("completion tool" in text for text in controls)
    assert trace["tool_calls"][-1]["completion_control"] is True
    # A fresh source observation follows the candidate-control instruction.
    assert trace["delivery_candidate"]["revision"] == 2
    assert trace["delivery_candidate"]["finalization_control"] == "candidate"
    assert trace["verification_events"][0]["kind"] == "services_stopped"
    outcome = derive_loop_outcome(result, usage, trace)
    assert outcome["outcome_axes"]["execution"]["status"] == "degraded"
    assert outcome["outcome_axes"]["execution"]["reason_code"] == "tool_failure"
    assert outcome["outcome_axes"]["execution"]["failure"]["verification_failures"][0][
        "status"
    ] == "artifact_output_error"


def test_deferred_child_result_prevents_clean_solved_outcome():
    from ouroboros.outcomes import derive_loop_outcome

    trace = {
        "tool_calls": [],
        "reasoning_notes": [],
        "child_result_dispositions": {
            "current": [{"child_task_id": "child1", "disposition": "deferred"}],
            "deferred_count": 1,
        },
    }
    outcome = derive_loop_outcome("Best available answer", {}, trace)
    assert outcome["outcome_axes"]["execution"]["status"] == "degraded"
    assert outcome["outcome_axes"]["objective"]["status"] == "best_effort"
    assert outcome["reason_code"] == "child_results_deferred"


def test_deferred_child_suffix_is_not_misclassified_as_delivery_control_failure():
    from ouroboros.outcomes import derive_loop_outcome

    trace = {
        "tool_calls": [],
        "reasoning_notes": [],
        "delivery_candidate": {
            "degraded": True,
            "degraded_reason": "host_child_status_suffix",
        },
        "child_result_dispositions": {
            "current": [{"child_task_id": "child1", "disposition": "deferred"}],
            "deferred_count": 1,
        },
    }
    outcome = derive_loop_outcome("Best available answer", {}, trace)
    assert outcome["outcome_axes"]["execution"]["failure"]["kind"] == "child_result_disposition"
    assert outcome["outcome_axes"]["objective"]["status"] == "best_effort"
    assert outcome["outcome_axes"]["objective"]["source"] == "child_result_disposition"
    assert outcome["reason_code"] == "child_results_deferred"

    trace["delivery_candidate"]["degraded_reason"] = (
        "invalid_delivery_control_after_repair"
    )
    invalid_control = derive_loop_outcome("Best available answer", {}, trace)
    assert invalid_control["outcome_axes"]["execution"]["failure"]["kind"] == (
        "finalization_control"
    )
    assert invalid_control["outcome_axes"]["objective"]["status"] == "degraded"
    assert invalid_control["outcome_axes"]["objective"]["source"] == (
        "delivery_finalization_control"
    )
    # The candidate's OWN typed cause survives: the generic code is reserved for a
    # degradation that reports no reason, so a record can no longer name a
    # malformed control object when the real cause was something else.
    assert invalid_control["reason_code"] == "invalid_delivery_control_after_repair"
    # The degradation FACT now reaches the record itself, which is what the
    # benchmark run-summary and result-index readers already look for.
    assert invalid_control["degraded"] is True
    assert invalid_control["degraded_reason"] == "invalid_delivery_control_after_repair"


def test_new_delivery_replacement_does_not_inherit_old_host_pass(tmp_path):
    import hashlib

    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    old_hash = hashlib.sha256(b"old accepted answer").hexdigest()
    trace = {
        "acceptance_decision": {"status": "accepted"},
        "review_decision": {"panel_id": "old-panel", "binding_hash": "old-binding"},
        "review_runs": [{
            "authority": "host_root",
            "candidate_hash": old_hash,
            "panel_id": "old-panel",
            "binding_hash": "old-binding",
            "evidence_revision": "old-review-evidence",
            "fence_hash": "old-fence",
            "aggregate_signal": "PASS",
        }],
        "tool_calls": [],
        "reasoning_notes": [],
    }
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "parent1"
    ctx = loop._RoundLimitContext(
        [{"role": "user", "content": "task"}],
        SimpleNamespace(),
        "test-model",
        "medium",
        0,
        tmp_path / "logs",
        "parent1",
        1,
        None,
        {},
        "",
        False,
        10,
        drive_root=tmp_path,
        status_drive_root=tmp_path,
        root_task_id="parent1",
    )
    loop._finalize_limit_ctx(ctx, registry, trace)

    replacement = loop._replace_delivery_candidate(
        registry, ctx, trace, "new replacement answer", control="replace",
    )

    binding = replacement.acceptance_binding
    assert replacement.content_sha256 != old_hash
    assert binding["candidate_sha256"] == replacement.content_sha256
    assert binding["acceptance_status"] == "unaccepted"
    assert binding["authoritative"] is False
    assert binding["panel_id"] == ""
    assert binding["binding_hash"] == ""
    assert "review_evidence_revision" not in binding


def test_same_text_replacement_after_evidence_change_does_not_inherit_old_pass(
    tmp_path,
):
    import hashlib

    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    answer = "Same complete text across evidence revisions."
    answer_hash = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    trace = {"tool_calls": [], "reasoning_notes": []}
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "parent1"
    ctx = loop._RoundLimitContext(
        [{"role": "user", "content": "task"}],
        SimpleNamespace(),
        "test-model",
        "medium",
        0,
        tmp_path / "logs",
        "parent1",
        1,
        None,
        {},
        "",
        False,
        10,
        drive_root=tmp_path,
        status_drive_root=tmp_path,
        root_task_id="parent1",
    )
    loop._finalize_limit_ctx(ctx, registry, trace)
    original = loop._replace_delivery_candidate(
        registry, ctx, trace, answer, control="candidate",
    )
    trace.update({
        "review_decision": {"panel_id": "old-panel", "binding_hash": "old-binding"},
        "review_runs": [{
            "authority": "host_root",
            "candidate_hash": answer_hash,
            "panel_id": "old-panel",
            "binding_hash": "old-binding",
            "evidence_revision": "old-review-evidence",
            "fence_hash": "old-fence",
            "aggregate_signal": "PASS",
        }],
    })
    # The authoritative binding the retired acceptance binder wrote for that panel, as a
    # candidate restored from a pre-fold record still carries it.
    original.acceptance_binding = {
        **original.acceptance_binding, "acceptance_status": "pass", "authoritative": True,
        "panel_id": "old-panel", "binding_hash": "old-binding", "fence_hash": "old-fence",
        "review_evidence_revision": "old-review-evidence",
    }
    assert original.acceptance_binding["candidate_sha256"] == answer_hash

    trace["tool_calls"].append({
        "tool": "write_file",
        "status": "ok",
        "result": "new evidence",
        "is_error": False,
    })
    replacement = loop._replace_delivery_candidate(
        registry, ctx, trace, answer, control="replace",
    )

    assert replacement.content_sha256 == original.content_sha256
    assert replacement.revision == original.revision + 1
    assert replacement.evidence_revision == original.evidence_revision + 1
    assert replacement.acceptance_binding["acceptance_status"] == "unaccepted"
    assert replacement.acceptance_binding["authoritative"] is False
    assert replacement.acceptance_binding["panel_id"] == ""
    assert replacement.acceptance_binding["binding_hash"] == ""
    assert "review_evidence_revision" not in replacement.acceptance_binding


def test_production_final_candidate_reaches_the_result_review_as_its_exact_bytes(tmp_path, monkeypatch):
    """The answer is the author's semantic reply (owner, 29.10): the result review, the candidate, its
    identity and the loop's return carry it as written; only the trace note behind the stored summary
    is redacted."""
    import hashlib

    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.utils import sanitize_tool_result_for_log

    answer = "Complete answer; removed ghp_abcdefghijklmnopqrstuvwxyz1234567890AB."
    safe_answer = sanitize_tool_result_for_log(answer)
    reviewed_contents = []
    reviewed_candidates = []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: ({"role": "assistant", "content": answer}, 0.0),
    )

    def record_the_reviewed_result(ctx, content, _llm_trace, _hold):
        reviewed_contents.append(content)
        reviewed_candidates.append(ctx._delivery_candidate)
        return False

    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", record_the_reviewed_result)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {
        "budget_drive_root": str(tmp_path),
        "root_task_id": "parent1",
    }

    result, _usage, trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": "do the work"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text, *, incident=None: None,
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )

    candidate = reviewed_candidates[0]
    binding = trace["delivery_candidate"]["acceptance_binding"]
    assert safe_answer != answer
    assert reviewed_contents == [answer]
    assert candidate.full_text == answer
    assert result == answer
    assert trace["reasoning_notes"][-1] == safe_answer
    assert trace["delivery_candidate"]["content_sha256"] == hashlib.sha256(
        answer.encode("utf-8")
    ).hexdigest()
    assert binding["candidate_sha256"] == trace["delivery_candidate"]["content_sha256"]
    assert binding["acceptance_status"] == "unaccepted"
    assert binding["authoritative"] is False


def test_budget_dispatch_rail_preserves_current_candidate_and_exact_binding(
    tmp_path, monkeypatch,
):
    import ouroboros.loop as loop
    import ouroboros.usage_accounting as accounting
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.tools.registry import ToolRegistry

    logs = tmp_path / "logs"
    logs.mkdir()
    events = queue.Queue()
    original = "Complete answer retained before the service re-loop."
    calls = 0
    historical_binding = {}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"role": "assistant", "content": original}, 0.0
        raise accounting.BudgetExceeded(
            "root limit closed", limit_scope="root", root_task_id="parent1",
        )

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {
        "budget_drive_root": str(tmp_path),
        "root_task_id": "parent1",
    }

    def request_another_round(ctx, _content, llm_trace, hold):
        from tests.test_delivery_forced_finalization import _bind_host_pass

        candidate = ctx._delivery_candidate
        _bind_host_pass(loop, registry, llm_trace, candidate)
        historical_binding.update(candidate.acceptance_binding)
        hold("[Result review] The review's findings were shown before delivery.")
        return True

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", request_another_round)
    monkeypatch.setattr(
        accounting,
        "usage_breakdown",
        lambda *_args, **_kwargs: {"physical_calls": 1, "integrity_degraded": False},
    )
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")

    result, usage, trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": "do the work"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=logs,
        emit_progress=lambda _text, *, incident=None: None,
        incoming_messages=queue.Queue(),
        event_queue=events,
        task_id="parent1",
        drive_root=tmp_path,
    )

    assert result == original
    assert calls == 2
    assert usage["reason_code"] == "budget_exhausted"
    assert usage["resource_limit"]["status"] == "resource_limited"
    assert usage["_best_effort_extracted"] is True
    assert trace["resource_limit"] == usage["resource_limit"]
    assert trace["delivery_candidate"]["finalization_control"] == "budget_preserve"
    assert trace["delivery_candidate"]["degraded"] is True
    assert trace["delivery_candidate"]["acceptance_binding"] == historical_binding
    assert trace["forced_finalization"]["source"] == "budget_preserve"
    outcome = derive_loop_outcome(result, usage, trace)
    assert outcome["outcome_axes"]["execution"]["status"] == "best_effort"
    assert outcome["outcome_axes"]["execution"]["resource_limit"] == usage["resource_limit"]


def test_round_limit_model_answer_replaces_candidate_with_unaccepted_binding(
    tmp_path, monkeypatch,
):
    import hashlib

    from ouroboros.outcomes import derive_loop_outcome

    original = "Complete answer before the finalization rail."
    forced = "Complete forced answer with the latest verified state."
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    result, usage, trace, calls = _run_loop(
        tmp_path,
        monkeypatch,
        [original, forced],
        review_holds=[True],
    )

    forced_sha = hashlib.sha256(forced.encode("utf-8")).hexdigest()
    candidate = trace["delivery_candidate"]
    binding = candidate["acceptance_binding"]
    assert result == forced
    assert len(calls) == 2
    assert usage["reason_code"] == "round_limit"
    assert usage["_best_effort_extracted"] is True
    assert candidate["content_sha256"] == forced_sha
    assert candidate["revision"] == 2
    assert candidate["finalization_control"] == "forced_replace:round_limit"
    assert candidate["degraded"] is True
    assert binding["candidate_sha256"] == forced_sha
    assert binding["acceptance_status"] == "unaccepted"
    assert binding["authoritative"] is False
    assert binding["binding_hash"] == ""
    assert trace["forced_finalization"]["source"] == "model"
    assert trace["forced_finalization"]["candidate_revision"] == 2
    outcome = derive_loop_outcome(result, usage, trace)
    assert outcome["outcome_axes"]["execution"]["status"] == "best_effort"
    assert outcome["outcome_axes"]["objective"]["status"] == "degraded"


def test_round_limit_caller_merges_distinct_returned_trace(tmp_path, monkeypatch):
    import ouroboros.loop as loop

    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")

    def fake_round_limit(ctx):
        ctx.accumulated_usage.update({
            "execution_status": "failed",
            "reason_code": "round_limit",
        })
        return "forced", ctx.accumulated_usage, {"forced_trace_marker": "merged"}

    monkeypatch.setattr(loop, "_handle_round_limit", fake_round_limit)
    result, _usage, trace, _calls = _run_loop(
        tmp_path,
        monkeypatch,
        ["candidate"],
        review_holds=[True],
    )

    assert result == "forced"
    assert trace["forced_trace_marker"] == "merged"


def test_forced_fallback_rejects_stale_delivery_candidate(tmp_path, monkeypatch):
    import hashlib

    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "parent1"
    registry._ctx.task_metadata = {
        "budget_drive_root": str(tmp_path),
        "root_task_id": "parent1",
    }
    trace = {"tool_calls": [], "reasoning_notes": []}
    ctx = loop._RoundLimitContext(
        [{"role": "user", "content": "task"}],
        SimpleNamespace(),
        "test-model",
        "medium",
        1,
        tmp_path / "logs",
        "parent1",
        2,
        None,
        {},
        "",
        False,
        10,
        drive_root=tmp_path,
    )
    loop._finalize_limit_ctx(ctx, registry, trace)
    candidate = loop._replace_delivery_candidate(
        registry, ctx, trace, "old complete answer", control="candidate",
    )
    trace["tool_calls"].append({
        "tool": "write_file",
        "status": "ok",
        "result": "new evidence",
        "is_error": False,
    })
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_args, **_kwargs: (None, 0.0))

    # The rail publishes the bytes it is given; a host notice quoting a provider sanitized that
    # quote where it was composed (``loop_transport``).
    fallback = "host fallback after ghp_abcdefghijklmnopqrstuvwxyz1234567890AB"
    text, usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text=fallback,
        reason_code="provider_unavailable",
    )

    rebound = ctx.delivery_candidate
    assert text == fallback
    assert rebound.full_text == fallback
    assert text != candidate.full_text
    assert usage["reason_code"] == "provider_unavailable"
    assert not usage.get("_best_effort_extracted")
    assert returned_trace["delivery_candidate"]["evidence_current"] is True
    assert returned_trace["delivery_candidate"]["finalization_control"] == (
        "forced_replace:provider_unavailable"
    )
    assert returned_trace["delivery_candidate"]["content_sha256"] == (
        hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    assert returned_trace["forced_finalization"]["source"] == "host_fallback"


def test_held_complete_prose_can_be_selected_by_its_exact_hash(tmp_path, monkeypatch):
    """No length classifier promotes prose; explicit selection preserves the whole held row."""
    from ouroboros.outcomes import derive_loop_outcome

    original = "Original complete answer covering the child conclusions."
    final_prose = "Implementation complete. " + "Every relevant correction is recorded here. " * 10
    invalid_json = json.dumps({"delivery_control": "finalize"})
    result, usage, trace, calls = _run_loop(tmp_path, monkeypatch,
        [original, invalid_json, final_prose, _finish_response(final_prose, by_hash=True)],
        child=True, bind_child_before_second=True)
    assert len(calls) == 4 and result == final_prose
    assert trace["delivery_candidate"]["degraded"] is False
    assert any(row.get("role") == "assistant" and row.get("content") == final_prose for row in calls[-1])
    assert calls[-1][-1]["role"] == "user"
    execution = derive_loop_outcome(result, usage, trace)["outcome_axes"]["execution"]
    assert execution["status"] == "ok" and execution["reason_code"] != "delivery_control_degraded"
    _assert_no_repair_round(calls)


@pytest.mark.parametrize("interim", ["OK.", "An interim note with much more text than the retained complete answer. " * 15])
def test_short_and_long_held_prose_never_select_themselves(tmp_path, monkeypatch, interim):
    original = "Original complete answer."
    progress = []
    result, _usage, trace, calls = _run_loop(tmp_path, monkeypatch,
        [original, interim, interim, _finish_response(original, by_hash=True)],
        child=True, bind_child_before_second=True, progress=progress)
    assert len(calls) == 4 and result == original
    assert trace["delivery_candidate"]["degraded"] is False
    assert sum(row.get("role") == "assistant" and row.get("content") == interim for row in calls[-1]) == 2
    assert calls[-1][-1]["role"] == "user"
    assert sum(row["text"] == interim.strip() and row.get("narration") is True for row in progress) == 2
    _assert_no_repair_round(calls)


def test_quiz_answer_and_parent_message_reopen_a_sealed_root_for_owner_follow_up(tmp_path, monkeypatch):
    """P1-6(g), pinned at the production seam (loop_delivery's post-answer admission
    drain): an owner quiz answer at a root and a parent's task message to a child grow
    the directive corpus, so a sealed root (``_release_for_owner_followup``) is reopened
    and another round runs; a host system frame (a settled plan wave) changes nothing."""
    import threading

    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, write_owner_message, write_task_message
    from tests.test_delivery_forced_finalization import _forced_test_context

    def run(write_followups):
        loop, registry, ctx, trace = _forced_test_context(tmp_path / str(len(released_runs)))
        drive = ctx.drive_root
        monkeypatch.setattr(loop, "_compute_subagent_handoff", lambda *_a, **_k: None)
        monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a, **_k: False)
        monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", lambda *_a, **_k: False)
        released = []
        monkeypatch.setattr(loop, "_release_for_owner_followup",
                            lambda *a, **k: released.append((a, k)))
        registry._ctx._task_acceptance_sealed_fence_token = "sealed-root"
        registry._ctx.owner_message_admission_lock = threading.Lock()
        registry._ctx.owner_message_admission_agent = SimpleNamespace(
            _accepting_owner_messages=True, _busy=True, _current_task_id="parent1")
        registry._ctx.budget_drive_root = str(drive)
        registry._ctx.task_attempt = 1
        write_followups(drive)
        result = loop._no_tool_final_answer("Final answer.", ctx, trace, registry, queue.Queue(), set(), lambda _m: None)
        released_runs.append(released)
        # A grown corpus forces another round (None); an unchanged one delivers the answer.
        return released, [row["source"] for row in getattr(registry._ctx, "_owner_directives", [])], result

    released_runs: list = []
    quiz = "[Owner quiz answer] quiz q1 — asked t0, answered t1.\nThe owner chose option 2: postgres"
    calls, sources, result = run(lambda drive: write_owner_message(drive, quiz, task_id="parent1", msg_id="qa-1", kind=KIND_QUIZ_ANSWER))
    assert len(calls) == 1 and sources == ["owner_quiz_answer"] and result is None
    calls, sources, result = run(lambda drive: write_task_message(drive, "Use the Q3 numbers only", "parent1",
                                                                  source_task_id="root-0", msg_id="pm-1"))
    assert len(calls) == 1 and sources == ["principal_task_message"] and result is None
    calls, sources, result = run(lambda drive: write_task_message(
        drive, "Plan review wave abcd1234: 3 released reviewer slot(s) settled", "parent1",
        source_task_id="parent1", provenance="system", msg_id="sys-1"))
    assert calls == [] and sources == [] and result is not None

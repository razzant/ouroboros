from __future__ import annotations

import json
import queue
from pathlib import Path
from types import SimpleNamespace

from tests._delivery_candidate_shared import (
    write_child as _write_child,
    write_confirmed_disposition_fixture as _write_confirmed_disposition,
)


def _run_loop(
    tmp_path,
    monkeypatch,
    responses,
    acceptance_results=None,
    *,
    child=False,
    bind_child_before_second=False,
):
    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    tmp_path.mkdir(parents=True, exist_ok=True)
    if child:
        _write_child(tmp_path)
    answers = iter(responses)
    calls = []

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

    review_states = iter(acceptance_results or [])

    def fake_acceptance(**kwargs):
        outcome = next(review_states, False)
        if outcome:
            # v6.71.1: the acceptance path no longer arms delivery-control (an
            # improvement pass is an ordinary answer round). These tests exercise
            # the CONTROL MECHANICS (keep/replace/repair/duplicate-key), which
            # still arm on the services/handoff/evidence-changed lanes — emulate
            # such a lane by arming through the real helper.
            ctx_shim = SimpleNamespace(
                messages=kwargs["messages"],
                task_id=kwargs["task_id"],
                drive_root=kwargs["drive_root"],
                status_drive_root=kwargs["drive_root"],
                drive_logs=Path(str(kwargs["drive_root"])) / "logs",
                root_task_id=kwargs["task_id"],
            )
            loop._arm_delivery_control(kwargs["tools"], ctx_shim, kwargs["llm_trace"])
        return outcome

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", fake_acceptance)
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
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )
    return result, usage, trace, calls


def test_handoff_service_notices_cannot_erase_full_candidate(tmp_path, monkeypatch):
    from ouroboros.outcomes import derive_loop_outcome

    original = "Complete original answer with all child conclusions."
    result, usage, trace, calls = _run_loop(
        tmp_path,
        monkeypatch,
        [
            original,
            "service notice: child status refreshed",
            "service notice again",
        ],
        child=True,
        bind_child_before_second=True,
    )
    assert result == original
    assert len(calls) == 3
    assert trace["delivery_candidate"]["finalization_control"] == "degraded_preserve"
    assert trace["delivery_candidate"]["degraded"] is True
    assert trace["delivery_candidate"]["evidence_current"] is True
    assert trace["delivery_candidate"]["acceptance_binding"]["authoritative"] is False
    outcome = derive_loop_outcome(result, usage, trace)
    assert outcome["outcome_axes"]["execution"]["status"] == "degraded"


def test_mutating_tool_acknowledgements_cannot_erase_full_candidate(tmp_path, monkeypatch):
    original = "Complete original answer with all implementation and verification details."
    result, _usage, trace, calls = _run_loop(
        tmp_path,
        monkeypatch,
        [
            original,
            {
                "content": None,
                "tool_calls": [{
                    "id": "write-1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({
                            "path": "effect.txt",
                            "content": "durable tool effect",
                        }),
                    },
                }],
            },
            "Review completed.",
            "Everything is done now.",
        ],
        acceptance_results=[True, False],
    )

    assert (tmp_path / "effect.txt").read_text(encoding="utf-8") == "durable tool effect"
    assert result == original
    assert len(calls) == 4
    assert trace["delivery_candidate"]["revision"] == 1
    assert trace["delivery_candidate"]["finalization_control"] == "degraded_preserve"
    assert trace["delivery_candidate"]["degraded"] is True
    assert trace["delivery_candidate"]["evidence_current"] is True
    assert trace["delivery_candidate"]["acceptance_binding"]["authoritative"] is False


def test_replace_control_rejects_non_string_full_answer(tmp_path, monkeypatch):
    original = "Complete original answer."
    result, _usage, trace, calls = _run_loop(
        tmp_path,
        monkeypatch,
        [
            original,
            json.dumps({"delivery_control": "replace", "full_answer": {"text": "not complete"}}),
            json.dumps({"delivery_control": "keep"}),
        ],
        acceptance_results=[True, False],
    )

    assert result == original
    assert len(calls) == 3
    assert trace["delivery_candidate"]["revision"] == 1
    assert trace["delivery_candidate"]["finalization_control"] == "keep"


def test_duplicate_delivery_control_key_enters_repair_then_keeps_candidate(
    tmp_path,
    monkeypatch,
):
    original = "Complete original answer."
    duplicate_control = (
        '{"delivery_control":"keep","delivery_control":"replace",'
        '"full_answer":"Ambiguous replacement."}'
    )
    result, _usage, trace, calls = _run_loop(
        tmp_path,
        monkeypatch,
        [original, duplicate_control, json.dumps({"delivery_control": "keep"})],
        acceptance_results=[True, False],
    )

    assert result == original
    assert len(calls) == 3
    assert any(
        "[DELIVERY_CONTROL_REPAIR]" in str(message.get("content") or "")
        for message in calls[2]
    )
    assert trace["delivery_candidate"]["finalization_control"] == "keep"
    assert trace["delivery_candidate"]["degraded"] is False


def test_duplicate_full_answer_key_twice_preserves_prior_candidate_degraded(
    tmp_path,
    monkeypatch,
):
    original = "Complete original answer."
    duplicate_answer = (
        '{"delivery_control":"replace","full_answer":"First answer.",'
        '"full_answer":"Second answer."}'
    )
    result, _usage, trace, calls = _run_loop(
        tmp_path,
        monkeypatch,
        [original, duplicate_answer, duplicate_answer],
        acceptance_results=[True, False],
    )

    assert result == original
    assert len(calls) == 3
    assert any(
        "[DELIVERY_CONTROL_REPAIR]" in str(message.get("content") or "")
        for message in calls[2]
    )
    assert trace["delivery_candidate"]["finalization_control"] == "degraded_preserve"
    assert trace["delivery_candidate"]["degraded"] is True
    assert trace["delivery_candidate"]["degraded_reason"] == (
        "invalid_delivery_control_after_repair"
    )


def test_service_round_can_keep_or_replace_complete_candidate(tmp_path, monkeypatch):
    original = "Complete original answer."
    result, _usage, trace, _calls = _run_loop(
        tmp_path,
        monkeypatch,
        [original, json.dumps({"delivery_control": "keep"})],
        acceptance_results=[True, False],
    )
    assert result == original
    assert trace["delivery_candidate"]["revision"] == 1
    assert trace["delivery_candidate"]["finalization_control"] == "keep"

    replacement = "Complete replacement answer."
    result2, _usage2, trace2, _calls2 = _run_loop(
        tmp_path / "replace",
        monkeypatch,
        [original, json.dumps({"delivery_control": "replace", "full_answer": replacement})],
        acceptance_results=[True, False],
    )
    assert result2 == replacement
    assert trace2["delivery_candidate"]["revision"] == 2
    assert trace2["delivery_candidate"]["finalization_control"] == "replace"


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


def test_service_outputs_finalize_before_acceptance_and_require_replacement(tmp_path, monkeypatch):
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
            json.dumps({"delivery_control": "replace", "full_answer": replacement}),
        ],
    )

    assert result == replacement
    assert len(model_calls) == 2
    assert "keep is NOT allowed" in model_calls[1][-1]["content"]
    assert trace["delivery_candidate"]["revision"] == 2
    assert trace["delivery_candidate"]["finalization_control"] == "replace"
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
    assert invalid_control["reason_code"] == "delivery_control_degraded"


def test_delivery_acceptance_binding_uses_exact_active_host_verdict(tmp_path):
    import hashlib

    from ouroboros import loop_delivery
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx._delivery_evidence_revision = 7
    registry._ctx._task_acceptance_sealed_fence_token = "current-fence"
    answer_hash = hashlib.sha256(b"exact answer").hexdigest()

    incomplete = loop_delivery._delivery_acceptance_binding(
        registry,
        {
            "review_runs": [{
                "authority": "host_root",
                "candidate_hash": answer_hash,
                "aggregate_signal": "PASS",
            }],
        },
        answer_hash,
    )
    assert incomplete["acceptance_status"] == "unaccepted"
    assert incomplete["authoritative"] is False

    complete_but_historical = loop_delivery._delivery_acceptance_binding(
        registry,
        {
            "review_runs": [{
                "authority": "host_root",
                "candidate_hash": answer_hash,
                "panel_id": "old-panel",
                "binding_hash": "old-binding",
                "evidence_revision": "old-evidence",
                "aggregate_signal": "PASS",
            }],
            "review_decision": {"eligibility": "not_eligible"},
        },
        answer_hash,
    )
    assert complete_but_historical["acceptance_status"] == "unaccepted"
    assert complete_but_historical["authoritative"] is False

    for verdict in ("PASS", "FAIL", "DEGRADED"):
        suffix = verdict.lower()
        trace = {
            "review_decision": {
                "panel_id": f"panel-{suffix}",
                "binding_hash": f"binding-{suffix}",
            },
            "review_runs": [{
                "authority": "host_root",
                "candidate_hash": answer_hash,
                "panel_id": f"panel-{suffix}",
                "binding_hash": f"binding-{suffix}",
                "evidence_revision": f"review-evidence-{suffix}",
                "fence_hash": f"review-fence-{suffix}",
                "aggregate_signal": verdict,
            }],
        }

        binding = loop_delivery._delivery_acceptance_binding(registry, trace, answer_hash)

        assert binding["candidate_sha256"] == answer_hash
        assert binding["evidence_revision"] == 7
        assert binding["review_evidence_revision"] == f"review-evidence-{suffix}"
        assert binding["acceptance_status"] == suffix
        assert binding["authoritative"] is True
        assert binding["panel_id"] == f"panel-{suffix}"
        assert binding["binding_hash"] == f"binding-{suffix}"
        assert binding["fence_hash"] == f"review-fence-{suffix}"


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
    from ouroboros import loop_delivery
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
    original.acceptance_binding = loop_delivery._delivery_acceptance_binding(
        registry, trace, answer_hash,
    )
    assert original.acceptance_binding["authoritative"] is True

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


def test_production_final_candidate_binds_exact_host_panel(tmp_path, monkeypatch):
    import hashlib

    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    answer = "Complete answer reviewed by the host panel."

    class FakeLLM:
        def default_model(self):
            return "test-model"

    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: ({"role": "assistant", "content": answer}, 0.0),
    )

    def record_exact_host_panel(*, content, llm_trace, **_kwargs):
        candidate_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        llm_trace["review_decision"] = {
            "panel_id": "panel-production",
            "binding_hash": "binding-production",
        }
        llm_trace.setdefault("review_runs", []).append({
            "authority": "host_root",
            "candidate_hash": candidate_hash,
            "panel_id": "panel-production",
            "binding_hash": "binding-production",
            "evidence_revision": "review-evidence-production",
            "fence_hash": "fence-production",
            "aggregate_signal": "PASS",
        })
        return False

    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", record_exact_host_panel)
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
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )

    binding = trace["delivery_candidate"]["acceptance_binding"]
    assert result == answer
    assert binding["candidate_sha256"] == hashlib.sha256(answer.encode("utf-8")).hexdigest()
    assert binding["acceptance_status"] == "pass"
    assert binding["authoritative"] is True
    assert binding["panel_id"] == "panel-production"
    assert binding["binding_hash"] == "binding-production"
    assert binding["review_evidence_revision"] == "review-evidence-production"


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

    def request_another_round(**_kwargs):
        candidate = registry._ctx._delivery_candidate
        historical_binding.update({
            "candidate_sha256": candidate.content_sha256,
            "evidence_revision": candidate.evidence_revision,
            "acceptance_status": "pass",
            "authoritative": True,
            "panel_id": "panel-exact",
            "binding_hash": "binding-exact",
        })
        candidate.acceptance_binding = dict(historical_binding)
        return True

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", request_another_round)
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
        emit_progress=lambda _text: None,
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
        acceptance_results=[True],
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
        acceptance_results=[True],
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

    text, usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="host fallback",
        reason_code="provider_unavailable",
    )

    assert text == "host fallback"
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

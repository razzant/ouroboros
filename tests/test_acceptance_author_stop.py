"""TZ-2 C4: an explicit author stop stays a stop (owner-confirmed defect).

The reported chain: Main stopped while the task still had running services; the
host stopped them at delivery (``services_stopped``), which changed the delivery
evidence fingerprint; the finish-style freshness check then rejected the stop, a
reviewer panel was bought, and an advisory finish turned the objective green — a
Done card over text saying "not ready".

A stop grants nothing, so nothing about it needs to be fresh: ``finish_task``
with ``action="stop"`` records ``task_completion`` at once and never reaches the
result review at delivery; only a finish is reviewed. The objective reads the
stop as a failed author stop and the task card carries the author's rationale —
for a stop recorded today and for the stop shapes historical records carry.
"""

from __future__ import annotations

import json
import queue
import subprocess
from types import SimpleNamespace

import pytest

STOP_RATIONALE = "Not ready: the export endpoint still fails on empty input."
STOP_TEXT = STOP_RATIONALE + " Stopping with the work unfinished."


def _fail_result_wave(monkeypatch, reviews: list) -> None:
    """Replace only the paid wave under the REAL delivery step: each new result
    subject is reviewed FAIL once (a settled ledger-shaped record the step binds);
    the same subject again is that settled record at $0."""
    from ouroboros.tools import review_change, review_result

    records: dict = {}
    asked: dict = {}

    def settled(_ctx, subject_sha, rows=None):
        asked["sha"] = subject_sha
        return records.get(subject_sha)

    def wave(_ctx, *, subject, answer="", **_kwargs):
        assert subject == "result"
        reviews.append(answer)
        sha = asked["sha"]
        records[sha] = {
            "record_id": f"rl-fixture-{len(records) + 1}", "state": "settled", "surface": "result",
            "subject": {"diff_sha": sha},
            "verdict": {"aggregate": "FAIL", "reason": "", "advisory_findings": [], "critical_findings": [
                {"severity": "critical", "seat_id": "critic", "item": "output", "reason": "Fix the output."}]},
        }
        return {"state": "settled", "record_id": records[sha]["record_id"]}

    monkeypatch.setattr(review_result, "settled_result_record", settled)
    monkeypatch.setattr(review_change, "run_review_change", wave)


def _run_stop_loop(tmp_path, monkeypatch, responses, *, stop_services, produced=False, mode="required"):
    """A real loop: real tool round, real delivery step, real result review at delivery.

    Only the model, the reviewers' paid wave and the service teardown are substituted.
    """
    import ouroboros.loop as loop
    from ouroboros.tools import services as services_mod
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_loop_acceptance_gate import _seed_acceptance_root

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", mode)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "3")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "12")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_args: False)
    panels: list = []
    _fail_result_wave(monkeypatch, panels)
    teardowns = {"count": 0}

    def fake_stop(_ctx):
        teardowns["count"] += 1
        if teardowns["count"] == 1 and stop_services:
            return [{"service_id": "preview", "name": "preview", "lifecycle": "stopped",
                     "artifact_outputs": "report.html was captured"}]
        return []

    monkeypatch.setattr(services_mod, "stop_task_services", fake_stop)
    answers = iter(responses)
    model_inputs: list = []

    def fake_call(_llm, request_messages, *_args, **kwargs):
        model_inputs.append([dict(row) for row in request_messages])
        answer = next(answers)
        # The scripted Main replaces the transport, including its actual-context
        # observer: the request whose response returns exposes the feedback it carried.
        if callable(kwargs.get("model_context_observer")):
            kwargs["model_context_observer"](request_messages)
        if isinstance(answer, dict):
            return {"role": "assistant", **answer}, 0.0
        return {"role": "assistant", "content": answer}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    for args in (["init"], ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                           "commit", "--allow-empty", "-m", "fixture baseline"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    ctx = registry._ctx
    task_id = "stop-root"
    _seed_acceptance_root(data, task_id, ctx)
    if produced:
        from ouroboros.artifacts import store_task_artifact_bytes
        store_task_artifact_bytes(data, task_id, "export.txt", b"Export implementation under review.\n")
    ctx.is_direct_chat = False
    ctx.task_attempt = 1
    ctx.current_chat_id = 1
    result, usage, trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": "Ship the export endpoint."}],
        tools=registry, llm=SimpleNamespace(default_model=lambda: "test-model"),
        drive_logs=data / "logs", emit_progress=lambda _text, *, incident=None: None,
        incoming_messages=queue.Queue(), task_id=task_id, drive_root=data,
    )
    return SimpleNamespace(result=result, usage=usage, trace=trace, panels=panels,
                           model_inputs=model_inputs, teardowns=teardowns["count"], ctx=ctx)


def _stop_tool_call():
    arguments = {"action": "stop", "answer": STOP_TEXT, "rationale": STOP_RATIONALE}
    return {"content": None, "tool_calls": [{
        "id": "stop-1", "type": "function",
        "function": {"name": "finish_task", "arguments": json.dumps(arguments)},
    }]}


def _finish_tool_call(answer):
    return {"content": None, "tool_calls": [{
        "id": "finish-selected", "type": "function",
        "function": {"name": "finish_task", "arguments": json.dumps({
            "action": "finish", "answer": answer,
        })},
    }]}


@pytest.mark.parametrize("mode", ["auto", "required", "off"])
def test_a_stop_survives_service_teardown_without_another_model_round(tmp_path, monkeypatch, mode):
    """A selected stop survives teardown without purchasing another author or critic."""
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.project_dialogue import _completion_verdict, completion_status_label

    run = _run_stop_loop(tmp_path, monkeypatch, [
        _stop_tool_call(),
    ], stop_services=True, mode=mode)
    assert len(run.model_inputs) == 1, "teardown must not buy another author reply for a stop"
    assert run.teardowns >= 1
    assert [event["kind"] for event in run.trace.get("verification_events") or []] == ["services_stopped"]
    assert run.trace["task_completion"]["action"] == "stop"
    assert run.trace["task_completion"]["rationale"] == STOP_RATIONALE
    assert run.panels == [], "an explicit stop must never buy a review"
    # #1602: the acknowledgement is a control fact, not a review that never ran.
    from ouroboros.outcomes import build_verification_ledger
    from ouroboros.review_projection import compact_review_projection

    assert not run.trace.get("review_runs")
    assert compact_review_projection(run.trace.get("review_runs"))["panels"] == []
    ledger = build_verification_ledger(task={"id": "stop-root"}, loop_outcome={"outcome_axes": {}},
                                       llm_trace=run.trace, artifact_bundle={})
    assert [row for row in ledger["entries"] if row["kind"] == "task_acceptance_review"] == []
    assert run.result == STOP_TEXT
    outcome = derive_loop_outcome(run.result, run.usage, run.trace)
    axes = outcome["outcome_axes"]
    assert axes["objective"]["status"] == "fail" and axes["objective"]["reason"] == "author_stop"
    record = {"status": "completed", "reason_code": str(outcome.get("reason_code") or "final_message"),
              "outcome_axes": axes}
    assert completion_status_label(record, {}) == "Failed"
    verdict = _completion_verdict(record, {})
    assert STOP_RATIONALE.rstrip(".") in verdict, verdict
    assert verdict.startswith("Ouroboros stopped with unfinished work")


def test_the_same_teardown_round_still_buys_the_review_for_a_finish(tmp_path, monkeypatch):
    """The control: a FINISH over changed evidence is not honoured unseen — its result is reviewed."""
    run = _run_stop_loop(tmp_path, monkeypatch, [
        _finish_tool_call("The export endpoint ships."),
        _finish_tool_call("The export endpoint ships."),
        _finish_tool_call("The export endpoint ships."),
        _finish_tool_call("The export endpoint ships."),
    ], stop_services=True, produced=True)
    assert run.panels, "a finish bound to stale evidence must still be reviewed"
    assert (run.trace.get("task_completion") or {}).get("action") != "stop"


def test_the_row_reason_slot_carries_the_stop_rationale():
    """The typed stop sentence, then the agent's own reason; a reviewer rationale stays off the row."""
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict, completion_status_label

    record = {"status": "completed", "reason_code": "final_message", "outcome_axes": {
        "execution": {"status": "ok"},
        "objective": {"status": "fail", "source": "task_acceptance_review", "reason": "author_stop",
                      "outcome_tier": "blocked_with_evidence"},
        "review": {"status": "skipped", "acceptance_decision": {
            "status": "finalized_unaccepted", "reason": "author_stop", "author_action": "stop",
            "enforcement": "advisory", "rationale": "reviewer prose that must not reach the row",
            "author_disposition": {"disposition": "", "action": "stop", "rationale": STOP_RATIONALE,
                                   "subject_hash": "s1", "reviewer_signal": "", "enforcement": "advisory",
                                   "recorded_at": "2026-09-25T00:00:00+00:00", "source": "author"}}}}}
    assert completion_status_label(record, {}) == "Failed"
    verdict = _completion_verdict(record, {})
    assert verdict == TASK_CAUSE_PHRASES["author_stop"][:-1] + " · " + STOP_RATIONALE
    assert "reviewer prose" not in verdict
    # No rationale recorded (a malformed or stale disposition): the typed sentence alone.
    record["outcome_axes"]["review"]["acceptance_decision"]["author_disposition"] = ""
    assert _completion_verdict(record, {}) == TASK_CAUSE_PHRASES["author_stop"]


def test_a_stop_after_an_earlier_review_finishes_without_a_keep_round(tmp_path, monkeypatch):
    """An informed stop after a FAIL review needs no keep/replacement or new review."""
    from ouroboros.outcomes import derive_loop_outcome

    run = _run_stop_loop(tmp_path, monkeypatch, [
        "The export endpoint ships.",                       # reviewed: FAIL, findings shown
        _stop_tool_call(),                                  # action-only stop after the findings
    ], stop_services=False, produced=True)
    assert len(run.model_inputs) == 2
    assert run.panels == ["The export endpoint ships."], "an honoured stop must never buy a second review"
    assert run.trace["task_completion"]["action"] == "stop"
    assert run.trace["task_completion"]["rationale"] == STOP_RATIONALE
    assert run.result == STOP_TEXT
    axes = derive_loop_outcome(run.result, run.usage, run.trace)["outcome_axes"]
    assert axes["objective"]["status"] == "fail" and axes["objective"]["reason"] == "author_stop"


def test_a_stop_recorded_under_exhausted_rounds_keeps_its_cause_and_its_rationale():
    """A2: when the review rounds ran out, ``_finish_advisory_author`` recorded the
    stop under its TRUE cause, ``review_cycles_exhausted``, with ``author_action``
    and the disposition's action ``stop``. The row keeps that cause's sentence and
    still carries the author's rationale; a finish, or a bare inherited stop action
    without the author's stop disposition, carries none."""
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _author_stop_rationale, _completion_verdict
    from ouroboros.review_records import recorded_author_stop

    author = {"action": "stop", "rationale": "Not ready:  the export endpoint still fails.", "source": "author"}
    structured = {"status": "finalized_unaccepted", "reason": "review_cycles_exhausted",
                  "author_action": "stop", "author_disposition": author}
    finish = {"status": "finalized_unaccepted", "reason": "review_cycles_exhausted", "author_action": "finish",
              "author_disposition": {"action": "finish", "rationale": "Shipping as is.", "source": "author"}}
    bare = {"status": "finalized_unaccepted", "reason": "review_cycles_exhausted", "author_action": "stop"}
    assert recorded_author_stop(structured) and recorded_author_stop({"reason": "author_stop"})
    assert _author_stop_rationale(structured) == "Not ready: the export endpoint still fails."
    for other in (finish, bare, {**structured, "reason": "capsule_spent"}, {}, None):
        assert not recorded_author_stop(other)
    assert _author_stop_rationale(finish) == _author_stop_rationale(bare) == ""
    record = {"status": "completed", "reason_code": "final_message", "outcome_axes": {
        "execution": {"status": "ok"},
        "objective": {"status": "fail", "source": "task_acceptance_review", "reason": "review_cycles_exhausted"},
        "review": {"status": "skipped", "acceptance_decision": structured}}}
    assert _completion_verdict(record, {}) == (
        TASK_CAUSE_PHRASES["review_cycles_exhausted"][:-1] + " · Not ready: the export endpoint still fails.")


@pytest.mark.parametrize("earlier_review", [True, False])
def test_evidence_changing_after_an_honoured_stop_neither_reopens_nor_replaces_it(tmp_path, monkeypatch, earlier_review):
    """(f) Evidence that changes only AFTER the stop was selected (a service seen stopped
    at the post-review evidence read) used to supersede the stop like a reviewed
    boundary and buy a review. A stop binds no evidence: no second author round or
    review runs, and the row reads the stop's cause, rationale and red objective."""
    import ouroboros.loop as loop
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.project_dialogue import _completion_verdict

    real_project, late = loop._project_child_result_dispositions, []

    def project(limit_ctx, llm_trace):
        if (getattr(limit_ctx.tools._ctx, "_completion_selected", None) or {}).get("action") == "stop" and not late:
            late.append(True)
            llm_trace.setdefault("verification_events", []).append({"kind": "services_stopped", "services": [
                {"service_id": "late", "name": "late", "lifecycle": "stopped"}]})
        return real_project(limit_ctx, llm_trace)

    monkeypatch.setattr(loop, "_project_child_result_dispositions", project)
    first = ["The export endpoint ships."] if earlier_review else []
    run = _run_stop_loop(tmp_path, monkeypatch, [*first, _stop_tool_call()], stop_services=False, produced=earlier_review)
    assert len(run.model_inputs) == len(first) + 1
    assert late and run.panels == first, "evidence after a stop must never buy a review"
    assert run.trace["task_completion"]["action"] == "stop"
    assert run.trace["task_completion"]["rationale"] == STOP_RATIONALE
    assert run.result == STOP_TEXT
    axes = derive_loop_outcome(run.result, run.usage, run.trace)["outcome_axes"]
    assert axes["objective"]["status"] == "fail" and axes["objective"]["reason"] == "author_stop"
    verdict = _completion_verdict({"status": "completed", "reason_code": "final_message", "outcome_axes": axes}, {})
    assert verdict.startswith("Ouroboros stopped with unfinished work") and STOP_RATIONALE.rstrip(".") in verdict

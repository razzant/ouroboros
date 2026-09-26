"""TZ-2 C4: an explicit author stop stays a stop (owner-confirmed defect).

The reported chain: Main calls ``task_acceptance_review(author_action=stop)``
while the task still has running services; the host stops them at delivery
(``services_stopped``), which changes the delivery evidence fingerprint; the
finish-style freshness check then rejected the stop, a reviewer panel was
bought, and the advisory ``author_finish`` recorded by ``_finish_cyber_acceptance``
turned the objective green — a Done card over text saying "not ready".

A stop grants nothing, so nothing about it needs to be fresh: it is recorded at
once as ``author_stop`` with no panel; only a finish binds to the reviewed
feedback and the three freshness facts. A later explicit finish still replaces
a stop through ``merge_agent_acceptance_stance`` (no new mechanism).
"""

from __future__ import annotations

import json
import queue
import subprocess
from types import SimpleNamespace

import pytest

STOP_RATIONALE = "Not ready: the export endpoint still fails on empty input."
STOP_TEXT = STOP_RATIONALE + " Stopping with the work unfinished."


def _fail_panel_result():
    from ouroboros.review_substrate import ReviewRunResult

    return ReviewRunResult(
        request={"surface": "task_acceptance", "policy": {"min_successful_slots": 1}},
        actors=[{"slot_id": "critic", "signal": "FAIL", "parsed": {
            "verdict": "FAIL", "outcome_tier": "best_effort", "completion_coach": "Fix the output.",
        }}], parsed_findings=[], aggregate_signal="FAIL",
    )


def _run_stop_loop(tmp_path, monkeypatch, responses, *, stop_services):
    """A real loop: real tool round, real nomination, real host acceptance pass.

    Only the model, the reviewer panel and the service teardown are substituted.
    """
    import ouroboros.loop as loop
    import ouroboros.review_substrate as review_substrate
    from ouroboros.tools import services as services_mod
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_loop_acceptance_gate import _seed_acceptance_root

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "required")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "3")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "12")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_args: False)
    monkeypatch.setattr(review_substrate, "triad_delivery_slots", lambda **_kw: [object()])
    panels: list = []

    def panel(ctx):
        panels.append(ctx.content)
        return _fail_panel_result()

    monkeypatch.setattr(loop, "_execute_task_acceptance_panel", panel)
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
    return {"content": None, "tool_calls": [{
        "id": "stop-1", "type": "function",
        "function": {"name": "task_acceptance_review", "arguments": json.dumps({
            "claim": STOP_TEXT, "author_action": "stop", "rationale": STOP_RATIONALE,
        })},
    }]}


def test_a_stop_survives_the_service_teardown_round_and_buys_no_panel(tmp_path, monkeypatch):
    """(a) stop → a running service → the host stops it at delivery → one more
    round between the cleanup and the panel → recorded ``author_stop``, no
    panel, not Done, and the agent's rationale in the row's reason slot."""
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.project_dialogue import _completion_verdict, completion_status_label

    run = _run_stop_loop(tmp_path, monkeypatch, [
        _stop_tool_call(),
        # The nomination stopped the service; the evidence changed; the host armed a
        # replacement round. Main restates the same unfinished stop.
        json.dumps({"delivery_control": "replace", "full_answer": STOP_TEXT}),
        STOP_TEXT,
    ], stop_services=True)
    assert run.teardowns >= 1
    assert [event["kind"] for event in run.trace.get("verification_events") or []] == ["services_stopped"]
    assert any("Task services were finalized before acceptance" in note
               for note in run.trace.get("reasoning_notes") or [])
    assert run.panels == [], "an explicit stop must never buy a reviewer panel"
    decision = run.trace["acceptance_decision"]
    assert decision["reason"] == "author_stop" and decision["author_action"] == "stop"
    assert decision["author_disposition"]["action"] == "stop"
    assert decision["author_disposition"]["rationale"] == STOP_RATIONALE
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


def test_the_same_teardown_round_still_buys_the_panel_for_a_finish(tmp_path, monkeypatch):
    """The control: a FINISH over changed evidence is not honoured — the panel runs."""
    run = _run_stop_loop(tmp_path, monkeypatch, [
        {"content": None, "tool_calls": [{
            "id": "finish-1", "type": "function",
            "function": {"name": "task_acceptance_review", "arguments": json.dumps({
                "claim": "The export endpoint ships.", "agent_disposition": "accepted",
                "author_action": "finish", "rationale": "Everything verified.",
            })},
        }]},
        json.dumps({"delivery_control": "replace", "full_answer": "The export endpoint ships."}),
        "The export endpoint ships.",
        "The export endpoint ships.",
    ], stop_services=True)
    assert run.panels, "a finish bound to stale evidence must still be reviewed"
    assert run.trace["acceptance_decision"]["reason"] != "author_stop"


def _finality_pass(tmp_path, monkeypatch):
    """The reviewer-bound host pass over a fake tool ctx (as test_review_author_finality)."""
    import ouroboros.loop as loop_mod
    import ouroboros.loop_acceptance_review as review
    from tests.test_loop_acceptance_gate import _seed_acceptance_root

    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "required")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "3")
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "advisory")
    tools_ctx = SimpleNamespace(_task_acceptance_reviewed=False, is_direct_chat=False,
                               drive_root=str(tmp_path), _owner_directives=[])
    _seed_acceptance_root(tmp_path, "author-root", tools_ctx)
    tools = SimpleNamespace(_ctx=tools_ctx)
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "answer.txt"}}]}
    messages = [{"role": "user", "content": "Solve the task."}]
    calls: list = []

    def panel(ctx):
        calls.append(ctx.content)
        return _fail_panel_result()

    monkeypatch.setattr(loop_mod, "_execute_task_acceptance_panel", panel)

    def run(text):
        return review._run_task_acceptance_review_once(
            tools=tools, content=text, task_id="author-root", task_type="task", llm_trace=trace,
            drive_root=tmp_path, messages=messages, emit_progress=lambda *_a, **_k: None)

    return SimpleNamespace(ctx=tools_ctx, trace=trace, messages=messages, calls=calls, run=run)


def test_a_later_explicit_finish_replaces_the_stop(tmp_path, monkeypatch):
    """(b) stop → later explicit finish → finish (the existing stance merge)."""
    from ouroboros.acceptance_settlement import expose_acceptance_feedback
    from ouroboros.loop_acceptance import merge_agent_acceptance_stance

    fx = _finality_pass(tmp_path, monkeypatch)
    assert fx.run("initial answer") is True
    expose_acceptance_feedback(fx.trace, fx.messages, "author-root")
    fx.trace["tool_calls"].append({"tool": "task_acceptance_review", "args": {}})
    merge_agent_acceptance_stance(fx.trace, {"explicit_finish": True, "author_action": "stop",
                                             "rationale": STOP_RATIONALE}, fx.ctx)
    assert fx.trace["acceptance_decision"]["author_action"] == "stop"
    fx.trace["tool_calls"].append({"tool": "task_acceptance_review", "args": {}})
    merge_agent_acceptance_stance(fx.trace, {"disposition": "partial", "explicit_finish": True,
                                             "author_action": "finish",
                                             "rationale": "Fixed the empty-input case after all."}, fx.ctx)
    assert fx.run("revised answer") is False
    decision = fx.trace["acceptance_decision"]
    assert decision["reason"] == "author_finish" and decision["author_action"] == "finish"
    assert decision["author_disposition"]["action"] == "finish"
    assert fx.calls == ["initial answer"]


@pytest.mark.parametrize("change", ["tools", "owner_directives", "evidence"])
def test_the_three_freshness_facts_bind_a_finish_but_never_a_stop(tmp_path, monkeypatch, change):
    from ouroboros.acceptance_settlement import expose_acceptance_feedback
    from ouroboros.loop_acceptance import merge_agent_acceptance_stance

    fx = _finality_pass(tmp_path, monkeypatch)
    assert fx.run("initial answer") is True
    expose_acceptance_feedback(fx.trace, fx.messages, "author-root")
    fx.trace["tool_calls"].append({"tool": "task_acceptance_review", "args": {}})
    merge_agent_acceptance_stance(fx.trace, {"explicit_finish": True, "author_action": "stop",
                                             "rationale": STOP_RATIONALE}, fx.ctx)
    if change == "tools":
        fx.trace["tool_calls"].append({"tool": "write_file", "args": {"path": "later.txt"}})
    elif change == "owner_directives":
        fx.ctx._owner_directives.append({"source": "owner_mailbox", "content": "any news?"})
    else:
        fx.trace["verification_events"] = [{"kind": "services_stopped",
                                            "services": [{"service_id": "preview", "lifecycle": "stopped"}]}]
    assert fx.run(STOP_TEXT) is False
    decision = fx.trace["acceptance_decision"]
    assert decision["reason"] == "author_stop" and decision["author_action"] == "stop"
    assert decision["author_disposition"]["rationale"] == STOP_RATIONALE
    assert fx.calls == ["initial answer"], "no second panel for a stop"


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


def test_a_stop_after_an_earlier_panel_keeps_its_finality_through_the_keep_round(tmp_path, monkeypatch):
    """(c) FAIL panel → action-only stop → keep: the stop honoured after an EARLIER
    panel used to lose its finality on the next delivery pass — the stale reviewed
    subject of that panel cleared the reviewed latch, the finish intent had been
    consumed, the keep/replace control re-armed, a SECOND panel ran and Cyber's
    advisory ``author_finish`` replaced the stop. One panel, final reason
    ``author_stop``, the agent's rationale on the record, and never Done."""
    from ouroboros.outcomes import derive_loop_outcome

    run = _run_stop_loop(tmp_path, monkeypatch, [
        "The export endpoint ships.",                       # reviewed: FAIL, capsule fed back
        _stop_tool_call(),                                  # action-only stop after the feedback
        json.dumps({"delivery_control": "keep"}),           # the host's control round: keep the stop
        STOP_TEXT, json.dumps({"delivery_control": "keep"}), STOP_TEXT,
        json.dumps({"delivery_control": "keep"}), STOP_TEXT,
    ], stop_services=False)
    assert run.panels == ["The export endpoint ships."], "an honoured stop must never buy a second panel"
    decision = run.trace["acceptance_decision"]
    assert decision["reason"] == "author_stop" and decision["author_action"] == "stop"
    assert decision["author_disposition"]["action"] == "stop"
    assert decision["author_disposition"]["rationale"] == STOP_RATIONALE
    assert run.result == STOP_TEXT
    axes = derive_loop_outcome(run.result, run.usage, run.trace)["outcome_axes"]
    assert axes["objective"]["status"] == "fail" and axes["objective"]["reason"] == "author_stop"


def _honoured_stop_after_a_panel(tmp_path, monkeypatch):
    """A FAIL panel, its feedback exposed, then an action-only stop the host honours."""
    from ouroboros.acceptance_settlement import expose_acceptance_feedback
    from ouroboros.loop_acceptance import merge_agent_acceptance_stance

    fx = _finality_pass(tmp_path, monkeypatch)
    assert fx.run("initial answer") is True
    expose_acceptance_feedback(fx.trace, fx.messages, "author-root")
    fx.trace["tool_calls"].append({"tool": "task_acceptance_review", "args": {}})
    merge_agent_acceptance_stance(fx.trace, {"explicit_finish": True, "author_action": "stop",
                                             "rationale": STOP_RATIONALE}, fx.ctx)
    assert fx.run(STOP_TEXT) is False
    assert fx.trace["acceptance_decision"]["reason"] == "author_stop"
    return fx


def test_an_honoured_stop_holds_until_the_authors_next_decision_replaces_it(tmp_path, monkeypatch):
    """(d) The earlier panel's reviewed subject no longer reopens review on a later
    delivery pass (the stop binds no subject), over the same or a changed text; the
    author's next explicit decision is still heard — a finish replaces the stop
    without another panel."""
    from ouroboros.loop_acceptance import merge_agent_acceptance_stance

    fx = _honoured_stop_after_a_panel(tmp_path, monkeypatch)
    for text in (STOP_TEXT, "A restated unfinished answer."):
        assert fx.run(text) is False
        assert fx.trace["acceptance_decision"]["reason"] == "author_stop"
    fx.trace["tool_calls"].append({"tool": "task_acceptance_review", "args": {}})
    merge_agent_acceptance_stance(fx.trace, {"disposition": "partial", "explicit_finish": True,
                                             "author_action": "finish",
                                             "rationale": "Fixed the empty-input case after all."}, fx.ctx)
    assert fx.run("revised answer") is False
    decision = fx.trace["acceptance_decision"]
    assert decision["reason"] == "author_finish" and decision["author_disposition"]["action"] == "finish"
    assert fx.calls == ["initial answer"], "neither the held stop nor the finish bought a panel"


def test_owner_input_after_an_honoured_stop_takes_the_ordinary_review_path(tmp_path, monkeypatch):
    """(e) New owner input supersedes the stop exactly as it supersedes any terminal
    acceptance: Main answers it and that answer is reviewed normally."""
    import ouroboros.loop as loop_mod

    fx = _honoured_stop_after_a_panel(tmp_path, monkeypatch)
    loop_mod._supersede_task_acceptance_for_owner_followup(fx.ctx, fx.trace)
    assert fx.trace["acceptance_decision"]["reason"] == "owner_followup"
    assert fx.run("The answer to the owner's follow-up.") is True
    assert fx.calls == ["initial answer", "The answer to the owner's follow-up."]
    assert fx.trace["acceptance_decision"]["reason"] != "author_stop"


def test_a_stop_recorded_under_exhausted_rounds_keeps_its_cause_and_its_rationale():
    """A2: when the review rounds ran out, ``_finish_advisory_author`` records the
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


@pytest.mark.parametrize("earlier_panel", [True, False])
def test_evidence_changing_after_an_honoured_stop_neither_reopens_nor_replaces_it(tmp_path, monkeypatch, earlier_panel):
    """(f) Evidence that changes only AFTER the stop was honoured (a service seen stopped
    at the post-acceptance evidence read) used to supersede the stop like a reviewed
    boundary: the reviewed latch reset, a panel was bought and its host exit replaced
    the stop. A stop binds no evidence: Main restates, no panel runs, and the row reads
    the stop's cause, rationale and red objective."""
    import ouroboros.loop as loop
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.project_dialogue import _completion_verdict

    real_project, late = loop._project_child_result_dispositions, []

    def project(limit_ctx, llm_trace):
        if (llm_trace.get("acceptance_decision") or {}).get("reason") == "author_stop" and not late:
            late.append(True)
            llm_trace.setdefault("verification_events", []).append({"kind": "services_stopped", "services": [
                {"service_id": "late", "name": "late", "lifecycle": "stopped"}]})
        return real_project(limit_ctx, llm_trace)

    monkeypatch.setattr(loop, "_project_child_result_dispositions", project)
    first = ["The export endpoint ships."] if earlier_panel else []
    keep = json.dumps({"delivery_control": "keep"})
    run = _run_stop_loop(tmp_path, monkeypatch, [*first, _stop_tool_call(),
                                                 *[step for _ in range(4) for step in (keep, STOP_TEXT)]],
                         stop_services=False)
    assert late and run.panels == first, "evidence after a stop must never buy a panel"
    decision = run.trace["acceptance_decision"]
    assert decision["reason"] == "author_stop" and decision["author_disposition"]["rationale"] == STOP_RATIONALE
    assert run.result == STOP_TEXT
    axes = derive_loop_outcome(run.result, run.usage, run.trace)["outcome_axes"]
    assert axes["objective"]["status"] == "fail" and axes["objective"]["reason"] == "author_stop"
    verdict = _completion_verdict({"status": "completed", "reason_code": "final_message", "outcome_axes": axes}, {})
    assert verdict.startswith("Ouroboros stopped with unfinished work") and STOP_RATIONALE.rstrip(".") in verdict


def test_owner_input_consumes_the_stop_so_a_later_exhausted_exit_is_not_read_as_one(tmp_path, monkeypatch):
    """(g) An explicit stop under exhausted rounds keeps its TRUE cause, its act and its
    rationale. New owner input supersedes it; the later host exit for the same exhausted
    rounds is the host's own, so it must not inherit the stop's act and re-read as the
    author stopping (formerly its row repeated the stale stop rationale). The historical
    stance stays on the record. The wallet is substituted with the panel: the substituted
    panel records no paid claim."""
    import ouroboros.loop as loop_mod
    import ouroboros.task_results as task_results
    from ouroboros.acceptance_settlement import expose_acceptance_feedback
    from ouroboros.loop_acceptance import merge_agent_acceptance_stance
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.project_dialogue import _completion_verdict
    from ouroboros.review_records import recorded_author_stop

    fx = _finality_pass(tmp_path, monkeypatch)
    assert fx.run("initial answer") is True
    expose_acceptance_feedback(fx.trace, fx.messages, "author-root")
    monkeypatch.setattr(task_results, "project_task_acceptance_review_capacity", lambda *_a, **_k: {
        "state": "unavailable", "reason": "review_cycles_exhausted", "claimed_cycles": 1, "cap_cycles": 1})

    def row():
        axes = derive_loop_outcome("answer", {}, fx.trace)["outcome_axes"]
        assert axes["objective"]["reason"] == "review_cycles_exhausted"
        return _completion_verdict({"status": "completed", "reason_code": "final_message", "outcome_axes": axes}, {})

    fx.trace["tool_calls"].append({"tool": "task_acceptance_review", "args": {}})
    merge_agent_acceptance_stance(fx.trace, {"explicit_finish": True, "author_action": "stop",
                                             "rationale": STOP_RATIONALE}, fx.ctx)
    assert fx.run(STOP_TEXT) is False
    stop = fx.trace["acceptance_decision"]
    assert (stop["reason"], stop["author_action"], stop["author_disposition"]["action"]) == (
        "review_cycles_exhausted", "stop", "stop")
    assert recorded_author_stop(stop) and STOP_RATIONALE.rstrip(".") in row()

    loop_mod._supersede_task_acceptance_for_owner_followup(fx.ctx, fx.trace)
    assert "author_action" not in fx.trace["acceptance_decision"]
    assert fx.run("The answer to the owner's follow-up.") is False
    later = fx.trace["acceptance_decision"]
    assert later["reason"] == "review_cycles_exhausted" and "author_action" not in later
    assert later["agent_rationale"] == STOP_RATIONALE  # history kept, never the act
    assert not recorded_author_stop(later) and STOP_RATIONALE.rstrip(".") not in row()
    assert fx.calls == ["initial answer"]


@pytest.mark.parametrize("change", ["material", "evidence"])
def test_a_local_preparation_stop_after_a_fail_panel_keeps_its_finality(tmp_path, monkeypatch, change):
    """(h) FAIL panel → the next host pass cannot assemble its evidence locally → the
    informed author stops → material (a new working-tree file) or evidence (a late
    ``services_stopped``) changes and Main restates its stop. The local-preparation stop
    kept the FAIL panel's reviewed subject, so ``preparation_delivery_choice`` refused
    the changed material, the stale subject cleared the reviewed latch and the host
    bought two more panels over a stop. Like the reviewer-bound stop it binds no
    subject: one panel, one failed assembly, the stop's local cause, act and rationale."""
    import ouroboros.loop as loop
    import ouroboros.loop_acceptance_review as review
    from ouroboros.outcomes import derive_loop_outcome

    real_build, builds = review._build_host_acceptance_evidence, []

    def build(ctx):
        builds.append(ctx.content)
        if len(builds) == 2:
            raise RuntimeError("local evidence assembly failed")
        return real_build(ctx)

    monkeypatch.setattr(review, "_build_host_acceptance_evidence", build)
    real_project, late = loop._project_child_result_dispositions, []

    def project(limit_ctx, llm_trace):
        if (llm_trace.get("acceptance_decision") or {}).get("reason") == "author_stop" and not late:
            late.append(True)
            if change == "material":
                (tmp_path / "repo" / "late.txt").write_text("changed after the stop\n", encoding="utf-8")
            else:
                llm_trace.setdefault("verification_events", []).append({"kind": "services_stopped", "services": [
                    {"service_id": "late", "name": "late", "lifecycle": "stopped"}]})
        return real_project(limit_ctx, llm_trace)

    monkeypatch.setattr(loop, "_project_child_result_dispositions", project)
    keep = json.dumps({"delivery_control": "keep"})
    run = _run_stop_loop(tmp_path, monkeypatch, [
        "The export endpoint ships.",                      # reviewed: FAIL, capsule fed back
        "The export endpoint ships, revised.",             # the host cannot assemble its evidence
        _stop_tool_call(),                                 # the informed author stops
        *[step for _ in range(3) for step in (STOP_TEXT, keep)],
    ], stop_services=False)
    assert late, "the change must land after the stop was honoured"
    assert run.panels == ["The export endpoint ships."], "a local-preparation stop must never buy a panel"
    assert len(builds) == 2, "the stopped material is never assembled again"
    assert run.result == STOP_TEXT
    decision = run.trace["acceptance_decision"]
    assert decision["origin"] == "local_acceptance_preparation"
    assert decision["reason"] == "author_stop" and decision["author_action"] == "stop"
    assert decision["author_disposition"]["rationale"] == STOP_RATIONALE
    axes = derive_loop_outcome(run.result, run.usage, run.trace)["outcome_axes"]
    assert axes["objective"]["status"] == "fail" and axes["objective"]["reason"] == "author_stop"


@pytest.mark.parametrize("reopen", ["owner_input", "author_finish"])
def test_owner_input_or_a_new_author_act_still_reopens_a_local_preparation_stop(tmp_path, monkeypatch, reopen):
    """(i) The subject-free local-preparation stop holds over changed text, yet owner
    input returns the answer to the ordinary host pass and the author's next explicit
    act is heard: a finish replaces the stop through the same incident, without a panel."""
    import ouroboros.loop as loop_mod
    import ouroboros.loop_acceptance_review as review
    from ouroboros.acceptance_settlement import expose_acceptance_feedback
    from ouroboros.loop_acceptance import merge_agent_acceptance_stance

    fx = _finality_pass(tmp_path, monkeypatch)
    assert fx.run("initial answer") is True
    assert fx.ctx._task_acceptance_reviewed_subject, "the FAIL panel bound its subject"

    def broken(_ctx):
        raise RuntimeError("local evidence assembly failed")

    monkeypatch.setattr(review, "_build_host_acceptance_evidence", broken)
    assert fx.run("revised answer") is True
    assert fx.trace["acceptance_decision"]["reason"] == "acceptance_preparation_failed"
    expose_acceptance_feedback(fx.trace, fx.messages, "author-root")
    fx.trace["tool_calls"].append({"tool": "task_acceptance_review", "args": {}})
    merge_agent_acceptance_stance(fx.trace, {"explicit_finish": True, "author_action": "stop",
                                             "rationale": STOP_RATIONALE}, fx.ctx)
    assert fx.run(STOP_TEXT) is False
    assert fx.trace["acceptance_decision"]["reason"] == "author_stop"
    assert fx.ctx._task_acceptance_reviewed_subject == ""
    assert fx.run("A restated unfinished answer.") is False
    assert fx.trace["acceptance_decision"]["reason"] == "author_stop"
    if reopen == "owner_input":
        loop_mod._supersede_task_acceptance_for_owner_followup(fx.ctx, fx.trace)
        assert fx.run("The answer to the owner's follow-up.") is False
        # The ordinary host pass decided again: the same unrepaired material is not
        # rebuilt, and the host's own honest ending replaced the consumed stop.
        decision = fx.trace["acceptance_decision"]
        assert decision["reason"] == "acceptance_preparation_failed" and "author_action" not in decision
        assert fx.trace["acceptance_preparation"]["attempts"] == 1
    else:
        fx.trace["tool_calls"].append({"tool": "task_acceptance_review", "args": {}})
        merge_agent_acceptance_stance(fx.trace, {"disposition": "partial", "explicit_finish": True,
                                                 "author_action": "finish",
                                                 "rationale": "Delivering the available result with its gap."}, fx.ctx)
        assert fx.ctx._task_acceptance_reviewed is False
        assert fx.run("A restated unfinished answer.") is False
        decision = fx.trace["acceptance_decision"]
        assert decision["reason"] == "author_finish" and decision["author_disposition"]["action"] == "finish"
    assert fx.calls == ["initial answer"]

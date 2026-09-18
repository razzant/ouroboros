"""Complete Main loop with the real reviewer coordinator and controlled transport."""
from __future__ import annotations

import copy
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros import loop, review_substrate
from ouroboros.loop_acceptance_review import acceptance_run_pending
from ouroboros.review_records import ReviewSlot
from ouroboros.tools.registry import ToolRegistry
from tests.test_loop_acceptance_gate import _order_acceptance_feedback, _seed_acceptance_root

ANSWER = "The complete report includes the requested budget."
STATUS = "How is it going?"


def call(name, arguments, identifier):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def test_queue_inspection_failure_does_not_invent_owner_generation_change():
    def inspect_unavailable(**_kwargs):
        raise OSError("queue unavailable")

    ctx = SimpleNamespace(
        _task_acceptance_fence_generation=1,
        _task_acceptance_fence_token="fence",
        _execution_trace={},
        inspect_acceptance_fence=inspect_unavailable,
    )
    assert loop._task_acceptance_owner_generation_changed(ctx) is False
    assert ctx._execution_trace["review_decision"]["admission_inspection"]["status"] == "unknown"


@pytest.fixture
def full_loop(tmp_path, monkeypatch):
    from ouroboros import review_custody
    from ouroboros.review_execution import ReviewAttemptResult

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "required")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "3")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "12")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_args: False)
    monkeypatch.setattr("ouroboros.tools.review_helpers.review_wave_budget_gate", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.review_evidence.acceptance_packet_budget_chars", lambda *_: 2_000_000)
    slot = ReviewSlot(slot_id="acceptance-one", model="fixture/reviewer", effort="high", timeout_sec=30)
    slots = [slot]
    monkeypatch.setattr(review_substrate, "triad_delivery_slots", lambda **_kw: slots)
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.repo_dir.mkdir()
    ctx = registry._ctx
    task_id = "async-loop-root"
    _seed_acceptance_root(ctx.drive_root, task_id, ctx)
    ctx.task_contract["expected_output"] = "A complete report including its budget."
    ctx.task_attempt = 1
    ctx.current_chat_id = 1
    ctx.is_direct_chat = False
    ctx.owner_message_admission_agent = SimpleNamespace(
        _owner_message_generation=0, _accepting_owner_messages=True,
        _busy=True, _current_task_id=task_id,
    )
    ctx.owner_message_admission_lock = threading.RLock()
    incoming, events = queue.Queue(), queue.Queue()
    fixture = SimpleNamespace(tools=registry, ctx=ctx, incoming=incoming, events=events,
                              model_inputs=[], review_requests=[], review_snapshots=[], review_sends=[],
                              entered=threading.Event(), release=threading.Event(), settled=threading.Event(),
                              waits=[], progress=[], model_step=0, condition=threading.Condition(),
                              settled_count=0, reviewer_verdict="PASS", slots=slots)
    original_settle = review_custody._settle_review_attempt
    def settle(*a, **kw):
        try:
            return original_settle(*a, **kw)
        finally:
            fixture.settled.set()
            with fixture.condition:
                fixture.settled_count += 1
                fixture.condition.notify_all()
    monkeypatch.setattr(review_custody, "_settle_review_attempt", settle)

    class HeldExecutor:
        def __init__(self, assignment):
            self.assignment = assignment
        def restore_custody(self, _state):
            return None
        def set_pending_invocation_checkpoint(self, _checkpoint):
            return None
        def prompt_payload(self):
            return {"messages": []}
        def prompt_chars(self):
            return 0
        def failure_custody(self):
            return {}
        def execute(self):
            from ouroboros.review_dispatch import invoke_review_paid_stamp
            invoke_review_paid_stamp(self.assignment.dispatch_stamp)
            request = self.assignment.request
            fixture.review_sends.append(self.assignment.call_id)
            fixture.review_requests.append(copy.deepcopy(request))
            fixture.review_snapshots.append(copy.deepcopy(ctx._execution_trace))
            fixture.entered.set()
            with fixture.condition:
                fixture.condition.notify_all()
            assert fixture.release.wait(10), "fixture did not release review"
            from ouroboros.review_evidence_refs import acceptance_evidence_ref_vocabulary
            vocabulary = acceptance_evidence_ref_vocabulary(request.evidence)
            reference = next(key for key, basis in vocabulary.items() if basis in {"tool_record", "packet_section"})
            response_text = json.dumps({
                "verdict": fixture.reviewer_verdict, "summary": "Independent review", "findings": [],
                "outcome_tier": "solved" if fixture.reviewer_verdict == "PASS" else "best_effort",
                "completion_coach": "Deliver the complete answer." if fixture.reviewer_verdict == "PASS" else "Add independent verification.",
                "criteria_used": [{"criterion": "full report", "status": "supported",
                                   "evidence_refs": [reference]}],
            })
            return ReviewAttemptResult(message={"content": response_text}, raw_text=response_text,
                                       usage={"prompt_tokens": 5, "completion_tokens": 3, "physical_attempt_state": "settled"})

    monkeypatch.setattr(review_substrate, "_review_route_executor", lambda assignment, **_kw: HeldExecutor(assignment))
    def park(_ctx, checkpoint):
        fixture.waits.append(copy.deepcopy(checkpoint))
        fixture.release.set()
        with fixture.condition:
            assert fixture.condition.wait_for(lambda: fixture.settled_count >= len(fixture.review_sends), timeout=10), "review did not settle"
    ctx.owner_wait_callback = park
    fixture.park = park
    fixture.run_args = dict(
        messages=[{"role": "system", "content": "Complete the owner task."},
                  {"role": "user", "content": "Prepare the complete report including its budget."}],
        tools=registry, llm=SimpleNamespace(default_model=lambda: "fixture/main"),
        drive_logs=ctx.drive_root / "logs", emit_progress=lambda text, **kw: fixture.progress.append(text),
        incoming_messages=incoming, task_id=task_id, drive_root=ctx.drive_root, event_queue=events,
    )
    def run():
        # The scripted Main replaces the transport, including its actual-context
        # observer. Observe only the request whose response is being returned.
        model = loop.call_llm_with_retry
        def observed(*args, **kwargs):
            result = model(*args, **kwargs)
            observer = kwargs.get("model_context_observer")
            if result[0] is not None and callable(observer):
                observer(args[1])
            return result
        with monkeypatch.context() as patcher:
            patcher.setattr(loop, "call_llm_with_retry", observed)
            return loop.run_llm_loop(**fixture.run_args)
    fixture.run = run
    yield fixture
    fixture.release.set()
    if fixture.entered.is_set():
        assert fixture.settled.wait(10)


def keep(f):
    observation = f.ctx._acceptance_observation
    assert observation["owner_source_sha256"] in str(f.model_inputs[-1])
    return {"content": json.dumps({"delivery_control": "keep", "acceptance_subject": {
        "owner_source_sha256": observation["owner_source_sha256"],
    }})}


def test_full_loop_explicit_batch_owner_status_and_free_collection(full_loop, monkeypatch):
    f = full_loop
    def main(_llm, messages, *_args, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "The report is ready; requesting its review.", "tool_calls": [
                call("task_acceptance_review", {"claim": ANSWER}, "nominate"),
                call("write_file", {"root": "task_drive", "path": "batch-proof.txt", "content": "last batch effect"}, "last-effect"),
            ]}, 0.0
        if f.model_step == 2:
            assert f.entered.wait(5), f.progress
            assert f.ctx._delivery_candidate.full_text == ANSWER
            f.incoming.put(STATUS)
            return {"content": "", "tool_calls": [call("read_file", {"root": "task_drive", "path": "batch-proof.txt"}, "read-proof")]}, 0.0
        if f.model_step == 3:
            assert STATUS in str(messages)
            assert not f.release.is_set()
            return {"content": "", "tool_calls": [call("send_user_message", {"text": "The report is ready; its review is still running."}, "status-answer")]}, 0.0
        assert f.model_step < 8, f.progress
        return keep(f), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == ANSWER, (result, f.progress, trace.get("review_decision"))
    assert len(f.review_sends) == 1
    from ouroboros.task_results import project_task_acceptance_review_capacity
    assert project_task_acceptance_review_capacity(f.ctx, task_id=f.ctx.task_id)["claimed_cycles"] == 1
    assert trace["acceptance_decision"]["status"] == "accepted"
    first = f.review_snapshots[0]
    assert [r["tool_call_id"] for r in first["tool_calls"]] == ["nominate", "last-effect"]
    assert first["tool_calls"][-1]["status"] == "ok"
    assert f.review_requests[0].subject == ANSWER
    assert "last batch effect" in json.dumps(f.review_requests[0].evidence)
    sends = list(f.events.queue)
    replies = [x for x in sends if x.get("type") == "send_message" and x.get("system_type") == "proactive_message"]
    assert len(replies) == 1 and "still running" in replies[0]["text"]
    assert all(x["subject"] == ANSWER for x in [r["request"] for r in trace["review_runs"] if r.get("authority") == "host_root"])
    assert f.waits and f.waits[0]["reason"] == "review" and not f.waits[0]["quiz_id"]
    assert f.ctx.owner_message_admission_agent._accepting_owner_messages is False


def test_new_criterion_same_answer_gets_one_new_panel_and_keeps_prior_request(full_loop, monkeypatch):
    f = full_loop
    criterion = "Show the budget as an explicit section."
    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "initial-review")]}, 0.0
        if f.model_step == 2:
            assert f.entered.wait(5)
            f.incoming.put(criterion)
            return {"content": "", "tool_calls": [call("send_user_message", {"text": "The full draft already includes the budget."}, "progress")]}, 0.0
        if f.model_step == 3:
            assert criterion in str(messages)
            observed = f.ctx._acceptance_observation
            return {"content": "", "tool_calls": [call("task_acceptance_review", {
                "claim": ANSWER, "acceptance_subject": {
                    "owner_source_sha256": observed["owner_source_sha256"],
                    "effective_criteria": "Complete report with the budget in its own explicit section.",
                },
            }, "new-subject-review")]}, 0.0
        assert f.model_step < 8, f.progress
        return keep(f), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == ANSWER
    assert len(f.review_sends) == len(set(f.review_sends)) == 2
    requests = f.review_requests
    assert requests[0].subject == requests[1].subject == ANSWER
    assert requests[0].retry_key != requests[1].retry_key
    assert criterion not in json.dumps(requests[0].evidence)
    assert criterion in json.dumps(requests[1].evidence)
    host = [r for r in trace["review_runs"] if r.get("authority") == "host_root"]
    assert len(host) == 2 and host[0]["superseded_by_revision"]
    assert host[0]["candidate_hash"] == host[1]["candidate_hash"]
    assert host[0]["subject_hash"] != host[1]["subject_hash"]
    from ouroboros.task_results import project_task_acceptance_review_capacity
    assert project_task_acceptance_review_capacity(f.ctx, task_id=f.ctx.task_id)["claimed_cycles"] == 2
    # The superseded panel was paid for: its verdicts must have been read, not stranded.
    assert not acceptance_run_pending(host[0])
    assert host[0]["actors"][0]["parsed"]["verdict"] == "PASS"


def test_reauthored_answer_collects_the_stranded_panel_before_paying_again(full_loop, monkeypatch):
    """A re-authored answer moves the paid identity, so the free-replay lookup no
    longer sees the running panel. Its verdicts were bought; the host collects
    them at $0 before assembling evidence for, or refusing, anything new."""
    f = full_loop
    reauthored = ANSWER + " Budget: $12."
    recorded = []

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "first-review")]}, 0.0
        if f.model_step == 2:
            assert f.entered.wait(5)
            # The paid panel settles while Main is still working; nothing has
            # read its verdicts yet and the settlement wake is the only signal.
            f.release.set()
            with f.condition:
                assert f.condition.wait_for(lambda: f.settled_count >= 1, timeout=10)
            return {"content": "", "tool_calls": [call("send_user_message", {"text": "Still writing the report."}, "status")]}, 0.0
        if f.model_step == 3:
            pending = [r for r in f.ctx._execution_trace["review_runs"] if r.get("authority") == "host_root"]
            recorded.append(pending[0]["actors"][0]["operation_id"])
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": reauthored}, "reauthored-review")]}, 0.0
        assert f.model_step < 8, f.progress
        return keep(f), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == reauthored
    # The re-authored subject still buys its own panel: no new refusal gate.
    assert len(f.review_sends) == len(set(f.review_sends)) == 2
    host = [r for r in trace["review_runs"] if r.get("authority") == "host_root"]
    assert len(host) == 2
    assert not acceptance_run_pending(host[0]), host[0]["actors"]
    # The EXACT recorded producer advanced, not a re-run.
    assert host[0]["actors"][0]["parsed"]["verdict"] == "PASS"
    assert host[0]["actors"][0]["operation_id"] == recorded[0]
    # The collected verdicts reached the next panel's dialogue history.
    history = f.review_requests[1].evidence["acceptance_dialogue_history"]
    assert history and history[0]["aggregate_signal"] == "PASS"
    # And the published projection, instead of a transport error on a row nobody read.
    from ouroboros.task_results import load_task_result, project_task_acceptance_review_capacity
    panels = {p["panel_id"]: p for p in
              load_task_result(f.ctx.drive_root, f.ctx.task_id)["review_projection"]["panels"]}
    collected = panels[host[0]["panel_id"]]
    assert collected["actors"][0]["transport_status"] == "success"
    assert collected["actors"][0]["parse_status"] == "valid"
    # Collection is free: exactly the two dispatched panels were ever claimed.
    assert project_task_acceptance_review_capacity(f.ctx, task_id=f.ctx.task_id)["claimed_cycles"] == 2


def test_reauthored_answer_on_a_one_cycle_install_is_refused_after_its_panel_was_collected(full_loop, monkeypatch):
    """The live incident shape (task 4525349b, OUROBOROS_REVIEW_MAX_CYCLES=1): the
    paid panel settles while Main is still working, Main re-authors, and the
    one-cycle cap refuses a second panel. The refusal is the product rule (owner
    14A) and stays; what the reconcile changes is that the paid verdicts are read
    BEFORE it — recorded as settled, present in the next evidence's dialogue
    history — and the decision carries no prose rationale claiming a quorum
    failure that never happened."""
    f = full_loop
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    reauthored = ANSWER + " Budget: $12."

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "first-review")]}, 0.0
        if f.model_step == 2:
            assert f.entered.wait(5)
            f.release.set()
            with f.condition:
                assert f.condition.wait_for(lambda: f.settled_count >= 1, timeout=10)
            return {"content": "", "tool_calls": [call("send_user_message", {"text": "Still writing the report."}, "status")]}, 0.0
        if f.model_step == 3:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": reauthored}, "reauthored-review")]}, 0.0
        assert f.model_step < 8, f.progress
        return keep(f), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == reauthored
    # One paid dispatch only: the cap refused the re-authored subject.
    assert len(f.review_sends) == 1
    host = [r for r in trace["review_runs"] if r.get("authority") == "host_root"]
    assert len(host) == 1
    # The paid panel was read at $0 before the refusal, not left as pending stubs.
    assert not acceptance_run_pending(host[0]), host[0]["actors"]
    assert host[0]["actors"][0]["parsed"]["verdict"] == "PASS"
    assert trace["review_decision"]["dispatch_refusal"]["reason"] == "review_cycles_exhausted"
    decision = trace.get("acceptance_decision") or {}
    assert decision.get("reason") == "review_cycles_exhausted"
    assert "no new panel" in decision["rationale"]
    from ouroboros.task_results import project_task_acceptance_review_capacity
    assert project_task_acceptance_review_capacity(f.ctx, task_id=f.ctx.task_id)["claimed_cycles"] == 1


def _terminal_record(trace):
    """The host's own fold of the trace, as the terminal row and the card read it."""
    from ouroboros import outcomes

    review = outcomes._review_axis(trace)
    return {"status": "completed", "reason_code": "final_message",
            "outcome_axes": {"execution": {"status": "ok"}, "review": review,
                             "objective": outcomes._objective_axis(review)}}


def test_a_rewritten_answer_delivers_under_the_running_panel_instead_of_buying_one(full_loop, monkeypatch):
    """The live incident shape (task 4525349b, cap 1) under owner D4=A and fork 1=B:
    Main nominates, the reviewers are still reading when it rewrites the answer
    through the delivery control (round 7 of the incident). The rewrite is a
    DELIVERY, not a nomination: it buys nothing and is refused nothing; the host
    waits for the panel it already paid for, and when that panel PASSES the earlier
    revision the task is accepted on the reviewers' word and the row says so."""
    from ouroboros.project_dialogue import _completion_verdict, outcome_phase

    f = full_loop
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    reauthored = ANSWER + " Budget: $12."

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "first-review")]}, 0.0
        if f.model_step == 2:
            assert f.entered.wait(5) and not f.release.is_set(), "the panel must still be running"
            observation = f.ctx._acceptance_observation
            return {"content": json.dumps({"delivery_control": "replace", "full_answer": reauthored,
                                           "acceptance_subject": {"owner_source_sha256": observation["owner_source_sha256"]}})}, 0.0
        assert f.model_step < 6, f.progress
        return keep(f), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == reauthored
    # The rewrite bought nothing and was refused nothing: one paid panel, no synthetic refusal run.
    assert len(f.review_sends) == 1
    host = [r for r in trace["review_runs"] if r.get("authority") == "host_root"]
    assert len(host) == 1 and not acceptance_run_pending(host[0]), host
    assert host[0]["actors"][0]["parsed"]["verdict"] == "PASS"
    assert not any("review_cycles_exhausted" in str(reason) for reason in host[0].get("degraded_reasons") or [])
    # The host waited for the panel it had paid for (blocking enforcement in this fixture).
    assert f.waits and f.waits[0]["reason"] == "review"
    assert any("holding the answer for its verdict" in line for line in f.progress), f.progress
    decision = trace["acceptance_decision"]
    assert decision["status"] == "accepted" and decision["reason"] == "previous_revision_accepted"
    assert decision["reviewer_signal"] == "PASS" and "rationale" not in decision
    from ouroboros.task_results import project_task_acceptance_review_capacity
    assert project_task_acceptance_review_capacity(f.ctx, task_id=f.ctx.task_id)["claimed_cycles"] == 1
    record = _terminal_record(trace)
    assert outcome_phase(record, {}) == "done", record["outcome_axes"]
    assert _completion_verdict(record, {}) == (
        "The reviewers approved an earlier version of this answer; the current version was not re-reviewed."
    )


def test_a_rejected_earlier_revision_is_not_a_verdict_on_the_rewrite(full_loop, monkeypatch):
    """Fork 1: a FAIL on the earlier revision must not paint the rewritten,
    unreviewed answer a critic verdict. At cap 1, the task instead ends blocked
    by the typed capacity refusal. The real earlier FAIL is retained and no
    actorless replacement panel is invented for the corrected answer."""
    from ouroboros.project_dialogue import outcome_phase

    f = full_loop
    f.reviewer_verdict = "FAIL"
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    reauthored = ANSWER + " Budget: $12."

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "first-review")]}, 0.0
        if f.model_step == 2:
            assert f.entered.wait(5) and not f.release.is_set()
            observation = f.ctx._acceptance_observation
            return {"content": json.dumps({"delivery_control": "replace", "full_answer": reauthored,
                                           "acceptance_subject": {"owner_source_sha256": observation["owner_source_sha256"]}})}, 0.0
        assert f.model_step < 6, f.progress
        return keep(f), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == reauthored and len(f.review_sends) == 1
    host = [r for r in trace["review_runs"] if r.get("authority") == "host_root"]
    assert [r.get("aggregate_signal") for r in host] == ["FAIL"], host
    assert host[0]["superseded_by_revision"]
    assert trace["acceptance_decision"]["reason"] == "review_cycles_exhausted"
    record = _terminal_record(trace)
    assert outcome_phase(record, {}) == "error", record["outcome_axes"]
    assert record["outcome_axes"]["objective"]["reason"] == "review_cycles_exhausted"


def _advisory(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")


def test_a_conscious_finish_releases_the_answer_while_the_panel_runs(full_loop, monkeypatch):
    """Owner D4=A point 3: under advisory enforcement Main chooses explicitly.
    ``"pending_review":"finish"`` on the delivery control delivers now, without a
    park; the verdict reaches Main as advice when it settles."""
    f = full_loop
    _advisory(monkeypatch)
    f.ctx.owner_wait_callback = lambda *_a, **_kw: pytest.fail("a conscious finish must not park")

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "nominate")]}, 0.0
        assert f.model_step == 2 and f.entered.wait(5) and not f.release.is_set()
        control = json.loads(keep(f)["content"])
        assert '"pending_review":"finish"' in str(messages), "the choice is offered while the panel runs"
        return {"content": json.dumps({**control, "pending_review": "finish"})}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == ANSWER and len(f.review_sends) == 1 and f.waits == []
    assert trace["acceptance_decision"]["reason"] == "author_finish"
    assert trace["review_decision"]["review_pending"] is True
    from ouroboros.task_results import project_task_acceptance_review_capacity
    assert project_task_acceptance_review_capacity(f.ctx, task_id=f.ctx.task_id)["claimed_cycles"] == 1


def test_a_panel_that_settles_after_the_loop_exited_is_attached_through_the_remembered_trace(full_loop, monkeypatch):
    """Fable review round 2 (CRITICAL): the loop exit restores the context's
    ``_execution_trace`` to its pre-loop value, so a wave settling after the
    turn ended found no trace and the late supplement never fired in production.
    The pending panel now remembers its trace; the settlement thread reads it
    back after the loop is gone and announces once in the task's room."""
    from ouroboros.task_results import load_task_result, write_task_result

    f = full_loop
    _advisory(monkeypatch)
    f.ctx.owner_wait_callback = lambda *_a, **_kw: pytest.fail("a conscious finish must not park")

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "nominate")]}, 0.0
        assert f.model_step == 2 and f.entered.wait(5) and not f.release.is_set()
        control = json.loads(keep(f)["content"])
        return {"content": json.dumps({**control, "pending_review": "finish"})}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == ANSWER and trace["review_decision"]["review_pending"] is True
    assert getattr(f.ctx, "_execution_trace", None) is None, "the loop exit detached the live trace"
    # The pipeline seals the task before the straggler answers.
    write_task_result(f.ctx.drive_root, f.ctx.task_id, "completed", chat_id=1, result=ANSWER)
    f.release.set()
    with f.condition:
        assert f.condition.wait_for(lambda: f.settled_count >= 1, timeout=10)
    events = list(f.events.queue)
    rows = [e for e in events if e.get("system_type") == "acceptance_late_settlement"]
    assert len(rows) == 1, [e.get("type") for e in events]
    assert rows[0]["task_id"] == f.ctx.task_id and rows[0]["chat_id"] == 1
    assert rows[0]["text"].startswith("Reviewers later passed this answer. They reviewed the answer that was delivered.")
    assert "- acceptance-one: PASS" in rows[0]["text"]
    stored = load_task_result(f.ctx.drive_root, f.ctx.task_id)
    assert stored["status"] == "completed"
    panel = stored["review_projection"]["panels"][-1]
    actor = panel["actors"][0]
    assert actor["transport_status"] == "success" and actor["parse_status"] == "valid"
    # The panel reviewed the bytes that shipped, so its settlement says so.
    assert panel["late_settlement"] == {"note": rows[0]["text"], "reviewed_revision": "delivered",
                                        "settled_after_terminal": True}
    assert rows[0]["progress_meta"]["card_row"] == "reviews"
    assert len(f.review_sends) == 1, "the supplement bought nothing"
    assert not getattr(f.ctx, "_acceptance_settlement_traces", {}), "a settled wave releases its remembered trace"


def test_a_rejected_earlier_revision_buys_a_panel_on_the_rewrite_when_the_cap_allows(full_loop, monkeypatch):
    """The other half of fork 1: after a FAIL on the earlier revision the ordinary
    path decides, and with review cycles left it buys a real panel on the
    rewritten bytes — the verdict that then accepts the task is about the
    delivered text, not the old one."""
    f = full_loop
    f.reviewer_verdict = "FAIL"
    reauthored = ANSWER + " Budget: $12."

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "first-review")]}, 0.0
        if f.model_step == 2:
            assert f.entered.wait(5) and not f.release.is_set()
            observation = f.ctx._acceptance_observation
            return {"content": json.dumps({"delivery_control": "replace", "full_answer": reauthored,
                                           "acceptance_subject": {"owner_source_sha256": observation["owner_source_sha256"]}})}, 0.0
        # The park released panel 1 (FAIL) before this round; the rewrite addresses the notes.
        f.reviewer_verdict = "PASS"
        assert f.model_step < 8, f.progress
        return keep(f), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == reauthored
    assert len(f.review_sends) == 2, "the rewrite got its own panel"
    host = [r for r in trace["review_runs"] if r.get("authority") == "host_root"]
    assert [r.get("aggregate_signal") for r in host] == ["FAIL", "PASS"], host
    assert host[0]["superseded_by_revision"] and not host[1].get("superseded_by_revision")
    assert f.review_requests[1].subject == reauthored
    assert trace["acceptance_decision"]["status"] == "accepted"
    assert trace["acceptance_decision"]["reason"] in {"clean_pass", "clean_pass_obligations_closed"}


@pytest.mark.parametrize("order", ["ready", "pending"])
@pytest.mark.parametrize("enforcement", ["advisory", "blocking"])
def test_an_older_fail_never_outvotes_the_pass_that_accepted_the_task(full_loop, monkeypatch, order, enforcement):
    """Astra review round 4: panel A rejects the first draft, Main re-nominates and
    panel B passes the second, Main rewrites once more under B. Both runs end up
    superseded; the decision names B. The review axis must read B alone — the
    old FAIL is audit evidence, not a vote against the accepted answer."""
    from ouroboros.project_dialogue import outcome_phase

    f = full_loop
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    f.reviewer_verdict = "FAIL"
    second = ANSWER + " Budget: $12."
    third = second + " Timeline: two weeks."

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "first-review")]}, 0.0
        if f.model_step == 2:
            assert f.entered.wait(5)
            f.release.set()
            with f.condition:
                assert f.condition.wait_for(lambda: f.settled_count >= 1, timeout=10)
            f.reviewer_verdict = "PASS"
            _order_acceptance_feedback(f, monkeypatch, second, order)
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": second}, "second-review")]}, 0.0
        if f.model_step == 3:
            # This scenario rewrites under B's settled PASS, not while B runs.
            with f.condition:
                assert f.condition.wait_for(lambda: f.settled_count >= 2, timeout=10)
            observation = f.ctx._acceptance_observation
            return {"content": json.dumps({"delivery_control": "replace", "full_answer": third,
                                           "acceptance_subject": {"owner_source_sha256": observation["owner_source_sha256"]}})}, 0.0
        assert f.model_step < 8, f.progress
        return keep(f), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == third and len(f.review_sends) == 2
    host = [r for r in trace["review_runs"] if r.get("authority") == "host_root"]
    assert [r.get("aggregate_signal") for r in host] == ["FAIL", "PASS"], host
    assert all(r.get("superseded_by_revision") for r in host)
    decision = trace["acceptance_decision"]
    assert decision["status"] == "accepted" and decision["reason"] == "previous_revision_accepted"
    assert decision["reviewed_panel_id"] == host[1]["panel_id"]
    record = _terminal_record(trace)
    assert record["outcome_axes"]["review"]["aggregate_signals"] == ["PASS"], record["outcome_axes"]["review"]
    assert outcome_phase(record, {}) == "done", record["outcome_axes"]
    # The verification ledger agrees: the older FAIL is superseded evidence, not a live failure.
    from ouroboros._outcome_receipts import review_run_ledger_status, select_current_review_runs
    selection = select_current_review_runs(trace["review_runs"], delivery_candidate=trace.get("delivery_candidate"),
                                           review_decision=trace.get("review_decision"))
    assert review_run_ledger_status(host[0], selection) == ("superseded", True)


def test_an_owner_followup_acknowledged_through_the_control_sets_the_panel_aside(full_loop, monkeypatch):
    """Astra review round 5: the owner changes the requirements while the panel
    runs; Main reads the message, acknowledges its source on the delivery control
    and rewrites. The rewrite is NOT a delivery under the old panel (it judged the
    old premises): the ordinary path buys a panel on the new answer and the old
    PASS never accepts it."""
    f = full_loop
    followup = "Also add a timeline section to the report."
    rewritten = ANSWER + " Timeline: two weeks."

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "first-review")]}, 0.0
        if f.model_step == 2:
            assert f.entered.wait(5) and not f.release.is_set()
            f.incoming.put(followup)
            return {"content": "", "tool_calls": [call("send_user_message", {"text": "Adding the timeline."}, "ack")]}, 0.0
        if f.model_step == 3:
            assert followup in str(messages), "the owner follow-up reached the model"
            observation = f.ctx._acceptance_observation
            return {"content": json.dumps({"delivery_control": "replace", "full_answer": rewritten,
                                           "acceptance_subject": {"owner_source_sha256": observation["owner_source_sha256"]}})}, 0.0
        assert f.model_step < 8, f.progress
        return keep(f), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == rewritten
    assert len(f.review_sends) == 2, ("the rewrite for the new premises got its own panel", f.progress)
    assert f.review_requests[1].subject == rewritten
    host = [r for r in trace["review_runs"] if r.get("authority") == "host_root"]
    assert host[0]["superseded_by_revision"] and host[0]["owner_source_sha256"] != host[1]["owner_source_sha256"]
    assert trace["acceptance_decision"]["reason"] != "previous_revision_accepted"
    assert trace["acceptance_decision"]["status"] == "accepted"


@pytest.mark.parametrize("install", ["advisory_default", "blocking_finish", "cyber_pro"])
def test_waiting_is_the_default_and_blocking_enforcement_never_offers_the_choice(full_loop, monkeypatch, install):
    """Waiting needs no key; blocking enforcement waits whatever the model says and
    is never offered the choice; Cyber Pro keeps its own rule (Main's final response
    is its decision) and is not offered the choice either."""
    f = full_loop
    if install == "advisory_default":
        _advisory(monkeypatch)
    if install == "cyber_pro":
        _advisory(monkeypatch)
        monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
        f.ctx.owner_wait_callback = lambda *_a, **_kw: pytest.fail("Cyber Pro never parks on a review")

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "nominate")]}, 0.0
        if f.model_step == 2:
            assert f.entered.wait(5)
            control = json.loads(keep(f)["content"])
            if install == "blocking_finish":
                control["pending_review"] = "finish"
            return {"content": json.dumps(control)}, 0.0
        assert f.model_step < 6, f.progress
        return keep(f), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == ANSWER and len(f.review_sends) == 1
    offered = any('"pending_review":"finish"' in str(inputs) for inputs in f.model_inputs)
    if install == "advisory_default":
        assert offered and f.waits and f.waits[0]["reason"] == "review"
        assert trace["acceptance_decision"]["status"] == "accepted"
    elif install == "blocking_finish":
        assert not offered and f.waits and f.waits[0]["reason"] == "review"
        assert trace["acceptance_decision"]["status"] == "accepted"
    else:
        assert not offered and f.waits == []
        assert trace["acceptance_decision"]["reason"] == "author_finish"


@pytest.mark.parametrize("pending_first", [False, True])
@pytest.mark.parametrize("changed_requirement", [False, True])
def test_explicit_renomination_replaces_the_held_answer_without_control_repair(
    full_loop, monkeypatch, pending_first, changed_requirement,
):
    f = full_loop
    revised = "The revised complete report includes the corrected budget of 200."
    criterion = "Use the corrected budget of 200 in the report."
    if not pending_first:
        f.release.set()
        f.ctx.owner_wait_callback = None
    original_executor = review_substrate._review_route_executor
    notified = False

    def executor(assignment, **kw):
        nonlocal notified
        if changed_requirement and not notified:
            notified = True
            f.incoming.put(criterion)
        return original_executor(assignment, **kw)

    monkeypatch.setattr(review_substrate, "_review_route_executor", executor)

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step <= 2:
            args = {"claim": ANSWER if f.model_step == 1 else revised}
            if f.model_step == 2:
                assert f.entered.wait(5)
                if changed_requirement:
                    assert criterion in str(messages)
                    args["acceptance_subject"] = {
                        "owner_source_sha256": f.ctx._acceptance_observation["owner_source_sha256"],
                        "effective_criteria": "Complete report with the corrected budget of 200.",
                    }
            return {"content": "", "tool_calls": [call("task_acceptance_review", args, f"nominate-{f.model_step}")]}, 0.0
        assert f.model_step < 8
        return keep(f), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == revised
    assert [request.subject for request in f.review_requests] == [ANSWER, revised]
    assert len(f.review_sends) == len(set(f.review_sends)) == 2
    host = [run for run in trace["review_runs"] if run.get("authority") == "host_root"]
    assert len(host) == 2 and host[0]["superseded_by_revision"]
    assert host[0]["request"]["subject"] == ANSWER and host[1]["request"]["subject"] == revised
    assert "DELIVERY_CONTROL_REPAIR" not in str(f.model_inputs)
    assert trace["acceptance_decision"]["status"] == "accepted"


def test_explicit_ready_panel_does_not_seal_before_main_finishes(full_loop, monkeypatch):
    f = full_loop
    f.release.set()
    # This supported standalone variant waits synchronously, guaranteeing the
    # real panel has settled during explicit nomination rather than after it.
    f.ctx.owner_wait_callback = None
    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "ready-review")]}, 0.0
        assert f.model_step == 2, f.progress
        assert f.settled.is_set()
        assert f.ctx.owner_message_admission_agent._accepting_owner_messages is True
        assert not getattr(f.ctx, "_task_acceptance_sealed_fence_token", None)
        assert f.ctx._delivery_candidate.full_text == ANSWER
        return keep(f), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == ANSWER and len(f.review_sends) == 1
    assert f.ctx.owner_message_admission_agent._accepting_owner_messages is False
    assert trace["acceptance_decision"]["status"] == "accepted", (trace["acceptance_decision"], [(r.get("aggregate_signal"), r.get("actors")) for r in trace["review_runs"]])


def test_final_seal_rechecks_input_arriving_after_early_review(full_loop, monkeypatch):
    f = full_loop
    f.release.set()
    f.ctx.owner_wait_callback = None
    begins, ends = [], []
    def begin(**_kw):
        token = f"fence-{len(begins) + 1}"
        begins.append(token)
        if len(begins) == 2:
            # Final delivery has drained the mailbox, but a previously admitted
            # message arrives before the last seal. It still belongs to Main.
            f.incoming.put("One last status question before delivery.")
        return {"token": token, "owner_message_generation": 0}
    def end(**kw):
        ends.append(dict(kw))
        return {"ok": True, "status": "sealed" if kw["outcome"] == "terminal" else "released"}
    f.ctx.begin_acceptance_fence, f.ctx.end_acceptance_fence = begin, end
    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "early")]}, 0.0
        assert f.model_step <= 3, f.progress
        if f.model_step == 3:
            assert "One last status question" in str(messages)
            assert ends[-1]["outcome"] == "revision"
            assert f.ctx.owner_message_admission_agent._accepting_owner_messages
        return keep(f), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == ANSWER and f.model_step == 3
    assert len(f.review_sends) == 1
    assert [row["outcome"] for row in ends] == ["revision", "revision", "terminal"]
    assert trace["acceptance_decision"]["status"] == "accepted"


def test_cold_loop_resume_collects_saved_roster_and_request_once(full_loop, monkeypatch):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.owner_wait import set_owner_wait
    f = full_loop
    f.slots.append(ReviewSlot("acceptance-two", "fixture/second-reviewer", effort="high", timeout_sec=30))
    class PlannedPause(BaseException):
        pass
    def pause(ctx, checkpoint):
        f.waits.append(copy.deepcopy(checkpoint))
        set_owner_wait(ctx.budget_drive_root or ctx.drive_root, ctx.task_id, {**checkpoint, "state": "waiting"})
        raise PlannedPause()
    f.ctx.owner_wait_callback = pause
    def first_main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "nominate")]}, 0.0
        assert f.model_step == 2
        return keep(f), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", first_main)
    with pytest.raises(PlannedPause):
        f.run()
    checkpoint = f.waits[-1]
    saved = json.loads(read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, checkpoint["source_ref"]))
    saved_run = saved["trace"]["review_runs"][-1]
    assert saved_run["request"]["subject"] == ANSWER
    assert saved_run["slot_roster"][0]["model"] == "fixture/reviewer"
    assert checkpoint["reason"] == "review" and not checkpoint["quiz_id"]
    f.release.set()
    with f.condition:
        assert f.condition.wait_for(lambda: f.settled_count == 2, timeout=10)
    old = f.ctx
    new_tools = ToolRegistry(repo_dir=old.repo_dir, drive_root=old.drive_root)
    new = new_tools._ctx
    for key in ("task_id", "task_attempt", "task_metadata", "task_contract", "budget_drive_root", "current_chat_id"):
        setattr(new, key, copy.deepcopy(getattr(old, key)))
    new.owner_message_admission_agent = SimpleNamespace(_owner_message_generation=0, _accepting_owner_messages=True,
                                                       _busy=True, _current_task_id=old.task_id)
    new.owner_message_admission_lock = threading.RLock()
    # Cognitive route rebuilding is independent of this operation-custody test.
    # The fixture has no assembled production ContextCore or live model catalog.
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *_a, **_kw: (None, "max"))
    new.context_fit_plan = None
    new.owner_wait_resume = {**checkpoint, "restart_transaction_id": "fixture-planned-restart"}
    def resumed(_ctx, handoff):
        assert handoff["wait_id"] == checkpoint["wait_id"]
        assert _ctx._delivery_candidate.full_text == ANSWER
    new.owner_wait_callback = resumed
    f.ctx, f.tools = new, new_tools
    f.run_args["tools"] = new_tools
    f.run_args["messages"] = []
    def resumed_main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        assert "planned restart" in str(messages)
        return keep(f), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", resumed_main)
    result, _usage, trace = f.run()
    assert result == ANSWER and len(f.review_sends) == 2
    run = trace["review_runs"][-1]
    assert run["request"] == saved_run["request"]
    assert run["slot_roster"] == saved_run["slot_roster"]
    assert len(run["slot_roster"]) == 2
    assert [r["operation_id"] for r in run["actors"]] == [r["operation_id"] for r in saved_run["actors"]]
    assert trace["acceptance_decision"]["status"] == "accepted"


def test_automatic_completion_uses_the_same_retained_candidate_and_free_collect(full_loop, monkeypatch):
    f = full_loop
    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": ANSWER}, 0.0
        assert f.model_step == 2, f.progress
        return keep(f), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == ANSWER and len(f.review_sends) == 1
    assert trace["acceptance_decision"]["status"] == "accepted"
    assert f.waits and f.waits[0]["reason"] == "review"


@pytest.mark.parametrize("failure", ["pending", "fail", "late_fail", "unavailable", "evidence_unavailable"])
def test_cyber_final_response_never_waits_for_or_obeys_critic_veto(full_loop, monkeypatch, failure):
    f = full_loop
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    if failure in {"fail", "late_fail"}:
        f.reviewer_verdict = "FAIL"
        if failure == "fail":
            f.release.set()
    if failure == "unavailable":
        monkeypatch.setattr(review_substrate, "triad_delivery_slots", lambda **_kw: [])
    if failure == "evidence_unavailable":
        def evidence_unavailable(*_a, **_kw):
            raise OSError("fixture evidence storage unavailable")
        monkeypatch.setattr("ouroboros.loop_acceptance_review._build_host_acceptance_evidence", evidence_unavailable)
    def forbidden_wait(*_a, **_kw):
        pytest.fail("Cyber final-response decision was parked by a review")
    f.ctx.owner_wait_callback = forbidden_wait
    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if failure in {"fail", "late_fail"} and f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "explicit-critic")]}, 0.0
        if failure in {"fail", "late_fail"}:
            if failure == "late_fail" and f.model_step == 2:
                assert "- acceptance-one: FAIL" not in str(messages)
                f.release.set()  # Settle after this request's ingress drain.
            with f.condition:
                assert f.condition.wait_for(lambda: f.settled_count == 1, timeout=10)
            assert f.model_step == 2
            return keep(f), 0.0
        assert f.model_step == 1
        return {"content": ANSWER}, 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == ANSWER
    assert trace["acceptance_decision"]["status"] == "finalized_unaccepted"
    assert trace["acceptance_decision"]["reason"] == "author_finish"
    assert trace["acceptance_decision"]["author_disposition"]["source"] == "author_final_response"
    assert not f.waits
    if failure == "pending":
        assert not f.release.is_set()
        assert trace["review_runs"][-1]["actors"][0]["operation_state"] in {"pending_dispatch", "in_flight"}
        assert trace["acceptance_decision"]["review_pending"]
    elif failure in {"fail", "late_fail"}:
        assert trace["review_runs"][-1]["aggregate_signal"] == "FAIL"
        assert trace["review_runs"][-1]["actors"][0]["parsed"]["verdict"] == "FAIL"
        assert len(f.review_sends) == 1
        if failure == "late_fail":
            # A late critic wake is not new owner input and cannot demand
            # another author round after Main has chosen to finish.
            assert f.model_step == 2
            assert not any("owner follow-up arrived" in text for text in f.progress)
    else:
        assert trace["review_runs"][-1]["aggregate_signal"] == "DEGRADED"
        assert not f.review_sends
        if failure == "evidence_unavailable":
            assert "evidence storage unavailable" in str(trace["review_runs"][-1]["degraded_reasons"])
            assert "binding_hash" not in trace["review_runs"][-1]
            assert trace["acceptance_decision"]["author_disposition"]["subject_hash"] == trace["delivery_candidate"]["subject_sha256"]


@pytest.mark.parametrize("failure", ["begin", "end", "inspect"])
def test_cyber_admission_unavailable_is_disclosed_without_review_veto(full_loop, monkeypatch, failure):
    f = full_loop
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    f.ctx.begin_acceptance_fence = lambda **_kw: None if failure == "begin" else {"token": "unreleased-fence", "owner_message_generation": 0}
    f.ctx.end_acceptance_fence = lambda **_kw: {"ok": False, "error": "fixture release unavailable"}
    if failure == "inspect":
        def inspect_unavailable(**_kw):
            raise OSError("fixture queue inspection unavailable")
        f.ctx.inspect_acceptance_fence = inspect_unavailable
    monkeypatch.setattr(loop, "_task_acceptance_subtree_snapshot", lambda *_a: (False, [{"task_id": "child", "status": "running"}]))
    f.ctx.owner_wait_callback = lambda *_a: pytest.fail("review admission withheld Cyber final")
    def main(*_a, **_kw):
        f.model_step += 1
        assert f.model_step == 1, "Unknown admission inspection was treated as a new owner request"
        return {"content": ANSWER}, 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == ANSWER
    assert trace["review_decision"]["admission_fence_available"] is (failure != "begin")
    assert trace["review_decision"]["admission_released"] is (failure == "begin")
    assert trace["review_decision"]["subtree_quiescent"] is False
    assert trace["acceptance_decision"]["status"] == "finalized_unaccepted"
    if failure == "inspect":
        assert trace["review_decision"]["admission_inspection"] == {
            "status": "unknown", "reason": "queue_inspection_failed", "error_type": "OSError",
        }


def test_cyber_unread_owner_message_still_reaches_same_main(full_loop, monkeypatch):
    f = full_loop
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    f.ctx.owner_wait_callback = lambda *_a: None  # unread inbox wakes rather than waits on a critic
    original = review_substrate._review_route_executor
    injected = False
    def executor(assignment, **kw):
        nonlocal injected
        result = original(assignment, **kw)
        if not injected:
            injected = True
            f.incoming.put(STATUS)
        return result
    monkeypatch.setattr(review_substrate, "_review_route_executor", executor)
    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            return {"content": ANSWER}, 0.0
        assert f.model_step == 2 and STATUS in str(messages)
        assert f.ctx._acceptance_ack_source_sha256 != f.ctx._acceptance_observation["owner_source_sha256"]
        return keep(f), 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == ANSWER and f.model_step == 2
    assert len(f.review_sends) == 1
    assert trace["acceptance_decision"]["status"] == "finalized_unaccepted"

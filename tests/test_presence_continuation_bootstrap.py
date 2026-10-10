"""#1536 production agent bootstrap coupled to the real review-yield lifecycle.

Only inference/reviewer transport and unrelated installation/post-task work are
scripted. Settings reload, context assembly, callback propagation, the loop,
acceptance policy, review wallet, round limits and terminal writes stay real.
The helper is also usable by an already-isolated Host subprocess.
"""
from __future__ import annotations

import contextvars
import copy
import json
from dataclasses import replace
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import agent as agent_module, config, loop, review_substrate
from ouroboros.presence_authority import PresenceToolGrant, presence_ceiling_payload
from ouroboros.presence_runner import (
    PresenceTurnExecutions, PresenceTurnGate, presence_event_identity,
    presence_turn_task_id, run_presence_turn,
)
from ouroboros.review_records import ReviewSlot
from ouroboros.task_results import load_task_result
from tests.test_presence_runner import _admission, _event

ANSWER = "The status report is complete: all three checks passed."
NEW_WORDS = "Please also mention the backup window."


def tool_call(name, arguments, identifier):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


def finish(identifier, **arguments):
    return {"content": None, "tool_calls": [tool_call(
        "presence_finish", {"outcome": "message", **arguments}, identifier)]}


def nominate(message=ANSWER):
    # Presence is a direct turn, so even Required needs actual eligibility.
    # This is the model's supported explicit review request, not a fabricated
    # expected_output written into the host's task contract.
    return {"content": None, "tool_calls": [
        tool_call("task_acceptance_review", {"claim": message,
                  "goal": "Prepare the status report"}, "review-request"),
        tool_call("presence_finish", {"outcome": "message", "message": message}, "nominate"),
    ]}


def admission():
    admitted = _admission()
    ceiling = replace(admitted.capability_ceiling, tool_grants=(
        *admitted.capability_ceiling.tool_grants, PresenceToolGrant("task_acceptance_review"),
        PresenceToolGrant("get_task_result")))
    ceiling = replace(ceiling, digest=presence_ceiling_payload(ceiling)["digest"])
    return replace(admitted, capability_ceiling=ceiling)


def event(number=42, text="Please prepare the status report"):
    return replace(_event(), source_event_id=f"telegram:bot-1:{number}",
                   message={"message_id": str(number)}, text=text, continuation_version=1)


def wait_for(predicate, timeout=20, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def install_bootstrap_harness(monkeypatch, data, *, repo=None, review_mode="required", enforcement="blocking"):
    """Install deterministic transports after the caller isolates DATA/HOME.

    No fixture commits, live configuration reads, or admission/cap overrides.
    ``h.script(messages)`` supplies model replies; ``h.release`` unblocks the
    reviewer. ``h.factory`` only observes the actual production constructor.
    """
    from ouroboros import agent_task_pipeline as pipeline, context_health, review_custody
    from ouroboros.review_execution import ReviewAttemptResult

    data = Path(data)
    repo = Path(repo or Path(__file__).resolve().parents[1])
    data.mkdir(parents=True, exist_ok=True)
    (data / "memory").mkdir(exist_ok=True)
    (data / "memory" / "WORLD.md").write_text("# Isolated test environment\n", encoding="utf-8")
    settings = data / "settings.json"
    settings.write_text(json.dumps({
        "OUROBOROS_TASK_REVIEW_MODE": review_mode,
        "OUROBOROS_REVIEW_ENFORCEMENT": enforcement,
        "MCP_ENABLED": False,
    }), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(settings))
    monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)
    # These do not participate in turn review or bootstrap context propagation.
    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda *_: None)
    monkeypatch.setattr(context_health, "_stray_server_note", lambda *_: "")
    monkeypatch.setattr(pipeline, "_dispatch_root_post_task", lambda *_a, **_k: None)
    monkeypatch.setattr(review_substrate, "triad_delivery_slots", lambda **_kw: [
        ReviewSlot(slot_id="bootstrap-reviewer", model="openai/gpt-4o-mini", effort="high", timeout_sec=30)])
    h = SimpleNamespace(data=data, repo=repo, gate=PresenceTurnGate(1, state_root=data / "state"),
                        executions=PresenceTurnExecutions(), release=threading.Event(), entered=threading.Event(),
                        settled=threading.Event(), inputs=[], calls=0, reviews=[], agents=[], verdict="PASS",
                        lock=threading.Lock(), script=[])
    original_settle = review_custody._settle_review_attempt

    def settle(*args, **kwargs):
        try:
            return original_settle(*args, **kwargs)
        finally:
            h.settled.set()

    monkeypatch.setattr(review_custody, "_settle_review_attempt", settle)

    class HeldReviewer:
        def __init__(self, assignment):
            self.assignment = assignment

        def restore_custody(self, _state):
            pass

        def set_pending_invocation_checkpoint(self, _checkpoint):
            pass

        def prompt_payload(self):
            return {"messages": []}

        def prompt_chars(self):
            return 0

        def failure_custody(self):
            return {}

        def execute(self):
            from ouroboros.review_dispatch import invoke_review_paid_stamp
            from ouroboros.review_evidence_refs import acceptance_evidence_ref_vocabulary

            invoke_review_paid_stamp(self.assignment.dispatch_stamp)
            h.reviews.append(copy.deepcopy(self.assignment.request))
            h.entered.set()
            assert h.release.wait(30), "test failed to release reviewer"
            vocabulary = acceptance_evidence_ref_vocabulary(self.assignment.request.evidence)
            reference = next(key for key, basis in vocabulary.items() if basis in {"tool_record", "packet_section"})
            payload = json.dumps({"verdict": h.verdict, "summary": "Independent review", "findings": [],
                "outcome_tier": "solved" if h.verdict == "PASS" else "best_effort",
                "completion_coach": "Deliver the report after reading the new context.",
                "criteria_used": [{"criterion": "complete report", "status": "supported",
                                   "evidence_refs": [reference]}]})
            return ReviewAttemptResult(message={"content": payload}, raw_text=payload,
                usage={"prompt_tokens": 5, "completion_tokens": 3, "physical_attempt_state": "settled"})

    monkeypatch.setattr(review_substrate, "_review_route_executor", lambda assignment, **_kw: HeldReviewer(assignment))

    def model(_llm, messages, *_args, **kwargs):
        with h.lock:
            h.inputs.append(copy.deepcopy(messages))
            h.calls += 1
            step = h.script.pop(0) if isinstance(h.script, list) else h.script
        observer = kwargs.get("model_context_observer")
        if callable(observer):
            observer(messages)
        return (step(messages) if callable(step) else step), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", model)
    make_agent = agent_module.make_agent

    def factory(**kwargs):
        actual = make_agent(**kwargs)
        h.agents.append(actual)
        return actual

    h.factory = factory

    def start(turn_event):
        admitted = admission()
        return h.executions.start_thread(
            presence_turn_task_id(admitted.binding_id, turn_event.source_event_id),
            identity=presence_event_identity(admitted.binding_id, turn_event),
            admit=lambda: h.gate.acquire(turn_event.conversation_key),
            run=lambda lease: run_presence_turn(admission=admitted, event=turn_event, repo_dir=repo,
                drive_root=data, agent_factory=factory, admitted=lease), context=contextvars.copy_context())[0]

    h.start = start
    return h


@pytest.mark.parametrize("review_mode", ["auto", "required"])
def test_full_agent_bootstrap_propagates_review_wait_and_reenters_same_author(tmp_path, monkeypatch, review_mode):
    h = install_bootstrap_harness(monkeypatch, tmp_path / "data", review_mode=review_mode)
    first = event()

    def model(messages):
        resumed = next((row["content"] for row in messages
                        if row.get("role") == "user" and isinstance(row.get("content"), str)
                        and row["content"].startswith("[PRESENCE CONVERSATION RESUMED]")), "")
        if resumed:
            assert NEW_WORDS in resumed
            assert "telegram:bot-1:43" in resumed
            return finish("final", answer_sha256=h.agents[0].tools._ctx._delivery_candidate.content_sha256)
        if any(row.get("role") == "user" and NEW_WORDS in str(row.get("content")) for row in messages):
            return finish("second", message="Noted.")
        ctx = h.agents[0].tools._ctx
        if getattr(ctx, "_task_acceptance_pending", ""):
            return finish("wait-selection", answer_sha256=ctx._delivery_candidate.content_sha256)
        return nominate()

    h.script = model
    execution = h.start(first)
    try:
        initial = execution.initial.result(timeout=40)
        assert initial.status == "continuing" and initial.outcome == "deferred" and not initial.text
        assert not execution.result.done() and h.entered.is_set()
        ctx = h.agents[0].tools._ctx
        assert isinstance(h.agents[0], agent_module.OuroborosAgent)
        assert ctx.review_wait_callback is h.agents[0].review_wait_callback
        assert ctx.model_wait_context.tool_context is ctx
        assert ctx.inline_max_rounds == admission().inline_max_rounds == 10
        assert ctx.task_contract["capability_ceiling"]["digest"] == admission().capability_ceiling.digest
        wait_for(lambda: (load_task_result(h.data, initial.task_id).get("owner_wait") or {}).get("state") == "waiting",
                 what="production bootstrap author parked")
        calls_before_second = h.calls
        second = run_presence_turn(admission=admission(), event=event(43, NEW_WORDS), repo_dir=h.repo,
            drive_root=h.data, agent_factory=h.factory, gate=h.gate)
        assert second.text == "Noted." and not execution.result.done()
        assert h.calls == calls_before_second + 1 and len(h.agents) == 2
        h.release.set()
        completed = execution.result.result(timeout=30)
        assert completed.text == ANSWER and completed.task_id == initial.task_id
        assert len(h.agents) == 2 and h.agents[0].tools._ctx is ctx and h.calls == calls_before_second + 2
        assert len(h.reviews) == 1
        row = load_task_result(h.data, initial.task_id)
        assert row["status"] == "completed" and row["review_status"]["eligibility"] == "eligible"
    finally:
        h.release.set()
        if h.entered.is_set():
            assert h.settled.wait(15)
        execution.result.result(timeout=30)

"""#1536 production agent bootstrap coupled to the real review-yield lifecycle.

Only inference/reviewer transport and unrelated installation/post-task work are
scripted. Settings reload, context assembly, callback propagation, the loop,
review policy, review wallet, round limits and terminal writes stay real. The
author parks through the kept trigger of ``test_presence_continuation``: it
reviews its own result with ``review_change(subject="result")``, an addressed
message reaches its mailbox while the held reviewer runs, it leaves only its own
wait and its next plain prose parks on that review. The helper is also usable by
an already-isolated Host subprocess (``tests/presence_lifecycle_host.py``).
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
from tests.test_presence_continuation import (
    ADDRESSED, LEFT, RESUMED, SETTLED, review, settlement_notices_done, waits,
)
from tests.test_presence_runner import _admission, _event

ANSWER = "The status report is complete: all three checks passed."
NEW_WORDS = "Please also mention the backup window."


def tool_call(name, arguments, identifier):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


def finish(identifier, **arguments):
    return {"content": None, "tool_calls": [tool_call(
        "presence_finish", {"outcome": "message", **arguments}, identifier)]}


def seen(messages, marker, role=None):
    return any(marker in str(row.get("content")) and (role is None or row.get("role") == role) for row in messages)


def left_review(messages):
    """Whether this author's review call returned because it left the wave for an addressed message."""
    return seen(messages, LEFT, role="tool")


def woke(messages):
    """Whether the review this author left settled or its conversation came back (host facts are user rows;
    the production system prompt may describe the same markers)."""
    return seen(messages, RESUMED, role="user") or seen(messages, SETTLED, role="user")


def resumed_note(messages):
    return next((str(row["content"]) for row in messages
                 if row.get("role") == "user" and RESUMED in str(row.get("content"))), "")


def admission():
    """The behavior grants ``review_change`` (and the own-work reader); a result review needs no resource."""
    admitted = _admission()
    ceiling = replace(admitted.capability_ceiling, tool_grants=(
        *admitted.capability_ceiling.tool_grants, PresenceToolGrant("get_task_result"),
        PresenceToolGrant("review_change")))
    ceiling = replace(ceiling, digest=presence_ceiling_payload(ceiling)["digest"])
    return replace(admitted, capability_ceiling=ceiling)


def event(number=42, text="Please prepare the status report"):
    return replace(_event(), source_event_id=f"telegram:bot-1:{number}",
                   message={"message_id": str(number)}, text=text, continuation_version=1)


def wait_for(predicate, timeout=60, what="condition"):
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
    reviewer, which writes ``h.addressed`` to its author's mailbox as it starts
    (the addressed message the author leaves the wave for). ``h.factory`` only
    observes the actual production constructor.
    """
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("supervisor.consciousness_carriers.autonomous_work_enabled", lambda *_: True, raising=False)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {
        "status": "available", "remaining_usd": 10})
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_args, **_kwargs: (True, ""))
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
    # One seat is the whole review pool a result review draws from.
    monkeypatch.setattr("ouroboros.reviewer_slot_config.review_pool_slots", lambda *_a, **_kw: [
        ReviewSlot(slot_id="bootstrap-reviewer", model="openai/gpt-4o-mini", effort="high", timeout_sec=180)])
    h = SimpleNamespace(data=data, repo=repo, gate=PresenceTurnGate(1, state_root=data / "state"),
                        executions=PresenceTurnExecutions(), release=threading.Event(), entered=threading.Event(),
                        settled=threading.Event(), inputs=[], calls=0, reviews=[], agents=[], verdict="PASS",
                        addressed=ADDRESSED, lock=threading.Lock(), script=[])
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
            from ouroboros.owner_mailbox import PROVENANCE_INDEPENDENT_TASK, write_task_message
            from ouroboros.review_dispatch import invoke_review_paid_stamp

            invoke_review_paid_stamp(self.assignment.dispatch_stamp)
            h.reviews.append(copy.deepcopy(self.assignment.request))
            if h.addressed:  # an addressed message reaches the author while this reviewer runs
                assert write_task_message(h.data, h.addressed, self.assignment.request.task_id,
                                          source_task_id="operations-room", provenance=PROVENANCE_INDEPENDENT_TASK)
            h.entered.set()
            assert h.release.wait(180), "test failed to release reviewer"
            # review_change's answer contract: an empty array is a clean answer.
            payload = "[]" if h.verdict == "PASS" else json.dumps([{
                "item": "backup_window", "verdict": "FAIL", "severity": "critical",
                "reason": "The status report should mention the backup window."}])
            return ReviewAttemptResult(message={"content": payload}, raw_text=payload,
                usage={"prompt_tokens": 5, "completion_tokens": 3, "physical_attempt_state": "settled"})

    monkeypatch.setattr(review_substrate, "_review_route_executor", lambda assignment, **_kw: HeldReviewer(assignment))

    def model(_llm, messages, *_args, **_kwargs):
        with h.lock:
            h.inputs.append(copy.deepcopy(messages))
            h.calls += 1
            step = h.script.pop(0) if isinstance(h.script, list) else h.script
        response = step(messages) if callable(step) else step
        if h.agents and h.agents[-1].tools._ctx.task_metadata.get("wake_cause") == "focus":
            from tests.presence_legacy_helpers import focus_response
            response = focus_response(response)
        return response, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", model)
    make_agent = agent_module.make_agent

    def factory(**kwargs):
        actual = make_agent(**kwargs)
        from tests.presence_legacy_helpers import register_legacy_completion
        register_legacy_completion(actual.tools)
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
        resumed = resumed_note(messages)
        if resumed:
            assert NEW_WORDS in resumed
            assert "telegram:bot-1:43" in resumed
            return finish("final", message=ANSWER)
        if any(row.get("role") == "user" and NEW_WORDS in str(row.get("content")) for row in messages):
            return finish("second", message="Noted.")
        return waits() if left_review(messages) else review(answer=ANSWER)

    h.script = model
    execution = h.start(first)
    try:
        initial = execution.initial.result(timeout=90)
        assert initial.status == "continuing" and initial.outcome == "deferred" and not initial.text
        # The parked author's initial envelope and the reviewer's start run on separate threads, in
        # either order: wait for the held reviewer itself. It cannot answer before `h.release`, so
        # the author's final result is still open once it is inside its call.
        assert h.entered.wait(30), "the held reviewer never started"
        assert not execution.result.done()
        ctx = h.agents[0].tools._ctx
        assert isinstance(h.agents[0], agent_module.OuroborosAgent)
        assert ctx.review_wait_callback is h.agents[0].review_wait_callback
        assert ctx.model_wait_context.tool_context is ctx
        assert ctx.inline_max_rounds == admission().inline_max_rounds == 10
        assert ctx.task_contract["capability_ceiling"]["digest"] == admission().capability_ceiling.digest
        wait_for(lambda: (load_task_result(h.data, initial.task_id).get("owner_wait") or {}).get("state") == "waiting",
                 what="production bootstrap author parked")
        assert load_task_result(h.data, initial.task_id)["owner_wait"]["reason"] == "review"
        calls_before_second = h.calls
        second = run_presence_turn(admission=admission(), event=event(43, NEW_WORDS), repo_dir=h.repo,
            drive_root=h.data, agent_factory=h.factory, gate=h.gate)
        assert second.text == "Noted." and not execution.result.done()
        assert h.calls == calls_before_second + 1 and len(h.agents) == 2
        h.release.set()
        completed = execution.result.result(timeout=90)
        assert completed.text == ANSWER and completed.task_id == initial.task_id
        assert len(h.agents) == 2 and h.agents[0].tools._ctx is ctx and h.calls == calls_before_second + 2
        assert len(h.reviews) == 1
        row = load_task_result(h.data, initial.task_id)
        # The delivery bound the author's own settled result review: the turn's review axis is eligible.
        assert row["status"] == "completed" and row["review_status"]["eligibility"] == "eligible"
    finally:
        h.release.set()
        if h.entered.is_set():
            assert h.settled.wait(60)
            settlement_notices_done()
        execution.result.result(timeout=90)

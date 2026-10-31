"""The shared complete-Main-loop fixture: the real loop, the real result review at
delivery and a controlled reviewer transport.

``full_loop`` runs ``loop.run_llm_loop`` for a root task under task review
Required and blocking enforcement, with one produced artifact, so a final
delivery buys the Required result review (``review_change(subject="result")``)
through the ordinary review machinery. Only the scripted Main and the reviewer's
physical send (``HeldExecutor``) are substituted. Reviewers answer at once; a test
that needs a review still running clears ``release`` (or arms a seat's own gate in
``slot_releases``) and sets it again.

The task-acceptance panel these tests once drove (nomination, its background
panel, the acceptance park and free collection) is gone; its successor's own
coverage is ``tests/test_result_review.py``.
"""
from __future__ import annotations

import copy
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros import loop, review_substrate
from ouroboros.review_records import ReviewSlot
from ouroboros.tools.registry import ToolRegistry
from tests.test_loop_acceptance_gate import _seed_acceptance_root

ANSWER = "The complete report includes the requested budget."
STATUS = "How is it going?"
# What a failing reviewer answers under the result review's JSON-array contract.
CRITICAL_FINDING = json.dumps([{"item": "missing_budget", "verdict": "FAIL", "severity": "critical",
                                "reason": "The report does not state the requested budget."}])


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
    assert loop._owner_generation_changed(ctx) is False
    assert ctx._execution_trace["review_decision"]["admission_inspection"]["status"] == "unknown"


@pytest.fixture
def full_loop(tmp_path, monkeypatch, request):
    from ouroboros import review_custody
    from ouroboros.review_execution import ReviewAttemptResult
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "required")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "3")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "12")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_args: False)
    monkeypatch.setattr("ouroboros.tools.review_helpers.review_wave_budget_gate", lambda *_a, **_k: None)
    slot = ReviewSlot(slot_id="acceptance-one", model="fixture/reviewer", effort="high", timeout_sec=30)
    slots = [slot]
    # A result review composes its panel from the review pool: one packet seat, answered below.
    set_review_pool(monkeypatch, pool_roster(pool_seat(slot.slot_id, slot.model, effort=slot.effort)))
    monkeypatch.setattr(review_substrate, "triad_delivery_slots", lambda **_kw: slots)
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.repo_dir.mkdir()
    # Review distinguishes an unreadable repository from a clean one.
    import subprocess
    for args in (["init"], ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                           "commit", "--allow-empty", "-m", "fixture baseline"]):
        subprocess.run(["git", *args], cwd=registry._ctx.repo_dir, check=True, capture_output=True)
    ctx = registry._ctx
    task_id = "async-loop-root"
    _seed_acceptance_root(ctx.drive_root, task_id, ctx)
    ctx.task_contract["expected_output"] = "A complete report including its budget."
    # Required reviews actual production, never the expected_output declaration.
    from ouroboros.artifacts import store_task_artifact_bytes
    if getattr(request, "param", True):
        store_task_artifact_bytes(ctx.drive_root, task_id, "fixture-report.txt", ANSWER.encode())
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
                              settled_count=0, settled_operations=set(), reviewer_verdict="PASS", slots=slots,
                              slot_releases={})
    fixture.release.set()  # reviewers answer at once; a test clears it to hold them
    original_settle = review_custody._settle_review_attempt
    def settle(*a, **kw):
        try:
            return original_settle(*a, **kw)
        finally:
            fixture.settled.set()
            with fixture.condition:
                fixture.settled_operations.add(a[0].operation_id)
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
            fixture.slot_releases.get(self.assignment.slot.slot_id, fixture.release).wait()
            response_text = "[]" if fixture.reviewer_verdict == "PASS" else CRITICAL_FINDING
            return ReviewAttemptResult(message={"content": response_text}, raw_text=response_text,
                                       usage={"prompt_tokens": 5, "completion_tokens": 3, "physical_attempt_state": "settled"})

    monkeypatch.setattr(review_substrate, "_review_route_executor", lambda assignment, **_kw: HeldExecutor(assignment))
    def park(_ctx, checkpoint):
        """A review park (a review its author left): record it, release the reviewers, await them."""
        fixture.waits.append(copy.deepcopy(checkpoint))
        fixture.release.set()
        with fixture.condition:
            assert fixture.condition.wait_for(
                lambda: set(fixture.review_sends) <= fixture.settled_operations, timeout=10,
            ), "awaited review operations did not settle"
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
    try:
        yield fixture
    finally:
        fixture.release.set()
        for gate in fixture.slot_releases.values():
            gate.set()
        if fixture.entered.is_set():
            assert fixture.settled.wait(10)


def select_completion(f, answer=None, **options):
    """Main explicitly selects complete bytes using the source it actually received."""
    observation = f.ctx._acceptance_observation
    assert observation["owner_source_sha256"] in str(f.model_inputs[-1])
    selected = ({"answer": answer} if answer is not None else
                {"answer_sha256": f.ctx._delivery_candidate.content_sha256})
    arguments = {"action": "finish", **selected, "acceptance_subject": {
        "owner_source_sha256": observation["owner_source_sha256"],
    }, **options}
    return {"content": None, "tool_calls": [call("finish_task", arguments, f"finish-{f.model_step}")]}


def keep(f, **options):
    return select_completion(f, **options)


def _terminal_record(trace):
    """The host's own fold of the trace, as the terminal row and the card read it."""
    from ouroboros import outcomes

    review = outcomes._review_axis(trace)
    return {"status": "completed", "reason_code": "final_message",
            "outcome_axes": {"execution": {"status": "ok"}, "review": review,
                             "objective": outcomes._objective_axis(review)}}


def test_automatic_completion_is_reviewed_once_through_the_real_seam_before_delivery(full_loop, monkeypatch):
    """A plain final answer with produced work, under Required: the host reviews the result
    once through the ordinary review machinery, binds the PASS and delivers on it, with no
    further author round and no park."""
    f = full_loop
    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        assert f.model_step == 1, f.progress
        return {"content": ANSWER}, 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, _usage, trace = f.run()
    assert result == ANSWER and len(f.review_sends) == 1 and not f.waits
    [run] = trace["review_runs"]
    assert (run["surface"], run["authority"], run["aggregate_signal"]) == ("result", "host_root", "PASS")
    record = _terminal_record(trace)
    assert record["outcome_axes"]["review"]["source"] == "result_review"
    assert record["outcome_axes"]["objective"]["status"] == "pass"

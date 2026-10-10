"""#1536: a Presence author yields its conversation at a qualified review wait and returns.

The real gate (cap 1, cross-process files), Host execution registry, ``run_presence_turn``,
loop, acceptance coordinator, direct wait, mailbox settlement and pipeline terminal run
here; only the model and the reviewer transport are scripted. Git subprocesses create
the disposable repository fixture.
"""

from __future__ import annotations

import contextvars
import copy
import json
import pathlib
import queue
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline, loop, review_substrate
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.presence_runner import (
    PresenceTurnEvent, PresenceTurnExecutions, PresenceTurnGate, presence_turn_replay, run_presence_turn,
)
from ouroboros.review_records import ReviewSlot
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolRegistry
from tests.test_presence_runner import _admission

ANSWER = "The status report is complete: all three checks passed."
NEW_WORDS = "Please also mention the backup window."
KEY = "telegram:bot-1:room-1:topic-1"
pytestmark = pytest.mark.serial


def call(name, arguments, identifier):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def finish(identifier, **arguments):
    return {"content": None, "tool_calls": [call("presence_finish", {"outcome": "message", **arguments}, identifier)]}


def event(number=42, text="Please prepare the status report", version=1):
    return PresenceTurnEvent(
        source_event_id=f"telegram:bot-1:{number}", provider="telegram", account_id="bot-1",
        conversation_id="room-1", thread_id="topic-1", conversation_key=KEY,
        actor={"platform_actor_id": "user-7", "username": "alex"}, conversation={"title": "Community"},
        message={"message_id": str(number)}, text=text, continuation_version=version)


def wait_for(predicate, timeout=20.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A reviewed Presence turn: a held reviewer, a scripted model, the real everything else."""
    from ouroboros import review_custody
    from ouroboros.review_execution import ReviewAttemptResult

    for name, value in {"OUROBOROS_TASK_REVIEW_MODE": "auto", "OUROBOROS_REVIEW_ENFORCEMENT": "blocking",
                        "OUROBOROS_REVIEW_MAX_CYCLES": "3", "OUROBOROS_MAX_ROUNDS": "12",
                        "OUROBOROS_SAFETY_MODE": "off", "MCP_ENABLED": "false"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a: False)
    monkeypatch.setattr("ouroboros.tools.review_helpers.review_wave_budget_gate", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.review_evidence.acceptance_packet_budget_chars", lambda *_: 2_000_000)
    slots = [ReviewSlot(slot_id="acceptance-one", model="fixture/reviewer", effort="high", timeout_sec=30)]
    monkeypatch.setattr(review_substrate, "triad_delivery_slots", lambda **_kw: slots)
    data, repo = tmp_path / "data", tmp_path / "repo"
    data.mkdir()
    repo.mkdir()
    for args in (["init"], ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                           "commit", "--allow-empty", "-m", "fixture baseline"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    h = SimpleNamespace(data=data, repo=repo, gate=PresenceTurnGate(1, state_root=data / "state"),
                        executions=PresenceTurnExecutions(), release=threading.Event(), entered=threading.Event(),
                        verdict="PASS", reviews=[], inputs=[], script=[], calls=0, ctxs=[], progress=[],
                        lock=threading.Lock())
    settled = threading.Event()
    original_settle = review_custody._settle_review_attempt

    def settle(*a, **kw):
        try:
            return original_settle(*a, **kw)
        finally:
            settled.set()

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
            from ouroboros.review_evidence_refs import acceptance_evidence_ref_vocabulary

            invoke_review_paid_stamp(self.assignment.dispatch_stamp)
            h.reviews.append(copy.deepcopy(self.assignment.request))
            h.entered.set()
            assert h.release.wait(30), "the test never released the reviewer"
            vocabulary = acceptance_evidence_ref_vocabulary(self.assignment.request.evidence)
            reference = next(key for key, basis in vocabulary.items() if basis in {"tool_record", "packet_section"})
            text = json.dumps({
                "verdict": h.verdict, "summary": "Independent review", "findings": [],
                "outcome_tier": "solved" if h.verdict == "PASS" else "best_effort",
                "completion_coach": "Deliver it." if h.verdict == "PASS" else "Mention the backup.",
                "criteria_used": [{"criterion": "complete report", "status": "supported",
                                   "evidence_refs": [reference]}],
            })
            return ReviewAttemptResult(message={"content": text}, raw_text=text,
                                       usage={"prompt_tokens": 5, "completion_tokens": 3,
                                              "physical_attempt_state": "settled"})

    monkeypatch.setattr(review_substrate, "_review_route_executor", lambda assignment, **_kw: HeldExecutor(assignment))

    def model(_llm, messages, *_a, **kwargs):
        h.inputs.append(copy.deepcopy(messages))
        with h.lock:
            h.calls += 1
            step = h.script.pop(0) if isinstance(h.script, list) else h.script
        response = step(messages) if callable(step) else step
        observer = kwargs.get("model_context_observer")
        if callable(observer):  # as the real transport does: the context actually sent is observed
            observer(messages)
        return response, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", model)

    class Agent:
        def handle_task(self, task):
            registry = ToolRegistry(repo_dir=repo, drive_root=data)
            ctx = registry._ctx
            ctx.is_direct_chat = True
            ctx.task_metadata = dict(task["metadata"])
            # Auto review by the typed contract: Auto/Required eligibility is unchanged.
            ctx.task_contract = {**task["task_contract"], "expected_output": "A complete status report."}
            # The agent's own RUNNING write: the contract is the root's review-cap authority.
            write_task_result(data, task["id"], "running", task_contract=ctx.task_contract, root_task_id=task["id"])
            ctx.task_attempt, ctx.current_chat_id = 1, task["chat_id"]
            ctx.review_wait_callback = getattr(self, "review_wait_callback", None)
            registry.override_handler("chat_history", lambda *_a, **_kw: "Synthetic history")
            h.ctxs.append(ctx)
            task["_skip_post_task_synthesis"] = True
            with task_model_wait_scope(task=task, drive_root=data, event_queue=None, worker_slot_held=False) as waiter:
                ctx.model_wait_context, waiter.tool_context = waiter, ctx
                text, usage, trace = loop.run_llm_loop(
                    [{"role": "system", "content": "Presence turn."}, {"role": "user", "content": task["text"]}],
                    registry, SimpleNamespace(default_model=lambda: "fixture/main"), data / "logs",
                    lambda text, **_kw: h.progress.append(text), queue.Queue(), task_id=task["id"], drive_root=data)
            events = []
            pipeline.emit_task_results(SimpleNamespace(drive_root=data, repo_dir=repo), None, None, events, task,
                                       text, usage, trace, 0.0, data / "logs", ctx=ctx)
            return events

    h.Agent = Agent

    def start(turn_event, agent=None):
        """Start one turn exactly as the Host does: its own execution, admission and thread."""
        from ouroboros.presence_runner import presence_event_identity, presence_turn_task_id

        admission = _admission()
        return h.executions.start_thread(
            presence_turn_task_id(admission.binding_id, turn_event.source_event_id),
            identity=presence_event_identity(admission.binding_id, turn_event),
            admit=lambda: h.gate.acquire(turn_event.conversation_key),
            run=lambda lease: run_presence_turn(admission=admission, event=turn_event, repo_dir=repo, drive_root=data,
                                                agent_factory=lambda **_kw: agent or Agent(), admitted=lease),
            context=contextvars.copy_context())[0]

    h.start = start
    yield h
    h.release.set()
    if h.entered.is_set():
        assert settled.wait(15)


class _QuickAgent:
    """A second turn of the same conversation: answers at once, no review."""

    def handle_task(self, task):
        write_task_result(pathlib.Path(task["_data"]), task["id"], "completed", metadata=task["metadata"],
                          result="Noted.", terminal_origin="model_final")
        return [{"type": "presence_result", "outcome": "message", "text": "Noted.", "work_ref": ""}]


def quick_turn(h, number=43, text=NEW_WORDS):
    class Agent(_QuickAgent):
        def handle_task(self, task):
            task["_data"] = str(h.data)
            return super().handle_task(task)

    return run_presence_turn(admission=_admission(), event=event(number, text), repo_dir=h.repo, drive_root=h.data,
                             agent_factory=lambda **_kw: Agent(), gate=h.gate)


def test_blocking_author_yields_cap_one_and_returns_with_the_new_words(harness):
    h = harness
    first = event()

    def after_wake(messages):
        note = next((row["content"] for row in messages if "[PRESENCE CONVERSATION RESUMED]" in str(row.get("content"))),
                    None)
        assert note is not None, h.progress
        assert NEW_WORDS in note and "telegram:bot-1:43" in note
        return finish("final", answer_sha256=h.ctxs[0]._delivery_candidate.content_sha256)

    h.script = [finish("nominate", message=ANSWER), after_wake]
    execution = h.start(first)
    initial = execution.initial.result(timeout=30)
    # The durable initial envelope: nothing released under Blocking, the same author continues.
    assert (initial.status, initial.outcome, initial.text, initial.output_ref) == ("continuing", "deferred", "", "")
    assert initial.continuation_ref == initial.task_id and not execution.result.done()
    wait_for(lambda: (load_task_result(h.data, initial.task_id).get("owner_wait") or {}).get("state") == "waiting",
             what="the review park")
    row = load_task_result(h.data, initial.task_id)
    assert row["status"] == "running" and row["presence_continuation"]["initial"]["outcome"] == "deferred"
    assert row["owner_wait"]["reason"] == "review"
    # Cap 1: the second event of the SAME conversation runs while the author is parked.
    second = quick_turn(h)
    assert second.text == "Noted." and not execution.result.done()
    assert h.calls == 1  # the parked author bought no model round while waiting
    # Replay of the first event returns the identical stored envelope, rerunning nothing.
    from ouroboros.presence_runner import presence_event_identity

    replayed = presence_turn_replay(h.data, initial.task_id, KEY, presence_event_identity(_admission().binding_id, first), 1)
    assert replayed == initial
    h.release.set()
    final = execution.result.result(timeout=60)
    assert (final.status, final.outcome, final.text) == ("completed", "message", ANSWER)
    assert final.output_ref.startswith("presence-output-") and final.continuation_ref == initial.task_id
    assert h.calls == 2 and len(h.reviews) == 1
    stored = load_task_result(h.data, initial.task_id)
    assert stored["status"] == "completed" and stored["presence_continuation"]["initial"] == row["presence_continuation"]["initial"]
    # The newer turn's pointer survives this late completion; the open ref is gone.
    from ouroboros.presence_runner import _read_previous_turn

    pointer = _read_previous_turn(h.data, KEY)
    assert pointer["task_id"] == second.task_id and not pointer.get("open_turns")


def test_advisory_early_release_keeps_the_author_and_never_sends_the_selection_twice(harness, monkeypatch):
    h = harness
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")

    def after_wake(messages):
        assert any("[PRESENCE CONVERSATION RESUMED]" in str(row.get("content")) for row in messages)
        released = h.ctxs[0]._presence_released[0]
        return finish("again", answer_sha256=released["sha256"])

    h.script = [finish("nominate", message=ANSWER, pending_review="finish"), after_wake]
    execution = h.start(event())
    initial = execution.initial.result(timeout=30)
    assert (initial.status, initial.outcome, initial.text) == ("continuing", "message", ANSWER)
    assert initial.output_ref.startswith("presence-output-") and not execution.result.done()
    h.release.set()
    final = execution.result.result(timeout=60)
    # The author re-finalized its released selection: no second send, the same author ended it.
    assert (final.outcome, final.text, final.output_ref) == ("silent", "", "")
    assert h.calls == 2
    record = load_task_result(h.data, initial.task_id)["presence_continuation"]
    assert record["outputs"] == [{"output_ref": initial.output_ref, "outcome": "message", "text": ANSWER}]
    from ouroboros.presence_runner import _read_previous_turn

    pointer = _read_previous_turn(h.data, KEY)  # no newer turn: the pointer names the released speech
    assert (pointer["task_id"], pointer["message"], pointer.get("open_turns")) == (initial.task_id, ANSWER, None)
    rows = [json.loads(line) for line in (h.data / "logs" / "chat.jsonl").read_text().splitlines()]
    # Mode 0 history: the released output is logged once as authored speech, never again at the terminal.
    assert [row["text"] for row in rows if row.get("direction") == "out"] == [ANSWER]


def test_a_correction_with_identical_text_is_a_new_output(harness, monkeypatch):
    h = harness
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    h.script = [finish("nominate", message=ANSWER, pending_review="finish"),
                lambda _messages: finish("correction", message=ANSWER, pending_review="wait")]
    execution = h.start(event())
    initial = execution.initial.result(timeout=30)
    h.release.set()
    final = execution.result.result(timeout=60)
    assert final.outcome == "message" and final.text == ANSWER
    assert final.output_ref and final.output_ref != initial.output_ref


@pytest.mark.parametrize("version", [0, 1])
def test_cyber_keeps_the_released_author_for_its_criticism(harness, monkeypatch, version):
    h = harness
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr("ouroboros.acceptance_settlement.acceptance_wait_chosen", lambda _ctx: False)
    h.script = [finish("nominate", message=ANSWER),
                lambda _messages: finish("again", answer_sha256=h.ctxs[0]._delivery_candidate.content_sha256)]
    execution = h.start(event(version=version))
    initial = execution.initial.result(timeout=30)
    assert initial.status == "continuing" and not execution.result.done()
    # Only a version-1 consumer takes the early output; a legacy one gets it with the terminal.
    assert (initial.outcome, initial.text) == (("message", ANSWER) if version else ("deferred", ""))
    h.release.set()
    final = execution.result.result(timeout=60)
    assert h.calls == 2
    assert (final.outcome, final.text) == (("silent", "") if version else ("message", ANSWER))


def test_stop_while_parked_ends_without_another_call(harness):
    from ouroboros.cancel_intents import request_cancel

    h = harness
    h.script = [finish("nominate", message=ANSWER)]
    execution = h.start(event())
    initial = execution.initial.result(timeout=30)
    request_cancel(h.data, initial.task_id, reason="owner stop", source="test")
    try:
        final = execution.result.result(timeout=60)
    except Exception as exc:  # the loop's cancellation rail may also end typed
        final = SimpleNamespace(outcome="", text="", error=exc)
    assert h.calls == 1 and not h.script and final.text == ""  # no further call and no speech
    # The conversation is free: the stopped author never took it back.
    assert quick_turn(h).text == "Noted."


def test_exhausted_rounds_never_publish_the_held_text_because_a_panel_passed(harness, monkeypatch):
    h = harness
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")  # the author parks on its last inline round
    h.script = [finish("nominate", message=ANSWER), {"content": "Forced record without a declaration."}]
    execution = h.start(event())
    execution.initial.result(timeout=30)
    h.release.set()
    final = execution.result.result(timeout=60)
    # Rounds were not reset: the one forced call wrote a record without a declaration, so nothing speaks.
    assert final.text == "" and final.outcome in {"silent", "deferred"}
    assert h.calls == 2 and not h.script


def test_live_tool_custody_keeps_the_conversation(harness, monkeypatch):
    h = harness
    monkeypatch.setattr("ouroboros.presence_continuation._custody_blockers",
                        lambda _root, _task: [{"kind": "tool_handoff", "detail": "still running"}])
    h.script = [finish("nominate", message=ANSWER),
                lambda _messages: finish("final", answer_sha256=h.ctxs[0]._delivery_candidate.content_sha256)]
    execution = h.start(event())
    wait_for(h.entered.is_set, what="the panel")
    wait_for(lambda: (load_task_result(h.data, execution.turn_id) or {}).get("owner_wait", {}).get("state") == "waiting",
             what="the park")
    assert not execution.initial.done()  # nothing was published or lent
    blocked = threading.Event()
    threading.Thread(target=lambda: (h.gate.acquire(KEY).release(), blocked.set()), daemon=True).start()
    assert not blocked.wait(1.5), "a held conversation was taken by another turn"
    h.release.set()
    final = execution.result.result(timeout=60)
    assert final.text == ANSWER and not load_task_result(h.data, execution.turn_id).get("presence_continuation")
    assert blocked.wait(10)


def test_stop_while_reacquiring_behind_another_turn_ends_without_a_call(harness):
    from ouroboros.cancel_intents import request_cancel

    h = harness
    h.script = [finish("nominate", message=ANSWER)]
    execution = h.start(event())
    initial = execution.initial.result(timeout=30)
    holding, done = threading.Event(), threading.Event()

    class Holder(_QuickAgent):
        def handle_task(self, task):
            task["_data"] = str(h.data)
            holding.set()
            assert done.wait(30)
            return super().handle_task(task)

    second = threading.Thread(target=lambda: run_presence_turn(
        admission=_admission(), event=event(43, NEW_WORDS), repo_dir=h.repo, drive_root=h.data,
        agent_factory=lambda **_kw: Holder(), gate=h.gate), daemon=True)
    second.start()
    assert holding.wait(10)
    h.release.set()  # the panel settles: the author wakes and queues behind the second turn
    wait_for(lambda: (load_task_result(h.data, initial.task_id).get("owner_wait") or {}).get("state") == "resumed",
             what="the wake")
    request_cancel(h.data, initial.task_id, reason="owner stop", source="test")
    try:
        final = execution.result.result(timeout=30)
    except Exception:  # the loop's cancellation rail may also end typed
        final = SimpleNamespace(text="")
    assert final.text == "" and h.calls == 1  # it never took the conversation back nor called the model
    assert second.is_alive()  # the conversation stayed with the turn that held it
    done.set()
    second.join(30)


def resumed(messages):
    return next(str(row["content"]) for row in messages if "[PRESENCE CONVERSATION RESUMED]" in str(row.get("content")))


@pytest.mark.parametrize("failure", ["raises", "lost"])
def test_an_unwritten_projection_lends_and_publishes_nothing(harness, monkeypatch, failure):
    """Finding 1: a successor must never run without seeing the yielded author as open."""
    from ouroboros import presence_runner

    h = harness
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    original = presence_runner.atomic_write_json

    def write(path, value):
        if pathlib.Path(path).name.startswith("last-") and value.get("continuing"):
            if failure == "raises":
                raise OSError("disk full")
            return None  # a write that silently never landed: only the read-back sees it
        return original(path, value)

    monkeypatch.setattr(presence_runner, "atomic_write_json", write)

    def after_panel(messages):
        notes = [str(row.get("content")) for row in messages if "[PRESENCE RELEASE NOT PUBLISHED]" in str(row.get("content"))]
        assert notes and "could not durably record it" in notes[-1]
        assert not any("[PRESENCE CONVERSATION RESUMED]" in str(row.get("content")) for row in messages)
        return finish("final", answer_sha256=h.ctxs[0]._delivery_candidate.content_sha256)

    h.script = [finish("nominate", message=ANSWER, pending_review="finish"), after_panel]
    execution = h.start(event())
    wait_for(h.entered.is_set, what="the panel")
    wait_for(lambda: (load_task_result(h.data, execution.turn_id) or {}).get("owner_wait", {}).get("state") == "waiting",
             what="the park")
    assert not execution.initial.done()  # nothing published, no reservation returned
    row = load_task_result(h.data, execution.turn_id)
    assert not row.get("presence_continuation") and not getattr(h.ctxs[0], "_presence_released", None)
    blocked = threading.Event()
    threading.Thread(target=lambda: (h.gate.acquire(KEY).release(), blocked.set()), daemon=True).start()
    assert not blocked.wait(1.5), "the conversation was lent without its projection"
    h.release.set()
    final = execution.result.result(timeout=60)
    # The chosen answer was never released early, so the terminal speaks it, exactly once.
    assert (final.status, final.outcome, final.text) == ("completed", "message", ANSWER) and final.output_ref
    assert execution.initial.result(timeout=1) is final and h.calls == 2
    assert blocked.wait(10)


def test_a_stop_after_the_wake_ends_before_a_free_conversation_is_taken_back(harness, monkeypatch):
    """Finding 2: an immediately free conversation is never handed to a stopped author."""
    from ouroboros import owner_wait
    from ouroboros.cancel_intents import request_cancel

    h = harness
    original = owner_wait.direct_owner_wait

    def wake_then_stop(ctx, checkpoint):
        outcome = original(ctx, checkpoint)
        request_cancel(h.data, ctx.task_id, reason="owner stop", source="test")
        return outcome

    monkeypatch.setattr(owner_wait, "direct_owner_wait", wake_then_stop)
    h.script = [finish("nominate", message=ANSWER)]
    execution = h.start(event())
    execution.initial.result(timeout=30)
    h.release.set()
    try:
        final = execution.result.result(timeout=60)
    except Exception:  # the loop's cancellation rail may also end typed
        final = SimpleNamespace(text="")
    assert final.text == "" and h.calls == 1 and not h.script
    assert h.ctxs[0]._presence_conversation_lost == "cancelled"  # seen at reacquisition, not later
    assert quick_turn(h).text == "Noted."  # nothing stayed held


def test_the_resuming_author_reads_across_a_rotation_of_the_chat_log(harness):
    """Finding 3: rows land in an archived generation and in the new live one; none are lost."""
    from supervisor.state import rotate_chat_log_if_needed

    h = harness
    later = "And schedule the restore drill."

    def after_wake(messages):
        note = resumed(messages)
        assert NEW_WORDS in note and later in note and "taken by turn" in note
        assert "Coverage: every row of this conversation's canonical log since you yielded is listed." in note
        return finish("final", answer_sha256=h.ctxs[0]._delivery_candidate.content_sha256)

    h.script = [finish("nominate", message=ANSWER), after_wake]
    execution = h.start(event())
    execution.initial.result(timeout=30)
    assert quick_turn(h).text == "Noted."
    rotate_chat_log_if_needed(h.data, max_bytes=1)
    assert list((h.data / "archive").glob("chat_*.jsonl"))
    assert quick_turn(h, 44, later).text == "Noted."
    h.release.set()
    assert execution.result.result(timeout=60).text == ANSWER and h.calls == 2


@pytest.mark.parametrize("stance", ["accepted", "rejected"])
def test_an_advisory_fail_is_answered_by_the_same_author_with_a_typed_stance(harness, monkeypatch, stance):
    """Finding 5: after a real FAIL the Presence author states its stance; no second panel is forced."""
    h = harness
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    h.verdict = "FAIL"
    corrected = ANSWER + " The backup window is 02:00-03:00 UTC."

    def after_feedback(messages):
        assert "[PRESENCE CONVERSATION RESUMED]" in str(messages)
        assert "acceptance-one: FAIL" in str(messages)  # the panel's verdict reached its author
        if stance == "accepted":
            return finish("correct", message=corrected, author_disposition="accepted",
                          rationale="Added the backup window the review asked for.")
        return finish("keep", answer_sha256=h.ctxs[0]._delivery_candidate.content_sha256, author_disposition="rejected",
                      rationale="The backup window is out of scope for a status report.")

    h.script = [finish("nominate", message=ANSWER), after_feedback]
    execution = h.start(event())
    initial = execution.initial.result(timeout=30)
    assert initial.status == "continuing"
    h.release.set()
    final = execution.result.result(timeout=60)
    assert final.text == (corrected if stance == "accepted" else ANSWER) and final.outcome == "message"
    assert h.calls == 2 and len(h.reviews) == 1
    decision = load_task_result(h.data, initial.task_id)["review_status"]["acceptance_decision"]
    assert decision["reason"] == "author_finish" and decision["status"] == "finalized_unaccepted"
    author = decision["author_disposition"]
    assert (author["disposition"], author["action"], author["reviewer_signal"]) == (stance, "finish", "FAIL")
    assert author["rationale"] and author["enforcement"] == "advisory"


def test_presence_finish_refuses_a_stance_without_its_rationale(harness):
    from ouroboros.tools.presence import _finish_presence

    ctx = SimpleNamespace(task_contract={"capability_ceiling": {}}, _completion_request=None)
    for disposition, rationale in (("accepted", ""), ("approved", "why")):
        text = _finish_presence(ctx, "message", message=ANSWER, author_disposition=disposition, rationale=rationale)
        assert "author_disposition" in text and ctx._completion_request is None

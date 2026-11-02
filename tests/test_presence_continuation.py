"""#1536: a Presence author yields its conversation at a review it left and returns.

The kept trigger: the author reviews its own result with ``review_change(subject="result")``
and waits; an addressed message reaches its mailbox while the reviewer runs (an independent
task's words, as ``tell_task`` writes them to an inline Presence mailbox), so the author
leaves only its own wait (``review_pause.author_detaches``) and the tool returns pending
(``review_pause.watch_left_review``). Its next plain prose parks on that review
(``owner_wait.park_for_left_review`` -> ``wait_after_tools(review_binding=...)``), where the
Presence review-wait owner (``presence_continuation.presence_review_wait``) persists the
initial envelope and lends the conversation; the settlement notice wakes it and it
reacquires before any further model or tool step.

The real gate (cap 1, cross-process files), Host execution registry, ``run_presence_turn``,
loop, review coordinator, direct wait, mailbox settlement and pipeline terminal run here;
only the model and the reviewer transport are scripted. Git subprocesses create the
disposable repository fixture. ``test_presence_continuation_host`` and
``test_presence_queue_observations`` build on this harness.

The host-run acceptance review at a Presence delivery left with the task-acceptance machine:
Presence speech is never held now (a Required result review of it runs with nobody waiting,
``review_result._presence_result_review``), and the early release and the author's typed
stance went with it.
"""

from __future__ import annotations

import contextvars
import copy
import dataclasses
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
from tests.presence_legacy_helpers import legacy_registry as ToolRegistry
from tests.test_presence_runner import _admission

ANSWER = "The status report is complete: all three checks passed."
NEW_WORDS = "Please also mention the backup window."
# The addressed message that reaches the author while its reviewer runs (an independent task's words).
ADDRESSED = "From the operations room: the nightly backup now starts at 02:00 UTC."
RESUMED = "[PRESENCE CONVERSATION RESUMED]"
# The opening of the settlement notice the left review writes to the author's mailbox (``watch_left_review``).
SETTLED = "[Review settled] The reviewers finished"
LEFT = "An addressed message reached you while the reviewers ran"
KEY = "telegram:bot-1:room-1:topic-1"
pytestmark = pytest.mark.serial


def call(name, arguments, identifier):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def finish(identifier, **arguments):
    return {"content": None, "tool_calls": [call("presence_finish", {"outcome": "message", **arguments}, identifier)]}


def review(identifier="review", answer=ANSWER):
    """The author asks for a review of its own result and waits for it."""
    return {"content": None, "tool_calls": [call("review_change", {
        "subject": "result", "answer": answer, "reason": "Check the report before it is sent."}, identifier)]}


def waits(_messages=None):
    """Plain prose with no completion choice: the author keeps the review it left as its intention."""
    return {"content": "I read the message from the operations room; I wait for the review of my report."}


def _seen(messages, marker, role=None):
    return any(marker in str(row.get("content")) and (role is None or row.get("role") == role) for row in messages)


def author(after_wake, answer=ANSWER):
    """The scripted author of the kept trigger, decided from its own transcript only (so one script serves
    several turns): review the result, then (the review call returned because it left the wave for the
    addressed message) wait in prose, then, once the review settled or the conversation came back,
    ``after_wake``."""
    def step(messages):
        # Host facts arrive as user rows; a system prompt may describe the same markers.
        if _seen(messages, RESUMED, role="user") or _seen(messages, SETTLED, role="user"):
            return after_wake(messages)
        if _seen(messages, LEFT, role="tool"):
            return waits()
        return review(answer=answer)

    return step


def deliver(_messages=None):
    return finish("final", message=ANSWER)


def resumed(messages):
    return next(str(row["content"]) for row in messages if RESUMED in str(row.get("content")))


def event(number=42, text="Please prepare the status report", version=1):
    return PresenceTurnEvent(
        source_event_id=f"telegram:bot-1:{number}", provider="telegram", account_id="bot-1",
        conversation_id="room-1", thread_id="topic-1", conversation_key=KEY,
        actor={"platform_actor_id": "user-7", "username": "alex"}, conversation={"title": "Community"},
        message={"message_id": str(number)}, text=text, continuation_version=version)


def reviewing_admission():
    """The test admission whose positive ceiling also grants ``review_change`` and no resource at all: a result
    review takes no workspace binding (``tool_resolution._target_binding_operation``), so the author may review its
    own result with the tool grant alone."""
    from ouroboros.presence_authority import PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload

    admission = _admission()
    grants = tuple(sorted((*admission.capability_ceiling.tool_grants, PresenceToolGrant("review_change")),
                          key=lambda grant: grant.name))
    ceiling = PresenceCapabilityCeiling(**{**admission.capability_ceiling.__dict__, "tool_grants": grants,
                                           "digest": "0" * 64})
    ceiling = PresenceCapabilityCeiling(**{**ceiling.__dict__, "digest": presence_ceiling_payload(ceiling)["digest"]})
    return dataclasses.replace(admission, capability_ceiling=ceiling)


def wait_for(predicate, timeout=60.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def parked(h, task_id):
    """Whether the author waits at its review park."""
    wait = (load_task_result(h.data, task_id) or {}).get("owner_wait") or {}
    return wait.get("state") == "waiting" and wait.get("reason") == "review"


def settlement_notices_done(timeout=30.0):
    """A left review's settlement notice runs on its own thread after the seats settle
    (``review_pause.notify_when_settled``); a teardown waits for it to finish."""
    wait_for(lambda: not any(thread.name == "review-left-settlement" and thread.is_alive()
                             for thread in threading.enumerate()), timeout, "the left review's settlement notice")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A reviewed Presence turn: a held reviewer, a scripted model, the real everything else."""
    from ouroboros import review_custody
    from ouroboros.review_execution import ReviewAttemptResult

    for name, value in {"OUROBOROS_TASK_REVIEW_MODE": "required", "OUROBOROS_REVIEW_ENFORCEMENT": "blocking",
                        "OUROBOROS_REVIEW_MAX_CYCLES": "3", "OUROBOROS_MAX_ROUNDS": "12",
                        "OUROBOROS_SAFETY_MODE": "off", "MCP_ENABLED": "false"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a: False)
    monkeypatch.setattr("ouroboros.tools.review_helpers.review_wave_budget_gate", lambda *_a, **_k: None)
    # One held reviewer seat is the whole review pool a result review draws from.
    slots = [ReviewSlot(slot_id="reviewer-one", model="fixture/reviewer", effort="high", timeout_sec=180)]
    monkeypatch.setattr("ouroboros.reviewer_slot_config.review_pool_slots", lambda *_a, **_kw: list(slots))
    data, repo = tmp_path / "data", tmp_path / "repo"
    data.mkdir()
    repo.mkdir()
    for args in (["init"], ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                           "commit", "--allow-empty", "-m", "fixture baseline"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    h = SimpleNamespace(data=data, repo=repo, gate=PresenceTurnGate(1, state_root=data / "state"),
                        executions=PresenceTurnExecutions(), release=threading.Event(), entered=threading.Event(),
                        verdict="PASS", reviews=[], inputs=[], script=[], calls=0, ctxs=[], progress=[],
                        addressed=ADDRESSED, lock=threading.Lock())
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
            from ouroboros.owner_mailbox import PROVENANCE_INDEPENDENT_TASK, write_task_message
            from ouroboros.review_dispatch import invoke_review_paid_stamp

            invoke_review_paid_stamp(self.assignment.dispatch_stamp)
            h.reviews.append(copy.deepcopy(self.assignment.request))
            if h.addressed:  # an addressed message reaches the author while this reviewer runs
                assert write_task_message(h.data, h.addressed, self.assignment.request.task_id,
                                          source_task_id="operations-room", provenance=PROVENANCE_INDEPENDENT_TASK)
            h.entered.set()
            assert h.release.wait(180), "the test never released the reviewer"
            # review_change's answer contract: an empty array is a clean answer.
            text = "[]" if h.verdict == "PASS" else json.dumps([{
                "item": "backup_window", "verdict": "FAIL", "severity": "critical",
                "reason": "The status report should mention the backup window."}])
            return ReviewAttemptResult(message={"content": text}, raw_text=text,
                                       usage={"prompt_tokens": 5, "completion_tokens": 3,
                                              "physical_attempt_state": "settled"})

    monkeypatch.setattr(review_substrate, "_review_route_executor", lambda assignment, **_kw: HeldExecutor(assignment))

    def model(_llm, messages, *_a, **_kwargs):
        h.inputs.append(copy.deepcopy(messages))
        with h.lock:
            h.calls += 1
            step = h.script.pop(0) if isinstance(h.script, list) else h.script
        response = step(messages) if callable(step) else step
        if h.ctxs and h.ctxs[-1].task_metadata.get("wake_cause") == "focus":
            from tests.presence_legacy_helpers import focus_response
            response = focus_response(response)
        return response, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", model)

    class Agent:
        def handle_task(self, task):
            registry = ToolRegistry(repo_dir=repo, drive_root=data)
            ctx = registry._ctx
            ctx.is_direct_chat = True
            ctx.task_metadata = dict(task["metadata"])
            ctx.task_contract = {**task["task_contract"], "expected_output": "A complete status report."}
            # The agent's own RUNNING write: the contract is the root's review-cap authority.
            write_task_result(data, task["id"], "running", task_contract=ctx.task_contract, root_task_id=task["id"])
            # The report the turn retained is the result its author reviews (#1731).
            from ouroboros.artifacts import store_task_artifact_bytes
            store_task_artifact_bytes(data, task["id"], "report.txt", ANSWER.encode())
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

        admission = reviewing_admission()
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
        assert settled.wait(60)
        settlement_notices_done()


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


def test_author_yields_cap_one_at_a_left_review_and_returns_with_the_new_words(harness):
    h = harness
    first = event()

    def after_wake(messages):
        note = resumed(messages)
        assert NEW_WORDS in note and "telegram:bot-1:43" in note
        assert _seen(messages, SETTLED)  # the left review's settlement notice reached the same author
        return deliver()

    h.script = author(after_wake)
    execution = h.start(first)
    initial = execution.initial.result(timeout=90)
    # The durable initial envelope: the same author continues, nothing was spoken yet.
    assert (initial.status, initial.outcome, initial.text, initial.output_ref) == ("continuing", "deferred", "", "")
    assert initial.continuation_ref == initial.task_id and not execution.result.done()
    wait_for(lambda: parked(h, initial.task_id), what="the review park")
    row = load_task_result(h.data, initial.task_id)
    assert row["status"] == "running" and row["presence_continuation"]["initial"]["outcome"] == "deferred"
    assert row["presence_continuation"]["review_binding"] == row["owner_wait"]["review_binding"]
    assert row["owner_wait"]["review_binding"].startswith("result-review:")
    # The author left the wave only for the addressed message: the review is still running, unsettled.
    assert h.entered.is_set() and ADDRESSED in str(h.inputs[-1])
    # Cap 1: the second event of the SAME conversation runs while the author is parked.
    second = quick_turn(h)
    assert second.text == "Noted." and not execution.result.done()
    assert h.calls == 2  # review, then the prose that parked: no model round while waiting
    # Replay of the first event returns the identical stored envelope, rerunning nothing.
    from ouroboros.presence_runner import presence_event_identity

    replayed = presence_turn_replay(h.data, initial.task_id, KEY, presence_event_identity(_admission().binding_id, first), 1)
    assert replayed == initial
    h.release.set()
    final = execution.result.result(timeout=60)
    assert (final.status, final.outcome, final.text) == ("completed", "message", ANSWER)
    assert final.output_ref.startswith("presence-output-") and final.continuation_ref == initial.task_id
    # The delivery binds the settled review of the same result: one review, no second wave.
    assert h.calls == 3 and len(h.reviews) == 1
    stored = load_task_result(h.data, initial.task_id)
    assert stored["status"] == "completed" and stored["presence_continuation"]["initial"] == row["presence_continuation"]["initial"]
    # The newer turn's pointer survives this late completion; the open ref is gone.
    from ouroboros.presence_runner import _read_previous_turn

    pointer = _read_previous_turn(h.data, KEY)
    assert pointer["task_id"] == second.task_id and not pointer.get("open_turns")


def test_stop_while_parked_ends_without_another_call(harness):
    from ouroboros.cancel_intents import request_cancel

    h = harness
    h.script = [review(), waits()]
    execution = h.start(event())
    initial = execution.initial.result(timeout=90)
    assert initial.status == "continuing"  # parked at the review it left
    request_cancel(h.data, initial.task_id, reason="owner stop", source="test")
    try:
        final = execution.result.result(timeout=60)
    except Exception as exc:  # the loop's cancellation rail may also end typed
        final = SimpleNamespace(outcome="", text="", error=exc)
    assert h.calls == 2 and not h.script and final.text == ""  # no further call and no speech
    # The conversation is free: the stopped author never took it back.
    assert quick_turn(h).text == "Noted."


def test_exhausted_rounds_never_publish_the_reviewed_answer_because_its_review_passed(harness, monkeypatch):
    h = harness
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "2")  # the author parks on its last inline round
    h.script = [review(), waits(), {"content": "Forced record without a declaration."}]
    execution = h.start(event())
    assert execution.initial.result(timeout=90).status == "continuing"
    h.release.set()  # the review passes the answer the author never declared
    final = execution.result.result(timeout=60)
    # Rounds were not reset: the one forced call wrote a record without a declaration, so nothing speaks.
    assert final.text == "" and final.outcome in {"silent", "deferred"}
    assert h.calls == 3 and not h.script


def test_live_tool_custody_keeps_the_conversation(harness, monkeypatch):
    h = harness
    monkeypatch.setattr("ouroboros.presence_continuation._custody_blockers",
                        lambda _root, _task: [{"kind": "tool_handoff", "detail": "still running"}])
    h.script = author(deliver)
    execution = h.start(event())
    wait_for(h.entered.is_set, what="the review")
    wait_for(lambda: parked(h, execution.turn_id), what="the park")
    assert not execution.initial.done()  # nothing was published or lent
    blocked = threading.Event()
    threading.Thread(target=lambda: (h.gate.acquire(KEY).release(), blocked.set()), daemon=True).start()
    assert not blocked.wait(1.5), "a held conversation was taken by another turn"
    h.release.set()
    final = execution.result.result(timeout=60)
    assert final.text == ANSWER and not load_task_result(h.data, execution.turn_id).get("presence_continuation")
    assert not any(_seen(messages, RESUMED) for messages in h.inputs)  # it never yielded, so it never came back
    assert blocked.wait(10)


def test_stop_while_reacquiring_behind_another_turn_ends_without_a_call(harness):
    from ouroboros.cancel_intents import request_cancel

    h = harness
    h.script = [review(), waits()]
    execution = h.start(event())
    initial = execution.initial.result(timeout=90)
    assert initial.status == "continuing"
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
    h.release.set()  # the review settles: the author wakes and queues behind the second turn
    wait_for(lambda: (load_task_result(h.data, initial.task_id).get("owner_wait") or {}).get("state") == "resumed",
             what="the wake")
    request_cancel(h.data, initial.task_id, reason="owner stop", source="test")
    try:
        final = execution.result.result(timeout=30)
    except Exception:  # the loop's cancellation rail may also end typed
        final = SimpleNamespace(text="")
    assert final.text == "" and h.calls == 2  # it never took the conversation back nor called the model
    assert second.is_alive()  # the conversation stayed with the turn that held it
    done.set()
    second.join(30)


@pytest.mark.parametrize("failure", ["raises", "lost"])
def test_an_unwritten_projection_lends_and_publishes_nothing(harness, monkeypatch, failure):
    """Finding 1: a successor must never run without seeing the yielded author as open."""
    from ouroboros import presence_runner

    h = harness
    original = presence_runner.atomic_write_json

    def write(path, value):
        if pathlib.Path(path).name.startswith("last-") and value.get("continuing"):
            if failure == "raises":
                raise OSError("disk full")
            return None  # a write that silently never landed: only the read-back sees it
        return original(path, value)

    monkeypatch.setattr(presence_runner, "atomic_write_json", write)

    def after_settlement(messages):
        assert not _seen(messages, RESUMED)  # the conversation was never lent, so it never came back
        return deliver()

    h.script = author(after_settlement)
    execution = h.start(event())
    wait_for(h.entered.is_set, what="the review")
    wait_for(lambda: parked(h, execution.turn_id), what="the park")
    assert not execution.initial.done()  # nothing published, no reservation returned
    assert not load_task_result(h.data, execution.turn_id).get("presence_continuation")
    blocked = threading.Event()
    threading.Thread(target=lambda: (h.gate.acquire(KEY).release(), blocked.set()), daemon=True).start()
    assert not blocked.wait(1.5), "the conversation was lent without its projection"
    h.release.set()
    final = execution.result.result(timeout=60)
    # Nothing was spoken while the author waited, so the terminal speaks its answer, exactly once.
    assert (final.status, final.outcome, final.text) == ("completed", "message", ANSWER) and final.output_ref
    assert execution.initial.result(timeout=1) is final and h.calls == 3
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
    h.script = [review(), waits()]
    execution = h.start(event())
    assert execution.initial.result(timeout=90).status == "continuing"
    h.release.set()
    try:
        final = execution.result.result(timeout=60)
    except Exception:  # the loop's cancellation rail may also end typed
        final = SimpleNamespace(text="")
    assert final.text == "" and h.calls == 2 and not h.script
    assert h.ctxs[0]._presence_conversation_lost == "cancelled"  # seen at reacquisition, not later
    assert quick_turn(h).text == "Noted."  # nothing stayed held


def test_the_resuming_author_reads_sql_rows_across_a_legacy_rotation_attempt(harness):
    """New chat rows are durable in SQL; the retired file rotator cannot hide them."""
    from supervisor.state import rotate_jsonl_log_if_needed

    h = harness
    later = "And schedule the restore drill."

    def after_wake(messages):
        note = resumed(messages)
        assert NEW_WORDS in note and later in note and "taken by turn" in note
        assert "Coverage: every row of this conversation's canonical log since you yielded is listed." in note
        return deliver()

    h.script = author(after_wake)
    execution = h.start(event())
    assert execution.initial.result(timeout=90).status == "continuing"
    assert quick_turn(h).text == "Noted."
    rotate_jsonl_log_if_needed(h.data, "chat.jsonl", "chat", max_bytes=1)
    assert not (h.data / "logs" / "chat.jsonl").exists()
    assert quick_turn(h, 44, later).text == "Noted."
    h.release.set()
    assert execution.result.result(timeout=60).text == ANSWER and h.calls == 3


def test_the_review_change_grant_alone_parks_for_a_result_but_a_change_needs_its_root(harness):
    """A result review takes no workspace binding: the tool grant alone lets the author review its own result
    and park, while a change subject from the same author stays bound to its root and is refused without one."""
    h = harness
    assert not reviewing_admission().capability_ceiling.resource_grants  # no vcs root at all
    change = {"content": None, "tool_calls": [call("review_change", {
        "subject": "index", "reason": "Check the staged change."}, "change")]}

    def step(messages):
        if _seen(messages, RESUMED, role="user") or _seen(messages, SETTLED, role="user"):
            return deliver()
        if _seen(messages, LEFT, role="tool"):
            return waits()
        if _seen(messages, "PRESENCE_RESOURCE_BLOCKED", role="tool"):
            return review()
        return change

    h.script = step
    execution = h.start(event())
    initial = execution.initial.result(timeout=90)
    assert initial.status == "continuing"  # the result review parked the author
    refusal = [str(row.get("content")) for row in h.inputs[1] if row.get("role") == "tool"]
    assert len(refusal) == 1 and "PRESENCE_RESOURCE_BLOCKED" in refusal[0]  # the change review, before any seat
    wait_for(lambda: parked(h, initial.task_id), what="the review park")
    assert load_task_result(h.data, initial.task_id)["owner_wait"]["review_binding"].startswith("result-review:")
    assert len(h.reviews) == 1
    h.release.set()
    final = execution.result.result(timeout=60)
    assert (final.outcome, final.text) == ("message", ANSWER) and h.calls == 4 and len(h.reviews) == 1

"""A delegated (agent_session) seat in a receptive review_change wave whose author left only
its own wait before the seat's first launch (live incident on 740a649ec, task b54e4bffb8744066;
the same refusal after a selected-outcome or wake-time exit in record rl-20261025T181307-e6e9a04aa6d8).

The drain records such a seat ``pending_dispatch`` ("preparing its first launch, which still
sends"). That seat still sends: its delegated start writes START_REQUESTED, then checkpoints
its invocation onto the attempt row, which must take it to ``in_flight``; a refusal there
("reserved commit-review operation is not in flight") leaves the paid seat unanswered. Only
the Claudexor gateway and the API seat's model are scripted; the wave, attempt rows, custody
and checkpoint are real.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody, review_pause
from ouroboros.owner_mailbox import write_owner_message
from ouroboros.review_state import load_state
from tests._review_session_route_shared import FakeGateway, _owned_gateway_uses_each_test_transport  # noqa: F401
from tests._review_session_route_shared import _terminal_detail, fake_route as _fake_route
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_receptive_commit_review import TASK, _git, _join, _notices, _parent, _produce, _read_mail, until
from tests.test_receptive_commit_review import author as _author

author, fake_route = _author, _fake_route
pytestmark = pytest.mark.serial
SESSION_ANSWER = "SESSION-SEAT-ANSWER"
_PAST = "2000-01-01T00:00:00Z"  # a wake time already reached when the wait starts


@pytest.fixture
def session_wave(author, fake_route, monkeypatch):
    """review_change over [API seat m1, delegated seat 'second'], the delegated seat held
    before its START_REQUESTED until ``release`` is set."""
    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS

    from ouroboros import config
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools import review
    from ouroboros.triad_review import REVIEW_TWO_PART_OBJECT_CONTRACT

    a = author
    set_review_pool(monkeypatch, pool_roster(pool_seat("first", "test/m1", delivery="native"),
                                             pool_seat("second", "fake-review=fake-small", kind="agent_session")))
    routes = [ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION]
    plan = {"models": ["m1", "fake-review=fake-small"], "routes": routes, "efforts": ["high"] * 2,
            "slot_ids": ["first", "second"], "use_local": [False] * 2, "subagent_ids": ["", ""],
            "session_targets": ["", "fake-review=fake-small"], "session_profiles": ["", ""],
            "retrieves": [True, True], "parts": [("change", "coupling")] * 2,
            "session_tasks": ["Review this candidate."] * 2,
            "session_policies": [{"output_contract": REVIEW_TWO_PART_OBJECT_CONTRACT}] * 2,
            "brief_shas": ["brief-sha"] * 2}

    def prepared(ctx, commit_message, review_rebuttal="", repo_dir=None, goal="", scope="", subject=None):
        from ouroboros.tools.review_helpers import review_enforcement_blocks
        root = subject.review_root if subject is not None else ctx.repo_dir
        return ({"blocking_review": review_enforcement_blocks(config.get_review_enforcement()),
                 "prompt": "Review this candidate.", "models": list(plan["models"]), "stable_prefix_len": 0,
                 "routes": list(routes), "session_task": "", "target_repo": root,
                 "row_plan": json.loads(json.dumps(plan, default=str)) | {"routes": list(routes)},
                 "layer": "core", "brief_texts": {"brief-sha": "Review this candidate."}}, None, False)
    monkeypatch.setattr(review, "_prepare_unified_review", prepared)
    answer = {"change": [], "change_clean": True, "coupling": [
        {"item": item, "verdict": "PASS", "severity": "advisory", "reason": SESSION_ANSWER}
        for item in sorted(SCOPE_REQUIRED_ITEMS)]}
    FakeGateway.detail = _terminal_detail(json.dumps(answer), conformance="passed")
    a.preparing, a.release = threading.Event(), threading.Event()
    real = delegate_custody.record_start_requested

    def held(*args, **kwargs):
        a.preparing.set()
        assert a.release.wait(60), "the delegated seat was never released"
        return real(*args, **kwargs)
    monkeypatch.setattr(delegate_custody, "record_start_requested", held)
    _git(a.repo, "add", "change.py")
    yield a
    a.release.set()


def _review_change(a, outcome, **wake):
    from ouroboros.tools.review_change import _handle_review_change

    def run():
        with _parent(a):
            outcome["result"] = _handle_review_change(a.ctx, root="system_repo", subject="index",
                                                      goal="Raise the value.", scope="change.py only", **wake)
    return _produce(a, run)


def _session_row(a):
    rows = [row for row in load_state(a.data).attempts[-1].triad_raw_results if row.get("slot_id") == "second"]
    assert len(rows) == 1, rows
    return rows[0]


def _posts():
    return [body for gateway in FakeGateway.instances for body in gateway.start_requests]


def _payload(text):
    """The tool's JSON object, after any leading message and before a delivered wake line."""
    return json.JSONDecoder().raw_decode(text, 0 if text.startswith("{") else text.index("\n{") + 1)[0]


@pytest.mark.parametrize("left", ["mail", "wake_at", None])
def test_a_delegated_seat_left_before_its_launch_still_checkpoints_and_answers(session_wave, left):
    """Left before its launch, on mail or at the author's own past wake time, the delegated seat
    still starts once after the author's wait returned (its checkpoint moves the row to in flight)
    and its answer is collected at $0 by the same unchanged call; a wave its author waits for
    settles with that answer, as before."""
    a, outcome = session_wave, {}
    thread = _review_change(a, outcome, **({"wake_on": {"wake_at": _PAST}} if left == "wake_at" else {}))
    assert a.reviewer.sent["m1"].wait(60) and a.preparing.wait(60)  # held before START_REQUESTED until release
    if left == "mail":
        write_owner_message(a.data, "Is the review done?", TASK)
    if left:
        _join(thread)
        returned = {"mail": "An addressed message reached you", "wake_at": "A selected outcome or wake time was ready"}
        assert outcome["result"].startswith(returned[left] + " while the reviewers ran"), outcome
        assert _session_row(a)["operation_state"] == "pending_dispatch"  # left before its first launch
        assert not _posts(), "nothing was sent before the author's wait returned"
    a.release.set()
    a.reviewer.release()
    if left:
        until(lambda: _notices(a, "change"), timeout=60)
        seat = _session_row(a)
        # The seat's own start reached the gateway after the wait returned; no later call started it.
        assert len(_posts()) == 1, f"settled unsent: {seat.get('failure_code')}: {seat.get('error')}"
        _read_mail(a)
        outcome = {}
        _join(_review_change(a, outcome))  # the same unchanged call collects the settled wave
    else:
        _join(thread)
    collected = _payload(outcome["result"])
    assert collected["state"] == "settled", outcome["result"]
    assert len(_posts()) == 1, "the delegated seat started exactly once and was never sent again"
    row = _session_row(a)
    assert "not in flight" not in str(row.get("raw_text") or "") + str(row.get("error") or ""), row
    assert SESSION_ANSWER in str(row.get("raw_text") or ""), row


def test_a_pause_after_the_wake_time_still_fences_the_held_seat(session_wave):
    """The owner pauses after the author's wake-time exit, while the delegated seat still prepares
    its first launch: that exit latched nothing, and the seat's own launch admission still refuses
    it, so nothing reaches the gateway."""
    from ouroboros import owner_pause

    a, outcome = session_wave, {}
    thread = _review_change(a, outcome, wake_on={"wake_at": _PAST})
    assert a.reviewer.sent["m1"].wait(60) and a.preparing.wait(60)  # held before START_REQUESTED until release
    _join(thread)
    assert _session_row(a)["operation_state"] == "pending_dispatch"
    owner_pause.install_fence(a.data, TASK, request_id="pause-after-wake")
    try:
        a.release.set()
        a.reviewer.release()
        until(lambda: _notices(a, "change"), timeout=60)  # every seat of the left wave settled
        seat = _session_row(a)
        assert not _posts(), f"sent under Pause: {seat.get('operation_state')}: {seat.get('error')}"
    finally:
        owner_pause.release_fence(a.data, TASK, reason="owner_resume")


@pytest.mark.parametrize("finished", ["before_the_checkpoint", "while_the_checkpoint_waits"])
def test_a_helper_finishing_while_the_seat_checkpoints_still_lets_it_send(session_wave, monkeypatch, finished):
    """The selected helper finishes while the delegated seat still prepares its first launch: before
    its start checkpoint, or while that checkpoint waits for the review-state lock. Either way the
    author's drain records the seat ``pending_dispatch``, the checkpoint reads the left wait at that
    row's transition under the lock, and the seat starts exactly once (a read taken before the lock
    refused the second case as "not in flight")."""
    from ouroboros import review_state, review_state_custody
    from ouroboros.task_results import write_task_result

    a, outcome = session_wave, {}
    waiting, resume = threading.Event(), threading.Event()
    real_update = review_state.update_state

    def seat_checkpoint_waits(*args, **kwargs):  # the seat's start checkpoint, before its lock and load
        if kwargs.get("lock_timeout_sec") == review_state_custody.SEAT_LAUNCH_LOCK_WAIT_SEC:
            waiting.set()
            assert resume.wait(60), "the seat's checkpoint was never resumed"
        return real_update(*args, **kwargs)
    monkeypatch.setattr(review_state, "update_state", seat_checkpoint_waits)
    write_task_result(a.data, "child", "running", parent_task_id=TASK)
    thread = _review_change(a, outcome, wake_on={"tasks": ["child"]})
    try:
        assert a.reviewer.sent["m1"].wait(60) and a.preparing.wait(60)  # held before START_REQUESTED until release
        if finished == "while_the_checkpoint_waits":
            a.release.set()
            assert waiting.wait(60), "the delegated seat never reached its start checkpoint"
        write_task_result(a.data, "child", "completed", parent_task_id=TASK)
        _join(thread)
        seat = _session_row(a)
        assert seat["operation_state"] == "pending_dispatch", seat
        assert ("A selected outcome or wake time was ready while this reviewer was preparing its first launch, "
                "which still sends") in seat["raw_text"], seat
        assert not _posts(), "nothing was sent before the author's wait returned"
    finally:
        resume.set()
        a.release.set()
        a.reviewer.release()
    until(lambda: _notices(a, "change"), timeout=60)
    seat = _session_row(a)
    assert len(_posts()) == 1, f"settled unsent: {seat.get('failure_code')}: {seat.get('error')}"


# What ended only the author's wait, and the reason the drain records for it.
_WAIT_ONLY = {"addressed_message": review_pause.ADDRESSED_EVENT, "helper_terminal": review_pause.HELPER_EVENT,
              "selected_review": review_pause.REVIEW_EVENT, "wake_at": review_pause.TIME_EVENT,
              "departed_author": review_pause.DEPARTED_EVENT}


def _leave_wait(tmp_path, ctx, operation, entry, cause):
    """End only the author's wait on ``cause`` through the drain's own observer (``author_detaches``),
    from a real wake source: mail, a selected child's terminal, a selected seat's published answer,
    a reached wake time, or an author nobody waits for."""
    from ouroboros import review_waiting
    from ouroboros.task_results import write_task_result

    entries = [entry]
    if cause == "departed_author":
        binding = review_pause.departed_binding(ctx)
    else:
        if cause == "helper_terminal":
            write_task_result(tmp_path, "child", "completed", parent_task_id=TASK)
        state = review_waiting.select_helpers(ctx, {"helper_terminal": {"tasks": ["child"]},
                                                    "wake_at": {"wake_at": _PAST}}.get(cause))
        if cause == "addressed_message":
            write_owner_message(tmp_path, "Is the review done?", TASK)
        if cause == "selected_review":
            state["selection"]["reviewers"] = [{"wave_id": "wave", "operation_id": "op-0", "slot_id": "first"}]
            answered = SimpleNamespace(operation_id="op-0", event=threading.Event(), actor=SimpleNamespace(
                review_wave_id="wave", outcome={"operation_id": "op-0", "slot_id": "first", "outcome_id": "op-0:first",
                                                "status": "responded", "additional": False, "verdict": "PASS"}))
            answered.event.set()
            entries.append(answered)
        binding = review_pause.receptive_binding(ctx, state=state)
    with review_pause.receptive_author(binding):
        assert review_pause.author_detaches(operation, entries)


@pytest.mark.parametrize("detach", [*_WAIT_ONLY, "pause_preparation", "wake_at_then_pause", "not_live",
                                    "other_live_operation", "row_of_another_operation"])
def test_a_seat_left_only_waiting_moves_to_in_flight_at_its_start(tmp_path, monkeypatch, detach):
    """Whatever ended only the author's wait before a seat's first launch (addressed mail, a selected
    helper terminal or reviewer answer, the wake time, a departed author), the drain's PENDING row says
    the preparation still sends and the seat's start checkpoint moves it to in flight: producer and
    checkpoint read one definition (``review_pause.left_wait_only``). A Pause preparation never sends,
    not even after an earlier wait-only exit, and neither does a row without this exact live seat."""
    from ouroboros import owner_pause, review_custody
    from ouroboros.review_custody import ActiveReviewAttempt
    from ouroboros.review_records import ReviewActorRecord, ReviewSlot
    from ouroboros.review_state import (CommitAttemptRecord, ReservedReviewSlotRefused,
                                        checkpoint_pending_review_invocation, update_state)
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import utc_now_iso
    from tests.test_model_sleep import _ctx

    write_task_result(tmp_path, TASK, "running")  # a live root the owner can pause
    ctx = _ctx(tmp_path, TASK, root=TASK)
    operation = SimpleNamespace(wait=SimpleNamespace(tool_context=ctx), result_root=tmp_path, owner_id="panel",
                                author_detached=False)
    # A fresh local seat preparing its first launch (its episode never armed); in other_live_operation
    # this process's live seat is another physical operation than the reserved row's.
    entry = ActiveReviewAttempt(key="attempt-key", operation_id="op-2" if detach == "other_live_operation" else "op-1",
                                started_at=utc_now_iso(),
                                episode={"root_task_id": TASK, "operation": None, "armed_generation": None})
    monkeypatch.setattr(review_custody, "_ACTIVE", {} if detach == "not_live" else {entry.key: entry})
    if detach == "pause_preparation":
        owner_pause.install_fence(tmp_path, TASK, request_id="pause")
        assert review_pause.author_detaches(operation, [entry])
    else:
        _leave_wait(tmp_path, ctx, operation, entry, detach if detach in _WAIT_ONLY else "wake_at")
    actor = review_pause.detached_actor(
        ReviewSlot("second", "fake-review=fake-small"), entry,
        lambda slot, error, operation_id="", operation_state="settled": ReviewActorRecord(
            slot.slot_id, slot.model, "error", error=error, operation_id=operation_id, operation_state=operation_state),
        usage_ctx=ctx, request=SimpleNamespace(surface="multi_model_review", retry_key="key"), operation=operation)
    assert (actor.operation_state, actor.late_result_pending) == ("pending_dispatch", True)
    said = {"pause_preparation": "cannot send after Resume", "departed_author": "Nobody waits for this reviewer"}
    assert said.get(detach, "which still sends") in actor.error
    if detach in _WAIT_ONLY:
        assert actor.usage["author_left_for"] == _WAIT_ONLY[detach] and not entry.pause_before_launch
    if detach == "wake_at_then_pause":  # the owner pauses a later drain over the same live seat
        owner_pause.install_fence(tmp_path, TASK, request_id="pause")
        assert review_pause.author_detaches(operation, [entry]) and entry.pause_before_launch
    row = {"slot_id": "second", "operation_id": "op-0" if detach == "row_of_another_operation" else "op-1",
           "operation_state": actor.operation_state, "late_result_pending": True}
    update_state(tmp_path, lambda state: state.record_attempt(CommitAttemptRecord(
        ts="2026-10-16T00:00:00+00:00", commit_message="c", status="reviewing", task_id=TASK, root_task_id=TASK,
        tool_name="review_change", attempt=1, repo_key="repo", paid=True, review_retry_key="key",
        triad_raw_results=[row])))

    def checkpoint():
        checkpoint_pending_review_invocation(
            tmp_path, repo_key="repo", tool_name="review_change", task_id=TASK, attempt=1, review_retry_key="key",
            surface="multi_model_review", slot_id="second", operation_id="op-1", invocation_id="inv-1")
    if detach not in _WAIT_ONLY:
        refusal = "operation identity changed" if detach == "row_of_another_operation" else "not in flight"
        with pytest.raises(ReservedReviewSlotRefused, match=refusal):
            checkpoint()
        [saved] = load_state(tmp_path).attempts[-1].triad_raw_results
        assert saved["operation_state"] == "pending_dispatch" and not saved.get("pending_invocation_id"), saved
        return
    checkpoint()
    [saved] = load_state(tmp_path).attempts[-1].triad_raw_results
    assert (saved["operation_state"], saved["pending_invocation_id"], saved["late_result_pending"]) == (
        "in_flight", "inv-1", True)

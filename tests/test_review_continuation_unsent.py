"""A recorded continuation nothing sent yet is repaired as that same choice, within its recorded window.

After the one canonical choice, everything else is a projection. A failure there (here the owner's
real Pause refusing the successor's START at launch admission) answers ``decided_unsent`` naming the
recorded choice, and the repeat sends that operation under its recorded token. A START takes what
remains of the RECORDED window and goes out only from the call that wrote it and carried it into its
roster row: a choice no row carried settles unsent past that window (no collection grace sends it),
a START some earlier call wrote but never carried is retired and offered again (a new choice starts
it within what remains, never on its stale bytes), and settling unsent and carrying meet on that row,
so two callers straddling the window end never leave an unsent seat with a sent run. An UNKNOWN send,
whose row carried it, replays its exact key and bytes. An identical reissue of the round, the result delivery's collection and a
commit's explicit collection reach the recorded choice before any cached roster, so none of
them mints a competing wave. Only the engine and the API seat's model are scripted; time moves by
patching ``deadline_utils.utc_now``.
"""
from __future__ import annotations

import json
import math
import time
from datetime import timedelta

import pytest

from ouroboros import delegate_custody, review_continuation
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.observability import read_blob_ref
from ouroboros.review_ledger import ledger_root, load_record
from ouroboros.review_state import load_state
from tests.test_review_explicit_continuation import _call, _commit_call, _seat, _wave
from tests.test_review_explicit_continuation import author as _author, fake_route as _fake_route
from tests.test_review_explicit_continuation import stopped_wave as _stopped_wave
from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport  # noqa: F401
from tests.test_receptive_commit_review import TASK, _git

author, fake_route, stopped_wave = _author, _fake_route, _stopped_wave  # fixtures by name


def _entered_at(monkeypatch):
    """When the executor enters each session run, by operation id (``time.monotonic``): it read its window no later,
    and a window only shrinks, so the time up to that instant bounds what a slow host can have taken from it."""
    from ouroboros import review_execution

    entered, run = {}, review_execution.run_delegated_review_session
    monkeypatch.setattr(review_execution, "run_delegated_review_session", lambda *, invocation, **kw: (
        entered.setdefault(invocation.operation_id, time.monotonic()), run(invocation=invocation, **kw))[1])
    return entered


def _timed_wave(a, monkeypatch):
    """The wave whose session seat runs with a real window (its START carries ``maxSeconds``)."""
    monkeypatch.setenv("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", "600")
    entered, began = _entered_at(monkeypatch), time.monotonic()
    first = _wave(a)
    lost = math.ceil(max(entered.values()) - began)  # the START carries what remained of the 600 s, rounded up
    assert 600 - lost <= a.engine.posts[0][0]["maxSeconds"] <= 600, a.engine.posts[0][0]
    return first


def _choice(first):
    return [{"seat_id": "second", "operation_id": first["continuable_seats"][0]["operation_id"]}]


def _recorded_then_crashed(a, monkeypatch, first):
    """The choice lands in the ledger; the process ends before its START (a failure after the choice)."""
    real = review_continuation._start_requested
    monkeypatch.setattr(review_continuation, "_start_requested", lambda *a_, **k: (_ for _ in ()).throw(
        OSError("process ended after the choice")))
    unsent = _call(a, review_reference=first["review_reference"], continue_seats=_choice(first))
    monkeypatch.setattr(review_continuation, "_start_requested", real)
    decision = _seat(a, first["record_id"])["continuations"][0]
    assert unsent["status"] == "decided_unsent" and unsent["recorded"][0]["operation_id"] == decision["operation_id"]
    assert delegate_custody.invocation_record(a.data, decision["invocation_id"]) is None and len(a.engine.posts) == 1
    return decision


def test_an_owner_pause_refusing_the_successor_start_leaves_the_choice_recorded_and_its_repeat_sends_it(
        stopped_wave, monkeypatch):
    from ouroboros import owner_pause

    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    reference, record_id = first["review_reference"], first["record_id"]
    real = review_continuation._record_choice

    def paused_after_choice(*args):  # the owner's Pause lands between the recorded choice and its START
        recorded = real(*args)
        owner_pause.install_fence(a.data, TASK, request_id="pause-after-choice")
        return recorded
    monkeypatch.setattr(review_continuation, "_record_choice", paused_after_choice)
    try:
        unsent = _call(a, review_reference=reference, continue_seats=_choice(first))
    finally:
        monkeypatch.setattr(review_continuation, "_record_choice", real)
        owner_pause.release_fence(a.data, TASK, reason="owner_resume")
    decision = _seat(a, record_id)["continuations"][0]
    assert unsent["status"] == "decided_unsent", unsent
    assert unsent["recorded"] == [{"seat_id": "second", "operation_id": decision["operation_id"],
                                   "continues": _choice(first)[0]["operation_id"]}]
    assert "nothing was sent" in unsent["consequence"] and "Nothing was decided" not in unsent["consequence"]
    assert delegate_custody.invocation_record(a.data, decision["invocation_id"]) is None and len(a.engine.posts) == 1

    sent = _call(a, review_reference=reference, continue_seats=_choice(first))  # the same call, after Resume
    assert sent["state"] == "settled" and sent["continuation"]["continued"][0]["rejoined"] is True, sent
    (_original, _k), (body, key) = a.engine.posts
    recorded = read_blob_ref(a.data, decision["request_ref"], expected_kind="json")
    assert key == decision["invocation_id"] and body["continueFrom"] == "run-1"
    assert {k: v for k, v in body.items() if k != "maxSeconds"} == {k: v for k, v in recorded.items() if k != "maxSeconds"}
    assert body["maxSeconds"] <= recorded["maxSeconds"]
    seat = _seat(a, record_id)
    assert len(seat["continuations"]) == 1 and seat["status"] == "responded"  # one choice, decided once


def test_a_recorded_choice_delayed_past_its_original_window_is_never_sent(stopped_wave, monkeypatch):
    from ouroboros import deadline_utils

    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    reference, record_id = first["review_reference"], first["record_id"]
    decision = _recorded_then_crashed(a, monkeypatch, first)
    assert decision["window_end"], "the stopped run's original window is recorded with the choice"
    later = parse_deadline_ts(decision["window_end"]) + timedelta(seconds=5)
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: later)  # the original window ended meanwhile
    collected = _call(a, review_reference=reference)  # a plain collection: the repair, under its settlement grace
    seat = _seat(a, record_id)
    assert collected["state"] == "settled" and collected["continuation"]["window_spent"] == ["second"], collected
    assert (seat["status"], seat["operation_state"], seat["failure_code"]) == (
        "not_dispatched", "not_dispatched", review_continuation.WINDOW_SPENT)
    assert delegate_custody.invocation_record(a.data, decision["invocation_id"]) is None and len(a.engine.posts) == 1
    assert review_continuation.continuable_seats(load_record(ledger_root(a.ctx), record_id)) == []  # offered no more
    again = _call(a, review_reference=reference, continue_seats=_choice(first))  # the same choice: nothing to send
    assert again["continuation"]["continued"][0]["window_spent"] is True and len(a.engine.posts) == 1


def test_a_recorded_choice_sent_later_gets_only_what_remains_of_its_window(stopped_wave, monkeypatch):
    from ouroboros import deadline_utils

    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    decision = _recorded_then_crashed(a, monkeypatch, first)
    recorded = read_blob_ref(a.data, decision["request_ref"], expected_kind="json")
    end = parse_deadline_ts(decision["window_end"])
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: end - timedelta(seconds=30))
    assert recorded["maxSeconds"] > 30, recorded  # the choice was recorded with more time than now remains
    collected = _call(a, review_reference=first["review_reference"])
    assert collected["state"] == "settled" and collected["continuation"]["repaired"] == ["second"], collected
    (_original, _k), (body, key) = a.engine.posts
    assert key == decision["invocation_id"] and body["maxSeconds"] <= 30  # never more than the window has left
    assert delegate_custody.invocation_record(a.data, key)["request"]["maxSeconds"] == body["maxSeconds"]


def _start_recorded_then_crashed(a, monkeypatch, first):
    """The START lands, the process ends before the roster row names the successor: recorded, provably unsent
    (collectors send only what a roster row carries)."""
    real = review_continuation.continued_row
    monkeypatch.setattr(review_continuation, "continued_row", lambda *a_, **k: (_ for _ in ()).throw(
        OSError("process ended after the START, before the roster row")))
    unsent = _call(a, review_reference=first["review_reference"], continue_seats=_choice(first))
    monkeypatch.setattr(review_continuation, "continued_row", real)
    decision = _seat(a, first["record_id"])["continuations"][0]
    start = delegate_custody.invocation_record(a.data, decision["invocation_id"])
    assert unsent["status"] == "decided_unsent" and start and len(a.engine.posts) == 1, unsent
    row = next(row for row in load_state(a.data).attempts[-1].triad_raw_results if row["slot_id"] == "second")
    assert row["operation_id"] == decision["supersedes"]
    return decision, start


def test_a_recorded_start_its_roster_never_carried_is_not_first_sent_after_its_window(stopped_wave, monkeypatch):
    from ouroboros import deadline_utils

    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    reference, record_id = first["review_reference"], first["record_id"]
    decision, _start = _start_recorded_then_crashed(a, monkeypatch, first)
    later = parse_deadline_ts(decision["window_end"]) + timedelta(seconds=5)
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: later)  # the original window ended meanwhile
    collected = _call(a, review_reference=reference)  # no settlement grace turns this into a first send
    seat = _seat(a, record_id)
    assert len(a.engine.posts) == 1, a.engine.posts[1:]
    assert collected["state"] == "settled" and collected["continuation"]["window_spent"] == ["second"], collected
    assert (seat["status"], seat["operation_state"], seat["failure_code"]) == (
        "not_dispatched", "not_dispatched", review_continuation.WINDOW_SPENT)
    assert review_continuation.continuable_seats(load_record(ledger_root(a.ctx), record_id)) == []
    again = _call(a, review_reference=reference, continue_seats=_choice(first))  # the same choice: nothing to send
    assert again["continuation"]["continued"][0]["window_spent"] is True and len(a.engine.posts) == 1
    assert _call(a, review_reference=reference)["state"] == "settled" and len(a.engine.posts) == 1
    # The never-sent START is retired as a definite zero-send: no collector, recovery or sweep can send it later.
    assert delegate_custody.invocation_record(a.data, decision["invocation_id"])["state"] == "failed_definite"
    assert decision["invocation_id"] not in {row["invocation_id"] for row in delegate_custody.pending_invocations(a.data)}


def _shifted_clock(monkeypatch, seconds):
    """One wall clock moved forward by ``seconds``: the window check, custody's rejoin window and the timestamp a
    START is written with (so a START written after the shift is as old as it really is)."""
    from ouroboros import deadline_utils, review_custody, utils

    real, holder = deadline_utils.utc_now, {"offset": timedelta(seconds=seconds)}
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: real() + holder["offset"])
    monkeypatch.setattr(review_custody, "utc_now", lambda: real() + holder["offset"])
    monkeypatch.setattr(utils, "utc_now_iso", lambda: (real() + holder["offset"]).isoformat())
    return holder


def _host_windows(monkeypatch):
    """The window the executor runs each successor under, which it cancels at (``timeout_sec``)."""
    from ouroboros import review_execution

    seen, run = {}, review_execution.run_delegated_review_session
    monkeypatch.setattr(review_execution, "run_delegated_review_session", lambda *, invocation, **kw: (
        seen.setdefault(invocation.operation_id, invocation.timeout_sec), run(invocation=invocation, **kw))[1])
    return seen


@pytest.mark.parametrize("left", [100, 10])
def test_a_recorded_start_its_roster_never_carried_is_retired_and_a_new_choice_gets_only_what_remains(
        stopped_wave, monkeypatch, left):
    """Its bytes were decided by an earlier call (maxSeconds then ~600): sending them now would let the engine run
    past the window, and below the settlement margin the host would too. So nothing sends them; the offer names
    the successor, and the explicit new choice starts with what remains, on the engine and on the host."""
    from ouroboros import deadline_utils

    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    reference, record_id = first["review_reference"], first["record_id"]
    decision, start = _start_recorded_then_crashed(a, monkeypatch, first)
    assert start["max_seconds"] > 500
    _shifted_clock(monkeypatch, (parse_deadline_ts(decision["window_end"]) - deadline_utils.utc_now()).total_seconds() - left)
    windows = _host_windows(monkeypatch)
    collected = _call(a, review_reference=reference)  # a plain collection after much of the window went by
    seat = _seat(a, record_id)
    assert len(a.engine.posts) == 1 and collected["continuation"]["not_sent"] == ["second"], collected
    assert (seat["status"], seat["failure_code"]) == ("not_dispatched", review_continuation.START_UNSENT)
    assert delegate_custody.invocation_record(a.data, decision["invocation_id"])["state"] == "failed_definite"
    [offer] = review_continuation.continuable_seats(load_record(ledger_root(a.ctx), record_id))
    assert offer["operation_id"] == decision["operation_id"] and offer["continues_run"] == "run-1"
    again = _call(a, review_reference=reference, continue_seats=[{"seat_id": "second", "operation_id": offer["operation_id"]}])
    assert again["state"] == "settled" and _seat(a, record_id)["status"] == "responded", again
    (_original, _k), (body, key) = a.engine.posts
    successor = _seat(a, record_id)["continuations"][-1]
    assert key == successor["invocation_id"] != decision["invocation_id"] and body["continueFrom"] == "run-1"
    assert body["maxSeconds"] <= left and min(body["maxSeconds"], windows[successor["operation_id"]]) <= left


@pytest.mark.parametrize("start_recorded", [False, True])
def test_two_collections_straddling_the_window_end_never_leave_an_unsent_seat_with_a_sent_run(
        stopped_wave, monkeypatch, start_recorded):
    """A reads the window just before its end and is held before its roster row; B runs just after the end. Before
    the row arbitrated them, B settled the seat unsent and A then posted it (a started run the ledger called never
    sent). Now the row decides: B claims it unsent, and A, whose window closed while it was held, writes no START
    and sends nothing."""
    import threading

    from ouroboros import deadline_utils
    from ouroboros.tools.review_change import _handle_review_change
    from tests.test_receptive_commit_review import _parent

    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    reference, record_id = first["review_reference"], first["record_id"]
    decision = (_start_recorded_then_crashed(a, monkeypatch, first)[0] if start_recorded
                else _recorded_then_crashed(a, monkeypatch, first))
    clock = _shifted_clock(monkeypatch, (parse_deadline_ts(decision["window_end"]) - deadline_utils.utc_now()).total_seconds() - 5)
    checked, release, outcome, transfer = threading.Event(), threading.Event(), {}, review_continuation.transfer_checking_copy

    def held(*args, **kw):  # the projection's next step after its window check
        if threading.current_thread().name == "collector-A":
            checked.set()
            release.wait(60)
        return transfer(*args, **kw)
    monkeypatch.setattr(review_continuation, "transfer_checking_copy", held)

    def run_a():
        with _parent(a):
            outcome["A"] = _handle_review_change(a.ctx, review_reference=reference)
    collector = threading.Thread(target=run_a, name="collector-A", daemon=True)
    collector.start()
    if start_recorded:  # a START an earlier call wrote: A treats it as never sent too, and claims the row
        collector.join(60)
        assert not checked.is_set() and len(a.engine.posts) == 1
        seat = _seat(a, record_id)
        assert (seat["status"], seat["failure_code"]) == ("not_dispatched", review_continuation.START_UNSENT)
        return
    assert checked.wait(60), "A never reached its projection"
    clock["offset"] += timedelta(seconds=10)  # now 5 s past the window end
    b = _call(a, review_reference=reference)
    release.set()
    collector.join(60)
    assert not collector.is_alive()
    seat = _seat(a, record_id)
    assert b["continuation"]["window_spent"] == ["second"] and len(a.engine.posts) == 1, a.engine.posts[1:]
    assert (seat["status"], seat["failure_code"]) == ("not_dispatched", review_continuation.WINDOW_SPENT)
    assert json.loads(outcome["A"])["state"] == "settled"
    assert delegate_custody.invocation_record(a.data, decision["invocation_id"]) is None  # A's window closed first: no START


def _fence_scope(a, fence):
    """The task's usage scope with a per-task fence of ``fence`` USD, as the loop binds it."""
    from ouroboros import usage_accounting as ua

    return ua.UsageScope(drive_root=a.data, task_id=TASK, root_task_id=TASK, root_limit_usd=fence)


def _settle_known_spend(a, usd):
    """Other work of the task settles ``usd`` of known spend in the actual usage ledger."""
    from ouroboros import usage_accounting as ua

    with ua.usage_scope(_fence_scope(a, usd)):
        row = ua.reserve_attempt(ua.AttemptRequest(model="test/m1", provider="test", source="main"))
        ua.mark_dispatched(row)
        ua.settle_attempt(row, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=usd, cost_final=True)


def _collect_at_fence(a, fence, **kwargs):
    """The author's ``review_change`` call under the task's usage scope with that fence."""
    import threading

    from ouroboros import usage_accounting as ua
    from ouroboros.tools.review_change import _handle_review_change
    from tests.test_receptive_commit_review import _parent

    out = {}

    def run():
        with _parent(a), ua.usage_scope(_fence_scope(a, fence)):
            out["text"] = _handle_review_change(a.ctx, **kwargs)
    thread = threading.Thread(target=run)
    thread.start()
    thread.join(60)
    return json.loads(out["text"])


def test_a_recorded_choice_repaired_at_a_reached_budget_fence_sends_nothing_until_there_is_room(stopped_wave, monkeypatch):
    """The choice was recorded before its START (a crash); by the repair the task's known spend has reached its fence.
    The START it would now write is new generation: the repair leaves the choice recorded and sends nothing."""
    from ouroboros import usage_accounting as ua

    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    decision = _recorded_then_crashed(a, monkeypatch, first)
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 0.5)  # the price catalog only
    _settle_known_spend(a, 3.0)
    held = _collect_at_fence(a, 3.0, review_reference=first["review_reference"])
    assert held["status"] == "decided_unsent" and held["reason"] == "budget_insufficient" and len(a.engine.posts) == 1, held
    assert delegate_custody.invocation_record(a.data, decision["invocation_id"]) is None  # no START was written
    sent = _collect_at_fence(a, 30.0, review_reference=first["review_reference"])
    assert sent["state"] == "settled" and sent["continuation"]["repaired"] == ["second"] and len(a.engine.posts) == 2, sent


def _carried_never_posted(a, monkeypatch, first, *, accepted=False):
    """The choice is projected (START written, roster row carries it) and its call ended before any POST; with
    ``accepted`` the engine did take a POST of those bytes whose reply was lost. Clocks are then set relative to
    the end of the window that START recorded."""
    from ouroboros import deadline_utils
    from ouroboros.tools.review_collection import continue_or_repair, resolve_reference
    from tests._review_session_route_shared import FakeGateway

    root, attempt, record = resolve_reference(a.ctx, first["review_reference"])
    continue_or_repair(a.ctx, root, attempt, record, _choice(first))  # carried: from now on it may have been sent
    decision = _seat(a, first["record_id"])["continuations"][0]
    start = delegate_custody.invocation_record(a.data, decision["invocation_id"])
    if accepted:
        FakeGateway().start_run(start["request"], idempotency_key=decision["invocation_id"])
    delegate_custody._CUSTODY.clear()
    end = _original_end(a)

    def at(seconds_from_end):
        return _shifted_clock(monkeypatch, (end - deadline_utils.utc_now()).total_seconds() + seconds_from_end)
    return decision, start, at


def _original_end(a):
    """The instant the stopped seat's ORIGINAL window ends: its first run's start plus that run's cap."""
    started, cap = delegate_custody.run_timing(a.data, "run-1")
    return parse_deadline_ts(started) + timedelta(seconds=cap)


def _left_unknown(a, first, decision, start):
    """Nothing POSTed it now (its window ended, or known spend reached a fence), and nothing proves no earlier call
    did: it stays UNKNOWN, never a definite absence, with its key and exact bytes kept."""
    seat = _seat(a, first["record_id"])
    left = delegate_custody.invocation_record(a.data, decision["invocation_id"])
    assert seat["operation_id"] == decision["operation_id"] and seat["operation_state"] == "custody_lost", seat
    assert left["state"] == "pending" and left["request"] == start["request"]  # its key and exact bytes are kept


@pytest.mark.parametrize("when", ["inside", "after"])
def test_an_unknown_start_replays_its_exact_key_and_bytes_only_while_its_window_lasts(stopped_wave, monkeypatch, when):
    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    decision, start, at = _carried_never_posted(a, monkeypatch, first, accepted=True)
    at(-100 if when == "inside" else 5)
    collected = _call(a, review_reference=first["review_reference"])
    if when == "after":
        assert len(a.engine.posts) == 2, a.engine.posts[2:]  # the original wave and the accepted POST only
        _left_unknown(a, first, decision, start)
        return
    assert collected["state"] == "settled" and _seat(a, first["record_id"])["status"] == "responded", collected
    (_first, _k), (lost, lost_key), (replayed, replay_key) = a.engine.posts
    assert replay_key == lost_key == decision["invocation_id"] and replayed == lost == start["request"]


def test_a_carried_start_that_was_never_posted_is_not_first_posted_after_its_window(stopped_wave, monkeypatch):
    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    decision, start, at = _carried_never_posted(a, monkeypatch, first)
    at(5)
    _call(a, review_reference=first["review_reference"])
    assert len(a.engine.posts) == 1, a.engine.posts[1:]
    _left_unknown(a, first, decision, start)


def _launch_delayed_past(monkeypatch, end, *, crosses=True):
    """The existing launcher seam of the POST (``review_execution.run_operation``), after the window was first read
    and the gateway, START and checkpoint steps ran: with ``crosses`` it waits until the window has ended."""
    import time

    from ouroboros import review_custody, review_execution

    launched, launch = [], review_execution.run_operation

    def delayed(source, function, *args, **kwargs):
        launched.append(review_custody.utc_now())
        while crosses and review_custody.utc_now() <= end + timedelta(milliseconds=50):
            time.sleep(0.02)
        return launch(source, function, *args, **kwargs)
    monkeypatch.setattr(review_execution, "run_operation", delayed)
    return launched


@pytest.mark.parametrize("crosses", [True, False])
def test_a_launch_delay_after_the_window_check_never_carries_a_first_post_past_the_window(stopped_wave, monkeypatch, crosses):
    """The recovery reads the window before its gateway, START and checkpoint steps and the launch admission; a delay
    there that crosses the end is refused inside the send callback itself (UNKNOWN, key and bytes kept, no POST).
    A delay that stays inside sends the same key and bytes."""
    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    decision, start, at = _carried_never_posted(a, monkeypatch, first)
    end = _original_end(a)
    at(-3)  # three seconds of the window left when the recovery reads it
    launched = _launch_delayed_past(monkeypatch, end, crosses=crosses)
    _call(a, review_reference=first["review_reference"])
    assert launched and launched[0] < end, launched  # the send was prepared while the window was open
    if crosses:
        assert len(a.engine.posts) == 1, a.engine.posts[1:]
        _left_unknown(a, first, decision, start)
        return
    (_original, _k), (body, key) = a.engine.posts
    assert key == decision["invocation_id"] and body == start["request"]
    assert _seat(a, first["record_id"])["status"] == "responded"


def test_a_carried_start_with_less_left_than_the_settlement_margin_generates_only_for_what_remains(stopped_wave, monkeypatch):
    """Its exact bytes still carry their earlier cap (~600 s) and custody still grants a rejoin its 30 s settlement
    margin, but the executor stops the successor generating where the window ends; the margin is for receipts."""
    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    decision, start, at = _carried_never_posted(a, monkeypatch, first)
    at(-10)
    entered = _entered_at(monkeypatch)
    granted, polled = _host_windows(monkeypatch), _polled(monkeypatch)
    began = time.monotonic()
    collected = _call(a, review_reference=first["review_reference"])
    assert collected["state"] == "settled" and _seat(a, first["record_id"])["status"] == "responded", collected
    (_first, _k), (body, key) = a.engine.posts
    assert key == decision["invocation_id"] and body == start["request"] and body["maxSeconds"] > 500  # bytes unchanged
    lost = math.ceil(entered[decision["operation_id"]] - began)  # the 30 s grant shrinks only until the executor reads it
    assert granted[decision["operation_id"]] >= 30 - lost and polled["run-2"] <= 10, (granted, polled)


def _polled(monkeypatch):
    """The window the executor polls each run for before it cancels it (``_poll_session_terminal``)."""
    from ouroboros import review_execution

    polled, poll = {}, review_execution._poll_session_terminal
    monkeypatch.setattr(review_execution, "_poll_session_terminal", lambda gateway, custody, drive, entry, run_id, seconds: (
        polled.setdefault(run_id, seconds), poll(gateway, custody, drive, entry, run_id, seconds))[1])
    return polled


def _preparation_takes(monkeypatch, clock, seconds):
    """One step of the projection after it read the window and before the START is written (the checking copy's
    transfer, standing for the admission read and project registration beside it) takes ``seconds``."""
    transfer = review_continuation.transfer_checking_copy

    def slow(*args, **kwargs):
        clock["offset"] += timedelta(seconds=seconds)
        return transfer(*args, **kwargs)
    monkeypatch.setattr(review_continuation, "transfer_checking_copy", slow)


def _start_persisted_late(monkeypatch, clock, seconds):
    """The successor START's own persistence (its body and billing binding, before the row is stamped) takes
    ``seconds``, once: after the cap was read from the window, before the START's timestamp."""
    record, done = delegate_custody.record_start_requested, []

    def slow(*args, **kwargs):
        if not done:
            done.append(True)
            clock["offset"] += timedelta(seconds=seconds)
        return record(*args, **kwargs)
    monkeypatch.setattr(delegate_custody, "record_start_requested", slow)


@pytest.mark.parametrize("delay", ["crosses_before_the_start", "crosses_after_the_start", "inside"])
def test_a_preparation_delay_before_the_start_is_written_never_moves_the_original_window_end(
        stopped_wave, monkeypatch, delay):
    """The projection reads the window, then its steps before the START is stamped take time. The successor still
    ends where the original window ends: through its START, the send callback and the host's polling. A delay crossing
    that end sends nothing (before the START: a definite zero-send, offered no more; after it, here with the START
    stamped 5 s after its cap was read: UNKNOWN, key and bytes kept); inside it, the cap sent and the window polled
    are only what remains then."""
    from ouroboros import deadline_utils

    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    end, record_id = _original_end(a), first["record_id"]
    left, takes, stamped_after = {"crosses_before_the_start": (10, 15, 0), "crosses_after_the_start": (13, 5, 5),
                                  "inside": (13, 5, 0)}[delay]
    clock = _shifted_clock(monkeypatch, (end - deadline_utils.utc_now()).total_seconds() - left)
    _preparation_takes(monkeypatch, clock, takes)
    _start_persisted_late(monkeypatch, clock, stamped_after)
    launched, polled = _launch_delayed_past(monkeypatch, end, crosses=delay == "crosses_after_the_start"), _polled(monkeypatch)
    chosen = _call(a, review_reference=first["review_reference"], continue_seats=_choice(first))
    decision = _seat(a, record_id)["continuations"][0]
    start = delegate_custody.invocation_record(a.data, decision["invocation_id"])
    if delay == "crosses_before_the_start":
        assert len(a.engine.posts) == 1 and not launched, a.engine.posts[1:]
        assert chosen["continuation"]["continued"][0]["window_spent"] is True, chosen
        assert start is None and (_seat(a, record_id)["status"], _seat(a, record_id)["failure_code"]) == (
            "not_dispatched", review_continuation.WINDOW_SPENT)
        assert review_continuation.continuable_seats(load_record(ledger_root(a.ctx), record_id)) == []
        return
    if delay == "crosses_after_the_start":
        assert launched and launched[0] < end and len(a.engine.posts) == 1, a.engine.posts[1:]
        _left_unknown(a, first, decision, start)
    else:
        (_original, _k), (body, key) = a.engine.posts
        assert key == decision["invocation_id"] and body == start["request"] and _seat(a, record_id)["status"] == "responded"
        assert polled["run-2"] <= left - takes, polled  # the host stops it generating where the original window ends
    assert start["max_seconds"] <= left - takes, start  # what remains when the START is written, never the earlier read


def test_a_window_closing_while_the_successor_project_registers_retires_it_and_writes_no_start(stopped_wave, monkeypatch):
    """A successor whose recorded scope names a registered project registers it before its START is written, and the
    window closes during that registration: no START will ever name the registration it created, so it is retired
    again, and the seat settles as never sent (a definite zero-send, offered no more)."""
    from ouroboros import deadline_utils, observability
    from tests._review_session_route_shared import FakeGateway

    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    decision = _recorded_then_crashed(a, monkeypatch, first)
    clock = _shifted_clock(monkeypatch, (_original_end(a) - deadline_utils.utc_now()).total_seconds() - 10)
    read, register, removed = observability.read_blob_ref, FakeGateway.register_project, []

    def registered_scope(*args, **kwargs):  # the recorded body as a review of a registered project carries it
        body = read(*args, **kwargs)
        if isinstance(body, dict) and isinstance(body.get("scope"), dict):
            body["scope"] = {key: value for key, value in body["scope"].items() if key != "ephemeral"}
        return body

    def slow_registration(self, root):  # the engine takes 15 s to register it
        clock["offset"] += timedelta(seconds=15)
        return register(self, root)
    monkeypatch.setattr(observability, "read_blob_ref", registered_scope)
    monkeypatch.setattr(FakeGateway, "project_unregistered", True)
    monkeypatch.setattr(FakeGateway, "register_project", slow_registration)
    monkeypatch.setattr(FakeGateway, "remove_project", lambda self, project_id: removed.append(project_id) or {"removed": True})
    collected = _call(a, review_reference=first["review_reference"])
    seat = _seat(a, first["record_id"])
    assert len(a.engine.posts) == 1 and removed == ["proj-new"], (a.engine.posts[1:], removed)
    assert delegate_custody.invocation_record(a.data, decision["invocation_id"]) is None
    assert (seat["status"], seat["failure_code"]) == ("not_dispatched", review_continuation.WINDOW_SPENT), seat
    assert collected["continuation"]["window_spent"] == ["second"], collected


def test_an_owner_deadline_inside_the_window_bounds_the_delayed_successor_start_and_its_polling(stopped_wave, monkeypatch):
    """The owner's current deadline, less the finalization reserve, ends before the original window: that narrower
    instant is the successor's end, and a delay before its START takes from it instead of adding to it."""
    from ouroboros import deadline_utils
    from ouroboros.config import get_finalization_grace_sec

    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    clock = _shifted_clock(monkeypatch, 0)
    owner_end = deadline_utils.utc_now() + timedelta(seconds=get_finalization_grace_sec() + 20)
    a.ctx.task_metadata = {**dict(a.ctx.task_metadata or {}), "deadline_at": owner_end.isoformat()}
    _preparation_takes(monkeypatch, clock, 8)
    polled = _polled(monkeypatch)
    _call(a, review_reference=first["review_reference"], continue_seats=_choice(first))
    decision = _seat(a, first["record_id"])["continuations"][0]
    (_original, _k), (body, key) = a.engine.posts
    assert decision["max_seconds"] <= 20 and key == decision["invocation_id"], decision  # the choice read the owner's bound
    assert body["maxSeconds"] <= 12 and polled["run-2"] <= 12, (body["maxSeconds"], polled)
    assert _seat(a, first["record_id"])["status"] == "responded"


@pytest.mark.parametrize("fence", [3.0, 30.0])
def test_a_carried_start_never_posted_is_first_posted_only_while_known_spend_is_below_its_fence(
        stopped_wave, monkeypatch, fence):
    """A carried START may never have been POSTed (its call ended first), so its replay can be the first paid POST; the
    send callback admits it on known spend as every review wave is admitted. With known spend at the fence nothing is
    POSTed and its outcome stays UNKNOWN, key and exact bytes kept; with room the same key sends the same bytes."""
    from ouroboros import usage_accounting as ua

    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 0.5)  # the price catalog only
    decision, start, at = _carried_never_posted(a, monkeypatch, first)
    at(-100)  # well inside its window
    _settle_known_spend(a, 3.0)
    _collect_at_fence(a, fence, review_reference=first["review_reference"])
    if fence == 3.0:
        assert len(a.engine.posts) == 1, a.engine.posts[1:]
        _left_unknown(a, first, decision, start)
        _collect_at_fence(a, 30.0, review_reference=first["review_reference"])  # room later: UNKNOWN is never re-posted
        assert len(a.engine.posts) == 1, a.engine.posts[1:]
        _left_unknown(a, first, decision, start)
        return
    (_original, _k), (body, key) = a.engine.posts
    assert key == decision["invocation_id"] and body == start["request"] and _seat(a, first["record_id"])["status"] == "responded"


def test_known_spend_reaching_its_fence_during_the_launch_steps_stops_the_post_inside_the_send_callback(
        stopped_wave, monkeypatch):
    """Admission is read in the send callback itself: spend another call settles while this recovery's gateway, START,
    checkpoint and launch steps run still stops its POST."""
    import threading

    from ouroboros import review_execution
    from ouroboros import usage_accounting as ua

    a = stopped_wave
    first = _timed_wave(a, monkeypatch)
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 0.5)
    decision, start, at = _carried_never_posted(a, monkeypatch, first)
    at(-100)
    launch, launched = review_execution.run_operation, []

    def settled_meanwhile(source, function, *args, **kwargs):
        if not launched:  # the task's other work, on its own thread, reaches the fence before this POST
            other = threading.Thread(target=_settle_known_spend, args=(a, 3.0))
            other.start()
            other.join(30)
        launched.append(True)  # the substrate's one retry of a refused send meets the same fence
        return launch(source, function, *args, **kwargs)
    monkeypatch.setattr(review_execution, "run_operation", settled_meanwhile)
    _collect_at_fence(a, 3.0, review_reference=first["review_reference"])
    assert launched and len(a.engine.posts) == 1, a.engine.posts[1:]
    _left_unknown(a, first, decision, start)


def test_an_identical_reissue_after_a_crash_after_the_choice_sends_that_choice_never_a_competing_wave(
        stopped_wave, monkeypatch):
    a = stopped_wave
    first = _wave(a)
    record_id = first["record_id"]
    attempts = len(load_state(a.data).attempts)
    decision = _recorded_then_crashed(a, monkeypatch, first)
    again = _wave(a)  # the same unchanged call, no reference
    assert again["record_id"] == record_id and again["state"] == "settled", again
    assert [key for _body, key in a.engine.posts][1:] == [decision["invocation_id"]]  # the recorded token, once
    assert len(load_state(a.data).attempts) == attempts  # no new attempt row, no second wave
    seat = _seat(a, record_id)
    assert seat["status"] == "responded" and len(seat["continuations"]) == 1


def test_a_commit_offers_its_stopped_seat_and_the_landing_after_the_explicit_choice_runs_its_own_gate(
        stopped_wave, monkeypatch):
    from ouroboros.tools import git

    a = stopped_wave
    head = _git(a.repo, "rev-parse", "HEAD")
    gate_tests = []
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *a_, **k: gate_tests.append(
        _git(a.repo, "rev-parse", "HEAD")) and None)
    returned = _commit_call(a)
    assert "continuable_seats" in returned and _git(a.repo, "rev-parse", "HEAD") == head, returned
    handback = json.JSONDecoder().raw_decode(returned, returned.index("{\"review_reference\""))[0]
    reference, offer = handback["review_reference"], handback["continuable_seats"]
    assert [item["seat_id"] for item in offer] == ["second"] and len(a.engine.posts) == 1
    _git(a.repo, "add", "change.py")  # the handback unstaged the candidate; the same reviewed bytes are staged again
    continued = _commit_call(a, review_reference=reference, continue_seats=[
        {"seat_id": "second", "operation_id": offer[0]["operation_id"]}])
    assert len(a.engine.posts) == 2 and a.engine.posts[1][0]["continueFrom"] == "run-1", continued
    assert _seat(a, reference["review_record_id"])["status"] == "responded"
    assert _git(a.repo, "rev-parse", "HEAD") == head and "Review outcome returned before commit" in continued, continued
    _git(a.repo, "add", "change.py")
    tested_before = len(gate_tests)
    landing = _commit_call(a, review_reference=json.JSONDecoder().raw_decode(
        continued, continued.index("{\"review_reference\""))[0]["review_reference"],
        author_disposition={"disposition": "accepted", "rationale": "test"})
    landed = _git(a.repo, "rev-parse", "HEAD")
    assert landed != head and _git(a.repo, "rev-list", "--count", f"{head}..{landed}") == "1", landing
    assert len(gate_tests) > tested_before and gate_tests[-1] == head  # the landing call's gate tests ran before Git
    assert len(a.engine.posts) == 2  # the landing buys no reviewer


def test_the_result_delivery_sends_a_choice_a_crash_left_unprojected_and_buys_no_other_wave(tmp_path, monkeypatch,
                                                                                             fake_route):
    from ouroboros import review_ledger
    from ouroboros.tools.review_change import _handle_review_change, run_review_change
    from ouroboros.tools.review_result import result_review_at_delivery
    from tests._review_session_route_shared import FakeGateway, _terminal_detail
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
    from tests.test_result_review import _floor_ctx, _produced_answer_protocol

    ctx = _floor_ctx(tmp_path, monkeypatch)  # Required: with no review running, the host's floor would buy one
    _produced_answer_protocol(monkeypatch)
    set_review_pool(monkeypatch, pool_roster(pool_seat("independent", "fake-review=fake-small", kind="agent_session",
                                                       access="readonly")))
    posts, runs = [], {}

    def start(self, request, *, idempotency_key=""):
        self.start_requests.append(dict(request))
        posts.append((dict(request), str(idempotency_key)))
        return {"runId": runs.setdefault(str(idempotency_key), f"run-{len(runs) + 1}")}

    def detail(self, run_id, **_kw):
        if run_id != "run-1":
            return _terminal_detail('{"findings": []}', conformance="passed")
        stopped = _terminal_detail("PARTIAL: half of the evidence read", state="failed")
        stopped["summary"].update(failure={"code": "credential_pool_exhausted", "safeMessage": "Account limit reached"},
                                  resumable={"cause": "pool_exhausted", "carriers": ["native"]})
        return stopped
    monkeypatch.setattr(FakeGateway, "start_run", start)
    monkeypatch.setattr(FakeGateway, "get_run", detail)
    first = run_review_change(ctx, subject="result", answer="FINAL ANSWER: 2", goal="Check the arithmetic")
    choice = [{"seat_id": "independent", "operation_id": first["continuable_seats"][0]["operation_id"]}]
    real = review_continuation._start_requested
    monkeypatch.setattr(review_continuation, "_start_requested", lambda *a_, **k: (_ for _ in ()).throw(
        OSError("process ended after the choice")))
    unsent = json.loads(_handle_review_change(ctx, review_reference=first["review_reference"], continue_seats=choice))
    monkeypatch.setattr(review_continuation, "_start_requested", real)
    assert unsent["status"] == "decided_unsent" and len(posts) == 1, unsent
    token = {row["seat_id"]: row for row in load_record(ledger_root(ctx), first["record_id"])["rows"]}[
        "independent"]["continuations"][0]["invocation_id"]
    records = review_ledger.recent_records(ledger_root(ctx), str(ctx.task_id), limit=50, surface="result")

    trace, shown = {"tool_calls": []}, []
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True
    assert len(shown) == 1 and "PASS" in shown[0]  # the newly collected outcome reaches its author first
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is False
    assert len(shown) == 1  # a received PASS does not require another handback
    assert [(body.get("continueFrom"), key) for body, key in posts[1:]] == [("run-1", token)]  # that choice, once
    assert len(review_ledger.recent_records(ledger_root(ctx), str(ctx.task_id), limit=50, surface="result")) == len(records)
    [run] = [run for run in trace["review_runs"] if not run.get("superseded_by_revision")]
    assert (run["record_id"], run["aggregate_signal"]) == (first["record_id"], "PASS")


def test_a_plain_identical_commit_after_a_crash_after_the_choice_sends_that_choice_never_a_new_wave(
        stopped_wave, monkeypatch):
    a = stopped_wave
    head = _git(a.repo, "rev-parse", "HEAD")
    first = _commit_call(a)
    handback = json.JSONDecoder().raw_decode(first, first.index("{\"review_reference\""))[0]
    reference, offer = handback["review_reference"], handback["continuable_seats"]
    record_id = reference["review_record_id"]
    _git(a.repo, "add", "change.py")  # the reviewed bytes, staged again for the commit tool's reference collection
    real = review_continuation._start_requested
    monkeypatch.setattr(review_continuation, "_start_requested", lambda *a_, **k: (_ for _ in ()).throw(
        OSError("process ended after the choice")))
    unsent = _commit_call(a, review_reference=reference, continue_seats=[
        {"seat_id": "second", "operation_id": offer[0]["operation_id"]}])
    monkeypatch.setattr(review_continuation, "_start_requested", real)
    assert "REVIEW_CONTINUATION_UNSENT" in unsent and len(a.engine.posts) == 1, unsent
    token = _seat(a, record_id)["continuations"][0]["invocation_id"]
    attempts = len(load_state(a.data).attempts)

    # The retry key binds bytes and intent: other bytes are not the choice's subject.
    (a.repo / "change.py").write_text("value = 3\n", encoding="utf-8")
    other = _commit_call(a)
    assert token not in [key for _body, key in a.engine.posts] and _git(a.repo, "rev-parse", "HEAD") == head, other
    assert _seat(a, record_id)["status"] == "pending"  # the recorded choice stays its own subject's, unsent
    (a.repo / "change.py").write_text("value = 2\n", encoding="utf-8")
    sent_before, attempts = len(a.engine.posts), len(load_state(a.data).attempts)

    again = _commit_call(a)  # the same unchanged commit call of the choice's subject, with no reference
    assert [key for _body, key in a.engine.posts][sent_before:] == [token], again  # the recorded choice, once
    assert len(load_state(a.data).attempts) == attempts, "no new attempt row: the same round's attempt rejoined"
    assert _seat(a, record_id)["status"] == "responded"
    # The collected outcome is handed back; only the author's continuation commits.
    assert _git(a.repo, "rev-parse", "HEAD") == head and "Review outcome returned before commit" in again, again

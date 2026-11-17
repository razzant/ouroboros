"""#2710: a review attempt whose roster rows settled stops reading as a reviewer still working.

A start token left on a row whose own named operation settled is a lease nobody released (a collector projects
``late_settled`` only when no invocation is pending; the start checkpoint never binds a settled row), so it holds
nothing: the attempt compacts out of the newest-50 window, an unpaid one is evictable, and its round's checkout
goes. A reviewer genuinely in flight, a running additional adviser, a lost seat and an outcome merged without its
operation keep their custody. Every case asserts both directions.
"""
from __future__ import annotations

import pytest

from ouroboros import review_state as rs
from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord

TOKEN = "inv-left-behind"


def _row(state, *, token=TOKEN, operation="op-1", slot="s1", **extra):
    return {"slot_id": slot, "model_id": "m", "status": "ok" if state.endswith("settled") else "error",
            "operation_id": operation, "operation_state": state, "pending_invocation_id": token,
            "late_result_pending": state in {"in_flight", "custody_lost"}, "raw_text": "x" * 4000, **extra}


def _attempt(rows, *, attempt=1, paid=True, status="reviewed", task="t-old", **extra):
    return CommitAttemptRecord(ts=f"2026-10-0{1 + attempt % 9}T00:00:00+00:00", commit_message="review",
                               status=status, phase="review", task_id=task, tool_name="review_change", attempt=attempt,
                               paid=paid, review_retry_key=f"round-{attempt}", triad_raw_results=rows, **extra)


ADVISER = {"review_plan": {"additional": True}}


@pytest.mark.parametrize("rows, held", [
    ([_row("late_settled")], False),  # the live shape: 354 rows here, every run terminal
    ([_row("settled")], False),
    ([_row("in_flight")], True),  # a reviewer genuinely in flight
    ([_row("late_settled"), _row("in_flight", slot="a1", **ADVISER)], True),  # a running additional adviser
    ([_row("custody_lost")], True),  # a lost seat keeps the token it may still be rejoined by
    ([_row("late_settled", operation="")], True),  # an outcome merged without its operation proves nothing
], ids=["late_settled", "settled", "in_flight", "adviser_running", "custody_lost", "unidentified"])
def test_a_rows_own_settled_operation_outranks_its_leftover_start_token(rows, held):
    attempt = _attempt(rows)
    assert rs._attempt_has_active_review_custody(attempt) is held
    assert rs._review_roster_row_is_pending(rows[-1]) is held
    assert rs.attempt_counted_seats_closed(attempt) is (not held or rows[-1].get("review_plan") is not None)


def _history(target):
    """``target`` as the oldest attempt, then fifty newer paid rows: one more review pushes it out of the window."""
    newer = [_attempt([], attempt=number, task=f"t-{number}") for number in range(2, 52)]
    state = AdvisoryReviewState()
    state.attempts = [target, *newer]
    state.record_attempt(_attempt([], attempt=99, task="t-new"))
    return next((row for row in state.attempts if row.task_id == "t-old"), None)


@pytest.mark.parametrize("paid", [True, False])
def test_a_settled_row_with_a_leftover_token_compacts_and_an_unpaid_one_is_evicted(paid):
    """A paid row (the money ledger) is compacted but never evicted; an unpaid one goes first."""
    kept = _history(_attempt([_row("late_settled")], paid=paid))
    if paid:
        assert kept is not None and kept.raw_stripped and kept.triad_raw_results == [], kept
    else:
        assert kept is None, "an unpaid settled row is evictable"


@pytest.mark.parametrize("rows", [[_row("in_flight")], [_row("late_settled"), _row("in_flight", slot="a1", **ADVISER)]],
                         ids=["in_flight", "adviser_running"])
@pytest.mark.parametrize("paid", [True, False])
def test_a_reviewer_still_working_keeps_its_token_and_payload_out_of_the_window(rows, paid):
    kept = _history(_attempt([dict(row) for row in rows], paid=paid))
    assert kept is not None and not kept.raw_stripped, "never evicted or compacted while a seat runs"
    assert kept.triad_raw_results[-1]["pending_invocation_id"] == TOKEN
    assert len(kept.triad_raw_results[-1]["raw_text"]) == 4000


def test_a_late_snapshot_cannot_reopen_a_settled_row_or_raise_its_attempts_late_flag():
    """A-6: the merge reads a settled row's leftover token as closed, so a late pending snapshot of the SAME operation
    neither replaces its outcome nor keeps the attempt's late flag raised; a seat still running keeps both."""
    snapshot = _row("in_flight", token=TOKEN)
    settled = rs._merge_attempt(_attempt([_row("late_settled")]),
                                _attempt([dict(snapshot)], status="reviewing", late_result_pending=True))
    assert settled.triad_raw_results[0]["operation_state"] == "late_settled" and not settled.late_result_pending
    running = rs._merge_attempt(_attempt([_row("in_flight")]),
                                _attempt([dict(snapshot)], status="reviewing", late_result_pending=True))
    assert running.triad_raw_results[0]["operation_state"] == "in_flight" and running.late_result_pending


@pytest.mark.parametrize("state", ["late_settled", "in_flight"])
def test_a_dead_owners_settled_rows_close_its_attempt_and_a_delegated_seat_stays_recoverable(state):
    """A-6: confirmed owner death closes an attempt whose rows all settled (a leftover token held it ``late_wait``
    forever); a seat whose delegated run may still go keeps the attempt reviewing for its exact rejoin."""
    attempt = _attempt([_row(state)], status="reviewing", late_result_pending=True, review_owner_pid=4242)
    state_doc = AdvisoryReviewState()
    state_doc.attempts = [attempt]
    state_doc.reconcile_process_local_review_custody_after_owner_loss(confirmed_dead_owner_pids={4242})
    if state == "late_settled":
        assert (attempt.status, attempt.phase, attempt.late_result_pending) == ("failed", "infra", False)
        assert not rs._attempt_has_active_review_custody(attempt)
    else:
        assert (attempt.status, attempt.phase, attempt.late_result_pending) == ("reviewing", "review", True)
        assert attempt.triad_raw_results[0]["pending_invocation_id"] == TOKEN


def _old_writers_rounds(tmp_path, monkeypatch, rounds):
    """ROUNDS finished commit rounds an old writer left ``reviewing`` with closed seats' start tokens, each proved
    settled by its exact record (``_prefix_shape``); every round has its own retry key, so its own checkout name."""
    import copy

    from ouroboros.review_state import update_state
    from tests.test_unwaited_review_settlement import _commit_world, _prefix_shape

    world = _commit_world(tmp_path, monkeypatch, phase="review")
    _prefix_shape(world)

    def more(state):
        for number in range(2, rounds + 1):
            clone = copy.deepcopy(state.attempts[0])
            clone.attempt, clone.review_retry_key = number, f"round-{number}"
            state.attempts.append(clone)
    update_state(world.data, more)
    return world


def _names(world):
    from ouroboros.review_state import load_state
    from ouroboros.tools.review_subject import checkout_token

    return {checkout_token(row.review_retry_key, task_id=row.task_id) for row in load_state(world.data).attempts}


def test_the_checkout_pass_repairs_every_proven_round_in_one_write_and_a_second_pass_writes_nothing(
        tmp_path, monkeypatch):
    from ouroboros import review_checking_prune
    from ouroboros.review_state import load_state
    from tests.test_unwaited_review_settlement import _state_writes

    world = _old_writers_rounds(tmp_path, monkeypatch, 3)
    assert review_checking_prune._review_rounds(world.data)[0] == _names(world), "all three read as custody"
    writes = _state_writes(monkeypatch)

    first, _owners = review_checking_prune._review_rounds(world.data, repair=True)
    assert len(writes) == 1 and first == set(), writes
    attempts = load_state(world.data).attempts
    assert [(row.status, row.phase) for row in attempts] == [("reviewed", "review_only")] * 3
    assert not any(row.get("pending_invocation_id") for item in attempts for row in item.triad_raw_results)
    second, _owners = review_checking_prune._review_rounds(world.data, repair=True)
    assert len(writes) == 1 and second == set(), "an identical second pass writes nothing"


@pytest.mark.parametrize("evidence", ["unreadable_record", "record_seat_naming_another_operation"])
def test_a_round_its_record_does_not_prove_stays_held_and_the_pass_writes_nothing(tmp_path, monkeypatch, evidence):
    from ouroboros import review_checking_prune
    from ouroboros import review_ledger as rl
    from ouroboros.review_state import load_state
    from tests.test_unwaited_review_settlement import _state_writes

    world = _old_writers_rounds(tmp_path, monkeypatch, 1)
    if evidence == "unreadable_record":
        rl.record_path(world.data, world.record_id).write_text("{not json", encoding="utf-8")
    else:
        rl.revise_record(world.data, world.record_id, lambda payload: {**payload, "rows": [
            {**row, "operation_id": f"{row.get('operation_id')}-successor"} for row in payload["rows"]]})
    before = load_state(world.data).attempts
    writes = _state_writes(monkeypatch)

    active, _owners = review_checking_prune._review_rounds(world.data, repair=True)

    assert active == _names(world) and writes == [] and load_state(world.data).attempts == before


def _start(data, slot, *, operation=None, fate="terminal"):
    """The custody record of seat SLOT's own start token (``inv-<slot>``), as the delegated transport writes it."""
    from ouroboros import delegate_custody as custody
    from tests.test_unwaited_review_settlement import TASK

    token, run = f"inv-{slot}", f"run-{slot}"
    assert custody.emit(data, custody.START_REQUESTED, {
        "invocation_id": token, "task_id": TASK, "surface": "multi_model_review", "slot_id": slot,
        "operation_id": operation or f"op-{slot}", "route": "claude", "request": {"prompt": "review"}})
    if fate == "refused":
        assert custody.emit(data, custody.START_FAILED, {"invocation_id": token, "task_id": TASK, "definite": True,
                                                         "error": "refused"})
    if fate in {"running", "terminal"}:
        assert custody.emit(data, custody.STARTED, {"run_id": run, "task_id": TASK, "invocation_id": token})
    if fate == "terminal":
        assert custody.emit(data, custody.TERMINAL_OBSERVED, {"run_id": run, "task_id": TASK, "state": "succeeded",
                                                              "observation_seq": 1, "source_ref": {"sha256": "0" * 64}})


@pytest.mark.parametrize("fate", ["terminal_run", "definite_refusal", "pending_invocation", "run_still_going",
                                  "answer_awaited", "foreign_invocation"])
def test_an_in_flight_row_of_an_ended_task_closes_only_by_its_invocations_own_custody_record(
        tmp_path, monkeypatch, fate):
    """O1(c): a seat left ``in_flight`` after its round ended closes from its start token's own custody record: a
    definite refusal (no outcome exists) or a run custody holds terminal whose outcome the wave's record holds. A
    pending or unbound start, a run still going, a foreign invocation and an answer the record still awaits (the
    wave's own $0 collection reads it) keep the seat's token, its row and the round's checkout. The task has ended in
    every case, and the control seat ``s1`` (a terminal run its record holds) closes in the same single write."""
    from ouroboros import review_checking_prune
    from ouroboros import review_ledger as rl
    from ouroboros.task_results import write_task_result
    from tests.test_review_ledger import _raw
    from tests.test_unwaited_review_settlement import TASK, _commit_world, _saved, _state_writes

    world = _commit_world(tmp_path, monkeypatch)
    write_task_result(world.data, TASK, "completed", result="done")
    recorded = [{**_raw("s1", "m1", parts=("change", "coupling")), "operation_id": "op-s1"}]
    if fate not in {"definite_refusal", "answer_awaited"}:  # a refused start has no outcome to record
        recorded.append({**_raw("s2", "m2"), "operation_id": "op-s2"})
    rl.settle_from_collected(world.data, world.record_id, recorded)
    _start(world.data, "s1")
    _start(world.data, "s2", operation="op-elsewhere" if fate == "foreign_invocation" else None, fate={
        "definite_refusal": "refused", "pending_invocation": "requested", "run_still_going": "running"}.get(fate, "terminal"))
    name = next(iter(review_checking_prune._review_rounds(world.data)[1]))
    writes = _state_writes(monkeypatch)

    first, _owners = review_checking_prune._review_rounds(world.data, repair=True)
    second, _owners = review_checking_prune._review_rounds(world.data, repair=True)

    closes = fate in {"terminal_run", "definite_refusal"}
    assert len(writes) == 1 and (name in first) is (not closes) and first == second, (writes, first)
    saved = _saved(world)
    rows = {row["slot_id"]: row for row in saved.triad_raw_results}
    assert rows["s1"]["operation_state"] == "late_settled" and "pending_invocation_id" not in rows["s1"]
    assert rows["s1"]["custody_closure"] == {"invocation_id": "inv-s1", "run_id": "run-s1", "run_state": "succeeded"}
    if not closes:
        assert (rows["s2"]["operation_state"], rows["s2"]["pending_invocation_id"]) == ("in_flight", "inv-s2")
        assert saved.status == "reviewing" and saved.late_result_pending
        return
    assert (saved.status, saved.phase, saved.late_result_pending) == ("reviewed", "review_only", False)
    assert "pending_invocation_id" not in rows["s2"]
    assert rows["s2"]["operation_state"] == ("not_dispatched" if fate == "definite_refusal" else "late_settled")


@pytest.mark.parametrize("phase", ["late_wait", "review"])
@pytest.mark.parametrize("fate", ["terminal_run", "run_still_going"])
def test_a_delegated_row_a_deferred_write_left_open_is_repaired_where_the_author_decides(tmp_path, monkeypatch,
                                                                                          phase, fate):
    """#2779 (Astra, 9 Nov): a commit row whose collection settled its record while a busy state lock refused the row
    write keeps its delegated seats in flight with their start tokens. The Advisory author's reference resolves after
    the exact repair closes them from each start's own custody record (a run custody holds terminal whose outcome the
    record holds), the task still running. The record alone proves nothing: a run still going keeps every seat, the
    row as it was and the refusal (still running, or no handback yet in the ``review`` phase)."""
    from ouroboros import review_ledger as rl
    from ouroboros.tools.commit_gate import resolve_commit_review_reference
    from tests.test_review_ledger import _raw
    from tests.test_unwaited_review_settlement import _commit_world, _reference, _saved

    world = _commit_world(tmp_path, monkeypatch, phase=phase)
    rl.settle_from_collected(world.data, world.record_id, [
        {**_raw("s1", "m1", parts=("change", "coupling")), "operation_id": "op-s1"},
        {**_raw("s2", "m2"), "operation_id": "op-s2"}])
    assert rl.load_record(world.data, world.record_id)["state"] == "settled"
    _start(world.data, "s1")
    _start(world.data, "s2", fate="running" if fate == "run_still_going" else "terminal")
    stale = _saved(world)
    assert (stale.status, stale.phase) == ("reviewing", phase)
    assert all(row["operation_state"] == "in_flight" and row["pending_invocation_id"] for row in stale.triad_raw_results)

    if fate == "run_still_going":
        with pytest.raises(ValueError, match="still running" if phase == "late_wait" else "no longer has its exact"):
            resolve_commit_review_reference(world.ctx, _reference(stale))
        assert _saved(world) == stale, "nothing is closed on the record's word alone"
        return
    source = resolve_commit_review_reference(world.ctx, _reference(stale))
    saved = _saved(world)
    assert (source.status, source.phase) == (saved.status, saved.phase) == ("reviewed", "review_only")
    rows = {row["slot_id"]: row for row in saved.triad_raw_results}
    assert all(row["operation_state"] == "late_settled" and "pending_invocation_id" not in row for row in rows.values())
    assert {row["custody_closure"]["run_id"] for row in rows.values()} == {"run-s1", "run-s2"}
    assert saved.paid and rl.load_record(world.data, world.record_id)["verdict"]["aggregate"] == "PASS"


def test_a_pass_whose_in_flight_seats_have_only_pending_starts_writes_nothing(tmp_path, monkeypatch):
    from ouroboros import review_checking_prune
    from ouroboros.review_state import load_state
    from tests.test_unwaited_review_settlement import _commit_world, _state_writes

    world = _commit_world(tmp_path, monkeypatch)
    for slot in ("s1", "s2"):
        _start(world.data, slot, fate="requested")
    before = load_state(world.data).attempts
    writes = _state_writes(monkeypatch)

    active, _owners = review_checking_prune._review_rounds(world.data, repair=True)

    assert writes == [] and len(active) == 1 and load_state(world.data).attempts == before

"""A paid review its author stopped waiting for settles its own record (owner 17.10.2026, Q3-2).

A wave that returns to its author with seats still running, and no addressed message behind it, owes
its settlement (one duty row of ``late_review_marks``): once its seats are collectable at $0 and its
author's task ended, the record settles to its verdict without a re-issue and the outcome is kept in
the author's room. A wave its author collected itself while running needs no mark.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger as rl
from ouroboros import review_pause
from tests.test_review_ledger import CRITICAL, _answer, _facts, _raw

TASK = "unwaited-author"


def _data(tmp_path):
    for sub in ("logs", "locks", "state"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


def _record(data, *, pending=True):
    second = (_raw("s2", "m2", "pending", operation_state="in_flight", late_result_pending=True) if pending
              else _raw("s2", "m2"))
    first = _raw("s1", "m1", parts=("change", "coupling"), raw_text="two-part answer")  # answers coupling too
    record = rl.build_wave_record(_facts([first, second], task_id=TASK, pending=pending),
                                  surface="change", drive_root=data)
    rl.write_record(data, record)
    return record.record_id


def _owed_rows(data):
    from ouroboros import store
    from ouroboros.late_review_marks import OWED_SET

    with store.open_store(data, write=False) as txn:
        return {key: row for key, row in txn.set_rows(OWED_SET).items() if "left_review" in row}


def _seats():
    return [{"surface": "change", "slot_id": slot, "parts": ["change"], "model": model, "status": "ok",
             "operation_state": "settled", "late_result_pending": False, "raw_text": "[]"}
            for slot, model in (("s1", "m1"), ("s2", "m2"))]


def _owe(data, record_id):
    ctx = SimpleNamespace(task_id=TASK, drive_root=data, task_metadata={"budget_drive_root": str(data)})
    assert review_pause.owe_unwaited_review(ctx, surface="change", retry_key="round-1", record_id=record_id)
    [row] = _owed_rows(data).values()
    return row


def test_an_unwaited_review_settles_its_record_once_its_author_ended(tmp_path, monkeypatch):
    data = _data(tmp_path)
    record_id = _record(data)
    row = _owe(data, record_id)
    assert row["after_end"] is True and row["left_review"]["left"] == "unwaited"
    assert row["left_review"]["record_id"] == record_id
    published = []
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (None, _seats(), ""))
    monkeypatch.setattr("ouroboros.late_review_marks.publish_late_review",
                        lambda root, mark: published.append(mark) or SimpleNamespace(status="saved"))

    assert review_pause.ended_left_review(row["left_review"]) == "saved"

    settled = rl.load_record(data, record_id)
    assert settled["state"] == "settled" and settled["verdict"]["aggregate"] == "PASS"
    [mark] = published
    head = mark["lines"][0]
    assert "that its author stopped waiting for" in head and "Its record now reads PASS" in head
    assert "kept here because its author's task ended" in head and "addressed message" not in head


@pytest.mark.parametrize("delivery", ["whole", "partial", "checkpoint_lost"])
def test_an_unwaited_review_its_author_collected_while_running_needs_no_mark(tmp_path, monkeypatch, delivery):
    from ouroboros import review_waiting, working_checkpoint
    from tests.test_review_waiting import _receipt

    data = _data(tmp_path)
    record_id = _record(data, pending=False)
    row = _owe(data, record_id)
    ctx = SimpleNamespace(task_id=TASK, drive_root=data, budget_drive_root=None)
    record = rl.load_record(data, record_id)
    rendered = None if delivery == "cursor_only" else record["rows"][:1] if delivery == "partial" else record["rows"]
    receipt = _receipt(ctx, record, rendered)
    review_waiting.acknowledge_delivery(ctx, receipt)
    if delivery != "checkpoint_lost":
        working_checkpoint.flush_content_acks(ctx)
    before = rl.load_record(data, record_id)
    published = []
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (None, _seats(), ""))
    monkeypatch.setattr("ouroboros.late_review_marks.publish_late_review",
                        lambda root, mark: published.append(mark) or SimpleNamespace(status="saved"))

    # No next delivery: Stop, a budget exit or the forced final may end the author here.
    assert review_pause.ended_left_review(row["left_review"]) == ("collected" if delivery == "whole" else "saved")
    assert len(published) == (0 if delivery == "whole" else 1)
    after = rl.load_record(data, record_id)
    assert (after["rows"], after["verdict"], after["revision"]) == (before["rows"], before["verdict"], before["revision"])


def test_a_left_review_whose_seats_still_run_keeps_its_record_pending_in_process(tmp_path, monkeypatch):
    """The in-process notice settles the record only when every seat has its outcome; a seat that may
    still answer keeps it pending and the notice names no verdict."""
    data = _data(tmp_path)
    record_id = _record(data)
    running = [_seats()[0], {**_seats()[1], "operation_state": "in_flight", "late_result_pending": True,
                             "status": "error", "raw_text": ""}]
    # S3 collection updates the paid attempt as well as its ledger record.
    from ouroboros.review_state import CommitAttemptRecord, update_state
    attempt = CommitAttemptRecord(ts="2026-10-22T00:00:00Z", commit_message="review", status="reviewed",
        task_id=TASK, tool_name="review_change", attempt=1, paid=True, review_record_id=record_id,
        review_retry_key="round-1", triad_raw_results=running, late_result_pending=True)
    update_state(data, lambda state: state.record_attempt(attempt))
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (attempt, running, ""))
    record = {"surface": "change", "task_id": TASK, "record_id": record_id, "retry_key": "round-1",
              "drive_root": str(data), "canonical_root": str(data)}
    assert review_pause._settled_aggregate(record) == ""
    assert rl.load_record(data, record_id)["state"] == "pending"
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (attempt, _seats(), ""))
    assert review_pause._settled_aggregate(record) == "PASS"
    assert rl.load_record(data, record_id)["state"] == "settled"


def test_a_headless_authors_left_seats_are_read_from_the_canonical_custody_root(tmp_path, monkeypatch):
    """The substrate writes a seat's assignment and outcome at the canonical custody root; a headless or forked
    author's own drive holds only the attempt row. Reading the seat from the author's drive called every owed
    seat "Original review assignment unavailable" (SC-C group A). Two-sided: the canonical root when the duty
    names one, the author's drive otherwise."""
    canonical, child = tmp_path / "data", tmp_path / "data" / "state" / "headless_tasks" / TASK / "data"
    attempt = SimpleNamespace(root_task_id=TASK, review_contract_fingerprint="cf")
    row = {"slot_id": "s2", "model": "m2", "operation_id": "op-2", "operation_state": "in_flight",
           "late_result_pending": True, "parts": ["change"]}
    roots = []
    monkeypatch.setattr(review_pause, "_left_attempt", lambda record: attempt)
    monkeypatch.setattr(review_pause, "_left_rows", lambda attempt: [("change", "s2", row)])
    monkeypatch.setattr("ouroboros.review_operation.collect_recorded_review_actor",
                        lambda root, **identity: roots.append(root) or {"status": "ok", "raw_text": "[]",
                                                                        "operation_state": "settled"})
    record = {"surface": "change", "task_id": TASK, "retry_key": "round-1", "drive_root": str(child),
              "canonical_root": str(canonical)}
    _attempt, seats, gap = review_pause._left_seats(record)
    assert gap == "" and roots == [canonical] and seats[0]["raw_text"] == "[]"
    review_pause._left_seats({**record, "canonical_root": ""})
    assert roots[-1] == child


def test_an_unwaited_result_review_settles_into_its_rooms_mark_with_each_seats_answer(tmp_path, monkeypatch):
    """A Presence result's Required review runs with nobody waiting: once its author ended, its record settles
    from the recorded seats' findings arrays and the outcome is kept in the room, each seat named by its
    result answer (a result seat answers contract A, never the two-part object)."""
    data = _data(tmp_path)
    second = _raw("s2", "m2", "pending", parts=("result",), operation_state="in_flight", late_result_pending=True)
    record = rl.build_wave_record(_facts([_raw("s1", "m1", parts=("result",)), second], task_id=TASK, pending=True),
                                  surface="result", drive_root=data)
    rl.write_record(data, record)
    ctx = SimpleNamespace(task_id=TASK, drive_root=data, task_metadata={"budget_drive_root": str(data)})
    assert review_pause.owe_unwaited_review(ctx, surface="result", retry_key="round-1", record_id=record.record_id)
    [row] = _owed_rows(data).values()
    seats = [{**seat, "surface": "result", "parts": ["result"]} for seat in _seats()]
    published = []
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (None, seats, ""))
    monkeypatch.setattr("ouroboros.late_review_marks.publish_late_review",
                        lambda root, mark: published.append(mark) or SimpleNamespace(status="saved"))

    assert review_pause.ended_left_review(row["left_review"]) == "saved"

    settled = rl.load_record(data, record.record_id)
    assert settled["state"] == "settled" and settled["verdict"]["aggregate"] == "PASS"
    [mark] = published
    assert "result review" in mark["lines"][0] and "Its record now reads PASS" in mark["lines"][0]
    body = "\n".join(mark["lines"][1:])
    assert "result PASS" in body and "retained source" not in body, mark["lines"]


# --- #2587/#2590: the one settlement writer ends a settled commit review's phase for its consumers -----

FAIL_TEXT = '[{"item": "bug", "verdict": "FAIL", "severity": "critical", "reason": "the amount is wrong"}]'


def _commit_world(tmp_path, monkeypatch, *, phase="late_wait", adviser=False, scope=False, record_rows=None):
    """A commit wave its author left by mail, persisted as the gate leaves it: every seat in flight with its start
    token, the record pending. ``adviser`` adds an additional seat; ``scope`` puts the second seat on the scope
    roster, ``"listed"`` in its ``raw_results`` or ``"single"`` as the historical wrapper that is its own row."""
    import subprocess

    from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state
    from ouroboros.tools.registry import ToolContext
    from tests.test_review_ledger import _plan

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    data, repo = _data(tmp_path / "data"), tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    seats = [("s1", "m1", ("change", "coupling")), ("s2", "m2", ("change",))] + ([("a1", "m3", ("change",))] if adviser else [])
    raws = [_raw(slot, model, "pending", parts=parts, operation_id=f"op-{slot}", operation_state="in_flight",
                 late_result_pending=True) for slot, model, parts in seats]
    plans = [_plan(slot, model, parts=parts, additional=slot == "a1") for slot, model, parts in seats]
    record = rl.build_wave_record(_facts(raws, rows=record_rows or plans, task_id=TASK, pending=True,
                                         binding_fingerprint="fp-reviewed"), surface="commit_gate", drive_root=data)
    rl.write_record(data, record)

    def row(slot, model, surface="multi_model_review"):
        return {"slot_id": slot, "model_id": model, "status": "error", "operation_id": f"op-{slot}",
                "operation_state": "in_flight", "late_result_pending": True, "pending_invocation_id": f"inv-{slot}",
                **({"review_plan": {"additional": True}} if slot == "a1" else {})}
    triad = [row(slot, model) for slot, model, _parts in seats if not (scope and slot == "s2")]
    attempt = CommitAttemptRecord(ts="2026-11-03T20:00:00+00:00", commit_message="candidate", status="reviewing",
        phase=phase, task_id=TASK, repo_key=make_repo_key(repo), tool_name="commit_reviewed", attempt=1, paid=True,
        late_result_pending=True, pre_review_fingerprint="fp-reviewed", review_retry_key="round-1",
        review_record_id=record.record_id, triad_raw_results=triad,
        scope_raw_result={"raw_results": [row("s2", "m2")]} if scope == "listed" else row("s2", "m2") if scope else {})
    update_state(data, lambda state: state.record_attempt(attempt))
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_id=TASK)
    left = {"surface": "commit", "task_id": TASK, "record_id": record.record_id, "retry_key": "round-1",
            "drive_root": str(data), "canonical_root": str(data), "attempt": 1}
    return SimpleNamespace(data=data, repo=repo, ctx=ctx, record_id=record.record_id, left=left, attempt=attempt)


def _two_part_pass():
    import json

    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
    return json.dumps({"change": [], "change_clean": True, "coupling": [
        {"item": item, "verdict": "PASS", "severity": "advisory", "reason": "checked the change and its consumers"}
        for item in sorted(SCOPE_REQUIRED_ITEMS)]})


def _seat(slot, surface="multi_model_review", text="", **extra):
    """A collected terminal seat in the production shape: its start token is not among the carried facts. ``s1``
    was asked both parts and answers the two-part object; the others answer the change question."""
    both = slot == "s1"
    return {"surface": surface, "slot_id": slot, "parts": ["change", "coupling"] if both else ["change"],
            "model": f"m-{slot}", "status": "ok", "operation_id": f"op-{slot}", "operation_state": "settled",
            "late_result_pending": False, "raw_text": text or (_two_part_pass() if both else "[]"), "usage": {}, **extra}


def _saved(world):
    from ouroboros.review_state import load_state

    return load_state(world.data).attempts[-1]


def _reference(attempt):
    return {"surface": "commit", "review_record_id": attempt.review_record_id, **{key: getattr(attempt, key) for key in (
        "repo_key", "task_id", "tool_name", "attempt", "pre_review_fingerprint")}}


@pytest.mark.parametrize("phase", ["late_wait", "review"])
@pytest.mark.parametrize("token", ["omitted", "empty"])
def test_a_settled_left_commit_wave_ends_its_review_phase_for_every_consumer(tmp_path, monkeypatch, phase, token):
    """#2587/#2590: the left wave settles FAIL through the real writer. The row takes the commit's settled projection
    (reviewed/review_only), each closed seat loses the start token its terminal outcome contradicts, and the next
    consumers read it as settled: the continuation reference resolves, the overlap check holds nothing, Storage
    no longer counts the round's checkout as in use and an identical re-issue would collect it. FAIL stays FAIL."""
    from ouroboros import review_checking_prune
    from ouroboros.review_state import _attempt_has_active_review_custody, settle_completed_round
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt, resolve_commit_review_reference
    from ouroboros.tools.review_subject import checkout_token

    world = _commit_world(tmp_path, monkeypatch, phase=phase)
    extra = {"pending_invocation_id": ""} if token == "empty" else {}
    seats = [_seat("s1", **extra), _seat("s2", text=FAIL_TEXT, **extra)]
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (world.attempt, seats, ""))

    assert review_pause._settled_aggregate(world.left) == "FAIL"

    saved = _saved(world)
    assert (saved.status, saved.phase, saved.late_result_pending) == ("reviewed", "review_only", False)
    assert all("pending_invocation_id" not in row or not row["pending_invocation_id"] for row in saved.triad_raw_results)
    assert not _attempt_has_active_review_custody(saved) and saved.triad_raw_results[1]["raw_text"] == FAIL_TEXT
    assert rl.load_record(world.data, world.record_id)["verdict"]["aggregate"] == "FAIL"
    assert resolve_commit_review_reference(world.ctx, _reference(saved)).attempt == 1
    assert _check_overlapping_review_attempt(world.ctx) is None and not world.ctx._review_resume_pending
    active, _owners = review_checking_prune._review_rounds(world.data)
    assert checkout_token("round-1", task_id=TASK) not in active
    reused = settle_completed_round(world.ctx, world.data, repo_key=saved.repo_key, tool_name="commit_reviewed",
                                    task_id=TASK, review_retry_key="round-1")
    assert reused is not None and reused.attempt == 1


@pytest.mark.parametrize("case", ["unresolved_usage_token", "other_operation", "superseded_successor"])
def test_a_seat_without_terminal_proof_keeps_its_custody(tmp_path, monkeypatch, case):
    """Only a terminal outcome of the row's own operation closes its token: a seat whose usage still holds the start
    token, another operation's outcome, a lost seat, an outcome naming no operation and a late outcome of the
    operation an explicit continuation superseded leave the row in custody (the successor keeps its own token) and
    the attempt reviewing, so nothing is projected settled."""
    from ouroboros.review_state import _attempt_has_active_review_custody, update_state
    from ouroboros.tools.review_change_custody import settle_collected_attempt

    world = _commit_world(tmp_path, monkeypatch)
    token = "inv-s1"
    if case == "superseded_successor":  # the seat was continued: its row now names the successor in flight
        token = "inv-s1b"
        update_state(world.data, lambda state: state.attempts[-1].triad_raw_results[0].update(
            operation_id="op-s1b", superseded_operation_ids=["op-s1"], pending_invocation_id=token))
    incoming = {"unresolved_usage_token": _seat("s1", operation_state="in_flight", late_result_pending=True,
                                                usage={"pending_invocation_id": "inv-s1"}),
                "other_operation": _seat("s1", operation_id="op-successor"),
                "custody_lost": _seat("s1", operation_state="custody_lost", late_result_pending=True),
                "unidentified": _seat("s1", operation_id=""), "superseded_successor": _seat("s1")}[case]
    if case == "unidentified":  # a historical row that never named its operation
        world.attempt.triad_raw_results[0].pop("operation_id")
    record = rl.load_record(world.data, world.record_id)

    settle_collected_attempt(world.data, world.attempt, [incoming, _seat("s2")], record)

    first = _saved(world).triad_raw_results[0]
    assert first["pending_invocation_id"] == token, first
    assert _saved(world).status == "reviewing" and _attempt_has_active_review_custody(_saved(world))


@pytest.mark.parametrize("shape", ["listed", "single"])
def test_a_legacy_scope_seat_settles_on_its_own_roster(tmp_path, monkeypatch, shape):
    world = _commit_world(tmp_path, monkeypatch, scope=shape)
    seats = [_seat("s1"), _seat("s2", surface="scope_review")]
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (world.attempt, seats, ""))

    assert review_pause._settled_aggregate(world.left) == "PASS"

    saved = _saved(world)
    assert [row["slot_id"] for row in saved.triad_raw_results] == ["s1"], "a scope seat never joins the triad"
    [scope_row] = saved.scope_raw_result["raw_results"] if shape == "listed" else [saved.scope_raw_result]
    assert scope_row["operation_state"] == "settled" and not scope_row.get("pending_invocation_id")
    assert scope_row["slot_id"] == "s2" and ("raw_results" in saved.scope_raw_result) == (shape == "listed")
    assert (saved.status, saved.phase) == ("reviewed", "review_only")


def test_an_additional_adviser_keeps_its_custody_and_gets_no_veto(tmp_path, monkeypatch):
    """Counted completion ends the review phase while the additional adviser still runs: its row keeps its start
    token and keeps Storage's checkout held, yet it holds no new candidate and the counted verdict is reusable
    by an identical re-issue (no second paid wave)."""
    from ouroboros import review_checking_prune
    from ouroboros.review_state import settle_completed_round
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt
    from ouroboros.tools.review_change_custody import settle_collected_attempt
    from ouroboros.tools.review_subject import checkout_token

    world = _commit_world(tmp_path, monkeypatch, adviser=True)
    rl.settle_from_collected(world.data, world.record_id, [
        {**_raw("s1", "m1", parts=("change", "coupling")), "operation_id": "op-s1"},
        {**_raw("s2", "m2"), "operation_id": "op-s2"}])
    record = rl.load_record(world.data, world.record_id)
    assert record["state"] == "pending" and record["verdict"]["aggregate"] == "PASS"  # the adviser is still open

    settle_collected_attempt(world.data, world.attempt, [_seat("s1"), _seat("s2")], record)

    saved = _saved(world)
    adviser = next(row for row in saved.triad_raw_results if row["slot_id"] == "a1")
    assert (saved.status, saved.phase, saved.late_result_pending) == ("reviewed", "review_only", True)
    assert adviser["operation_state"] == "in_flight" and adviser["pending_invocation_id"] == "inv-a1"
    assert _check_overlapping_review_attempt(world.ctx) is None and not world.ctx._review_resume_pending
    active, _owners = review_checking_prune._review_rounds(world.data)
    assert checkout_token("round-1", task_id=TASK) in active, "the live adviser's checkout stays held"
    reused = settle_completed_round(world.ctx, world.data, repo_key=saved.repo_key, tool_name="commit_reviewed",
                                    task_id=TASK, review_retry_key="round-1")
    assert reused is not None and reused.attempt == 1


def _prefix_shape(world):
    """The persisted pre-fix shape: the old writer merged each terminal seat over its row but kept the row's start
    token, status and phase; the record settled FAIL."""
    from ouroboros.review_state import update_state

    def old_merge(state):
        row = state.attempts[-1]
        for seat in row.triad_raw_results:
            seat.update(operation_state="settled", late_result_pending=False, status="responded", raw_text="[]")
        row.late_result_pending = False
    rl.settle_from_collected(world.data, world.record_id, [
        {**_raw("s1", "m1", parts=("change", "coupling")), "operation_id": "op-s1"},
        {**_raw("s2", "m2", answers={"change": _answer("change", "FAIL", [CRITICAL])}), "operation_id": "op-s2"}])
    update_state(world.data, old_merge)


@pytest.mark.parametrize("consumer", ["reference", "identical_reissue", "review_status"])
def test_a_persisted_prefix_row_is_repaired_exactly_where_a_consumer_would_hold_it(tmp_path, monkeypatch, consumer):
    """A row the old writer left reviewing with closed seats' tokens is repaired from its exact bound settled record,
    outside the state lock, only when a consumer would refuse or hold it; ``review_status`` reads the same fact
    and offers the reference without writing anything."""
    import json as _json

    from ouroboros.review_state import load_state, settle_completed_round
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt, resolve_commit_review_reference
    from ouroboros.tools.preflight_review import _handle_review_status

    world = _commit_world(tmp_path, monkeypatch, phase="review")
    _prefix_shape(world)
    stale = _saved(world)
    assert (stale.status, stale.phase) == ("reviewing", "review") and stale.triad_raw_results[0]["pending_invocation_id"]

    if consumer == "reference":
        assert resolve_commit_review_reference(world.ctx, _reference(stale)).phase == "review_only"
    elif consumer == "overlap":
        assert _check_overlapping_review_attempt(world.ctx) is None and not world.ctx._review_resume_pending
    elif consumer == "identical_reissue":
        assert settle_completed_round(world.ctx, world.data, repo_key=stale.repo_key, tool_name="commit_reviewed",
                                      task_id=TASK, review_retry_key="round-1") is not None
    else:
        before = load_state(world.data).attempts
        payload = _json.loads(_handle_review_status(world.ctx, task_id=TASK))
        assert payload["review_reference"]["review_record_id"] == world.record_id, payload
        assert load_state(world.data).attempts == before, "review_status writes nothing"
        return
    repaired = _saved(world)
    assert (repaired.status, repaired.phase) == ("reviewed", "review_only")
    assert not any(row.get("pending_invocation_id") for row in repaired.triad_raw_results)
    assert rl.load_record(world.data, world.record_id)["verdict"]["aggregate"] == "FAIL"


@pytest.mark.parametrize("evidence", ["missing_record_id", "unreadable_record", "record_seat_naming_no_operation"])
def test_a_prefix_row_without_exact_settled_evidence_stays_held(tmp_path, monkeypatch, evidence):
    """Settlement is proved only positively: a record with no counted seat, a counted seat reopened to a successor,
    or one naming no operation proves nothing, whatever ``counted_pending`` alone would say."""
    from ouroboros.review_state import update_state
    from ouroboros.tools.commit_gate import resolve_commit_review_reference

    world = _commit_world(tmp_path, monkeypatch, phase="review")
    _prefix_shape(world)
    if evidence == "missing_record_id":
        update_state(world.data, lambda state: setattr(state.attempts[-1], "review_record_id", ""))
    elif evidence == "foreign_binding":
        rl.revise_record(world.data, world.record_id, lambda payload: {
            **payload, "fingerprints": {**payload["fingerprints"], "binding": "another-subject"}})
    elif evidence == "unreadable_record":
        rl.record_path(world.data, world.record_id).write_text("{not json", encoding="utf-8")
    elif evidence in {"empty_record_roster", "reopened_record_seat", "record_seat_naming_no_operation"}:
        def seats(payload):
            rows = [] if evidence == "empty_record_roster" else [dict(row) for row in payload["rows"]]
            for row in rows[1:]:
                if evidence == "reopened_record_seat":
                    row.update(operation_id="op-s2b", status="pending", operation_state="in_flight",
                               continuations=[{"operation_id": "op-s2b", "supersedes": "op-s2"}])
                else:
                    row.pop("operation_id", None)
            return {**payload, "rows": rows}
        rl.revise_record(world.data, world.record_id, seats)
    else:
        update_state(world.data, lambda state: state.attempts[-1].triad_raw_results[1].update(
            operation_state="in_flight", late_result_pending=True))

    with pytest.raises(ValueError, match="missing or no longer has its exact evidence"):
        resolve_commit_review_reference(world.ctx, _reference(_saved(world)))

    kept = _saved(world)
    assert (kept.status, kept.phase) == ("reviewing", "review") and kept.triad_raw_results[0]["pending_invocation_id"]


@pytest.mark.parametrize("record", ["settled", "missing"])
def test_a_compacted_settled_row_still_names_its_review_for_an_explicit_continuation(tmp_path, monkeypatch, record):
    """A settled row compacted outside the newest-50 window keeps continuation through its exact settled record
    (whose seats keep their retained sources); without that record the refusal stays honest."""
    from ouroboros.review_state import update_state
    from ouroboros.review_state_custody import _strip_attempt_heavy_payload
    from ouroboros.tools.commit_gate import resolve_commit_review_reference

    world = _commit_world(tmp_path, monkeypatch)
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (world.attempt, [_seat("s1"), _seat("s2")], ""))
    assert review_pause._settled_aggregate(world.left) == "PASS"
    update_state(world.data, lambda state: _strip_attempt_heavy_payload(state.attempts[-1]))
    assert _saved(world).raw_stripped and _saved(world).triad_raw_results == []
    if record == "missing":
        rl.record_path(world.data, world.record_id).unlink()
        with pytest.raises(ValueError, match="missing or no longer has its exact evidence"):
            resolve_commit_review_reference(world.ctx, _reference(_saved(world)))
        return
    assert resolve_commit_review_reference(world.ctx, _reference(_saved(world))).raw_stripped


def test_a_definite_pre_dispatch_refusal_ends_custody_but_is_no_verdict(tmp_path, monkeypatch):
    """A counted seat whose start was definitely refused (``not_dispatched``) is that operation's terminal fact: its
    token closes and the phase ends, so nothing holds the next candidate, yet no verdict is read into it. Its round
    is not collected as a decided review, so the next identical call dispatches a new wave."""
    from ouroboros.review_state import _attempt_has_active_review_custody, settle_completed_round
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt, resolve_commit_review_reference

    world = _commit_world(tmp_path, monkeypatch)
    refused = {**_seat("s2"), "status": "not_dispatched", "operation_state": "not_dispatched", "raw_text": "",
               "failure_code": "START_UNSENT", "error": "the start was refused before any send"}
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (world.attempt, [_seat("s1"), refused], ""))

    aggregate = review_pause._settled_aggregate(world.left)

    assert aggregate not in ("PASS", "FAIL"), aggregate
    saved = _saved(world)
    second = saved.triad_raw_results[1]
    assert second["operation_state"] == "not_dispatched" and not second.get("pending_invocation_id"), second
    assert (saved.status, saved.phase, saved.late_result_pending) == ("reviewed", "review_only", False)
    assert not _attempt_has_active_review_custody(saved)
    assert _check_overlapping_review_attempt(world.ctx) is None and not world.ctx._review_resume_pending
    assert settle_completed_round(world.ctx, world.data, repo_key=saved.repo_key, tool_name="commit_reviewed",
                                  task_id=TASK, review_retry_key="round-1") is None
    assert resolve_commit_review_reference(world.ctx, _reference(saved)).phase == "review_only"


@pytest.mark.parametrize("roster", ["empty", "reserved"])
def test_a_roster_reserved_before_dispatch_stays_held_and_unrepaired(tmp_path, monkeypatch, roster):
    """A row whose roster is still empty or only reserved (``pending_dispatch``) proves no outcome, even beside a
    settled record a naive repair could read: it is not repaired or projected, its own task's next call rejoins it
    instead of buying a new wave, its round is not collected, and its reference stays refused."""
    from ouroboros.review_state import settle_completed_round, update_state
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt, resolve_commit_review_reference
    from ouroboros.tools.review_change_custody import repair_settled_attempt

    world = _commit_world(tmp_path, monkeypatch, phase="review")
    rl.settle_from_collected(world.data, world.record_id, [
        {**_raw("s1", "m1", parts=("change", "coupling")), "operation_id": "op-s1"},
        {**_raw("s2", "m2"), "operation_id": "op-s2"}])
    rows = [] if roster == "empty" else [{"slot_id": slot, "operation_id": f"op-{slot}", "status": "pending",
                                          "operation_state": "pending_dispatch"} for slot in ("s1", "s2")]
    update_state(world.data, lambda state: setattr(state.attempts[-1], "triad_raw_results", rows))
    before = _saved(world)

    assert repair_settled_attempt(world.data, world.data, before) is None
    with pytest.raises(ValueError, match="missing or no longer has its exact evidence"):
        resolve_commit_review_reference(world.ctx, _reference(before))
    assert _check_overlapping_review_attempt(world.ctx) is None and world.ctx._review_resume_pending
    assert settle_completed_round(world.ctx, world.data, repo_key=before.repo_key, tool_name="commit_reviewed",
                                  task_id=TASK, review_retry_key="round-1") is None
    after = _saved(world)
    assert (after.status, after.phase, after.triad_raw_results) == ("reviewing", "review", rows)


@pytest.mark.parametrize("adviser", [False, True])
def test_storage_holds_a_rounds_checkout_by_custody_until_settled_then_by_its_own_rules(tmp_path, monkeypatch,
                                                                                         adviser):
    """Storage's checkout prune reads the same custody fact. While the wave runs, its round's checkout is held as
    ``review_custody_active``; once the counted seats settle, that hold ends and Storage's own retention rules
    decide (here the author's task has not ended). A running additional adviser keeps it held. Only judged here:
    nothing is deleted."""
    import json

    from ouroboros import review_checking_prune
    from ouroboros.tools.review_change_custody import settle_collected_attempt
    from ouroboros.tools.review_subject import CHECKOUT_MARKER, checkout_token

    world = _commit_world(tmp_path, monkeypatch, adviser=adviser)
    entry = world.data / "review_checkouts" / checkout_token("round-1", task_id=TASK)
    entry.mkdir(parents=True)
    (entry / CHECKOUT_MARKER).write_text(json.dumps({"schema_version": 1, "task_id": TASK}), encoding="utf-8")

    def hold():
        return review_checking_prune._checkout_hold(world.data, entry, rounds=review_checking_prune._review_rounds(
            world.data), sources=set(), live=lambda _task: False, cutoff=0.0)

    assert hold() == "review_custody_active"
    if adviser:
        rl.settle_from_collected(world.data, world.record_id, [
            {**_raw("s1", "m1", parts=("change", "coupling")), "operation_id": "op-s1"},
            {**_raw("s2", "m2"), "operation_id": "op-s2"}])
        settle_collected_attempt(world.data, world.attempt, [_seat("s1"), _seat("s2")],
                                 rl.load_record(world.data, world.record_id))
    else:
        monkeypatch.setattr(review_pause, "_left_seats", lambda record: (world.attempt, [_seat("s1"), _seat("s2")], ""))
        assert review_pause._settled_aggregate(world.left) == "PASS"

    assert hold() == ("review_custody_active" if adviser else "task_not_terminal")
    assert entry.is_dir()


def _late_wait_write(world, rows, **kwargs):
    """A pending write by a call still holding its snapshot of this attempt's seats (``_finalize_pending_review``)."""
    from ouroboros.tools.commit_gate import _record_commit_attempt

    world.ctx._current_review_attempt_number = 1
    _record_commit_attempt(world.ctx, "candidate", kwargs.pop("status", "reviewing"), triad_raw_results=rows,
                           scope_raw_result={}, pre_review_fingerprint="fp-reviewed", review_retry_key="round-1",
                           **kwargs)


@pytest.mark.parametrize("case", ["late_wait", "late_wait_adviser_running", "cyber_terminal_write"])
def test_a_stale_pending_write_after_settlement_reopens_nothing(tmp_path, monkeypatch, case):
    """#2587: once a collector settled the counted seats, a write from a call that still holds its earlier pending
    snapshot of the same operations adds nothing: no second attempt row (a late one never expires), no reopened
    phase, no token back on a settled seat and no late flag raised by Cyber's terminal write. The settled FAIL stays,
    nothing holds the next candidate, a running adviser keeps its own custody and the counted verdict stays
    collectible at $0."""
    import copy

    from ouroboros import config
    from ouroboros.review_state import _attempt_has_active_review_custody, load_state, settle_completed_round
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt
    from ouroboros.tools.review_change_custody import settle_collected_attempt

    adviser = case == "late_wait_adviser_running"
    world = _commit_world(tmp_path, monkeypatch, adviser=adviser)
    stale = copy.deepcopy(world.attempt.triad_raw_results)
    seats = [_seat("s1"), _seat("s2", text=FAIL_TEXT)]
    if adviser:
        rl.settle_from_collected(world.data, world.record_id, [
            {**_raw("s1", "m1", parts=("change", "coupling")), "operation_id": "op-s1"},
            {**_raw("s2", "m2", answers={"change": _answer("change", "FAIL", [CRITICAL])}), "operation_id": "op-s2"}])
        settle_collected_attempt(world.data, world.attempt, seats, rl.load_record(world.data, world.record_id))
    else:
        monkeypatch.setattr(review_pause, "_left_seats", lambda record: (world.attempt, seats, ""))
        assert review_pause._settled_aggregate(world.left) == "FAIL"
    settled = _saved(world)

    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline("cyber_pro" if case == "cyber_terminal_write" else "pro")
    try:
        if case == "cyber_terminal_write":
            _late_wait_write(world, stale, status="reviewed", phase="review_only")
        else:
            _late_wait_write(world, stale, phase="late_wait", late_result_pending=True,
                             block_reason="review_late_result_pending")
    finally:
        config.reset_runtime_mode_baseline_for_tests()

    attempts = load_state(world.data).attempts
    assert [row.attempt for row in attempts] == [1], "no second row for a wave already settled"
    saved = attempts[0]
    assert (saved.status, saved.phase, saved.late_result_pending) == ("reviewed", "review_only", adviser)
    counted = [row for row in saved.triad_raw_results if row["slot_id"] != "a1"]
    assert all(row["operation_state"] == "settled" and not row.get("pending_invocation_id") for row in counted)
    assert counted[1]["raw_text"] == FAIL_TEXT and saved.triad_raw_results == settled.triad_raw_results
    assert _attempt_has_active_review_custody(saved) == adviser
    assert _check_overlapping_review_attempt(world.ctx) is None and not world.ctx._review_resume_pending
    reused = settle_completed_round(world.ctx, world.data, repo_key=saved.repo_key, tool_name="commit_reviewed",
                                    task_id=TASK, review_retry_key="round-1")
    assert reused is not None and reused.attempt == 1


@pytest.mark.parametrize("row_state", ["reviewing", "closed"])
def test_a_later_send_after_a_definite_refusal_is_recorded(tmp_path, monkeypatch, row_state):
    """A definite non-send is no outcome: when that same operation is later sent (a recorded choice sent on its
    repeat), the pending write that carries it is kept, so its custody is never hidden behind the refusal."""
    import copy

    from ouroboros.review_state import load_state, update_state

    world = _commit_world(tmp_path, monkeypatch)

    def refuse(state):
        row = state.attempts[-1]
        row.triad_raw_results[1].update(status="not_dispatched", operation_state="not_dispatched",
                                        late_result_pending=False, pending_invocation_id="")
        if row_state == "closed":
            row.triad_raw_results[0].update(status="ok", operation_state="settled", late_result_pending=False,
                                            pending_invocation_id="", raw_text=_two_part_pass())
            row.status, row.phase, row.late_result_pending = "reviewed", "review_only", False
    update_state(world.data, refuse)
    sent = copy.deepcopy(world.attempt.triad_raw_results)
    sent[1]["pending_invocation_id"] = "inv-s2-sent"

    _late_wait_write(world, sent if row_state == "reviewing" else sent[1:], phase="late_wait",
                     late_result_pending=True, block_reason="review_late_result_pending")

    carried = [row for attempt in load_state(world.data).get_active_attempts() for row in attempt.triad_raw_results
               if row["slot_id"] == "s2"]
    assert [row["pending_invocation_id"] for row in carried] == ["inv-s2-sent"], carried
    assert carried[0]["operation_state"] == "in_flight"


def test_a_recorded_choice_beside_a_running_adviser_is_this_rounds_open_work(tmp_path, monkeypatch):
    """A round whose counted seats settled while an additional adviser still runs, and whose record holds an explicit
    continuation a crash left unprojected, is still this task's open work for the next plain commit call
    (``decided_round_attempts``): the running adviser no longer hides it, so that call rejoins the recorded choice
    rather than buying a competing wave."""
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt
    from ouroboros.tools.review_change_custody import settle_collected_attempt

    world = _commit_world(tmp_path, monkeypatch, adviser=True)
    rl.settle_from_collected(world.data, world.record_id, [
        {**_raw("s1", "m1", parts=("change", "coupling")), "operation_id": "op-s1"},
        {**_raw("s2", "m2"), "operation_id": "op-s2"}])
    settle_collected_attempt(world.data, world.attempt, [_seat("s1"), _seat("s2")],
                             rl.load_record(world.data, world.record_id))

    def choose(payload):  # the explicit choice reached the record; its projection onto the attempt never ran
        rows = [dict(row) for row in payload["rows"]]
        for row in rows:
            if row.get("seat_id") == "s2":
                row.update(operation_id="op-s2b", status="pending", operation_state="in_flight", continuations=[
                    {"operation_id": "op-s2b", "supersedes": "op-s2", "invocation_id": "inv-s2b"}])
        return {**payload, "rows": rows}
    rl.revise_record(world.data, world.record_id, choose)

    assert _check_overlapping_review_attempt(world.ctx) is None
    assert [attempt.attempt for attempt in world.ctx._review_open_peers] == [1]


@pytest.mark.parametrize("next_reader", ["review_status_and_reference", "another_tasks_commit"])
def test_an_ended_authors_settled_wave_ends_its_review_phase(tmp_path, monkeypatch, next_reader):
    """Independent review, critical: the ended author's producer (``ended_left_review``) settles the wave's record and
    hands the same collected seats to the shared writer before it returns, so the reference ``review_status`` offers
    resolves and another task's next commit is not held by finished work. The FAIL stays the record's."""
    import json

    from ouroboros import config
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt, resolve_commit_review_reference
    from ouroboros.tools.preflight_review import _handle_review_status
    from ouroboros.tools.registry import ToolContext

    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline("pro")
    try:
        world = _commit_world(tmp_path, monkeypatch)
        monkeypatch.setattr(review_pause, "_left_seats", lambda record: (
            world.attempt, [_seat("s1"), _seat("s2", text=FAIL_TEXT)], ""))
        monkeypatch.setattr("ouroboros.late_review_marks.publish_late_review",
                            lambda root, mark: SimpleNamespace(status="saved"))

        assert review_pause.ended_left_review({**world.left, "left": "unwaited"}) == "saved"

        assert rl.load_record(world.data, world.record_id)["verdict"]["aggregate"] == "FAIL"
        saved = _saved(world)
        assert (saved.status, saved.phase, saved.late_result_pending) == ("reviewed", "review_only", False)
        if next_reader == "review_status_and_reference":
            status = json.loads(_handle_review_status(world.ctx, task_id=TASK))
            assert status["review_reference"]["review_record_id"] == world.record_id
            assert resolve_commit_review_reference(world.ctx, _reference(saved)).attempt == 1
        else:
            other = ToolContext(repo_dir=world.repo, drive_root=world.data, task_id="next-author")
            assert _check_overlapping_review_attempt(other) is None
    finally:
        config.reset_runtime_mode_baseline_for_tests()


def test_a_left_waves_counted_outcome_ends_its_phase_at_its_producer_while_an_adviser_runs(tmp_path, monkeypatch):
    """Independent review, critical: the left-wave producer (``_settled_aggregate``) projects counted completion through
    the shared writer even though its record stays pending for a running additional adviser, and sends no whole-wave
    notice yet. A fresh context then collects the counted FAIL at $0 instead of buying a wave, the overlap check holds
    nothing, and the adviser keeps its own custody and checkout."""
    from ouroboros import review_checking_prune
    from ouroboros.review_state import settle_completed_round
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_subject import checkout_token

    world = _commit_world(tmp_path, monkeypatch, adviser=True)
    running = {**world.attempt.triad_raw_results[-1], "surface": "multi_model_review", "parts": ["change"]}
    seats = [_seat("s1"), _seat("s2", text=FAIL_TEXT), running]
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (world.attempt, seats, ""))

    assert review_pause._settled_aggregate(world.left) == "", "no whole-wave notice while the adviser runs"

    record = rl.load_record(world.data, world.record_id)
    assert record["state"] == "pending" and record["verdict"]["aggregate"] == "FAIL"
    saved = _saved(world)
    adviser = next(row for row in saved.triad_raw_results if row["slot_id"] == "a1")
    assert (saved.status, saved.phase, saved.late_result_pending) == ("reviewed", "review_only", True)
    assert adviser["operation_state"] == "in_flight" and adviser["pending_invocation_id"] == "inv-a1"
    fresh = ToolContext(repo_dir=world.repo, drive_root=world.data, task_id=TASK)
    assert _check_overlapping_review_attempt(fresh) is None and not fresh._review_resume_pending
    reused = settle_completed_round(fresh, world.data, repo_key=saved.repo_key, tool_name="commit_reviewed",
                                    task_id=TASK, review_retry_key="round-1", exclude_attempt=2)
    assert reused is not None and reused.attempt == 1, "the counted FAIL is collected at $0"
    active, _owners = review_checking_prune._review_rounds(world.data)
    assert checkout_token("round-1", task_id=TASK) in active, "the running adviser keeps its checkout"


def test_a_stale_pending_write_after_history_compaction_opens_no_row_that_would_hold_the_next_candidate(
        tmp_path, monkeypatch):
    """Independent review: once the settled row is compacted out of the newest 50, a late pending write from a call
    that still holds its old snapshot finds no rows on it. Its bound settled record now stands for them, so no
    second row opens. Such a row would never expire (it is late), would make this task's next, different intent
    rejoin it and wait, and would keep Storage holding the round's checkout."""
    import copy

    from ouroboros import review_checking_prune
    from ouroboros.review_state import CommitAttemptRecord, load_state, update_state
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt
    from ouroboros.tools.review_subject import checkout_token

    world = _commit_world(tmp_path, monkeypatch)
    snapshot = copy.deepcopy(world.attempt.triad_raw_results)
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (
        world.attempt, [_seat("s1"), _seat("s2", text=FAIL_TEXT)], ""))
    assert review_pause._settled_aggregate(world.left) == "FAIL"

    def add_history(state):  # ordinary bounded history: fifty newer rows push the settled row out
        for number in range(2, 52):
            state.record_attempt(CommitAttemptRecord(
                ts=f"2026-11-04T{number // 60:02}:{number % 60:02}:00+00:00", commit_message="another review",
                status="reviewed", phase="review_only", paid=True, task_id=f"other-task-{number}",
                repo_key=world.attempt.repo_key, tool_name="commit_reviewed", attempt=1))
    update_state(world.data, add_history)
    assert next(row for row in load_state(world.data).attempts if row.task_id == TASK).raw_stripped
    world.ctx._current_review_record_id = world.record_id

    _late_wait_write(world, snapshot, phase="late_wait", late_result_pending=True,
                     block_reason="review_late_result_pending")

    assert [row.attempt for row in load_state(world.data).attempts if row.task_id == TASK] == [1]
    assert _check_overlapping_review_attempt(world.ctx) is None and not world.ctx._review_resume_pending
    active, _owners = review_checking_prune._review_rounds(world.data)
    assert checkout_token("round-1", task_id=TASK) not in active
    assert rl.load_record(world.data, world.record_id)["verdict"]["aggregate"] == "FAIL"


def _state_writes(monkeypatch):
    """Every actual write of the shared review-state file from here on (a save its no-op comparison skipped is none)."""
    from ouroboros import review_state

    writes, real = [], review_state.write_text_atomic

    def counting(path, *args, **kwargs):
        if pathlib.Path(path).name == "advisory_review.json":
            writes.append(path)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(review_state, "write_text_atomic", counting)
    return writes


def test_an_ended_authors_waiting_wave_rewrites_its_row_only_when_a_seat_closes(tmp_path, monkeypatch):
    """Independent review (perf_lifecycle): the 300 s owed-duty replay of an ended author's wave that still waits for a
    counted seat hands the closed seat to the shared writer on every pass. Only the pass that closes it writes the
    review state; the later passes, which change nothing, write nothing. The running seat keeps its token."""
    world = _commit_world(tmp_path, monkeypatch)
    running = {**world.attempt.triad_raw_results[1], "surface": "multi_model_review", "parts": ["change"]}
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (world.attempt, [_seat("s1"), running], ""))
    monkeypatch.setattr("ouroboros.late_review_marks.publish_late_review", lambda root, mark: SimpleNamespace(status="saved"))
    writes = _state_writes(monkeypatch)

    statuses, per_pass = [], []
    for _ in range(3):
        before = len(writes)
        statuses.append(review_pause.ended_left_review({**world.left, "left": "unwaited"}))
        per_pass.append(len(writes) - before)

    assert statuses == ["waiting"] * 3 and per_pass == [1, 0, 0], (statuses, per_pass)
    saved = _saved(world)
    assert (saved.status, saved.phase) == ("reviewing", "late_wait")
    assert [row.get("pending_invocation_id") for row in saved.triad_raw_results] == [None, "inv-s2"]


def test_a_repeated_collection_of_seats_with_named_evidence_costs_no_state_rewrite(tmp_path, monkeypatch):
    """The saved row keeps a seat's named-evidence inventory as a reference to its producer payload; a repeated
    collection brings the full inventory back, which saves to the same bytes, so nothing is rewritten."""
    from ouroboros.tools.review_change_custody import settle_collected_attempt

    world = _commit_world(tmp_path, monkeypatch, adviser=True)
    rl.settle_from_collected(world.data, world.record_id, [
        {**_raw("s1", "m1", parts=("change", "coupling")), "operation_id": "op-s1"},
        {**_raw("s2", "m2"), "operation_id": "op-s2"}])
    record = rl.load_record(world.data, world.record_id)
    usage = {"checking_evidence": {"named_evidence": [{"path": "ouroboros/a.py", "lines": "1-9"}]}}
    seats = [_seat(slot, response_ref=f"obs:{slot}", usage=usage) for slot in ("s1", "s2")]
    settle_collected_attempt(world.data, world.attempt, seats, record)
    saved = _saved(world)
    held = [row for row in saved.triad_raw_results if row.get("slot_id") in ("s1", "s2")]
    assert held and all(set(row["usage"]["checking_evidence"]["named_evidence"]) == {
        "retained_in", "sha256", "bytes", "file_count"} for row in held)
    writes = _state_writes(monkeypatch)

    settle_collected_attempt(world.data, world.attempt, seats, record)

    assert writes == [] and _saved(world) == saved


def test_a_counted_outcome_beside_a_running_adviser_costs_no_state_rewrite(tmp_path, monkeypatch):
    """Independent review (perf_lifecycle): once counted completion is projected while an additional adviser still
    runs, every commit call's overlap check and a repeated collection of the same seats leave the review state
    unwritten; the row stays exactly as projected and the adviser keeps its custody."""
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt
    from ouroboros.tools.review_change_custody import settle_collected_attempt

    world = _commit_world(tmp_path, monkeypatch, adviser=True)
    rl.settle_from_collected(world.data, world.record_id, [
        {**_raw("s1", "m1", parts=("change", "coupling")), "operation_id": "op-s1"},
        {**_raw("s2", "m2"), "operation_id": "op-s2"}])
    record = rl.load_record(world.data, world.record_id)
    settle_collected_attempt(world.data, world.attempt, [_seat("s1"), _seat("s2")], record)
    projected = _saved(world)
    assert (projected.status, projected.phase, projected.late_result_pending) == ("reviewed", "review_only", True)
    writes = _state_writes(monkeypatch)

    for _ in range(3):
        assert _check_overlapping_review_attempt(world.ctx) is None and not world.ctx._review_resume_pending
    settle_collected_attempt(world.data, world.attempt, [_seat("s1"), _seat("s2")], record)

    assert writes == [] and _saved(world) == projected
    adviser = next(row for row in projected.triad_raw_results if row["slot_id"] == "a1")
    assert adviser["operation_state"] == "in_flight" and adviser["pending_invocation_id"] == "inv-a1"


@pytest.mark.parametrize("stale", ["counted", "scope", "adviser"])
def test_a_projected_row_whose_closed_seat_kept_its_token_holds_nothing(tmp_path, monkeypatch, stale):
    """An old writer's row already in the settled projection whose closed seat (counted, on the scope roster, or a
    finished adviser its record confirms) kept its start token holds nothing (#2710: the seat's own settled
    operation outranks the leftover token), so the overlap check neither waits on it nor rewrites the state."""
    from ouroboros.review_state import _attempt_has_active_review_custody, _attempt_review_roster_rows, update_state
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt

    world = _commit_world(tmp_path, monkeypatch, phase="review", adviser=stale == "adviser",
                          scope="listed" if stale == "scope" else False)
    rl.settle_from_collected(world.data, world.record_id, [
        {**_raw("s1", "m1", parts=("change", "coupling")), "operation_id": "op-s1"},
        {**_raw("s2", "m2"), "operation_id": "op-s2"}, *([{**_raw("a1", "m3"), "operation_id": "op-a1"}]
                                                           if stale == "adviser" else [])])

    def old_projection(state):  # each seat terminal, its token kept, the row projected
        row = state.attempts[-1]
        for seat in _attempt_review_roster_rows(row):
            seat.update(operation_state="settled", late_result_pending=False, status="responded", raw_text="[]")
        row.status, row.phase, row.late_result_pending = "reviewed", "review_only", False
    update_state(world.data, old_projection)
    assert not _attempt_has_active_review_custody(_saved(world))
    writes = _state_writes(monkeypatch)

    for _ in range(2):
        assert _check_overlapping_review_attempt(world.ctx) is None and not world.ctx._review_resume_pending

    kept = _saved(world)
    assert writes == [] and not _attempt_has_active_review_custody(kept), writes
    assert all(row.get("pending_invocation_id") for row in _attempt_review_roster_rows(kept)), "nothing was rewritten"
    assert (kept.status, kept.phase) == ("reviewed", "review_only")


@pytest.mark.parametrize("adviser", ["none", "settled", "running", "different_operation"])
def test_storages_checkout_pass_repairs_an_old_writers_finished_row_by_its_exact_record(tmp_path, monkeypatch, adviser):
    """#2710: before it judges custody the checkout pass repairs a finished row an old writer left reviewing with
    closed seats' start tokens, from its exact settled record, in one locked write (the next pass writes nothing), so
    Storage's own rules decide its checkout (here the author's task still runs). An adviser its attempt still holds
    in flight keeps the review hold whatever its record says; the owner's Free listing only reads."""
    import json

    from ouroboros import review_checking_prune
    from ouroboros.review_state import load_state, update_state
    from ouroboros.tools.review_subject import CHECKOUT_MARKER, checkout_token

    running = adviser in {"running", "different_operation"}
    world = _commit_world(tmp_path, monkeypatch, phase="review", adviser=adviser != "none")
    _prefix_shape(world)
    if running:  # the old writer merged terminal seats only: a seat still in flight kept its row
        update_state(world.data, lambda state: state.attempts[-1].triad_raw_results[-1].update(
            operation_state="in_flight", late_result_pending=True, status="error"))
    if adviser in {"settled", "different_operation"}:
        rl.settle_from_collected(world.data, world.record_id, [
            {**_raw("a1", "m3"), "operation_id": "op-a1"}])
    if adviser == "different_operation":
        def change_operation(payload):
            rows = [dict(row) for row in payload["rows"]]
            next(row for row in rows if row["seat_id"] == "a1")["operation_id"] = "op-a1-successor"
            return {**payload, "rows": rows}
        rl.revise_record(world.data, world.record_id, change_operation)
    entry = world.data / "review_checkouts" / checkout_token("round-1", task_id=TASK)
    entry.mkdir(parents=True)
    (entry / CHECKOUT_MARKER).write_text(json.dumps({"schema_version": 1, "task_id": TASK}), encoding="utf-8")
    before = load_state(world.data).attempts
    assert entry.name in review_checking_prune._review_rounds(world.data)[0], "the old row reads as custody"
    assert load_state(world.data).attempts == before, "Free's listing writes nothing"
    writes = _state_writes(monkeypatch)

    holds, per_pass = [], []
    for _ in range(2):
        count = len(writes)
        holds.append(review_checking_prune._checkout_hold(world.data, entry, rounds=review_checking_prune._review_rounds(
            world.data, repair=True), sources=set(), live=lambda: set(), cutoff=0.0))
        per_pass.append(len(writes) - count)

    assert holds == ["review_custody_active" if running else "task_not_terminal"] * 2 and entry.is_dir(), holds
    assert per_pass == [1, 0], per_pass
    repaired = _saved(world)
    assert (repaired.status, repaired.phase) == ("reviewed", "review_only")
    seats = {row["slot_id"]: row for row in repaired.triad_raw_results}
    assert not any(seats[slot].get("pending_invocation_id") for slot in ("s1", "s2"))
    if running:
        assert seats["a1"]["operation_state"] == "in_flight" and seats["a1"]["pending_invocation_id"] == "inv-a1"

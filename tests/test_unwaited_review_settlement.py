"""A paid review its author stopped waiting for settles its own record (owner 17.10.2026, Q3-2).

A wave that returns to its author with seats still running, and no addressed message behind it, owes
its settlement (one duty row of ``late_review_marks``): once its seats are collectable at $0 and its
author's task ended, the record settles to its verdict without a re-issue and the outcome is kept in
the author's room. A wave its author collected itself while running needs no mark.
"""
from __future__ import annotations

from types import SimpleNamespace

from ouroboros import review_ledger as rl
from ouroboros import review_pause
from tests.test_review_ledger import _facts, _raw

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


def test_an_unwaited_review_its_author_collected_while_running_needs_no_mark(tmp_path, monkeypatch):
    data = _data(tmp_path)
    record_id = _record(data, pending=False)  # the author's own re-issue already settled it
    row = _owe(data, record_id)
    published = []
    monkeypatch.setattr(review_pause, "_left_seats", lambda record: (None, _seats(), ""))
    monkeypatch.setattr("ouroboros.late_review_marks.publish_late_review",
                        lambda root, mark: published.append(mark) or SimpleNamespace(status="saved"))

    assert review_pause.ended_left_review(row["left_review"]) == "collected"
    assert published == [] and rl.load_record(data, record_id)["revision"] == 1


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

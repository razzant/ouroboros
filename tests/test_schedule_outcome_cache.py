"""Outcome observation keeps terminal facts honest and never republishes quiet words."""
import copy

import pytest

from ouroboros import task_results
from ouroboros.task_results import write_task_result
from supervisor import queue_schedules as store, schedule_occurrence as occurrence
from tests.test_schedule_occurrence import q, _row, _rows  # noqa: F401


def history(host, task_id="last", **extra):
    row = {"id": "s1", "enabled": False, "trigger": {"type": "once", "run_at": "2099-01-01T00:00:00+00:00"},
           "completed_at": "2099-01-01T00:00:00+00:00", "last_task_id": task_id, "task": {"text": "Check"}, **extra}
    store._write_scheduled_tasks({"tasks": [row]}, host.root)
    return row


SENTENCE = "Checked the disk: 30 GiB free, 30 less than ninety minutes ago; " * 3  # longer than an answer's 160-character end


@pytest.mark.parametrize("status,silent,notice,summary,sent,quiet,line", [
    ("completed", False, "", "", False, False, "Own final line"),
    ("failed", False, "", "", False, False, "Own final line"),
    ("cancelled", False, "", "", False, False, "Own final line"),
    ("rejected_duplicate", False, "", "", False, False, "Own final line"),
    ("completed", True, "", "", False, True, ""),
    ("failed", True, "Host notice", "", False, False, "Host notice"),
    ("failed", False, "Host notice", "", False, False, "Host notice"),
    # The run's own sentence is its line, whole; a host notice outranks it; a message already sent
    # makes the finish not quiet while the internal note still never leaves the result.
    ("completed", True, "", SENTENCE, False, True, SENTENCE.strip()),
    ("completed", False, "", SENTENCE, False, False, SENTENCE.strip()),
    ("completed", True, "Host notice", SENTENCE, False, False, "Host notice"),
    ("completed", True, "", "", True, False, ""),
], ids=["spoken", "failed", "cancelled", "duplicate", "quiet", "quiet-notice", "spoken-notice",
        "quiet-sentence", "spoken-sentence", "quiet-notice-sentence", "quiet-after-message"])
def test_settled_outcomes_are_strictly_read_off_lock_on_disabled_consumed_rows(q, monkeypatch, status, silent, notice, summary, sent, quiet, line):  # noqa: F811
    history(q)
    write_task_result(q.root, "last", status, result="Private first line\nOwn final line", wake_silent=silent,
                      terminal_host_notice=notice, _terminal_observed=True,
                      **{k: v for k, v in (("wake_summary", summary), ("owner_message_sent", sent)) if v})
    reader, reads = task_results.load_task_result, []
    def read(root, tid, **kw):
        reads.append((tid, kw, q.queue._queue_lock._is_owned()))
        return reader(root, tid, **kw)
    monkeypatch.setattr(task_results, "load_task_result", read)
    q.queue.check_scheduled_tasks()
    outcome = _rows(q)["s1"]["last_outcome"]
    assert outcome == {"task_id": "last", "state": status, "quiet": quiet, "notice": bool(notice),
                       "ended_at": outcome["ended_at"], "line": line}
    assert outcome["ended_at"] and reads == [("last", {"strict": True}, False)]
    path = q.root / "state/scheduled_tasks.json"
    before = path.read_bytes(), path.stat().st_mtime_ns
    q.queue.init(q.root)  # a process restart needs no in-memory cache to remember the result
    q.queue.check_scheduled_tasks()
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before and len(reads) == 1


@pytest.mark.parametrize("initial", ["interrupted", "running", "scheduled", "missing", "error"])
def test_unobserved_results_retry_under_the_same_id(q, monkeypatch, initial):  # noqa: F811
    history(q)
    if initial not in {"missing", "error"}:
        write_task_result(q.root, "last", initial, result="Not final")
    reader = task_results.load_task_result
    if initial == "error":
        def unreadable(*a, **kw):
            raise OSError("read unavailable")
        monkeypatch.setattr(task_results, "load_task_result", unreadable)
    q.queue.check_scheduled_tasks()
    assert "last_outcome" not in _rows(q)["s1"]
    monkeypatch.setattr(task_results, "load_task_result", reader)
    write_task_result(q.root, "last", "completed", result="Settled")
    q.queue.check_scheduled_tasks()
    assert _rows(q)["s1"]["last_outcome"]["line"] == "Settled"
    assert _rows(q)["s1"]["last_outcome"]["ended_at"] is None  # updated_at is not a finish time


def test_result_observed_for_an_old_task_cannot_overwrite_a_new_last_run(q, monkeypatch):  # noqa: F811
    history(q)
    write_task_result(q.root, "last", "completed", result="Old")
    reader = task_results.load_task_result
    def read(root, tid, **kw):
        result = reader(root, tid, **kw)
        data = store.load_schedule_store(root)
        data["tasks"][0]["last_task_id"] = "new"
        store._write_scheduled_tasks(data, root)
        return result
    monkeypatch.setattr(task_results, "load_task_result", read)
    q.queue.check_scheduled_tasks()
    assert _rows(q)["s1"]["last_task_id"] == "new" and "last_outcome" not in _rows(q)["s1"]
    monkeypatch.setattr(task_results, "load_task_result", reader)
    write_task_result(q.root, "new", "completed", result="New")
    q.queue.check_scheduled_tasks()
    assert _rows(q)["s1"]["last_outcome"]["line"] == "New"


def test_due_previous_run_fact_is_set_once_then_cleared(q):  # noqa: F811
    _row(q, intent={"kind": "system_repo"}, cron=True)
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert occurrence.record_dispatch_possible(task)
    data = store.load_schedule_store(q.root)
    data["tasks"][0]["next_run_at"] = "2099-01-01T00:00:00+00:00"
    store._write_scheduled_tasks(data, q.root)
    q.queue.check_scheduled_tasks()  # settle admission, but the next cron point is still future
    assert "previous_run_open" not in _rows(q)["s1"]
    data = store.load_schedule_store(q.root)
    data["tasks"][0]["next_run_at"] = "2000-01-01T00:00:00+00:00"
    store._write_scheduled_tasks(data, q.root)
    q.queue.check_scheduled_tasks()
    fact = _rows(q)["s1"]["previous_run_open"]
    assert fact["task_id"] == task["id"] and fact["since"]
    q.queue.check_scheduled_tasks()
    assert _rows(q)["s1"]["previous_run_open"] == fact and len(q.pending) == 1
    q.pending.clear()
    write_task_result(q.root, task["id"], "completed", result="Previous result")
    q.queue.check_scheduled_tasks()
    row = _rows(q)["s1"]
    assert "previous_run_open" not in row
    [next_task] = q.pending
    timing = next_task["metadata"]["schedule_occurrence"]
    assert timing["previous_task_id"] == task["id"]
    assert timing["previous"]["line"] == "Previous result" and timing["previous"]["state"] == "completed"


@pytest.mark.parametrize("cached_id", ["previous", "older"])
def test_new_claim_copies_only_its_previous_runs_outcome(q, cached_id):  # noqa: F811
    _row(q, intent={"kind": "system_repo"})
    data = store.load_schedule_store(q.root)
    row = data["tasks"][0]
    row["last_task_id"] = "previous"
    row["last_outcome"] = {"task_id": cached_id, "state": "completed", "quiet": True, "line": ""}
    store._write_scheduled_tasks(data, q.root)
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    fact = task["metadata"]["schedule_occurrence"]
    assert fact["previous_task_id"] == "previous"
    assert fact["previous"] == (row["last_outcome"] if cached_id == "previous" else {
        "task_id": "previous", "state": "pending_observation"})
    frozen = copy.deepcopy(task)
    # A republish claim carries its schedule row beside the frozen task (occurrence.view).
    prepared = occurrence._prepare(occurrence.view(_rows(q)["s1"], frozen))
    assert prepared["task"] == frozen and "hold" not in prepared


def test_related_followup_outcome_is_observed_even_when_policy_is_ineligible(q):  # noqa: F811
    history(q, source="task_followup", followup_relation={"kind": "related"},
            followup_origin={"task_id": "missing-origin", "root_task_id": "missing-origin"})
    write_task_result(q.root, "last", "completed", result="Own result" * 30)
    q.queue.check_scheduled_tasks()
    assert _rows(q)["s1"]["last_outcome"]["line"] == ("Own result" * 30)[:160]


def test_outcome_and_liveness_are_host_fields_and_not_fingerprint_fields(q):  # noqa: F811
    _row(q, intent={"kind": "system_repo"})
    row = _rows(q)["s1"]
    forged = {"last_outcome": {"task_id": "fake", "state": "completed"}, "previous_run_open": {"task_id": "fake"}}
    saved = q.queue.upsert_scheduled_task({**row, **forged})
    assert all(key not in saved for key in forged)
    assert occurrence.fingerprint(row) == occurrence.fingerprint({**row, **forged})


@pytest.mark.parametrize("due", [False, True])
def test_previous_run_fact_tracks_that_task_even_if_another_run_is_queued(q, due):  # noqa: F811
    history(q, previous_run_open={"task_id": "last", "since": "then"}, **({
        "enabled": True, "completed_at": "", "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
    } if due else {}))
    q.pending.extend([{"id": "last"}, {"id": "new", "metadata": {"schedule_id": "s1"}}])
    q.queue.check_scheduled_tasks()
    assert _rows(q)["s1"]["previous_run_open"]["task_id"] == "last"
    q.pending.pop(0)
    q.queue.check_scheduled_tasks()
    assert "previous_run_open" not in _rows(q)["s1"]


def test_unknown_queue_state_cannot_claim_a_live_previous_run(q, monkeypatch):  # noqa: F811
    _row(q, intent={"kind": "system_repo"})
    data = store.load_schedule_store(q.root)
    data["tasks"][0]["last_task_id"] = "previous"
    store._write_scheduled_tasks(data, q.root)
    monkeypatch.setattr(store, "_schedule_running_or_queued", lambda *a, **kw: None)
    q.queue.check_scheduled_tasks()
    assert "previous_run_open" not in _rows(q)["s1"] and not q.pending
    monkeypatch.setattr(store, "_schedule_running_or_queued", lambda *a, **kw: True)
    q.queue.check_scheduled_tasks()
    assert _rows(q)["s1"]["previous_run_open"]["task_id"] == "previous" and not q.pending

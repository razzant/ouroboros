"""A call's wire evidence never makes the call loop wait on the disk.

27.10: an open-append-close for every 20 ms frame on the call thread stood 132 s of a 2.3-minute OpenAI
call under disk load; the browser's uplink backed up and barge-ins dropped seconds of the owner's speech.
"""
import json
import threading
import time

from ouroboros.live_conversation import WireJournal


class SlowPath:
    """A path whose every open takes 50 ms, as under the disk load of the owner's night calls."""

    def __init__(self, real):
        self.real, self.threads = real, set()

    def open(self, mode, encoding=None):
        self.threads.add(threading.current_thread().name)
        time.sleep(.05)
        return self.real.open(mode, encoding=encoding)


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_appends_never_wait_for_the_disk_and_close_keeps_every_record_in_order(tmp_path):
    path = SlowPath(tmp_path / "wire.jsonl")
    journal = WireJournal(path)
    started = time.monotonic()
    for number in range(50):
        journal.append({"n": number})
    assert time.monotonic() - started < 1.0  # writing inline would take 2.5 s here
    journal.close()
    assert [row["n"] for row in lines(tmp_path / "wire.jsonl")] == list(range(50))
    assert threading.current_thread().name not in path.threads  # the caller never opened the file
    journal.append({"n": "late"})  # a record after close is written at once, after the others
    assert lines(tmp_path / "wire.jsonl")[-1] == {"n": "late"}


def test_a_journal_that_saw_no_record_closes_without_a_file(tmp_path):
    journal = WireJournal(tmp_path / "wire.jsonl")
    journal.close()
    assert not (tmp_path / "wire.jsonl").exists()


def test_flush_waits_for_the_records_so_far_and_the_journal_stays_open(tmp_path):
    journal = WireJournal(SlowPath(tmp_path / "wire.jsonl"))
    journal.append({"n": 1})
    journal.flush()
    assert lines(tmp_path / "wire.jsonl") == [{"n": 1}]  # a reader during the call sees what came before
    journal.append({"n": 2})
    journal.close()
    assert lines(tmp_path / "wire.jsonl") == [{"n": 1}, {"n": 2}]


def test_the_writer_stops_by_itself_once_nothing_more_arrives(tmp_path):
    """No owner has to stop it: a measuring tool or a test that never ends a call leaks no thread."""
    journal = WireJournal(tmp_path / "wire.jsonl")
    journal.append({"n": 1})
    deadline = time.monotonic() + 2
    while any(t.name == "call-wire-journal" for t in threading.enumerate()) and time.monotonic() < deadline:
        time.sleep(.02)
    assert not any(t.name == "call-wire-journal" for t in threading.enumerate())
    assert lines(tmp_path / "wire.jsonl") == [{"n": 1}]
    journal.append({"n": 2})  # a later record starts a new writer
    journal.close()
    assert lines(tmp_path / "wire.jsonl") == [{"n": 1}, {"n": 2}]

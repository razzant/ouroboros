"""Worker readiness identity: which process a ``worker_ready`` row names (owner, 14 Oct 21:09 #1).

What is pinned, on the fake slots and the real seam of tests/test_worker_readiness.py (the one
child spawned here is the last test's real worker loop, on macOS):

* births are compared only when both the parent and the child measured one, else the pid decides
  and a parent without a birth adopts the child's; mismatching births never open the slot;
* kernel births of one validated kind (``kernel_birth``, #2382) are measured births that decide
  first, whatever zone spelled ``ps -o lstart=``, even where a missing spelling would fall back to
  the pid; a row or answer without a comparable one keeps the rule above, so an older writer's
  spelling read in another zone is still refused (unfixed here: the birth-else-pid rule stands);
* a stale row of a reused pid never names a slot whose birth was measured; an equal kernel birth
  wins over an exact spelling; without a parent birth the newest row for the pid wins (an earlier
  holder wrote its row before it died);
* the seam (the spawn's readiness thread) ledgers a birthless slot itself, never a measured one
  (tests/test_worker_demand.py pins that the supervisor loop ledgers nothing);
* the idle probe's answer follows the same rule.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from ouroboros.utils import append_jsonl
from tests.test_worker_readiness import _READY_ROW, _booting_slot, _FakeProc, _rows  # the seam's fake slots
from tests.test_worker_readiness import pool, seam  # noqa: F401 -- the real seam fixtures

_KERNEL = "darwin-start:1793143271.251840"
_REUSED = "darwin-start:1793143271.251841"  # another process at the pid, started in the same ps second
_ZONE = "Wed Oct 28 03:53:55 2026"  # the parent's "test-birth", spelled in another zone
_LINUX, _LINUX_OTHER = "812." + "0123456789abcdef" * 2, "811." + "0123456789abcdef" * 2
_FILETIME, _FILETIME_OTHER = "win-filetime:134058912345678901", "win-filetime:134058912345678902"


@pytest.fixture
def unmeasured_parent(monkeypatch):
    """The parent's own birth read fails (its ledger write raises): the slot keeps no birth."""
    from ouroboros import process_custody

    attempted = []

    def fail(_root, *, pid, **_kwargs):
        attempted.append(pid)
        raise OSError("ps unavailable")

    monkeypatch.setattr(process_custody, "record_process", fail)
    return attempted


@pytest.mark.parametrize("parent,child", [
    ("", "child-birth"), ("test-birth", ""), ("", ""), ("test-birth", "test-birth"),
], ids=["absent_parent_birth", "absent_child_birth", "both_absent", "matching"])
def test_readiness_compares_births_only_when_both_sides_measured_one(
        pool, seam, unmeasured_parent, parent, child):  # noqa: F811
    slot = _booting_slot(pool, 0, 5001)
    slot.process_birth = parent
    append_jsonl(seam.events, {**_READY_ROW, "process_birth": child})

    seam.run({0: slot}, seam.cursor, 1)

    assert seam.killed == [] and seam.respawned == []
    assert pool.lifecycle.worker_ready(slot), "the child's own row opened the slot"
    assert slot.process_birth == (parent or child), "a parent without a birth adopts the child's own"
    assert slot.ready_identity == (5001, parent or child)
    assert unmeasured_parent == ([] if parent else [5001]), "only a birthless slot is ledgered, by its seam"


def test_mismatching_births_never_open_the_slot(pool, seam, unmeasured_parent):  # noqa: F811
    slot = _booting_slot(pool, 0, 5001)
    append_jsonl(seam.events, {**_READY_ROW, "process_birth": "another-process"})

    seam.run({0: slot}, seam.cursor, 1)

    assert not pool.lifecycle.worker_ready(slot) and slot.reaping and slot.process_birth == "test-birth"
    assert seam.killed == [5001] and seam.respawned == [(0, {"ready_attempt": 2})]


@pytest.mark.parametrize("order", ["stale_first", "own_first"])
def test_a_stale_row_of_a_reused_pid_never_names_a_slot_whose_birth_was_measured(pool, seam, order):  # noqa: F811
    slot = _booting_slot(pool, 0, 5001)
    stale = {**_READY_ROW, "process_birth": "earlier-holder"}
    for row in ([stale, _READY_ROW] if order == "stale_first" else [_READY_ROW, stale]):
        append_jsonl(seam.events, row)

    seam.run({0: slot}, seam.cursor, 1)

    assert pool.lifecycle.worker_ready(slot) and slot.ready_identity == (5001, "test-birth")
    assert seam.killed == []


def test_pid_reuse_without_a_parent_birth_adopts_the_newest_row_for_the_pid(
        pool, seam, unmeasured_parent):  # noqa: F811
    """An earlier holder of a reused pid died, and so wrote its row, before this child received it."""
    slot = _booting_slot(pool, 0, 5001)
    slot.process_birth = ""
    append_jsonl(seam.events, {**_READY_ROW, "process_birth": "earlier-holder"})
    append_jsonl(seam.events, {**_READY_ROW, "process_birth": "child-birth"})

    seam.run({0: slot}, seam.cursor, 1)

    assert slot.process_birth == "child-birth" and slot.ready_identity == (5001, "child-birth")
    assert pool.lifecycle.worker_ready(slot)


@pytest.mark.parametrize("slot_birth,event_pid,event_birth,confirmed", [
    ("test-birth", 5001, "test-birth", True),
    ("test-birth", 5001, "", True),
    ("", 5001, "", True),
    ("test-birth", 5001, "another-process", False),
    ("test-birth", 4999, "test-birth", False),
], ids=["matching", "absent_child_birth", "both_absent", "mismatching", "foreign_pid"])
def test_the_idle_probe_answer_compares_births_only_when_both_are_known(
        pool, slot_birth, event_pid, event_birth, confirmed):  # noqa: F811
    lifecycle, workers = pool.lifecycle, pool.workers
    slot = workers.Worker(wid=0, proc=_FakeProc(5001), in_q=MagicMock(), busy_task_id=None,
                          process_birth=slot_birth, ready_identity=(5001, slot_birth), quiescence_token="probe")
    workers.WORKERS[0] = slot

    lifecycle.handle_worker_quiescent(
        {"worker_id": 0, "pid": event_pid, "process_birth": event_birth, "token": "probe"}, workers)

    assert slot.quiescence_confirmed is confirmed


@pytest.mark.parametrize("row_kernel,opens", [(_KERNEL, True), (None, False)], ids=["kernel_row", "legacy_row"])
def test_an_equal_kernel_birth_opens_the_slot_whatever_zone_spelled_the_birth(pool, seam,  # noqa: F811
                                                                              row_kernel, opens):
    """#2382: the parent and the child read ``ps`` in different zones; the legacy row is refused as before."""
    slot = _booting_slot(pool, 0, 5001)
    slot.kernel_birth = _KERNEL
    append_jsonl(seam.events, {**_READY_ROW, "process_birth": _ZONE,
                               **({"kernel_birth": row_kernel} if row_kernel else {})})

    seam.run({0: slot}, seam.cursor, 1)

    assert pool.lifecycle.worker_ready(slot) is opens
    assert slot.ready_identity == ((5001, "test-birth") if opens else None), "the parent's own spelling stays"
    assert seam.killed == ([] if opens else [5001])


def test_a_different_kernel_birth_never_opens_the_slot_even_with_its_exact_spelling(pool, seam):  # noqa: F811
    slot = _booting_slot(pool, 0, 5001)
    slot.kernel_birth = _KERNEL
    append_jsonl(seam.events, {**_READY_ROW, "kernel_birth": _REUSED})

    seam.run({0: slot}, seam.cursor, 1)

    assert not pool.lifecycle.worker_ready(slot) and seam.killed == [5001]


@pytest.mark.parametrize("order", ["spelling_first", "kernel_first"])
def test_an_equal_kernel_birth_wins_over_an_exact_spelling_without_one(pool, seam, order):  # noqa: F811
    slot = _booting_slot(pool, 0, 5001)
    slot.kernel_birth = _KERNEL
    spelled = {**_READY_ROW, "git_sha": "b" * 40}  # the exact spelling from a writer without a kernel birth
    own = {**_READY_ROW, "process_birth": _ZONE, "kernel_birth": _KERNEL}
    for row in ([spelled, own] if order == "spelling_first" else [own, spelled]):
        append_jsonl(seam.events, row)

    seam.run({0: slot}, seam.cursor, 1)

    assert pool.lifecycle.worker_ready(slot)
    assert [row["observed_sha"] for row in _rows(seam.supervisor, "worker_sha_verify")] == ["abc123"]


@pytest.mark.parametrize("slot_birth,slot_kernel,event,now,before", [
    ("test-birth", _KERNEL, {"process_birth": _ZONE, "kernel_birth": _KERNEL}, True, False),
    ("test-birth", _KERNEL, {"process_birth": "test-birth", "kernel_birth": _REUSED}, False, True),
    ("test-birth", _KERNEL, {"process_birth": "", "kernel_birth": _REUSED}, False, True),
    ("", _KERNEL, {"process_birth": _ZONE, "kernel_birth": _KERNEL}, True, True),
    ("test-birth", _KERNEL, {"process_birth": _ZONE}, False, False),
    ("test-birth", _KERNEL, {"process_birth": "test-birth"}, True, True),
    ("test-birth", _KERNEL, {"process_birth": ""}, True, True),
    ("test-birth", "", {"process_birth": _ZONE, "kernel_birth": _KERNEL}, False, False),
    ("test-birth", "", {"process_birth": "", "kernel_birth": _KERNEL}, True, True),
    ("test-birth", _KERNEL, {"process_birth": _ZONE, "kernel_birth": "darwin-start:1793143271"}, False, False),
    ("test-birth", _KERNEL, {"process_birth": "test-birth", "kernel_birth": "Tue Oct 27 14:57:17 2026"}, True, True),
    ("test-birth", _KERNEL, {"process_birth": _ZONE, "kernel_birth": _FILETIME}, False, False),
    (_LINUX, _LINUX, {"process_birth": _LINUX, "kernel_birth": _LINUX}, True, True),
    (_LINUX, _LINUX, {"process_birth": _LINUX_OTHER, "kernel_birth": _LINUX_OTHER}, False, False),
    (_FILETIME, _FILETIME, {"process_birth": _FILETIME, "kernel_birth": _FILETIME}, True, True),
    (_FILETIME, _FILETIME, {"process_birth": _FILETIME_OTHER, "kernel_birth": _FILETIME_OTHER}, False, False),
], ids=["new_other_zone", "reused_same_second", "reused_beats_pid_fallback", "kernel_without_parent_spelling",
        "old_other_zone_unfixed", "old_same", "old_unmeasured_pid_fallback", "parent_unread_other_zone_unfixed",
        "parent_unread_pid_fallback", "malformed_other_zone", "malformed_same", "two_kinds",
        "linux_same", "linux_reused", "windows_same", "windows_reused"])
def test_the_idle_probe_answer_lets_comparable_kernel_births_decide(pool, slot_birth, slot_kernel, event,  # noqa: F811
                                                                     now, before):
    """``before`` is the prior answer: the legacy rule the same answer meets without kernel births."""
    def confirmed(kernel, answer):
        slot = pool.workers.Worker(wid=0, proc=_FakeProc(5001), in_q=MagicMock(), busy_task_id=None,
                                   process_birth=slot_birth, ready_identity=(5001, slot_birth),
                                   quiescence_token="probe")
        slot.kernel_birth = kernel
        pool.workers.WORKERS[0] = slot
        pool.lifecycle.handle_worker_quiescent({"worker_id": 0, "pid": 5001, "token": "probe", **answer},
                                               pool.workers)
        return slot.quiescence_confirmed

    assert confirmed(slot_kernel, event) is now
    assert confirmed("", {key: value for key, value in event.items() if key != "kernel_birth"}) is before


@pytest.mark.serial  # a real spawned worker loop
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS ps spells lstart in the reader's zone")
def test_a_real_worker_in_another_zone_opens_its_slot_and_answers_its_probe(pool, seam, tmp_path,  # noqa: F811
                                                                             monkeypatch):
    """The real worker loop reports its births in its own zone; the real readiness thread ledgers its
    slot in this process's zone (TZ of these two processes only; no OS setting changes)."""
    from tests.test_worker_local_quiescence import _probe_worker_process

    monkeypatch.setenv("TZ", "Pacific/Chatham")  # inherited by the spawned worker
    with _probe_worker_process(tmp_path, monkeypatch, "clean") as worker:
        monkeypatch.setenv("TZ", "UTC")
        row = _rows(tmp_path / "data" / "logs" / "events.jsonl", "worker_ready")[-1]
        append_jsonl(seam.events, row)
        slot = pool.workers.Worker(wid=0, proc=worker.proc, in_q=MagicMock(), busy_task_id=None,
                                   reaping=True, booting=True)
        pool.workers.WORKERS[0] = slot

        seam.run({0: slot}, seam.cursor, 1)

        assert row["pid"] == worker.proc.pid and slot.process_birth and row["process_birth"]
        assert slot.process_birth != row["process_birth"], "precondition: two zones spell one birth differently"
        assert slot.kernel_birth == row["kernel_birth"] and slot.kernel_birth.startswith("darwin-start:")
        assert pool.lifecycle.worker_ready(slot) and seam.killed == [] and seam.respawned == []
        slot.quiescence_token = "probe-1"
        worker.release.set()
        worker.in_q.put({"type": "worker_quiescence", "token": "probe-1"})
        answer = worker.receive(lambda event: event.get("type") == "worker_quiescent", 15)
        assert answer is not None and answer["kernel_birth"] == slot.kernel_birth
        pool.lifecycle.handle_worker_quiescent(answer, pool.workers)
        assert slot.quiescence_confirmed
        # Unfixed: the same births without the kernel field (an older writer) are judged by spelling.
        assert pool.lifecycle._own_ready_row(slot, [{**row, "kernel_birth": ""}]) is None
        assert not pool.lifecycle._births_agree(slot, {**answer, "kernel_birth": ""})

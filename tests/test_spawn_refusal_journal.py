"""A spawn that stays refused is journaled once per change of its cause, and once when it clears.

The health tick finds a dead slot and hands it to the reaper; while the replacement's spawn stays refused, the reaper
clears ``reaping`` for recovery and the next tick finds the same dead process again. The journal then grew by time,
not by change: one ``worker_dead_detected`` row and one WARNING traceback per tick (39 in 60 s on the stand, LG open
question 9). A death is one fact, written once per process; a refusal is another, written when its cause changes and
when it ends. The retry itself (one spawn attempt per tick) and the crash-storm clock are unchanged.
"""
from __future__ import annotations

import logging

import pytest

from ouroboros.task_carrier import LANE_CONVERSATION
from supervisor import queue, task_reaper, worker_health, workers
from tests.test_owner_wait_pool import InputQueue, Process
from tests.test_owner_wait_pool import pool as pool  # noqa: F401 -- fixture
from tests.worker_pool_support import affirm_ready



def _dead(wid: int = 3):
    slot = workers.Worker(wid, Process(6000 + wid), InputQueue(), lane=LANE_CONVERSATION)
    affirm_ready(slot)
    slot.proc.alive, slot.proc.exitcode = False, -9
    workers.WORKERS[wid] = slot
    return slot


def _refusing(error: BaseException):
    def respawn(_wid, **_kwargs):
        raise error
    return respawn


def _tick(slot) -> None:
    """One health tick, then the reaper's attempt to replace the slot (a refusal hands it back to the next tick)."""
    worker_health._ensure_workers_healthy_locked(queue)
    while not queue._reap_queue.empty():
        queue._reap_queue.get_nowait()
    task_reaper._respawn_after_reap(queue, workers, slot.wid, expected_worker=slot)


@pytest.fixture
def journal(pool, monkeypatch, caplog):
    rows: list = []
    monkeypatch.setattr(workers, "append_jsonl", lambda _path, row: rows.append(row) or True)
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0.0)
    monkeypatch.setattr(task_reaper, "_RESPAWN_REFUSALS", {})
    monkeypatch.setattr(worker_health, "_DEATHS_RECORDED", {})
    caplog.set_level(logging.DEBUG, logger="supervisor.task_reaper")

    def warnings():
        return [record for record in caplog.records if record.name == "supervisor.task_reaper"
                and record.levelno == logging.WARNING]

    return rows, warnings, caplog


def test_a_refusal_that_stands_is_written_once_and_its_end_once(journal, monkeypatch):
    rows, warnings, caplog = journal
    slot = _dead()
    monkeypatch.setattr(workers, "respawn_worker", _refusing(OSError(35, "Resource temporarily unavailable")))
    for _beat in range(5):
        _tick(slot)
        assert not slot.reaping, "every tick still retries: recovery is unchanged"
    assert [row["type"] for row in rows].count("worker_dead_detected") == 1, "one death, one row"
    assert len(warnings()) == 1 and warnings()[0].exc_info, "the first refusal keeps its traceback as evidence"
    monkeypatch.setattr(workers, "respawn_worker", _refusing(PermissionError(1, "Operation not permitted")))
    _tick(slot)
    _tick(slot)
    assert len(warnings()) == 2, "a changed cause is written again, once"
    monkeypatch.setattr(workers, "respawn_worker", lambda _wid, **_kwargs: True)
    _tick(slot)
    cleared = [record for record in caplog.records if record.name == "supervisor.task_reaper"
               and record.levelno == logging.INFO and "refus" in record.getMessage()]
    assert len(cleared) == 1 and "7" in cleared[0].getMessage(), "its end is written once, with the refusals it ended"
    assert task_reaper._RESPAWN_REFUSALS == {}
    assert [row["type"] for row in rows].count("worker_dead_detected") == 1


def test_a_new_process_s_death_in_the_same_slot_is_a_new_row(journal, monkeypatch):
    rows, _warnings, _caplog = journal
    first = _dead()
    monkeypatch.setattr(workers, "respawn_worker", _refusing(OSError(35, "Resource temporarily unavailable")))
    _tick(first)
    _tick(first)
    second = _dead()  # the slot got a new process, which died as well
    _tick(second)
    assert [row["type"] for row in rows].count("worker_dead_detected") == 2
    assert {row["exitcode"] for row in rows if row["type"] == "worker_dead_detected"} == {-9}

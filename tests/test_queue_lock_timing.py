"""The queue lock Q times its outermost acquisitions without changing what it is.

``supervisor.queue._queue_lock`` is a ``MeasuredRLock``: every existing ``with`` and
``acquire``/``release`` caller keeps RLock semantics (ownership, re-entrancy, a release by a
thread that does not hold it raising the RLock's own error, ``_is_owned``), while the
per-minute ``supervisor_load`` row learns how long outermost acquisitions waited and held Q
and where the longest hold came from. Pinned both ways: re-entry is measured once, a refused
attempt is not measured, and the reader drains without ever waiting for Q.
"""

from __future__ import annotations

import threading
import time

import pytest

from supervisor.loop_load import MeasuredRLock


def _script(lock) -> list:
    """One fixed sequence of lock operations and what each answered."""
    seen: list = []
    other: list = []

    def from_another_thread(action):
        thread = threading.Thread(target=lambda: other.append(action()))
        thread.start()
        thread.join(5)
        return other.pop()

    def released_by_non_owner():
        try:
            lock.release()
        except RuntimeError as exc:
            return str(exc)
        return "released"

    seen.append(lock._is_owned())
    seen.append(lock.acquire())
    seen.append(lock.acquire(blocking=False))  # re-entrant: granted at once
    with lock:
        seen.append(lock._is_owned())
    seen.append(from_another_thread(lambda: lock.acquire(blocking=False)))
    seen.append(from_another_thread(lambda: lock.acquire(timeout=0.05)))
    seen.append(from_another_thread(released_by_non_owner))
    lock.release()
    seen.append(lock._is_owned())
    lock.release()
    seen.append(lock._is_owned())
    seen.append(released_by_non_owner())
    seen.append(from_another_thread(lambda: (lock.acquire(timeout=1), lock.release())[0]))
    seen.append(bool(lock))
    return seen


def test_q_is_the_measured_lock_with_rlock_semantics_unchanged():
    from supervisor import queue

    assert isinstance(queue._queue_lock, MeasuredRLock)
    expected = _script(threading.RLock())
    assert expected == [False, True, True, True, False, False, "cannot release un-acquired lock",
                        True, False, "cannot release un-acquired lock", True, True]
    assert _script(MeasuredRLock()) == expected
    with queue._queue_lock:
        assert queue._queue_lock._is_owned()
    assert not queue._queue_lock._is_owned()


def _hold(lock, seconds, entered=None, release=None):
    with lock:
        if entered is not None:
            entered.set()
        if release is not None:
            assert release.wait(5)
        time.sleep(seconds)


def test_only_outermost_acquisitions_are_measured_with_their_wait_hold_and_site():
    lock = MeasuredRLock()
    with lock:  # outermost: one measured hold
        with lock:  # re-entrant: depth only
            time.sleep(0.07)  # a Windows sleep may end a ~15.6 ms tick early
    facts = lock.drain()
    assert facts["count"] == 1, "a re-entrant acquisition is not measured again"
    assert facts["hold_max_sec"] >= 0.05 and facts["hold_total_sec"] == facts["hold_max_sec"]
    assert facts["wait_max_sec"] < 0.05
    assert facts["hold_max_site"].startswith("tests/test_queue_lock_timing.py:")
    assert facts["hold_max_site"].endswith(" in test_only_outermost_acquisitions_are_measured_with_their_wait_hold_and_site")

    entered = threading.Event()
    holder = threading.Thread(target=_hold, args=(lock, 0.22, entered))
    holder.start()
    assert entered.wait(5)
    assert lock.acquire(blocking=False) is False and lock.acquire(timeout=0.01) is False, "refused, not measured"
    with lock:  # waits for the holder
        pass
    holder.join(5)
    facts = lock.drain()
    assert facts["count"] == 2
    assert facts["wait_max_sec"] >= 0.1, "the contended acquisition waited for the holder"
    assert facts["hold_max_sec"] >= 0.2 and facts["hold_max_site"].endswith(" in _hold")
    assert lock.drain() == {"count": 0, "wait_total_sec": 0.0, "wait_max_sec": 0.0,
                            "hold_total_sec": 0.0, "hold_max_sec": 0.0, "hold_max_site": None}


def test_the_reader_drains_while_another_thread_holds_q():
    lock = MeasuredRLock()
    with lock:
        pass
    entered, release = threading.Event(), threading.Event()
    holder = threading.Thread(target=_hold, args=(lock, 0.0, entered, release))
    holder.start()
    try:
        assert entered.wait(5)
        drained: list = []
        reader = threading.Thread(target=lambda: drained.append(lock.drain()))
        reader.start()
        reader.join(2)
        assert not reader.is_alive(), "the drain waited for Q"
        assert drained[0]["count"] == 1, "the open hold lands in the window where it ends"
    finally:
        release.set()
        holder.join(5)
    assert lock.drain()["count"] == 1


def test_a_measured_cycle_stays_within_microseconds_of_a_plain_rlock():
    """Micro-benchmark (printed with ``-s``): an uncontended outermost ``with`` cycle."""
    cycles = 20000

    def per_cycle(lock) -> float:
        started = time.perf_counter()
        for _ in range(cycles):
            with lock:
                pass
        return (time.perf_counter() - started) / cycles

    plain, measured = min(per_cycle(threading.RLock()) for _ in range(3)), min(per_cycle(MeasuredRLock()) for _ in range(3))
    print(f"\nQ cycle: RLock {plain * 1e9:.0f} ns, MeasuredRLock {measured * 1e9:.0f} ns")
    assert measured < 25e-6, f"a measured cycle costs {measured * 1e6:.1f} µs"


@pytest.mark.parametrize("lock_type", [threading.RLock, MeasuredRLock])
def test_with_releases_on_an_exception_for_both_lock_types(lock_type):
    lock = lock_type()
    with pytest.raises(ValueError):
        with lock:
            raise ValueError("inside")
    assert lock._is_owned() is False and lock.acquire(blocking=False) is True
    lock.release()

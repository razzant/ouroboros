"""Self-pin of the root conftest's thread-hygiene guard: the baseline is taken after fixtures are set up (a module-scoped fixture's thread is not a test's leak); after every fixture of an item is
torn down, a thread the item started and left alive is named at the item (the leaker), a
thread that finishes within the grace is not, and the baseline is held by thread OBJECT so a
recycled ident cannot hide a leak. The guard itself lives in tests/conftest.py next to the
password-resolver guard; this file exercises it directly with a fake item."""
from __future__ import annotations

import pathlib
import sys
import threading
from types import SimpleNamespace

import pytest

conftest = next(
    m for m in list(sys.modules.values())
    if getattr(m, "__file__", "")
    and pathlib.Path(m.__file__).as_posix().endswith("tests/conftest.py")
    and hasattr(m, "_fail_if_a_thread_leaked")
)


def _item(nodeid: str, baseline) -> SimpleNamespace:
    stash = pytest.Stash()
    stash[conftest._THREADS_BEFORE_ITEM] = baseline
    return SimpleNamespace(nodeid=nodeid, stash=stash)


def test_a_thread_left_alive_after_teardown_names_the_item(monkeypatch):
    monkeypatch.setattr(conftest, "_THREAD_LEAK_GRACE_SEC", 0.2)
    release = threading.Event()
    baseline = set(threading.enumerate())
    leaker = threading.Thread(target=release.wait, name="probe-leaker", daemon=True)
    leaker.start()
    try:
        with pytest.raises(pytest.fail.Exception, match="tests/x.py::leaks leaked 1 thread.*probe-leaker"):
            conftest._fail_if_a_thread_leaked(_item("tests/x.py::leaks", baseline))
        assert ("tests/x.py::leaks", ["probe-leaker"]) in conftest._THREAD_LEAKS
    finally:
        release.set()
        leaker.join(2)
        conftest._THREAD_LEAKS[:] = [row for row in conftest._THREAD_LEAKS if row[0] != "tests/x.py::leaks"]


def test_a_thread_that_finishes_within_the_grace_is_not_a_leak(monkeypatch):
    monkeypatch.setattr(conftest, "_THREAD_LEAK_GRACE_SEC", 2.0)
    baseline = set(threading.enumerate())
    quick = threading.Thread(target=lambda: None, name="probe-quick", daemon=True)
    quick.start()
    conftest._fail_if_a_thread_leaked(_item("tests/x.py::quick", baseline))   # no failure
    quick.join(2)


def test_a_baseline_thread_is_held_by_object_not_by_ident(monkeypatch):
    """Through the REAL hooks: the setup hookwrapper snapshots the baseline while a baseline
    thread is alive; that thread exits; a new thread is started and its ident is recycled to
    the exited one (``Thread.ident`` is a property over ``_ident``); the leak checker must
    still name the new thread — a baseline held by ident would have let it pass."""
    monkeypatch.setattr(conftest, "_THREAD_LEAK_GRACE_SEC", 0.2)
    exit_old = threading.Event()
    old = threading.Thread(target=exit_old.wait, name="probe-old", daemon=True)
    old.start()
    item = SimpleNamespace(nodeid="tests/x.py::recycled", stash=pytest.Stash())
    snapshot = conftest.pytest_runtest_call(item)   # the call-phase hookwrapper: snapshot, then yield
    next(snapshot)
    with pytest.raises(StopIteration):
        next(snapshot)
    baseline = item.stash[conftest._THREADS_BEFORE_ITEM]
    assert old in baseline and old.ident is not None
    exit_old.set()
    old.join(2)
    assert not old.is_alive()
    release = threading.Event()
    new = threading.Thread(target=release.wait, name="probe-new", daemon=True)
    new.start()
    real_ident = new.ident
    try:
        new._ident = old.ident   # what the OS may hand out once the old thread is gone
        assert new.ident in {thread.ident for thread in baseline}   # an ident set would skip it
        with pytest.raises(pytest.fail.Exception, match="tests/x.py::recycled leaked 1 thread.*probe-new"):
            conftest._fail_if_a_thread_leaked(item)
        assert ("tests/x.py::recycled", ["probe-new"]) in conftest._THREAD_LEAKS
    finally:
        new._ident = real_ident
        release.set()
        new.join(2)
        conftest._THREAD_LEAKS[:] = [row for row in conftest._THREAD_LEAKS if row[0] != "tests/x.py::recycled"]

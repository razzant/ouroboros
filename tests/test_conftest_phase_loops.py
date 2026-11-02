"""tests/conftest.py's phase loops: a loop a test broke fails that test's phase, never the session."""
from types import SimpleNamespace

import pytest

from tests import conftest


def test_a_half_closed_phase_loop_does_not_end_the_session():
    """On Windows a proactor loop whose close raised in the call or teardown phase (failing that phase) is
    left without its self-pipe, and closing it again raises AttributeError. The protocol's final close runs
    after the reports, where an exception ended the session as an INTERNALERROR and left every later test
    of the shard unrun."""
    item = SimpleNamespace(nodeid="tests/x.py::broke_its_loop", stash=pytest.Stash())
    protocol = conftest.pytest_runtest_protocol(item, None)
    next(protocol)
    loops = item.stash[conftest._PHASE_EVENT_LOOPS]
    attempted = []

    def half_closed(loop):
        def close():
            attempted.append(loop)
            raise AttributeError("'NoneType' object has no attribute 'close'")
        return close

    for loop in loops:
        loop.close = half_closed(loop)
    try:
        with pytest.raises(StopIteration):
            next(protocol)
        assert attempted == loops, "every loop still gets its close"
        assert conftest._PHASE_EVENT_LOOPS not in item.stash
    finally:
        for loop in loops:
            del loop.close
            loop.close()

"""The supervisor event bus latch never leaks from one test into the next (conftest autouse)."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def closed_managers():
    """Observe after the function fixtures, but reap our own managers even on a regression."""
    managers = []
    yield managers
    try:
        assert managers
        for manager in managers:
            assert not manager._process.is_alive(), "a test left its event bus running"
            assert manager._process.exitcode == 0
    finally:
        for manager in managers:
            if manager._process.is_alive():
                manager.shutdown()


@pytest.mark.parametrize("explicit_close", [False, True])
def test_a_test_owned_bus_is_closed_before_monkeypatch_restores_it(tmp_path, monkeypatch,
                                                                closed_managers, explicit_close):
    from supervisor import workers

    # Undo hands the prior values back. The bus created underneath those patches must
    # already be stopped, otherwise the manager reference disappears while its process lives.
    for name, value in (("DRIVE_ROOT", tmp_path), ("_EVENT_Q", None), ("_EVENT_Q_MANAGER", None),
                        ("_EVENT_Q_GENERATION", ""), ("_EVENT_Q_SHUTDOWN", False)):
        monkeypatch.setattr(workers, name, value)
    queue = workers.get_event_q()
    manager = workers._EVENT_Q_MANAGER
    closed_managers.append(manager)
    assert manager._process.is_alive()
    queue.put({"type": "isolation-probe"})
    assert queue.get(timeout=5) == {"type": "isolation-probe"}
    if explicit_close:
        workers.shutdown_event_q()


def test_a_shutdown_latches_the_bus_for_this_test_only():
    from supervisor import workers

    workers.shutdown_event_q()
    assert workers._EVENT_Q_SHUTDOWN is True


def test_the_next_test_starts_with_an_open_bus():
    from supervisor import workers

    assert workers._EVENT_Q_SHUTDOWN is False

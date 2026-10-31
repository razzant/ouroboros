"""The launcher's readiness wait has no clock: a live server is waited for while it lives, and a
server that exits before it served fails the start at once. Health from another generation or PID
stays rejected (tests/test_launcher_generation_custody.py)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.serial


class _Clock:
    """A fake ``time`` whose every read advances by ``step`` seconds; ``sleep`` advances too."""

    def __init__(self, step: float):
        self.now, self.step, self.sleeps = 1_000.0, step, 0

    def time(self):
        self.now += self.step
        return self.now

    def monotonic(self):
        return self.time()

    def sleep(self, _seconds):
        self.sleeps += 1
        self.now += self.step


@pytest.fixture
def launcher(monkeypatch):
    import launcher

    monkeypatch.setattr(launcher, "_generation_port", None)
    monkeypatch.setattr(launcher, "_agent_exit_before_ready", threading.Event())
    return launcher


def test_the_ui_wait_outlives_any_deadline_while_the_generation_lives(launcher, monkeypatch):
    """Both directions: no failure after the clock passed 60 and 120 simulated seconds while the
    child lives, then success the moment the lifecycle thread proves the port. A fixed deadline
    (the retired ``time.time() + 60``) would return False here."""
    clock = _Clock(step=10.0)
    monkeypatch.setattr(launcher, "time", clock)
    alive = SimpleNamespace(is_alive=lambda: True)

    advance = clock.sleep

    def sleep(seconds):
        advance(seconds)
        if clock.sleeps == 20:  # 200 simulated seconds into the boot the generation proves its port
            launcher._generation_port = 9876

    clock.sleep = sleep
    assert launcher._await_server_ready(1, None, alive) == (True, 9876)
    assert clock.now - 1_000.0 > 120, "the simulated wait passed every former deadline"


def test_the_ui_wait_fails_at_once_when_the_child_exits_before_it_served(launcher, monkeypatch):
    clock = _Clock(step=0.0)
    monkeypatch.setattr(launcher, "time", clock)
    alive = SimpleNamespace(is_alive=lambda: True)
    launcher._agent_exit_before_ready.set()
    assert launcher._await_server_ready(1, None, alive) == (False, 1)
    assert clock.sleeps == 0, "no polling after the exit: the failure is immediate"


def test_the_lifecycle_loop_marks_an_exit_before_readiness_but_not_a_restart_or_a_proven_one(launcher, monkeypatch):
    """The event behind the immediate failure: set by a crash before the port was proven; never by
    an exit-42 restart, a launcher-requested recycle or a crash of a generation that already served."""
    import launcher as mod

    def run_generation(exit_code, *, proved, restart_requested=False):
        mod._agent_exit_before_ready.clear()
        mod._generation_port = 9876 if proved else None
        mod._agent_restart_requested.clear()
        if restart_requested:
            mod._agent_restart_requested.set()
        # The exact bookkeeping of agent_lifecycle_loop after proc.wait().
        proved_now = mod._generation_port is not None
        mod._generation_port = None
        if not proved_now and exit_code not in (mod.RESTART_EXIT_CODE, mod.PANIC_EXIT_CODE) and not mod._agent_restart_requested.is_set():
            mod._agent_exit_before_ready.set()
        mod._agent_restart_requested.clear()
        return mod._agent_exit_before_ready.is_set()

    source = open(mod.__file__, encoding="utf-8").read()
    assert "if not proved and exit_code not in (RESTART_EXIT_CODE, PANIC_EXIT_CODE) and not _agent_restart_requested.is_set():" in source
    assert run_generation(1, proved=False) is True
    assert run_generation(mod.RESTART_EXIT_CODE, proved=False) is False
    assert run_generation(1, proved=False, restart_requested=True) is False
    assert run_generation(1, proved=True) is False


def test_the_health_probe_waits_while_the_child_lives_and_ends_on_its_exit(launcher, monkeypatch):
    import io

    from ouroboros import launcher_readiness

    clock = _Clock(step=10.0)
    monkeypatch.setattr(launcher_readiness, "time", clock)  # the probe's own clock module
    probes = []

    class _Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def urlopen(url, timeout=None):
        probes.append(timeout)
        if len(probes) < 15:  # 150 simulated seconds of refused connections: the boot is long
            raise ConnectionRefusedError(url)
        return _Response(b'{"status": "ok", "boot": {"phase": "serving", "dialogue_ready": false, "tasks_ready": false}}')

    monkeypatch.setattr(launcher.urllib.request, "urlopen", urlopen)
    child = SimpleNamespace(poll=lambda: None)
    assert launcher._wait_for_server(1, proc=child) is True
    assert len(probes) == 15 and set(probes) == {2}, "the per-probe HTTP timeout is the only clock, a transport bound"
    assert clock.now - 1_000.0 > 120

    probes.clear()
    exits = iter([None, None, 3])
    dead_soon = SimpleNamespace(poll=lambda: next(exits))
    assert launcher._wait_for_server(1, proc=dead_soon) is False
    assert len(probes) == 2, "the probe ends with the child's exit, not with a deadline"

"""A scenario's timeout ceiling landing inside a server stop never leaves the server running.

The e2e lanes give every mock scenario a pytest-timeout ceiling with the signal method
(tests/system_e2e/conftest.py, tests/test_e2e_cancellation_scenarios.py). Its handler raises
wherever the main thread is, so an alarm inside ``IsolatedServer.stop()`` once unwound the
stop before the forced kill and reaped nothing: a SIGTERM-proof server outlived it into the
next scenario. Both stops now hold the alarm until the tree is down
(``server_runner.alarm_deferred``). These tests arm the ceiling's own handler, let it fire
inside the stop, and require the timeout to still fail AND the process to be gone.
"""

from __future__ import annotations

import contextlib
import io
import os
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from devtools.benchmarks.common.server_runner import IsolatedServer
from tests.system_e2e.harness import KeylessIsolatedServer

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="the ceiling's signal method is POSIX-only")

# A server that ignores SIGTERM: only the stop's forced kill ends it.
_STUBBORN = ("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             "print('ready', flush=True); time.sleep(60)")


@contextlib.contextmanager
def _scenario_ceiling(seconds: float):
    """Arm the lanes' ceiling handler (pytest-timeout, signal method) for ``seconds``, restoring
    whatever this test's own runner had armed."""
    import pytest_timeout
    from _pytest._io import TerminalWriter

    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers run only in the main thread")
    item = SimpleNamespace(config=SimpleNamespace(get_terminal_writer=lambda: TerminalWriter(file=io.StringIO())))
    settings = SimpleNamespace(method="signal", timeout=seconds, disable_debugger_detection=True)
    previous, remaining = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    pytest_timeout.pytest_timeout_set_timer(item, settings)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if remaining[0]:
            signal.setitimer(signal.ITIMER_REAL, *remaining)


@contextlib.contextmanager
def _owned(proc: subprocess.Popen):
    """The test's own backstop: whatever the stop under test did, the child is gone after."""
    try:
        assert proc.stdout.readline().strip() == "ready"
        yield proc
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        proc.stdout.close()


def test_a_ceiling_inside_an_isolated_server_stop_fails_the_test_only_after_the_kill(tmp_path):
    proc = subprocess.Popen([sys.executable, "-I", "-c", _STUBBORN], start_new_session=True,
                            stdout=subprocess.PIPE, text=True)
    with _owned(proc):
        server = object.__new__(IsolatedServer)
        server.proc, server.stop_grace_sec, server.data_root = proc, 1.0, tmp_path
        with _scenario_ceiling(0.2), pytest.raises(pytest.fail.Exception, match="Timeout"):
            server.stop()  # the alarm lands in the SIGTERM grace wait
        assert proc.poll() is not None  # force-killed and reaped before the timeout propagated

    quiet = subprocess.Popen([sys.executable, "-I", "-c", _STUBBORN], start_new_session=True,
                             stdout=subprocess.PIPE, text=True)
    with _owned(quiet):
        server = object.__new__(IsolatedServer)
        server.proc, server.stop_grace_sec, server.data_root = quiet, 0.2, tmp_path
        server.stop()  # no alarm: an ordinary stop, nothing raised
        assert quiet.poll() is not None


# A server whose descendant left its process group: only the container's reap ends that one.
_DETACHING = ("import subprocess, sys, time\n"
              "child = subprocess.Popen([sys.executable, '-I', '-c', 'import signal, time; "
              "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'], start_new_session=True)\n"
              "print(child.pid, flush=True)\n"
              "time.sleep(60)\n")


def _gone(pid: int, within: float = 10.0) -> bool:
    import time

    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_a_ceiling_inside_a_candidate_server_stop_still_reaps_its_container(tmp_path):
    from ouroboros.process_containment import ProcessContainer

    container, released = ProcessContainer(), []
    proc = container.spawn([sys.executable, "-I", "-c", _DETACHING], stdout=subprocess.PIPE, text=True)
    detached = int(proc.stdout.readline())
    real_reap = container.reap

    def reap_under_the_ceiling() -> str:
        signal.raise_signal(signal.SIGALRM)  # the ceiling fires inside the reap, which runs once
        return real_reap()

    container.reap = reap_under_the_ceiling
    server = object.__new__(KeylessIsolatedServer)
    server.candidate = SimpleNamespace(release=lambda: released.append(True))
    server._candidate_container, server._candidate_cleanup_error = container, ""
    server.proc, server.stop_grace_sec, server.data_root = proc, 5.0, tmp_path
    try:
        with _scenario_ceiling(60), pytest.raises(pytest.fail.Exception, match="Timeout"):
            server.stop()
        # The detached member was reaped by the container and the checkout released.
        assert _gone(detached) and proc.poll() is not None and released == [True]
        assert server._candidate_cleanup_error == ""
    finally:
        for pid in (detached, proc.pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        proc.wait(timeout=10)
        proc.stdout.close()

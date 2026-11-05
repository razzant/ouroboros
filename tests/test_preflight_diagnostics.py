"""The fixture observes nested pytest completion and timeout without changing either."""
import subprocess
import sys
import threading
import time
from types import SimpleNamespace


from ouroboros import preflight_runner
from tests import stack_samples



def test_samples_are_whole_and_stop_with_a_finite_join(tmp_path, monkeypatch):
    path = tmp_path / "taken.samples"
    stop = stack_samples.start(path, 0.05)
    deadline = time.monotonic() + 10
    while " in test_samples_are_whole_and_stop_with_a_finite_join" not in path.read_text(encoding="utf-8"):
        assert time.monotonic() < deadline, path.read_text(encoding="utf-8")
        time.sleep(0.05)
    assert stop() is True
    assert path.read_text(encoding="utf-8").startswith("Stack sample 1 (every 0.05s):\n")
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(stack_samples, "faulthandler", SimpleNamespace(
        dump_traceback=lambda **_: (entered.set(), release.wait(10))))
    stop = stack_samples.start(tmp_path / "stuck.samples", 0.01)
    assert entered.wait(10)
    began = time.monotonic()
    assert stop(timeout=0.2) is False and time.monotonic() - began < 5  # never waits out a stuck sampler
    release.set()
    assert stop() is True


def test_a_failed_exit_event_still_stops_the_sampler(tmp_path, preflight_timeout_diagnostics):
    """The probe's own exit handler stops and joins its sampler even when its last event cannot be written."""
    preflight_timeout_diagnostics(dump_after=60)
    module = preflight_runner.install_worker_probe(tmp_path)
    script = ("import atexit, sys, threading\n"  # registered first, so it runs after the probe's handler
              "atexit.register(lambda: print('sampler alive:', any(t.name == 'preflight-stack-sampler'"
              " for t in threading.enumerate())))\n"
              f"sys.path.insert(0, {str(preflight_runner.probe_dir(tmp_path))!r})\n"
              f"__import__({module!r})._trace.close()\n")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "sampler alive: False" in result.stdout, result.stdout + result.stderr
    assert "I/O operation on closed file" in result.stderr  # the exit event write did fail

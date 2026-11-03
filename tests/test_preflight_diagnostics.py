"""The fixture observes nested pytest completion and timeout without changing either."""
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import preflight_runner
from tests import stack_samples

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("hang", [False, True])
def test_preflight_diagnostics_preserve_exit_and_timeout(tmp_path, monkeypatch, preflight_timeout_diagnostics, hang):
    trace_dir = preflight_timeout_diagnostics(dump_after=2)
    root = tmp_path / "nested"
    repo = root / "repo"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (repo / "conftest.py").write_text(
        "import atexit, os, time\ndef hold_exit():\n    time.sleep(60)\n"
        # xdist has its own ten-second worker reap. Hold the controller instead
        # so startup has room and the unchanged preflight timeout owns the kill.
        f"if {hang!r} and not os.environ.get('PYTEST_XDIST_WORKER'):\n    atexit.register(hold_exit)\n",
        encoding="utf-8",
    )
    (tests / "test_probe.py").write_text("def test_probe():\n    assert True\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    monkeypatch.setenv("OURO_TEST_WORKER_BUDGET", "2")  # nested slots: the probe expects two real workers
    module = preflight_runner._install_worker_probe(root)
    # Own the nested pytest temporary tree. Otherwise its controller may spend
    # teardown cleaning unrelated numbered runs instead of reaching hold_exit.
    code, output, containment = preflight_runner._execute_pytest_pass(
        sys.executable, repo, root,
        ["tests", "-q", "-n", "2", "-p", module, "--basetemp", str(root / "pytest")], 20,
    )
    assert code == (None if hang else 0)
    assert not containment
    logs = {path.name: path.read_text(encoding="utf-8") for path in trace_dir.glob("*.log")}
    samples = [path.read_text(encoding="utf-8") for path in trace_dir.glob("*.samples")]
    # A process's log holds whole event lines only; its stack samples have their own file.
    events = [json.loads(line) for name, text in logs.items() for line in text.splitlines()
              if name != "parent.log" or line.startswith('{"event":')]
    assert {row["worker"] for row in events if row["event"] == "probe_import"} == {"controller", "gw0", "gw1"}
    assert "Fatal Python error" not in output + "".join(samples)  # the observer never kills what it observes
    assert not any(line.startswith("Timeout") for text in samples for line in text.splitlines())
    if hang:
        assert "[100%]" in output  # The test passed; interpreter teardown is still live.
        assert any(row["event"] == "before_kill" and row["returncode"] is None for row in events)
        assert any("Stack sample" in text and "hold_exit" in text for text in samples)
    else:
        assert not any(row["event"] == "before_kill" for row in events)
        assert {row["worker"] for row in events if row["event"] == "atexit"} == {"controller", "gw0", "gw1"}


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
    module = preflight_runner._install_worker_probe(tmp_path)
    script = ("import atexit, sys, threading\n"  # registered first, so it runs after the probe's handler
              "atexit.register(lambda: print('sampler alive:', any(t.name == 'preflight-stack-sampler'"
              " for t in threading.enumerate())))\n"
              f"sys.path.insert(0, {str(preflight_runner._probe_dir(tmp_path))!r})\n"
              f"__import__({module!r})._trace.close()\n")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "sampler alive: False" in result.stdout, result.stdout + result.stderr
    assert "I/O operation on closed file" in result.stderr  # the exit event write did fail

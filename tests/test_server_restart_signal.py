"""A single signal must not turn final restart cleanup into unfinished work."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ouroboros.process_containment import ProcessContainer
from ouroboros.test_environment import isolated_environment, retain_tree
from tests.test_server_capture_exit import prepare_capture, wait_for


def test_server_tick_observes_existing_stop_latch(monkeypatch):
    import asyncio
    import threading
    from ouroboros import server_process

    stopped = threading.Event()
    monkeypatch.setattr(server_process, "_exit_signalled", stopped)
    calls = []
    async def tick(self, counter):
        calls.append(counter)
        return False
    monkeypatch.setattr(server_process.uvicorn.Server, "on_tick", tick)
    server = server_process._SignalStopServer(None)
    assert asyncio.run(server.on_tick(1)) is False
    stopped.set()
    assert asyncio.run(server.on_tick(2)) is True
    assert calls == [1]


@pytest.mark.skipif(sys.platform == "win32", reason="Real PID signals require POSIX delivery")
@pytest.mark.parametrize("entrypoint,ingress", [
    ("console", "control"), ("console", "sigint"),
    ("console", "sigterm"), ("imported", "sigint"),
])
@pytest.mark.timeout(80)
def test_signal_during_restart_join_preserves_cleanup(tmp_path, entrypoint, ingress):
    prepare_capture(tmp_path)
    gates = tmp_path / "held"
    gates.mkdir()
    env = isolated_environment(tmp_path, REPO)
    env["OUROBOROS_HOST_SERVICE_PORT"] = "0"
    env["OUROBOROS_SERVER_PORT"] = "0"
    container = ProcessContainer(lifeline=True, overload=True)
    proc = None
    facts = {"entrypoint": entrypoint, "ingress": ingress}
    try:
        with (gates / "console.log").open("wb") as log:
            proc = container.spawn([
                sys.executable, "-u", str(Path(__file__).with_name("_server_restart_signal.py")),
                str(tmp_path), "held", entrypoint,
            ], cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            wait_for(lambda: all((gates / name).exists() for name in (
                "ready.json", "ordinary-entered.json", "review-entered.json")), proc)
            (gates / "request-restart").touch()
            wait_for(lambda: all((gates / name).exists() for name in (
                "restart-cleanup-entered.json", "main-socket-closed.json",
                "lifespan-finished.json", "server-finished.json")), proc)
            assert json.loads((gates / "main-socket-closed.json").read_text())["closed"]
            assert not (gates / "release").exists()
            assert not (gates / "release-restart-cleanup").exists()
            started = time.monotonic()
            if ingress != "control":
                proc.send_signal(signal.SIGINT if ingress == "sigint" else signal.SIGTERM)
            try:
                code = proc.wait(timeout=.75)
            except subprocess.TimeoutExpired:
                code = None
            facts.update(returncode_before_release=code,
                         elapsed_seconds=time.monotonic() - started,
                         cleanup_completed_before_release=(gates / "restart-cleanup-completed.json").exists(),
                         switch_completed_before_release=(gates / "restart-switch-completed.json").exists())
            (gates / "observation.json").write_text(json.dumps(facts, indent=2) + "\n")
            print("RESTART_SIGNAL_OBSERVATION", json.dumps(facts), flush=True)
            assert code is None, "executable exited while required restart cleanup was still held"
            (gates / "release-restart-cleanup").touch()
            assert proc.wait(timeout=10) == 42
            assert (gates / "restart-cleanup-completed.json").exists()
            assert (gates / "restart-switch-completed.json").exists()
    finally:
        (gates / "release").touch()
        (gates / "release-restart-cleanup").touch()
        if proc is not None:
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                pass
        error = container.reap()
        container.close()
        if proc is not None:
            proc.wait(timeout=5)
        (gates / "reaped.json").write_text(json.dumps({
            "error": error, "returncode": proc.returncode if proc else None,
        }))
        evidence = Path(os.environ["OUROBOROS_TEST_EVIDENCE_ROOT"])
        shutil.copytree(gates, evidence / f"restart-signal-{entrypoint}-{ingress}")
        if error:
            retain_tree(tmp_path, error)
        assert not error, error

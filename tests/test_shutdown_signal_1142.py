"""Ordinary close must not end as a SIGKILL or a false "supervisor died" alarm (#1142).

Two mechanisms, two halves:

* server half — ``server._SignalStopServer.handle_exit`` sets ``_supervisor_stop`` at the
  signal, and ``uvicorn.Config`` carries ``timeout_graceful_shutdown`` so the lifespan
  teardown (the terminal-custody write) starts inside the launcher's stop budget;
* launcher half — the POSIX graceful phase signals only the server PID, leaving the group
  SIGKILL fallback and the post-exit group sweep untouched.

The real-process test deliberately reproduces the OLD launcher behaviour (SIGTERM to the
whole group), because packaged launchers are immutable until the next release: the server
half must be self-sufficient against it.
"""

from __future__ import annotations

from tests import _dialogue as dialogue_test

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_json(url: str, key: str, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310 - local test server
                payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("supervisor_error"):
                    raise RuntimeError(f"supervisor failed to initialize: {payload['supervisor_error']}")
                if payload.get(key) is True:
                    return
        except Exception as exc:  # pragma: no cover - diagnostic only
            last = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"{url} never reported {key}=true: {last}")


@pytest.mark.skipif(sys.platform == "win32", reason="process groups and SIGTERM are POSIX")
def test_group_sigterm_reaches_terminal_custody_without_a_false_supervisor_alarm(tmp_path):
    _check_real_shutdown(tmp_path, launcher_pipe=False)


def _check_real_shutdown(tmp_path, *, launcher_pipe):
    """A real server in an isolated data root, SIGTERMed the way the OLD launcher does (whole
    group, so its Manager dies first) while a browser WebSocket is open: the lifespan teardown
    must still run inside the launcher budget and the owner chat must get no supervisor_failure."""
    from ouroboros.process_containment import ProcessContainer
    from ouroboros.runtime_limits import LAUNCHER_STOP_GRACE_SEC

    port = _free_port()
    data_dir = tmp_path / "data"
    (data_dir / "state").mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    model = "openai-compatible::never-called"
    (data_dir / "settings.json").write_text(json.dumps({
        "OPENAI_COMPATIBLE_API_KEY": "shutdown-test-key",
        "OPENAI_COMPATIBLE_BASE_URL": f"http://127.0.0.1:{port + 2}/v1",
        "OUROBOROS_MODEL": model, "OUROBOROS_MODEL_LIGHT": model, "OUROBOROS_MODEL_FALLBACKS": model,
        "OUROBOROS_MAX_WORKERS": 1,
        "OUROBOROS_RUNTIME_MODE": "light",
        # A fresh root with a ready provider runs the starter installer at boot; an explicit empty
        # list keeps it from cloning the skill sources from GitHub and installing skills mid-test.
        "OUROBOROS_STARTER_SKILLS": "[]",
    }), encoding="utf-8")
    # An owner chat is bound, so the false alarm WOULD be written if the crash counter fired.
    from devtools.benchmarks.common.server_runner import seed_owner_state
    seed_owner_state(data_dir)
    env = {
        **os.environ,
        "HOME": str(home),
        "OUROBOROS_APP_ROOT": str(tmp_path),
        "OUROBOROS_DATA_DIR": str(data_dir),
        "OUROBOROS_SETTINGS_PATH": str(data_dir / "settings.json"),
        "OUROBOROS_REPO_DIR": REPO_ROOT,
        "OUROBOROS_SERVER_HOST": "127.0.0.1",
        "OUROBOROS_SERVER_PORT": str(port),
        "OUROBOROS_HOST_SERVICE_PORT": str(port + 1),
        "OUROBOROS_MANAGED_BY_LAUNCHER": "1",
    }
    if launcher_pipe:
        env["OUROBOROS_LAUNCHER_STOP_STDIN"] = "1"
    url = f"http://127.0.0.1:{port}"
    container = ProcessContainer()
    proc = container.spawn(
        [sys.executable, "server.py"], cwd=REPO_ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.PIPE if launcher_pipe else subprocess.DEVNULL,
    )
    ws = None
    stuck = None
    try:
        _wait_json(f"{url}/api/state", "supervisor_ready", 90)
        time.sleep(1.0)  # let the loop run a few ticks against the live Manager
        from websockets.sync.client import connect

        ws = connect(f"ws://127.0.0.1:{port}/ws", open_timeout=10)  # the desktop window's socket
        # An in-flight request whose body never completes: uvicorn's graceful drain waits for
        # it, which is what parked the lifespan teardown behind the launcher budget.
        stuck = socket.create_connection(("127.0.0.1", port), timeout=5)
        stuck.sendall(
            b"POST /api/settings HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\nContent-Length: 100000\r\n\r\n{\"OUROBOROS_MODEL\": \""
        )
        time.sleep(0.5)
        started = time.monotonic()
        if launcher_pipe:
            proc.stdin.write(b"quit\n")
            proc.stdin.flush()
            proc.stdin.close()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # pre-#1142 launcher
        try:
            code = proc.wait(timeout=LAUNCHER_STOP_GRACE_SEC)
        except subprocess.TimeoutExpired:
            pytest.fail("server did not exit within the launcher stop budget — the next launcher would SIGKILL it")
        elapsed = time.monotonic() - started
    finally:
        if stuck is not None:
            stuck.close()
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if proc.poll() is None:
            proc.kill()
        try:
            error = container.reap()
        finally:
            container.close()
        proc.wait(timeout=10)  # collect the direct child; reap() skips zombies by design
        assert not error, error

    # The command boundary exits after Uvicorn's replay and main's cleanup unwind.
    expected_code = 0 if launcher_pipe else 128 + signal.SIGTERM
    assert code == expected_code, f"graceful exit expected, got {code}"
    chat_rows = dialogue_test.log_rows(data_dir / "logs" / "chat.jsonl")
    alarms = [row for row in chat_rows if row.get("system_type") == "supervisor_failure"
              or "Supervisor loop died" in str(row.get("text") or "")]
    assert alarms == [], alarms
    shutdown_rows = [row for row in _read_rows(data_dir / "logs" / "supervisor.jsonl")
                     if row.get("type") == "server_shutdown"]
    cause = "launcher_quit" if launcher_pipe else "external_signal"
    assert shutdown_rows and shutdown_rows[-1].get("cause") == cause, shutdown_rows
    # The door's entry, written before any wait, names the same cause the late row confirms.
    from datetime import datetime
    entry = json.loads((data_dir / "state" / "shutdown_entry.json").read_text(encoding="utf-8"))
    assert (entry["cause"], entry["restart_exit"]) == (cause, False), entry
    assert datetime.fromisoformat(entry["entered_at"]) <= datetime.fromisoformat(shutdown_rows[-1]["ts"])
    assert elapsed < LAUNCHER_STOP_GRACE_SEC, elapsed
    # The event-bus manager ignores the group's SIGTERM (ARCHITECTURE §5): it was ALIVE when the teardown
    # reached it, after the terminal custody that emits on it, and the RPC ended it (no kill needed). A manager
    # the signal had killed would leave its exit code here and the bus_lost door would have taken the stop.
    stopped = [row for row in _read_rows(data_dir / "logs" / "supervisor.jsonl")
               if row.get("type") == "event_queue_generation_stopped"]
    assert stopped and stopped[-1]["exitcode_before_shutdown"] is None and stopped[-1]["killed"] is False, stopped
    assert not any(row.get("type") == "supervisor_generation_fault" for row in _read_rows(data_dir / "logs" / "supervisor.jsonl"))


_PIPE_OWNER_PROBE = r'''
import io, json, os, pathlib, subprocess, sys, threading
import uvicorn
from ouroboros import server_process as state
root, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
server = state._SignalStopServer(uvicorn.Config(lambda scope, receive, send: None))
opened, fdopen = [], os.fdopen
def record_stream(*args, **kwargs):
    stream = fdopen(*args, **kwargs)
    opened.append(stream)
    return stream
os.fdopen = record_stream
try:
    server.watch_launcher_stop()
finally:
    os.fdopen = fdopen
facts = {"private_streams": len(opened), "opt_in_consumed": "OUROBOROS_LAUNCHER_STOP_STDIN" not in os.environ}
if opened:
    facts.update(raw=isinstance(opened[0], io.FileIO), inheritable=os.get_inheritable(opened[0].fileno()))
if sys.platform == "win32":
    import ctypes, msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetStdHandle.argtypes = (ctypes.c_uint32,)
    kernel.GetStdHandle.restype = ctypes.c_void_p
    facts["win32_stdin_matches_crt"] = kernel.GetStdHandle(-10) == msvcrt.get_osfhandle(0)
try:
    child = subprocess.run([sys.executable, "-c", "import sys; print(repr(sys.stdin.buffer.read()))"],
                           capture_output=True, text=True, timeout=3)
    facts["inherited_stdin_eof"] = child.returncode == 0 and child.stdout.strip() == "b''"
except subprocess.TimeoutExpired:
    facts["inherited_stdin_eof"] = False
if facts["inherited_stdin_eof"]:
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3)
    facts["git_completed"] = git.returncode == 0 and len(git.stdout.strip()) == 40
(root / "ready.json").write_text(json.dumps(facts), encoding="utf-8")
if mode == "exit-open":
    sys.exit(0)  # The owner still holds the write end; the raw reader must not stall finalization.
readers = [thread for thread in threading.enumerate() if thread.name == "launcher-stop"]
for reader in readers:
    reader.join(5)
(root / "finished.json").write_text(json.dumps({
    "should_exit": server.should_exit, "exit_latched": state._exit_signalled.is_set(),
    "supervisor_stopped": state._supervisor_stop.is_set(), "reader_alive": any(t.is_alive() for t in readers),
    "private_closed": all(stream.closed for stream in opened), "source": state.server_stop_source(),
}), encoding="utf-8")
'''


def _launch_pipe_owner_probe(tmp_path, mode):
    import sysconfig
    from pathlib import Path

    script = tmp_path / "pipe_owner.py"
    script.write_text(_PIPE_OWNER_PROBE, encoding="utf-8")
    env = {**os.environ, "OUROBOROS_LAUNCHER_STOP_STDIN": "1"}
    env["PYTHONPATH"] = os.pathsep.join([REPO_ROOT, sysconfig.get_path("purelib"), env.get("PYTHONPATH", "")])
    executable = getattr(sys, "_base_executable", sys.executable) if sys.platform == "win32" else sys.executable
    proc = subprocess.Popen([executable, str(script), str(tmp_path), mode], cwd=REPO_ROOT, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            return proc, json.loads((Path(tmp_path) / "ready.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if proc.poll() is not None:
                break
            time.sleep(.02)
    proc.kill()
    output = proc.communicate(timeout=5)
    pytest.fail(f"private-stdin probe never became ready: {output}")


@pytest.mark.parametrize("message,source", [(b"quit\n", "launcher_quit"), (b"", "launcher_gone"),
                                            (b"other", "launcher_gone")])
def test_launcher_pipe_quit_or_launcher_exit_stops_the_server(tmp_path, message, source):
    """Quit is the ordinary stop; EOF means the launcher is gone, so the opt-in server stops
    too instead of serving on as an orphan (every platform, not only under a Windows Job)."""
    proc, ready = _launch_pipe_owner_probe(tmp_path, "receive")
    try:
        assert ready["inherited_stdin_eof"], "ordinary children must not read the open control pipe"
        assert ready["git_completed"]
        assert ready["private_streams"] == 1 and ready["raw"] and not ready["inheritable"]
        assert ready["opt_in_consumed"]
        if sys.platform == "win32":
            assert ready["win32_stdin_matches_crt"], "both native stdin tables must name NUL"
        stdout, stderr = proc.communicate(input=message, timeout=10)
        assert proc.returncode == 0, (stdout, stderr)
        finished = json.loads((tmp_path / "finished.json").read_text(encoding="utf-8"))
        assert finished == {"should_exit": True, "exit_latched": True, "supervisor_stopped": True,
                            "reader_alive": False, "private_closed": True, "source": source}
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5)


def test_private_launcher_reader_allows_normal_exit_with_owner_pipe_still_open(tmp_path):
    proc, ready = _launch_pipe_owner_probe(tmp_path, "exit-open")
    try:
        assert ready["inherited_stdin_eof"] and ready["raw"]
        proc.wait(timeout=5)  # Keep proc.stdin OPEN until the process has exited naturally.
        assert proc.returncode == 0
        assert not proc.stdin.closed
    finally:
        if proc.poll() is None:
            proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
    assert b"_enter_buffered_busy" not in stderr, (stdout, stderr)


def test_direct_server_keeps_ordinary_child_stdin_without_launcher_opt_in(tmp_path):
    import sysconfig

    script = tmp_path / "direct_stdin.py"
    script.write_text('''
import json, subprocess, sys
import uvicorn
from ouroboros.server_process import _SignalStopServer
server = _SignalStopServer(uvicorn.Config(lambda scope, receive, send: None))
server.watch_launcher_stop()
child = subprocess.run([sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
                       capture_output=True, text=True, timeout=3)
print(json.dumps({"stdin": child.stdout, "child_exit": child.returncode, "should_exit": server.should_exit}))
''', encoding="utf-8")
    env = {**os.environ}
    env.pop("OUROBOROS_LAUNCHER_STOP_STDIN", None)
    env["PYTHONPATH"] = os.pathsep.join([REPO_ROOT, sysconfig.get_path("purelib"), env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, env=env,
                          input="ordinary stdin survives", capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"stdin": "ordinary stdin survives", "child_exit": 0, "should_exit": False}

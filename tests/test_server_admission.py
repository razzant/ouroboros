"""One server per actual data root: the lifetime admission of ``server_entrypoint.admit_data_root``.

Real kernel locks across real processes. A second holder of the same data root, spelled
through a symlink and a path with spaces, is refused whatever its repository, PID file or
port; the hold survives everything but process exit; exec does not inherit it and a
forked child closes its copy without unlocking its parent; a spawned Restart successor is
admitted once its parent exits; imports alone admit nothing. The real-server scenario
starts two ``server.py`` processes on one data root.
"""

from __future__ import annotations

import errno
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

_PROBE = r'''
import os, pathlib, subprocess, sys, time
from ouroboros.server_entrypoint import ServerAdmissionRefused, admit_data_root
root, wait, mode = sys.argv[1], float(sys.argv[2]), sys.argv[3]
try:
    admit_data_root(pathlib.Path(root), wait_sec=wait)
except ServerAdmissionRefused as refusal:
    print("refused:", refusal, flush=True)
    sys.exit(3)
print("admitted", flush=True)
if mode == "hold":
    time.sleep(60)
elif mode.startswith("exit-after:"):
    time.sleep(float(mode.split(":", 1)[1]))
elif mode == "fork-and-exec-then-exit":
    child = os.fork()
    if child == 0:
        time.sleep(30)
        os._exit(0)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], close_fds=False)
    print(f"survivors {child} {sleeper.pid}", flush=True)
    os._exit(0)
'''


def _probe(tmp_path, root, *, wait=0.0, mode="exit"):
    script = tmp_path / "admission_probe.py"
    script.write_text(_PROBE, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    # A Windows venv redirector has a different PID from the interpreter holding the lock.
    executable = (getattr(sys, "_base_executable", None) or sys.executable) if sys.platform == "win32" else sys.executable
    if executable != sys.executable:
        env["__PYVENV_LAUNCHER__"] = sys.executable
    return subprocess.Popen([executable, str(script), str(root), str(wait), mode], cwd=str(REPO), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _first_line(proc, timeout=20.0):
    """The child's first non-blank stdout line, or "" when none arrives in time. The blocking read
    runs on a reader thread so the bound holds even while the child stalls before printing."""
    lines: list = []

    def read():
        for raw in proc.stdout:
            if raw.strip():
                lines.append(raw.strip())
                return

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(timeout)
    return lines[0] if lines else ""


@pytest.fixture
def data_root(tmp_path):
    """The data root (a path with spaces) and a second spelling of it: a directory symlink where
    the host can create one, else the same path, so a Windows host without the symlink privilege
    keeps its LockFileEx coverage."""
    root = tmp_path / "data root with spaces"
    (root / "state").mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError):
        alias = root
    return root, alias


def test_a_second_process_is_refused_through_any_spelling_until_the_holder_exits(tmp_path, data_root):
    root, alias = data_root
    holder = _probe(tmp_path, root, mode="hold")
    try:
        assert _first_line(holder) == "admitted"
        contender = _probe(tmp_path, alias, wait=0.2)
        out, _ = contender.communicate(timeout=20)
        assert contender.returncode == 3, out
        assert "another Ouroboros server serves this data directory" in out and f"pid={holder.pid}" in out
    finally:
        holder.kill()
        holder.wait(timeout=10)
    admitted = _probe(tmp_path, alias)
    out, _ = admitted.communicate(timeout=20)
    assert admitted.returncode == 0 and out.strip() == "admitted"
    assert (root / "state" / "server.lock").exists(), "the lock file is never unlinked"


def test_a_restart_successor_is_admitted_once_its_parent_exits(tmp_path, data_root):
    root, _alias = data_root
    parent = _probe(tmp_path, root, mode="exit-after:1.0")
    try:
        assert _first_line(parent) == "admitted"
        started = time.monotonic()
        successor = _probe(tmp_path, root, wait=8.0)
        out, _ = successor.communicate(timeout=30)
        assert successor.returncode == 0 and out.strip() == "admitted", out
        assert time.monotonic() - started < 8.0
    finally:
        parent.kill()
        parent.wait(timeout=10)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fork")
def test_neither_a_forked_child_nor_an_exec_survivor_keeps_a_dead_servers_admission(tmp_path, data_root):
    root, _alias = data_root
    parent = _probe(tmp_path, root, mode="fork-and-exec-then-exit")
    survivors = []
    try:
        assert _first_line(parent) == "admitted"
        line = _first_line(parent)
        assert line.startswith("survivors "), line
        survivors = [int(pid) for pid in line.split()[1:]]
        parent.wait(timeout=10)
        for pid in survivors:
            os.kill(pid, 0)  # both still alive while the next server asks
        successor = _probe(tmp_path, root, wait=0.5)
        out, _ = successor.communicate(timeout=20)
        assert successor.returncode == 0 and out.strip() == "admitted", out
    finally:
        for pid in survivors:
            try:
                os.kill(pid, 9)
            except OSError:
                pass


def test_readmission_is_idempotent_and_the_stamp_names_the_holder(tmp_path, data_root):
    from ouroboros import server_entrypoint

    root, alias = data_root
    server_entrypoint.admit_data_root(root, wait_sec=0)
    first = server_entrypoint._admission["fd"]
    server_entrypoint.admit_data_root(alias, wait_sec=0)  # the same actual root
    assert server_entrypoint._admission["fd"] == first
    assert (root / "state" / "server.lock").read_text(encoding="ascii").strip() == f"pid={os.getpid()}"
    assert not os.get_inheritable(first)


def test_a_filesystem_without_kernel_locks_refuses_with_the_reason(tmp_path, data_root, monkeypatch):
    from ouroboros import platform_layer, server_entrypoint

    def unsupported(_fd):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", unsupported)
    with pytest.raises(server_entrypoint.ServerAdmissionRefused, match="kernel file locks are unavailable"):
        server_entrypoint.admit_data_root(data_root[0], wait_sec=5)
    assert server_entrypoint._admission == {}


def test_importing_the_server_admits_nothing(tmp_path, data_root):
    root, _alias = data_root
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(root), "PYTHONPATH": str(REPO)}
    importer = subprocess.Popen(
        [sys.executable, "-c", "import server, sys, time; print('imported', flush=True); time.sleep(30)"],
        cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert _first_line(importer, timeout=60) == "imported"
        contender = _probe(tmp_path, root)
        out, _ = contender.communicate(timeout=20)
        assert contender.returncode == 0 and out.strip() == "admitted", out
    finally:
        importer.kill()
        importer.wait(timeout=10)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as response:  # noqa: S310
            return response.status == 200
    except Exception:
        return False


def _server_env(tmp_path, data_dir, port, *, pid_file):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return {
        **os.environ,
        "HOME": str(home),
        "OUROBOROS_APP_ROOT": str(tmp_path / "app"),
        "OUROBOROS_DATA_DIR": str(data_dir),
        "OUROBOROS_SETTINGS_PATH": str(pathlib.Path(data_dir) / "settings.json"),
        "OUROBOROS_PID_FILE": str(pid_file),
        "OUROBOROS_SERVER_HOST": "127.0.0.1",
        "OUROBOROS_SERVER_PORT": str(port),
        "OUROBOROS_HOST_SERVICE_PORT": str(_free_port()),
    }


def test_two_real_servers_on_one_data_root_admit_one_and_a_launcher_exit_ends_it(tmp_path, data_root):
    """Server A is launcher-shaped (private stop pipe), server B a direct start through a
    symlinked spelling with its own port and PID file: B exits 2 without serving while A
    keeps serving; A's launcher pipe reaching EOF ends A; B then starts on that root."""
    from ouroboros.process_containment import ProcessContainer

    root, alias = data_root
    port_a, port_b = _free_port(), _free_port()
    env_a = {**_server_env(tmp_path, root, port_a, pid_file=tmp_path / "a.pid"),
             "OUROBOROS_MANAGED_BY_LAUNCHER": "1", "OUROBOROS_LAUNCHER_STOP_STDIN": "1"}
    env_b = _server_env(tmp_path, alias, port_b, pid_file=tmp_path / "b.pid")
    env_b.pop("OUROBOROS_LAUNCHER_STOP_STDIN", None)
    container = ProcessContainer()
    try:
        server_a = container.spawn([sys.executable, "server.py"], cwd=str(REPO), env=env_a, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 120
        while not _healthy(port_a):
            assert server_a.poll() is None and time.monotonic() < deadline, "server A never became healthy"
            time.sleep(0.5)
        server_b = container.spawn([sys.executable, "server.py"], cwd=str(REPO), env=env_b, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = server_b.communicate(timeout=90)
        assert server_b.returncode == 2, out
        assert "Server not started" in out and "another Ouroboros server" in out, out
        assert not _healthy(port_b) and _healthy(port_a)

        started = time.monotonic()
        server_a.stdin.close()  # the launcher is gone: EOF on its private pipe
        server_a.wait(timeout=30)
        assert server_a.returncode == 0 and time.monotonic() - started < 25
        bindings = json.loads((root / "state" / "server_port.bindings.json").read_text(encoding="utf-8") or "{}")
        assert "main" not in bindings, "a stopped server retires its endpoint publication"

        server_b2 = container.spawn([sys.executable, "server.py"], cwd=str(REPO), env=env_b, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 120
        while not _healthy(port_b):
            assert server_b2.poll() is None and time.monotonic() < deadline, "server B was not admitted after A exited"
            time.sleep(0.5)
    finally:
        try:
            error = container.reap()
        finally:
            container.close()
        assert not error, error

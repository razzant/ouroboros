"""``IsolatedServer.stop`` also stops the restart successor its server spawned.

When a server cannot exec in place (always on Windows; after a failed ``os.execvpe`` on
POSIX) ``server_control.restart_current_process`` spawns its successor through
``process_custody.spawn_supervised`` and the original exits 42, so the benchmark's own
``Popen`` no longer names the server. The fixture below is that real restart path with
exec refused: the successor is a real ``server_restart_fallback`` row in a private data
root's ledger. Stops go only through the measured identity of such rows; a foreign,
reused or unmeasurable identity is never signalled, and what cannot be proven gone is
reported, including the group of a successor that died before the stop. Native Windows
is not exercised here.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

from devtools.benchmarks.common import server_runner
from devtools.benchmarks.common.server_runner import IsolatedServer
from ouroboros import process_custody
from ouroboros.process_containment import ProcessContainer, pid_is_zombie
from ouroboros.platform_layer import deliver_signal, pid_is_alive

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX exec fallback; native Windows needs its own host")

REPO = pathlib.Path(__file__).resolve().parents[1]
PURPOSE = "server_restart_fallback"  # server_control.restart_current_process

# One server generation. The last one serves until stopped; every earlier one hands over
# exactly as ``server.py`` does after its restart request: ``restart_current_process`` with
# the given exec behaviour, then exit 42 (``exit_after_shutdown(RESTART_EXIT_CODE)``).
_SERVER = r'''
import json, logging, os, pathlib, signal, subprocess, sys, time
repo, marks, gen, last, mode = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sys.path.insert(0, repo)

def mark(name, pid=0):
    # Published whole by rename: a poller never reads a created but still empty mark.
    staged = marks / f".{name}-{gen}"
    staged.write_text(str(pid or os.getpid()))
    os.replace(staged, marks / f"{name}-{gen}")

if gen == last:
    if mode == "crash":
        # Dies while a plain subprocess child, which shares its group, still runs.
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        mark("child", child.pid)
        mark("serving")
        os._exit(70)
    if mode == "ignore_term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        def graceful(_signum, _frame):
            mark("graceful")
            os._exit(0)
        signal.signal(signal.SIGTERM, graceful)
    mark("serving")
    while True:
        time.sleep(0.1)

from ouroboros import server_control
if mode != "exec":
    def refused(*_args, **_kwargs):
        raise OSError("exec refused by the fixture")
    os.execvpe = refused
os.environ["OUROBOROS_SERVER_REEXEC_ARGV_JSON"] = json.dumps(
    [sys.argv[0], repo, str(marks), str(gen + 1), str(last), mode])
server_control.restart_current_process("127.0.0.1", 0, repo_dir=pathlib.Path(repo),
                                       log=logging.getLogger("restart-fixture"))
mark("handed_over")
os._exit(42)
'''


def _alive(pid: int) -> bool:
    return pid_is_alive(pid) and not pid_is_zombie(pid)


def _wait_for(path: pathlib.Path, within: float = 60.0) -> int:
    deadline = time.monotonic() + within
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.05)
    return int(path.read_text())


def _gone(pid: int, within: float = 10.0) -> bool:
    deadline = time.monotonic() + within
    while _alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


class _Run:
    """One isolated data root whose server restarts ``depth`` times before serving."""

    def __init__(self, tmp_path: pathlib.Path, container: ProcessContainer, *, depth: int = 1, mode: str = "spawn"):
        self.data_root, self.marks = tmp_path / "data", tmp_path / "marks"
        self.data_root.mkdir()
        self.marks.mkdir()
        script = tmp_path / "server_generation.py"
        script.write_text(_SERVER, encoding="utf-8")
        env = {**os.environ, "OUROBOROS_DATA_DIR": str(self.data_root),
               "OUROBOROS_SETTINGS_PATH": str(self.data_root / "settings.json")}
        env.pop("OUROBOROS_SERVER_REEXEC_ARGV_JSON", None)
        self.server = IsolatedServer(tmp_path / "clone", self.data_root, self.data_root / "settings.json")
        self.server.stop_grace_sec = 3.0
        self.server.proc = container.spawn(
            [sys.executable, str(script), str(REPO), str(self.marks), "0", str(depth), mode],
            cwd=str(REPO), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.serving = _wait_for(self.marks / f"serving-{depth}")

    def rows(self) -> list[dict]:
        readable, rows = process_custody._read_ledger_strict(self.data_root)
        assert readable
        return [row for row in rows if row.get("purpose") == PURPOSE]


@pytest.fixture
def container():
    box = ProcessContainer()
    try:
        yield box
    finally:
        box.reap()  # the test's own backstop, whatever the stop under test did
        box.close()


def test_a_spawned_successor_outliving_the_original_is_stopped_gracefully(tmp_path, container):
    run = _Run(tmp_path, container)
    assert run.server.proc.wait(timeout=30) == 42, "the original handed over and exited"
    assert run.serving != run.server.proc.pid and _alive(run.serving)
    (row,) = run.rows()
    assert (row["pid"], row["scope"], row["pgid"]) == (run.serving, "daemon", run.serving)
    unconfirmed = run.server.stop()
    assert _gone(run.serving), "the successor outlived the stop"
    assert _wait_for(run.marks / "graceful-1", 5) == run.serving, "the successor got the graceful stop first"
    assert unconfirmed == [] and run.server.stop_unconfirmed == []
    assert run.rows() == [row], "the stop rewrote no custody record"


def test_a_chained_restart_stops_only_the_live_last_successor(tmp_path, container):
    run = _Run(tmp_path, container, depth=2)
    assert run.server.proc.wait(timeout=30) == 42
    intermediate = _wait_for(run.marks / "handed_over-1")
    assert _gone(intermediate), "the intermediate successor exited 42 itself"
    assert sorted(row["pid"] for row in run.rows()) == sorted([intermediate, run.serving])
    unconfirmed = run.server.stop()
    assert _gone(run.serving) and _wait_for(run.marks / "graceful-2", 5) == run.serving
    assert unconfirmed == []


def test_an_ordinary_same_pid_exec_is_stopped_through_the_original_handle(tmp_path, container):
    run = _Run(tmp_path, container, mode="exec")
    assert run.serving == run.server.proc.pid and run.server.proc.poll() is None
    assert run.rows() == [], "an exec in place spawns no successor"
    unconfirmed = run.server.stop()
    assert run.server.proc.poll() is not None
    assert _wait_for(run.marks / "graceful-1", 5) == run.serving
    assert unconfirmed == []


def test_a_successor_ignoring_the_graceful_stop_is_forced_and_a_repeat_stop_is_a_no_op(tmp_path, container):
    run = _Run(tmp_path, container, mode="ignore_term")
    run.server.stop_grace_sec = 0.5
    assert run.server.proc.wait(timeout=30) == 42
    unconfirmed = run.server.stop()
    assert _gone(run.serving) and not (run.marks / "graceful-1").exists()
    assert unconfirmed == []
    assert run.server.stop() == [], "nothing left to stop, nothing claimed unconfirmed"


def test_a_failed_forced_stop_stays_unconfirmed_and_keeps_the_files(tmp_path, container, monkeypatch):
    run = _Run(tmp_path, container, mode="ignore_term")
    run.server.stop_grace_sec = 0.3
    assert run.server.proc.wait(timeout=30) == 42
    monkeypatch.setattr(server_runner, "kill_pid_tree", lambda *_args, **_kwargs: None)
    unconfirmed = run.server.stop()
    assert _alive(run.serving) and run.rows() and run.data_root.is_dir()
    assert unconfirmed == [f"restart successor {run.serving}: exit unconfirmed after a forced stop"]


def test_a_successor_that_died_leaving_its_group_running_is_unconfirmed_and_never_signalled(tmp_path, container):
    run = _Run(tmp_path, container, mode="crash")
    assert run.server.proc.wait(timeout=30) == 42
    child = _wait_for(run.marks / "child-1")
    assert _gone(run.serving), "the successor died before any stop"
    (row,) = run.rows()
    assert (row["pid"], row["pgid"]) == (run.serving, run.serving) and _alive(child)
    unconfirmed = run.server.stop()
    assert _alive(child), "a group whose leader is gone grants no signal"
    assert unconfirmed == [f"restart successor {run.serving}: process group {run.serving} "
                           "still has live members, not signalled"]
    assert run.rows() == [row], "the stop rewrote no custody record"
    deliver_signal(child, graceful=False)  # the test's own child, by the pid it published
    assert _gone(child)
    assert run.server.stop() == [], "once the group is empty the same stop is confirmed"


def test_an_unreadable_ledger_is_unconfirmed_and_nothing_is_signalled(tmp_path, container):
    run = _Run(tmp_path, container)
    assert run.server.proc.wait(timeout=30) == 42
    ledger = process_custody.ledger_path(run.data_root)
    intact = ledger.read_bytes()
    ledger.write_bytes(intact + b"{not json\n")
    unconfirmed = run.server.stop()
    assert _alive(run.serving), "no weaker evidence may stand in for the unreadable record"
    assert len(unconfirmed) == 1 and unconfirmed[0].startswith("restart successor custody unreadable")
    ledger.write_bytes(intact)  # once readable again, the same stop completes
    unconfirmed = run.server.stop()
    assert _gone(run.serving) and unconfirmed == []


def _foreign_sleeper(container: ProcessContainer) -> subprocess.Popen:
    proc = container.spawn([sys.executable, "-c", "import time; time.sleep(120)"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 10
    while not process_custody.process_fingerprint(proc.pid).get("start_time"):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    return proc


def _record(data_root: pathlib.Path, row: dict) -> None:
    ledger = process_custody.ledger_path(data_root)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _row(pid: int, fingerprint: dict) -> dict:
    return {"ts": "2026-10-29T00:00:00+00:00", "pid": pid, "pgid": pid, "fingerprint": fingerprint,
            "purpose": PURPOSE, "scope": "daemon", "owner_task": "",
            "session_id": "foreign"}


def test_a_reused_pid_under_a_successor_row_is_never_signalled(tmp_path, container):
    data_root = tmp_path / "data"
    sleeper = _foreign_sleeper(container)
    measured = process_custody.process_fingerprint(sleeper.pid)
    _record(data_root, _row(sleeper.pid, {**measured, "start_time": "Thu Jan  1 00:00:00 1970",
                                          "start_time_boot": ""}))
    server = IsolatedServer(tmp_path / "clone", data_root, data_root / "settings.json")
    unconfirmed = server.stop()
    assert sleeper.poll() is None, "the reused pid was not signalled"
    assert unconfirmed == [], "a measured different birth is someone else's process"


def test_an_unmeasurable_identity_is_unconfirmed_and_never_signalled(tmp_path, container, monkeypatch):
    data_root = tmp_path / "data"
    sleeper = _foreign_sleeper(container)
    _record(data_root, _row(sleeper.pid, process_custody.process_fingerprint(sleeper.pid)))
    monkeypatch.setattr(process_custody, "process_start_time", lambda _pid: "")
    monkeypatch.setattr(process_custody, "_live_cmd_sha256", lambda _pid: "")
    server = IsolatedServer(tmp_path / "clone", data_root, data_root / "settings.json")
    unconfirmed = server.stop()
    assert sleeper.poll() is None
    assert unconfirmed == [f"restart successor {sleeper.pid}: identity unmeasurable, not signalled"]

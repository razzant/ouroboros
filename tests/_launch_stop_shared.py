"""What the launch-stop tests share: a service that puts its server in another group, and how they watch it.

Real consumers drive real processes. Each service is a runner that starts a loopback server on port 0 in
ANOTHER process group (``start_new_session``), the shape a test fixture or a dev-server runner has. The
server answering a connection is the observation that matters; its measured identity (pid and birth,
``process_census``) is checked beside it, never a bare pid. Faults are injected only at the OS boundary the
stop reads (a denied signal, an unreadable process table or ledger, a failed write). The controlled identities
at the end stand in for that boundary where a host cannot stage the case: a dead leader's pid given to another.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATIENCE = 120.0  # a bound on causal waits (an interpreter starting on a loaded host), never a pacing sleep


SERVER = r'''
import os, socket, sys
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
sock.listen(16)
tmp = sys.argv[1] + ".tmp"
with open(tmp, "w") as handle:
    handle.write(f"{sock.getsockname()[1]} {os.getpid()}")
os.replace(tmp, sys.argv[1])
while True:
    conn, _ = sock.accept()
    conn.sendall(b"pong\n")
    conn.close()
'''


# argv: server script, port file, mode, then ``key=value`` options. ``separate`` puts the server in its
# own process group; ``same`` keeps it in the runner's; a mode ending in ``-exit`` makes the runner exit
# once the server answers, as a crashed test runner or a launcher script does. TERM keeps its default
# (immediate death, no cleanup), like pytest's. ``out=FILE`` writes an output in the cwd before READY;
# ``hidden=1`` starts the server without any scope marker, as a binary hiding its environment shows on macOS
# (only the runner's parent chain reaches it); ``late=FILE`` starts the server only once FILE exists, while the
# runner keeps running; ``daemon=PIDFILE`` also starts a child that sheds every scope
# marker and leads its own session, the shape of the installation's managed daemon (``daemon_child=PIDFILE``:
# that child starts one of its own).
RUNNER = r'''
import os, pathlib, re, subprocess, sys, time
server, port_file, mode = sys.argv[1], sys.argv[2], sys.argv[3]
options = dict(arg.split("=", 1) for arg in sys.argv[4:])
shed = {key: value for key, value in os.environ.items()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*_SCOPE_[0-9a-f]{32}", key)}
while "late" in options and not os.path.exists(options["late"]):
    time.sleep(0.01)
subprocess.Popen([sys.executable, server, port_file], stdin=subprocess.DEVNULL,
                 start_new_session=mode.startswith("separate"), env=shed if "hidden" in options else None)
if "daemon" in options:
    code = "import time; time.sleep(120)"
    if "daemon_child" in options:
        code = ("import pathlib, subprocess, sys, time; child = subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(120)']); pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(120)")
    child = subprocess.Popen([sys.executable, "-c", code, options.get("daemon_child", "")], env=shed,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    pathlib.Path(options["daemon"]).write_text(str(child.pid), encoding="utf-8")
while not os.path.exists(port_file):
    time.sleep(0.01)
if "out" in options:
    pathlib.Path(options["out"]).write_text("generated", encoding="utf-8")
print("READY", flush=True)
if mode.endswith("-exit"):
    sys.exit(0)
while True:
    time.sleep(60)
'''


HIDDEN_TOP_RUNNER = r'''
import os, subprocess, sys, time
server, port_file = sys.argv[1], sys.argv[2]
# ``sh -c 'cmd; :'`` keeps a shell alive above the server; a platform /bin/sh hides its environment on macOS
subprocess.Popen(["/bin/sh", "-c", f"{sys.executable} {server} {port_file}; :"], stdin=subprocess.DEVNULL,
                 start_new_session=True)
while not os.path.exists(port_file):
    time.sleep(0.01)
print("READY", flush=True)
'''


def write_scripts(tmp_path: Path) -> Path:
    root = tmp_path / "scripts"
    root.mkdir()
    (root / "server.py").write_text(SERVER, encoding="utf-8")
    (root / "runner.py").write_text(RUNNER, encoding="utf-8")
    return root


def _runner_cmd(scripts: Path, port_file: Path, mode: str = "separate", *extra: str) -> list[str]:
    return [sys.executable, str(scripts / "runner.py"), str(scripts / "server.py"), str(port_file), mode, *extra]


def _server(port_file: Path) -> tuple[int, int, str]:
    """(port, pid, birth) of the server the runner started; the runner prints READY only after this file exists."""
    from ouroboros.process_census import measure_birth

    deadline = time.monotonic() + _PATIENCE
    while not port_file.exists():
        assert time.monotonic() < deadline, f"{port_file.name}: the server never started"
        time.sleep(0.02)
    port, pid = (int(part) for part in port_file.read_text(encoding="utf-8").split())
    birth = measure_birth(pid)
    assert birth, "the server must be alive with a measurable birth"
    return port, pid, birth


def _pid_from(pidfile: Path) -> tuple[int, str]:
    """(pid, birth) a fixture wrote to ``pidfile``."""
    from ouroboros.process_census import measure_birth

    deadline = time.monotonic() + _PATIENCE
    while not pidfile.exists() or not pidfile.read_text(encoding="utf-8").strip():
        assert time.monotonic() < deadline, f"{pidfile.name} was never written"
        time.sleep(0.02)
    pid = int(pidfile.read_text(encoding="utf-8"))
    birth = measure_birth(pid)
    assert birth, f"{pidfile.name}: process {pid} must be alive"
    return pid, birth


def _answers(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
            return conn.recv(16) == b"pong\n"
    except OSError:
        return False


def _same_process_alive(pid: int, birth: str) -> bool:
    from ouroboros.process_census import IDENTITY_ALIVE, read_identities

    identity = read_identities([pid])[pid]
    return identity.state == IDENTITY_ALIVE and identity.birth == birth


def _end(pid: int, birth: str) -> None:
    """Test teardown: end a process this test started, only while its measured identity still matches."""
    if birth and _same_process_alive(pid, birth):
        os.kill(pid, signal.SIGKILL)


def _payload(text: str) -> dict:
    """The JSON object a stop result carries, after an optional ``⚠️ CODE ...`` head line."""
    body = text.partition("\n\n")[2] if text.startswith("⚠️") else text
    return json.JSONDecoder().raw_decode(body)[0]


def _registry(tmp_path, monkeypatch, task_id="stop-task"):
    from ouroboros import config as cfg
    from ouroboros.tools.registry import ToolRegistry

    cfg.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, raising=False)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    (tmp_path / "repo").mkdir(exist_ok=True)
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.task_id = task_id
    return registry


def _start(registry, scripts, port_file, name, mode="separate", options=(), **extra):
    started = json.loads(registry.execute("start_service", {
        "name": name, "cmd": _runner_cmd(scripts, port_file, mode, *options),
        "readiness": {"log_contains": "READY", "timeout_sec": 20}, **extra}))
    assert started["state"] in ("running", "exited"), started  # ``_server`` waits for the server itself
    return started


def _wait_exited(registry, name: str) -> None:
    deadline = time.monotonic() + _PATIENCE
    while json.loads(registry.execute("service_status", {"name": name}))["state"] != "exited":
        assert time.monotonic() < deadline, f"{name}'s runner never exited"
        time.sleep(0.05)


def _count_census(monkeypatch) -> list:
    from ouroboros import process_scope

    calls, real = [], process_scope.census
    monkeypatch.setattr(process_scope, "census", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def _order_spy(monkeypatch, by_thread: dict | None = None) -> list:
    """Every signal sent (``kill``/``killpg``, signal 0 excluded), process-table scan (``ps -A``, ``ps -axo``,
    ``pgrep``) and identity read (``ps -p``), in the order they happen; ``by_thread`` also files each under
    its thread's name."""
    events, run, kill, killpg = [], subprocess.run, os.kill, os.killpg

    def note(kind, value):
        events.append((kind, value))
        if by_thread is not None:
            by_thread.setdefault(threading.current_thread().name, []).append((kind, value))

    def spy_run(argv, *args, **kwargs):
        words = [str(part) for part in argv] if isinstance(argv, (list, tuple)) else [str(argv)]
        if words and (words[0] == "pgrep" or (words[0] == "ps" and ("-A" in words or "-axo" in words))):
            note("scan", words[0])
        elif words and words[0] == "ps" and "-p" in words:
            note("read", words[words.index("-p") + 1])
        return run(argv, *args, **kwargs)

    def spy_kill(pid, sig):
        if sig:
            note("kill", int(pid))
        return kill(pid, sig)

    def spy_killpg(pgid, sig):
        if sig:
            note("kill", -int(pgid))
        return killpg(pgid, sig)

    monkeypatch.setattr(subprocess, "run", spy_run)
    monkeypatch.setattr(os, "kill", spy_kill)
    monkeypatch.setattr(os, "killpg", spy_killpg)
    return events


def _read_through_ps(monkeypatch) -> None:
    """Select ``process_census``'s ``ps`` reader on any POSIX host, for a test of THAT reader's own bound: only this
    module stops seeing ``/proc``. It proves nothing about ``/proc``, Linux's active reader: those reads are in-process
    and honour no ``timeout``, so a read that stalls there holds a shutdown's measured pass (unresolved, not asserted)."""
    from ouroboros import process_census

    if shutil.which("ps") is None:
        pytest.skip("no ps on this host (the documented Docker image ships none): it has no such reader")

    class Host:
        path = SimpleNamespace(isdir=lambda path: path != "/proc" and os.path.isdir(path))

        def __getattr__(self, name):
            return getattr(os, name)

    monkeypatch.setattr(process_census, "os", Host())


def _deny_signals_to(monkeypatch, pids: set):
    """An EPERM for these pids at the one OS call the measured stop signals through; returns the undo."""
    from ouroboros import platform_layer

    real = platform_layer.deliver_signal

    def deliver(pid, *, graceful):
        return platform_layer.SIGNAL_DENIED if int(pid) in pids else real(pid, graceful=graceful)

    monkeypatch.setattr(platform_layer, "deliver_signal", deliver)
    return lambda: monkeypatch.setattr(platform_layer, "deliver_signal", real)


def _executor_ctx(tmp_path, task_id="exec-task"):
    from ouroboros.tools.registry import ToolContext

    workspace = tmp_path / "project"
    workspace.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=tmp_path / "system", drive_root=tmp_path / "data", task_id=task_id,
                      workspace_root=workspace, workspace_mode="external")
    ctx.executor_ref = {"type": "local", "workspace_host_path": str(workspace), "workspace_backend_path": "/workspace"}
    return ctx, workspace


def _executor_start(ctx, workspace, scripts, port_file, name, mode="separate", options=()):
    from ouroboros.tools import services

    started = json.loads(services._start_service(ctx, name=name, cmd=_runner_cmd(scripts, port_file, mode, *options),
                                                 cwd=str(workspace), readiness={"log_contains": "READY",
                                                                                "timeout_sec": 20}))
    assert started["executor"]["type"] == "local", started
    return started


def _ledgered_service(tmp_path, scripts, name, mode="separate", *, scope="session", owner_task="t", drive=None,
                      options=()):
    """A service launch through the one spawn chokepoint, with its launch marker in its env and its row."""
    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX, SCOPE_VALUE, mint_marker
    from ouroboros.process_custody import spawn_supervised

    marker = mint_marker(LAUNCH_SCOPE_PREFIX)
    leader = spawn_supervised(_runner_cmd(scripts, tmp_path / f"{name}.port", mode, *options), drive_root=drive or tmp_path,
                              purpose=f"service:{name}", scope=scope, owner_task_id=owner_task, scope_marker=marker,
                              env={**os.environ, marker: SCOPE_VALUE}, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    assert leader.stdout.readline().strip() == b"READY"
    return leader, _server(tmp_path / f"{name}.port")


def _stranger(other_than: str) -> tuple[subprocess.Popen, str]:
    """A live innocent process in its own group, measurably not the process ``other_than`` names (a birth
    is measured to the second: another one is awaited rather than assumed)."""
    from ouroboros.process_census import measure_birth

    while True:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True,
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        birth = measure_birth(proc.pid)
        if birth and birth != other_than:
            return proc, birth
        proc.kill()
        proc.wait(timeout=_PATIENCE)
        time.sleep(0.05)


def _settle(leader, *procs, identities=()) -> None:
    """Test teardown: the Popen handles this test holds, then the measured identities it started."""
    for proc in (leader, *procs):
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=_PATIENCE)
    for pid, birth in identities:
        _end(pid, birth)


# --------------------------------------------------------------------------- controlled identities
# Table-and-signal doubles for launches whose leader pid is reused: no OS process is signalled.

_CONTROLLED_PIDS = 880000  # from here up, a pid is this table's own
_PID, _CHILD, _CARRIER, _OTHER = 880201, 880202, 880203, 880299
_OLD_BIRTH, _NEW_BIRTH = "Mon Nov 2 12:00:00 2026", "Mon Nov 2 13:00:00 2026"
_HEX_A, _HEX_B = "b" * 32, "c" * 32
_MARK_A, _MARK_B = "OURO_LAUNCH_SCOPE_" + _HEX_A, "OURO_LAUNCH_SCOPE_" + _HEX_B


def _launch_row(owner, *, marker=_MARK_A, birth=_OLD_BIRTH, purpose="service:old", session="previous-generation",
                pid=_PID):
    row = {"pid": pid, "pgid": pid, "purpose": purpose, "scope": "task", "owner_task": owner,
           "session_id": session, "fingerprint": {}}
    return {**row, "scope_birth": birth, "scope_marker": marker} if marker else row


def _controlled(tmp_path, monkeypatch, rows):
    """Controlled identity readers, signals and kills over the pid table ``rows`` (no OS process is signalled,
    files stay in ``tmp_path``). A row's own leader is the live pid with its recorded launch birth; an unmarked row
    records none, so any live process on its pid is its leader. ``world.unknown`` makes the census unreadable and
    ``world.unreadable`` the identity of the pids it names. It relabels no platform: custody files and file locks stay
    this host's own (Windows' too), and a Windows ``end_scope`` still answers ``unsupported``, so only a test of what
    custody merges runs there. A controlled pid that reaches an OS kill past these doubles is refused and named in
    ``world.native``, on every host."""
    from ouroboros import owned_shutdown, platform_layer, process_census, process_custody, process_scope
    from ouroboros.utils import append_jsonl

    world = SimpleNamespace(signals=[], kills=[], native=[], denied=set(), unknown=False, censuses=[], unreadable=set())

    def census(*args, **kwargs):  # one process table per call, unreadable while ``world.unknown``
        world.censuses.append(args)
        return None if world.unknown else dict(rows)

    def identities(pids, **_bound):  # a running shutdown bounds each read (``timeout``)
        return {item: process_census._Identity("unreadable") if item in world.unreadable else
                process_census._Identity(rows[item].state, rows[item].birth, rows[item].pgid)
                if item in rows else process_census._Identity("absent") for item in pids}

    def signal(target, *, graceful):
        world.signals.append((target, rows[target].birth if target in rows else ""))
        if target in world.denied:
            return platform_layer.SIGNAL_DENIED
        end(target)
        return platform_layer.SIGNAL_SENT

    def end(target):
        rows.pop(target, None)
        for row in list(rows.values()):
            if row.ppid == target:
                row.ppid = 1

    def kill(target, **_kw):  # an older row's group or tree kill
        world.kills.append(target)
        end(target)

    def same(row):
        live = rows.get(int(row["pid"]))
        return live is not None and live.birth == str(row.get("scope_birth") or live.birth)

    def refusing(name, real):  # a pid no process holds, whatever the OS would find there; signal 0 only probes POSIX
        def call(target, sig):
            if abs(int(target)) >= _CONTROLLED_PIDS and (sig or os.name == "nt"):  # negative: its group
                world.native.append((name, abs(int(target))))
                raise ProcessLookupError(errno.ESRCH, f"controlled pid {target} has no process")
            return real(target, sig)
        return call

    def hidden_run(command, *args, _real=platform_layer._hidden_run, **kwargs):  # Windows' one tree and pid kill
        named = [int(word) for word in command[1:] if str(word).isdigit() and int(word) >= _CONTROLLED_PIDS]
        if command[0] == "taskkill" and named:
            world.native.append(("taskkill", named[0]))
            return subprocess.CompletedProcess(command, 128)
        return _real(command, *args, **kwargs)

    for name in ("kill", "killpg"):  # Windows has no killpg
        if hasattr(os, name):
            monkeypatch.setattr(os, name, refusing(name, getattr(os, name)))
    monkeypatch.setattr(platform_layer, "_hidden_run", hidden_run)
    alive = platform_layer.pid_is_alive
    monkeypatch.setattr(process_custody, "_fingerprint_matches", lambda row, **kw: same(row))
    monkeypatch.setattr(process_custody, "_service_group_survives_leader", lambda row: False)
    monkeypatch.setattr(process_custody, "measured_identity_mismatch", lambda row: (
        {"recorded": {"start_time": row.get("scope_birth")}, "measured": {"start_time": rows[int(row["pid"])].birth},
         "mismatches": ["start_time"]} if int(row["pid"]) in rows and not same(row) else {}))
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda root: set())
    monkeypatch.setattr(process_custody, "live_daemon_root_pids", lambda *a, **kw: set())
    monkeypatch.setattr(process_custody, "kill_process_group_id", kill)
    monkeypatch.setattr(platform_layer, "kill_process_group_id", kill)
    monkeypatch.setattr(platform_layer, "kill_pid_tree", kill)
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda target: target in rows if target >= _CONTROLLED_PIDS else alive(target))
    monkeypatch.setattr(process_custody, "pid_is_alive", lambda target: target in rows)
    monkeypatch.setattr(process_scope, "census", census)
    monkeypatch.setattr(process_scope, "read_identities", identities)
    monkeypatch.setattr(process_scope, "birth_epoch", lambda value: 123.0 if value else 0.0)
    monkeypatch.setattr(platform_layer, "deliver_signal", signal)
    monkeypatch.setattr(process_scope, "_RESCAN_SETTLE_SEC", 0)
    monkeypatch.setattr(owned_shutdown, "_exit_proven", lambda target: target not in rows)
    process_custody.ledger_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)

    def spawned(row):  # what record_process writes after a Popen: one appended row, then its ownership record
        assert append_jsonl(process_custody.ledger_path(tmp_path), row)
        assert owned_shutdown.record_ledgered_process(tmp_path, row)

    world.spawned = spawned
    return world


def _view(root):
    from ouroboros import owned_shutdown, process_custody

    readable, rows = process_custody._read_ledger_strict(root)
    assert readable
    return rows, {entry["record_id"]: entry for entry in owned_shutdown.owned_records(root, strict=True)}


def _owed(root, leader: int) -> tuple[list, list]:
    """The pids the latest ledger row and the ownership copy of the launch of ``leader`` still name."""
    from ouroboros import owned_shutdown, process_custody

    rows = [row for row in process_custody._read_ledger(root) if int(row.get("pid") or 0) == leader]
    copies = [record["ledger_entry"] for record in owned_shutdown.owned_records(root)
              if int((record.get("ledger_entry") or {}).get("pid") or 0) == leader]

    def names(found):
        return sorted(int(item["pid"]) for row in found for item in row.get("known_members") or [])

    return names(rows), names(copies)


def _reader(tmp_path, monkeypatch, reader):
    """A reader of ``tmp_path``'s custody and the task id its launch row names."""
    if reader == "presence":
        from ouroboros.presence_continuation import _custody_blockers

        return (lambda: _custody_blockers(tmp_path, "t-a")), "t-a"
    if reader == "cold sleep":
        from ouroboros.model_sleep import cold_blockers
        from tests.test_batch4_producer_custody import _registry

        registry, _queue, _workers = _registry(tmp_path, monkeypatch)
        return (lambda: cold_blockers(registry._ctx)), "root"
    if reader == "continuation":
        from supervisor.continuation_admission import conflicting_writers
        from tests._budget_pause_exact_helpers import _install_queue
        from tests.test_owner_continue import _interrupted

        queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
        _interrupted(tmp_path)
        return (lambda: conflicting_writers(queue, "pred-1")), "pred-1"
    from ouroboros import body_candidate

    return (lambda: body_candidate.live_lineage_processes(tmp_path, "t-a")), "t-a"


def _held(found, pid):
    """``live``: a running owned process; ``unknown``: the launch is held unobserved; ``none``: not held."""
    if all(isinstance(item, int) for item in found):  # a body candidate's pids, -1 for an unknown record
        return "live" if pid in found else "unknown" if -1 in found else "none"
    if any(item.get("kind") == "owned_process" and str(item.get("detail") or item.get("pid")) == str(pid)
           for item in found):
        return "live"
    return "unknown" if any(item.get("detail") == f"launch of process {pid} unobserved" for item in found) else "none"

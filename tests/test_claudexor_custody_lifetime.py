"""The installation daemon survives worker teardown and generation sweeps; Panic ends it.

A task worker may be the process that first needs the shared Claudexor daemon, so the
daemon is a PID-tree descendant of that worker while its process group, custody row and
paid runs belong to the installation. Every worker tree-kill therefore spares the
ledger's live ``daemon``-scope roots, both server sweeps keep the purpose's legacy
session rows, and the explicit stop signals only a root whose recorded identity was
measured live — never a name, a port, or a row the reaper merely could not measure.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import time

import pytest

from ouroboros import claudexor_daemon, process_custody
from ouroboros.platform_layer import pid_is_alive

_POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups; Windows Job Objects")

_DAEMON = """
import http.server, json
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'work-survives')
    def do_POST(self):
        if self.headers.get('Authorization') != 'Bearer fixture-token':
            self.send_response(401); self.end_headers(); return
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({'compatible': True, 'protocolMajor': 3,
            'engine': {'version': '3.9.8', 'sha': 'a' * 40}}).encode())
server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
"""
_WORKER = """
import json, pathlib, subprocess, sys, time
from ouroboros.process_custody import spawn_supervised
proc = spawn_supervised([sys.executable, '-u', '-c', sys.argv[2]],
    drive_root=pathlib.Path(sys.argv[1]), purpose=sys.argv[3], scope='daemon',
    stdout=subprocess.PIPE, text=True)
print(json.dumps({'pid': proc.pid, 'port': int(proc.stdout.readline())}), flush=True)
time.sleep(60)
"""


def _echoes(port: int) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/alive", timeout=1) as response:
            return response.read() == b"work-survives"
    except OSError:
        return False


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_is_alive(pid) or process_custody.pid_is_zombie(pid):
            return True
        time.sleep(0.05)
    return False


def _supervisor_rows(root: pathlib.Path, kind: str) -> list:
    path = root / "logs" / "supervisor.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == kind]


@pytest.fixture
def worker_with_daemon(tmp_path, monkeypatch):
    """A real dummy worker that spawned a real echo daemon into ``tmp_path``'s ledger.

    Never Claudexor itself: the contract under test is process lifetime, and a socket
    echo AFTER each teardown is the proof that work continued, not merely that a pid
    existed. Both processes are torn down by exact handle in teardown.
    """
    import ouroboros.config as config_mod
    from ouroboros.platform_layer import kill_pid_tree, subprocess_new_group_kwargs
    from supervisor import queue as q

    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    worker = subprocess.Popen(
        [sys.executable, "-u", "-c", _WORKER, str(tmp_path), _DAEMON, claudexor_daemon.CUSTODY_PURPOSE],
        stdout=subprocess.PIPE, text=True, **subprocess_new_group_kwargs(),
    )
    daemon = json.loads(worker.stdout.readline())
    assert _echoes(daemon["port"])
    claudexor_daemon._write_ownership_marker()
    descriptor = claudexor_daemon.owned_descriptor_path()
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    token = descriptor.parent / "token"
    token.write_text("fixture-token")
    descriptor.write_text(json.dumps({"host": "127.0.0.1", "port": daemon["port"], "tokenPath": str(token)}))
    try:
        yield worker, daemon
    finally:
        if worker.poll() is None:
            kill_pid_tree(worker.pid)
        worker.wait(timeout=5)
        worker.stdout.close()
        if pid_is_alive(daemon["pid"]):
            kill_pid_tree(daemon["pid"])


@_POSIX
@pytest.mark.serial
def test_worker_tree_kill_spares_the_daemon_and_a_new_manager_stops_it(tmp_path, monkeypatch, worker_with_daemon):
    from supervisor.worker_pool_lifecycle import kill_worker_tree

    worker, daemon = worker_with_daemon
    assert process_custody.live_daemon_root_pids(tmp_path) == {daemon["pid"]}
    # A generation change: the sweep keeps it, then the REAL shared worker tree-kill.
    monkeypatch.setattr(process_custody, "_SESSION_ID", "next-server-generation")
    assert process_custody.reap_orphaned_processes(
        tmp_path, retained_purposes={claudexor_daemon.CUSTODY_PURPOSE}) == []
    kill_worker_tree(worker.pid)
    worker.wait(timeout=5)
    assert _echoes(daemon["port"]), "the daemon's work must survive its spawning worker"
    assert [entry["pid"] for entry in process_custody._read_ledger(tmp_path)] == [daemon["pid"]]

    # Panic's stop from a NEW manager that never spawned anything: identity is the ledger.
    manager = claudexor_daemon.OwnedClaudexorDaemon()
    assert manager._proc is None
    assert manager.stop() is True
    assert _wait_gone(daemon["pid"]), "a confirmed own daemon root dies on the explicit stop"
    assert not _echoes(daemon["port"])
    assert process_custody._read_ledger(tmp_path) == []
    rows = _supervisor_rows(tmp_path, "process_stopped")
    assert [(r["pid"], r["purpose"], r["scope"], r["reason"]) for r in rows] == [
        (daemon["pid"], claudexor_daemon.CUSTODY_PURPOSE, "daemon", "owner_stop"),
    ]
    assert manager.stop() is False, "nothing left to stop"


@_POSIX
@pytest.mark.serial
def test_cancel_and_timeout_kill_spares_the_daemon_beside_kept_services(tmp_path, monkeypatch, worker_with_daemon):
    from supervisor.worker_pool_lifecycle import kill_worker_tree

    worker, daemon = worker_with_daemon
    kill_worker_tree(worker.pid, keep_services=True)
    worker.wait(timeout=5)
    assert _echoes(daemon["port"])


@_POSIX
@pytest.mark.serial
def test_stop_and_sparing_refuse_a_row_whose_live_identity_does_not_match(tmp_path, monkeypatch):
    """A live pid under a daemon row is not a daemon: the recorded command hash must match."""
    import ouroboros.config as config_mod
    from ouroboros.platform_layer import subprocess_new_group_kwargs

    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path)
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                **subprocess_new_group_kwargs())
    try:
        entry = process_custody.record_process(
            tmp_path, pid=stranger.pid, cmd=["sleep"], purpose=claudexor_daemon.CUSTODY_PURPOSE,
            scope="daemon",
        )
        entry["fingerprint"]["cmd_sha256"] = "0" * 64
        process_custody._rewrite_ledger(tmp_path, [entry])
        assert process_custody.live_daemon_root_pids(tmp_path) == set()
        assert process_custody.stop_ledgered_processes(
            tmp_path, {claudexor_daemon.CUSTODY_PURPOSE}) == []
        assert claudexor_daemon.OwnedClaudexorDaemon().stop() is False
        assert stranger.poll() is None, "a mismatching row never authorizes a signal"
        assert len(process_custody._read_ledger(tmp_path)) == 1, "left for the reaper to judge"
        assert _supervisor_rows(tmp_path, "process_stopped") == []
    finally:
        stranger.kill()
        stranger.wait(timeout=5)


@_POSIX
@pytest.mark.serial
def test_stop_needs_measured_identity_and_an_owned_home(tmp_path, monkeypatch):
    """The reaper's permissive keep is never stop authority, and a foreign home stops nothing."""
    import ouroboros.config as config_mod
    from ouroboros.platform_layer import subprocess_new_group_kwargs

    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            **subprocess_new_group_kwargs())
    try:
        process_custody.record_process(
            tmp_path, pid=proc.pid, cmd=["sleep"], purpose=claudexor_daemon.CUSTODY_PURPOSE,
            scope="session",  # a legacy row is ours too
        )
        # Unmeasurable start time: `_fingerprint_matches` keeps the row (the sweep's safe
        # direction), the teardown still spares it, but the explicit stop refuses.
        monkeypatch.setattr(process_custody, "process_start_time", lambda _pid: "")
        assert process_custody._fingerprint_matches(process_custody._read_ledger(tmp_path)[0])
        assert process_custody.stop_ledgered_processes(
            tmp_path, {claudexor_daemon.CUSTODY_PURPOSE}) == []
        assert proc.poll() is None
        monkeypatch.undo()
        monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path)

        # Measured identity under a FOREIGN home: refused before the ledger is read.
        marker = claudexor_daemon.ownership_marker_path()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"owner": "ouroboros", "data_dir": str(tmp_path / "elsewhere")}))
        assert claudexor_daemon.verify_owned_home()
        assert claudexor_daemon.OwnedClaudexorDaemon().stop() is False
        assert proc.poll() is None
        marker.unlink()

        # Missing marker is not stop authority, although first provisioning remains allowed.
        assert claudexor_daemon.verify_owned_home() == ""
        assert claudexor_daemon.OwnedClaudexorDaemon().stop() is False
        claudexor_daemon._write_ownership_marker()
        monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "_classify_liveness", lambda *_a, **_kw: (object(), "running", ""))
        # This unit case supplies authenticated endpoint evidence; the HTTP fixture above
        # exercises the actual descriptor/token/handshake path before a real process stop.
        assert claudexor_daemon.OwnedClaudexorDaemon().stop() is True
        assert _wait_gone(proc.pid)
        assert [r["scope"] for r in _supervisor_rows(tmp_path, "process_stopped")] == ["session"]
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


def test_retained_daemon_pids_is_best_effort(monkeypatch):
    from supervisor import queue as q

    monkeypatch.setattr(process_custody, "live_daemon_root_pids", lambda _root, **_kw: (_ for _ in ()).throw(OSError("x")))
    assert q._retained_daemon_pids() == set()


def _function_source(module_path: pathlib.Path, name: str) -> str:
    """The exact source of one module-level function by AST (decorators such as the
    lifecycle serializer hide the body from ``inspect.getsource``)."""
    import ast

    text = module_path.read_text(encoding="utf-8")
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"{name} not found in {module_path.name}")


def test_every_supervisor_worker_tree_kill_is_the_shared_one():
    """The chokepoint: no supervisor path reaches ``kill_pid_tree`` except the one owner."""
    root = pathlib.Path(__file__).resolve().parent.parent
    direct = {
        path.name for path in (root / "supervisor").glob("*.py")
        if re.search(r"\bkill_pid_tree\(", path.read_text(encoding="utf-8"))
    }
    assert direct == {"worker_pool_lifecycle.py"}
    for module, name, keep in (
        ("workers", "kill_workers", False),
        ("worker_pool_lifecycle", "_kill_survivors", False),
        ("worker_pool_lifecycle", "kill_workers_for_update", False),
        ("worker_pool_lifecycle", "_replace_unready_slot", False),
        ("worker_pool_lifecycle", "respawn_worker", False),
        ("task_reaper", "_kill_and_confirm_worker_dead", True),
        ("task_lifecycle", "_finish_captured_running", True),
    ):
        source = _function_source(root / "supervisor" / f"{module}.py", name)
        assert "kill_worker_tree(" in source, name
        assert ("keep_services=True" in source) is keep, name


def test_retained_purpose_never_rescues_a_stale_identity(tmp_path, monkeypatch):
    row = {"pid": 123, "purpose": claudexor_daemon.CUSTODY_PURPOSE, "scope": "session",
           "session_id": "old-generation"}
    monkeypatch.setattr(process_custody, "_read_ledger", lambda _: [row])
    monkeypatch.setattr(process_custody, "_fingerprint_matches", lambda _: False)
    kept = []
    monkeypatch.setattr(process_custody, "_rewrite_ledger", lambda _, entries, **_kw: kept.extend(entries))
    assert process_custody.reap_orphaned_processes(
        tmp_path, retained_purposes={claudexor_daemon.CUSTODY_PURPOSE}) == []
    assert kept == []


@pytest.mark.parametrize("surface", ["startup", "periodic"])
def test_both_server_sweeps_preserve_legacy_daemon_records(tmp_path, monkeypatch, surface):
    from ouroboros import server_maintenance

    calls = []
    monkeypatch.setattr(server_maintenance, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_maintenance, "_installed_skill_names", lambda: None)
    monkeypatch.setattr(server_maintenance, "_reconcile_delegated_runs", lambda _: None)
    monkeypatch.setattr(server_maintenance, "_cursor_refresh_settled_terminals", lambda: None)
    monkeypatch.setattr(server_maintenance, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])
    monkeypatch.setattr(process_custody, "reap_orphaned_processes", lambda root, **kw: calls.append((root, kw)) or [])
    monkeypatch.setattr("ouroboros.delegate_terminal.backfill_terminal_reconciliations", lambda _: [])
    monkeypatch.setattr("supervisor.terminal_delivery.replay_pending_deliveries", lambda _: None)
    monkeypatch.setattr("ouroboros.delegate_state_sweep.sweep_settled_delegate_state", lambda _: {})
    monkeypatch.setattr("ouroboros.model_send_seal.reconcile_model_send_seals", lambda _: {})
    if surface == "startup":
        server_maintenance._startup_custody_sweep()
    else:
        server_maintenance._periodic_supervisor_maintenance([0], [time.time()])
    assert len(calls) == 1
    assert calls[0][0] == tmp_path
    assert calls[0][1]["retained_purposes"] == {claudexor_daemon.CUSTODY_PURPOSE}


@_POSIX
@pytest.mark.serial
@pytest.mark.parametrize("evidence", ["missing_marker", "malformed_marker", "foreign_token"])
def test_attached_stop_requires_marker_and_authenticated_endpoint(worker_with_daemon, evidence):
    worker, child = worker_with_daemon
    marker = claudexor_daemon.ownership_marker_path()
    if evidence == "missing_marker":
        marker.unlink()
    elif evidence == "malformed_marker":
        marker.write_text("{broken")
    else:
        token = claudexor_daemon.owned_descriptor_path().parent / "token"
        token.write_text("wrong-fixture-token")
    manager = claudexor_daemon.OwnedClaudexorDaemon()
    assert manager.stop() is False
    assert _echoes(child["port"])
    assert worker.poll() is None


@_POSIX
@pytest.mark.serial
def test_worker_teardown_retains_a_legacy_session_daemon(tmp_path, worker_with_daemon):
    from supervisor.worker_pool_lifecycle import kill_worker_tree
    worker, child = worker_with_daemon
    # Replay the historical daemon producer's scope on the same exact process.
    entries = process_custody._read_ledger(tmp_path)
    entries[0]["scope"] = "session"
    process_custody._rewrite_ledger(tmp_path, entries)
    kill_worker_tree(worker.pid)
    worker.wait(timeout=5)
    assert _echoes(child["port"])
    assert claudexor_daemon.OwnedClaudexorDaemon().stop() is True
    assert _wait_gone(child["pid"])

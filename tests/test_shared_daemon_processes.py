"""Shared-daemon custody: portable control flow and separate native Windows proof.

The engine/CLI below are loopback fixtures, never a provider-backed Claudexor.
They exercise the real host spawn, subprocess stop, Job and cleanup primitives.
"""
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_daemon as daemon
from ouroboros import platform_layer as pl
from ouroboros import process_custody as custody


def _fixture_python() -> str:
    """Use the real Windows interpreter, bypassing the venv PID redirector."""
    if os.name == "nt":
        return str(getattr(sys, "_base_executable", sys.executable))
    return sys.executable


def test_selective_windows_tree_uses_one_snapshot_without_taskkill_tree(monkeypatch):
    calls, snapshots = [], []
    # Worker 10 owns ordinary 11/12 and shared daemon 20 with client work 21/22.
    children = {10: [11, 20], 11: [12], 20: [21], 21: [22]}
    monkeypatch.setattr(pl, "IS_WINDOWS", True)
    monkeypatch.setattr(pl, "_process_children", lambda: snapshots.append(True) or children)
    monkeypatch.setattr(pl, "_hidden_run", lambda argv, **kw: calls.append(argv))
    pl.kill_pid_tree(10, exclude_pids={20})
    assert snapshots == [True]
    assert calls == [["taskkill", "/F", "/PID", str(pid)] for pid in (12, 11, 10)]
    calls.clear()
    pl.kill_pid_tree(20, exclude_pids={20})
    assert calls == []


def test_posix_forced_tree_preserves_shared_branch_even_in_same_group(monkeypatch):
    if pl.IS_WINDOWS:
        pytest.skip("POSIX group branch")
    killed, groups = [], []
    monkeypatch.setattr(pl, "_process_children", lambda: {10: [11, 20], 11: [12], 20: [21]})
    monkeypatch.setattr(pl, "process_group_id", lambda pid: 10)
    monkeypatch.setattr(pl, "kill_process_group_id", lambda pgid, **kw: groups.append(pgid))
    monkeypatch.setattr(pl, "force_kill_pid", killed.append)
    pl.kill_process_tree(SimpleNamespace(pid=10), exclude_pids={20})
    assert groups == []
    assert killed == [12, 11, 10]


@pytest.mark.parametrize("count", [1, 15])
@pytest.mark.parametrize("attached", [False, True])
def test_posix_tree_reads_one_snapshot_for_target_and_whole_retained_branch(monkeypatch, count, attached):
    monkeypatch.setattr(pl, "IS_WINDOWS", False)
    calls = []
    # A nested retained branch, either under the worker or owned by another worker.
    lines = ["10 1", "11 10", "12 11", f"20 {10 if attached else 1}"]
    lines.extend(f"{21 + index} {20 + index}" for index in range(count))
    def snapshot(argv, **kwargs):
        calls.append(argv)
        assert argv == ["ps", "-axo", "pid=,ppid="]
        assert kwargs["timeout"] == 3 and kwargs["check"] is True
        return SimpleNamespace(stdout="\n".join(lines), returncode=0)
    monkeypatch.setattr(pl.subprocess, "run", snapshot)
    targets, spared = pl._tree_kill_targets(10, {20})
    assert targets == [12, 11, 10]
    assert spared == set(range(20, 21 + count))
    assert len(calls) == 1, "retained subtree size must not add subprocesses per worker"


@pytest.mark.parametrize("failure", ["missing", "timeout", "failed", "empty", "malformed", "negative"])
def test_posix_snapshot_failure_retains_legacy_descendants_and_sparing(monkeypatch, failure):
    monkeypatch.setattr(pl, "IS_WINDOWS", False)
    calls = []
    children = {10: [11, 20], 11: [12], 20: [21]}
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "ps":
            if failure == "missing":
                raise FileNotFoundError("ps unavailable")
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, 3)
            if failure == "failed":
                raise subprocess.CalledProcessError(1, argv)
            return SimpleNamespace(stdout={"empty": "", "malformed": "10 1\nunreadable",
                                           "negative": "10 1\n-20 10"}[failure])
        assert argv[:2] == ["pgrep", "-P"]
        return SimpleNamespace(stdout="\n".join(map(str, children.get(int(argv[-1]), []))))
    monkeypatch.setattr(pl.subprocess, "run", run)
    targets, spared = pl._tree_kill_targets(10, {20})
    assert targets == [12, 11, 10]
    assert spared == {20, 21}
    assert calls[0] == ["ps", "-axo", "pid=,ppid="]
    assert any(call[0] == "pgrep" for call in calls)


def test_windows_legacy_argv_hash_is_not_native_measurement(monkeypatch):
    monkeypatch.setattr(custody, "IS_WINDOWS", True)
    monkeypatch.setattr(custody, "pid_is_alive", lambda pid: True)
    monkeypatch.setattr(custody, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(custody, "_live_cmd_sha256", lambda pid: pytest.fail("legacy hash is not measured"))
    row = {"pid": 123, "fingerprint": {"start_time": "", "cmd_sha256": "old argv spelling"}}
    assert custody._fingerprint_matches(row)
    assert not custody._fingerprint_matches(row, require_measured=True)


def test_stop_snapshot_does_not_signal_a_new_custody_row(tmp_path, monkeypatch):
    row = {"pid": 123, "purpose": daemon.CUSTODY_PURPOSE, "fingerprint": {"start_time": "old"}}
    custody.append_jsonl(custody.ledger_path(tmp_path), row)
    monkeypatch.setattr(custody, "_fingerprint_matches", lambda *a, **kw: True)
    original = custody.process_stop_snapshot(tmp_path, {daemon.CUSTODY_PURPOSE})
    successor = {**row, "fingerprint": {"start_time": "new"}}
    custody.append_jsonl(custody.ledger_path(tmp_path), successor)
    monkeypatch.setattr(pl, "kill_pid_tree", lambda *a, **kw: pytest.fail("signalled successor"))
    assert custody.stop_ledgered_processes(
        tmp_path, {daemon.CUSTODY_PURPOSE}, expected_entries=original,
    ) == []
    assert custody._read_ledger(tmp_path) == [successor]


@pytest.mark.parametrize("receipt,rc,accepted", [
    ({"ok": True, "stopped": True, "outcome": "exited"}, 0, True),
    ({"ok": True, "stopped": True, "outcome": "killed"}, 0, True),
    ({"ok": True}, 0, False),
    ({"ok": True, "stopped": True, "outcome": "still_alive"}, 0, False),
    ({"ok": True, "stopped": True, "outcome": "exited"}, 1, False),
    ("not-json", 0, False),
])
def test_operator_cli_requires_terminal_receipt_and_owned_environment(tmp_path, monkeypatch, receipt, rc, accepted):
    from ouroboros import claudexor_runtime, config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("CLAUDEXOR_CONFIG_DIR", "foreign")
    monkeypatch.setenv("CLAUDEXOR_DAEMON_SOCK", "foreign-socket")
    monkeypatch.setenv("CLAUDEXOR_CONTROL_PORT", "1234")
    def cli_command(*, require_npm):
        assert require_npm is False
        return ["verified-node", "verified-cli"]
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(
        resolve_cli_command=cli_command,
    ))
    def run(argv, **kw):
        assert argv == ["verified-node", "verified-cli", "daemon", "stop", "--json"]
        assert kw["env"]["CLAUDEXOR_CONFIG_DIR"] == str(tmp_path / "claudexor")
        assert "CLAUDEXOR_DAEMON_SOCK" not in kw["env"]
        assert "CLAUDEXOR_CONTROL_PORT" not in kw["env"]
        assert kw["timeout"] > 30
        return SimpleNamespace(returncode=rc, stdout=receipt if isinstance(receipt, str) else json.dumps(receipt))
    monkeypatch.setattr(daemon.subprocess, "run", run)
    assert daemon.OwnedClaudexorDaemon()._request_operator_stop() is accepted


def test_operator_stop_receipt_does_not_stop_or_claim_a_live_successor(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    daemon._write_ownership_marker()
    manager = daemon.OwnedClaudexorDaemon()
    endpoints = iter([(object(), "running", ""), (object(), "running", "")])
    monkeypatch.setattr(manager, "_classify_liveness", lambda **kw: next(endpoints))
    requests = []
    monkeypatch.setattr(manager, "_request_operator_stop", lambda: requests.append(True) or True)
    monkeypatch.setattr(custody, "stop_ledgered_processes", lambda *a, **kw: pytest.fail("chased successor"))
    assert manager.stop_outcome() == "unconfirmed"
    assert requests == [True]
    assert "endpoint remains" in manager._last_error


_ENGINE = r'''
print('fixture-engine: Python entered', flush=True)
import http.server, json, os, pathlib, socketserver, subprocess, sys, threading
from ouroboros.platform_layer import subprocess_new_group_kwargs
print('fixture-engine: imports ready', flush=True)

class LoopbackHTTPServer(http.server.HTTPServer):
    def server_bind(self):
        # The fixture has no hostname semantics; avoid external reverse DNS.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

home = pathlib.Path(os.environ['CLAUDEXOR_CONFIG_DIR'])
home.mkdir(parents=True, exist_ok=True)
progress = home / 'client-work.txt'
print('fixture-engine: starting harness', flush=True)
harness = subprocess.Popen([sys.executable, '-c',
    "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); n=0\nwhile True:\n n+=1; p.write_text(str(n)); time.sleep(.05)",
    str(progress)], **subprocess_new_group_kwargs())
print('fixture-engine: harness spawned', flush=True)
(home / 'fixture-engine.json').write_text(json.dumps({'pid': os.getpid(), 'harness_pid': harness.pid}))
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'client-B-keeps-working')
    def do_POST(self):
        print('fixture-engine: POST ' + self.path, flush=True)
        self.rfile.read(int(self.headers.get('Content-Length', '0')))
        if self.headers.get('Authorization') != 'Bearer fixture-token':
            self.send_response(401); self.end_headers(); return
        self.send_response(200); self.end_headers()
        if self.path == '/fixture-stop':
            harness.terminate(); harness.wait(timeout=5)
            self.wfile.write(b'{}')
            threading.Thread(target=server.shutdown, daemon=True).start()
        else:
            self.wfile.write(json.dumps({'compatible': True, 'protocolMajor': 3,
                'engine': {'version': '3.9.8', 'sha': 'a' * 40}}).encode())
        print('fixture-engine: POST complete ' + self.path, flush=True)
print('fixture-engine: binding loopback', flush=True)
server = LoopbackHTTPServer(('127.0.0.1', 0), Handler)
control = home / 'daemon'; control.mkdir(exist_ok=True)
(control / 'token').write_text('fixture-token')
(control / 'control-api.json').write_text(json.dumps({'host':'127.0.0.1',
    'port':server.server_port, 'tokenPath':str(control / 'token')}))
(home / 'fixture-engine.json').write_text(json.dumps({'pid':os.getpid(),
    'port':server.server_port, 'harness_pid':harness.pid}))
print('fixture-engine: control published', flush=True)
try: server.serve_forever(poll_interval=.05)
finally:
    server.server_close()
    if harness.poll() is None: harness.terminate(); harness.wait(timeout=5)
'''

_WORKER = r'''
import json, os, pathlib, subprocess, sys, time
from types import SimpleNamespace
root = pathlib.Path(sys.argv[1])
os.environ.update(OUROBOROS_DATA_DIR=str(root), OUROBOROS_SETTINGS_PATH=str(root/'settings.json'))
from ouroboros import claudexor_daemon as daemon, claudexor_runtime, platform_layer as pl
command = [sys.executable, '-u', '-c', sys.argv[2]]
claudexor_runtime.get_runtime_manager = lambda: SimpleNamespace(
    ensure=lambda: command, pin=None, status=lambda: {'version':'3.9.8','build_sha':'a'*40,'source':'fixture'})
probe = daemon.OwnedClaudexorDaemon._classify_liveness
def observe_probe(self, **kwargs):
    started = time.monotonic()
    print('fixture-worker: probe begin', file=sys.stderr, flush=True)
    result = probe(self, **kwargs)
    print('fixture-worker: probe %.3fs %s %s' % (time.monotonic() - started, result[1], result[2]), file=sys.stderr, flush=True)
    return result
daemon.OwnedClaudexorDaemon._classify_liveness = observe_probe
daemon.ensure_owned_gateway(startup_wait_sec=30).close()
owned = daemon.get_owned_daemon()
custody_pid = int(getattr(getattr(owned, '_proc', None), 'pid', 0) or 0)
ordinary = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], **pl.subprocess_new_group_kwargs())
info = json.loads((root/'claudexor'/'fixture-engine.json').read_text())
info.update(worker_pid=os.getpid(), ordinary_pid=ordinary.pid, custody_pid=custody_pid)
(root/'ready.json.tmp').write_text(json.dumps(info))
(root/'ready.json.tmp').replace(root/'ready.json')
time.sleep(120)
'''

_LAUNCHER = r'''
import subprocess, sys, time
from ouroboros import platform_layer as pl
kwargs = pl.subprocess_new_group_kwargs()
use_job = pl.IS_WINDOWS and sys.argv[4] != 'headless'
if use_job: kwargs.update(pl.subprocess_new_group_kwargs(breakaway_from_job=True, suspended=True))
worker = subprocess.Popen([sys.executable, '-u', '-c', sys.argv[2], sys.argv[1], sys.argv[3]], **kwargs)
if use_job:
    job = pl.create_kill_on_close_job(allow_breakaway=sys.argv[4] != 'legacy')
    if not job or not pl.assign_pid_to_job(job, worker.pid) or not pl.resume_process(worker.pid):
        worker.kill(); worker.wait(timeout=5)
        if job: pl.close_job(job)
        raise RuntimeError('fixture worker could not enter its launcher Job')
raise SystemExit(worker.wait(timeout=120))
'''

_CLI = r'''
import json, os, pathlib, sys, time, urllib.request
from ouroboros.platform_layer import pid_is_alive
from ouroboros.process_containment import pid_is_zombie
assert sys.argv[1:] == ['daemon','stop','--json']
assert not os.environ.get('CLAUDEXOR_DAEMON_SOCK')
home = pathlib.Path(os.environ['CLAUDEXOR_CONFIG_DIR'])
info = json.loads((home/'fixture-engine.json').read_text())
request = urllib.request.Request('http://127.0.0.1:%s/fixture-stop'%info['port'],
    data=b'{}', headers={'Authorization':'Bearer '+(home/'daemon'/'token').read_text()})
urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=5).close()
deadline = time.monotonic()+5
while pid_is_alive(info['pid']) and not pid_is_zombie(info['pid']):
    if time.monotonic() > deadline: raise SystemExit(1)
    time.sleep(.05)
print(json.dumps({'ok':True,'stopped':True,'outcome':'exited','detail':'fixture cooperative shutdown'}))
'''


def _wait(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    assert predicate(), "fixture did not reach the expected state"


def _gone(pid):
    return not pl.pid_is_alive(pid) or custody.pid_is_zombie(pid)


@pytest.mark.serial
def test_loopback_fixture_bind_does_not_require_reverse_dns(monkeypatch):
    import ast
    import http.server
    import socket
    import socketserver

    node = next(item for item in ast.parse(_ENGINE).body
                if isinstance(item, ast.ClassDef) and item.name == "LoopbackHTTPServer")
    namespace = {"http": http, "socketserver": socketserver}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "fixture-server", "exec"), namespace)
    monkeypatch.setattr(socket, "getfqdn", lambda *_a: pytest.fail("fixture performed reverse DNS"))
    with namespace["LoopbackHTTPServer"](("127.0.0.1", 0), http.server.BaseHTTPRequestHandler) as server:
        assert server.server_name == "127.0.0.1"
        assert server.server_port == server.server_address[1] > 0


@pytest.fixture
def shared_tree(tmp_path, monkeypatch, request):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    source = str(pathlib.Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([source, env.get("PYTHONPATH", "")])
    if os.name == "nt":
        site = pathlib.Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
        if site.is_dir():
            env["PYTHONPATH"] = os.pathsep.join([str(site), env["PYTHONPATH"]])
    # Bind the whole fixture process tree before the launcher starts. Setting
    # DATA_DIR only inside _WORKER leaves the launcher and its first imports
    # carrying a caller's real data-plane overrides.
    env.update(
        OUROBOROS_DATA_DIR=str(tmp_path),
        OUROBOROS_SETTINGS_PATH=str(tmp_path / "settings.json"),
    )
    for crossing in ("CLAUDEXOR_CONFIG_DIR", "CLAUDEXOR_DAEMON_SOCK", "CLAUDEXOR_CONTROL_PORT"):
        env.pop(crossing, None)
    log_path = tmp_path / "launcher.log"
    with log_path.open("wb") as log:
        parent = subprocess.Popen(
            [_fixture_python(), "-u", "-c", _LAUNCHER, str(tmp_path), _WORKER, _ENGINE,
             getattr(request, "param", "launcher")], env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log,
            **pl.subprocess_new_group_kwargs(),
        )
    info = {}
    try:
        try:
            _wait(lambda: (tmp_path / "ready.json").exists() or parent.poll() is not None, timeout=60)
        except AssertionError as exc:
            def read_log(path):
                try:
                    return path.read_text()
                except OSError:
                    return "<missing>"
            raise AssertionError(
                f"{exc}\nlauncher.log:\n{read_log(log_path)}\n"
                f"daemon.log:\n{read_log(tmp_path / 'claudexor' / 'daemon.log')}"
            ) from exc
        assert parent.poll() is None, (
            log_path.read_text() + "\ndaemon.log:\n"
            + ((tmp_path / "claudexor" / "daemon.log").read_text()
               if (tmp_path / "claudexor" / "daemon.log").is_file() else "<missing>")
        )
        info = json.loads((tmp_path / "ready.json").read_text())
        _wait(lambda: (tmp_path / "claudexor" / "client-work.txt").exists())
        yield parent, info
    finally:
        # Exact fixture PIDs only; on failed startup recover the fixture's own receipt.
        receipt = tmp_path / "claudexor" / "fixture-engine.json"
        if not info and receipt.exists():
            info = json.loads(receipt.read_text())
        if parent.poll() is None:
            pl.kill_process_tree(parent)
        parent.wait(timeout=5)
        if not info:
            custody.stop_ledgered_processes(tmp_path, {daemon.CUSTODY_PURPOSE})
        for name in ("ordinary_pid", "harness_pid", "worker_pid", "custody_pid", "pid"):
            if info.get(name) and not _gone(info[name]):
                pl.kill_pid_tree(info[name])
                _wait(lambda: _gone(info[name]))


def _continues(root, info):
    def diagnostics():
        logs = [str(info)]
        for path in (root / "launcher.log", root / "claudexor" / "daemon.log"):
            logs.append(f"{path.name}:\n" + (path.read_text() if path.is_file() else "<missing>"))
        return "\n".join(logs)
    assert not _gone(info["pid"]), "fixture daemon exited after ancestor cleanup:\n" + diagnostics()
    assert not _gone(info["harness_pid"]), "other fixture client exited after ancestor cleanup:\n" + diagnostics()
    def alive():
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"http://127.0.0.1:{info['port']}/alive", timeout=2) as response:
                return response.read() == b"client-B-keeps-working"
        except (OSError, urllib.error.URLError):
            return False
    try:
        _wait(alive, timeout=10)
    except AssertionError as exc:
        raise AssertionError("fixture daemon has no HTTP response:\n" + diagnostics()) from exc
    progress = root / "claudexor" / "client-work.txt"
    previous = progress.read_text()
    _wait(lambda: progress.read_text() not in ("", previous))


def _fixture_cli(root, monkeypatch):
    from ouroboros import claudexor_runtime
    path = root / "operator-cli.py"
    source = str(pathlib.Path(__file__).resolve().parents[1])
    # Like the worker fixture, this separate Python process needs the tested checkout.
    path.write_text(f"import sys\nsys.path.insert(0, {source!r})\n" + _CLI, encoding="utf-8")
    def cli_command(*, require_npm):
        assert require_npm is False
        return [sys.executable, str(path)]
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(
        resolve_cli_command=cli_command,
    ))


@pytest.mark.serial
def test_real_ancestor_cleanup_spares_daemon_and_other_client(tmp_path, shared_tree):
    parent, info = shared_tree
    roots = custody.live_daemon_root_pids(tmp_path)
    assert roots == {info["custody_pid"]}
    pl.kill_process_tree(parent, exclude_pids=roots)
    parent.wait(timeout=5)
    _wait(lambda: _gone(info["worker_pid"]) and _gone(info["ordinary_pid"]))
    _continues(tmp_path, info)


@pytest.mark.serial
def test_real_cli_process_stops_legacy_custody_without_signalling(tmp_path, monkeypatch, shared_tree):
    _, info = shared_tree
    _fixture_cli(tmp_path, monkeypatch)
    rows = custody._read_ledger(tmp_path)
    rows[0]["fingerprint"].pop("start_time_boot", None)
    rows[0]["fingerprint"]["start_time"] = ""
    custody._rewrite_ledger(tmp_path, rows)
    assert not custody._fingerprint_matches(rows[0], require_measured=True)
    monkeypatch.setattr(custody, "stop_ledgered_processes", lambda *a, **kw: pytest.fail("forced fallback"))
    manager = daemon.OwnedClaudexorDaemon()
    assert manager._proc is None
    assert manager.stop_outcome() == "stopped"
    assert _gone(info["pid"]) and _gone(info["harness_pid"])
    assert custody._read_ledger(tmp_path)[0]["fingerprint"]["start_time"] == ""


@pytest.mark.serial
def test_prior_session_custody_tree_reap_preserves_shared_daemon(tmp_path, shared_tree):
    _, info = shared_tree
    worker = custody.record_process(
        tmp_path, pid=info["worker_pid"], cmd="worker", purpose="worker", scope="session",
        reap_process_group=False,
    )
    worker["session_id"] = "previous-server"
    custody.append_jsonl(custody.ledger_path(tmp_path), worker)
    assert custody.reap_orphaned_processes(
        tmp_path, retained_purposes={daemon.CUSTODY_PURPOSE},
    ) == [info["worker_pid"]]
    _wait(lambda: _gone(info["worker_pid"]) and _gone(info["ordinary_pid"]))
    _continues(tmp_path, info)


def test_launcher_forced_stop_passes_retained_subtree(monkeypatch):
    import launcher
    calls = []
    class Process:
        pid = 9990011
        waits = 0
        def terminate(self):
            calls.append("graceful")
        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("server", timeout)
    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "_agent_proc", Process())
    monkeypatch.setattr(launcher, "_agent_job", None)
    monkeypatch.setattr(launcher, "_retained_shared_daemon_pids", lambda: {9990022})
    monkeypatch.setattr(launcher, "kill_process_tree", lambda proc, **kw: calls.append((proc.pid, kw)))
    monkeypatch.setattr(launcher, "_cleanup_recorded_server_group_for_pid", lambda *a: None)
    monkeypatch.setattr(launcher, "_kill_stale_on_port", lambda *a: None)
    launcher.stop_agent()
    assert calls == ["graceful", (9990011, {"exclude_pids": {9990022}})]


def test_recorded_server_cleanup_preserves_daemon_exclusions(tmp_path, monkeypatch):
    import launcher
    path = tmp_path / "server_process.json"
    path.write_text(json.dumps({"pid": 9990011, "pgid": 0}))
    calls = []
    monkeypatch.setattr(launcher, "_server_process_record_path", lambda: path)
    monkeypatch.setattr(launcher, "_server_process_identity_matches", lambda row: True)
    monkeypatch.setattr(launcher, "_retained_shared_daemon_pids", lambda: {9990022})
    monkeypatch.setattr(launcher, "pid_is_alive", lambda pid: True)
    monkeypatch.setattr(launcher, "kill_pid_tree", lambda pid, **kw: calls.append((pid, kw)))
    launcher._cleanup_recorded_server_process()
    assert calls == [(9990011, {"exclude_pids": {9990022}})]
    assert not path.exists()


def test_stray_reaper_filters_descendants_before_final_root_revalidation(monkeypatch):
    from ouroboros import launcher_server_reaper as reaper
    events = []
    monkeypatch.setattr(pl, "IS_WINDOWS", False)
    def snapshot():
        events.append("snapshot")
        return {10: [11, 20], 11: [12], 20: [21]}
    monkeypatch.setattr(pl, "_process_children", snapshot)
    monkeypatch.setattr(reaper, "_runs_our_server", lambda *a: events.append("command") or True)
    monkeypatch.setattr(reaper, "_is_launcher_managed", lambda *a: events.append("environment") or True)
    monkeypatch.setattr(reaper, "_signal_pid", lambda pid: events.append(("signal", pid)))
    monkeypatch.setattr(reaper, "_pid_gone", lambda pid: True)
    assert reaper._revalidate_and_kill(10, set(), set(), {20})
    assert events == ["snapshot", "command", "environment",
                      ("signal", 10), ("signal", 12), ("signal", 11)]


@pytest.mark.serial
@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Job inheritance")
def test_windows_forced_launcher_exit_preserves_previous_generation_daemon_then_panic_stops_it(
    tmp_path, monkeypatch, shared_tree,
):
    from tests.test_server_control_panic_daemon import _run_panic
    parent, info = shared_tree
    assert pl.process_start_time(info["pid"]).startswith("win-filetime:")
    assert pl.process_command(info["pid"])
    assert custody._fingerprint_matches(custody._read_ledger(tmp_path)[0], require_measured=True)
    # Actual death of the Job handle owner; no manual Job-close simulation.
    parent.kill()
    parent.wait(timeout=5)
    _wait(lambda: _gone(info["worker_pid"]) and _gone(info["ordinary_pid"]))
    _continues(tmp_path, info)
    _fixture_cli(tmp_path, monkeypatch)
    manager = daemon.OwnedClaudexorDaemon()
    assert manager._proc is None
    outcomes = []
    def stop():
        result = manager.stop_outcome()
        outcomes.append(result)
        return result == "stopped"
    with monkeypatch.context() as panic_patch:
        _run_panic(panic_patch, tmp_path, daemon_stop=stop)
    assert outcomes == ["stopped"]
    assert _gone(info["pid"]) and _gone(info["harness_pid"])


@pytest.mark.serial
@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows process identity")
def test_windows_birth_and_full_command_round_trip():
    argv = [sys.executable, "-c", "import time;time.sleep(30)", 'путь с пробелом', 'quoted"value', "tail\\"]
    proc = subprocess.Popen(argv, **pl.subprocess_new_group_kwargs())
    try:
        assert pl.process_start_time(proc.pid) == pl.process_start_time_legacy(proc.pid)
        assert pl.process_start_time(proc.pid).startswith("win-filetime:")
        assert pl.process_command(proc.pid) == subprocess.list2cmdline(argv)
    finally:
        proc.kill()
        proc.wait(timeout=5)


@pytest.mark.serial
@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows Job capability")
@pytest.mark.parametrize("shared_tree", ["legacy", "headless"], indirect=True)
def test_windows_first_use_keeps_working_with_legacy_job_or_without_launcher_job(
    tmp_path, shared_tree, request,
):
    parent, info = shared_tree
    _continues(tmp_path, info)
    if request.node.callspec.params["shared_tree"] == "legacy":
        # The old immutable launcher still starts a usable daemon, but cannot
        # provide the new survive-close guarantee until its package is updated.
        parent.kill()
        parent.wait(timeout=5)
        _wait(lambda: _gone(info["pid"]) and _gone(info["harness_pid"]))
    else:
        pl.kill_process_tree(parent, exclude_pids={info["custody_pid"]})
        parent.wait(timeout=5)
        _wait(lambda: _gone(info["worker_pid"]))
        _continues(tmp_path, info)

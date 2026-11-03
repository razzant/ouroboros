"""Real startup custody across Python consumers, with event-controlled readiness.

The fake engine owns an exclusive writer election and a real authenticated socket.
It never reads accounts, calls a provider, or consumes an installed engine runtime.
"""

import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_daemon as owned, claudexor_runtime, config, process_custody
from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable
from ouroboros.platform_layer import kill_process_tree, pid_is_alive, subprocess_new_group_kwargs

pytestmark = [pytest.mark.serial, pytest.mark.skipif(os.name == "nt", reason="POSIX measured process custody")]

_ENGINE = '''
import http.server, json, os, pathlib, socketserver, subprocess, sys, time
root = pathlib.Path(os.environ["CLAUDEXOR_CONFIG_DIR"])
with (root / "spawned.jsonl").open("a") as out:
    out.write(json.dumps({"pid": os.getpid()}) + "\\n")
try:
    (root / "writer").mkdir()
except FileExistsError:
    sys.exit(1)
def wait(name):
    deadline = time.monotonic() + 60
    while not (root / name).exists():
        if time.monotonic() >= deadline: sys.exit(2)
        time.sleep(.01)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
(root / "elected.json").write_text(json.dumps({"pid": os.getpid(), "child": child.pid}))
wait("publish")
if (root / "crash").exists():
    child.terminate(); child.wait(timeout=5)
    print("current fixture startup exited", flush=True)
    sys.exit(7)
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args): pass
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.headers.get("Authorization") != "Bearer fixture-startup-token":
            self.send_response(401); self.end_headers(); return
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({"compatible": True, "protocolMajor": 3,
            "engine": {"version": "9.9.9", "sha": "c" * 40},
            "servingMode": "normal" if (root / "normal").exists() else "recovery_only"}).encode())
# HTTPServer.server_bind resolves reverse DNS before listen; this numeric loopback fixture needs none.
server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
descriptor = root / "daemon" / "control-api.json"
descriptor.parent.mkdir(parents=True, exist_ok=True)
token = descriptor.parent / "token"
token.write_text("fixture-startup-token")
descriptor.write_text(json.dumps({"host": "127.0.0.1", "port": server.server_address[1],
    "tokenPath": str(token)}))
server.serve_forever()
'''

_CONSUMER = '''
import json, pathlib, sys, time
from types import SimpleNamespace
from ouroboros import claudexor_daemon as owned, claudexor_runtime
from ouroboros.gateways.claudexor import ClaudexorUnavailable
root, name, engine = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def wait(path):
    deadline = time.monotonic() + 30
    while not path.exists():
        if time.monotonic() >= deadline: raise RuntimeError("fixture barrier expired")
        time.sleep(.01)
def prepare():
    (root / (name + ".prepared")).touch()
    wait(root / (name + ".release"))
    return [sys.executable, engine]
claudexor_runtime.get_runtime_manager = lambda: SimpleNamespace(
    ensure=prepare, status=lambda: {"source": "fixture", "version": "9.9.9", "build_sha": "c" * 40})
manager = owned.OwnedClaudexorDaemon()
try:
    endpoint = manager.ensure_running(startup_wait_sec=.2)
    result = {"code": "ready", "port": endpoint.port}
except ClaudexorUnavailable as exc:
    result = {"code": exc.code, "detail": str(exc)}
(root / (name + ".result.json")).write_text(json.dumps(result))
wait(root / "finish")
if manager._proc is not None:
    manager._proc.wait(timeout=5)
'''


def _wait_for(read, *, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = read()
            if result:
                return result
        except (OSError, ValueError):
            pass
        time.sleep(.01)
    raise AssertionError("fixture event did not arrive")


def _read_json(path):
    return json.loads(path.read_text())


def _gone(pid):
    return not pid_is_alive(pid) or process_custody.pid_is_zombie(pid)


@pytest.fixture
def startup(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))  # parent and consumers share one install
    home = root / "claudexor"
    home.mkdir(parents=True)
    engine = tmp_path / "fixture_engine.py"
    engine.write_text(_ENGINE)
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(
        ensure=lambda: [sys.executable, str(engine)],
        status=lambda: {"source": "fixture", "version": "9.9.9", "build_sha": "c" * 40},
    ))
    consumers = []

    def consumer(name):
        env = dict(os.environ)
        env.update(OUROBOROS_APP_ROOT=str(tmp_path), OUROBOROS_DATA_DIR=str(root),
                   OUROBOROS_SETTINGS_PATH=str(root / "settings.json"),
                   OUROBOROS_REPO_DIR=str(pathlib.Path(__file__).resolve().parents[1]))
        process = subprocess.Popen(
            [sys.executable, "-c", _CONSUMER, str(home), name, str(engine)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            **subprocess_new_group_kwargs(),
        )
        consumers.append(process)
        return process

    fixture = SimpleNamespace(root=root, home=home, engine=engine, consumer=consumer)
    try:
        yield fixture
    finally:
        # Exact private fixture custody, never name/port sweeps or a live daemon.
        process_custody.stop_ledgered_processes(root, {owned.CUSTODY_PURPOSE})
        (home / "finish").touch()
        for process in consumers:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                kill_process_tree(process)
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()


def test_wait_expiry_preserves_startup_and_a_new_manager_joins_it(startup, monkeypatch):
    manager = owned.OwnedClaudexorDaemon()
    with pytest.raises(ClaudexorUnavailable) as pending:
        manager.ensure_running(startup_wait_sec=.1)
    assert pending.value.code == "daemon_starting"
    assert pending.value.status_code == 503
    elected = _wait_for(lambda: _read_json(startup.home / "elected.json"))
    assert manager._proc.pid == elected["pid"] and manager._proc.poll() is None
    peer = owned.OwnedClaudexorDaemon()
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(
        pin=SimpleNamespace(version="9.9.9", build_sha="c" * 40),
        ensure=lambda: pytest.fail("a joined startup must not provision or spawn again")))
    with pytest.raises(ClaudexorUnavailable, match="retry joins"):
        peer.ensure_running(startup_wait_sec=.05)
    (startup.home / "normal").touch()
    (startup.home / "publish").touch()
    endpoint = peer.ensure_running(startup_wait_sec=3)
    with ClaudexorGateway(endpoint) as gateway:
        assert gateway.handshake()["servingMode"] == "normal"
    assert len((startup.home / "spawned.jsonl").read_text().splitlines()) == 1
    assert peer.stop() is True
    manager._proc.wait(timeout=5)
    assert _gone(elected["child"])


def test_two_python_consumers_join_one_elected_startup_after_prepare_barrier(startup):
    first, second = startup.consumer("first"), startup.consumer("second")
    for name in ("first", "second"):
        _wait_for(lambda name=name: (startup.home / (name + ".prepared")).exists())
    for name in ("first", "second"):
        (startup.home / (name + ".release")).touch()
    results = [_wait_for(lambda name=name: _read_json(startup.home / (name + ".result.json")))
               for name in ("first", "second")]
    assert [row["code"] for row in results] == ["daemon_starting", "daemon_starting"]
    elected = _wait_for(lambda: _read_json(startup.home / "elected.json"))
    # A caller's readiness wait can expire before the election loser exits.
    contenders = process_custody.live_daemon_root_pids(startup.root, purposes={owned.CUSTODY_PURPOSE}, strict=True)
    _wait_for(lambda: all(_gone(pid) for pid in contenders - {elected["pid"]}))
    live = process_custody.live_daemon_root_pids(startup.root, purposes={owned.CUSTODY_PURPOSE}, strict=True)
    assert live == {elected["pid"]}, "the losing physical contender must not count as an owner"
    assert first.poll() is None and second.poll() is None
    count = len((startup.home / "spawned.jsonl").read_text().splitlines())
    for _ in range(2):
        with pytest.raises(ClaudexorUnavailable) as pending:
            owned.OwnedClaudexorDaemon().ensure_running(startup_wait_sec=.03)
        assert pending.value.code == "daemon_starting"
    assert len((startup.home / "spawned.jsonl").read_text().splitlines()) == count
    (startup.home / "normal").touch()
    (startup.home / "publish").touch()
    assert owned.OwnedClaudexorDaemon().ensure_running(startup_wait_sec=3).port > 0
    assert owned.OwnedClaudexorDaemon().stop() is True
    assert _gone(elected["child"])


def test_delayed_runtime_preparation_rechecks_the_ready_peer_before_spawn(startup):
    delayed = startup.consumer("delayed")
    _wait_for(lambda: (startup.home / "delayed.prepared").exists())
    (startup.home / "normal").touch()
    (startup.home / "publish").touch()
    winner = owned.OwnedClaudexorDaemon()
    endpoint = winner.ensure_running(startup_wait_sec=3)
    (startup.home / "delayed.release").touch()
    result = _wait_for(lambda: _read_json(startup.home / "delayed.result.json"))
    assert result == {"code": "ready", "port": endpoint.port}
    assert len((startup.home / "spawned.jsonl").read_text().splitlines()) == 1
    assert delayed.poll() is None
    assert winner.stop() is True


@pytest.mark.parametrize("descriptor", [False, True])
@pytest.mark.parametrize("action", ["stop", "panic"])
def test_peer_stop_works_before_readiness_and_reaps_the_owned_child(startup, monkeypatch, descriptor, action):
    manager = owned.OwnedClaudexorDaemon()
    with pytest.raises(ClaudexorUnavailable):
        manager.ensure_running(startup_wait_sec=.1)
    elected = _wait_for(lambda: _read_json(startup.home / "elected.json"))
    if descriptor:
        (startup.home / "publish").touch()
        _wait_for(lambda: owned.owned_descriptor_path().is_file())
        monkeypatch.setattr(owned, "get_owned_daemon", lambda: manager)
        with pytest.raises(ClaudexorUnavailable) as recovery:
            owned.ensure_owned_gateway(admission_wait_sec=0)
        assert recovery.value.code == "daemon_recovery_only"
    started = time.monotonic()
    stop = owned.OwnedClaudexorDaemon().stop
    if action == "panic":
        from tests.test_server_control_panic_daemon import _run_panic
        with monkeypatch.context() as patch:
            # Panic no longer invokes the legacy cooperative worker callback;
            # the real owned daemon + its child must be gone below.
            _run_panic(patch, startup.root, daemon_stop=stop)
    else:
        assert stop() is True
    assert time.monotonic() - started < 5
    manager._proc.wait(timeout=5)
    assert _gone(elected["pid"]) and _gone(elected["child"])


def test_stop_retires_an_ensure_waiting_in_runtime_preparation(startup, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def prepare():
        entered.set()
        assert release.wait(5)
        return [sys.executable, str(startup.engine)]
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(ensure=prepare))
    manager = owned.OwnedClaudexorDaemon()
    result = []
    def ensure():
        try:
            manager.ensure_running(startup_wait_sec=.1)
        except ClaudexorUnavailable as exc:
            result.append(exc.code)
    thread = threading.Thread(target=ensure)
    thread.start()
    try:
        assert entered.wait(5)
        assert manager.stop() is False
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert result == ["daemon_start_cancelled"]
    assert not (startup.home / "spawned.jsonl").exists()


def test_stop_does_not_wait_for_the_same_managers_control_window(startup):
    manager = owned.OwnedClaudexorDaemon()
    result = []
    def ensure():
        try:
            manager.ensure_running(startup_wait_sec=20)
        except ClaudexorUnavailable as exc:
            result.append(exc.code)
    thread = threading.Thread(target=ensure)
    thread.start()
    try:
        elected = _wait_for(lambda: _read_json(startup.home / "elected.json"))
        started = time.monotonic()
        assert manager.stop() is True
        assert time.monotonic() - started < 5
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert result == ["daemon_start_cancelled"]
    assert manager._proc is None
    assert _gone(elected["child"])


@pytest.mark.parametrize("damage", ["corrupt", "unreadable", "stat_unreadable", "invalid_utf8", "dangling"])
def test_unknown_custody_refuses_a_new_spawn(startup, monkeypatch, damage):
    ledger = process_custody.ledger_path(startup.root)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if damage == "dangling":
        ledger.symlink_to(ledger.parent / "missing")
    elif damage == "corrupt":
        ledger.write_bytes(b"{broken\n")
    elif damage == "invalid_utf8":
        ledger.write_bytes(b'{"pid":1,"purpose":"\xff"}\n')
    else:
        ledger.write_bytes(b'{"pid":1}\n')
    if damage in {"unreadable", "stat_unreadable"}:
        operation = "read_bytes" if damage == "unreadable" else "stat"
        original = getattr(pathlib.Path, operation)
        def read(path, *args, **kwargs):
            if path == ledger:
                raise PermissionError("fixture ledger read refused")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(pathlib.Path, operation, read)
    with pytest.raises(ClaudexorUnavailable) as refused:
        owned.OwnedClaudexorDaemon().ensure_running(startup_wait_sec=0)
    assert refused.value.code == "daemon_startup_unknown"
    assert not (startup.home / "spawned.jsonl").exists()


def test_other_daemon_purposes_are_not_joined_as_claudexor(startup):
    stranger = process_custody.spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(60)"], drive_root=startup.root,
        purpose="fixture_companion", scope="daemon")
    try:
        assert process_custody.live_daemon_root_pids(startup.root) == {stranger.pid}
        manager = owned.OwnedClaudexorDaemon()
        with pytest.raises(ClaudexorUnavailable) as pending:
            manager.ensure_running(startup_wait_sec=.1)
        assert pending.value.code == "daemon_starting"
        elected = _wait_for(lambda: _read_json(startup.home / "elected.json"))
        assert manager._proc.pid == elected["pid"] != stranger.pid
        assert owned.OwnedClaudexorDaemon().stop() is True
        manager._proc.wait(timeout=5)
        assert stranger.poll() is None
    finally:
        kill_process_tree(stranger)
        stranger.wait(timeout=5)


@pytest.mark.parametrize("damage", ["marker", "foreign_marker", "descriptor", "token"])
def test_peer_stop_never_promotes_invalid_or_foreign_discovery_to_authority(startup, damage):
    manager = owned.OwnedClaudexorDaemon()
    with pytest.raises(ClaudexorUnavailable):
        manager.ensure_running(startup_wait_sec=.1)
    elected = _wait_for(lambda: _read_json(startup.home / "elected.json"))
    marker = owned.ownership_marker_path()
    if damage == "marker":
        marker.write_text("{broken")
    elif damage == "foreign_marker":
        marker.write_text(json.dumps({"owner": "ouroboros", "data_dir": str(startup.root / "foreign")}))
    elif damage == "descriptor":
        descriptor = owned.owned_descriptor_path()
        descriptor.parent.mkdir(parents=True, exist_ok=True)
        descriptor.write_text("{broken")
    else:
        (startup.home / "publish").touch()
        _wait_for(lambda: owned.owned_descriptor_path().is_file())
        (owned.owned_descriptor_path().parent / "token").write_text("wrong-fixture-token")
    assert owned.OwnedClaudexorDaemon().stop() is False
    assert manager._proc.poll() is None and not _gone(elected["child"])
    # Direct handle custody still permits this test's own explicit cleanup.
    assert manager._terminate_child() is True


def test_crashed_startup_reports_current_pid_build_and_log_interval(startup):
    old = b"old runtime 3.8.2 failed for an unrelated reason\n"
    (startup.home / "daemon.log").write_bytes(old)
    (startup.home / "crash").touch()
    (startup.home / "publish").touch()
    manager = owned.OwnedClaudexorDaemon()
    with pytest.raises(ClaudexorUnavailable) as failed:
        manager.ensure_running(startup_wait_sec=.5)
    elected = _read_json(startup.home / "elected.json")
    assert failed.value.code == "daemon_spawn_failed"
    text = str(failed.value)
    assert f"spawn_pid={elected['pid']}" in text and "poll=7" in text
    assert "version=9.9.9" in text and "build_sha=" + "c" * 40 in text
    assert f"startup log interval={len(old)}.." in text
    assert "old runtime" not in text and str(startup.home / "daemon.log") in text
    assert manager._proc is None
    assert manager._startup_attempt == {}
    assert _gone(elected["pid"])
    if sys.platform == "linux":
        assert not pathlib.Path(f"/proc/{elected['pid']}").exists(), "Popen.poll reaped the exited child"


def test_stopped_manager_joins_peer_without_reusing_its_old_diagnostics(startup):
    first, peer = owned.OwnedClaudexorDaemon(), owned.OwnedClaudexorDaemon()
    processes = []
    try:
        with pytest.raises(ClaudexorUnavailable, match="daemon is still starting"):
            first.ensure_running(startup_wait_sec=.1)
        old = first._proc
        processes.append(old)
        _wait_for(lambda: _read_json(startup.home / "elected.json"))
        assert first.stop() is True and old.poll() is not None
        assert first._startup_attempt == {}
        # The synthetic engine's election marker has no restart cleanup.
        (startup.home / "writer").rmdir()
        (startup.home / "elected.json").unlink()
        with pytest.raises(ClaudexorUnavailable, match="daemon is still starting"):
            peer.ensure_running(startup_wait_sec=.1)
        current = peer._proc
        processes.append(current)
        _wait_for(lambda: _read_json(startup.home / "elected.json"))
        with pytest.raises(ClaudexorUnavailable) as joined:
            first.ensure_running(startup_wait_sec=.03)
        detail = str(joined.value)
        assert f"live_pids=[{current.pid}]" in detail
        assert "joining another manager" in detail
        assert "spawn_pid=" not in detail and "selected_build_sha=" not in detail
        assert "startup log interval=" not in detail
    finally:
        first.stop()
        peer.stop()
        for process in processes:
            process.wait(timeout=5)

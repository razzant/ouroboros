"""The launcher's custody of one server generation: identity, stop, readiness and intent.

Real sentinel processes stand in for a previous server: only a re-proved birth/command
identity is ever signalled, a wrong birth leaves the sentinel alive, an unmeasurable or
unconfirmed one keeps its record and blocks the next generation, a reaped root's former
group is reported but never signalled, and an unconfirmed ordinary stop keeps the captured handle. Readiness is the endpoint the CHILD published:
a stale binding holding the same PID is refused, the plain port file counts only after
its confirmed removal, and a dead child ends the wait. The parsed launch intent governs
the child over an inherited one; the relaunch argv drops ``--runtime-venv`` and the real
historical v7.6.0 parser accepts it.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import threading
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups and birth tokens")


@pytest.fixture
def sentinel():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    time.sleep(0.3)  # ps reports a stable command line once the interpreter started
    yield proc
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)


@pytest.fixture
def launcher_root(monkeypatch, tmp_path):
    import launcher

    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    monkeypatch.setattr(launcher, "DATA_DIR", data)
    monkeypatch.setattr(launcher, "PORT_FILE", data / "state" / "server_port")
    monkeypatch.setattr(launcher, "_retained_shared_daemon_pids", lambda: set())
    return launcher, data


def _record(launcher, data, proc, **fingerprint_changes):
    from ouroboros.process_custody import process_fingerprint

    fingerprint = {**process_fingerprint(proc.pid, proc.args), **fingerprint_changes}
    record = {"pid": proc.pid, "pgid": proc.pid, "server_path": "/elsewhere/server.py", "fingerprint": fingerprint}
    path = data / "state" / "server_process.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


@pytest.mark.serial
def test_a_wrong_birth_is_another_process_and_is_never_signalled(launcher_root, sentinel):
    launcher, data = launcher_root
    path = _record(launcher, data, sentinel, start_time="Thu Jan  1 00:00:00 1970", start_time_boot="1.0")
    assert launcher._recorded_server_identity(json.loads(path.read_text())) == "foreign"
    assert launcher._cleanup_recorded_server_process("test") == []
    time.sleep(0.3)
    assert sentinel.poll() is None, "a reused PID must never be signalled"
    assert not path.exists(), "a refuted record retires"


@pytest.mark.serial
def test_an_unmeasurable_identity_keeps_the_record_and_blocks(launcher_root, sentinel, monkeypatch):
    from ouroboros import process_custody

    launcher, data = launcher_root
    path = _record(launcher, data, sentinel)
    monkeypatch.setattr(process_custody, "process_start_time", lambda _pid: "")
    assert launcher._recorded_server_identity(json.loads(path.read_text())) == "unknown"
    assert launcher._cleanup_recorded_server_process("test") == [sentinel.pid]
    assert sentinel.poll() is None and path.exists()


@pytest.mark.serial
def test_a_proven_predecessor_is_stopped_and_its_record_retired(launcher_root, sentinel):
    launcher, data = launcher_root
    path = _record(launcher, data, sentinel)
    assert launcher._recorded_server_identity(json.loads(path.read_text())) == "ours"
    assert launcher._cleanup_recorded_server_process("test") == []
    sentinel.wait(timeout=10)
    assert not path.exists()


@pytest.mark.serial
def test_an_unconfirmed_stop_keeps_the_record_and_the_next_generation_waits(launcher_root, sentinel, monkeypatch):
    launcher, data = launcher_root
    path = _record(launcher, data, sentinel)
    for name in ("terminate_process_group_id", "kill_process_group_id", "kill_pid_tree", "force_kill_pid"):
        monkeypatch.setattr(launcher, name, lambda *a, **k: None)  # a server that does not die
    confirm = launcher._confirm_gone
    monkeypatch.setattr(launcher, "_confirm_gone", lambda pid, timeout=5.0: confirm(pid, 0.3))
    monkeypatch.setattr(launcher, "_reap_same_install_strays", lambda reason: [])
    assert launcher._pre_generation_cleanup() == [sentinel.pid]
    assert path.exists() and sentinel.poll() is None


@pytest.mark.serial
def test_a_reaped_roots_former_group_is_reported_and_never_signalled(launcher_root):
    """Both callers reap the child first; after that its PID and group id are names a later
    process may hold, so a group that still has members proves nothing and is not signalled."""
    from ouroboros.platform_layer import force_kill_pid, pid_is_alive, process_group_id

    launcher, data = launcher_root
    root = subprocess.Popen([sys.executable, "-c", "import subprocess, sys; print(subprocess.Popen([sys.executable, "
                             "'-c', 'import time; time.sleep(120)']).pid, flush=True)"],
                            start_new_session=True, stdout=subprocess.PIPE, text=True)
    member = int(root.stdout.readline())
    try:
        root.wait(timeout=10)  # reaped, as stop_agent and the lifecycle loop do before the call
        assert process_group_id(member) == root.pid, "the member still names the reaped root's group"
        path = data / "state" / "server_process.json"
        path.write_text(json.dumps({"pid": root.pid, "pgid": root.pid}), encoding="utf-8")
        assert launcher._cleanup_recorded_server_group_for_pid(root.pid, "test") is True
        time.sleep(0.5)
        assert pid_is_alive(member), "a reaped root's group id is no authority to signal"
        assert not path.exists(), "the reaped root's record retires"
    finally:
        force_kill_pid(member)


def test_an_unconfirmed_ordinary_stop_keeps_the_captured_handle(launcher_root, monkeypatch):
    launcher, data = launcher_root

    class Hung:
        pid = 987654
        stdin = None

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("server", timeout)

        def poll(self):
            return None

    proc = Hung()
    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "_agent_proc", proc)
    monkeypatch.setattr(launcher, "_agent_job", None)
    monkeypatch.setattr(launcher, "kill_process_tree", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_cleanup_recorded_server_group_for_pid",
                        lambda *a: pytest.fail("an unconfirmed stop retires nothing"))
    assert launcher.stop_agent() is False
    assert launcher._agent_proc is proc


def _child(seconds="120"):
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"], start_new_session=True)


@pytest.mark.serial
def test_readiness_is_the_endpoint_the_child_published(launcher_root):
    from ouroboros.server_process import record_service_binding

    launcher, data = launcher_root
    child = _child()
    try:
        time.sleep(0.3)
        bindings = data / "state" / "server_port.bindings.json"
        stale = {"pid": child.pid, "host": "127.0.0.1", "port": 41001,
                 "fingerprint": {"source": "process_start_time", "start_time": "Thu Jan  1 00:00:00 1970",
                                 "cmd_sha256": "0" * 64}}
        bindings.write_text(json.dumps({"main": stale}), encoding="utf-8")
        def still_waiting(**kwargs):
            # The wait has no clock of its own: a live child that published nothing keeps it
            # waiting; the test's own abort (a timer) ends it, never the launcher.
            abort = threading.Event()
            threading.Timer(1.0, abort.set).start()
            return launcher._await_generation_endpoint(child, abort_event=abort, **kwargs) is None

        assert still_waiting(port_file_cleared=False), "a stale binding holding the same PID is no readiness"
        (data / "state" / "server_port").write_text("41002", encoding="utf-8")
        assert still_waiting(port_file_cleared=False), \
            "a port file whose removal was not confirmed may predate this generation"
        assert launcher._await_generation_endpoint(child, port_file_cleared=True) == 41002
        bindings.unlink()
        record_service_binding(data, "main", "127.0.0.1", 41003, pid=child.pid)
        assert launcher._await_generation_endpoint(child, port_file_cleared=False) == 41003
    finally:
        child.kill()
        child.wait(timeout=10)
    started = time.monotonic()
    assert launcher._await_generation_endpoint(child, port_file_cleared=True) is None
    assert time.monotonic() - started < 2.0, "a dead child ends the wait at once"
    assert launcher._wait_for_server(41003, proc=child) is False


def test_the_clear_reports_only_a_confirmed_removal(launcher_root, monkeypatch, tmp_path):
    launcher, data = launcher_root
    (data / "state" / "server_port").write_text("1", encoding="utf-8")
    assert launcher._clear_port_file() is True and not (data / "state" / "server_port").exists()
    stuck = tmp_path / "stuck"
    stuck.mkdir()
    (stuck / "server_port").mkdir()  # unlink of a directory fails: removal is not confirmed
    monkeypatch.setattr(launcher, "PORT_FILE", stuck / "server_port")
    assert launcher._clear_port_file() is False


@pytest.mark.parametrize("parsed,inherited", [("owner", "automatic"), ("automatic", "owner")])
def test_the_parsed_intent_governs_the_child_over_an_inherited_one(launcher_root, monkeypatch, tmp_path,
                                                                   parsed, inherited):
    launcher, data = launcher_root
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "server.py").write_text("", encoding="utf-8")
    captured = {}

    class FakeProcess:
        pid = 4244
        stdout = type("Out", (), {"readline": lambda self, _limit=-1: b""})()

    monkeypatch.setenv("OUROBOROS_LAUNCH_INTENT", inherited)
    monkeypatch.setattr(launcher, "_launch_intent", parsed)
    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "REPO_DIR", repo)
    monkeypatch.setattr(launcher, "_load_settings", lambda: {})
    monkeypatch.setattr(launcher, "_apply_settings_to_env", lambda _s: None)
    monkeypatch.setattr(launcher, "_hidden_popen", lambda cmd, **kw: captured.update(kw) or FakeProcess())
    launcher.start_agent(port=1)
    assert captured["env"]["OUROBOROS_LAUNCH_INTENT"] == parsed


def test_an_automatic_launcher_does_not_spawn_a_generation_after_panic(launcher_root, monkeypatch):
    launcher, data = launcher_root
    (data / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    shutdown = threading.Event()
    monkeypatch.setattr(launcher, "_shutdown_event", shutdown)
    monkeypatch.setattr(launcher, "_launch_intent", "automatic")
    monkeypatch.setattr(launcher, "_pre_generation_cleanup", lambda: [])
    monkeypatch.setattr(launcher, "start_agent", lambda port: pytest.fail("Panic must hold the automatic start"))
    launcher.agent_lifecycle_loop(1)
    assert shutdown.is_set()
    assert (data / "state" / "panic_stop.flag").read_text(encoding="utf-8") == "panic"


def _historical_parser():
    show = subprocess.run(["git", "show", "v7.6.0:ouroboros/launcher_bootstrap.py"], cwd=str(REPO),
                          capture_output=True, text=True)
    if show.returncode != 0:
        pytest.skip("the v7.6.0 release tag is not available in this clone")
    import types

    module = types.ModuleType("launcher_bootstrap_v760")
    sys.modules[module.__name__] = module  # its dataclasses resolve their module by name
    try:
        exec(compile(show.stdout, "v7.6.0:ouroboros/launcher_bootstrap.py", "exec"), module.__dict__)  # noqa: S102
    finally:
        sys.modules.pop(module.__name__, None)
    return module.parse_launch_options


@pytest.mark.parametrize("venv_args", [["--runtime-venv", "/ouroboros/body/venv"], ["--runtime-venv=/v"]])
def test_the_relaunch_argv_drops_the_runtime_venv_for_a_historical_launcher(venv_args):
    from ouroboros.launcher_bootstrap import parse_launch_options, source_handoff_argv

    argv = ["--no-ui", "--seed-bundle", "/opt/ouroboros/seed", *venv_args, "--launch-intent", "automatic"]
    assert parse_launch_options(argv).runtime_venv is not None
    handoff = source_handoff_argv(argv)
    assert handoff == ["--no-ui", "--seed-bundle", "/opt/ouroboros/seed", "--launch-intent", "automatic"]
    historical = _historical_parser()(handoff)
    assert historical.no_ui and historical.launch_intent == "automatic"
    assert str(historical.seed_bundle) == "/opt/ouroboros/seed"

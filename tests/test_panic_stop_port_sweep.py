"""Panic sweeps no port, and in what order it hard-exits.

Characterization of the Emergency Stop contract: the main and Host Service
listeners belong to the exiting server, so the hard exit closes them and no
listener is killed by port — an unrelated process on a default or configured
Ouroboros port survives. Owned work is requested first; persistence precedes
the hard exit.

Every destructive operation is neutralized here except the one under test: a
real sentinel listener stands on the ports the old sweeps targeted.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest



class _ExitCalled(RuntimeError):
    pass


def _harness(monkeypatch, tmp_path, *, port_kill=None):
    """Neutralize every destructive teardown op and record the panic timeline."""
    from ouroboros import server_control

    events: list = []

    def _record(name, value=None):
        events.append((name, value) if value is not None else (name,))

    def _phase(name, *, request_only=False):
        _record(name + ("_request" if request_only else "_settlement"))
        return [{"requested": True, "scope": "group", "pid": 123}] if request_only else None

    def _kill_port(port):
        _record("kill_process_on_port", port)
        if port_kill is not None:
            port_kill(port)

    from ouroboros.startup_historical_audit import audit

    monkeypatch.setattr(audit, "stop", lambda: None)
    monkeypatch.setattr(server_control, "_persist_panic_controls", lambda _root: _record("persist_controls"))
    original_flag = server_control._write_panic_flag

    def write_flag(root):
        _record("write_flag")
        original_flag(root)

    monkeypatch.setattr(server_control, "_write_panic_flag", write_flag)
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda **kw: SimpleNamespace(
        panic_stop=lambda **kw: _phase("local_model", **kw),
        stop_server=lambda: _phase("local_model")))
    monkeypatch.setattr("ouroboros.claudexor_daemon.get_owned_daemon", lambda **kw: SimpleNamespace(
        panic_stop=lambda **kw: _phase("owned_daemon", **kw),
        stop_outcome=lambda: _phase("owned_daemon")))
    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda **kw: _phase("shells", **kw))
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground",
                        lambda *a, **kw: _phase("foreground", request_only=kw.get("request_only", False)))
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services",
                        lambda *a, **kw: _phase("services", request_only=kw.get("request_only", False)))
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda **kw: _phase("companions", **kw))
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.platform_layer.force_kill_pid", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", _kill_port)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    before = set(threading.enumerate())

    def exit_for_test(code):
        _record("hard_exit", code)
        # Only the test joins already-launched, neutralized settlement helpers.
        # Their asynchronous completion is not an os._exit guarantee.
        for thread in set(threading.enumerate()) - before:
            if thread.name.startswith("panic-"):
                thread.join(timeout=5)
                assert not thread.is_alive()
        raise _ExitCalled(code)

    monkeypatch.setattr(server_control.os, "_exit", exit_for_test)
    return events, lambda **kw: _record("kill_workers", kw)

def _sentinel_listener():
    """A foreign process listening on a free loopback port; the caller stops it."""
    code = ("import socket, sys, time; s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
            "print(s.getsockname()[1], flush=True); time.sleep(120)")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    return proc, int(proc.stdout.readline())


@pytest.mark.skipif(sys.platform == "win32", reason="the sentinel port probe uses POSIX lsof semantics")
def test_panic_sweeps_no_port_and_a_foreign_listener_survives(monkeypatch, tmp_path):
    """The removed sweeps targeted the bound main port and the Host Service port: a stranger
    there survives Panic, and the real port primitive is never reached."""
    import server
    from ouroboros import platform_layer

    sentinel, port = _sentinel_listener()
    try:
        # The recorder forwards to the REAL primitive: the old sweep would have killed the sentinel.
        events, kill_workers = _harness(monkeypatch, tmp_path, port_kill=platform_layer.kill_process_on_port)
        monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: port)
        monkeypatch.setattr(server, "DEFAULT_PORT", port)
        monkeypatch.setattr(server, "DATA_DIR", tmp_path)
        with pytest.raises(_ExitCalled) as exit_info:
            server._execute_panic_stop(SimpleNamespace(stop=lambda: None), kill_workers)
        assert exit_info.value.args[0] == server.PANIC_EXIT_CODE
        time.sleep(0.5)
        assert sentinel.poll() is None, "Panic killed an unrelated listener by port"
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass
        assert not any(event[0] == "kill_process_on_port" for event in events)
    finally:
        sentinel.kill()
        sentinel.wait(timeout=5)


def test_requests_precede_settlement_and_persistence_before_hard_exit(monkeypatch, tmp_path):
    """Physical owned-worker requests, not the legacy cooperative callback,
    precede cleanup and control persistence."""
    import server

    events, kill_workers = _harness(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    child = SimpleNamespace(pid=12345, _ouroboros_stop_backstop=None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [child])
    monkeypatch.setattr("supervisor.worker_pool_lifecycle.kill_worker_tree",
                        lambda pid, *, panic_process: events.append(("owned_worker_request", pid, panic_process)))

    with pytest.raises(_ExitCalled):
        server._execute_panic_stop(SimpleNamespace(stop=lambda: None), kill_workers)

    names = [event[0] for event in events]
    direct_requests = {"local_model_request", "owned_daemon_request", "shells_request",
                       "foreground_request", "services_request", "companions_request",
                       "owned_worker_request"}
    assert set(names[:7]) == direct_requests
    assert all(name.endswith("_request") for name in names[:7])
    assert not any(name.endswith("_request") for name in names[7:])
    assert ("owned_worker_request", child.pid, child) in events[:7]
    assert names.index("write_flag") > 6
    assert names.index("persist_controls") > 6
    assert names.index("hard_exit") > names.index("persist_controls")
    assert {name for name in names if name.endswith("_settlement")} == {
        "local_model_settlement", "owned_daemon_settlement", "shells_settlement",
        "foreground_settlement", "services_settlement", "companions_settlement"}
    assert not any(name == "kill_workers" for name in names)
    assert (tmp_path / "state" / "panic_stop.flag").read_text(encoding="utf-8") == "panic"


def test_no_server_host_leaf_imports_the_composition_root():
    """The lazy `import server` inside the panic port sweep was the last back-edge
    from a host leaf to the composition root. Scanned as a class, at any depth, so
    a future lazy import inside a function cannot quietly restore it."""
    import ast
    import pathlib

    import server

    leaves = sorted((pathlib.Path(server.__file__).parent / "ouroboros").glob("server_*.py"))
    # 5 pre-split host leaves + the 6 D11 server-split leaves (liveness,
    # maintenance, owner_routing, process, restart, routing_context). The floor
    # guards against the glob silently matching nothing.
    assert len(leaves) >= 11
    for leaf in leaves:
        for node in ast.walk(ast.parse(leaf.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert not any(
                    alias.name == "server" or alias.name.startswith("server.")
                    for alias in node.names
                ), leaf.name
            if isinstance(node, ast.ImportFrom):
                assert node.module != "server", leaf.name


def test_emergency_process_cleanup_stays_a_separate_path_from_panic(monkeypatch, tmp_path):
    """The uvicorn-hang cleanup is NOT the panic: it finalizes running tasks with an
    honest interrupted reason and returns, where panic hard-exits. Keeping them
    separate is an explicit design decision, not an oversight."""
    import server

    worker_calls = []
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda: None)
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *a, **k: None)
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **kw: worker_calls.append(kw))
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.platform_layer.force_kill_pid", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port",
                        lambda _port: pytest.fail("the restart cleanup sweeps no port"))
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)

    # Returns normally: no os._exit patch is needed, which is the whole point.
    server._emergency_process_cleanup()

    assert worker_calls and worker_calls[0]["force"] is True

"""Panic requests captured daemon termination before independent settlement.

A worker tree kill preserves installation daemons during ordinary close; Panic
therefore sends its own daemon request. Request/settlement failure is disclosed,
never a reason to prevent the other requests or hard exit.
"""
from __future__ import annotations

import copy
import logging
import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.serial


class _ExitCalled(RuntimeError):
    pass


def _run_panic(monkeypatch, tmp_path, *, daemon_stop, panic_request=None, diagnostics=None,
               worker_stop=None, children=(), configure=None, ingress=False):
    """Neutralize destructive owners; keep request and settlement separate.

    Legacy diagnostics pass the actual isolated manager.stop as daemon_stop.
    Supplying it never turns that potentially blocking call into a request.
    Test-owned settlement threads are joined AFTER the mocked hard exit so
    their writes cannot escape into another fixture; production does not join.
    """
    from ouroboros import server_control
    from ouroboros.maintenance import child as packer
    from ouroboros.startup_historical_audit import audit
    from ouroboros.storage_free import free as storage_free

    monkeypatch.setattr(audit, "stop", lambda: None)
    monkeypatch.setattr(packer, "stop", lambda: None)
    monkeypatch.setattr(storage_free, "stop", lambda: None)
    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda **kw: [])
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda *a, **kw: [])
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *a, **kw: [])
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda **kw: [])
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda **kw: SimpleNamespace(
        panic_stop=lambda **kw: [], stop_server=lambda: None))
    monkeypatch.setattr(server_control, "_persist_panic_controls", lambda _root: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: list(children))
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    monkeypatch.setattr("ouroboros.claudexor_daemon.get_owned_daemon", lambda **kw: SimpleNamespace(
        panic_stop=lambda *, request_only: panic_request(request_only=request_only) if panic_request else [],
        stop_outcome=daemon_stop))
    monkeypatch.setattr(server_control.os, "_exit", lambda code: (_ for _ in ()).throw(_ExitCalled(code)))

    if configure is not None:
        configure()
    worker_calls = []
    before = set(threading.enumerate())

    def workers(**kw):
        worker_calls.append(kw)
        if worker_stop is not None:
            worker_stop()

    def report(message, *args):
        if diagnostics is not None:
            diagnostics.append({"requests": copy.deepcopy(args[0]), "settlements": copy.deepcopy(args[1]),
                                "flag_written": args[2], "controls_written": args[3]})
        logging.getLogger(__name__).critical(message, *args)

    try:
        with pytest.raises(_ExitCalled) as exit_info:
            def stop():
                server_control.execute_panic_stop(
                    consciousness=SimpleNamespace(stop=lambda: None), kill_workers_fn=workers,
                    data_dir=tmp_path, panic_exit_code=99, log=SimpleNamespace(critical=report),
                )
            if ingress:
                server_control.PanicIngress(stop).request("/panic")
            else:
                stop()
        assert exit_info.value.args == (99,)
    finally:
        for thread in set(threading.enumerate()) - before:
            if thread.name.startswith("panic-"):
                thread.join(timeout=5)
                assert not thread.is_alive(), "test-owned Panic settlement did not finish"
    return worker_calls


def test_panic_stop_requests_and_settles_owned_claudexor_daemon(monkeypatch, tmp_path):
    calls = []
    diagnostics = []
    worker_calls = _run_panic(
        monkeypatch, tmp_path,
        panic_request=lambda **kw: calls.append(("request", kw)) or [{"requested": True, "scope": "group", "pid": 123}],
        daemon_stop=lambda: calls.append(("settle", {})) or True,
        diagnostics=diagnostics,
    )
    assert calls == [("request", {"request_only": True}), ("settle", {})]
    assert diagnostics[0]["requests"]["daemon"] == [{"requested": True, "scope": "group", "pid": 123}]
    assert worker_calls == []  # root-first pool cleanup must not race worker-local requests


@pytest.mark.parametrize("phase", ["request", "settlement"])
def test_panic_discloses_daemon_failure_and_continues(monkeypatch, tmp_path, phase):
    diagnostics = []

    def fail(**_kwargs):
        raise RuntimeError(f"daemon {phase} failed")

    worker_calls = _run_panic(
        monkeypatch, tmp_path,
        panic_request=fail if phase == "request" else lambda **kw: [],
        daemon_stop=fail if phase == "settlement" else lambda: True,
        diagnostics=diagnostics,
    )
    observed = diagnostics[0]["requests" if phase == "request" else "settlements"]["daemon"]
    assert observed == {"requested": False, "error": f"RuntimeError: daemon {phase} failed"}
    assert worker_calls == []  # root-first pool cleanup must not race worker-local requests


def test_panic_requests_daemon_before_worker_owned_requests(monkeypatch, tmp_path):
    """The worker's private request precedes independent daemon settlement."""
    order = []
    child = SimpleNamespace(pid=123)
    monkeypatch.setattr("supervisor.worker_pool_lifecycle.kill_worker_tree",
                        lambda *_a, **_k: order.append("worker_request") or {"requested": True})
    _run_panic(monkeypatch, tmp_path,
               panic_request=lambda **kw: order.append("daemon_request") or [],
               daemon_stop=lambda: order.append("daemon_settlement") or True,
               children=(child,))
    assert order == ["daemon_request", "worker_request", "daemon_settlement"]


def test_panic_requests_the_held_restart_successor_in_its_request_phase(monkeypatch, tmp_path):
    """A successor this generation spawned is stopped by Panic, never waited for or left serving."""
    from ouroboros import platform_layer, server_control

    successor = SimpleNamespace(pid=4242)
    monkeypatch.setattr(server_control, "_restart_successors", [successor])
    monkeypatch.setattr(server_control, "_restart_stop_requested", False)
    monkeypatch.setattr(platform_layer, "request_process_tree_kill",
                        lambda proc: {"pid": proc.pid, "requested": True, "scope": "process"})
    diagnostics = []
    _run_panic(monkeypatch, tmp_path, daemon_stop=lambda: True, diagnostics=diagnostics)
    assert diagnostics[0]["requests"]["restart-successor"] == [{"pid": 4242, "requested": True, "scope": "process"}]
    assert server_control._restart_stop_requested is True  # a spawn still publishing meets this intent


@pytest.mark.parametrize("fault", [
    "audit", "packer", "storage", "free", "model", "daemon", "executor-import", "request-thread", "bounded-first",
    "bounded-all", "diagnostic", "ingress-start", "ingress-construction",
])
def test_accepted_panic_completes_despite_early_failure(monkeypatch, tmp_path, fault):
    """An accepted Panic keeps termination, independent requests and bounded writes.

    The failed ingress races Quit and the restart watcher before the real executor;
    OS stops are test-owned doubles, while the flag write and completion path are real.
    """
    import builtins
    from ouroboros import server_control
    from ouroboros.maintenance import child as packer
    from ouroboros.startup_historical_audit import audit
    from ouroboros.storage import count as storage_count
    from ouroboros.storage_free import free as storage_free

    calls, diagnostics = [], []
    child = SimpleNamespace(pid=123)
    real_start, real_thread = threading.Thread.start, threading.Thread
    real_import = builtins.__import__
    failed_writes, created_threads = [], []

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected " + fault)

    def racing_exits():
        assert server_control._restart_stop_requested
        with monkeypatch.context() as patch:
            for exit_fn, code, name in ((server_control.exit_after_restart, 42, "watcher-parked"),
                                        (server_control.exit_after_shutdown, 0, "quit-parked")):
                patch.setattr(threading, "Event", lambda: SimpleNamespace(wait=lambda: calls.append(name)))
                exit_fn(code)
        fail()

    def start(thread):
        if thread.name == "panic-ingress" and fault == "ingress-start":
            racing_exits()
        if thread.name == "panic-bounded-write" and (fault == "bounded-all" or
                                                     fault == "bounded-first" and not failed_writes):
            failed_writes.append(thread.name)
            fail()
        return real_start(thread)

    def thread(*args, **kwargs):
        created_threads.append(kwargs.get("name"))
        if kwargs.get("name") == "panic-ingress" and fault == "ingress-construction":
            racing_exits()
        if kwargs.get("name") == "panic-commands" and fault == "request-thread":
            fail()
        return real_thread(*args, **kwargs)

    def importing(name, *args, **kwargs):
        if name == "multiprocessing":
            fail()
        return real_import(name, *args, **kwargs)

    def configure():
        monkeypatch.setattr(server_control, "_restart_stop_requested", False)
        monkeypatch.setattr(server_control, "stop_restart_successor", lambda: calls.append("successor") or [])
        monkeypatch.setattr("supervisor.worker_pool_lifecycle.kill_worker_tree",
                            lambda *_a, **_k: calls.append("child") or {"requested": True})
        monkeypatch.setattr(server_control, "_persist_panic_controls", lambda _: calls.append("controls"))
        monkeypatch.setattr(threading.Thread, "start", start)
        monkeypatch.setattr(threading, "Thread", thread)
        if fault == "audit":
            monkeypatch.setattr(audit, "stop", fail)
        else:
            monkeypatch.setattr(audit, "stop", lambda: calls.append("audit-stop"))
        monkeypatch.setattr(packer, "stop", fail if fault == "packer" else lambda: calls.append("packer-stop"))
        monkeypatch.setattr(storage_count, "stop", fail if fault == "storage" else lambda: calls.append("storage-stop"))
        monkeypatch.setattr(storage_free, "stop", fail if fault == "free" else lambda: calls.append("free-stop"))
        if fault == "model":
            monkeypatch.setattr("ouroboros.local_model.get_manager", fail)
        elif fault == "daemon":
            monkeypatch.setattr("ouroboros.claudexor_daemon.get_owned_daemon", fail)
        elif fault == "executor-import":
            monkeypatch.setattr(builtins, "__import__", importing)
        elif fault == "diagnostic":
            monkeypatch.setattr(logging.Logger, "critical", fail)

    _run_panic(monkeypatch, tmp_path, daemon_stop=lambda: True, diagnostics=diagnostics,
               children=(child,), configure=configure, ingress=fault.startswith("ingress-"))
    assert server_control._restart_stop_requested is True
    assert not {"panic-startup-audit", "panic-maintenance", "panic-storage-count", "panic-storage-free"} & set(created_threads)
    # The four native stops run synchronously first; one failing never skips the others.
    first = 2 if fault.startswith("ingress-") else 0
    head = [name for name, broken in (("audit-stop", fault == "audit"), ("packer-stop", fault == "packer"),
                                      ("storage-stop", fault == "storage"), ("free-stop", fault == "free"))
            if not broken]
    assert calls[first:first + len(head)] == head
    if fault != "executor-import":
        assert "successor" in calls and "child" in calls
        if fault != "bounded-all":
            assert calls.index("child") < calls.index("controls")
    assert (tmp_path / "state/panic_stop.flag").exists() == (fault not in {"bounded-first", "bounded-all"})
    assert ("controls" in calls) == (fault != "bounded-all")
    if fault.startswith("ingress-"):
        assert calls[:2] == ["watcher-parked", "quit-parked"]
    if fault != "bounded-all":
        assert diagnostics[0]["flag_written"] is (fault != "bounded-first")
        assert diagnostics[0]["controls_written"] is True
        if fault != "executor-import":
            assert set(diagnostics[0]["requests"]) == {
                "local-model", "daemon", "commands", "executors", "services", "companions",
                "restart-successor", "scope-census", "child-123", "startup-audit", "maintenance", "storage-count",
                "storage-free",
            }
            # The installation-scope census is one more independent request; this root planted no marker.
            assert diagnostics[0]["requests"]["scope-census"] == {"requested": False, "reason": "no installation marker planted"}
    if fault in {"audit", "packer", "storage", "free", "model", "daemon", "request-thread"}:
        owner = {"audit": "startup-audit", "packer": "maintenance", "storage": "storage-count", "free": "storage-free",
                 "model": "local-model", "daemon": "daemon", "request-thread": "commands"}[fault]
        assert diagnostics[0]["requests"][owner]["requested"] is False
        assert "injected " + fault in diagnostics[0]["requests"][owner]["error"]

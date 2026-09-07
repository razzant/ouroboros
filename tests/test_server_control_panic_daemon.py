"""Panic stop must take the owned Claudexor daemon down with it.

``execute_panic_stop`` kills workers, tracked shells, services, companions and
ports, but it used to leave the owned ``claudexord`` — and the delegated
harness runs living in its process group — running through a panic. The BIBLE
Emergency Stop Invariant requires ALL subprocess trees to die on panic, and
nothing (including a failing daemon stop) may delay the hard exit.
"""

from types import SimpleNamespace

import pytest


class _ExitCalled(RuntimeError):
    pass


def _run_panic(monkeypatch, tmp_path, *, daemon_stop):
    """Run execute_panic_stop with every destructive teardown op neutralized.

    Returns the recorded kill_workers_fn calls (proof that teardown continued
    past the daemon-stop block).
    """
    from ouroboros import server_control

    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda: None)
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *a, **k: None)
    monkeypatch.setattr(
        "ouroboros.local_model.get_manager",
        lambda: SimpleNamespace(stop_server=lambda: None),
    )
    monkeypatch.setattr("supervisor.state.load_state", lambda: {})
    monkeypatch.setattr("supervisor.state.save_state", lambda _state: None)
    monkeypatch.setattr(
        "supervisor.evolution_lifecycle.complete_evolution_campaign", lambda *a, **k: {}
    )
    monkeypatch.setattr("ouroboros.post_task_evolution.drop_pending_request", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.platform_layer.force_kill_pid", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    monkeypatch.setattr(
        "ouroboros.claudexor_daemon.get_owned_daemon",
        lambda: SimpleNamespace(stop=daemon_stop),
    )
    monkeypatch.setattr(
        server_control.os, "_exit", lambda code: (_ for _ in ()).throw(_ExitCalled(code))
    )

    worker_calls = []
    with pytest.raises(_ExitCalled):
        server_control.execute_panic_stop(
            consciousness=SimpleNamespace(stop=lambda: None),
            kill_workers_fn=lambda **kw: worker_calls.append(kw),
            data_dir=tmp_path,
            panic_exit_code=120,
            log=SimpleNamespace(critical=lambda *a, **k: None),
        )
    return worker_calls


def test_panic_stop_stops_owned_claudexor_daemon(monkeypatch, tmp_path):
    """Panic stops the owned claudexord: delegated runs die with its process group."""
    stops = []
    worker_calls = _run_panic(
        monkeypatch, tmp_path, daemon_stop=lambda: stops.append(True) or True
    )
    assert stops == [True]
    assert worker_calls == [{
        "force": True, "archive_service_logs": False,
        "reconcile_delegate_custody": False,
    }]


def test_panic_stop_survives_daemon_stop_failure(monkeypatch, tmp_path):
    """A failing daemon stop never blocks the panic: the remaining teardown still
    runs (workers killed) and the hard exit still happens."""

    def _boom():
        raise RuntimeError("daemon stop failed")

    worker_calls = _run_panic(monkeypatch, tmp_path, daemon_stop=_boom)
    assert worker_calls == [{
        "force": True, "archive_service_logs": False,
        "reconcile_delegate_custody": False,
    }]


def test_panic_stops_the_daemon_before_the_worker_trees(monkeypatch, tmp_path):
    """The worker tree-kill spares the installation's daemon roots by design, so the
    explicit daemon stop must come FIRST and is what ends them under Panic."""
    order = []
    from ouroboros import server_control

    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda: None)
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.local_model.get_manager",
                        lambda: SimpleNamespace(stop_server=lambda: None))
    monkeypatch.setattr("supervisor.state.load_state", lambda: {})
    monkeypatch.setattr("supervisor.state.save_state", lambda _state: None)
    monkeypatch.setattr("supervisor.evolution_lifecycle.complete_evolution_campaign",
                        lambda *a, **k: {})
    monkeypatch.setattr("ouroboros.post_task_evolution.drop_pending_request", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.platform_layer.force_kill_pid", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    monkeypatch.setattr("ouroboros.claudexor_daemon.get_owned_daemon",
                        lambda: SimpleNamespace(stop=lambda: order.append("daemon_stop") or True))
    monkeypatch.setattr(server_control.os, "_exit",
                        lambda code: (_ for _ in ()).throw(_ExitCalled(code)))
    with pytest.raises(_ExitCalled):
        server_control.execute_panic_stop(
            consciousness=SimpleNamespace(stop=lambda: None),
            kill_workers_fn=lambda **kw: order.append("kill_workers"),
            data_dir=tmp_path, panic_exit_code=120,
            log=SimpleNamespace(critical=lambda *a, **k: None),
        )
    assert order == ["daemon_stop", "kill_workers"]

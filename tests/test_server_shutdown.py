import json
import os
import pathlib
import threading
from types import SimpleNamespace

import time
import pytest


def _stop_restart_watcher(server):
    """Ordinary server return retires its watcher without inventing a restart."""
    for thread in threading.enumerate():
        if thread.name.endswith("(_check_restart)"):
            thread.join(timeout=5)
            assert not thread.is_alive(), "restart watcher did not stop"
    server._restart_requested.clear()


def test_owned_stop_runs_every_in_memory_owner_and_reads_records_from_the_set(monkeypatch, tmp_path):
    """One failing owner does not skip the others, and its failure keeps the outcome unconfirmed
    beside the durable records (none here); those come from the set, never a walk."""
    from ouroboros import owned_shutdown

    order = []
    def shell_failure(*, deadline):
        order.append("shell")
        assert deadline == owned_shutdown._GENERATION_STOP.deadline  # the first door's one grace
        raise OSError("shell custody unavailable")
    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", shell_failure)
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground",
                        lambda *_a, **_kw: pytest.fail("the stop settles durable records itself"))
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services",  # its list contract: nothing left unconfirmed
                        lambda root, **kw: order.append(("services", root, kw)) or [])
    monkeypatch.setattr(os, "walk", lambda *_a, **_kw: pytest.fail("the exit never walks data/state"))
    outcome = owned_shutdown.stop_owned_work(tmp_path)
    assert outcome["state"] == "unconfirmed" and outcome["targets"] == 0 and outcome["unconfirmed"] == []
    assert outcome["memory_unconfirmed"] == {"commands": {"state": "failed"}}
    assert sorted(order, key=str) == sorted(["shell", ("services", tmp_path.resolve(),
                                                       {"wait": True, "durable": False})], key=str)


def test_shutdown_task_cleanup_args_never_reports_crash_storm():
    """Graceful shutdown (requested restart or external signal) must finalize a
    running task as cancelled/interrupted, never as a worker crash storm."""
    import server

    status_restart, reason_restart = server._shutdown_task_cleanup_args(restart_requested=True)
    status_signal, reason_signal = server._shutdown_task_cleanup_args(restart_requested=False)

    assert status_restart == "cancelled"
    assert status_signal == "cancelled"
    # The misleading crash-storm label must never be used for a graceful shutdown.
    assert "crash storm" not in reason_restart.lower()
    assert "crash storm" not in reason_signal.lower()
    assert "restart" in reason_restart.lower()
    assert "interrupted" in reason_signal.lower()


def test_managed_update_restart_preserves_pending_queue(monkeypatch, tmp_path):
    import server

    worker_calls = []
    state = {"owner_chat_id": 0}
    ctx = SimpleNamespace(
        load_state=lambda: dict(state),
        save_state=lambda updated: state.update(updated),
        safe_restart=lambda **_kwargs: (True, "ok"),
        kill_workers=lambda **kwargs: worker_calls.append(kwargs),
        persist_queue_snapshot=lambda **_kwargs: None,
        # The evolution restart-receipt check reads pending_restart_verify.json
        # from ctx.DRIVE_ROOT before any restart proceeds.
        DRIVE_ROOT=tmp_path,
        REPO_DIR=tmp_path,
    )
    monkeypatch.setattr(
        server,
        "_managed_update_pending_kwargs",
        lambda: {"preserve_pending": True},
    )
    monkeypatch.setattr(server, "_request_restart_exit", lambda: None)

    server._perform_supervisor_restart(ctx)

    assert worker_calls[0]["preserve_pending"] is True
    assert worker_calls[0]["terminal_status"] == "cancelled"


def test_pre_transaction_update_quiesce_also_preserves_pending(monkeypatch):
    import server
    import supervisor.update_merge as update_merge
    import supervisor.workers as workers

    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {})
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "managed_update:smart")
    monkeypatch.setattr(
        workers,
        "worker_pool_admission_state",
        lambda: {"disabled_reason": "managed_update"},
    )

    assert server._managed_update_pending_kwargs() == {"preserve_pending": True}


def test_managed_update_quiesce_passes_owner_wait_handoffs(monkeypatch, tmp_path):
    import ouroboros.gateway.control as control
    import ouroboros.owner_wait as owner_wait
    import ouroboros.delegate_recovery as delegate_recovery
    import supervisor.git_ops as git_ops
    import supervisor.workers as workers

    captured = {}
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda _reason: None)
    monkeypatch.setattr(workers, "drain_repo_writers", lambda: [])
    monkeypatch.setattr(owner_wait, "prepare_owner_wait_handoffs", lambda *_args: {"waiting"})
    monkeypatch.setattr(
        delegate_recovery,
        "prepare_planned_restart_handoffs",
        lambda *_args, **kwargs: set(kwargs["additional_task_ids"]),
    )

    def fake_kill(**kwargs):
        captured.update(kwargs)
        return ["worker:still-running"]

    monkeypatch.setattr(workers, "kill_workers_for_update", fake_kill)

    blockers = control._quiesce_repo_writers("smart")

    assert blockers == ["worker:still-running"]
    assert captured["preserve_running_task_ids"] == {"waiting"}


def test_manual_rollback_quiesce_keeps_cancellation_semantics(monkeypatch, tmp_path):
    """A rollback returns the tree to an OLDER runtime, so no owner wait may be
    handed to it: the pool stop keeps its ordinary interrupt/cancel semantics."""
    import ouroboros.gateway.control as control
    import ouroboros.delegate_recovery as delegate_recovery
    import ouroboros.owner_wait as owner_wait
    import supervisor.git_ops as git_ops
    import supervisor.workers as workers

    captured = {}
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda _reason: None)
    monkeypatch.setattr(workers, "drain_repo_writers", lambda: [])
    monkeypatch.setattr(
        owner_wait, "prepare_owner_wait_handoffs",
        lambda *_args: pytest.fail("a rollback must not park an owner wait"),
    )
    monkeypatch.setattr(
        delegate_recovery, "prepare_planned_restart_handoffs",
        lambda *_args, **_kwargs: pytest.fail("a rollback must not prepare a handoff"),
    )
    monkeypatch.setattr(workers, "kill_workers_for_update", lambda **kwargs: captured.update(kwargs) or [])
    monkeypatch.setattr(
        "ouroboros.tools.services.kill_all_services", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        "ouroboros.process_custody.quiesce_custodied_services", lambda *_a: (True, []))

    assert control._quiesce_repo_writers("manual_rollback") == []
    assert captured["preserve_running_task_ids"] == set()
    assert captured["terminal_status"] == "interrupted"


def test_failed_owner_wait_handoff_blocks_the_update_instead_of_killing_the_pool(
    monkeypatch, tmp_path,
):
    """Custody preparation is the gate: if it cannot park the wait, the pool is
    never stopped and repo-writer admission re-opens."""
    import ouroboros.gateway.control as control
    import ouroboros.owner_wait as owner_wait
    import supervisor.git_ops as git_ops
    import supervisor.workers as workers

    reopened = []
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda _reason: None)
    monkeypatch.setattr(workers, "drain_repo_writers", lambda: [])
    monkeypatch.setattr(workers, "open_repo_writer_admission", lambda: reopened.append(True))

    def refuse(*_args):
        raise OSError("owner wait source bytes are unreadable")

    monkeypatch.setattr(owner_wait, "prepare_owner_wait_handoffs", refuse)
    monkeypatch.setattr(
        workers, "kill_workers_for_update",
        lambda **_kwargs: pytest.fail("the pool must not stop after a failed handoff"),
    )

    blockers = control._quiesce_repo_writers("smart")

    assert blockers == ["owner_wait_handoff:OSError: owner wait source bytes are unreadable"]
    assert reopened == [True]


def test_managed_update_restart_arms_prepared_transaction_for_direct_reexec(
    monkeypatch, tmp_path,
):
    import ouroboros.delegate_recovery as delegate_recovery
    import ouroboros.gateway.control as control
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge
    from ouroboros import server_process, server_restart
    import server

    prepared = {
        "transaction_id": "tx-owner-wait",
        "status": "prepared",
        "supervisor_pid": os.getpid(),
    }
    tx_dir = tmp_path / "state" / "delegate_recovery_transactions"
    tx_dir.mkdir(parents=True)
    (tx_dir / "active.json").write_text(
        '{"transaction_id":"tx-owner-wait"}', encoding="utf-8",
    )
    (tx_dir / "tx-owner-wait.json").write_text(
        json.dumps(prepared), encoding="utf-8",
    )
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())
    owner, restart = threading.Event(), threading.Event()
    monkeypatch.setattr(server_process, "_owner_restart_requested", owner)
    monkeypatch.setattr(server_process, "_restart_requested", restart)
    monkeypatch.setattr(server, "_owner_restart_requested", owner)
    monkeypatch.setattr(server_restart, "_owner_restart_requested", owner)
    monkeypatch.setattr(server, "_restart_current_process_impl", lambda *_a, **_kw: None)
    monkeypatch.setattr(update_merge, "read_update_tx_strict", lambda: ("valid", {"phase": "pending_boot_smoke"}))
    monkeypatch.delenv(delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV, raising=False)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        request_restart=server_process._request_restart_exit)))
    response = control._restart_response(request, strategy="auto_merge", plan={})

    assert response.status_code == 200
    assert owner.is_set() and restart.is_set()
    assert delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV not in os.environ
    server._restart_current_process("127.0.0.1", 8765)
    assert os.environ[delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV] == "tx-owner-wait"
    assert not delegate_recovery.owner_restart_in_progress(tmp_path)


def test_failed_managed_update_restart_disarms_transaction_token(monkeypatch, tmp_path):
    import ouroboros.delegate_recovery as delegate_recovery
    import ouroboros.gateway.control as control

    monkeypatch.setenv(delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV, "stale-tx")
    request = type("Request", (), {"app": type("App", (), {"state": type("State", (), {
        "request_restart": None,
    })()})()})()

    response = control._restart_response(request, strategy="auto_merge", plan={})

    assert response.status_code == 200
    assert json.loads(response.body)["status"] == "restart_required"
    assert delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV not in os.environ


def test_ordinary_restart_disarms_orphan_update_intent(monkeypatch):
    import server
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge

    calls = []
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda **_kw: object())
    monkeypatch.setattr(update_merge, "release_update_lock", lambda _lock: None)
    monkeypatch.setattr(update_merge, "read_update_tx_strict", lambda: ("absent", {}))
    monkeypatch.setattr(git_ops, "_clear_update_intent", lambda: calls.append("clear") or True)

    ok, message = server._safe_restart_serialized(
        lambda **_kwargs: (calls.append("restart") or True, "ok"),
        reason="owner_restart",
        unsynced_policy="rescue_and_reset",
    )

    assert (ok, message) == (True, "ok")
    assert calls == ["clear", "restart"]


def test_restart_deferred_while_assisted_merge_is_being_resolved(monkeypatch):
    """Regression pin for the restart guard: while an assisted managed-update merge is
    mid-resolution (any assisted phase), _safe_restart_serialized must DEFER the restart
    instead of running the checkout/reset that would wipe the resolver's worktree."""
    import server
    import supervisor.update_merge as update_merge

    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda **_kw: object())
    monkeypatch.setattr(update_merge, "release_update_lock", lambda _lock: None)
    for phase in ("materializing_assisted", "assisted_resolution", "committing_assisted"):
        monkeypatch.setattr(
            update_merge, "read_update_tx_strict",
            lambda phase=phase: ("valid", {"phase": phase, "task_id": "resolver"}),
        )

        ok, message = server._safe_restart_serialized(
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("restart must be deferred during assisted resolution")
            ),
            reason="owner_restart",
            unsynced_policy="rescue_and_reset",
        )

        assert ok is False, phase
        assert "deferred" in message.lower()


def test_supervisor_startup_restores_queue_before_worker_reset():
    """A fresh process must not overwrite the durable queue with its empty memory."""
    import inspect
    import server

    source = inspect.getsource(server._run_supervisor)
    restore = source.index("restored_pending = restore_pending_from_snapshot(")
    reset = source.index("kill_workers(preserve_pending=True)")
    spawn = source.index("spawn_workers()")  # one ready worker; demand grows the rest
    assert restore < reset < spawn


def test_update_finalizer_waits_for_real_supervisor_outcome(monkeypatch):
    """The waiter blocks on the init OUTCOME latch, never on readiness: a failed
    init is an outcome that must not hang the boot, and it must not have to
    fake readiness to unblock this waiter (the API would paint Online)."""
    import server

    calls = []
    outcome = SimpleNamespace(wait=lambda: calls.append("wait"))
    monkeypatch.setattr(server, "_supervisor_init_done", outcome)
    monkeypatch.setattr(server, "_supervisor_ready", threading.Event())  # never set here
    monkeypatch.setattr(server, "_supervisor_error", None)

    assert server._wait_for_supervisor_update_finalize() is True
    assert calls == ["wait"]
    monkeypatch.setattr(server, "_supervisor_error", "Supervisor init failed: boom")
    assert server._wait_for_supervisor_update_finalize() is False
    assert calls == ["wait", "wait"]


def test_boot_update_check_notifies_the_live_ui():
    import inspect
    import server

    source = inspect.getsource(server._boot_managed_update_tasks)
    assert '"type": "update_status_ready"' in source
    assert source.index("compute_managed_update_status(fetch=True)") < source.index(
        '"type": "update_status_ready"'
    )


def test_successful_boot_rollback_requests_restart_and_preserves_queue(monkeypatch):
    import server
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge
    import supervisor.workers as workers

    calls = []
    monkeypatch.setattr(server, "_wait_for_supervisor_update_finalize", lambda: False)
    monkeypatch.setattr(
        update_merge, "finalize_managed_update_on_boot",
        lambda supervisor_ready, **_kwargs: {"finalized": False, "rolled_back": True},
    )
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {"phase": "marker_cleanup_retry"})
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda reason: calls.append(("gate", reason)))
    monkeypatch.setattr(server, "_request_restart_exit", lambda: calls.append(("restart", "")))
    monkeypatch.setattr(
        git_ops, "compute_managed_update_status",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("restarting generation must not check feed")),
    )

    server._boot_managed_update_tasks()

    assert calls == [
        ("gate", "managed_update:rollback_restart"),
        ("restart", ""),
    ]


def test_failed_boot_rollback_does_not_restart(monkeypatch):
    import server
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge

    calls = []
    monkeypatch.setattr(server, "_wait_for_supervisor_update_finalize", lambda: False)
    monkeypatch.setattr(
        update_merge, "finalize_managed_update_on_boot",
        lambda supervisor_ready, **_kwargs: {"finalized": False, "rolled_back": False},
    )
    monkeypatch.setattr(
        git_ops, "compute_managed_update_status",
        lambda fetch: {"available": False, "check_ok": True},
    )
    monkeypatch.setattr(server, "_request_restart_exit", lambda: calls.append("restart"))
    monkeypatch.setattr(server, "broadcast_ws_sync", lambda payload: calls.append(payload["type"]))

    server._boot_managed_update_tasks()

    assert calls == ["update_status_ready"]


@pytest.mark.parametrize("intent,marker,starts", [
    ("automatic", "panic", False),
    ("automatic", "owner_restart_no_resume", True),
    (None, "panic", True),
])
def test_main_automatic_start_preserves_panic(monkeypatch, tmp_path, capsys, intent, marker, starts):
    import server

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    if intent is None:
        monkeypatch.delenv("OUROBOROS_LAUNCH_INTENT", raising=False)
    else:
        monkeypatch.setenv("OUROBOROS_LAUNCH_INTENT", intent)
    flag = tmp_path / "state" / "panic_stop.flag"
    if marker is not None:
        flag.parent.mkdir()
        flag.write_text(marker, encoding="utf-8")

    class ReachedStartup(Exception):
        pass

    def start():
        raise ReachedStartup()

    monkeypatch.setattr(server, "verify_settings_integrity", start)
    if starts:
        with pytest.raises(ReachedStartup):
            server.main()
    else:
        assert server.main() == 0
        assert "Ouroboros is stopped. Use Start to resume." in capsys.readouterr().out
    assert flag.exists() is (marker is not None)
    if marker is not None:
        assert flag.read_text(encoding="utf-8") == marker


def test_main_normal_exit_does_not_run_emergency_cleanup(monkeypatch, tmp_path):
    import server

    cleanup_calls = []

    class FakeServer:
        def __init__(self, _config):
            self.should_exit = False
            self.stop_watcher_bound = False

        def watch_launcher_stop(self):
            self.stop_watcher_bound = True

        def run(self, *, sockets):
            assert self.stop_watcher_bound
            assert len(sockets) == 1 and sockets[0].getsockname()[1] > 0
            return None

    monkeypatch.setattr(server, "load_settings", lambda: {"OUROBOROS_SERVER_HOST": "127.0.0.1"})
    monkeypatch.setattr(server, "parse_server_args", lambda *_a, **_k: SimpleNamespace(host="127.0.0.1", port=0, host_explicit=False))
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "get_network_auth_startup_warning", lambda _host: "")
    monkeypatch.setattr(server, "validate_network_auth_configuration", lambda _host: "")
    monkeypatch.setattr(server, "find_free_port", lambda _host, port, **_: port)
    monkeypatch.setattr(server, "write_port_file", lambda *_a, **_k: None)
    monkeypatch.setattr(server.uvicorn, "Config", lambda *a, **k: object())
    monkeypatch.setattr(server, "_SignalStopServer", FakeServer)  # the main() server seam (#1142)
    monkeypatch.setattr(server, "_emergency_process_cleanup", lambda: cleanup_calls.append("cleanup"))
    monkeypatch.setattr(server, "_event_loop", None)  # the watcher's close_all_ws hop needs no loop here
    server._restart_requested.clear()

    try:
        assert server.main() == 0
    finally:
        _stop_restart_watcher(server)
    assert cleanup_calls == []


@pytest.mark.parametrize("launcher_managed", [True, False])
def test_main_graceful_restart_cleanup_avoids_port_sweep(monkeypatch, tmp_path, launcher_managed):
    import server

    cleanup_calls, bound = [], []

    class FakeServer:
        def __init__(self, _config):
            self.should_exit = False
            self.stop_watcher_bound = False

        def watch_launcher_stop(self):
            self.stop_watcher_bound = True

        def run(self, *, sockets):
            assert self.stop_watcher_bound
            assert len(sockets) == 1 and sockets[0].getsockname()[1] > 0
            bound.append(sockets[0].getsockname()[1])
            server._restart_requested.set()
            return None

    monkeypatch.setattr(server, "load_settings", lambda: {"OUROBOROS_SERVER_HOST": "127.0.0.1"})
    monkeypatch.setattr(server, "parse_server_args", lambda *_a, **_k: SimpleNamespace(host="127.0.0.1", port=0, host_explicit=False))
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "get_network_auth_startup_warning", lambda _host: "")
    monkeypatch.setattr(server, "validate_network_auth_configuration", lambda _host: "")
    monkeypatch.setattr(server, "find_free_port", lambda _host, port, **_: port)
    monkeypatch.setattr(server, "write_port_file", lambda *_a, **_k: None)
    monkeypatch.setattr(server.uvicorn, "Config", lambda *a, **k: object())
    monkeypatch.setattr(server, "_SignalStopServer", FakeServer)  # the main() server seam (#1142)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", launcher_managed)
    census = {"worker_exits": {"unconfirmed": []}, "live_children": [], "owned_stop": {"state": "completed"}}
    monkeypatch.setattr(server, "_emergency_process_cleanup", lambda **kw: cleanup_calls.append(kw) or census)
    seam_calls = []  # the authority switch seam runs on the cleanup's census, before the exit
    monkeypatch.setattr(server.store_switch, "switch_at_restart",
                        lambda root, repo_dir, cleanup: seam_calls.append((root, repo_dir, cleanup)) or {"status": "no_intent"})
    transfers = []

    def transfer(host, port):
        assert cleanup_calls == [{}]
        assert seam_calls == [(tmp_path, server.REPO_DIR, census)]
        transfers.append((host, port))

    monkeypatch.setattr(server, "_restart_current_process", transfer)
    exits = []
    monkeypatch.setattr(server.os, "_exit", exits.append)
    monkeypatch.setattr(server, "_event_loop", None)  # the watcher's close_all_ws hop needs no loop here
    server._restart_requested.clear()

    try:
        server.main()
    finally:
        _stop_restart_watcher(server)

    assert cleanup_calls == [{}]
    assert seam_calls == [(tmp_path, server.REPO_DIR, census)]
    assert transfers == ([] if launcher_managed else [("127.0.0.1", bound[0])])  # the port it bound
    assert exits == [server.RESTART_EXIT_CODE]


@pytest.mark.parametrize("launcher_managed", [True, False])
def test_main_fault_exit_restarts_through_the_installs_own_restarter(monkeypatch, tmp_path, launcher_managed):
    """The Fault door's exit (ARCHITECTURE §9) at main()'s return. Launcher-managed: 1, no cleanup here (the launcher's
    crash path restarts; never 42, never 99). Direct: the pre-exec census first, the planned-restart transaction
    dropped from the env, no store switch, then ``restart_current_process(owner_initiated=False)`` with the bound
    port; a failed successor returns 1. The quiet direction: without a fault main returns 0 and runs no cleanup
    (``test_main_normal_exit_does_not_run_emergency_cleanup``)."""
    import server
    from ouroboros import server_process
    from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV

    calls, bound = [], []
    exit_event, stop_event = threading.Event(), threading.Event()

    class FakeServer:
        def __init__(self, _config):
            self.should_exit = False

        def watch_launcher_stop(self):
            pass

        def run(self, *, sockets):
            bound.append(sockets[0].getsockname()[1])
            server_process.latch_fault_stop("bus_lost")  # the loop's door, latched during the run
            return None

    monkeypatch.setattr(server, "load_settings", lambda: {"OUROBOROS_SERVER_HOST": "127.0.0.1"})
    monkeypatch.setattr(server, "parse_server_args", lambda *_a, **_k: SimpleNamespace(host="127.0.0.1", port=0, host_explicit=False))
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "get_network_auth_startup_warning", lambda _host: "")
    monkeypatch.setattr(server, "validate_network_auth_configuration", lambda _host: "")
    monkeypatch.setattr(server, "find_free_port", lambda _host, port, **_: port)
    monkeypatch.setattr(server, "write_port_file", lambda *_a, **_k: None)
    monkeypatch.setattr(server.uvicorn, "Config", lambda *a, **k: object())
    monkeypatch.setattr(server, "_SignalStopServer", FakeServer)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", launcher_managed)
    monkeypatch.setattr(server, "_event_loop", None)
    for module in (server, server_process):
        monkeypatch.setattr(module, "_exit_signalled", exit_event)
        monkeypatch.setattr(module, "_supervisor_stop", stop_event)
    monkeypatch.setattr(server_process, "_stop_source", "external_signal")
    monkeypatch.setattr(server_process, "_arm_stop_backstop", lambda source, action=None: calls.append(("backstop", source)))
    monkeypatch.setattr(server_process, "_FAULT_EXIT", None)
    monkeypatch.setattr(server_process, "_FAULT_EXIT_CODE", None)
    census = {"worker_exits": {"unconfirmed": []}, "live_children": [], "owned_stop": {"state": "completed"}}
    monkeypatch.setattr(server, "_emergency_process_cleanup", lambda: calls.append(("cleanup",)) or census)
    monkeypatch.setattr(server.store_switch, "switch_at_restart",
                        lambda *_a, **_k: pytest.fail("a fault exit switches no store and adopts nothing"))
    successor = {"fails": False}

    def restart(host, port, **kw):
        calls.append(("restart", host, port, kw))
        if successor["fails"]:
            raise OSError("exec refused")

    monkeypatch.setattr(server, "_restart_current_process_impl", restart)
    monkeypatch.setenv(PLANNED_RESTART_TRANSACTION_ENV, "stale-tx")
    server._restart_requested.clear()

    try:
        code = server.main()
    finally:
        _stop_restart_watcher(server)
    assert not server._restart_requested.is_set() and server_process.server_stop_source() == "bus_lost"
    if launcher_managed:
        assert code == 1 and calls == [("backstop", "bus_lost")]
        assert os.environ[PLANNED_RESTART_TRANSACTION_ENV] == "stale-tx"  # not this branch's business
        return
    assert code == 0  # the exec never returns; a Windows successor's parent exit 0 is its handoff
    assert calls == [("backstop", "bus_lost"), ("cleanup",),
                     ("restart", "127.0.0.1", bound[0], {"repo_dir": server.REPO_DIR, "log": server_process.log,
                                                         "owner_initiated": False})]
    assert PLANNED_RESTART_TRANSACTION_ENV not in os.environ

    # A failed successor returns 1: no half-alive process, no false restart code.
    calls.clear()
    exit_event.clear()
    server_process._FAULT_EXIT_CODE = None  # the next generation's own decision (monkeypatch restores it)
    successor["fails"] = True
    try:
        assert server.main() == 1
    finally:
        _stop_restart_watcher(server)
    assert [call[0] for call in calls] == ["backstop", "cleanup", "restart"]


@pytest.mark.parametrize("launcher_managed", [True, False])
def test_the_fault_backstop_takes_the_same_door_as_mains_return(monkeypatch, launcher_managed):
    """The backstop that ends a fault teardown outliving the launcher grace IS the fault door, never a bare
    exit that leaves a direct install down (IMPL-1 open item 1, accepted as a defect). Launcher-managed: exit 1
    and no census here, the launcher restarts. Direct: the pre-exec census and the successor (the exec never
    returns; its double does, and a Windows handoff's parent exits 0); a failed successor exits 1. The decision
    is taken once: main's return and the timer read one answer, so there is one census and one successor.
    Without a fault the door answers 0 (quiet)."""
    from ouroboros import server_process

    calls, exits = [], []
    monkeypatch.setattr(server_process, "_restart_requested", threading.Event())
    monkeypatch.setattr(server_process, "_FAULT_EXIT", None)
    monkeypatch.setattr(server_process, "_FAULT_EXIT_CODE", None)
    monkeypatch.setattr(server_process.os, "_exit", exits.append)
    successor = {"fails": False}

    def restart(host, port, **kw):
        calls.append(("restart", host, port, kw["owner_initiated"]))
        if successor["fails"]:
            raise OSError("exec refused")

    server_process.bind_fault_exit("127.0.0.1", 8765, launcher_managed=launcher_managed, repo_dir=pathlib.Path("/repo"),
                                   cleanup=lambda: calls.append(("cleanup",)), restart=restart)
    monkeypatch.setattr(server_process, "_stop_source", "external_signal")
    assert server_process.fault_exit() == 0 and calls == []  # no fault: an ordinary return, nothing taken
    monkeypatch.setattr(server_process, "_stop_source", "bus_lost")
    server_process._fault_backstop("bus_lost")
    if launcher_managed:
        assert exits == [1] and calls == []
        assert server_process.fault_exit() == 1 and calls == []  # main's return reads the same answer
        return
    assert exits == [0] and calls == [("cleanup",), ("restart", "127.0.0.1", 8765, False)]
    assert server_process.fault_exit() == 0 and len(calls) == 2, "one decision: no second census or successor"
    server_process._FAULT_EXIT_CODE = None  # another generation: its successor fails
    calls.clear()
    exits.clear()
    successor["fails"] = True
    server_process._fault_backstop("bus_lost")
    assert exits == [1] and [call[0] for call in calls] == ["cleanup", "restart"]


def test_a_stuck_census_does_not_hang_the_fault_door(monkeypatch):
    """The Fault door's bounded census: on a direct install the pre-exec census (``_emergency_process_cleanup``) can
    block behind a teardown holding the queue lock while ``fault_exit`` holds its own; the one decision is still
    taken: the census is bounded by the launcher grace and the successor starts over an unfinished census (the next
    generation's custody sweep settles the rest). The other direction: a census that finishes runs whole, before the
    successor."""
    from ouroboros import server_process

    calls, release = [], threading.Event()
    monkeypatch.setattr(server_process, "_restart_requested", threading.Event())
    monkeypatch.setattr(server_process, "_FAULT_EXIT", None)
    monkeypatch.setattr(server_process, "_FAULT_EXIT_CODE", None)
    monkeypatch.setattr(server_process, "_FAULT_CENSUS_BOUND_SEC", 0.2)
    monkeypatch.setattr(server_process, "_stop_source", "bus_lost")

    def stuck_census():
        calls.append("census-start")
        release.wait(10)
        calls.append("census-end")

    server_process.bind_fault_exit("127.0.0.1", 8765, launcher_managed=False, repo_dir=pathlib.Path("/repo"),
                                   cleanup=stuck_census, restart=lambda *_a, **_k: calls.append("restart"))
    started = time.monotonic()
    try:
        assert server_process.fault_exit() == 0
        assert time.monotonic() - started < 5 and calls == ["census-start", "restart"]
    finally:
        release.set()
    for thread in threading.enumerate():
        if thread.name == "fault-census":
            thread.join(5)
    assert calls == ["census-start", "restart", "census-end"]

    calls.clear()
    server_process._FAULT_EXIT_CODE = None
    server_process.bind_fault_exit("127.0.0.1", 8765, launcher_managed=False, repo_dir=pathlib.Path("/repo"),
                                   cleanup=lambda: calls.append("census"), restart=lambda *_a, **_k: calls.append("restart"))
    assert server_process.fault_exit() == 0 and calls == ["census", "restart"]


def test_a_fault_teardown_cancels_unsaved_rows_as_server_shutdown_naming_the_cause(monkeypatch):
    """The Fault door (ARCHITECTURE §9), the documented stop-cause/Continue mapping (no new source): an unsaved row the
    fault teardown cancels keeps ``stop_source="server_shutdown"`` (a technical interruption the owner may Continue)
    and its reason text names the cause; every other door keeps its own text (the mapping answers '')."""
    from ouroboros import server_process
    from ouroboros.owner_continue import TECHNICAL_CANCEL_SOURCES
    from ouroboros.server_restart import _shutdown_task_cleanup_args

    assert "server_shutdown" in TECHNICAL_CANCEL_SOURCES
    assert server_process.fault_cancel_reason() == ""  # an ordinary stop: the door's own text stands
    assert "crash" not in _shutdown_task_cleanup_args(False)[1].split("not a worker crash")[0]
    monkeypatch.setattr(server_process, "_stop_source", "bus_lost")
    bus_lost = server_process.fault_cancel_reason()
    assert "event-bus process exited" in bus_lost and bus_lost.endswith("not a worker crash.")
    monkeypatch.setattr(server_process, "_stop_source", "loop_failed")
    assert "failed three times in a row" in server_process.fault_cancel_reason()
    monkeypatch.setattr(server_process, "_stop_source", "launcher_quit")
    assert server_process.fault_cancel_reason() == ""


@pytest.mark.parametrize("stop_source", ["bus_lost", "external_signal"])
def test_the_lifespan_teardown_hands_a_faults_cause_to_kill_workers(tmp_path, monkeypatch, request, stop_source):
    """The real lifespan's one ``kill_workers`` call (the Fault door, ARCHITECTURE §9): after a fault the rows it cancels
    carry the fault's sentence as their reason, still ``stop_source="server_shutdown"``; after an ordinary stop the
    door's own text stands (the quiet direction). After that call a fault ends the delegated runs of the rows it
    holds with their saved work (``end_fault_held_runs``); an ordinary stop addresses none."""
    from starlette.testclient import TestClient

    import server as srv
    from ouroboros import server_process
    from ouroboros.server_restart import _shutdown_task_cleanup_args
    from supervisor import workers
    from supervisor.restart_retention import RETAINED_FOR_BOOT
    from tests.test_extensions_api import _patch_lifespan_for_drive_root_test

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    srv.app.app.state.drive_root = drive_root
    srv.app.app.state.repo_dir = tmp_path / "repo"
    _patch_lifespan_for_drive_root_test(monkeypatch, srv, {}, request)
    monkeypatch.setattr(srv, "_boot_managed_update_tasks", lambda: None)
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: "")
    calls, ended = [], []
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **kw: calls.append(kw) or workers.RUNNING.update(
        {"t-held": {"task": {"id": "t-held"}, RETAINED_FOR_BOOT: True}}))
    monkeypatch.setattr(server_process, "end_task_runs", lambda root, ids, **kw: ended.append(list(ids)))
    with TestClient(srv.app):
        monkeypatch.setattr(server_process, "_stop_source", stop_source)
    [call] = calls
    assert call["stop_source"] == "server_shutdown"
    _status, ordinary = _shutdown_task_cleanup_args(False)
    if stop_source == "bus_lost":
        assert call["result_reason"] == server_process.fault_cancel_reason() != ordinary
        assert "event-bus process exited" in call["result_reason"]
        assert ended == [["t-held"]], "the fault's held task keeps its saved work; its own runs end after the kill"
    else:
        assert call["result_reason"] == ordinary
        assert ended == [], "an ordinary stop leaves the held rows' runs as they were"


@pytest.mark.parametrize("stop_source", ["bus_lost", "launcher_quit"])
def test_a_fault_ends_its_held_tasks_own_runs_and_a_quit_leaves_them(tmp_path, monkeypatch, stop_source):
    """Owner 1A at the Fault door: a task the fault holds with its saved work continues at the next boot, and its own
    live delegated run is cancelled now with the abrupt-loss verdict (``owner_task_gone``) over the attach-only
    gateway; a review panel's run on that task stays its panel's. A Quit holds the same row and addresses no run."""
    from ouroboros import claudexor_daemon, delegate_custody as custody, owned_shutdown, server_process
    from ouroboros.task_results import write_task_result
    from supervisor import workers
    from supervisor.restart_retention import RETAINED_FOR_BOOT
    from tests._delegated_transport_shared import _LiveRunStub

    write_task_result(tmp_path, "t-held", "running", root_task_id="t-held")
    for run_id, source in (("run-held", "delegated_subagent"), ("run-panel", "review_substrate")):
        custody._CUSTODY.clear()
        custody.record_started(tmp_path, custody.RunCustody(
            run_id=run_id, task_id="t-held", route_id="r", model="m", project_id="p", project_owned=False,
            root_task_id="t-held", ledger_root=str(tmp_path), source=source))
    custody._CUSTODY.clear()
    live = _LiveRunStub()
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: live)
    monkeypatch.setattr(owned_shutdown, "stop_deadline", lambda: time.monotonic() + 20)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "RUNNING", {"t-held": {"task": {"id": "t-held"}, RETAINED_FOR_BOOT: True}})
    monkeypatch.setattr(server_process, "_stop_source", stop_source)
    try:
        server_process.end_fault_held_runs()
    finally:
        custody._CUSTODY.clear()
    assert live.cancels == ([("run-held", "owner_task_gone")] if stop_source == "bus_lost" else [])


def _record_runs(root, *rows):
    from ouroboros import delegate_custody as custody

    for run_id, task_id, source in rows:
        custody._CUSTODY.clear()
        custody.record_started(root, custody.RunCustody(
            run_id=run_id, task_id=task_id, route_id="r", model="m", project_id="p", project_owned=False,
            root_task_id=task_id, ledger_root=str(root), source=source))
    custody._CUSTODY.clear()


@pytest.mark.parametrize("stop_source", ["bus_lost", "launcher_quit"])
def test_a_fault_ends_a_sleeping_tasks_own_run_and_a_quit_leaves_it(tmp_path, monkeypatch, stop_source):
    """Owner 1A for a task asleep on its own delegated run: the teardown parks the sleep with its saved work under an
    app-stop hold (the list the next boot names), and after a fault its own run ends now with the abrupt-loss verdict;
    the panel's run stays, and a hold an earlier stop placed is not this fault's (its run stays too). A Quit parks
    the same sleep and addresses no run."""
    from ouroboros import claudexor_daemon, delegate_custody as custody, owned_shutdown, server_process
    from ouroboros.utils import utc_now_iso
    from supervisor.events_budget import HOLD_SAVED_WORK, hold_budget_row
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._delegated_transport_shared import _LiveRunStub
    from tests.test_restart_retention import _pool_events
    from tests.test_warm_restart_retention import _park

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _park(tmp_path, workers, "sleep")
    older = {**hold_budget_row({"id": "t-older"}, reason=HOLD_SAVED_WORK), "held_at": "2026-01-01T00:00:00+00:00"}
    workers.PENDING.append({"id": "t-older", "_budget_pause_hold": older})
    _record_runs(tmp_path, ("run-sleep", "saved-warm", "delegated_subagent"),
                 ("run-panel", "saved-warm", "review_substrate"), ("run-older", "t-older", "delegated_subagent"))
    live = _LiveRunStub()
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: live)
    monkeypatch.setattr(owned_shutdown, "stop_deadline", lambda: time.monotonic() + 20)
    monkeypatch.setattr(owned_shutdown._GENERATION_STOP, "entry", {"entered_at": utc_now_iso(), "cause": stop_source})
    monkeypatch.setattr(server_process, "_stop_source", stop_source)
    try:
        workers.kill_workers(force=True, terminal_status="cancelled",
                             result_reason=server_process.fault_cancel_reason() or "Server shutdown.",
                             stop_source="server_shutdown", retain_saved_work=True)
        server_process.end_fault_held_runs()
    finally:
        custody._CUSTODY.clear()
    parked = next(task for task in workers.PENDING if task.get("id") == "saved-warm")
    assert parked["_budget_pause_hold"]["reason"] == "saved_sleep_recovery", "the sleep keeps its saved work"
    assert live.cancels == ([("run-sleep", "owner_task_gone")] if stop_source == "bus_lost" else [])


def test_the_fault_door_texts_say_its_kept_works_runs_end():
    """The two tests above in words: the Fault door's row (ARCHITECTURE §9) and DEPLOYMENT's fault paragraph say the
    kept work's own delegated runs end, which a Close does not do, while the shared engine stays."""
    docs = pathlib.Path(__file__).resolve().parents[1] / "docs"
    rows = (docs / "architecture" / "09-shutdown-and-process-cleanup.md").read_text(encoding="utf-8").splitlines()
    row = next(line for line in rows if line.startswith("| Fault ("))
    assert row.split(" | ")[1] == "as Close; kept work's delegated runs end"
    deployment = " ".join((docs / "DEPLOYMENT.md").read_text(encoding="utf-8").split())
    assert "daemon is untouched, while the kept tasks' own delegated runs end." in deployment


@pytest.mark.parametrize("door", ["restart", "panic"])
def test_a_fault_under_a_restart_or_an_accepted_panic_ends_no_held_run(tmp_path, monkeypatch, door):
    """The Fault door's run end belongs to the fault alone: a restart keeps its own run rule, and an accepted Panic
    owns the exit (the Emergency Stop is never a fault), so neither ends a held task's run here."""
    from ouroboros import claudexor_daemon, delegate_custody as custody, owned_shutdown, server_control, server_process
    from ouroboros.task_results import write_task_result
    from supervisor import workers
    from supervisor.restart_retention import RETAINED_FOR_BOOT
    from tests._delegated_transport_shared import _LiveRunStub

    write_task_result(tmp_path, "t-held", "running", root_task_id="t-held")
    _record_runs(tmp_path, ("run-held", "t-held", "delegated_subagent"))
    live = _LiveRunStub()
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: live)
    monkeypatch.setattr(owned_shutdown, "stop_deadline", lambda: time.monotonic() + 20)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "RUNNING", {"t-held": {"task": {"id": "t-held"}, RETAINED_FOR_BOOT: True}})
    monkeypatch.setattr(server_process, "_stop_source", "bus_lost")
    if door == "restart":
        restart = threading.Event()
        restart.set()
        monkeypatch.setattr(server_process, "_restart_requested", restart)
    else:
        monkeypatch.setattr(server_control, "_restart_stop_requested", True)
    try:
        server_process.end_fault_held_runs()
    finally:
        custody._CUSTODY.clear()
    assert live.cancels == []


def test_ending_kept_runs_returns_within_half_the_stop_grace_on_a_hung_engine(tmp_path, monkeypatch):
    """A sick engine never holds a door: the pass joins for at most half the generation's remaining stop grace, the
    rest stays with the owned-process stop, and the pass goes on in the background while the process lives."""
    from ouroboros import owned_shutdown, server_process
    from ouroboros.task_results import write_task_result

    release = threading.Event()

    class _HungEngine:
        def handshake(self, **_kw): return {}
        def get_run(self, rid, *, timeout_sec=None):
            release.wait(5)
            raise RuntimeError("never answered")
        def cancel_run(self, rid, reason=""):
            release.wait(5)
            return {}
        def close(self): pass

    write_task_result(tmp_path, "t-kept", "running", root_task_id="t-kept")
    _record_runs(tmp_path, ("run-kept", "t-kept", "delegated_subagent"))
    deadline = time.monotonic() + 2.0
    monkeypatch.setattr(owned_shutdown, "stop_deadline", lambda: deadline)
    started = time.monotonic()
    try:
        server_process.end_task_runs(tmp_path, ["t-kept"], gateway_factory=_HungEngine)
        elapsed = time.monotonic() - started
        hung = [t for t in threading.enumerate() if t.name == "kept-task-runs-end" and t.is_alive()]
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "kept-task-runs-end":
                thread.join(10)
    assert elapsed < 1.5 and hung, "the door returned at its bound while the pass still waits on the engine"


def test_the_startup_notice_names_the_previous_faults_cause_and_the_held_count(monkeypatch):
    """The Fault door (ARCHITECTURE §9): after a fault exit the next start says what ended the previous generation and
    how many restored rows THIS fault holds with their saved work (a hold placed before the faulted generation entered
    its shutdown, by an earlier Quit or Restart nobody resumed, is not counted); after an ordinary stop the clause is
    absent (quiet)."""
    from ouroboros import owned_shutdown, server_process
    from supervisor.events_budget import HOLD_SAVED_WORK, hold_budget_row

    held = [hold_budget_row({"id": "a"}, reason=HOLD_SAVED_WORK), hold_budget_row({"id": "b"}, reason=HOLD_SAVED_WORK)]
    older = {**hold_budget_row({"id": "c"}, reason=HOLD_SAVED_WORK), "held_at": "2026-01-01T00:00:00+00:00"}
    pending = [{"id": "a", "_budget_pause_hold": held[0]}, {"id": "b", "_budget_pause_hold": held[1]},
               {"id": "c", "_budget_pause_hold": older}, {"id": "plain"}]
    fault = {"cause": "bus_lost", "entered_at": "2026-06-01T00:00:00+00:00"}
    monkeypatch.setattr(owned_shutdown._GENERATION_STOP, "previous", fault)
    assert server_process.held_saved_work_count(pending) == 2, "the earlier Quit's unresumed hold is not this fault's"
    assert server_process.held_saved_work_ids(pending, "") == ["a", "b", "c"]
    monkeypatch.setattr(owned_shutdown._GENERATION_STOP, "previous", {"cause": "launcher_quit"})
    assert server_process.previous_fault_sentence(2) == "" and server_process.held_saved_work_count(pending) == 0
    monkeypatch.setattr(owned_shutdown._GENERATION_STOP, "previous", {"cause": "bus_lost"})
    assert server_process.previous_fault_sentence(2) == (
        "The server restarted itself after its event-bus process exited. 2 tasks keep their saved work.")
    monkeypatch.setattr(owned_shutdown._GENERATION_STOP, "previous", {"cause": "loop_failed"})
    assert server_process.previous_fault_sentence(0).endswith("its task loop failed three times in a row. No task holds saved work.")


def test_a_panic_during_a_faults_teardown_takes_the_fault_clause_and_turn_away(tmp_path, monkeypatch):
    """A fault latched, then the owner pressed Panic before the teardown ended: the entry keeps the fault's first
    cause, yet the stop was the Emergency Stop. While its input is unconsumed the next start states no fault (no
    clause, no recovery turn); once consumed, a later fault entry speaks again."""
    from ouroboros import owned_shutdown, server_process

    monkeypatch.setattr(server_process, "DATA_DIR", tmp_path)
    monkeypatch.setattr(owned_shutdown._GENERATION_STOP, "previous",
                        {"cause": "bus_lost", "entered_at": "2026-06-01T00:00:00+00:00", "generation": "g-prev"})
    flag = tmp_path / "state" / "panic_stop.flag"
    flag.parent.mkdir(parents=True)
    flag.write_text("panic", encoding="utf-8")
    assert server_process.previous_fault_sentence(1) == ""
    assert server_process.core_loss_recovery_turn(
        tmp_path, 7, [], admit=lambda *_a, **_k: pytest.fail("no recovery turn after a Panic"),
        live=lambda: False, gate=lambda: True) is False
    flag.unlink()
    assert server_process.previous_fault_sentence(1) == (
        "The server restarted itself after its event-bus process exited. 1 task keeps its saved work.")


def test_emergency_cleanup_joins_the_one_owned_stop(monkeypatch):
    import server

    stop_calls = []
    worker_calls = []

    monkeypatch.setattr(server, "stop_owned_work", lambda root: stop_calls.append(root))
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **kw: worker_calls.append(kw))
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    server._restart_requested.clear()

    server._emergency_process_cleanup()

    assert stop_calls == [server.DATA_DIR]
    assert worker_calls == [{"force": True, "archive_service_logs": False, "retain_saved_work": True}]  # #1563


def test_emergency_cleanup_during_restart_marks_tasks_cancelled(monkeypatch):
    """A hung restart that reaches emergency cleanup must finalize running tasks
    as interrupted-by-restart, never as a worker crash storm."""
    import server

    worker_calls = []

    monkeypatch.setattr(server, "stop_owned_work", lambda _root: None)
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **kw: worker_calls.append(kw))
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    server._restart_requested.set()

    try:
        server._emergency_process_cleanup()
    finally:
        server._restart_requested.clear()

    assert len(worker_calls) == 1
    call = worker_calls[0]
    assert call["force"] is True
    assert call["archive_service_logs"] is False
    assert call["terminal_status"] == "cancelled"
    assert "crash storm" not in call["result_reason"].lower()


def test_panic_stop_kills_services_without_log_finalization(monkeypatch, tmp_path):
    from ouroboros import server_control

    foreground_calls = []
    service_calls = []
    worker_calls = []

    class ExitCalled(RuntimeError):
        pass

    monkeypatch.setattr("ouroboros.startup_historical_audit.audit.stop", lambda: None)
    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda **kw: None)
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda *a, **k: foreground_calls.append((a, k)))
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *a, **k: service_calls.append((a, k)))
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda **kw: SimpleNamespace(
        panic_stop=lambda **kw: [], stop_server=lambda: None))
    monkeypatch.setattr("ouroboros.claudexor_daemon.get_owned_daemon", lambda **kw: SimpleNamespace(
        panic_stop=lambda **kw: [], stop_outcome=lambda: None))
    monkeypatch.setattr(server_control, "_persist_panic_controls", lambda _data: None)
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda **kw: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [SimpleNamespace(pid=4321)])
    native_calls = []
    monkeypatch.setattr("supervisor.worker_pool_lifecycle.kill_worker_tree",
                        lambda pid, panic_process=None: native_calls.append(pid) or {"requested": True})
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    monkeypatch.setattr(server_control.os, "_exit", lambda code: (_ for _ in ()).throw(ExitCalled(code)))

    before = set(threading.enumerate())
    try:
        with pytest.raises(ExitCalled):
            server_control.execute_panic_stop(
            consciousness=SimpleNamespace(stop=lambda: None),
            kill_workers_fn=lambda **kw: worker_calls.append(kw),
            data_dir=tmp_path,
            panic_exit_code=120,
            log=SimpleNamespace(critical=lambda *a, **k: None),
            )
    finally:
        for thread in set(threading.enumerate()) - before:
            if thread.name.startswith("panic-"):
                thread.join(timeout=5)
                assert not thread.is_alive()

    assert native_calls == [4321]  # physical process request before settlement
    assert any(k.get("request_only") for _, k in foreground_calls)
    assert any(k.get("wait") is False for _, k in foreground_calls)
    assert any(k.get("request_only") for _, k in service_calls)
    assert any(k.get("wait") is False for _, k in service_calls)
    assert worker_calls == []  # no root-first cooperative callback dependency
    assert (tmp_path / "state/panic_stop.flag").read_text(encoding="utf-8") == "panic"


# ---------------------------------------------------- shutdown-aware supervisor loop

class _FakeStopEvent:
    """A stop event whose backoff wait returns at once and records its timeout."""

    def __init__(self):
        self.flag = False
        self.waits = []

    def is_set(self):
        return self.flag

    def set(self):
        self.flag = True

    def clear(self):
        self.flag = False

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self.flag


class _Recorder:
    def __init__(self):
        self.alerts = []
        self.watchdog_stops = []
        self.steps = []
        self.stop = _FakeStopEvent()
        self.restart = None
        self.ready = None
        # Snapshot restore reports revived PENDING rows in its return value and
        # the RUNNING rows it fenced for cancellation custody through the
        # keyword-only out-parameter; a test can script either.
        self.restored_pending = 0
        self.fenced_running: list = []

    def restore(self, *_args, terminalized=None, **_kwargs):
        if terminalized is not None:
            terminalized.extend(self.fenced_running)
        return self.restored_pending


def _supervisor_harness(monkeypatch, tmp_path, steps):
    """Drive the REAL server._run_supervisor with every init/tick collaborator
    stubbed (no processes, ports, Manager or live data root). The scripted
    ``steps`` fire from the first call of each tick: ``ok`` = healthy tick,
    ``raise`` = a crash, ``stop``/``restart`` = set the flag (the loop exits at
    its next ``while`` check), ``raise_after_stop``/``raise_after_restart`` =
    the flag is set and the same tick then crashes (the shutdown race),
    ``bus_dead`` = from this tick on every ``get_event_q`` call raises
    ``EventBusLost(-9)``. ``rec.health`` lists the tick of each
    ``ensure_workers_healthy`` call."""
    import threading
    import queue as queue_mod

    import server
    import supervisor.events as events_mod
    import supervisor.message_bus as bus_mod
    import supervisor.queue as queue_pkg
    import supervisor.state as state_mod
    import supervisor.workers as workers_mod
    from ouroboros import server_control

    rec = _Recorder()
    rec.steps = list(steps)
    rec.restart = threading.Event()
    rec.ready = threading.Event()
    rec.ticks, rec.health, rec.bus_dead = 0, [], False

    def _tick_head(_data_dir):
        step = rec.steps.pop(0) if rec.steps else "stop"
        rec.ticks += 1
        rec.bus_dead = rec.bus_dead or step == "bus_dead"
        if step in ("stop", "raise_after_stop"):
            rec.stop.set()
        if step in ("restart", "raise_after_restart"):
            rec.restart.set()
        if step == "bus_lost":  # the manager process ended: the fact every get_event_q call reads
            raise workers_mod.EventBusLost(-15)
        if step == "panic_bus_lost":  # an accepted Panic killed the bus itself: its fact reaches the tick
            server_control._restart_stop_requested = True
            raise workers_mod.EventBusLost(-9)
        if step.startswith("raise"):
            raise BrokenPipeError(32, "Broken pipe")

    class _Bridge:
        def __init__(self, _settings):
            self._broadcast_fn = None

    class _Consciousness:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def tick(self, now=None):
            pass

        def notify(self, reason):
            pass

    import time as time_mod

    noop = lambda *_a, **_k: None  # noqa: E731
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    # Patch the module's bound name, not the process-wide time.sleep.
    monkeypatch.setattr(server, "time", SimpleNamespace(
        sleep=noop, monotonic=time_mod.monotonic, time=time_mod.time,
        thread_time=time_mod.thread_time))  # the loop samples its OWN thread's CPU per phase stamp
    monkeypatch.setattr(server, "_supervisor_stop", rec.stop)
    monkeypatch.setattr(server, "_restart_requested", rec.restart)
    monkeypatch.setattr(server, "_supervisor_ready", rec.ready)
    # The fault door latches through server_process (the same globals request_shutdown latches):
    # the harness owns those too, records the backstop instead of arming a 15 s os._exit timer,
    # and reads the bus as live unless a test says otherwise.
    from ouroboros import server_process
    rec.exit = threading.Event()
    rec.backstops = []
    monkeypatch.setattr(server, "_exit_signalled", rec.exit)
    monkeypatch.setattr(server_process, "_exit_signalled", rec.exit)
    monkeypatch.setattr(server_process, "_supervisor_stop", rec.stop)
    monkeypatch.setattr(server_process, "_stop_source", "external_signal")
    monkeypatch.setattr(server_process, "_arm_stop_backstop", lambda source, action=None: rec.backstops.append(source))
    monkeypatch.setattr(workers_mod, "event_bus_exitcode", lambda: None)
    monkeypatch.setattr(server_control, "_restart_stop_requested", False)  # no Panic unless a step accepts one
    monkeypatch.setattr(workers_mod, "RUNNING", {})
    monkeypatch.setattr(workers_mod, "PENDING", [])
    monkeypatch.setattr(server, "_supervisor_error", None)
    monkeypatch.setattr(server, "_supervisor_thread", None)
    monkeypatch.setattr(server, "_consciousness", None)
    monkeypatch.setattr(server, "_apply_settings_to_env", noop)
    monkeypatch.setattr("ouroboros.usage_store.migrate_from_journal", noop)
    monkeypatch.setattr(server, "_bootstrap_supervisor_repo", lambda _s: (True, "ok"))
    monkeypatch.setattr(server, "_runtime_branch_defaults", lambda: ("dev", "stable"))
    # Startup notices have their own real delivery/state consumer tests. This
    # harness records supervisor crash alerts and stubs other boot side effects.
    monkeypatch.setattr("ouroboros.upgrade_notices.startup_upgrade_notices", noop)
    for name in (
        "_resume_interrupted_project_deletions", "_startup_prune_sweeps", "_startup_custody_sweep",
        "_prune_worktrees", "_prune_delegated_snapshots", "_periodic_supervisor_maintenance",
        # Its own owner notice is another startup side effect, asserted by
        # tests/test_retired_settings_chat_notice.py; here it would be a
        # crash alert (this harness's send_with_budget double records every
        # send, and the boot/crash assertions below own that list).
        "_startup_retired_settings_notice", "_startup_review_pool_notice",
    ):
        monkeypatch.setattr(server, name, noop)
    monkeypatch.setattr(server, "_start_supervisor_liveness_watchdog",
                        lambda _liveness, stop_event=None: rec.watchdog_stops.append(stop_event))
    monkeypatch.setattr(server, "_process_bridge_updates", lambda _bridge, offset, _ctx, **_kw: offset)
    monkeypatch.setattr(server, "_check_pending_restart_drain", lambda _ctx: True)
    monkeypatch.setattr(bus_mod, "init", noop)
    monkeypatch.setattr(bus_mod, "LocalChatBridge", _Bridge)
    monkeypatch.setattr(bus_mod, "send_with_budget", lambda chat_id, text, **kw: rec.alerts.append((chat_id, text)))
    monkeypatch.setattr("ouroboros.utils.set_log_sink", noop)
    monkeypatch.setattr(events_mod, "make_server_log_sink", lambda *_a, **_k: None)
    monkeypatch.setattr(events_mod, "dispatch_event", noop)
    monkeypatch.setattr(state_mod, "init", noop)
    monkeypatch.setattr(state_mod, "init_state", lambda **_k: state_mod.StateRead("current", "primary", {}))
    monkeypatch.setattr(state_mod, "load_state", lambda: {"owner_chat_id": 7})
    for name in ("save_state", "update_state", "append_jsonl", "update_budget_from_usage"):
        monkeypatch.setattr(state_mod, name, noop)
    # The tick's first rotator call (events.jsonl; the dialogue chains no longer rotate) is the tick head.
    monkeypatch.setattr(state_mod, "rotate_jsonl_log_if_needed",
                        lambda root, name, *_a, **_k: _tick_head(root) if name == "events.jsonl" else None)
    for name in ("enqueue_task", "enforce_due_task_timeouts", "enqueue_evolution_task_if_needed",
                 "cancel_task_by_id", "queue_deep_self_review_task",
                 "sort_pending", "check_scheduled_tasks"):
        monkeypatch.setattr(queue_pkg, name, noop)
    monkeypatch.setattr(queue_pkg, "persist_queue_snapshot", lambda **_k: True)
    monkeypatch.setattr(queue_pkg, "restore_pending_from_snapshot", rec.restore)
    for name in ("init", "spawn_workers", "kill_workers", "assign_tasks", "auto_resume_after_restart"):
        monkeypatch.setattr(workers_mod, name, noop)
    monkeypatch.setattr(workers_mod, "ensure_workers_healthy", lambda: rec.health.append(rec.ticks))

    def _event_q():
        if rec.bus_dead:
            raise workers_mod.EventBusLost(-9)
        return queue_mod.Queue()

    monkeypatch.setattr(workers_mod, "get_event_q", _event_q)
    monkeypatch.setattr("ouroboros.delegate_recovery.pre_adopt_planned_handoffs", noop)
    monkeypatch.setattr("ouroboros.tools.services.prune_service_logs", lambda _root: {})
    monkeypatch.setattr("ouroboros.consciousness.BackgroundConsciousness", _Consciousness)
    return rec


def _run(rec, server):
    server._run_supervisor({})
    assert rec.ready.is_set() or server._supervisor_error or rec.exit.is_set()  # init reached the loop
    return rec


@pytest.mark.parametrize(("places_state", "restored", "fenced", "expected"), [
    pytest.param("paused", 0, ["a"], "♻️ Settling 1 task that was still running when the server stopped.", id='paused-0-fenced0-♻️ Settling 1 task that was still running when the server stopped.'),
    pytest.param("empty", 2, ["a", "b"], "♻️ Restored pending queue: 2 tasks. "
     "Settling 2 tasks that were still running when the server stopped.", id='empty-2-fenced1-♻️ Restored pending queue: 2 tasks. Settling 2 tasks that were still running when the server stopped.'),
    pytest.param("unreadable", 0, ["a"], "♻️ Settling 1 task that was still running when the server stopped.", id='unreadable-0-fenced0-♻️ Settling 1 task that was still running when the server stopped.'),
])
def test_boot_notice_states_recovery_as_an_intent(
        monkeypatch, tmp_path, restored, fenced, expected, places_state):
    """A window closed on live work: the boot line names how many tasks the
    restart interrupted and says they are BEING settled. It cannot claim they
    ENDED: restore only minted the durable intent, and cancellation custody
    writes each terminal result a watchdog window later."""
    import server

    rec = _supervisor_harness(monkeypatch, tmp_path, ["stop"])
    from tests.test_places import paused_inventory
    suffix = paused_inventory(tmp_path, places_state)
    expected += f" {suffix}" if suffix else ""
    rec.restored_pending = restored
    rec.fenced_running = list(fenced)
    _run(rec, server)

    assert [text for _chat, text in rec.alerts if text.startswith("♻️")] == [expected]


@pytest.mark.parametrize("cause", ["bus_lost", "launcher_quit"])
def test_a_fault_boot_with_nothing_restored_still_states_the_fault(monkeypatch, tmp_path, cause):
    """After the Fault door (ARCHITECTURE §9) the next start says what ended
    the previous generation even when no queue row was restored and no task interrupted; after an ordinary stop
    with nothing restored the boot posts no notice (the quiet direction)."""
    import server
    from ouroboros import owned_shutdown

    rec = _supervisor_harness(monkeypatch, tmp_path, ["stop"])
    monkeypatch.setattr(owned_shutdown._GENERATION_STOP, "previous", {"cause": cause})
    _run(rec, server)

    expected = ["♻️ The server restarted itself after its event-bus process exited. No task holds saved work."]
    assert [text for _chat, text in rec.alerts if text.startswith("♻️")] == (expected if cause == "bus_lost" else [])


def test_exception_while_stopping_exits_quietly_without_counting_a_crash(monkeypatch, tmp_path, caplog):
    """The graceful-shutdown race: the teardown sets the stop flag, then the tick
    meets the torn-down bus (BrokenPipe). That is not a crash: no owner alarm,
    readiness untouched, no supervisor error — and the watchdog generation stops."""
    import logging

    import server

    rec = _supervisor_harness(monkeypatch, tmp_path, ["ok", "raise_after_stop"])
    with caplog.at_level(logging.INFO, logger="server"):
        _run(rec, server)

    assert rec.alerts == []
    assert rec.ready.is_set() is True
    assert server._supervisor_error is None
    assert rec.watchdog_stops and rec.watchdog_stops[0].is_set()
    assert server._supervisor_thread is None
    assert any("exiting on shutdown" in record.getMessage() for record in caplog.records)
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


def test_exception_while_restarting_exits_quietly_too(monkeypatch, tmp_path):
    import server

    rec = _supervisor_harness(monkeypatch, tmp_path, ["raise_after_restart"])
    _run(rec, server)

    assert rec.alerts == []
    assert rec.ready.is_set() is True
    assert server._supervisor_error is None
    assert rec.watchdog_stops[0].is_set()


def test_the_third_consecutive_failure_on_a_live_bus_ends_the_generation(monkeypatch, tmp_path):
    """Two causes, one door (ARCHITECTURE §5 and §9): the third consecutive tick failure on a LIVE bus posts one
    Main row in owner words (what happened, what happens next; the raw error stays in the log), clears
    readiness, latches the stop exactly as a signal would (``_exit_signalled``, ``_stop_source=loop_failed``,
    the launcher-grace backstop armed), requests no restart, leaves no ``_supervisor_error`` (so ``boot.phase``
    never reads ``failed`` for a loop death), stops the watchdog generation, and the backoff between the
    failures waited on the stop event, never on time.sleep. The quiet direction, a healthy tick between
    failures resetting the count, is ``test_healthy_tick_between_crashes_resets_the_count``."""
    import server
    from ouroboros import server_process

    rec = _supervisor_harness(monkeypatch, tmp_path, ["raise", "raise", "raise", "ok"])
    _run(rec, server)

    assert [chat for chat, _text in rec.alerts] == [7]
    text = rec.alerts[0][1]
    assert text == "🛑 The task loop failed three times in a row; the server is restarting."
    assert "Broken pipe" not in text, "the raw error belongs to the log, not the owner's line"
    assert rec.ready.is_set() is False and server._supervisor_error is None
    assert rec.exit.is_set() and rec.stop.is_set() and server_process.server_stop_source() == "loop_failed"
    assert rec.backstops == ["loop_failed"] and not rec.restart.is_set()
    assert rec.watchdog_stops[0].is_set() and server._supervisor_thread is None
    assert rec.stop.waits == [2, 4]
    assert rec.steps == ["ok"]  # the generation ended: the next tick never ran


def test_a_lost_event_bus_ends_the_generation_at_once_without_retries(monkeypatch, tmp_path):
    """ARCHITECTURE §5: one transient failure on a live bus keeps the loop (a backoff, no row: the quiet direction);
    ``EventBusLost`` ends it at the tick that read it: no backoff, no count, one ``bus_lost`` row (one fact in
    the main line; the exit code is a journal detail), readiness cleared, the stop latched with
    ``_stop_source=bus_lost``."""
    import server
    from ouroboros import server_process

    rec = _supervisor_harness(monkeypatch, tmp_path, ["raise", "ok", "bus_lost", "ok"])
    _run(rec, server)

    assert len(rec.alerts) == 1, rec.alerts
    assert rec.alerts[0][1] == "🛑 The event-bus process of this server exited; the server is restarting."
    assert rec.stop.waits == [2]  # the transient's backoff; the bus loss waited for nothing
    assert rec.steps == ["ok"] and rec.ready.is_set() is False and server._supervisor_error is None
    assert server_process.server_stop_source() == "bus_lost" and rec.exit.is_set() and rec.backstops == ["bus_lost"]
    assert not rec.restart.is_set()


def test_the_tick_reads_the_bus_fact_before_worker_health(monkeypatch, tmp_path):
    """The Fault door's bus fact comes first in a tick (ARCHITECTURE §5): on a dead bus the
    tick ends the generation before ``ensure_workers_healthy`` could hand a dead worker to the reaper, where its
    one retry would be spent and its delegated runs released on a bus that is already gone. With a live bus each
    tick checks worker health once (the quiet direction)."""
    import server
    from ouroboros import server_process

    rec = _supervisor_harness(monkeypatch, tmp_path, ["ok", "ok", "bus_dead", "ok"])
    _run(rec, server)

    assert rec.health == [1, 2], "one health check per live tick, none on the tick that read a dead bus"
    assert rec.alerts == [(7, "🛑 The event-bus process of this server exited; the server is restarting.")]
    assert server_process.server_stop_source() == "bus_lost" and rec.exit.is_set() and rec.steps == ["ok"]


def test_a_proxy_error_is_bus_lost_only_when_the_manager_is_confirmed_dead(monkeypatch, tmp_path):
    """ARCHITECTURE §5: a proxy error (BrokenPipe) counts as the bus lost only once the manager's exit code confirms
    it; with a live manager it is one failure of three (quiet), with a dead one the generation ends at once."""
    import server
    import supervisor.workers as workers
    from ouroboros import server_process

    rec = _supervisor_harness(monkeypatch, tmp_path, ["raise", "stop"])
    _run(rec, server)
    assert rec.alerts == [] and rec.stop.waits == [2] and server_process.server_stop_source() == "external_signal"

    rec = _supervisor_harness(monkeypatch, tmp_path, ["raise", "stop"])
    monkeypatch.setattr(workers, "event_bus_exitcode", lambda: -9)
    _run(rec, server)
    assert len(rec.alerts) == 1 and rec.alerts[0][1].startswith("🛑 The event-bus process of this server exited")
    assert rec.stop.waits == []
    assert server_process.server_stop_source() == "bus_lost" and rec.exit.is_set()


def test_panic_never_takes_the_fault_door(monkeypatch, tmp_path, caplog):
    """Panic owns the exit (ARCHITECTURE §9, the Fault door defers to it):
    an accepted Panic kills the workers and the bus itself, so the tick that then reads ``EventBusLost`` is
    Panic's own work, not a fault. The loop leaves quietly: no ``supervisor_failure`` row, no ``bus_lost``
    cause (so no Continue-eligible ``server_shutdown`` cancellation of its rows), nothing latched or armed.
    The door's own bodies defer too, whichever Panic reaches first: ``end_generation_on_fault`` posts and
    latches nothing, the backstop exits nothing, ``fault_exit`` takes no census and starts no successor; Panic
    owns the exit (``exit_after_shutdown``). The other direction, on the same collaborators: without Panic the
    same bus loss is the Fault door, one row with one fact, the exit code a journal detail, the latch, the backstop."""
    import logging

    import server
    from ouroboros import server_control, server_process

    rec = _supervisor_harness(monkeypatch, tmp_path, ["ok", "panic_bus_lost", "ok"])
    with caplog.at_level(logging.INFO, logger="server"):
        _run(rec, server)
    assert rec.alerts == [] and rec.backstops == [] and not rec.exit.is_set()
    assert server_process.server_stop_source() == "external_signal" and server_process.fault_cancel_reason() == ""
    assert server._supervisor_error is None and rec.steps == ["ok"] and rec.stop.waits == []
    assert any("exiting on shutdown" in record.getMessage() for record in caplog.records)

    rows, exits, calls = [], [], []
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda _path, row: rows.append(row))
    monkeypatch.setattr(server_process.os, "_exit", exits.append)
    monkeypatch.setattr(server_process, "_restart_requested", threading.Event())
    monkeypatch.setattr(server_process, "_FAULT_EXIT", None)
    monkeypatch.setattr(server_process, "_FAULT_EXIT_CODE", None)
    server_process.bind_fault_exit("127.0.0.1", 8765, launcher_managed=False, repo_dir=pathlib.Path("/repo"),
                                   cleanup=lambda: calls.append("census"), restart=lambda *_a, **_k: calls.append("restart"))
    monkeypatch.setattr(server_control, "_restart_stop_requested", True)
    assert server_process.end_generation_on_fault("bus_lost", exitcode=-9) is False
    assert rows == [] and rec.alerts == [] and server_process.server_stop_source() == "external_signal"
    server_process._fault_backstop("bus_lost")
    assert exits == [] and calls == []
    monkeypatch.setattr(server_process, "_stop_source", "bus_lost")  # a fault latched just before Panic was accepted
    assert server_process.fault_exit() == 1 and calls == [], "Panic owns the exit: no census, no successor"

    monkeypatch.setattr(server_control, "_restart_stop_requested", False)
    monkeypatch.setattr(server_process, "_stop_source", "external_signal")
    assert server_process.end_generation_on_fault("bus_lost", exitcode=-9) is True
    assert [row["type"] for row in rows] == ["supervisor_generation_fault"] and rows[0]["exitcode"] == -9
    assert rec.alerts == [(7, "🛑 The event-bus process of this server exited; the server is restarting.")]
    assert server_process.server_stop_source() == "bus_lost" and rec.backstops == ["bus_lost"] and rec.exit.is_set()


def test_healthy_tick_between_crashes_resets_the_count(monkeypatch, tmp_path):
    import server

    rec = _supervisor_harness(monkeypatch, tmp_path, ["raise", "raise", "ok", "raise", "raise", "stop"])
    _run(rec, server)

    assert rec.alerts == []
    assert rec.ready.is_set() is True
    assert server._supervisor_error is None
    assert rec.steps == []
    assert rec.stop.waits == [2, 4, 2, 4]


@pytest.mark.parametrize("published", [None, False])
def test_a_kept_saved_queue_is_not_a_failed_tick(monkeypatch, tmp_path, published):
    """A process that has not taken the saved queue into custody (its restore could not read it) keeps
    that snapshot at every tick's publication (None): durable as it is, so the loop runs on. Only a
    failed publication (False) is a crash. The tick's publisher is ``publish_queue_changes``
    (``queue_publication``); the mark every transition leaves is not the tick's outcome."""
    import server
    import supervisor.queue as queue_pkg

    rec = _supervisor_harness(monkeypatch, tmp_path, ["ok", "ok", "ok", "stop"])
    monkeypatch.setattr(queue_pkg, "publish_queue_changes", lambda: published)
    _run(rec, server)
    if published is None:
        assert rec.alerts == [] and server._supervisor_error is None and rec.stop.waits == []
    else:
        assert len(rec.alerts) == 1 and rec.alerts[0][1].startswith("🛑 The task loop failed three times in a row")


def test_lifespan_teardown_stops_and_joins_the_loop_before_the_bus_goes_down():
    """Source-order pin (the file's style for lifespan ordering): the stop flag is
    the FIRST teardown statement, and the bounded join precedes the worker kill,
    the bridge shutdown and the event-bus shutdown; a fresh lifespan clears the
    flag before it starts a generation."""
    import inspect
    import server

    source = inspect.getsource(server.lifespan)
    assert source.index("_supervisor_stop.clear()") < source.index("_start_supervisor_if_needed(settings)")
    finally_idx = source.index("\n    finally:\n") + 1
    stop_idx = source.index("_supervisor_stop.set()")
    join_idx = source.index("supervisor_thread.join(timeout=2)")
    kill_idx = source.index("kill_workers(")
    bridge_idx = source.index("get_bridge().shutdown()")
    bus_idx = source.index("_shutdown_supervisor_event_bus()")
    assert finally_idx < stop_idx < join_idx < kill_idx < bridge_idx < bus_idx
    # Nothing between `finally:` and the stop flag but whitespace.
    assert source[finally_idx + len("    finally:\n"):stop_idx].strip() == ""


def test_terminal_custody_precedes_every_best_effort_wait_in_the_teardown():
    """The one IRREVERSIBLE durable write of the teardown goes first.

    An external SIGKILL arrives about ten seconds after SIGTERM, while the
    extension-reconcile and host-service waits ahead of kill_workers were worth
    roughly thirty-nine seconds: an interrupted task was simply never
    terminalized. Order is now stop flag, bounded join, terminal custody, then
    the optional waits and the kill-all sweeps.

    The diagnostic `server_shutdown` row is one of those best-effort steps and
    goes AFTER the custody write too: `append_jsonl` waits up to two seconds for
    the log lock and then does file I/O, so on slow storage or a held lock it can
    eat the launcher's force-exit budget before a single worker is terminalized.
    Nothing between the join and `kill_workers` may do I/O. The early cause is the
    door's lock-free entry write, made before the join, so a kill before the row
    still leaves the cause."""
    import inspect
    import server

    teardown = inspect.getsource(server.lifespan).split("\n    finally:\n", 1)[1]
    stop_idx = teardown.index("_supervisor_stop.set()")
    entry_idx = teardown.index("begin_owned_stop(lifespan_drive_root, cause=server_stop_cause())")
    join_idx = teardown.index("supervisor_thread.join(timeout=2)")
    kill_idx = teardown.index("kill_workers(")
    append_idx = teardown.index('"server_shutdown"')
    extension_idx = teardown.index("extension_reconcile_task.cancel()")
    host_idx = teardown.index("host_service_listener.close()")
    sweeps_idx = teardown.index("stop_owned_work(lifespan_drive_root)")
    assert stop_idx < entry_idx < join_idx < kill_idx < extension_idx < host_idx < sweeps_idx
    assert kill_idx < append_idx < extension_idx
    assert teardown.count("begin_owned_stop(") == 1 and "shutdown_entry" not in teardown[join_idx:kill_idx]


def test_supervisor_revival_clears_a_stale_stop_flag(monkeypatch):
    import server

    started = []

    class _Thread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self.kwargs["target"])

    import threading

    monkeypatch.setattr(server, "has_startup_ready_provider", lambda _s: True)
    monkeypatch.setattr(server, "_supervisor_thread", None)
    monkeypatch.setattr(server, "_supervisor_error", "stale")
    monkeypatch.setattr(server, "threading", SimpleNamespace(Thread=_Thread, Event=threading.Event))
    server._supervisor_stop.set()
    try:
        assert server._start_supervisor_if_needed({}) is True
        assert server._supervisor_stop.is_set() is False
        assert started == [server._supervisor_generation]  # the latch-checking thread body (#1142)
    finally:
        server._supervisor_stop.clear()

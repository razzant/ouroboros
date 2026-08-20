from types import SimpleNamespace


def test_lifespan_shutdown_kills_executor_foreground_before_services():
    import inspect
    import server

    source = inspect.getsource(server.lifespan)

    shell_idx = source.index("kill_all_tracked_subprocesses()")
    foreground_idx = source.index("kill_all_foreground(lifespan_drive_root)")
    service_idx = source.index("kill_all_services(lifespan_drive_root)")
    assert shell_idx < foreground_idx < service_idx


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
    from ouroboros import server_restart

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
        server_restart,
        "_managed_update_pending_kwargs",
        lambda: {"preserve_pending": True},
    )
    monkeypatch.setattr(server_restart, "_request_restart_exit", lambda: None)

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


def test_ordinary_restart_disarms_orphan_update_intent(monkeypatch):
    import server
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge

    calls = []
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: object())
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

    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: object())
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
    restore = source.index("restored_pending = restore_pending_from_snapshot()")
    reset = source.index("kill_workers(preserve_pending=True)")
    spawn = source.index("spawn_workers(max_workers)")
    assert restore < reset < spawn


def test_update_finalizer_waits_for_real_supervisor_outcome(monkeypatch):
    import server

    calls = []
    ready = SimpleNamespace(wait=lambda: calls.append("wait"))
    monkeypatch.setattr(server, "_supervisor_ready", ready)
    monkeypatch.setattr(server, "_supervisor_error", None)

    assert server._wait_for_supervisor_update_finalize() is True
    assert calls == ["wait"]


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
        lambda supervisor_ready: {"finalized": False, "rolled_back": True},
    )
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
        lambda supervisor_ready: {"finalized": False, "rolled_back": False},
    )
    monkeypatch.setattr(
        git_ops, "compute_managed_update_status",
        lambda fetch: {"available": False, "check_ok": True},
    )
    monkeypatch.setattr(server, "_request_restart_exit", lambda: calls.append("restart"))
    monkeypatch.setattr(server, "broadcast_ws_sync", lambda payload: calls.append(payload["type"]))

    server._boot_managed_update_tasks()

    assert calls == ["update_status_ready"]


def test_main_normal_exit_does_not_run_emergency_cleanup(monkeypatch):
    import server

    cleanup_calls = []

    class FakeServer:
        def __init__(self, _config):
            self.should_exit = False

        def run(self):
            return None

    monkeypatch.setattr(server, "load_settings", lambda: {"OUROBOROS_SERVER_HOST": "127.0.0.1"})
    monkeypatch.setattr(server, "parse_server_args", lambda *_a, **_k: SimpleNamespace(host="127.0.0.1", port=8765))
    monkeypatch.setattr(server, "get_network_auth_startup_warning", lambda _host: "")
    monkeypatch.setattr(server, "validate_network_auth_configuration", lambda _host: "")
    monkeypatch.setattr(server, "find_free_port", lambda _host, port: port)
    monkeypatch.setattr(server, "write_port_file", lambda *_a, **_k: None)
    monkeypatch.setattr(server.uvicorn, "Config", lambda *a, **k: object())
    monkeypatch.setattr(server.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(server, "_emergency_process_cleanup", lambda: cleanup_calls.append("cleanup"))
    server._restart_requested.clear()

    assert server.main() == 0
    assert cleanup_calls == []


def test_main_graceful_restart_cleanup_avoids_port_sweep(monkeypatch):
    import server

    cleanup_calls = []

    class FakeServer:
        def __init__(self, _config):
            self.should_exit = False

        def run(self):
            server._restart_requested.set()
            return None

    class ExitCalled(RuntimeError):
        pass

    monkeypatch.setattr(server, "load_settings", lambda: {"OUROBOROS_SERVER_HOST": "127.0.0.1"})
    monkeypatch.setattr(server, "parse_server_args", lambda *_a, **_k: SimpleNamespace(host="127.0.0.1", port=8765))
    monkeypatch.setattr(server, "get_network_auth_startup_warning", lambda _host: "")
    monkeypatch.setattr(server, "validate_network_auth_configuration", lambda _host: "")
    monkeypatch.setattr(server, "find_free_port", lambda _host, port: port)
    monkeypatch.setattr(server, "write_port_file", lambda *_a, **_k: None)
    monkeypatch.setattr(server.uvicorn, "Config", lambda *a, **k: object())
    monkeypatch.setattr(server.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", True)
    monkeypatch.setattr(server, "_emergency_process_cleanup", lambda **kw: cleanup_calls.append(kw))
    monkeypatch.setattr(server.os, "_exit", lambda code: (_ for _ in ()).throw(ExitCalled(code)))
    server._restart_requested.clear()

    try:
        server.main()
    except ExitCalled:
        pass
    finally:
        server._restart_requested.clear()

    assert cleanup_calls == [{"port_sweep": False}]


def test_emergency_cleanup_kills_services_without_log_finalization(monkeypatch):
    import server

    foreground_calls = []
    service_calls = []
    worker_calls = []

    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda: None)
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda *a, **k: foreground_calls.append((a, k)))
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *a, **k: service_calls.append((a, k)))
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **kw: worker_calls.append(kw))
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    server._restart_requested.clear()

    server._emergency_process_cleanup(port_sweep=False)

    assert foreground_calls == [((server.DATA_DIR,), {"wait": False})]
    assert service_calls == [((server.DATA_DIR,), {"wait": False})]
    assert worker_calls == [{"force": True, "archive_service_logs": False}]


def test_emergency_cleanup_during_restart_marks_tasks_cancelled(monkeypatch):
    """A hung restart that reaches emergency cleanup must finalize running tasks
    as interrupted-by-restart, never as a worker crash storm."""
    import server

    worker_calls = []

    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda: None)
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *a, **k: None)
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **kw: worker_calls.append(kw))
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    server._restart_requested.set()

    try:
        server._emergency_process_cleanup(port_sweep=False)
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

    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda: None)
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda *a, **k: foreground_calls.append((a, k)))
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *a, **k: service_calls.append((a, k)))
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda: SimpleNamespace(stop_server=lambda: None))
    monkeypatch.setattr("supervisor.state.load_state", lambda: {})
    monkeypatch.setattr("supervisor.state.save_state", lambda _state: None)
    monkeypatch.setattr("supervisor.evolution_lifecycle.complete_evolution_campaign", lambda *a, **k: {})
    monkeypatch.setattr("ouroboros.post_task_evolution.drop_pending_request", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    monkeypatch.setattr(server_control.os, "_exit", lambda code: (_ for _ in ()).throw(ExitCalled(code)))

    try:
        server_control.execute_panic_stop(
            consciousness=SimpleNamespace(stop=lambda: None),
            kill_workers_fn=lambda **kw: worker_calls.append(kw),
            data_dir=tmp_path,
            panic_exit_code=120,
            log=SimpleNamespace(critical=lambda *a, **k: None),
        )
    except ExitCalled:
        pass

    assert foreground_calls == [((tmp_path,), {"wait": False})]
    assert service_calls == [((tmp_path,), {"wait": False})]
    assert worker_calls == [{"force": True, "archive_service_logs": False}]

import threading
from types import SimpleNamespace


def _stop_restart_watcher(server):
    """Stop the restart watcher ``server.main()`` starts (an unnamed daemon polling
    ``_restart_requested`` on a 0.5 s ``time.sleep``). Its only stop seam is the flag it polls,
    so a test that returns from ``main()`` with the flag clear — or clears it before the next
    poll — leaves the poller running on the xdist worker for good (the ``sleeps`` polluter
    tests/test_delegate_hold.py pinned around). Raise the flag, join, then restore it."""
    server._restart_requested.set()
    for thread in threading.enumerate():
        if thread.name.endswith("(_check_restart)"):
            thread.join(timeout=5)
            assert not thread.is_alive(), "restart watcher did not stop"
    server._restart_requested.clear()


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


def test_main_normal_exit_does_not_run_emergency_cleanup(monkeypatch, tmp_path):
    import server

    cleanup_calls = []

    class FakeServer:
        def __init__(self, _config):
            self.should_exit = False

        def run(self, *, sockets):
            assert len(sockets) == 1 and sockets[0].getsockname()[1] > 0
            return None

    monkeypatch.setattr(server, "load_settings", lambda: {"OUROBOROS_SERVER_HOST": "127.0.0.1"})
    monkeypatch.setattr(server, "parse_server_args", lambda *_a, **_k: SimpleNamespace(host="127.0.0.1", port=0))
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "_ACTUAL_BOUND_PORT", None)
    monkeypatch.setattr(server, "get_network_auth_startup_warning", lambda _host: "")
    monkeypatch.setattr(server, "validate_network_auth_configuration", lambda _host: "")
    monkeypatch.setattr(server, "find_free_port", lambda _host, port: port)
    monkeypatch.setattr(server, "write_port_file", lambda *_a, **_k: None)
    monkeypatch.setattr(server.uvicorn, "Config", lambda *a, **k: object())
    monkeypatch.setattr(server.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(server, "_emergency_process_cleanup", lambda: cleanup_calls.append("cleanup"))
    monkeypatch.setattr(server, "_event_loop", None)  # the watcher's close_all_ws hop needs no loop here
    server._restart_requested.clear()

    try:
        assert server.main() == 0
    finally:
        _stop_restart_watcher(server)
    assert cleanup_calls == []


def test_main_graceful_restart_cleanup_avoids_port_sweep(monkeypatch, tmp_path):
    import server

    cleanup_calls = []

    class FakeServer:
        def __init__(self, _config):
            self.should_exit = False

        def run(self, *, sockets):
            assert len(sockets) == 1 and sockets[0].getsockname()[1] > 0
            server._restart_requested.set()
            return None

    class ExitCalled(RuntimeError):
        pass

    monkeypatch.setattr(server, "load_settings", lambda: {"OUROBOROS_SERVER_HOST": "127.0.0.1"})
    monkeypatch.setattr(server, "parse_server_args", lambda *_a, **_k: SimpleNamespace(host="127.0.0.1", port=0))
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "_ACTUAL_BOUND_PORT", None)
    monkeypatch.setattr(server, "get_network_auth_startup_warning", lambda _host: "")
    monkeypatch.setattr(server, "validate_network_auth_configuration", lambda _host: "")
    monkeypatch.setattr(server, "find_free_port", lambda _host, port: port)
    monkeypatch.setattr(server, "write_port_file", lambda *_a, **_k: None)
    monkeypatch.setattr(server.uvicorn, "Config", lambda *a, **k: object())
    monkeypatch.setattr(server.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", True)
    monkeypatch.setattr(server, "_emergency_process_cleanup", lambda **kw: cleanup_calls.append(kw))
    monkeypatch.setattr(server.os, "_exit", lambda code: (_ for _ in ()).throw(ExitCalled(code)))
    monkeypatch.setattr(server, "_event_loop", None)  # the watcher's close_all_ws hop needs no loop here
    server._restart_requested.clear()

    try:
        server.main()
    except ExitCalled:
        pass
    finally:
        _stop_restart_watcher(server)

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
    assert worker_calls == [{
        "force": True, "archive_service_logs": False,
        "reconcile_delegate_custody": False,
    }]


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


def _supervisor_harness(monkeypatch, tmp_path, steps):
    """Drive the REAL server._run_supervisor with every init/tick collaborator
    stubbed (no processes, ports, Manager or live data root). The scripted
    ``steps`` fire from the first call of each tick: ``ok`` = healthy tick,
    ``raise`` = a crash, ``stop``/``restart`` = set the flag (the loop exits at
    its next ``while`` check), ``raise_after_stop``/``raise_after_restart`` =
    the flag is set and the same tick then crashes (the shutdown race)."""
    import threading
    import queue as queue_mod

    import server
    import supervisor.events as events_mod
    import supervisor.message_bus as bus_mod
    import supervisor.queue as queue_pkg
    import supervisor.state as state_mod
    import supervisor.workers as workers_mod

    rec = _Recorder()
    rec.steps = list(steps)
    rec.restart = threading.Event()
    rec.ready = threading.Event()

    def _tick_head(_data_dir):
        step = rec.steps.pop(0) if rec.steps else "stop"
        if step in ("stop", "raise_after_stop"):
            rec.stop.set()
        if step in ("restart", "raise_after_restart"):
            rec.restart.set()
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

    import time as time_mod

    noop = lambda *_a, **_k: None  # noqa: E731
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    # Patch the module's bound name, not the process-wide time.sleep.
    monkeypatch.setattr(server, "time", SimpleNamespace(sleep=noop, monotonic=time_mod.monotonic, time=time_mod.time))
    monkeypatch.setattr(server, "_supervisor_stop", rec.stop)
    monkeypatch.setattr(server, "_restart_requested", rec.restart)
    monkeypatch.setattr(server, "_supervisor_ready", rec.ready)
    monkeypatch.setattr(server, "_supervisor_error", None)
    monkeypatch.setattr(server, "_supervisor_thread", None)
    monkeypatch.setattr(server, "_consciousness", None)
    monkeypatch.setattr(server, "_apply_settings_to_env", noop)
    monkeypatch.setattr(server, "ensure_legacy_imported", noop)
    monkeypatch.setattr(server, "_bootstrap_supervisor_repo", lambda _s: (True, "ok"))
    monkeypatch.setattr(server, "_runtime_branch_defaults", lambda: ("dev", "stable"))
    for name in (
        "_resume_interrupted_project_deletions", "_startup_prune_sweeps", "_startup_custody_sweep",
        "_startup_worktree_prune", "_prune_delegated_snapshots", "_periodic_supervisor_maintenance",
    ):
        monkeypatch.setattr(server, name, noop)
    monkeypatch.setattr(server, "_start_supervisor_liveness_watchdog",
                        lambda _liveness, stop_event=None: rec.watchdog_stops.append(stop_event))
    monkeypatch.setattr(server, "_process_bridge_updates", lambda _bridge, offset, _ctx: offset)
    monkeypatch.setattr(server, "_check_pending_restart_drain", lambda _ctx: True)
    monkeypatch.setattr(bus_mod, "init", noop)
    monkeypatch.setattr(bus_mod, "LocalChatBridge", _Bridge)
    monkeypatch.setattr(bus_mod, "send_with_budget", lambda chat_id, text: rec.alerts.append((chat_id, text)))
    monkeypatch.setattr("ouroboros.utils.set_log_sink", noop)
    monkeypatch.setattr(events_mod, "make_server_log_sink", lambda *_a, **_k: None)
    monkeypatch.setattr(events_mod, "dispatch_event", noop)
    monkeypatch.setattr(state_mod, "init", noop)
    monkeypatch.setattr(state_mod, "init_state", noop)
    monkeypatch.setattr(state_mod, "load_state", lambda: {"owner_chat_id": 7})
    for name in ("save_state", "update_state", "append_jsonl", "update_budget_from_usage",
                 "rotate_jsonl_log_if_needed"):
        monkeypatch.setattr(state_mod, name, noop)
    monkeypatch.setattr(state_mod, "rotate_chat_log_if_needed", _tick_head)
    for name in ("enqueue_task", "enforce_task_timeouts", "enqueue_evolution_task_if_needed",
                 "persist_queue_snapshot", "cancel_task_by_id", "queue_deep_self_review_task",
                 "sort_pending", "check_scheduled_tasks"):
        monkeypatch.setattr(queue_pkg, name, noop)
    monkeypatch.setattr(queue_pkg, "restore_pending_from_snapshot", lambda: 0)
    for name in ("init", "spawn_workers", "kill_workers", "assign_tasks", "ensure_workers_healthy",
                 "auto_resume_after_restart"):
        monkeypatch.setattr(workers_mod, name, noop)
    monkeypatch.setattr(workers_mod, "get_event_q", lambda: queue_mod.Queue())
    monkeypatch.setattr("ouroboros.delegate_recovery.pre_adopt_planned_handoffs", noop)
    monkeypatch.setattr("ouroboros.observability.prune_observability_blobs", lambda _root: {})
    monkeypatch.setattr("ouroboros.tools.services.prune_service_logs", lambda _root: {})
    monkeypatch.setattr("ouroboros.consciousness.BackgroundConsciousness", _Consciousness)
    return rec


def _run(rec, server):
    server._run_supervisor({})
    assert rec.ready.is_set() or server._supervisor_error  # init reached the loop
    return rec


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


def test_three_genuine_consecutive_crashes_still_die_visibly(monkeypatch, tmp_path):
    """Without a shutdown the contract stays: the third consecutive crash records
    the error, clears readiness, alerts the owner exactly once, stops the
    watchdog generation — and the backoff between crashes waits on the stop
    event (prompt shutdown), never on time.sleep."""
    import server

    rec = _supervisor_harness(monkeypatch, tmp_path, ["raise", "raise", "raise", "ok"])
    _run(rec, server)

    assert len(rec.alerts) == 1
    assert rec.alerts[0][0] == 7
    assert "died after repeated crashes" in rec.alerts[0][1]
    assert rec.ready.is_set() is False
    assert "3 consecutive crashes" in str(server._supervisor_error)
    assert rec.watchdog_stops[0].is_set()
    assert server._supervisor_thread is None
    assert rec.stop.waits == [2, 4]
    assert rec.steps == ["ok"]  # the loop is dead: the next tick never ran


def test_healthy_tick_between_crashes_resets_the_count(monkeypatch, tmp_path):
    import server

    rec = _supervisor_harness(monkeypatch, tmp_path, ["raise", "raise", "ok", "raise", "raise", "stop"])
    _run(rec, server)

    assert rec.alerts == []
    assert rec.ready.is_set() is True
    assert server._supervisor_error is None
    assert rec.steps == []
    assert rec.stop.waits == [2, 4, 2, 4]


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
        assert started == [server._run_supervisor]
    finally:
        server._supervisor_stop.clear()

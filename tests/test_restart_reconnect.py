"""Regression checks for restart/reconnect client behavior."""

import inspect
import os
import pathlib

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_server_enables_ws_ping_and_heartbeat():
    server_source = _read("server.py")
    helper_source = _read("ouroboros/server_runtime.py")
    assert "ws_heartbeat_loop" in server_source or "ws_heartbeat_loop" in helper_source
    assert '"type": "heartbeat"' in server_source or '"type": "heartbeat"' in helper_source
    assert "ws_ping_interval=20" in server_source
    assert "ws_ping_timeout=20" in server_source


def test_index_page_disables_cache():
    server_source = _read("server.py")
    helper_source = _read("ouroboros/server_web.py")
    assert "cache-control" in server_source.lower() or "cache-control" in helper_source.lower()


def test_find_free_port_waits_for_preferred_port():
    """find_free_port should retry the preferred port before falling back."""
    import socket
    import threading
    from ouroboros.server_entrypoint import find_free_port

    # Grab any available port for isolation
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        preferred = s.getsockname()[1]

    # Block the preferred port with a listener that releases after a short delay
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", preferred))
    blocker.listen(1)

    def _release():
        import time
        time.sleep(1.0)
        blocker.close()

    threading.Thread(target=_release, daemon=True).start()

    # With wait_retries=6 × 0.5s = 3s budget, it should get the preferred port
    result = find_free_port("127.0.0.1", preferred, wait_retries=6, wait_interval=0.5)
    assert result == preferred, f"Expected preferred port {preferred}, got {result}"


def test_find_free_port_falls_back_when_stuck():
    """find_free_port should fall back to a nearby port if preferred stays busy.

    On Windows, SO_REUSEADDR allows binding to a port already in LISTEN state,
    so the real-socket blocking approach does not reliably prevent rebind.
    We use monkeypatching to guarantee the preferred port stays unavailable.
    """
    from ouroboros import server_entrypoint

    preferred = 51234  # arbitrary high port unlikely to collide

    _original = getattr(server_entrypoint, "_can_bind_port", None)

    def _fake_can_bind(host: str, port: int) -> bool:
        if port == preferred:
            return False  # always blocked
        # For fallback ports, allow the first one
        return True

    server_entrypoint._can_bind_port = _fake_can_bind
    try:
        result = server_entrypoint.find_free_port(
            "127.0.0.1", preferred, max_tries=10,
            wait_retries=2, wait_interval=0.05,
        )
        assert result != preferred, f"Should have fallen back, but got preferred {preferred}"
    finally:
        if _original is not None:
            server_entrypoint._can_bind_port = _original


def test_find_free_port_retries_fallback_range_until_port_frees(monkeypatch):
    """Fallback scanning should keep retrying instead of returning the busy preferred port."""
    from ouroboros import server_entrypoint

    preferred = 41000
    attempts = {preferred: 0, preferred + 1: 0, preferred + 2: 0, preferred + 3: 0}

    def fake_can_bind(_host: str, port: int) -> bool:
        attempts[port] += 1
        if port == preferred:
            return False
        if port == preferred + 1:
            return attempts[port] >= 3
        return False

    monkeypatch.setattr(server_entrypoint, "_can_bind_port", fake_can_bind)

    result = server_entrypoint.find_free_port(
        "127.0.0.1",
        preferred,
        max_tries=4,
        wait_retries=3,
        wait_interval=0,
    )

    assert result == preferred + 1
    assert attempts[preferred] == 3
    assert attempts[preferred + 1] == 3


def test_find_free_port_raises_when_range_stays_busy(monkeypatch):
    """find_free_port should fail clearly instead of returning a known-busy port."""
    from ouroboros import server_entrypoint

    monkeypatch.setattr(server_entrypoint, "_can_bind_port", lambda _host, _port: False)

    try:
        server_entrypoint.find_free_port(
            "127.0.0.1",
            42000,
            max_tries=3,
            wait_retries=2,
            wait_interval=0,
        )
    except OSError as exc:
        assert "42000-42002" in str(exc)
    else:
        raise AssertionError("Expected find_free_port to raise when every candidate stays busy")


def test_task_done_live_summary_distinguishes_typed_failure():
    source = _read("web/modules/log_events.js")
    assert "export function taskOutcomeSeverity" in source
    assert "export function taskPresentation" in source
    assert "outcome_axes?.execution?.status" in source
    assert "['degraded', 'best_effort'].includes(execution)" in source
    assert "['degraded', 'best_effort'].includes(objective)" in source
    assert "const severity = taskOutcomeSeverity(evt);" in source
    # v6.82 (P5): the severity -> card-phase mapping is ONE shared reducer, so
    # live task_done, history task_summary rows and the terminal replay
    # fallback cannot drift (a cancelled root must never render as done).
    assert "export function taskTerminalPhase" in source
    assert "if (severity === 'cancelled') return 'cancelled';" in source
    assert "if (severity === 'error') return 'error';" in source
    assert "if (severity === 'warn') return 'warn';" in source
    assert source.count("taskTerminalPhase(evt)") >= 2
    assert "const terminal = taskDoneIsTerminal(evt);" in source
    assert "const presentation = taskPresentation(terminal || outcome === 'error' ? outcome : 'working'," in source
    assert "headline: presentation.headline" in source


def test_no_severity_keyed_visibility_writer_survives_restart_paths():
    """A warn/error/cancelled task keeps its block WITHOUT any writer forcing it.

    Card presence used to be a sticky flag written from two places for exactly
    this case: the live terminal frame (`summary.terminal && summary.phase ===
    'warn'`) and the history replay's `needsVisibleTerminal` branch. Two
    writers for one fact is how a cancelled root could come back from a reload
    as Done, so both are gone and the ONE predicate decides from the record's
    own facts. The predicate's clauses are pinned in the static contract
    fixture, and the BEHAVIOUR (a zero-tool failed turn keeps its block live
    and after a reload, a done one does not) in
    `web/tests/chat_activity_block.test.js`; this guards only that no
    severity-keyed visibility writer comes back on either restart path.
    """
    source = _read("web/modules/chat.js")
    assert "needsVisibleTerminal" not in source
    assert "summary.phase === 'warn'" not in source


def test_restart_watchdog_waits_for_uvicorn_exit():
    source = _read("server.py")
    assert "_uvicorn_exited = threading.Event()" in source
    assert "_uvicorn_exited.wait(timeout=force_exit_timeout_sec)" in source
    assert "_uvicorn_exited.set()" in source


def test_owner_restart_copy_is_explicit_about_stopped_task(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import server

    from ouroboros import server_restart

    source = _read("ouroboros/server_restart.py")
    assert "Stopping active task. New settings apply to the next message." in source
    assert "prepare_owner_restart" in source
    assert "owner_restart_no_resume" not in source
    # Checkout gate first (a refusal leaves the server intact), then the durable
    # no-resume intent, then the owned-work stop, then the owner's stop notice.
    owner_restart = _read("server.py").split('lowered.startswith("/restart"):', 1)[1].split(
        'elif lowered == "/review"', 1
    )[0]
    assert "_perform_owner_restart(ctx, reply)" in owner_restart
    flags = tmp_path / "state"
    calls = []
    ctx = SimpleNamespace(safe_restart=object())

    def checked(function, **kwargs):
        assert function is ctx.safe_restart and not flags.exists()
        assert kwargs == {"reason": "owner_restart", "unsynced_policy": "rescue_and_reset"}
        calls.append("checked")
        return True, "ok"

    def stopped(actual):
        assert actual is ctx
        from ouroboros.delegate_recovery import active_restart_transaction
        assert active_restart_transaction(tmp_path)["door"] == "owner_restart"
        assert not (flags / "panic_stop.flag").exists()
        calls.append("stopped")
        return ["active-task"]

    def notice(text, _suffix):
        assert calls == ["checked", "stopped"]
        assert text == "Stopping active task. New settings apply to the next message."
        calls.append("notice")

    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_restart, "_safe_restart_serialized", checked)
    monkeypatch.setattr(server_restart, "_stop_owned_work", stopped)
    monkeypatch.setattr(server_restart, "_request_restart_exit", lambda owner: calls.append(("exit", owner)))
    assert server._perform_owner_restart(ctx, notice) == (True, "")
    assert calls == ["checked", "stopped", "notice", ("exit", True)]
    stop = _read("ouroboros/server_restart.py").split("def _stop_owned_work", 1)[1]
    assert (stop.index("request_cancel(") < stop.index("ctx.kill_workers(")
            < stop.index("reconcile_orphaned_runs(") < stop.index("stop_outcome()"))


def test_owner_restart_cleanup_disables_second_custody_reconcile(monkeypatch):
    import server
    import threading

    owner, restart = threading.Event(), threading.Event()
    monkeypatch.setattr(server, "_owner_restart_requested", owner)
    monkeypatch.setattr(server, "_restart_requested", restart)
    monkeypatch.setattr(server, "_managed_update_pending_kwargs", lambda: {})
    monkeypatch.setattr(server, "_stop_owned_daemon_for_new_pin", lambda: None)
    monkeypatch.setattr(server, "stop_owned_work", lambda *a, **k: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    calls = []
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **kwargs: calls.append(kwargs))
    for owner_requested in (False, True):
        owner.set() if owner_requested else owner.clear()
        for restarting in (False, True):
            restart.set() if restarting else restart.clear()
            calls.clear()
            server._emergency_process_cleanup()
            expected = {"force": True, "archive_service_logs": False, "retain_saved_work": True}  # #1563
            if owner_requested:
                expected["reconcile_delegate_custody"] = False
            if restarting:
                status, reason = server._shutdown_task_cleanup_args(True)
                expected.update(terminal_status=status, result_reason=reason, stop_source="server_shutdown")
            assert calls == [expected]
    lifespan = inspect.getsource(server.lifespan)
    assert "**_restart_cleanup_kwargs()" in lifespan


def test_auto_resume_skips_owner_restart_no_resume_flag(tmp_path, monkeypatch):
    import supervisor.queue as queue
    import supervisor.workers as workers

    # The marker is consumed only after the queue holding its Restart holds is published (#1982),
    # so the queue snapshot belongs to this test's root too: the worker-shared root may hold an
    # earlier test's snapshot naming work this process never restored, which (correctly) keeps it.
    snapshot = tmp_path / "state" / "queue_snapshot.json"
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", snapshot)
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    flag = tmp_path / "state" / "owner_restart_no_resume.flag"
    flag.parent.mkdir(parents=True)
    flag.write_text("owner_restart", encoding="utf-8")
    compat_flag = tmp_path / "state" / "panic_stop.flag"
    compat_flag.write_text("owner_restart_no_resume", encoding="utf-8")

    def fail_load_state():
        raise AssertionError("owner restart flag should be checked before load_state")

    monkeypatch.setattr(workers, "load_state", fail_load_state)
    workers.auto_resume_after_restart()

    assert snapshot.is_file(), "the Restart holds were not published"
    assert not flag.exists()
    assert not compat_flag.exists()


def test_owner_restart_proceeds_when_worker_shutdown_fails(tmp_path, monkeypatch):
    """A worker shutdown that raises is a diagnostic, not a veto: the no-resume
    intent stays, the owner is told what the restart changes, and the process
    exits. Nothing is owned here, so the notice claims no stopped task: the
    owned-work branch is pinned in tests/test_manual_restart_execution.py."""
    import server
    import supervisor.message_bus as message_bus
    from ouroboros import config, server_restart

    messages = []
    exits = []

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{
                "update_id": offset,
                "message": {
                    "chat": {"id": 1},
                    "from": {"id": 1},
                    "text": "/restart",
                },
            }]

    class Ctx:
        consciousness = None
        RUNNING = {}

        def load_state(self):
            return {}

        def save_state(self, _state):
            return None

        def update_state(self, mutator):
            live = {}
            mutator(live)
            return live

        def send_with_budget(self, _chat_id, text, **kwargs):
            messages.append(text)

        def safe_restart(self, **_kwargs):
            return True, "OK"

        def kill_workers(self, force=True, **kwargs):
            assert kwargs["terminal_status"] == "cancelled"
            assert kwargs["result_reason"] == "Owner restart stopped this task before process restart."
            raise RuntimeError("shutdown failed")

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(server_restart, "_request_restart_exit", lambda owner=False: exits.append(owner))

    server._process_bridge_updates(Bridge(), 0, Ctx())

    from ouroboros.delegate_recovery import active_restart_transaction
    assert active_restart_transaction(tmp_path)["door"] == "owner_restart"
    assert not (tmp_path / "state" / "panic_stop.flag").exists()
    assert "New settings apply to the next message." in messages
    assert not any("Stopping active task" in text for text in messages)
    assert not any("cancelled" in text or "deferred" in text for text in messages)
    assert exits == [True]


def test_only_an_owner_restart_asks_for_the_runtime_mode_to_be_re_read(tmp_path, monkeypatch):
    """The owner raises the mode in Settings, presses Restart, and the mode never changes.

    ``restart_current_process`` re-execs with ``os.environ.copy()``, so the ratchet
    pin ``OUROBOROS_BOOT_RUNTIME_MODE`` rode into the replacement and the child
    re-pinned the OLD baseline from it — verified live: the process held
    ``OUROBOROS_RUNTIME_MODE=advanced`` under ``OUROBOROS_BOOT_RUNTIME_MODE=light``
    and ``/api/state`` kept reporting light, on that restart and every later one.

    One fact decides it, and the two directions are pinned here: the owner's
    ``/restart`` marks the restart owner-initiated, the agent's restart request
    (via the supervisor) does not.
    """
    import server
    import supervisor.message_bus as message_bus

    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{
                "update_id": offset,
                "message": {"chat": {"id": 1}, "from": {"id": 1}, "text": "/restart"},
            }]

    class Ctx:
        consciousness = None

        def load_state(self):
            return {}

        def save_state(self, _state):
            return None

        def update_state(self, mutator):
            live = {}
            mutator(live)
            return live

        def send_with_budget(self, _chat_id, _text, **kwargs):
            return None

        def safe_restart(self, **_kwargs):
            return True, "OK"

        def kill_workers(self, force=True, **_kwargs):
            return None

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)
    from ouroboros import server_restart
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    # The owned-work stop has its own suite; this pin is about the owner flag alone.
    monkeypatch.setattr(server_restart, "_stop_owned_work", lambda ctx: None)

    server._owner_restart_requested.clear()
    server._restart_requested.clear()
    try:
        server._process_bridge_updates(Bridge(), 0, Ctx())
        assert server._owner_restart_requested.is_set(), \
            "the owner's Restart button must mark the restart owner-initiated"

        # The agent's own restart request goes through the same exit signal and
        # must NOT claim owner intent: its baseline stays pinned.
        server._owner_restart_requested.clear()
        server._restart_requested.clear()
        server._request_restart_exit()
        assert server._restart_requested.is_set()
        assert not server._owner_restart_requested.is_set()
    finally:
        server._owner_restart_requested.clear()
        server._restart_requested.clear()


def _owner_restart_world(tmp_path, monkeypatch, stop):
    from types import SimpleNamespace

    import ouroboros.server_restart as server_restart
    from supervisor import queue

    exits, notices = [], []
    monkeypatch.setattr(queue, "RETURNS_NAMED", {})
    monkeypatch.setattr(server_restart, "_LAST_RETURNING", set())
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_restart, "_safe_restart_serialized", lambda *a, **k: (True, ""))
    monkeypatch.setattr(server_restart, "_stop_owned_work", stop)
    monkeypatch.setattr(server_restart, "_request_restart_exit", lambda owner=False: exits.append(owner))
    run = lambda: server_restart._perform_owner_restart(SimpleNamespace(safe_restart=None),
                                                        lambda text, _suffix: notices.append(text))
    return SimpleNamespace(run=run, exits=exits, notices=notices, queue=queue,
                           flag=tmp_path / "state" / "delegate_recovery_transactions" / "active.json")


def test_a_raised_owner_stop_still_restarts_so_its_durable_flags_stay_true(tmp_path, monkeypatch):
    """W05-2 (PROV_W05_POOL): a stop that raised left this generation running with automatic
    admission closed. Reopening it instead would keep the generation running under the durable
    no-resume flags just written, which review operations and budget pauses read as a stop and
    follow-ups as a restart in progress. The stop may not veto: the restart follows."""
    from supervisor import queue

    def broken(_ctx):
        queue.name_queue_returns([], "owner_restart")  # what the real stop names first
        raise OSError("owned-work stop raised")

    world = _owner_restart_world(tmp_path, monkeypatch, broken)
    assert world.run() == (True, "")
    assert world.exits == [True], "the owner's restart follows"
    from ouroboros.delegate_recovery import active_restart_transaction
    assert active_restart_transaction(world.flag.parents[2])["door"] == "owner_restart"
    assert world.notices == ["New settings apply to the next message."], "no stopped task is claimed"


def test_an_interrupted_owner_stop_reopens_admission_without_an_exit(tmp_path, monkeypatch):
    """A BaseException (an interrupt) propagates; if this generation survives it, it keeps admitting."""
    import pytest
    from supervisor import queue

    def interrupted(_ctx):
        queue.name_queue_returns([], "owner_restart")
        raise KeyboardInterrupt

    world = _owner_restart_world(tmp_path, monkeypatch, interrupted)
    with pytest.raises(KeyboardInterrupt):
        world.run()
    assert world.queue.RETURNS_NAMED == {} and world.exits == []

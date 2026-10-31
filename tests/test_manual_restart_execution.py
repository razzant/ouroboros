"""The owner's Restart: the checkout gate first, then a clean stop of this generation's
own work, then the re-exec — never deferred once the gate has passed."""

import contextlib
import inspect
import json
import logging
import os
import threading
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents, claudexor_daemon as owned, claudexor_runtime, config
from ouroboros import delegate_custody as custody, process_custody, server_restart
from ouroboros.post_task_checkpoint import POST_TASK_SYNTHESIS_INFLIGHT, POST_TASK_SYNTHESIS_LOCK
from supervisor import active_activity
from tests.test_claudexor_startup_lifetime import startup  # noqa: F401 - fixture reuse (real fake engine)

STOP_NOTICE = "Stopping active task. New settings apply to the next message."


def _rows(root):
    """Supervisor rows other than the store's own first-touch import receipt."""
    path = root / "logs" / "supervisor.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") != "current_store_import"]


def _gate(callback, **kwargs):
    """A passing update gate; the real one is pinned separately below."""
    return callback(**kwargs)


def _bridge():
    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{"update_id": offset,
                     "message": {"chat": {"id": 1}, "from": {"id": 1}, "text": "/restart"}}]

    return Bridge()


@pytest.fixture(autouse=True)
def _fresh_owner_stop_fact(monkeypatch):
    """Each Restart here begins its own stop; that process-local fact never reaches another test."""
    monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())


@pytest.fixture
def restart_root(tmp_path, monkeypatch):
    import server
    import supervisor.message_bus as message_bus

    for module in (config, server_restart, server):
        monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "log_chat", lambda *args, **kwargs: None)
    return tmp_path


@pytest.fixture
def owners(restart_root, monkeypatch):
    """Record every existing owner the stop reuses; the daemon is a real manager on the empty root."""
    calls = []
    manager = owned.OwnedClaudexorDaemon()
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: manager)
    monkeypatch.setattr(owned, "ensure_owned_gateway",
                        lambda **kw: pytest.fail("manual Restart must never ensure (start) the daemon"))
    monkeypatch.setattr(active_activity, "get_direct_activity_registry",
                        lambda: SimpleNamespace(snapshot=lambda: [{"activity_id": "direct-1"}]))
    monkeypatch.setattr(cancel_intents, "request_cancel",
                        lambda root, task_id, **kw: calls.append(("cancel", task_id, kw)) or {})
    monkeypatch.setattr(custody, "reconcile_orphaned_runs",
                        lambda root, running_task_ids=None, gateway_factory=None, recoverable_task_ids=None:
                        calls.append(("reconcile", running_task_ids, gateway_factory, recoverable_task_ids)) or [])
    real_outcome = manager.stop_outcome
    monkeypatch.setattr(manager, "stop_outcome", lambda: calls.append(("daemon_stop",)) or real_outcome())
    return SimpleNamespace(calls=calls, manager=manager)


def _ctx(calls, *, kill=None, checkout=None, running=None, messages=None):
    def default_kill(**kw):
        from ouroboros.delegate_recovery import active_restart_transaction
        flags = active_restart_transaction(server_restart.DATA_DIR).get("door") == "owner_restart"
        calls.append(("kill", kw, flags))
        return True

    return SimpleNamespace(
        RUNNING=running or {}, consciousness=None,
        load_state=lambda: {}, save_state=lambda _state: None, update_state=lambda mutator: mutator({}),
        send_with_budget=lambda _chat_id, text, **_kw: (messages if messages is not None else []).append(text),
        kill_workers=kill or default_kill,
        safe_restart=checkout or (lambda **kw: calls.append(("checkout", kw)) or (True, "OK: ouroboros")),
    )


def _restart(ctx, monkeypatch, *, gate=_gate):
    """Drive the real bridge handler; returns the owner-restart exits it requested."""
    import server

    exits = []
    monkeypatch.setattr(server_restart, "_safe_restart_serialized", gate)
    monkeypatch.setattr(server_restart, "_request_restart_exit", lambda owner=False: exits.append(owner))
    server._process_bridge_updates(_bridge(), 0, ctx)
    return exits


def test_restart_checks_out_first_then_stops_owned_work_in_order(owners, restart_root, monkeypatch, caplog):
    messages = []
    synthesis_key = (str(restart_root.resolve()), "synthesis-1")
    ctx = _ctx(owners.calls, running={"task-a": {"task": {"id": "task-a"}}}, messages=messages)
    with POST_TASK_SYNTHESIS_LOCK:
        POST_TASK_SYNTHESIS_INFLIGHT[synthesis_key] = None
    try:
        with caplog.at_level(logging.WARNING):
            exits = _restart(ctx, monkeypatch)
    finally:
        with POST_TASK_SYNTHESIS_LOCK:
            POST_TASK_SYNTHESIS_INFLIGHT.pop(synthesis_key, None)
    assert exits == [True]
    assert [c[0] for c in owners.calls] == [
        "checkout", "cancel", "cancel", "cancel", "kill", "reconcile", "daemon_stop",
    ]
    checkout, *cancels, kill, reconcile, _ = owners.calls
    assert checkout[1] == {"reason": "owner_restart", "unsynced_policy": "rescue_and_reset"}
    assert [c[1] for c in cancels] == ["task-a", "direct-1", "synthesis-1"]
    assert all(c[2]["source"] == "owner_restart" and c[2]["requested_stop_policy"] == "immediate"
               and c[2]["allow_settled_target"] is True for c in cancels)
    assert kill[1]["force"] is True and kill[1]["terminal_status"] == "cancelled"
    assert kill[1]["reconcile_delegate_custody"] is False, "kill-path reconcile would ensure (start) the daemon"
    assert kill[2] is True, "the durable no-resume intent precedes the stop"
    assert reconcile == ("reconcile", set(), owned.read_owned_gateway, None)
    from ouroboros.delegate_recovery import active_restart_transaction
    assert active_restart_transaction(restart_root)["door"] == "owner_restart"
    assert not (restart_root / "state" / "panic_stop.flag").exists()
    assert messages == ["♻️ Restarting.", STOP_NOTICE]
    # Nothing to stop on the empty root: the daemon stop is quiet, not a diagnostic.
    assert _rows(restart_root) == [] and not [r for r in caplog.records if r.levelno >= logging.CRITICAL]


def test_restart_pauses_admission_before_it_names_returns(owners, restart_root, monkeypatch):
    """The stop's first step: a launch or Resume already writing its marks hands over to the still-running
    pool (``pause_admission``) before the returns are named, the cancels recorded and the pool retired."""
    from supervisor import workspace_preparation

    from ouroboros.delegate_recovery import _active_restart_transaction_path
    flag = _active_restart_transaction_path(restart_root)
    monkeypatch.setattr(workspace_preparation, "pause_admission",
                        lambda timeout=None: owners.calls.append(("pause", flag.exists())) or True)
    real_returns = server_restart._prepare_owner_restart_returns
    monkeypatch.setattr(server_restart, "_prepare_owner_restart_returns",
                        lambda ctx: owners.calls.append(("returns",)) or real_returns(ctx))
    assert _restart(_ctx(owners.calls), monkeypatch) == [True]
    order = [c[0] for c in owners.calls]
    assert order.index("checkout") < order.index("pause") < order.index("returns") < order.index("kill")
    # Before the Restart's own follow-up restriction exists, which would refuse a committed launch's put.
    assert ("pause", False) in owners.calls and order.count("pause") == 1


def test_a_restart_whose_flag_cannot_be_written_reopens_admission(owners, restart_root, monkeypatch):
    from supervisor import workspace_preparation

    events = []
    monkeypatch.setattr(workspace_preparation, "pause_admission", lambda timeout=None: events.append("pause") or True)
    monkeypatch.setattr(workspace_preparation, "resume_admission", lambda: events.append("resume"))
    monkeypatch.setattr("supervisor.queue_schedules.schedule_transaction",
                        lambda _root: (_ for _ in ()).throw(OSError("read-only state")))
    assert _restart(_ctx(owners.calls), monkeypatch) == []
    assert events == ["pause", "resume"] and "kill" not in [c[0] for c in owners.calls]


def test_assisted_update_refusal_still_refuses_before_any_stop(owners, restart_root, monkeypatch):
    from supervisor import update_merge

    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: object())
    monkeypatch.setattr(update_merge, "release_update_lock", lambda _lock: None)
    monkeypatch.setattr(update_merge, "read_update_tx_strict",
                        lambda: ("valid", {"phase": "assisted_resolution", "task_id": "resolver"}))
    messages = []
    ctx = _ctx(owners.calls, messages=messages,
               checkout=lambda **kw: pytest.fail("the gate refuses before the checkout"),
               kill=lambda **kw: pytest.fail("refused before any stop"))
    assert _restart(ctx, monkeypatch, gate=server_restart._safe_restart_serialized) == []
    assert messages == ["♻️ Restarting.",
                        "⚠️ Restart cancelled: Managed update merge is still being resolved; restart was deferred."]
    assert owners.calls == [] and not (restart_root / "state").exists()


def test_checkout_refusal_leaves_the_server_intact(owners, restart_root, monkeypatch):
    messages = []
    ctx = _ctx(owners.calls, messages=messages,
               checkout=lambda **kw: owners.calls.append(("checkout", kw)) or (False, "Failed checkout: fixture"),
               kill=lambda **kw: pytest.fail("a refused checkout stops nothing"))
    assert _restart(ctx, monkeypatch) == []
    assert messages == ["♻️ Restarting.", "⚠️ Restart cancelled: Failed checkout: fixture"]
    assert [c[0] for c in owners.calls] == ["checkout"] and not (restart_root / "state").exists()


@pytest.mark.parametrize("kill", [
    lambda **kw: False,
    lambda **kw: (_ for _ in ()).throw(RuntimeError("shutdown failed")),
])
def test_unconfirmed_worker_shutdown_is_a_diagnostic_and_the_restart_proceeds(
        owners, restart_root, monkeypatch, caplog, kill):
    messages = []
    with caplog.at_level(logging.CRITICAL):
        assert _restart(_ctx(owners.calls, kill=kill, messages=messages), monkeypatch) == [True]
    assert "worker shutdown" in caplog.text.lower()
    assert [c[0] for c in owners.calls][-2:] == ["reconcile", "daemon_stop"]
    from ouroboros.delegate_recovery import active_restart_transaction
    assert active_restart_transaction(restart_root)["door"] == "owner_restart"
    assert messages == ["♻️ Restarting.", STOP_NOTICE]


def test_unconfirmed_daemon_stop_is_reported_and_the_restart_proceeds(owners, restart_root, monkeypatch, caplog):
    owned._write_ownership_marker()
    # An authenticated endpoint with no ledgered root and no self-started child: stop cannot confirm.
    monkeypatch.setattr(owners.manager, "_classify_liveness", lambda **kw: (object(), "running", ""))
    messages = []
    with caplog.at_level(logging.CRITICAL):
        assert _restart(_ctx(owners.calls, messages=messages), monkeypatch) == [True]
    assert "stop unconfirmed" in caplog.text and "custody retained" in caplog.text
    assert "next generation attaches" in caplog.text
    rows = _rows(restart_root)
    assert rows[-1]["type"] == "process_stop_unconfirmed" and rows[-1]["purpose"] == owned.CUSTODY_PURPOSE
    assert owners.calls[-1] == ("daemon_stop",)
    assert messages == ["♻️ Restarting.", STOP_NOTICE], "one honest message; never 'deferred'"


def test_restart_teardown_still_stops_a_newer_engine_after_an_unconfirmed_stop(owners, restart_root, monkeypatch):
    """Planned doors keep a newer engine on a lower pin; the owner's Restart, whose own stop could not
    confirm, still gets the teardown's attested stop (the fact its own stop sets tells the two apart)."""
    owned._write_ownership_marker()
    owned.owned_descriptor_path().parent.mkdir(parents=True, exist_ok=True)
    owned.owned_descriptor_path().write_text("{}", encoding="utf-8")
    monkeypatch.setattr(owners.manager, "_classify_liveness", lambda **kw: (object(), "running", ""))
    assert not server_restart._OWNER_STOP_BEGAN.is_set()
    assert _restart(_ctx(owners.calls), monkeypatch) == [True]
    assert server_restart._OWNER_STOP_BEGAN.is_set()
    monkeypatch.setattr(claudexor_runtime, "load_runtime_pin",
                        lambda path=None: SimpleNamespace(version="4.3.0", build_sha="a" * 40))
    monkeypatch.setattr(owned, "read_owned_gateway", lambda: contextlib.nullcontext(
        SimpleNamespace(engine_version="4.3.1", engine_build_sha="b" * 40)))
    server_restart._stop_owned_daemon_for_new_pin()  # the lifespan teardown's check
    assert [c[0] for c in owners.calls].count("daemon_stop") == 2


def test_an_abandoned_restart_does_not_keep_its_stop_fact(owners, restart_root, monkeypatch):
    """A BaseException inside the stop returns the queue and keeps this generation admitting: a later
    planned door must then keep a newer engine again."""
    reopened = []
    monkeypatch.setattr("supervisor.queue.reopen_queue_returns", lambda door: reopened.append(door))
    monkeypatch.setattr(server_restart, "_safe_restart_serialized", _gate)

    def interrupted(**kw):
        assert server_restart._OWNER_STOP_BEGAN.is_set()
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        server_restart._perform_owner_restart(_ctx(owners.calls, kill=interrupted))
    assert reopened == ["owner_restart"] and not server_restart._OWNER_STOP_BEGAN.is_set()


def test_raising_daemon_stop_is_recorded_like_panic_and_the_restart_proceeds(owners, restart_root, monkeypatch):
    def fail():
        raise RuntimeError("fixture stop failed")

    monkeypatch.setattr(owners.manager, "stop_outcome", fail)
    assert _restart(_ctx(owners.calls), monkeypatch) == [True]
    row = _rows(restart_root)[-1]
    assert row == {"ts": row["ts"], "type": "process_stop_unconfirmed",
                   "purpose": owned.CUSTODY_PURPOSE, "reason": "stop raised RuntimeError"}


def test_the_restart_notice_claims_a_stopped_task_only_when_one_was_owned(
        owners, restart_root, monkeypatch):
    """The owner is told what actually happened. With nothing owned the restart
    sends the settings sentence alone; the stop sentence above is what an owned
    live task earns. The stop sequence itself is unchanged in both branches."""
    monkeypatch.setattr(active_activity, "get_direct_activity_registry",
                        lambda: SimpleNamespace(snapshot=lambda: []))
    messages = []
    ctx = _ctx(owners.calls, messages=messages)

    assert _restart(ctx, monkeypatch) == [True]

    assert messages == ["♻️ Restarting.", "New settings apply to the next message."]
    assert [c[0] for c in owners.calls] == ["checkout", "kill", "reconcile", "daemon_stop"]


def test_stop_outcome_types_the_two_non_stops(restart_root, monkeypatch, caplog):
    """Nothing to stop is quiet; an unconfirmed remainder is disclosed by the stop itself."""
    manager = owned.OwnedClaudexorDaemon()
    assert manager.stop_outcome() == "nothing_to_stop" and manager.stop() is False
    assert not caplog.records and _rows(restart_root) == []
    owned._write_ownership_marker()
    monkeypatch.setattr(manager, "_classify_liveness", lambda **kw: (object(), "running", ""))
    with caplog.at_level(logging.CRITICAL):
        assert manager.stop_outcome() == "unconfirmed" and manager.stop() is False
    assert [row["type"] for row in _rows(restart_root)] == ["process_stop_unconfirmed"] * 2


def _join_warmup_thread(timeout=5):
    for thread in threading.enumerate():
        if thread.name == "owned-daemon-warmup":
            thread.join(timeout)
            assert not thread.is_alive(), "warmup thread did not finish"


def test_warmup_is_one_background_ensure_for_a_provisioned_home_only(restart_root, monkeypatch, caplog):
    import server

    ensures = []

    class Gateway:
        def close(self):
            ensures.append("closed")

    monkeypatch.setattr(owned, "ensure_owned_gateway", lambda **kw: ensures.append(("ensure", kw)) or Gateway())
    assert owned.warm_owned_daemon() is False and ensures == []
    descriptor = owned.owned_descriptor_path()
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text("{}")
    assert owned.warm_owned_daemon() is True
    _join_warmup_thread()
    assert ensures == [("ensure", {"admission_wait_sec": 40.0}), "closed"]

    def unreachable(**kw):
        raise RuntimeError("fixture daemon down")

    monkeypatch.setattr(owned, "ensure_owned_gateway", unreachable)
    with caplog.at_level(logging.INFO):
        assert owned.warm_owned_daemon() is True
        _join_warmup_thread()
    assert "warmup did not reach readiness" in caplog.text
    # The boot coordinator's services step makes exactly this call, guarded like every other
    # real-data side effect: the lifespan passes ``warm_daemon=live`` (``not pytest_default_real_data_dir``).
    from ouroboros import server_boot

    services = inspect.getsource(server_boot.start_background_services)
    assert services.index("if warm_daemon:") < services.index("warm_owned_daemon()")
    lifespan = inspect.getsource(server.lifespan)
    assert "live = not pytest_default_real_data_dir" in lifespan and "warm_daemon=live" in lifespan


@pytest.mark.serial
@pytest.mark.skipif(os.name == "nt", reason="POSIX measured process custody")
def test_unconfirmed_daemon_stop_leaves_one_daemon_for_the_next_generation(startup, monkeypatch):  # noqa: F811
    """After an unconfirmed stop the re-exec'd generation attaches to the live daemon; no second spawn."""
    (startup.home / "normal").touch()
    (startup.home / "publish").touch()
    manager = owned.OwnedClaudexorDaemon()
    endpoint = manager.ensure_running(startup_wait_sec=3)
    process = manager._proc
    monkeypatch.setattr(server_restart, "DATA_DIR", startup.root)
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: manager)
    monkeypatch.setattr(owned, "ensure_owned_gateway",
                        lambda **kw: pytest.fail("manual Restart must never ensure (start) the daemon"))
    ctx = SimpleNamespace(RUNNING={}, kill_workers=lambda **kw: True)
    with monkeypatch.context() as patched:
        # Controlled: the ledger root and the self-started child both stay unconfirmed, the daemon alive.
        patched.setattr(manager, "_terminate_child", lambda: False)
        patched.setattr(process_custody, "stop_ledgered_processes",
                        lambda root, purposes, unconfirmed=None, **kw: (
                            unconfirmed.append("controlled unconfirmed root") if unconfirmed is not None else None
                        ) or [])
        server_restart._stop_owned_work(ctx)
    assert process.poll() is None
    assert _rows(startup.root)[-1]["type"] == "process_stop_unconfirmed"
    successor = owned.OwnedClaudexorDaemon()  # the next server generation's manager
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(
        pin=SimpleNamespace(version="9.9.9", build_sha="c" * 40),
        ensure=lambda: pytest.fail("the next generation must attach to the live daemon, not spawn")))
    assert successor.ensure_running(startup_wait_sec=3).port == endpoint.port
    assert len((startup.home / "spawned.jsonl").read_text().splitlines()) == 1
    assert successor.stop() is True
    process.wait(timeout=5)

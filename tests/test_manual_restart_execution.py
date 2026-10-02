"""The owner's Restart: the checkout gate first, then a clean stop of this generation's
own work, then the re-exec — never deferred once the gate has passed."""

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

STOP_NOTICE = "Stopping active task. Restart confirmed. New settings apply to the next message."


def _rows(root):
    path = root / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _gate(callback, **kwargs):
    """A passing update gate; the real one is pinned separately below."""
    return callback(**kwargs)


def _bridge():
    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{"update_id": offset,
                     "message": {"chat": {"id": 1}, "from": {"id": 1}, "text": "/restart"}}]

    return Bridge()


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
        flags = (server_restart.DATA_DIR / "state" / "owner_restart_no_resume.flag").exists()
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
    assert (restart_root / "state" / "owner_restart_no_resume.flag").read_text() == "owner_restart"
    assert (restart_root / "state" / "panic_stop.flag").read_text() == "owner_restart_no_resume"
    assert messages == ["♻️ Restarting.", STOP_NOTICE]
    # Nothing to stop on the empty root: the daemon stop is quiet, not a diagnostic.
    assert _rows(restart_root) == [] and not [r for r in caplog.records if r.levelno >= logging.CRITICAL]


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
    assert (restart_root / "state" / "owner_restart_no_resume.flag").exists()
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

    assert messages == ["♻️ Restarting.", "Restart confirmed. New settings apply to the next message."]
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
    # The lifespan makes exactly this call, guarded like every other real-data side effect.
    lifespan = inspect.getsource(server.lifespan)
    assert "warm_owned_daemon()" in lifespan
    assert lifespan.index("if not pytest_default_real_data_dir:") < lifespan.index("warm_owned_daemon()")


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

"""A planned restart ends the owned daemon only when the landed checkout pins another engine.

Self-restart after self-modification and the managed update (or its rollback) share
one seam, the lifespan teardown of a requested restart; the comparison reads the pin
the next generation selects and the serving engine's own handshake, and the stop is
the manual Restart's attested stop, never a veto.
"""

import inspect
import json
import logging
import os
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_daemon as owned, claudexor_runtime, config, server_restart
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests.test_claudexor_startup_lifetime import startup  # noqa: F401 - fixture reuse (real fake engine)

PIN = SimpleNamespace(version="3.10.0", build_sha="a" * 40)


def _rows(root):
    path = root / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class _Gateway:
    """What ``read_owned_gateway`` hands back: a handshaken client that must be closed."""

    def __init__(self, version, sha):
        self.engine_version, self.engine_build_sha, self.closed = version, sha, False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.closed = True


@pytest.fixture
def planned(tmp_path, monkeypatch):
    """A provisioned home, the pin the checkout landed, and a recording real manager."""
    for module in (config, server_restart):
        monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    descriptor = owned.owned_descriptor_path()
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text("{}")
    calls = []
    manager = owned.OwnedClaudexorDaemon()
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: manager)
    monkeypatch.setattr(owned, "ensure_owned_gateway",
                        lambda **kw: pytest.fail("a restart teardown must never ensure (start) the daemon"))
    monkeypatch.setattr(claudexor_runtime, "load_runtime_pin", lambda path=None: PIN)
    real_outcome = manager.stop_outcome
    monkeypatch.setattr(manager, "stop_outcome", lambda: calls.append("daemon_stop") or real_outcome())

    def serving(version, sha):
        gateway = _Gateway(version, sha)
        monkeypatch.setattr(owned, "read_owned_gateway", lambda: gateway)
        return gateway

    return SimpleNamespace(calls=calls, manager=manager, root=tmp_path, serving=serving)


def test_off_pin_daemon_is_stopped_and_the_restart_proceeds(planned, monkeypatch, caplog):
    gateway = planned.serving("3.9.8", "b" * 40)
    monkeypatch.setattr(planned.manager, "stop_outcome", lambda: planned.calls.append("daemon_stop") or "stopped")
    with caplog.at_level(logging.INFO):
        assert server_restart._stop_owned_daemon_for_new_pin() is None
    assert planned.calls == ["daemon_stop"] and gateway.closed
    assert "3.9.8" in caplog.text and "3.10.0" in caplog.text and "pinned engine" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.CRITICAL] and _rows(planned.root) == []


@pytest.mark.parametrize("serving", [("3.10.0", "a" * 40)], ids=["same version and build"])
def test_on_pin_daemon_is_left_serving(planned, caplog, serving):
    gateway = planned.serving(*serving)
    with caplog.at_level(logging.INFO):
        server_restart._stop_owned_daemon_for_new_pin()
    assert planned.calls == [] and gateway.closed
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("serving", [("3.10.0", "b" * 40), ("3.9.8", "a" * 40)],
                         ids=["build differs", "version differs"])
def test_either_pin_field_differing_counts(planned, monkeypatch, serving):
    planned.serving(*serving)
    monkeypatch.setattr(planned.manager, "stop_outcome", lambda: planned.calls.append("daemon_stop") or "stopped")
    server_restart._stop_owned_daemon_for_new_pin()
    assert planned.calls == ["daemon_stop"]


def test_unconfirmed_stop_is_the_same_diagnostic_and_the_restart_proceeds(planned, caplog):
    """The real manager on a home with a descriptor and no marker cannot confirm; nothing vetoes."""
    planned.serving("3.9.8", "b" * 40)
    with caplog.at_level(logging.CRITICAL):
        assert server_restart._stop_owned_daemon_for_new_pin() is None
    assert planned.calls == ["daemon_stop"]
    assert "Planned restart: owned Claudexor stop unconfirmed" in caplog.text
    assert "custody retained" in caplog.text and "next generation attaches" in caplog.text
    rows = _rows(planned.root)
    assert rows[-1]["type"] == "process_stop_unconfirmed" and rows[-1]["purpose"] == owned.CUSTODY_PURPOSE


def test_raising_stop_is_recorded_like_panic_and_the_restart_proceeds(planned, monkeypatch, caplog):
    planned.serving("3.9.8", "b" * 40)

    def fail():
        raise RuntimeError("fixture stop failed")

    monkeypatch.setattr(planned.manager, "stop_outcome", fail)
    with caplog.at_level(logging.CRITICAL):
        assert server_restart._stop_owned_daemon_for_new_pin() is None
    assert "Planned restart: owned Claudexor stop raised RuntimeError" in caplog.text
    row = _rows(planned.root)[-1]
    assert row == {"ts": row["ts"], "type": "process_stop_unconfirmed",
                   "purpose": owned.CUSTODY_PURPOSE, "reason": "stop raised RuntimeError"}


def test_unreachable_foreign_or_stopped_daemon_changes_nothing(planned, monkeypatch, caplog):
    """Discovery/handshake refusals (typed) keep the existing handoff: the next generation attaches."""
    def refused():
        raise ClaudexorUnavailable("daemon_unreachable", "fixture: nothing answers", status_code=503)

    monkeypatch.setattr(owned, "read_owned_gateway", refused)
    with caplog.at_level(logging.INFO):
        server_restart._stop_owned_daemon_for_new_pin()
    assert planned.calls == []
    assert "keeps the owned Claudexor daemon" in caplog.text and "nothing answers" in caplog.text


@pytest.mark.parametrize("pin", [
    lambda path=None: None,
    lambda path=None: (_ for _ in ()).throw(
        claudexor_runtime.ClaudexorRuntimeError("runtime_pin_missing", "fixture: no pin")),
], ids=["unpublished pin", "unreadable pin"])
def test_unpublished_or_unreadable_pin_changes_nothing(planned, monkeypatch, pin):
    monkeypatch.setattr(claudexor_runtime, "load_runtime_pin", pin)
    monkeypatch.setattr(owned, "read_owned_gateway",
                        lambda: pytest.fail("no pin to compare against: the daemon is not even read"))
    server_restart._stop_owned_daemon_for_new_pin()
    assert planned.calls == []


def test_unprovisioned_home_is_never_read(tmp_path, monkeypatch):
    for module in (config, server_restart):
        monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(claudexor_runtime, "load_runtime_pin",
                        lambda path=None: pytest.fail("an unprovisioned home reads no pin"))
    monkeypatch.setattr(owned, "read_owned_gateway", lambda: pytest.fail("an unprovisioned home is never read"))
    server_restart._stop_owned_daemon_for_new_pin()


def test_pin_is_read_from_the_landed_checkout_not_the_cached_manager(planned, monkeypatch):
    """The process's runtime manager still holds the pin it booted with; the checkout decides."""
    planned.serving("3.9.8", "b" * 40)
    monkeypatch.setattr(planned.manager, "stop_outcome", lambda: planned.calls.append("daemon_stop") or "stopped")
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager",
                        lambda: pytest.fail("the cached manager pin is this process's, not the next generation's"))
    server_restart._stop_owned_daemon_for_new_pin()
    assert planned.calls == ["daemon_stop"]


def test_the_check_lives_in_the_restart_teardown_and_the_handoff_performer_is_untouched(planned, monkeypatch, tmp_path):
    """Source pin in the file's style: the teardown makes the check after the worker kill and before
    the bridge goes down, only for a requested restart; ``_perform_supervisor_restart`` keeps its
    handoff unchanged (an on-pin daemon is never consulted there)."""
    import server

    source = inspect.getsource(server.lifespan)
    kill = source.index("kill_workers(\n                force=True,\n                terminal_status=cleanup_status")
    check = source.index("_stop_owned_daemon_for_new_pin()")
    assert kill < check < source.index("get_bridge().shutdown()")
    assert check < source.index("if extension_reconcile_task is not None:")
    assert check < source.index("stop_owned_work(lifespan_drive_root)")
    assert source.rindex("if _restart_requested.is_set():", 0, check) > kill
    assert "_stop_owned_daemon_for_new_pin" not in inspect.getsource(server._perform_supervisor_restart)

    planned.serving("3.10.0", "a" * 40)
    monkeypatch.setattr(owned, "read_owned_gateway", lambda: pytest.fail("the performer never reads the daemon"))
    worker_calls = []
    state = {"owner_chat_id": 0}
    ctx = SimpleNamespace(
        load_state=lambda: dict(state), save_state=lambda updated: state.update(updated),
        safe_restart=lambda **_kwargs: (True, "ok"),
        kill_workers=lambda **kwargs: worker_calls.append(kwargs),
        persist_queue_snapshot=lambda **_kwargs: None, DRIVE_ROOT=tmp_path, REPO_DIR=tmp_path, RUNNING={},
    )
    def preserve_delegated(root, running, restart_transaction_id="", additional_task_ids=None):
        assert not additional_task_ids  # This fixture has no native owner-wait participants.
        return {"sleeping-parent"}

    monkeypatch.setattr("ouroboros.delegate_recovery.prepare_planned_restart_handoffs", preserve_delegated)
    monkeypatch.setattr(server, "_request_restart_exit", lambda: None)
    server._perform_supervisor_restart(ctx)
    assert worker_calls[0]["preserve_running_task_ids"] == {"sleeping-parent"}
    assert worker_calls[0]["preserve_pending"] is True
    assert planned.calls == []


@pytest.mark.serial
@pytest.mark.skipif(os.name == "nt", reason="POSIX measured process custody")
def test_real_engine_ends_only_when_the_checkout_pins_another_engine(startup, monkeypatch):  # noqa: F811
    """On the real fake engine (9.9.9 / c*40) the same generation keeps it while the pin matches
    and ends it, confirmed, when the landed checkout pins another engine."""
    (startup.home / "normal").touch()
    (startup.home / "publish").touch()
    manager = owned.OwnedClaudexorDaemon()
    manager.ensure_running(startup_wait_sec=3)
    process = manager._proc
    monkeypatch.setattr(server_restart, "DATA_DIR", startup.root)
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: manager)
    monkeypatch.setattr(owned, "ensure_owned_gateway",
                        lambda **kw: pytest.fail("a restart teardown must never ensure (start) the daemon"))

    monkeypatch.setattr(claudexor_runtime, "load_runtime_pin",
                        lambda path=None: SimpleNamespace(version="9.9.9", build_sha="c" * 40))
    server_restart._stop_owned_daemon_for_new_pin()
    assert process.poll() is None, "an unchanged pin leaves the daemon serving"

    monkeypatch.setattr(claudexor_runtime, "load_runtime_pin",
                        lambda path=None: SimpleNamespace(version="9.10.0", build_sha="d" * 40))
    server_restart._stop_owned_daemon_for_new_pin()
    process.wait(timeout=5)
    assert not [row for row in _rows(startup.root) if row["type"] == "process_stop_unconfirmed"]
    assert manager.stop_outcome() == "nothing_to_stop"

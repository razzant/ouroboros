"""Emergency Stop's remote gate (``delegate_recovery.remote_engines_named``) after the owner removes the last remote
engine connection: work already recorded on that engine keeps its owed stop (a run, a start whose answer is unknown, an
open model operation), on its recorded engine only, never retargeted or run locally instead, and stays in view; an
installation with neither a connection nor recorded remote work stays on the cheap path, and one that never reached a
remote engine reads nothing. No real engine, SSH, model or Panic exit.
"""

from __future__ import annotations

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import remote_stops
from ouroboros.delegate_recovery import remote_engines_named
from ouroboros.server_control import _remote_panic_step

_REF = {"kind": "remote", "connection": "removed", "ssh_host": "unreachable.invalid", "engine_home": "/engine",
        "store": "s", "remote_root": "/copies"}


@pytest.fixture
def local_only(monkeypatch, tmp_path):
    """An installation that names no connection and never reached a remote engine, over this test's own data root
    (the parent's reproduction used ``config.DATA_DIR``; a shared root would carry one test's remote custody into
    the next)."""
    monkeypatch.setenv("OUROBOROS_ENGINE_CONNECTIONS", "[]")
    root = tmp_path / "data"
    root.mkdir()
    return root


@pytest.fixture
def removed(local_only):
    """The owner's normal Settings removal of the last connection (the raw setting names none) on an installation
    that reached it: its tunnel home stays (``remote_engines`` never removes it), as on every root remote work was
    recorded at."""
    local_only.joinpath(*remote_stops.TUNNEL_HOME, "removed").mkdir(parents=True)
    return local_only


def test_a_removed_last_connection_keeps_its_recorded_runs_owed_stop(removed):
    """The parent's reproduction: an accepted remote run, then the last connection removed; Panic still records the
    stop it owes, on the run's own recorded engine."""
    row = custody.RunCustody(run_id="run-existing", task_id="test-owner", invocation_id="inv-existing",
                             route_id="codex", model="m", access="full", mode="agent", ledger_root=str(removed),
                             engine_ref=dict(_REF), daemon_store_id="s")
    assert custody.record_started(removed, row)
    assert remote_engines_named(removed) and not remote_engines_named()  # custody decides once no row names one
    _remote_panic_step(removed)
    actual = current.run(removed, "run-existing")
    assert actual.containment_fault.get("reason") == remote_stops.PANIC_STOP_UNCONFIRMED
    assert actual.engine_ref == _REF and not actual.settled  # never retargeted, never read as stopped


def test_a_sent_start_whose_answer_is_unknown_keeps_its_owed_stop(removed):
    """A start that reached the POST and has no run id: Panic marks it owed on its recorded engine; nothing re-sends
    it and its intent survives the removal."""
    assert custody.record_start_requested(
        removed, run_id="", task_id="test-owner", invocation_id="inv-sent", idempotency_key="inv-sent",
        request={"prompt": "p", "scope": {"kind": "project", "root": "/x"}}, route="codex", engine_ref=dict(_REF))
    assert custody.emit(removed, custody.REMOTE_START_PHASE, {"run_id": "", "invocation_id": "inv-sent",
                                                                "phase": "posting", "connection": "removed"})
    assert remote_engines_named(removed)
    _remote_panic_step(removed)
    [record] = [row for row in custody.pending_invocations(removed) if row["invocation_id"] == "inv-sent"]
    assert remote_stops.owed(record) and record["engine_ref"] == _REF


def test_an_open_remote_model_operation_keeps_the_gate_open(removed):
    """An open model operation recorded on the removed engine holds the gate open too, so the boot after a Panic still
    asks its own engine to cancel it (here unreachable: nothing is sent elsewhere)."""
    from ouroboros import usage_accounting as usage
    from ouroboros.observability import persist_call

    reservation = usage.reserve_attempt(usage.AttemptRequest(
        model="claudexor@removed::codex=m", provider="claudexor", drive_root=removed, task_id="t-op",
        root_task_id="t-op", parent_task_id="", reservation_usd=1.0, global_limit_usd=100.0))
    usage.mark_dispatched(reservation)
    persist_call(removed, task_id="t-op", call_id=f"{reservation.attempt_id}_model_request",
                 call_type="llm_claudexor_request", payload={}, keep_raw=True,
                 manifest={"invocation_id": reservation.attempt_id, "engine_ref": dict(_REF),
                           "operation_id": "op-removed"})
    assert remote_engines_named(removed)
    outcome = remote_stops.cancel_open_model_operations(removed)
    assert list(outcome) == ["op-removed"] and outcome["op-removed"] != "requested"  # its engine is gone: unconfirmed


def test_the_gates_witness_is_the_folder_every_tunnel_is_published_in(local_only, monkeypatch):
    """``remote_stops.TUNNEL_HOME`` names exactly the folder ``remote_engines`` publishes each tunnel under."""
    from ouroboros import config, remote_engines

    monkeypatch.setattr(config, "DATA_DIR", local_only)
    assert remote_engines._tunnel_dir("removed").parent == local_only.joinpath(*remote_stops.TUNNEL_HOME)


@pytest.mark.parametrize("reached", [False, True], ids=["never-reached", "reached-then-removed"])
def test_a_local_only_installation_stays_on_the_cheap_path(local_only, reached):
    """No connection and only local work: the gate stays closed and Panic's remote step does nothing, whether or not
    a remote engine was ever reached here (once one was, current custody is read and holds nothing remote)."""
    if reached:
        local_only.joinpath(*remote_stops.TUNNEL_HOME, "removed").mkdir(parents=True)
    local = custody.RunCustody(run_id="run-local", task_id="test-owner", invocation_id="inv-local", route_id="codex",
                               model="m", access="full", mode="agent", ledger_root=str(local_only))
    assert custody.record_started(local_only, local)
    assert not remote_engines_named(local_only)
    assert _remote_panic_step(local_only) == _remote_panic_step(local_only, settle=True) == {
        "requested": True, "remote_engines": 0}
    assert not current.run(local_only, "run-local").containment_fault
    assert not (local_only / "state" / "usage.sqlite").exists()  # the gate reads; it never imports or creates money


def test_unreadable_custody_is_unknown_once_a_remote_engine_was_reached_and_never_read_before(local_only, monkeypatch):
    """The gate's one cheap fact: without the tunnel home no remote engine was ever reached here, so custody is not
    read at all (a store that cannot be read holds nothing remote); once one was, a read that fails keeps the gate
    open (unknown is never none)."""
    def unreadable(*_args, **_kwargs):
        raise OSError("custody store unreadable")

    monkeypatch.setattr(current, "open_held_runs", unreadable)
    assert not remote_engines_named(local_only)
    local_only.joinpath(*remote_stops.TUNNEL_HOME, "removed").mkdir(parents=True)
    assert remote_engines_named(local_only)


def test_the_remote_engines_view_keeps_a_removed_connections_owed_stop_in_sight(removed, monkeypatch):
    """What the Remote engines section reads (``remote_engines.status``): a stop Emergency Stop owes on a connection
    no longer saved is listed as that removed connection, never as connected or as nothing owed."""
    from ouroboros import config, remote_engines

    monkeypatch.setattr(config, "DATA_DIR", removed)
    row = custody.RunCustody(run_id="run-existing", task_id="test-owner", invocation_id="inv-existing",
                             route_id="codex", model="m", access="full", mode="agent", ledger_root=str(removed),
                             engine_ref=dict(_REF), daemon_store_id="s")
    assert custody.record_started(removed, row)
    assert remote_engines.status() == []  # nothing owed yet: a removed connection is not listed for its history
    _remote_panic_step(removed)
    assert remote_engines.status() == [{
        "name": "removed", "label": "removed", "ssh_host": "", "engine_home": "", "remote_root": "", "workspaces": [],
        "enabled": False, "observation": {"state": "removed"}, "owed_stops": {"runs": 1, "starts": 0},
        "removed": True}]

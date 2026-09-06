"""The installation daemon survives generation sweeps, including legacy rows."""

import os
import subprocess
import sys
import time

import pytest

from ouroboros import claudexor_daemon, process_custody


@pytest.mark.serial
@pytest.mark.skipif(os.name == "nt", reason="POSIX real process-group stop; Windows uses Job Objects")
@pytest.mark.parametrize("scope", ["session", "daemon"])
def test_generation_change_keeps_daemon_work_and_explicit_stop_still_works(tmp_path, monkeypatch, scope):
    proc = process_custody.spawn_supervised(
        [sys.executable, "-u", "-c", "import sys; print('ready'); "
         "[print(line.strip()) for line in sys.stdin]"],
        drive_root=tmp_path, purpose=claudexor_daemon.CUSTODY_PURPOSE, scope=scope,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    manager = claudexor_daemon.OwnedClaudexorDaemon()
    manager._proc = proc
    try:
        assert proc.stdout.readline().strip() == "ready"
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-server-generation")
        assert process_custody.reap_orphaned_processes(
            tmp_path, retained_purposes={claudexor_daemon.CUSTODY_PURPOSE},
        ) == []
        proc.stdin.write("work survives\n")
        proc.stdin.flush()
        assert proc.stdout.readline().strip() == "work survives"
        assert len(process_custody._read_ledger(tmp_path)) == 1
        assert manager.stop() is True
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            manager.stop()
        proc.wait(timeout=10)
        proc.stdin.close()
        proc.stdout.close()


def test_retained_purpose_never_rescues_a_stale_identity(tmp_path, monkeypatch):
    row = {"pid": 123, "purpose": claudexor_daemon.CUSTODY_PURPOSE, "scope": "session",
           "session_id": "old-generation"}
    monkeypatch.setattr(process_custody, "_read_ledger", lambda _: [row])
    monkeypatch.setattr(process_custody, "_fingerprint_matches", lambda _: False)
    kept = []
    monkeypatch.setattr(process_custody, "_rewrite_ledger", lambda _, entries: kept.extend(entries))
    assert process_custody.reap_orphaned_processes(
        tmp_path, retained_purposes={claudexor_daemon.CUSTODY_PURPOSE}) == []
    assert kept == []


@pytest.mark.parametrize("surface", ["startup", "periodic"])
def test_both_server_sweeps_preserve_legacy_daemon_records(tmp_path, monkeypatch, surface):
    from ouroboros import server_maintenance

    calls = []
    monkeypatch.setattr(server_maintenance, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_maintenance, "_installed_skill_names", lambda: None)
    monkeypatch.setattr(server_maintenance, "_reconcile_delegated_runs", lambda _: None)
    monkeypatch.setattr(server_maintenance, "_cursor_refresh_settled_terminals", lambda: None)
    monkeypatch.setattr(server_maintenance, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])
    monkeypatch.setattr(process_custody, "reap_orphaned_processes", lambda root, **kw: calls.append((root, kw)) or [])
    monkeypatch.setattr("ouroboros.delegate_terminal.backfill_terminal_reconciliations", lambda _: [])
    monkeypatch.setattr("supervisor.terminal_delivery.replay_pending_deliveries", lambda _: None)
    monkeypatch.setattr("ouroboros.delegate_state_sweep.sweep_settled_delegate_state", lambda _: {})
    monkeypatch.setattr("ouroboros.model_send_seal.reconcile_model_send_seals", lambda _: {})
    if surface == "startup":
        server_maintenance._startup_custody_sweep()
    else:
        server_maintenance._periodic_supervisor_maintenance([0], [time.time()])
    assert len(calls) == 1
    assert calls[0][0] == tmp_path
    assert calls[0][1]["retained_purposes"] == {claudexor_daemon.CUSTODY_PURPOSE}

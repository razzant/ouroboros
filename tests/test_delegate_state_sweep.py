"""CPL4-C13 pins: terminal+age sweep of delegate recovery/supervision state.

Only PROVEN-dead-and-old files move: terminal-status recovery rows past GC
retention, orphaned restart transactions, supervision files of settled tasks.
Live/resumable rows, referenced transactions, ``active.json``, unreadable
rows and anything younger than retention all stay; an unreadable custody log
skips the sweep entirely (the ``_prune_delegated_snapshots`` idiom).
"""

from __future__ import annotations

import json
import os
import time

from ouroboros.delegate_state_sweep import sweep_settled_delegate_state
from ouroboros.task_result_schema import stamp_task_result_schema

_OLD = time.time() - 400 * 86400


def _write_row(root, family, name, payload, *, old=True):
    path = root / "state" / family / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    if old:
        os.utime(path, (_OLD, _OLD))
    return path


def _settle_task(root, task_id):
    results = root / "task_results"
    results.mkdir(parents=True, exist_ok=True)
    row = stamp_task_result_schema({"task_id": task_id, "status": "completed"})
    (results / f"{task_id}.json").write_text(json.dumps(row), encoding="utf-8")


def test_terminal_old_rows_swept_live_rows_kept(tmp_path):
    vetoed = _write_row(tmp_path, "delegate_recovery", "t-vetoed", {"status": "vetoed"})
    adopted = _write_row(tmp_path, "delegate_recovery", "t-adopted", {"status": "adopted"})
    reserved = _write_row(
        tmp_path, "delegate_recovery", "t-reserved",
        {"status": "reserved", "restart_transaction_id": "tx-live"},
    )
    fresh_terminal = _write_row(
        tmp_path, "delegate_recovery", "t-fresh", {"status": "vetoed"}, old=False,
    )

    report = sweep_settled_delegate_state(tmp_path)

    assert not vetoed.exists() and not adopted.exists()
    assert reserved.exists() and fresh_terminal.exists()
    assert sorted(report["removed"]) == [
        "delegate_recovery/t-adopted.json", "delegate_recovery/t-vetoed.json",
    ]
    assert not report["errors"] and not report["skipped"]


def test_transactions_referenced_or_active_survive(tmp_path):
    _write_row(tmp_path, "delegate_recovery", "t-reserved",
               {"status": "reserved", "restart_transaction_id": "tx-live"})
    live_tx = _write_row(tmp_path, "delegate_recovery_transactions", "tx-live",
                         {"status": "prepared"})
    orphan_tx = _write_row(tmp_path, "delegate_recovery_transactions", "tx-orphan",
                           {"status": "normal_exit_acknowledged"})
    active = _write_row(tmp_path, "delegate_recovery_transactions", "active",
                        {"transaction_id": "tx-live"})

    report = sweep_settled_delegate_state(tmp_path)

    assert live_tx.exists() and active.exists()
    assert not orphan_tx.exists()
    assert report["removed"] == ["delegate_recovery_transactions/tx-orphan.json"]


def test_unreadable_recovery_row_keeps_itself_and_all_transactions(tmp_path):
    broken = _write_row(tmp_path, "delegate_recovery", "t-broken", {})
    broken.write_text("{not json", encoding="utf-8")
    os.utime(broken, (_OLD, _OLD))
    orphan_tx = _write_row(tmp_path, "delegate_recovery_transactions", "tx-orphan",
                           {"status": "prepared"})

    report = sweep_settled_delegate_state(tmp_path)

    assert broken.exists() and orphan_tx.exists()
    assert report["errors"] and report["skipped"] == "transactions_kept_unreadable_recovery_row"


def test_supervision_swept_only_for_settled_old_tasks(tmp_path):
    settled = _write_row(tmp_path, "delegate_supervision", "t-done",
                         {"schema": 1, "run_id": "r1"})
    _settle_task(tmp_path, "t-done")
    no_result = _write_row(tmp_path, "delegate_supervision", "t-noresult",
                           {"schema": 1, "run_id": "r2"})
    fresh_settled = _write_row(tmp_path, "delegate_supervision", "t-fresh",
                               {"schema": 1, "run_id": "r3"}, old=False)
    _settle_task(tmp_path, "t-fresh")

    report = sweep_settled_delegate_state(tmp_path)

    assert not settled.exists()
    assert no_result.exists()      # fail-closed: no durable result
    assert fresh_settled.exists()  # younger than retention
    assert report["removed"] == ["delegate_supervision/t-done.json"]


def test_unreadable_custody_log_skips_the_sweep(tmp_path, monkeypatch):
    import ouroboros.delegate_state_sweep as sweep_mod

    victim = _write_row(tmp_path, "delegate_recovery", "t-vetoed", {"status": "vetoed"})
    monkeypatch.setattr(
        "ouroboros.delegate_custody.custody_log_unreadable", lambda root: True,
    )

    report = sweep_mod.sweep_settled_delegate_state(tmp_path)

    assert victim.exists()
    assert report["skipped"] == "custody_log_unreadable" and not report["removed"]


def test_startup_custody_sweep_runs_the_state_sweep():
    import inspect

    import ouroboros.server_maintenance as sm

    assert "sweep_settled_delegate_state" in inspect.getsource(sm._startup_custody_sweep)

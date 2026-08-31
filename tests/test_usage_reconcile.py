"""Reconciler tests for the abandoned unresolved-attempt write-off lifecycle.

A transient provider failure (429/5xx, transport reset) leaves a TERMINAL
``unresolved`` ledger row that no writer can ever resolve; without a
reconciliation path it blocked ``cost_final`` forever (the benchmark incident
class). These tests pin the production path: ``terminalize_abandoned_attempt``
gains the typed bound write-off, and ``reconcile_abandoned_unresolved_attempts``
drives it from the terminal cost authority (any age) and the periodic sweep
(TTL-gated).
"""

from __future__ import annotations

import json
import time

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.usage_ledger import UNRESOLVED_WRITEOFF_REASON, UsageLedgerCorrupt
from ouroboros.usage_reconcile import reconcile_abandoned_unresolved_attempts


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    (root / "state").mkdir(parents=True)
    return root


def _request(data_root, **overrides):
    values = {
        "model": "openai/gpt-5.2",
        "provider": "openai",
        "reservation_usd": 1.0,
        "drive_root": data_root,
        "task_id": "child",
        "root_task_id": "root",
        "source": "test",
    }
    values.update(overrides)
    return ua.AttemptRequest(**values)


def _ledger(data_root):
    path = data_root / ua.LEDGER_REL
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _unresolved_reservation(data_root, **overrides):
    reservation = ua.reserve_attempt(_request(data_root, **overrides))
    ua.mark_dispatched(reservation)
    ua.mark_unresolved(reservation, "RateLimitError: transient 429")
    return reservation


def test_terminal_task_reconcile_writes_off_immediately_and_clears_finality(data_root):
    """The incident shape: a completed task's stuck unresolved row must not
    block cost_final once the terminal cost authority runs."""
    reservation = _unresolved_reservation(data_root)
    before = ua.usage_projection(data_root)
    assert before["cost_final"] is False
    assert before["non_final_rows"] == 1
    assert before["unresolved_upper_bound_usd"] == 1.0
    assert before["accounted_usd"] == 1.0

    outcome = reconcile_abandoned_unresolved_attempts(data_root, task_id="child")

    assert outcome["terminalized"] == [reservation.attempt_id]
    assert outcome["kept_unknown_bound"] == []
    after = ua.usage_projection(data_root)
    # Finality flips; money does not move — the carried bound becomes the cost.
    assert after["cost_final"] is True
    assert after["non_final_rows"] == 0
    assert after["unresolved_upper_bound_usd"] == 0.0
    assert after["accounted_usd"] == before["accounted_usd"] == 1.0
    assert after["settled_usd"] == 1.0
    final_row = _ledger(data_root)[-1]
    assert final_row["state"] == "settled"
    assert final_row["cost_usd"] == 1.0
    assert final_row["cost_final"] is True
    assert final_row["settle_reason"] == UNRESOLVED_WRITEOFF_REASON
    assert "429" in final_row["origin_reason"]


def test_subtree_rows_reconcile_at_root_terminalization(data_root):
    """A child's dead row gates the ROOT's frame: root_task_id scoping covers it."""
    _unresolved_reservation(data_root, task_id="child", root_task_id="root")

    outcome = reconcile_abandoned_unresolved_attempts(data_root, task_id="root")

    assert len(outcome["terminalized"]) == 1
    assert ua.usage_projection(data_root, root_task_id="root")["cost_final"] is True


def test_task_scoped_reconcile_leaves_other_tasks_rows_alone(data_root):
    mine = _unresolved_reservation(data_root, task_id="child", root_task_id="root")
    other = _unresolved_reservation(data_root, task_id="other", root_task_id="other")

    outcome = reconcile_abandoned_unresolved_attempts(data_root, task_id="child")

    assert outcome["terminalized"] == [mine.attempt_id]
    projection = ua.usage_projection(data_root)
    assert projection["non_final_rows"] == 1  # the other task's row stays open
    states = {row["attempt_id"]: row["state"] for row in _ledger(data_root)}
    assert states[other.attempt_id] == "unresolved"


def test_sweep_is_ttl_gated(data_root):
    """Before the TTL the row stays honestly pending; past it, written off."""
    reservation = _unresolved_reservation(data_root)
    row = _ledger(data_root)[-1]
    import datetime as _dt

    born = _dt.datetime.fromisoformat(row["ts"]).timestamp()

    early = reconcile_abandoned_unresolved_attempts(data_root, now=born + 10, max_age_sec=900)
    assert early["terminalized"] == []
    assert ua.usage_projection(data_root)["cost_final"] is False

    late = reconcile_abandoned_unresolved_attempts(data_root, now=born + 901, max_age_sec=900)
    assert late["terminalized"] == [reservation.attempt_id]
    assert ua.usage_projection(data_root)["cost_final"] is True


def test_sweep_uses_configured_ttl_default(data_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_USAGE_UNRESOLVED_WRITEOFF_SEC", "120")
    reservation = _unresolved_reservation(data_root)
    import datetime as _dt

    born = _dt.datetime.fromisoformat(_ledger(data_root)[-1]["ts"]).timestamp()
    assert reconcile_abandoned_unresolved_attempts(
        data_root, now=born + 119
    )["terminalized"] == []
    assert reconcile_abandoned_unresolved_attempts(data_root, now=born + 121)["terminalized"] == [
        reservation.attempt_id
    ]


def test_unknown_bound_row_stays_honestly_unresolved(data_root):
    """No fabricated number: a row with unknown pricing never gets a write-off cost."""
    _unresolved_reservation(data_root, reservation_usd=None, force_unknown_reservation=True)

    outcome = reconcile_abandoned_unresolved_attempts(data_root, task_id="child")

    assert outcome["terminalized"] == []
    assert len(outcome["kept_unknown_bound"]) == 1
    projection = ua.usage_projection(data_root)
    assert projection["cost_final"] is False
    assert projection["non_final_rows"] == 1


def test_legacy_metadata_unresolved_rows_are_not_touched(data_root):
    """Legacy ambiguous-call rows are a separate class: no bound, never finality-blocking."""
    with ua._locked(data_root):
        records = ua._read_records_locked_cached(data_root)
        ua._append_rows_locked(
            data_root,
            records,
            [
                {
                    "kind": "legacy_metadata",
                    "attempt_id": "legacy-meta-1",
                    "state": "unresolved",
                    "model": "",
                    "provider": "legacy",
                    "reservation_upper_bound_usd": None,
                    "ambiguous_call_count": 3,
                    "task_id": "",
                    "root_task_id": "",
                    "parent_task_id": "",
                    "category": "legacy",
                    "source": "legacy_state_call_delta",
                }
            ],
        )
    outcome = reconcile_abandoned_unresolved_attempts(data_root, task_id="", max_age_sec=0)
    assert outcome["terminalized"] == []
    assert outcome["kept_unknown_bound"] == []
    assert _ledger(data_root)[-1]["state"] == "unresolved"


def test_reconcile_is_idempotent(data_root):
    reservation = _unresolved_reservation(data_root)
    first = reconcile_abandoned_unresolved_attempts(data_root, task_id="child")
    second = reconcile_abandoned_unresolved_attempts(data_root, task_id="child")
    assert first["terminalized"] == [reservation.attempt_id]
    assert second["terminalized"] == []
    assert [row["state"] for row in _ledger(data_root)] == ["reserved", "dispatched", "unresolved", "settled"]


def test_terminalize_abandoned_attempt_unresolved_writeoff_direct(data_root):
    """The terminalizer itself: unresolved -> settled at bound; idempotent after."""
    reservation = _unresolved_reservation(data_root)

    assert ua.terminalize_abandoned_attempt(reservation, reason="reconcile") == "settled"
    # Idempotent: a terminal attempt is never transitioned again.
    assert ua.terminalize_abandoned_attempt(reservation, reason="again") == "settled"
    assert [row["state"] for row in _ledger(data_root)] == ["reserved", "dispatched", "unresolved", "settled"]


def test_terminalize_abandoned_attempt_prior_semantics_preserved(data_root):
    """reserved -> released; dispatched without usage -> unresolved (bound carried)."""
    reserved = ua.reserve_attempt(_request(data_root, task_id="t1"))
    assert ua.terminalize_abandoned_attempt(reserved, reason="never started") == "released"

    dispatched = ua.reserve_attempt(_request(data_root, task_id="t2"))
    ua.mark_dispatched(dispatched)
    assert ua.terminalize_abandoned_attempt(dispatched, reason="child aborted") == "unresolved"
    # ...and THAT unresolved row is write-off eligible on the next pass.
    assert ua.terminalize_abandoned_attempt(dispatched, reason="reconcile") == "settled"
    assert ua.usage_projection(data_root)["cost_final"] is True


def test_unresolved_exit_without_the_typed_reason_is_corrupt():
    """The substrate guard: unresolved -> settled exists ONLY as the typed write-off."""
    rows = [
        {"seq": 1, "attempt_id": "a", "state": "reserved", "kind": "attempt"},
        {"seq": 2, "attempt_id": "a", "state": "dispatched", "kind": "attempt"},
        {"seq": 3, "attempt_id": "a", "state": "unresolved", "kind": "attempt"},
        {"seq": 4, "attempt_id": "a", "state": "settled", "kind": "attempt", "cost_usd": 1.0},
    ]
    with pytest.raises(UsageLedgerCorrupt):
        ua._validate_records(rows)
    rows[3]["settle_reason"] = UNRESOLVED_WRITEOFF_REASON
    rows[3]["reservation_upper_bound_usd"] = 1.0
    rows[3]["cost_final"] = True
    ua._validate_records(rows)  # typed write-off at the carried bound, final: legal
    rows[3]["state"] = "released"
    with pytest.raises(UsageLedgerCorrupt):
        ua._validate_records(rows)


def test_writeoff_transition_rejects_a_discounted_or_nonfinal_cost():
    """Sol review pin: the authority rejects bound $1.00 settling as final $0.00."""
    base = [
        {"seq": 1, "attempt_id": "a", "state": "reserved", "kind": "attempt",
         "reservation_upper_bound_usd": 1.0},
        {"seq": 2, "attempt_id": "a", "state": "dispatched", "kind": "attempt",
         "reservation_upper_bound_usd": 1.0},
        {"seq": 3, "attempt_id": "a", "state": "unresolved", "kind": "attempt",
         "reservation_upper_bound_usd": 1.0},
    ]
    writeoff = {
        "seq": 4, "attempt_id": "a", "state": "settled", "kind": "attempt",
        "settle_reason": UNRESOLVED_WRITEOFF_REASON,
        "reservation_upper_bound_usd": 1.0, "cost_usd": 1.0, "cost_final": True,
    }
    ua._validate_records([*base, dict(writeoff)])  # exact bound, final: accepted

    discounted = dict(writeoff, cost_usd=0.0)
    with pytest.raises(UsageLedgerCorrupt):
        ua._validate_records([*base, discounted])

    under_bound = dict(writeoff, cost_usd=0.5)
    with pytest.raises(UsageLedgerCorrupt):
        ua._validate_records([*base, under_bound])

    non_final = dict(writeoff, cost_final=False)
    with pytest.raises(UsageLedgerCorrupt):
        ua._validate_records([*base, non_final])

    missing_cost = dict(writeoff)
    del missing_cost["cost_usd"]
    with pytest.raises(UsageLedgerCorrupt):
        ua._validate_records([*base, missing_cost])


def test_writeoff_event_lands_in_events_log(data_root):
    """The audit disclosure: every sweep write-off is visible in events.jsonl."""
    reservation = _unresolved_reservation(data_root)

    outcome = reconcile_abandoned_unresolved_attempts(data_root, task_id="child")

    assert outcome["terminalized"] == [reservation.attempt_id]
    events = [
        json.loads(line)
        for line in (data_root / "logs" / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    writeoffs = [row for row in events if row.get("type") == "usage_unresolved_writeoff"]
    assert len(writeoffs) == 1
    assert writeoffs[0]["attempt_ids"] == [reservation.attempt_id]
    assert writeoffs[0]["task_id"] == "child"
    assert writeoffs[0]["ts"]


def test_writeoff_ttl_getter_clamps(monkeypatch):
    from ouroboros.config import get_usage_unresolved_writeoff_sec

    monkeypatch.delenv("OUROBOROS_USAGE_UNRESOLVED_WRITEOFF_SEC", raising=False)
    assert get_usage_unresolved_writeoff_sec() == 900.0
    monkeypatch.setenv("OUROBOROS_USAGE_UNRESOLVED_WRITEOFF_SEC", "30")
    assert get_usage_unresolved_writeoff_sec() == 60.0
    monkeypatch.setenv("OUROBOROS_USAGE_UNRESOLVED_WRITEOFF_SEC", "999999")
    assert get_usage_unresolved_writeoff_sec() == 86400.0
    monkeypatch.setenv("OUROBOROS_USAGE_UNRESOLVED_WRITEOFF_SEC", "garbage")
    assert get_usage_unresolved_writeoff_sec() == 900.0


def test_periodic_supervisor_maintenance_runs_the_reconciler(monkeypatch):
    """The production seam: the supervisor maintenance tick drives the sweep."""
    import sys

    # server.py inserts OUROBOROS_REPO_DIR (the live repo on this host, which may
    # diverge from this checkout) into sys.path at import time and sets
    # OUROBOROS_AGENT_PYTHON; keep both from leaking into later tests.
    monkeypatch.delenv("OUROBOROS_REPO_DIR", raising=False)
    monkeypatch.delenv("OUROBOROS_AGENT_PYTHON", raising=False)
    saved_path = list(sys.path)
    import server

    sys.path[:] = saved_path

    calls = []
    monkeypatch.setattr(
        "ouroboros.usage_reconcile.reconcile_abandoned_unresolved_attempts",
        lambda drive_root: calls.append(drive_root) or {"terminalized": []},
    )
    # Only the usage cadence is due; every sibling cadence just ran.
    monkeypatch.setattr(server, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])
    monkeypatch.setattr(server, "_LAST_USAGE_RECONCILE", [0.0])
    now = time.time()
    server._periodic_supervisor_maintenance([now], [now])
    assert calls == [server.DATA_DIR]
    # Cadence: a second pass inside the 300s window does not re-fire.
    server._periodic_supervisor_maintenance([now], [now])
    assert len(calls) == 1


def test_reconstruct_task_cost_terminalizes_the_tasks_dead_rows(data_root):
    """The production seam: the terminal cost authority reconciles before reading."""
    from supervisor.state import reconstruct_task_cost

    _unresolved_reservation(data_root, task_id="child", root_task_id="root")
    fields = reconstruct_task_cost("child", fields=True, drive_root=data_root)
    assert fields["cost_accounting_status"] == "available"
    assert fields["cost_final"] is True
    assert fields["non_final_rows"] == 0
    assert fields["cost_usd"] == 1.0

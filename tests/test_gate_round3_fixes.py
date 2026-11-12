"""GATE ROUND-3 (v6.98.0 phase A) — probe-backed regression tests.

GR3-1  cascade intent is postcondition-owned ALWAYS (replay summary included,
       widened-mid-flight survival, dead-descendants defer);
GR3-2  a live claim is never stolen by age + the minimal pre-write claim fence;
GR3-3  evolution closure gated on completeness (settle-time backstop);
GR3-4  outbox registration failure is a real failure;
GR3-5  the persist→register crash window is closed by ORDERING;
GR3-6  an unowned lifecycle fault rides the normal terminal dispatch;
GR3-7  audit failure is typed UNKNOWN, never clean;
GR3-8  stale sweep failures do not veto a converged tree;
GR3-9  strict registry reads refuse a corrupt projection;
GR3-11 import coherence (the fail_tasks half was retired with the
       function itself in the 7.0 ABI window, owner Q10=A).
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_RUNNING,
    load_task_result,
    write_task_result,
)


class _DeadProc:
    """A worker proc surface that is already dead (no real subprocess)."""

    pid = 0

    def is_alive(self):
        return False

    def terminate(self):
        return None

    def join(self, timeout=None):
        return None


def _trail_rows(drive, log_name="supervisor.jsonl"):
    path = pathlib.Path(drive) / "logs" / log_name
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture()
def qenv(tmp_path, monkeypatch):
    import supervisor.queue as q
    from supervisor import task_lifecycle, workers

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {}, raising=False)
    return types.SimpleNamespace(q=q, tl=task_lifecycle, workers=workers, drive=tmp_path)


# --------------------------------------------------------------------------
# GR3-1 — cascade intent is postcondition-owned, ALWAYS
# --------------------------------------------------------------------------
# GR3-2 — minimal write fence on the custody kill path
# --------------------------------------------------------------------------


def test_lost_claim_before_the_terminal_write_aborts_publication(qenv):
    """GR3-2 write fence: the claim is re-verified between the kill/join window
    and the durable terminal write; a stolen claim aborts (CANCEL_FAILED,
    custody restored, no terminal write) instead of publishing over the new
    owner's teardown."""
    stolen = {"done": False}
    store = qenv.drive / "state" / "cancel_intents.json"

    class _StealingProc(_DeadProc):
        def join(self, timeout=None):
            if not stolen["done"]:
                data = json.loads(store.read_text(encoding="utf-8"))
                row = data["intents"]["steal1"]
                row["generation"] = int(row["generation"]) + 1
                row["claim_pid"] = 2 ** 22
                row["claim_owner"] = "other-custody"
                data["intents"]["steal1"] = row
                store.write_text(json.dumps(data), encoding="utf-8")
                stolen["done"] = True

    task = {"id": "steal1", "chat_id": 2}
    worker = types.SimpleNamespace(
        wid=0, proc=_StealingProc(), busy_task_id="steal1", reaping=False,
    )
    qenv.workers.WORKERS[0] = worker
    qenv.q.RUNNING["steal1"] = {"task": task, "worker_id": 0}
    write_task_result(qenv.drive, "steal1", STATUS_RUNNING, result="working")
    ci.request_cancel(qenv.drive, "steal1", reason="stop")

    outcome = qenv.tl.cancel_task_custody("steal1")

    assert outcome == qenv.tl.CANCEL_FAILED
    assert load_task_result(qenv.drive, "steal1")["status"] == STATUS_RUNNING, (
        "no terminal write over a lost claim"
    )
    assert "steal1" in qenv.q.RUNNING, "custody restored"
    assert worker.reaping is False


# --------------------------------------------------------------------------
# GR3-3 — evolution closure gated on completeness
# --------------------------------------------------------------------------


def _stop_outcomes(*, failed=()):
    return {
        "cancelled": [], "already_settled": [], "not_found": [],
        "failed": list(failed), "intent_write_failed": [],
    }


def test_incomplete_evolution_stop_leaves_the_campaign_open_until_settle(tmp_path, monkeypatch):
    import supervisor.queue as q
    import supervisor.state as state
    from supervisor import events as events_mod
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    q.init(tmp_path)
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "RUNNING", {})
    assert el.start_evolution_campaign("Improve", source="test").get("status") == "active"
    state.update_state(lambda live: live.update(
        owner_chat_id=7, evolution_mode_enabled=True, evolution_owner_stopped=False,
    ))
    sent: list = []
    ctx = types.SimpleNamespace(
        sort_pending=lambda: None,
        persist_queue_snapshot=lambda reason="": None,
        send_with_budget=lambda chat_id, text, **kw: sent.append(text),
        load_state=state.load_state,
    )
    monkeypatch.setattr(
        q, "stop_evolution_tasks", lambda reason: _stop_outcomes(failed=["evo-live"]),
    )

    events_mod._handle_toggle_evolution({"enabled": False}, ctx)

    assert bool(state.load_state().get("evolution_owner_stopped")) is True
    assert el._read_evolution_campaign().get("status") == "active", (
        "an INCOMPLETE stop must not close the campaign"
    )
    assert any("INCOMPLETE" in text for text in sent)

    # The settle-time backstop closes it once the live task settles — invoked
    # from the evolution terminal path even for a rejected terminal (finally).
    events_mod._handle_evolution_task_done(
        ctx, evt={}, task_id="evo-live", task={},
        task_done_event={"status": "failed"}, outcome_axes={}, cost=None, rounds=0,
    )
    closed = el._read_evolution_campaign()
    assert closed.get("status") == "stopped"
    assert "owner stop" in str(closed.get("completion_reason") or "")


def test_owner_evolution_stop_gates_closure_on_completeness(tmp_path, monkeypatch):
    import server as srv
    import supervisor.queue as q
    import supervisor.state as state
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    q.init(tmp_path)
    assert el.start_evolution_campaign("Improve", source="test").get("status") == "active"
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        sort_pending=lambda: None,
        persist_queue_snapshot=lambda reason="": None,
        send_with_budget=lambda chat_id, text, **kw: None,
    )
    monkeypatch.setattr(
        q, "stop_evolution_tasks", lambda reason: _stop_outcomes(failed=["evo-live"]),
    )
    wording = srv._owner_evolution_stop(ctx, 7)
    assert "INCOMPLETE" in wording
    assert el._read_evolution_campaign().get("status") == "active", "left open"

    # A later CLEAN pass closes the campaign terminally.
    monkeypatch.setattr(q, "stop_evolution_tasks", lambda reason: _stop_outcomes())
    wording = srv._owner_evolution_stop(ctx, 7)
    assert "INCOMPLETE" not in wording
    assert el._read_evolution_campaign().get("status") == "stopped"


# --------------------------------------------------------------------------
# GR3-4 — outbox registration failure is a real failure
# --------------------------------------------------------------------------


def test_failed_owed_registration_keeps_the_cancel_intent_open(qenv, monkeypatch):
    """GR3-4 cancel path: an answer that could not be durably owed must NOT be
    settled over — the intent stays open (claim released) for the watchdog,
    whose retry finds the settled result and re-delivers on the miss lane."""
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *a, **kw: None)
    monkeypatch.setattr(
        "supervisor.terminal_delivery.register_pending_delivery",
        lambda *a, **kw: False,
    )
    task = {"id": "owed1", "chat_id": 4}
    worker = types.SimpleNamespace(wid=0, proc=_DeadProc(), busy_task_id="owed1", reaping=False)
    qenv.workers.WORKERS[0] = worker
    qenv.q.RUNNING["owed1"] = {"task": task, "worker_id": 0}
    write_task_result(qenv.drive, "owed1", STATUS_RUNNING, result="working")
    ci.request_cancel(qenv.drive, "owed1", reason="stop")

    outcome = qenv.tl.cancel_task_custody("owed1")

    assert outcome == qenv.tl.CANCEL_CANCELLED, "the teardown itself is not gated"
    assert load_task_result(qenv.drive, "owed1")["status"] == STATUS_CANCELLED
    row = ci.active_intent(qenv.drive, "owed1")
    assert row is not None and row["state"] == ci.INTENT_REQUESTED, (
        "the intent stays OPEN (claim released) over the unowed answer"
    )


# --------------------------------------------------------------------------
# GR3-5 — the persist→register crash window is closed by ordering
# --------------------------------------------------------------------------
# GR3-6 — unowned lifecycle fault uses normal terminal dispatch
# --------------------------------------------------------------------------
# GR3-7 — audit failure is typed UNKNOWN, never clean
# --------------------------------------------------------------------------


def test_delegated_audit_failure_is_typed_unknown_never_clean(qenv, monkeypatch):
    import ouroboros.delegate_custody as dc
    from supervisor.terminal_delivery import (
        RUN_STATE_UNKNOWN_PREFIX,
        build_unreviewed_salvage_event,
    )

    monkeypatch.setattr(dc, "reconcile_task_runs", lambda *a, **kw: None)

    def _boom(*a, **kw):
        raise RuntimeError("custody rows unreadable")

    monkeypatch.setattr(dc, "open_runs", _boom)
    still = qenv.tl._reconcile_delegated_runs_on_kill(qenv.q, "au1")
    assert still == [f"{RUN_STATE_UNKNOWN_PREFIX}:audit_failed"]
    rows = [
        r for r in _trail_rows(qenv.drive, "events.jsonl")
        if r.get("type") == "delegated_runs_unreconciled"
    ]
    assert rows and rows[0]["flavor"] == "audit_failed" and rows[0]["run_ids"] == still

    # The marker rides the delivery note as an honest UNKNOWN sentence.
    event = build_unreviewed_salvage_event(
        qenv.drive, {"id": "au1", "chat_id": 3}, "au1",
        outcome="cancelled", salvaged_text="partial", unreconciled_runs=still,
        settled_status="cancelled",
    )
    assert "DELEGATED RUN STATE UNKNOWN" in event["text"]

    # A failed addressed snapshot stays unknown without falling back to global readers.
    from ouroboros import delegate_terminal

    monkeypatch.setattr(dc, "open_runs", lambda *a, **kw: [])
    monkeypatch.setattr(delegate_terminal, "custody_audit_snapshot",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("snapshot unreadable")))

    def _boom2(*a, **kw):
        raise RuntimeError("pending rows unreadable")

    monkeypatch.setattr(dc, "pending_invocations", _boom2)
    still2 = qenv.tl._reconcile_delegated_runs_on_kill(qenv.q, "au2")
    assert still2 == [f"{RUN_STATE_UNKNOWN_PREFIX}:snapshot_unavailable"]


# --------------------------------------------------------------------------
# GR3-8 — stale sweep failures do not veto a converged tree
# --------------------------------------------------------------------------
# GR3-11a — import coherence
# --------------------------------------------------------------------------

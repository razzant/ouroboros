"""Custody: the one settle owner, and the reaping slot it must never strand.

Split out of ``tests/test_cancel_intents_phase_a.py`` by theme: custody over a task that
is neither queued nor running, the watchdog sweep that feeds it open and stale claimed
intents, and every path where a slot could be left stranded — a raising teardown, an
abandoned claim, a dead custody, two concurrent custodies, and a losing takeover.
"""

from __future__ import annotations

import json
import types

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_RUNNING,
    load_task_result,
    write_task_result,
)

from tests._cancel_intents_shared import _live_split_drive_task
from tests._cancel_intents_shared import (  # noqa: F401  (autouse fixture applies on import)
    _reap_spawned_live_procs,
)
from tests._cancel_intents_shared import qenv as _qenv

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
qenv = _qenv


def test_custody_settles_an_intent_for_a_missing_task(qenv):
    """The incident's wedge: intent recorded, task neither queued nor running —
    custody's finalize-on-miss settles it as cancelled with the parent decision
    stamped at OUTCOME (never at intent time)."""
    ci.request_cancel(qenv.drive, "ghost1", reason="tree teardown",
                      requested_by="parent9")
    write_task_result(qenv.drive, "ghost1", STATUS_RUNNING, result="was running")

    outcome = qenv.tl.cancel_task_custody("ghost1")

    assert outcome == qenv.tl.CANCEL_CANCELLED
    stored = load_task_result(qenv.drive, "ghost1")
    assert stored["status"] == STATUS_CANCELLED
    assert stored["parent_decision"] == "cancelled"
    assert stored["parent_decision_reason"] == "tree teardown"
    # Honest accounting: reconstructed (confirmed zero here), never a missing block.
    assert "cost_accounting_status" in stored
    assert ci.active_intent(qenv.drive, "ghost1") is None


def test_watchdog_sweep_feeds_open_and_stale_claimed_intents(qenv, monkeypatch):
    fed: list[str] = []
    monkeypatch.setattr(qenv.tl, "cancel_task_custody",
                        lambda tid, **_kw: fed.append(tid) or "cancelled")

    now = 1_000_000.0
    # Open old intent: fed.
    ci.request_cancel(qenv.drive, "old1")
    # Freshly claimed intent: custody in flight — left alone.
    ci.request_cancel(qenv.drive, "claimed1")
    ci.claim_intent(qenv.drive, "claimed1", owner="cancel_task_custody")

    from datetime import datetime, timezone
    aged = datetime.fromtimestamp(now - 60, tz=timezone.utc).isoformat()
    stale = datetime.fromtimestamp(now - ci.CLAIM_STALE_SEC - 5, tz=timezone.utc).isoformat()
    # Rewrite provenance directly (test-only): age the open intent past the
    # watchdog min-age and make one claim stale.
    store = qenv.drive / "state" / "cancel_intents.json"
    data = json.loads(store.read_text(encoding="utf-8"))
    data["intents"]["old1"]["requested_at"] = aged
    claimed_now = datetime.fromtimestamp(now - 1, tz=timezone.utc).isoformat()
    data["intents"]["claimed1"]["claimed_at"] = claimed_now
    data["intents"]["claimed1"]["requested_at"] = aged
    store.write_text(json.dumps(data), encoding="utf-8")

    outcomes = qenv.tl.sweep_cancel_intents(now=now)
    assert fed == ["old1"]
    assert outcomes == {"old1": "cancelled"}
    ci.settle_intent(qenv.drive, "old1", outcome="cancelled")  # what real custody does

    # GR3-2: the same claim gone STALE while its claimant pid (this test
    # process) probes ALIVE is NEVER stolen by age — the live owner settles or
    # releases; stealing it would let two custodies double-settle.
    data = json.loads(store.read_text(encoding="utf-8"))
    data["intents"]["claimed1"]["claimed_at"] = stale
    store.write_text(json.dumps(data), encoding="utf-8")
    fed.clear()
    qenv.tl.sweep_cancel_intents(now=now)
    assert fed == []

    # Stale with liveness UNKNOWN (pid missing — the incident shape: custody
    # died mid-teardown before/without a readable pid) IS still recoverable.
    data = json.loads(store.read_text(encoding="utf-8"))
    data["intents"]["claimed1"].pop("claim_pid", None)
    store.write_text(json.dumps(data), encoding="utf-8")
    fed.clear()
    qenv.tl.sweep_cancel_intents(now=now)
    assert fed == ["claimed1"]

    # A brand-new intent is left one tick for its own control event.
    fed.clear()
    ci.request_cancel(qenv.drive, "young1")
    data = json.loads(store.read_text(encoding="utf-8"))
    data["intents"]["young1"]["requested_at"] = datetime.fromtimestamp(
        now - 1, tz=timezone.utc,
    ).isoformat()
    store.write_text(json.dumps(data), encoding="utf-8")
    qenv.tl.sweep_cancel_intents(now=now)
    assert "young1" not in fed


def test_lifecycle_fault_never_frees_a_reaping_slot(tmp_path):
    """A ``reaping`` slot is owned by the reaper/custody: releasing it here would
    hand a mid-kill process back to assignment."""
    from ouroboros.utils import append_jsonl
    from supervisor.events import _handle_task_done

    running = {"t12": {"task": {"id": "t12"}}}
    slot = types.SimpleNamespace(busy_task_id="t12", reaping=True)
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING=running,
        WORKERS={0: slot},
        append_jsonl=append_jsonl,
        persist_queue_snapshot=lambda **_kw: True,
    )
    _handle_task_done({"task_id": "t12", "status": "running", "worker_id": 0}, ctx)

    assert slot.busy_task_id == "t12" and slot.reaping is True


def test_concurrent_custody_on_a_pending_task_settles_exactly_once(qenv):
    """A-F11 probe shape: the second custody must give the capture back."""
    ci.request_cancel(qenv.drive, "pending-race", reason="stop")
    qenv.q.PENDING[:] = [{"id": "pending-race", "chat_id": 1}]
    write_task_result(qenv.drive, "pending-race", "scheduled")
    # Custody-1 holds a FRESH claim (it is mid-teardown).
    ci.claim_intent(qenv.drive, "pending-race", owner="custody-1")

    outcome = qenv.tl.cancel_task_custody("pending-race")

    assert outcome == qenv.tl.CANCEL_FAILED
    assert [t["id"] for t in qenv.q.PENDING] == ["pending-race"], "capture returned"
    assert load_task_result(qenv.drive, "pending-race")["status"] == "scheduled"
    assert ci.active_intent(qenv.drive, "pending-race")["claim_owner"] == "custody-1"


@pytest.mark.serial
def test_custody_raising_mid_teardown_releases_the_reaping_slot(qenv, monkeypatch):
    """A-F1a: a crash between capture and respawn must not strand the slot."""
    task_id = "raiser"
    task, _child_drive, proc = _live_split_drive_task(qenv, task_id)
    write_task_result(qenv.drive, task_id, STATUS_RUNNING, result="working")
    ci.request_cancel(qenv.drive, task_id)
    from supervisor import cancel_custody

    monkeypatch.setattr(
        cancel_custody, "_finish_captured_running",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("teardown exploded")),
    )
    try:
        outcome = qenv.tl.cancel_task_custody(task_id)
    finally:
        proc.terminate()

    assert outcome == qenv.tl.CANCEL_FAILED
    assert qenv.workers.WORKERS[0].reaping is False, "the slot must be reopened"
    # The intent stays OPEN (back to requested) so the watchdog retries.
    intent = ci.active_intent(qenv.drive, task_id)
    assert intent is not None and intent["state"] == ci.INTENT_REQUESTED


@pytest.mark.serial
def test_custody_takes_over_a_slot_stranded_by_an_abandoned_claim(qenv):
    """A-F1c: the infinite CANCEL_FAILED loop a dead custody used to cause."""
    task_id = "stranded"
    task, _child_drive, proc = _live_split_drive_task(qenv, task_id)
    write_task_result(qenv.drive, task_id, STATUS_RUNNING, result="working")
    ci.request_cancel(qenv.drive, task_id)
    ci.claim_intent(qenv.drive, task_id, owner="dead-custody")
    qenv.workers.WORKERS[0].reaping = True  # marker its owner never cleared

    # A FRESH claim is respected: no takeover, honest failure.
    assert qenv.tl.cancel_task_custody(task_id) == qenv.tl.CANCEL_FAILED

    store = qenv.drive / "state" / "cancel_intents.json"
    data = json.loads(store.read_text(encoding="utf-8"))
    data["intents"][task_id]["claim_pid"] = 2 ** 22  # the owner's process is gone
    store.write_text(json.dumps(data), encoding="utf-8")

    try:
        outcome = qenv.tl.cancel_task_custody(task_id)
    finally:
        proc.terminate()
    assert outcome == qenv.tl.CANCEL_CANCELLED
    assert load_task_result(qenv.drive, task_id)["status"] == STATUS_CANCELLED
    assert ci.active_intent(qenv.drive, task_id) is None


def test_settled_branch_recovers_a_slot_stranded_by_a_dead_custody(qenv):
    """A-F1b: the task settled on its own — nothing else revisits that worker."""
    task_id = "stranded-settled"
    respawned: list = []
    qenv.workers.WORKERS[0] = types.SimpleNamespace(
        wid=0, busy_task_id=task_id, reaping=True,
        proc=types.SimpleNamespace(pid=None, is_alive=lambda: False),
    )
    import supervisor.workers as workers_mod
    qenv_respawn = workers_mod.respawn_worker
    assert qenv_respawn is not None
    workers_mod.respawn_worker = lambda wid: respawned.append(wid)
    try:
        write_task_result(qenv.drive, task_id, STATUS_COMPLETED, result="finished")
        ci.request_cancel(qenv.drive, task_id)  # settled: no intent minted
        # Force the wedged shape: an intent whose claim owner is a dead process.
        ci.request_cancel(qenv.drive, task_id + "-x")  # keep the store non-empty
        store = qenv.drive / "state" / "cancel_intents.json"
        data = json.loads(store.read_text(encoding="utf-8"))
        data["intents"][task_id] = {
            "request_id": "ci_dead", "task_id": task_id, "state": ci.INTENT_CLAIMED,
            "claim_owner": "dead-custody", "claim_pid": 2 ** 22,
            "claimed_at": ci.utc_now_iso(), "generation": 1, "scope": "single",
            "requested_at": ci.utc_now_iso(),
        }
        store.write_text(json.dumps(data), encoding="utf-8")

        assert qenv.tl.cancel_task_custody(task_id) == qenv.tl.CANCEL_ALREADY_SETTLED
    finally:
        workers_mod.respawn_worker = qenv_respawn
    assert respawned == [0], "a dead worker behind an abandoned claim is respawned"
    assert ci.active_intent(qenv.drive, task_id) is None


def test_custody_refuses_when_the_claim_cannot_be_read(qenv, monkeypatch):
    """AR2-2: a claim attempt that RAISED cannot prove exclusivity — custody
    refuses and gives the capture back instead of settling unfenced."""
    ci.request_cancel(qenv.drive, "claim-io", reason="stop")
    qenv.q.PENDING[:] = [{"id": "claim-io", "chat_id": 1}]
    write_task_result(qenv.drive, "claim-io", "scheduled")
    monkeypatch.setattr(
        "ouroboros.cancel_intents.claim_intent",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("intent store io")),
    )

    assert qenv.tl.cancel_task_custody("claim-io") == qenv.tl.CANCEL_FAILED
    assert [t["id"] for t in qenv.q.PENDING] == ["claim-io"], "capture returned"
    assert load_task_result(qenv.drive, "claim-io")["status"] == "scheduled"


def test_custody_without_any_intent_is_the_documented_legacy_path(qenv, monkeypatch):
    """AR2-2: claim → None (no active intent) is the legacy/no-intent path —
    capture under the queue lock is the mutual exclusion and custody proceeds."""
    qenv.q.PENDING[:] = [{"id": "no-intent", "chat_id": 1}]
    write_task_result(qenv.drive, "no-intent", "scheduled")
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)

    assert qenv.tl.cancel_task_custody("no-intent") == qenv.tl.CANCEL_CANCELLED
    assert load_task_result(qenv.drive, "no-intent")["status"] == STATUS_CANCELLED


def test_two_concurrent_custodies_on_a_pending_task_settle_exactly_once(qenv, monkeypatch):
    """GR2-2 (sol's repro shape): two threads racing custody over one pending
    task used to produce TWO cancelled writes and TWO task_done events — the
    loser entered the miss lane before the winner claimed. Claim-before-capture
    makes exactly one settle owner in every interleaving."""
    import threading

    ci.request_cancel(qenv.drive, "race-2t", reason="stop")
    qenv.q.PENDING[:] = [{"id": "race-2t", "chat_id": 1}]
    write_task_result(qenv.drive, "race-2t", "scheduled")
    done_events: list = []
    monkeypatch.setattr(
        qenv.q, "_emit_cancel_task_done",
        lambda t, tid, **kw: done_events.append(tid),
    )
    barrier = threading.Barrier(2)
    outcomes: list = []

    def _run():
        barrier.wait()
        outcomes.append(qenv.tl.cancel_task_custody("race-2t"))

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert outcomes.count(qenv.tl.CANCEL_CANCELLED) == 1, outcomes
    assert done_events == ["race-2t"], "exactly ONE task_done"
    assert load_task_result(qenv.drive, "race-2t")["status"] == STATUS_CANCELLED
    assert ci.active_intent(qenv.drive, "race-2t") is None
    assert qenv.q.PENDING == [], "the loser must not re-insert the captured row"


def test_double_takeover_loser_restores_the_reaping_marker_as_found(qenv, monkeypatch):
    """AR2-11 (fable probe: two custodies over one abandoned claim): the LOSER'S
    refused-claim restore must put the reaping marker back exactly as found —
    blanking it would hand the winner's mid-kill process to assignment."""
    task_id = "double-takeover"
    worker = types.SimpleNamespace(
        wid=0, busy_task_id=task_id, reaping=True,  # marker left by the dead custody
        proc=types.SimpleNamespace(pid=None, is_alive=lambda: True,
                                   join=lambda timeout=None: None,
                                   terminate=lambda: None),
    )
    qenv.workers.WORKERS[0] = worker
    qenv.q.RUNNING[task_id] = {"task": {"id": task_id, "chat_id": 1}, "worker_id": 0}
    write_task_result(qenv.drive, task_id, STATUS_RUNNING, result="working")
    ci.request_cancel(qenv.drive, task_id)
    # The on-disk claim is ABANDONED (dead pid): the takeover gate passes.
    store = qenv.drive / "state" / "cancel_intents.json"
    data = json.loads(store.read_text(encoding="utf-8"))
    data["intents"][task_id].update({
        "state": ci.INTENT_CLAIMED, "claim_owner": "dead-custody",
        "claim_pid": 2 ** 22, "claimed_at": ci.utc_now_iso(), "generation": 3,
    })
    store.write_text(json.dumps(data), encoding="utf-8")
    # ...but the WINNER claims in the window between this loser's capture and its
    # own claim: the claim comes back REFUSED.
    refused = {**data["intents"][task_id], "claim_refused": True}
    monkeypatch.setattr("ouroboros.cancel_intents.claim_intent",
                        lambda *_a, **_kw: refused)

    assert qenv.tl.cancel_task_custody(task_id) == qenv.tl.CANCEL_FAILED
    assert qenv.workers.WORKERS[0].reaping is True, (
        "the loser must restore the marker as found — the winner is mid-kill behind it"
    )

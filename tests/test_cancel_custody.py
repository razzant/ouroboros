"""Custody: the one settle owner, and the reaping slot it must never strand.

Split out of ``tests/test_cancel_intents_phase_a.py`` by theme: custody over a task that
is neither queued nor running, the watchdog sweep that feeds it open and stale claimed
intents, and every path where a slot could be left stranded — a raising teardown, an
abandoned claim, a dead custody, two concurrent custodies, and a losing takeover.

The v7next tip adds the upstream retry-race custody family (bea08137 "Harden
cancellation custody across retry races"). The idle rail no longer mints retries
(#1376); a stored pre-upgrade retry keeps its physical-leaf/logical-root boundary, and
the hard-timeout rail still yields to a Stop that owns its logical root.
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

from tests._cancel_intents_shared import _live_split_drive_task, _write_root_retry_pair

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
    monkeypatch.setattr(qenv.q, "cancel_task_custody",
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

def _root_retry_task(task_id: str) -> dict:
    return {
        "id": task_id,
        "type": "task",
        "chat_id": 0,
        "depth": 0,
        "root_task_id": task_id,
        "parent_task_id": "",
        "delegation_role": "root",
    }


def _historical_root_retry(qenv, old_id: str, new_id: str) -> dict:
    """The stored pair and restored leaf row a pre-#1376 idle-timeout retry of a root left."""
    _write_root_retry_pair(qenv.drive, old_id, new_id)
    admitted = qenv.q.enqueue_task(
        {**_root_retry_task(old_id), "id": new_id, "original_task_id": old_id,
         "timeout_retry_from": old_id, "_attempt": 2},
        front=True, restoring_snapshot=True,
    )
    assert not admitted.get("_admission_blocked"), admitted
    return admitted

def test_a_stored_retry_leaf_takes_a_late_stop_on_its_predecessor(qenv):
    from ouroboros.task_status import load_effective_task_result

    old_id, new_id = "admit-before-old", "admit-before-new"
    _historical_root_retry(qenv, old_id, new_id)
    assert [row["id"] for row in qenv.q.PENDING] == [new_id]
    intent = ci.request_cancel(qenv.drive, old_id, reason="late stop")
    assert intent["task_id"] == new_id
    effective = load_effective_task_result(qenv.drive, old_id)
    assert effective["task_id"] == new_id
    assert effective["cancel_state"] == "pending"

    assert qenv.tl.cancel_task_custody(new_id) == qenv.tl.CANCEL_CANCELLED

    assert qenv.q.PENDING == []
    assert load_task_result(qenv.drive, new_id)["status"] == STATUS_CANCELLED
    assert load_effective_task_result(qenv.drive, old_id)["status"] == STATUS_CANCELLED
    assert ci.active_intents(qenv.drive) == {}

@pytest.mark.parametrize("stopped", [True, False], ids=["root_cascade", "no_stop_control"])
def test_timeout_precheck_yields_retry_leaf_to_logical_root_cascade(
    qenv, monkeypatch, stopped,
):
    """The leaf of a stored retry chain reaches its absolute ceiling. A cascade Stop on its
    logical root owns the kill (``_validated_retry_root_cancel_key``): no reap job, the row
    stays for cancellation custody. Without that Stop the same bound queues the reap."""
    from supervisor import workers

    now = 20_000.0
    root_id, leaf_id = "precheck-root", "precheck-leaf"
    _write_root_retry_pair(
        qenv.drive, root_id, leaf_id, new_status=STATUS_RUNNING,
    )
    leaf_task = {
        "id": leaf_id,
        "type": "task",
        "chat_id": 0,
        "depth": 0,
        "root_task_id": root_id,
        "parent_task_id": "",
        "delegation_role": "root",
        "original_task_id": root_id,
        "timeout_retry_from": root_id,
    }
    qenv.q.RUNNING[leaf_id] = {
        "task": leaf_task,
        "started_at": now - 1000.0,
        "last_heartbeat_at": now - 1000.0,
        "last_progress_at": now - 1000.0,
        "attempt": 2,
        "worker_id": -1,
    }
    monkeypatch.setattr(qenv.q, "FINALIZATION_GRACE_SEC", 0.0)
    monkeypatch.setattr(qenv.q, "get_task_idle_timeout_sec", lambda: 60.0)
    monkeypatch.setattr(qenv.q, "get_per_call_timeout_ceiling_sec", lambda: 0.0)
    monkeypatch.setattr(qenv.q, "get_task_abs_ceiling_sec", lambda: 500.0)  # it ran 1000 s
    monkeypatch.setattr(qenv.q, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(qenv.q, "persist_queue_snapshot", lambda reason="": True)
    jobs: list[dict] = []
    monkeypatch.setattr(qenv.q, "_reap_queue", types.SimpleNamespace(put=jobs.append))
    monkeypatch.setattr(workers, "WORKERS", {})
    if stopped:
        ci.request_cancel(
            qenv.drive,
            root_id,
            reason="stop every retry",
            scope=ci.SCOPE_CASCADE,
        )

    qenv.q._enforce_task_timeouts_locked(workers, now, 0, {})

    if stopped:
        assert jobs == []
        assert leaf_id in qenv.q.RUNNING
    else:
        [job] = jobs
        assert (job["task_id"], job["terminal_reason"], job["ceiling_reached"]) == (
            leaf_id, "absolute_ceiling", True)
        assert leaf_id not in qenv.q.RUNNING

def test_same_id_timeout_retry_cancels_exactly(qenv):
    task_id = "same-id-timeout-child"
    task = {
        "id": task_id,
        "chat_id": 0,
        "root_task_id": "root",
        "parent_task_id": "root",
        "delegation_role": "subagent",
        "timeout_retry_from": task_id,
        "original_task_id": task_id,
    }
    qenv.q.PENDING[:] = [task]
    write_task_result(
        qenv.drive,
        task_id,
        "interrupted",
        retry_task_id=task_id,
        root_task_id="root",
        parent_task_id="root",
        delegation_role="subagent",
    )
    ci.request_cancel(qenv.drive, task_id)

    assert qenv.tl.cancel_task_custody(task_id) == qenv.tl.CANCEL_CANCELLED
    assert qenv.q.PENDING == []
    assert load_task_result(qenv.drive, task_id)["status"] == STATUS_CANCELLED

def test_retry_leaf_completion_between_request_and_custody_wins(qenv):
    from ouroboros.task_status import load_effective_task_result

    old_id, new_id = "completion-race-old", "completion-race-new"
    _historical_root_retry(qenv, old_id, new_id)
    intent = ci.request_cancel(qenv.drive, old_id, reason="late stop")
    assert intent["task_id"] == new_id
    qenv.q.PENDING.clear()
    write_task_result(qenv.drive, new_id, STATUS_COMPLETED, result="won the race")

    assert qenv.tl.cancel_task_custody(new_id) == qenv.tl.CANCEL_ALREADY_SETTLED

    assert load_task_result(qenv.drive, old_id)["status"] == "interrupted"
    assert load_task_result(qenv.drive, new_id)["status"] == STATUS_COMPLETED
    assert load_effective_task_result(qenv.drive, old_id)["status"] == STATUS_COMPLETED
    assert ci.active_intents(qenv.drive) == {}

def test_graceful_single_retry_targets_leaf_and_stop_now_hardens_same_intent(
    qenv, monkeypatch,
):
    from supervisor import owner_stop
    from supervisor import task_reaper as tr

    old_id, new_id = "graceful-old", "graceful-new"
    _historical_root_retry(qenv, old_id, new_id)
    retry_task = qenv.q.PENDING.pop()
    qenv.q.RUNNING[new_id] = {
        "task": retry_task,
        "worker_id": 0,
        "attempt": 2,
    }
    monkeypatch.setattr(qenv.q, "FINALIZATION_GRACE_SEC", 120.0)
    armed: list[tuple[str, str]] = []

    def _arm(_drive, task_id, _reason, **kwargs):
        armed.append((task_id, str(kwargs.get("control_msg_id") or "")))
        return str(kwargs.get("control_msg_id") or "")

    monkeypatch.setattr(tr, "request_finalization_grace", _arm)
    graceful = ci.request_cancel(
        qenv.drive,
        old_id,
        reason="wrap up",
        requested_stop_policy=ci.STOP_POLICY_FINALIZE,
    )
    assert graceful["task_id"] == new_id

    owner_stop.begin_graceful_stop(new_id)

    assert armed and armed[0][0] == new_id
    assert qenv.q.RUNNING[new_id]["finalization_control_msg_id"].startswith(
        "ownerstop:ci_"
    )
    hardened = ci.request_cancel(
        qenv.drive,
        old_id,
        reason="stop now",
        requested_stop_policy=ci.STOP_POLICY_IMMEDIATE,
    )
    assert hardened["task_id"] == new_id
    assert hardened["request_id"] == graceful["request_id"]
    assert ci.stop_policy(ci.active_intent(qenv.drive, new_id)) == ci.STOP_POLICY_IMMEDIATE
    assert ci.active_intent(qenv.drive, old_id) is None

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
    monkeypatch.setattr(
        qenv.tl, "_finish_captured_running",
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

def test_custody_without_any_intent_is_a_typed_noop(qenv, monkeypatch):
    """A2 (invariant 14): claim → None (no active intent) is NO authority. Custody
    captures nothing, kills nothing and writes nothing; the answer is typed from the
    stored result (``not_found`` for a live task, ``already_settled`` for a settled one)."""
    qenv.q.PENDING[:] = [{"id": "no-intent", "chat_id": 1}]
    write_task_result(qenv.drive, "no-intent", "scheduled")
    emitted: list = []
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: emitted.append(_a))

    assert qenv.tl.cancel_task_custody("no-intent") == qenv.tl.CANCEL_NOT_FOUND
    assert [t["id"] for t in qenv.q.PENDING] == ["no-intent"]
    assert load_task_result(qenv.drive, "no-intent")["status"] == "scheduled"
    assert emitted == []
    write_task_result(qenv.drive, "no-intent", "completed")
    assert qenv.tl.cancel_task_custody("no-intent") == qenv.tl.CANCEL_ALREADY_SETTLED
    assert [t["id"] for t in qenv.q.PENDING] == ["no-intent"]

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


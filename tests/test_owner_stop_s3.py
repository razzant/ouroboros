"""S3 graceful owner stop ("Wrap up") test matrix — Q1/Q2/Q3=A/Q4=A/Q6=A.

Covers the policy axis end to end against production seams: the typed
``stop_policy`` vocabulary and its monotonic hardening in
``ouroboros/cancel_intents.py``; the graceful HTTP ingress (immediate 202
pending acknowledgement, no synchronous teardown); the policy-aware episode
predicates and orchestration in ``supervisor/owner_stop.py`` (deterministic
``ownerstop:<request_id>`` control identity, idempotent arming, custody feed on
settle/expiry/pending); the Q4=A summary suppression; the Q6=A bounded child
projection; and the reload-visible ``stop_policy`` projection in
``cancel_state_fields``. Absence of the policy stays byte-identical immediate
hard cancellation (§13.1) — proven by the explicit-immediate legacy envelope.
"""

from __future__ import annotations

import json
import pathlib
import queue
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import pytest

import ouroboros.cancel_intents as ci
import supervisor.owner_stop as ostop
from ouroboros.config import OWNER_STOP_OUTER_CAP_SEC
from ouroboros.gateway.tasks import api_task_cancel
from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, _mailbox_path, write_owner_message
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import utc_now_iso


def _isolate_queue(monkeypatch, tmp_path, *, pending=(), running=None):
    from supervisor import queue as q
    from supervisor import workers

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [dict(t) for t in pending])
    monkeypatch.setattr(q, "RUNNING", dict(running or {}))
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    return q


def _client(tmp_path):
    app = Starlette(routes=[
        Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"]),
    ])
    app.state.drive_root = tmp_path
    return TestClient(app)


def _finalize_rows(drive_root, task_id):
    path = _mailbox_path(pathlib.Path(drive_root), task_id)
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("kind") == "finalize_now"
    ]


# ---------------------------------------------------------------------------
# Vocabulary, monotonic hardening, reload projection
# ---------------------------------------------------------------------------


def test_stop_policy_reads_immediate_for_absent_or_unknown():
    assert ci.stop_policy(None) == ci.STOP_POLICY_IMMEDIATE
    assert ci.stop_policy({}) == ci.STOP_POLICY_IMMEDIATE
    assert ci.stop_policy({"stop_policy": "nonsense"}) == ci.STOP_POLICY_IMMEDIATE
    assert ci.stop_policy({"stop_policy": "finalize_then_cancel"}) == ci.STOP_POLICY_FINALIZE


def test_immediate_hardens_pending_graceful_and_never_softens(tmp_path):
    graceful = ci.request_cancel(
        tmp_path, "t-h", requested_stop_policy=ci.STOP_POLICY_FINALIZE,
    )
    rid = graceful["request_id"]
    assert ci.stop_policy(graceful) == ci.STOP_POLICY_FINALIZE
    # Stop-now during the wait: the SAME durable request hardens in place.
    hardened = ci.request_cancel(
        tmp_path, "t-h", requested_stop_policy=ci.STOP_POLICY_IMMEDIATE,
    )
    assert hardened["request_id"] == rid                # single kill-owner
    assert ci.stop_policy(hardened) == ci.STOP_POLICY_IMMEDIATE
    assert hardened["hardened_at"]
    # Graceful over an accepted immediate is the forbidden softening direction.
    softened = ci.request_cancel(
        tmp_path, "t-h", requested_stop_policy=ci.STOP_POLICY_FINALIZE,
    )
    assert softened["request_id"] == rid
    assert ci.stop_policy(ci.active_intent(tmp_path, "t-h")) == ci.STOP_POLICY_IMMEDIATE
    # The hardening left a typed forensic row.
    ledger = (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    assert "stop_policy_hardened" in ledger


def test_cancel_state_fields_projects_the_graceful_policy_for_reload(tmp_path):
    ci.request_cancel(tmp_path, "t-proj", requested_stop_policy=ci.STOP_POLICY_FINALIZE)
    fields = ci.cancel_state_fields(tmp_path, "t-proj")
    assert fields["cancel_state"] == "pending"
    assert fields["stop_policy"] == ci.STOP_POLICY_FINALIZE
    # Immediate intents keep the pre-S3 shape: NO stop_policy key at all.
    ci.request_cancel(tmp_path, "t-imm")
    assert "stop_policy" not in ci.cancel_state_fields(tmp_path, "t-imm")


# ---------------------------------------------------------------------------
# HTTP ingress: 202 pending acknowledgement vs synchronous legacy teardown
# ---------------------------------------------------------------------------


def test_graceful_post_returns_202_pending_without_synchronous_teardown(tmp_path, monkeypatch):
    q = _isolate_queue(
        monkeypatch, tmp_path,
        pending=[{"id": "root-g", "chat_id": 0, "root_task_id": "root-g"}],
    )
    kicks = []
    monkeypatch.setattr(ostop, "begin_graceful_stop", lambda tid: kicks.append(tid))
    with _client(tmp_path) as client:
        resp = client.post(
            "/api/tasks/root-g/cancel", json={"stop_policy": "finalize_then_cancel"},
        )
    assert resp.status_code == 202
    assert resp.json() == {
        "ok": True, "task_id": "root-g",
        "cancel_state": "pending", "stop_policy": "finalize_then_cancel",
    }
    # The durable intent IS the whole owner will; nothing was torn down yet.
    intent = ci.active_intent(tmp_path, "root-g")
    assert ci.stop_policy(intent) == ci.STOP_POLICY_FINALIZE
    assert [t["id"] for t in q.PENDING] == ["root-g"]
    assert (load_task_result(tmp_path, "root-g") or {}).get("status") != "cancelled"
    # One orchestration pass was kicked off the HTTP thread.
    deadline = time.time() + 5
    while not kicks and time.time() < deadline:
        time.sleep(0.01)
    assert kicks == ["root-g"]


def test_bad_policy_is_400_and_explicit_immediate_keeps_the_legacy_contract(tmp_path, monkeypatch):
    _isolate_queue(
        monkeypatch, tmp_path,
        pending=[{"id": "root-i", "chat_id": 0, "root_task_id": "root-i"}],
    )
    with _client(tmp_path) as client:
        bad = client.post("/api/tasks/root-i/cancel", json={"stop_policy": "graceful"})
        typed = client.post("/api/tasks/root-i/cancel", json={"stop_policy": "immediate"})
    assert bad.status_code == 400
    # Explicit immediate stays the synchronous legacy teardown + envelope.
    assert typed.status_code == 200
    assert typed.json() == {"ok": True, "task_id": "root-i"}
    assert load_task_result(tmp_path, "root-i")["status"] == "cancelled"
    intent = ci.active_intent(tmp_path, "root-i") or {}
    assert ci.stop_policy(intent) == ci.STOP_POLICY_IMMEDIATE


def test_graceful_escalates_to_stop_now_through_the_same_intent(tmp_path, monkeypatch):
    """The owner presses "Stop now" during the graceful wait: the
    same durable request hardens and the synchronous teardown runs."""
    q = _isolate_queue(
        monkeypatch, tmp_path,
        pending=[{"id": "root-e", "chat_id": 0, "root_task_id": "root-e"}],
    )
    monkeypatch.setattr(ostop, "begin_graceful_stop", lambda tid: None)
    with _client(tmp_path) as client:
        graceful = client.post(
            "/api/tasks/root-e/cancel", json={"stop_policy": "finalize_then_cancel"},
        )
        rid = ci.active_intent(tmp_path, "root-e")["request_id"]
        hard = client.post("/api/tasks/root-e/cancel", json={})
    assert graceful.status_code == 202
    assert hard.status_code == 200
    assert q.PENDING == []
    assert load_task_result(tmp_path, "root-e")["status"] == "cancelled"
    # Same request id end to end: one stop episode, monotonically hardened.
    ledger = (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    hardened_rows = [
        json.loads(line) for line in ledger.splitlines()
        if line.strip() and "stop_policy_hardened" in line
    ]
    assert [row["request_id"] for row in hardened_rows] == [rid]


# ---------------------------------------------------------------------------
# Episode predicates and sweep orchestration
# ---------------------------------------------------------------------------


def _graceful_intent(tmp_path, task_id, **kw):
    return ci.request_cancel(
        tmp_path, task_id, requested_stop_policy=ci.STOP_POLICY_FINALIZE, **kw,
    )


def _write_root_retry_pair(tmp_path, root_id, leaf_id, *, leaf_status="running"):
    write_task_result(
        tmp_path,
        root_id,
        "interrupted",
        root_task_id=root_id,
        delegation_role="root",
        superseded_by=leaf_id,
        retry_task_id=leaf_id,
    )
    write_task_result(
        tmp_path,
        leaf_id,
        leaf_status,
        root_task_id=root_id,
        parent_task_id="",
        delegation_role="root",
        supersedes_task_id=root_id,
        original_task_id=root_id,
        timeout_retry_from=root_id,
    )


def test_owner_stop_active_matrix(tmp_path):
    intent = _graceful_intent(tmp_path, "t-act")
    now = time.time()
    assert ostop.owner_stop_active(intent, now=now, grace_sec=120.0) is True
    # A custody claim means the kill already started.
    assert ostop.owner_stop_active(
        {**intent, "state": ci.INTENT_CLAIMED}, now=now, grace_sec=120.0,
    ) is False
    # UNDRAINED (owner 1=A): the grace budget has not started ticking yet —
    # only the outer request-time cap (2=A) bounds the episode, so 121s after
    # the request the summary turn is still owed.
    assert ostop.owner_stop_active(intent, now=now + 121.0, grace_sec=120.0) is True
    # Past the outer cap the episode no longer owns the intent (the SWEEP
    # feeds custody) — but the intent stays OPEN, so the enforcement HOLD
    # deliberately keeps covering the task (MAJOR-B).
    assert ostop.owner_stop_active(
        intent, now=now + OWNER_STOP_OUTER_CAP_SEC + 1.0, grace_sec=120.0,
    ) is False
    # DRAINED: the grace budget runs from the delivery instant.
    drained = {**intent, "control_drained_at": utc_now_iso()}
    assert ostop.owner_stop_active(drained, now=now + 119.0, grace_sec=120.0) is True
    assert ostop.owner_stop_active(drained, now=now + 121.0, grace_sec=120.0) is False
    assert ostop.owner_stop_open(intent) is True
    assert ostop.owner_stop_open({**intent, "state": ci.INTENT_CLAIMED}) is False
    # Immediate intents never form an episode and are never held.
    immediate = ci.request_cancel(tmp_path, "t-act2")
    assert ostop.owner_stop_active(immediate, now=now, grace_sec=120.0) is False
    assert ostop.owner_stop_open(immediate) is False


def test_running_owner_stop_tasks_reads_the_durable_projection(tmp_path):
    _graceful_intent(tmp_path, "t-held")
    ci.request_cancel(tmp_path, "t-hard")            # immediate: not held
    held = ostop.running_owner_stop_tasks(tmp_path, grace_sec=120.0)
    assert held == {"t-held"}
    # A claimed intent releases the hold: custody owns the kill from there.
    ci.claim_intent(tmp_path, "t-held", owner="custody-test")
    assert ostop.running_owner_stop_tasks(tmp_path, grace_sec=120.0) == set()
    assert ostop.running_owner_stop_tasks(tmp_path, grace_sec=0.0) == set()


@pytest.mark.parametrize(
    ("hard_axis", "expected_reason"),
    [
        ("deadline", "deadline"),
        ("absolute_ceiling", "absolute_ceiling"),
    ],
)
def test_owner_stop_hold_never_bypasses_hard_task_axes(
    tmp_path, monkeypatch, hard_axis, expected_reason,
):
    """Owner stop replaces idle/grace, never an earlier hard task boundary."""
    from supervisor import queue as q

    task_id = f"t-{hard_axis}"
    _graceful_intent(tmp_path, task_id)
    now = time.time()
    task = {"id": task_id, "chat_id": 0, "type": "task"}
    started_at = now - 10.0
    absolute_ceiling = 10_000_000.0
    if hard_axis == "deadline":
        task["deadline_at"] = datetime.fromtimestamp(
            now - 1.0, tz=timezone.utc,
        ).isoformat()
    else:
        started_at = now - 1000.0
        absolute_ceiling = 60.0
    meta = {
        "task": task,
        "started_at": started_at,
        "last_heartbeat_at": now,
        "last_progress_at": now,
        "attempt": 1,
    }
    q_isolated = _isolate_queue(
        monkeypatch, tmp_path, running={task_id: meta},
    )
    monkeypatch.setattr(q, "FINALIZATION_GRACE_SEC", 120.0, raising=False)
    monkeypatch.setattr(q, "get_task_abs_ceiling_sec", lambda: absolute_ceiling)
    requested = []
    monkeypatch.setattr(
        q,
        "_request_finalization_grace",
        lambda _drive, tid, reason, **_kwargs: requested.append((tid, reason))
        or f"hard:{tid}",
    )

    intent = ci.active_intent(tmp_path, task_id)
    assert intent is not None
    assert ostop.sweep_owner_stop_hold(
        q_isolated, task_id, intent, now=now,
    ) is False

    q_isolated._enforce_task_timeouts_locked(None, now, 0, {})

    assert requested == [(task_id, expected_reason)]
    assert meta["finalization_reason"] == expected_reason
    assert meta["finalization_control_msg_id"] == f"hard:{task_id}"


def test_owner_stop_loop_deadline_can_only_narrow_an_explicit_task_deadline(
    tmp_path, monkeypatch,
):
    import ouroboros.loop as loop_mod

    task_id = "t-loop-hard-deadline"
    intent = _graceful_intent(tmp_path, task_id)
    task_deadline = time.time() + 5.0
    owner_deadline = ostop.owner_stop_deadline_ts(intent, 120.0)
    assert owner_deadline > task_deadline
    ctx = SimpleNamespace(
        status_drive_root=tmp_path,
        drive_root=tmp_path,
        task_id=task_id,
        deadline_ts=task_deadline,
    )
    monkeypatch.setattr(
        loop_mod.task_pacing, "get_finalization_grace_sec", lambda: 120.0,
    )
    monkeypatch.setattr(loop_mod.time, "time", lambda: task_deadline + 1.0)

    assert loop_mod._owner_stop_window_elapsed(ctx) is True
    assert ctx.deadline_ts == task_deadline
    assert loop_mod._narrow_round_deadline(ctx, owner_deadline) == task_deadline


def _expiry_enforcement_queue(monkeypatch, tmp_path, task_id, meta):
    """An isolated queue tuned so the generic rail WOULD reap/retry the task
    if it were not held: small idle timeout, huge ceiling, captured reap jobs."""
    from supervisor import queue as q
    from supervisor import workers

    q_isolated = _isolate_queue(monkeypatch, tmp_path, running={task_id: meta})
    monkeypatch.setattr(q, "FINALIZATION_GRACE_SEC", 120.0, raising=False)
    monkeypatch.setattr(q, "get_task_idle_timeout_sec", lambda: 60.0)
    monkeypatch.setattr(q, "get_per_call_timeout_ceiling_sec", lambda: 0.0)
    monkeypatch.setattr(q, "get_task_abs_ceiling_sec", lambda: 10_000_000.0)
    monkeypatch.setattr(q, "_ensure_reaper_started", lambda: None)
    jobs = []
    monkeypatch.setattr(q, "_reap_queue", SimpleNamespace(put=jobs.append))
    return q_isolated, workers, jobs


def test_tool_call_spanning_expiry_keeps_the_episode_whole(tmp_path, monkeypatch):
    """MAJOR-B trace (a): a PROGRESSING task whose tool call spans the old
    request+grace horizon must not have its episode withdrawn — no control
    revocation, no false retraction toast — and (owner 1=A) the sweep keeps
    HOLDING while the summary-turn control is undrained, so the late drain
    still buys the bounded final turn; custody kills only past the effective
    (drain-anchored, outer-capped) deadline."""
    from supervisor import workers

    toasts = []
    monkeypatch.setattr(
        workers, "get_event_q", lambda: SimpleNamespace(put=toasts.append),
    )
    intent = _graceful_intent(tmp_path, "t-span")
    now = time.time()
    meta = {
        "task": {"id": "t-span", "chat_id": 0, "type": "task"},
        "started_at": now - 300.0,
        "last_heartbeat_at": now,
        "attempt": 1,
    }
    q_isolated, workers_mod, jobs = _expiry_enforcement_queue(
        monkeypatch, tmp_path, "t-span", meta,
    )
    # Arm the real episode (control + latch) through the production sweep.
    assert ostop.sweep_owner_stop_hold(q_isolated, "t-span", intent, now=now) is True
    control_id = ostop.owner_stop_control_id(intent)
    assert meta["finalization_control_msg_id"] == control_id
    armed_toasts = len(toasts)
    # A tool call is still producing progress as the deadline (requested+120s)
    # passes; the enforcement tick runs 50s past expiry.
    expiry = now + 170.0
    meta["last_progress_at"] = expiry - 1.0
    q_isolated._enforce_task_timeouts_locked(workers_mod, expiry, 0, {})
    # Held whole: no withdraw (latch + control intact), no retraction toast,
    # no reap job, the task still RUNNING.
    assert q_isolated.RUNNING["t-span"] is meta
    assert meta["finalization_control_msg_id"] == control_id
    assert meta["finalization_reason"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert len(toasts) == armed_toasts
    assert jobs == []
    rows = _finalize_rows(tmp_path, "t-span")
    assert len(rows) == 1 and rows[0]["msg_id"] == control_id
    # The summary-turn control is still DELIVERABLE (never revoked): a drain
    # yields it, so the loop's owner-stop rail can produce the final answer.
    from ouroboros.owner_mailbox import drain_owner_entries

    drained = drain_owner_entries(tmp_path, "t-span")
    assert [e["msg_id"] for e in drained] == [control_id]
    # THE LIVE DEFECT'S FIX (owner 1=A): 50s past the old request+120 horizon
    # the control is STILL undrained (the tool call is in flight) — the sweep
    # keeps holding instead of feeding custody, so the final turn stays owed.
    assert ostop.sweep_owner_stop_hold(q_isolated, "t-span", intent, now=expiry) is True
    # The loop finally drains the control as the tool call ends: the grace
    # budget starts HERE, and custody becomes the killer only 120s later.
    drain_iso = datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat()
    assert ci.mark_finalize_control_drained(tmp_path, "t-span", drained_at=drain_iso) is True
    updated = ci.active_intent(tmp_path, "t-span")
    assert ostop.sweep_owner_stop_hold(q_isolated, "t-span", updated, now=expiry + 119.0) is True
    assert ostop.sweep_owner_stop_hold(q_isolated, "t-span", updated, now=expiry + 121.0) is False


def test_non_progressing_task_at_expiry_is_not_reaped_or_cloned(tmp_path, monkeypatch):
    """MAJOR-B trace (b): a NON-progressing task at episode expiry must not be
    idle_timeout-reaped by the generic rail nor cloned into a new-id retry that
    escapes the intent; the owner-requested reason stays with custody."""
    intent = _graceful_intent(tmp_path, "t-idle")
    now = time.time()
    requested = ostop._requested_ts(intent)
    meta = {
        "task": {"id": "t-idle", "chat_id": 0, "type": "task"},
        "started_at": now - 1000.0,
        "last_heartbeat_at": now - 500.0,
        "last_progress_at": now - 500.0,               # idle >> 60s idle timeout
        "attempt": 1,
        "finalization_requested_at": requested,        # armed episode latch
        "finalization_reason": REASON_OWNER_REQUESTED_FINALIZATION,
        "finalization_control_msg_id": ostop.owner_stop_control_id(intent),
    }
    q_isolated, workers_mod, jobs = _expiry_enforcement_queue(
        monkeypatch, tmp_path, "t-idle", meta,
    )
    q_isolated._enforce_task_timeouts_locked(workers_mod, requested + 170.0, 0, {})
    # No idle_timeout terminal, no reap job, no new-id retry clone.
    assert q_isolated.RUNNING["t-idle"] is meta
    assert jobs == []
    assert q_isolated.PENDING == []
    assert (load_task_result(tmp_path, "t-idle") or {}).get("status") != "cancelled"
    # The owner-requested intent is still the one will custody settles.
    live = ci.active_intent(tmp_path, "t-idle")
    assert ci.stop_policy(live) == ci.STOP_POLICY_FINALIZE


def test_active_intent_keeps_timeout_reaper_from_cloning_task(tmp_path, monkeypatch):
    """Cancellation custody, not the generic timeout reaper, owns the task."""
    intent = _graceful_intent(tmp_path, "t-claimed")
    assert ci.claim_intent(tmp_path, "t-claimed", owner="custody-test")
    now = time.time()
    meta = {
        "task": {"id": "t-claimed", "chat_id": 0, "type": "task"},
        "started_at": now - 1000.0,
        "last_heartbeat_at": now - 500.0,
        "last_progress_at": now - 500.0,               # idle: would-be retry shape
        "attempt": 1,
        "finalization_requested_at": now - 300.0,      # grace long elapsed
        "finalization_reason": "idle_timeout",
        "finalization_control_msg_id": ostop.owner_stop_control_id(intent),
    }
    q_isolated, workers_mod, jobs = _expiry_enforcement_queue(
        monkeypatch, tmp_path, "t-claimed", meta,
    )
    q_isolated._enforce_task_timeouts_locked(workers_mod, now, 0, {})
    assert jobs == []
    assert q_isolated.RUNNING["t-claimed"] is meta
    assert q_isolated.PENDING == []


def _fake_queue(tmp_path, running):
    return SimpleNamespace(
        _queue_lock=threading.Lock(),
        RUNNING=running,
        PENDING=[],
        DRIVE_ROOT=tmp_path,
        FINALIZATION_GRACE_SEC=120.0,
        _task_drive_for_task=lambda task, tid: tmp_path,
        _task_deadline_ts=lambda task: 0.0,
        get_task_abs_ceiling_sec=lambda: 10_000_000.0,
    )


def test_sweep_hold_arms_the_episode_idempotently(tmp_path, monkeypatch):
    from supervisor import workers

    toasts = []
    monkeypatch.setattr(
        workers, "get_event_q", lambda: SimpleNamespace(put=toasts.append),
    )
    intent = _graceful_intent(tmp_path, "t-arm")
    running = {"t-arm": {"task": {"id": "t-arm", "chat_id": 5}, "started_at": time.time()}}
    q = _fake_queue(tmp_path, running)
    assert ostop.sweep_owner_stop_hold(q, "t-arm", intent, now=time.time()) is True
    control_id = ostop.owner_stop_control_id(intent)
    assert control_id == f"ownerstop:{intent['request_id']}"
    # The coupled control + RUNNING latch both carry the deterministic identity.
    assert running["t-arm"]["finalization_control_msg_id"] == control_id
    assert running["t-arm"]["finalization_reason"] == REASON_OWNER_REQUESTED_FINALIZATION
    rows = _finalize_rows(tmp_path, "t-arm")
    assert len(rows) == 1
    assert rows[0]["msg_id"] == control_id
    assert rows[0]["text"].startswith(REASON_OWNER_REQUESTED_FINALIZATION)
    # The owner-facing toast replaced the generic reached-terminal wording.
    assert len(toasts) == 1
    assert toasts[0]["chat_id"] == 5 and toasts[0]["is_progress"] is True
    assert "summarize and stop" in toasts[0]["text"]
    assert "Stop now remains available" in toasts[0]["text"]
    # A watchdog/restart replay re-arms the SAME id: no duplicate control/toast.
    assert ostop.sweep_owner_stop_hold(q, "t-arm", intent, now=time.time()) is True
    assert len(_finalize_rows(tmp_path, "t-arm")) == 1
    assert len(toasts) == 1


def test_stop_now_revokes_the_queued_owner_finalization_control(
    tmp_path, monkeypatch,
):
    from ouroboros.owner_mailbox import (
        KIND_CONTROL_REVOKED,
        drain_owner_entries,
    )
    from supervisor import task_lifecycle as lifecycle
    from supervisor import workers

    task_id = "t-hardened-control"
    meta = {
        "task": {"id": task_id, "chat_id": 0, "type": "task"},
        "started_at": time.time(),
    }
    q = _isolate_queue(monkeypatch, tmp_path, running={task_id: meta})
    monkeypatch.setattr(q, "FINALIZATION_GRACE_SEC", 120.0, raising=False)
    monkeypatch.setattr(
        workers, "get_event_q", lambda: SimpleNamespace(put=lambda _event: None),
    )
    graceful = _graceful_intent(tmp_path, task_id)
    assert ostop.sweep_owner_stop_hold(q, task_id, graceful, now=time.time()) is True
    control_id = ostop.owner_stop_control_id(graceful)
    assert meta["finalization_control_msg_id"] == control_id

    hardened = ci.request_cancel(
        tmp_path,
        task_id,
        requested_stop_policy=ci.STOP_POLICY_IMMEDIATE,
    )
    assert hardened["request_id"] == graceful["request_id"]
    monkeypatch.setattr(
        q,
        "cancel_task_custody",
        lambda _task_id, **_kwargs: q.CANCEL_CANCELLED,
    )

    assert lifecycle.drive_cancel_intent_scope(task_id) == q.CANCEL_CANCELLED
    assert "finalization_requested_at" not in meta
    assert "finalization_reason" not in meta
    assert "finalization_control_msg_id" not in meta
    assert drain_owner_entries(tmp_path, task_id) == []
    mailbox_rows = [
        json.loads(line)
        for line in _mailbox_path(tmp_path, task_id).read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]
    assert [
        row["text"]
        for row in mailbox_rows
        if row.get("kind") == KIND_CONTROL_REVOKED
    ] == [control_id]


def test_graceful_cascade_arms_and_drains_the_physical_retry_leaf(
    tmp_path, monkeypatch,
):
    """A logical cascade intent owns its current physical root retry attempt."""
    from supervisor import workers

    root_id, leaf_id, child_id = "retry-root", "retry-leaf", "retry-child"
    _write_root_retry_pair(tmp_path, root_id, leaf_id)
    write_task_result(
        tmp_path,
        child_id,
        "scheduled",
        root_task_id=root_id,
        parent_task_id=leaf_id,
        delegation_role="subagent",
    )
    running = {
        leaf_id: {
            "task": {
                "id": leaf_id,
                "chat_id": 0,
                "type": "task",
                "root_task_id": root_id,
                "parent_task_id": "",
                "delegation_role": "root",
                "original_task_id": root_id,
                "timeout_retry_from": root_id,
            },
            "worker_id": -1,
            "attempt": 2,
        },
    }
    q = _isolate_queue(
        monkeypatch,
        tmp_path,
        pending=[{
            "id": child_id,
            "chat_id": 0,
            "type": "task",
            "root_task_id": root_id,
            "parent_task_id": leaf_id,
            "delegation_role": "subagent",
        }],
        running=running,
    )
    monkeypatch.setattr(q, "FINALIZATION_GRACE_SEC", 120.0, raising=False)
    monkeypatch.setattr(
        workers, "get_event_q", lambda: SimpleNamespace(put=lambda _event: None),
    )
    intent = _graceful_intent(
        tmp_path,
        root_id,
        scope=ci.SCOPE_CASCADE,
    )

    assert ostop.sweep_owner_stop_hold(q, root_id, intent, now=time.time()) is True

    assert ci.active_intent(tmp_path, root_id)["request_id"] == intent["request_id"]
    assert ci.active_intent(tmp_path, leaf_id) is None
    assert ostop.running_owner_stop_tasks(tmp_path, grace_sec=120.0) == {
        root_id,
        leaf_id,
    }
    assert leaf_id in q.RUNNING
    assert q.PENDING == []
    assert load_task_result(tmp_path, child_id)["status"] == "cancelled"
    control_id = ostop.owner_stop_control_id(intent)
    assert q.RUNNING[leaf_id]["finalization_control_msg_id"] == control_id
    assert [row["msg_id"] for row in _finalize_rows(tmp_path, leaf_id)] == [control_id]

    # The production mailbox drain runs under the PHYSICAL id, but stamps the
    # durable intent under the logical root. The worker deadline lookup uses
    # the same reverse resolution.
    controls = _production_drain(tmp_path, leaf_id)
    assert controls["finalize_now"].startswith(REASON_OWNER_REQUESTED_FINALIZATION)
    updated = ci.active_intent(tmp_path, root_id)
    assert updated["control_drained_at"]
    import ouroboros.loop as loop_mod

    deadline = ostop.owner_stop_deadline_ts(updated, 120.0)
    limit_ctx = SimpleNamespace(
        status_drive_root=tmp_path,
        drive_root=tmp_path,
        task_id=leaf_id,
        deadline_ts=None,
    )
    monkeypatch.setattr(loop_mod.task_pacing, "get_finalization_grace_sec", lambda: 120.0)
    monkeypatch.setattr(loop_mod.time, "time", lambda: deadline + 1.0)
    assert loop_mod._owner_stop_window_elapsed(limit_ctx) is True
    assert limit_ctx.deadline_ts == deadline


def test_descendant_custody_failure_never_arms_the_root_final_turn(
    tmp_path, monkeypatch,
):
    root_id, child_id = "blocked-root", "blocked-child"
    root_meta = {
        "task": {"id": root_id, "chat_id": 0, "type": "task"},
        "started_at": time.time(),
    }
    child = {
        "id": child_id,
        "chat_id": 0,
        "type": "task",
        "root_task_id": root_id,
        "parent_task_id": root_id,
        "delegation_role": "subagent",
    }
    q = _isolate_queue(
        monkeypatch,
        tmp_path,
        pending=[child],
        running={root_id: root_meta},
    )
    monkeypatch.setattr(q, "FINALIZATION_GRACE_SEC", 120.0, raising=False)
    custody_calls = []

    def _refuse_child(task_id, **_kwargs):
        custody_calls.append(task_id)
        return q.CANCEL_FAILED

    monkeypatch.setattr(q, "cancel_task_custody", _refuse_child)
    intent = _graceful_intent(tmp_path, root_id, scope=ci.SCOPE_CASCADE)

    assert ostop.sweep_owner_stop_hold(q, root_id, intent, now=time.time()) is True

    assert custody_calls and set(custody_calls) == {child_id}
    assert [task["id"] for task in q.PENDING] == [child_id]
    assert "finalization_control_msg_id" not in root_meta
    assert _finalize_rows(tmp_path, root_id) == []
    forensic = (tmp_path / "logs" / "supervisor.jsonl").read_text(
        encoding="utf-8",
    )
    assert "owner_stop_descendants_pending" in forensic


def test_late_descendant_is_reswept_before_exactly_one_root_control(
    tmp_path, monkeypatch,
):
    from supervisor import workers

    root_id = "resweep-root"
    first_id, late_id = "resweep-first", "resweep-late"
    root_meta = {
        "task": {"id": root_id, "chat_id": 0, "type": "task"},
        "started_at": time.time(),
    }

    def _child(task_id):
        return {
            "id": task_id,
            "chat_id": 0,
            "type": "task",
            "root_task_id": root_id,
            "parent_task_id": root_id,
            "delegation_role": "subagent",
        }

    q = _isolate_queue(
        monkeypatch,
        tmp_path,
        pending=[_child(first_id)],
        running={root_id: root_meta},
    )
    monkeypatch.setattr(q, "FINALIZATION_GRACE_SEC", 120.0, raising=False)
    monkeypatch.setattr(
        workers, "get_event_q", lambda: SimpleNamespace(put=lambda _event: None),
    )
    real_custody = q.cancel_task_custody
    custody_calls = []
    late_injected = []

    def _custody(task_id, **kwargs):
        custody_calls.append(task_id)
        outcome = real_custody(task_id, **kwargs)
        if task_id == first_id and not late_injected:
            with q._queue_lock:
                q.PENDING.append(_child(late_id))
            late_injected.append(late_id)
        return outcome

    monkeypatch.setattr(q, "cancel_task_custody", _custody)
    intent = _graceful_intent(tmp_path, root_id, scope=ci.SCOPE_CASCADE)

    assert ostop.sweep_owner_stop_hold(q, root_id, intent, now=time.time()) is True

    assert custody_calls == [first_id, late_id]
    assert q.PENDING == []
    assert load_task_result(tmp_path, first_id)["status"] == "cancelled"
    assert load_task_result(tmp_path, late_id)["status"] == "cancelled"
    control_id = ostop.owner_stop_control_id(intent)
    assert root_meta["finalization_control_msg_id"] == control_id
    assert [row["msg_id"] for row in _finalize_rows(tmp_path, root_id)] == [
        control_id,
    ]
    assert ostop.sweep_owner_stop_hold(q, root_id, intent, now=time.time()) is True
    assert len(_finalize_rows(tmp_path, root_id)) == 1


def test_completed_retry_leaf_settles_logical_cascade_without_duplicate_summary(
    tmp_path, monkeypatch,
):
    from supervisor import task_lifecycle as lifecycle

    root_id, leaf_id = "complete-root", "complete-leaf"
    _write_root_retry_pair(tmp_path, root_id, leaf_id)
    _isolate_queue(monkeypatch, tmp_path)
    intent = _graceful_intent(tmp_path, root_id, scope=ci.SCOPE_CASCADE)
    write_task_result(tmp_path, leaf_id, "completed", result="final answer")
    summaries = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.deliver_cascade_summary",
        lambda *_args, **_kwargs: summaries.append(1) or True,
    )

    outcomes = lifecycle.sweep_cancel_intents(
        now=ostop._requested_ts(intent) + 11.0,
    )

    assert outcomes == {root_id: lifecycle.CANCEL_CANCELLED}
    assert summaries == []
    assert ci.active_intent(tmp_path, root_id) is None
    assert load_task_result(tmp_path, leaf_id)["status"] == "completed"
    assert not lifecycle.task_subtree_is_live(root_id, ignore_intents=True)
    forensic = (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    assert "owner_stop_summary_suppressed" in forensic


def test_stop_now_hardens_logical_cascade_and_cancels_retry_leaf_once(
    tmp_path, monkeypatch,
):
    from supervisor import task_lifecycle as lifecycle

    root_id, leaf_id = "harden-root", "harden-leaf"
    _write_root_retry_pair(tmp_path, root_id, leaf_id, leaf_status="scheduled")
    q = _isolate_queue(
        monkeypatch,
        tmp_path,
        pending=[{
            "id": leaf_id,
            "chat_id": 0,
            "type": "task",
            "root_task_id": root_id,
            "parent_task_id": "",
            "delegation_role": "root",
            "original_task_id": root_id,
            "timeout_retry_from": root_id,
        }],
    )
    graceful = _graceful_intent(tmp_path, root_id, scope=ci.SCOPE_CASCADE)
    hardened = ci.request_cancel(
        tmp_path,
        root_id,
        scope=ci.SCOPE_CASCADE,
        requested_stop_policy=ci.STOP_POLICY_IMMEDIATE,
    )
    assert hardened["request_id"] == graceful["request_id"]
    calls = []
    real_custody = q.cancel_task_custody

    def _custody(task_id, **kwargs):
        calls.append(task_id)
        return real_custody(task_id, **kwargs)

    monkeypatch.setattr(q, "cancel_task_custody", _custody)
    summaries = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.deliver_cascade_summary",
        lambda *_args, **_kwargs: summaries.append(1) or True,
    )

    outcomes = lifecycle.sweep_cancel_intents(
        now=ostop._requested_ts(hardened) + 11.0,
    )

    assert outcomes == {root_id: lifecycle.CANCEL_CANCELLED}
    assert calls.count(leaf_id) == 1
    assert summaries == [1]
    assert q.PENDING == []
    assert load_task_result(tmp_path, leaf_id)["status"] == "cancelled"
    assert ci.active_intent(tmp_path, root_id) is None


def test_corrupt_retry_lineage_holds_until_grace_deadline_then_releases_hard_path(
    tmp_path, monkeypatch,
):
    from supervisor import task_lifecycle as lifecycle

    root_id, leaf_id = "corrupt-root", "corrupt-leaf"
    write_task_result(tmp_path, root_id, "running", root_task_id=root_id)
    intent = _graceful_intent(tmp_path, root_id, scope=ci.SCOPE_CASCADE)
    # Corrupt the lineage only after the owner will is durable: predecessor
    # names the leaf, but the leaf is bound to a foreign logical root.
    write_task_result(
        tmp_path,
        root_id,
        "interrupted",
        root_task_id=root_id,
        superseded_by=leaf_id,
        retry_task_id=leaf_id,
    )
    write_task_result(
        tmp_path,
        leaf_id,
        "running",
        root_task_id="foreign-root",
        parent_task_id="",
        delegation_role="root",
        supersedes_task_id=root_id,
        original_task_id=root_id,
        timeout_retry_from=root_id,
    )
    q = _isolate_queue(
        monkeypatch,
        tmp_path,
        running={
            leaf_id: {
                "task": {
                    "id": leaf_id,
                    "root_task_id": root_id,
                    "parent_task_id": "",
                    "delegation_role": "root",
                },
                "worker_id": -1,
            },
        },
    )
    monkeypatch.setattr(q, "FINALIZATION_GRACE_SEC", 120.0, raising=False)
    hard_calls = []
    monkeypatch.setattr(
        q,
        "cancel_task_by_id",
        lambda task_id, **kwargs: hard_calls.append((task_id, kwargs)) or True,
    )

    before = lifecycle.sweep_cancel_intents(
        now=ostop._requested_ts(intent) + 11.0,
    )

    assert before == {root_id: ostop.OWNER_STOP_HOLDING}
    assert hard_calls == []
    assert leaf_id in q.RUNNING
    assert leaf_id in ostop.running_owner_stop_tasks(tmp_path, grace_sec=120.0)

    after = lifecycle.sweep_cancel_intents(
        now=ostop._requested_ts(intent) + OWNER_STOP_OUTER_CAP_SEC + 1.0,
    )
    assert after == {root_id: lifecycle.CANCEL_CANCELLED}
    assert hard_calls == [(root_id, {"cascade": True})]


def test_sweep_feeds_custody_for_settled_pending_or_expired_roots(tmp_path, monkeypatch):
    from supervisor import workers

    monkeypatch.setattr(
        workers, "get_event_q", lambda: SimpleNamespace(put=lambda _e: None),
    )
    now = time.time()
    # Settled root: natural completion won — custody settles the intent honestly.
    intent = _graceful_intent(tmp_path, "t-done", allow_settled_target=True)
    write_task_result(tmp_path, "t-done", "completed", result="finished naturally")
    q = _fake_queue(tmp_path, {"t-done": {"task": {"id": "t-done", "chat_id": 0}}})
    assert ostop.sweep_owner_stop_hold(q, "t-done", intent, now=now) is False
    # Pending root (never started): zero model turns, custody feed.
    pending_intent = _graceful_intent(tmp_path, "t-pend")
    assert ostop.sweep_owner_stop_hold(
        _fake_queue(tmp_path, {}), "t-pend", pending_intent, now=now,
    ) is False
    # Expired OUTER cap (undrained control never delivered): the episode is
    # over; the generic path proceeds.
    expired = dict(_graceful_intent(tmp_path, "t-exp"))
    assert ostop.sweep_owner_stop_hold(
        _fake_queue(tmp_path, {"t-exp": {"task": {"id": "t-exp", "chat_id": 0}}}),
        "t-exp", expired, now=now + OWNER_STOP_OUTER_CAP_SEC + 100.0,
    ) is False
    # And no episode was armed anywhere along the way.
    for tid in ("t-done", "t-pend", "t-exp"):
        assert _finalize_rows(tmp_path, tid) == []


# ---------------------------------------------------------------------------
# Q4=A summary suppression + Q6=A child projection
# ---------------------------------------------------------------------------


def test_graceful_summary_suppressed_only_for_completed_finalize_roots(tmp_path):
    q = _fake_queue(tmp_path, {})
    # SUCCESS: finalize intent + COMPLETED durable result -> suppressed + forensic.
    _graceful_intent(tmp_path, "t-ok", allow_settled_target=True)
    write_task_result(tmp_path, "t-ok", "completed", result="final summary answer")
    assert ostop.graceful_summary_suppressed(q, "t-ok") is True
    forensics = (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    assert "owner_stop_summary_suppressed" in forensics
    # Expiry -> cancelled keeps the tree's ONE receipt.
    _graceful_intent(tmp_path, "t-exp2", allow_settled_target=True)
    write_task_result(tmp_path, "t-exp2", "cancelled", result="expired")
    assert ostop.graceful_summary_suppressed(q, "t-exp2") is False
    # An immediate stop never suppresses.
    ci.request_cancel(tmp_path, "t-imm2", allow_settled_target=True)
    write_task_result(tmp_path, "t-imm2", "completed", result="done")
    assert ostop.graceful_summary_suppressed(q, "t-imm2") is False


def test_child_result_projection_is_bounded_and_includes_cancelled_children(tmp_path):
    q = _fake_queue(tmp_path, {})
    for i in range(3):
        write_task_result(
            tmp_path, f"kid-{i}", "cancelled",
            result=f"child {i} partial result " + "x" * 400,
            root_task_id="t-root", parent_task_id="t-root",
        )
    projection = ostop._child_result_projection(q, "t-root")
    assert projection.startswith("[CHILD_RESULTS]")
    for i in range(3):
        assert f"kid-{i} (cancelled):" in projection
    # Each preview is bounded to the cap (240 chars + ellipsis).
    for line in projection.splitlines()[1:]:
        assert len(line) < 340
    # A childless root projects nothing (the control stays the bare reason).
    assert ostop._child_result_projection(q, "t-lonely") == ""


def test_owner_requested_finalization_is_a_best_effort_reason_and_bench_truncation_code():
    from devtools.benchmarks.common.result_index import RUNTIME_TRUNCATION_REASON_CODES
    from ouroboros.outcomes import BEST_EFFORT_REASON_CODES

    assert REASON_OWNER_REQUESTED_FINALIZATION in BEST_EFFORT_REASON_CODES
    assert REASON_OWNER_REQUESTED_FINALIZATION in RUNTIME_TRUNCATION_REASON_CODES


def test_owner_stop_deadline_is_immutable_from_its_anchors(tmp_path):
    requested = utc_now_iso()
    intent = {"requested_at": requested, "stop_policy": ci.STOP_POLICY_FINALIZE}
    deadline = ostop.owner_stop_deadline_ts(intent, 120.0)
    assert deadline > 0
    # UNDRAINED: the outer request-time cap is the only bound (owner 2=A).
    assert deadline == ostop._requested_ts(intent) + OWNER_STOP_OUTER_CAP_SEC
    # Progress/heartbeats never extend it: the same intent yields the same deadline.
    assert ostop.owner_stop_deadline_ts(intent, 120.0) == deadline
    assert ostop.owner_stop_deadline_ts({}, 120.0) == 0.0
    # DRAINED: min(drain + grace, request + outer cap) — grace from delivery…
    drained = {**intent, "control_drained_at": requested}
    assert ostop.owner_stop_deadline_ts(drained, 120.0) == (
        ostop._requested_ts(intent) + 120.0
    )
    # …and a drain near the outer cap can never extend the episode past it.
    late = datetime.fromtimestamp(
        ostop._requested_ts(intent) + OWNER_STOP_OUTER_CAP_SEC - 10.0, tz=timezone.utc,
    ).isoformat()
    assert ostop.owner_stop_deadline_ts(
        {**intent, "control_drained_at": late}, 120.0,
    ) == deadline


# ---------------------------------------------------------------------------
# Drain-anchored episode budget + outer cap (owner decisions 2026-08-15, 1=A/2=A)
# ---------------------------------------------------------------------------


def _backdate_intent(tmp_path, task_id, *, seconds):
    """Move the durable intent's requested_at into the past (test-only)."""
    path = tmp_path / "state" / "cancel_intents.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    row = data["intents"][task_id]
    row["requested_at"] = datetime.fromtimestamp(
        time.time() - seconds, tz=timezone.utc,
    ).isoformat()
    path.write_text(json.dumps(data), encoding="utf-8")


def _production_drain(tmp_path, task_id):
    """Run the PRODUCTION loop mailbox drain once with a fresh seen-set."""
    import ouroboros.loop as loop_mod

    return loop_mod._drain_incoming_messages(
        [], queue.Queue(), tmp_path, task_id, None, set(),
        owner_ctx=SimpleNamespace(budget_drive_root=str(tmp_path)),
    )


def test_loop_drain_discards_hardened_owner_control_before_and_during_stamp(
    tmp_path, monkeypatch,
):

    stale_root = tmp_path / "already-hardened"
    stale_root.mkdir()
    stale_task = "t-stale-control"
    stale_intent = _graceful_intent(stale_root, stale_task)
    assert write_owner_message(
        stale_root,
        REASON_OWNER_REQUESTED_FINALIZATION,
        stale_task,
        msg_id=ostop.owner_stop_control_id(stale_intent),
        kind=KIND_FINALIZE_NOW,
    )
    ci.request_cancel(
        stale_root,
        stale_task,
        requested_stop_policy=ci.STOP_POLICY_IMMEDIATE,
    )
    assert _production_drain(stale_root, stale_task) == {}

    race_root = tmp_path / "hardening-race"
    race_root.mkdir()
    race_task = "t-raced-control"
    race_intent = _graceful_intent(race_root, race_task)
    assert write_owner_message(
        race_root,
        REASON_OWNER_REQUESTED_FINALIZATION,
        race_task,
        msg_id=ostop.owner_stop_control_id(race_intent),
        kind=KIND_FINALIZE_NOW,
    )

    def _harden_during_stamp(_owner_ctx, _drive_root, task_id):
        assert task_id == race_task
        ci.request_cancel(
            race_root,
            race_task,
            requested_stop_policy=ci.STOP_POLICY_IMMEDIATE,
        )
        return True

    import supervisor.owner_stop as owner_stop_mod

    monkeypatch.setattr(
        owner_stop_mod,
        "_mark_owner_stop_control_drained",
        _harden_during_stamp,
    )

    assert _production_drain(race_root, race_task) == {}
    assert ci.stop_policy(ci.active_intent(race_root, race_task)) == (
        ci.STOP_POLICY_IMMEDIATE
    )
    for root in (stale_root, race_root):
        event_path = root / "logs" / "events.jsonl"
        assert not event_path.exists() or "owner_stop_stamp_failed" not in (
            event_path.read_text(encoding="utf-8")
        )


def test_late_drain_starts_the_episode_budget_at_delivery(tmp_path):
    """(a) The control is drained LONG after the request (a blocking tool call
    held the round boundary, the live 39e0f183 defect): the grace budget starts
    at the drain, so the bounded summary turn still runs instead of the episode
    expiring 120s after the button press."""
    _graceful_intent(tmp_path, "t-drain")
    _backdate_intent(tmp_path, "t-drain", seconds=300.0)
    intent = ci.active_intent(tmp_path, "t-drain")
    now = time.time()
    # 300s after the request the episode is STILL active (old semantics: dead).
    assert ostop.owner_stop_active(intent, now=now, grace_sec=120.0) is True
    # The PRODUCTION loop drain delivers the control and stamps the intent.
    control_id = ostop.owner_stop_control_id(intent)
    assert write_owner_message(
        tmp_path, REASON_OWNER_REQUESTED_FINALIZATION, "t-drain",
        msg_id=control_id, kind=KIND_FINALIZE_NOW,
    )
    controls = _production_drain(tmp_path, "t-drain")
    # The control reached the owner-stop rail (the ZERO-or-ONE-turn summary
    # itself is pinned by the rail tests above)…
    assert controls["finalize_now"] == REASON_OWNER_REQUESTED_FINALIZATION
    updated = ci.active_intent(tmp_path, "t-drain")
    drained_ts = ostop._drained_ts(updated)
    assert drained_ts > 0
    # …and the budget runs from the DRAIN: active 119s past it, over at 121s,
    # after which the task settles through ordinary custody.
    assert ostop.owner_stop_active(updated, now=drained_ts + 119.0, grace_sec=120.0) is True
    assert ostop.owner_stop_active(updated, now=drained_ts + 121.0, grace_sec=120.0) is False


def test_never_drained_episode_ends_at_the_outer_cap(tmp_path, monkeypatch):
    """(b) The control is NEVER drained (a tool call hung for the whole
    episode): the sweep keeps holding until the 10-minute outer cap, then
    releases so the existing honest custody-cancel path applies unchanged."""
    from supervisor import workers

    monkeypatch.setattr(
        workers, "get_event_q", lambda: SimpleNamespace(put=lambda _e: None),
    )
    intent = _graceful_intent(tmp_path, "t-hung")
    now = time.time()
    running = {"t-hung": {"task": {"id": "t-hung", "chat_id": 0}, "started_at": now}}
    q_fake = _fake_queue(tmp_path, running)
    # Undrained far past the old request+120 horizon: the sweep still HOLDS.
    assert ostop.sweep_owner_stop_hold(q_fake, "t-hung", intent, now=now + 400.0) is True
    # Past the outer 10-minute cap the hold releases: the generic custody feed
    # (settled cancelled, receipt stop_reason preserved) proceeds unchanged.
    assert ostop.sweep_owner_stop_hold(
        q_fake, "t-hung", intent, now=now + OWNER_STOP_OUTER_CAP_SEC + 1.0,
    ) is False
    # The intent stays the single owner will for custody to settle.
    assert ci.stop_policy(ci.active_intent(tmp_path, "t-hung")) == ci.STOP_POLICY_FINALIZE


def test_crash_restart_preserves_both_deadlines(tmp_path):
    """(c) A worker crash/restart between request and drain — or after the
    drain — cannot resurrect an unlimited episode: both anchors live on the
    durable intent, and a restart re-drain never moves the drain stamp."""
    _graceful_intent(tmp_path, "t-crash")
    _backdate_intent(tmp_path, "t-crash", seconds=100.0)
    before = ci.active_intent(tmp_path, "t-crash")
    outer_deadline = ostop.owner_stop_deadline_ts(before, 120.0)
    # Pre-drain restart: the re-read durable intent yields the SAME deadline.
    assert ostop.owner_stop_deadline_ts(
        ci.active_intent(tmp_path, "t-crash"), 120.0,
    ) == outer_deadline
    # The first drain stamps the delivery durably…
    control_id = ostop.owner_stop_control_id(before)
    assert write_owner_message(
        tmp_path, REASON_OWNER_REQUESTED_FINALIZATION, "t-crash",
        msg_id=control_id, kind=KIND_FINALIZE_NOW,
    )
    _production_drain(tmp_path, "t-crash")
    stamped = ci.active_intent(tmp_path, "t-crash")["control_drained_at"]
    assert stamped
    deadline = ostop.owner_stop_deadline_ts(ci.active_intent(tmp_path, "t-crash"), 120.0)
    assert deadline == ostop._parse_ts(stamped) + 120.0        # under the outer cap
    # …a post-crash re-drain (fresh seen set — the control is replayable until
    # terminal cleanup) is a no-op on the stamp: FIRST DRAIN WINS…
    _production_drain(tmp_path, "t-crash")
    after = ci.active_intent(tmp_path, "t-crash")
    assert after["control_drained_at"] == stamped
    # …and even an explicit later stamp attempt is refused.
    assert ci.mark_finalize_control_drained(
        tmp_path, "t-crash", drained_at="2099-01-01T00:00:00+00:00",
    ) is False
    assert ostop.owner_stop_deadline_ts(
        ci.active_intent(tmp_path, "t-crash"), 120.0,
    ) == deadline


def test_zero_grace_is_feature_off_pre_drain_and_post_drain(tmp_path):
    """M1: ``grace_sec<=0`` means the graceful-stop feature is OFF everywhere
    (the immediate custody path) — never a request+outer-cap window, not even
    pre-drain: no episode deadline, no sweep hold, no enforcement bypass."""
    intent = _graceful_intent(tmp_path, "t-zero")
    now = time.time()
    assert ostop.owner_stop_deadline_ts(intent, 0.0) == 0.0
    assert ostop.owner_stop_active(intent, now=now, grace_sec=0.0) is False
    # Post-drain too: a drain stamp never resurrects a window under grace 0.
    assert ci.mark_finalize_control_drained(tmp_path, "t-zero") is True
    assert ostop.owner_stop_deadline_ts(
        ci.active_intent(tmp_path, "t-zero"), 0.0,
    ) == 0.0
    # The sweep feeds custody immediately; enforcement needs no bypass set.
    running = {"t-zero": {"task": {"id": "t-zero", "chat_id": 0}, "started_at": now}}
    q = _fake_queue(tmp_path, running)
    q.FINALIZATION_GRACE_SEC = 0.0
    assert ostop.sweep_owner_stop_hold(q, "t-zero", intent, now=now) is False
    assert ostop.running_owner_stop_tasks(tmp_path, grace_sec=0.0) == set()
    # A positive grace keeps the normal pre-drain outer-cap window.
    assert ostop.owner_stop_deadline_ts(intent, 120.0) > 0


def test_failed_drain_stamp_blocks_control_and_keeps_conservative_deadline(
    tmp_path, monkeypatch,
):
    """An unconfirmed drain never buys a turn or extends the durable window."""
    intent = _graceful_intent(tmp_path, "t-stamp")
    attempts = []

    def _failing_stamp(*_a, **_k):
        attempts.append(1)
        return False

    monkeypatch.setattr(ci, "mark_finalize_control_drained", _failing_stamp)
    control_id = ostop.owner_stop_control_id(intent)
    assert write_owner_message(
        tmp_path, REASON_OWNER_REQUESTED_FINALIZATION, "t-stamp",
        msg_id=control_id, kind=KIND_FINALIZE_NOW,
    )
    controls = _production_drain(tmp_path, "t-stamp")
    assert controls == {}
    assert len(attempts) == 2                       # one retry, then stop
    updated = ci.active_intent(tmp_path, "t-stamp")
    assert not str(updated.get("control_drained_at") or "")
    # Conservative deadline: undrained -> request + outer cap (sweep semantics).
    assert ostop.owner_stop_deadline_ts(updated, 120.0) == (
        ostop._requested_ts(updated) + OWNER_STOP_OUTER_CAP_SEC
    )
    events = (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert '"owner_stop_stamp_failed"' in events
    assert '"t-stamp"' in events


# ---------------------------------------------------------------------------
# Loop owner-stop rail: ZERO or ONE tool-less model turn (Q1/Q3=A, CF-02/CF-03)
# ---------------------------------------------------------------------------


def test_owner_stop_rail_retained_candidate_spends_zero_model_turns(tmp_path, monkeypatch):
    """A current valid DeliveryCandidate is reused verbatim: NO model call, the
    typed owner_requested_finalization reason stamped (never the deadline's)."""
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    loop._replace_delivery_candidate(
        registry, ctx, trace, "Retained complete final answer.", control="candidate",
    )
    calls = []
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *a, **k: calls.append(1) or ({"role": "assistant", "content": "fresh"}, 0.0),
    )
    text, usage, _returned = loop._handle_forced_finalization(
        ctx, REASON_OWNER_REQUESTED_FINALIZATION,
    )
    assert calls == []                                  # zero paid turns
    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert "Retained complete final answer." in text


def test_owner_stop_rail_without_candidate_spends_exactly_one_turn(tmp_path, monkeypatch):
    """No retained candidate: exactly ONE logical tool-less finalization call
    (single_semantic_turn — the generic second semantic refresh is disabled),
    fed the control's bounded child projection, typed reason stamped."""
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, _registry, ctx, _trace = _forced_test_context(tmp_path)
    calls = []

    def _one_call(_llm, messages, *_args, **_kwargs):
        calls.append(messages[-1]["content"])
        return {"role": "assistant", "content": "Synthesized final summary."}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", _one_call)
    control = (
        REASON_OWNER_REQUESTED_FINALIZATION
        + "\n[CHILD_RESULTS] kid-1 (cancelled): partial child answer"
    )
    text, usage, _returned = loop._handle_forced_finalization(ctx, control)
    assert len(calls) == 1                              # exactly one paid turn
    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert "Synthesized final summary." in text
    # The one prompt is the OWNER_STOP rail's (not the deadline's) and carries
    # the bounded durable child projection.
    assert "[OWNER_STOP]" in calls[0]
    assert "[CHILD_RESULTS] kid-1 (cancelled)" in calls[0]


def test_expired_control_at_consume_never_starts_a_paid_summary(tmp_path, monkeypatch):
    """M2: a finalize control consumed only AFTER the effective deadline
    (undrained -> request + outer cap) must not buy a paid summary turn — the
    rail returns the honest fallback on the same typed rail and custody
    settles the episode as today. Inside the window the bounded summary runs."""
    from tests.test_delivery_forced_finalization import _forced_test_context

    calls = []

    def _paid(*_a, **_k):
        calls.append(1)
        return {"role": "assistant", "content": "Paid summary."}, 0.0

    # Inside the window (fresh intent): exactly one paid summary turn runs.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    loop, _registry, ctx, _trace = _forced_test_context(fresh)
    monkeypatch.setattr(loop, "call_llm_with_retry", _paid)
    _graceful_intent(fresh, "parent1")
    text, usage, _t = loop._handle_forced_finalization(
        ctx, REASON_OWNER_REQUESTED_FINALIZATION,
    )
    assert len(calls) == 1 and "Paid summary." in text
    # Past the outer cap: ZERO paid turns; the typed reason and the honest
    # fallback stand (no new outcome invented — custody settles the episode).
    expired = tmp_path / "expired"
    expired.mkdir()
    loop, _registry, ctx2, _trace2 = _forced_test_context(expired)
    _graceful_intent(expired, "parent1")
    _backdate_intent(expired, "parent1", seconds=OWNER_STOP_OUTER_CAP_SEC + 30.0)
    calls.clear()
    text, usage, _t = loop._handle_forced_finalization(
        ctx2, REASON_OWNER_REQUESTED_FINALIZATION,
    )
    assert calls == []                                  # no paid summary
    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert usage["execution_status"] == "failed"
    assert "no final answer could be produced" in text


def test_stopped_direct_turn_ends_with_zero_calls_and_the_retained_candidate(tmp_path, monkeypatch):
    """"Stop now" on an in-process direct-chat turn (custody's control carries
    REASON_OWNER_STOPPED_DIRECT_TURN): the turn ends at its round boundary with
    the delivery candidate it already holds and NO further model call — the
    honest twin of killing a pooled worker, never the graceful rail's paid
    final turn."""
    from tests.test_delivery_forced_finalization import _forced_test_context
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    loop._replace_delivery_candidate(
        registry, ctx, trace, "Retained partial answer.", control="candidate",
    )
    calls = []
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *a, **k: calls.append(1) or ({"role": "assistant", "content": "fresh"}, 0.0),
    )
    text, usage, _returned = loop._handle_forced_finalization(ctx, REASON_OWNER_STOPPED_DIRECT_TURN)
    assert calls == []
    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert usage["execution_status"] == "failed"
    assert "Retained partial answer." in text


def test_stopped_direct_turn_without_a_candidate_ends_with_the_typed_fallback(tmp_path, monkeypatch):
    from tests.test_delivery_forced_finalization import _forced_test_context
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    loop, _registry, ctx, _trace = _forced_test_context(tmp_path)
    calls = []
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *a, **k: calls.append(1) or ({"role": "assistant", "content": "fresh"}, 0.0),
    )
    text, usage, _returned = loop._handle_forced_finalization(ctx, REASON_OWNER_STOPPED_DIRECT_TURN)
    assert calls == []
    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert "owner stopped this chat turn" in text


def test_stopped_direct_turn_hard_stop_wins_over_the_transport_diversion(tmp_path, monkeypatch):
    """Adversarial finding: during an active transport-wait episode every
    finalize_now flavour was diverted to the provider-outage terminal, so the
    owner's zero-call stop was reported as an outage. The direct-turn control
    is routed first."""
    from tests.test_delivery_forced_finalization import _forced_test_context
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    from ouroboros import loop_round_limits

    _loop, _registry, ctx, _trace = _forced_test_context(tmp_path)
    routed = []
    monkeypatch.setattr(loop_round_limits, "_handle_direct_turn_hard_stop", lambda c: routed.append("hard_stop") or ("stopped", ctx.accumulated_usage, {}))
    monkeypatch.setattr(loop_round_limits, "_finalize_now_transport_terminal", lambda *a, **k: routed.append("transport") or ("outage", {}, {}))
    text, _usage, _trace2 = loop_round_limits._maybe_early_finalize(
        ctx, None, {"finalize_now": REASON_OWNER_STOPPED_DIRECT_TURN}, transport_episode=object(),
    )
    assert routed == ["hard_stop"]
    assert text == "stopped"


def test_stopped_direct_turn_marks_post_task_synthesis_skipped_but_wrap_up_does_not(tmp_path, monkeypatch):
    """"Stop now" reaches past the loop: the hard stop records the EXISTING
    ``_skip_post_task_synthesis`` marker on the tool context (the seam
    ``emit_task_results`` copies onto the task record), so the paid post-task
    summary/reflection/consolidation never dispatches for a stopped direct
    turn — while the graceful "Wrap up" rail keeps its memory write."""
    from tests.test_delivery_forced_finalization import _forced_test_context
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    loop, registry, ctx, _trace = _forced_test_context(tmp_path)
    assert not getattr(registry._ctx, "_skip_post_task_synthesis", False)
    loop._handle_forced_finalization(ctx, REASON_OWNER_STOPPED_DIRECT_TURN)
    assert registry._ctx._skip_post_task_synthesis is True

    graceful = tmp_path / "graceful"
    graceful.mkdir()
    loop2, registry2, ctx2, trace2 = _forced_test_context(graceful)
    loop2._replace_delivery_candidate(
        registry2, ctx2, trace2, "Retained partial answer.", control="candidate",
    )
    loop2._handle_forced_finalization(ctx2, REASON_OWNER_REQUESTED_FINALIZATION)
    assert not getattr(registry2._ctx, "_skip_post_task_synthesis", False)

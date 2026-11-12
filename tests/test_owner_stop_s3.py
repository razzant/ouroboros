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
# HTTP ingress: 202 pending acknowledgement vs synchronous legacy teardown
# ---------------------------------------------------------------------------


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
    assert hard_calls == [(root_id, {})]


# ---------------------------------------------------------------------------
# Q4=A summary suppression + Q6=A child projection
# ---------------------------------------------------------------------------


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

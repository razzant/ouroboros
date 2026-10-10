"""Batch4 (owner 7A): what a stop retains, through the REAL consumers.

One predicate (``supervisor/restart_retention.py``) is shared by the owner
Restart's cancel census, ``kill_workers``'s running and pending handling, the
snapshot restore and the boot-marker consumer. These tests drive those
consumers themselves — never the predicate alone — so a census that skips a
paused id while the kill still terminalizes it, or a restore that re-queues a
held row as dispatchable work, fails here.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from tests._budget_pause_exact_helpers import _install_queue, _parked, _pause

RESTART_REASON = "Owner restart stopped this task before process restart."


def _pool_events(workers, monkeypatch):
    events: list = []
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False, raising=False)
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append), raising=False)
    return events


def _done_ids(events):
    return sorted(e["task_id"] for e in events if e.get("type") == "task_done")


def _supervisor_rows(root):
    path = root / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _queued(task_id, **extra):
    return {"admitted_dispatch": "possible" if extra.get("_attempt", 1) > 1 else "none", "id": task_id, "type": "task", "chat_id": 0, "_attempt": 1, "text": task_id, **extra}


def test_a_checkpoint_before_park_survives_every_door_as_the_same_paused_task(tmp_path, monkeypatch):
    """The worker stored the continuation and wrote ``pausing`` + source, then the
    stop came before the supervisor handled the park event: the row is still
    RUNNING. Panic's kill (no new prerequisite: the workers are already dead
    when this runs) parks it under its exact marker instead of terminalizing
    it — no ``task_done``, no cancelled status — and the next boot keeps it."""
    from ouroboros import budget_pause
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    events = _pool_events(workers, monkeypatch)
    ctx, _limit, pause = _pause(tmp_path, monkeypatch, task_id="saved-run")
    budget_pause.end_dispatch_fence(ctx.task_id)
    assert budget_pause.budget_pause_row(tmp_path, "saved-run")["state"] == budget_pause.STATE_PAUSING
    workers.RUNNING["saved-run"] = {"task": {"id": "saved-run", "type": "task", "chat_id": 0,
                                             "root_task_id": "saved-run"}, "worker_id": 0, "attempt": 1}
    workers.RUNNING["plain-run"] = {"task": {"id": "plain-run", "type": "task", "chat_id": 0,
                                             "root_task_id": "plain-run"}, "worker_id": 1, "attempt": 1}

    # The exact Panic call (server_control): no hold_never_started, no reconcile.
    assert workers.kill_workers(force=True, archive_service_logs=False, reconcile_delegate_custody=False)

    assert workers.RUNNING == {}
    assert [row["id"] for row in workers.PENDING] == ["saved-run"]
    parked = workers.PENDING[0]
    assert parked["_budget_pause"]["exact_continuation"] is True
    assert parked["_budget_pause"]["checkpoint"]["pause_id"] == pause["pause_id"]
    row = load_task_result(tmp_path, "saved-run")
    assert row["status"] != STATUS_CANCELLED
    assert row["budget_pause"]["state"] == budget_pause.STATE_PAUSED
    assert row["budget_pause"]["pause_source"] == "stopped_during_pausing"
    assert _done_ids(events) == ["plain-run"]
    cleanup = [r for r in _supervisor_rows(tmp_path) if r["type"] == "zombie_prevention_cleanup"][-1]
    assert cleanup["retained_budget_paused"] == ["saved-run"]
    # The next boot re-validates the durable authority and keeps the same id, unwoken.
    workers.PENDING[:] = []
    assert queue.restore_pending_from_snapshot() == 1
    assert workers.PENDING[0]["id"] == "saved-run" and workers.PENDING[0]["_budget_pause"]["exact_continuation"]


def test_an_explicit_stop_outranks_a_saved_pause_and_request_only_pause_is_interrupted(tmp_path, monkeypatch):
    """Stop wins over a stored checkpoint; a ``pausing`` row without a source is
    pause INTENT only — both are ordinary interrupted work, never a paused task."""
    from ouroboros import budget_pause
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import STATUS_CANCELLED, STATUS_RUNNING, load_task_result, write_task_result
    from supervisor.restart_retention import PAUSE_REQUEST_ONLY, pause_retention

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    events = _pool_events(workers, monkeypatch)
    ctx, _limit, _pause_row = _pause(tmp_path, monkeypatch, task_id="stopped-saved")
    budget_pause.end_dispatch_fence(ctx.task_id)
    request_cancel(tmp_path, "stopped-saved", reason="owner stop", source="owner", requested_by="owner")
    write_task_result(tmp_path, "intent-only", STATUS_RUNNING)
    budget_pause.set_budget_pause(tmp_path, "intent-only", {
        "pause_id": "p-intent", "state": budget_pause.STATE_PAUSING, "reason": "budget",
        "task_attempt": 1, "source_ref": None})
    assert pause_retention(tmp_path, "intent-only", 1) == PAUSE_REQUEST_ONLY
    for task_id in ("stopped-saved", "intent-only"):
        workers.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": 0,
                                             "root_task_id": task_id}, "worker_id": 0, "attempt": 1}

    workers.kill_workers(force=True, terminal_status="cancelled", result_reason=RESTART_REASON,
                         hold_never_started=True)

    assert workers.PENDING == [] and workers.RUNNING == {}
    assert load_task_result(tmp_path, "stopped-saved")["status"] == STATUS_CANCELLED
    assert load_task_result(tmp_path, "intent-only")["status"] == STATUS_CANCELLED
    assert _done_ids(events) == ["intent-only", "stopped-saved"]


def test_owner_restart_holds_the_never_started_queue_and_keeps_it_through_later_doors(tmp_path, monkeypatch):
    """Owner 7A: a never-started independent root keeps its id under a typed
    Restart hold; a never-started child remains held without adoption. A retry
    of work that already ran keeps the ordinary cancellation. The lifespan
    teardown's second kill (no hold flag) and a stale-snapshot boot keep the
    held row; only an explicit Resume releases it — same id, still unstarted."""
    from ouroboros.task_results import (
        STATUS_CANCELLED, STATUS_RUNNING, STATUS_SCHEDULED, load_task_result, write_task_result,
    )
    from supervisor.events_budget import HOLD_OWNER_RESTART, budget_hold_fact

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    events = _pool_events(workers, monkeypatch)
    write_task_result(tmp_path, "root-live", STATUS_RUNNING, chat_id=0)
    workers.RUNNING["root-live"] = {"task": {"id": "root-live", "type": "task", "chat_id": 0,
                                             "root_task_id": "root-live"}, "worker_id": 0, "attempt": 1}
    for task_id in ("child-of-live", "queued-root", "retry-row"):
        write_task_result(tmp_path, task_id, STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.extend([
        _queued("child-of-live", parent_task_id="root-live", root_task_id="root-live"),
        _queued("queued-root", root_task_id="queued-root"),
        _queued("retry-row", root_task_id="retry-row", _attempt=2),
    ])

    workers.kill_workers(force=True, terminal_status="cancelled", result_reason=RESTART_REASON,
                         reconcile_delegate_custody=False, hold_never_started=True)

    assert [row["id"] for row in workers.PENDING] == ["child-of-live", "queued-root"]
    held = workers.PENDING[1]
    hold = budget_hold_fact(held)
    assert hold is not None and hold["reason"] == HOLD_OWNER_RESTART and hold["dispatchable"] is False
    projected = load_task_result(tmp_path, "queued-root")
    assert projected["status"] == STATUS_SCHEDULED and projected["reason_code"] == HOLD_OWNER_RESTART
    for task_id in ("root-live", "retry-row"):
        assert load_task_result(tmp_path, task_id)["status"] == STATUS_CANCELLED
    assert _done_ids(events) == ["retry-row", "root-live"]
    cleanup = [r for r in _supervisor_rows(tmp_path) if r["type"] == "zombie_prevention_cleanup"][-1]
    assert cleanup["held_for_owner_restart"] == ["child-of-live", "queued-root"]

    # The lifespan teardown's own kill (no hold flag): the held row survives.
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason=RESTART_REASON)
    assert [row["id"] for row in workers.PENDING] == ["child-of-live", "queued-root"]
    assert load_task_result(tmp_path, "queued-root")["status"] == STATUS_SCHEDULED

    # A boot long after the stop: the held row is retained at ANY snapshot age.
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    snap["ts"] = "2000-01-01T00:00:00+00:00"
    queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snap))
    workers.PENDING[:] = []
    assert queue.restore_pending_from_snapshot() == 2
    restored = next(t for t in workers.PENDING if t["id"] == "queued-root")
    assert restored["id"] == "queued-root" and budget_hold_fact(restored)["reason"] == HOLD_OWNER_RESTART

    # Only the owner's explicit Resume releases it: same id, zero dispatches.
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    outcome = queue.resume_budget_paused_task("queued-root")
    assert outcome["ok"] is True and outcome["task_id"] == "queued-root"
    assert budget_hold_fact(restored) is None
    assert restored["_budget_pause_hold"]["selected_by"] == "owner"


def test_a_queued_unused_resume_grant_returns_to_its_pause_instead_of_being_cancelled(tmp_path, monkeypatch):
    """A Resume grant minted but never dispatched is not running work: the stop
    revokes it back to the exact pause (the next Resume mints a fresh one).
    Before this, the drain saw no ``_budget_pause`` marker and cancelled it."""
    from ouroboros import budget_pause
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="granted")
    assert queue.resume_budget_paused_task("granted")["ok"] is True
    grant_id = task["_budget_pause_resume"]["grant_id"]

    workers.kill_workers(force=True, terminal_status="cancelled", result_reason=RESTART_REASON,
                         hold_never_started=True)

    assert [row["id"] for row in workers.PENDING] == ["granted"]
    kept = workers.PENDING[0]
    assert kept["_budget_pause"]["exact_continuation"] is True and "_budget_pause_resume" not in kept
    row = load_task_result(tmp_path, "granted")
    assert row["status"] != STATUS_CANCELLED
    assert row["budget_pause"]["state"] == budget_pause.STATE_PAUSED
    assert row["budget_pause"]["grant"]["grant_id"] == grant_id
    assert row["budget_pause"]["grant"]["revoke_reason"] == "restart_before_dispatch"


def test_a_revocation_that_cannot_be_written_stays_held_through_the_restart(tmp_path, monkeypatch):
    """The unused grant's revocation write fails at the owner Restart: the row is
    neither cancelled nor dispatchable — it keeps its exact pause under a typed
    hold, the durable grant is untouched, and the next Resume is refused."""
    from ouroboros import budget_pause
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result
    from supervisor.events_budget import HOLD_RESTART_REVOCATION_UNWRITTEN, budget_hold_fact

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    events = _pool_events(workers, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _parked(tmp_path, monkeypatch, task_id="unwritten")
    assert queue.resume_budget_paused_task("unwritten")["ok"] is True

    def refuse(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(budget_pause, "set_budget_pause", refuse)
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason=RESTART_REASON,
                         hold_never_started=True)

    assert [row["id"] for row in workers.PENDING] == ["unwritten"] and _done_ids(events) == []
    kept = workers.PENDING[0]
    assert kept["_budget_pause"]["exact_continuation"] is True
    assert budget_hold_fact(kept)["reason"] == HOLD_RESTART_REVOCATION_UNWRITTEN
    row = load_task_result(tmp_path, "unwritten")
    assert row["status"] != STATUS_CANCELLED
    assert row["budget_pause"]["state"] == budget_pause.STATE_RESUME_GRANTED, "nothing was written over it"
    assert queue.resume_budget_paused_task("unwritten")["ok"] is False


def test_the_restart_census_skips_saved_pauses_including_a_direct_actor(tmp_path, monkeypatch):
    """A cancel intent minted for a saved pause would outrank it at restore; the
    census the owner Restart cancels from therefore excludes every saved pause,
    a still-registered direct actor's included."""
    from ouroboros import budget_pause, server_restart
    from supervisor import active_activity

    _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    for task_id in ("pooled-saved", "direct-saved"):
        ctx, _limit, _row = _pause(tmp_path, monkeypatch, task_id=task_id)
        budget_pause.end_dispatch_fence(ctx.task_id)
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: SimpleNamespace(
        snapshot=lambda: [{"activity_id": "direct-saved"}, {"activity_id": "direct-live"}]))
    ctx = SimpleNamespace(RUNNING={"pooled-saved": {}, "pooled-live": {}})

    assert server_restart._owned_live_task_ids(ctx) == ["pooled-live", "direct-live"]


def test_owner_restart_passes_the_hold_only_through_its_own_kill(tmp_path, monkeypatch):
    """Owner 2026-10-08 (quiz d2f7532b): the owner Restart's stop keeps saved work and
    the queue for the next boot (its transaction names what returns) instead of
    asking ``kill_workers`` to hold never-started rows; an existing hold stays and
    the transport notice still names what was held."""
    from ouroboros import cancel_intents, server_restart
    from ouroboros import delegate_custody as custody
    from supervisor.events_budget import HOLD_OWNER_RESTART

    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cancel_intents, "request_cancel", lambda *a, **k: {})
    monkeypatch.setattr(custody, "reconcile_orphaned_runs", lambda *a, **k: [])
    monkeypatch.setattr(server_restart, "_stop_owned_daemon", lambda label: None)
    monkeypatch.setattr(server_restart, "_managed_update_pending_kwargs", lambda: {})
    kills: list = []
    pending = [{"id": "held-1", "_budget_pause_hold": {"reason": HOLD_OWNER_RESTART, "selected": False}}]
    ctx = SimpleNamespace(RUNNING={}, PENDING=pending, kill_workers=lambda **kw: kills.append(kw) or True)

    server_restart._stop_owned_work(ctx)

    assert kills and kills[0]["retain_saved_work"] is True and kills[0]["preserve_pending"] is True
    assert "hold_never_started" not in kills[0]
    assert kills[0]["reconcile_delegate_custody"] is False
    assert server_restart._owner_restart_held_count(ctx) == 1


def test_boot_holds_the_restored_queue_before_the_restart_marker_is_consumed(tmp_path, monkeypatch):
    """If the stop could not write its final snapshot, the boot restores an older
    queue without the per-row holds. While the owner-Restart marker is still
    present, every never-started row restored takes the hold, and the marker is
    consumed only after the queue carrying those holds is durably persisted."""
    from supervisor import worker_chat_lane
    from supervisor.events_budget import HOLD_OWNER_RESTART, budget_hold_fact

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

    write_task_result(tmp_path, "older-row", STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.append(_queued("older-row", root_task_id="older-row"))
    queue.persist_queue_snapshot(reason="before_stop")
    workers.PENDING[:] = []
    flag = tmp_path / "state" / "owner_restart_no_resume.flag"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("owner_restart", encoding="utf-8")
    (tmp_path / "state" / "panic_stop.flag").write_text("owner_restart_no_resume", encoding="utf-8")

    assert queue.restore_pending_from_snapshot() == 1
    assert budget_hold_fact(workers.PENDING[0])["reason"] == HOLD_OWNER_RESTART

    monkeypatch.setattr(worker_chat_lane, "_pool", lambda: workers)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": False)
    worker_chat_lane.auto_resume_after_restart()
    assert flag.exists(), "a marker consumed before its holds are durable could auto-wake the queue"

    persisted: list = []
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": persisted.append(reason) or True)
    worker_chat_lane.auto_resume_after_restart()
    assert persisted == ["owner_restart_holds"] and not flag.exists()
    assert not (tmp_path / "state" / "panic_stop.flag").exists()


def test_panic_adds_no_prerequisite_before_the_kill(tmp_path, monkeypatch):
    """Retention reads the durable pause only AFTER every worker was signalled:
    Panic's kill is not delayed by parking, persistence or quiescence."""
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    order: list = []
    from supervisor import restart_retention

    real_park = restart_retention.park_saved_pause
    monkeypatch.setattr(restart_retention, "park_saved_pause",
                        lambda *a, **k: order.append("park") or real_park(*a, **k))
    monkeypatch.setattr(workers, "kill_worker_tree", lambda pid, **_k: order.append("kill"))
    monkeypatch.setattr(workers, "_kill_survivors", lambda: None)
    proc = SimpleNamespace(pid=4242, is_alive=lambda: False, join=lambda timeout=None: None,
                           terminate=lambda: None)
    workers.WORKERS[0] = SimpleNamespace(wid=0, proc=proc, busy_task_id="t-run")
    workers.RUNNING["t-run"] = {"task": {"id": "t-run", "type": "task", "chat_id": 0}, "worker_id": 0,
                                "attempt": 1, "started_at": time.time()}

    workers.kill_workers(force=True, archive_service_logs=False, reconcile_delegate_custody=False)

    assert order[0] == "kill" and order.index("kill") < order.index("park")


def test_only_the_planned_restart_record_triggers_the_boot_auto_resume(tmp_path, monkeypatch):
    """A ``launcher_start`` line in the log tail is not evidence of unfinished
    work; ``pending_restart_verify.json`` stays the one real trigger."""
    import threading

    from supervisor import worker_chat_lane

    (tmp_path / "logs").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "scratchpad.md").write_text("# Scratchpad\n- finish the report\n", encoding="utf-8")
    (tmp_path / "logs" / "supervisor.jsonl").write_text(json.dumps({"type": "launcher_start"}) + "\n",
                                                        encoding="utf-8")
    pool = SimpleNamespace(DRIVE_ROOT=tmp_path, load_state=lambda: {"owner_chat_id": 7},
                           chat_turn_liveness=lambda: False)
    monkeypatch.setattr(worker_chat_lane, "_pool", lambda: pool)
    monkeypatch.setattr(worker_chat_lane.time, "sleep", lambda _s: None)
    woke = threading.Event()
    monkeypatch.setattr(worker_chat_lane, "handle_chat_direct", lambda *_a, **_k: woke.set())

    worker_chat_lane.auto_resume_after_restart()
    assert not woke.wait(0.2), "a log-tail launcher line alone never auto-resumes"

    (tmp_path / "state" / "pending_restart_verify.json").write_text("{}", encoding="utf-8")
    worker_chat_lane.auto_resume_after_restart()
    assert woke.wait(5), "the planned restart's verify record still resumes the scratchpad work"


# --- Batch4 F1: the stop door's typed cause reaches the owner's Continue ---

def _running_root(root, workers, task_id, *, worker_id=0):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(root, task_id, STATUS_RUNNING, chat_id=7, root_task_id=task_id, title=task_id,
                      origin_message_text=f"owner words for {task_id}",
                      origin_message_ref={"chat_id": 7, "client_message_id": f"m-{task_id}"},
                      billing_group={"billing_group_id": task_id, "billing_group_limit_usd": 20.0,
                                     "billing_group_limit_source": "initial_task_admission",
                                     "billing_group_limit_revision": "admission-1"})
    workers.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": 7, "root_task_id": task_id},
                                "worker_id": worker_id, "attempt": 1}


def _restart_door(tmp_path, monkeypatch, workers):
    """The owner Restart's REAL stop (``server_restart._stop_owned_work``): the real
    cancel intents and the real ``kill_workers``; only the daemon/custody legs are inert."""
    from ouroboros import server_restart
    from ouroboros import delegate_custody as custody

    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(custody, "reconcile_orphaned_runs", lambda *a, **k: [])
    monkeypatch.setattr(server_restart, "_stop_owned_daemon", lambda label: None)
    monkeypatch.setattr(server_restart, "_managed_update_pending_kwargs", lambda: {})
    ctx = SimpleNamespace(RUNNING=workers.RUNNING, PENDING=workers.PENDING, kill_workers=workers.kill_workers)
    return server_restart._stop_owned_work(ctx)


def test_the_owner_restart_records_its_cause_so_continue_is_offered_and_an_earlier_stop_stays_a_stop(
        tmp_path, monkeypatch):
    """F1: the Restart door's kill used to write a bare ``cancelled`` (no origin), so
    every root it interrupted refused Continue as ``interruption_cause_unrecorded``.
    The door now carries its typed cause through the existing failure writer; an owner
    Stop requested before the Restart keeps its own cause (never overwritten)."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.owner_continue import continuation_eligibility
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result
    from supervisor.continuation_admission import admit_continuation

    _install_queue(tmp_path, monkeypatch)
    from supervisor import workers
    events = _pool_events(workers, monkeypatch)
    _running_root(tmp_path, workers, "restart-root")
    _running_root(tmp_path, workers, "stopped-root", worker_id=1)
    request_cancel(tmp_path, "stopped-root", reason="Stop now", source="http_single", requested_by="owner",
                   requested_stop_policy="immediate")

    assert sorted(_restart_door(tmp_path, monkeypatch, workers)) == ["restart-root", "stopped-root"]

    restarted = load_task_result(tmp_path, "restart-root")
    assert restarted["status"] == STATUS_CANCELLED
    assert restarted["cancel_origin"]["source"] == "owner_restart"
    assert restarted["cancel_origin"]["requested_by"] == "owner"
    assert continuation_eligibility(restarted, "restart-root") == {
        "eligible": True, "cause": "owner_restart", "refusal": ""}
    stopped = load_task_result(tmp_path, "stopped-root")
    assert stopped["cancel_origin"]["source"] == "http_single", "the earlier Stop outranks the Restart"
    assert continuation_eligibility(stopped, "stopped-root")["refusal"] == "stopped_by_owner"
    assert _done_ids(events) == ["restart-root", "stopped-root"]

    admitted = admit_continuation("restart-root", action_nonce="restart-press-0001")
    assert admitted["ok"] is True and admitted["held"] is False, admitted
    assert admit_continuation("stopped-root", action_nonce="stopped-press-0001")["error"] == "stopped_by_owner"


def test_a_graceful_shutdown_records_its_cause_and_panic_records_none(tmp_path, monkeypatch):
    """The graceful doors (planned restart, SIGTERM teardown, the hung-restart
    cleanup) pass ``server_shutdown``: a technical interruption. Panic's exact
    call passes nothing, so its kill stays ineligible."""
    import inspect

    import server
    from ouroboros.owner_continue import continuation_eligibility
    from ouroboros.server_restart import _shutdown_task_cleanup_args
    from ouroboros.task_results import load_task_result

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _running_root(tmp_path, workers, "graceful-root")
    status, reason = _shutdown_task_cleanup_args(restart_requested=False)
    workers.kill_workers(force=True, terminal_status=status, result_reason=reason, stop_source="server_shutdown")
    graceful = load_task_result(tmp_path, "graceful-root")
    assert graceful["status"] == "cancelled"
    assert graceful["cancel_origin"] == {"source": "server_shutdown", "reason": reason}
    assert continuation_eligibility(graceful, "graceful-root")["eligible"] is True

    _running_root(tmp_path, workers, "panic-root")
    workers.kill_workers(force=True, archive_service_logs=False, reconcile_delegate_custody=False)  # Panic
    panicked = load_task_result(tmp_path, "panic-root")
    assert panicked["status"] == "failed" and "cancel_origin" not in panicked
    assert continuation_eligibility(panicked, "panic-root")["refusal"] == "interruption_cause_unrecorded"

    # Every graceful kill door of the composition root passes the typed cause.
    for door in (server.lifespan, server._emergency_process_cleanup, server._perform_supervisor_restart):
        source = inspect.getsource(door)
        kills = source.count("terminal_status=cleanup_status")
        assert kills and source.count('stop_source="server_shutdown"') == kills, door.__name__


def test_an_unreadable_intent_store_records_no_cause_and_a_retried_write_keeps_the_doors(tmp_path, monkeypatch):
    """No cause is better than a guessed one: an unreadable cancel-intent store could
    hide an owner Stop, so the write carries no origin (Continue stays refused). A
    kill whose terminal write failed keeps the door's cause in its retry custody."""
    from ouroboros import task_results
    from ouroboros.owner_continue import continuation_eligibility
    from ouroboros.task_results import load_task_result

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _running_root(tmp_path, workers, "unknown-cause")
    (tmp_path / "state" / "cancel_intents.json").write_text("{not json", encoding="utf-8")
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason=RESTART_REASON,
                         stop_source="owner_restart")
    unknown = load_task_result(tmp_path, "unknown-cause")
    assert unknown["status"] == "cancelled" and "cancel_origin" not in unknown
    assert continuation_eligibility(unknown, "unknown-cause")["refusal"] == "interruption_cause_unrecorded"
    (tmp_path / "state" / "cancel_intents.json").unlink()

    _running_root(tmp_path, workers, "retried-root")
    real_write = task_results.write_task_result
    failures = {"left": 2}  # the kill's own write, then the in-kill retry of its custody marker

    def failing_once(root, task_id, status, **fields):
        if task_id == "retried-root" and status == "cancelled" and failures["left"]:
            failures["left"] -= 1
            raise OSError("disk full")
        return real_write(root, task_id, status, **fields)

    monkeypatch.setattr(task_results, "write_task_result", failing_once)
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason=RESTART_REASON,
                         stop_source="owner_restart")
    assert failures["left"] == 0
    retained = next(row for row in workers.PENDING if row["id"] == "retried-root")
    assert workers._terminalization_retry_spec(retained)["stop_source"] == "owner_restart"
    assert load_task_result(tmp_path, "retried-root")["status"] == "running"
    assert workers._settle_terminalization_retry_task(retained) is True
    retried = load_task_result(tmp_path, "retried-root")
    assert retried["status"] == "cancelled" and retried["cancel_origin"]["source"] == "owner_restart"

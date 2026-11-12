"""External work retained by the body during one exact cold park: a model-chosen
sleep, or a technical park on an unreadable authority (reason ``authority``).

The existing pause document owns the request and receipt. The runtime accepts
exact run identities before the actor releases its process; its identity is
provenance, not a lease or a Resume grant. Discovery uses current actor/queue
carriers, never task history. Notifications are optional hints. A sleep binds
its sleep identity and may wake; an authority park binds by its pause identity
alone and never wakes by itself: only an explicit Resume returns its author.
A started review panel's run keeps its own controller and is not transferred.
"""
from __future__ import annotations

import logging
import pathlib
import time
from typing import Any

log = logging.getLogger(__name__)


RETAINED_REASONS = frozenset({"sleep", "authority"})


def _binding(task_id: str, row: dict) -> dict:
    binding = {"task_id": str(task_id), **{key: row.get(key) for key in (
        "pause_id", "pause_generation", "task_attempt", "source_ref", "root_task_id", "project_id")}}
    if row.get("reason") == "authority":
        return {**binding, "reason": "authority"}  # no sleep identity is invented
    return {**binding, "sleep_id": str((row.get("sleep") or {}).get("sleep_id") or "")}


def accepted_receipt(task_id: str, row: dict) -> dict:
    """Logical receipt only; callers separately check current controls/runtime."""
    receipt = row.get("external_custody") or {}
    binding = _binding(task_id, row)
    required = ("pause_id", "task_attempt", "source_ref") + (("sleep_id",) if row.get("reason") == "sleep" else ())
    if (row.get("reason") not in RETAINED_REASONS or row.get("state") not in {"pausing", "paused", "resume_granted"}
            or receipt.get("state") != "accepted" or receipt.get("binding") != binding
            or not all(binding.get(k) for k in required)):
        return {}
    return receipt


def transferable_pairs(root: Any, task_id: str, *, panel_keeps_reviews: bool = False) -> list[list[str]]:
    """The exact task's bound ordinary runs; unknown starts remain blockers. A sleep refuses
    to leave a panel run behind; an authority park leaves it with its panel's controller."""
    from ouroboros.delegate_custody_current import task_snapshot
    from ouroboros.delegate_registration_policy import review_owned_source

    view = task_snapshot(root, task_id)
    if view["pending"]:
        raise ValueError("unbound external invocation cannot transfer to cold custody")
    pairs = []
    for run in view["runs"].values():
        if run.settled or (panel_keeps_reviews and review_owned_source(run.source)):
            continue
        if not run.run_id or not run.invocation_id or review_owned_source(run.source):
            raise ValueError("external run has no transferable ordinary owner binding")
        pairs.append([str(run.run_id), str(run.invocation_id)])
    return sorted(pairs)


def request_fields(ctx: Any, row: dict) -> dict:
    """Open custody keeps the tree's project lane whatever a run may write (``project_lease.lane_holders``)."""
    from ouroboros.delegate_custody import custody_root

    pairs = transferable_pairs(custody_root(ctx), str(ctx.task_id), panel_keeps_reviews=row.get("reason") == "authority")
    if not pairs:
        return {}
    return {"state": "requested", "binding": _binding(str(ctx.task_id), row), "runs": pairs,
            "retains_project_lane": bool(row.get("project_id")), "requested_at": time.time()}


def seen_interactions(ctx: Any, pairs: Any = None) -> dict:
    """Notification exposure carried by the actor, never question consumption."""
    remembered = dict(getattr(ctx, "_sleep_interaction_seen", None) or {})
    # Old transcripts lacking this map may receive one extra notification;
    # a latest-run supervision ACK alone proves no question was in this view.
    ids = remembered if pairs is None else (run_id for run_id, _invocation_id in pairs)
    return {run_id: sorted(set(remembered.get(run_id) or ())) for run_id in ids}


def record_question_notice(ctx: Any, external: list) -> None:
    """Called only after this actor's question-source notice reaches its transcript."""
    seen = seen_interactions(ctx)
    for run in external:
        if not isinstance(run, dict):
            continue
        rid = str(run.get("run_id") or "")
        if rid and not run.get("review_owned") and run.get("interaction_source"):
            seen[rid] = sorted(set(seen.get(rid) or ()) | set(run.get("pending_interaction_ids") or ()))
    ctx._sleep_interaction_seen = seen


def _replace(root: Any, task_id: str, expected: dict, replacement: dict) -> dict:
    """One conditional writer for acceptance AND withdrawal; neither can win late."""
    from ouroboros.budget_pause import BudgetPauseSuperseded
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, require_writable_task_result_schema
    from ouroboros.task_store import patch_document

    def update(current):
        require_writable_task_result_schema(current)
        if current.get("status") in _TRULY_TERMINAL_STATUSES or current.get("budget_pause") != expected:
            raise BudgetPauseSuperseded("cold custody request changed")
        return {**current, "budget_pause": replacement}

    patch_document(root, task_id, update)
    return replacement


def confirm_release(ctx: Any, prepared: dict, seed: dict, *, control: Any) -> dict | None:
    """Save, then wait for the body; awake return requires a proved withdrawal.

    Confirmation expiry is transport uncertainty, never proof the host died.
    A lost withdrawal keeps the interruptible hold and observes the winner.
    """
    from ouroboros.budget_pause import budget_pause_row, end_dispatch_fence, _HOLD_POLL_SEC
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.routing_wait import _confirm_wait_sec

    root, tid = pathlib.Path(ctx.budget_drive_root or ctx.drive_root), str(ctx.task_id)
    current = budget_pause_row(root, tid)
    if current == seed:
        current = _replace(root, tid, seed, prepared)
    deadline = time.monotonic() + _confirm_wait_sec(None)
    while True:
        reason = control(ctx)
        if reason:
            raise ModelWaitInterrupted(reason)
        current = budget_pause_row(root, tid)
        if _binding(tid, current) != _binding(tid, prepared):
            from ouroboros.budget_pause import BudgetPauseSuperseded
            raise BudgetPauseSuperseded("cold sleep source/attempt changed during transfer")
        if accepted_receipt(tid, current) and current.get("reason") == "authority":
            # A member-only technical park: its own producers already settled, and the
            # owner's Pause reaches these runs through the accepted receipt afterwards.
            return current
        if accepted_receipt(tid, current):
            from ouroboros.model_sleep import cold_blockers
            from ouroboros.owner_pause import member_fence
            fence = member_fence(ctx)
            if fence.get("state") == "unknown":
                time.sleep(_HOLD_POLL_SEC)
                continue
            if fence:
                try:
                    _replace(root, tid, current, {**current, "state": "abandoned",
                        "abandon_reason": "owner_pause_before_cold_release", "abandoned_at": time.time()})
                except Exception:
                    time.sleep(_HOLD_POLL_SEC)
                    continue
                end_dispatch_fence(tid)
                ctx._budget_pausing, ctx._model_sleep = False, None
                return None  # the independent owner fence remains closed
            blockers = cold_blockers(ctx, transferred=current["external_custody"]["runs"])
            if blockers:
                # Acceptance cannot certify a newly appeared local/unknown writer.
                from ouroboros.budget_pause import ColdSleepWritersUnsettled
                raise ColdSleepWritersUnsettled(f"cold release still has local custody: {blockers}")
            return current
        transfer = current.get("external_custody") or {}
        if transfer.get("state") == "refused" or time.monotonic() >= deadline:
            if current.get("state") == "pausing" and transfer.get("state") in {"requested", "refused"}:
                why = str(transfer.get("reason") or "cold_custody_confirmation_unconfirmed")
                try:
                    _replace(root, tid, current, {**current, "state": "abandoned", "abandon_reason": why,
                                                  "abandoned_at": time.time()})
                except Exception:
                    log.debug("Cold withdrawal unconfirmed; retain the hold", exc_info=True)
                else:
                    end_dispatch_fence(tid)
                    ctx._budget_pausing, ctx._model_sleep = False, None
                    return None
        time.sleep(_HOLD_POLL_SEC)


def _runtime(root: Any):
    from ouroboros.delegate_collection import body_runtime
    return body_runtime(root)


def accepted_owner_ids(q: Any = None, *, direct: Any = ()) -> set[str]:
    """Passive orphan exclusion from exact current cold carriers, never grants.

    A persisted accepted locator remains a keeper when a canonical read or the
    current runtime is unavailable. Rebinding and granting have their own strict
    reads; this frequent liveness reader neither writes nor opens transport.
    """
    if q is None:
        from supervisor import queue as q
    with q._queue_lock:
        tasks = [dict(t) for t in q.PENDING]
        tasks.extend(dict(meta.get("task") or {}) for meta in q.RUNNING.values())
    tasks.extend(dict(t) for t in direct if isinstance(t, dict))
    return {str(t["id"]) for t in tasks if t.get("id") and carrier_receipt(t)}


def reconcile(q: Any = None, *, direct: Any = (), runtime: Any = None, stop_requested: Any = None) -> set[str]:
    """Off-loop acceptance/rebinding hook. Return exact retained semantic owners.

    Call on startup before orphan handling and on the body's ordinary pass.
    ``direct`` contains current direct actors' task records; RUNNING requests
    are discovered even before the park event exists or when its hint is lost.
    One read per addressed store selects the rows carrying a saved request or
    receipt, so a pass costs what those rows cost, not one read per live actor.
    """
    from ouroboros.budget_pause import BudgetPauseSuperseded
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.store_hot_facts import cold_custody_rows

    if q is None:
        from supervisor import queue as q
    with q._queue_lock:
        tasks = [dict(meta.get("task") or {}) for meta in q.RUNNING.values()]
        tasks.extend(dict(t) for t in q.PENDING if carrier_receipt(t))
    tasks.extend(dict(t) for t in direct if isinstance(t, dict))
    # An unreadable current row must not turn an accepted parked owner into an
    # orphan. The canonical read still gates acceptance and every return grant.
    retained = {str(t["id"]) for t in tasks if t.get("id") and carrier_receipt(t)}
    by_store: dict = {}
    for task in tasks:
        if task.get("id"):
            by_store.setdefault(pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT), set()).add(str(task["id"]))
    for root, ids in by_store.items():
        if stop_requested is not None and stop_requested():
            break
        try:
            rows = cold_custody_rows(root, ids)
        except Exception:
            log.warning("Cold custody observation unavailable for %s; no acceptance inferred", root, exc_info=True)
            continue
        for tid, row in rows.items():
            if stop_requested is not None and stop_requested():
                break
            try:
                transfer = row.get("external_custody") or {}
                accepted = accepted_receipt(tid, row)
                if not accepted and not (row.get("reason") in RETAINED_REASONS and row.get("state") == "pausing"
                                         and transfer.get("state") == "requested"):
                    continue
                if transfer.get("binding") != _binding(tid, row) or not row.get("source_ref"):
                    continue
                if accepted:
                    retained.add(tid)
                else:
                    read_actor_source_bytes(root, tid, row["source_ref"])
                owner = runtime if runtime is not None else _runtime(root)
                ready = owner is not None and owner.ensure_runs(transfer["runs"], task_id=tid)
                if accepted:
                    # An unavailable caretaker is not anonymous owner loss. Return
                    # separately requires current runtime responsibility.
                    retained.add(tid)
                    continue
                if stop_requested is not None and stop_requested():
                    continue
                if not ready:
                    _replace(root, tid, row, {**row, "external_custody": {
                        **transfer, "state": "refused", "reason": "body_custody_unavailable"}})
                    continue
                from ouroboros.owner_pause import member_fence
                from ouroboros.cancel_intents import cancel_pending
                from types import SimpleNamespace
                # Authority custody preserves already-admitted runs; it grants no execution.
                # The owner's fence still gates Resume and reaches these runs through this
                # receipt. Sleep retains its stricter execution-fence prerequisite.
                if cancel_pending(root, tid) or (row.get("reason") != "authority" and member_fence(
                        SimpleNamespace(task_id=tid, root_task_id=row.get("root_task_id") or tid,
                                        budget_drive_root=str(root)))):
                    continue
                receipt = {**transfer, "state": "accepted", "accepted_at": time.time(),
                           "runtime": owner.identity()}
                _replace(root, tid, row, {**row, "external_custody": receipt})
                retained.add(tid)
            except BudgetPauseSuperseded:
                continue
            except Exception:
                log.warning("Cold custody observation unavailable for %s; no acceptance inferred", tid, exc_info=True)
    return retained


def observe_return(root: Any, task_id: str, external: dict) -> dict:
    """Off-Q current physical responsibility for a logically accepted sleep."""
    from ouroboros.budget_pause import budget_pause_row

    row = budget_pause_row(root, task_id)
    receipt = accepted_receipt(task_id, row)
    if not receipt or external.get("custody_read") != "ok":
        return {}
    pairs = {tuple(pair) for pair in receipt["runs"]}
    panel_kept = row.get("reason") == "authority"  # a started panel's run is its controller's, not returned
    if any((str(run.get("run_id") or ""), str(run.get("invocation_id") or "")) not in pairs
           for run in external.get("runs") or [] if not (panel_kept and run.get("review_owned"))):
        return {}
    runtime = _runtime(root)
    if runtime is None or not runtime.ensure_runs(receipt["runs"], task_id=task_id):
        return {}
    return {"binding": receipt["binding"], "active": True, "runs": receipt["runs"]}


def return_allowed(task_id: str, row: dict, external: dict) -> bool:
    receipt, observed = accepted_receipt(task_id, row), external.get("body_custody") or {}
    return bool(receipt and observed.get("active") and observed.get("binding") == receipt["binding"]
                and observed.get("runs") == receipt.get("runs") and external.get("custody_read") == "ok"
                and all([str(run.get("run_id") or ""), str(run.get("invocation_id") or "")] in receipt["runs"]
                        for run in external.get("runs") or []))


def carrier_receipt(task: dict) -> dict:
    """Pure queue projection, qualified when the exact pause is installed/restored."""
    marker = task.get("_budget_pause") or (task.get("_budget_pause_resume") or {}).get("pause") or {}
    checkpoint = marker.get("checkpoint") or {}
    row = {**checkpoint, "reason": marker.get("reason"), "state": "paused", "sleep": marker.get("sleep"),
           "root_task_id": marker.get("root_task_id"), "project_id": task.get("project_id") or "",
           "external_custody": marker.get("external_custody")}
    return accepted_receipt(str(task.get("id") or ""), row)

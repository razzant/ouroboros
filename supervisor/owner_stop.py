"""Owner graceful stop ("Wrap up" / finalize-then-stop, Q1/Q2/Q3=A/Q6=A, 2026-08-15).

The policy half of the S3 cancel-finalization design, kept OUT of the pinned
``supervisor/task_lifecycle.py`` (which keeps a line-neutral dispatch at the
intent sweep) and ``supervisor/queue.py`` (which keeps one typed predicate in
its timeout enforcement):

- the durable cancel intent (``ouroboros/cancel_intents.py``) with
  ``stop_policy=finalize_then_cancel`` is the ONLY owner will — no second
  ledger, timer, or lease;
- the grace episode REUSES the existing coupled ``finalize_now`` control +
  RUNNING-row latch (``task_reaper.request_finalization_grace``), with the
  episode/control identity derived deterministically from the durable stop
  ``request_id`` so watchdog/restart replays never mint a duplicate control;
- ``sweep_cancel_intents`` calls ``sweep_owner_stop_hold`` per open intent:
  before the shared deadline the episode is armed/held (custody NOT fed);
  at the deadline, on a settled result, or after an immediate upgrade the
  generic custody feed proceeds — the existing custody stays the only killer;
- Q6=A cascade: live descendants are hard-settled deepest-first with ZERO paid
  turns through the existing subtree sweep, then only the root's one bounded
  tool-less turn runs over the preserved child results.

Panic and every non-graceful cancel are untouched: absence of the explicit
policy is byte-identical immediate hard cancellation (§13.1).
"""

from __future__ import annotations

import logging
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

# The sweep outcome recorded while an owner-stop episode holds an open intent.
OWNER_STOP_HOLDING = "owner_stop_finalizing"

_CHILD_PROJECTION_MAX_ROWS = 20
_CHILD_PROJECTION_PREVIEW_CHARS = 240


def _parse_ts(raw: Any) -> float:
    text = str(raw or "").replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _requested_ts(intent: Dict[str, Any]) -> float:
    return _parse_ts(intent.get("requested_at"))


def _drained_ts(intent: Dict[str, Any]) -> float:
    """When the loop actually DELIVERED the finalize control to the model.

    Stamped by the worker's production mailbox drain
    (``cancel_intents.mark_finalize_control_drained``, first drain wins) and
    read back here by the sweep; 0.0 while the control is still undelivered.
    """
    return _parse_ts(intent.get("control_drained_at"))


def owner_stop_deadline_ts(intent: Dict[str, Any], grace_sec: float) -> float:
    """The episode's EFFECTIVE deadline (owner decisions 2026-08-15, 1=A + 2=A).

    Two immutable anchors, never extended by progress:

    - before the finalize control is DELIVERED to the model (no durable drain
      stamp yet): the OUTER safety cap alone applies — stop request time +
      ``OWNER_STOP_OUTER_CAP_SEC`` — so a task inside a long blocking tool
      call keeps its bounded final turn instead of being killed 120s after
      the button press;
    - after delivery: ``min(drain + grace SSOT, request + outer cap)`` — the
      episode budget starts ticking at the drain, and the outer cap still
      bounds the whole episode from the owner's request.

    ``grace_sec<=0`` means the graceful-stop feature is OFF (same semantics
    as ``running_owner_stop_tasks``): NO episode window exists anywhere,
    pre-drain included — never a request+outer-cap window. The sweep then
    feeds custody immediately (the immediate custody path).
    """
    from ouroboros.config import OWNER_STOP_OUTER_CAP_SEC

    if float(grace_sec or 0.0) <= 0:
        return 0.0
    requested = _requested_ts(intent)
    if not requested:
        return 0.0
    outer_deadline = requested + float(OWNER_STOP_OUTER_CAP_SEC)
    drained = _drained_ts(intent)
    if drained:
        return min(drained + float(grace_sec), outer_deadline)
    return outer_deadline


def owner_stop_open(intent: Any) -> bool:
    """Whether an UNCLAIMED finalize-policy intent is still OPEN.

    Open means: the durable intent carries the explicit finalize policy and no
    custody claim holds it (a claim means the kill already started). This is
    the HOLD predicate for the generic timeout rails (§12.2 item 8): a running
    task stays held against the generic idle/finalization-grace machinery while
    the deadline gates the sweep's arm-vs-feed-custody decision
    (``sweep_owner_stop_hold``).  A task's earlier explicit deadline and
    absolute safety ceiling remain independent hard axes; neither the hold nor
    the final model turn may widen them.
    """
    if not isinstance(intent, dict):
        return False
    from ouroboros.cancel_intents import INTENT_CLAIMED, STOP_POLICY_FINALIZE, stop_policy

    if stop_policy(intent) != STOP_POLICY_FINALIZE:
        return False
    return intent.get("state") != INTENT_CLAIMED


def owner_stop_active(intent: Any, *, now: float, grace_sec: float) -> bool:
    """Whether an OPEN graceful stop episode is still inside its window.

    Active means OPEN (``owner_stop_open``) plus the EFFECTIVE deadline
    (``owner_stop_deadline_ts``: outer cap before the control is delivered,
    ``min(drain + grace, request + outer cap)`` after) has not passed.
    Own/descendant progress NEVER extends either anchor (§12.2 item 8).
    Consulted by the SWEEP only (arm vs feed custody); the enforcement hold
    uses the deadline-free ``owner_stop_open`` but still yields to the task's
    explicit deadline and absolute ceiling.
    """
    if not owner_stop_open(intent):
        return False
    deadline = owner_stop_deadline_ts(intent, grace_sec)
    return bool(deadline) and now < deadline


def queue_grace_sec(q: Any) -> float:
    """The shared finalization-grace SSOT as the queue currently holds it."""
    try:
        return float(getattr(q, "FINALIZATION_GRACE_SEC", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def running_owner_stop_tasks(drive_root: Any, *, grace_sec: float) -> set:
    """Task ids whose OPEN owner-stop intent bypasses generic idle/grace rails.

    Read once per enforcement pass (one small locked projection read), then
    checked per RUNNING row — the typed predicate ``supervisor/queue.py``
    consults before generic spare-withdraw, spare-reset, second-grace,
    idle-kill, reaping, and retry branches (§12.2 item 8). The task's explicit
    deadline and absolute safety ceiling are checked separately by the queue
    and never inherit this hold.  Within those hard bounds the hold covers the
    whole open-intent window, including the short expiry window before custody
    consumes the intent, so the generic idle rail cannot clone a retry.
    ``grace_sec<=0`` means the graceful-stop feature is off (the sweep feeds
    custody immediately), so no hold is needed.
    """
    if float(grace_sec or 0.0) <= 0:
        return set()
    try:
        from ouroboros.cancel_intents import active_intents

        intents = active_intents(drive_root)
    except Exception:
        log.debug("owner-stop enforcement read failed", exc_info=True)
        return set()
    held: set[str] = set()
    for task_id, intent in intents.items():
        if not owner_stop_open(intent):
            continue
        held.add(task_id)
        try:
            from ouroboros.cancel_intents import _validated_single_cancel_target

            held.add(_validated_single_cancel_target(drive_root, task_id))
        except Exception:
            # A corrupt chain must not let the generic timeout rail kill a
            # possible graceful root attempt. Include every live root-shaped
            # row that still names the logical id; the intent sweep will expose
            # the authority failure instead of silently bypassing the hold.
            try:
                from supervisor import queue as q

                with q._queue_lock:
                    for running_id, meta in q.RUNNING.items():
                        task = meta.get("task") if isinstance(meta, dict) else None
                        if (
                            isinstance(task, dict)
                            and str(task.get("root_task_id") or "") == task_id
                            and str(task.get("delegation_role") or "") == "root"
                        ):
                            held.add(str(running_id))
            except Exception:
                log.debug(
                    "owner-stop retry hold resolution failed for %s",
                    task_id,
                    exc_info=True,
                )
    return held


def sweep_owner_stop_hold(q: Any, task_id: str, intent: Dict[str, Any], *, now: float) -> bool:
    """The policy-aware sweep decision for ONE open intent (§12.2 item 9).

    True  = the graceful episode holds this tick: descendants (cascade) are
            hard-settled with zero paid turns, the root's grace episode is
            idempotently armed/held, custody is NOT fed.
    False = the generic custody feed proceeds (immediate policy, hardened
            intent, expired deadline, settled root, or a root that never
            started — pending tasks never buy a model turn).
    """
    grace = queue_grace_sec(q)
    if not owner_stop_active(intent, now=now, grace_sec=grace):
        return False
    try:
        from ouroboros.cancel_intents import (
            _validated_single_cancel_target,
            settled_status,
        )

        physical_task_id = _validated_single_cancel_target(q.DRIVE_ROOT, task_id)
        if _task_hard_bound_reached(q, physical_task_id, now=now):
            # A later graceful-stop request cannot buy a final turn beyond a
            # caller-supplied deadline or the global absolute safety ceiling.
            # Feed the existing cancellation custody instead.
            return False
        if settled_status(q.DRIVE_ROOT, physical_task_id):
            # Natural completion won (or an earlier terminal landed): feed
            # custody so completion-wins settles the intent honestly.
            return False
    except Exception:
        # Unknown lineage/result authority is not permission to bypass the
        # active grace window and feed hard custody.  Hold until the existing
        # durable deadline expires; the outer predicate then releases the
        # generic hard path without extending owner authority indefinitely.
        log.warning("owner-stop settled read failed for %s", task_id, exc_info=True)
        return True
    try:
        return orchestrate_graceful_stop(q, task_id, intent, now=now)
    except Exception:
        log.warning("owner-stop orchestration failed for %s", task_id, exc_info=True)
        return True


def _task_hard_bound_reached(q: Any, task_id: str, *, now: float) -> bool:
    """Mirror the queue's two hard time axes without introducing a new SSOT."""
    try:
        with q._queue_lock:
            meta = q.RUNNING.get(task_id) if isinstance(q.RUNNING, dict) else None
            if not isinstance(meta, dict):
                return False
            task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
            started_at = float(meta.get("started_at") or 0.0)
        deadline_ts = float(q._task_deadline_ts(task) or 0.0)
        if deadline_ts and now >= deadline_ts:
            return True
        if started_at > 0:
            absolute_ceiling = float(q.get_task_abs_ceiling_sec())
            return max(0.0, now - started_at) >= absolute_ceiling
        return False
    except Exception:
        # Unreadable hard-bound authority is not permission to extend a task.
        log.warning(
            "owner-stop hard-bound read failed for %s",
            task_id,
            exc_info=True,
        )
        return True


def begin_graceful_stop(task_id: str) -> None:
    """Ingress kick-off: run one orchestration pass off the HTTP thread.

    The HTTP handler answered with the immediate durable pending
    acknowledgement already; this pass arms the episode without waiting for the
    ~20s sweep tick. Crash-safe: the durable intent alone replays the whole
    episode through ``sweep_cancel_intents`` -> ``sweep_owner_stop_hold``.
    """
    from supervisor import queue as q

    try:
        from ouroboros.cancel_intents import active_intent

        intent = active_intent(q.DRIVE_ROOT, task_id)
        if isinstance(intent, dict):
            sweep_owner_stop_hold(q, task_id, intent, now=time.time())
    except Exception:
        log.warning("owner-stop ingress orchestration failed for %s", task_id, exc_info=True)


def orchestrate_graceful_stop(q: Any, task_id: str, intent: Dict[str, Any], *, now: float) -> bool:
    """One idempotent hold tick: settle descendants (Q6=A), arm the root episode.

    Returns True while the episode genuinely holds a LIVE running root; a root
    that is pending/missing returns False so custody settles it immediately
    (zero model turns — §13.1).
    """
    from ouroboros.cancel_intents import (
        SCOPE_CASCADE,
        _validated_single_cancel_target,
    )

    cascade = str(intent.get("scope") or "") == SCOPE_CASCADE
    physical_task_id = _validated_single_cancel_target(q.DRIVE_ROOT, task_id)
    descendants_settled = True
    if cascade:
        descendants_settled = _settle_descendants_hard(
            q,
            task_id,
            exclude_task_ids={task_id, physical_task_id},
        )
    with q._queue_lock:
        running_meta = (
            q.RUNNING.get(physical_task_id)
            if isinstance(q.RUNNING, dict) else None
        )
    if not isinstance(running_meta, dict):
        # The in-process direct-chat turn has no RUNNING row but drains the
        # same owner mailbox on the canonical drive: arm its episode from the
        # queue-shaped record the chat agent exposes.
        from supervisor import workers

        turn = workers.direct_chat_turn(physical_task_id)
        if turn is None:
            # PENDING (never started -> zero turns) or gone (miss lane): feed
            # custody. The sweep's generic path settles both shapes.
            return False
        running_meta = {"task": turn, "started_at": float(turn.get("_started_at") or now)}
    if not descendants_settled:
        # The cascade root never receives its paid final turn while a child is
        # still live.  Keep the owner-stop intent open and retry the bounded
        # descendant sweep on the next watchdog tick; expiry still releases
        # the ordinary hard-custody path.
        _forensic(q, task_id, "owner_stop_descendants_pending", intent)
        return True
    return _arm_owner_stop_episode(
        q,
        physical_task_id,
        intent,
        running_meta,
        now=now,
        cascade=cascade,
        logical_task_id=task_id,
    )


def _settle_descendants_hard(
    q: Any, task_id: str, *, exclude_task_ids: set[str] | None = None,
) -> bool:
    """Q6=A: live descendants are hard-stopped deepest-first, zero paid turns.

    Reuses the existing cascade subtree sweep with the ROOT excluded — each
    descendant gets its own durable intent and custody teardown, per-task
    delivery suppressed (the tree's story is the root's finalization or, on
    expiry, the one cascade summary). Idempotent: a settled subtree yields an
    empty sweep. The root id is fenced FIRST so late descendant admission is
    refused while the root finalizes (§12.2 item 3).
    """
    from supervisor.task_lifecycle import (
        CANCELLED_ROOT_FENCES,
        _cancel_subtree_sweep,
        _live_descendants_locked,
        _prune_cancellation_fences,
    )

    excluded = set(exclude_task_ids or {task_id})
    with q._queue_lock:
        CANCELLED_ROOT_FENCES[task_id] = utc_now_iso()
        _prune_cancellation_fences(protected={task_id})
    try:
        settled = set(excluded)
        # Reuse the ordinary cascade's bounded re-sweep shape.  The root and
        # its physical retry leaf are pre-settled only for this descendant
        # pass, so every other live node is hard-stopped deepest-first.
        for _sweep in range(4):
            _cancelled, outcomes = _cancel_subtree_sweep(
                q, task_id, settled,
            )
            settled.update(
                child_id
                for child_id, outcome in outcomes.items()
                if outcome in q._CANCEL_TERMINALIZED
            )
            if not outcomes:
                break
        with q._queue_lock:
            remaining = {
                str(row.get("task_id") or "")
                for row in _live_descendants_locked(q, task_id)
                if str(row.get("task_id") or "") not in excluded
            }
        if remaining:
            log.warning(
                "owner-stop descendants remain live for %s: %s",
                task_id,
                sorted(remaining),
            )
            return False
        return True
    except Exception:
        log.warning("owner-stop descendant sweep failed for %s", task_id, exc_info=True)
        return False


def owner_stop_control_id(intent: Dict[str, Any]) -> str:
    """Deterministic episode/control identity from the durable stop request."""
    return f"ownerstop:{str(intent.get('request_id') or '')}"


# The finalize control custody writes for a STOPPED in-process direct-chat turn
# (immediate policy). The loop's forced-finalization branch recognizes it and
# ends the turn WITHOUT a further model call: a retained delivery candidate or
# the typed fallback text — the honest twin of killing a worker.
REASON_OWNER_STOPPED_DIRECT_TURN = "owner_stopped_direct_chat_turn"


def _mark_owner_stop_control_drained(
    owner_ctx: Any, drive_root: Optional[pathlib.Path], task_id: str,
) -> bool:
    """Stamp the owner-stop finalize control's DELIVERY on the durable intent.

    The intent lives on the CANONICAL data root (``budget_drive_root`` first;
    a forked task's mailbox drive differs). Idempotent (first drain wins). A
    failed stamp is retried ONCE; still unconfirmed, a typed forensic event
    is appended and no extended budget is assumed: the sweep keeps the
    request+outer-cap deadline, and ``_owner_stop_window_elapsed`` reads the
    same unstamped intent, bounding the worker by that anchor."""
    try:
        from ouroboros.cancel_intents import (
            mark_finalize_control_drained,
            resolve_owner_stop_intent,
        )

        root = (
            str(getattr(owner_ctx, "budget_drive_root", "") or "")
            or (str(drive_root) if drive_root is not None else "")
        )
        if not (root and task_id):
            return False
        root_path = pathlib.Path(root)
        for _ in range(2):
            if mark_finalize_control_drained(root_path, task_id):
                return True
            _intent_task_id, row = resolve_owner_stop_intent(root_path, task_id)
            if isinstance(row, dict) and str(row.get("control_drained_at") or ""):
                return True  # already stamped: the durable anchor is confirmed
            if not isinstance(row, dict) or not row:
                # The same stop request was hardened/settled while this
                # mailbox row was being drained.  It is a stale control, not a
                # persistence failure and must not buy a final model turn.
                return False

        append_jsonl(root_path / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": "owner_stop_stamp_failed",
            "task_id": task_id,
        })
        return False
    except Exception:
        log.debug("owner-stop drain stamp failed for %s", task_id, exc_info=True)
        return False


def _owner_stop_control_is_current(
    owner_ctx: Any,
    drive_root: Optional[pathlib.Path],
    task_id: str,
    control_msg_id: str,
) -> bool:
    """Whether this deterministic mailbox control still names the live policy."""
    try:
        from ouroboros.cancel_intents import resolve_owner_stop_intent

        root = (
            str(getattr(owner_ctx, "budget_drive_root", "") or "")
            or (str(drive_root) if drive_root is not None else "")
        )
        if not (root and task_id and control_msg_id):
            return False
        _intent_task_id, intent = resolve_owner_stop_intent(
            pathlib.Path(root), task_id,
        )
        return bool(intent) and owner_stop_control_id(intent) == str(control_msg_id)
    except Exception:
        log.debug(
            "owner-stop control validation failed for %s",
            task_id,
            exc_info=True,
        )
        return False


def _narrow_round_deadline(ctx: Any, candidate: Any) -> Optional[float]:
    """Apply a later control deadline without ever widening task authority."""
    try:
        deadline = float(candidate)
    except (TypeError, ValueError, OverflowError):
        return ctx.deadline_ts
    current = ctx.deadline_ts
    if current is None:
        ctx.deadline_ts = deadline
    else:
        try:
            ctx.deadline_ts = min(float(current), deadline)
        except (TypeError, ValueError, OverflowError):
            ctx.deadline_ts = deadline
    return ctx.deadline_ts


def _owner_stop_window_elapsed(ctx: Any) -> bool:
    """Bind and check the owner-stop deadline."""
    try:
        from ouroboros import task_pacing
        from ouroboros.cancel_intents import (
            STOP_POLICY_FINALIZE,
            resolve_owner_stop_intent,
            stop_policy,
        )

        root = getattr(ctx, "status_drive_root", None) or ctx.drive_root
        if root is None or not ctx.task_id:
            return False
        _intent_task_id, intent = resolve_owner_stop_intent(
            pathlib.Path(root), ctx.task_id,
        )
        if not isinstance(intent, dict) or stop_policy(intent) != STOP_POLICY_FINALIZE:
            return False
        deadline = owner_stop_deadline_ts(
            intent, float(task_pacing.get_finalization_grace_sec()),
        )
        if deadline:
            effective_deadline = _narrow_round_deadline(ctx, deadline)
            return time.time() >= float(effective_deadline)
        return True
    except Exception:
        log.debug("owner-stop window check failed for %s", ctx.task_id, exc_info=True)
        return False


def revoke_hardened_owner_stop_control(
    q: Any, task_id: str, intent: Dict[str, Any],
) -> bool:
    """Best-effort retract an unread graceful control after Stop-now hardens it.

    The durable immediate policy is the authority and cancellation proceeds
    even if mailbox cleanup fails.  This helper removes the ordinary queued
    case through the existing append-only revocation protocol and clears the
    paired RUNNING latch without emitting the unrelated "task resumed" toast.
    The loop independently validates the current durable policy at drain time,
    closing the race where the control was already read.
    """
    try:
        from ouroboros.cancel_intents import (
            STOP_POLICY_IMMEDIATE,
            _validated_single_cancel_target,
            stop_policy,
        )

        if (
            stop_policy(intent) != STOP_POLICY_IMMEDIATE
            or not str(intent.get("hardened_at") or "")
        ):
            return False
        physical_task_id = _validated_single_cancel_target(
            q.DRIVE_ROOT, task_id,
        )
        control_id = owner_stop_control_id(intent)
        with q._queue_lock:
            meta = q.RUNNING.get(physical_task_id)
            if not isinstance(meta, dict):
                return False
            if str(meta.get("finalization_control_msg_id") or "") != control_id:
                return False
            task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
            task_drive = q._task_drive_for_task(task, physical_task_id)
        from ouroboros.owner_mailbox import revoke_owner_control

        if not revoke_owner_control(
            pathlib.Path(task_drive), physical_task_id, control_id,
        ):
            return False
        with q._queue_lock:
            current = q.RUNNING.get(physical_task_id)
            if (
                isinstance(current, dict)
                and str(current.get("finalization_control_msg_id") or "")
                == control_id
            ):
                current.pop("finalization_requested_at", None)
                current.pop("finalization_reason", None)
                current.pop("finalization_control_msg_id", None)
                q.RUNNING[physical_task_id] = current
        _forensic(q, task_id, "owner_stop_control_revoked_on_hardening", intent)
        return True
    except Exception:
        log.warning(
            "owner-stop hardened-control revocation failed for %s",
            task_id,
            exc_info=True,
        )
        return False


def _arm_owner_stop_episode(
    q: Any, task_id: str, intent: Dict[str, Any], running_meta: Dict[str, Any],
    *, now: float, cascade: bool, logical_task_id: str = "",
) -> bool:
    """Idempotently arm the coupled finalize_now control + RUNNING latch."""
    from supervisor.task_reaper import request_finalization_grace
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION

    control_id = owner_stop_control_id(intent)
    direct_turn: Optional[Dict[str, Any]] = None
    with q._queue_lock:
        meta = q.RUNNING.get(task_id)
        if not isinstance(meta, dict):
            # The in-process direct-chat turn: the caller synthesized
            # ``running_meta`` from the chat agent's record; its armed-control
            # latch lives on that record (``workers.stamp_direct_chat_turn``).
            candidate = running_meta.get("task") if isinstance(running_meta, dict) else None
            if not (isinstance(candidate, dict) and candidate.get("_is_direct_chat")):
                return False
            direct_turn = candidate
            meta = running_meta
        task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
        latch = meta if direct_turn is None else task
        if str(latch.get("finalization_control_msg_id") or "") == control_id:
            return True  # already armed; the mailbox drain dedupes by msg_id
        chat_id = int(task.get("chat_id") or 0)
        task_drive = q._task_drive_for_task(task, task_id)
    control_text = REASON_OWNER_REQUESTED_FINALIZATION
    if cascade:
        projection = _child_result_projection(q, logical_task_id or task_id)
        if projection:
            control_text = f"{control_text}\n{projection}"
    grace_deadline = owner_stop_deadline_ts(intent, queue_grace_sec(q))
    remaining = max(0, int(grace_deadline - now)) if grace_deadline else 0
    def _write_control(_turn: Any = None) -> str:
        return request_finalization_grace(
            pathlib.Path(task_drive), task_id, REASON_OWNER_REQUESTED_FINALIZATION,
            chat_id=chat_id, stamp=int(_requested_ts(intent) or now),
            control_msg_id=control_id, control_text=control_text,
            toast_text=(
                f"⏳ The owner asked task {task_id} to summarize and stop. One final "
                f"answer is being produced now (≤{remaining}s); Stop now remains "
                "available and escalates the same stop request immediately."
            ),
        )

    if direct_turn is not None:
        # Atomic against the turn's completion (its admission lock): a turn
        # that already ended arms nothing — custody settles it on the sweep.
        from supervisor import workers

        armed = workers.arm_direct_chat_turn(
            task_id, _write_control,
            latch_key="finalization_control_msg_id",
            finalization_requested_at=_requested_ts(intent) or now,
            finalization_reason=REASON_OWNER_REQUESTED_FINALIZATION,
        )
        if armed is None:
            return False
        written = str(armed.get("finalization_control_msg_id") or "")
    else:
        written = _write_control()
    if not written:
        # The control write failed; hold anyway (the deadline still bounds the
        # episode) and let the next sweep tick retry the same deterministic id.
        _forensic(q, task_id, "owner_stop_arm_failed", intent)
        return True
    with q._queue_lock:
        meta = q.RUNNING.get(task_id)
        if isinstance(meta, dict):
            meta["finalization_requested_at"] = _requested_ts(intent) or now
            meta["finalization_reason"] = REASON_OWNER_REQUESTED_FINALIZATION
            meta["finalization_control_msg_id"] = written
            q.RUNNING[task_id] = meta
    _forensic(q, task_id, "owner_stop_armed", intent)
    return True


def graceful_summary_suppressed(q: Any, task_id: str) -> bool:
    """Q4=A: suppress the cascade receipt after a SUCCESSFUL graceful stop.

    True only when the root's open cascade intent carries the finalize policy
    AND the root's durable result is COMPLETED — the owner already received the
    model's own final answer through normal delivery, and the card state says
    factual "Done" with the owner-stop marker in its details; a second summary
    message would be the duplicate Q4 forbids. Every other outcome (expiry ->
    cancelled, failed, replayed crash)
    keeps the tree's ONE receipt. The suppression is recorded as a typed
    forensic row so the crash-order evidence shows a conscious decision.
    """
    try:
        from ouroboros.cancel_intents import (
            STOP_POLICY_FINALIZE,
            _validated_single_cancel_target,
            active_intent,
            stop_policy,
        )
        from ouroboros.task_results import STATUS_COMPLETED, load_task_result

        intent = active_intent(q.DRIVE_ROOT, task_id) or {}
        if stop_policy(intent) != STOP_POLICY_FINALIZE:
            return False
        physical_task_id = _validated_single_cancel_target(q.DRIVE_ROOT, task_id)
        status = str(
            (load_task_result(q.DRIVE_ROOT, physical_task_id) or {}).get("status")
            or ""
        )
        if status != STATUS_COMPLETED:
            return False
        _forensic(q, task_id, "owner_stop_summary_suppressed", intent)
        return True
    except Exception:
        log.debug("graceful summary suppression check failed for %s", task_id, exc_info=True)
        return False


def _forensic(q: Any, task_id: str, event: str, intent: Dict[str, Any]) -> None:
    try:
        append_jsonl(
            pathlib.Path(q.DRIVE_ROOT) / "logs" / "supervisor.jsonl",
            {"ts": utc_now_iso(), "type": "owner_stop", "event": event,
             "task_id": task_id, "request_id": str(intent.get("request_id") or "")},
        )
    except Exception:
        log.debug("owner-stop forensic append failed for %s", task_id, exc_info=True)


def _child_result_projection(q: Any, task_id: str) -> str:
    """Q6=A: the bounded durable child projection for the root's ONE final turn.

    Built from the existing seams — cascade ancestry enumeration
    (``terminal_delivery._cascade_descendant_rows``) over durable rows plus the
    queue snapshot, and each child's own durable result — DELIBERATELY including
    settled-cancelled children (§12.2 item 4). Bounded: at most
    ``_CHILD_PROJECTION_MAX_ROWS`` rows with the exact omitted count, each
    result previewed to ``_CHILD_PROJECTION_PREVIEW_CHARS``. No ledger is added.
    """
    try:
        from supervisor.terminal_delivery import _cascade_descendant_rows
        from ouroboros.task_results import load_task_result

        rows = _cascade_descendant_rows(pathlib.Path(q.DRIVE_ROOT), task_id)
    except Exception:
        log.debug("owner-stop child projection failed for %s", task_id, exc_info=True)
        return ""
    if not rows:
        return ""
    lines = [
        "[CHILD_RESULTS] Your subtasks were stopped for this owner-requested "
        "finalization; their preserved durable results:",
    ]
    for index, (tid, status) in enumerate(sorted(rows.items())):
        if index >= _CHILD_PROJECTION_MAX_ROWS:
            lines.append(f"- … {len(rows) - _CHILD_PROJECTION_MAX_ROWS} more descendant(s) omitted")
            break
        preview = ""
        try:
            result = load_task_result(pathlib.Path(q.DRIVE_ROOT), tid) or {}
            status = str(result.get("status") or status or "")
            preview = " ".join(str(result.get("result") or "").split())
        except Exception:
            preview = ""
        if len(preview) > _CHILD_PROJECTION_PREVIEW_CHARS:
            preview = preview[:_CHILD_PROJECTION_PREVIEW_CHARS] + "…"
        lines.append(f"- {tid} ({status or 'unknown'}): {preview or '(no result text)'}")
    return "\n".join(lines)


def handle_finalize_now_entry(entry, owner_ctx, drive_root, task_id, controls) -> None:
    """Apply a drained KIND_FINALIZE_NOW mailbox control (extracted verbatim
    from the loop's drain dispatch; the stop-control currency/latch checks
    belong beside the rest of the owner-stop semantics in this module).

    Sets ``controls["finalize_now"]`` (and the deadline timestamp for the
    non-owner-stop reason) unless the owner-stop control is stale or its
    first-drain latch was already spent.
    """
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from ouroboros import task_pacing

    text = str(entry.get("text") or "deadline")
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line == REASON_OWNER_REQUESTED_FINALIZATION:
        if not _owner_stop_control_is_current(
            owner_ctx, drive_root, task_id, str(entry.get("msg_id") or ""),
        ):
            return
        # Owner-stop budget starts at delivery; first drain wins.
        if not _mark_owner_stop_control_drained(owner_ctx, drive_root, task_id):
            return
        if not _owner_stop_control_is_current(
            owner_ctx, drive_root, task_id, str(entry.get("msg_id") or ""),
        ):
            return
    else:
        opened = parse_deadline_ts(entry.get("ts"))
        if opened is not None:
            controls["finalize_deadline_ts"] = (
                opened.timestamp()
                + task_pacing.effective_finalization_reserve_sec(owner_ctx)
            )
    controls["finalize_now"] = text

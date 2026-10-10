"""Exact-continuation budget Resume grants and their revocation (#1196).

The owner's explicit Resume of a task paused MID-RUN mints ONE single-use grant
bound to one pause id and one resume generation; the grant rides the queue row
as ``_budget_pause_resume`` until a worker consumes it, and returns to the
pause — never to a replay or a terminal — when money vanishes, a restart
intervenes, or its revocation cannot be written (then the row is HELD, typed,
with its pause marker retained). Split out of ``supervisor/queue_transitions.py``
at that module's band ceiling: the grant lifecycle is one owner with its own
reason to change (owner Q7/Q9/Q10 semantics), and ``queue_transitions`` keeps
the general resume seam (``resume_budget_paused_task``) that calls into it.
Every call here runs with the queue lock held by that seam or by restore,
except the answered root's late-phase Resume (D10, ``resume_late_phase``):
it observes tree custody off-lock and takes the queue lock briefly itself.
"""

from __future__ import annotations

import logging
import pathlib
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


def grant_exact_budget_resume(task: Dict[str, Any], pause: Dict[str, Any],
                              *, selected_by: str = "",
                              external: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if selected_by == "sleep_wake":
        return {"ok": False, "error": "sleep_policy_required"}
    return _grant_exact_resume(task, pause, selected_by=selected_by, external=external)


def grant_exact_sleep_resume(task: Dict[str, Any], pause: Dict[str, Any],
                             *, external: Dict[str, Any]) -> Dict[str, Any]:
    """Sleep readiness has its own policy; it is never a model's child selection."""
    return _grant_exact_resume(task, pause, external=external, sleep_wake=True)


def _sleep_policy_allows(task, row, result_root):
    from types import SimpleNamespace
    from ouroboros.owner_pause import member_fence
    from supervisor.events_budget import budget_hold_fact

    return bool(row.get("reason") == "sleep" and isinstance(row.get("sleep"), dict)
                and row.get("sleep_ready") and not budget_hold_fact(task)
                and not member_fence(SimpleNamespace(task_id=task.get("id"),
                    root_task_id=task.get("root_task_id") or task.get("id"), budget_drive_root=result_root)))


def _resume_custody_refusal(result_root, task_id, row, external):
    """Exact source and off-lock custody observation must both authorize Resume."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.budget_pause import EXTERNAL_STOP_CONFIRMED, set_budget_pause

    try:
        read_actor_source_bytes(result_root, task_id, row["source_ref"])
    except Exception:
        return {"ok": False, "error": "pause_source_unreadable", "action": "cancel_or_new_run"}
    # Custody observed outside the queue lock by the caller, else read now.
    if not isinstance(external, dict):
        return {"ok": False, "error": "custody_observation_required"}
    if external.get("custody_read") != "ok":
        return {"ok": False, "error": "external_custody_unreadable",
                "detail": str(external.get("error") or ""), "action": "retry_or_cancel"}
    # ``stop_confirmed`` is the only terminal fact; every other run stays under custody.
    # An owner Pause's started critic is not the member's own writer: it finishes
    # separately and its result is collected later (owner 2026-10-08, full variant).
    owner_paused = str(row.get("reason") or "") == "owner"
    unsettled = [run for run in (external.get("runs") or [])
                 if isinstance(run, dict) and str(run.get("state") or "") != EXTERNAL_STOP_CONFIRMED
                 and not (owner_paused and run.get("review_owned"))]
    if unsettled:
        try:
            set_budget_pause(result_root, task_id, {**row, "external_runs": external},
                             expected_pause_id=str(row.get("pause_id") or ""), expected_state=str(row.get("state") or ""))
        except Exception:
            log.debug("Fresh custody observation could not be recorded on %s", task_id, exc_info=True)
        return {"ok": False, "error": "external_runs_unsettled",
                "runs": [{key: run.get(key) for key in ("run_id", "state", "stop_outcome")} for run in unsettled],
                "action": "wait_for_delegated_runs_to_settle_or_cancel_them"}
    return None


def _owner_resume_fence(result_root, task_id, external, *, root_task_id="", selected_by=""):
    """Bind the fresh whole-tree observation to the still-closed owner fence."""
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    root_task_id = root_task_id or task_id
    observed = (external or {}).get("owner_pause_tree")
    if not isinstance(observed, dict) or observed.get("error"):
        return "", {"ok": False, "error": "owner_pause_custody_unreadable"}
    try:
        if root_task_id != task_id:
            origin = load_task_result(result_root, root_task_id, strict=True) or {}
            if selected_by or origin.get("status") not in _TRULY_TERMINAL_STATUSES:
                return "", {"ok": False, "error": "root_still_paused", "root_task_id": root_task_id}
        current_fence = read_fence(result_root, root_task_id)
    except Exception:
        return "", {"ok": False, "error": "owner_pause_custody_unreadable"}
    if not fence_closed(current_fence) or current_fence != observed.get("fence"):
        return "", {"ok": False, "error": "selection_authority_changed"}
    blockers = _owner_pause_blockers(observed.get("blockers"))
    if blockers:
        return "", {"ok": False, "error": "owner_pause_effects_unsettled",
                    "blockers": blockers, "action": "wait_for_effect_settlement"}
    return str(current_fence.get("fence_id") or ""), None


def _owner_pause_blockers(blockers: Any) -> List[Dict[str, Any]]:
    """The tree custody an owner Pause's Resume waits for: everything except the
    reviewers the Pause lets finish (their delegated runs and model sends)."""
    from supervisor.owner_pause_control import review_finishing

    return [b for b in (blockers or []) if isinstance(b, dict) and not review_finishing(b)]


def resume_warm_owner_pause_root(task_id: str, *, selected_by: str = "") -> Optional[Dict[str, Any]]:
    """The owner's Resume of a tree whose ROOT parked WARM under its Pause; None when not one.

    Nothing was re-queued for such a root: its worker keeps the stack while a
    started critic finishes (owner 2026-10-08, full variant), so there is no
    exact grant to mint on a pause row. The same refusals apply as for a cold
    root (Restart/Panic hold, a live Stop, unsettled task-owned custody, global
    money and the root tree's own cap); then ONE single-use ``resume_grant`` is
    recorded on the fence as it is released, the queue latch is lifted, cold
    descendants become eligible exactly as after a cold root's Resume (owner
    Q9), and the parked worker wakes itself on the open fence. A warm CHILD
    under a terminal root is selected into ``selected_members`` the same way;
    under a live root it answers ``root_still_paused``.
    """
    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.owner_pause import (
        FENCE_RELEASED, fence_closed, launch_lock, read_fence, set_fence_state,
    )
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result
    from supervisor import queue as q
    from supervisor.continuation_admission import action_writers
    from supervisor.events_budget import hold_root_resume_descendants
    from supervisor.owner_pause_control import _warm_paused_direct_turn, warm_paused_member
    from supervisor.state import budget_remaining
    from supervisor.workers import direct_chat_turn

    task_id = str(task_id or "").strip()
    with q._queue_lock:
        meta = q.RUNNING.get(task_id)
        task = dict(meta.get("task") or {}) if isinstance(meta, dict) else {}
    direct = None if task else direct_chat_turn(task_id)
    task = task or dict(direct or {})
    if not task:
        return None
    result_root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    root_task_id = str(task.get("root_task_id") or task_id)
    try:
        fence = read_fence(result_root, root_task_id)
    except Exception:
        return {"ok": False, "error": "owner_pause_custody_unreadable"}
    # A pooled member's parked row rides RUNNING; a direct actor's only its durable row.
    warm = warm_paused_member(meta) if direct is None else _warm_paused_direct_turn(result_root, task_id)
    if not fence_closed(fence) or not warm:
        return None
    fence_id = str(fence.get("fence_id") or "")
    if any((pathlib.Path(q.DRIVE_ROOT) / "state" / name).exists()
           for name in ("owner_restart_no_resume.flag", "panic_stop.flag")):
        return {"ok": False, "error": "restart_no_resume", "action": "wait_or_cancel"}
    try:
        if has_active_intent(result_root, task_id, strict=True):
            return {"ok": False, "error": "cancel_intent_active"}
    except Exception:
        return {"ok": False, "error": "cancellation_authority_unavailable"}
    if root_task_id != task_id:
        try:
            origin = load_task_result(result_root, root_task_id, strict=True) or {}
        except Exception:
            return {"ok": False, "error": "owner_pause_custody_unreadable"}
        if selected_by or origin.get("status") not in _TRULY_TERMINAL_STATUSES:
            return {"ok": False, "error": "root_still_paused", "root_task_id": root_task_id,
                    "action": "resume_root_first"}
    try:
        blockers = _owner_pause_blockers(action_writers(
            q, root_task_id, drive_root=result_root, owner_pause_fence_id=fence_id))
    except Exception as exc:
        return {"ok": False, "error": "owner_pause_custody_unreadable", "detail": str(exc)[:200]}
    if blockers:
        return {"ok": False, "error": "owner_pause_effects_unsettled", "blockers": blockers,
                "action": "wait_for_effect_settlement"}
    refusal = _global_money_refusal(q, budget_remaining) or _root_money_refusal(result_root, task, root_task_id)
    if refusal:
        return refusal
    grant = {"grant_id": uuid.uuid4().hex, "granted_at": utc_now_iso(), "granted_at_ts": time.time(),
             "single_use": True, "selected_by": str(selected_by or "owner"), "authority": "explicit_resume",
             "generation": int(fence.get("generation") or 0), "warm": True}
    try:
        with launch_lock(result_root, root_task_id):
            current = read_fence(result_root, root_task_id)
            if current != fence:
                return {"ok": False, "error": "selection_authority_changed"}
            if root_task_id == task_id:
                set_fence_state(result_root, root_task_id, fence_id=fence_id, state=FENCE_RELEASED,
                                expected_state=str(fence.get("state") or ""),
                                release_reason="owner_resume", resume_grant=grant)
            else:
                selected = dict(fence.get("selected_members") or {})
                selected[task_id] = {"grant_id": grant["grant_id"], "selected_at": grant["granted_at"], "warm": True}
                set_fence_state(result_root, root_task_id, fence_id=fence_id, state=str(fence.get("state") or ""),
                                expected_state=str(fence.get("state") or ""), selected_members=selected)
    except Exception as exc:
        return {"ok": False, "error": "grant_not_recorded", "detail": str(exc)[:200]}
    held_siblings: List[str] = []
    if root_task_id == task_id:
        with q._queue_lock:
            latch = q.BUDGET_ROOT_FENCES.get(root_task_id)
            if isinstance(latch, dict) and latch.get("cause") == "owner_pause" and str(latch.get("fence_id") or "") == fence_id:
                q.BUDGET_ROOT_FENCES.pop(root_task_id, None)
                held_siblings, _markers, _rebound = hold_root_resume_descendants(q, root_task_id, latch, grant)
            persisted = q.persist_queue_snapshot(reason="owner_pause_warm_resumed")
    else:
        persisted = True
    q.append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                   {"ts": utc_now_iso(), "type": "owner_pause_resumed", "task_id": task_id,
                    "root_task_id": root_task_id, "fence_id": fence_id, "grant_id": grant["grant_id"],
                    "warm": True, "selected_by": grant["selected_by"], "held_siblings": held_siblings,
                    "owner_visible": True, "toast_once": f"{task_id}:owner-resume:{grant['grant_id']}"})
    return {"ok": True, "task_id": task_id, "root_task_id": root_task_id, "warm": True,
            "grant_id": grant["grant_id"], "held_siblings": held_siblings, "snapshot_persisted": bool(persisted)}


def _global_money_refusal(q: Any, budget_remaining: Any) -> Optional[Dict[str, Any]]:
    try:
        # Authoritative, never the admit-only stale snapshot: a grant is money.
        remaining = budget_remaining(q.load_state(), strict=True, allow_stale=False)
    except Exception:
        return {"ok": False, "error": "monetary_authority_unavailable"}
    if remaining <= 0:
        return {"ok": False, "error": "budget_still_exhausted", "action": "increase_budget_then_resume"}
    return None


def _root_money_refusal(result_root: Any, task: Dict[str, Any], root_task_id: str) -> Optional[Dict[str, Any]]:
    from ouroboros.usage_accounting import refresh_root_accounting
    from ouroboros.usage_admission import task_accounting_key

    # ONE fresh strict ledger read is this grant's monetary authority (a Continue's
    # successor: its whole-work GROUP under the original cap), never the display cache.
    tree = refresh_root_accounting(result_root, task_accounting_key(result_root, task, root_task_id), strict=True)
    trees = [tree]
    if isinstance(tree, dict):
        from ouroboros.usage_admission import task_money_snapshot
        trees.append(task_money_snapshot(result_root, task, root_task_id))
    for tree in trees:
        if not isinstance(tree, dict):
            # Unknown tree spend is not room: an unreadable ledger refuses typed.
            return {"ok": False, "error": "root_accounting_unavailable",
                    "action": "retry_or_cancel"}
        if tree.get("integrity_degraded"):
            return {"ok": False, "error": "root_accounting_degraded",
                    "action": "retry_or_cancel"}
        # Known spend decides, the reservation's own rule (#1487); holds do not.
        limit, known = tree.get("root_limit_usd"), tree.get("settled_usd")
        if limit is not None:
            if known is None:
                return {"ok": False, "error": "root_accounting_degraded",
                        "action": "retry_or_cancel"}
            if float(known) >= float(limit) - 1e-9:
                return {"ok": False, "error": "root_hard_cap_exhausted",
                        "action": "increase_budget_then_resume"}
    return None


def _grant_exact_resume(task: Dict[str, Any], pause: Dict[str, Any], *, selected_by: str = "",
                        external: Optional[Dict[str, Any]] = None,
                        sleep_wake: bool = False) -> Dict[str, Any]:
    """Grant from durable pause, fresh custody, money, Stop and clock checks.
    Explicit model selection needs a root grant; sleep has its own policy.
    """
    from ouroboros.budget_pause import (
        LIVE_PAUSE_STATES, STATE_PAUSED, STATE_RESUME_GRANTED,
        exact_pause_marker, set_budget_pause,
    )
    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.config import get_task_abs_ceiling_sec
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now
    from ouroboros.model_wait import execution_elapsed_seconds
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result
    from supervisor.events_budget import (
        BUDGET_HOLD_KEY, HOLD_REVOCATION_UNWRITTEN, HOLD_MALFORMED_RESUME_IDENTITY,
        hold_budget_row, live_root_resume_grant, hold_root_resume_descendants,
    )
    from supervisor import queue as q
    from supervisor.state import budget_remaining

    task_id = str(task.get("id") or "")
    checkpoint = pause.get("checkpoint") if isinstance(pause.get("checkpoint"), dict) else {}
    pause_id = str(checkpoint.get("pause_id") or "")
    if not pause_id.strip():
        return {"ok": False, "error": "malformed_pause_identity"}
    result_root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    if any((pathlib.Path(q.DRIVE_ROOT) / "state" / name).exists()
           for name in ("owner_restart_no_resume.flag", "panic_stop.flag")):
        return {"ok": False, "error": "restart_no_resume", "action": "wait_or_cancel"}
    try:
        result_row = load_task_result(result_root, task_id, strict=True) or {}
    except Exception:
        return {"ok": False, "error": "pause_record_unreadable", "action": "cancel_or_new_run"}
    row = result_row.get("budget_pause") if isinstance(result_row.get("budget_pause"), dict) else {}
    if task.get("_owner_hold") or result_row.get("_owner_hold"):
        return {"ok": False, "error": "owner_held"}
    if result_row.get("status") in _TRULY_TERMINAL_STATUSES:
        return {"ok": False, "error": "task_terminal"}
    if (row and row.get("state") in LIVE_PAUSE_STATES and row.get("source_ref")
            and str(row.get("pause_id") or "") and str(row.get("pause_id") or "") != pause_id):
        # Refresh the stale locator from the durable pause, retaining the queue's fence id.
        refreshed = exact_pause_marker(row, default_root=str(pause.get("root_task_id")
                                                             or task.get("root_task_id") or task_id))
        if pause.get("fence_id"):
            refreshed["fence_id"] = pause["fence_id"]
        task["_budget_pause"] = refreshed
        q.append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                       {"ts": utc_now_iso(), "type": "budget_pause_marker_refreshed", "task_id": task_id,
                        "stale_pause_id": pause_id, "pause_id": str(row.get("pause_id") or ""),
                        "pause_generation": int(row.get("pause_generation") or 0)})
        pause, checkpoint = refreshed, refreshed["checkpoint"]
        pause_id = str(row.get("pause_id") or "")
    if (not row or row.get("pause_id") != pause_id or row.get("state") not in LIVE_PAUSE_STATES
            or not row.get("source_ref")):
        return {"ok": False, "error": "pause_record_missing", "action": "cancel_or_new_run"}
    from ouroboros.owner_pause import fence_closed

    owner_fence_id = ""
    root_task_id = str(task.get("root_task_id") or task_id)
    observed_fence = ((external or {}).get("owner_pause_tree") or {}).get("fence") or {}
    if (fence_closed(observed_fence) or root_task_id == task_id
            and (row.get("reason") == "owner" or fence_closed(result_row.get("owner_pause")))):
        owner_fence_id, refusal = _owner_resume_fence(result_root, task_id, external,
                                                     root_task_id=root_task_id, selected_by=selected_by)
        if refusal:
            return refusal
    if int(row.get("task_attempt") or 0) != int(task.get("_attempt") or 1):
        # Another attempt than the checkpoint's: the loop would refuse the grant on arrival.
        return {"ok": False, "error": "pause_attempt_mismatch", "action": "cancel_or_new_run",
                "row_attempt": int(row.get("task_attempt") or 0), "queue_attempt": int(task.get("_attempt") or 1)}
    hold = task.get(BUDGET_HOLD_KEY) if isinstance(task.get(BUDGET_HOLD_KEY), dict) else {}
    if sleep_wake and not _sleep_policy_allows(task, row, result_root):
        return {"ok": False, "error": "sleep_wake_vetoed"}
    if hold.get("reason") == HOLD_MALFORMED_RESUME_IDENTITY:
        return {"ok": False, "error": HOLD_MALFORMED_RESUME_IDENTITY}
    live_grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    if "grant" in row and not str(live_grant.get("grant_id") or "").strip():
        return {"ok": False, "error": "malformed_grant_identity"}
    if row.get("state") == STATE_RESUME_GRANTED and not live_grant.get("revoked_at"):
        # Orphaned = no queue carrier holds this undispatched grant any more. That
        # covers a typed revocation hold AND a crash between the durable grant and
        # the snapshot (restore saw only `_budget_pause`), which would otherwise
        # answer resume_already_granted forever (Astra run-a882315dbcd7 #1).
        orphaned = bool(
            not any(isinstance(item.get("_budget_pause_resume"), dict)
                        and str(item["_budget_pause_resume"].get("grant_id") or "") == str(live_grant.get("grant_id") or "")
                        for item in list(q.PENDING) + [m.get("task") for m in q.RUNNING.values() if isinstance(m, dict)]
                        if isinstance(item, dict))
        )
        if not orphaned:
            return {"ok": False, "error": "resume_already_granted",
                    "grant_id": live_grant.get("grant_id")}
        revoked = {**live_grant, "revoked_at": utc_now_iso(),
                   "revoke_reason": (f"deferred:{hold.get('reason')}:{str(hold.get('detail') or '')[:120]}"
                                     if hold else "orphaned_grant_without_carrier")}
        try:
            set_budget_pause(result_root, task_id, {**row, "state": STATE_PAUSED, "grant": revoked},
                             expected_pause_id=pause_id, expected_state=STATE_RESUME_GRANTED,
                             expected_grant_id=str(live_grant.get("grant_id") or ""))
        except Exception as exc:
            return {"ok": False, "error": HOLD_REVOCATION_UNWRITTEN, "detail": str(exc)[:200],
                    "grant_id": live_grant.get("grant_id"), "action": "retry_or_cancel"}
        row = {**row, "state": STATE_PAUSED, "grant": revoked}
    refusal = _resume_custody_refusal(result_root, task_id, row, external)
    if refusal:
        return refusal
    row = {**row, "external_runs": external,  # an owner Pause's sent work: settled per THIS read
           **({"settlement": "settled"} if row.get("settlement") else {})}
    try:
        if has_active_intent(pathlib.Path(q.DRIVE_ROOT), task_id, strict=True):
            return {"ok": False, "error": "cancel_intent_active"}
    except Exception:
        return {"ok": False, "error": "cancellation_authority_unavailable"}
    deadline = parse_deadline_ts(task.get("deadline_at") or (task.get("task_contract") or {}).get("deadline_at"))
    if deadline is not None and deadline <= utc_now():
        return {"ok": False, "error": "deadline_passed"}
    now = time.time()
    started = float(row.get("started_at") or checkpoint.get("started_at") or 0.0)
    paused_at = float(row.get("paused_at") or checkpoint.get("paused_at") or now)
    prior_paused = float(row.get("paused_duration_sec") or 0.0)
    # ONE shared clock: wall time minus the quota union minus the paused carrier.
    executed_sec = execution_elapsed_seconds(
        {"started_at": started, "budget_paused_sec": prior_paused,
         "model_wait_quota_clock": row.get("model_wait_quota_clock") or {}}, paused_at)
    ceiling = get_task_abs_ceiling_sec()  # None = unlimited lifetime; 0 = exhausted
    if ceiling is not None and started and executed_sec >= float(ceiling):
        return {"ok": False, "error": "lifetime_exhausted", "executed_sec": round(executed_sec, 1)}
    refusal = _global_money_refusal(q, budget_remaining)
    if refusal:
        return refusal
    # The grant binds to the queue row's own lineage (``root_task_id`` above), the
    # one ``budget_resume_dispatch_allowed`` revalidates: a saved pause's root field
    # never overrides it (an older writer saved the member's own id there).
    root_grant = live_root_resume_grant(q, root_task_id, result_root) if root_task_id != task_id else {}
    if selected_by and root_task_id != task_id and not sleep_wake:
        # Q9: lineage alone grants nothing; model selection needs this root's live grant.
        if not root_grant:
            return {"ok": False, "error": "root_resume_grant_missing",
                    "root_task_id": root_task_id, "action": "resume_root_first"}
    if str(pause.get("scope") or "") == "root":
        refusal = _root_money_refusal(result_root, task, root_task_id)
        if refusal:
            return refusal
    # Owner Q9: a descendant cannot be resumed under a root that is itself still paused.
    if root_task_id != task_id and any(
            str(item.get("id") or "") == root_task_id and isinstance(item.get("_budget_pause"), dict)
            for item in q.PENDING):
        return {"ok": False, "error": "root_still_paused", "root_task_id": root_task_id,
                "action": "resume_root_first"}
    # A later Resume raises the generation; older grants cannot become live again.
    generation = int(row.get("resume_generation") or 0) + 1
    pause_generation = int(row.get("pause_generation") or 0)
    grant = {
        **({"owner_pause_fence_id": owner_fence_id} if owner_fence_id else {}),
        "grant_id": uuid.uuid4().hex, "granted_at": utc_now_iso(), "granted_at_ts": now,
        "single_use": True, "paused_duration_sec": prior_paused + max(0.0, now - paused_at),
        "executed_sec_before_pause": round(executed_sec, 3),
        "pause_id": pause_id, "pause_generation": pause_generation, "generation": generation,
        "selected_by": "sleep_wake" if sleep_wake else str(selected_by or "owner"),
        "authority": "sleep_readiness" if sleep_wake else "explicit_resume",
        "sleep_id": str((row.get("sleep") or {}).get("sleep_id") or "") if sleep_wake else "",
        "root_grant_id": str(root_grant.get("grant_id") or ""),
        "root_resume_generation": int(root_grant.get("generation") or 0),
        "root_fence_id": (str((q.BUDGET_ROOT_FENCES.get(root_task_id) or {}).get("fence_id") or "")
                          if root_task_id != task_id else ""),
        "refresh_planning_threshold": str(row.get("rail") or "") in {
            "graceful_ceiling", "wrapup_last_fit", "soft_land"},
    }
    prior_pause = dict(pause)
    try:
        # CAS on the validated state: a concurrent writer refuses here, typed.
        set_budget_pause(result_root, task_id,
                         {**row, "state": STATE_RESUME_GRANTED, "grant": grant,
                          "resume_generation": generation},
                         expected_pause_id=pause_id, expected_state=str(row.get("state") or ""))
    except Exception as exc:
        return {"ok": False, "error": "grant_not_recorded", "detail": str(exc)[:200]}
    task.pop("_budget_pause", None)
    task["_budget_pause_resume"] = {
        **checkpoint, "grant_id": grant["grant_id"], "granted_at": grant["granted_at"],
        "grant_generation": generation, "pause_id": pause_id, "pause_generation": pause_generation,
        "paused_duration_sec": grant["paused_duration_sec"], "pause": prior_pause,
        "external_runs": external,
        **({"sleep_exclusion_since": now} if row.get("reason") == "sleep" else {}),
        **{key: grant[key] for key in ("selected_by", "authority", "sleep_id", "root_grant_id", "root_resume_generation", "root_fence_id")},
    }
    task["budget_resumed_at"] = grant["granted_at"]
    # Release a re-validated restore/revocation hold in place (``selected`` flips,
    # nothing is erased); the prior hold is kept for the snapshot rollback below.
    released_hold = hold if hold and not hold.get("selected") else None
    if released_hold is not None:
        task[BUDGET_HOLD_KEY] = {**released_hold, "selected": True, "selected_at": utc_now_iso(),
                                 "selected_by": grant["selected_by"], "released_reason": "exact_resume_granted"}
    fence = q.BUDGET_ROOT_FENCES.get(root_task_id)
    fence_released = False
    held_siblings: List[str] = []
    released_markers: Dict[str, Dict[str, Any]] = {}
    rebound_holds: Dict[str, Dict[str, Any]] = {}
    if (task_id == root_task_id and isinstance(fence, dict) and str(fence.get("fence_id") or "")
            == str(owner_fence_id or prior_pause.get("fence_id") or fence.get("fence_id"))):
        # The root's own Resume lifts its admission latch: exact descendants keep
        # their OWN `_budget_pause` rows and are only ELIGIBLE; the model selects each (Q9).
        q.BUDGET_ROOT_FENCES.pop(root_task_id, None)
        fence_released = True
        held_siblings, released_markers, rebound_holds = hold_root_resume_descendants(q, root_task_id, fence, grant)
    if not q.persist_queue_snapshot(reason="budget_exact_resume_granted"):
        task.pop("_budget_pause_resume", None)
        task["_budget_pause"] = prior_pause
        if released_hold is not None:
            task[BUDGET_HOLD_KEY] = released_hold
        if fence_released:
            q.BUDGET_ROOT_FENCES[root_task_id] = fence
        for member in q.PENDING:
            member_id = str(member.get("id") or "")
            if member_id in set(held_siblings):
                member.pop(BUDGET_HOLD_KEY, None)
                if member_id in released_markers:
                    member["_budget_pause"] = released_markers[member_id]
            elif member_id in rebound_holds:
                member[BUDGET_HOLD_KEY] = rebound_holds[member_id]
        try:
            set_budget_pause(result_root, task_id, row, expected_pause_id=pause_id,
                             expected_state=STATE_RESUME_GRANTED,
                             expected_grant_id=str(grant["grant_id"]))
            return {"ok": False, "error": "snapshot_not_persisted"}
        except Exception as rollback_error:
            detail = str(rollback_error)[:120]
            log.warning("Exact resume grant rollback remains unpersisted for %s", task_id, exc_info=True)
        # Rollback failed: the hold keeps this orphaned grant's identity so the
        # next Resume writes its deferred revocation before minting.
        hold_budget_row(
            task, reason=HOLD_REVOCATION_UNWRITTEN,
            detail=f"snapshot_not_persisted:{detail}",
            extra={"pause_id": pause_id, "grant_id": str(grant["grant_id"]),
                   "root_task_id": root_task_id,
                   **({"prior_hold_reason": str(released_hold.get("reason") or "")}
                      if released_hold else {})},
            result_root=result_root)
        task["_budget_pause"] = prior_pause
        return {"ok": False, "error": "snapshot_not_persisted",
                "held": HOLD_REVOCATION_UNWRITTEN, "grant_id": str(grant["grant_id"])}
    try:
        from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

        write_task_result(
            result_root, task_id, STATUS_SCHEDULED, reason_code="",
            resource_limit={**prior_pause, "status": "resume_granted", "resumed_at": grant["granted_at"],
                            "grant_id": grant["grant_id"], "auto_resume": False},
        )
    except Exception:
        log.debug("Failed to project exact budget resume for %s", task_id, exc_info=True)
    eligible = [str(r.get("id") or "") for r in q.PENDING
                if isinstance(r.get("_budget_pause"), dict) and r["_budget_pause"].get("exact_continuation")
                and str(r.get("root_task_id") or "") == root_task_id and str(r.get("id") or "") != task_id]
    q.append_jsonl(
        q.DRIVE_ROOT / "logs" / "events.jsonl",
        {"ts": utc_now_iso(), "type": "budget_task_explicitly_resumed", "task_id": task_id,
         "root_task_id": root_task_id, "same_generation": True, "exact_continuation": True,
         "grant_id": grant["grant_id"], "grant_generation": generation,
         "selected_by": grant["selected_by"],
         "paused_duration_sec": grant["paused_duration_sec"],
         "eligible_descendants": eligible if task_id == root_task_id else [],
         "held_siblings": held_siblings, "rebound_held_siblings": sorted(rebound_holds),
         "released_hold": str((released_hold or {}).get("reason") or "")},
    )
    return {"ok": True, "task_id": task_id, "root_task_id": root_task_id, "exact_continuation": True,
            "grant_id": grant["grant_id"], "grant_generation": generation,
            "paused_duration_sec": round(grant["paused_duration_sec"], 1),
            "eligible_descendants": eligible if task_id == root_task_id else [],
            "held_siblings": held_siblings, "rebound_held_siblings": sorted(rebound_holds),
            **({"released_hold": str(released_hold.get("reason") or "")} if released_hold else {})}


def revoke_exact_budget_resume(task: Dict[str, Any], reason: str) -> bool:
    """Return a granted-but-undispatched task to its exact pause (queue lock held).

    Money can vanish between the grant and the dispatch (a sibling spent it) and
    a restart may intervene; the grant is single-use and must not be dispatched
    into a refused send. Identity decides what may be written: a revocation is
    recorded ONLY against the pause, state and grant this handoff names
    (compare-and-set on all three). A grant the durable row says was CONSUMED
    is never re-armed: the task ran on, the queue row is stale — it takes a
    typed hold, its handoff leaves, and no ``_budget_pause`` marker is
    re-minted over a task that is not paused. If the durable row already
    carries a NEWER pause (pauseA -> Resume -> pauseB), or a different grant,
    nothing is written over it — the spent handoff simply leaves the queue
    row, which re-reads the current pause or holds; at restore, a newer grant
    that never reached a worker either is revoked too, so the next Resume finds
    a pause, not a grant no row carries. A failed write leaves the row
    un-dispatchable rather than carrying a stale grant.
    """
    from ouroboros.budget_pause import (
        LIVE_PAUSE_STATES, STATE_PAUSED, STATE_RESUME_GRANTED, STATE_RESUMED, budget_pause_row,
        exact_pause_marker, set_budget_pause,
    )
    from supervisor.events_budget import (
        HOLD_GRANT_CONSUMED_STALE_ROW, HOLD_RECORD_UNREADABLE_AT_REVOKE, HOLD_RESTART_REVOCATION_UNWRITTEN,
        HOLD_REVOCATION_UNWRITTEN, HOLD_STALE_GRANT_SUPERSEDED, HOLD_MALFORMED_RESUME_IDENTITY, hold_budget_row,
    )
    from supervisor import queue as q

    handoff = task.get("_budget_pause_resume") if isinstance(task.get("_budget_pause_resume"), dict) else None
    if handoff is None:
        return False
    task_id = str(task.get("id") or "")
    result_root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    prior_pause = dict(handoff.get("pause") or {}) if isinstance(handoff.get("pause"), dict) else {}
    expected_pause_id = str(handoff.get("pause_id") or "").strip()
    handoff_grant_id = str(handoff.get("grant_id") or "")
    if not expected_pause_id or not handoff_grant_id.strip() or not prior_pause:
        task["_budget_pause"] = prior_pause
        hold_budget_row(task, reason=HOLD_MALFORMED_RESUME_IDENTITY, detail="malformed_resume_identity",
                        extra={"pause_id": expected_pause_id, "grant_id": handoff_grant_id}, result_root=None)
        return False
    try:
        row = budget_pause_row(result_root, task_id)
    except Exception:
        log.warning("Exact resume grant revocation could not read the pause row for %s",
                    task_id, exc_info=True)
        # The saved pause is retained: the marker stays the locator the owner's
        # next Resume validates; the hold keeps the row off the dispatch path.
        task["_budget_pause"] = prior_pause
        hold_budget_row(task, reason=HOLD_RECORD_UNREADABLE_AT_REVOKE,
                         detail=str(reason or ""),
                         extra={"pause_id": expected_pause_id, "grant_id": handoff_grant_id},
                         result_root=result_root)
        return False
    current_pause_id = str(row.get("pause_id") or "")
    row_state = str(row.get("state") or "")
    grant = dict(row["grant"]) if isinstance(row.get("grant"), dict) else {}
    if (not current_pause_id.strip() or (current_pause_id == expected_pause_id or row_state == STATE_RESUME_GRANTED)
            and not str(grant.get("grant_id") or "").strip()):
        task["_budget_pause"] = prior_pause
        hold_budget_row(task, reason=HOLD_MALFORMED_RESUME_IDENTITY, detail="malformed_durable_resume_identity",
                        extra={"pause_id": expected_pause_id, "grant_id": handoff_grant_id}, result_root=None)
        return False
    same_pause = current_pause_id == expected_pause_id
    same_grant = str(grant.get("grant_id") or "") == handoff_grant_id
    consumed = bool(grant.get("consumed_at")) or row_state == STATE_RESUMED
    if same_pause and same_grant and consumed:
        # NEVER re-armed: the loop consumed this grant, so the task ran (or
        # ran and ended). This queue row is a stale carrier; it is held typed,
        # off the dispatch path, with no pause marker — and a restore fences it
        # as the running work it names.
        task.pop("_budget_pause_resume", None)
        if isinstance(task.get("_owner_wait_resume"), dict):
            # The task ran ON past this grant and LATER parked in an owner wait:
            # the spent carrier is simply retired and the newer planned-restart
            # handoff decides the row (its own restore gate re-validates it).
            # Fencing the row here would drop that valid continuation (#1196, F3).
            q.append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                           {"ts": utc_now_iso(), "type": "budget_resume_carrier_retired",
                            "task_id": task_id, "reason": str(reason or ""),
                            "grant_id": str(grant.get("grant_id") or ""),
                            "pause_id": current_pause_id,
                            "consumed_at": grant.get("consumed_at"),
                            "retained": "owner_wait_resume"})
            return False
        task["_budget_pause_consumed"] = {
            "pause_id": current_pause_id, "grant_id": str(grant.get("grant_id") or ""),
            "consumed_at": grant.get("consumed_at"), "reason": str(reason or ""),
        }
        hold_budget_row(task, reason=HOLD_GRANT_CONSUMED_STALE_ROW, detail=str(reason or ""),
                         extra={"pause_id": current_pause_id, "grant_id": str(grant.get("grant_id") or "")},
                         result_root=None)  # the task's own status is not ours to rewrite
        q.append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                       {"ts": utc_now_iso(), "type": "budget_resume_grant_revoke_refused_consumed",
                        "task_id": task_id, "reason": str(reason or ""),
                        "grant_id": str(grant.get("grant_id") or ""), "pause_id": current_pause_id,
                        "consumed_at": grant.get("consumed_at")})
        return False
    superseded = not same_pause or not same_grant
    if superseded:
        log.warning("Stale exact-resume handoff for %s (grant %s, pause %s) is NOT written over the "
                    "current pause %s", task_id, handoff_grant_id, expected_pause_id,
                    current_pause_id or "<none>")
        task.pop("_budget_pause_resume", None)
        newer_undispatched = (
            str(reason or "") == "restart_before_dispatch"
            and row_state == STATE_RESUME_GRANTED
            and str(grant.get("grant_id") or "")
            and not grant.get("revoked_at") and not grant.get("consumed_at")
        )
        if newer_undispatched:
            # The snapshot lagged a NEWER grant; no worker survives a restart, so
            # that grant never reached one either. Revoked under its OWN
            # identity, or the row is held naming it (the next Resume writes
            # the deferred revocation before minting again).
            revoked = {**grant, "revoked_at": utc_now_iso(),
                       "revoke_reason": "restart_before_dispatch:superseding_grant_undispatched"}
            try:
                set_budget_pause(result_root, task_id, {**row, "state": STATE_PAUSED, "grant": revoked},
                                 expected_pause_id=current_pause_id, expected_state=STATE_RESUME_GRANTED,
                                 expected_grant_id=str(grant.get("grant_id") or ""))
                row = {**row, "state": STATE_PAUSED, "grant": revoked}
                row_state = STATE_PAUSED
            except Exception:
                log.warning("Superseding grant of %s could not be revoked at restore; held", task_id, exc_info=True)
                task["_budget_pause"] = exact_pause_marker(
                    row, default_root=str(task.get("root_task_id") or task_id))
                hold_budget_row(task, reason=HOLD_RESTART_REVOCATION_UNWRITTEN,
                                 detail=str(reason or ""),
                                 extra={"pause_id": current_pause_id, "grant_id": str(grant.get("grant_id") or "")},
                                 result_root=result_root)
                return False
        if row_state in LIVE_PAUSE_STATES and row.get("source_ref"):
            task["_budget_pause"] = exact_pause_marker(
                row, default_root=str(task.get("root_task_id") or task_id))
        else:
            hold_budget_row(task, reason=HOLD_STALE_GRANT_SUPERSEDED,
                             detail=str(reason or ""),
                             extra={"handoff_pause_id": expected_pause_id,
                                    "current_pause_id": current_pause_id},
                             result_root=result_root)
        q.append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                       {"ts": utc_now_iso(), "type": "budget_resume_grant_revoke_superseded",
                        "task_id": task_id, "reason": str(reason or ""),
                        "handoff_grant_id": handoff_grant_id,
                        "handoff_pause_id": expected_pause_id, "current_pause_id": current_pause_id,
                        "current_grant_id": grant.get("grant_id"),
                        "superseding_grant_revoked": bool(newer_undispatched)})
        return False
    try:
        grant.update(revoked_at=utc_now_iso(), revoke_reason=str(reason or ""))
        set_budget_pause(result_root, task_id, {**row, "state": STATE_PAUSED, "grant": grant},
                         expected_pause_id=current_pause_id, expected_state=row_state,
                         expected_grant_id=str(grant.get("grant_id") or ""))
    except Exception:
        log.warning("Exact resume grant revocation not recorded for %s", task_id, exc_info=True)
        # Retained, typed, un-dispatchable: the marker stays so the owner's next
        # Resume finds the exact pause; the grant named here is proven
        # undispatched (its handoff leaves the row with this hold), so that
        # Resume writes the deferred revocation before minting a new grant.
        task["_budget_pause"] = prior_pause
        hold_budget_row(task, reason=(HOLD_RESTART_REVOCATION_UNWRITTEN
                                      if str(reason or "") == "restart_before_dispatch"
                                      else HOLD_REVOCATION_UNWRITTEN),
                         detail=str(reason or ""),
                         extra={"pause_id": current_pause_id, "grant_id": grant.get("grant_id")},
                         result_root=result_root)
        return False
    task["_budget_pause"] = prior_pause
    task.pop("_budget_pause_resume", None)
    q.append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                   {"ts": utc_now_iso(), "type": "budget_resume_grant_revoked", "task_id": task_id,
                    "reason": str(reason or ""), "grant_id": grant.get("grant_id"),
                    "pause_id": current_pause_id})
    return True


_LATE_RESUMING: set = set()
_LATE_RESUMING_LOCK = threading.Lock()


def _late_phase_refusal(q: Any, root: pathlib.Path, task_id: str, row: Dict[str, Any],
                        record: Dict[str, Any], fence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The exact Resume's own refusals for a saved late phase; nothing is consumed or degraded."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now
    from supervisor.continuation_admission import action_writers
    from supervisor.events_budget import budget_hold_fact
    from supervisor.state import budget_remaining

    if any((pathlib.Path(q.DRIVE_ROOT) / "state" / name).exists()
           for name in ("owner_restart_no_resume.flag", "panic_stop.flag")):
        return {"ok": False, "error": "restart_no_resume", "action": "wait_or_cancel"}
    try:
        if has_active_intent(root, task_id, strict=True):
            return {"ok": False, "error": "cancel_intent_active"}
    except Exception:
        return {"ok": False, "error": "cancellation_authority_unavailable"}
    with q._queue_lock:
        # A Continue of this answered root that may already write would overlap the remainder.
        successors = [str(t.get("id") or "") for t in [*q.PENDING, *(m.get("task") or {} for m in q.RUNNING.values())]
                      if isinstance(t, dict) and ((t.get("metadata") or {}).get("continuation") or {}).get(
                          "predecessor_task_id") == task_id and (t.get("id") in q.RUNNING or not budget_hold_fact(t))]
    # The whole tree's sent work must have settled, exactly as for a paused loop's Resume.
    # Reviewers already launched finish separately; Resume does not wait for them.
    blockers = [{"kind": "continuation_successor", "task_id": tid} for tid in successors] + _owner_pause_blockers(
        action_writers(q, task_id, drive_root=root, owner_pause_fence_id=str(fence.get("fence_id") or "")))
    if blockers:
        return {"ok": False, "error": "owner_pause_effects_unsettled", "action": "wait_for_effect_settlement",
                "blockers": blockers}
    deadline = parse_deadline_ts(row.get("deadline_at") or (row.get("task_contract") or {}).get("deadline_at")
                                 or (row.get("metadata") or {}).get("deadline_at"))
    if deadline is not None and deadline <= utc_now():
        return {"ok": False, "error": "deadline_passed"}
    refusal = _global_money_refusal(q, budget_remaining) or _root_money_refusal(
        root, {**row, "id": task_id}, str(row.get("root_task_id") or task_id))
    if refusal:
        return refusal
    if record:
        try:
            read_actor_source_bytes(root, task_id, record["payload_ref"])
        except Exception:
            return {"ok": False, "error": "pause_source_unreadable", "action": "cancel_or_new_run"}
    return None


def _release_late_fence(q: Any, root: pathlib.Path, task_id: str, row: dict, fence: Dict[str, Any]) -> Dict[str, Any]:
    """Resume of an answered root's Pause with no saved post-task remainder (D10).

    Nothing left at all: the tree census already releases it. Only deferred,
    unsent late work left (a late review still preparing): this explicit
    Resume reopens the fence once nothing sent remains in flight.
    """
    from ouroboros.owner_pause import FENCE_RELEASED, launch_lock, set_fence_state
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight
    from ouroboros.review_operation import task_has_live_review_operation
    from supervisor.continuation_admission import conflicting_writers, release_settled_continuations
    from supervisor.owner_pause_control import refresh_owner_pause_tree

    remainder = refresh_owner_pause_tree(task_id) != FENCE_RELEASED
    if remainder:
        refusal = _late_phase_refusal(q, root, task_id, row, {}, fence)
        if refusal:
            return refusal
        try:
            blocked = (post_task_synthesis_in_flight(root, task_id)
                       or task_has_live_review_operation(root, task_id, sent_only=True)
                       or conflicting_writers(q, task_id, drive_root=root,
                                              owner_pause_fence_id=str(fence.get("fence_id") or "")))
        except Exception:
            blocked = True
        if blocked:
            return {"ok": False, "error": "owner_pause_effects_unsettled", "action": "wait_for_effect_settlement"}
        try:
            with launch_lock(root, task_id):
                from ouroboros.owner_pause import read_fence
                from ouroboros.acceptance_late import resume_paused_acceptance_preparations
                if read_fence(root, task_id) != fence:
                    return {"ok": False, "error": "selection_authority_changed"}
                resume_paused_acceptance_preparations(root, task_id, fence)
                set_fence_state(root, task_id, fence_id=str(fence.get("fence_id") or ""), state=FENCE_RELEASED,
                                expected_state=str(fence.get("state") or ""), release_reason="owner_resume_late")
        except Exception as exc:
            return {"ok": False, "error": "selection_authority_changed", "detail": str(exc)[:200]}
        with q._queue_lock:
            if (q.BUDGET_ROOT_FENCES.get(task_id) or {}).get("cause") == "owner_pause":
                q.BUDGET_ROOT_FENCES.pop(task_id, None)
            q.persist_queue_snapshot(reason="owner_pause_late_released")
    release_settled_continuations(task_id)
    return {"ok": True, "task_id": task_id, "root_task_id": task_id, "late_phase": "resumed" if remainder else "settled",
            "owner_pause_released": True}


def resume_late_phase(task_id: str) -> Optional[Dict[str, Any]]:
    """The owner's Resume of an answered root's paused remainder (D10); None when not one.

    The exact Resume's refusals apply unchanged (Restart/Panic hold, a live
    Stop, unsettled tree custody, the calendar deadline, global money and the
    root tree's own cap): a refusal consumes nothing and leaves the saved phase
    and its fence as they were. Otherwise ONE single-use grant is recorded on
    the pause and handed to the existing late-phase executor, which consumes it
    where the work starts. A closed fence whose late work already ended is
    settled by the tree census (released when nothing remains).
    """
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.post_task_checkpoint import late_phase_pause_record, update_late_phase_pause
    from ouroboros.post_task_synthesis import revoke_late_phase_grant
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result
    from supervisor import queue as q
    from supervisor import workers

    root = pathlib.Path(q.DRIVE_ROOT)
    try:
        row = load_task_result(root, task_id, strict=True) or {}
        fence = read_fence(root, task_id) if row else {}
    except Exception:
        return {"ok": False, "error": "pause_record_unreadable"}
    if row.get("status") not in _TRULY_TERMINAL_STATUSES:
        return None
    record = late_phase_pause_record(row)
    if not record:
        return _release_late_fence(q, root, task_id, row, fence) if fence_closed(fence) else None
    with _LATE_RESUMING_LOCK:
        if task_id in _LATE_RESUMING:
            return {"ok": False, "error": "resume_already_granted"}
        _LATE_RESUMING.add(task_id)
    try:
        live = record.get("grant") if isinstance(record.get("grant"), dict) else {}
        if live and not live.get("consumed_at") and not live.get("revoked_at"):
            # No executor holds it in this process: an orphan of a failed start.
            revoke_late_phase_grant(root, task_id, reason="orphaned_grant_without_executor")
        refusal = _late_phase_refusal(q, root, task_id, row, record, fence)
        if refusal:
            return refusal
        grant = {"grant_id": uuid.uuid4().hex, "granted_at": utc_now_iso(), "single_use": True,
                 "selected_by": "owner", "authority": "explicit_resume",
                 "fence_id": str(fence.get("fence_id") or "") if fence_closed(fence) else ""}

        def mint(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            prior = current.get("grant") if isinstance(current.get("grant"), dict) else {}
            if (current.get("pause_id") != record.get("pause_id") or current.get("stopped_at")
                    or prior and not prior.get("consumed_at") and not prior.get("revoked_at")):
                return None
            return {**current, "grant": grant, "resume_generation": int(current.get("resume_generation") or 0) + 1}

        try:
            if update_late_phase_pause(root, task_id, mint) is None:
                return {"ok": False, "error": "selection_authority_changed"}
        except Exception as exc:
            return {"ok": False, "error": "grant_not_recorded", "detail": str(exc)[:200]}
        from ouroboros.agent_task_pipeline import recover_pending_root_post_task_synthesis

        if not recover_pending_root_post_task_synthesis(root, getattr(workers, "REPO_DIR", None),
                                                        resume_task_id=task_id):
            revoke_late_phase_grant(root, task_id, reason="late_phase_resume_not_started")
            return {"ok": False, "error": "selection_authority_changed"}
    finally:
        with _LATE_RESUMING_LOCK:
            _LATE_RESUMING.discard(task_id)
    with q._queue_lock:
        latch = q.BUDGET_ROOT_FENCES.get(task_id) or {}
        # A newer Pause minted over this grant keeps its own latch (the remainder parks again).
        if latch.get("cause") == "owner_pause" and str(latch.get("fence_id") or "") in {"", grant["fence_id"]}:
            q.BUDGET_ROOT_FENCES.pop(task_id, None)
        persisted = q.persist_queue_snapshot(reason="owner_pause_late_resumed")
    q.append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                   {"ts": utc_now_iso(), "type": "owner_pause_resumed", "task_id": task_id, "root_task_id": task_id,
                    "fence_id": grant["fence_id"], "grant_id": grant["grant_id"], "late_phase": True})
    return {"ok": True, "task_id": task_id, "root_task_id": task_id, "late_phase": "resumed",
            "grant_id": grant["grant_id"], "stage": record.get("stage"), "snapshot_persisted": bool(persisted)}

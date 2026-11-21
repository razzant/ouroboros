"""Exact single-use grants bind one task, source, pause and generation.
An explicit Resume checks custody, controls, money and clocks under the queue's control seam.
Grant and revocation mutate the queued carrier under the queue lock; the exact grant's
custody observation runs off-lock, as do the tree observations of warm and answered roots.
Automatic wakes stage a blocked handoff and publish outside Q. The durable row CAS opens it.
An interrupted unconsumed grant returns to the same pause. A failed revocation keeps a typed hold.
"""
from __future__ import annotations

import logging
import pathlib
import time
import uuid
from typing import Any, Dict, List, Optional

from ouroboros.control_actor import control_actor, observed_control_stop, resume_author
from ouroboros.delegate_recovery import owner_restart_in_progress
from ouroboros.work_stop import SELECTION_KEY, clear_refusal, continuation_refusal, release_refusal, retain_consumed_selection
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)
# Facts of one ``observe_exact_resume`` read: the grant decides on them and records none of them.
_OBSERVATION_ONLY = ("resume_fence_id", "local_resume_blockers", "start_money_refusal")


def observe_exact_resume(q, task, task_id, result_root, *, gaps):
    """Off-lock custody for this saved task. Return its observation or an immediate refusal.

    Every saved reason is observed, a known sleep included: a money pause also asks its unconfirmed runs
    to stop (``request_stop``); an owner Pause or a retained sleep/authority park observes without cancelling.
    Money, Stop and the explicit choice about an unconfirmed start are decided on this observation by the grant.
    """
    from ouroboros.budget_pause import observe_task_runs
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.sleep_custody import RETAINED_REASONS
    from supervisor.continuation_admission import action_writers

    tree = str(task.get("root_task_id") or task_id)
    try:
        fence = read_fence(result_root, tree)
    except Exception:
        return None, {"ok": False, "error": "owner_pause_custody_unreadable"}
    reason = str(task["_budget_pause"].get("reason") or "")
    closed = fence_closed(fence)
    owner_paused = closed or reason == "owner"  # the owner's fence can overlay a budget/sleep checkpoint
    external = observe_task_runs(result_root, task_id, reason="budget_resume_uncovered_cost",
                                 request_stop=not (owner_paused or reason in RETAINED_REASONS))
    fence_id = str(fence.get("fence_id") or "")
    external["resume_fence_id"] = fence_id
    # A saved loop is not proof that its handed tools/processes ended.
    try:
        blockers = action_writers(q, tree, drive_root=result_root, gaps=gaps if owner_paused else None,
                                  owner_pause_fence_id=fence_id if closed else "")
    except Exception as exc:
        return None, {"ok": False, "error": "owner_pause_custody_unreadable", "detail": type(exc).__name__}
    # A global checkpoint can also hold an ordinary start. Its decision spends
    # the tree's authority just as a root-scoped Resume does.
    if (any(isinstance(run, dict) and run.get("ordinary_start") for run in external.get("runs") or ())
            or (closed and any(b.get("ordinary_start") for b in blockers))):
        external["start_money_refusal"] = _root_money_refusal(result_root, task, tree)
    if owner_paused:
        external["owner_pause_tree"] = {"fence": fence, "blockers": blockers}
    if closed:
        return external, None  # the grant revalidates the fence before using the observation
    # After the fence reopens, a selected child's own local sender still owns its call.
    observed_separately = {"running_member", "dispatchable_member", "delegated_run"}
    local = [b for b in blockers if not b.get("task_id") or
             (b["task_id"] == task_id and b.get("kind") not in observed_separately and not b.get("review_owned"))]
    if local:
        external["local_resume_blockers"] = local
    return external, None


def grant_exact_budget_resume(task: Dict[str, Any], pause: Dict[str, Any],
                              *, selected_by: str = "",
                              external: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if selected_by in {"sleep_wake", "authority_readable"}:
        return {"ok": False, "error": "sleep_policy_required"}
    return _grant_exact_resume(task, pause, selected_by=selected_by, external=external)


def grant_exact_sleep_resume(task: Dict[str, Any], pause: Dict[str, Any],
                             *, external: Dict[str, Any], batch=None) -> Dict[str, Any]:
    """Readiness acts through the host regardless of which caller observed it."""
    from ouroboros.control_actor import host_control
    with host_control():
        return _grant_exact_resume(task, pause, external=external, sleep_wake=True, batch=batch)


def _sleep_policy_allows(task, row, result_root):
    from supervisor import queue as q
    from supervisor.sleep_wake import _vetoed, authority_readable
    from ouroboros.installation_stop import installation_panic
    if _vetoed(q, task) or installation_panic(result_root):
        return False
    if row.get("reason") == "authority":
        return authority_readable(task, row, result_root)
    return row.get("reason") == "sleep" and isinstance(row.get("sleep"), dict) and bool(row.get("sleep_ready"))


def _resume_custody_refusal(result_root, task_id, row, external, owner_paused=False):
    """Exact source and off-lock custody observation must both authorize Resume."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.budget_pause import EXTERNAL_STOP_CONFIRMED, set_budget_pause
    from ouroboros.sleep_custody import RETAINED_REASONS, return_allowed

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
    # An owner Pause's started critic is not the member's own writer: it finishes separately, its result collected
    # later (owner 2026-10-08, full variant), whatever saved row the Pause's closed fence overlays; an authority park
    # leaves a started panel's run to its controller. A review that cannot apply to the work is never a writer.
    reason = str(row.get("reason") or "")
    panel_owned = owner_paused or reason in {"owner", "authority"}
    unsettled = [run for run in (external.get("runs") or [])
                 if isinstance(run, dict) and str(run.get("state") or "") != EXTERNAL_STOP_CONFIRMED
                 and not (panel_owned and run.get("review_owned")) and run.get("independent_review") is not True]
    if unsettled and reason in RETAINED_REASONS:
        if return_allowed(task_id, row, external if reason == "sleep" else {**external, "runs": unsettled}):
            unsettled = []  # same author's retained dependency, not a stop receipt
    if unsettled:
        try:
            set_budget_pause(result_root, task_id, {**row, "external_runs": external},
                             expected_pause_id=str(row.get("pause_id") or ""), expected_state=str(row.get("state") or ""))
        except Exception:
            log.debug("Fresh custody observation could not be recorded on %s", task_id, exc_info=True)
        return {"ok": False, "error": "external_runs_unsettled",
                "runs": [{key: run[key] for key in ("run_id", "state", "stop_outcome", "task_id", "invocation_id",
                                                    "ordinary_start", "resume_decisions") if key in run} for run in unsettled],
                "action": ("wait_for_delegated_runs_to_settle" if reason in RETAINED_REASONS
                           else "wait_for_delegated_runs_to_settle_or_cancel_them")}
    return None


def _owner_resume_fence(result_root, task_id, external, *, actor, root_task_id="", selected_by=""):
    """Bind the fresh whole-tree observation to the still-closed owner fence."""
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    root_task_id = root_task_id or task_id
    observed = (external or {}).get("owner_pause_tree")
    if not isinstance(observed, dict) or observed.get("error"):
        return "", {"ok": False, "error": "owner_pause_custody_unreadable"}
    try:
        if not actor:
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
        return str(current_fence["fence_id"]), {"ok": False, "error": "owner_pause_effects_unsettled",
                    "blockers": blockers, "action": "wait_for_effect_settlement"}
    return str(current_fence.get("fence_id") or ""), None


def _owner_pause_blockers(blockers: Any) -> List[Dict[str, Any]]:
    """The tree custody an owner Pause's Resume waits for: everything except the
    reviewers the Pause lets finish (their delegated runs and model sends)."""
    from supervisor.owner_pause_control import review_finishing

    return [b for b in (blockers or []) if isinstance(b, dict) and not review_finishing(b)]


def _resume_actor(root, tree, task_id):
    """Cascade Stop belongs to the logical root and single Stop to the selected physical attempt."""
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.owner_pause import root_actor

    try:
        if any(cancel_pending(root, tid, strict=True) for tid in {tree, task_id}):
            return False, {"ok": False, "error": "cancel_intent_active"}
    except Exception:
        return False, {"ok": False, "error": "cancellation_authority_unavailable"}
    try:
        return root_actor(root, tree, task_id, strict=True), None
    except Exception:
        return False, {"ok": False, "error": "owner_pause_custody_unreadable"}


def _exact_custody_refusal(root, tree, task_id, row, external, fence_id, tree_refusal, actor):
    from ouroboros.delegate_resume import decide
    from ouroboros.owner_pause import read_fence

    if not isinstance(external, dict):
        return {"ok": False, "error": "custody_observation_required"}
    observed_id, local = external.get("resume_fence_id"), external.get("local_resume_blockers", [])
    external = {k: v for k, v in external.items() if k not in _OBSERVATION_ONLY}
    own = _resume_custody_refusal(root, task_id, row, external, bool(fence_id))
    if own and own["error"] != "external_runs_unsettled":
        return own
    blockers = list((tree_refusal or {}).get("blockers") or local)
    for run in (own or {}).get("runs", ()):
        if not any(b.get("kind") == "delegated_run" and all(b.get(k) == v for k, v in run.items()) for b in blockers):
            blockers.append({"kind": "delegated_run", **run})
    if observed_id is not None and not fence_id and any(b.get("ordinary_start") for b in blockers):
        try:
            current_id = str(read_fence(root, tree).get("fence_id") or "")
        except Exception:
            return {"ok": False, "error": "owner_pause_custody_unreadable"}
        if current_id != observed_id:
            return {"ok": False, "error": "selection_authority_changed"}
    cause = tree_refusal or ({} if local else own) or {}
    refusal = decide(root, tree, task_id, blockers, fence_id=observed_id or fence_id,
                     pause_id=row["pause_id"], tree_actor=actor, error=cause.get("error", "owner_pause_effects_unsettled"),
                     action=cause.get("action", "wait_for_effect_settlement"))
    if refusal and refusal["error"] == "external_runs_unsettled":
        refusal["runs"] = [{k: v for k, v in block.items() if k != "kind"} for block in refusal["blockers"]]
    return refusal


def _restart_or_update_refusal(q: Any, task: Any, result_root: pathlib.Path, task_id: str) -> Optional[Dict[str, Any]]:
    """No Resume while an owner Restart runs, or while an update holds repo-writer admission closed (#2191)."""
    from ouroboros.task_results import load_task_result
    from supervisor import workers
    if owner_restart_in_progress(q.DRIVE_ROOT):
        return {"ok": False, "error": "restart_no_resume", "action": "wait_or_cancel"}
    reason = str(getattr(q, "repo_writer_admission_closed", workers.repo_writer_admission_closed)() or "")
    if not reason or (isinstance(task, dict) and getattr(
            q, "repo_writer_task_allowed", workers.repo_writer_task_allowed)(task)):
        return None
    chat_id = task.get("chat_id") if isinstance(task, dict) else None
    try:  # the typed refusal stands even when the owner notice cannot be sent
        if chat_id is None:
            chat_id = (load_task_result(result_root, task_id, strict=False) or {}).get("chat_id")
        if chat_id is not None:
            from ouroboros.task_finalization import host_operation_reply_kwargs
            getattr(q, "send_with_budget", workers.send_with_budget)(
                int(chat_id), "🔒 An update is in progress; press Resume after the restart.",
                **host_operation_reply_kwargs(None, "", ""), role="system", system_type="managed_update_notice")
    except Exception:
        log.warning("update-admission Resume notice for %s was not sent", task_id, exc_info=True)
    return {"ok": False, "error": "update_in_progress", "reason": reason, "action": "resume_after_restart"}


def resume_warm_owner_pause_root(task_id: str, *, selected_by: str = "",
                                 gaps: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
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
    under a live root it answers ``root_still_paused``. ``gaps`` receives the
    damaged rows its tree census ties to no tree.
    """
    from ouroboros.owner_pause import (
        FENCE_RELEASED, fence_closed, launch_lock, read_fence, set_fence_state,
    )
    from ouroboros.model_wait import execution_elapsed_seconds
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result
    from supervisor import queue as q
    from supervisor.continuation_admission import action_writers
    from supervisor.events_budget import (
        _root_resume_state, hold_root_resume_descendants,
    )
    from supervisor.owner_pause_control import (
        _warm_paused_direct_turn, complete_owner_pause_latch, warm_paused_member,
    )
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
    if refusal := release_refusal(observed_control_stop(
            result_root, task, budget=q.BUDGET_ROOT_FENCES), control_actor()):
        return refusal
    root_task_id = str(task.get("root_task_id") or task_id)
    try:
        fence = read_fence(result_root, root_task_id)
    except Exception:
        return {"ok": False, "error": "owner_pause_custody_unreadable"}
    # A pooled member's parked row rides RUNNING; a direct actor's only its durable row.
    warm = warm_paused_member(meta) if direct is None else _warm_paused_direct_turn(result_root, task_id)
    tree_selection = bool(fence.get("state") == FENCE_RELEASED and (fence.get("resume_grant") or {}).get("tree_only"))
    if not (fence_closed(fence) or tree_selection) or not warm:
        return None
    fence_id = str(fence.get("fence_id") or "")
    if refusal := _restart_or_update_refusal(q, task, result_root, task_id):
        return refusal
    actor, refusal = _resume_actor(result_root, root_task_id, task_id)
    if refusal:
        return refusal
    if not actor:
        try:
            origin = load_task_result(result_root, root_task_id, strict=True) or {}
        except Exception:
            return {"ok": False, "error": "owner_pause_custody_unreadable"}
        if (selected_by and not tree_selection) or origin.get("status") not in _TRULY_TERMINAL_STATUSES:
            return {"ok": False, "error": "root_still_paused", "root_task_id": root_task_id,
                    "action": "resume_root_first"}
    try:
        blockers = _owner_pause_blockers(action_writers(
            q, root_task_id, drive_root=result_root, owner_pause_fence_id=fence_id, gaps=gaps))
    except Exception as exc:
        return {"ok": False, "error": "owner_pause_custody_unreadable", "detail": str(exc)[:200]}
    refusal = _global_money_refusal(q, budget_remaining, [task]) or _root_money_refusal(result_root, task, root_task_id)
    if refusal:
        return refusal
    try:
        with q._queue_lock:
            root_state = _root_resume_state(q, root_task_id, result_root,
                                           actor_id=task_id if actor else None)
            clock = {key: meta[key] for key in ("started_at", "budget_paused_sec", "model_wait_quota_clock",
                                              "sleep_parked_at") if isinstance(meta, dict) and key in meta}
    except Exception:
        return {'ok': False, 'error': 'owner_pause_custody_unreadable'}
    from ouroboros.delegate_resume import decide, decisions_in_use
    record = load_task_result(result_root, task_id) or {}
    wait = record.get("owner_wait") or {}
    wait_id = str(wait.get("wait_id") or "")
    blocked_at = _warm_pause_clock_at(clock, wait)
    if direct is not None and "_execution_owner" not in task and record.get("execution_owner"):
        task["_execution_owner"] = record["execution_owner"]
    # Check at confirmation, not at the offer or before custody observation. Execution
    # stops at the park boundary; the calendar deadline does not. Neither a choice nor
    # a warm grant may acquire authority from an already ended clock.
    now = time.time()
    if refusal := resume_clock_refusal(task, execution_elapsed_seconds(
            {**wait, **clock, "task": task}, min(now, blocked_at) if blocked_at is not None else now), now):
        return refusal
    if refusal := decide(result_root, root_task_id, task_id, blockers, fence_id=fence_id,
                         wait_id=wait_id, tree_actor=actor):
        return refusal
    from ouroboros.presence_carrier import kept_allowance_lift, lift_allowance, lifted_record, owner_allowance_lift
    # A warm stack's record is its durable row (its queue row may be a lane seed). The owner's decision lands there and on
    # its queue row with this release; every warm grant carries it to the stack, which a newer Pause may have parked
    # before it took an earlier grant up (#2566).
    lift = {} if selected_by or not actor else owner_allowance_lift(record, result_root)
    decision = lift or kept_allowance_lift(record)
    grant = {"by": resume_author(), "grant_id": uuid.uuid4().hex, "granted_at": utc_now_iso(),
             **({"accepted_start_risks": risks} if (risks := decisions_in_use(task_id)) else {}),
             "granted_at_ts": time.time(), "single_use": True, "selected_by": str(selected_by or "owner"),
             "authority": "explicit_resume",
             'root_resume_basis': root_state['basis'], "generation": int(fence.get("generation") or 0), "warm": True,
             **({"producer_allowance_lifted": decision} if decision else {})}
    projected = None
    if actor:
        # The census above passed never-started members on the durable fence alone when the
        # Pause was ``latch_pending``. Complete its latch BEFORE the release: the lift below
        # then holds them for explicit selection (Q9) exactly as after a projected latch.
        with q._queue_lock:
            projected = complete_owner_pause_latch(root_task_id, fence)
    try:
        with launch_lock(result_root, root_task_id):
            current = read_fence(result_root, root_task_id)
            if wait_id and decisions_in_use(task_id):
                parked = (load_task_result(result_root, task_id, strict=True) or {}).get("owner_wait") or {}
                if parked.get("wait_id") != wait_id or not warm_paused_member({"owner_wait": parked}):
                    return {"ok": False, "error": "selection_authority_changed"}
            if current == fence and actor:
                set_fence_state(result_root, root_task_id, fence_id=fence_id, state=FENCE_RELEASED,
                                expected_state=str(fence.get("state") or ""),
                                release_reason="owner_resume", resume_grant=grant,
                                row_update=(lambda row: lifted_record(row, lift)) if lift else None)
            elif current == fence:
                selected = dict(fence.get("selected_members") or {})
                selected[task_id] = {"grant_id": grant["grant_id"], "selected_at": grant["granted_at"], "warm": True}
                set_fence_state(result_root, root_task_id, fence_id=fence_id, state=str(fence.get("state") or ""),
                                expected_state=str(fence.get("state") or ""), selected_members=selected)
    except Exception as exc:
        return {"ok": False, "error": "grant_not_recorded", "detail": str(exc)[:200]}
    if current != fence:
        if projected and not (fence_closed(current) and str(current.get("fence_id") or "") == fence_id):
            with q._queue_lock:  # its generation ended meanwhile: withdraw the latch completed for it
                if (q.BUDGET_ROOT_FENCES.get(root_task_id) or {}).get("fence_id") == fence_id:
                    q.BUDGET_ROOT_FENCES.pop(root_task_id, None)
        return {"ok": False, "error": "selection_authority_changed"}
    held_siblings: List[str] = []
    if actor:
        with q._queue_lock:
            if lift and isinstance((q.RUNNING.get(task_id) or {}).get("task"), dict):
                lift_allowance(q.RUNNING[task_id]["task"], lift)  # the row a later park keeps, published below
            latch = q.BUDGET_ROOT_FENCES.get(root_task_id)
            if isinstance(latch, dict) and latch.get("cause") == "owner_pause" and str(latch.get("fence_id") or "") == fence_id:
                # This newer warm grant supersedes the root's queue carriers,
                # not its consumed durable history. Mutate the LIVE row, not
                # the observation copy above; a newer latch keeps its fields.
                live = (q.RUNNING.get(task_id) or {}).get("task")
                if isinstance(live, dict):
                    live.pop('_budget_pause_resume', None)
                    # The released durable fence now holds this warm grant: it is the row's queue selection.
                    basis = {**root_state['basis'], 'owner': {
                        **root_state['basis']['owner'],
                        'grant_id': grant['grant_id'],
                        'grant_generation': int(grant['generation']),
                        'revoked': False,
                    }}
                    live[SELECTION_KEY] = {
                        'by': grant['by'], 'selected_by': 'owner',
                        'selected_at': grant['granted_at'],
                        'root_task_id': root_task_id,
                        'root_grant_id': grant['grant_id'],
                        'root_resume_generation': int(grant['generation']),
                        'root_resume_basis': basis,
                    }
                q.BUDGET_ROOT_FENCES.pop(root_task_id, None)
                held_siblings, _markers, _rebound = hold_root_resume_descendants(
                    q, root_task_id, latch, grant, actor_id=task_id)
            persisted = q.publish_queue_snapshot(reason="owner_pause_warm_resumed")  # durable before release
    else:
        persisted = True
    q.append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                   {"ts": utc_now_iso(), "type": "owner_pause_resumed", "task_id": task_id,
                    "root_task_id": root_task_id, "fence_id": fence_id, "grant_id": grant["grant_id"],
                    "warm": True, "selected_by": grant["selected_by"], "held_siblings": held_siblings,
                    "owner_visible": True, "toast_once": f"{task_id}:owner-resume:{grant['grant_id']}"})
    return {"ok": True, "task_id": task_id, "root_task_id": root_task_id, "warm": True,
            "grant_id": grant["grant_id"], "held_siblings": held_siblings, "snapshot_persisted": bool(persisted)}


def resume_clock_refusal(task, executed_sec, now):
    from ouroboros.config import resolve_task_execution_timeout_sec
    from supervisor.queue_timeouts import _task_deadline_ts

    if _task_deadline_ts(task, missing=float("inf")) <= now:
        return {"ok": False, "error": "deadline_passed"}
    ceiling = resolve_task_execution_timeout_sec(task)
    if ceiling is not None and executed_sec >= ceiling:
        return {"ok": False, "error": "lifetime_exhausted", "executed_sec": round(executed_sec, 1)}
    return None


def _warm_pause_clock_at(clock, wait):
    from ouroboros.deadline_utils import parse_deadline_ts

    started = clock.get("sleep_parked_at") or wait.get("sleep_started_at")
    if isinstance(started, (int, float)) and started > 0:
        return started
    parked = parse_deadline_ts((wait.get("owner_pause") or {}).get("parked_at") or wait.get("parked_at"))
    return parked.timestamp() if parked else None


def _global_money_refusal(q: Any, budget_remaining: Any,
                          starts: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    try:
        # Authoritative, never the admit-only stale snapshot: a grant is money.
        remaining = budget_remaining(q.load_state(), strict=True, allow_stale=False)
    except Exception:
        return {"ok": False, "error": "monetary_authority_unavailable"}
    from supervisor.worker_assignment import needs_paid_admission

    # A reached global limit refuses paid admission only: a Resume proceeds when nothing it ``starts`` needs it,
    # and their every send still meets the reservation. Without ``starts`` (a money pause names a refused paid
    # send) it refuses.
    if remaining <= 0 and (starts is None or any(needs_paid_admission(task) for task in starts)):
        return {"ok": False, "error": "budget_still_exhausted", "action": "increase_budget_then_resume"}
    return None


def _root_money_refusal(result_root: Any, task: Dict[str, Any], root_task_id: str) -> Optional[Dict[str, Any]]:
    from ouroboros.usage_accounting import refresh_root_accounting
    from ouroboros.usage_admission import task_accounting_key, task_billing_fields
    from ouroboros.usage_ledger import BillingAuthorityUnavailable

    try:
        task_billing_fields(task, root_task_id, None, result_root)
        # A successor keeps the original group cap. Both spend readings must be fresh.
        tree = refresh_root_accounting(result_root, task_accounting_key(result_root, task, root_task_id), strict=True)
        trees = [tree]
        if isinstance(tree, dict):
            from ouroboros.usage_admission import task_money_snapshot
            trees.append(task_money_snapshot(result_root, task, root_task_id))
    except BillingAuthorityUnavailable as exc:
        return {"ok": False, "error": "billing_authority_unavailable", "billing_authority": exc.kind, "cause": exc.cause,
                "action": "fix_billing_authority_then_resume" if exc.kind == "unreadable" else "cancel_or_new_run"}
    except Exception:
        log.warning("Resume could not read accounting for %s", root_task_id, exc_info=True)
        return {"ok": False, "error": "root_accounting_unavailable", "action": "retry_or_cancel"}
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


def _unlifted_record(row: Dict[str, Any], lift: Dict[str, Any]) -> Dict[str, Any]:
    """``presence_carrier.lifted_record`` undone in the write withdrawing its grant: number back, decision gone."""
    record = dict(row)
    metadata = record["metadata"] = dict(record.get("metadata") or {})
    metadata.pop("producer_allowance_lifted", None)
    metadata["root_cost_ceiling_usd"] = lift["ceiling_usd"]
    return record


def take_back_owner_resume_lift(task: Dict[str, Any], before: Dict[str, Any], result_root: Any) -> None:
    """A grantless release that did not land decided nothing (``presence_carrier.owner_resume_lift``): the durable row
    takes its number back; the queue row is its caller's to restore. ``before`` is that row as the lift found it: a
    decision it already kept stays. A store that refuses keeps the owner's decision on that record, nothing released,
    until the row's next release reads it; the door still refuses, and the failure is logged."""
    from ouroboros.presence_carrier import kept_allowance_lift
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema
    from ouroboros.task_store import patch_document

    lift = kept_allowance_lift(task)
    if not lift or kept_allowance_lift(before):
        return

    def undo(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        require_writable_task_result_schema(row)
        return stamp_task_result_schema(_unlifted_record(row, lift)) if row else None

    try:
        patch_document(pathlib.Path(result_root), str(task.get("id") or ""), undo)
    except Exception:
        log.warning("The owner's decision on %s stays on its record: its release did not land", task.get("id"),
                    exc_info=True)


def _revoke_uncarried_grant(root, task_id, row, hold):
    """CAS away the disproven handoff, leaving this exact pause available to grant."""
    from ouroboros.budget_pause import STATE_PAUSED, STATE_RESUME_GRANTED, set_budget_pause
    from supervisor.events_budget import HOLD_REVOCATION_UNWRITTEN

    live_grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    grant_id = str(live_grant.get("grant_id") or "")
    revoked = {**live_grant, "revoked_at": utc_now_iso(), "revoke_reason": (
        f"deferred:{hold.get('reason')}:{str(hold.get('detail') or '')[:120]}" if hold else "orphaned_grant_without_carrier")}
    restored = {**row, "state": STATE_PAUSED, "grant": revoked}
    try:
        set_budget_pause(root, task_id, restored, expected_pause_id=row["pause_id"],
                         expected_state=STATE_RESUME_GRANTED, expected_grant_id=grant_id)
    except Exception as exc:
        return row, {"ok": False, "error": HOLD_REVOCATION_UNWRITTEN, "detail": str(exc)[:200],
                     "grant_id": grant_id, "action": "retry_or_cancel"}
    return restored, None


def _retire_orphaned_grant(q, task_id, result_root, row, hold, *, defer_revoke=False):
    from ouroboros.budget_pause import STATE_RESUME_GRANTED

    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    grant_id = str(grant.get("grant_id") or "")
    if "grant" in row and not grant_id.strip():
        return row, {"ok": False, "error": "malformed_grant_identity"}
    if row.get("state") != STATE_RESUME_GRANTED or grant.get("revoked_at"):
        return row, None
    tasks = list(q.PENDING) + [meta.get("task") for meta in q.RUNNING.copy().values() if isinstance(meta, dict)]
    if any(isinstance(task, dict) and isinstance(carrier := task.get("_budget_pause_resume"), dict)
           and carrier.get("grant_id") == grant_id for task in tasks):
        return row, {"ok": False, "error": "resume_already_granted", "grant_id": grant_id}
    if defer_revoke:
        # A named start decision is in flight: the caller re-checks this orphan
        # against fresh custody before the handoff is CAS'd away.
        return row, None
    return _revoke_uncarried_grant(result_root, task_id, row, hold)


def _grant_exact_resume(task: Dict[str, Any], pause: Dict[str, Any], *, selected_by: str = "",
                        external: Optional[Dict[str, Any]] = None,
                        sleep_wake: bool = False, batch=None) -> Dict[str, Any]:
    """Grant from the durable pause, fresh custody, money, Stop and clock checks."""
    from ouroboros.budget_pause import (
        LIVE_PAUSE_STATES, STATE_PAUSED, STATE_RESUME_GRANTED,
        exact_pause_marker, saved_budget_pause, set_budget_pause,
    )
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.config import explicit_execution_timeout_sec
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now
    from ouroboros.model_wait import execution_elapsed_seconds
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result
    from supervisor.events_budget import (
        BUDGET_HOLD_KEY, HOLD_REVOCATION_UNWRITTEN,
        hold_budget_row, _root_resume_state, _root_resume_grants, hold_root_resume_descendants,
    )
    from supervisor import queue as q
    from supervisor.state import budget_remaining

    task_id = str(task.get("id") or "")
    checkpoint = pause.get("checkpoint") if isinstance(pause.get("checkpoint"), dict) else {}
    pause_id = str(checkpoint.get("pause_id") or "")
    if not pause_id.strip():
        return {"ok": False, "error": "malformed_pause_identity"}
    result_root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    if refusal := release_refusal(observed_control_stop(
            result_root, task, budget=q.BUDGET_ROOT_FENCES), control_actor()):
        return refusal
    if refusal := _restart_or_update_refusal(q, task, result_root, task_id):
        return refusal
    try:
        result_row = load_task_result(result_root, task_id, strict=True) or {}
    except Exception:
        return {"ok": False, "error": "pause_record_unreadable", "action": "cancel_or_new_run"}
    row = saved_budget_pause(result_row)
    if task.get("_owner_hold") or result_row.get("_owner_hold"):
        return {"ok": False, "error": "owner_held"}
    if result_row.get("status") in _TRULY_TERMINAL_STATUSES:
        return {"ok": False, "error": "task_terminal"}
    if (not sleep_wake and row.get("reason") == "authority"
            and row.get("cause") in {"dispatch_fence_invalid", "dispatch_fence_unreadable", "start_custody_unreadable"}):
        from supervisor.sleep_wake import authority_readable
        if not authority_readable(task, row, result_root):
            return {"ok": False, "error": row.get("rail") or "control_authority_unreadable", "cause": row["cause"]}
    if sleep_wake and (row.get("state") != STATE_PAUSED or row.get("pause_id") != pause_id):
        return {"ok": False, "error": "sleep_wake_vetoed"}
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
    from ouroboros.owner_pause import fence_closed, recovered_start_over_latch

    owner_fence_id = ""
    root_task_id = str(task.get("root_task_id") or task_id)
    actor, refusal = _resume_actor(result_root, root_task_id, task_id)
    if refusal:
        return refusal
    observed_fence = ((external or {}).get("owner_pause_tree") or {}).get("fence") or {}
    tree_refusal = None
    if (fence_closed(observed_fence) or actor
            and (row.get("reason") == "owner" or fence_closed(result_row.get("owner_pause")))):
        owner_fence_id, tree_refusal = _owner_resume_fence(result_root, task_id, external,
                                                     actor=actor, root_task_id=root_task_id, selected_by=selected_by)
        if tree_refusal and tree_refusal["error"] != "owner_pause_effects_unsettled":
            return tree_refusal
    from ouroboros.delegate_resume import decisions_in_use, names_start
    if int(row.get("task_attempt") or 0) != int(task.get("_attempt") or 1):
        # Another attempt than the checkpoint's: the loop would refuse the grant on arrival.
        return {"ok": False, "error": "pause_attempt_mismatch", "action": "cancel_or_new_run",
                "row_attempt": int(row.get("task_attempt") or 0), "queue_attempt": int(task.get("_attempt") or 1)}
    hold = continuation_refusal(task)
    if sleep_wake and not _sleep_policy_allows(task, row, result_root):
        return {"ok": False, "error": "sleep_wake_vetoed"}
    row, refusal = _retire_orphaned_grant(q, task_id, result_root, row, hold, defer_revoke=names_start())
    if refusal:
        return refusal
    named_orphan = row.get("state") == STATE_RESUME_GRANTED and not (row.get("grant") or {}).get("revoked_at")
    deadline = parse_deadline_ts(task.get("deadline_at") or (task.get("task_contract") or {}).get("deadline_at"))
    now = time.time()
    started = float(row.get("started_at") or checkpoint.get("started_at") or 0.0)
    paused_at = float(row.get("paused_at") or checkpoint.get("paused_at") or now)
    prior_paused = float(row.get("paused_duration_sec") or 0.0)
    # ONE shared clock: wall time minus the quota union minus the paused carrier.
    executed_sec = execution_elapsed_seconds(
        {"task": task, "started_at": started, "budget_paused_sec": prior_paused,
         "model_wait_quota_clock": row.get("model_wait_quota_clock") or {}}, paused_at)
    if refusal := resume_clock_refusal(task, executed_sec, now):
        return refusal
    if refusal := _global_money_refusal(q, budget_remaining, None if str(row.get("reason") or "budget") == "budget" else [task]):
        return refusal
    # Dispatch revalidates the queue row's lineage, including for older pause records.
    try:
        root_state = _root_resume_state(q, root_task_id, result_root,
                                       actor_id=task_id if actor else None)
        root_grant = _root_resume_grants(root_state)[0] if not actor else {}
    except Exception:
        return {'ok': False, 'error': 'owner_pause_custody_unreadable'}
    if selected_by and not actor and not sleep_wake and not root_grant:
        return {"ok": False, "error": "root_resume_grant_missing", "root_task_id": root_task_id,
                "action": "resume_root_first"}
    priced = isinstance(external, dict) and "start_money_refusal" in external  # read with an ordinary start's custody
    if priced or str(pause.get("scope") or "") == "root" or row.get("reason") == "authority":
        # A technical pause on an unreadable billing authority rereads it strictly here too.
        if refusal := external["start_money_refusal"] if priced else _root_money_refusal(result_root, task, root_task_id):
            return refusal
    if not actor and any(str(item.get("id") or "") == root_task_id and isinstance(item.get("_budget_pause"), dict)
                         and item["_budget_pause"].get("reason") != "sleep" for item in q.PENDING):
        return {"ok": False, "error": "root_still_paused", "root_task_id": root_task_id,
                "action": "resume_root_first"}
    if refusal := _exact_custody_refusal(result_root, root_task_id, task_id, row, external, owner_fence_id, tree_refusal, actor):
        return refusal
    if named_orphan:
        # Check a named repeat against current custody before revoking: a new
        # start decision cannot be written in STATE_RESUME_GRANTED.
        row, refusal = _revoke_uncarried_grant(result_root, task_id, row, hold)
        if refusal:
            return refusal
    risks = decisions_in_use(task_id)
    external = {k: v for k, v in external.items() if k not in _OBSERVATION_ONLY}
    row = {**row, "external_runs": external,
           **({"settlement": "start_outcome_unknown" if risks else "settled"} if row.get("settlement") else {})}
    technical = sleep_wake and row.get("reason") == "authority"
    automatic = "authority_readable" if technical else "sleep_wake"
    author = {} if sleep_wake else resume_author()
    direct = row.get("reason") == "authority" and not sleep_wake and not selected_by
    generation = int(row.get("resume_generation") or 0) + 1
    pause_generation = int(row.get("pause_generation") or 0)
    from ouroboros.presence_carrier import lift_allowance, lifted_record, owner_allowance_lift
    lift = {} if selected_by or sleep_wake else owner_allowance_lift(task, result_root)  # decided once (#2566)
    grant = {
        "by": author, **({"selection_door": "direct"} if direct else {}),
        **({"accepted_start_risks": risks} if risks else {}),
        **({"owner_pause_fence_id": owner_fence_id} if owner_fence_id else {}),
        "grant_id": uuid.uuid4().hex, "granted_at": utc_now_iso(), "granted_at_ts": now,
        "single_use": True, "paused_duration_sec": prior_paused + max(0.0, now - paused_at),
        "executed_sec_before_pause": round(executed_sec, 3),
        "pause_id": pause_id, "pause_generation": pause_generation, "generation": generation,
        "selected_by": automatic if sleep_wake else str((author.get("actor") or "host") if direct else selected_by or "owner"),
        "authority": ("authority_readable" if technical else "sleep_readiness") if sleep_wake else "explicit_resume",
        "sleep_id": str((row.get("sleep") or {}).get("sleep_id") or "") if sleep_wake else "",
        'root_resume_basis': root_state['basis'], "root_grant_id": str(root_grant.get("grant_id") or ""),
        "root_resume_generation": int(root_grant.get("generation") or 0),
        "root_fence_id": (str((q.BUDGET_ROOT_FENCES.get(root_task_id) or {}).get("fence_id") or "")
                          if not actor else ""),
        "refresh_planning_threshold": str(row.get("rail") or "") in {
            "graceful_ceiling", "wrapup_last_fit", "soft_land"},
        **({"producer_allowance_lifted": lift} if lift else {}),
    }
    prior_pause = dict(pause)
    handoff = {
        **checkpoint, "grant_id": grant["grant_id"], "granted_at": grant["granted_at"],
        "grant_generation": generation, "pause_id": pause_id, "pause_generation": pause_generation,
        "paused_duration_sec": grant["paused_duration_sec"],
        "pause": {key: value for key, value in prior_pause.items() if key != "refusal"},
        "external_runs": external,
        # A sleep, and an explicitly bounded task's every exact Resume, stays excluded until
        # the worker consumes the grant: a capacity-delayed start is still that Pause (#939,
        # owner A). A task without an execution budget keeps its legacy clock.
        **({"sleep_exclusion_since": now}
           if row.get("reason") == "sleep" or explicit_execution_timeout_sec(task) is not None else {}),
        **{key: grant[key] for key in (
            'selected_by', 'by', 'authority', 'sleep_id', 'root_grant_id', 'selection_door',
            'root_resume_generation', 'root_fence_id', 'root_resume_basis') if key in grant},
    }

    def record_grant():
        if sleep_wake:
            if deadline is not None and deadline <= utc_now():
                return {"ok": False, "error": "deadline_passed"}
            if any(cancel_pending(result_root, tid, strict=True) for tid in {root_task_id, task_id}):
                return {"ok": False, "error": "cancel_intent_active"}
            if refusal := _global_money_refusal(q, budget_remaining, [task]):
                return refusal
            if technical and (refusal := _root_money_refusal(result_root, task, root_task_id)):
                return refusal
        set_budget_pause(result_root, task_id,
                         {**row, "state": STATE_RESUME_GRANTED, "grant": grant,
                          "resume_generation": generation},
                         expected_pause_id=pause_id, expected_state=str(row.get("state") or ""),
                         expected_grant_id=str((row.get("grant") or {}).get("grant_id") or ""),
                         row_update=(lambda current: lifted_record(current, lift)) if lift else None)

    held_siblings, rebound_holds, released_hold = [], {}, None
    def granted():
        try:
            from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

            write_task_result(
                result_root, task_id, STATUS_SCHEDULED, reason_code="",
                resource_limit={**handoff["pause"], "status": "resume_granted", "resumed_at": grant["granted_at"],
                                "grant_id": grant["grant_id"], "auto_resume": sleep_wake},
            )
        except Exception:
            log.debug("Failed to project exact budget resume for %s", task_id, exc_info=True)
        eligible = [str(r.get("id") or "") for r in q.PENDING
                    if isinstance(r.get("_budget_pause"), dict) and r["_budget_pause"].get("exact_continuation")
                    and str(r.get("root_task_id") or "") == root_task_id and str(r.get("id") or "") != task_id]
        q.append_jsonl(
            q.DRIVE_ROOT / "logs" / "events.jsonl",
            {"ts": utc_now_iso(), "type": "budget_task_woken" if sleep_wake else "budget_task_explicitly_resumed", "task_id": task_id,
             "root_task_id": root_task_id, "same_generation": True, "exact_continuation": True,
             "grant_id": grant["grant_id"], "grant_generation": generation,
             "selected_by": grant["selected_by"], "by": grant["by"],
             "paused_duration_sec": grant["paused_duration_sec"],
             "eligible_descendants": eligible if actor else [],
             "held_siblings": held_siblings, "rebound_held_siblings": sorted(rebound_holds),
             "released_hold": str((released_hold or {}).get("reason") or ""),
             "cleared_refusal": str(hold.get("reason") or "")},
        )
        return {"ok": True, "task_id": task_id, "root_task_id": root_task_id, "exact_continuation": True,
                "grant_id": grant["grant_id"], "grant_generation": generation,
                "paused_duration_sec": round(grant["paused_duration_sec"], 1),
                "eligible_descendants": eligible if actor else [],
                "held_siblings": held_siblings, "rebound_held_siblings": sorted(rebound_holds),
                **({"released_hold": str(released_hold.get("reason") or "")} if released_hold else {})}

    if sleep_wake:
        from supervisor.sleep_wake import publish_wake_grant
        return publish_wake_grant(q, task, pause, row, handoff, record_grant, granted, batch)
    try:
        record_grant()
    except Exception as exc:
        return {"ok": False, "error": "grant_not_recorded", "detail": str(exc)[:200]}
    task.pop("_budget_pause", None)
    task["_budget_pause_resume"] = handoff
    task["budget_resumed_at"] = grant["granted_at"]
    lifted_from = lift_allowance(task, lift)  # stands with the snapshot below, or is rolled back with the grant
    # The exact handoff owns selection until consumption; old stops survive only
    # in these local rollback copies, never in the new grant or selection.
    released_hold = task.pop(BUDGET_HOLD_KEY, None)
    prior_selection = task.pop(SELECTION_KEY, None)
    fence = q.BUDGET_ROOT_FENCES.get(root_task_id)
    fence_released = False
    released_markers: Dict[str, Dict[str, Any]] = {}
    if actor and isinstance(fence, dict) and (
            str(fence.get("fence_id") or "") == str(owner_fence_id or prior_pause.get("fence_id") or fence.get("fence_id"))
            or recovered_start_over_latch(observed_fence, fence)):
        # The root's own Resume lifts its admission latch: exact descendants keep
        # their OWN `_budget_pause` rows and are only ELIGIBLE; the model selects each (Q9).
        q.BUDGET_ROOT_FENCES.pop(root_task_id, None)
        fence_released = True
        held_siblings, released_markers, rebound_holds = hold_root_resume_descendants(
            q, root_task_id, fence, grant, actor_id=task_id)
    elif actor and owner_fence_id and fence is None:
        # The owner Pause's latch was never projected (``latch_pending``, or a park that could not
        # read its fence): this Resume's fresh read of the closed current fence stands in for it,
        # so its descendants wait for explicit selection exactly as when a latch lifts (Q9).
        held_siblings, released_markers, rebound_holds = hold_root_resume_descendants(
            q, root_task_id, {"fence_id": owner_fence_id, "cause": "owner_pause"}, grant, actor_id=task_id)
    if not q.publish_queue_snapshot(reason="budget_exact_resume_granted"):  # durable before the grant stands
        task.pop("_budget_pause_resume", None)
        task.update(lifted_from, _budget_pause=prior_pause)  # the allowance a lift moved returns with the pause
        if released_hold is not None:
            task[BUDGET_HOLD_KEY] = released_hold
        if prior_selection is not None:
            task[SELECTION_KEY] = prior_selection
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
                             expected_grant_id=str(grant["grant_id"]),
                             row_update=(lambda current: _unlifted_record(current, lift)) if lift else None)
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
                   "root_task_id": root_task_id},
            result_root=result_root)
        return {"ok": False, "error": "snapshot_not_persisted",
                "held": HOLD_REVOCATION_UNWRITTEN, "grant_id": str(grant["grant_id"])}
    return granted()


def newer_continuation_basis(task: Dict[str, Any], drive_root: Any) -> bool:
    """Custody after a spent Resume, never permission to dispatch that custody.

    Each carrier's existing admission validates its source/transaction. An
    unreadable authority raises: neither a continuation nor absence is proven.
    """
    from ouroboros.budget_pause import LIVE_PAUSE_STATES, budget_pause_row
    from ouroboros.delegate_recovery import _read, planned_handoff_matches

    if any(isinstance(task.get(key), dict) for key in ("_owner_wait_resume", "_working_recovery")):
        return True
    root = pathlib.Path(task.get("budget_drive_root") or drive_root)
    pause = task.get("_budget_pause") or {}
    if pause.get("exact_continuation"):
        row = budget_pause_row(root, str(task.get("id") or ""))
        if (row.get("state") in LIVE_PAUSE_STATES and row.get("source_ref")
                and row.get("pause_id") == (pause.get("checkpoint") or {}).get("pause_id")):
            return True
    return int(task.get("_attempt") or 1) > 1 and planned_handoff_matches(
        _read(root, str(task.get("id") or ""), strict=True), task)


def resume_hold_requires_fence(task: Dict[str, Any], drive_root: Any) -> bool:
    """Re-decide spent, marker-less custody using its durable hold after restore."""
    task.pop("_budget_pause_consumed", None)  # obsolete local copy, never a grant
    refusal = continuation_refusal(task)
    if refusal.get("grant_state") not in {"consumed", "superseded"}:
        return False
    pause = task.get("_budget_pause") or {}
    if refusal.get("grant_state") == "superseded" and pause.get("exact_continuation"):
        return False  # the live exact locator owns its revalidation
    if not newer_continuation_basis(task, drive_root):
        return True
    clear_refusal(task)
    return False


def revalidate_consumed_resume_holds() -> None:
    """Call first in ``supervisor.task_admission.revalidate_project_holds()`` under Q,
    before its empty-project-holds early return and before normal admission.

    Safe every tick: without consumed or marker-less superseded holds, the queue
    scan is in-memory only, with no store reads/writes or snapshot publication;
    with them its store reads are zero-wait, contention retrying next tick.
    """
    from supervisor import queue as q
    from supervisor.queue_snapshot import _fence_snapshot_running_rows

    from ouroboros.store import StoreError, admission_acquisition

    changed = False
    for task in q.PENDING:
        before = continuation_refusal(task)
        try:
            with admission_acquisition():  # under Q nothing waits: a busy read keeps the hold for the next tick
                # A live attempt of the same id owns the task: its stale row waits, never fences it.
                if resume_hold_requires_fence(task, q.DRIVE_ROOT) and task.get("id") not in q.RUNNING:
                    _fence_snapshot_running_rows([{"id": task["id"]}], restored_ids=set())
            changed |= before != continuation_refusal(task)
        except (StoreError, OSError, ValueError, TypeError, RuntimeError):
            log.warning("Continuation basis unreadable for %s; hold retained", task.get("id"), exc_info=True)
    if changed:
        q.persist_queue_snapshot(reason="consumed_resume_hold_revalidated")


def repark_after_active_start(q, task):
    """An unconsumed, queued grant predating a recovered writer returns to its exact pause. Q held."""
    from ouroboros.budget_pause import budget_pause_row
    from ouroboros.owner_pause import read_fence, fence_closed
    from ouroboros.store import acquisition_budget

    root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    task_id = task["id"]
    try:
        with acquisition_budget(0):
            fence = read_fence(root, task.get("root_task_id") or task_id)
            if not fence_closed(fence) or not fence.get("active_delegated_start"):
                return None
            grant = (budget_pause_row(root, task_id) or {}).get("grant") or {}
            if grant.get("owner_pause_fence_id") == fence["fence_id"]:
                return None
            if not revoke_exact_budget_resume(task, "active_delegated_start"):
                return {"ok": False, "error": "selection_authority_changed"}
    except Exception:
        return {"ok": False, "error": "owner_pause_custody_unreadable"}
    if not q.publish_queue_snapshot(reason="active_delegated_start_reparked"):
        return {"ok": False, "error": "snapshot_not_persisted"}
    return None


def revoke_exact_budget_resume(task: Dict[str, Any], reason: str) -> bool:
    """Return a granted-but-undispatched task to its exact pause.

    Money can vanish between the grant and the dispatch (a sibling spent it) and a restart may intervene;
    the grant is single-use and must not be dispatched into a refused send. A revocation is recorded ONLY
    against the pause, state and grant this handoff names (compare-and-set on all three). A grant the
    durable row says was consumed is never re-armed: the task ran on, so its spent carrier leaves (simply
    retired when a newer continuation basis carries the task, otherwise the stale row takes a typed hold),
    and no ``_budget_pause`` marker is minted over a task that is not paused. A newer pause or a different
    grant on the durable row is not written over: the handoff leaves and the row re-reads the current
    pause, except that at a restart (``restart_before_dispatch``) a newer grant no worker received is
    revoked under its own identity too, so the next Resume finds a pause. A failed write leaves the row
    undispatchable under a hold rather than carrying a stale grant.
    A caller may run it on a copy outside Q and carry the result back under Q (``_parking_lock``).
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
    if (row_state == STATE_PAUSED and current_pause_id == expected_pause_id and task.get("_budget_pause")
            and handoff.get("authority") in {"sleep_readiness", "authority_readable"}):
        task.pop("_budget_pause_resume", None)  # saved proposal, never confirmed
        return True
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
        try:
            continuing = newer_continuation_basis(task, result_root)
        except (OSError, ValueError, TypeError, RuntimeError):
            continuing = False  # retained hold; restore/tick re-reads before deciding absence
        if continuing:
            retain_consumed_selection(task, row)
            task.pop("_budget_pause_resume", None)
            q.append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                           {"ts": utc_now_iso(), "type": "budget_resume_carrier_retired",
                            "task_id": task_id, "reason": str(reason or ""),
                            "grant_id": str(grant.get("grant_id") or ""),
                            "pause_id": current_pause_id,
                            "consumed_at": grant.get("consumed_at"),
                            "retained": "newer_continuation_basis"})
            return False
        task.pop("_budget_pause_resume", None)
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


def _late_phase_refusal(q: Any, root: pathlib.Path, task_id: str, row: Dict[str, Any], fence: Dict[str, Any],
                        *, gaps: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
    """The exact Resume's own refusals for an answered root's late work; nothing is consumed."""
    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now
    from supervisor.continuation_admission import action_writers
    from supervisor.events_budget import budget_hold_fact
    from supervisor.state import budget_remaining

    if refusal := _restart_or_update_refusal(q, row, root, task_id):
        return refusal
    try:
        if has_active_intent(root, task_id, strict=True):
            return {"ok": False, "error": "cancel_intent_active"}
    except Exception:
        return {"ok": False, "error": "cancellation_authority_unavailable"}
    with q._queue_lock:
        rows = [t for t in [*q.PENDING, *(m.get("task") or {} for m in q.RUNNING.values())] if isinstance(t, dict)]
        # A Continue of this answered root that may already write would overlap the late work.
        successors = [str(t.get("id") or "") for t in rows
                      if ((t.get("metadata") or {}).get("continuation") or {}).get(
                          "predecessor_task_id") == task_id and (t.get("id") in q.RUNNING or not budget_hold_fact(t))]
        # All this Resume can start: the tree's queued and warm members. The answered root runs nothing again and
        # its unsent late review is never sent again (``review_pause``), so review-only work starts no actor.
        members = [dict(t) for t in rows if str(t.get("root_task_id") or "") == task_id and t.get("id") != task_id]
    # The whole tree's sent work must have settled, exactly as for a paused loop's Resume.
    # Reviewers already launched finish separately; Resume does not wait for them.
    blockers = [{"kind": "continuation_successor", "task_id": tid} for tid in successors] + _owner_pause_blockers(
        action_writers(q, task_id, drive_root=root, owner_pause_fence_id=str(fence.get("fence_id") or ""), gaps=gaps))
    deadline = parse_deadline_ts(row.get("deadline_at") or (row.get("task_contract") or {}).get("deadline_at")
                                 or (row.get("metadata") or {}).get("deadline_at"))
    if deadline is not None and deadline <= utc_now():
        return {"ok": False, "error": "deadline_passed"}
    refusal = _global_money_refusal(q, budget_remaining, members) or _root_money_refusal(
        root, {**row, "id": task_id}, str(row.get("root_task_id") or task_id))
    if refusal:
        return refusal
    from ouroboros.delegate_resume import decide, names_start
    if not members and names_start():
        return {"ok": False, "error": "selection_authority_changed"}
    return decide(root, task_id, task_id, blockers, fence_id=str(fence["fence_id"]),
                  tree_actor=True, allow_starts=bool(members))


def resume_late_phase(task_id: str, *, gaps: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
    """The owner's Resume of an answered root's Pause; None when not one.

    The Pause already ended that root's post-answer upkeep, so nothing of it is
    resumed. Nothing left at all: the tree census already releases the fence.
    Only deferred, unsent late work left (a review operation not yet sent): this
    explicit Resume reopens the fence once nothing sent remains in flight. The
    exact Resume's refusals apply unchanged (Restart/Panic hold, a live Stop,
    unsettled tree custody, the calendar deadline, global money and the root
    tree's own cap): a refusal leaves the fence as it was. ``gaps`` receives
    the damaged rows its own censuses tie to no tree.
    """
    from ouroboros.owner_pause import FENCE_RELEASED, fence_closed, launch_lock, read_fence, set_fence_state
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight
    from ouroboros.review_operation import task_has_live_review_operation
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result
    from supervisor.continuation_admission import conflicting_writers, release_settled_continuations
    from supervisor.owner_pause_control import refresh_owner_pause_tree
    from supervisor import queue as q

    root = pathlib.Path(q.DRIVE_ROOT)
    try:
        row = load_task_result(root, task_id, strict=True) or {}
        fence = read_fence(root, task_id) if row else {}
    except Exception:
        return {"ok": False, "error": "pause_record_unreadable"}
    if row.get("status") not in _TRULY_TERMINAL_STATUSES or not fence_closed(fence):
        return None
    if refusal := release_refusal(observed_control_stop(
            root, row, budget=q.BUDGET_ROOT_FENCES, owner=fence), control_actor()):
        return refusal
    remainder = refresh_owner_pause_tree(task_id, gaps=gaps) != FENCE_RELEASED
    if remainder:
        if refusal := _late_phase_refusal(q, root, task_id, row, fence, gaps=gaps):
            return refusal
        try:
            blocked = (post_task_synthesis_in_flight(root, task_id)
                       or task_has_live_review_operation(root, task_id, sent_only=True)
                       or conflicting_writers(q, task_id, drive_root=root,
                                              owner_pause_fence_id=str(fence.get("fence_id") or ""), gaps=gaps))
        except Exception:
            blocked = True
        if blocked:
            return {"ok": False, "error": "owner_pause_effects_unsettled", "action": "wait_for_effect_settlement"}
        try:
            with launch_lock(root, task_id):
                if read_fence(root, task_id) != fence:
                    return {"ok": False, "error": "selection_authority_changed"}
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

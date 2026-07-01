"""Evolution campaign state and transaction lifecycle.

Owns the campaign file (``state/evolution_campaign.json``), the campaign/
transaction lifecycle transitions, and the owner-facing cycle reporting.
``supervisor.queue`` imports this module top-level; anything here that needs
queue state (``DRIVE_ROOT``, locks) must import the queue lazily at call time
to avoid a module-load cycle.
"""

from __future__ import annotations

import logging
import os
import pathlib
import uuid
from typing import Any, Dict, Optional

from ouroboros.evolution_fingerprint import canonical_objective_fingerprint
from ouroboros.outcomes import normalize_outcome_axes
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

EVOLUTION_CAMPAIGN_FILE = pathlib.Path("state") / "evolution_campaign.json"


def _evolution_campaign_path() -> pathlib.Path:
    from supervisor import queue

    return pathlib.Path(queue.DRIVE_ROOT) / EVOLUTION_CAMPAIGN_FILE


def _read_evolution_campaign() -> Dict[str, Any]:
    data = read_json_dict(_evolution_campaign_path()) or {}
    return data if isinstance(data, dict) else {}


def _write_evolution_campaign(data: Dict[str, Any]) -> None:
    path = _evolution_campaign_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data, trailing_newline=True)


def evolution_block_reason() -> str:
    """Refusal message when evolution may not run in the current runtime mode.

    Evolution campaigns are self-modification work, so they require runtime
    mode ``advanced`` or ``pro``. In ``light`` (conversation-only) mode they are
    hard-blocked before any campaign state, queue entry, or expensive round.
    Returns ``""`` when evolution is allowed.
    """
    from ouroboros.config import get_runtime_mode

    if get_runtime_mode() == "light":
        return (
            "🧬 Evolution campaigns are self-modification work and require runtime "
            "mode 'advanced' or 'pro'. The runtime is in 'light' mode "
            "(self-modification is disabled), so no campaign was started. Switch "
            "the runtime mode in Settings to evolve."
        )
    return ""


def start_evolution_campaign(objective: str = "", *, source: str = "owner") -> Dict[str, Any]:
    """Start or resume the active evolution campaign."""
    campaign = _read_evolution_campaign()
    now = utc_now_iso()
    objective = str(objective or "").strip()
    if campaign.get("status") not in {"active", "paused"}:
        campaign = {
            "schema_version": 1,
            "id": uuid.uuid4().hex[:8],
            "status": "active",
            "objective": objective or "Autonomously improve Ouroboros by acting on the highest-value backlog or process-memory signal.",
            "source": source,
            "started_at": now,
            "updated_at": now,
            "cycles_done": 0,
            "absorbed_cycles_done": 0,
            "objective_repeat_counts": {},  # BUG3: fp -> non-absorbing-cycle count
            "dropped_objective_fps": [],  # BUG3 Layer B: attempted-and-dropped objective fps
            "budget_spent_usd": 0.0,
            "last_task_id": "",
            "progress_notes": "",
            "completed_at": "",
            "completion_reason": "",
        }
    else:
        if objective:
            campaign["objective"] = objective
        campaign["status"] = "active"
        campaign["updated_at"] = now
    _write_evolution_campaign(campaign)
    return campaign


def pause_evolution_campaign(reason: str = "") -> Dict[str, Any]:
    """Pause the active evolution campaign without deleting its state.

    RESUMABLE: a later ``/evolve start`` resumes the SAME campaign in place
    (start_evolution_campaign treats ``paused`` as resumable). Used by the system
    breakers (light mode, consecutive failures, objective-repeat cap, budget reserve,
    restart-blocked) — NOT by an owner stop. For an owner stop use
    ``complete_evolution_campaign`` (terminal).
    """
    campaign = _read_evolution_campaign()
    if campaign:
        campaign["status"] = "paused"
        campaign["updated_at"] = utc_now_iso()
        campaign["pause_reason"] = str(reason or "")
        _write_evolution_campaign(campaign)
    return campaign


def complete_evolution_campaign(
    reason: str = "", *, status: str = "stopped", cleanup_worktree: bool = True
) -> Dict[str, Any]:
    """Terminally CLOSE the active campaign — the OWNER-stop counterpart of the
    resumable pause. ``status`` is non-{active,paused}, so a later ``/evolve start``
    mints a FRESH campaign instead of resurrecting this one. Archives + pops any
    in-flight ``active_transaction`` (and ``post_task_backlog_id``) so a terminally
    stopped campaign carries no dangling commit for a boot reconcile to absorb. The
    durable gate against autonomous re-arm is the ``evolution_owner_stopped`` state
    flag set at the owner-stop sites (read by ``apply_pending_request``); this terminal
    status is the observability/audit marker plus a clean campaign. Never raises.

    ``cleanup_worktree`` (default True) runs the deterministic per-cycle worktree reset
    for an in-flight transaction. PANIC passes ``False``: the Emergency Stop Invariant
    (BIBLE) forbids delaying panic, so panic must NOT run git stash/reset work before its
    hard exit — the panic flag + boot reconcile own that recovery instead."""
    try:
        campaign = _read_evolution_campaign()
        if not campaign:
            return campaign or {}
        now = utc_now_iso()
        tx = campaign.get("active_transaction")
        if isinstance(tx, dict):
            tx = {**tx, "cycle_outcome": tx.get("cycle_outcome") or "owner_stopped"}
            # Owner stop mid-cycle: this terminal 'stopped' status makes
            # update_evolution_campaign_after_task early-return when the cancelled task's
            # (async) task_done later fires, so the normal per-cycle worktree cleanup would
            # be SKIPPED — leaking the abandoned, unreviewed evolution edits into the live
            # repo. Run that same deterministic cleanup here before popping the tx. It is
            # self-guarded (skips with a recorded reason while a task still holds the shared
            # worktree — hence owner-stop sites cancel the running cycle BEFORE this close —
            # kill-switch-able, never raises). SKIPPED under panic (cleanup_worktree=False):
            # the Emergency Stop Invariant forbids any git work before the panic hard-exit.
            if cleanup_worktree:
                try:
                    _cleanup_worktree_after_cycle(tx, str(tx.get("task_id") or ""))
                except Exception:
                    pass
            try:
                append_unique_transaction(campaign, tx)
            except Exception:
                pass
            campaign.pop("active_transaction", None)
        campaign.pop("post_task_backlog_id", None)
        campaign.pop("pause_reason", None)
        campaign["status"] = str(status or "stopped")
        campaign["updated_at"] = now
        campaign["completed_at"] = now
        campaign["completion_reason"] = str(reason or "")
        _write_evolution_campaign(campaign)
        return campaign
    except Exception:
        log.debug("complete_evolution_campaign failed", exc_info=True)
        return {}


def begin_evolution_transaction(task_id: str, *, cycle: int, campaign: Dict[str, Any]) -> Dict[str, Any]:
    """Attach a compact self-modification transaction to the active campaign."""
    try:
        from supervisor import git_ops

        rc_head, head, _ = git_ops.git_capture(["git", "rev-parse", "HEAD"])
        rc_branch, branch, _ = git_ops.git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        base_head = head.strip() if rc_head == 0 else ""
        base_branch = branch.strip() if rc_branch == 0 else ""
    except Exception:
        base_head = ""
        base_branch = ""
    transaction = {
        "schema_version": 1,
        "transaction_id": uuid.uuid4().hex[:12],
        "campaign_id": str((campaign or {}).get("id") or ""),
        "task_id": str(task_id or ""),
        "cycle": int(cycle or 0),
        # BUG3: capture the objective this cycle will run, at cycle START, as the SSOT
        # per-cycle fingerprint. Read here (not at outcome time) because campaign["objective"]
        # can be overwritten by a later promotion before the outcome is recorded.
        "objective_fp": canonical_objective_fingerprint(str((campaign or {}).get("objective") or "")),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "base_head": base_head,
        "base_branch": base_branch,
        "preflight_status": "pending",
        "advisory_status": "pending",
        "triad_scope_status": "pending",
        "commit_sha": "",
        "push_status": "pending",
        "restart_decision": "",
        "restart_required": False,
        "restart_verified": False,
        "restart_verified_at": "",
        "rescue_ref": "",
        "rescue_path": "",
        "recovery_hint": "",
    }
    current = _read_evolution_campaign()
    if current.get("id") == campaign.get("id"):
        current["active_transaction"] = transaction
        current["updated_at"] = utc_now_iso()
        _write_evolution_campaign(current)
    return transaction


def _bump_objective_repeat_count(campaign: Dict[str, Any], tx: Dict[str, Any]) -> None:
    """BUG3: count one non-absorbing cycle against its objective fingerprint.

    Cumulative PER-FINGERPRINT (not a consecutive streak), so a blocked objective that is
    re-proposed NON-consecutively (interleaved with other no_op work) still accumulates toward
    the pause gate. ``setdefault`` tolerates campaigns persisted before this field existed; a
    transaction without an ``objective_fp`` (e.g. a tx-less idle cycle) is skipped, never
    bucketed under the empty key.
    """
    fp = str((tx or {}).get("objective_fp") or "")
    if not fp:
        return
    counts = campaign.setdefault("objective_repeat_counts", {})
    counts[fp] = int(counts.get(fp, 0) or 0) + 1
    # Layer B: also mark this objective attempted-and-dropped so the chooser (Layer A) can be
    # told not to re-propose it. This is a campaign-local signal, NOT a backlog status flip:
    # the backlog item stays "open" (the work is genuinely unsolved), we only stop FEEDING it
    # back to the evolution objective chooser.
    dropped = campaign.setdefault("dropped_objective_fps", [])
    if fp not in dropped:
        dropped.append(fp)


def _clear_objective_repeat_count(campaign: Dict[str, Any], tx: Dict[str, Any]) -> None:
    """BUG3: a genuine absorb clears ONLY this objective's repeat tally and dropped flag.

    Called at every site that sets ``cycle_outcome == "absorbed"`` (task-done here, plus the
    two durable boot/restart-verify absorb sites in agent_startup_checks). Keyed on the same
    SSOT fingerprint as the bump so success on the looping objective resets exactly its bucket
    and un-drops it (it landed, so it is no longer do-not-re-propose).
    """
    fp = str((tx or {}).get("objective_fp") or "")
    if not fp:
        return
    counts = campaign.setdefault("objective_repeat_counts", {})
    counts.pop(fp, None)
    dropped = campaign.setdefault("dropped_objective_fps", [])
    if fp in dropped:
        dropped.remove(fp)


def update_evolution_transaction(task_id: str, **updates: Any) -> None:
    """Best-effort update of the active/lightweight evolution transaction."""
    campaign = _read_evolution_campaign()
    tx = campaign.get("active_transaction")
    if not isinstance(tx, dict) or str(tx.get("task_id") or "") != str(task_id or ""):
        return
    for key, value in updates.items():
        if value is not None:
            tx[key] = value
    tx["updated_at"] = utc_now_iso()
    campaign["active_transaction"] = tx
    campaign["updated_at"] = utc_now_iso()
    _write_evolution_campaign(campaign)


def _cleanup_worktree_after_cycle(tx: Dict[str, Any], task_id: str) -> None:
    """Deterministic worktree cleanup when a cycle closes WITHOUT absorption.

    A no_op/abandoned evolution cycle must leave the repo at its recorded
    ``base_head``: abandoned edits or unreviewed local commits otherwise leak
    into the next cycle (and into unrelated tasks) as mystery state. Recovery
    is never silent — dirty files go into a git stash and an ahead HEAD is
    preserved as a local branch before the hard reset; both refs are recorded
    on the transaction. Skipped (with a recorded reason) when other tasks are
    running in the shared worktree or the base is unknown. Never raises.
    Kill-switch: OUROBOROS_EVOLUTION_CYCLE_CLEANUP=false.
    """
    if str(os.environ.get("OUROBOROS_EVOLUTION_CYCLE_CLEANUP", "true") or "true").lower() in {"0", "false", "no", "off"}:
        tx["cleanup_status"] = "disabled"
        return
    base_head = str(tx.get("base_head") or "").strip()
    if not base_head:
        tx["cleanup_status"] = "skipped_no_base"
        return
    try:
        from supervisor import git_ops, queue

        # Same protection class as git_ops._guard_live_repo_destructive_git, but
        # covering the stash too: a unit test that never re-pointed
        # git_ops.REPO_DIR must not stash/reset the LIVE repo's working tree.
        if os.environ.get("OUROBOROS_ALLOW_LIVE_REPO_TESTS") != "1":
            import sys as _sys
            try:
                live_repo = git_ops.REPO_DIR.resolve(strict=False) == (
                    pathlib.Path.home() / "Ouroboros" / "repo"
                ).resolve(strict=False)
            except OSError:
                live_repo = False
            if live_repo and ("PYTEST_CURRENT_TEST" in os.environ or "pytest" in _sys.modules):
                tx["cleanup_status"] = "skipped_live_repo_test_guard"
                return

        # Lock-free RUNNING snapshot is safe because this runs on the SAME
        # single supervisor thread that assigns tasks (dispatch_event ->
        # assign_tasks are sequential); only cancel paths mutate RUNNING from
        # HTTP threads, which can only shrink the set.
        running_others = [tid for tid in list(queue.RUNNING.keys()) if str(tid) != str(task_id)]
        if running_others:
            # The live worktree is shared: a reset would destroy concurrent
            # tasks' work. Leave state for the boot reconcile / next cycle.
            tx["cleanup_status"] = "skipped_other_tasks_running"
            return

        rc_status, status_out, _ = git_ops.git_capture(["git", "status", "--porcelain"])
        rc_head, head_out, _ = git_ops.git_capture(["git", "rev-parse", "HEAD"])
        if rc_status != 0 or rc_head != 0:
            tx["cleanup_status"] = "skipped_git_unavailable"
            return
        dirty = bool(status_out.strip())
        head = head_out.strip()
        if not dirty and head == base_head:
            tx["cleanup_status"] = "already_clean"
            return

        if dirty:
            stash_label = f"evolution-cycle-cleanup-{tx.get('transaction_id') or task_id}"
            rc_stash, _, stash_err = git_ops.git_capture(
                ["git", "stash", "push", "--include-untracked", "-m", stash_label]
            )
            if rc_stash != 0:
                from ouroboros.utils import truncate_review_artifact

                # Refuse to reset over unsaved changes (P1: no silent loss).
                tx["cleanup_status"] = "skipped_stash_failed"
                tx["recovery_hint"] = (
                    "worktree dirty and stash failed: "
                    + truncate_review_artifact(str(stash_err or "").strip(), 400)
                )
                return
            tx["cleanup_stash"] = stash_label

        if head != base_head:
            preserved, ref_name = git_ops.preserve_local_ref_branch("HEAD", prefix="evolution-leftover")
            if not preserved:
                tx["cleanup_status"] = "skipped_preserve_failed"
                tx["recovery_hint"] = (
                    "HEAD ahead of base_head and preserve-branch failed; left as-is"
                    + (f" (dirty files already saved in stash {tx['cleanup_stash']})." if tx.get("cleanup_stash") else ".")
                )
                return
            tx["cleanup_preserved_ref"] = ref_name
            rc_reset, _, reset_err = git_ops.git_capture(["git", "reset", "--hard", base_head])
            if rc_reset != 0:
                from ouroboros.utils import truncate_review_artifact

                tx["cleanup_status"] = "reset_failed"
                tx["recovery_hint"] = (
                    f"git reset --hard {base_head[:12]} failed: "
                    + truncate_review_artifact(str(reset_err or "").strip(), 400)
                )
                return
            tx["cleanup_status"] = "reset_to_base"
        else:
            tx["cleanup_status"] = "stashed_dirty"  # HEAD already at base; only the stash happened
        log.info(
            "Evolution cycle %s cleanup: worktree restored to base %s (stash=%s, preserved=%s)",
            tx.get("transaction_id") or task_id, base_head[:12],
            tx.get("cleanup_stash") or "-", tx.get("cleanup_preserved_ref") or "-",
        )
    except Exception:
        tx["cleanup_status"] = "error"
        log.debug("Evolution cycle worktree cleanup failed", exc_info=True)


def update_evolution_campaign_after_task(
    task_id: str,
    *,
    cost_usd: float,
    outcome_axes: Dict[str, Any],
    rounds: int,
    transaction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record an evolution cycle outcome in the active campaign file."""
    from supervisor import queue

    campaign = _read_evolution_campaign()
    if campaign.get("status") not in {"active", "paused"}:
        return {}
    metadata_tx = transaction if isinstance(transaction, dict) else {}
    active_tx = campaign.get("active_transaction") if isinstance(campaign.get("active_transaction"), dict) else {}
    if str(active_tx.get("task_id") or "") == str(task_id or ""):
        tx = {**metadata_tx, **active_tx}
    elif str(metadata_tx.get("task_id") or "") == str(task_id or ""):
        tx = dict(metadata_tx)
    else:
        tx = {}
    axes = normalize_outcome_axes({"outcome_axes": outcome_axes or {}})
    tx_id = str(tx.get("transaction_id") or "") if tx else ""
    for existing in list(campaign.get("history") or []):
        if not isinstance(existing, dict) or str(existing.get("task_id") or "") != str(task_id or ""):
            continue
        existing_tx = existing.get("transaction") if isinstance(existing.get("transaction"), dict) else {}
        if not tx_id or str(existing_tx.get("transaction_id") or "") == tx_id:
            return {**dict(campaign.get("active_transaction") or existing_tx or tx), "_replay": True}
    if tx:
        tx["outcome_axes"] = axes
        tx["updated_at"] = utc_now_iso()
    history = list(campaign.get("history") or [])
    row = {
        "task_id": str(task_id or ""),
        "ts": utc_now_iso(),
        "cost_usd": float(cost_usd or 0.0),
        "outcome_axes": axes,
        "rounds": int(rounds or 0),
    }
    if tx:
        row["transaction"] = tx
    history.append(row)
    campaign["history"] = history[-50:]
    if tx:
        has_commit = bool(str(tx.get("commit_sha") or "").strip())
        restart_verified = bool(tx.get("restart_verified"))
        has_rescue = bool(str(tx.get("rescue_ref") or "").strip())
        if str((campaign.get("active_transaction") or {}).get("task_id") or "") == str(task_id or ""):
            if has_commit and restart_verified:
                tx["cycle_outcome"] = "absorbed"
                campaign["absorbed_cycles_done"] = int(campaign.get("absorbed_cycles_done") or 0) + 1
                append_unique_transaction(campaign, tx)
                campaign.pop("active_transaction", None)
                _clear_objective_repeat_count(campaign, tx)  # BUG3: genuine progress clears this fp
            elif has_rescue:
                tx["cycle_outcome"] = "abandoned"
                tx["abandoned_reason"] = "rescue_ref_present"
                _cleanup_worktree_after_cycle(tx, str(task_id or ""))
                append_unique_transaction(campaign, tx)
                campaign.pop("active_transaction", None)
                campaign.pop("post_task_backlog_id", None)  # BUG3 Layer B: detach (was missing on abandoned)
                _bump_objective_repeat_count(campaign, tx)  # BUG3: non-absorbing cycle counts
            elif not has_commit:
                tx["cycle_outcome"] = "no_op"
                tx["restart_required"] = False
                tx["recovery_hint"] = ""
                _cleanup_worktree_after_cycle(tx, str(task_id or ""))
                append_unique_transaction(campaign, tx)
                campaign.pop("active_transaction", None)
                campaign.pop("post_task_backlog_id", None)
                _bump_objective_repeat_count(campaign, tx)  # BUG3: non-absorbing cycle counts
            else:
                tx["cycle_outcome"] = "waiting_for_restart"
                tx["recovery_hint"] = tx.get("recovery_hint") or (
                    "Task ended without a reviewed commit plus restart verification; active "
                    "transaction retained until restart verifies, repo state is recovered, or it is superseded."
                )
                tx["restart_required"] = True
                if not tx.get("restart_decision"):
                    tx["restart_decision"] = "supervisor_auto_requested"
                campaign["active_transaction"] = tx
                request_evolution_restart(queue.DRIVE_ROOT, tx, log=log)
        # WS-13.5 (e5=ux_absorb_report): tell the owner in chat what a completed
        # self-evolution cycle did. Absorbed -> short what/why; abandoned ->
        # honest warning; no-op / waiting -> quiet (event only). No web edits.
        try:
            notify_owner_cycle_outcome(campaign, tx)
        except Exception:
            log.debug("Failed to send evolution cycle owner report", exc_info=True)
    campaign["last_task_id"] = str(task_id or "")
    campaign["cycles_done"] = int(campaign.get("cycles_done") or 0) + 1
    execution_status = str((axes.get("execution") or {}).get("status") or "unknown")
    objective_status = str((axes.get("objective") or {}).get("status") or "not_evaluated")
    campaign["progress_notes"] = (
        f"Last cycle {task_id}: execution={execution_status}, objective={objective_status}, "
        f"rounds={int(rounds or 0)}, cost=${float(cost_usd or 0.0):.4f}."
    )
    campaign["budget_spent_usd"] = round(
        float(campaign.get("budget_spent_usd") or 0.0) + float(cost_usd or 0.0),
        6,
    )
    campaign["updated_at"] = utc_now_iso()
    _write_evolution_campaign(campaign)
    return tx


def build_evolution_task_text(cycle: int) -> str:
    """Build the next evolution-campaign task prompt."""
    from ouroboros.config import get_evolution_persistent_objective

    campaign = _read_evolution_campaign()
    if campaign.get("status") != "active":
        return f"EVOLUTION #{cycle}"
    parts = [
        f"EVOLUTION CAMPAIGN {campaign.get('id') or 'active'} — CYCLE #{cycle}",
        "",
        "## Objective",
        str(campaign.get("objective") or "Autonomously improve Ouroboros."),
    ]
    steer = get_evolution_persistent_objective()
    if steer:
        parts.extend([
            "",
            "## Owner Standing Steer (optional bias — does NOT override the Objective above)",
            steer,
        ])
    progress = str(campaign.get("progress_notes") or "").strip()
    if progress:
        parts.extend(["", "## Progress So Far", progress])
    history = list(campaign.get("history") or [])[-3:]
    if history:
        parts.extend(["", "## Recent Campaign Cycles"])
        for row in history:
            axes = normalize_outcome_axes(row)
            execution_status = str((axes.get("execution") or {}).get("status") or "unknown")
            objective_status = str((axes.get("objective") or {}).get("status") or "not_evaluated")
            parts.append(
                f"- {row.get('task_id')}: execution={execution_status}, objective={objective_status}; "
                f"rounds={row.get('rounds', 0)}; cost=${float(row.get('cost_usd') or 0):.4f}"
            )
    # Fix B (C10.2): surface the durable improvement backlog and recent solve-capability
    # as optional CONTEXT, never a directive. Ouroboros decides what (if anything) to act
    # on — an evolution cycle is NOT obligated to draw from the backlog or repeat past
    # patterns. Injecting them is LLM-first steering, not a hardcoded work order.
    try:
        from ouroboros.evolution_checkpoints import build_solve_capability_digest
        from ouroboros.improvement_backlog import format_backlog_digest
        from ouroboros.utils import truncate_review_artifact
        from supervisor import queue

        _digest_root = pathlib.Path(queue.DRIVE_ROOT)
        _backlog_digest = format_backlog_digest(_digest_root, limit=8, max_chars=3000)
        if _backlog_digest:
            parts.extend([
                "",
                "## Improvement Backlog (context only — NOT a work order)",
                "Standing nominations from past cycles. Weigh them if useful, but you are "
                "free to pursue the Objective however you judge best; you need not pick "
                "from this list.",
                "",
                _backlog_digest,
            ])
        _capability_digest = truncate_review_artifact(build_solve_capability_digest(_digest_root), 2000)
        if _capability_digest:
            parts.extend([
                "",
                "## Recent Solve-Capability (context only)",
                _capability_digest,
            ])
    except Exception:
        log.debug("evolution task digest injection failed", exc_info=True)
    parts.extend([
        "",
        "## Execution Contract",
        "- Work as a normal Ouroboros self-improvement task.",
        "- Use standard tests and the normal advisory + triad + scope review flow before committing code.",
        "- Land at most ONE reviewed self-modification commit in this cycle. Fold reviewer fixes into that commit before committing; do not churn follow-up commits.",
        "- After a reviewed commit lands, call request_restart once and stop. Restart verification is the absorption boundary for the cycle.",
        "- An honest no-op is a legitimate outcome when the objective is unsafe, already solved, too broad, or needs owner input; do not commit just to make a cycle non-empty.",
        "- If the best next step is memory/identity/backlog rather than code, update those durable artifacts with provenance, but do not treat that as an absorbed self-evolution cycle.",
        "- A true absorbed self-evolution cycle requires one reviewed self-modification commit followed by successful restart verification before the next campaign cycle.",
        "- The review enforcement mode (advisory vs blocking) is the owner's setting. Do NOT hardcode review findings to always block (or always pass) regardless of that mode: forcing per-finding blocks under an owner-chosen advisory mode is forbidden self-modification (BIBLE P3), not a hardening. If advisory pass-through of a critical finding feels wrong, surface it to the owner — never patch the enforcement gate to override their choice.",
        "- If the objective is complete or needs owner input, say so clearly in the final result.",
    ])
    return "\n".join(parts)


def notify_owner_cycle_outcome(campaign: Dict[str, Any], tx: Dict[str, Any]) -> None:
    """WS-13.5 (e5=ux_absorb_report): owner-facing chat note for a finished
    self-evolution cycle. Absorbed -> short what/why; abandoned -> honest
    warning; no_op / waiting -> quiet (the lifecycle event already records it).
    No web/UI edits; chat only, budget-gated. Lazy imports avoid an import
    cycle with supervisor.state / supervisor.message_bus.
    """
    outcome = str(tx.get("cycle_outcome") or "")
    if outcome not in ("absorbed", "abandoned"):
        return  # no_op / waiting_for_restart: event-only, stay quiet
    from supervisor.state import load_state
    from supervisor.message_bus import send_with_budget
    owner_chat_id = int(load_state().get("owner_chat_id") or 0)
    if not owner_chat_id:
        return
    objective = str(campaign.get("objective") or "").strip()
    obj_short = (objective[:160] + "…") if len(objective) > 160 else objective
    if outcome == "absorbed":
        commit_sha = str(tx.get("commit_sha") or "").strip()[:12]
        msg = (
            f"🧬 Evolution cycle absorbed (commit {commit_sha}).\n"
            f"Objective: {obj_short or 'autonomous self-improvement'}\n"
            "The reviewed self-modification is now live (restart verified). Reply if you want it reverted."
        )
    else:
        reason = str(tx.get("abandoned_reason") or "unspecified")
        msg = (
            f"⚠️ Evolution cycle abandoned (reason: {reason}).\n"
            f"Objective: {obj_short or 'autonomous self-improvement'}\n"
            "No change was absorbed; the transaction was rolled back/closed to unblock the next cycle."
        )
    send_with_budget(owner_chat_id, msg)


def append_unique_transaction(campaign: Dict[str, Any], tx: Dict[str, Any]) -> None:
    tx_history = list(campaign.get("transaction_history") or [])
    tx_id = str(tx.get("transaction_id") or "")
    if tx_id and any(isinstance(item, dict) and str(item.get("transaction_id") or "") == tx_id for item in tx_history):
        campaign["transaction_history"] = tx_history[-50:]
        return
    tx_history.append(dict(tx))
    campaign["transaction_history"] = tx_history[-50:]


def request_evolution_restart(drive_root: pathlib.Path, tx: Dict[str, Any], log: Any = None) -> None:
    if str(os.environ.get("OUROBOROS_EVOLUTION_AUTO_RESTART", "true") or "true").lower() in {"0", "false", "no", "off"}:
        return
    commit_sha = str(tx.get("commit_sha") or "").strip()
    if not commit_sha:
        return
    try:
        atomic_write_json(
            pathlib.Path(drive_root) / "state" / "pending_restart_verify.json",
            {
                "ts": utc_now_iso(),
                "expected_sha": commit_sha,
                "expected_branch": str(tx.get("base_branch") or ""),
                "reason": "supervisor_auto_evolution_restart",
            },
            trailing_newline=True,
        )
        from supervisor import workers

        workers.get_event_q().put({
            "type": "restart_request",
            "reason": "supervisor_auto_evolution_restart",
            "ts": utc_now_iso(),
        })
    except Exception:
        if log is not None:
            log.debug("Failed to request automatic evolution restart", exc_info=True)

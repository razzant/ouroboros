"""What the evolution campaign contributes to the queue, and when.

Reads the campaign's status for the owner-facing snapshot, and admits the next
evolution cycle only when the budget reserve, the campaign state and the objective
repeat cap all allow it.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Dict, Optional
from supervisor.state import EVOLUTION_BUDGET_RESERVE
from ouroboros.contracts.task_contract import attach_task_contract
from ouroboros.utils import utc_now_iso
from supervisor.evolution_lifecycle import (
    build_evolution_task_text,
    disable_evolution_authority,
    disable_evolution_projection,
    deliver_pending_owner_report,
    evolution_block_reason,
    pause_evolution_campaign,
)


def _queue():
    """The parent module, read at call time.

    The queue owns PENDING/RUNNING, the drive root, the liveness settings and the lock that guards them, and ``init``/``init_queue_refs`` REBIND those names. Reading them through the module is what keeps one binding: a from-import here would freeze the value this module saw at import time.
    """
    from supervisor import queue

    return queue


log = logging.getLogger(__name__)


def queue_deep_self_review_task(reason: str, model: str = "", force: bool = False, chat_id: Optional[int] = None) -> Optional[str]:
    """Queue a deep self-review task.

    ``chat_id`` targets a specific chat (e.g. the external transport chat that ran
    ``/review``) so the queued ack and the task results return to the requester
    instead of always defaulting to the web owner's ``owner_chat_id``.
    """
    target_chat_id = chat_id if chat_id else _queue().load_state().get("owner_chat_id")
    if not target_chat_id:
        return None
    if (not force) and _queue().queue_has_task_type("deep_self_review"):
        return None
    tid = uuid.uuid4().hex[:8]
    _queue().enqueue_task({
        "id": tid,
        "type": "deep_self_review",
        "chat_id": int(target_chat_id),
        "text": reason or "Deep self-review",
        "model": model,
    })
    _queue().persist_queue_snapshot(reason="deep_self_review_enqueued")
    _queue().send_with_budget(int(target_chat_id), f"🔎 Deep self-review queued: {tid} ({reason})")
    return tid


def get_evolution_status_snapshot(*, budget_projection: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a non-mutating evolution scheduling snapshot.

    ``budget_projection``: optional pre-computed global usage projection from a
    caller that already replayed the ledger this request (``/api/state``), so the
    snapshot does not replay it again. Default ``None`` keeps the self-computing,
    strict fail-closed behavior — a caller whose own computation FAILED must pass
    nothing, so the paused-evolution disclosure still comes from this snapshot.
    """
    st = _queue().load_state()
    enabled = bool(st.get("evolution_mode_enabled"))
    owner_chat_id = int(st.get("owner_chat_id") or 0)
    consecutive_failures = int(st.get("evolution_consecutive_failures") or 0)
    try:
        remaining: Optional[float] = round(float(_queue().budget_remaining(st, strict=True, projection=budget_projection)), 2)
        accounting_available = True
    except Exception:
        remaining = None
        accounting_available = False
    queued_task = next((t for t in _queue().PENDING if str(t.get("type") or "") == "evolution"), None)
    running_task = next(
        (
            (meta.get("task") if isinstance(meta, dict) else None)
            for meta in _queue().RUNNING.values()
            if isinstance(meta, dict)
            and isinstance(meta.get("task"), dict)
            and str(meta["task"].get("type") or "") == "evolution"
        ),
        None,
    )
    status = "disabled"
    detail = "Evolution mode is off."

    campaign = _queue()._read_evolution_campaign()
    active_tx = campaign.get("active_transaction") if isinstance(campaign.get("active_transaction"), dict) else {}
    restart_blocked = bool(
        active_tx
        and str(active_tx.get("commit_sha") or "").strip()
        and (bool(active_tx.get("restart_required")) or not bool(active_tx.get("restart_verified")))
    )

    if restart_blocked:
        status = "waiting_for_restart_verify"
        detail = "Waiting for restart verification before the next absorbed evolution cycle."
    elif isinstance(running_task, dict):
        status = "running"
        detail = "Evolution task is running now."
    elif isinstance(queued_task, dict):
        status = "queued"
        detail = "Evolution task is queued and waiting for a worker."
    elif not accounting_available:
        status = "accounting_unavailable"
        detail = "Cost accounting is unavailable; evolution dispatch is paused without changing the campaign."
    elif consecutive_failures >= 3:
        status = "paused_failures"
        detail = (
            f"Paused after {consecutive_failures} consecutive failures. "
            "Use Evolve again after investigating the failure."
        )
    elif enabled and not owner_chat_id:
        status = "waiting_for_owner_chat"
        detail = "Waiting for the first owner chat binding before scheduling evolution."
    elif enabled and remaining is not None and remaining < EVOLUTION_BUDGET_RESERVE:
        status = "budget_blocked"
        detail = (
            f"Budget reserve active: ${remaining:.2f} remaining, "
            f"${EVOLUTION_BUDGET_RESERVE:.0f} reserved for conversations."
        )
    elif enabled and (_queue().PENDING or _queue().RUNNING):
        status = "waiting_for_idle"
        detail = "Waiting for active tasks to finish before the next evolution cycle."
    elif enabled:
        status = "idle_ready"
        detail = "Idle and ready to queue the next evolution cycle."
    elif remaining is not None and remaining < EVOLUTION_BUDGET_RESERVE and str(st.get("last_evolution_task_at") or "").strip():
        status = "budget_stopped"
        detail = (
            f"Evolution auto-stopped because only ${remaining:.2f} remains, "
            f"below the ${EVOLUTION_BUDGET_RESERVE:.0f} conversation reserve."
        )

    return {
        "enabled": enabled,
        "status": status,
        "detail": detail,
        "campaign": campaign,
        "cycle": int(st.get("evolution_cycle") or 0),
        "owner_chat_bound": bool(owner_chat_id),
        "last_task_at": str(st.get("last_evolution_task_at") or ""),
        "consecutive_failures": consecutive_failures,
        "cost_accounting_status": "available" if accounting_available else "unavailable",
        # Unbounded budget (supervisor not initialized / TOTAL_BUDGET<=0)
        # is float('inf'), which strict JSON cannot carry — surface None so
        # /api/state stays serializable on onboarding installs.
        "budget_remaining_usd": remaining if remaining is not None and math.isfinite(remaining) else None,
        "budget_reserve_usd": float(EVOLUTION_BUDGET_RESERVE),
        "pending_count": len(_queue().PENDING),
        "running_count": len(_queue().RUNNING),
        "queued_task_id": str((queued_task or {}).get("id") or ""),
        "running_task_id": str((running_task or {}).get("id") or ""),
    }


def _deliver_pending_owner_report() -> None:
    deliver_pending_owner_report(_queue().notify_owner_cycle_outcome)


def enqueue_evolution_task_if_needed() -> None:
    """Queue evolution only when idle, enabled, within budget, and not failure-paused."""
    _deliver_pending_owner_report()
    if _queue().PENDING or _queue().RUNNING:
        return
    st = _queue().load_state()
    if not bool(st.get("evolution_mode_enabled")):
        return
    owner_chat_id = st.get("owner_chat_id")
    if not owner_chat_id:
        return
    campaign = _queue()._read_evolution_campaign()
    from supervisor.state import update_state
    has_authority = all(str(campaign.get(key) or "").strip() for key in ("id", "source"))
    if campaign.get("status") != "active" or not has_authority:
        disable_evolution_authority("bare_flag_disabled", campaign_id=str(campaign.get("id") or ""))
        _queue().send_with_budget(
            int(owner_chat_id),
            "🧬 Evolution stayed off: the enable flag had no active campaign authority. Use /evolve start to begin a fresh campaign.",
        )
        return
    active_tx = campaign.get("active_transaction") if isinstance(campaign.get("active_transaction"), dict) else {}
    if active_tx and (
        str(active_tx.get("commit_sha") or "").strip()
        or str(active_tx.get("dispatch_status") or "") == "reaping"
    ):
        return

    # Defensive net: light mode must never run evolution even if the flag was
    # left enabled (e.g. carried across a restart into light mode). Disable and
    # pause once; entry points already refuse new starts up front.
    block = evolution_block_reason()
    if block:
        pause_evolution_campaign("blocked in light runtime mode")
        disable_evolution_projection()
        _queue().send_with_budget(int(owner_chat_id), block)
        return

    consecutive_failures = int(st.get("evolution_consecutive_failures") or 0)
    if consecutive_failures >= 3:
        pause_evolution_campaign("paused after consecutive failures")
        disable_evolution_projection()
        _queue().send_with_budget(
            int(owner_chat_id),
            f"🧬⚠️ Evolution paused: {consecutive_failures} consecutive failures. "
            f"Use /evolve start to resume after investigating the issue."
        )
        return

    # BUG3: pause if the SAME objective has been re-proposed and no-op'd
    # OBJECTIVE_REPEAT_CAP times without ever absorbing. This is a SEPARATE breaker from
    # consecutive_failures above: that counter is reset to 0 by ANY non-failing cycle
    # (events.py), so it cannot catch a self-maintenance loop where a blocked objective is
    # re-proposed NON-consecutively (interleaved with other no_op work). The per-objective
    # count is keyed on the same canonical fingerprint the transaction stamps, accumulates
    # across non-consecutive recurrence, and is cleared only on a genuine absorb.
    from ouroboros.evolution_fingerprint import canonical_objective_fingerprint

    _objective_repeat_counts = campaign.get("objective_repeat_counts") or {}
    _active_objective_fp = canonical_objective_fingerprint(str(campaign.get("objective") or ""))
    _objective_repeats = int(_objective_repeat_counts.get(_active_objective_fp, 0)) if _active_objective_fp else 0
    if _objective_repeats >= _queue().OBJECTIVE_REPEAT_CAP:
        pause_evolution_campaign("paused: objective re-proposed without ever absorbing")
        disable_evolution_projection()
        _queue().send_with_budget(
            int(owner_chat_id),
            f"🧬⚠️ Evolution paused: the current objective ran {_objective_repeats} reviewed "
            f"cycles WITHOUT ever being absorbed — it keeps getting re-proposed and never lands "
            f"(a self-maintenance loop, not progress). A plain resume won't help; use "
            f"/evolve start with a DIFFERENT objective."
        )
        return

    try:
        remaining = _queue().budget_remaining(st, strict=True)
    except Exception:
        log.error("Evolution scheduling deferred: cost accounting unavailable", exc_info=True)
        _queue().append_jsonl(_queue().DRIVE_ROOT / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": "evolution_accounting_unavailable",
            "action": "dispatch_deferred", "owner_visible": True,
        })
        return
    if remaining < EVOLUTION_BUDGET_RESERVE:
        pause_evolution_campaign("budget reserve reached")
        disable_evolution_projection()
        _queue().send_with_budget(int(owner_chat_id), f"💸 Evolution stopped: ${remaining:.2f} remaining (reserve ${EVOLUTION_BUDGET_RESERVE:.0f} for conversations).")
        return
    cycle = int(st.get("evolution_cycle") or 0) + 1
    tid = uuid.uuid4().hex[:8]
    transaction = _queue().begin_evolution_transaction(tid, cycle=cycle, campaign=campaign)
    if not transaction:
        disable_evolution_authority("transaction_attach_failed", campaign_id=str(campaign.get("id") or ""), task_id=tid)
        _queue().send_with_budget(
            int(owner_chat_id),
            "🧬 Evolution stayed off: the campaign changed before its next task could be attached. Start it again when ready.",
        )
        return
    task = {
        "id": tid, "type": "evolution",
        "chat_id": int(owner_chat_id),
        "text": build_evolution_task_text(cycle),
        "metadata": {"evolution_transaction": transaction},
    }
    attach_task_contract(task)
    _queue().enqueue_task(task)

    def _record_cycle(live: Dict[str, Any]) -> None:
        live["evolution_cycle"] = cycle
        live["last_evolution_task_at"] = utc_now_iso()

    update_state(_record_cycle)

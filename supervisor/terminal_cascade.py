"""The cascade's one owner-facing terminal message: root salvage plus a children digest.

Moved from ``supervisor/terminal_delivery.py`` at its module ceiling. The shared delivery seams
(owed registration, the unreviewed-salvage builder and sender) are reached through the
``terminal_delivery`` module object at call time, so that module stays the one import and
patch surface; it re-exports these names for its callers.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)
_GAP_NAMES = 10  # damaged rows a membership gap names before it counts the rest


def _td():
    from supervisor import terminal_delivery

    return terminal_delivery


def deliver_cascade_summary(
    drive_root: Any, task_id: str, root_task_row: Dict[str, Any], outcomes: Dict[str, str],
) -> bool:
    """ONE chat message for a settled cascade: root salvage + children digest.

    A2: the sweeps ran custody with ``deliver=False``, so this is the tree's only
    owner-facing terminal message. Fail-soft; durable dedupe inside the seam.
    The cascade postcondition calls this UNCONDITIONALLY before it settles the
    root intent (GR3-1c) — including the replay/already-down path — so a tree
    with no resolvable lineage chat records a typed ``terminal_delivery_handoff``
    row instead of vanishing (the crash-order evidence must show the summary was
    consciously not owed, never silently dropped).

    DELIVERY IDENTITY (GR4-2): the summary's delivery id is
    ``cascade:<root_tid>:<intent request_id>`` — deterministic per cascade
    INTENT, not per message content. A watchdog replay of the SAME intent
    (crash between the owed summary and the settle) rebuilds a digest whose
    content may differ (children settled meanwhile), and a content-derived id
    would deliver a second summary; the intent-derived id dedups it through
    the delivered registry, while a LATER separate cancel request (new
    request_id) legitimately delivers its own. Falls back to the content id
    when no intent row exists (direct internal callers).

    Returns whether the summary is DURABLY accounted for (GR4-1): owed /
    already delivered / consciously handed off. ``False`` = the owed
    registration failed; the caller leaves the cascade intent OPEN so the
    watchdog re-feeds (loud by the typed ``terminal_delivery_unregistered``
    event each attempt).
    """
    try:
        from ouroboros.cancel_intents import STOP_POLICY_FINALIZE, active_intent, stop_policy
        from ouroboros.observability import preserved_salvage_path
        from ouroboros.owner_stop_receipt import capture_receipt, has_wrap_up_receipt, resolve_receipt_target
        from ouroboros.task_results import load_task_result
        from ouroboros.task_status import SETTLED_STATUSES

        try:
            intent = active_intent(pathlib.Path(drive_root), task_id) or {}
        except Exception:
            intent = {}
        stop = (task_id, str(intent.get("request_id") or ""))
        physical_task_id, _pred_ids, retry_root_ids = resolve_receipt_target(
            pathlib.Path(drive_root), task_id, stop=stop,
        )
        wrap_up = (
            stop_policy(intent) == STOP_POLICY_FINALIZE
            or bool(intent.get("hardened_at"))
            or has_wrap_up_receipt(pathlib.Path(drive_root), physical_task_id, stop=stop)
        )

        def _durable_child_outcome(tid: str, sweep_outcome: str) -> str:
            # GR4-4: each child's line reports its CURRENT durable status at
            # digest build time — a child a concurrent actor settled after the
            # sweep must not be shown with the sweep's stale ``failed``.
            try:
                status = str(
                    (load_task_result(pathlib.Path(drive_root), tid) or {}).get("status") or ""
                )
            except Exception:
                status = ""
            return status if status in SETTLED_STATUSES else sweep_outcome

        # GR5-4: digest MEMBERSHIP comes from the durable tree, not only this
        # run's sweep outcomes — a watchdog replay after the children already
        # terminalized runs with EMPTY outcomes and would otherwise deliver a
        # root summary whose digest silently omits the whole subtree. GR6-2:
        # membership is ANCESTRY rooted at the cancelled node (a parent-chain
        # walk over the durable rows plus the queue snapshot — the same
        # reachability the cascade sweep uses), never a root_task_id equality:
        # a MID-TREE target's grandchildren keep the ORIGINAL tree's root id
        # and match no equality clause, and non-subagent descendants belong in
        # the digest too. Sweep outcomes win only for ids with no durable row
        # yet (fresh kills whose write is still in flight); every line still
        # reads its CURRENT durable status through ``_durable_child_outcome``
        # (GR4-4). Bounded exactly as today: the durable ``cancel_receipt``
        # block caps the persisted digest at 40 with the exact omitted count.
        merged: Dict[str, str] = {
            str(tid): str(outcome) for tid, outcome in outcomes.items()
        }
        membership_gap = ""
        try:
            gaps: list = []
            for tid, status in _cascade_descendant_rows(
                pathlib.Path(drive_root), task_id, strict=wrap_up, gaps=gaps,
            ).items():
                if tid and tid not in merged:
                    merged[tid] = status
            membership_gap = "; ".join(gaps)  # a named damaged row keeps the known children, honestly partial
        except Exception:
            membership_gap = "descendant_enumeration_unavailable"
            log.debug(
                "cascade digest durable-tree enumeration failed for %s",
                task_id, exc_info=True,
            )
        children = [
            {"task_id": tid, "outcome": _durable_child_outcome(tid, outcome),
             "salvaged": bool(preserved_salvage_path(pathlib.Path(drive_root), tid))}
            for tid, outcome in sorted(merged.items())
            if tid not in retry_root_ids
        ]
        root_result = (
            load_task_result(pathlib.Path(drive_root), physical_task_id)
            or load_task_result(pathlib.Path(drive_root), task_id)
            or {}
        )
        # Expiry/crash may precede the final control. The same receipt (one row
        # identity per intent) still accompanies this fallback, without another
        # model turn; a failed capture never holds the tree's one summary.
        if wrap_up:
            try:
                capture_receipt(pathlib.Path(drive_root), physical_task_id, merged, gap=membership_gap,
                                stop=stop)
            except Exception:
                log.warning("Wrap up receipt capture failed for %s", task_id, exc_info=True)
            root_result = (
                load_task_result(pathlib.Path(drive_root), physical_task_id)
                or load_task_result(pathlib.Path(drive_root), task_id)
                or root_result
            )
        root_status = str(root_result.get("status") or "")
        salvage_path = (
            preserved_salvage_path(pathlib.Path(drive_root), physical_task_id)
            or preserved_salvage_path(pathlib.Path(drive_root), task_id)
        )
        salvage_text = ""
        if root_status == "completed":
            # Completion-wins root: its own completed result is the answer.
            salvage_text = str(root_result.get("result") or "")
        elif salvage_path:
            try:
                salvage_text = pathlib.Path(salvage_path).read_text(encoding="utf-8")
            except Exception:
                salvage_text = ""
        # A partial digest census rides the summary's own receipt and words; Wrap up's row already names it.
        summary_gap = "" if wrap_up else membership_gap
        delivery_id = ""
        try:
            from ouroboros.cancel_intents import active_intent

            intent = active_intent(pathlib.Path(drive_root), task_id) or {}
            if intent.get("request_id"):
                delivery_id = f"cascade:{task_id}:{intent['request_id']}"
        except Exception:
            log.debug("cascade delivery-id intent read failed for %s", task_id, exc_info=True)
        event = _td().build_unreviewed_salvage_event(
            pathlib.Path(drive_root),
            root_task_row or root_result,
            task_id,
            outcome=_cascade_root_outcome(root_status),
            salvaged_text=salvage_text,
            preserved_path=salvage_path,
            children=children,
            membership_gap=summary_gap,
            settled_status=root_status,
            delivery_id=delivery_id,
        )
        # GR4-1: register the owed row FIRST so the accounting result is
        # observable; deliver_unreviewed_salvage re-registers idempotently.
        owed = True
        if event is not None:
            owed = _td().register_pending_delivery(pathlib.Path(drive_root), event)
        _td().deliver_unreviewed_salvage(
            pathlib.Path(drive_root),
            root_task_row or root_result,
            task_id,
            outcome=_cascade_root_outcome(root_status),
            salvaged_text=salvage_text,
            preserved_path=salvage_path,
            children=children,
            membership_gap=summary_gap,
            # GR2-12: the typed root status decides the completed-vs-salvage
            # framing; the prose in ``outcome`` is presentation only.
            settled_status=root_status,
            delivery_id=delivery_id,
        )
        return owed
    except Exception:
        log.warning("Cascade summary delivery failed for %s", task_id, exc_info=True)
        return False


def _cascade_descendant_rows(drive_root: pathlib.Path, task_id: str, *, strict: bool = False,
                             gaps: Optional[List[str]] = None) -> Dict[str, str]:
    """``{descendant_id: status}`` for every task whose ancestry reaches ``task_id``.

    GR6-2: the digest's membership walk. Rows come from the durable task
    results UNION the queue snapshot (a still-live or freshly-killed child may
    have no durable row yet); reachability is the recorded ``root_task_id``
    shortcut (covers a whole tree whose intermediate parents left no rows)
    plus a depth-bounded parent-chain walk over the collected rows — the same
    reachability shape the cascade sweep and ``_has_live_ancestor_in_set``
    use. Deliberately NOT filtered to subagents: a non-subagent descendant
    torn down by the cascade belongs in the tree's one summary too. A damaged
    row the census could not place, or a descendant whose own record it could
    not read (still listed), keeps every known row and is named in ``gaps``
    (the receipt's membership gap); so is a queue-only fallback after the
    store itself failed, which ``strict`` raises instead.
    """
    target = str(task_id or "").strip()
    queued = []
    try:
        from ouroboros.utils import read_json_dict

        snapshot = read_json_dict(
            pathlib.Path(drive_root) / "state" / "queue_snapshot.json",
        ) or {}
        for group, status in (("pending", "scheduled"), ("running", "running")):
            for item in snapshot.get(group) or []:
                if not isinstance(item, dict):
                    continue
                task = item.get("task") if isinstance(item.get("task"), dict) else item
                tid = str(item.get("id") or task.get("id") or "")
                if tid:
                    queued.append({**task, "task_id": tid, "status": status})
    except Exception:
        if strict:
            raise
        log.debug("cascade digest snapshot scan failed for %s", target, exc_info=True)

    try:
        from ouroboros.task_store import tree_facts

        census = tree_facts(drive_root, target, descendants=True, queue_rows=queued)
        rows, unresolved = census.members, census.unresolved
    except Exception:
        if strict:
            raise
        log.debug("cascade digest durable-row selection failed for %s", target, exc_info=True)
        rows, unresolved = {row["task_id"]: row for row in reversed(queued)}, None
    if gaps is not None and unresolved is None:
        gaps.append("descendant_enumeration_unavailable")
    elif gaps is not None and set(unresolved) - {target}:  # the target's own record is no descendant's
        named = [f"{tid} ({fact['reason']})" for tid, fact in sorted(unresolved.items()) if tid != target]
        gaps.append("unresolved_task_rows: " + ", ".join(named[:_GAP_NAMES])
                    + (f" and {len(named) - _GAP_NAMES} more" if len(named) > _GAP_NAMES else ""))

    def _reaches_target(tid: str) -> bool:
        row = rows.get(tid) or {}
        if str(row.get("root_task_id") or "") == target:
            return True
        parent = str(row.get("parent_task_id") or "")
        seen: set = set()
        while parent and parent not in seen and len(seen) < 100:
            if parent == target:
                return True
            seen.add(parent)
            parent = str(rows.get(parent, {}).get("parent_task_id") or "")
        return False

    return {
        tid: str(row.get("status") or "")
        for tid, row in rows.items()
        if tid != target and _reaches_target(tid)
    }


def _cascade_root_outcome(root_status: str) -> str:
    """How the tree's ONE message describes its root — from the ROOT'S OWN status.

    A cascade over a root that had already ``failed`` (or was killed on a budget
    hard stop) used to describe it as "stopped", which is the summary telling
    the owner a different story than the task card. Only a root this cascade
    actually cancelled is called stopped.
    """
    status = str(root_status or "").strip().lower()
    if status == "completed":
        return "completed before the stop (result preserved)"
    if status and status != "cancelled":
        return f"already settled as {status} before the stop (result preserved)"
    return "stopped"

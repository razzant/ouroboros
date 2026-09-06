"""Absorbing a child: reading one result, or waiting on a batch of them.

A parent takes a child's work back through two surfaces — the full single-child
read and the compact batch projection — and they must agree about what matters:
the outcome axes, the pinned result hash, the receipts, and any capability the
child did not actually have. The waits add the facts a blocking parent cannot
see for itself: an attention beacon raised mid-flight, siblings still running,
an id this tree never minted, and a prompt cache that expired while it waited.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from ouroboros.outcomes import normalize_outcome_axes
from ouroboros.task_results import (
    STATUS_COMPLETED,
    STATUS_REJECTED_DUPLICATE,
    validate_task_id,
)
from ouroboros.task_status import load_effective_task_result, wait_for_effective_tasks
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import truncate_review_artifact
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result


def disclosable_capability_delta(data: Dict[str, Any]) -> Dict[str, Any]:
    """The child's delta when it has something to SAY, else ``{}`` — ONE predicate.

    THE terminal parent-facing disclosure, and since v6.87.28 the only parent-facing
    one: the reduction is not known until the child is dispatched, so no scheduling
    result can carry it. It is a predicate rather than an inline test because the
    parent absorbs a child through TWO surfaces — `get_task_result`/`wait_task` read
    one child in full, `wait_tasks` projects a batch compactly — and the batch one
    is the surface a fan-out parent actually uses. It had the test in neither place
    and the disclosure in one, so a parent that scheduled five children and absorbed
    them in a burst was told nothing about any of them.

    A delta that took nothing away and ignored nothing is noise in every payload.
    """
    delta = data.get("capability_delta") if isinstance(data.get("capability_delta"), dict) else {}
    return delta if (delta.get("reduced") or delta.get("legacy_note")) else {}


def _subtask_outcome_summary(data: Dict[str, Any], receipts: list | None = None) -> str:
    ledger = data.get("verification_ledger") if isinstance(data.get("verification_ledger"), dict) else {}
    summary: Dict[str, Any] = {
        "outcome_axes": normalize_outcome_axes(data),
    }
    # R5: CURRENT delegated-custody reconciliation state next to the historical
    # axes, on the full single-child handoff surfaces only (get_task_result /
    # wait_task — wait_tasks' batch projection is a pinned compact contract).
    # The frozen delegated_runs_* counters are a historical snapshot (owner
    # Q2=B); the envelope's trigger + open_run_ids are the liveness surface a
    # parent can trust after a refresh/backfill healed the row. One bounded
    # entry: capped lists with an exact omitted count.
    unreconciled = data.get("delegated_runs_unreconciled")
    unreconciled = (
        [str(item) for item in unreconciled] if isinstance(unreconciled, list) else []
    )
    envelope = data.get("delegate_terminal_reconciliation")
    envelope = envelope if isinstance(envelope, dict) else {}
    if unreconciled or envelope:
        custody: Dict[str, Any] = {"unreconciled": unreconciled[:10]}
        omitted_any = False
        if len(unreconciled) > 10:
            custody["unreconciled_omitted"] = len(unreconciled) - 10
            omitted_any = True
        if envelope:
            open_ids = envelope.get("open_run_ids")
            open_ids = [str(item) for item in open_ids] if isinstance(open_ids, list) else []
            custody["trigger"] = str(envelope.get("trigger") or "")
            custody["open_run_ids"] = open_ids[:10]
            if len(open_ids) > 10:
                custody["open_run_ids_omitted"] = len(open_ids) - 10
                omitted_any = True
        if omitted_any:
            # A bound must name a source the actor can resolve (BIBLE P1).
            # Retry lineage unions the ORIGINAL row's disclosure into this
            # projection, so the omitted identifiers may live on either row —
            # name both when the lineage exists.
            tid = str(data.get("task_id") or "")
            origin = str(
                data.get("original_task_id") or data.get("timeout_retry_from") or ""
            )
            paths = [f"task_results/{tid}.json"]
            if origin and origin != tid:
                paths.append(f"task_results/{origin}.json")
            custody["full_source"] = [
                f"read_file(root='runtime_data', path='{p}')" for p in paths
            ]
        summary["delegated_custody"] = custody
    if isinstance(data.get("task_contract"), dict):
        summary["task_contract"] = data.get("task_contract")
    _delta = disclosable_capability_delta(data)
    if _delta:
        summary["capability_delta"] = _delta
    if isinstance(data.get("artifact_bundle"), dict):
        summary["artifact_bundle"] = data.get("artifact_bundle")
    if ledger:
        # An omitted-to-artifact stub carries no entries; its summary is the
        # count authority, and for a full ledger the two always agree.
        ledger_summary = ledger.get("summary") if isinstance(ledger.get("summary"), dict) else {}
        summary["verification_ledger"] = {
            "schema_version": ledger.get("schema_version"),
            "summary": ledger_summary,
            "entry_count": ledger_summary.get("entry_count", len(ledger.get("entries") or []) if isinstance(ledger.get("entries"), list) else 0),
        }
    if receipts:
        # W2: bounded per-receipt rows for the FULL single-child handoff ONLY
        # (get_task_result/wait_task — already uncapped surfaces): which checks
        # passed, not just counts, so a parent can absorb a child on receipt-level
        # green/red instead of prose. The wait_tasks BATCH projection deliberately
        # stays counts-compact (v6.17.0 birth shape + v6.71.2 measured compaction,
        # 694K->25K). Rows render through the SSOT identity projection + disclosed
        # bound (hard cap, exact omitted count).
        #
        # The bound is OUTSTANDING-FIRST, then newest: a plain newest-10 window let
        # a child that failed a check early and then produced ten greens hand the
        # parent an affirmatively all-green list, with the red only implied by a
        # count. The still-unreconciled SET is this repo's SSOT for exactly that
        # problem ("a newer red would let a latest-pointer erase an older still-red
        # one"), so every outstanding red / masked pass is carried first — tagged so
        # the parent sees WHY it is here — and the rest of the cap is filled with the
        # newest remaining receipts. The cap and its exact omitted count are unchanged.
        from ouroboros._outcome_receipts import (
            disclosed_list_projection,
            receipt_identity_projection,
            unreconciled_failed,
            unreconciled_masked,
        )

        rows = [r for r in receipts if isinstance(r, dict)]
        outstanding_kind: Dict[int, str] = {}
        for _receipt in unreconciled_failed(rows):
            outstanding_kind[id(_receipt)] = "unreconciled_failed"
        for _receipt in unreconciled_masked(rows):
            outstanding_kind.setdefault(id(_receipt), "unreconciled_masked_pass")
        ordered = [r for r in reversed(rows) if id(r) in outstanding_kind]
        ordered += [r for r in reversed(rows) if id(r) not in outstanding_kind]

        def _receipt_row(receipt: Any) -> Any:
            if not isinstance(receipt, dict):
                return truncate_review_artifact(str(receipt), limit=200)
            row = {"status": str(receipt.get("status") or "")}
            outstanding = outstanding_kind.get(id(receipt), "")
            if outstanding:
                row["outstanding"] = outstanding
            if "matched" in receipt:
                row["matched"] = receipt.get("matched")
            row.update(receipt_identity_projection(receipt, check_cap=200))
            return row

        summary.update(disclosed_list_projection(
            ordered, key="verification_receipts", limit=10, item=_receipt_row,
        ))
    return json.dumps(summary, ensure_ascii=False, indent=2, default=str)


def _get_task_result(
    ctx: ToolContext, task_id: str, include_authority: bool = False,
    include_work_order_source: bool = False, source_start_char: Any = None,
    source_end_char: Any = None, include_completion_source: bool = False,
) -> str:
    """Read a task result, or a bounded canonical work-order/completion source range."""
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    status_drive_root = Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    data = load_effective_task_result(status_drive_root, task_id)
    if not data:
        return _publish_tool_result(ctx, ToolResult(
            status="unavailable", code="LEGACY_UNAVAILABLE",
            text=f"Task {task_id}: unknown or not yet registered",
        ))
    if bool(include_authority) or bool(include_work_order_source) or bool(include_completion_source):
        from ouroboros.agent_startup_checks import task_result_authority_projection

        authority = task_result_authority_projection(data, drive_root=status_drive_root)
        payload: Dict[str, Any] = {
            "status": "available", "authority": authority,
            "source": {"tool": "get_task_result", "task_id": str(task_id)},
        }
        if bool(include_work_order_source):
            from ouroboros.subagent_work_order import (
                _source_task_from_context,
                work_order_source_projection,
            )

            projection, reason = work_order_source_projection(
                _source_task_from_context(ctx, str(task_id)),
                source_start_char,
                source_end_char,
            )
            source = {
                "kind": "task_result",
                "task_id": str(task_id),
                "tool": "get_task_result",
                "arguments": {
                    "task_id": str(task_id),
                    "include_authority": True,
                    "include_work_order_source": True,
                },
                "projection": "canonical_work_order",
            }
            payload["source"] = source
            payload["work_order_source"] = projection or {
                "schema": 1, "kind": "canonical_work_order", "status": "unavailable",
            }
            if reason:
                payload["work_order_source"]["reason"] = reason
        if bool(include_completion_source):
            from ouroboros.task_finalization import completion_source_projection

            payload["completion_source"] = completion_source_projection(
                status_drive_root, str(task_id), data, source_start_char, source_end_char,
            )
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    status = data.get("status", "unknown")
    result = data.get("result", "")
    trace = data.get("trace_summary", "")
    try:
        from ouroboros.outcomes import read_verification_receipts_from_roots
        from ouroboros.task_status import _child_drive_candidates

        # During the pre-copy-back window ordinary verification lives on the
        # isolated child drive while a zero-run lifecycle receipt is already on
        # the canonical root. Merge both replicas; a non-empty canonical file is
        # not evidence that the local one contains nothing new.
        receipts = read_verification_receipts_from_roots(
            [*_child_drive_candidates(data), status_drive_root], task_id,
        )
    except Exception:
        receipts = []
    outcome_summary = _subtask_outcome_summary(data, receipts=receipts)
    from ouroboros.tools.join_ledger import _child_result_sha256

    child_result_sha256 = _child_result_sha256(data)
    # SSOT cost projection (C2): unknown never renders as $0.00 (and a null in
    # the stored result no longer crashes the f-string with a TypeError).
    from ouroboros.cost_projection import cost_display

    if status == STATUS_COMPLETED:
        output = (
            f"Task {task_id} [{status}]: cost={cost_display(data)}\n"
            f"child_result_sha256={child_result_sha256}\n\n"
            f"[SUBTASK_OUTCOME]\n{outcome_summary}\n[/SUBTASK_OUTCOME]\n\n"
            f"[BEGIN_SUBTASK_OUTPUT]\n{result}\n[END_SUBTASK_OUTPUT]"
        )
    elif status == STATUS_REJECTED_DUPLICATE:
        duplicate_of = str(data.get("duplicate_of") or "?")
        output = (
            f"Task {task_id} [{status}]: duplicate_of={duplicate_of}\n"
            f"child_result_sha256={child_result_sha256}\n\n"
            f"[SUBTASK_OUTCOME]\n{outcome_summary}\n[/SUBTASK_OUTCOME]\n\n"
            f"{result or f'Task was rejected as a duplicate of {duplicate_of}.'}"
        )
    else:
        output = (
            f"Task {task_id} [{status}]\n"
            f"child_result_sha256={child_result_sha256}\n\n"
            f"[SUBTASK_OUTCOME]\n{outcome_summary}\n[/SUBTASK_OUTCOME]\n\n"
            f"{result or 'No details available.'}"
        )
    if trace:
        output += f"\n\n[SUBTASK_TRACE]\n{trace}\n[/SUBTASK_TRACE]"
    return output


def _wait_attention_poll(
    ctx: ToolContext, after_ts: str, task_ids: List[str],
) -> Callable[..., Any]:
    """on_poll hook: break a sliced wait for the actor's mailbox or a child attention beacon
    (blocker/question/interface_contract/review_requested/delegation_constraint).

    The cursor is context-local and per child: a beacon written before this
    particular tool call is still delivered, while a later wait in the same
    actor context does not replay it.  Equal-timestamp rows use their stable
    content identity, so the five-row response bound cannot strand the rest.
    """
    # tree_note/tree_read live in ouroboros/tools/task_tree.py (extracted for module size).
    from ouroboros.tools.task_tree import tree_root_id
    from ouroboros.owner_mailbox import OwnerMailboxPeek

    rid = tree_root_id(ctx)
    mailbox_peek = OwnerMailboxPeek()

    cursor_store = getattr(ctx, "_wait_attention_cursors", None)
    if not isinstance(cursor_store, dict):
        cursor_store = {}
        try:
            setattr(ctx, "_wait_attention_cursors", cursor_store)
        except Exception:
            # An exotic immutable context still gets correct delivery within
            # this hook instance; ordinary ToolContext objects retain it across
            # subsequent wait_task/wait_tasks calls.
            pass

    child_cursors: Dict[str, Dict[str, Any]] = {}
    for task_id in task_ids:
        key = f"{rid}:{task_id}"
        cursor = cursor_store.get(key)
        if not isinstance(cursor, dict):
            cursor = {"after_ts": str(after_ts or ""), "seen_ids": set()}
            cursor_store[key] = cursor
        if not isinstance(cursor.get("seen_ids"), set):
            cursor["seen_ids"] = {
                str(item) for item in (cursor.get("seen_ids") or []) if str(item)
            }
        child_cursors[str(task_id)] = cursor

    def _hook(_results: Dict[str, Any], _terminal: Dict[str, bool]) -> Any:
        from ouroboros.loop_transport import _owner_signal_pending

        # Reuse transport-wait's non-destructive peek. The ordinary round-top
        # drain still owns delivery and acknowledgement; child work stays live.
        if _owner_signal_pending(
            None, getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or ""),
            getattr(ctx, "_loop_mailbox_seen_ids", None), getattr(ctx, "task_attempt", None) or 1,
            mailbox_peek,
        ):
            return {"reason": "owner_mailbox_pending", "delivery": "pending_loop_drain"}
        if not rid:
            return None
        try:
            from ouroboros.task_tree_ledger import (
                tree_ledger_attention_after,
                tree_ledger_row_id,
            )

            attention = tree_ledger_attention_after(rid, "", task_ids=set(task_ids))
        except Exception:
            return None
        pending: List[tuple[Dict[str, Any], str]] = []
        for row in attention:
            task_id = str(row.get("task_id") or "")
            cursor = child_cursors.get(task_id)
            if cursor is None:
                continue
            ts = str(row.get("ts") or "")
            cursor_ts = str(cursor.get("after_ts") or "")
            row_id = tree_ledger_row_id(row)
            if ts < cursor_ts:
                continue
            if ts == cursor_ts and row_id in cursor["seen_ids"]:
                continue
            pending.append((row, row_id))
        if not pending:
            return None

        delivered = pending[:5]
        for row, row_id in delivered:
            cursor = child_cursors[str(row.get("task_id") or "")]
            ts = str(row.get("ts") or "")
            cursor_ts = str(cursor.get("after_ts") or "")
            if ts > cursor_ts:
                cursor["after_ts"] = ts
                cursor["seen_ids"] = set()
            cursor["seen_ids"].add(row_id)
        return {
            "reason": "child_attention_beacon",
            "beacons": [row for row, _row_id in delivered],
            "beacons_remaining": len(pending) - len(delivered),
        }

    return _hook


def cache_horizon_note(ctx: Any, elapsed_sec: Any) -> str:
    """One factual line when a blocking wait outlived the APPLIED prompt-cache TTL.

    Reads the RECORDED fact of this task's latest send — ``_last_prompt_cache_ttl``
    in the loop's accumulated usage (published on the tool ctx), converted by
    ``llm.cache_ttl_seconds`` — never a route-level prediction (a second predictor
    can disagree with the payload after route-filter/promotion/cap). Empty string
    when the horizon is unknown or not yet elapsed. UNKNOWN covers three cases,
    all silent: no cached send recorded, a route that carries no markers at all,
    and a send whose markers were BARE (reported ``"default"``) — a bare marker
    names no tier, so its horizon is the provider's business and inventing one
    would mislead the agent into re-planning its waits around a number nobody
    established. Only the explicitly stamped ``5m``/``1h`` tiers speak here.
    Deliberately NO token-count predictions: the submarine forensics showed the
    fact ("the wait outlived the cache") is what changes the agent's next decision
    (batch waits, longer single windows), while "~X tokens will re-write" is a
    counterfactual — the next send may reroute, compact, or still hit a live cache.

    REACHABILITY, honestly (each wait tool clamps its own window, so "all three
    wait tools carry the line" is a capability, not a per-configuration promise):
    at the shipped default TTL ``1h`` (3600s horizon) only ``wait_tasks`` (7200s
    clamp) can genuinely emit it; ``wait_task`` clamps at exactly 3600s and can
    only cross by a poll overshoot of a couple of seconds, and ``delegate_wait``
    clamps its WINDOW at ``config.DELEGATE_WAIT_WINDOW_MAX_SEC`` (1800s; the
    2100s ToolEntry ceiling above it is the kill timeout, not the window — F5)
    and cannot cross at all.
    At ``5m`` all three emit it. Pinned by
    tests/test_cache_optimization.py::test_cache_horizon_reachability_matches_the_wait_clamps —
    the call sites stay on all three because the tier is an owner setting, not a
    constant, and a wait tool that silently could not disclose would be worse.
    """
    try:
        elapsed = float(elapsed_sec)
    except (TypeError, ValueError):
        return ""
    usage = getattr(ctx, "_accumulated_usage", None)
    if not isinstance(usage, dict):
        return ""
    applied_ttl = str(usage.get("_last_prompt_cache_ttl") or "").strip()
    from ouroboros.llm import cache_ttl_seconds

    horizon = cache_ttl_seconds(applied_ttl)
    if horizon is None or elapsed <= horizon:
        return ""
    return (
        f"⚠️ configured prompt-cache horizon ({applied_ttl}, {horizon}s) elapsed during "
        f"this wait ({elapsed:.0f}s); the next model send may be cold."
    )


def _wait_for_task(ctx: ToolContext, task_id: str, timeout_sec: int = 180) -> str:
    """Wait for a subtask to reach a terminal status."""
    try:
        tid = validate_task_id(task_id)
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text=f"⚠️ TOOL_ARG_ERROR (wait_task): {exc}",
        ))
    try:
        timeout = max(0, min(int(timeout_sec), 3600))
    except (TypeError, ValueError):
        timeout = 180
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    status_drive_root = Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    waited = wait_for_effective_tasks(
        status_drive_root, [tid], timeout_sec=timeout,
        on_poll=_wait_attention_poll(ctx, "", [tid]), poll_interval_sec=2.0,
    )
    early = waited.get("early_return")
    if early and early.get("reason") == "owner_mailbox_pending":
        header = "Task wait interrupted by an unread message for this task"
        extra = "\n\nThe ordinary loop will deliver and acknowledge the message. This does not stop the child."
    elif early:
        header = "Task wait interrupted by a child attention beacon"
        extra = f"\n\n[CHILD_BEACONS]\n{json.dumps(early, ensure_ascii=False, indent=2)}\n[/CHILD_BEACONS]"
    else:
        header = "Task wait completed" if waited.get("all_terminal") else "Task wait timed out"
        extra = ""
    # B2 advisory (never a gate): if ANY other child of THIS parent is still in flight
    # while we block on this one, point at wait_tasks(any_terminal) so the agent absorbs
    # whichever finishes first instead of blocking serially on one id at a time.
    other_live = _count_live_sibling_children(ctx, status_drive_root, exclude_task_id=tid)
    if other_live >= 1:
        extra += (
            f"\n\n[ADVISORY] {other_live} other child(ren) still running/scheduled — consider "
            "wait_tasks(any_terminal) to absorb whichever finishes first instead of waiting one at a time."
        )
    horizon_note = cache_horizon_note(ctx, waited.get("elapsed_sec"))
    if horizon_note:
        extra += f"\n\n{horizon_note}"
    return f"{header} after {waited.get('elapsed_sec', 0):.1f}s.{extra}\n\n{_get_task_result(ctx, tid)}"


def _count_live_sibling_children(ctx: ToolContext, status_drive_root: Path, *, exclude_task_id: str) -> int:
    """Count this parent's children still running/scheduled/requested (excluding the one
    just waited on). Advisory only — a failure returns 0 so it never breaks wait_task."""
    parent_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not parent_id:
        return 0
    try:
        from ouroboros.task_results import (
            STATUS_REQUESTED,
            STATUS_RUNNING,
            STATUS_SCHEDULED,
            list_task_results,
        )

        live = 0
        for item in list_task_results(status_drive_root, statuses=[STATUS_RUNNING, STATUS_SCHEDULED, STATUS_REQUESTED]):
            if str(item.get("task_id") or item.get("id") or "") == exclude_task_id:
                continue
            if str(item.get("parent_task_id") or "") == parent_id:
                live += 1
        return live
    except Exception:
        return 0


_UNMINTED_WAIT_GRACE_SEC = 30.0


def _unminted_wait_ids(ctx: ToolContext, status_drive_root: Path, task_ids: List[str]) -> List[str]:
    """Ids with no trace on ANY surface this tree mints ids through: no task
    result, no queue-snapshot row, and no tree-ledger row naming them (v6.91).

    wave2's root blocked 900s slices on three hallucinated ids that wait_tasks
    silently polled as 'unknown' — while the real lead was missing from the wait
    set. The typed marker (plus the actual children roster) lets the parent
    repair its wait set instead of starving on phantoms. Fail-soft per probe: an
    unreadable surface treats the id as KNOWN — a real child must never be
    branded unknown on an I/O error."""
    from ouroboros.task_status import _load_queue_snapshot, _queue_task_status

    try:
        snapshot = _load_queue_snapshot(status_drive_root)
    except Exception:
        snapshot = {"_snapshot_invalid": True}
    ledger_ids: set = set()
    try:
        from ouroboros.task_tree_ledger import tree_ledger_rows
        from ouroboros.tools.task_tree import tree_root_id

        for row in tree_ledger_rows(tree_root_id(ctx)):
            for key in ("task_id", "child_task_id", "parent_task_id"):
                value = str(row.get(key) or "").strip()
                if value:
                    ledger_ids.add(value)
    except Exception:
        pass
    unknown: List[str] = []
    for tid in task_ids:
        try:
            if load_effective_task_result(status_drive_root, tid):
                continue
            queue_status, _ = _queue_task_status(snapshot, tid)
            if queue_status:  # running/scheduled row, or "unknown" on a missing snapshot (fail-soft)
                continue
            if tid in ledger_ids:
                continue
        except Exception:
            continue  # unreadable surface: treat as known
        unknown.append(tid)
    return unknown


def _children_roster_projection(
    ctx: ToolContext, status_drive_root: Path, *, limit: int = 30,
) -> Dict[str, Any]:
    """This parent's DIRECT children in the v6.71.2 compact field set (task_id/
    status/accounted_upper_bound_usd/sha/outcome_axes, ABI-3 honest name) —
    never result envelopes; missing
    accounting projects null, never a confirmed-looking $0. The bound is
    DISCLOSED through the shared ``disclosed_list_projection`` (BIBLE P1): the
    payload carries ``children_roster`` plus ``children_roster_omitted``, the
    exact count of real children the cap hid — a silent ``[:limit]`` here could
    hide the very replacement id this repair surface exists to show. Fail-soft:
    an empty roster with omitted=0."""
    from ouroboros._outcome_receipts import disclosed_list_projection
    from ouroboros.task_status import find_child_tasks
    from ouroboros.tools.join_ledger import _child_result_sha256

    empty = {"children_roster": [], "children_roster_omitted": 0}
    my_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not my_id:
        return empty
    try:
        rows = find_child_tasks(
            status_drive_root, parent_task_id=my_id, root_task_id="",
            exclude_task_id=my_id, scope="direct",
        )
    except Exception:
        return empty
    from ouroboros.cost_projection import cost_projection

    roster: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        _cost = cost_projection(row)
        roster.append({
            "task_id": str(row.get("task_id") or row.get("id") or ""),
            "status": row.get("status"),
            "accounted_upper_bound_usd": _cost["accounted_upper_bound_usd"],
            "child_result_sha256": _child_result_sha256(row),
            "outcome_axes": normalize_outcome_axes(row),
        })
    return disclosed_list_projection(
        roster, key="children_roster", limit=max(1, int(limit)), item=lambda entry: entry,
    )


def _wait_for_tasks(
    ctx: ToolContext,
    task_ids: List[str],
    timeout_sec: int = 600,
    mode: str = "all_terminal",
) -> str:
    """Wait for multiple subtasks and return a compact structural projection per child.

    A wait set whose ids were ALL unminted at entry ends after the registration
    grace instead of the full requested window (disclosed as
    ``wait_short_circuited``); any id that turns real during the grace makes it
    an ordinary wait again, with the remaining window intact."""
    if not isinstance(task_ids, list) or not task_ids:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text="⚠️ TOOL_ARG_ERROR (wait_tasks): task_ids must be a non-empty list.",
        ))
    from ouroboros.config import MAX_ACTIVE_SUBAGENTS_HARD_CAP
    from ouroboros.cost_projection import cost_projection

    if len(task_ids) > MAX_ACTIVE_SUBAGENTS_HARD_CAP:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text=(
                "⚠️ TOOL_ARG_ERROR (wait_tasks): task_ids is capped at "
                f"{MAX_ACTIVE_SUBAGENTS_HARD_CAP}."
            ),
        ))
    normalized_ids: List[str] = []
    for item in task_ids:
        try:
            tid = validate_task_id(item)
        except ValueError as exc:
            return _publish_tool_result(ctx, ToolResult(
                status="error", code="TOOL_ARG_ERROR",
                text=f"⚠️ TOOL_ARG_ERROR (wait_tasks): {exc}",
            ))
        if tid not in normalized_ids:
            normalized_ids.append(tid)
    try:
        timeout = max(0, min(int(timeout_sec), 7200))
    except (TypeError, ValueError):
        timeout = 600
    normalized_mode = str(mode or "all_terminal").strip().lower()
    if normalized_mode not in {"all_terminal", "any_terminal"}:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text="⚠️ TOOL_ARG_ERROR (wait_tasks): mode must be all_terminal or any_terminal.",
        ))
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    status_drive_root = Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    # Typed unknown-id detection (v6.91): flagged ids KEEP polling — "not YET
    # registered" is a real state for a just-scheduled child — but a phantom id
    # is disclosed instead of silently starving the wait (wave2: three
    # hallucinated ids blocked 900s slices while the real lead went unwaited).
    entry_unknown_ids = _unminted_wait_ids(ctx, status_drive_root, normalized_ids)
    # One beacon cursor for the whole wait, so a two-phase window cannot skip an
    # attention beacon emitted during its first phase.
    _wait_since = ""
    # A wait set in which EVERY id is unminted cannot be satisfied by waiting —
    # nothing was ever scheduled to terminate. Spend only the registration-race
    # grace on it (wave1's root blocked its whole window on three hallucinated
    # ids), then re-probe; the moment any id turns real this becomes an ordinary
    # wait and gets the rest of the requested window.
    _phantom_only = bool(entry_unknown_ids) and len(entry_unknown_ids) == len(normalized_ids)
    first_window = min(float(timeout), _UNMINTED_WAIT_GRACE_SEC) if _phantom_only else float(timeout)
    waited = wait_for_effective_tasks(
        status_drive_root, normalized_ids, timeout_sec=first_window, mode=normalized_mode,
        on_poll=_wait_attention_poll(ctx, _wait_since, normalized_ids), poll_interval_sec=2.0,
    )
    if _phantom_only and first_window < float(timeout) and waited.get("early_return") is None:
        entry_unknown_ids = _unminted_wait_ids(ctx, status_drive_root, normalized_ids)
        if len(entry_unknown_ids) < len(normalized_ids):
            elapsed = float(waited.get("elapsed_sec") or 0.0)
            resumed = wait_for_effective_tasks(
                status_drive_root, normalized_ids,
                timeout_sec=max(0.0, float(timeout) - elapsed), mode=normalized_mode,
                on_poll=_wait_attention_poll(ctx, _wait_since, normalized_ids), poll_interval_sec=2.0,
            )
            resumed["elapsed_sec"] = float(resumed.get("elapsed_sec") or 0.0) + elapsed
            resumed["timeout_sec"] = float(timeout)
            waited = resumed
        else:
            # Disclosed, not silent: the wait ended early and says why.
            waited["wait_short_circuited"] = {
                "reason": "all_task_ids_unminted",
                "requested_timeout_sec": float(timeout),
                "waited_sec": round(float(waited.get("elapsed_sec") or 0.0), 1),
                "note": (
                    "Every requested task_id was unminted at entry and still unminted after "
                    f"the {int(_UNMINTED_WAIT_GRACE_SEC)}s registration grace, so the wait "
                    "returned instead of blocking for the full timeout. Fix the wait set from "
                    "children_roster / your schedule_subagent results, then wait again."
                ),
            }
    tasks = waited.get("tasks")
    if isinstance(tasks, dict):
        from ouroboros.tools.join_ledger import _child_result_sha256

        # Re-probe the entry-time unknowns once: an id minted mid-wait (queue
        # row or result appeared) is a real child, not a phantom.
        unknown_ids = [tid for tid in entry_unknown_ids if not tasks.get(tid)]
        if unknown_ids:
            unknown_ids = _unminted_wait_ids(ctx, status_drive_root, unknown_ids)

        # Compact STRUCTURAL projection (v6.71.2): the full public_task_result
        # envelope duplicated forensics (trace_refs, loop_outcome internals,
        # verification_ledger) into the parent context on every batch absorb.
        # The parent decision needs the semantic handoff only; the full envelope
        # stays on disk in task_results/<id>.json, addressable by
        # child_result_sha256 (the join-ledger SSOT hash), and is fetched with
        # get_task_result — a DISCLOSED omission (BIBLE P1), not silent
        # truncation. Single-task wait_task/get_task_result stay full.
        public_tasks: Dict[str, Any] = {}
        for tid, data in tasks.items():
            if str(tid) in unknown_ids:
                public_tasks[str(tid)] = {
                    "task_id": str(tid),
                    "status": None,
                    "unknown_task_id": True,
                    "note": (
                        "UNKNOWN_TASK_ID: not yet registered or never scheduled — no task "
                        "result, no queue row, and no tree-ledger row names this id in this "
                        "tree. Check it against your schedule_subagent results / the "
                        "children_roster below; an all_terminal wait cannot complete while "
                        "it stays unscheduled."
                    ),
                }
                continue
            if not isinstance(data, dict):
                public_tasks[str(tid)] = data
                continue
            # SSOT cost projection (C2/ABI-3): honest null (never a
            # confirmed-looking $0), the honest name only, and finality
            # only when the child's own record claims it.
            _cost = cost_projection(data)
            projected: Dict[str, Any] = {
                "task_id": str(data.get("task_id") or data.get("id") or tid),
                "status": data.get("status"),
                "accounted_upper_bound_usd": _cost["accounted_upper_bound_usd"],
                "cost_final": _cost["cost_final"],
                "child_result_sha256": _child_result_sha256(data),
                "outcome_axes": normalize_outcome_axes(data),
                "result": data.get("result"),
                "trace_summary": data.get("trace_summary"),
            }
            if data.get("duplicate_of"):
                projected["duplicate_of"] = str(data.get("duplicate_of"))
            # A capability reduction is a SEMANTIC handoff fact, not forensics: it is
            # what decides how far to trust this answer, and this is the surface a
            # fan-out parent absorbs its children through. Same predicate as the
            # single-child read, so the batch and the singleton cannot disagree.
            _delta = disclosable_capability_delta(data)
            if _delta:
                projected["capability_delta"] = _delta
            # Delegation honesty (Q1A, 2026-08-10 amendments): whether a
            # harness-dispatched child ACTUALLY delegated is a handoff fact the
            # fan-out parent absorbs here — the e9108a09 incident hid nine
            # native-only "harness" children behind this very projection.
            # Compact counts only; the full evidence stays in the envelope.
            _envelope = data.get("subagent_envelope") if isinstance(data.get("subagent_envelope"), dict) else {}
            _evidence = _envelope.get("execution_evidence") if isinstance(_envelope.get("execution_evidence"), dict) else {}
            if _evidence or str(data.get("effective_executor") or "") == "harness":
                _ee: Dict[str, Any] = {
                    "dispatch_executor": str(data.get("effective_executor") or ""),
                }
                if _evidence.get("evidence_read_failed"):
                    # Unreadable custody log (v6.94.0 landing-gate scope fix):
                    # the counts are UNKNOWN — emitting them as 0 beside the
                    # marker fabricated a "no runs" receipt for a log that was
                    # never read. The compact projection carries ONLY the typed
                    # marker; counts AND the substrate claim are omitted, the
                    # same omission rule subagents.envelope_from_task applies.
                    _ee["evidence_read_failed"] = True
                else:
                    if _evidence:
                        # Counts only when the envelope actually attested them:
                        # a result with no evidence recorded (pre-6.94) gets NO
                        # zero counts — absence means "no evidence yet", not
                        # "no runs".
                        _ee["delegated_runs_started"] = int(_evidence.get("delegated_runs_started") or 0)
                        _ee["delegated_runs_settled"] = int(_evidence.get("delegated_runs_settled") or 0)
                        _ee["delegated_runs_succeeded"] = int(_evidence.get("delegated_runs_succeeded") or 0)
                        _ee["delegated_runs_failed"] = int(_evidence.get("delegated_runs_failed") or 0)
                        _ee["delegated_runs_source_unresolved"] = int(
                            _evidence.get("delegated_runs_source_unresolved") or 0
                        )
                    # The substrate claim rides only when the envelope made one.
                    _substrate = str(data.get("actual_substrate") or _envelope.get("actual_substrate") or "")
                    if _substrate:
                        _ee["actual_substrate"] = _substrate
                        # C3: counters are delegated-run facts; the native
                        # (metered) contribution beside them is unknown.
                        _ee["native_contribution"] = "unknown"
                projected["execution_evidence"] = _ee
            public_tasks[str(tid)] = projected
        waited["tasks"] = public_tasks
        waited["tasks_note"] = (
            "Compact per-child projection. The full result envelope (trace_refs, "
            "loop_outcome, verification_ledger) remains on disk in task_results/"
            "<task_id>.json, addressable by child_result_sha256; get_task_result "
            "returns the full result text plus trace/outcome summaries."
        )
        if unknown_ids:
            waited["unknown_task_ids"] = unknown_ids
            # The repair surface: the ACTUAL direct children, compact v6.71.2
            # field set only (never envelopes), so the parent can fix its wait
            # set instead of re-polling phantoms. Carries children_roster plus
            # the disclosed children_roster_omitted count (never a silent cap).
            waited.update(_children_roster_projection(ctx, status_drive_root))
    horizon_note = cache_horizon_note(ctx, waited.get("elapsed_sec"))
    if horizon_note:
        waited["cache_horizon_note"] = horizon_note
    return json.dumps(waited, ensure_ascii=False, indent=2)

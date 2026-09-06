"""D#7 soft join-ledger tools (extracted from control.py to keep it under the module
size gate). The parent's explicit, structured (P5 — not parsed-from-prose) controls for
not orphaning spawned subagent children:

  - peek_task: inspect a child's status / latest beacons / result tail (a PURE READ —
    makes no finalization decision and does not alter the change-based handoff reminder).
  - discard_child_result: explicitly abandon a child's exact result through the
    shared typed task-tree decision ledger, lineage-gated to OWN children.

The shared lineage/ledger helpers (_status_drive_root, _is_own_child,
_record_child_decision_beacon) and the cancel_task handler (moved here from control.py,
upgraded with a recorded reason + lineage gate) live here too.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

from ouroboros.task_results import validate_task_id
from ouroboros.task_status import load_effective_task_result, observe_cancellation_target
from ouroboros.task_tree_ledger import (
    CHILD_RESULT_DISPOSITIONS,
    CHILD_RESULT_DISPOSITION_TYPE,
    child_result_disposition_row,
    child_result_disposition_violations,
    normalize_child_result_disposition_payload,
    tree_ledger_append,
)
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import utc_now_iso

log = logging.getLogger("ouroboros.tools.join_ledger")

_ARTIFACT_IDENTITY_FIELDS = (
    "id",
    "artifact_id",
    "kind",
    "name",
    "relpath",
    "sha256",
    "status",
)


def _stable_artifact_identities(result: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return stable artifact identities without volatile paths/timestamps."""

    candidates: list[Any] = []
    if isinstance(result.get("artifacts"), list):
        candidates.extend(result.get("artifacts") or [])
    bundle = result.get("artifact_bundle") if isinstance(result.get("artifact_bundle"), dict) else {}
    if isinstance(bundle.get("artifacts"), list):
        candidates.extend(bundle.get("artifacts") or [])
    identities: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        if isinstance(item, str) and item.strip():
            item = {"path": item}
        if not isinstance(item, dict):
            continue
        identity = {
            key: item.get(key)
            for key in _ARTIFACT_IDENTITY_FIELDS
            if item.get(key) not in (None, "")
        }
        raw_path = str(item.get("path") or item.get("abs_path") or "").strip()
        if raw_path:
            artifact_path = Path(raw_path)
            if not artifact_path.is_absolute():
                identity.setdefault("relpath", artifact_path.as_posix())
            if "name" not in identity:
                identity["name"] = Path(raw_path).name
        if not identity:
            continue
        encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)
        if encoded in seen:
            continue
        seen.add(encoded)
        identities.append(identity)
    return sorted(
        identities,
        key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, default=str),
    )


def _child_result_sha256(result: Dict[str, Any]) -> str:
    """Hash the exact semantic child result consumed by a parent decision.

    Cost, timestamps, queue diagnostics, and parent-decision fields are omitted by
    construction. A content/status/artifact change therefore invalidates a prior
    disposition, while accounting or coordination telemetry does not.
    """

    semantic_result = result
    bundle = (
        semantic_result.get("artifact_bundle")
        if isinstance(semantic_result.get("artifact_bundle"), dict)
        else {}
    )
    payload = {
        "status": str(semantic_result.get("status") or ""),
        "result": semantic_result.get("result"),
        "trace_summary": semantic_result.get("trace_summary"),
        "artifact_status": str(semantic_result.get("artifact_status") or bundle.get("status") or ""),
        "artifacts": _stable_artifact_identities(semantic_result),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _child_disposition_lineage(result: Dict[str, Any]) -> tuple[str, str, str]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    child_task_id = str(result.get("task_id") or result.get("id") or "").strip()
    parent_task_id = str(
        result.get("parent_task_id") or metadata.get("parent_task_id") or ""
    ).strip()
    root_task_id = str(
        result.get("root_task_id") or metadata.get("root_task_id") or parent_task_id
    ).strip()
    return root_task_id, parent_task_id, child_task_id


def _current_child_result_disposition(result: Dict[str, Any]) -> str:
    """Return the exact-hash disposition from the derived task-tree projection."""

    if str(result.get("child_result_disposition_source") or "") != "task_tree_ledger":
        return ""
    disposition = str(result.get("child_result_disposition") or "").strip().lower()
    expected = str(result.get("child_result_disposition_sha256") or "").strip().lower()
    if disposition not in CHILD_RESULT_DISPOSITIONS or len(expected) != 64:
        return ""
    return disposition if expected == _child_result_sha256(result) else ""


def _record_child_result_disposition(
    ctx: ToolContext,
    payload: Dict[str, Any],
    rationale: str,
) -> str:
    """Append the sole authoritative, exact-hash child disposition row."""

    # Aggregated diagnostics (W2): name EVERY violated constraint in ONE reply —
    # the old one-error-per-round shape cost a live parent 9 paid rounds. The
    # typed exact-hash authority is unchanged: no truncation on the caller's
    # behalf, no superset keys accepted, malformed = atomic no-op.
    problems = child_result_disposition_violations(payload)
    reason_text = " ".join(str(rationale or "").split())
    if not reason_text:
        problems.append("tree_note text is required as the rationale")
    elif len(reason_text) > 500:
        problems.append(
            f"tree_note rationale must be at most 500 characters (got {len(reason_text)}; "
            "shorten it yourself — it is never truncated for you)"
        )
    if problems:
        return (
            "⚠️ CHILD_RESULT_DISPOSITION_INVALID: " + "; ".join(problems) + ". "
            "Correct example: tree_note(kind='decision', text='<short why, ≤500 chars>', "
            "payload={'type': 'child_result_disposition', 'child_task_id': '<id>', "
            "'disposition': 'integrated'|'irrelevant'|'deferred', "
            "'child_result_sha256': '<the 64-hex sha from [SUBTASK_OUTCOME]/get_task_result>'})."
            " Nothing was recorded (atomic no-op)."
        )
    normalized = normalize_child_result_disposition_payload(payload)
    if normalized is None:  # unreachable: violations above are the same authority
        return "⚠️ CHILD_RESULT_DISPOSITION_INVALID: payload failed normalization."
    tid = normalized["child_task_id"]
    disposition = normalized["disposition"]
    expected = normalized["child_result_sha256"]

    status_drive_root = _status_drive_root(ctx)
    if not _is_own_child(ctx, status_drive_root, tid):
        return (
            f"⚠️ CHILD_RESULT_LINEAGE_FORBIDDEN: {tid} is not a direct child of this task; "
            "no disposition was recorded."
        )
    data = load_effective_task_result(status_drive_root, tid) or {}
    if not data:
        return f"⚠️ CHILD_RESULT_STALE: {tid} has no current result to bind."
    actual = _child_result_sha256(data)
    if actual != expected:
        return (
            f"⚠️ CHILD_RESULT_STALE: {tid} changed (expected {expected[:12]}, current "
            f"{actual[:12]}); inspect it again and submit the current hash."
        )

    from ouroboros.tools.task_tree import tree_root_id

    root_task_id = tree_root_id(ctx)
    parent_task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if _child_disposition_lineage(data) != (root_task_id, parent_task_id, tid):
        return (
            f"⚠️ CHILD_RESULT_LINEAGE_FORBIDDEN: {tid} does not carry the exact "
            "root/parent lineage required for a tree-ledger disposition."
        )

    existing = child_result_disposition_row(
        root_task_id,
        parent_task_id,
        tid,
        expected,
        data_root=status_drive_root,
    )
    if (
        existing
        and existing.get("payload") == normalized
        and str(existing.get("text") or "") == reason_text
    ):
        return (
            f"OK: child {tid} is already marked {disposition} for result "
            f"{expected[:12]} (idempotent)."
        )

    metadata = (
        getattr(ctx, "task_metadata", {})
        if isinstance(getattr(ctx, "task_metadata", {}), dict)
        else {}
    )
    role = str(metadata.get("role") or getattr(ctx, "role", "") or "")
    appended = tree_ledger_append(
        root_task_id,
        "decision",
        reason_text,
        task_id=parent_task_id,
        role=role,
        payload=normalized,
        allow_child_result_disposition=True,
        data_root=status_drive_root,
    )
    if not appended.startswith("OK:"):
        return (
            f"⚠️ CHILD_RESULT_DISPOSITION_WRITE_FAILED: failed to append the "
            f"authoritative task-tree decision for {tid}."
        )

    current = load_effective_task_result(status_drive_root, tid) or {}
    if _child_result_sha256(current) != expected:
        return (
            f"⚠️ CHILD_RESULT_STALE: {tid} changed while its disposition was recorded; "
            "the old row remains audit evidence but does not close the new result."
        )
    if _child_disposition_lineage(current) != (root_task_id, parent_task_id, tid):
        return (
            f"⚠️ CHILD_RESULT_LINEAGE_FORBIDDEN: {tid} lineage changed while its "
            "disposition was recorded."
        )
    authoritative = child_result_disposition_row(
        root_task_id,
        parent_task_id,
        tid,
        expected,
        data_root=status_drive_root,
    )
    if (
        authoritative.get("payload") != normalized
        or str(authoritative.get("text") or "") != reason_text
    ):
        return (
            f"⚠️ CHILD_RESULT_DISPOSITION_WRITE_FAILED: a later decision replaced "
            f"the disposition for {tid}; this caller did not overwrite it."
        )
    return f"OK: child {tid} marked {disposition} for result {expected[:12]}."


def _record_child_result_disposition_batch(
    ctx: ToolContext,
    payload: Dict[str, Any],
    rationale: str,
) -> str:
    """Record dispositions for MANY children in ONE tree_note call.

    Each ``children`` entry is validated and recorded exactly like the single
    form (same exact-hash binding, lineage gates, and idempotency — the batch
    expands into the same individual authoritative ledger rows, so every
    existing reader is unchanged). Entries are independent: an invalid entry is
    rejected with a clear per-entry error naming it, while valid entries still
    record. The shared tree_note text is the rationale for every entry.
    """

    envelope_extra = sorted(set(payload) - {"type", "children"})
    children = payload.get("children")
    if envelope_extra or not isinstance(children, list) or not children:
        return (
            "⚠️ CHILD_RESULT_DISPOSITION_INVALID: the batch form is exactly "
            "{'type': 'child_result_disposition', 'children': [{'child_task_id', "
            "'disposition', 'child_result_sha256'}, ...]} with a non-empty array"
            + (f" (unknown key(s): {', '.join(envelope_extra)})" if envelope_extra else "")
            + ". Nothing was recorded (atomic no-op)."
        )
    lines: list[str] = []
    recorded = 0
    for index, entry in enumerate(children):
        if not isinstance(entry, dict):
            lines.append(f"[entry {index}] ⚠️ CHILD_RESULT_DISPOSITION_INVALID: entry must be a JSON object.")
            continue
        single = dict(entry)
        single.setdefault("type", CHILD_RESULT_DISPOSITION_TYPE)
        label = str(entry.get("child_task_id") or f"entry {index}")
        outcome = _record_child_result_disposition(ctx, single, rationale)
        if outcome.startswith("OK:"):
            recorded += 1
        lines.append(f"[{label}] {outcome}")
    total = len(children)
    if recorded == total:
        header = f"OK: batch child disposition recorded for {recorded} child(ren)."
    elif recorded:
        header = (
            f"⚠️ CHILD_RESULT_DISPOSITION_PARTIAL: {recorded}/{total} entries recorded; "
            "the failed entries below were rejected individually and must be corrected."
        )
    else:
        header = (
            f"⚠️ CHILD_RESULT_DISPOSITION_INVALID: 0/{total} batch entries were recorded."
        )
    return header + "\n" + "\n".join(lines)


def _record_current_child_result_disposition(
    ctx: ToolContext,
    child_task_id: str,
    disposition: str,
    rationale: str,
) -> str:
    """Bind an operation that genuinely consumed or rejected the current result."""

    try:
        tid = validate_task_id(child_task_id)
    except ValueError as exc:
        return f"⚠️ CHILD_RESULT_DISPOSITION_INVALID: {exc}"
    data = load_effective_task_result(_status_drive_root(ctx), tid) or {}
    if not data:
        return f"⚠️ CHILD_RESULT_STALE: {tid} has no current result to bind."
    return _record_child_result_disposition(
        ctx,
        {
            "type": CHILD_RESULT_DISPOSITION_TYPE,
            "child_task_id": tid,
            "disposition": disposition,
            "child_result_sha256": _child_result_sha256(data),
        },
        rationale,
    )


def _record_child_decision_beacon(ctx: ToolContext, task_id: str, text: str) -> None:
    """Record a parent coordination decision about a child on the task-tree ledger
    (D#7) so the decision is durable + visible across the tree. Best-effort."""
    try:
        from ouroboros.tools.task_tree import tree_root_id

        rid = tree_root_id(ctx)
        if not rid:
            return
        from ouroboros.task_tree_ledger import tree_ledger_append

        meta = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
        role = str(meta.get("role") or getattr(ctx, "role", "") or "")
        tree_ledger_append(rid, "decision", text, task_id=str(task_id or ""), role=role)
    except Exception:
        log.debug("Failed to record child decision beacon for %s", task_id, exc_info=True)


def _status_drive_root(ctx: ToolContext) -> Path:
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    return Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or ctx.drive_root))


def _is_own_child(ctx: ToolContext, status_drive_root: Path, tid: str) -> bool:
    """True if ``tid`` is a DIRECT child of the CURRENT task (D#7 safety): a parent
    decision may only describe the caller's OWN children, never an unrelated parent's
    join ledger. Fail-CLOSED — any error returns False."""
    try:
        from ouroboros.task_status import find_child_tasks

        meta = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
        my_id = str(getattr(ctx, "task_id", "") or meta.get("task_id") or "")
        if not my_id or not tid:
            return False
        children = find_child_tasks(
            Path(status_drive_root), parent_task_id=my_id, root_task_id="", exclude_task_id=my_id
        )
        return any(str(c.get("task_id") or c.get("id") or "") == tid for c in children)
    except Exception:
        return False


def _is_delegated_task(ctx: ToolContext) -> bool:
    """Whether the current caller has a delegated-child tool profile."""

    try:
        from ouroboros.tool_access import active_tool_profile

        return active_tool_profile(ctx) in {
            "acting_subagent", "local_readonly_subagent",
        }
    except Exception:
        # Preserve the descendant-only authority even if profile projection
        # itself fails: a delegated lineage marker may narrow this caller but
        # must never widen it to arbitrary task inspection/control.
        metadata = (
            getattr(ctx, "task_metadata", {})
            if isinstance(getattr(ctx, "task_metadata", {}), dict)
            else {}
        )
        constraint = getattr(ctx, "task_constraint", None)
        return (
            str(getattr(constraint, "mode", "") or "") in {
                "acting_subagent", "local_readonly_subagent",
            }
            or str(metadata.get("delegation_role") or "") == "subagent"
        )


def _clip(text: object, limit: int, *, tail: bool = False) -> str:
    """Truncate to ``limit`` chars with an EXPLICIT omission marker so a peek never
    silently drops cognitive content (P1 — no silent horizon cut; the agent can then
    get_task_result the full body if it needs the omitted part)."""
    s = str(text or "")
    if len(s) <= limit:
        return s
    omitted = len(s) - limit
    if tail:
        return f"…(+{omitted} earlier chars omitted — get_task_result for the full body)\n{s[-limit:]}"
    return f"{s[:limit]}…(+{omitted} more chars omitted)"


def _peek_task(ctx: ToolContext, task_id: str, view: str = "summary") -> str:
    """Read a child's CURRENT status + latest coordination beacons + result tail (D#7 — the
    parent's 'see intermediate findings' right). A PURE READ: it changes no state. The
    pre-finalization handoff reminder is CHANGE-BASED (it re-surfaces whenever a child's
    status/result changes and is suppressed only by an explicit discard_child_result /
    cancel_task, or by being unchanged since last shown) — peeking neither suppresses nor
    re-triggers it. view: summary | partials | tail."""
    try:
        tid = validate_task_id(task_id)
    except ValueError as exc:
        return f"⚠️ TOOL_ARG_ERROR (peek_task): {exc}"
    v = str(view or "summary").strip().lower()
    status_drive_root = _status_drive_root(ctx)
    if _is_delegated_task(ctx) and not _is_own_child(ctx, status_drive_root, tid):
        return (
            f"⚠️ peek_task: {tid} is not a child of this task — a delegated task "
            "may inspect only its own children."
        )
    data = load_effective_task_result(status_drive_root, tid) or {}
    status = str(data.get("status") or "unknown")
    # SSOT cost projection (C2): a missing/unknown cost says "unknown", never a
    # confident $0.00, and an open amount is labelled as the upper bound it is.
    from ouroboros.cost_projection import cost_display

    parts = [
        f"Task {tid} [{status}] cost={cost_display(data)} (peek — NOT absorbed)",
        f"child_result_sha256={_child_result_sha256(data)}",
    ]
    # Latest beacons this child posted to the shared ledger (partial_finding / blocker /
    # question / milestone), newest last.
    try:
        from ouroboros.tools.task_tree import tree_root_id
        from ouroboros.task_tree_ledger import tree_ledger_rows

        rid = tree_root_id(ctx)
        if rid:
            rows = [r for r in tree_ledger_rows(rid) if str(r.get("task_id") or "") == tid]
            if v in ("partials", "summary"):
                beacons = [
                    r for r in rows
                    if str(r.get("kind")) in (
                        "partial_finding", "blocker", "question", "milestone",
                        "interface_contract", "review_requested",
                    )
                ]
                if len(beacons) > 8:
                    parts.append(f"  …(+{len(beacons) - 8} older beacon(s) omitted; showing newest 8)")
                for r in beacons[-8:]:
                    detail = ""
                    if str(r.get("kind") or "") == "review_requested":
                        payload = r.get("payload") if isinstance(r.get("payload"), dict) else {}
                        detail = (
                            f" evidence_ref={_clip(payload.get('evidence_ref'), 1000)}"
                            f" evidence_sha256={str(payload.get('evidence_sha256') or '')}"
                        )
                    parts.append(
                        f"  • [{r.get('kind')}] {_clip(r.get('text'), 400)}{detail}"
                    )
    except Exception:
        log.debug("peek_task ledger read failed for %s", tid, exc_info=True)
    if v in ("tail", "summary"):
        result = str(data.get("result") or "")
        if result:
            parts.append(f"[PEEK_RESULT_TAIL]\n{_clip(result, 1200, tail=True)}\n[/PEEK_RESULT_TAIL]")
    trace = str(data.get("trace_summary") or "")
    if trace and v == "tail":
        parts.append(f"[PEEK_TRACE]\n{_clip(trace, 800)}\n[/PEEK_TRACE]")
    return "\n".join(parts)


def _discard_child_result(ctx: ToolContext, task_id: str, reason: str) -> str:
    """Explicitly decide to ABANDON a child's result (D#7). This is the EXPLICIT,
    structured signal (P5 — not a parsed-from-prose phrase) that lets the parent finalize
    without that child: it records an exact-hash ``irrelevant`` decision in the shared
    task-tree ledger. A reason is REQUIRED so the abandon is an auditable judgment, not a
    silent loss. The raw child result is not mutated. Lineage-gated to OWN children."""
    try:
        tid = validate_task_id(task_id)
    except ValueError as exc:
        return f"⚠️ TOOL_ARG_ERROR (discard_child_result): {exc}"
    reason_text = _clip(" ".join(str(reason or "").split()), 500)
    if not reason_text:
        return "⚠️ TOOL_ARG_ERROR (discard_child_result): a non-empty reason is required."
    status_drive_root = _status_drive_root(ctx)
    # D#7 safety: a parent may abandon only its OWN child's result.
    if not _is_own_child(ctx, status_drive_root, tid):
        return f"⚠️ discard_child_result: {tid} is not a child of this task — refusing to discard."
    recorded = _record_current_child_result_disposition(
        ctx,
        tid,
        "irrelevant",
        reason_text,
    )
    if not recorded.startswith("OK:"):
        return recorded.replace("CHILD_RESULT_DISPOSITION", "discard_child_result")
    return f"Discarded child result {tid} (reason: {reason_text}). It will not block finalization."


def _override_delegation_constraint(ctx: ToolContext, constraint_id: str, reason: str) -> str:
    """Explicitly override an unresolved delegation constraint in this task tree."""

    cid = " ".join(str(constraint_id or "").split())
    if not cid:
        return "⚠️ TOOL_ARG_ERROR (override_delegation_constraint): constraint_id is required."
    reason_text = _clip(" ".join(str(reason or "").split()), 500)
    if not reason_text:
        return "⚠️ TOOL_ARG_ERROR (override_delegation_constraint): a non-empty reason is required."
    try:
        from ouroboros.tools.task_tree import tree_root_id
        from ouroboros.task_tree_ledger import open_delegation_constraints, tree_ledger_append

        rid = tree_root_id(ctx)
        if not rid:
            return "⚠️ override_delegation_constraint: no task-tree scope."
        open_rows = open_delegation_constraints(rid)
        target_row = next((
            row for row in open_rows
            if isinstance(row.get("payload"), dict)
            and str(row["payload"].get("constraint_id") or "") == cid
        ), None)
        if target_row is None:
            return f"⚠️ override_delegation_constraint: constraint {cid!r} is not open in this task tree."
        emitter_task_id = str(target_row.get("task_id") or "").strip()
        if emitter_task_id:
            status_drive_root = _status_drive_root(ctx)
            if not _is_own_child(ctx, status_drive_root, emitter_task_id):
                return (
                    "⚠️ override_delegation_constraint: only the parent of the task that raised "
                    f"constraint {cid!r} may override it."
                )
        meta = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
        role = str(meta.get("role") or getattr(ctx, "role", "") or "")
        return tree_ledger_append(
            rid,
            "decision",
            f"overrode delegation constraint {cid}: {reason_text}",
            task_id=str(getattr(ctx, "task_id", "") or ""),
            role=role,
            allow_constraint_override=True,
            payload={
                "constraint_id": cid,
                "decision": "overridden",
                "reason": reason_text,
                "parent_task_id": str(getattr(ctx, "task_id", "") or ""),
            },
        )
    except Exception:
        log.debug("Failed to override delegation constraint %s", cid, exc_info=True)
        return f"⚠️ override_delegation_constraint: failed to record override for {cid}."


def _cancel_task(ctx: ToolContext, task_id: str, reason: str = "") -> str:
    try:
        tid = validate_task_id(task_id)
    except ValueError as exc:
        return f"⚠️ TOOL_ARG_ERROR (cancel_task): {exc}"
    reason_text = _clip(" ".join(str(reason or "").split()), 500)
    status_drive_root = _status_drive_root(ctx)
    # Only stamp the join-ledger parent_decision (+ post to the tree ledger) when the
    # target is THIS task's own child — a cancel must not rewrite an unrelated task's
    # parent_decision and hide it from its real parent's reminder (D#7 safety).
    own = _is_own_child(ctx, status_drive_root, tid)
    # Subagent isolation: a delegated child may cancel only its own children.
    # Project focus does not narrow an ordinary top-level principal; workspace
    # parents keep the same task-control authority as other top-level tasks.
    if not own and _is_delegated_task(ctx):
        return f"⚠️ cancel_task: {tid} is not a child of this task — a delegated task may only cancel its own children."
    # Durable cancel intent — the ONE ingress (phase A, owner batch-4 1=A). The
    # canonical status never carries intent: the supervisor's cancellation
    # custody claims this intent, tears the task down, and settles the terminal
    # outcome (writing parent_decision only at that OUTCOME — never here, so a
    # child that finishes before the kill keeps its completed result; completion
    # wins, and discarding a kept result stays a separate explicit action).
    intent: Dict[str, Any] = {}
    # GR6-1 live-ownership check at the ingress: the durable terminal result
    # is persisted BEFORE post-task cognition ends, so a settled STATUS with a
    # live RUNNING row means a worker still spending — the mint must not
    # no-op as ``already_settled`` over it. Worker-side, the queue snapshot is
    # the ownership projection (the live maps belong to the supervisor).
    live_ownership = False
    try:
        from ouroboros.cancel_intents import _validated_single_cancel_target
        from ouroboros.task_status import task_has_live_queue_ownership

        live_ownership = task_has_live_queue_ownership(status_drive_root, tid)
        # This pre-read is only a fail-open liveness hint. request_cancel
        # resolves the target again under the projection lock before minting.
        retry_hint = _validated_single_cancel_target(status_drive_root, tid)
        if retry_hint != tid:
            live_ownership = live_ownership or task_has_live_queue_ownership(
                status_drive_root, retry_hint,
            )
    except Exception:
        live_ownership = True
        log.debug("cancel_task live-ownership read failed for %s", tid, exc_info=True)
    try:
        from ouroboros.cancel_intents import CancelIntentProjectionCorrupt, request_cancel

        observation = observe_cancellation_target(status_drive_root, tid, include_execution=True,
            request_origin={"kind": "agent_task", "task_id": str(getattr(ctx, "task_id", "") or "")})
        intent = request_cancel(
            status_drive_root,
            tid,
            reason=reason_text,
            source="agent_tool",
            requested_by=str(getattr(ctx, "task_id", "") or "") if own else "",
            allow_settled_target=live_ownership,
            observation=observation,
        )
    except CancelIntentProjectionCorrupt:
        # GR4-8: a corrupt projection is not a transient — "retry" cannot
        # succeed until the file is repaired. The malformed file was preserved
        # (never overwritten) and a projection_corrupt_refused forensic row was
        # recorded in logs/supervisor.jsonl.
        log.error("cancel_task refused for %s: intent projection corrupt", tid)
        return (
            f"⚠️ CANCEL_INTENT_PROJECTION_CORRUPT: the cancel-intent projection "
            f"(state/cancel_intents.json) is corrupt; nothing was cancelled for {tid} "
            "and retrying cannot succeed until the file is repaired. The malformed "
            "file was preserved (no overwrite) and a projection_corrupt_refused "
            "forensic row was recorded in logs/supervisor.jsonl."
        )
    except Exception:
        log.debug("Failed to record durable cancel intent for %s", tid, exc_info=True)
        return (
            f"⚠️ CANCEL_INTENT_WRITE_FAILED: durable cancel intent for {tid} could not "
            "be recorded; nothing was cancelled. Retry, or report the failure."
        )
    observation["matches_cancel_target"] = observation.get("observed_task_id") == str(intent.get("task_id") or tid)
    observed_note = ("\n[OBSERVED_TARGET]\n" + json.dumps(observation, ensure_ascii=False)
                     + "\n[/OBSERVED_TARGET]\nThese source observations are separate from the caller's reason; "
                     "missing or stale facts do not prove that no work or spending occurred.")
    if intent.get("already_settled"):
        # Completion wins (owner 4=A): nothing to tear down, and minting an intent
        # for a settled task would show a false "Cancelling…" state on a finished
        # card until the watchdog cleaned it up.
        return (
            f"Nothing to cancel: {tid} had already finished ({intent.get('status')}). "
            "Its result is preserved — use discard_child_result if you want to drop it." + observed_note
        )
    if own:
        _record_child_decision_beacon(
            ctx, tid,
            f"requested cancellation of child {tid}" + (f": {reason_text}" if reason_text else ""),
        )
    # Emit live so the supervisor processes the cancellation within one loop tick;
    # the durable intent survives a lost event (the supervisor watchdog re-feeds it).
    from ouroboros.tools.control import _emit_control_event

    physical_task_id = str(intent.get("task_id") or tid)
    emitted = _emit_control_event(ctx, {
        "type": "cancel_task",
        "task_id": physical_task_id,
        "requested_task_id": tid,
        "reason": reason_text,
        "ts": utc_now_iso(),
    })
    note = " (live)" if emitted == "live" else " (deferred to round end)"
    already = " (already requested earlier — idempotent)" if intent.get("already_requested") else ""
    return (
        f"Cancel requested: {tid}{(' — ' + reason_text) if reason_text else ''}{note}{already}. "
        "cancel_state=pending until the supervisor confirms teardown; a child that "
        "already finished keeps its completed result (use discard_child_result to drop it)." + observed_note
    )


def get_tools() -> list[ToolEntry]:
    return [
        ToolEntry("cancel_task", {
            "name": "cancel_task",
            "description": "Request cancellation of a running/scheduled task by ID (durable intent; "
                           "the supervisor confirms teardown and settles the outcome — the child shows "
                           "cancel_state=pending until then). A child that already finished keeps its "
                           "completed result: cancel means 'stop spending', not 'discard the result' "
                           "(use discard_child_result for that). Delegated children may target only "
                           "their own children. Give a short reason — it is recorded on the shared "
                           "task-tree ledger, so a stopped child is an auditable decision, not a "
                           "silent disappearance.",
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string"},
                "reason": {"type": "string", "default": "", "description": "Why you are stopping it (recorded for the tree + review)."},
            }, "required": ["task_id"]},
        }, _cancel_task),
        ToolEntry("peek_task", {
            "name": "peek_task",
            "description": "Look at a child task's CURRENT status, its latest coordination beacons "
                           "(partial_finding/blocker/question/milestone) and a tail of its result — "
                           "a PURE READ. Use this to check intermediate findings or decide whether to keep "
                           "waiting / steer / cancel, without committing to a finalization decision. It "
                           "changes no state: the pre-finalization reminder is change-based and is cleared "
                           "only by discard_child_result / cancel_task, not by reading.",
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string"},
                "view": {"type": "string", "enum": ["summary", "partials", "tail"], "default": "summary",
                         "description": "summary = status+beacons+result tail; partials = beacons only; tail = result+trace tail."},
            }, "required": ["task_id"]},
        }, _peek_task),
        ToolEntry("discard_child_result", {
            "name": "discard_child_result",
            "description": "Explicitly decide to finalize WITHOUT a child's result (abandon it on purpose). "
                           "Requires a reason. This is the structured way to drop a child you no longer need "
                           "so it stops being flagged before you finalize — use it instead of just ignoring "
                           "the child, so the abandon is a logged judgment rather than a silent loss.",
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string"},
                "reason": {"type": "string", "description": "Why this child's result is not needed."},
            }, "required": ["task_id", "reason"]},
        }, _discard_child_result),
        ToolEntry("override_delegation_constraint", {
            "name": "override_delegation_constraint",
            "description": "Explicitly override an unresolved delegation_constraint in this task tree. Requires a reason; records an append-only decision row so a future schedule_subagent call may proceed audibly.",
            "parameters": {"type": "object", "properties": {
                "constraint_id": {"type": "string"},
                "reason": {"type": "string", "description": "Why overriding this constraint is correct."},
            }, "required": ["constraint_id", "reason"]},
        }, _override_delegation_constraint),
    ]

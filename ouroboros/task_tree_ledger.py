"""Task-tree coordination ledger — the swarm blackboard + typed child->parent beacons.

Scoped by ROOT_TASK_ID (the whole task tree), so it works for ANY swarm — project or
not (email triage, research, a presentation, an OS from scratch). One append-only JSONL
holds both coordination artifacts and beacons; durable project milestones still belong in
the project journal (this ledger is EPHEMERAL coordination for one swarm run).

Domain-agnostic by design: a 'contract' is code-module APIs OR presentation
section-ownership+style OR a research claim/source schema OR an email-triage category
schema — whatever the integration seam is for THIS task. Deterministic code enforces only
form (scope, kinds, append-only, size caps); the LLM interprets meaning (BIBLE P5).
"""

from __future__ import annotations

import json
import logging
import pathlib
from hashlib import sha256
from typing import Any, Dict, List

from ouroboros.config import DATA_DIR
from ouroboros.task_results import validate_task_id
from ouroboros.utils import (
    append_jsonl,
    iter_jsonl_objects,
    jsonl_generation_signature,
    utc_now_iso,
)

log = logging.getLogger(__name__)

# Coordination artifacts + typed child->parent beacons, in one append-only ledger.
COORDINATION_KINDS = ("contract", "decision", "fact", "note")
DELEGATION_CONSTRAINT_KIND = "delegation_constraint"
REVIEW_REQUESTED_KIND = "review_requested"
BEACON_KINDS = (
    "milestone", "partial_finding", "blocker", "question", "interface_contract",
    REVIEW_REQUESTED_KIND, DELEGATION_CONSTRAINT_KIND,
)
LEDGER_KINDS = COORDINATION_KINDS + BEACON_KINDS
# Beacons that ask the parent to look NOW (surface an early return from a sliced wait): a child is
# stuck (blocker), needs an answer (question), or needs the shared seam/contract changed
# (interface_contract) — each requires the parent to reconcile before the child can safely proceed.
from ouroboros.owner_mailbox import TASK_ATTENTION_KINDS as ATTENTION_KINDS  # noqa: E402

DELEGATION_CONSTRAINT_DIRECTIVES = ("halt_fanout", "cap_children", "require_lane", "block_surface")
CHILD_RESULT_DISPOSITION_TYPE = "child_result_disposition"
CHILD_RESULT_DISPOSITIONS = frozenset({"integrated", "irrelevant", "deferred"})
_CHILD_RESULT_DISPOSITION_FIELDS = frozenset({
    "type",
    "child_task_id",
    "disposition",
    "child_result_sha256",
})

_MAX_TEXT_CHARS = 4000
# Bound runaway growth — this is a coordination ledger, not a bulk-data store.
_MAX_LEDGER_BYTES = 2 * 1024 * 1024
_MAX_EVIDENCE_REF_CHARS = 1000


def normalize_review_request_payload(payload: Any) -> Dict[str, str] | None:
    """Validate the advisory, exact-evidence review-request capsule.

    The row is only a non-blocking request to the parent.  It never launches a
    reviewer or grants acceptance authority, so the generic reference remains
    caller-authored; a parent/critic must still inspect the referenced bytes.
    Keeping the two-key shape closed gives a later parent-selected review one
    stable evidence hash for its existing accounting/deduplication path without
    introducing a review job or a second accounting ledger here.  Distinct hints
    are intentionally retained: identical bytes can need different reviews, and
    every emitter's direct parent must receive its own notification.
    """

    if not isinstance(payload, dict) or set(payload) != {
        "evidence_ref", "evidence_sha256",
    }:
        return None
    evidence_ref = str(payload.get("evidence_ref") or "").strip()
    evidence_sha256 = str(payload.get("evidence_sha256") or "").strip().lower()
    if not evidence_ref or len(evidence_ref) > _MAX_EVIDENCE_REF_CHARS:
        return None
    if len(evidence_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in evidence_sha256
    ):
        return None
    return {
        "evidence_ref": evidence_ref,
        "evidence_sha256": evidence_sha256,
    }


def tree_ledger_path(
    root_id: str,
    *,
    data_root: pathlib.Path | None = None,
) -> pathlib.Path:
    # Strict: a root_id is always an internally-generated task id, so validate_task_id RAISES on a
    # malformed id and a typo can never build a bogus task-tree path. Read callers treat the raise as
    # "no such tree" (fail-soft); the write path (tree_ledger_append) surfaces it as a TOOL_ARG_ERROR.
    root = pathlib.Path(data_root) if data_root is not None else pathlib.Path(DATA_DIR)
    return root / "task_trees" / validate_task_id(root_id) / "blackboard.jsonl"


def child_result_disposition_violations(payload: Any) -> list[str]:
    """EVERY violated constraint of the typed child-disposition row, in ONE pass.

    Returns [] when the payload is valid. The aggregated list exists because the
    old one-error-per-round shape cost a live parent 9 paid rounds discovering
    the constraints serially (W2). The key set stays CLOSED and a malformed
    payload stays an atomic no-op — unknown keys are rejected, never silently
    ignored (a silently-dropped key would swallow future typed fields), and
    nothing is truncated on the caller's behalf."""
    if not isinstance(payload, dict):
        return ["payload must be a JSON object"]
    violations: list[str] = []
    unknown = sorted(set(payload) - _CHILD_RESULT_DISPOSITION_FIELDS)
    missing = sorted(_CHILD_RESULT_DISPOSITION_FIELDS - set(payload))
    if unknown:
        violations.append(
            f"unknown key(s) {', '.join(unknown)} (the key set is CLOSED: exactly "
            "type, child_task_id, disposition, child_result_sha256)"
        )
    if missing:
        violations.append(f"missing key(s): {', '.join(missing)}")
    if "type" not in missing and str(payload.get("type") or "") != CHILD_RESULT_DISPOSITION_TYPE:
        violations.append(f"type must be exactly '{CHILD_RESULT_DISPOSITION_TYPE}'")
    if "child_task_id" not in missing:
        try:
            validate_task_id(payload.get("child_task_id"))
        except ValueError:
            violations.append("child_task_id must be a valid task id")
    if "disposition" not in missing:
        disposition = str(payload.get("disposition") or "").strip().lower()
        if disposition not in CHILD_RESULT_DISPOSITIONS:
            violations.append(
                f"disposition must be one of {'|'.join(sorted(CHILD_RESULT_DISPOSITIONS))}"
            )
    if "child_result_sha256" not in missing:
        sha = str(payload.get("child_result_sha256") or "").strip().lower()
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            violations.append(
                "child_result_sha256 must be the 64-char hex sha of the child result "
                "you actually read (from [SUBTASK_OUTCOME]/get_task_result)"
            )
    return violations


def normalize_child_result_disposition_payload(payload: Any) -> Dict[str, str] | None:
    """Validate the one typed child-disposition row shape.

    The task-tree row is the sole durable authority. Consumers deliberately
    ignore malformed rows instead of interpreting free text or task-result
    compatibility fields as a decision. Validity derives from
    ``child_result_disposition_violations`` so the normalizer and the aggregated
    diagnostics can never disagree."""
    if child_result_disposition_violations(payload):
        return None
    return {
        "type": CHILD_RESULT_DISPOSITION_TYPE,
        "child_task_id": validate_task_id(payload.get("child_task_id")),
        "disposition": str(payload.get("disposition") or "").strip().lower(),
        "child_result_sha256": str(payload.get("child_result_sha256") or "").strip().lower(),
    }


def tree_ledger_append(
    root_id: str,
    kind: str,
    text: str,
    *,
    task_id: str = "",
    role: str = "",
    needs_parent_attention: bool = False,
    payload: Dict[str, Any] | None = None,
    allow_constraint_override: bool = False,
    allow_child_result_disposition: bool = False,
    data_root: pathlib.Path | None = None,
) -> str:
    try:
        rid = validate_task_id(root_id)
    except ValueError:
        return "⚠️ TOOL_ARG_ERROR (tree_note): no/invalid task-tree scope (root_task_id missing or malformed)."
    kind_norm = str(kind or "note").strip().lower()
    if kind_norm not in LEDGER_KINDS:
        return f"⚠️ TOOL_ARG_ERROR (tree_note): kind must be one of {LEDGER_KINDS}"
    body = str(text or "").strip()
    if not body:
        return "⚠️ TOOL_ARG_ERROR (tree_note): text is required"
    if len(body) > _MAX_TEXT_CHARS:
        return (
            f"⚠️ TOOL_ARG_ERROR (tree_note): entry exceeds {_MAX_TEXT_CHARS} chars "
            f"({len(body)}) — a ledger entry is a short coordination note; keep it terse "
            "and move bulk detail to an artifact."
        )
    payload_out: Dict[str, Any] = {}
    if kind_norm == REVIEW_REQUESTED_KIND:
        normalized_review = normalize_review_request_payload(payload)
        if normalized_review is None:
            return (
                "⚠️ TOOL_ARG_ERROR (tree_note): review_requested payload must contain "
                "exactly evidence_ref (1..1000 chars) and evidence_sha256 (64-char "
                "lower/upper-case hex). The request is advisory and does not start review."
            )
        payload_out = normalized_review
    elif kind_norm == DELEGATION_CONSTRAINT_KIND:
        if not isinstance(payload, dict):
            return "⚠️ TOOL_ARG_ERROR (tree_note): delegation_constraint requires a structured payload object."
        directive = str(payload.get("directive") or "").strip().lower()
        if directive not in DELEGATION_CONSTRAINT_DIRECTIVES:
            return (
                "⚠️ TOOL_ARG_ERROR (tree_note): delegation_constraint payload.directive "
                f"must be one of {DELEGATION_CONSTRAINT_DIRECTIVES}"
            )
        scope = payload.get("scope")
        if scope is not None and not isinstance(scope, (str, dict)):
            return "⚠️ TOOL_ARG_ERROR (tree_note): delegation_constraint payload.scope must be a string or object."
        raw_constraint_id = str(payload.get("constraint_id") or "").strip()
        if not raw_constraint_id:
            seed = "|".join([
                str(root_id or ""),
                str(task_id or ""),
                directive,
                str(scope or ""),
                body,
            ])
            raw_constraint_id = "dc_" + sha256(seed.encode("utf-8")).hexdigest()[:16]
        payload_out = {
            "constraint_id": raw_constraint_id,
            "directive": directive,
            "scope": scope if scope is not None else "",
            "rationale": str(payload.get("rationale") or body)[:1000],
            "created_by": str(payload.get("created_by") or task_id or role or ""),
            "advisory": bool(payload.get("advisory")),
        }
    elif payload:
        if kind_norm == "decision" and allow_constraint_override:
            payload_out = dict(payload)
        elif kind_norm == "decision" and allow_child_result_disposition:
            normalized = normalize_child_result_disposition_payload(payload)
            if normalized is None:
                # ONE diagnostic authority for the closed contract: render the same
                # aggregated violations `child_result_disposition_violations` gives
                # the tool surface, instead of a second, weaker one-line message
                # that could drift away from the constraints actually enforced.
                return (
                    "⚠️ CHILD_RESULT_DISPOSITION_INVALID: "
                    + "; ".join(child_result_disposition_violations(payload))
                )
            # This flag is used only by join_ledger after direct-lineage and
            # current-result validation. Generic tree_note cannot bypass it.
            payload_out = normalized
        else:
            return (
                "⚠️ TOOL_ARG_ERROR (tree_note): structured payload is supported only for "
                "delegation_constraint and validated decision contracts."
            )
    path = tree_ledger_path(rid, data_root=data_root)
    try:
        # Validated child-result dispositions are the sole disposition authority
        # and are bounded by the number of children — a chatty swarm filling the
        # blackboard must not be able to lock finalization out of its ledger.
        disposition_exempt = bool(
            kind_norm == "decision" and allow_child_result_disposition
        )
        if (
            not disposition_exempt
            and path.is_file()
            and path.stat().st_size > _MAX_LEDGER_BYTES
        ):
            return (
                "⚠️ TOOL_ARG_ERROR (tree_note): the task-tree ledger is full (>2MB) — it is for "
                "coordination artifacts, not bulk data; summarize or move detail to artifacts."
            )
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    # Informational milestones stay informational even when an older caller
    # supplied the former blanket override. The kind is the typed contract.
    attention = kind_norm in ATTENTION_KINDS
    row = {
        "ts": utc_now_iso(),
        "kind": kind_norm,
        "text": body,
        "task_id": str(task_id or ""),
        "role": str(role or ""),
        "needs_parent_attention": attention,
    }
    if payload_out:
        row["payload"] = payload_out
    try:
        written = append_jsonl(path, row)
    except Exception:
        log.warning("Failed to append task-tree ledger row for %s", rid, exc_info=True)
        written = False
    if not written:
        return (
            "⚠️ TREE_LEDGER_WRITE_FAILED: the task-tree ledger row was not "
            "durably appended; no success acknowledgement was issued."
        )
    return f"OK: task-tree ledger[{rid}] += {kind_norm} entry ({len(body)} chars)."


def record_subscription_window_exhausted(
    root_id: str,
    *,
    child_task_id: str,
    reset_at: str = "",
    route: str = "",
    executor: str = "",
    data_root: pathlib.Path | None = None,
) -> bool:
    """Host-written ADVISORY ``delegation_constraint`` beacon for D28 exhaustion.

    When a child's substrate resolution lands on ``subscription_window_exhausted``
    (every plan window of the delegated route is spent — the child fell back to
    metered tokens under ``auto`` or blocked under a ``harness`` pin), the parent
    used to learn it only at absorption, typically AFTER blocking a full
    wait_tasks window and after sibling fan-out kept scheduling onto the spent
    route. This row rides the EXISTING attention channel — the kind is in
    ``ATTENTION_KINDS``, so ``tree_ledger_attention_after`` (the wait tools'
    early-wake poll) surfaces it mid-wait. ``advisory=True`` keeps it a
    disclosure: the enforcement reducer (``effective_delegation_budget``) skips
    advisory rows by contract, so nothing is gated — the woken parent decides
    (wait for reset_at, accept metered spend, or reshape the fan-out).
    Fail-soft: returns False on any write problem, never raises into dispatch.
    """
    try:
        rid = validate_task_id(root_id)
        child = validate_task_id(child_task_id)
    except ValueError:
        return False
    reset = str(reset_at or "").strip()
    text = (
        f"Delegated substrate window exhausted for child {child}: every plan window of "
        f"route {route or '?'} is spent"
        + (f" (resets at {reset})" if reset else " (no reset instant reported)")
        + f"; the child runs executor={executor or '?'}."
    )
    seed = "|".join([rid, child, "subscription_window_exhausted", reset])
    row = {
        "ts": utc_now_iso(),
        "kind": DELEGATION_CONSTRAINT_KIND,
        "text": text,
        "task_id": child,
        "role": "system",
        "needs_parent_attention": True,
        "payload": {
            "constraint_id": "dc_" + sha256(seed.encode("utf-8")).hexdigest()[:16],
            "directive": "halt_fanout",
            "scope": {"route": str(route or ""), "executor": str(executor or "")},
            "rationale": text[:1000],
            "created_by": child,
            "advisory": True,
            "reason": "subscription_window_exhausted",
            "reset_at": reset,
            "child_task_id": child,
        },
    }
    try:
        path = tree_ledger_path(rid, data_root=data_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(append_jsonl(path, row))
    except Exception:
        log.debug("Failed to append subscription-window-exhausted beacon for %s", rid, exc_info=True)
        return False


def tree_ledger_rows(
    root_id: str,
    *,
    data_root: pathlib.Path | None = None,
) -> List[Dict[str, Any]]:
    try:
        path = tree_ledger_path(root_id, data_root=data_root)  # raises on malformed root_id
    except ValueError:
        return []  # reads are fail-soft: a bad/unknown scope simply has no rows
    if not path.is_file():
        return []
    return [r for r in iter_jsonl_objects(path) if isinstance(r, dict)]


def tree_ledger_row_id(row: Dict[str, Any]) -> str:
    """Stable content identity shared by ephemeral ledger cursors."""

    encoded = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def child_result_disposition_row(
    root_id: str,
    parent_task_id: str,
    child_task_id: str,
    child_result_sha256: str,
    *,
    data_root: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """Fold the ledger to the latest decision for one exact child result.

    Rows for older semantic hashes remain audit evidence but never close the
    changed result. A later valid row for the same hash intentionally replaces
    an earlier decision; append order is the deterministic conflict rule.
    """

    try:
        rid = validate_task_id(root_id)
        parent = validate_task_id(parent_task_id)
        child = validate_task_id(child_task_id)
    except ValueError:
        return {}
    expected = str(child_result_sha256 or "").strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        return {}
    current: Dict[str, Any] = {}
    for row in tree_ledger_rows(rid, data_root=data_root):
        if str(row.get("kind") or "") != "decision":
            continue
        if str(row.get("task_id") or "").strip() != parent:
            continue
        payload = normalize_child_result_disposition_payload(row.get("payload"))
        if payload is None:
            continue
        if payload["child_task_id"] != child or payload["child_result_sha256"] != expected:
            continue
        if not str(row.get("text") or "").strip():
            continue
        current = {**row, "payload": payload}
    return current


def _tree_ledger_snapshot(source: Dict[str, Any], root_id: str) -> str:
    payload = {
        "schema_version": 1,
        "root_task_id": root_id,
        "source": source,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _tree_ledger_snapshot_rows(
    root_id: str,
    *,
    data_root: pathlib.Path | None = None,
) -> tuple[List[Dict[str, Any]], str, bool, int]:
    try:
        path = tree_ledger_path(root_id, data_root=data_root)
    except ValueError:
        return [], "", True, 0
    if not path.is_file():
        return [], "", True, 0
    rows: List[Dict[str, Any]] = []
    snapshot = ""
    unreadable = 0
    for _attempt in range(2):
        before = jsonl_generation_signature(path)
        rows = []
        unreadable = 0
        try:
            with path.open("rb") as handle:
                for raw in handle:
                    try:
                        line = raw.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        unreadable += 1
                        continue
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        unreadable += 1
                        continue
                    if not isinstance(entry, dict):
                        unreadable += 1
                        continue
                    rows.append(entry)
        except OSError:
            unreadable += 1
        after = jsonl_generation_signature(path)
        snapshot = _tree_ledger_snapshot(after, root_id)
        if before and before == after:
            return rows, snapshot, True, unreadable
    return rows, snapshot, False, unreadable


def _format_tree_ledger_row(row: Dict[str, Any]) -> str:
    flag = " ⚠needs_parent_attention" if row.get("needs_parent_attention") else ""
    # Role AND the full task id: a row is addressable with forward_to_worker/peek_task.
    who = " ".join(part for part in (str(row.get("role") or ""), str(row.get("task_id") or "")) if part)
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    payload_note = ""
    if str(row.get("kind") or "") == DELEGATION_CONSTRAINT_KIND and payload:
        payload_note = (
            f" {{id={payload.get('constraint_id')}, directive={payload.get('directive')}, "
            f"scope={payload.get('scope')}}}"
        )
    elif str(payload.get("type") or "") == "child_result_disposition":
        payload_note = (
            f" {{child={payload.get('child_task_id')}, disposition={payload.get('disposition')}, "
            f"sha256={str(payload.get('child_result_sha256') or '')[:12]}}}"
        )
    elif str(row.get("kind") or "") == REVIEW_REQUESTED_KIND and payload:
        payload_note = (
            f" {{evidence={payload.get('evidence_ref')}, "
            f"sha256={str(payload.get('evidence_sha256') or '')}}}"
        )
    return (
        f"- [{str(row.get('ts') or '')[:16]}] {str(row.get('kind') or 'note')}{flag} "
        f"({who}){payload_note}: {str(row.get('text') or '')}"
    )


def tree_ledger_page(
    root_id: str,
    *,
    limit: int = 40,
    offset: int = 0,
    snapshot: str = "",
    data_root: pathlib.Path | None = None,
) -> str:
    rows, current_snapshot, stable, unreadable = _tree_ledger_snapshot_rows(
        root_id, data_root=data_root,
    )
    if not rows and not current_snapshot:
        if str(snapshot or "").strip():
            return (
                "TREE_READ_SNAPSHOT_CHANGED: the task-tree ledger source is no longer "
                "available; no mixed page was returned; restart with offset=0 and no snapshot."
            )
        return ""
    if not stable:
        return (
            "TREE_READ_SNAPSHOT_CHANGED_DURING_READ: the task-tree ledger changed "
            "while the page was captured; no mixed page was returned; retry with "
            "offset=0 and no snapshot."
        )
    requested_snapshot = str(snapshot or "").strip().lower()
    if requested_snapshot and requested_snapshot != current_snapshot:
        return (
            "TREE_READ_SNAPSHOT_CHANGED: the task-tree ledger changed after the prior "
            "page; no mixed page was returned; restart with offset=0 and no snapshot."
        )
    try:
        take = max(1, min(int(limit or 40), 200))
    except (TypeError, ValueError):
        take = 40
    try:
        skip = max(0, int(offset or 0))
    except (TypeError, ValueError):
        skip = 0
    valid_total = len(rows)
    total = valid_total + unreadable
    end = max(0, valid_total - skip)
    start = max(0, end - take)
    page = rows[start:end]
    remaining = start
    lines = [
        f"Page: total={total} valid_total={valid_total} unreadable={unreadable} "
        f"returned={len(page)} offset={skip} remaining={remaining} "
        f"remaining_scope=valid_rows coverage={'partial' if unreadable else 'complete'} "
        f"snapshot={current_snapshot}"
    ]
    if unreadable:
        lines.append(
            f"TREE_READ_GAP: coverage is partial; {unreadable} non-empty physical "
            "row(s) were malformed or non-object JSON. Valid rows remain pageable."
        )
    if remaining:
        lines.append(
            f"Next older page: tree_read(limit={take}, offset={skip + len(page)}, "
            f"snapshot='{current_snapshot}')"
        )
    lines.extend(_format_tree_ledger_row(row) for row in page)
    return "\n".join(lines)


def tree_ledger_tail_digest(
    root_id: str,
    *,
    limit: int = 40,
    data_root: pathlib.Path | None = None,
) -> str:
    """Recent ledger entries for context injection (no ctx needed). Each entry shown in
    full; older entries beyond the tail represented by a visible pointer to tree_read."""
    rows, snapshot, stable, unreadable = _tree_ledger_snapshot_rows(
        root_id, data_root=data_root,
    )
    if not rows and not unreadable:
        return ""
    page_size = max(1, int(limit))
    take = rows[-page_size:]
    omitted = len(rows) - len(take)
    lines: List[str] = []
    if unreadable:
        lines.append(
            f"- ⚠ coverage partial: {unreadable} unreadable task-tree row(s); "
            "inspect valid rows with tree_read"
        )
    if omitted:
        if stable:
            lines.append(
                f"- …[{omitted} earlier ledger entries; source=tree_read; next="
                f"tree_read(limit={page_size}, offset={len(take)}, "
                f"snapshot='{snapshot}') ]"
            )
        else:
            lines.append(
                f"- …[{omitted} observed earlier ledger entries; ledger changed during "
                f"capture; restart with tree_read(limit={page_size})]"
            )
    lines.extend(_format_tree_ledger_row(row) for row in take)
    return "\n".join(lines)


def tree_ledger_attention_after(
    root_id: str,
    after_ts: str,
    *,
    task_ids: set[str] | None = None,
    seen_ids: set[str] | None = None,
    data_root: pathlib.Path | None = None,
) -> List[Dict[str, Any]]:
    """Return new attention beacons, optionally filtered to direct children.

    The ledger is whole-tree scoped, but a sleeping nanny may react only to its
    own direct children. Task waits pass their exact waited ids; callers that
    deliberately omit the optional filters keep the whole-tree projection.
    """
    out: List[Dict[str, Any]] = []
    allowed = {str(item) for item in task_ids} if task_ids is not None else None
    seen = {str(item) for item in seen_ids} if seen_ids is not None else set()
    for r in tree_ledger_rows(root_id, data_root=data_root):
        if r.get("kind") not in ATTENTION_KINDS:
            continue
        if allowed is not None and str(r.get("task_id") or "") not in allowed:
            continue
        ts = str(r.get("ts") or "")
        if after_ts and ts < after_ts:
            continue
        if after_ts and ts == after_ts:
            # Historical callers rely on the original strict-after contract.
            # The sleeping-nanny rail passes seen_ids so equal-timestamp rows can
            # still be admitted once and then de-duplicated by content hash.
            if seen_ids is None:
                continue
            row_id = tree_ledger_row_id(r)
            if row_id in seen:
                continue
        out.append(r)
    return out


def open_delegation_constraints(root_id: str) -> List[Dict[str, Any]]:
    """Unresolved delegation constraints for a task tree.

    A constraint is resolved by a later decision row carrying
    payload.decision="overridden" for the same payload.constraint_id. Malformed
    rows are ignored by consumers (coordination must fail open).
    """

    rows = tree_ledger_rows(root_id)
    constraints: List[Dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        constraint_id = str(payload.get("constraint_id") or "").strip()
        if not constraint_id:
            continue
        if str(row.get("kind") or "") == "decision" and str(payload.get("decision") or "").strip().lower() == "overridden":
            constraints = [
                existing for existing in constraints
                if str((existing.get("payload") if isinstance(existing.get("payload"), dict) else {}).get("constraint_id") or "").strip()
                != constraint_id
            ]
            continue
        if str(row.get("kind") or "") == DELEGATION_CONSTRAINT_KIND:
            constraints.append(row)
    return constraints


__all__ = [
    "LEDGER_KINDS",
    "COORDINATION_KINDS",
    "BEACON_KINDS",
    "ATTENTION_KINDS",
    "DELEGATION_CONSTRAINT_DIRECTIVES",
    "DELEGATION_CONSTRAINT_KIND",
    "REVIEW_REQUESTED_KIND",
    "CHILD_RESULT_DISPOSITION_TYPE",
    "CHILD_RESULT_DISPOSITIONS",
    "normalize_child_result_disposition_payload",
    "normalize_review_request_payload",
    "child_result_disposition_row",
    "record_subscription_window_exhausted",
    "tree_ledger_path",
    "tree_ledger_append",
    "tree_ledger_rows",
    "tree_ledger_row_id",
    "tree_ledger_page",
    "tree_ledger_tail_digest",
    "tree_ledger_attention_after",
    "open_delegation_constraints",
]

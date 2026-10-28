"""Helpers for durable task result/status files."""

from __future__ import annotations

import copy
import json
import logging
import pathlib
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from ouroboros.cost_projection import COST_ALIAS_PAIRS, COST_OPENNESS_FIELDS, normalize_task_result_cost_planes
from ouroboros.task_store import list_documents, patch_document, read_document
from ouroboros.utils import utc_now_iso
# Projection owner retains read custody; this historical import remains compatible.
from ouroboros.review_projection import merge_review_projection as merge_review_projection
from ouroboros.review_records import validate_author_disposition
from ouroboros.terminal_time import preserve_terminal_attempt, task_attempt_witness, terminal_time_patch

log = logging.getLogger(__name__)

STATUS_REQUESTED = "requested"
STATUS_SCHEDULED = "scheduled"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_REJECTED_DUPLICATE = "rejected_duplicate"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"
STATUS_CANCELLED = "cancelled"

# ABI 7.0 (Q8=B): re-export schema admission, stamping and quarantine for callers.
from ouroboros.task_result_schema import (  # noqa: F401
    QUARANTINED_SCHEMA_REASON, TASK_RESULT_QUARANTINE_DIR, TASK_RESULT_SCHEMA_VERSION,
    emit_quarantine_event as _emit_quarantine_event,
    quarantine_task_result as _quarantine_task_result,
    require_writable_task_result_schema, stamp_task_result_schema,
    task_result_schema_refusal,
)


# Cancel intent outranks running mirrors but not terminal states: stale progress
# cannot resurrect the task, and the supervisor's final CANCELLED write still lands.
STATUS_CANCEL_REQUESTED = "cancel_requested"

# Events, progress replay, chat summaries and results share the cost-projection
# SSOT's current names and ALL openness/integrity markers. Legacy cost_usd aliases
# are read-only compatibility: use carry_cost_meta, not a key loop, to copy them.
# In particular, non_final_rows explains cost_final; omitting it or
# ledger_integrity_degraded would silently lose the authority's uncertainty.
TASK_COST_META_FIELDS = tuple(dict.fromkeys(
    [new for new, _old in COST_ALIAS_PAIRS] + list(COST_OPENNESS_FIELDS)
))

# Monotonic lifecycle: stale scheduled/running mirrors cannot overwrite cancellation
# or terminal outcomes (the ghost-subagent class). Unknown statuses never block.
_TRULY_TERMINAL_STATUSES = frozenset({
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED,
    STATUS_REJECTED_DUPLICATE,
})
_STATUS_RANK = {
    STATUS_REQUESTED: 0,
    STATUS_SCHEDULED: 1,
    STATUS_RUNNING: 2,
    STATUS_INTERRUPTED: 2,
    STATUS_CANCEL_REQUESTED: 3,
    STATUS_COMPLETED: 4,
    STATUS_FAILED: 4,
    STATUS_CANCELLED: 4,
    STATUS_REJECTED_DUPLICATE: 4,
}
# Regressions are only blocked once a task reaches the cancel-intent latch or a
# terminal state; normal forward progress (requested->scheduled->running) and
# unknown statuses are always allowed.
_REGRESSION_GUARD_FLOOR = _STATUS_RANK[STATUS_CANCEL_REQUESTED]

_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

PLAN_REVIEW_STATE_KEY = "plan_review_state"
_PLAN_REVIEW_STATE_VERSION = 2
_PLAN_REVIEW_STATE_MAX_BYTES = 1_000_000
_PLAN_REVIEW_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_PLAN_REVIEW_REASON_MAX_CHARS = 2_000

def cancellation_blocks_child_result(result: Any) -> bool:
    """Return whether canonical cancellation forbids child-drive promotion.

    Only a supervisor-SETTLED ``cancelled`` blocks: cancel INTENT no longer rides
    the canonical status (it lives in the durable ``cancel_intents`` projection),
    and natural completion WINS a late cancel (owner decision 4=A, 2026-08-11) —
    a child that finished before the teardown keeps its completed result and
    artifacts, so copy-back paths must promote it rather than refuse it.
    """

    if not isinstance(result, dict):
        return False
    return str(result.get("status") or "").strip().lower() == STATUS_CANCELLED


def resolve_task_lineage(
    task_id: Any,
    *,
    metadata: Any = None,
    root_task_id: Any = None,
    parent_task_id: Any = None,
    delegation_role: Any = None,
    original_task_id: Any = None,
    timeout_retry_from: Any = None,
) -> Dict[str, Any]:
    """Return one typed lineage projection for root-owned lifecycle gates.

    ``root_task_id`` is the logical subtree/budget authority and intentionally
    survives a top-level hard-timeout retry that receives a fresh physical
    ``task_id``.  Such a retry is a root *attempt* only when the two independent
    host-written retry markers agree.  This keeps malformed lineage fail-closed
    without splitting budget, fence, task-tree, or cost authorities.
    """

    meta = metadata if isinstance(metadata, dict) else {}

    def _field(explicit: Any, key: str) -> str:
        # ``None`` means the canonical carrier is absent.  An explicit empty
        # parent is meaningful and must override stale copied metadata.
        value = explicit if explicit is not None else meta.get(key)
        return str(value or "").strip()

    resolved_task_id = str(task_id or "").strip()
    resolved_root_id = _field(root_task_id, "root_task_id") or resolved_task_id
    resolved_parent_id = _field(parent_task_id, "parent_task_id")
    resolved_role = _field(delegation_role, "delegation_role").lower()
    resolved_original_id = _field(original_task_id, "original_task_id")
    resolved_retry_from = _field(timeout_retry_from, "timeout_retry_from")
    is_regular_root = bool(resolved_task_id and resolved_root_id == resolved_task_id
                           and not resolved_parent_id and resolved_role != "subagent")
    is_retry_root = bool(resolved_task_id and resolved_root_id and resolved_root_id != resolved_task_id
                         and not resolved_parent_id and resolved_role == "root" and resolved_original_id
                         and resolved_original_id == resolved_retry_from and resolved_original_id != resolved_task_id)
    return {
        "task_id": resolved_task_id,
        "root_task_id": resolved_root_id,
        "parent_task_id": resolved_parent_id,
        "delegation_role": resolved_role,
        "original_task_id": resolved_original_id,
        "timeout_retry_from": resolved_retry_from,
        "is_retry_root_attempt": is_retry_root,
        "is_root_task": bool(is_regular_root or is_retry_root),
    }


def resolve_ctx_lineage(ctx: Any, task_id: str = "") -> Dict[str, Any]:
    """``resolve_task_lineage`` over a live tool context's own lineage facts."""
    meta = getattr(ctx, "task_metadata", {})
    return resolve_task_lineage(
        task_id or getattr(ctx, "task_id", ""),
        metadata=meta if isinstance(meta, dict) else {},
        root_task_id=getattr(ctx, "root_task_id", None),
        parent_task_id=getattr(ctx, "parent_task_id", None),
        delegation_role=getattr(ctx, "delegation_role", None),
        original_task_id=getattr(ctx, "original_task_id", None),
        timeout_retry_from=getattr(ctx, "timeout_retry_from", None),
    )


def _is_status_regression(existing_status: str, new_status: str) -> bool:
    """Return True when writing *new_status* over *existing_status* would
    regress or corrupt a task that has already reached cancel-intent or a
    terminal state.

    Rules:
      - Unknown statuses never block (forward-compatible).
      - Truly-terminal is sticky: once completed/failed/cancelled/rejected, only
        a same-status rewrite is allowed (result/trace enrichment). Switching to
        a *different* terminal status (e.g. cancelled -> completed) is blocked.
      - cancel-intent (cancel_requested) blocks regress to running/scheduled but
        still allows the supervisor's eventual terminal write (rank 3 -> 4).
    """
    existing = str(existing_status or "")
    new = str(new_status or "")
    # Sticky terminal FIRST — independent of whether the new status is ranked, so
    # a typo/unknown/future status can never overwrite a terminal one. Only an
    # identical-status rewrite (result/trace enrichment) is allowed.
    if existing in _TRULY_TERMINAL_STATUSES:
        return new != existing
    if existing == STATUS_CANCEL_REQUESTED:
        # LEGACY read-path only (pre-intent files): the latch status is no longer
        # written — cancel intent lives in the durable ``cancel_intents``
        # projection. Natural completion WINS (owner decision 4=A): any terminal
        # write, including a racing ``completed``, may land over an old latch;
        # only a regression to scheduled/running is refused.
        new_rank = _STATUS_RANK.get(new)
        return new_rank is not None and new_rank < _STATUS_RANK[STATUS_CANCEL_REQUESTED]
    existing_rank = _STATUS_RANK.get(existing)
    new_rank = _STATUS_RANK.get(new)
    if existing_rank is None or new_rank is None:
        return False
    if existing_rank >= _REGRESSION_GUARD_FLOOR:
        return new_rank < existing_rank
    return False


def lifecycle_named(axes: Any, status: str) -> Any:
    """``axes`` whose lifecycle states ``status``, as a copy; the caller's dict stays as given.

    The lifecycle axis is the row's own status (``normalize_outcome_axes`` reads it so), yet a
    loop that ended derives ``completed`` before the writer settles ``failed`` or loses a
    terminal race. Every other axis is evidence and stays; absent or malformed lifecycle data
    stays as written."""
    lifecycle = axes.get("lifecycle") if isinstance(axes, dict) else None
    if not isinstance(lifecycle, dict) or lifecycle.get("status") == status:
        return axes
    return {**axes, "lifecycle": {**lifecycle, "status": status}}


def is_reconciled_presence_placeholder(row: Dict[str, Any]) -> bool:
    """The orphan reconciler's failed mark on a lost presence turn: no terminal of its own ever ran."""
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return (row.get("status") == STATUS_FAILED and not row.get("terminal_origin")
            and row.get("status_reconciled_from") in {STATUS_RUNNING, STATUS_INTERRUPTED}
            and row.get("reason_code") in {"orphaned_running_after_worker_restart", "interrupted_retry_lost"}
            and (meta.get("source") == "presence" or isinstance(meta.get("presence"), dict)))


def reopen_reconciled_presence_placeholder(drive_root: Any, task_id: str) -> bool:
    """Return a placeholder to ``running`` for its re-run; the host mark moves to ``superseded_placeholder``."""
    path, reopened = f"task:{validate_task_id(task_id)}", []
    # The terminal-projection bookkeeping of the placeholder's failed transition goes with it: the
    # re-run's own terminal transition must originate its own room row, never inherit a failed one.
    projection = ("canonical_terminal_projection", "canonical_terminal_projection_ready",
                  "canonical_terminal_projection_origin", "terminal_time")
    cleared = {"reason_code", "outcome_axes", "artifact_status", "artifact_bundle", "result",
               "status_reconciled_from", *projection}

    def _reopen(existing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not is_reconciled_presence_placeholder(existing):
            return None
        require_writable_task_result_schema(existing, path)
        mark = {key: existing.get(key) for key in ("status", "reason_code", "status_reconciled_from", "ts", *projection)
                if key in existing}
        reopened.append(utc_now_iso())
        return stamp_task_result_schema({
            **{key: value for key, value in existing.items() if key not in cleared},
            "status": STATUS_RUNNING, "ts": reopened[0], "updated_at": reopened[0],
            "superseded_placeholder": {**mark, "result": str(existing.get("result") or "")[-500:]},
        })

    patch_document(drive_root, task_id, _reopen)
    return bool(reopened)


def validate_task_id(task_id: Any) -> str:
    text = str(task_id or "").strip()
    if not _TASK_ID_RE.fullmatch(text):
        raise ValueError("task_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
    return text


def task_results_dir(drive_root: Any, *, create: bool = True) -> pathlib.Path:
    """Resolve ``<drive_root>/task_results``; writers create it, readers pass ``create=False``.

    A read-side mkdir once let a scan with a MagicMock-derived root create a stray
    ``MagicMock/.../task_results`` tree in cwd, so a never-provisioned root reads as empty."""
    path = pathlib.Path(drive_root) / "task_results"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def task_result_path(drive_root: Any, task_id: str, *, create: bool = True) -> pathlib.Path:
    return task_results_dir(drive_root, create=create) / f"{validate_task_id(task_id)}.json"


def _normalize_swarm_efficiency(data: Dict[str, Any]) -> Dict[str, Any]:
    """Read the two retired fan-out names without rewriting historical records."""
    value = data.get("swarm_efficiency")
    if not isinstance(value, dict):
        return data
    value = dict(value)
    for old, new in (("wave_count", "fanout_count"),
                     ("inter_wave_latency_sec_total", "fanout_interval_sec_total")):
        if old in value:
            value.setdefault(new, value.pop(old))
    return {**data, "swarm_efficiency": value}


def load_task_result(
    drive_root: Any, task_id: str, *, strict: bool = False, signed: bool = False,
) -> Any:
    """Read one exact result through schema admission, tolerating old fan-out names.

    Fail-soft refusal quarantines with a batched event and returns None; strict
    refusal raises without moving, so unreadable identities cannot be readmitted.
    Neither mode materializes deliverables or independent state axes. ``signed``
    returns ``(result, signature)`` from ONE read transaction: an ownership check
    (``supervisor.task_ownership``) re-checks the stamp of the very body it holds,
    so a writer landing between two reads is a change and a busy store is unknown,
    never a false change."""
    try:
        tid = validate_task_id(task_id)
    except ValueError:
        if strict:
            raise
        return (None, None) if signed else None
    data, stamp = read_document(drive_root, tid, signed=True) if signed else (read_document(drive_root, tid), None)
    if data is None:
        if strict:
            _refuse_occupied(drive_root, tid, write=False)
        result = None  # This read saw absence even if a writer publishes immediately after it.
    else:
        _admit_stored(data, tid, strict=strict, noun="task result authority")
        result = _normalize_swarm_efficiency(data)
    return (result, stamp) if signed else result


def admitted_results(drive_root: Any, task_ids: Iterable[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """``load_task_result(strict=True)`` of many ids in ONE read, keyed by the ids as given, where a
    refusal (an invalid id, a row whose identity or status does not bind) and an absence are both
    ``None``: what a reader that turns every refusal into "no fact" needs
    (``dialogue_provenance.task_lineage_lookup``). A store that cannot be read still raises."""
    from ouroboros.task_store import documents_by_ids

    valid: Dict[str, str] = {}
    for task_id in task_ids:
        try:
            valid[str(task_id)] = validate_task_id(task_id)
        except ValueError:
            valid[str(task_id)] = ""
    documents = documents_by_ids(drive_root, sorted({tid for tid in valid.values() if tid}))
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    for task_id, tid in valid.items():
        data = documents.get(tid) if tid else None
        try:
            if data is not None:
                _admit_stored(data, tid, strict=True, noun="task result authority")
        except ValueError:
            data = None
        results[task_id] = None if data is None else _normalize_swarm_efficiency(data)
    return results


def _refuse_occupied(drive_root: Any, task_id: str, *, write: bool) -> None:
    """ABI-2: an identity whose file the import refused or quarantined has no row, yet a strict
    reader or a create-only writer must not readmit it as a fresh absence."""
    from ouroboros.store_import import occupied_reason

    if occupied_reason(drive_root, task_id, write=write):
        raise ValueError(f"task result authority is unreadable or invalid: task:{task_id}")


def _admit_stored(data: Dict[str, Any], task_id: str, *, strict: bool, noun: str) -> None:
    """A stored row was admitted when it entered the store; the strict reader still refuses an
    identity or status it cannot bind (the file-era message, so callers keep their wording)."""
    if strict and (str(data.get("task_id") or "") != task_id or not isinstance(data.get("status"), str)
                   or not str(data.get("status") or "").strip()):
        raise ValueError(f"{noun} is unreadable or invalid: task:{task_id}")


def list_task_results(
    drive_root: Any,
    *,
    statuses: Optional[List[str]] = None,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """List task results; ``statuses`` is the store's WHERE prefilter on the stored status.

    Strict authority readers cannot treat an unbindable row as absent: one such row refuses
    the whole list. Rows come back in task-id order.
    """
    wanted = {str(item) for item in list(statuses or []) if str(item).strip()}
    results: List[Dict[str, Any]] = []
    for data in list_documents(drive_root, statuses=wanted or None):
        _admit_stored(data, str(data.get("task_id") or ""), strict=strict, noun="task result")
        results.append(_normalize_swarm_efficiency(data))
    return results


def write_task_result(
    results_drive_root: Any,
    task_id: str,
    status: str,
    *,
    _field_projector: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    strict_existing_dict: bool = False,
    create_only: bool = False,
    _terminal_observed: bool = False,
    _terminal_time_source: Optional[Dict[str, Any]] = None,
    **fields: Any,
) -> Dict[str, Any]:
    """Merge-write a task result inside one store write transaction.

    Workers, supervisor and gateways share CURRENT under that transaction: the monotonic
    reducer decides terminal races, so natural completion wins a late cancel.
    Discarding a result remains the separate explicit ``discard_child_result``.
    ``_field_projector`` runs after review-publication selection under the same
    lock, deriving fields/status from CURRENT or publishing its verified refs;
    repeating the incoming-review merge afterward would undo that handoff.
    ``strict_existing_dict`` is accepted for its callers: every stored row is a dict.
    ``create_only`` initializes only absence, before projection or timestamps.
    The body is one store transaction: the merge over CURRENT, the typed columns and
    obligation flags of the merged document, the custody and handoff riders, COMMIT.
    """
    path = f"task:{validate_task_id(task_id)}"
    if create_only:
        _refuse_occupied(results_drive_root, task_id, write=True)
    explicit_ts = str(fields.pop("ts", "") or "")
    from ouroboros.task_custody import capture_owner_mail, capture_unread_mail, merge_unread_mail
    from ouroboros.task_store import Reprepare, prepared_write

    prepared: Dict[str, Any] = {}
    transition: Dict[str, bool] = {}  # set by the reducer when THIS write turns the row truly terminal

    def _write(needs: set) -> Dict[str, Any]:
        # TZ-1 V10: the mailbox bytes are read BEFORE the row lock (a bounded union happens
        # under it); a write the projector turns terminal asks once
        # (``Reprepare``) and the whole write repeats with that input prepared out here
        # (what an earlier round prepared is kept, not read again).
        if "mail" not in prepared and (status in _TRULY_TERMINAL_STATUSES or "mail" in needs):
            prepared["mail"] = (capture_unread_mail(results_drive_root, task_id), capture_owner_mail(results_drive_root, task_id))
        return patch_document(results_drive_root, task_id, lambda existing: _merge(existing, prepared))

    def _merge(existing: Dict[str, Any], prepared: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        transition.clear()
        if strict_existing_dict and existing and (
            str(existing.get("task_id") or "") != str(task_id)
            or not isinstance(existing.get("status"), str)
            or not str(existing.get("status") or "").strip()
        ):
            raise ValueError(
                f"task result authority is unreadable or invalid: {path}"
            )
        # ABI 7.0: every write stamps the row; another schema's row is never downgraded.
        require_writable_task_result_schema(existing, path)
        if create_only and existing:
            return None
        prepared_fields = dict(fields)
        if not existing and "swarm_fanout_rollup" not in prepared_fields:
            from ouroboros.swarm_rollup import initial_swarm_rollup
            prepared_fields["swarm_fanout_rollup"] = initial_swarm_rollup(results_drive_root)
        if "review_projection" in prepared_fields:
            prepared_fields["review_projection"] = merge_review_projection(
                existing.get("review_projection"), prepared_fields["review_projection"],
            )
        projected_fields = _field_projector(existing, {**prepared_fields, "status": status}) if _field_projector else prepared_fields
        if projected_fields is None:  # projector saw a terminal/stale row: no mutation
            return None
        # The store reconciles fan-out base witnesses with addressed event union,
        # including an independently repaired execution-root base on copy-back.
        projected_status = str(projected_fields.pop("status", status))
        projected_fields.pop("terminal_time", None)  # ordinary fields cannot manufacture clock authority
        # Monotonic lifecycle: no stale mirror may overwrite a terminal outcome.
        existing_status = str(existing.get("status") or "")
        if existing and _is_status_regression(existing_status, projected_status):
            # Debugging a "stuck" task: the only signal that a stale/late write was dropped.
            log.debug("Blocked status regression %s -> %s for task %s",
                      existing.get("status"), projected_status, task_id)
            return None
        # The lifecycle axis names the status this accepted write lands: in the axes it carries,
        # or in the current axes it keeps across a status change. Same-status enrichment leaves
        # stored axes alone, so untouched history is never migrated.
        carried = "outcome_axes" in projected_fields
        if carried or projected_status != existing_status:
            axes = projected_fields["outcome_axes"] if carried else existing.get("outcome_axes")
            named = lifecycle_named(axes, projected_status)
            if named is not axes:
                projected_fields["outcome_axes"] = named
        if isinstance(projected_fields.get("delegate_terminal_reconciliation"), dict):
            # #1573: no writer, a wholesale terminal one included, unpublishes a retained output.
            from ouroboros.delegate_terminal import keep_published_receipts
            projected_fields["delegate_terminal_reconciliation"] = keep_published_receipts(
                existing, projected_fields["delegate_terminal_reconciliation"])
        # Only this accepted transition can originate terminal delivery debt:
        # enrichment/replica fields cannot adopt historical terminal rows or
        # erase provenance persisted before the separate readiness write.
        projected_fields.pop("canonical_terminal_projection_origin", None)
        if projected_status in _TRULY_TERMINAL_STATUSES and existing_status not in _TRULY_TERMINAL_STATUSES:
            if "mail" not in prepared:
                raise Reprepare("mail")  # the projector turned the row terminal: the capture happens outside
            transition["terminal"] = True
            captured, owner_mail = prepared["mail"]
            merged = {**existing, **projected_fields}
            lineage_keys = ("root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")
            if resolve_task_lineage(task_id, metadata=merged.get("metadata"),
                                    **{key: merged.get(key) for key in lineage_keys})["is_root_task"]:
                projected_fields["canonical_terminal_projection_origin"] = "terminal_transition"
                # Batch4: the owner's exact rows, ACKed ones included, outlive the mailbox cleanup.
                projected_fields["owner_mailbox"] = merge_unread_mail(existing.get("owner_mailbox"), owner_mail)
            # TZ-1 V10: this accepted transition keeps the mail no attempt read (no ACK written);
            # later late mail joins through settlement, never through a rejected write.
            projected_fields["unread_mailbox"] = merge_unread_mail(projected_fields.get("unread_mailbox"), captured)
        # Unread-mail custody only grows: no replica or partial write can shrink it.
        projected_fields["unread_mailbox"] = merge_unread_mail(
            existing.get("unread_mailbox"), projected_fields.get("unread_mailbox"))
        if projected_fields["unread_mailbox"] is None:
            projected_fields.pop("unread_mailbox")
        projected_fields = preserve_terminal_attempt(existing, projected_fields)
        if ((existing.get("wake_summary") or existing.get("suggested_name")) and not projected_fields.get("wake_summary")
                and (transition.get("terminal") or task_attempt_witness(existing)
                     != task_attempt_witness({**existing, **projected_fields}))):
            from ouroboros.task_finalization import wake_cause_title

            title = wake_cause_title(projected_fields.get("metadata", existing.get("metadata")) or {})
            if title or existing.get("wake_summary"):
                projected_fields.update(wake_summary=None, suggested_name=title or existing.get("title") or None)
        now = utc_now_iso()
        projected_fields.update(terminal_time_patch(existing, projected_fields, status=projected_status,
            task_id=task_id, observed_at=now if _terminal_observed else None, replica=_terminal_time_source))
        # ABI-3: shared deep cost normalization handles both rows/nested planes; incoming values win.
        return stamp_task_result_schema(normalize_task_result_cost_planes({
            **normalize_task_result_cost_planes(existing),
            **projected_fields,
            "task_id": task_id,
            "status": projected_status,
            "ts": explicit_ts or str(existing.get("ts") or now),
            "updated_at": now,
        }))

    stored = prepared_write(_write)
    if transition.get("terminal"):
        # Prior safety flags are in-process retry authority, never a durable grant.
        # Child completion cannot clear its still-running root's history.
        if stored and str(stored.get("root_task_id") or (stored.get("metadata") or {}).get("root_task_id") or task_id) == task_id:
            from ouroboros.safety import clear_safety_work
            clear_safety_work(task_id)
        # The accepted transition into a truly terminal status ends the task: processes its commands left behind are
        # ended by scope (marker + recorded groups) outside this store transaction, on positive evidence only; a pause,
        # retained hold, handoff or reopenable placeholder never reaches here (DEVELOPMENT "Process Custody Rule").
        try:  # a task that launched nothing costs nothing
            from ouroboros.process_scope import note_task_terminal_transition
            note_task_terminal_transition(results_drive_root, task_id)
        except Exception:
            log.debug("task scope cleanup could not be started for %s; the sweep owns it", task_id, exc_info=True)
    from ouroboros.swarm_receipt import refresh_for_task
    refresh_for_task(results_drive_root, stored)
    return stored


# plan_review_state (v1/v2) is the earlier planning engine's record, read-only since the
# planning organ (``plan_record``): finished tasks, and tasks in flight at an upgrade,
# still hold it. The readers below validate the bounded state and resolve its exact
# sources (``tools/plan_review_artifacts.py``); v1 stays wrapped under legacy_v1, never
# migrated, and nothing writes either schema.

_PLAN_REVIEW_MAX_WAVES = 64
_PLAN_REVIEW_ATTEMPT_STATUSES = frozenset({"open", "unavailable", "rail_degraded", "cycles_exhausted"})
_PLAN_REVIEW_AGGREGATES = frozenset({"GREEN", "REVIEW_REQUIRED", "REVISE_PLAN", "DEGRADED"})


def _empty_plan_review_state() -> Dict[str, Any]:
    return {
        "schema_version": _PLAN_REVIEW_STATE_VERSION,
        "series_id": "",
        "cycles_paid": 0,
        "need_evidence_seen": [],
        "current_attempt": {},
        "waves": [],
        "waves_omitted": 0,
    }


def legacy_plan_review_projection(value: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only projection of a v1 record: ``{fingerprint, status, outcome, closed,
    acceptance_claims}`` where status ∈ closed | rail_degraded | open (every non-closed
    v1 ATTEMPT: open review, unavailable) | pending (a wave without any attempt) | absent."""
    attempt = value.get("current_attempt") if isinstance(value.get("current_attempt"), dict) else {}
    fingerprint = str(attempt.get("fingerprint") or "") or str(value.get("latest_review_fingerprint") or "")
    waves = value.get("waves") if isinstance(value.get("waves"), list) else []
    wave = next((w for w in waves if isinstance(w, dict)
                 and str(w.get("request_fingerprint") or "") == fingerprint), None) if fingerprint else None
    review = wave.get("review") if isinstance((wave or {}).get("review"), dict) else {}
    outcome = str(review.get("aggregate_signal") or "")
    integrated = bool(review and str((wave or {}).get("phase") or "") == "reviewed"
                      and str((wave or {}).get("review_evidence_status") or "integrated") != "pending")
    closed = integrated and bool(review.get("closed")) and outcome in {"GREEN", "REVIEW_REQUIRED"}
    if closed:
        status = "closed"
    elif str(attempt.get("status") or "") == "rail_degraded":
        status = "rail_degraded"
    elif fingerprint:
        status = "open"
    elif waves:
        status = "pending"  # a v1 wave that never reached a panel: hold under every policy
    else:
        status = "absent"
    return {
        "fingerprint": fingerprint, "status": status, "outcome": outcome, "closed": closed,
        "reason": str(attempt.get("reason") or ""),
        "acceptance_claims": list((wave or {}).get("acceptance_claims") or []),
    }


def _validated_plan_review_state(value: Any) -> Dict[str, Any]:
    """Validate bounded v2 state; wrap v1 read-only without migrating authority."""
    if value in (None, {}):
        return _empty_plan_review_state()
    if not isinstance(value, dict):
        raise ValueError("PLAN_REVIEW_STATE_INVALID: unsupported or malformed schema")
    if value.get("schema_version") == 1:
        state = _empty_plan_review_state()
        state["legacy_v1"] = copy.deepcopy(value)
        state["legacy_v1_projection"] = legacy_plan_review_projection(value)
        return state
    if value.get("schema_version") != _PLAN_REVIEW_STATE_VERSION:
        raise ValueError("PLAN_REVIEW_STATE_INVALID: unsupported or malformed schema")
    waves = value.get("waves")
    if not isinstance(waves, list) or len(waves) > _PLAN_REVIEW_MAX_WAVES:
        raise ValueError("PLAN_REVIEW_STATE_INVALID: waves must be a bounded list")
    seen: set[str] = set()
    for wave in waves:
        if not isinstance(wave, dict):
            raise ValueError("PLAN_REVIEW_STATE_INVALID: wave must be an object")
        fingerprint = str(wave.get("request_fingerprint") or "")
        if not _PLAN_REVIEW_HASH_RE.fullmatch(fingerprint) or fingerprint in seen:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: wave fingerprints must be unique")
        if str(wave.get("aggregate") or "") not in _PLAN_REVIEW_AGGREGATES:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: invalid wave aggregate")
        if not isinstance(wave.get("closed"), bool):
            raise ValueError("PLAN_REVIEW_STATE_INVALID: wave closed must be boolean")
        if wave["aggregate"] == "GREEN" and not wave["closed"]:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: GREEN wave must be closed")
        if wave["aggregate"] in {"REVISE_PLAN", "DEGRADED"} and wave["closed"]:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: REVISE_PLAN/DEGRADED wave cannot be closed")
        if not wave.get("compact"):
            if not isinstance(wave.get("spec"), dict) or not isinstance(wave.get("findings"), list):
                raise ValueError("PLAN_REVIEW_STATE_INVALID: full wave needs spec and findings")
            if not isinstance(wave.get("dispositions", []), list):
                raise ValueError("PLAN_REVIEW_STATE_INVALID: dispositions must be a list")
        if "author_disposition" in wave and validate_author_disposition(
            wave["author_disposition"], subject_hash=fingerprint,
        ) is None:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: author_disposition is malformed or stale")
        seen.add(fingerprint)
    cycles_paid = value.get("cycles_paid", 0)
    if not isinstance(cycles_paid, int) or isinstance(cycles_paid, bool) or cycles_paid < 0:
        raise ValueError("PLAN_REVIEW_STATE_INVALID: cycles_paid must be a non-negative count")
    if not isinstance(value.get("need_evidence_seen", []), list):
        raise ValueError("PLAN_REVIEW_STATE_INVALID: need_evidence_seen must be a list")
    copied = {**_empty_plan_review_state(), **copy.deepcopy(value)}
    attempt = copied.get("current_attempt")
    if not isinstance(attempt, dict):
        raise ValueError("PLAN_REVIEW_STATE_INVALID: current_attempt must be an object")
    if attempt:
        if set(attempt) - {"fingerprint", "status", "reason", "author_subject", "submitted_subject"} or not {"fingerprint", "status", "reason"} <= set(attempt):
            raise ValueError("PLAN_REVIEW_STATE_INVALID: current_attempt shape is invalid")
        if not _PLAN_REVIEW_HASH_RE.fullmatch(str(attempt.get("fingerprint") or "")):
            raise ValueError("PLAN_REVIEW_STATE_INVALID: current attempt fingerprint is invalid")
        if str(attempt.get("status") or "") not in _PLAN_REVIEW_ATTEMPT_STATUSES:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: current attempt status is invalid")
        if "submitted_subject" in attempt and not isinstance(attempt["submitted_subject"], dict):
            raise ValueError("PLAN_REVIEW_STATE_INVALID: submitted subject source is invalid")
        if "author_subject" in attempt:
            subject = attempt["author_subject"]
            if (not isinstance(subject, dict) or not isinstance(subject.get("source_ref"), dict)
                    or not _PLAN_REVIEW_HASH_RE.fullmatch(str(subject.get("review_fingerprint") or ""))
                    or validate_author_disposition(subject.get("author_disposition"), subject_hash=attempt["fingerprint"]) is None):
                raise ValueError("PLAN_REVIEW_STATE_INVALID: current author subject is invalid")
        if len(str(attempt.get("reason") or "")) > _PLAN_REVIEW_REASON_MAX_CHARS:
            raise ValueError("PLAN_REVIEW_STATE_INVALID: current attempt reason is too large")
    if len(json.dumps(copied, ensure_ascii=False, default=str).encode("utf-8")) > _PLAN_REVIEW_STATE_MAX_BYTES:
        raise ValueError("PLAN_REVIEW_STATE_INVALID: state exceeds the bounded size limit")
    return copied


def load_plan_review_state(results_drive_root: Any, task_id: str) -> Dict[str, Any]:
    """Read bounded state, then resolve the full specs used by decision consumers."""
    result = read_document(results_drive_root, validate_task_id(task_id))
    if result is None:
        return _empty_plan_review_state()
    from ouroboros.tools.plan_review_artifacts import authority_state

    return authority_state(
        results_drive_root, task_id, _validated_plan_review_state(result.get(PLAN_REVIEW_STATE_KEY)),
    )


def plan_review_wave(state: Dict[str, Any], fingerprint: str) -> Optional[Dict[str, Any]]:
    """The wave recorded under ``fingerprint`` (full or compact), as a private copy."""
    for wave in (state or {}).get("waves") or []:
        if str(wave.get("request_fingerprint") or "") == fingerprint:
            return copy.deepcopy(wave)
    return None


def current_plan_review_wave(state: Any) -> Optional[Dict[str, Any]]:
    """Copy the current fingerprint's wave; only an absent fingerprint selects the latest."""
    if not isinstance(state, dict):
        return None
    attempt = state.get("current_attempt") if isinstance(state.get("current_attempt"), dict) else {}
    fingerprint = str(attempt.get("fingerprint") or "")
    if fingerprint:
        return plan_review_wave(state, fingerprint)
    waves = state.get("waves") if isinstance(state.get("waves"), list) else []
    return copy.deepcopy(waves[-1]) if waves else None


def _compact_plan_review_wave(wave: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.tools.plan_review_artifacts import compact_wave

    return compact_wave(wave)


def plan_review_authority_core(state: Dict[str, Any], *, source_ref: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Project plan authority through the exact-wave companion."""
    from ouroboros.tools.plan_review_artifacts import plan_review_authority_core as project

    return project(state, source_ref=source_ref)

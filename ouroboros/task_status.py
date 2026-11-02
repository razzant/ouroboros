"""Effective task status helpers shared by tools and gateways.

Only pooled execution owners enter worker-boot orphan inference; direct,
presence and unknown owners are never inferred dead from a later worker boot.
"""

from __future__ import annotations

import logging
import pathlib
from functools import lru_cache, partial
import time
from datetime import datetime, timezone
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, List, Optional

from ouroboros.observability import timed_phase
from ouroboros.finalization_timing import cached_finalization_children
from ouroboros.headless import (
    ARTIFACT_STATUS_FAILED,
    ARTIFACT_STATUS_FINALIZING,
    ARTIFACT_STATUS_MISSING,
    ARTIFACT_STATUS_PENDING,
    ARTIFACT_STATUS_READY as ARTIFACT_STATUS_READY,
    ARTIFACT_TERMINAL_STATUSES,
    task_requires_workspace_capture,
)
from ouroboros.outcomes import (
    EXECUTION_FAILED,
    EXECUTION_INFRA_FAILED,
    OBJECTIVE_FAIL,
    infra_failed_axes,
    normalize_outcome_axes,
)
from ouroboros.budget_pause import LIVE_PAUSE_STATES
from ouroboros.post_task_checkpoint import (
    _TERMINAL_ACCOUNTING_SCRUB_FIELDS,
    project_replica_task_result_fields,
)
from ouroboros.task_store import select_lineage
from ouroboros.task_results import (
    STATUS_CANCEL_REQUESTED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_REJECTED_DUPLICATE,
    STATUS_REQUESTED,
    STATUS_RUNNING,
    STATUS_SCHEDULED,
    cancellation_blocks_child_result,
    load_task_result,
    validate_task_id,
)
from ouroboros.queue_observation import load_queue_document, queue_stamp
from ouroboros.utils import iter_jsonl_objects

log = logging.getLogger(__name__)


# Terminal task statuses. Since the cancel redesign (Poltergeist sprint phase A)
# the ``cancel_requested`` latch is GONE from this set: cancel intent is a
# durable ``cancel_intents`` projection row, never a status value, so
# terminality has exactly one definition again. ``FINAL_STATUSES`` and
# ``SETTLED_STATUSES`` are now the same set; both names stay exported because
# the split documented two historical semantics ("terminal for handoff" vs
# "truly settled") and consumers of either must agree from here on.
FINAL_STATUSES: frozenset[str] = frozenset({
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED,
    STATUS_REJECTED_DUPLICATE,
})
NONTERMINAL_STATUSES: frozenset[str] = frozenset({
    STATUS_REQUESTED,
    STATUS_SCHEDULED,
    STATUS_RUNNING,
    # Transient pre-requeue marker written by the reaper/update teardown before a
    # retry is enqueued (formalized in the lifecycle projection, phase A.11).
    STATUS_INTERRUPTED,
})
SETTLED_STATUSES: frozenset[str] = FINAL_STATUSES
ARTIFACT_NONTERMINAL_STATUSES: frozenset[str] = frozenset({
    ARTIFACT_STATUS_PENDING,
    ARTIFACT_STATUS_FINALIZING,
})
_ORPHAN_RUNNING_GRACE_SECONDS = 30.0
_ARTIFACT_LIFECYCLE_FIELDS: frozenset[str] = frozenset({
    "artifact_status",
    "artifact_error",
    "artifact_bundle",
    "artifact_finalized_at",
})


def _outcome_execution_status(result: Dict[str, Any]) -> str:
    axes = normalize_outcome_axes(result)
    execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else {}
    return str(execution.get("status") or "").strip().lower()


def _outcome_objective_status(result: Dict[str, Any]) -> str:
    axes = normalize_outcome_axes(result)
    objective = axes.get("objective") if isinstance(axes.get("objective"), dict) else {}
    return str(objective.get("status") or "").strip().lower()


def _terminal_failure_from_outcome(result: Dict[str, Any]) -> bool:
    status = str(result.get("status") or "").strip().lower()
    if status == STATUS_CANCELLED:
        return True
    if status == STATUS_FAILED:
        return True
    execution = _outcome_execution_status(result)
    objective = _outcome_objective_status(result)
    if execution in {EXECUTION_FAILED, EXECUTION_INFRA_FAILED}:
        return True
    return objective == OBJECTIVE_FAIL


def _fail_nonterminal_artifact_bundle(bundle: Dict[str, Any], message: str) -> Dict[str, Any]:
    updated = dict(bundle or {})
    updated["status"] = ARTIFACT_STATUS_FAILED
    errors = list(updated.get("errors") or []) if isinstance(updated.get("errors"), list) else []
    if message not in errors:
        errors.append(message)
    updated["errors"] = errors
    artifacts = updated.get("artifacts")
    if isinstance(artifacts, list):
        patched_artifacts = []
        for artifact in artifacts:
            if isinstance(artifact, dict):
                item = dict(artifact)
                if str(item.get("status") or "").strip().lower() in ARTIFACT_NONTERMINAL_STATUSES:
                    item["status"] = ARTIFACT_STATUS_FAILED
                    item_errors = list(item.get("errors") or []) if isinstance(item.get("errors"), list) else []
                    if message not in item_errors:
                        item_errors.append(message)
                    item["errors"] = item_errors
                patched_artifacts.append(item)
            else:
                patched_artifacts.append(artifact)
        updated["artifacts"] = patched_artifacts
    return updated


def _child_drive_candidates(result: Dict[str, Any]) -> List[pathlib.Path]:
    from ouroboros.task_custody import child_drive_candidates  # the one child-root vocabulary

    return child_drive_candidates(result)


# Legacy mirrored disposition/sha fields stripped from every effective read; the
# typed task-tree ledger row is the sole disposition authority (re-projected below
# only on full `materialize_artifacts=True` reads).
_CHILD_DISPOSITION_FIELDS = (
    "child_result_disposition",
    "child_result_disposition_sha256",
    "child_result_disposition_reason",
    "child_result_disposition_source",
    "child_result_disposition_beacon_state",
    "child_result_disposition_beacon_sha256",
    "parent_decision_child_result_sha256",
    "terminal_child_result_snapshot",
)


def _project_child_result_disposition(
    drive_root: pathlib.Path,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive the current exact-hash disposition from the task-tree ledger.

    The raw task result is never a disposition authority. Legacy mirrored fields
    are removed from the effective read before the sole typed decision row is
    projected for existing consumers.
    """

    projected = dict(result)
    for field in _CHILD_DISPOSITION_FIELDS:
        projected.pop(field, None)
    try:
        from ouroboros.task_tree_ledger import child_result_disposition_row
        from ouroboros.tools.join_ledger import (
            _child_disposition_lineage,
            _child_result_sha256,
        )

        root_task_id, parent_task_id, child_task_id = _child_disposition_lineage(projected)
        if not all((root_task_id, parent_task_id, child_task_id)):
            return projected
        semantic_hash = _child_result_sha256(projected)
        row = child_result_disposition_row(
            root_task_id,
            parent_task_id,
            child_task_id,
            semantic_hash,
            data_root=pathlib.Path(drive_root),
        )
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if not payload:
            return projected
        projected.update(
            child_result_disposition=str(payload.get("disposition") or ""),
            child_result_disposition_sha256=semantic_hash,
            child_result_disposition_reason=str(row.get("text") or ""),
            child_result_disposition_source="task_tree_ledger",
        )
    except Exception:
        return projected
    return projected


def _load_queue_snapshot(drive_root: pathlib.Path) -> Dict[str, Any]:
    return load_queue_document(drive_root)


# GR7-1a freshness bound for the live-ownership twin. The supervisor persists
# queue.alive on every main-loop pass (nominally 0.5s apart), so a stamp
# this old means the writer is gone or badly wedged — the twin can no longer
# trust it to prove a DEAD worker. Sized well above 2× the nominal tick to
# absorb ordinary loop hitches (task_done processing, git ops) without turning
# every cancel into a fail-open pass.
_SNAPSHOT_OWNERSHIP_FRESH_SEC = 10.0


def queue_snapshot_observation(snapshot: Dict[str, Any], *, boot: bool = False) -> Dict[str, Any]:
    """Date the existing queue observation, without asserting current capacity.

    Context, scheduling receipts and cancellation observations share the same
    freshness bound as live ownership. Counts belong to the captured snapshot;
    even a fresh observation is not a reservation or a continuously live view.
    """
    stamp = queue_stamp(snapshot, boot=boot)
    observation = {"source": "state/queue_snapshot.json", "ts": stamp.get("ts"),
                   "age_sec": None, "freshness": "unknown", "fresh": False}
    if snapshot.get("_snapshot_missing") or snapshot.get("_snapshot_invalid"):
        return observation
    raw = str(stamp.get("ts") or "").strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
        stamped = (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
    except (TypeError, ValueError, OverflowError, OSError):
        return observation
    age = time.time() - stamped
    fresh = age <= _SNAPSHOT_OWNERSHIP_FRESH_SEC
    return {**observation, "age_sec": round(age, 3), "fresh": fresh,
            "freshness": "fresh" if fresh else "stale"}


def _snapshot_is_stale(snapshot: Dict[str, Any]) -> bool:
    """Missing, unreadable or old observations cannot prove a dead worker."""
    return not queue_snapshot_observation(snapshot)["fresh"]


def execution_owner_record(drive_root: Any, task: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """Bind the actual execution class to its existing task-result attempt.

    Assignment replaces this fact when an inline turn resumes in the pool; it
    is ownership provenance, never a process-death or replay authorization. A
    direct turn lives only inside the server process that runs it, so its record
    also names that process generation (``direct_roots.owner_process_ended``). A
    Presence turn runs on a thread of whichever process admitted it (the Host, or
    an initiating tool's worker): its record names that controller, so its end is
    a fact rather than a guess (``presence_recovery.attempt_ended``).
    """
    record = {"kind": kind, "data_root": str(pathlib.Path(drive_root).resolve()),
              "task_id": str(task.get("id") or task.get("task_id") or ""),
              "task_attempt": int(task.get("_attempt") or task.get("task_attempt") or 0)}
    if kind == "direct":
        from ouroboros.process_custody import current_custody_session_id

        record["process"] = current_custody_session_id()
    elif kind == "presence":
        from ouroboros.review_operation import controller_identity

        record["controller"] = controller_identity()
    return record


@lru_cache(maxsize=64)
def _resolved_root(root: str) -> pathlib.Path:
    """A data root's resolved path, once per process: a page observes each of its tasks, and
    resolving walks every component of the path (an ``lstat`` each, a GIL hand-off each)."""
    return pathlib.Path(root).resolve()


def task_execution_observation(
    drive_root: Any, result: Dict[str, Any], *, queue_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Recorded ownership and a separately dated positive live observation.

    Missing inline evidence is unknown, never pooled death. Local observations
    are root-bound. The direct fragment lists addressable turns, not every live
    stack, so absence cannot settle a task. No observation authorizes replay.
    ``queue_ownership`` is the queue's custody of a row, not admission to run: the
    effective read adds a PENDING row's ``dispatch`` from its ``current_stop``.
    """
    root = _resolved_root(str(drive_root))
    tid = str(result.get("task_id") or result.get("id") or "")
    owner = result.get("execution_owner")
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    kind, reason = "unknown", "execution_owner_unavailable"
    if isinstance(owner, dict):
        if (owner.get("data_root") == str(root) and owner.get("task_id") == tid
                and owner.get("task_attempt") == (result.get("task_attempt") or result.get("_attempt") or 0)
                and owner.get("kind") in {"pooled", "direct", "presence"}):
            kind, reason = owner["kind"], "live_execution_unobserved"
        else:
            reason = "execution_owner_binding_mismatch"
    elif owner is None:
        # A consumed exact Resume proves the pool took over, unlike origin.
        pause = result.get("budget_pause") if isinstance(result.get("budget_pause"), dict) else {}
        grant = pause.get("grant") if isinstance(pause.get("grant"), dict) else {}
        if pause.get("state") == "resumed" or grant.get("consumed_at"):
            kind = "pooled"
        elif result.get("_is_direct_chat") or metadata.get("source") == "presence":
            kind = "presence" if isinstance(metadata.get("presence"), dict) else "direct"
        else:
            kind = "pooled"  # legacy managed-task rows keep their existing custody path
        reason = "legacy_execution_owner"
    observation = {"kind": kind, "state": "unknown", "source": "task_result", "reason": reason}
    if str(result.get("status") or "").lower() in FINAL_STATUSES:
        return {**observation, "state": "terminal", "reason": "recorded_terminal"}
    if kind == "unknown":
        return observation
    snapshot = queue_snapshot if queue_snapshot is not None else _load_queue_snapshot(root)
    queue_seen = queue_snapshot_observation(snapshot)
    queued, _task = _queue_task_status(snapshot, tid)
    if queue_seen["fresh"] and queued in {STATUS_RUNNING, STATUS_SCHEDULED}:
        return {**observation, **queue_seen, "kind": "pooled", "state": "active",
                "reason": "queue_ownership", "status": queued}
    try:
        from ouroboros.presence_runner import presence_turn_is_live
        from supervisor.active_activity import get_direct_activity_registry

        if presence_turn_is_live(tid, drive_root=root):
            return {**observation, "kind": "presence", "state": "active", "source": "local_presence", "reason": "local_execution"}
        entry = get_direct_activity_registry().get(tid)
        actor_root = getattr(getattr(getattr(entry, "actor", None), "env", None), "drive_root", None)
        if actor_root is not None and _resolved_root(str(actor_root)) == root:
            return {**observation, "kind": "direct", "state": "active", "source": "local_direct", "reason": "local_execution"}
    except Exception:
        pass  # absence of a positive witness is never a negative observation
    fragment = load_queue_document(root, "direct_roots.json")
    # Both existing supervisor fragments use the same 10s ownership freshness
    # (peer_roster._projection_observation); retain the distinct source name.
    direct_seen = {**queue_snapshot_observation(fragment), "source": "state/direct_roots.json"}
    rows = fragment.get("roots")
    if direct_seen["fresh"] and isinstance(rows, list) and any(
        isinstance(row, dict) and str(row.get("task_id") or "") == tid for row in rows
    ):
        return {**observation, **direct_seen, "kind": "direct", "state": "active", "reason": "listed_direct"}
    return {**observation, "queue_snapshot": queue_seen,
            "direct_roots": {**direct_seen, "incomplete": bool(fragment.get("incomplete"))}}


def task_has_live_queue_ownership(drive_root: pathlib.Path, task_id: str) -> bool:
    """Worker-side half of the GR6-1 live-ownership predicate.

    The agent's cancel ingress runs in the worker process, where the
    supervisor's live maps are unreachable — the queue snapshot is the durable
    projection of the same fact. A RUNNING row means live physical ownership:
    the durable terminal result is persisted BEFORE post-task cognition ends,
    so a settled status alone must never make the cancel tool no-op while the
    worker keeps spending (``already_settled`` is terminal only when no live
    ownership remains).

    Fail-OPEN toward liveness (GR7-1a): a MISSING, unreadable, or STALE
    snapshot cannot prove the worker is dead, so it answers True (assume
    live). The asymmetry decides the polarity — a false "live" costs one
    custody pass that finds a dead process and takes the fast
    already-settled path; a false "dead" makes the cancel ingress no-op
    ("Nothing to cancel", no durable intent) while the worker keeps burning.
    Only a FRESH snapshot that positively lacks a RUNNING row answers False.
    """
    try:
        snapshot = _load_queue_snapshot(pathlib.Path(drive_root))
        if snapshot.get("_snapshot_missing") or snapshot.get("_snapshot_invalid"):
            return True
        if _snapshot_is_stale(snapshot):
            return True
        status, _task = _queue_task_status(snapshot, str(task_id or "").strip())
        return status == STATUS_RUNNING
    except Exception:
        return True


def task_subtree_has_live_queue_rows(drive_root: pathlib.Path, task_id: str) -> bool:
    """Worker-side twin of the owner Stop's live-subtree check for a settled root.

    A RUNNING or PENDING snapshot row whose ``root_task_id`` is the target is a
    live member of its tree (the target's own row is the ownership half above):
    an ended root's helpers still spend. Fail-OPEN exactly like
    ``task_has_live_queue_ownership``: only a FRESH snapshot without such a row
    answers False.
    """
    try:
        snapshot = _load_queue_snapshot(pathlib.Path(drive_root))
        if snapshot.get("_snapshot_missing") or snapshot.get("_snapshot_invalid") or _snapshot_is_stale(snapshot):
            return True
        target = str(task_id or "").strip()
        return any(isinstance(row, dict) and isinstance(row.get("task"), dict)
                   and str(row["task"].get("root_task_id") or "") == target
                   and str(row.get("id") or row["task"].get("id") or "") != target
                   for lane in ("running", "pending") for row in snapshot.get(lane) or [])
    except Exception:
        return True


def _queue_task_status(snapshot: Dict[str, Any], task_id: str) -> tuple[str, Dict[str, Any]]:
    if snapshot.get("_snapshot_missing") or snapshot.get("_snapshot_invalid"):
        return "unknown", {}
    for row in snapshot.get("running") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("id") or row.get("task_id") or "") == task_id:
            task = row.get("task") if isinstance(row.get("task"), dict) else {}
            return STATUS_RUNNING, task
    for row in snapshot.get("pending") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("id") or row.get("task_id") or "") == task_id:
            task = row.get("task") if isinstance(row.get("task"), dict) else {}
            return STATUS_SCHEDULED, task
    return "", {}


def observe_cancellation_target(
    drive_root: Any, task_id: str, *, include_execution: bool = False,
    request_origin: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read separate source facts about the resolved target, never cancel authority.

    The stored result, queue snapshot and custody rows are separate observations;
    their own timestamps/freshness survive. A failed liveness probe is UNKNOWN,
    never the fail-open ownership hint used to admit an actual cancellation.
    """
    from ouroboros.cancel_intents import _validated_single_cancel_target
    from ouroboros.cost_projection import cost_projection
    from ouroboros.utils import utc_now_iso

    observation: Dict[str, Any] = {"observed_at": utc_now_iso(), "requested_task_id": task_id,
                                   "request_origin": dict(request_origin or {})}
    try:
        target = _validated_single_cancel_target(drive_root, task_id)
    except Exception:
        return {**observation, "observed_task_id": None, "status": "retry_target_unavailable"}
    observation["observed_task_id"] = target
    try:
        record = load_task_result(drive_root, target, strict=True) or {}
        observation["task_result"] = {"status": record.get("status"), "updated_at": record.get("updated_at"),
                                      "started_at": record.get("started_at"), "cost": cost_projection(record)}
    except Exception:
        observation["task_result"] = {"status": None, "coverage": "unavailable"}
    try:
        snapshot = _load_queue_snapshot(pathlib.Path(drive_root))
        queue_observation = queue_snapshot_observation(snapshot)
        fresh = queue_observation["fresh"]
        state, _task = _queue_task_status(snapshot, target) if fresh else ("unknown", {})
        observation["queue_snapshot"] = {**queue_observation, "status": state or "not_listed"}
    except Exception:
        observation["queue_snapshot"] = {**queue_snapshot_observation({}), "status": "unknown"}
    if include_execution:
        try:
            from ouroboros.delegate_evidence import task_execution_evidence

            evidence = task_execution_evidence(drive_root, target)
            keys = ("delegated_runs_started", "delegated_runs_settled", "delegated_runs_succeeded",
                    "delegated_runs_failed", "subscription_cost_usd", "subscription_cost_estimated",
                    "subscription_cost_partial")
            observation["delegated_execution"] = ({"evidence_read_failed": True} if evidence.get("evidence_read_failed")
                                                   else {key: evidence[key] for key in keys if key in evidence})
        except Exception:
            observation["delegated_execution"] = {"evidence_read_failed": True}
    return observation


class _EventsTailIndex:
    """One lazily-parsed events.jsonl tail shared across a batch of orphan checks.

    ``_is_stale_orphan_running_task`` needs two facts from the same 2MB events
    tail: the latest event ts per task id and the latest ``worker_boot`` ts.
    Reading that tail per RUNNING row made a task-list request over N stale
    running rows pay N full tail parses (v6.9x P2, review fix GPT#8). One index
    instance parses the tail at most once — and not at all when no caller ever
    consults it — and answers every row from memory. The instance is scoped to
    a single request/batch; it is never cached across requests (the tail moves)."""

    def __init__(self, drive_root: pathlib.Path) -> None:
        self._drive_root = pathlib.Path(drive_root)
        self._latest_by_task: Optional[Dict[str, float]] = None
        self._worker_boot = 0.0

    def _ensure_parsed(self) -> None:
        if self._latest_by_task is not None:
            return
        latest: Dict[str, float] = {}
        for event in iter_jsonl_objects(self._drive_root / "logs" / "events.jsonl", tail_bytes=2_000_000):
            try:
                parsed = datetime.fromisoformat(str(event.get("ts") or "").strip().replace("Z", "+00:00"))
                ev_ts = float((parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp())
            except Exception:
                ev_ts = 0.0
            event_task_id = str(event.get("task_id") or "")
            if not event_task_id and isinstance(event.get("task"), dict):
                event_task_id = str((event.get("task") or {}).get("id") or "")
            if event_task_id:
                latest[event_task_id] = max(latest.get(event_task_id, 0.0), ev_ts)
            if str(event.get("type") or "") == "worker_boot":
                self._worker_boot = max(self._worker_boot, ev_ts)
        self._latest_by_task = latest

    def latest_event_ts(self, task_id: str) -> float:
        self._ensure_parsed()
        return float((self._latest_by_task or {}).get(str(task_id), 0.0))

    def latest_worker_boot(self) -> float:
        self._ensure_parsed()
        return self._worker_boot


def _still_orphan_at_write(drive_root: pathlib.Path, task_id: str, observed: tuple, applied: List[bool], existing: Dict[str, Any],
                           fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Projector for the reconciler's terminal write (bound with ``partial``): the decision was taken
    outside the row lock, so a presence retry that went live meanwhile, or a row that moved on (requeued,
    restarted as a new attempt even under the same status, settled by another writer), cancels the write:
    ``observed`` is the row's attempt basis plus ``updated_at`` at decision time. ``applied`` gets True
    only on a real write."""
    from ouroboros.task_custody import attempt_basis

    execution = task_execution_observation(drive_root, existing)
    inferred_orphan = fields.get("reason_code") in {"orphaned_running_after_worker_restart", "interrupted_retry_lost"}
    if execution["state"] == "active" or (inferred_orphan and execution["kind"] != "pooled"):
        return None
    if (attempt_basis(existing), existing.get("updated_at")) != observed \
            or str(existing.get("status") or "").lower() not in {STATUS_RUNNING, STATUS_INTERRUPTED}:
        return None
    applied.append(True)
    return fields


def _is_stale_orphan_running_task(
    drive_root: pathlib.Path,
    task_id: str,
    result: Dict[str, Any],
    events_index: Optional[_EventsTailIndex] = None,
    queue_snapshot: Optional[Dict[str, Any]] = None,
    *, boot: bool = False,
) -> bool:
    observation = task_execution_observation(
        drive_root, {**result, "task_id": task_id}, queue_snapshot=queue_snapshot,
    )
    if observation["state"] == "active" or observation["kind"] != "pooled":
        return False
    pause = result.get("budget_pause") if isinstance(result.get("budget_pause"), dict) else {}
    if pause.get("state") in LIVE_PAUSE_STATES:
        # Unconsumed Resume belongs to pause custody, which revokes/reparks it
        # after confirmed worker death. A generic orphan must not spend that grant.
        return False
    status = str(result.get("status") or "").lower()
    # ``interrupted`` is the transient pre-requeue marker (A.11): a record still
    # carrying it with no queued retry after a worker restart is the same orphan
    # class as a stale ``running`` row and reconciles the same way.
    if status not in {STATUS_RUNNING, STATUS_INTERRUPTED}:
        return False
    if status == STATUS_RUNNING and isinstance(result.get("outcome_axes"), dict):
        return False
    if status == STATUS_INTERRUPTED and str(
        result.get("superseded_by") or result.get("retry_task_id") or ""
    ).strip():
        # A named retry is followed by the retry-lineage branch above; only a
        # retry-less interrupted record can wedge.
        return False
    legacy_result_status = str(result.get("result_status") or "").strip().lower()
    if legacy_result_status:
        return False
    heartbeat = 0.0
    try:
        parsed = datetime.fromisoformat(str(result.get("ts") or result.get("started_at") or result.get("created_at") or "").strip().replace("Z", "+00:00"))
        heartbeat = float((parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp())
    except Exception:
        pass
    if heartbeat and time.time() - heartbeat < _ORPHAN_RUNNING_GRACE_SECONDS:
        return False
    # A stale/missing snapshot cannot prove that a pooled owner is gone (the
    # GR7-1a polarity ``task_has_live_queue_ownership`` already uses).  The
    # destructive reconciler must keep the old row until a fresh snapshot or a
    # separate positive recovery fact exists.  A batch caller passes the
    # snapshot it already read, like ``events_index`` above.
    try:
        snapshot = (
            queue_snapshot if isinstance(queue_snapshot, dict)
            else _load_queue_snapshot(pathlib.Path(drive_root))
        )
        if not queue_snapshot_observation(snapshot, boot=boot)["fresh"]:
            return False
    except Exception:
        return False
    if events_index is None:
        events_index = _EventsTailIndex(pathlib.Path(drive_root))
    latest_task_event = max(heartbeat, events_index.latest_event_ts(task_id))
    latest_worker_boot = events_index.latest_worker_boot()
    return bool(latest_worker_boot and latest_worker_boot > latest_task_event)


PRESENCE_ATTEMPT_LOST = "presence_attempt_lost"


def _project_lost_attempt(merged: Dict[str, Any], reason: str, prior_status: str, notice: str) -> None:
    """Project a non-terminal row whose executor positively ended as an infra failure, in place."""
    merged.update(status=STATUS_FAILED, reason_code=reason, outcome_axes=infra_failed_axes(reason),
                  status_reconciled_from=prior_status)
    merged["result"] = (str(merged.get("result") or "Task was interrupted before a terminal result was recorded.")
                        + "\n\n⚠️ " + notice)
    if str(merged.get("artifact_status") or "").strip().lower() in ARTIFACT_NONTERMINAL_STATUSES:
        merged["artifact_status"] = ARTIFACT_STATUS_FAILED
        bundle = dict(merged["artifact_bundle"]) if isinstance(merged.get("artifact_bundle"), dict) else {}
        merged["artifact_bundle"] = _fail_nonterminal_artifact_bundle(bundle, "task interrupted before artifact finalization")


def _presence_attempt_lost(drive_root: pathlib.Path, result: Dict[str, Any],
                           queue_snapshot: Optional[Dict[str, Any]] = None) -> bool:
    """A non-terminal row of a Presence turn whose attempt can no longer change.

    The positive end facts are one rule shared with the Host work view
    (``presence_recovery.attempt_ended``); age, absence and worker boots prove nothing.
    """
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    if (str(result.get("status") or "").lower() not in {STATUS_RUNNING, STATUS_INTERRUPTED}
            or not isinstance(metadata.get("presence"), dict)):
        return False
    observation = task_execution_observation(drive_root, result, queue_snapshot=queue_snapshot)
    if observation["kind"] != "presence" or observation["state"] == "active":
        return False
    from ouroboros.presence_recovery import attempt_ended

    return attempt_ended(result, drive_root)


def _normalize_workspace_artifact_status(result: Dict[str, Any]) -> Dict[str, Any]:
    if not task_requires_workspace_capture(result):
        return result
    status = str(result.get("status") or "").lower()
    if status not in FINAL_STATUSES:
        return result
    artifact_status = str(result.get("artifact_status") or "").strip().lower()
    if artifact_status in ARTIFACT_TERMINAL_STATUSES:
        return result
    if status == STATUS_CANCELLED:
        # LEGACY cancelled results only: the phase-A cancel path captures real
        # workspace artifacts before the settled write (A4), so a new cancelled
        # record arrives with a terminal artifact_status and never reaches here.
        normalized = dict(result)
        normalized["artifact_status"] = ARTIFACT_STATUS_MISSING
        try:
            from ouroboros.outcomes import artifact_bundle_from_result

            normalized.pop("artifact_bundle", None)
            normalized["artifact_bundle"] = artifact_bundle_from_result(normalized)
        except Exception:
            pass
        axes = normalize_outcome_axes(normalized)
        artifact_axis = dict(axes.get("artifacts") or {})
        artifact_axis["status"] = ARTIFACT_STATUS_MISSING
        axes["artifacts"] = artifact_axis
        normalized["outcome_axes"] = axes
        return normalized
    normalized = dict(result)
    normalized.setdefault("child_status", status)
    normalized["status"] = STATUS_RUNNING
    normalized["artifact_status"] = ARTIFACT_STATUS_FINALIZING
    return normalized


def stored_list_order_is_final(result: Dict[str, Any]) -> bool:
    """Whether this body's status and ts suffice for filtered-list membership.

    Derived by the store writer, never a declaration that the task cannot change.
    Retry and replica reads, workspace finalization and every nonterminal/unknown
    status still need the ordinary effective reader. A later body write rederives
    the fact; a pre-projection writer leaves NULL. No external state is consulted.
    """
    status = result.get("status")
    ts = result.get("ts")
    if status not in FINAL_STATUSES or (ts is not None and (not isinstance(ts, str) or ts != ts.strip())):
        return False
    if result.get("superseded_by") or result.get("retry_task_id"):
        return False
    if not cancellation_blocks_child_result(result) and _child_drive_candidates(result):
        return False
    return _normalize_workspace_artifact_status(result).get("status") == status


def _parent_workspace_artifact_lifecycle_fields(result: Dict[str, Any]) -> frozenset[str]:
    if not task_requires_workspace_capture(result):
        return frozenset()
    artifact_status = str(result.get("artifact_status") or "").strip().lower()
    if artifact_status in ARTIFACT_TERMINAL_STATUSES or artifact_status in ARTIFACT_NONTERMINAL_STATUSES:
        return _ARTIFACT_LIFECYCLE_FIELDS
    return frozenset()


# A canonical row carrying a LIVE exact budget pause (#1196) owns its lifecycle
# and accounting planes: the forked child-drive replica is the worker's
# pre-pause ``running`` row, so it must neither resurrect ``running`` nor blank
# the ledger-derived cost fields the park wrote onto the canonical row.
_LIVE_PAUSE_CANONICAL_FIELDS = frozenset({
    "status", "reason_code", "resource_limit", "result", "error", "ts", "outcome_axes",
    "budget_pause", *_TERMINAL_ACCOUNTING_SCRUB_FIELDS,
})


def _budget_pause_marker(queue_task: Any) -> bool:
    """A TYPED budget-pause marker on a PENDING row: exact (``exact_continuation``
    is a bool, True for an exact mid-run pause, False for a pre-dispatch hold).
    ``{}`` or a malformed dict is no marker, so it cannot defeat the ordinary
    requeue race where the running mirror wins."""
    pause = queue_task.get("_budget_pause") if isinstance(queue_task, dict) else None
    return isinstance(pause, dict) and isinstance(pause.get("exact_continuation"), bool)


def _merge_queue_status(
    current_status: str, queue_status: str, queue_task: Optional[Dict[str, Any]] = None,
) -> str:
    current = str(current_status or "").lower()
    queued = str(queue_status or "").lower()
    if not queued or current in FINAL_STATUSES:
        return current
    if current == STATUS_RUNNING and queued == STATUS_SCHEDULED and not _budget_pause_marker(queue_task):
        # A PENDING row parked under a budget-pause marker is not running,
        # whatever a stale mirror says (#1196); an ordinary PENDING row beside a
        # ``running`` mirror is the requeue race the running mirror wins.
        return current
    return queued


def load_effective_task_result(
    drive_root: pathlib.Path,
    task_id: str,
    *,
    materialize_artifacts: bool = True,
    reads: Any = None,
) -> Dict[str, Any]:
    try:
        tid = validate_task_id(task_id)
    except ValueError:
        return {}
    return effective_task_result(
        drive_root,
        (reads.load(drive_root, tid) if reads is not None else load_task_result(drive_root, tid)) or {},
        materialize_artifacts=materialize_artifacts,
        **({"reads": reads} if reads is not None else {}),
    )


def reconcile_orphaned_running_tasks(
    drive_root: Any, *, exclude_task_ids: frozenset[str] = frozenset(),
    expired_quizzes: Optional[List[Any]] = None,
    write_guard: Optional[Callable[[str], Any]] = None, boot: bool = True,
) -> int:
    """Durably finalize stored RUNNING task results the effective-status
    projection already considers terminal.

    A task whose worker crashed / was SIGKILLed / manually stopped can leave
    a task-store record at ``running`` forever (a misleading zombie). The
    read-time ``effective_task_result`` already projects such an orphan to a
    terminal status, but never persists it, so a headless/no-UI run that never
    re-reads the result keeps the stale ``running`` on disk.

    Boot accepts legacy snapshot timestamps; periodic callers pass ``boot=False``.
    Grace, worker-boot-after-task and readable/fresh snapshot gates still apply.
    The status-only (``materialize_artifacts=False``) projection means no child file is
    listed or copied here: the child store stays served read-only from its drive
    and reaches the canonical store through ``task_custody.settle_child_drive``
    before that drive can go. The write is fenced by the row's attempt basis and
    ``updated_at`` at decision time, and, given ``write_guard`` (the supervisor's
    queue-ownership fence, queue -> row lock order), by live queue ownership; the
    monotonic guard in ``write_task_result`` protects a newer terminal/cancel write.
    Idempotent; safe at boot and on a periodic tick.

    ``expired_quizzes`` collects ``(task_id, quiz_id)`` for every question this
    sweep expired, so the supervisor-side caller can send the same live frame the
    task-done seam sends. This module stays free of a supervisor import.
    """
    from ouroboros.task_custody import attempt_basis
    from ouroboros.task_results import write_task_result
    from ouroboros.obligations import result_rows

    root = pathlib.Path(drive_root)
    healed = 0
    try:
        running = [row for row in result_rows(root, "nonterminal", exclude=exclude_task_ids)
                   if row.get("status") in {STATUS_RUNNING, STATUS_INTERRUPTED}]
    except Exception:
        return 0
    for row in running:
        task_id = str(row.get("task_id") or row.get("id") or "")
        if not task_id or task_id in exclude_task_ids:
            continue
        # An ACTIVE cancel intent means cancellation custody already owns this
        # row and will settle it with its own outcome and text; at boot that
        # custody is still inside the watchdog's minimum age, so healing here
        # would win the race and publish infra_failed for a task the owner was
        # told is being cancelled. An UNREADABLE intent store is the same
        # refusal: this sweep never settles over an unknown cancel authority.
        try:
            from ouroboros.cancel_intents import has_active_intent

            if has_active_intent(root, task_id, strict=True):
                continue
        except Exception:
            log.debug("Orphan reconcile skipped %s: cancel authority unreadable", task_id, exc_info=True)
            continue
        try:
            effective = effective_task_result(root, row, materialize_artifacts=False, _boot=boot)
        except Exception:
            continue
        eff_status = str(effective.get("status") or "").strip().lower()
        if eff_status not in SETTLED_STATUSES:
            continue
        persist_fields = {
            key: effective[key]
            for key in (
                "result",
                "reason_code",
                "outcome_axes",
                "status_reconciled_from",
                "artifact_status",
                "artifact_bundle",
            )
            if effective.get(key) is not None
        }
        try:
            applied: List[bool] = []
            observed = (attempt_basis(row), row.get("updated_at"))
            with (write_guard(task_id) if write_guard else nullcontext(True)) as allowed:
                if not allowed:
                    continue
                # The projection is this write's replica: a child terminal lends the occurrence
                # its canonical attempt matches, as copyback would; a recovery projection has none.
                write_task_result(root, task_id, status=eff_status,
                                  _field_projector=partial(_still_orphan_at_write, root, task_id, observed, applied),
                                  _terminal_time_source=effective, **persist_fields)
                if not applied:
                    continue  # the row moved on between the effective read and this write: nothing was settled here
                healed += 1
                # This sweep is a terminal writer that never passes the task-done seam, so it closes the
                # same per-task owner-control projections that seam closes (otherwise the record says
                # "ended" while the card keeps an open question), before any next attempt can open new
                # ones. Both legs are idempotent and fail-soft, as in the seam's own coordinator.
                try:
                    from ouroboros.owner_hurry import reconcile_terminal as reconcile_hurry

                    reconcile_hurry(root, task_id)
                except Exception:
                    log.debug("owner_hurry reconcile failed for healed %s", task_id, exc_info=True)
                try:
                    from ouroboros.owner_quiz import reconcile_terminal as reconcile_quiz

                    expired = reconcile_quiz(root, task_id)
                    if expired_quizzes is not None:
                        expired_quizzes.extend((task_id, quiz_id) for quiz_id in expired)
                except Exception:
                    log.debug("owner_quiz reconcile failed for healed %s", task_id, exc_info=True)
            if persist_fields.get("reason_code") == PRESENCE_ATTEMPT_LOST:
                # Outside the write guard: the room learns once that this turn ended unanswered.
                from ouroboros.presence_runner import _notify_unresolved_turn

                _notify_unresolved_turn(root, task_id)
        except Exception:
            continue
    return healed


def _apply_cancel_state_projection(
    drive_root: pathlib.Path,
    requested_task_id: str,
    merged: Dict[str, Any],
    reads: Any = None,
) -> None:
    """Overlay durable cancel intent on one effective, non-authoritative view.

    The stable historical handle wins (cascade intents intentionally stay on
    that logical root); a canonical SINGLE intent may instead live on the
    physical timeout-retry leaf.  Both the ordinary path and the early
    retry-follow return must use this same ordering.
    """
    try:
        eff_status = str(merged.get("status") or "").strip().lower()
        if eff_status == STATUS_CANCEL_REQUESTED:
            merged.setdefault("cancel_state", "pending")
            return
        if eff_status in SETTLED_STATUSES:
            return
        from ouroboros.cancel_intents import cancel_state_fields

        intents = reads.cancel_intents(drive_root) if reads is not None else None
        cancel_fields = cancel_state_fields(
            pathlib.Path(drive_root), requested_task_id, intents=intents,
        )
        if not cancel_fields:
            physical_task_id = str(merged.get("task_id") or "").strip()
            if physical_task_id and physical_task_id != requested_task_id:
                cancel_fields = cancel_state_fields(
                    pathlib.Path(drive_root), physical_task_id, intents=intents,
                )
        merged.update(cancel_fields)
    except Exception:
        pass


def effective_task_result(
    drive_root: pathlib.Path,
    result: Dict[str, Any],
    *,
    materialize_artifacts: bool = True,
    _seen: frozenset[str] = frozenset(),
    _events_index: Optional[_EventsTailIndex] = None,
    _boot: bool = False,
    reads: Any = None,
) -> Dict[str, Any]:
    """Merge parent result, child-drive result, and active queue state.

    Every read is pure (TZ-1 A): no file copy, hash, registration or manifest
    write. Child files reach the canonical store only through copy-back and
    ``task_custody.settle_child_drive``. The default read adds the artifact VIEW
    (``task_custody.store_artifact_view``: recorded canonical rows, the task's
    OWN child stores' recorded rows and stat-only ``measured: False`` listings)
    plus the task-tree disposition lookup. ``materialize_artifacts=False`` (the
    historical name) is the "status/cost projection only" read: no view, no
    disposition lookup and no sha-bearing/disposition claims; its ``artifacts``
    are the raw recorded rows. Read-only display surfaces (chat history
    annotation, ``api_tasks_list``, the SSE follow loop, ``api_logs_tail``
    discovery) pass ``False``; child-result consumers (join_ledger,
    wait_*/get_task_result, api_task_get/artifact, prune) keep the ``True``
    default. The orphan reconciler decides and persists on a ``False`` read.
    ``_events_index`` optionally shares ONE parsed events-tail across a batch of
    rows (the task-list request), so N stale-running rows cost one tail read
    instead of N; ``None`` keeps the per-call read for single-row callers.
    ``reads`` (``task_reads.TaskReads``) serves every document, the queue snapshot, the cancel
    intents and the events tail of a read of many tasks from its batches.
    """
    load = reads.load if reads is not None else load_task_result
    if reads is not None and _events_index is None:
        _events_index = reads.events_index

    if not result:
        return {}
    task_id = str(result.get("task_id") or result.get("id") or "").strip()
    if not task_id:
        return dict(result)
    retry_id = str(result.get("superseded_by") or result.get("retry_task_id") or "").strip()
    if retry_id and retry_id != task_id and retry_id not in _seen:
        retry_result = load(drive_root, retry_id) or {}
        if retry_result:
            effective_retry = effective_task_result(
                pathlib.Path(drive_root),
                retry_result,
                materialize_artifacts=materialize_artifacts,
                _seen=frozenset(set(_seen) | {task_id}),
                _events_index=_events_index, _boot=_boot, **({"reads": reads} if reads is not None else {}),
            )
            if effective_retry:
                merged_retry = dict(effective_retry)
                lineage = list(merged_retry.get("retry_lineage") or [])
                # #1573: the successor's answer must not hide retained outputs THIS
                # physical row published; each receipt keeps its own physical read call.
                audit = result.get("delegate_terminal_reconciliation")
                published = [dict(row) for row in ((audit.get("terminal_runs") if isinstance(audit, dict)
                                                    else None) or [])
                             if isinstance(row, dict) and any(
                                 row.get(k) for k in ("staged_output", "other_staged_outputs",
                                                      "terminal_observation", "other_terminal_observations"))]
                lineage.insert(0, {
                    "task_id": task_id,
                    "status": result.get("status"),
                    "outcome_axes": normalize_outcome_axes(result),
                    "reason_code": result.get("reason_code"),
                    "retry_task_id": retry_id,
                    **({"terminal_runs": published} if published else {}),
                })
                merged_retry["retry_lineage"] = lineage
                merged_retry.setdefault("original_task_id", task_id)
                merged_retry.setdefault("supersedes_task_id", task_id)
                if result.get("cancel_receipt") and "cancel_receipt" not in merged_retry:
                    merged_retry["cancel_receipt"] = result["cancel_receipt"]
                # The retry answer does not erase the original's unknown effects.
                # Invocation IDs and their original task/attempt attribution stay
                # factual disclosure only, never authority to repeat an operation.
                retired = result.get("retired_tool_invocations") or {}
                if retired:
                    merged_retry["retired_tool_invocations"] = {
                        **retired, **(merged_retry.get("retired_tool_invocations") or {})}
                # GR6-5b: the interrupted original's unreconciled delegated
                # runs are a fact about runs that may STILL be live — the
                # retry projection must not drop the disclosure the raw row
                # retains. Union with the retry's own list, order-preserving.
                inherited = [
                    str(rid) for rid in (result.get("delegated_runs_unreconciled") or [])
                    if str(rid)
                ]
                if inherited:
                    own = [
                        str(rid)
                        for rid in (merged_retry.get("delegated_runs_unreconciled") or [])
                        if str(rid)
                    ]
                    merged_retry["delegated_runs_unreconciled"] = own + [
                        rid for rid in inherited if rid not in own
                    ]
                _apply_cancel_state_projection(
                    pathlib.Path(drive_root), task_id, merged_retry, reads,
                )
                return merged_retry

    merged = dict(result)
    child_result: Dict[str, Any] = {}
    child_text = ""
    if not cancellation_blocks_child_result(result):
        from ouroboros.install_identity import StoreForeign

        for child_drive in _child_drive_candidates(result):
            try:
                child_result = load(child_drive, task_id) or {}
            except StoreForeign:  # another install's drive: this install keeps only its own row
                child_result = {}
            if child_result:
                child_text = str(child_drive)
                break

    if child_result:
        parent_status = str(result.get("status") or "").lower()
        child_status = str(child_result.get("status") or "").lower()
        copied_child_status = str(result.get("child_status") or "").lower()
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        result_is_workspace = bool(str(result.get("workspace_root") or "").strip() or str(metadata.get("workspace_root") or "").strip())
        copied_child_terminal = (
            result_is_workspace
            and copied_child_status in FINAL_STATUSES
            and parent_status == copied_child_status
        )
        preserve_parent_terminal = (
            (parent_status in {STATUS_FAILED, STATUS_CANCELLED, STATUS_REJECTED_DUPLICATE} and not copied_child_terminal)
            or (parent_status in FINAL_STATUSES and child_status not in FINAL_STATUSES)
        )
        preserve_parent_retry = (
            child_status not in FINAL_STATUSES
            and parent_status not in {STATUS_REQUESTED, STATUS_SCHEDULED, STATUS_RUNNING}
        )
        parent_authoritative_fields = (
            {"status", "result", "error", "ts"}
            if preserve_parent_terminal or preserve_parent_retry
            else set()
        )
        parent_authoritative_fields = parent_authoritative_fields | _parent_workspace_artifact_lifecycle_fields(result)
        # Pool assignment can rebind a paused direct turn before its old child
        # replica receives the new start. Owner and its attempt remain one fact.
        if "execution_owner" in result:
            parent_authoritative_fields = parent_authoritative_fields | {"execution_owner", "task_attempt", "_attempt"}
        canonical_pause = result.get("budget_pause") if isinstance(result.get("budget_pause"), dict) else {}
        # Only a STALE nonterminal replica (the worker's pre-pause ``running``
        # row) yields to the live pause; a replica that already reached a
        # terminal status is the child's real outcome and is never suppressed
        # by a canonical pause the copyback has not yet cleared.
        if str(canonical_pause.get("state") or "") in LIVE_PAUSE_STATES and child_status not in FINAL_STATUSES:
            parent_authoritative_fields = parent_authoritative_fields | _LIVE_PAUSE_CANONICAL_FIELDS
        child_overlay = project_replica_task_result_fields(result, child_result)
        for key, value in child_overlay.items():
            if key in {"task_id", "parent_task_id", "root_task_id", "session_id", "actor_id", "delegation_role"}:
                continue
            if key in parent_authoritative_fields:
                continue
            if key == "artifacts":
                continue
            merged[key] = value
        merged.setdefault("child_drive_root", child_text)
        merged.setdefault("headless_child_drive_root", child_text)
        metadata = merged.get("metadata") if isinstance(merged.get("metadata"), dict) else {}
        merged_is_workspace = bool(str(merged.get("workspace_root") or "").strip() or str(metadata.get("workspace_root") or "").strip())
        if merged_is_workspace and child_status in FINAL_STATUSES and (parent_status not in {STATUS_FAILED, STATUS_CANCELLED, STATUS_REJECTED_DUPLICATE} or copied_child_terminal):
            merged = _normalize_workspace_artifact_status(merged)

    merged = _normalize_workspace_artifact_status(merged)

    queue_snapshot = None
    terminal_projection = ""
    parent_status = str(merged.get("status") or "").lower()
    if parent_status not in FINAL_STATUSES:
        queue_snapshot = (reads.queue_snapshot(drive_root) if reads is not None
                          else _load_queue_snapshot(pathlib.Path(drive_root)))
        queue_status, queue_task = _queue_task_status(queue_snapshot, task_id)
        if queue_status and queue_status != "unknown":
            merged["status"] = _merge_queue_status(parent_status, queue_status, queue_task)
            from ouroboros.project_admission import project_hold_fact
            from ouroboros.work_stop import task_current_stop

            merged["project_admission_hold"] = project_hold_fact(queue_task)
            # The census's reading of the same row: a held row stays ``scheduled`` in its queue.
            merged["current_stop"] = task_current_stop(drive_root, queue_snapshot, task_id)
            for key in (
                "parent_task_id",
                "root_task_id",
                "session_id",
                "actor_id",
                "delegation_role",
                "role",
                "memory_mode",
                "drive_root",
                "child_drive_root",
                "budget_drive_root",
                "task_constraint",
            ):
                if not merged.get(key) and queue_task.get(key):
                    merged[key] = queue_task.get(key)
        else:
            if _presence_attempt_lost(pathlib.Path(drive_root), merged, queue_snapshot):
                # The queue never owns a Presence turn; its own end fact decides, as the work view does.
                terminal_projection = "presence_attempt_lost"
                _project_lost_attempt(merged, PRESENCE_ATTEMPT_LOST, parent_status, (
                    "PRESENCE_ATTEMPT_LOST: this Presence turn's executor ended without a terminal result; "
                    "whether its reply or tools took effect is unknown, and it is never re-run."))
            elif queue_status == "unknown":
                merged["queue_reconciliation_warning"] = "queue snapshot missing or invalid"
            elif _terminal_failure_from_outcome(merged):
                terminal_projection = "recorded_outcome"
                merged["status"] = STATUS_CANCELLED if str(merged.get("status") or "").strip().lower() == STATUS_CANCELLED else STATUS_FAILED
                merged["status_reconciled_from"] = parent_status
                artifact_status = str(merged.get("artifact_status") or "").strip().lower()
                if artifact_status in ARTIFACT_NONTERMINAL_STATUSES:
                    merged["artifact_status"] = ARTIFACT_STATUS_FAILED
                    bundle = dict(merged.get("artifact_bundle") or {}) if isinstance(merged.get("artifact_bundle"), dict) else {}
                    merged["artifact_bundle"] = _fail_nonterminal_artifact_bundle(
                        bundle,
                        "task ended before artifact finalization",
                    )
            elif _is_stale_orphan_running_task(
                pathlib.Path(drive_root), task_id, merged, _events_index,
                queue_snapshot=queue_snapshot, boot=_boot,
            ):
                terminal_projection = "pooled_worker_restart"
                orphan_reason = (
                    "interrupted_retry_lost"
                    if parent_status == STATUS_INTERRUPTED
                    else "orphaned_running_after_worker_restart"
                )
                _project_lost_attempt(merged, orphan_reason, parent_status, (
                    "TASK_ORPHAN_RECONCILED: queue is empty and worker restarted after this task; "
                    f"marking the stale {parent_status} task as infra_failed."))
    # Typed public cancel projection (phase A): an ACTIVE durable cancel intent
    # rides every effective read as ``cancel_state: "pending"`` — status itself
    # stays honest (running/scheduled) until the supervisor settles the teardown.
    # A legacy ``cancel_requested`` status (old files awaiting boot migration)
    # projects the same pending state.
    _apply_cancel_state_projection(pathlib.Path(drive_root), task_id, merged, reads)
    merged["execution_observation"] = task_execution_observation(drive_root, merged, queue_snapshot=queue_snapshot)
    observed = merged["execution_observation"]
    if observed.get("reason") == "queue_ownership" and observed.get("status") == STATUS_SCHEDULED:
        held = (merged.get("current_stop") or {}).get("state")
        observed["dispatch"] = "unknown" if held == "unknown" else "held" if held else "queued"
    if terminal_projection:
        # This read inferred a terminal; a later persisted terminal is a recorded
        # fact even when its historical status_reconciled_from marker survives.
        merged["execution_observation"].update(
            source="effective_task_result", reason="projected_terminal", basis=terminal_projection)

    if not materialize_artifacts:
        # Status/cost projection only: no artifact view (no store listing) and no
        # disposition lookup. Strip the legacy mirrored fields so a False row
        # never carries sha-bearing/disposition claims.
        projected = dict(merged)
        for field in _CHILD_DISPOSITION_FIELDS:
            projected.pop(field, None)
        if isinstance(projected.get("artifact_bundle"), dict) and projected["artifact_bundle"].get("status"):
            projected["artifact_status"] = normalize_outcome_axes(projected)["artifacts"]["status"]
        return projected
    try:
        from ouroboros.outcomes import artifact_bundle_from_result
        from ouroboros.task_custody import own_child_drives, store_artifact_view

        # In this artifact view, canonical admission/copy-back facts authorize child store
        # reads, never scratch existence. Keep child claims even after cancellation; the
        # view copies, hashes and registers nothing.
        child_rows = {}
        for drive in own_child_drives(drive_root, task_id, store_row=result):
            rows = (load_task_result(drive, task_id) or {}).get("artifacts")
            child_rows[drive / "task_results" / "artifacts" / task_id] = rows if isinstance(rows, list) else []
        recorded = [item for item in (merged.get("artifacts") or []) if isinstance(item, dict)]
        listed = store_artifact_view(drive_root, task_id, recorded, child_rows)
        if listed:
            merged["artifacts"] = listed
            merged["artifact_bundle"] = artifact_bundle_from_result(merged)
            merged["artifact_status"] = merged["artifact_bundle"].get("status")
    except Exception:
        log.debug("Artifact view unavailable for %s", task_id, exc_info=True)
    return _project_child_result_disposition(pathlib.Path(drive_root), merged)


def wait_for_effective_tasks(
    drive_root: pathlib.Path,
    task_ids: Iterable[str],
    *,
    timeout_sec: float,
    mode: str = "all_terminal",
    poll_interval_sec: float = 0.5,
    on_poll: Optional[Callable[[Dict[str, Any], Dict[str, bool]], Any]] = None,
    terminal_probe: Optional[Callable[[Dict[str, Any]], Dict[str, bool]]] = None,
) -> Dict[str, Any]:
    """Poll effective task results until the wait ``mode`` is satisfied.

    Terminality is ``SETTLED_STATUSES``: a wait loop's jobs is to surface the
    FINAL record, and since the phase-A cancel redesign cancel intent is a
    durable ``cancel_intents`` projection (surfaced as ``cancel_state:
    "pending"``), never a status value — the supervisor's custody settles every
    intent to a real terminal shortly after. The wait stays bounded by
    ``timeout_sec`` either way, and ``live_child_status`` reports a pending
    cancellation honestly (``cancel_pending``) instead of collapsing it to
    terminal/unknown, and a row its ``current_stop`` holds as ``held``."""
    ids = []
    for item in task_ids:
        try:
            tid = validate_task_id(item)
        except ValueError:
            tid = str(item or "").strip()
        if tid and tid not in ids:
            ids.append(tid)
    start = time.monotonic()
    deadline = start + max(0.0, float(timeout_sec or 0))
    results: Dict[str, Dict[str, Any]] = {}
    timed_out = False
    early: Any = None
    while True:
        # Poll reads are status/cancel_state projections only (no store listing
        # or disposition lookup per 2s tick). The one full read happens after the
        # loop, on every exit path, so wait/get consumers keep their place in the
        # child-result sha economy.
        results = {
            tid: load_effective_task_result(
                pathlib.Path(drive_root), tid, materialize_artifacts=False
            )
            for tid in ids
        }
        terminal = (terminal_probe(results) if terminal_probe is not None else
                    {tid: str(data.get("status") or "").strip().lower() in SETTLED_STATUSES for tid, data in results.items()})
        # Sliced wait hook: a child->parent attention beacon (including review_requested)
        # gets one preflight even when the child already terminalized, so a beacon
        # written before this wait is not hidden by the terminal fast path.
        if callable(on_poll):
            try:
                signal = on_poll(results, terminal)
            except Exception:
                signal = None
            if signal is not None:
                early = signal
                break
        if mode == "any_terminal" and any(terminal.values()):
            break
        if mode != "any_terminal" and all(terminal.values()):
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(max(0.05, min(2.0, float(poll_interval_sec or 0.5), deadline - time.monotonic())))
    # The single full read, on EVERY exit path (terminal, timeout, early
    # beacon): the returned rows re-enter the sha/artifact economy.
    results = {tid: load_effective_task_result(pathlib.Path(drive_root), tid) for tid in ids}
    out: Dict[str, Any] = {
        "mode": mode,
        "timeout_sec": float(timeout_sec or 0),
        "elapsed_sec": max(0.0, time.monotonic() - start),
        "timed_out": timed_out,
        "all_terminal": (all(terminal_probe(results).values()) if terminal_probe is not None else
                         all(str(data.get("status") or "").strip().lower() in SETTLED_STATUSES for data in results.values())) if ids else True,
        "tasks": results,
    }
    if early is not None:
        out["early_return"] = early
    # Keep the legacy status projection with the date of its queue evidence.
    # Terminal result and worker ownership can differ during post-task work;
    # a stale snapshot cannot establish what is still running now.
    try:
        _snap = _load_queue_snapshot(pathlib.Path(drive_root))
        out["queue_snapshot_observation"] = queue_snapshot_observation(_snap)
        live: Dict[str, str] = {}
        for tid in ids:
            _st, _ = _queue_task_status(_snap, tid)
            row = results.get(tid) or {}
            eff_status = str(row.get("status") or "").strip().lower()
            if str(row.get("cancel_state") or "") == "pending" and eff_status not in SETTLED_STATUSES:
                # A durable cancel intent is a real, known state — report it as a
                # typed pending cancellation, never as terminal (the settle is
                # pending) or unknown. Covers the legacy latch too (the effective
                # read projects it as cancel_state=pending).
                _st = "cancel_pending"
            elif (row.get("current_stop") or {}).get("state") == "held" and eff_status not in SETTLED_STATUSES:
                _st = "held"  # in the queue, but a current stop keeps it from running
            live[tid] = _st or ("terminal" if eff_status in SETTLED_STATUSES else "unknown")
        out["live_child_status"] = live
    except Exception:
        pass
    return out


@cached_finalization_children
@timed_phase("child_lookup")
def find_child_tasks(
    drive_root: pathlib.Path,
    *,
    parent_task_id: str = "",
    root_task_id: str = "",
    exclude_task_id: str = "",
    scope: str = "subtree",
    materialize_artifacts: bool = True,
) -> List[Dict[str, Any]]:
    """Project matching subagents, including retry lineage and live queue rows.

    Subtree includes direct children OR members of the named root; direct scope
    includes only children of parent_task_id. A leaf must not absorb siblings.
    """
    parent = str(parent_task_id or "").strip()
    root = str(root_task_id or "").strip()
    excluded = str(exclude_task_id or "").strip()
    direct_only = str(scope or "subtree").strip().lower() == "direct"

    # One queue read serves the overlay below and seeds the store's lineage closure: a queued item
    # naming the parent or root may carry its parent/role only in the queue. The store returns this
    # lineage alone (``select_lineage``) in ``task_id`` order, so the retry projection and the
    # last-row-wins fold below are deterministic. Lineage-less live rows reappear in the overlay.
    snapshot = _load_queue_snapshot(pathlib.Path(drive_root))
    queued = [(status, item, item["task"] if isinstance(item.get("task"), dict) else {})
              for group, status in (("pending", STATUS_SCHEDULED), ("running", STATUS_RUNNING))
              for item in snapshot.get(group) or [] if isinstance(item, dict)]
    members = {identity for _status, item, task in queued
               if (parent and str(task.get("parent_task_id") or "").strip() == parent)
               or (not direct_only and root and str(task.get("root_task_id") or "").strip() == root)
               for identity in (item.get("id") or item.get("task_id"), item.get("id") or task.get("id"))}
    selected = select_lineage(pathlib.Path(drive_root), parent, root, direct_only=direct_only, queue_members=members)
    events_index = _EventsTailIndex(pathlib.Path(drive_root))
    rows: Dict[str, Dict[str, Any]] = {}
    for row in (
        effective_task_result(
            pathlib.Path(drive_root), item,
            materialize_artifacts=materialize_artifacts,
            _events_index=events_index,
        )
        for item in selected
    ):
        tid = str(row.get("task_id") or "")
        if not tid or tid == excluded:
            continue
        if str(row.get("delegation_role") or "") != "subagent":
            continue
        if parent and str(row.get("parent_task_id") or "") == parent:
            rows[tid] = row
        elif not direct_only and root and str(row.get("root_task_id") or "") == root:
            rows[tid] = row

    for status, item, task in queued:
        tid = str(item.get("id") or task.get("id") or "")
        if not tid or tid == excluded:
            continue
        if str(task.get("delegation_role") or "") != "subagent":
            continue
        if parent and str(task.get("parent_task_id") or "") == parent:
            row = dict(task)
        elif not direct_only and root and str(task.get("root_task_id") or "") == root:
            row = dict(task)
        else:
            continue
        row.setdefault("task_id", tid)
        row["status"] = status
        existing = rows.get(tid, {})
        if not existing:
            # Queue lineage can recover a row skipped by the prefilter;
            # retain its disk cost, result and artifacts as well.
            existing = load_effective_task_result(
                pathlib.Path(drive_root), tid, materialize_artifacts=materialize_artifacts
            )
        if not existing:
            rows[tid] = row
            continue
        combined = dict(existing)
        for key, value in row.items():
            if key == "status":
                combined["status"] = _merge_queue_status(str(existing.get("status") or ""), str(value or ""), row)
            elif not combined.get(key) and value:
                combined[key] = value
        rows[tid] = combined
    return sorted(rows.values(), key=lambda item: (str(item.get("ts") or ""), str(item.get("task_id") or "")))


def _artifact_stat_marker(path: str) -> str:
    """GROUND-TRUTH existence fact for a child's claimed artifact path. An ABSOLUTE path
    that does not exist is flagged ⚠ MISSING (a child can report a deliverable it never
    actually wrote — the cyber-racing failure); a relative pointer is not resolvable here
    so it is marked unresolved rather than falsely missing."""
    try:
        p = pathlib.Path(path)
        # A relative pointer is unresolved here REGARDLESS of whether it happens to exist
        # under the current cwd (the absorbing parent's cwd is not the child's), so check
        # absoluteness FIRST — never let a stray cwd match read as a confirmed deliverable.
        if not p.is_absolute():
            return "[? unresolved path]"
        if p.is_file():
            try:
                return f"[✓ present, {p.stat().st_size} bytes]"
            except OSError:
                return "[✓ present]"
        return "[✓ present]" if p.exists() else "[⚠ MISSING]"
    except (OSError, ValueError):
        return ""


def _child_artifact_pointers(child: Dict[str, Any]) -> List[str]:
    """Artifact name+path pointers for a child (IDENTIFIERS, not content dumps), each STAT'd
    as a GROUND-TRUTH fact (✓ present / ⚠ MISSING) so the parent absorbs whether a claimed
    deliverable actually exists. LLM-first: this is a structural fact for the agent to react
    to — it does NOT change the child's status or force a review (I, v6.39)."""
    out: List[str] = []
    bundle = child.get("artifact_bundle") if isinstance(child.get("artifact_bundle"), dict) else {}
    arts = bundle.get("artifacts") if isinstance(bundle.get("artifacts"), list) else (
        child.get("artifacts") if isinstance(child.get("artifacts"), list) else []
    )
    for art in arts or []:
        if isinstance(art, dict):
            name = str(art.get("name") or "").strip()
            path = str(art.get("abs_path") or art.get("path") or "").strip()
            if path:
                label = f"{name} -> {path}" if name else path
                out.append(f"{label} {_artifact_stat_marker(path)}".strip())
            elif name:
                out.append(name)
    return out[:20]


def format_subagent_absorption_message(
    children: List[Dict[str, Any]],
    *,
    parent_task_id: str,
    budget_chars: int = 160_000,
) -> str:
    """Inject completed DIRECT children's FULL authored result before finalization so
    the parent absorbs their work (the cyber-racing parent finalized without ever
    reading its 3 children). Whole-artifact-or-pointer: each terminal direct child is
    injected in FULL while the aggregate fits ``budget_chars``; once exceeded, the
    remaining children are replaced WHOLE by a get_task_result pointer — a child's
    result plus its separately labelled host notice is NEVER mid-truncated, and
    the full output is always durable + pullable (P1). Grandchildren roll up to
    their direct parent: the root sees their STATUS
    only, not their raw output (avoids deep-tree context explosion)."""
    from ouroboros.tools.join_ledger import _child_result_sha256
    from ouroboros.task_finalization import provider_terminal_body

    parent = str(parent_task_id or "").strip()
    direct = [c for c in children if str(c.get("parent_task_id") or "") == parent]
    descendants = [c for c in children if str(c.get("parent_task_id") or "") != parent]
    terminal = [c for c in direct if str(c.get("status") or "").strip().lower() in FINAL_STATUSES]
    pending = [c for c in direct if c not in terminal]

    lines: List[str] = [
        "[SUBAGENT_RESULTS — absorb your children's work before finalizing. "
        "Full outputs are durable and pullable via get_task_result.]"
    ]
    spent = 0
    omitted = 0
    from ouroboros.cost_projection import cost_display

    from ouroboros.task_finalization import terminal_host_notice_text

    for child in terminal:
        cid = str(child.get("task_id") or child.get("id") or "")
        role = str(child.get("role") or "")
        result = provider_terminal_body(
            str(child.get("result") or "").strip(),
            terminal_host_notice_text(child),
        )
        terminal_status = str(child.get("child_status") or "")
        status_suffix = (
            f", terminal_result_status={terminal_status}"
            if terminal_status and terminal_status != str(child.get("status") or "")
            else ""
        )
        # A child's typed custody debt travels WITH its result: the parent could
        # absorb the work while its child's own delegated patch sat undisposed,
        # and nothing in the digest said so. Bounded, and silent for a clean
        # child. Visibility only — authority over that patch is the orphan rule.
        debt = child.get("delegated_runs_unreconciled")
        debt = [str(x) for x in debt] if isinstance(debt, list) else []
        debt_suffix = f", custody_debt={','.join(debt[:10])}" if debt else ""
        if len(debt) > 10:
            debt_suffix += f' (+{len(debt) - 10} more — get_task_result("{cid}"))'
        lines.append(
            f"\n## child {cid} ({role}) — status={child.get('status')}{status_suffix}"
            f"{debt_suffix}, "
            # SSOT cost projection (C2): unknown says unknown, never $0.0000.
            f"cost={cost_display(child, decimals=4)}, child_result_sha256={_child_result_sha256(child)}"
        )
        if result and spent + len(result) <= budget_chars:
            lines.append(result)
            spent += len(result)
        elif result:
            omitted += 1
            lines.append(
                f"[FULL RESULT OMITTED to fit context: {len(result)} chars — pull it with "
                f'get_task_result("{cid}")]'
            )
        else:
            lines.append("[no result text returned]")
        pointers = _child_artifact_pointers(child)
        if pointers:
            lines.append("artifacts: " + "; ".join(pointers))
    if omitted:
        lines.append(
            f"\n[NOTE] {omitted} child result(s) omitted for the context budget — "
            "pull them explicitly with get_task_result if you need them."
        )
    import json

    from ouroboros.work_stop import stop_for_reader

    # A queued status is no permission to run: a child its current stop holds is listed apart, beside that stop
    # as this parent may act on it (``stop_for_reader``: the request door when it is this parent's to ask for).
    stops = [stop_for_reader(child.get("current_stop") or {}, child, parent) for child in pending]
    for held, header in ((False, "[STILL RUNNING — not yet absorbable]"),
                         (True, "[HELD BY A CURRENT STOP — not yet absorbable: each reaches no terminal state until "
                                "its stop is released; a door named here is a request the supervisor still checks]")):
        group = [(child, stop) for child, stop in zip(pending, stops) if (stop.get("state") == "held") is held]
        if group:
            lines.append("\n" + header)
        for child, stop in group:
            cancel_note = (
                " (cancel pending — supervisor teardown in progress)"
                if str(child.get("cancel_state") or "") == "pending"
                else ""
            )
            lines.append(
                f"- {child.get('task_id') or child.get('id')}: {child.get('status')}{cancel_note}, "
                + (f"current_stop={json.dumps(stop, ensure_ascii=False, sort_keys=True)}, " if stop else "")
                + f"child_result_sha256={_child_result_sha256(child)}"
            )
    if descendants:
        lines.append(
            f"\n[DEEPER DESCENDANTS — rolled up to their direct parents; status only] ({len(descendants)}):"
        )
        for child in descendants[:40]:
            lines.append(
                f"- {child.get('task_id') or child.get('id')} "
                f"(parent {child.get('parent_task_id')}): {child.get('status')}"
            )
        if len(descendants) > 40:
            # Visible omission, never a silent clip of cognitive status (BIBLE P1).
            lines.append(
                f"- ⚠️ OMISSION NOTE: {len(descendants) - 40} additional descendants omitted "
                f"(full status via get_task_result on each root/child id)"
            )
    return "\n".join(lines)

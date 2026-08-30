"""Effective task status helpers shared by tools and gateways."""

from __future__ import annotations

import json
import logging
import pathlib
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

from ouroboros.headless import (
    ARTIFACT_STATUS_FAILED,
    ARTIFACT_STATUS_FINALIZING,
    ARTIFACT_STATUS_MISSING,
    ARTIFACT_STATUS_PENDING,
    ARTIFACT_STATUS_READY,
)
from ouroboros.outcomes import (
    EXECUTION_FAILED,
    EXECUTION_INFRA_FAILED,
    OBJECTIVE_FAIL,
    infra_failed_axes,
    normalize_outcome_axes,
)
from ouroboros.post_task_checkpoint import project_replica_task_result_fields
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
    list_task_results,
    load_task_result,
    validate_task_id,
)
from ouroboros.utils import iter_jsonl_objects, read_json_dict

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
ARTIFACT_TERMINAL_STATUSES: frozenset[str] = frozenset({
    ARTIFACT_STATUS_READY,
    ARTIFACT_STATUS_FAILED,
    "ready_with_changes",
    "ready_no_changes",
    "missing",
})
ARTIFACT_NONTERMINAL_STATUSES: frozenset[str] = frozenset({
    ARTIFACT_STATUS_PENDING,
    ARTIFACT_STATUS_FINALIZING,
})
HANDOFF_SNIPPET_CHARS = 240
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
    paths: List[pathlib.Path] = []
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    for source in (result, metadata):
        for key in ("child_drive_root", "headless_child_drive_root", "drive_root"):
            text = str(source.get(key) or "").strip()
            if not text:
                continue
            path = pathlib.Path(text)
            if path not in paths:
                paths.append(path)
    return paths


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
    path = pathlib.Path(drive_root) / "state" / "queue_snapshot.json"
    if not path.exists():
        return {"_snapshot_missing": True}
    data = read_json_dict(path)
    if not isinstance(data, dict):
        return {"_snapshot_invalid": True}
    return data


# GR7-1a freshness bound for the live-ownership twin. The supervisor persists
# the snapshot on every main-loop pass (nominally 0.5s apart), so a snapshot
# this old means the writer is gone or badly wedged — the twin can no longer
# trust it to prove a DEAD worker. Sized well above 2× the nominal tick to
# absorb ordinary loop hitches (task_done processing, git ops) without turning
# every cancel into a fail-open pass.
_SNAPSHOT_OWNERSHIP_FRESH_SEC = 10.0


def _snapshot_is_stale(snapshot: Dict[str, Any]) -> bool:
    """Whether the snapshot is too old to prove a dead worker (GR7-1a).

    A missing or unparseable ``ts`` cannot prove freshness either, so it
    reads as stale.
    """
    raw = str(snapshot.get("ts") or "").strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
        stamped = (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
    except (TypeError, ValueError):
        return True
    return (time.time() - stamped) > _SNAPSHOT_OWNERSHIP_FRESH_SEC


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


def _is_stale_orphan_running_task(
    drive_root: pathlib.Path,
    task_id: str,
    result: Dict[str, Any],
    events_index: Optional[_EventsTailIndex] = None,
) -> bool:
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
    if events_index is None:
        events_index = _EventsTailIndex(pathlib.Path(drive_root))
    latest_task_event = max(heartbeat, events_index.latest_event_ts(task_id))
    latest_worker_boot = events_index.latest_worker_boot()
    return bool(latest_worker_boot and latest_worker_boot > latest_task_event)


def _normalize_workspace_artifact_status(result: Dict[str, Any]) -> Dict[str, Any]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    if not (str(result.get("workspace_root") or "").strip() or str(metadata.get("workspace_root") or "").strip()):
        return result
    task_constraint = result.get("task_constraint") if isinstance(result.get("task_constraint"), dict) else {}
    if not task_constraint and isinstance(metadata.get("task_constraint"), dict):
        task_constraint = metadata.get("task_constraint") or {}
    if (
        str(result.get("delegation_role") or metadata.get("delegation_role") or "").strip() == "subagent"
        and str(task_constraint.get("mode") or "").strip() == "local_readonly_subagent"
    ):
        return result
    status = str(result.get("status") or "").lower()
    if status not in FINAL_STATUSES:
        return result
    artifact_status = str(result.get("artifact_status") or "").lower()
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


def _parent_workspace_artifact_lifecycle_fields(result: Dict[str, Any]) -> frozenset[str]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    if not (str(result.get("workspace_root") or "").strip() or str(metadata.get("workspace_root") or "").strip()):
        return frozenset()
    task_constraint = result.get("task_constraint") if isinstance(result.get("task_constraint"), dict) else {}
    if not task_constraint and isinstance(metadata.get("task_constraint"), dict):
        task_constraint = metadata.get("task_constraint") or {}
    if (
        str(result.get("delegation_role") or metadata.get("delegation_role") or "").strip() == "subagent"
        and str(task_constraint.get("mode") or "").strip() == "local_readonly_subagent"
    ):
        return frozenset()
    artifact_status = str(result.get("artifact_status") or "").strip().lower()
    if artifact_status in ARTIFACT_TERMINAL_STATUSES or artifact_status in ARTIFACT_NONTERMINAL_STATUSES:
        return _ARTIFACT_LIFECYCLE_FIELDS
    return frozenset()


def _merge_queue_status(current_status: str, queue_status: str) -> str:
    current = str(current_status or "").lower()
    queued = str(queue_status or "").lower()
    if not queued or current in FINAL_STATUSES:
        return current
    if current == STATUS_RUNNING and queued == STATUS_SCHEDULED:
        return current
    return queued


def load_effective_task_result(
    drive_root: pathlib.Path,
    task_id: str,
    *,
    materialize_artifacts: bool = True,
) -> Dict[str, Any]:
    try:
        tid = validate_task_id(task_id)
    except ValueError:
        return {}
    return effective_task_result(
        drive_root,
        load_task_result(drive_root, tid) or {},
        materialize_artifacts=materialize_artifacts,
    )


def reconcile_orphaned_running_tasks(drive_root: Any) -> int:
    """Durably finalize on-disk RUNNING task results the effective-status
    projection already considers terminal.

    A task whose worker crashed / was SIGKILLed / manually stopped can leave
    ``task_results/<id>.json`` at ``running`` forever (a misleading zombie). The
    read-time ``effective_task_result`` already projects such an orphan to a
    terminal status, but never persists it, so a headless/no-UI run that never
    re-reads the result keeps the stale ``running`` on disk.

    This sweep reuses ``load_effective_task_result`` so the persisted file matches
    the read projection exactly and inherits ALL of its liveness gates: the grace
    window, the worker-boot-after-task evidence, and the refusal to reconcile when
    the queue snapshot is missing. A task that is still pending/running in the
    queue, or whose worker has not booted after the task's last event, is never
    reconciled. The monotonic guard in ``write_task_result`` additionally protects
    a genuinely newer terminal/cancel write. Idempotent; safe at boot and on a
    periodic supervisor tick.
    """
    from ouroboros.task_results import list_task_results, write_task_result

    root = pathlib.Path(drive_root)
    healed = 0
    try:
        running = list_task_results(root, statuses=[STATUS_RUNNING, STATUS_INTERRUPTED])
    except Exception:
        return 0
    for row in running:
        task_id = str(row.get("task_id") or row.get("id") or "")
        if not task_id:
            continue
        try:
            effective = load_effective_task_result(root, task_id)
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
            write_task_result(root, task_id, status=eff_status, **persist_fields)
            healed += 1
        except Exception:
            continue
        # The healed orphan is provably dead (the projection's liveness gates):
        # forget its reaping-registry id and release any acceptance fence it
        # owned (a wedged kill holds both past the orphan's terminalization).
        try:
            from supervisor.queue import release_acceptance_fence_for_dead_owner
            from supervisor.task_reaper import _forget_task_reaping

            _forget_task_reaping(task_id)
            release_acceptance_fence_for_dead_owner(task_id)
        except Exception:
            log.debug("Post-heal acceptance-fence release failed for %s", task_id, exc_info=True)
    return healed


def _apply_cancel_state_projection(
    drive_root: pathlib.Path,
    requested_task_id: str,
    merged: Dict[str, Any],
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

        cancel_fields = cancel_state_fields(
            pathlib.Path(drive_root), requested_task_id,
        )
        if not cancel_fields:
            physical_task_id = str(merged.get("task_id") or "").strip()
            if physical_task_id and physical_task_id != requested_task_id:
                cancel_fields = cancel_state_fields(
                    pathlib.Path(drive_root), physical_task_id,
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
) -> Dict[str, Any]:
    """Merge parent result, child-drive result, and active queue state.

    ``materialize_artifacts=False`` yields a "status/cost projection only" read
    (v6.90.x P2): the entire artifact block — including the mutating child-artifact
    rebase (``copy_file_to_task_artifacts``), ``collect_task_artifact_records``
    scans, and the task-tree ``_project_child_result_disposition`` hash lookup — is
    skipped, and the projection never carries sha-bearing/disposition claims.
    ``artifacts`` on a False row are the raw admission-time recorded entries,
    not the merged/rebased set a materializing read would produce.
    Read-only display surfaces (chat history annotation, ``api_tasks_list``, the
    SSE follow loop, ``api_logs_tail`` discovery) pass ``False``; every consumer
    that participates in the child-result sha economy or artifact durability
    (join_ledger, wait_*/get_task_result, api_task_get/artifact, reconcile, prune)
    keeps the ``True`` default.
    ``_events_index`` optionally shares ONE parsed events-tail across a batch of
    rows (the task-list request), so N stale-running rows cost one tail read
    instead of N; ``None`` keeps the per-call read for single-row callers.
    """

    if not result:
        return {}
    task_id = str(result.get("task_id") or result.get("id") or "").strip()
    if not task_id:
        return dict(result)
    retry_id = str(result.get("superseded_by") or result.get("retry_task_id") or "").strip()
    if retry_id and retry_id != task_id and retry_id not in _seen:
        retry_result = load_task_result(drive_root, retry_id) or {}
        if retry_result:
            effective_retry = effective_task_result(
                pathlib.Path(drive_root),
                retry_result,
                materialize_artifacts=materialize_artifacts,
                _seen=frozenset(set(_seen) | {task_id}),
                _events_index=_events_index,
            )
            if effective_retry:
                merged_retry = dict(effective_retry)
                lineage = list(merged_retry.get("retry_lineage") or [])
                lineage.insert(0, {
                    "task_id": task_id,
                    "status": result.get("status"),
                    "outcome_axes": normalize_outcome_axes(result),
                    "reason_code": result.get("reason_code"),
                    "retry_task_id": retry_id,
                })
                merged_retry["retry_lineage"] = lineage
                merged_retry.setdefault("original_task_id", task_id)
                merged_retry.setdefault("supersedes_task_id", task_id)
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
                    pathlib.Path(drive_root), task_id, merged_retry,
                )
                return merged_retry

    merged = dict(result)
    child_result: Dict[str, Any] = {}
    child_text = ""
    if not cancellation_blocks_child_result(result):
        for child_drive in _child_drive_candidates(result):
            child_result = load_task_result(child_drive, task_id) or {}
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

    parent_status = str(merged.get("status") or "").lower()
    if parent_status not in FINAL_STATUSES:
        queue_status, queue_task = _queue_task_status(_load_queue_snapshot(pathlib.Path(drive_root)), task_id)
        if queue_status and queue_status != "unknown":
            merged["status"] = _merge_queue_status(parent_status, queue_status)
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
            if queue_status == "unknown":
                merged["queue_reconciliation_warning"] = "queue snapshot missing or invalid"
            elif _terminal_failure_from_outcome(merged):
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
            elif _is_stale_orphan_running_task(pathlib.Path(drive_root), task_id, merged, _events_index):
                orphan_reason = (
                    "interrupted_retry_lost"
                    if parent_status == STATUS_INTERRUPTED
                    else "orphaned_running_after_worker_restart"
                )
                merged["status"] = STATUS_FAILED
                merged["reason_code"] = orphan_reason
                merged["outcome_axes"] = infra_failed_axes(orphan_reason)
                merged["status_reconciled_from"] = parent_status
                merged["result"] = (
                    str(merged.get("result") or "Task was interrupted before a terminal result was recorded.")
                    + "\n\n⚠️ TASK_ORPHAN_RECONCILED: queue is empty and worker restarted after this task; "
                    f"marking the stale {parent_status} task as infra_failed."
                )
                artifact_status = str(merged.get("artifact_status") or "").strip().lower()
                if artifact_status in ARTIFACT_NONTERMINAL_STATUSES:
                    merged["artifact_status"] = ARTIFACT_STATUS_FAILED
                    bundle = dict(merged.get("artifact_bundle") or {}) if isinstance(merged.get("artifact_bundle"), dict) else {}
                    merged["artifact_bundle"] = _fail_nonterminal_artifact_bundle(
                        bundle,
                        "task interrupted before artifact finalization",
                    )
    # Typed public cancel projection (phase A): an ACTIVE durable cancel intent
    # rides every effective read as ``cancel_state: "pending"`` — status itself
    # stays honest (running/scheduled) until the supervisor settles the teardown.
    # A legacy ``cancel_requested`` status (old files awaiting boot migration)
    # projects the same pending state.
    _apply_cancel_state_projection(pathlib.Path(drive_root), task_id, merged)

    if not materialize_artifacts:
        # Status/cost projection only: skip the whole artifact block (incl. the
        # mutating child-artifact rebase and collect_task_artifact_records file
        # scans) AND the disposition hash lookup. Strip the legacy mirrored
        # fields so a False row never carries sha-bearing/disposition claims.
        projected = dict(merged)
        for field in _CHILD_DISPOSITION_FIELDS:
            projected.pop(field, None)
        return projected
    try:
        from ouroboros.artifacts import (
            collect_task_artifact_records,
            copy_file_to_task_artifacts,
            merge_artifact_records,
        )
        from ouroboros.outcomes import artifact_bundle_from_result

        if child_result:
            parent_artifacts = [item for item in (result.get("artifacts") or []) if isinstance(item, dict)]
            child_artifacts_for_merge = [item for item in (child_result.get("artifacts") or []) if isinstance(item, dict)]
            if parent_artifacts or child_artifacts_for_merge:
                merged["artifacts"] = merge_artifact_records(parent_artifacts, child_artifacts_for_merge)

        rebased_child_artifacts: List[Dict[str, Any]] = []
        if child_text:
            parent_artifact_ctx = SimpleNamespace(drive_root=pathlib.Path(drive_root), task_id=task_id)
            child_artifacts = merge_artifact_records(
                [item for item in (child_result.get("artifacts") or []) if isinstance(item, dict)],
                collect_task_artifact_records(pathlib.Path(child_text), task_id),
            )
            from ouroboros.outcome_receipt_store import (
                is_verification_receipts_path,
                publish_verification_receipt_union,
            )

            for child_artifact in child_artifacts:
                source_text = str(child_artifact.get("path") or "").strip()
                if not source_text:
                    continue
                source = pathlib.Path(source_text).expanduser().resolve(strict=False)
                if not source.is_file():
                    continue
                if is_verification_receipts_path(child_text, task_id, source):
                    # Historical child results may already list the receipt
                    # stream as a generic artifact.  Reconcile it through its
                    # one locked owner and never feed it to shutil.copy2.
                    publish_verification_receipt_union(
                        pathlib.Path(drive_root), task_id, pathlib.Path(child_text),
                    )
                    continue
                copied = copy_file_to_task_artifacts(
                    parent_artifact_ctx,
                    source,
                    kind=str(child_artifact.get("kind") or "child_artifact"),
                )
                if copied:
                    rebased_child_artifacts.append(copied)

        collected_artifacts = collect_task_artifact_records(drive_root, task_id)
        if collected_artifacts or rebased_child_artifacts:
            existing_artifacts = [item for item in (merged.get("artifacts") or []) if isinstance(item, dict)]
            rebased_names = {
                str(item.get("name") or pathlib.Path(str(item.get("path") or "")).name)
                for item in rebased_child_artifacts
                if isinstance(item, dict)
            }
            if rebased_names:
                existing_artifacts = [
                    item
                    for item in existing_artifacts
                    if str(item.get("name") or pathlib.Path(str(item.get("path") or "")).name) not in rebased_names
                ]
                collected_artifacts = collect_task_artifact_records(drive_root, task_id)
            merged["artifacts"] = merge_artifact_records(existing_artifacts, rebased_child_artifacts, collected_artifacts)
            merged["artifact_bundle"] = artifact_bundle_from_result(merged)
            if not merged.get("artifact_status"):
                merged["artifact_status"] = merged["artifact_bundle"].get("status")
    except Exception:
        pass
    return _project_child_result_disposition(pathlib.Path(drive_root), merged)


def wait_for_effective_tasks(
    drive_root: pathlib.Path,
    task_ids: Iterable[str],
    *,
    timeout_sec: float,
    mode: str = "all_terminal",
    poll_interval_sec: float = 0.5,
    on_poll: Optional[Callable[[Dict[str, Any], Dict[str, bool]], Any]] = None,
) -> Dict[str, Any]:
    """Poll effective task results until the wait ``mode`` is satisfied.

    Terminality is ``SETTLED_STATUSES``: a wait loop's jobs is to surface the
    FINAL record, and since the phase-A cancel redesign cancel intent is a
    durable ``cancel_intents`` projection (surfaced as ``cancel_state:
    "pending"``), never a status value — the supervisor's custody settles every
    intent to a real terminal shortly after. The wait stays bounded by
    ``timeout_sec`` either way, and ``live_child_status`` reports a pending
    cancellation honestly (``cancel_pending``) instead of collapsing it to
    terminal/unknown."""
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
        results = {tid: load_effective_task_result(pathlib.Path(drive_root), tid) for tid in ids}
        terminal = {tid: str(data.get("status") or "").strip().lower() in SETTLED_STATUSES for tid, data in results.items()}
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
        time.sleep(max(0.05, min(2.0, float(poll_interval_sec or 0.5))))
    out: Dict[str, Any] = {
        "mode": mode,
        "timeout_sec": float(timeout_sec or 0),
        "elapsed_sec": max(0.0, time.monotonic() - start),
        "timed_out": timed_out,
        "all_terminal": all(str(data.get("status") or "").strip().lower() in SETTLED_STATUSES for data in results.values()) if ids else True,
        "tasks": results,
    }
    if early is not None:
        out["early_return"] = early
    # Live per-child status from the queue snapshot — kills the false "starved"/"dead"
    # claim: the parent sees which children are actually RUNNING/SCHEDULED vs terminal.
    try:
        _snap = _load_queue_snapshot(pathlib.Path(drive_root))
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
            live[tid] = _st or ("terminal" if eff_status in SETTLED_STATUSES else "unknown")
        out["live_child_status"] = live
    except Exception:
        pass
    return out


def find_child_tasks(
    drive_root: pathlib.Path,
    *,
    parent_task_id: str = "",
    root_task_id: str = "",
    exclude_task_id: str = "",
    scope: str = "subtree",
    materialize_artifacts: bool = True,
) -> List[Dict[str, Any]]:
    """Collect a task's subagent children.

    ``scope`` selects the matching semantics (structural, never prose-parsed — P5):

    - ``"subtree"`` (default): a row matches if it is a DIRECT child
      (``parent_task_id == parent``) OR any subagent of the same tree
      (``root_task_id == root``). This is the whole-subtree view UI/API surfaces
      (`gateway/tasks.py`, `gateway/logs.py`) render as descendant cards.
    - ``"direct"``: ONLY direct children (``parent_task_id == parent``); the
      root-subtree fallback is skipped. This is what per-node absorption/handoff
      wants — a LEAF task must see only its OWN children (none), not its parent
      and siblings. The v6.56 root-fallback made a childless grandchild receive a
      false ``children_unabsorbed`` reminder about its parent/sibling.
    """
    parent = str(parent_task_id or "").strip()
    root = str(root_task_id or "").strip()
    excluded = str(exclude_task_id or "").strip()
    direct_only = str(scope or "subtree").strip().lower() == "direct"
    rows: Dict[str, Dict[str, Any]] = {}
    for row in (
        effective_task_result(
            pathlib.Path(drive_root), item, materialize_artifacts=materialize_artifacts
        )
        for item in list_task_results(pathlib.Path(drive_root))
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

    snapshot = _load_queue_snapshot(pathlib.Path(drive_root))
    for group, status in (("pending", STATUS_SCHEDULED), ("running", STATUS_RUNNING)):
        for item in snapshot.get(group) or []:
            if not isinstance(item, dict):
                continue
            task = item.get("task") if isinstance(item.get("task"), dict) else {}
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
                rows[tid] = row
                continue
            combined = dict(existing)
            for key, value in row.items():
                if key == "status":
                    combined["status"] = _merge_queue_status(str(existing.get("status") or ""), str(value or ""))
                elif not combined.get(key) and value:
                    combined[key] = value
            rows[tid] = combined
    return sorted(rows.values(), key=lambda item: (str(item.get("ts") or ""), str(item.get("task_id") or "")))


def compute_cost_with_children(
    drive_root: pathlib.Path, task_id: str, own_cost_usd: float
) -> tuple[float, bool]:
    """Recursive per-task cost rollup (v6.57.0, P6b): own cost PLUS the cost of every
    direct child. Each child already stored its own ``cost_usd_with_children``, so
    summing the direct children's rolled-up value makes this correct for the whole
    subtree without re-walking it. Returns ``(total, partial)`` where ``partial`` is
    True when any direct child is still non-terminal (its cost is not final yet).

    Why a NEW field and not a change to ``cost_usd``: existing consumers (per-task
    accounting, the global session budget ledger) read ``cost_usd`` as this task's own
    spend; the parent card / Logs read the rollup. Never mutate ``cost_usd`` — the
    site/PB incident showed a parent under-reporting because children weren't summed,
    but the fix is an ADDITIVE field, not a semantics change (P7 SSOT)."""
    total = float(own_cost_usd or 0.0)
    partial = False
    try:
        children = find_child_tasks(
            pathlib.Path(drive_root), parent_task_id=str(task_id), root_task_id="", scope="direct"
        )
    except Exception:
        return round(total, 6), True
    for child in children:
        accounting_available = str(child.get("cost_accounting_status") or "available") == "available"
        child_total = child.get("cost_usd_with_children")
        if child_total is None:
            child_total = child.get("cost_usd")
        try:
            if child_total is None or not accounting_available:
                raise ValueError("child cost unavailable")
            total += float(child_total)
        except (TypeError, ValueError):
            partial = True
        if (
            str(child.get("status") or "").strip().lower() not in FINAL_STATUSES
            or child.get("cost_final") is not True
            or bool(child.get("cost_with_children_partial"))
        ):
            partial = True
    return round(total, 6), partial


def _handoff_snippet(value: Any) -> Dict[str, Any]:
    text = str(value or "")
    stripped = text.strip()
    if not stripped:
        return {"available": False, "chars": 0, "preview": ""}
    preview = stripped.replace("\n", " ")
    if len(preview) > HANDOFF_SNIPPET_CHARS:
        preview = preview[: HANDOFF_SNIPPET_CHARS - 3] + "..."
    return {"available": True, "chars": len(text), "preview": preview}


def format_handoff_message(children: List[Dict[str, Any]]) -> str:
    from ouroboros.tools.join_ledger import _child_result_sha256

    from ouroboros.cost_projection import cost_projection

    payload = []
    for child in children:
        result_info = _handoff_snippet(child.get("result"))
        trace_info = _handoff_snippet(child.get("trace_summary"))
        _cost = cost_projection(child)
        payload.append({
            "task_id": str(child.get("task_id") or child.get("id") or ""),
            "status": str(child.get("status") or ""),
            "role": str(child.get("role") or ""),
            "description": str(child.get("description") or child.get("objective") or ""),
            # SSOT cost projection (C2): honest null — a child with no accounting
            # reads null here, never a fabricated $0 — plus the additive name.
            "cost_usd": _cost["cost_usd"],
            "accounted_upper_bound_usd": _cost["accounted_upper_bound_usd"],
            "cost_final": _cost["cost_final"],
            "artifact_status": str(child.get("artifact_status") or ""),
            "terminal_result_status": (
                str(child.get("child_status") or "")
                if str(child.get("child_status") or "") != str(child.get("status") or "")
                else ""
            ),
            "child_result_sha256": _child_result_sha256(child),
            "result_available": result_info["available"],
            "result_chars": result_info["chars"],
            "result_preview": result_info["preview"],
            "trace_available": trace_info["available"],
            "trace_chars": trace_info["chars"],
            "trace_preview": trace_info["preview"],
            "full_output": "Use get_task_result or wait_task for the full untruncated child output (wait_tasks returns a compact batch projection).",
        })
    return (
        "[SUBAGENT_HANDOFF_STATUS]\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n[/SUBAGENT_HANDOFF_STATUS]"
    )


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
    result is NEVER mid-truncated, and the full output is always durable + pullable
    (P1). Grandchildren roll up to their direct parent: the root sees their STATUS
    only, not their raw output (avoids deep-tree context explosion)."""
    from ouroboros.tools.join_ledger import _child_result_sha256

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

    for child in terminal:
        cid = str(child.get("task_id") or child.get("id") or "")
        role = str(child.get("role") or "")
        result = str(child.get("result") or "").strip()
        terminal_status = str(child.get("child_status") or "")
        status_suffix = (
            f", terminal_result_status={terminal_status}"
            if terminal_status and terminal_status != str(child.get("status") or "")
            else ""
        )
        lines.append(
            f"\n## child {cid} ({role}) — status={child.get('status')}{status_suffix}, "
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
    if pending:
        lines.append("\n[STILL RUNNING — not yet absorbable]")
        for child in pending:
            cancel_note = (
                " (cancel pending — supervisor teardown in progress)"
                if str(child.get("cancel_state") or "") == "pending"
                else ""
            )
            lines.append(
                f"- {child.get('task_id') or child.get('id')}: {child.get('status')}{cancel_note}, "
                f"child_result_sha256={_child_result_sha256(child)}"
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

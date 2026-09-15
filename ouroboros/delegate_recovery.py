"""Narrow successor proofs for configured-session worker/restart recovery."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import uuid
from dataclasses import asdict
from typing import Any, Mapping, Optional

from ouroboros import delegate_custody as custody
from ouroboros.subagent_work_order import work_order_fingerprint
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

CAUSE_WORKER_CRASH = "non_signal_worker_crash"
CAUSE_PLANNED_SELF_RESTART = "planned_self_restart"
_ALLOWED_CAUSES = {CAUSE_WORKER_CRASH, CAUSE_PLANNED_SELF_RESTART}
NO_RESUME_CAUSES = (
    "owner_restart", "panic", "external_signal", "worker_signal",
    "deadline", "timeout", "explicit_cancellation", "abrupt_whole_app_loss",
)
PLANNED_RESTART_TRANSACTION_ENV = "OUROBOROS_PLANNED_RESTART_TRANSACTION_ID"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        .encode("utf-8")
    ).hexdigest()


def authority_fingerprint_from_task(task: Mapping[str, Any]) -> str:
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.contracts.task_contract import build_task_contract

    contract = build_task_contract(task)
    normalized_constraint = normalize_task_constraint(task.get("task_constraint"))
    constraint = asdict(normalized_constraint) if normalized_constraint is not None else {}

    def _root(value: Any) -> str:
        text = str(value or "").strip()
        return str(pathlib.Path(text).resolve(strict=False)) if text else ""

    return _canonical_hash({
        "task_id": str(task.get("id") or ""),
        "workspace_root": _root(task.get("workspace_root")),
        "workspace_mode": str(task.get("workspace_mode") or ""),
        "drive_root": _root(task.get("drive_root")),
        "task_constraint": constraint,
        "task_contract": contract,
        "executor_ref": task.get("executor_ref") if isinstance(task.get("executor_ref"), dict) else {},
    })


def authority_fingerprint_from_context(ctx: Any) -> str:
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    return authority_fingerprint_from_task({
        **meta,
        "id": str(getattr(ctx, "task_id", "") or ""),
        "drive_root": str(getattr(ctx, "drive_root", "") or ""),
        "workspace_root": str(getattr(ctx, "workspace_root", "") or meta.get("workspace_root") or ""),
        "workspace_mode": str(getattr(ctx, "workspace_mode", "") or meta.get("workspace_mode") or ""),
        "task_constraint": getattr(ctx, "task_constraint", None) or meta.get("task_constraint") or {},
        "task_contract": getattr(ctx, "task_contract", None) or meta.get("task_contract") or {},
        "executor_ref": getattr(ctx, "executor_ref", None) or meta.get("executor_ref") or {},
    })


def _path(drive_root: Any, task_id: str) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / "delegate_recovery" / f"{task_id}.json"


def _read(drive_root: Any, task_id: str) -> dict[str, Any]:
    try:
        data = json.loads(_path(drive_root, task_id).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(drive_root: Any, row: dict[str, Any]) -> None:
    path = _path(drive_root, str(row.get("task_id") or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    row["updated_at"] = utc_now_iso()
    atomic_write_json(path, row)


def _restart_transaction_path(drive_root: Any, transaction_id: str) -> pathlib.Path:
    return (
        pathlib.Path(drive_root) / "state" / "delegate_recovery_transactions"
        / f"{transaction_id}.json"
    )


def _active_restart_transaction_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / "delegate_recovery_transactions" / "active.json"


def arm_active_planned_restart_transaction(drive_root: Any) -> str:
    """Pass a prepared restart transaction to a direct re-exec successor.

    Launcher-managed exits acknowledge the same durable transaction by waiting
    for exit code 42.  A direct server re-exec has no launcher, so it carries
    the already-created transaction id through the existing one-shot
    environment handoff consumed by ``_ack_direct_exec_successor``.
    """
    try:
        active = json.loads(_active_restart_transaction_path(drive_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(active, dict):
        return ""
    transaction_id = str(active.get("transaction_id") or "")
    row = _read_restart_transaction(drive_root, transaction_id) if transaction_id else {}
    if (
        not transaction_id
        or row.get("status") != "prepared"
        or int(row.get("supervisor_pid") or 0) != os.getpid()
    ):
        return ""
    os.environ[PLANNED_RESTART_TRANSACTION_ENV] = transaction_id
    return transaction_id


def _read_restart_transaction(drive_root: Any, transaction_id: str) -> dict[str, Any]:
    try:
        data = json.loads(
            _restart_transaction_path(drive_root, transaction_id).read_text(encoding="utf-8")
        )
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_restart_transaction(drive_root: Any, row: dict[str, Any]) -> None:
    path = _restart_transaction_path(drive_root, str(row.get("transaction_id") or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    row["updated_at"] = utc_now_iso()
    atomic_write_json(path, row)


def acknowledge_observed_restart_exit(
    drive_root: Any, *, supervisor_pid: int, exit_code: int,
) -> bool:
    """Launcher-side proof that the exact prepared generation exited with code 42."""

    try:
        active = json.loads(
            _active_restart_transaction_path(drive_root).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False
    active = active if isinstance(active, dict) else {}
    transaction_id = str(active.get("transaction_id") or "")
    row = _read_restart_transaction(drive_root, transaction_id)
    if (
        not transaction_id
        or row.get("status") != "prepared"
        or int(row.get("supervisor_pid") or 0) != int(supervisor_pid)
        or int(exit_code) != 42
    ):
        return False
    row.update({
        "status": "normal_exit_acknowledged", "exit_code": 42,
        "exit_acknowledged_at": utc_now_iso(), "ack_source": "launcher_waitpid",
    })
    _write_restart_transaction(drive_root, row)
    custody.emit(drive_root, "delegate_restart_transaction_acknowledged", {
        "restart_transaction_id": transaction_id,
        "supervisor_pid": int(supervisor_pid), "exit_code": 42,
        "task_ids": list(row.get("task_ids") or []), "ack_source": "launcher_waitpid",
    })
    return True


def _ack_direct_exec_successor(drive_root: Any) -> None:
    """A same-PID successor carrying the one-shot token proves exec succeeded."""

    transaction_id = str(os.environ.pop(PLANNED_RESTART_TRANSACTION_ENV, "") or "")
    if not transaction_id:
        return
    row = _read_restart_transaction(drive_root, transaction_id)
    if (
        row.get("status") != "prepared"
        or int(row.get("supervisor_pid") or 0) != os.getpid()
    ):
        return
    row.update({
        "status": "normal_exit_acknowledged", "exit_code": 42,
        "exit_acknowledged_at": utc_now_iso(), "ack_source": "direct_exec_successor",
    })
    _write_restart_transaction(drive_root, row)
    custody.emit(drive_root, "delegate_restart_transaction_acknowledged", {
        "restart_transaction_id": transaction_id,
        "supervisor_pid": os.getpid(), "exit_code": 42,
        "task_ids": list(row.get("task_ids") or []), "ack_source": "direct_exec_successor",
    })


def _selected_session(task: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = task.get("configured_subagent") if isinstance(task.get("configured_subagent"), dict) else {}
    route = snapshot.get("route") if isinstance(snapshot.get("route"), dict) else {}
    return snapshot if str(route.get("kind") or "") == "agent_session" else {}


def _review_substrate_run(source: Any) -> bool:
    """True for every durable `review_substrate*` custody source spelling."""

    return str(source or "").startswith("review_substrate")


def unsettled_start_ids(
    drive_root: Any, task_id: str, *, rows: Optional[list[dict[str, Any]]] = None,
) -> dict[str, list[str]]:
    """Durable run/start blockers from one consistent custody-log snapshot.

    A review run is the review substrate's obligation, not the actor's
    delegation slot, so it never blocks the actor's own start.
    """

    mine = str(task_id or "")
    snapshot = list(rows) if rows is not None else list(
        custody._iter_rows(custody.event_log_path(drive_root))
    )
    runs = custody.replay(drive_root, rows=snapshot)
    return {
        "open_run_ids": [
            row.run_id for row in runs.values()
            if row.task_id == mine and not row.settled
            and not _review_substrate_run(row.source)
        ],
        "pending_invocation_ids": [
            str(row.get("invocation_id") or "")
            for row in custody.pending_invocations(drive_root, rows=snapshot)
            if str(row.get("task_id") or "") == mine
            and not _review_substrate_run(row.get("source"))
        ],
        "undisposed_patch_run_ids": [
            row.run_id for row in runs.values()
            if row.task_id == mine and row.snapshot_id and row.settled
            and not row.patch_disposed
            and not _review_substrate_run(row.source)
        ],
    }


def reconcile_unrecoverable_task(drive_root: Any, task_id: str) -> None:
    """Fail-soft cancellation/settlement for a task without a valid successor."""

    try:
        from ouroboros import delegate_terminal

        result = delegate_terminal.terminal_reconcile_task(
            drive_root, str(task_id or ""), trigger="unrecoverable_successor",
        )
        delegate_terminal.record_terminal_reconciliation(
            drive_root, str(task_id or ""), result,
        )
    except Exception:
        log.debug("Unrecoverable delegate reconciliation failed", exc_info=True)


def prepare_worker_crash_handoff(
    drive_root: Any,
    task: Mapping[str, Any],
    *,
    old_attempt: int,
    new_attempt: int,
    worker_id: int,
    exitcode: Optional[int],
) -> dict[str, Any]:
    """Reserve one same-generation successor and settle anything unprovable."""

    try:
        row = prepare_handoff(
            drive_root, task, cause=CAUSE_WORKER_CRASH,
            old_attempt=old_attempt, new_attempt=new_attempt,
            worker_id=worker_id, exitcode=exitcode,
        )
    except Exception:
        log.debug("Worker-crash delegate handoff preparation failed", exc_info=True)
        row = {}
    if not row and isinstance(task.get("configured_subagent"), dict):
        reconcile_unrecoverable_task(drive_root, str(task.get("id") or ""))
    return row


def veto_worker_retry_handoff(
    drive_root: Any, task_id: str, handoff: Mapping[str, Any], admission_block: str,
) -> None:
    if not handoff:
        return
    try:
        veto_handoff(drive_root, task_id, f"retry_admission_blocked:{admission_block}")
    except Exception:
        log.debug("Admission-blocked handoff veto failed", exc_info=True)


def _holder_matches(row: Mapping[str, Any], holder: Any, *, pending: bool) -> bool:
    identity = "pending_invocation_id" if pending else "run_id"
    holder_identity = _holder_value(holder, "invocation_id" if pending else "run_id", pending=pending)
    if holder_identity != str(row.get(identity) or ""):
        return False
    return all(
        _holder_value(holder, name, pending=pending) == str(row.get(name) or "")
        for name in (
            "selected_subagent_id", "config_fingerprint", "authority_fingerprint",
            "work_order_fingerprint", "snapshot_id", "execution_root",
            "baseline_sha", "target_root",
        )
    )


def _holder_value(holder: Any, name: str, *, pending: bool) -> str:
    return (
        str(holder.get(name) or "")
        if pending else str(getattr(holder, name, "") or "")
    )


def _run_probe_error(run_id: str) -> str:
    gateway = None
    try:
        from ouroboros.claudexor_daemon import ensure_owned_gateway
        gateway = ensure_owned_gateway()
        gateway.get_run(run_id)
        return ""
    except Exception as exc:
        return type(exc).__name__
    finally:
        if gateway is not None:  # pragma: no branch - paired with the acquisition above
            gateway.close()


def _successor_binding_mismatch(row: Mapping[str, Any], task: Mapping[str, Any]) -> str:
    """Re-prove task authority plus the exact leaf binding reserved at handoff.

    The task's first physical invocation, when one exists, is bound to the task
    snapshot.  An explicit same-nanny replacement (or a root-direct leaf) has its
    own later durable actor/work-order binding, so recovery follows that unique
    custody holder rather than retargeting it back to the task's initial actor.
    """

    if str(row.get("authority_fingerprint") or "") != authority_fingerprint_from_task(task):
        return "authority_mismatch"
    binding_source = str(row.get("actor_binding_source") or "task_snapshot")
    if binding_source == "current_custody_holder":
        if all(
            str(row.get(name) or "")
            for name in (
                "selected_subagent_id", "config_fingerprint", "work_order_fingerprint",
            )
        ):
            return ""
        return "current_holder_binding_incomplete"
    if binding_source != "task_snapshot":
        return "actor_binding_source_invalid"
    snapshot = _selected_session(task)
    if str(row.get("config_fingerprint") or "") != str(snapshot.get("config_fingerprint") or ""):
        return "config_mismatch"
    if str(row.get("selected_subagent_id") or "") != str(snapshot.get("selected_subagent_id") or ""):
        return "actor_mismatch"
    if str(row.get("work_order_fingerprint") or "") != work_order_fingerprint(task):
        return "work_order_mismatch"
    return ""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    from ouroboros.platform_layer import pid_is_alive

    return bool(pid_is_alive(pid))


def _restore_wait_checkpoint(drive_root: Any, row: Mapping[str, Any]) -> None:
    """Restore cursor/mailbox/interaction facts before successor wait resumes."""

    task_id = str(row.get("task_id") or "")
    run_id = str(row.get("run_id") or "")
    path = pathlib.Path(drive_root) / "state" / "delegate_supervision" / f"{task_id}.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    state = state if isinstance(state, dict) and str(state.get("run_id") or run_id) == run_id else {}
    pending_wake = row.get("pending_wake") if isinstance(row.get("pending_wake"), dict) else {}
    state.update({
        "schema": 1, "run_id": run_id,
        "status": "wake_pending" if pending_wake else "adopted",
    })
    state["journal_cursor"] = max(
        int(state.get("journal_cursor") or 0), int(row.get("journal_cursor") or 0)
    )
    state["mailbox_acknowledged_ids"] = sorted({
        str(item) for item in [
            *(state.get("mailbox_acknowledged_ids") or []),
            *(row.get("mailbox_acknowledged_ids") or []),
        ] if str(item)
    })
    state["interaction_acknowledged_ids"] = sorted({
        str(item) for item in [
            *(state.get("interaction_acknowledged_ids") or []),
            *(row.get("interaction_acknowledged_ids") or []),
        ] if str(item)
    })
    if pending_wake:
        state["pending_wake"] = dict(pending_wake)
        payload = pending_wake.get("payload") if isinstance(pending_wake.get("payload"), dict) else {}
        if payload:
            state["last_wake"] = dict(payload)
    if isinstance(row.get("checkpoint"), dict):
        state["checkpoint"] = dict(row["checkpoint"])
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, state)
    interaction_ids = frozenset(
        str(item) for item in (row.get("interaction_acknowledged_ids") or []) if str(item)
    )
    if interaction_ids and run_id:
        from ouroboros.delegate_interactions import _REPORTED_INTERACTIONS

        _REPORTED_INTERACTIONS[run_id] = interaction_ids


def _successor_pending_wake(value: Any) -> dict[str, Any]:
    """Remove attempt-local loop controls from a successor's replay payload."""
    wake = dict(value) if isinstance(value, dict) else {}
    payload = wake.get("payload") if isinstance(wake.get("payload"), dict) else {}
    events = payload.get("wake_events") if isinstance(payload.get("wake_events"), list) else []
    if not events:
        return wake
    from ouroboros.delegate_supervision import _LOOP_CONTROL_KINDS, _QUIET_STATUSES

    kept = [
        dict(item) for item in events if isinstance(item, dict)
        and str(item.get("kind") or "") not in _LOOP_CONTROL_KINDS
    ]
    if len(kept) == len(events):
        return wake
    payload = dict(payload)
    if kept:
        payload["wake_events"] = kept
    else:
        payload.pop("wake_events", None)
    if not kept and str(payload.get("status") or "") in _QUIET_STATUSES:
        return {}
    return {**wake, "payload": payload}


def prepare_handoff(
    drive_root: Any,
    task: Mapping[str, Any],
    *,
    cause: str,
    old_attempt: int,
    new_attempt: int,
    worker_id: int = 0,
    exitcode: Optional[int] = None,
    restart_transaction_id: str = "",
) -> dict[str, Any]:
    """Persist one exact successor reservation, or return no reservation."""

    if cause not in _ALLOWED_CAUSES:
        return {}
    task_id = str(task.get("id") or "")
    if not task_id:
        return {}
    supervision_path = (
        pathlib.Path(drive_root) / "state" / "delegate_supervision" / f"{task_id}.json"
    )
    try:
        supervision = json.loads(supervision_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        supervision = {}
    supervision = supervision if isinstance(supervision, dict) else {}
    pending_wake = (
        supervision.get("pending_wake")
        if isinstance(supervision.get("pending_wake"), dict)
        and not supervision["pending_wake"].get("acknowledged_at")
        else {}
    )
    if not pending_wake and supervision.get("status") == "awake":
        acknowledged = (
            supervision.get("last_acknowledged_wake")
            if isinstance(supervision.get("last_acknowledged_wake"), dict) else {}
        )
        payload = (
            acknowledged.get("payload")
            if isinstance(acknowledged.get("payload"), dict) else {}
        )
        if payload:
            replay_wake_id = uuid.uuid4().hex
            pending_wake = {
                **acknowledged,
                "wake_id": replay_wake_id,
                "payload": {
                    **payload, "supervision_wake_id": replay_wake_id,
                    "replayed_after_worker_loss": True,
                },
                "acknowledged_at": "",
                "replay_reason": "acknowledged_in_dead_transcript",
            }
    pending_wake = _successor_pending_wake(pending_wake)
    runs = [row for row in custody.open_runs(drive_root) if row.task_id == task_id]
    pending = [
        row for row in custody.pending_invocations(drive_root)
        if str(row.get("task_id") or "") == task_id
    ]
    settled_holder = None
    if not runs and not pending:
        pending_run_id = str(supervision.get("run_id") or "")
        candidate = custody.replay(drive_root).get(pending_run_id)
        if candidate is not None and candidate.task_id == task_id and candidate.settled:
            settled_holder = candidate
    if len(runs) + len(pending) + int(settled_holder is not None) != 1:
        return {}
    holder: Any = runs[0] if runs else pending[0] if pending else settled_holder
    holder_is_pending = bool(pending)
    config_fingerprint = _holder_value(holder, "config_fingerprint", pending=holder_is_pending)
    selected_subagent_id = _holder_value(holder, "selected_subagent_id", pending=holder_is_pending)
    holder_work_order = _holder_value(holder, "work_order_fingerprint", pending=holder_is_pending)
    holder_authority = _holder_value(holder, "authority_fingerprint", pending=holder_is_pending)
    authority_fp = authority_fingerprint_from_task(task)
    if (
        not config_fingerprint
        or not selected_subagent_id
        or not holder_work_order
        or holder_authority != authority_fp
    ):
        return {}
    snapshot = _selected_session(task)
    task_snapshot_matches_holder = bool(
        snapshot
        and selected_subagent_id == str(snapshot.get("selected_subagent_id") or "")
        and config_fingerprint == str(snapshot.get("config_fingerprint") or "")
        and holder_work_order == work_order_fingerprint(task)
    )
    # A pending invocation is replayed through its original idempotency key, which
    # still performs a POST.  The replacement/root-direct extension is adoption-only:
    # it follows an already-started (or already-settled) exact leaf and never widens
    # into replaying a selector that was not the task-start snapshot.
    if holder_is_pending and not task_snapshot_matches_holder:
        return {}
    row = {
        "schema": 1,
        "status": "reserved",
        "cause": cause,
        "task_id": task_id,
        "old_attempt": int(old_attempt),
        "new_attempt": int(new_attempt),
        "worker_id": int(worker_id),
        "supervisor_pid": os.getpid(),
        "exitcode": exitcode,
        "restart_transaction_id": str(restart_transaction_id or ""),
        "selected_subagent_id": selected_subagent_id,
        "config_fingerprint": config_fingerprint,
        "authority_fingerprint": authority_fp,
        "work_order_fingerprint": holder_work_order,
        "actor_binding_source": (
            "task_snapshot" if task_snapshot_matches_holder else "current_custody_holder"
        ),
        "run_id": runs[0].run_id if runs else settled_holder.run_id if settled_holder else "",
        "pending_invocation_id": str(pending[0].get("invocation_id") or "") if pending else "",
        "settled_terminal": settled_holder is not None,
        "snapshot_id": holder.snapshot_id if (runs or settled_holder) else str(holder.get("snapshot_id") or ""),
        "execution_root": holder.execution_root if (runs or settled_holder) else str(holder.get("execution_root") or ""),
        "baseline_sha": holder.baseline_sha if (runs or settled_holder) else str(holder.get("baseline_sha") or ""),
        "target_root": holder.target_root if (runs or settled_holder) else str(holder.get("target_root") or ""),
        "journal_cursor": int(supervision.get("journal_cursor") or 0),
        "mailbox_acknowledged_ids": list(supervision.get("mailbox_acknowledged_ids") or []),
        "interaction_acknowledged_ids": list(
            supervision.get("interaction_acknowledged_ids") or []
        ),
        "pending_wake": dict(pending_wake),
        "checkpoint": supervision.get("checkpoint") if isinstance(supervision.get("checkpoint"), dict) else {},
        "no_resume_veto_causes": list(NO_RESUME_CAUSES),
        "created_at": utc_now_iso(),
    }
    _write(drive_root, row)
    return row


def recoverable_task_ids(drive_root: Any) -> set[str]:
    root = pathlib.Path(drive_root) / "state" / "delegate_recovery"
    if not root.exists():
        return set()
    task_ids: set[str] = set()
    for path in root.glob("*.json"):
        row = _read(drive_root, path.stem)
        same_generation_crash = (
            row.get("cause") == CAUSE_WORKER_CRASH
            and int(row.get("supervisor_pid") or 0) == os.getpid()
        )
        planned_restart = (
            row.get("cause") == CAUSE_PLANNED_SELF_RESTART
            and (
                int(row.get("supervisor_pid") or 0) == os.getpid()
                or (
                    (tx := _read_restart_transaction(
                        drive_root, str(row.get("restart_transaction_id") or "")
                    )).get("status") == "normal_exit_acknowledged"
                    and str(row.get("task_id") or "") in set(tx.get("task_ids") or [])
                )
            )
        )
        if row.get("status") in {"reserved", "pre_adopted"} and (same_generation_crash or planned_restart):
            task_ids.add(str(row.get("task_id") or ""))
    return {task_id for task_id in task_ids if task_id}


def has_planned_restart_handoffs(drive_root: Any) -> bool:
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    active = _read_restart_transaction(drive_root, "active")
    transaction = _read_restart_transaction(drive_root, str(active.get("transaction_id") or ""))
    if transaction.get("status") == "prepared" and transaction.get("supervisor_pid") == os.getpid():
        for task_id in transaction.get("task_ids", []):
            row = load_task_result(drive_root, task_id, strict=True) or {}
            if (row.get("status") not in _TRULY_TERMINAL_STATUSES
                    and (row.get("owner_wait") or {}).get("state") == "waiting"):
                return True
    root = pathlib.Path(drive_root) / "state" / "delegate_recovery"
    if not root.exists():
        return False
    return any(
        (row := _read(drive_root, path.stem)).get("status") in {"reserved", "pre_adopted"}
        and row.get("cause") == CAUSE_PLANNED_SELF_RESTART
        for path in root.glob("*.json")
    )


def prepare_planned_restart_handoffs(
    drive_root: Any,
    running: Mapping[str, Any],
    *,
    restart_transaction_id: str = "",
    additional_task_ids: Optional[set[str]] = None,
) -> set[str]:
    """Reserve only exact tasks durably in event-only supervising sleep."""

    transaction_id = str(restart_transaction_id or uuid.uuid4().hex)
    preserved: set[str] = set(additional_task_ids or ())
    for task_id, meta in dict(running or {}).items():
        task = meta.get("task") if isinstance(meta, dict) else None
        if not isinstance(task, dict):
            continue
        checkpoint_path = (
            pathlib.Path(drive_root) / "state" / "delegate_supervision" / f"{task_id}.json"
        )
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(checkpoint, dict) or checkpoint.get("status") not in {
            "sleeping", "wake_pending",
        }:
            continue
        old_attempt = int(meta.get("attempt") or task.get("_attempt") or 1)
        row = prepare_handoff(
            drive_root,
            task,
            cause=CAUSE_PLANNED_SELF_RESTART,
            old_attempt=old_attempt,
            new_attempt=old_attempt + 1,
            worker_id=int(meta.get("worker_id") or 0),
            exitcode=None,
            restart_transaction_id=transaction_id,
        )
        if not row or str(row.get("run_id") or "") != str(checkpoint.get("run_id") or ""):
            if row:
                veto_handoff(drive_root, str(task_id), "sleep_checkpoint_run_mismatch")
            continue
        row["journal_cursor"] = int(checkpoint.get("journal_cursor") or 0)
        row["mailbox_acknowledged_ids"] = list(
            checkpoint.get("mailbox_acknowledged_ids") or []
        )
        row["interaction_acknowledged_ids"] = list(
            checkpoint.get("interaction_acknowledged_ids") or []
        )
        row["pending_wake"] = (
            dict(checkpoint.get("pending_wake"))
            if isinstance(checkpoint.get("pending_wake"), dict) else {}
        )
        row["checkpoint"] = checkpoint.get("checkpoint") if isinstance(checkpoint.get("checkpoint"), dict) else {}
        _write(drive_root, row)
        preserved.add(str(task_id))
    if preserved:
        transaction = {
            "schema": 1, "transaction_id": transaction_id, "status": "prepared",
            "supervisor_pid": os.getpid(), "task_ids": sorted(preserved),
            "prepared_at": utc_now_iso(), "expected_exit_code": 42,
        }
        _write_restart_transaction(drive_root, transaction)
        active_path = _active_restart_transaction_path(drive_root)
        active_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(active_path, {
            "transaction_id": transaction_id, "supervisor_pid": os.getpid(),
            "prepared_at": transaction["prepared_at"],
        })
        custody.emit(drive_root, "delegate_restart_transaction_prepared", {
            "restart_transaction_id": transaction_id, "supervisor_pid": os.getpid(),
            "task_ids": sorted(preserved), "expected_exit_code": 42,
        })
    return preserved


def pre_adopt_planned_handoffs(
    drive_root: Any, pending_tasks: list[Mapping[str, Any]],
) -> set[str]:
    """Durably adopt exact planned-restart runs before startup orphan cleanup."""

    _ack_direct_exec_successor(drive_root)
    adopted: set[str] = set()
    pending_tasks_by_id = {
        str(task.get("id") or ""): task
        for task in list(pending_tasks or []) if str(task.get("id") or "")
    }
    root = pathlib.Path(drive_root) / "state" / "delegate_recovery"
    no_resume_flag = next((
        name for name in ("owner_restart_no_resume.flag", "panic_stop.flag")
        if (pathlib.Path(drive_root) / "state" / name).exists()
    ), "")
    for path in root.glob("*.json") if root.exists() else ():
        task_id = path.stem
        row = _read(drive_root, task_id)
        if row.get("cause") != CAUSE_PLANNED_SELF_RESTART or row.get("status") not in {
            "reserved", "pre_adopted",
        }:
            continue
        task = pending_tasks_by_id.get(task_id)
        old_pid = int(row.get("supervisor_pid") or 0)
        transaction_id = str(row.get("restart_transaction_id") or "")
        transaction = _read_restart_transaction(drive_root, transaction_id)
        mismatch = ""
        if no_resume_flag:
            mismatch = f"startup_no_resume:{no_resume_flag}"
        elif not transaction_id:
            mismatch = "restart_transaction_missing"
        elif transaction.get("status") != "normal_exit_acknowledged":
            mismatch = "restart_normal_exit_unproven"
        elif int(transaction.get("supervisor_pid") or 0) != old_pid:
            mismatch = "restart_transaction_pid_mismatch"
        elif int(transaction.get("exit_code") or 0) != 42:
            mismatch = "restart_transaction_exit_mismatch"
        elif task_id not in set(transaction.get("task_ids") or []):
            mismatch = "restart_transaction_task_mismatch"
        elif task is None:
            mismatch = "startup_restored_task_missing"
        elif old_pid != os.getpid() and _pid_alive(old_pid):
            mismatch = "previous_supervisor_generation_not_dead"
        elif int(row.get("new_attempt") or 0) != int(task.get("_attempt") or 1):
            mismatch = "startup_attempt_mismatch"
        elif binding_mismatch := _successor_binding_mismatch(row, task):
            mismatch = f"startup_{binding_mismatch}"
        if mismatch:
            veto_handoff(drive_root, task_id, mismatch)
            continue
        runs = [candidate for candidate in custody.open_runs(drive_root) if candidate.task_id == task_id]
        pending_invocations_for_task = [
            candidate for candidate in custody.pending_invocations(drive_root)
            if str(candidate.get("task_id") or "") == task_id
        ]
        settled = custody.replay(drive_root).get(str(row.get("run_id") or ""))
        if row.get("settled_terminal") is True:
            if (
                runs
                or pending_invocations_for_task
                or settled is None
                or not settled.settled
                or not _holder_matches(row, settled, pending=False)
            ):
                veto_handoff(drive_root, task_id, "startup_settled_binding_mismatch")
                continue
            adopted_run_id = settled.run_id
        elif str(row.get("pending_invocation_id") or ""):
            if (
                runs
                or len(pending_invocations_for_task) != 1
                or not _holder_matches(row, pending_invocations_for_task[0], pending=True)
            ):
                veto_handoff(drive_root, task_id, "startup_invocation_binding_mismatch")
                continue
            adopted_run_id = ""
        else:
            if len(runs) != 1 or not _holder_matches(row, runs[0], pending=False):
                veto_handoff(drive_root, task_id, "startup_run_binding_mismatch")
                continue
            gateway = None
            try:
                from ouroboros.claudexor_daemon import ensure_owned_gateway

                gateway = ensure_owned_gateway()
                gateway.get_run(runs[0].run_id)
            except Exception:
                veto_handoff(drive_root, task_id, "startup_run_unprovable")
                continue
            finally:
                if gateway is not None:
                    gateway.close()
            adopted_run_id = runs[0].run_id
        row.update({"status": "pre_adopted", "pre_adopted_at": utc_now_iso()})
        try:
            _write(drive_root, row)
        except Exception:
            try:
                custody.reconcile_task_runs(drive_root, task_id)
            except Exception:
                pass
            continue
        custody.emit(
            drive_root,
            "delegate_recovery_pre_adopted" if not adopted_run_id else "delegate_run_adopted",
            {
                "task_id": task_id,
                "run_id": adopted_run_id,
                "pending_invocation_id": str(row.get("pending_invocation_id") or ""),
                "cause": CAUSE_PLANNED_SELF_RESTART,
                "phase": "before_startup_orphan_sweep",
                "config_fingerprint": str(row.get("config_fingerprint") or ""),
                "authority_fingerprint": str(row.get("authority_fingerprint") or ""),
            },
        )
        adopted.add(task_id)
    return adopted


def veto_handoff(drive_root: Any, task_id: str, reason: str) -> dict[str, Any]:
    row = _read(drive_root, task_id)
    if not row:
        return {}
    row.update({"status": "vetoed", "veto_reason": str(reason or "recovery_mismatch")})
    _write(drive_root, row)
    try:
        custody.reconcile_task_runs(drive_root, task_id)
    except Exception:
        pass
    return row


def adopt_handoff(ctx: Any, task: Mapping[str, Any]) -> dict[str, Any]:
    """Adopt the exact live run/pending invocation before any new start or LLM."""

    drive = custody.custody_root(ctx)
    task_id = str(task.get("id") or "")
    row = _read(drive, task_id)
    if not row or row.get("status") not in {"reserved", "pre_adopted"}:
        return {"status": "none"}
    expected_attempt = int(task.get("_attempt") or 1)
    planned_restart_mismatch = ""
    if row.get("cause") == CAUSE_PLANNED_SELF_RESTART:
        transaction_id = str(row.get("restart_transaction_id") or "")
        transaction = _read_restart_transaction(drive, transaction_id)
        if not transaction_id:
            planned_restart_mismatch = "restart_transaction_missing"
        elif transaction.get("status") != "normal_exit_acknowledged":
            planned_restart_mismatch = "restart_normal_exit_unproven"
        elif int(transaction.get("supervisor_pid") or 0) != int(row.get("supervisor_pid") or 0):
            planned_restart_mismatch = "restart_transaction_pid_mismatch"
        elif int(transaction.get("exit_code") or 0) != 42:
            planned_restart_mismatch = "restart_transaction_exit_mismatch"
        elif task_id not in set(transaction.get("task_ids") or []):
            planned_restart_mismatch = "restart_transaction_task_mismatch"
    if (
        planned_restart_mismatch
        or row.get("cause") not in _ALLOWED_CAUSES
        or (
            row.get("cause") == CAUSE_WORKER_CRASH
            and int(row.get("supervisor_pid") or 0) != os.getpid()
        )
        or int(row.get("new_attempt") or 0) != expected_attempt
        or _successor_binding_mismatch(row, task)
    ):
        reason = planned_restart_mismatch or "successor_binding_mismatch"
        veto_handoff(drive, task_id, reason)
        return {"status": "recovery_required", "reason": reason}
    runs = [candidate for candidate in custody.open_runs(drive) if candidate.task_id == task_id]
    pending = [
        candidate for candidate in custody.pending_invocations(drive)
        if str(candidate.get("task_id") or "") == task_id
    ]
    pending_wake = row.get("pending_wake") if isinstance(row.get("pending_wake"), dict) else {}
    wake_payload = (
        dict(pending_wake.get("payload"))
        if isinstance(pending_wake.get("payload"), dict) else {}
    )
    if row.get("settled_terminal") is True:
        settled = custody.replay(drive).get(str(row.get("run_id") or ""))
        if runs or pending or settled is None or not settled.settled or not _holder_matches(
            row, settled, pending=False,
        ):
            veto_handoff(drive, task_id, "settled_terminal_binding_mismatch")
            return {"status": "recovery_required", "reason": "settled_terminal_binding_mismatch"}
        try:
            _restore_wait_checkpoint(drive, row)
        except Exception:
            return {"status": "recovery_required", "reason": "checkpoint_restore_failed"}
        row.update({"status": "adopted", "adopted_at": utc_now_iso()})
        try:
            _write(drive, row)
        except Exception:
            return {"status": "recovery_required", "reason": "adoption_record_unwritable"}
        custody.emit(drive, "delegate_run_adopted", {
            "task_id": task_id, "run_id": settled.run_id,
            "cause": str(row.get("cause") or ""), "phase": "settled_wake_replay",
            "config_fingerprint": str(row.get("config_fingerprint") or ""),
            "authority_fingerprint": str(row.get("authority_fingerprint") or ""),
        })
        return {
            "status": "settled_recovered", "run_id": settled.run_id,
            "cause": row.get("cause"), "wake": wake_payload,
        }
    if len(runs) + len(pending) != 1:
        veto_handoff(drive, task_id, "ambiguous_or_missing_leaf")
        return {"status": "recovery_required", "reason": "ambiguous_or_missing_leaf"}
    if runs:
        run = runs[0]
        if not _holder_matches(row, run, pending=False):
            veto_handoff(drive, task_id, "run_binding_mismatch")
            return {"status": "recovery_required", "reason": "run_binding_mismatch"}
        if probe_error := _run_probe_error(run.run_id):
            veto_handoff(drive, task_id, f"run_unprovable:{probe_error}")
            return {"status": "recovery_required", "reason": "run_unprovable"}
        run_id = run.run_id
    else:
        pending_id = str(pending[0].get("invocation_id") or "")
        if not _holder_matches(row, pending[0], pending=True):
            veto_handoff(drive, task_id, "invocation_binding_mismatch")
            return {"status": "recovery_required", "reason": "invocation_binding_mismatch"}
        from ouroboros.tools.delegate import exact_start
        request = pending[0].get("request") if isinstance(pending[0], dict) else None
        replay_prompt = request.get("prompt") if isinstance(request, dict) else None
        if not isinstance(replay_prompt, str) or not replay_prompt:
            veto_handoff(drive, task_id, "pending_request_prompt_missing")
            return {"status": "recovery_required", "reason": "pending_request_prompt_missing"}
        retry_spec = {"retry_of": pending_id}
        source_request = pending[0].get("work_order_source_request")
        if isinstance(source_request, dict) and source_request:
            retry_spec["work_order_source_request"] = source_request
        # Consume the reservation before transport: RUNNING protects this owner;
        # a later crash can mint a fresh handoff from durable pending/run custody.
        row.update({"status": "adopted", "adopted_at": utc_now_iso()})
        try:
            _write(drive, row)
        except Exception:
            return {"status": "recovery_required", "reason": "adoption_record_unwritable"}
        # The exact-start primitive answers with the family's NATIVE result; the
        # durable invocation fate below, not this payload, decides the outcome.
        from ouroboros.delegate_shared import delegate_payload

        parsed = delegate_payload(exact_start(ctx, replay_prompt, retry_spec))
        run_id = str(parsed.get("run_id") or "")
        if str(parsed.get("status") or "") != "started" or not run_id:
            # Durable fate, not transport prose, decides whether this exact POST
            # remains pending, failed definitely, or became an adoptable run.
            # Missing/ambiguous evidence stays fail-closed without a false pending id.
            fate = custody.invocation_record(drive, pending_id) or {}
            fate_state = str(fate.get("state") or "unknown")
            if fate_state == "pending":
                return {
                    "status": "recovery_required", "reason": "pending_recovery_deferred",
                    "pending_invocation_id": pending_id,
                    "detail": parsed,
                }
            if fate_state == "failed_definite":
                veto_handoff(drive, task_id, "pending_recovery_definitely_refused")
                return {
                    "status": "recovery_required",
                    "reason": "pending_recovery_definitely_refused", "detail": parsed,
                }
            if fate_state == "started":
                run_id = str(fate.get("run_id") or "")
                started = custody.replay(drive).get(run_id)
                rebound = {**row, "run_id": run_id}
                if (
                    started is None
                    or str(started.invocation_id or "") != pending_id
                    or not _holder_matches(rebound, started, pending=False)
                ):
                    run_id = ""
                elif not started.settled and _run_probe_error(run_id):
                    return {
                        "status": "recovery_required",
                        "reason": "pending_recovery_run_unprovable", "run_id": run_id,
                        "detail": parsed,
                    }
            if not run_id:
                return {
                    "status": "recovery_required",
                    "reason": "pending_recovery_state_unprovable",
                    "invocation_id": pending_id,
                    "invocation_state": fate_state,
                    "detail": parsed,
                }
    try:
        _restore_wait_checkpoint(drive, {**row, "run_id": run_id})
    except Exception:
        try:
            custody.reconcile_task_runs(drive, task_id)
        except Exception:
            pass
        return {"status": "recovery_required", "reason": "checkpoint_restore_failed"}
    row.update({"status": "adopted", "run_id": run_id, "adopted_at": utc_now_iso()})
    try:
        _write(drive, row)
    except Exception:
        try:
            custody.reconcile_task_runs(drive, task_id)
        except Exception:
            pass
        return {"status": "recovery_required", "reason": "adoption_record_unwritable"}
    custody.emit(drive, "delegate_run_adopted", {
        "task_id": task_id,
        "run_id": run_id,
        "cause": str(row.get("cause") or ""),
        "config_fingerprint": str(row.get("config_fingerprint") or ""),
        "authority_fingerprint": str(row.get("authority_fingerprint") or ""),
        "new_attempt": expected_attempt,
    })
    return {
        "status": "adopted", "run_id": run_id, "cause": row.get("cause"),
        **({"wake": wake_payload} if wake_payload else {}),
    }


__all__ = [
    "CAUSE_PLANNED_SELF_RESTART",
    "CAUSE_WORKER_CRASH",
    "NO_RESUME_CAUSES",
    "PLANNED_RESTART_TRANSACTION_ENV",
    "arm_active_planned_restart_transaction",
    "acknowledge_observed_restart_exit",
    "adopt_handoff",
    "authority_fingerprint_from_context",
    "authority_fingerprint_from_task",
    "has_planned_restart_handoffs",
    "prepare_handoff",
    "prepare_planned_restart_handoffs",
    "prepare_worker_crash_handoff",
    "pre_adopt_planned_handoffs",
    "reconcile_unrecoverable_task",
    "recoverable_task_ids",
    "unsettled_start_ids",
    "veto_handoff",
    "veto_worker_retry_handoff",
]

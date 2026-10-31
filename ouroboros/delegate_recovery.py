"""Narrow successor proofs for configured-session worker/restart recovery."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Mapping, Optional

from ouroboros import delegate_custody as custody
from ouroboros import delegate_recovery_holders as holders
from ouroboros import delegate_supervision_state as checkpoints
from ouroboros.platform_layer import IS_WINDOWS
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
PLANNED_RESTART_PARENT_ENV = "OUROBOROS_PLANNED_RESTART_PARENT_HANDLE"
PLANNED_RESTART_BINDING_ENV = "OUROBOROS_PLANNED_RESTART_BINDING_HANDLE"
WINDOWS_PARENT_ACK = "windows_direct_parent_handle"
_restart_parent: Optional[dict[str, Any]] = None  # this process's one observation of an inherited parent


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


def _read(drive_root: Any, task_id: str, *, strict: bool = False) -> dict[str, Any]:
    try:
        data = json.loads(_path(drive_root, task_id).read_text(encoding="utf-8"))
        if not isinstance(data, dict) and strict:
            raise ValueError("delegate handoff is not an object")
        return holders.read_view(data)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        if strict:
            raise
        return {}


def _write(drive_root: Any, row: dict[str, Any]) -> None:
    path = _path(drive_root, str(row.get("task_id") or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, holders.write_view(row))
    from ouroboros.maintenance_retirement import artifact_changed
    artifact_changed(drive_root)


def _restart_transaction_path(drive_root: Any, transaction_id: str) -> pathlib.Path:
    return (
        pathlib.Path(drive_root) / "state" / "delegate_recovery_transactions"
        / f"{transaction_id}.json"
    )


def _active_restart_transaction_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / "delegate_recovery_transactions" / "active.json"


_PROCESS_GENERATION = uuid.uuid4().hex
_LEGACY_RESTART_FLAG = "owner_restart_no_resume.flag"
_LEGACY_RESTART_BYTE = "owner_restart_no_resume"


def active_restart_transaction(root: Any) -> dict:
    active = _read_restart_transaction(root, "active")
    return _read_restart_transaction(root, str(active.get("transaction_id") or "")) if active else {}


def prepare_owner_restart(root: Any, *, legacy: bool = False) -> dict:
    """Publish the door before stopping work; restore the prior pointer on failure."""
    path = _active_restart_transaction_path(root)
    previous = path.read_bytes() if path.exists() else None
    row = {"schema": 1, "transaction_id": uuid.uuid4().hex, "door": "owner_restart",
           "status": "prepared", "supervisor_pid": os.getpid(),
           "generation": "legacy" if legacy else _PROCESS_GENERATION,
           "door_applied_at": "", "task_ids": [], "prepared_at": utc_now_iso(), "expected_exit_code": 42}
    try:
        _write_restart_transaction(root, row)
        atomic_write_json(path, {key: row[key] for key in ("transaction_id", "supervisor_pid", "prepared_at")})
    except Exception:
        from ouroboros.utils import write_bytes_atomic

        if previous is None:
            path.unlink(missing_ok=True)
        else:
            write_bytes_atomic(path, previous)
        _restart_transaction_path(root, row["transaction_id"]).unlink(missing_ok=True)
        raise
    return row


def import_legacy_restart(root: Any) -> None:
    """Translate old flags once at boot, never from a work decision."""
    row = active_restart_transaction(root)
    state = pathlib.Path(root) / "state"
    old_flag = state / _LEGACY_RESTART_FLAG
    compat = state / "panic_stop.flag"
    try:
        old_byte = compat.read_text(encoding="utf-8").strip() == _LEGACY_RESTART_BYTE
    except FileNotFoundError:
        old_byte = False
    if old_flag.exists() or old_byte:
        if row.get("door") != "owner_restart" or row.get("door_applied_at"):
            # Preserve an unconsumed transaction and its ACK. A legacy flag has
            # no transaction identity and cannot rename an active update's door.
            if not row or row.get("returns_restored_at") or row.get("door_applied_at"):
                row = prepare_owner_restart(root, legacy=True)
        old_flag.unlink(missing_ok=True)
        if old_byte:
            compat.unlink(missing_ok=True)


def boot_door(root) -> str:
    """Translate unconsumed installation inputs at snapshot restoration only."""
    from ouroboros.installation_stop import installation_panic

    import_legacy_restart(root)

    if installation_panic(root):
        return "panic"
    return "owner_restart" if pending_owner_restart(root) else ""


def pending_owner_restart(root: Any) -> dict:
    """The unconsumed prior-generation door; ACK separately governs returns."""
    row = active_restart_transaction(root)
    if (row.get("door") == "owner_restart" and not row.get("door_applied_at")
            and row.get("status") in {"prepared", "normal_exit_acknowledged"}
            and (row.get("generation") != _PROCESS_GENERATION or row.get("supervisor_pid") != os.getpid())):
        return row
    return {}


def owner_restart_in_progress(root: Any) -> bool:
    """Server latches, or a worker's own parent's prepared door; never Panic."""
    import multiprocessing

    parent = multiprocessing.parent_process()
    if parent is not None:
        row = active_restart_transaction(root)
        return bool(row.get("door") == "owner_restart" and row.get("status") == "prepared"
                    and not row.get("door_applied_at") and row.get("supervisor_pid") == parent.pid)
    from ouroboros import server_restart

    return bool(pathlib.Path(root) == pathlib.Path(server_restart.DATA_DIR)
                and server_restart._OWNER_STOP_BEGAN.is_set())


def consume_owner_restart(root: Any, row: dict) -> None:
    current = _read_restart_transaction(root, row["transaction_id"])
    _write_restart_transaction(root, {**current, "door_applied_at": utc_now_iso()})


def arm_active_planned_restart_transaction(drive_root: Any) -> str:
    """Pass a prepared restart transaction to a direct exec/spawn successor.

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
    from ouroboros.maintenance_retirement import artifact_changed
    artifact_changed(drive_root)


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


@contextmanager
def direct_restart_transaction(drive_root: Any, transaction_id: str):
    """Serialize spawn publication with the child's first read of this exact handoff."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    if not transaction_id:
        yield {}
        return
    path = _restart_transaction_path(drive_root, transaction_id).with_suffix(".lock")
    fd = acquire_exclusive_file_lock(path, timeout_sec=5.0, owner_aware_stale=True)
    try:
        active = _read_restart_transaction(drive_root, "active")
        row = _read_restart_transaction(drive_root, transaction_id)
        valid = (fd is not None and row.get("status") == "prepared" and not row.get("returns_restored_at")
                 and row.get("expected_exit_code") == 42
                 and all(active.get(key) == row.get(key) for key in
                         ("transaction_id", "supervisor_pid", "prepared_at"))
                 and row.get("transaction_id") == transaction_id)
        yield row if valid else {}
    finally:
        release_exclusive_file_lock(path, fd)


def direct_spawn_successor_matches(row: Mapping[str, Any]) -> bool:
    """Only the measured child of this successful supervised spawn may acknowledge it."""
    from ouroboros.process_custody import _fingerprint_matches

    entry = row.get("direct_spawn_successor")
    return bool(isinstance(entry, dict) and entry.get("pid") == os.getpid()
                and entry.get("purpose") == "server_restart_fallback" and entry.get("scope") == "daemon"
                and _fingerprint_matches(entry, require_measured=True))


def spawned_successor_is_current(row: Mapping[str, Any]) -> bool:
    """A spawn acknowledgement lends its returns only to the exact successor it bound."""
    if row.get("ack_source") == WINDOWS_PARENT_ACK:
        from ouroboros.platform_layer import process_start_time

        return (row.get("successor_pid"), row.get("successor_birth")) == (os.getpid(), process_start_time(os.getpid()))
    return row.get("ack_source") != "direct_spawn_successor" or direct_spawn_successor_matches(row)


def bind_restart_successor(drive_root: Any, row: dict[str, Any], proc: Any) -> None:
    """Inside the held transaction, the spawning parent names its one successor.

    POSIX (exec failed): the exact custodied spawn. Windows: this generation's birth
    and the successor's PID/birth, which count only after the successor has observed
    this exact parent's exit 42 through its inherited handle.
    """
    if IS_WINDOWS:
        from ouroboros.platform_layer import process_start_time

        births = process_start_time(os.getpid()), process_start_time(int(proc.pid))
        if not all(births):
            raise ValueError(f"restart transaction {row.get('transaction_id')!r} cannot bind its successor")
        row.update(supervisor_birth=births[0], successor_pid=int(proc.pid), successor_birth=births[1])
    else:
        row["direct_spawn_successor"] = proc._ouroboros_custody
    _write_restart_transaction(drive_root, row)


def observe_restart_parent() -> dict[str, Any]:
    """Wait once for the parent a Windows direct restart passed as an inherited handle.

    Returns its ``pid``, ``birth`` and ``exit_code`` (empty without a handle or an
    observation); later readers share this one result, and children never inherit it.
    """
    global _restart_parent
    if _restart_parent is None:
        _restart_parent, handle = {}, os.environ.pop(PLANNED_RESTART_PARENT_ENV, "")
        binding = os.environ.pop(PLANNED_RESTART_BINDING_ENV, "")
        if binding:
            try:
                from ouroboros.platform_layer import await_restart_binding

                await_restart_binding(int(binding))
            except Exception as exc:
                log.warning("The bootstrap restart binding wait failed: %s", exc)
        if handle:
            try:
                from ouroboros.platform_layer import await_process_handle

                _restart_parent = await_process_handle(int(handle))
            except Exception as exc:
                log.warning("The restart parent handle could not be observed: %s", exc)
    return _restart_parent


def _windows_successor_problem(row: Mapping[str, Any], parent: Mapping[str, Any]) -> str:
    from ouroboros.platform_layer import process_start_time

    if not row:
        return "the transaction is not this restart's active prepared one"
    if parent.get("exit_code") != 42:
        return f"parent exit {parent.get('exit_code', 'unobserved')}"
    if (parent.get("pid"), parent.get("birth")) != (int(row.get("supervisor_pid") or 0), row.get("supervisor_birth")):
        return "the exited parent is not the prepared generation"
    if (row.get("successor_pid"), row.get("successor_birth")) != (os.getpid(), process_start_time(os.getpid())):
        return "this process is not the bound successor"
    return ""


def _transfer_proof(transaction_id: str, row: Mapping[str, Any], parent: Mapping[str, Any]) -> str:
    """This process's acknowledgement source for a valid prepared row, else ``""``.

    POSIX: same-PID exec, or the exact custodied spawn after a failed exec. Windows
    accepts neither: only the inherited parent's observed exit 42 and the identities
    that parent bound, read after that wait.
    """
    if IS_WINDOWS:
        if problem := _windows_successor_problem(row, parent):
            log.warning("Restart %s continuation unproven, its handoffs are not resumed: %s", transaction_id, problem)
            return ""
        return WINDOWS_PARENT_ACK
    if not row:
        return ""
    if "direct_spawn_successor" in row:
        return "direct_spawn_successor" if direct_spawn_successor_matches(row) else ""
    return "direct_exec_successor" if int(row.get("supervisor_pid") or 0) == os.getpid() else ""


def _ack_direct_exec_successor(drive_root: Any) -> None:
    """A one-shot token plus this platform's transfer proof acknowledges the restart."""

    transaction_id = str(os.environ.pop(PLANNED_RESTART_TRANSACTION_ENV, "") or "")
    if not transaction_id:
        return
    parent = observe_restart_parent() if IS_WINDOWS else {}
    with direct_restart_transaction(drive_root, transaction_id) as row:
        ack_source = _transfer_proof(transaction_id, row, parent)
        if not ack_source:
            return
        row.update({
            "status": "normal_exit_acknowledged", "exit_code": 42,
            "exit_acknowledged_at": utc_now_iso(), "ack_source": ack_source,
        })
        _write_restart_transaction(drive_root, row)
    custody.emit(drive_root, "delegate_restart_transaction_acknowledged", {
        "restart_transaction_id": transaction_id,
        "supervisor_pid": row["supervisor_pid"], "exit_code": 42,
        "task_ids": list(row.get("task_ids") or []), "ack_source": ack_source,
    })


def _selected_session(task: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = task.get("configured_subagent") if isinstance(task.get("configured_subagent"), dict) else {}
    route = snapshot.get("route") if isinstance(snapshot.get("route"), dict) else {}
    return snapshot if str(route.get("kind") or "") == "agent_session" else {}


def unsettled_start_ids(
    drive_root: Any, task_id: str, *, rows: Optional[list[dict[str, Any]]] = None, continuing: str = "",
) -> dict[str, list[str]]:
    """Durable run/start obligations from one consistent custody-log snapshot.

    A review run is the review substrate's obligation, not the actor's
    delegation slot, so it never blocks the actor's own start; nor does the
    undisposed candidate patch of the run a start is ``continuing``, nor that run
    once its terminal is observed (its accounting finishes in the background, #1969).
    Evidence-only captures owe no apply debt. A readonly continuation leaves a
    candidate patch waiting for its own disposition; it is not a fresh replacement.
    """

    mine = str(task_id or "")
    if rows is not None:
        snapshot: Optional[list[dict[str, Any]]] = list(rows)
        runs = custody.replay(drive_root, rows=snapshot)
    else:
        from ouroboros.delegate_custody_current import runs as current_runs

        snapshot, runs = None, current_runs(drive_root, "task_id=?", (mine,))
    return {
        "open_run_ids": [
            row.run_id for row in runs.values()
            if row.task_id == mine and not row.settled and not row.review_owned
            and not (row.run_id == continuing and row.terminal_state in custody.TERMINAL_STATES)
        ],
        "pending_invocation_ids": [
            str(row.get("invocation_id") or "")
            for row in custody.pending_invocations(drive_root, rows=snapshot)
            if str(row.get("task_id") or "") == mine
            and not custody.review_owned_source(row.get("source"))
        ],
        "undisposed_patch_run_ids": [
            row.run_id for row in runs.values()
            if row.task_id == mine and row.snapshot_id and row.settled
            and custody.candidate_apply_possible(row)
            and not row.patch_disposed and not row.review_owned and row.run_id != continuing
        ],
    }


def reconcile_unrecoverable_task(drive_root: Any, task_id: str) -> None:
    """Fail-soft cancellation/settlement for a task without a valid successor."""

    try:
        from ouroboros import delegate_terminal
        from ouroboros.delegate_custody_reconcile import attach_only_gateway

        result = delegate_terminal.terminal_reconcile_task(  # a release: attach only, no replay
            drive_root, str(task_id or ""), trigger="unrecoverable_successor",
            gateway_factory=attach_only_gateway,
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


def _saved_handoff_holder(drive_root: Any, row: Mapping[str, Any]) -> tuple[Any, dict, str]:
    """Prove the reserved identity locally, including pending→bound and late settlement.

    One current task snapshot sees both sides of STARTED. Absence from open
    custody alone never loses a settled run. An unreadable store retains the
    reservation; a proved different holder still vetoes it. A pending
    reservation whose exact invocation was definitively refused, or bound and
    settled beside newer custody, is ``saved_invocation_resolved``.
    """
    from ouroboros.delegate_custody_current import task_snapshot

    try:
        view = task_snapshot(drive_root, str(row.get("task_id") or ""))
    except Exception:
        return None, {}, "custody_unreadable"
    runs = [run for run in view["runs"].values() if not run.review_owned]
    pending = [item for item in view["pending"] if not custody.review_owned_source(item.get("source"))]
    pending_id, run_id = str(row.get("pending_invocation_id") or ""), str(row.get("run_id") or "")
    holders = [run for run in runs if (run.run_id == run_id if run_id else
                                      bool(pending_id) and run.invocation_id == pending_id)]
    requests = [item for item in pending if item.get("invocation_id") == pending_id]
    if not holders and len(requests) == 1 and not row.get("settled_terminal"):
        request = requests[0]
        if _holder_matches(row, request, pending=True):
            return None, request, ""
    if len(holders) == 1:
        run = holders[0]
        if ((not row.get("settled_terminal") or run.settled)
                and (not pending_id or run.invocation_id == pending_id)
                and _holder_matches({**row, "run_id": run.run_id}, run, pending=False)):
            return run, {}, ""
    if pending_id and not run_id and not requests and _pending_resolved(drive_root, row, holders):
        return None, {}, "saved_invocation_resolved"
    return None, {}, "saved_holder_binding_mismatch"


def _pending_resolved(drive_root: Any, row: Mapping[str, Any], bound: list) -> bool:
    """Whether the reserved invocation's own exact fate is final; unknown or foreign is not."""
    if bound:
        return (len(bound) == 1 and bound[0].settled
                and _holder_matches({**row, "run_id": bound[0].run_id}, bound[0], pending=False))
    try:
        fate = custody.invocation_record(drive_root, str(row.get("pending_invocation_id") or "")) or {}
    except Exception:
        return False
    return (fate.get("state") == "failed_definite" and fate.get("task_id") == row.get("task_id")
            and _holder_matches(row, {**fate, "invocation_id": row.get("pending_invocation_id")}, pending=True))


def restore_bound_handoff(ctx: Any, run_id: str) -> None:
    """Carry a deferred pending wait only once its exact invocation has a run.

    Until then its cursor, wake and ACK sources remain in the adopted handoff;
    no empty run-id supervision state is written. Called by the actual wait.
    """
    drive = custody.custody_root(ctx)
    row = _read(drive, str(getattr(ctx, "task_id", "") or ""))
    if not row or int(row.get("new_attempt") or 0) != int(getattr(ctx, "task_attempt", None) or 1):
        return
    for holder in row["holders"]:
        if holder.get("status") != "adopted" or holder.get("run_id") or not holder.get("pending_invocation_id"):
            continue
        run, _, error = _saved_handoff_holder(drive, holders.scoped(row, holder))
        if not error and run is not None and run.run_id == run_id:
            _restore_wait_checkpoint(drive, {**holders.scoped(row, holder), "run_id": run_id})
            holder["run_id"] = run_id
            _write(drive, row)


def _holder_value(holder: Any, name: str, *, pending: bool) -> str:
    return (
        str(holder.get(name) or "")
        if pending else str(getattr(holder, name, "") or "")
    )


def _attach_only():
    """The fail-soft settlement after a vetoed or unwritable handoff is a release:
    it attaches to a serving daemon and never starts one or replays a pending start."""
    from ouroboros.delegate_custody_reconcile import attach_only_gateway

    return attach_only_gateway


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

    if "holders" in row:
        # The task may resume when at least one independently proved holder remains.
        failures = [_successor_binding_mismatch(holders.scoped(dict(row), h), task) for h in row["holders"]]
        return "" if any(not error for error in failures) else failures[0] if failures else "holders_missing"
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
    if not run_id:
        return
    path = pathlib.Path(drive_root) / "state" / "delegate_supervision" / f"{task_id}.json"
    state = checkpoints.run_view(checkpoints.read(path), run_id)
    pending_wake = row.get("pending_wake") if isinstance(row.get("pending_wake"), dict) else {}
    state.update({"status": "wake_pending" if pending_wake else "adopted",
                  "journal_cursor": max(int(state.get("journal_cursor") or 0), int(row.get("journal_cursor") or 0)),
                  "interaction_acknowledged_ids": sorted(set(state.get("interaction_acknowledged_ids") or [])
                                                         | set(row.get("interaction_acknowledged_ids") or []))})
    if pending_wake:
        state["pending_wake"] = dict(pending_wake)
        state["last_wake"] = dict(pending_wake.get("payload") or {})
    if isinstance(row.get("checkpoint"), dict):
        state["checkpoint"] = dict(row["checkpoint"])
    for key, value in checkpoints.task_facts(row).items():
        if key not in state:
            state[key] = value
    checkpoints.save_view(path, drive_root, task_id, state)
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


def _handoff_wait_source(drive_root: Any, task: Mapping[str, Any], attempt: int) -> Optional[dict]:
    """All run checkpoints, plus unbound reservations whose wait sources remain deferred."""
    task_id = str(task.get("id") or "")
    try:
        previous = _read(drive_root, task_id, strict=True)
        state = checkpoints.read(pathlib.Path(drive_root) / "state" / "delegate_supervision" / f"{task_id}.json")
    except (OSError, ValueError, RuntimeError):
        return None
    deferred, blocked = {}, {}
    for holder in previous.get("holders", []):
        if holder.get("status") not in {"adopted", "reserved", "pre_adopted", "deferred"} or not holder.get("pending_invocation_id"):
            continue
        source_attempt = previous.get("new_attempt" if holder.get("status") in {"adopted", "deferred"} else "old_attempt")
        scoped = holders.scoped(previous, holder)
        error = ("pending_wait_owner_mismatch" if previous.get("task_id") != task_id or int(source_attempt or 0) != attempt
                 else _successor_binding_mismatch(scoped, task))
        run, custody_error = None, ""
        if not error:
            run, _, custody_error = _saved_handoff_holder(drive_root, scoped)
            error = custody_error
        if error == "saved_invocation_resolved":
            continue
        if error:
            blocked[holder["pending_invocation_id"]] = {**holder, "status": "deferred", "deferred_reason": error}
            continue
        for key, value in checkpoints.task_facts(previous).items():
            state.setdefault(key, value)
        if run is not None:
            # Resolved old work need not displace the current run's delivered view.
            if not run.settled or holder.get("pending_wake"):
                state["runs"].setdefault(run.run_id, {**holder, "status": "wake_pending" if holder.get("pending_wake") else "adopted"})
        else:
            deferred[holder["pending_invocation_id"]] = dict(holder)
    state.update(deferred_holders=deferred, blocked_holders=blocked)
    return state


def prepare_handoff(
    drive_root: Any, task: Mapping[str, Any], *, cause: str, old_attempt: int, new_attempt: int,
    worker_id: int = 0, exitcode: Optional[int] = None, restart_transaction_id: str = "",
) -> dict[str, Any]:
    """Reserve every exact ordinary holder independently; review custody stays with its panel."""
    if cause not in _ALLOWED_CAUSES or not task.get("id"):
        return {}
    task_id = str(task["id"])
    supervision = _handoff_wait_source(drive_root, task, int(old_attempt))
    if supervision is None or custody.custody_log_unreadable(drive_root):
        return {}
    from ouroboros.delegate_custody_current import task_snapshot

    try:
        view = task_snapshot(drive_root, task_id)
    except Exception:
        return {}
    active = any(not run.settled and not run.review_owned for run in view["runs"].values()) or bool(view["pending"])
    candidates = [(run, False) for run in view["runs"].values() if not run.review_owned
                  and (not run.settled or (run.run_id in supervision["runs"] and
                       (supervision["runs"][run.run_id].get("pending_wake") or not active)))]
    candidates.extend((item, True) for item in view["pending"] if not custody.review_owned_source(item.get("source")))
    reserved = list(supervision["blocked_holders"].values())
    authority = authority_fingerprint_from_task(task)
    for candidate, pending in candidates:
        if pending and candidate.get("invocation_id") in supervision["blocked_holders"]:
            continue
        identity = {key: _holder_value(candidate, key, pending=pending) for key in holders.IDENTITY_FIELDS}
        invocation_id = str(candidate.get("invocation_id") or "") if pending else ""
        run_id = "" if pending else candidate.run_id
        wait = (supervision["deferred_holders"].get(invocation_id, checkpoints.task_facts(supervision)) if pending
                else checkpoints.run_view(supervision, run_id))
        fields = holders.wait_fields(wait, replay_acknowledged=cause == CAUSE_WORKER_CRASH)
        fields["pending_wake"] = _successor_pending_wake(fields["pending_wake"])
        binding_valid = (all(identity[key] for key in ("selected_subagent_id", "config_fingerprint", "work_order_fingerprint"))
                         and identity["authority_fingerprint"] == authority)
        reserved.append({**identity, **fields, "run_id": run_id, "pending_invocation_id": invocation_id,
                         "actor_binding_source": ("task_snapshot" if _selected_session(task)
                             and identity["selected_subagent_id"] == _selected_session(task).get("selected_subagent_id")
                             and identity["config_fingerprint"] == _selected_session(task).get("config_fingerprint")
                             and identity["work_order_fingerprint"] == work_order_fingerprint(task) else "current_custody_holder"),
                         "settled_terminal": not pending and candidate.settled,
                         "status": "reserved" if binding_valid else "refused",
                         **({} if binding_valid else {"veto_reason": "holder_binding_incomplete_or_foreign"})})
    if not any(h.get("status") == "reserved" for h in reserved):
        return {}
    row = {"schema": 2, "task_id": task_id, "status": "reserved", "holders": reserved,
           "cause": cause, "old_attempt": int(old_attempt), "new_attempt": int(new_attempt),
           "worker_id": int(worker_id), "supervisor_pid": os.getpid(), "exitcode": exitcode,
           "restart_transaction_id": str(restart_transaction_id or ""),
           "no_resume_veto_causes": list(NO_RESUME_CAUSES), "created_at": utc_now_iso(),
           **checkpoints.task_facts(supervision)}
    _write(drive_root, row)
    return holders.read_view(row)


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
        if row.get("status") in {"reserved", "pre_adopted", "deferred"} and (same_generation_crash or planned_restart):
            task_ids.add(str(row.get("task_id") or ""))
    return {task_id for task_id in task_ids if task_id}


def recoverable_handles(drive_root: Any) -> set[tuple[str, str, str]]:
    """Orphan protection is exact custody, never every run sharing a reserved task id."""
    result = set()
    for task_id in recoverable_task_ids(drive_root):
        row = _read(drive_root, task_id)
        for holder in row.get("holders", []):
            if holder.get("status") not in {"reserved", "pre_adopted", "deferred"}:
                continue
            run, pending, error = _saved_handoff_holder(drive_root, holders.scoped(row, holder))
            if run is not None and not error:
                result.add((task_id, run.run_id, run.invocation_id))
            elif pending and not error:
                result.add((task_id, "", str(pending["invocation_id"])))
            elif error == "custody_unreadable":
                # Retain only the saved identity while the authority is unknown.
                result.add((task_id, str(holder.get("run_id") or ""), str(holder.get("pending_invocation_id") or "")))
    return result


def continuation_handles(drive_root: Any, task_ids: set[str]) -> set[tuple[str, str, str]]:
    """Snapshot the custody of admitted saved continuations, without lending a task-wide exemption."""
    from ouroboros.delegate_custody_current import task_snapshot

    result = set()
    for task_id in task_ids:
        view = task_snapshot(drive_root, task_id)
        result.update((task_id, run.run_id, run.invocation_id) for run in view["runs"].values() if not run.review_owned)
        result.update((task_id, "", str(item["invocation_id"])) for item in view["pending"]
                      if not custody.review_owned_source(item.get("source")))
    return result


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
        old_attempt = int(meta.get("attempt") or task.get("_attempt") or 1)
        checkpoint = _handoff_wait_source(drive_root, task, old_attempt)
        if not checkpoint or not (checkpoint.get("deferred_holders") or any(
                (state.get("status") in {"sleeping", "wake_pending"} or state.get("pending_wake")) for state in checkpoint.get("runs", {}).values())):
            continue
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
        if not row or not any(h.get("status") == "reserved" for h in row["holders"]):
            continue
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
    """Reserve exact planned-restart custody before startup orphan cleanup."""

    _ack_direct_exec_successor(drive_root)
    adopted: set[str] = set()
    pending_tasks_by_id = {
        str(task.get("id") or ""): task
        for task in list(pending_tasks or []) if str(task.get("id") or "")
    }
    root = pathlib.Path(drive_root) / "state" / "delegate_recovery"
    from ouroboros.work_stop import current_stop
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
        if pending_owner_restart(drive_root) or current_stop(task):
            mismatch = "startup_work_stopped"
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
        elif not spawned_successor_is_current(transaction):
            mismatch = "restart_successor_generation_mismatch"
        elif task is None:
            mismatch = "startup_restored_task_missing"
        elif (old_pid != os.getpid()
              # The inherited handle observed this exact generation's exit; its number may be reused.
              and transaction.get("ack_source") != WINDOWS_PARENT_ACK
              and _pid_alive(old_pid)):
            mismatch = "previous_supervisor_generation_not_dead"
        elif int(row.get("new_attempt") or 0) != int(task.get("_attempt") or 1):
            mismatch = "startup_attempt_mismatch"
        if mismatch:
            veto_handoff(drive_root, task_id, mismatch)
            continue
        from ouroboros.owner_wait import governing_continuation

        saved = bool(governing_continuation(task))
        any_kept = False
        for holder in row["holders"]:
            if holder.get("status") == "refused":
                continue
            scoped = holders.scoped(row, holder)
            error = _successor_binding_mismatch(scoped, task)
            run, _, custody_error = _saved_handoff_holder(drive_root, scoped)
            error = error or custody_error
            if not error and run is not None and not run.settled and not saved:
                error = "run_unobserved" if _run_probe_error(run.run_id) else ""
            if error:
                holder.update(status="deferred" if error in {"custody_unreadable", "run_unobserved"} else "refused",
                              deferred_reason=error)
                any_kept = any_kept or holder["status"] == "deferred"
                continue
            holder.update(status="pre_adopted", pre_adopted_at=utc_now_iso())
            if run is not None:
                holder["run_id"] = run.run_id
            any_kept = True
            custody.emit(drive_root, "delegate_recovery_pre_adopted", {
                "task_id": task_id, "run_id": holder.get("run_id", ""),
                "pending_invocation_id": holder.get("pending_invocation_id", ""),
                "cause": CAUSE_PLANNED_SELF_RESTART, "phase": "before_startup_orphan_sweep"})
        row["status"] = "pre_adopted" if any_kept else "vetoed"
        try:
            _write(drive_root, row)
        except Exception:
            continue
        if any_kept:
            adopted.add(task_id)
    return adopted


def planned_handoff_matches(row: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    """Exact successor custody; transaction and physical adoption still gate dispatch."""
    return bool(task.get("id") and row.get("task_id") == task["id"]
                and row.get("cause") == CAUSE_PLANNED_SELF_RESTART
                and row.get("status") in {"reserved", "pre_adopted"}
                and int(row.get("new_attempt") or 0) == int(task.get("_attempt") or 1)
                and not _successor_binding_mismatch(row, task))


def planned_handoff_resume_allowed(drive_root: Any, task: Mapping[str, Any]) -> bool:
    """Release no-dispatch custody only for this exact, pre-adopted continuation.

    Startup owns pre-adoption and its physical-run checks. Admission only reads
    that authority; it must neither adopt a run nor authorize ordinary replay.
    """
    try:
        task_id = str(task.get("id") or "")
        row = _read(drive_root, task_id)
        if row.get("status") != "pre_adopted" or not planned_handoff_matches(row, task):
            return False
        if owner_restart_in_progress(drive_root) or pending_owner_restart(drive_root):
            return False
        transaction_id = str(row.get("restart_transaction_id") or "")
        transaction = _read_restart_transaction(drive_root, transaction_id)
        return bool(transaction_id and transaction.get("transaction_id") == transaction_id
                    and transaction.get("status") == "normal_exit_acknowledged"
                    and int(transaction.get("supervisor_pid") or 0) == int(row.get("supervisor_pid") or 0)
                    and int(transaction.get("exit_code") or 0) == 42
                    and task_id in set(transaction.get("task_ids") or []))
    except (OSError, ValueError, TypeError, RuntimeError):
        return False


def veto_handoff(drive_root: Any, task_id: str, reason: str, *, handle: str = "") -> dict[str, Any]:
    """Refuse only the named holder, or all holders for a task-wide no-resume cause."""
    row = _read(drive_root, task_id)
    if not row:
        return {}
    for holder in row["holders"]:
        if handle and handle not in {holder.get("run_id"), holder.get("pending_invocation_id")}:
            continue
        holder.update(status="refused", veto_reason=str(reason or "recovery_mismatch"))
        # A refusal never replays pending work or cancels a different holder.
        run, _, error = _saved_handoff_holder(drive_root, holders.scoped(row, holder))
        try:
            custody.reconcile_task_runs(drive_root, task_id, gateway_factory=_attach_only(),
                                        run_ids={run.run_id} if not error and run is not None else set())
        except Exception:
            log.debug("Refused holder cleanup deferred", exc_info=True)
    if all(h.get("status") == "refused" for h in row["holders"]):
        row.update(status="vetoed", veto_reason=str(reason))
    _write(drive_root, row)
    return row


def _adoption_problem(drive: Any, row: dict, task: Mapping[str, Any]) -> str:
    from ouroboros.work_stop import current_stop
    from ouroboros.cancel_intents import cancel_pending

    if current_stop(task) or cancel_pending(drive, str(task.get("id") or "")) or pending_owner_restart(drive):
        return "work_stopped"
    if row.get("cause") not in _ALLOWED_CAUSES or int(row.get("new_attempt") or 0) != int(task.get("_attempt") or 1):
        return "successor_binding_mismatch"
    if row.get("cause") == CAUSE_WORKER_CRASH:
        return "" if int(row.get("supervisor_pid") or 0) == os.getpid() else "successor_generation_mismatch"
    tx_id = str(row.get("restart_transaction_id") or "")
    tx = _read_restart_transaction(drive, tx_id)
    if not tx_id:
        return "restart_transaction_missing"
    if tx.get("status") != "normal_exit_acknowledged":
        return "restart_normal_exit_unproven"
    if int(tx.get("supervisor_pid") or 0) != int(row.get("supervisor_pid") or 0):
        return "restart_transaction_pid_mismatch"
    if int(tx.get("exit_code") or 0) != 42:
        return "restart_transaction_exit_mismatch"
    if task.get("id") not in set(tx.get("task_ids") or []):
        return "restart_transaction_task_mismatch"
    return ""


def _adopt_holder(ctx: Any, task: Mapping[str, Any], row: dict, holder: dict, *, saved: bool) -> dict:
    drive = custody.custody_root(ctx)
    scoped = holders.scoped(row, holder)
    identity = {key: holder.get(key, "") for key in ("run_id", "pending_invocation_id")}
    error = _successor_binding_mismatch(scoped, task)
    run, pending, custody_error = _saved_handoff_holder(drive, scoped)
    error = error or custody_error
    if error:
        return {**identity, "status": "deferred" if error == "custody_unreadable" else "refused", "reason": error}
    if pending and not saved:
        # No task row is substituted for this invocation's immutable saved binding.
        if problem := _adoption_problem(drive, row, task):
            return {**identity, "status": "refused", "reason": problem}
        from ouroboros.tools.delegate import exact_start
        from ouroboros.delegate_shared import delegate_payload

        pending_id = str(pending["invocation_id"])
        holder.update(status="adopted", adopted_at=utc_now_iso())
        _write(drive, row)  # Persist consumption before the replay's paid boundary.
        retry = {"retry_of": pending_id}
        if pending.get("work_order_source_request"):
            retry["work_order_source_request"] = pending["work_order_source_request"]
        from ouroboros.delegate_pending import request_body
        prompt = (request_body(drive, pending) or {}).get("prompt")
        if not isinstance(prompt, str) or not prompt:
            return {**identity, "status": "refused", "reason": "pending_request_prompt_missing"}
        parsed = delegate_payload(exact_start(ctx, prompt, retry))
        run, pending, error = _saved_handoff_holder(drive, scoped)
        if error or run is None:
            fate = custody.invocation_record(drive, pending_id) or {}
            return {**({"invocation_id": pending_id} if fate.get("state") == "failed_definite" else identity),
                    "status": "refused" if fate.get("state") == "failed_definite" else "deferred",
                    "reason": "pending_recovery_definitely_refused" if fate.get("state") == "failed_definite" else "pending_recovery_deferred",
                    "detail": parsed}
    if run is not None and not run.settled and not saved and _run_probe_error(run.run_id):
        return {**identity, "run_id": run.run_id, "status": "deferred", "reason": "run_unobserved"}
    run_id = run.run_id if run is not None else ""
    if run_id:
        _restore_wait_checkpoint(drive, {**scoped, "run_id": run_id})
    holder.update(status="adopted", run_id=run_id, adopted_at=utc_now_iso())
    _write(drive, row)
    custody.emit(drive, "delegate_run_adopted" if run_id else "delegate_recovery_adopted", {
        "task_id": str(task["id"]), "run_id": run_id,
        "pending_invocation_id": holder.get("pending_invocation_id", ""),
        "cause": row.get("cause"), "new_attempt": row.get("new_attempt"),
        **{key: holder.get(key, "") for key in holders.IDENTITY_FIELDS}})
    wake = (holder.get("pending_wake") or {}).get("payload") or {}
    return {"status": "adopted", "run_id": run_id, "cause": row.get("cause"),
            **({"liveness": "unobserved"} if saved else {}),
            **({"settled": run.settled, "terminal_state": run.terminal_state} if run is not None else
               {"pending_invocation_id": pending["invocation_id"], "retry_of": pending["invocation_id"]}),
            **({"wake": wake} if wake else {})}


def adopt_handoff(ctx: Any, task: Mapping[str, Any]) -> dict[str, Any]:
    """Consume every holder's receipt independently, before any configured-session return."""
    drive, task_id = custody.custody_root(ctx), str(task.get("id") or "")
    try:
        row = _read(drive, task_id, strict=True)
    except (OSError, ValueError):
        return {"status": "recovery_required", "reason": "handoff_unreadable"}
    if not row or row.get("status") not in {"reserved", "pre_adopted", "adopted", "deferred"}:
        return {"status": "none"}
    if problem := _adoption_problem(drive, row, task):
        veto_handoff(drive, task_id, problem)
        return {"status": "recovery_required", "reason": problem}
    from ouroboros.owner_wait import governing_continuation

    saved = bool(governing_continuation(task))
    receipts = []
    for holder in row["holders"]:
        if holder.get("status") == "refused":
            receipts.append({"status": "refused", "run_id": holder.get("run_id", ""),
                             "pending_invocation_id": holder.get("pending_invocation_id", ""),
                             "reason": holder.get("veto_reason", "holder_refused")})
            continue
        try:
            receipt = _adopt_holder(ctx, task, row, holder, saved=saved)
        except Exception:
            receipt = {"status": "deferred", "run_id": holder.get("run_id", ""),
                       "pending_invocation_id": holder.get("pending_invocation_id", ""), "reason": "adoption_record_unwritable"}
        if len(row["holders"]) == 1 and receipt.get("reason") == "custody_unreadable":
            return {"status": "recovery_required", "reason": "custody_unreadable"}
        holder["status"] = receipt["status"]
        if receipt.get("reason"):
            holder["deferred_reason"] = receipt["reason"]
        receipts.append(receipt)
        if receipt["status"] == "refused":
            try:
                custody.reconcile_task_runs(drive, task_id, gateway_factory=_attach_only(), run_ids=set())
            except Exception:
                pass
    row["status"] = ("adopted" if all(h["status"] == "adopted" for h in row["holders"]) else
                     "vetoed" if all(h["status"] == "refused" for h in row["holders"]) else "deferred")
    try:
        _write(drive, row)
    except Exception:
        return {"status": "recovery_required", "reason": "adoption_record_unwritable", "holders": receipts}
    if len(receipts) == 1:
        receipt = receipts[0]
        if receipt["status"] == "adopted":
            return {**receipt, "status": "settled_recovered" if receipt.get("settled") and not saved else "adopted"}
        return {**receipt, "status": "recovery_required"}
    return {"status": "adopted" if any(r["status"] == "adopted" for r in receipts) else "recovery_required",
            "holders": receipts, "cause": row.get("cause"), "liveness": "unobserved" if saved else "per_holder"}


__all__ = [
    "CAUSE_PLANNED_SELF_RESTART",
    "CAUSE_WORKER_CRASH",
    "NO_RESUME_CAUSES",
    "PLANNED_RESTART_PARENT_ENV",
    "PLANNED_RESTART_TRANSACTION_ENV",
    "arm_active_planned_restart_transaction",
    "acknowledge_observed_restart_exit",
    "adopt_handoff",
    "authority_fingerprint_from_context",
    "authority_fingerprint_from_task",
    "bind_restart_successor",
    "has_planned_restart_handoffs",
    "observe_restart_parent",
    "planned_handoff_resume_allowed",
    "prepare_handoff",
    "prepare_planned_restart_handoffs",
    "prepare_worker_crash_handoff",
    "pre_adopt_planned_handoffs",
    "reconcile_unrecoverable_task",
    "recoverable_task_ids",
    "recoverable_handles",
    "continuation_handles",
    "unsettled_start_ids",
    "veto_handoff",
    "veto_worker_retry_handoff",
]

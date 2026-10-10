"""Host projections of existing tool claims and physical custody, never results.

An invocation that unwound cannot launch again. Its known executor or supervisor
receipt can still be pending. These readers preserve that distinction for Pause,
Continue and cold sleep without a second ledger or business-status classification.
Untracked descendants/remote effects remain the same disclosed gap as exit zero.
"""
from __future__ import annotations

from typing import Any


def invocation_binding() -> dict:
    from ouroboros.owner_pause import _TOOL_OPERATION, _member_coordinates

    active = _TOOL_OPERATION.get()
    if not active:
        return {}
    root, tree, task = _member_coordinates(active[0])
    return {"drive_root": root, "root_task_id": tree, "task_id": task,
            "launch_operation_id": active[2].get("operation_id", "")}


def _update_active_claim(update) -> None:
    from ouroboros.owner_pause import _TOOL_OPERATION
    from ouroboros.task_results import stamp_task_result_schema
    from ouroboros.utils import update_json_locked

    active = _TOOL_OPERATION.get()
    if not active or not active[2].get("claim_path"):
        return
    outcome = active[2]
    def change(current):
        claims = dict(current.get("launch_handoffs") or {})
        claim = claims.get(outcome["operation_id"])
        if claim is None:
            return None  # A standalone/unbound tool has no durable task claim.
        claims[outcome["operation_id"]] = update(dict(claim))
        return stamp_task_result_schema({**current, "launch_handoffs": claims})
    update_json_locked(outcome["claim_path"], change, strict_existing_dict=True)


def retain_unconfirmed_host_operation(reason: str) -> None:
    """A host owner could not publish its independent custody; never tool meta."""
    _update_active_claim(lambda claim: {**claim, "host_unconfirmed": reason})


def record_control_handoff(event: dict) -> None:
    """Bind the existing supervisor receipt before its event can be emitted."""
    descriptor = {key: str(event.get(key) or "") for key in
                  ("type", "task_id", "client_message_id", "routing_token")}
    _update_active_claim(lambda claim: {**claim, "control_receipts":
                                      [*(claim.get("control_receipts") or []), descriptor]})


def forget_unemitted_control(event: dict) -> None:
    token = str(event.get("routing_token") or "")
    _update_active_claim(lambda claim: {**claim, "control_receipts":
        [row for row in claim.get("control_receipts") or [] if row.get("routing_token") != token]})


def claim_still_owns_effect(root: Any, claim: dict) -> bool:
    if claim.get("host_unconfirmed"):
        return True
    from pathlib import Path
    from ouroboros.routing_wait import wait_for_promotion_admission, wait_for_routing_annotation

    for receipt in claim.get("control_receipts") or []:
        token = str(receipt.get("routing_token") or "")
        if not token:
            return True
        if receipt.get("type") == "promote_chat_to_task":
            observed = wait_for_promotion_admission(Path(root), receipt.get("task_id", ""), token,
                client_message_id=receipt.get("client_message_id", ""), timeout_sec=0)
        else:
            observed = wait_for_routing_annotation(Path(root), receipt.get("client_message_id", ""),
                                                   token, timeout_sec=0)
        if observed.get("status") not in {"scheduled", "delivered", "rejected", "needs_manual_target"}:
            return True
    return False


def retained_tool_custody(root: Any, task_id: str, row: dict, *, excluding: str = "") -> list[dict]:
    """One read-only projection of still-live local invocations and known effects."""
    blockers = []
    for operation_id, claim in (row.get("launch_handoffs") or {}).items():
        if operation_id == excluding:
            continue
        if (claim.get("state") not in {"returned", "owner_dead"}
                or claim_still_owns_effect(root, claim)):
            blockers.append({"kind": "tool_handoff", "task_id": task_id, "operation": claim})
    from ouroboros.merge_receipts import effect_is_definite
    for receipt in row.get("merge_receipts") or []:
        if receipt.get("launch_operation_ids") and not effect_is_definite(receipt):
            blockers.append({"kind": "merge_operation", "task_id": task_id,
                             "receipt_id": receipt.get("receipt_id")})
    return blockers


def retire_tool_invocations(root: Any, task_id: str, root_task_id: str, *,
                            pid: int, process_birth: str, task_attempt: int, holder: str = "") -> None:
    """Called only with positive evidence the exact local owner ended; exact claim match.

    Evidence: a retained-Process confirmed death, or ``local_custody_repair``'s
    witness / platform-qualified absence. ``holder`` names the task result the
    claim lives on when it is not the member's own (a member not yet published
    claims on its root's row). Keep unknown-effect history and independent
    custody. Legacy/unattributable claims are not evidence and remain untouched.
    """
    from ouroboros.task_results import (
        require_writable_task_result_schema, stamp_task_result_schema, task_result_path,
    )
    from ouroboros.utils import update_json_locked, utc_now_iso

    if not pid or not process_birth or type(task_attempt) is not int:
        return
    identity = {"pid": pid, "process_birth": process_birth, "task_attempt": task_attempt}
    def retire(current):
        require_writable_task_result_schema(current)
        if current.get("task_id") != (holder or task_id) or not current.get("status"):
            raise ValueError("tool_invocation_authority_unreadable")
        claims = dict(current.get("launch_handoffs") or {})
        retired = dict(current.get("retired_tool_invocations") or {})
        for op, claim in list(claims.items()):
            if (not isinstance(claim, dict) or claim.get("local_owner") != identity or claim.get("task_id") != task_id
                    or claim.get("root_task_id") != root_task_id):
                continue
            fact = {**claim, "state": "owner_dead", "retired_at": utc_now_iso(),
                    "effect_outcome": "unknown", "replay_authorized": False}
            retired[op] = fact
            if claim_still_owns_effect(root, fact):
                claims[op] = fact
            else:
                claims.pop(op)
        if retired == current.get("retired_tool_invocations", {}):
            return None
        return stamp_task_result_schema({**current, "launch_handoffs": claims,
                                         "retired_tool_invocations": retired})
    update_json_locked(task_result_path(root, holder or task_id), retire, strict_existing_dict=True)

def task_process_blockers(drive_root: Any, task_ids: set[str]) -> list[dict]:
    """Strict read of existing canonical executor custody without backend I/O.

    Kill identity (command hash/signalability) is not a completion predicate:
    failed probes, access denial and exec must not hide a still-live process.
    Legacy unattributed records cannot prove a specific task's custody.
    """
    import pathlib
    from ouroboros.platform_layer import pid_provably_gone
    from ouroboros.workspace_executor import _PROCESS_STATE_DIR, _load_process_record, _valid_process_record

    folder = pathlib.Path(drive_root) / "state" / _PROCESS_STATE_DIR
    try:
        paths = list(folder.iterdir())
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise ValueError("workspace_executor_custody_unreadable") from exc
    blockers = []
    for path in paths:
        if path.suffix != ".json":
            continue
        record = _load_process_record(path)
        if record is None:
            raise ValueError("workspace_executor_custody_unreadable")
        task_id = str(record.get("task_id") or "")
        if task_id not in task_ids:
            continue
        if not _valid_process_record(path, record, check_identity=False):
            raise ValueError("workspace_executor_custody_invalid")
        pid = int(record.get("host_pid") or 0)
        if record.get("executor_type") == "local":
            if pid <= 0:
                raise ValueError("workspace_executor_custody_invalid")
            if pid_provably_gone(pid):
                continue
        elif record.get("backend_completed") is True and (pid <= 0 or pid_provably_gone(pid)):
            continue
        blockers.append({"kind": "workspace_executor", "task_id": task_id,
                         "record_id": record["id"], "executor_type": record["executor_type"]})
    return blockers

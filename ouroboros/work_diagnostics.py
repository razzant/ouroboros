"""The source-control binding of task-related diagnostics.

Only the supervisor's silence admission mints this reference. Parent lineage
alone never selects it. This reads existing controls and owns none; a revoked
diagnostic ends only through the single-scope cancellation owner. The tool
profile lives in ``tool_capabilities.TASK_DIAGNOSTIC_TOOL_NAMES``.
"""

from __future__ import annotations


def diagnostic_ref(task: dict) -> dict:
    metadata = task.get("metadata") or {}
    value = metadata.get("work_diagnostic")
    if (isinstance(value, dict) and value.get("kind") == "silence"
            and all(isinstance(value.get(k), str) and value[k] for k in ("task_id", "root_task_id", "episode_id"))
            and type(value.get("task_attempt")) is int and value["task_attempt"] > 0):
        return value
    return {}


def context_ref(ctx) -> dict:
    return diagnostic_ref({"metadata": getattr(ctx, "task_metadata", None) or {}})


def control_target_reason(task: dict, target: str, root, *, current_source=None) -> str | None:
    """A lifecycle choice needs the same current source authority as a new send.

    Only the observed source is a target: a silent child gives no power over its
    root or siblings. The control owner still decides, so Pause of a child source
    is refused honestly (``not_a_root_task``) and a retry root keeps its fence.
    """
    ref = diagnostic_ref(task)
    if ref and target != ref["task_id"]:
        return "diagnostic_control_target"
    return control_reason(task, root, before_handoff=True, current_source=current_source)


def control_actor_reason(root, actor: str, target: str, *, current_source=None) -> str | None:
    """Classify the actor by its host binding; a name or event grants no authority."""
    from ouroboros.store import StoreBusy, StoreError
    from ouroboros.task_results import load_task_result

    try:
        task = load_task_result(root, actor, strict=True) or {}
        if "work_diagnostic" not in (task.get("metadata") or {}):
            return None
        return control_target_reason({**task, "id": actor}, target, root, current_source=current_source)
    except StoreBusy:
        raise
    except (OSError, ValueError, StoreError):
        return "owner_launch_authority_unavailable:diagnostic_source_unreadable"


def withdraw_revoked(task: dict, root) -> None:
    """A diagnostic whose source authority ended stops through the existing
    single-scope cancellation owner, keyed to the diagnostic alone: its custody
    settles it ``cancelled`` (never a worker exception), and the source's own
    controls and owner actions stay untouched. A pending Panic owns every stop."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.installation_stop import installation_panic

    if diagnostic_ref(task) and not installation_panic(root):
        request_cancel(root, str(task.get("id") or task.get("task_id") or ""),
                       reason="Source authority ended before this diagnostic's next model call",
                       source="silence_resolved", requested_by="supervisor", scope="single")


def _current_source_reason(ref, fact, meta, *, terminal=False) -> str | None:
    """The supervisor's current attempt/progress can lead its durable projection."""
    from ouroboros.work_stop import current_stop

    if not meta or terminal and not meta.get("started_at"):
        return "cancelled"
    task = meta.get("task") or {}
    stop = current_stop(meta) or current_stop(task)
    if stop and stop.get("cause") != "wait":
        return "owner_pause" if stop.get("cause") in {"pause", "awaiting_selection"} else "cancelled"
    attempt = meta.get("attempt") or task.get("_attempt")
    if (attempt != ref["task_attempt"] or str(task.get("root_task_id") or task.get("id")) != ref["root_task_id"]
            or float(meta.get("last_progress_at") or 0) > fact["last_progress_at"]):
        return "cancelled"
    current = meta.get("silence_attention")
    if current is not None and (current.get("episode_id") != ref["episode_id"]
                               or current.get("state") != "pending_attention"):
        return "cancelled"
    return None


def control_reason(task: dict, root, *, before_handoff: bool, current_source=None) -> str | None:
    """Recheck accepted source identity and its controls at every physical send.

    Pause still permits already-sent work to settle. Terminal answer bytes alone
    do not revoke the observation: the source worker may still own processes.
    """
    ref = diagnostic_ref(task)
    if not ref:
        if "work_diagnostic" in (task.get("metadata") or {}):
            return "owner_launch_authority_unavailable:diagnostic_binding_invalid"
        return None
    if ((task.get("id") or task.get("task_id")) != "silence-" + str(ref["episode_id"])
            or task.get("parent_task_id") != ref["task_id"]
            or task.get("root_task_id") != ref["root_task_id"]
            or task.get("delegation_role") != "subagent"
            or (task.get("task_constraint") or {}).get("mode") != "local_readonly_subagent"):
        return "owner_launch_authority_unavailable:diagnostic_binding_mismatch"
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.deadline_utils import seconds_until
    from ouroboros.installation_stop import installation_panic
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.store import StoreBusy, StoreError
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import FINAL_STATUSES, _load_queue_snapshot, queue_snapshot_observation
    from ouroboros.work_stop import current_stop

    try:
        if installation_panic(root, strict=True):
            return "cancelled"
        for task_id in dict.fromkeys((ref["task_id"], ref["root_task_id"])):
            if cancel_pending(root, task_id, strict=True):
                return "cancelled"
            row = load_task_result(root, task_id, strict=True)
            if not row:
                return "owner_launch_authority_unavailable:diagnostic_source_missing"
            if row.get("followup_stop") or row.get("cancel_origin"):
                return "cancelled"
            if row.get("reason_code") in {"deadline", "absolute_ceiling"}:
                return row["reason_code"]
            stop = current_stop(row)
            if stop and stop.get("cause") != "wait":
                if stop.get("cause") in {"panic", "app_stop"}:
                    return "cancelled"
                if before_handoff:
                    return ("owner_pause" if stop.get("cause") in {"pause", "awaiting_selection"}
                            else "owner_launch_authority_unavailable:diagnostic_source_held")
            for source in (row, row.get("metadata") or {}, row.get("task_contract") or {}):
                if source.get("deadline_at") and seconds_until(source["deadline_at"]) == 0:
                    return "deadline"
            if task_id == ref["task_id"]:
                fact = row.get("silence_attention") or {}
                if before_handoff and (str(row.get("root_task_id") or task_id) != ref["root_task_id"]
                        or row.get("task_attempt") != ref["task_attempt"]
                        or fact.get("task_id") != task_id
                        or fact.get("episode_id") != ref["episode_id"]
                        or fact.get("task_attempt") != ref["task_attempt"]
                        or fact.get("state") != "pending_attention"):
                    # The diagnostic loses its reason to start; already-sent work still settles.
                    return "cancelled"
                if before_handoff and (not isinstance(fact.get("last_progress_at"), (int, float))):
                    return "owner_launch_authority_unavailable:diagnostic_observation_missing"
                if before_handoff and current_source is None:
                    # The queue can observe progress before the recovery lane publishes
                    # it to the result, including while that result is still running.
                    # Answer bytes alone also do not prove process exit.
                    snapshot = _load_queue_snapshot(root)
                    if not queue_snapshot_observation(snapshot)["fresh"] or not isinstance(snapshot.get("running"), list):
                        return "owner_launch_authority_unavailable:diagnostic_source_custody_unknown"
                    meta = next((m for m in snapshot["running"] if m.get("id") == task_id), None)
                    if reason := _current_source_reason(ref, fact, meta, terminal=row.get("status") in FINAL_STATUSES):
                        return reason
                source_fact = fact
                source_terminal = row.get("status") in FINAL_STATUSES
        if before_handoff and fence_closed(read_fence(root, ref["root_task_id"])):
            return "owner_pause"
        if before_handoff and current_source is not None:
            # Read memory last, immediately before the control owner takes the action.
            if reason := _current_source_reason(ref, source_fact, current_source(ref["task_id"]), terminal=source_terminal):
                return reason
    except StoreBusy:
        raise  # The admission/model-wait owner retries the same unsent operation.
    except (OSError, ValueError, StoreError):
        return "owner_launch_authority_unavailable:diagnostic_source_unreadable"
    return None

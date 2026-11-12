"""The owner's work-local primary: one durable decision and a text-free wake.

The task document is the authority, including after a worker's acknowledgement
and before its next checkpoint. The queue mirrors that document under Q. Mail
only asks the loop to reread it, so delayed deliveries cannot undo a later act.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

from ouroboros import route_choice


class ModelActRefused(ValueError):
    def __init__(self, code: str, status: int = 409):
        super().__init__(code.replace("_", " "))
        self.code, self.status = code, status


def eligibility(task: Mapping) -> str:
    from ouroboros.task_carrier import execution_kind
    from ouroboros.task_results import resolve_task_lineage

    if execution_kind(dict(task)) == "presence":
        return "presence_model_unsupported"
    lineage = resolve_task_lineage(str(task.get("id") or task.get("task_id") or ""), **{
        key: task.get(key) for key in ("metadata", "root_task_id", "parent_task_id", "delegation_role",
                                      "original_task_id", "timeout_retry_from")})
    return "" if lineage["is_root_task"] else "not_a_root_task"


def carrier_route(carrier: Mapping) -> tuple[str, bool]:
    if carrier.get("source") == "main":
        return route_choice.following_route("main")
    return str(carrier["model"]), bool(carrier.get("use_local"))


def primary_role(row: Mapping) -> str:
    metadata = row.get("metadata") or {}
    carrier = route_choice.lead_snapshot(metadata.get("lead"))
    if carrier:
        return "main" if carrier["source"] == "main" else "lead"
    role = str(metadata.get("model_role") or "main")
    return role if role in route_choice.MODEL_ROLE_SETTINGS and role != "fallback" else "main"


def recorded_primary(row: Mapping) -> dict:
    """An owner carrier, or the role a carrier-less root already follows."""
    carrier = route_choice.lead_snapshot((row.get("metadata") or {}).get("lead"))
    role = primary_role(row)
    if carrier and carrier["source"] != "main":
        model, local = carrier_route(carrier)
    else:
        model, local = route_choice.following_route(role)
    return {"model": model, "use_local": local, "role": role}


def admit(task_id: str, body: dict, *, source: str = "owner") -> tuple[dict, dict, bool]:
    """Write under Q, returning the durable receipt and the current queue task.

    Receipts live with the task. Even an older request repeated after a newer
    decision returns its original answer without rewriting either carrier.
    """
    if source not in {"owner", "owner_words"}:
        raise ValueError("A work model act requires an owner source")
    from ouroboros.lead_fact import project_lead
    from ouroboros.task_results import load_task_result
    from ouroboros.task_store import patch_document
    from supervisor import queue as q
    from supervisor.queue_transitions import settled_status_of

    with q._queue_lock:
        meta = q.RUNNING.get(task_id)
        task = meta.get("task") if isinstance(meta, dict) else None
        running = isinstance(task, dict)
        if not running:
            task = next((row for row in q.PENDING if row.get("id") == task_id), None)
        if not isinstance(task, dict):
            raise ModelActRefused("task_not_live", 404)
        if refusal := eligibility(task):
            raise ModelActRefused(refusal)
        root = task.get("budget_drive_root") or q.DRIVE_ROOT
        existing = load_task_result(root, task_id, strict=True)
        if not existing:
            raise ModelActRefused("model_control_unavailable")
        if settled_status_of(existing):
            raise ModelActRefused("task_not_live", 404)
        from ouroboros.cancel_intents import cancel_pending

        if cancel_pending(root, task_id, strict=True):
            raise ModelActRefused("cancel_pending")
        fact = (meta.get("lead") or {}) if running else {}
        action, request_id = body["action"], body["request_id"]
        duplicate = False

        def mutate(row):
            nonlocal duplicate
            if not row or settled_status_of(row):
                raise ModelActRefused("task_not_live", 404)
            receipts = (row.get("owner_model") or {}).get("receipts") or {}
            if request_id in receipts:
                duplicate = True
                return None
            carrier = route_choice.lead_snapshot((row.get("metadata") or {}).get("lead"))
            if action in {"return", "wait", "keep"} and (not running or fact.get("basis") != "fallback"):
                raise ModelActRefused("model_action_state")
            if action == "follow":
                carrier = {"source": "main"}
            elif action in {"set", "keep"}:
                chosen = body if action == "set" else fact
                carrier = {"source": source, "model": chosen["model"], "use_local": chosen.get("use_local", False)}
                if source == "owner_words":
                    carrier["cited"] = body["cited"]
            metadata = {**(row.get("metadata") or {}), **({"lead": carrier} if carrier else {})}
            if action == "follow":
                metadata["model_role"] = "main"
            projected = project_lead(fact, {"metadata": metadata, "owner_model": {"request_id": request_id}}, queued=not running)
            if not projected:
                projected = project_lead(task={"metadata": metadata}, queued=True)
            receipt = {"ok": True, "task_id": task_id, "applied": "next_step" if running else "at_start",
                       "lead": projected}
            return {**row, "metadata": metadata, "owner_model": {
                "request_id": request_id, "action": action, "recent_recorded": action not in {"set", "keep"},
                "receipts": {**receipts, request_id: receipt}}}

        stored = patch_document(root, task_id, mutate)
        task["metadata"] = {**(task.get("metadata") or {}), **copy.deepcopy(stored["metadata"])}
        task["owner_model"] = copy.deepcopy(stored["owner_model"])
        result = copy.deepcopy(stored["owner_model"]["receipts"][request_id])
        current = copy.deepcopy(task)
    if not stored["owner_model"]["recent_recorded"]:
        from ouroboros.gateway.ui_preferences import record_model_recent

        chosen = stored["metadata"]["lead"]
        try:
            record_model_recent(q.DRIVE_ROOT, chosen["model"], chosen["use_local"])
        except Exception as exc:
            raise ModelActRefused("model_recents_pending", 503) from exc
        # A newer decision may commit while this suggestion is written. Only the
        # decision whose suggestion completed is acknowledged; a retry repairs the rest.
        acknowledged = patch_document(root, task_id, lambda row: {
            **row, "owner_model": {**row["owner_model"], "recent_recorded": True}}
            if (row.get("owner_model") or {}).get("request_id") == stored["owner_model"]["request_id"] else None)
        with q._queue_lock:  # mirror only; both disk writes above are outside Q
            if (task["owner_model"]["request_id"] == acknowledged["owner_model"]["request_id"]
                    == stored["owner_model"]["request_id"]):
                task["owner_model"]["recent_recorded"] = current["owner_model"]["recent_recorded"] = True
    # The task document committed before the queue snapshot. A restart can always
    # rebase a stale saved queue/checkpoint from that authority.
    if q.persist_queue_snapshot(reason="owner_model") is False:
        raise ModelActRefused("model_queue_publish_failed", 503)
    return result, current, duplicate


def request(task_id: str, body: Mapping, *, source: str = "owner") -> dict:
    """Both the card and a cited tool ask the supervisor to save, then wake the same decision."""
    from ouroboros.owner_mailbox import KIND_OWNER_MODEL, write_owner_message
    from supervisor.queue import _task_drive_for_task

    result, task, _duplicate = admit(task_id, body, source=source)
    control = task["owner_model"]
    if result["applied"] == "next_step" and not write_owner_message(
            _task_drive_for_task(task, task_id), "owner_model", task_id,
            msg_id=f"owner_model:{control['request_id']}", kind=KIND_OWNER_MODEL):
        raise ModelActRefused("model_wake_failed", 503)
    return result


def preserve_carrier(existing: Mapping, fields: dict) -> dict:
    """A worker's older task copy cannot replace a host-owned primary decision."""
    carrier = (existing.get("metadata") or {}).get("lead")
    if carrier is not None and isinstance(fields.get("metadata"), dict):
        fields = {**fields, "metadata": {**fields["metadata"], "lead": copy.deepcopy(carrier)}}
    if existing.get("owner_model") and isinstance(fields.get("metadata"), dict):
        fields = {**fields, "metadata": dict(fields["metadata"])}
        fields["metadata"].pop("model_role", None)
        if role := (existing.get("metadata") or {}).get("model_role"):
            fields["metadata"]["model_role"] = role
    if existing.get("owner_model"):
        fields = {**fields, "owner_model": copy.deepcopy(existing["owner_model"])}
    return fields


def primary_action(ctx: Any, action: str) -> str:
    """The same symbolic return/wait for the mind's tool and the owner's card."""
    from ouroboros.lead_fact import requested_return

    requested_return(ctx)
    route = ctx.primary_route
    waiter = getattr(ctx, "model_wait_context", None)
    chosen = ((getattr(waiter, "overrides", None) or {}).get(route["role"]) or {})
    ctx.active_model_override = str(chosen.get("model") or route["model"])
    ctx.active_use_local_override = bool(chosen.get("use_local", route["use_local"]))
    ctx.active_role_override = route["role"]
    ctx.route_wait_on_primary = action == "wait"
    ctx.route_choice = route_choice.explicit("wait_card") if chosen else route_choice.opening_choice(ctx)
    return (f"primary route {ctx.active_model_override}{' (local)' if ctx.active_use_local_override else ''}"
            f" (role {route['role']}; if it refuses: "
            f"{'wait for it; once its content filter stops a request, nothing is there to wait for, so this run ends as interrupted, without configured routes' if action == 'wait' else 'configured routes again'})")


def read_current(ctx: Any) -> dict:
    from ouroboros.task_results import load_task_result

    root = getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None)
    task_id = str(getattr(ctx, "task_id", "") or "")
    return (load_task_result(root, task_id, strict=True) or {}) if root and task_id else {}


def waiting_wake(waiter: Any) -> str | None:
    """A refused call's resource wait can yield to the loop; a sent call never polls this."""
    from pathlib import Path
    from ouroboros.owner_mailbox import KIND_OWNER_MODEL, OwnerMailboxPeek, drain_owner_entries
    from ouroboros.store import StoreBusy, acquisition_budget

    ctx = waiter.tool_context
    if ctx is None:
        return None
    seen = set(getattr(ctx, "_loop_mailbox_seen_ids", ()) or ())
    root = Path(waiter.drive_root)
    applied = getattr(ctx, "_owner_model_applied_id", None)
    try:
        stamp = (OwnerMailboxPeek._fingerprint(root, waiter.task_id, waiter.attempt, seen), applied)
        if stamp == getattr(waiter, "_owner_model_mail_stamp", None):
            return None
        status = {}
        entries = drain_owner_entries(root, waiter.task_id, seen, waiter.attempt,
                                      kinds={KIND_OWNER_MODEL}, _read_status=status)
        if not status.get("complete"):
            return None
        with acquisition_budget(0):
            control = read_current(ctx).get("owner_model") or {} if entries else {}
    except (OSError, StoreBusy):
        return None  # leave the read owed; contention is not a task control
    if control.get("request_id") and control["request_id"] != applied:
        return "owner_model"
    waiter._owner_model_mail_stamp = stamp
    return None


def apply_current(ctx: Any) -> bool:
    """Read the latest durable act, never the value or order of a mail entry.

    Prepares the ordinary round handover. Its successful rebind marks the request
    applied, so a checkpoint cannot acknowledge a handover that never happened.
    """
    row = read_current(ctx)
    control = row.get("owner_model") or {}
    request_id = control.get("request_id")
    if not request_id:
        return False
    carrier = route_choice.lead_snapshot((row.get("metadata") or {}).get("lead"))
    primary = recorded_primary(row)
    model, local, role = primary["model"], primary["use_local"], primary["role"]
    following = role != "lead"
    ctx.task_metadata = {**(getattr(ctx, "task_metadata", None) or {})}
    ctx.task_metadata.pop("model_role", None)
    if slot := (row.get("metadata") or {}).get("model_role"):
        ctx.task_metadata["model_role"] = slot
    if carrier:
        ctx.task_metadata["lead"] = carrier
    else:
        ctx.task_metadata.pop("lead", None)
    ctx.task_model_override = None if following else model
    ctx.task_use_local_override = None if following else local
    ctx.task_route_source = carrier["source"] if carrier and not following else None
    ctx.primary_route = primary
    ctx._route_binding = None
    if request_id == getattr(ctx, "_owner_model_applied_id", None):
        ctx._owner_model_request_id, ctx._owner_model_primary_changed = None, False
        return False
    action = control["action"]
    changed_primary = action in {"set", "follow", "keep"}
    if changed_primary:
        waiter = getattr(ctx, "model_wait_context", None)
        if waiter is not None:
            # An older wait-card choice belongs to the primary it replaced.
            waiter.overrides.pop("lead", None)
            waiter.overrides.pop(role, None)
        ctx.route_choice = route_choice.opening_choice(ctx)
        ctx.active_model_override, ctx.active_use_local_override, ctx.active_role_override = model, local, role
        ctx.route_wait_on_primary = False
        ctx._lead_return_from = ctx._lead_episode = ctx._lead_role = ctx._primary_view = None
    else:
        primary_action(ctx, action)
    ctx._owner_model_primary_changed = changed_primary
    ctx._owner_model_request_id = request_id
    words = {"follow": "set this work to follow Main", "return": "requested a return to this work's primary",
             "wait": "requested waiting for this work's primary"}.get(action, "set this work's primary")
    ctx._route_facts_pending = f"[ROUTE FACTS] My human {words}: {model}{' (local)' if local else ''}."
    return True


def restore_current(ctx: Any, messages: list | None = None) -> None:
    """A saved route never outranks a primary written since that checkpoint."""
    if apply_current(ctx):
        from ouroboros.llm import LLMClient

        if messages is not None:
            messages[:] = LLMClient.sanitize_reasoning_on_model_switch(
                messages, ctx.active_model, ctx.active_model_override)
        ctx.active_model, ctx.active_use_local = ctx.active_model_override, ctx.active_use_local_override

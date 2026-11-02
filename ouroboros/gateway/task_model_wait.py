"""Owner decisions for an existing live model wait, over the ordinary mailbox."""

from __future__ import annotations

import asyncio
import copy
import json
import pathlib
from contextlib import nullcontext
from typing import Any

from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_error
from ouroboros.model_wait import mutate_wait
from ouroboros.task_results import validate_task_id


def history_wait_row(entry: dict) -> dict | None:
    """A model-wait event is a typed card reference, never prose or completion."""
    if entry.get("type") != "task_model_wait" or not entry.get("wait_id") or not entry.get("task_id"):
        return None
    row = {key: value for key, value in entry.items()
           if key not in {"type", "ts", "task_id", "quota_clock", "is_progress"} and not key.startswith("_")}
    return {"text": "", "role": "system", "ts": str(entry.get("ts") or ""), "is_progress": False,
            "system_type": "task_model_wait", "task_id": str(entry["task_id"]),
            "model_waits": {str(entry["wait_id"]): row}}


def history_wait_overlay(messages: list[dict], owner_limit: int) -> tuple[list[dict], list[dict], bool]:
    """Fold latest revision per wait, bounded by the existing hydration window."""
    other, by_owner = [], {}
    for message in messages:
        if message.get("system_type") != "task_model_wait":
            other.append(message)
            continue
        task_id = message["task_id"]
        current = by_owner.setdefault(task_id, {**message, "model_waits": {}})
        current["ts"] = max(current["ts"], message["ts"])
        for wait_id, row in message["model_waits"].items():
            previous = current["model_waits"].get(wait_id) or {}
            if int(row.get("revision") or 0) > int(previous.get("revision") or 0):
                current["model_waits"][wait_id] = copy.deepcopy(row)
    ordered = sorted(by_owner.values(), key=lambda row: row["ts"])
    return other, ordered[-owner_limit:] if owner_limit > 0 else [], len(ordered) > owner_limit


class WaitDecisionRefused(ValueError):
    def __init__(self, reason: str, row: dict | None = None, status: int = 409):
        super().__init__(reason)
        self.reason = reason
        self.row = row or {}
        self.status = status


def _action(body: dict) -> tuple[str, str, dict]:
    """Validate this decision family before any settings or mailbox side effect."""
    request_id, decision_id = body.get("request_id"), body.get("decision_id")
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128:
        raise WaitDecisionRefused("request_id_required", status=400)
    if not isinstance(decision_id, str):
        raise WaitDecisionRefused("malformed_decision_id", status=400)
    parts = decision_id.split(":", 2)
    if len(parts) != 3 or parts[0] != "model_wait" or not parts[2]:
        raise WaitDecisionRefused("malformed_decision_id", status=400)
    try:
        task_id, wait_id = validate_task_id(parts[1]), validate_task_id(parts[2])
    except ValueError:
        raise WaitDecisionRefused("malformed_decision_id", status=400) from None
    revision = body.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise WaitDecisionRefused("revision_required", status=400)
    action = body.get("action")
    common = {"request_id", "decision_id", "revision", "action"}
    action_fields = {"auto_continue": {"auto_continue"},
                     "switch": {"model", "credential_profile_id", "use_local", "persist_role"}, "retry": set()}
    if not isinstance(action, str) or action not in action_fields or set(body) - common - action_fields[action]:
        raise WaitDecisionRefused("invalid_model_wait_action", status=400)
    value = {key: body[key] for key in common if key != "decision_id"}
    if action == "auto_continue":
        if not isinstance(body.get("auto_continue"), bool):
            raise WaitDecisionRefused("auto_continue_must_be_boolean", status=400)
        value["auto_continue"] = body["auto_continue"]
    elif action == "switch":
        if (not isinstance(body.get("model"), str) or not body["model"].strip()
                or not isinstance(body.get("credential_profile_id"), str)
                or not isinstance(body.get("use_local"), bool)
                or not isinstance(body.get("persist_role", False), bool)):
            raise WaitDecisionRefused("invalid_model_selection", status=400)
        value.update(model=body["model"].strip(), credential_profile_id=body["credential_profile_id"].strip(),
                     use_local=body["use_local"], persist_role=body.get("persist_role", False))
    return task_id, wait_id, value


def _live_task(task_id: str) -> dict:
    from supervisor import queue
    from supervisor.workers import direct_chat_turn
    from ouroboros.cancel_intents import cancel_pending

    with queue._queue_lock:
        meta = queue.RUNNING.get(task_id)
        task = dict(meta.get("task") or {}) if isinstance(meta, dict) else direct_chat_turn(task_id)
        if not task:
            raise WaitDecisionRefused("task_not_live")
        if isinstance(meta, dict):
            task["_attempt"] = int(meta.get("attempt") or task.get("_attempt") or 1)
        if cancel_pending(queue.DRIVE_ROOT, task_id):
            raise WaitDecisionRefused("cancel_pending")
        return task


def _decide(root: Any, body: dict) -> JSONResponse:
    from ouroboros.gateway.owner_settings import CommitBoundary
    from ouroboros.owner_mailbox import KIND_MODEL_WAIT, write_owner_message
    from supervisor.queue import _task_drive_for_task

    boundary = CommitBoundary()
    row: dict = {}
    action: dict = {}
    duplicate = False
    claimed = False
    transformation_completed = False
    mailbox_attempted = False
    owner = None
    operation_wait = False
    attempt = 0

    def phase_owner():
        from ouroboros.post_task_checkpoint import post_task_model_wait
        from ouroboros.review_operation import review_operation_controller

        # A paid review operation owns its reviewers' waits, during and after
        # the author's turn: only its exact live controller may consume them.
        if operation_wait:
            return review_operation_controller(root, task_id, wait_id)
        return post_task_model_wait(root, task_id)

    def live_task():
        if owner is None:
            if operation_wait:
                raise WaitDecisionRefused("task_not_live")  # never the author's mailbox
            return _live_task(task_id)
        current = phase_owner()
        if owner.closed or current is None or (current is not owner and (
                not getattr(owner, "owner_id", "") or getattr(current, "owner_id", "") != owner.owner_id)):
            raise WaitDecisionRefused("task_not_live")
        return owner.task

    def mutate(wait_id, transform):
        return owner.mutate_row(wait_id, transform) if owner is not None else mutate_wait(root, task_id, wait_id, transform)

    def saved():
        return boundary.committed or bool(action and row.get("saved_request_id") == action["request_id"])

    def discard_definitely_unapplied():
        nonlocal row
        if not claimed or saved() or transformation_completed or mailbox_attempted:
            return

        def discard(previous):
            if previous and previous.get("pending_action") == action and previous.get("applied_request_id") != action["request_id"]:
                value = dict(previous)
                value.pop("pending_action", None)
                return value
            return None

        row = mutate(wait_id, discard)

    def receipt(current):
        """This exact claim applied by its own controller: request, wait owner and attempt."""
        return bool(action and current.get("applied_request_id") == action["request_id"]
                    and current.get("task_attempt") == attempt
                    and str(current.get("model_wait_owner_id") or "") == str(getattr(owner, "owner_id", "") or ""))

    def claim_as_it_is():
        """After any discard, the claim's current row answers whether its controller applied it."""
        nonlocal row
        discard_definitely_unapplied()
        if claimed:
            try:
                row = mutate(wait_id, lambda previous: None)
            except Exception:
                pass  # the last observed row stays the answer
        return claimed and receipt(row)

    def stop_clear():
        """Whether the owner's separately confirmed Settings save may still land. A reviewer's wait follows
        its operation's own stop state: Wrap up lets the reviewer finish and keeps this save (owner choice,
        2026-10-14); Stop now, Panic, a cancelled result or an unreadable state withhold it. Any other wait
        keeps it only while no Stop intent at all (Wrap up included) is recorded."""
        from ouroboros.cancel_intents import cancel_pending, resolve_owner_stop_intent
        from ouroboros.review_operation import operation_stop_state

        try:
            stopped = operation_stop_state(pathlib.Path(root), task_id)
            if operation_wait:
                return not stopped
            return not (stopped or cancel_pending(root, task_id, strict=True)
                        or resolve_owner_stop_intent(root, task_id)[1])
        except Exception:
            return False
    try:
        task_id, wait_id, action = _action(body)
        from ouroboros.review_operation import names_review_operation

        operation_wait = names_review_operation(root, task_id, wait_id)
        owner = phase_owner()
        task = live_task()
        attempt = int(task.get("_attempt") or 1)

        def claim(previous):
            nonlocal duplicate
            if not previous:
                raise WaitDecisionRefused("model_wait_not_found", status=404)
            if previous.get("task_attempt") != attempt:
                raise WaitDecisionRefused("stale_model_wait", previous)
            if previous.get("applied_request_id") == action["request_id"]:
                duplicate = True
                return None
            pending = previous.get("pending_action")
            if pending:
                if pending != action:
                    reason = "request_id_conflict" if pending.get("request_id") == action["request_id"] else "model_wait_action_pending"
                    raise WaitDecisionRefused(reason, previous)
                duplicate = True
                return None
            if previous.get("state") != "waiting" or previous.get("revision") != action["revision"]:
                raise WaitDecisionRefused("stale_model_wait", previous)
            if action.get("persist_role"):
                from ouroboros.gateway.owner_settings import _owner_read_settings_raw
                from ouroboros.model_slots import apply_model_role_override

                try:
                    apply_model_role_override(
                        _owner_read_settings_raw(), role=previous["role"], model=action["model"],
                        credential_profile_id=action["credential_profile_id"], use_local=action["use_local"])
                except ValueError as error:
                    raise WaitDecisionRefused("invalid_role_update", previous, 400) from error
            return {**previous, "pending_action": action}

        row = mutate(wait_id, claim)
        claimed = True
        applied = row.get("applied_request_id") == action["request_id"]
        if not applied:
            if action.get("persist_role"):
                from ouroboros.gateway.owner_settings import _owner_update_settings, settings_document_mutation
                from ouroboros.model_slots import apply_model_role_override, model_role_setting_keys

                if row.get("saved_request_id") != action["request_id"]:
                    def save_allowed():
                        nonlocal row
                        try:
                            live_task()
                        except WaitDecisionRefused:
                            if not getattr(owner, "consumes_pending_action", False):
                                raise
                            # A controller consuming the claim itself may apply it while this save
                            # waits for the lock; its row then stops waiting and it reads closed,
                            # so its own receipt stands in for liveness.
                            current = mutate(wait_id, lambda previous: None)
                            if not receipt(current):
                                raise WaitDecisionRefused("task_not_live", current) from None
                            row = current
                        # One stop rule for a live and an applied claim (``stop_clear``).
                        return stop_clear()

                    def transform(settings):
                        nonlocal transformation_completed
                        if not save_allowed():
                            return None  # the decision stands; Settings are not written
                        value = apply_model_role_override(
                            settings, role=row["role"], model=action["model"],
                            credential_profile_id=action["credential_profile_id"], use_local=action["use_local"])
                        transformation_completed = True
                        return value

                    with settings_document_mutation():  # a chosen model is authored, the shown default included
                        _owner_update_settings(transform, boundary=boundary,
                                               authored_keys=model_role_setting_keys(row["role"]), allow_safety_lowering=True)
                    if transformation_completed:
                        row = mutate(wait_id,
                                     lambda previous: {**previous, "saved_request_id": action["request_id"]})
            # Re-check after the optional settings write. A persistent owner
            # choice may have landed even when cancellation now fences the task.
            with owner.lock if owner is not None else nullcontext():
                if getattr(owner, "consumes_pending_action", False):
                    # A review operation consumes the claimed row itself (the author's
                    # mailbox is not its ingress): its controller already applied it, or
                    # it is still pending before that exact live controller.
                    row = mutate(wait_id, lambda previous: None)
                    applied = receipt(row)
                    if not applied and (row.get("pending_action") != action or owner.closed):
                        # A remote controller closes by consuming this very claim
                        # between the two reads: its durable receipt, read after the
                        # closed observation, answers; any other closure refuses.
                        row = mutate(wait_id, lambda previous: None)
                        applied = receipt(row)
                        if not applied:
                            raise WaitDecisionRefused("task_not_live", row)
                else:
                    task = live_task()
                    control = {**action, "wait_id": wait_id, "task_attempt": attempt}
                    mailbox_attempted = True
                    if not write_owner_message(owner.drive_root if owner is not None else _task_drive_for_task(task, task_id), json.dumps(control), task_id,
                                               msg_id=f"model_wait:{wait_id}:{action['request_id']}", kind=KIND_MODEL_WAIT):
                        raise WaitDecisionRefused("mailbox_write_failed", row, 503)
        return JSONResponse({"ok": True, "decision_id": body["decision_id"], "request_id": action["request_id"],
                             "state": row["state"], "wait": row, "duplicate": duplicate, "applied": applied,
                             "saved": saved()},
                            status_code=200 if duplicate else 202)
    except WaitDecisionRefused as error:
        # Applied, saved and the refusal are separate facts; after a claim its row is read as it now is.
        applied = claim_as_it_is()
        current = row if claimed else error.row
        return json_error(error.reason, error.status, ok=False, reason_code=error.reason,
                          state=current.get("state", ""), wait=current, applied=applied, saved=saved())
    except Exception as error:
        applied = claim_as_it_is()
        return json_error(str(error), 503, ok=False, reason_code="model_wait_decision_failed", wait=row,
                          applied=applied, saved=saved())


async def answer_model_wait_decision(root: Any, body: dict) -> tuple[int, dict]:
    """Share the existing wait effect and settings-writer receipts across transports."""
    decide = _decide
    if body.get("persist_role") is True:
        from ouroboros.gateway.settings import _run_settings_writer

        response = await _run_settings_writer(decide, root, body)
    else:
        response = await asyncio.to_thread(decide, root, body)
    return response.status_code, json.loads(response.body)

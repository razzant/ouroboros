"""Where a chat turn becomes a task, and where a project scope is bound.

Owns the routing acknowledgement the decision actor reads, the off-loop
preparation of the promoted source, the durable rejection record, the
rollback of a promoted task that never reached the queue, and the registry
side of project scope and per-cycle project digests.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional
from ouroboros.utils import utc_now_iso
from ouroboros.task_results import STATUS_FAILED, STATUS_SCHEDULED, write_task_result

log = logging.getLogger(__name__)


def _events():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from supervisor import events

    return events


def _emit_routing_receipt(
    ctx: Any,
    evt: Dict[str, Any],
    *,
    action: str,
    target: str = "",
    target_label: str = "",
    status: str,
    reason: str = "",
    detail: str = "",
    options: Optional[list] = None,
    attachment_manifest: Optional[list] = None,
    publish: bool = True,
) -> Dict[str, Any]:
    """Persist and publish one token-bound routing annotation receipt."""
    if target and not str(target_label or "").strip():
        from ouroboros.project_dialogue import routing_target_label

        target_label = routing_target_label(ctx.DRIVE_ROOT, action, target, task=evt, project_id=str(evt.get("project_id") or ""))
    client_message_id = str(evt.get("client_message_id") or "").strip()
    routing_token = str(evt.get("routing_token") or "").strip()
    annotation_status = "not_applicable"
    if client_message_id:
        try:
            from ouroboros.project_dialogue import append_chat_annotation

            annotation_status = (
                "persisted"
                if append_chat_annotation(
                    ctx.DRIVE_ROOT,
                    client_message_id,
                    action=action,
                    target=target,
                    target_label=target_label,
                    status=status,
                    routing_token=routing_token,
                    reason=reason,
                    detail=detail,
                    options=options,
                    attachment_manifest=attachment_manifest,
                )
                else "failed"
            )
        except Exception:
            annotation_status = "failed"
            log.debug("Routing annotation append failed", exc_info=True)

    effective_status = str(status or "needs_manual_target")
    effective_reason = str(reason or "")
    if annotation_status == "failed" and effective_status in {"scheduled", "delivered"}:
        effective_status = "unconfirmed"
        effective_reason = "routing_annotation_persist_failed"

    receipt: Dict[str, Any] = {
        "persisted": annotation_status in {"persisted", "not_applicable"},
        "status": effective_status,
        "reason": effective_reason,
        "detail": str(detail or ""),
        "annotation_status": annotation_status,
        "routing_token": routing_token,
        "target_label": str(target_label or ""),
    }
    if attachment_manifest is not None:
        receipt["attachment_manifest"] = _events()._routing_attachments(attachment_manifest) or []
    if not receipt["persisted"]:
        return receipt
    if publish:
        _publish_routing_ack(
            ctx,
            evt,
            action=action,
            target=target,
            target_label=target_label,
            status=effective_status,
            options=options,
            attachment_manifest=attachment_manifest,
        )
    return receipt


def _publish_routing_ack(
    ctx: Any,
    evt: Dict[str, Any],
    *,
    action: str,
    target: str,
    target_label: str = "",
    status: str,
    options: Optional[list] = None,
    attachment_manifest: Optional[list] = None,
) -> None:
    """Publish a live non-bubble acknowledgement after durable authority exists."""
    try:
        if target and not str(target_label or "").strip():
            from ouroboros.project_dialogue import routing_target_label

            target_label = routing_target_label(ctx.DRIVE_ROOT, action, target, task=evt, project_id=str(evt.get("project_id") or ""))
        client_message_id = str(evt.get("client_message_id") or "").strip()
        try:
            chat_id = int(evt.get("chat_id") or 0)
        except (TypeError, ValueError):
            chat_id = 0
        bridge = getattr(ctx, "bridge", None)
        ack = getattr(bridge, "send_routing_ack", None)
        if callable(ack):
            ack_kwargs = {
                "client_message_id": client_message_id,
                "action": action,
                "target": target,
                "target_label": target_label,
                "status": status,
            }
            if options is not None:
                ack_kwargs["options"] = options
            if attachment_manifest is not None:
                ack_kwargs["attachment_manifest"] = attachment_manifest
            if str(evt.get("routing_token") or ""):
                ack_kwargs["routing_token"] = str(evt.get("routing_token"))
            ack(
                chat_id,
                **ack_kwargs,
            )
    except Exception:
        log.debug("Routing typed ack failed", exc_info=True)


def _handle_project_digest(evt: Dict[str, Any], ctx: Any) -> None:
    """Surface a concise per-project cycle completion digest to consciousness.

    Full project awareness (v6.32.0): the one identity already sees the project's
    chat thread in its unified memory, so this is a crisp "task finished" summary
    (project_id + full objective + outcome statuses), NOT an isolation boundary.
    Per-cycle RAW internal facts stay in the per-project knowledge/journal store
    (scoped tools); the единый agent decides what to do with the digest — backlog,
    identity, or nothing (BIBLE P5).
    """
    pid = str(evt.get("project_id") or "").strip()
    if not pid:
        return
    try:
        from ouroboros.projects_registry import touch_project

        touch_project(ctx.DRIVE_ROOT, pid)
    except Exception:
        log.debug("project_digest touch failed", exc_info=True)
    try:
        # Digest into the штаб's consciousness: carry the objective WHOLE (BIBLE P1
        # — no silent/lossy clip of cognitive text). The one mind is aware of its
        # project work in full; only raw per-cycle facts stay in the project store.
        digest = (
            f"Project '{pid}' task {str(evt.get('task_id') or '')} finished: "
            f"execution={str(evt.get('execution_status') or 'unknown')}, "
            f"objective={str(evt.get('objective_status') or 'not_evaluated')}. "
            f"Goal: {str(evt.get('objective') or '')}"
        )
        consciousness = getattr(ctx, "consciousness", None)
        if consciousness is not None:
            consciousness.inject_observation(digest)
    except Exception:
        log.debug("project_digest consciousness injection failed", exc_info=True)


def _rollback_promoted_pending(
    ctx: Any, task_id: str, admission_token: str, *, reason: str,
) -> bool:
    """Remove an unconfirmed promote before the supervisor can assign it."""
    from supervisor import queue as supervisor_queue

    removed = False
    with supervisor_queue._queue_lock:
        pending = getattr(ctx, "PENDING", supervisor_queue.PENDING)
        survivors = [
            task for task in pending
            if not (
                str(task.get("id") or "") == task_id
                and str(
                    task.get("_admission_owner_token")
                    or task.get("promotion_admission_token")
                    or ""
                ) == admission_token
            )
        ]
        removed = len(survivors) != len(pending)
        if removed:
            pending[:] = survivors
    if removed:
        persist = getattr(ctx, "persist_queue_snapshot", None)
        if callable(persist):
            try:
                persist(reason=reason)
            except Exception:
                log.warning("Failed to persist promote rollback for %s", task_id, exc_info=True)
    return removed


def _persist_promote_rejection(
    ctx: Any,
    evt: Dict[str, Any],
    outcome: Dict[str, Any],
    *,
    status: str = "rejected",
) -> None:
    task_id = str(outcome.get("task_id") or evt.get("task_id") or "")
    reason = str(outcome.get("reason") or "admission_rejected")
    if reason == "task_id_lookup_failed":
        return  # preserve the unreadable exact-id authority byte-for-byte
    write_task_result(
        ctx.DRIVE_ROOT,
        task_id,
        STATUS_FAILED,
        reason_code=reason,
        project_id=str(evt.get("project_id") or ""),
        description=str(evt.get("objective") or ""),
        expected_output=str(evt.get("expected_output") or ""),
        promotion_admission={
            "status": status,
            "routing_token": str(evt.get("routing_token") or ""),
            "reason": reason,
            "detail": str(outcome.get("detail") or ""),
            "worker_pool_disabled_reason": str(
                outcome.get("worker_pool_disabled_reason") or ""
            ),
            "confirmed_at": utc_now_iso(),
        },
        result=(
            f"Promotion was not scheduled: {reason}. "
            f"{str(outcome.get('detail') or '')}"
        ).strip(),
    )


def _prepare_promote_source_off_loop(evt: Dict[str, Any], ctx: Any) -> None:
    """Resolve a potentially 900s clone away from the supervisor drain loop."""
    continuation = dict(evt)
    continuation["_source_prepared"] = True
    try:
        from ouroboros.promotion_source import resolve_promote_source

        folder, note, error, project_id, source_created = resolve_promote_source(
            ctx,
            str(evt.get("source") or ""),
            str(evt.get("project_id") or ""),
        )
        continuation["project_id"] = project_id
        continuation["_source_note"] = note
        continuation["_source_error"] = error
        continuation["_source_created"] = bool(source_created)
        if folder and not str(continuation.get("workspace_root") or "").strip():
            continuation["workspace_root"] = folder
    except Exception as exc:
        continuation["_source_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from supervisor.workers import get_event_q

        get_event_q().put(continuation)
    except Exception as exc:
        log.exception("Failed to publish promote source continuation")
        from supervisor import queue as supervisor_queue

        task_id = str(evt.get("task_id") or "")
        routing_token = str(evt.get("routing_token") or "")
        supervisor_queue.release_task_admission(task_id, routing_token)
        failed = {
            "status": "unconfirmed",
            "reason": "source_continuation_publish_failed",
            "detail": f"{type(exc).__name__}: {exc}",
            "task_id": task_id,
        }
        try:
            _persist_promote_rejection(ctx, evt, failed, status="unconfirmed")
            _emit_routing_receipt(
                ctx,
                evt,
                action=(
                    "route_to_project"
                    if bool(evt.get("routed_from_main"))
                    else "promote_chat_to_task"
                ),
                target=task_id,
                status="unconfirmed",
                reason=failed["reason"],
                detail=failed["detail"],
            )
        except Exception:
            log.exception("Failed to persist promote source continuation failure")


def _handle_promote_chat_to_task(evt: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Spawn a first-class pooled owner task from a conversation-lane promote.

    Unlike ``schedule_subagent`` the child is NOT a subagent: it is a normal
    owner task (live card, canonical drive, project lease participation). The
    conversation lane that emitted the event stays free.
    """
    from supervisor.workers import (
        _broadcast_task_named,
        promote_chat_to_task,
        worker_pool_admission_state,
    )
    receipt_action = (
        "route_to_project" if bool(evt.get("routed_from_main"))
        else "promote_chat_to_task"
    )

    task_id = str(evt.get("task_id") or "")
    routing_token = str(evt.get("routing_token") or "")
    try:
        from supervisor import queue as supervisor_queue

        reservation = supervisor_queue.reserve_task_admission(
            task_id,
            routing_token,
            require_worker_pool=True,
            drive_root=ctx.DRIVE_ROOT,
            worker_pool=getattr(ctx, "WORKERS", None),
        )
        reservation_status = str(reservation.get("status") or "")
        if reservation_status == "already_reserved" and evt.get("_admission_reserved"):
            reservation_status = "reserved"
        if reservation_status != "reserved":
            if reservation_status == "existing_same_token":
                admission = reservation.get("promotion_admission")
                return {
                    "status": str((admission or {}).get("status") or "unconfirmed"),
                    "task_id": task_id,
                    "reason": str((admission or {}).get("reason") or ""),
                }
            if reservation_status == "already_reserved":
                return {"status": "preparing", "task_id": task_id}
            blocked = {
                "status": "needs_manual_target",
                "reason": str(reservation.get("reason") or "admission_reservation_failed"),
                "worker_pool_disabled_reason": str(
                    reservation.get("worker_pool_disabled_reason") or ""
                ),
                "task_id": task_id,
                "reservation_owned": False,
            }
            if blocked["reason"] != "duplicate_task_id":
                _persist_promote_rejection(ctx, evt, blocked)
            _emit_routing_receipt(
                ctx,
                evt,
                action=receipt_action,
                target=task_id,
                status="needs_manual_target",
                reason=blocked["reason"],
            )
            ctx.append_jsonl(
                ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "promote_chat_to_task_rejected",
                    "task_id": task_id,
                    "reason": blocked["reason"],
                    "worker_pool_disabled_reason": blocked[
                        "worker_pool_disabled_reason"
                    ],
                },
            )
            return blocked
        evt = {**evt, "_admission_reserved": True}
        if str(evt.get("source") or "").strip() and not evt.get("_source_prepared"):
            threading.Thread(
                target=_prepare_promote_source_off_loop,
                args=(dict(evt), ctx),
                daemon=True,
                name=f"promote-source-{task_id[:12]}",
            ).start()
            return {"status": "preparing", "task_id": task_id}
        source_error = str(evt.get("_source_error") or "")
        if source_error:
            outcome = {
                "status": "needs_manual_target",
                "reason": "project_source_error",
                "detail": source_error,
                "task_id": task_id,
                "reservation_owned": True,
            }
        else:
            outcome = None
        pool_state = worker_pool_admission_state(ctx)
        if outcome is None and not pool_state["available"]:
            outcome = {
                "status": "needs_manual_target",
                "reason": "worker_pool_unavailable",
                "worker_pool_disabled_reason": str(pool_state.get("disabled_reason") or ""),
                "task_id": task_id,
            }
        elif outcome is None:
            outcome = promote_chat_to_task(evt, ctx)
        outcome = outcome if isinstance(outcome, dict) else {"status": "scheduled"}
        admitted_task_contract = outcome.pop("_admitted_task_contract", None)
        if str(outcome.get("status") or "") == "scheduled":
            title = str(evt.get("title") or "").strip()[:80]
            receipt = _emit_routing_receipt(
                ctx,
                evt,
                action=receipt_action,
                target=str(outcome.get("task_id") or task_id),
                status="scheduled",
                detail=str(outcome.get("source_note") or ""),
                attachment_manifest=_events()._routing_attachments(outcome.get("attachment_manifest")),
                publish=False,
            )
            admission_status = (
                "scheduled"
                if receipt.get("persisted") and str(receipt.get("status") or "") == "scheduled"
                else "unconfirmed"
            )
            stored = write_task_result(
                ctx.DRIVE_ROOT,
                str(outcome.get("task_id") or task_id),
                STATUS_SCHEDULED,
                root_task_id=str(outcome.get("task_id") or task_id),
                delegation_role="root",
                task_contract=admitted_task_contract,
                project_id=str(outcome.get("project_id") or evt.get("project_id") or ""),
                description=str(evt.get("objective") or ""),
                expected_output=str(evt.get("expected_output") or ""),
                suggested_name=title,
                promotion_admission={
                    "status": admission_status,
                    "routing_token": str(evt.get("routing_token") or ""),
                    "reason": str(receipt.get("reason") or ""),
                    "confirmed_at": utc_now_iso(),
                    "queue_snapshot_persisted": True,
                    "routing_receipt_required": bool(str(evt.get("client_message_id") or "")),
                    "routing_receipt_status": str(receipt.get("annotation_status") or ""),
                    "source_note": str(outcome.get("source_note") or ""),
                },
                result=(
                    "Task accepted and durably scheduled."
                    if admission_status == "scheduled"
                    else "Task is scheduled, but its owner-facing routing receipt was not confirmed."
                ),
                attachment_manifest=list(outcome.get("attachment_manifest") or []),
            )
            admission = stored.get("promotion_admission") if isinstance(stored, dict) else {}
            if (
                str((admission or {}).get("status") or "") != admission_status
                or str((admission or {}).get("routing_token") or "")
                != str(evt.get("routing_token") or "")
            ):
                raise RuntimeError("scheduled promotion result was not persisted")
            supervisor_queue.release_task_admission(task_id, routing_token)
            if admission_status != "scheduled":
                return {
                    **outcome,
                    "status": "unconfirmed",
                    "reason": str(receipt.get("reason") or "routing_receipt_persist_failed"),
                }
            _publish_routing_ack(
                ctx,
                evt,
                action=receipt_action,
                target=str(outcome.get("task_id") or task_id),
                target_label=str(receipt.get("target_label") or ""),
                status="scheduled",
                attachment_manifest=_events()._routing_attachments(outcome.get("attachment_manifest")),
            )
            if title:
                _broadcast_task_named(
                    {"type": "task_named", "task_id": str(outcome.get("task_id") or task_id),
                     "suggested_name": title}
                )
            try:
                ctx.append_jsonl(
                    ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "promote_chat_to_task_admitted",
                        "task_id": str(outcome.get("task_id") or task_id),
                    },
                )
            except Exception:
                log.warning("Failed to record admitted promote %s", task_id, exc_info=True)
            return outcome

        _rollback_promoted_pending(
            ctx,
            str(outcome.get("task_id") or task_id),
            routing_token,
            reason="promote_chat_to_task_rejected",
        )
        supervisor_queue.release_task_admission(task_id, routing_token)
        if str(outcome.get("reason") or "") != "attachment_admission_rejected":
            _persist_promote_rejection(ctx, evt, outcome)
        _emit_routing_receipt(
            ctx,
            evt,
            action=receipt_action,
            target=str(outcome.get("task_id") or task_id),
            status="needs_manual_target",
            reason=str(outcome.get("reason") or "admission_rejected"),
            detail=str(outcome.get("detail") or ""),
            attachment_manifest=_events()._routing_attachments(outcome.get("attachment_manifest")),
        )
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "promote_chat_to_task_rejected",
                "task_id": str(outcome.get("task_id") or evt.get("task_id") or ""),
                "reason": str(outcome.get("reason") or "admission_rejected"),
                "project_lifecycle": str(outcome.get("project_lifecycle") or ""),
                "worker_pool_disabled_reason": str(
                    outcome.get("worker_pool_disabled_reason") or ""
                ),
            },
        )
        return outcome
    except Exception as exc:
        log.warning("promote_chat_to_task event failed", exc_info=True)
        _rollback_promoted_pending(
            ctx, task_id, routing_token, reason="promote_chat_to_task_failed",
        )
        try:
            from supervisor import queue as supervisor_queue

            supervisor_queue.release_task_admission(task_id, routing_token)
        except Exception:
            pass
        failed_outcome = {
            "status": "unconfirmed",
            "reason": "promotion_persistence_failed",
            "task_id": task_id,
            "detail": f"{type(exc).__name__}: {exc}",
        }
        try:
            _persist_promote_rejection(ctx, evt, failed_outcome, status="unconfirmed")
        except Exception:
            log.warning("Failed to persist promote failure for %s", task_id, exc_info=True)
        _emit_routing_receipt(
            ctx,
            evt,
            action=receipt_action,
            target=str(evt.get("task_id") or ""),
            status="unconfirmed",
            reason="promotion_persistence_failed",
            detail=f"{type(exc).__name__}: {exc}",
        )
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "promote_chat_to_task_failed",
                "task_id": task_id,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return failed_outcome


def _handle_ensure_project_scope(evt: Dict[str, Any], ctx: Any) -> None:
    """Create/attach the registry project for an in-task ensure_project_scope call
    and bind the CURRENT task to it (the worker already set ctx.project_id locally)."""
    from supervisor.workers import ensure_project_scope

    try:
        ensure_project_scope(evt, ctx)
    except Exception:
        log.warning("ensure_project_scope event failed", exc_info=True)


def _handle_routing_manual_target(evt: Dict[str, Any], ctx: Any) -> None:
    """Publish the decision actor's typed abstention without routing work."""
    from ouroboros.project_dialogue import routing_options_with_labels

    options = routing_options_with_labels(ctx.DRIVE_ROOT, evt.get("options"))
    _emit_routing_receipt(
        ctx,
        evt,
        action="route_decision",
        target=str(evt.get("requested_target") or evt.get("reason") or "")[:200],
        status="needs_manual_target",
        reason=str(evt.get("reason") or "target_unspecified"),
        options=options,
        # Durable carrier: the picker click re-forwards these staged specs to
        # the chosen destination long after the routing turn's metadata died.
        attachment_manifest=_events()._routing_attachments(evt.get("attachment_uploads")),
    )

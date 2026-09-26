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

from ouroboros.dialogue_provenance import presence_root_carrier
from ouroboros.task_results import STATUS_FAILED, STATUS_SCHEDULED, write_task_result
from ouroboros.utils import utc_now_iso

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


def _routing_project_address(ctx: Any, target: str, status: str) -> Dict[str, Any]:
    """Navigation address from the successful destination's registered binding."""
    if not target or status not in {"scheduled", "delivered"}:
        return {}
    try:
        from ouroboros.projects_registry import get_project, project_binding_for_task

        binding = project_binding_for_task(ctx.DRIVE_ROOT, target) or {}
        project = get_project(ctx.DRIVE_ROOT, str(binding.get("project_id") or ""))
        if project and project.get("chat_id") is not None and project.get("lifecycle") not in {"deleting", "deleted"}:
            return {"project_id": project["id"], "project_chat_id": int(project["chat_id"])}
    except Exception:
        log.debug("Routing destination projection unavailable", exc_info=True)
    return {}


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
    from ouroboros.project_dialogue import routing_refusal_cause

    if target and not str(target_label or "").strip():
        from ouroboros.project_dialogue import routing_target_label

        target_label = routing_target_label(ctx.DRIVE_ROOT, action, target, task=evt, project_id=str(evt.get("project_id") or ""))
    client_message_id = str(evt.get("client_message_id") or "").strip()
    routing_token = str(evt.get("routing_token") or "").strip()
    annotation_status = "not_applicable"
    project_address = _routing_project_address(ctx, target, status)
    # Q3=A: the owner-facing sentence for a REFUSED act (host table; "" for a
    # landed row or the picker), computed once for the durable row and the ack.
    cause = routing_refusal_cause(action, status, reason, options)
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
                    cause=cause,
                    options=options,
                    attachment_manifest=attachment_manifest,
                    **project_address,
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
        # True only when a row was actually written: an event that carries no
        # receipt id has no receipt, and saying "persisted" for it is how an
        # ensure_project_scope handler reported a bind nobody could read.
        # ``annotation_status`` keeps the three-way fact for callers whose
        # positive authority lives elsewhere (a promote's admission record).
        "persisted": annotation_status == "persisted",
        "status": effective_status,
        "reason": effective_reason,
        "detail": str(detail or ""),
        "annotation_status": annotation_status,
        "routing_token": routing_token,
        "target_label": str(target_label or ""),
    }
    if attachment_manifest is not None:
        receipt["attachment_manifest"] = _events()._routing_attachments(attachment_manifest) or []
    if annotation_status == "failed":
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
            cause=cause,
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
    cause: str = "",
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
                **_routing_project_address(ctx, target, status),
            }
            if options is not None:
                ack_kwargs["options"] = options
            if attachment_manifest is not None:
                ack_kwargs["attachment_manifest"] = attachment_manifest
            if str(cause or ""):
                ack_kwargs["cause"] = str(cause)
            if str(evt.get("routing_token") or ""):
                ack_kwargs["routing_token"] = str(evt.get("routing_token"))
            ack(
                chat_id,
                **ack_kwargs,
            )
    except Exception:
        log.debug("Routing typed ack failed", exc_info=True)


def _handle_project_digest(evt: Dict[str, Any], ctx: Any) -> None:
    """A project task finished: touch the project and wake consciousness early.

    Full project awareness (v6.32.0): the one identity already sees the project's
    chat thread in its unified memory and the finished task in ``task_results``,
    so the digest is a wake REASON, not a message — the next wake-up renders the
    facts itself (``consciousness_wake``) and the one agent decides what to do
    with them — backlog, identity, or nothing (BIBLE P5). A digest of a tree
    consciousness itself started never re-arms the clock (its own finish is not
    news to it; the chain would never sleep).
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
        from ouroboros.consciousness_authority import is_consciousness_origin

        consciousness = getattr(ctx, "consciousness", None)
        if consciousness is not None and not is_consciousness_origin(evt):
            digest_task_id = str(evt.get("task_id") or "").strip()
            reason = f"project_digest:{pid}:{digest_task_id}" if digest_task_id else f"project_digest:{pid}"
            consciousness.notify(reason)
    except Exception:
        log.debug("project_digest consciousness notify failed", exc_info=True)


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


def _own_emitted_stub(ctx: Any, task_id: str, routing_token: str) -> bool:
    """Whether the row on disk is THIS promote's emitted pre-receipt (#1160). A
    refusal that writes no result of its own must still replace it, or the
    reconciliation read answers "admission pending" for ever."""
    try:
        from ouroboros.routing_wait import is_own_admission_stub
        from ouroboros.task_results import load_task_result

        return is_own_admission_stub(load_task_result(ctx.DRIVE_ROOT, task_id), routing_token)
    except Exception:
        log.warning("promote: emitted-stub lookup failed for %s", task_id, exc_info=True)
        return False


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
            project_name=str(evt.get("project_name") or ""),
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
        # No host producer (skill card, Swarm, picker click) stamps `source`, so
        # this exit never bypasses the wrapper's host-initiated refusal notice.
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


def _promote_chat_to_task_outcome(evt: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Spawn a first-class pooled owner task from a conversation-lane promote.

    Unlike ``schedule_subagent`` the child is NOT a subagent: it is a normal
    owner task (live card, canonical drive, project lease participation). The
    conversation lane that emitted the event stays free. Every exit returns the
    typed outcome to ``_handle_promote_chat_to_task``, the one publication
    boundary that tells the owner about a host-initiated refusal.
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
                    # A replay of an already-settled admission: the owner was
                    # told once, so the refusal notice stays silent.
                    "replayed": True,
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
            if blocked["reason"] != "duplicate_task_id" or _own_emitted_stub(ctx, task_id, routing_token):
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
            suggested_name = str(outcome.pop("_admitted_suggested_name", "") or "")
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
            # The admission record is the positive authority; the annotation is
            # required only where an owner message exists to carry it.
            admission_status = (
                "scheduled"
                if str(receipt.get("annotation_status") or "") in {"persisted", "not_applicable"}
                and str(receipt.get("status") or "") == "scheduled"
                else "unconfirmed"
            )
            transfer = outcome.pop("force_plan_transfer", None)
            carrier = presence_root_carrier(evt, task_contract=evt.get("task_contract"))
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
                suggested_name=suggested_name,
                promotion_admission={
                    "status": admission_status,
                    "routing_token": str(evt.get("routing_token") or ""),
                    "reason": str(receipt.get("reason") or ""),
                    "confirmed_at": utc_now_iso(),
                    "queue_snapshot_persisted": True,
                    "routing_receipt_required": bool(str(evt.get("client_message_id") or "")),
                    "routing_receipt_status": str(receipt.get("annotation_status") or ""),
                    "source_note": str(outcome.get("source_note") or ""),
                    # Owner 3=A: the promoter's unmet planning obligation moved onto
                    # this root; the tool reads it back with the admission receipt.
                    **({"force_plan_transfer": dict(transfer)} if isinstance(transfer, dict) and transfer else {}),
                },
                result=(
                    "Task accepted and durably scheduled."
                    if admission_status == "scheduled"
                    else "Task is scheduled, but its owner-facing routing receipt was not confirmed."
                ),
                attachment_manifest=list(outcome.get("attachment_manifest") or []),
                # A Presence promotion's host-carried provenance is canonical from
                # admission, so its binding finds, polls and controls the work while
                # it is still queued (the worker's running write keeps the same value).
                **({"metadata": carrier, "source": "presence_promote"} if carrier else {}),
            )
            admission = stored.get("promotion_admission") if isinstance(stored, dict) else {}
            if (
                str((admission or {}).get("status") or "") != admission_status
                or str((admission or {}).get("routing_token") or "")
                != str(evt.get("routing_token") or "")
            ):
                raise RuntimeError("scheduled promotion result was not persisted")
            supervisor_queue.release_task_admission(task_id, routing_token)
            if isinstance(transfer, dict) and transfer:
                _record_obligation_transfer(ctx, transfer)
            if admission_status != "scheduled":
                return {
                    **outcome,
                    "status": "unconfirmed",
                    "reason": str(receipt.get("reason") or "routing_receipt_persist_failed"),
                }
            from ouroboros.project_handoff import enqueue_project_handoff

            enqueue_project_handoff(ctx.DRIVE_ROOT, str(outcome.get("task_id") or task_id))
            _publish_routing_ack(
                ctx,
                evt,
                action=receipt_action,
                target=str(outcome.get("task_id") or task_id),
                target_label=str(receipt.get("target_label") or ""),
                status="scheduled",
                attachment_manifest=_events()._routing_attachments(outcome.get("attachment_manifest")),
            )
            if suggested_name:
                _broadcast_task_named(
                    {"type": "task_named", "task_id": str(outcome.get("task_id") or task_id),
                     "suggested_name": suggested_name}
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
        if (str(outcome.get("reason") or "") != "attachment_admission_rejected"
                or _own_emitted_stub(ctx, task_id, routing_token)):
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


def _notify_host_initiated_refusal(ctx: Any, evt: Dict[str, Any], outcome: Any) -> None:
    """The ONE place a HOST-issued promote (skill card, Swarm, picker click)
    tells the owner it did not start (Q2=A). No model turn narrates such a
    refusal, so exactly one typed System row lands in the chat the OWNER wrote
    in — never ``task["chat_id"]``, which project admission may have rewritten
    to a new room — bound to the task that never started. A tool-issued promote
    gets nothing here (its receipt plus the failed call are the record and the
    model narrates); preparing, scheduled and replayed outcomes send nothing.
    """
    if not evt.get("host_initiated") or not isinstance(outcome, dict) or outcome.get("replayed"):
        return
    status = str(outcome.get("status") or "")
    if status not in {"needs_manual_target", "unconfirmed"}:
        return
    try:
        from ouroboros.project_dialogue import routing_refusal_cause
        from supervisor.message_bus import notification_chat_route

        chat = notification_chat_route(evt.get("chat_id"))
        if chat is None:
            return
        first_line = next(iter(str(evt.get("objective") or "").strip().splitlines()), "")
        if len(first_line) > 60:
            # An untitled act is named by its request's first words, cut at a
            # word boundary so the row never ends mid-word.
            first_line = first_line[:60].rsplit(" ", 1)[0].rstrip() + "…"
        title = str(evt.get("title") or evt.get("suggested_name") or "").strip() or first_line or "Task"
        action = "route_to_project" if bool(evt.get("routed_from_main")) else "promote_chat_to_task"
        reason = str(outcome.get("reason") or ("admission_rejected" if status == "needs_manual_target" else ""))
        ctx.send_with_budget(
            chat, f"{title} · {routing_refusal_cause(action, status, reason, None)}", role="system",
            system_type="task_start_unconfirmed" if status == "unconfirmed" else "task_not_started",
            task_id=str(outcome.get("task_id") or evt.get("task_id") or ""),
        )
    except Exception:
        log.debug("host-initiated refusal row failed for %s", evt.get("task_id"), exc_info=True)


def _handle_promote_chat_to_task(evt: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """The one publication boundary of a promote: every outcome of the handler
    body — the reservation-blocked and unconfirmed early returns included —
    passes the host-initiated refusal notice before it is returned."""
    outcome = _promote_chat_to_task_outcome(evt, ctx)
    _notify_host_initiated_refusal(ctx, evt, outcome)
    return outcome


def _record_obligation_transfer(ctx: Any, transfer: Dict[str, Any]) -> None:
    """The promoter's task details name where its planning obligation went."""
    promoter = str(transfer.get("from") or "")
    if not promoter:
        return
    try:
        write_task_result(
            ctx.DRIVE_ROOT, promoter, "running",
            _field_projector=lambda current, _patch: {
                "status": str(current.get("status") or "running"),
                "force_plan_transfer": dict(transfer),
            },
        )
    except Exception:
        log.warning("force_plan transfer record failed for %s", promoter, exc_info=True)


def _handle_ensure_project_scope(evt: Dict[str, Any], ctx: Any) -> None:
    """Create/attach the registry project for an in-task ensure_project_scope call,
    bind the CURRENT task to it, and answer on the receipt rail.

    The worker already scoped itself in memory; what it waits for is the DURABLE
    outcome: a landed bind (``delivered``) or a typed refusal (``rejected`` with
    the reason -- bound elsewhere, a refused bind, a registration failure) under
    the act's own synthetic receipt id. Before this the tool said "OK: created"
    before any bind existed and the handler's receipt writer reported an unwritten
    row as persisted."""
    from supervisor.workers import ensure_project_scope

    try:
        outcome = ensure_project_scope(evt, ctx)
    except Exception as exc:
        log.warning("ensure_project_scope event failed", exc_info=True)
        outcome = {"status": "rejected", "reason": "ensure_project_scope_failed",
                   "detail": f"{type(exc).__name__}: {exc}"}
    if not isinstance(outcome, dict):
        outcome = {"status": "unconfirmed", "reason": "handler_returned_no_outcome"}
    target = str(outcome.get("project_id") or evt.get("project_id") or "")
    if outcome.get("status") == "delivered":
        from ouroboros.project_handoff import enqueue_project_handoff

        enqueue_project_handoff(ctx.DRIVE_ROOT, str(evt.get("task_id") or ""))
    label = ""
    try:
        from ouroboros.projects_registry import get_project

        label = str((get_project(ctx.DRIVE_ROOT, target) or {}).get("name") or "") if target else ""
    except Exception:
        log.debug("ensure_project_scope: project name lookup failed", exc_info=True)
    _emit_routing_receipt(
        ctx, evt, action="ensure_project_scope", target=target, target_label=label or target,
        status=str(outcome.get("status") or "unconfirmed"),
        reason=str(outcome.get("reason") or ""), detail=str(outcome.get("detail") or ""),
        publish=False,
    )
    try:
        ctx.append_jsonl(ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "ensure_project_scope_settled",
            "task_id": str(evt.get("task_id") or ""), "project_id": target,
            "status": str(outcome.get("status") or ""), "reason": str(outcome.get("reason") or ""),
            "routing_token": str(evt.get("routing_token") or ""),
        })
    except Exception:
        log.debug("ensure_project_scope settle row failed", exc_info=True)


def _handle_routing_manual_target(evt: Dict[str, Any], ctx: Any) -> None:
    """Publish the decision actor's typed abstention without routing work."""
    from ouroboros.project_dialogue import routing_options_with_labels

    options = routing_options_with_labels(ctx.DRIVE_ROOT, evt.get("options"))
    _emit_routing_receipt(
        ctx,
        evt,
        action="route_decision",
        # An unnamed target stays "": the typed reason is a code, never a task
        # id to label.
        target=str(evt.get("requested_target") or "")[:200],
        status="needs_manual_target",
        reason=str(evt.get("reason") or "target_unspecified"),
        # The model's own words about the abstention, kept beside the typed
        # code on the durable row (replay-only; no owner surface reads it).
        detail=str(evt.get("detail") or ""),
        options=options,
        # Durable carrier: the picker click re-forwards these staged specs to
        # the chosen destination long after the routing turn's metadata died.
        attachment_manifest=_events()._routing_attachments(evt.get("attachment_uploads")),
    )

"""Owner-facing delivery of worker chat events: text, media, and typing.

One owner for what reaches the owner's chat and for the bound chat id every
other handler routes onto. Final answers are deduplicated twice - an in-memory
deque for the fast path and the durable delivery registry that survives a
restart - because the worker deliberately sends the terminal answer over both
the live queue and the buffered return.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


# A progress frame's ``task_id`` is a ROUTING address — it says which live card the
# line lands on, NOT who wrote the line. The supervisor narrates a task's terminal
# path (grace requested, grace withdrawn) onto that task's own card, so those frames
# carry the task's id while the task itself did nothing. Host-authored frames set
# this key; ``_handle_send_message`` refuses to count them as the task's work.
# Without it the supervisor's own voice answers its own question — the grace toast
# stamped last_progress_at, the next 0.5s tick read the task as resumed, and the
# episode it had just opened was withdrawn before the worker could ever drain it.
HOST_NARRATION = "host_narration"


def _bound_project_chat_id(ctx: Any, task_id: Any, parent_task_id: Any = "", root_task_id: Any = "") -> int:
    """Resolve project chat for a task by LINEAGE (own binding -> parent -> root), so a
    subagent of a project task routes to the project thread, not the main chat — only
    the root is bound (post-hoc via UI or ensure_project_scope), children inherit."""
    tid = str(task_id or "").strip()
    if not tid:
        return 0
    try:
        from ouroboros.projects_registry import project_chat_for_task_tree

        return int(project_chat_for_task_tree(ctx.DRIVE_ROOT, tid, parent_task_id, root_task_id) or 0)
    except Exception:
        return 0


def _handle_typing_start(evt: Dict[str, Any], ctx: Any) -> None:
    try:
        chat_id = int(evt.get("chat_id") or 0)
        task_id = str(evt.get("task_id") or "")
        phase = str(evt.get("phase") or "thinking")
        client_msg_id = ""
        kind = ""
        if task_id:
            try:
                from supervisor.active_activity import get_direct_activity_registry
                # A registry hit identifies a direct/ephemeral turn; queued
                # managed tasks also emit typing_start but are not tracked here,
                # so their frames go out without a kind stamp.
                entry = get_direct_activity_registry().get(task_id)
                if entry:
                    client_msg_id = entry.client_message_id
                    kind = entry.kind
            except Exception:
                pass
        if not kind and task_id:
            # A RUNNING queue ROOT is stamped "managed_task" so the client can
            # reconcile its entry against the /api/state activity snapshot
            # (which lists queue roots). Subagent typing keeps the legacy
            # no-kind exemption: no snapshot source enumerates children.
            try:
                running = getattr(ctx, "RUNNING", None)
                meta = running.get(task_id) if isinstance(running, dict) else None
                task_row = meta.get("task") if isinstance(meta, dict) else None
                if isinstance(task_row, dict):
                    from ouroboros.task_results import resolve_task_lineage

                    lineage = resolve_task_lineage(
                        task_id,
                        metadata=task_row.get("metadata"),
                        root_task_id=task_row.get("root_task_id"),
                        parent_task_id=task_row.get("parent_task_id"),
                        delegation_role=task_row.get("delegation_role"),
                        original_task_id=task_row.get("original_task_id"),
                        timeout_retry_from=task_row.get("timeout_retry_from"),
                    )
                    if lineage["is_root_task"]:
                        kind = "managed_task"
            except Exception:
                log.debug("managed typing kind resolution failed for %s", task_id, exc_info=True)
        if chat_id:
            ctx.bridge.send_chat_action(
                chat_id,
                "typing",
                activity_id=task_id,
                client_message_id=client_msg_id,
                phase=phase,
                kind=kind,
            )
    except Exception:
        log.debug("Failed to send typing action to chat", exc_info=True)
        pass


# Delivered final-answer dedupe (mirror of the already_done terminal dedupe, for
# this one event kind): the worker sends the final send_message BOTH over the
# live queue (before blocking post-task) AND in the buffered return — queue.put
# is not a delivery receipt, so neither copy is dropped worker-side; instead
# both carry the same delivery_id and the second one is suppressed here. The
# in-memory deque is a fast-path cache; the durable registry
# (``supervisor.terminal_delivery``, phase A2) is the LOGICAL dedupe shared by
# the natural, cancel, and reap delivery paths and survives a restart.
_DELIVERED_MESSAGE_IDS: "deque[str]" = deque(maxlen=256)


def _register_delivered(ctx: Any, delivery_id: str) -> None:
    """Durably mark one delivery id as delivered — and clear what it owed.

    Both halves of the same fact: the id joins the restart-surviving registry AND
    leaves the pending outbox in one write, so a replay can never re-send an
    answer that landed (phase A2/F7). Fail-soft: a registry write must never cost
    a delivery that already happened.
    """
    try:
        from supervisor.terminal_delivery import register_delivery

        register_delivery(ctx.DRIVE_ROOT, delivery_id)
    except Exception:
        log.debug("durable delivery registration failed", exc_info=True)


def _handle_send_message(evt: Dict[str, Any], ctx: Any) -> None:
    try:
        delivery_id = str(evt.get("delivery_id") or "")
        if delivery_id and delivery_id in _DELIVERED_MESSAGE_IDS:
            log.debug("send_message suppressed as duplicate (delivery_id=%s)", delivery_id)
            # This copy is suppressed because the FIRST one was sent, so record
            # that durably: it also clears any pending-outbox row, which would
            # otherwise be replayed (and suppressed again) until it gave up.
            _register_delivered(ctx, delivery_id)
            return
        if delivery_id:
            try:
                from supervisor.terminal_delivery import already_delivered

                if already_delivered(ctx.DRIVE_ROOT, delivery_id):
                    log.debug(
                        "send_message suppressed as durably delivered (delivery_id=%s)",
                        delivery_id,
                    )
                    return
            except Exception:
                # Fail open toward delivery — never lose an answer to a dedupe read.
                log.debug("durable delivery dedupe read failed", exc_info=True)
        log_text = evt.get("log_text")
        fmt = str(evt.get("format") or "")
        is_progress = bool(evt.get("is_progress"))
        raw_ts = evt.get("ts")
        task_id = str(evt.get("task_id") or "")
        # Real-progress signal (activity model): a progress narration line is genuine work,
        # so stamp the EMITTING task's last_progress_at. (A productively-waiting parent is
        # kept alive separately by _subtree_progressing detecting fresh DESCENDANT progress,
        # not by re-stamping its own last_progress_at from child narration.) HOST_NARRATION
        # frames are addressed to the task's card but authored by the supervisor, so they
        # are narration ABOUT the task, never work BY it.
        progress_meta = evt.get("progress_meta") if isinstance(evt.get("progress_meta"), dict) else None
        _running = getattr(ctx, "RUNNING", None)
        if is_progress and task_id and isinstance(_running, dict):
            _m = _running.get(task_id)
            # Mutate in place (see _handle_llm_usage): no write-back, so a cross-thread
            # cancel that popped this task is never resurrected.
            if isinstance(_m, dict):
                if not evt.get(HOST_NARRATION):
                    _m["last_progress_at"] = time.time()
                # v6.82 (P5): host-attested cancelable marker. RUNNING membership is
                # the supervisor's own truth that this frame belongs to a queue task
                # that /api/tasks/{id}/cancel can force-cancel. An in-process
                # direct-chat turn is never in RUNNING, so its card never shows a
                # dead "Cancel run" button. Covers pooled roots that skip the
                # scheduled notice (e.g. promote_chat_to_task). The marker is
                # LINEAGE-GATED here and carries the RUNNING row's authoritative
                # lineage: only a resolved ROOT is stamped (a timeout-retry root
                # counts — its root_task_id names the original, which is exactly
                # why the frontend must trust this attestation rather than
                # re-deriving rootness from frame shape), and a subagent's
                # narration never mints a root-shaped card with a live Cancel.
                # Copy-on-write: the worker's own event dict is never mutated.
                task_row = _m.get("task") if isinstance(_m.get("task"), dict) else {}
                progress_meta = dict(progress_meta or {})
                for lineage_key in ("root_task_id", "parent_task_id", "delegation_role"):
                    value = str(task_row.get(lineage_key) or "").strip()
                    if value and not progress_meta.get(lineage_key):
                        progress_meta[lineage_key] = value
                try:
                    from ouroboros.task_results import resolve_task_lineage

                    lineage = resolve_task_lineage(
                        task_id,
                        metadata=task_row.get("metadata"),
                        root_task_id=task_row.get("root_task_id"),
                        parent_task_id=task_row.get("parent_task_id"),
                        delegation_role=task_row.get("delegation_role"),
                        original_task_id=task_row.get("original_task_id"),
                        timeout_retry_from=task_row.get("timeout_retry_from"),
                    )
                    if bool(lineage["is_root_task"]):
                        progress_meta["cancelable"] = True
                except Exception:
                    log.debug("cancelable lineage resolution failed for %s", task_id, exc_info=True)
        bound_chat = _bound_project_chat_id(ctx, task_id, evt.get("parent_task_id"), evt.get("root_task_id"))
        chat_id = bound_chat or int(evt["chat_id"])
        ctx.send_with_budget(
            chat_id,
            str(evt.get("text") or ""),
            log_text=(str(log_text) if isinstance(log_text, str) else None),
            fmt=fmt,
            is_progress=is_progress,
            task_id=task_id,
            progress_meta=progress_meta,
            ts=(str(raw_ts) if raw_ts else None),
            # S3 (Q4): a typed system receipt keeps its role/type end to end.
            role=str(evt.get("role") or ""),
            system_type=str(evt.get("system_type") or ""),
        )
        # Registered only AFTER a successful send: if the live copy's send
        # raises, the buffered copy must NOT be suppressed later — "never
        # lost" outranks "never doubled". The durable registration is the
        # restart-surviving half of the same rule.
        if delivery_id:
            _DELIVERED_MESSAGE_IDS.append(delivery_id)
            _register_delivered(ctx, delivery_id)
    except Exception as e:
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "send_message_event_error", "error": repr(e),
            },
        )


def _handle_send_photo(evt: Dict[str, Any], ctx: Any) -> None:
    """Send a photo to the owner's chat."""
    import base64 as b64mod
    try:
        # Binding precedence (matches _handle_send_message/_handle_log_event): a
        # post-hoc bound task keeps its original main chat_id, so its media must
        # still route to the project panel.
        chat_id = _bound_project_chat_id(
            ctx, evt.get("task_id"), evt.get("parent_task_id"), evt.get("root_task_id")
        ) or int(evt.get("chat_id") or 0)
        image_b64 = str(evt.get("image_base64") or "")
        caption = str(evt.get("caption") or "")
        mime = str(evt.get("mime") or "image/png")
        if not chat_id or not image_b64:
            return
        photo_bytes = b64mod.b64decode(image_b64)
        ok, err = ctx.bridge.send_photo(chat_id, photo_bytes, caption=caption, mime=mime)
        if not ok:
            ctx.append_jsonl(
                ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "send_photo_error",
                    "chat_id": chat_id, "error": err,
                },
            )
    except Exception as e:
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "send_photo_event_error", "error": repr(e),
            },
        )


def _handle_send_video(evt: Dict[str, Any], ctx: Any) -> None:
    """Send a video to the owner's chat."""
    import base64 as b64mod
    try:
        # Binding precedence (matches the sibling handlers): a post-hoc bound
        # task's media routes to its project panel, not the old main thread.
        bound_chat = _bound_project_chat_id(
            ctx, evt.get("task_id"), evt.get("parent_task_id"), evt.get("root_task_id")
        )
        raw_chat_id = evt.get("chat_id")
        if not bound_chat and (raw_chat_id is None or raw_chat_id == ""):
            return
        chat_id = bound_chat or int(raw_chat_id)
        video_b64 = str(evt.get("video_base64") or "")
        caption = str(evt.get("caption") or "")
        mime = str(evt.get("mime") or "video/mp4")
        if not video_b64:
            return
        video_bytes = b64mod.b64decode(video_b64)
        ok, err = ctx.bridge.send_video(chat_id, video_bytes, caption=caption, mime=mime)
        if not ok:
            ctx.append_jsonl(
                ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "send_video_error",
                    "chat_id": chat_id, "error": err,
                },
            )
    except Exception as e:
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "send_video_event_error", "error": repr(e),
            },
        )


def _handle_send_document(evt: Dict[str, Any], ctx: Any) -> None:
    """Send an arbitrary document/file to the owner's chat."""
    import base64 as b64mod
    try:
        # Binding precedence (matches the sibling media handlers): a post-hoc
        # bound task's file routes to its project panel, not the old main thread.
        bound_chat = _bound_project_chat_id(
            ctx, evt.get("task_id"), evt.get("parent_task_id"), evt.get("root_task_id")
        )
        raw_chat_id = evt.get("chat_id")
        if not bound_chat and (raw_chat_id is None or raw_chat_id == ""):
            return
        chat_id = bound_chat or int(raw_chat_id)
        file_b64 = str(evt.get("file_base64") or "")
        caption = str(evt.get("caption") or "")
        filename = str(evt.get("filename") or "file")
        mime = str(evt.get("mime") or "application/octet-stream")
        download_url = str(evt.get("download_url") or "")
        task_id = str(evt.get("task_id") or "")
        if not file_b64:
            return
        file_bytes = b64mod.b64decode(file_b64)
        ok, err = ctx.bridge.send_document(
            chat_id, file_bytes, filename=filename, caption=caption, mime=mime,
            download_url=download_url, task_id=task_id,
        )
        if not ok:
            ctx.append_jsonl(
                ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "send_document_error",
                    "chat_id": chat_id, "error": err,
                },
            )
    except Exception as e:
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "send_document_event_error", "error": repr(e),
            },
        )

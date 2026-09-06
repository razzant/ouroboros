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
from supervisor.log_addressing import bound_project_chat_id as _bound_project_chat_id
from supervisor.message_bus import notification_chat_route
from ouroboros.subagent_messages import subagent_message_meta


HOST_NARRATION = "host_narration"


def _handle_typing_start(evt: Dict[str, Any], ctx: Any) -> None:
    try:
        # Membership, not truthiness: absence skips the indicator, an explicit
        # id — the hidden partition included — is a destination.
        chat_id = notification_chat_route(evt.get("chat_id"))
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
        if chat_id is not None:
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


_DELIVERED_MESSAGE_IDS: "deque[str]" = deque(maxlen=256)


def _register_delivered(ctx: Any, delivery_id: str) -> None:
    """Atomically mark one id delivered and clear its pending-outbox row."""
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
        task_row: Dict[str, Any] = {}
        if task_id and isinstance(_running, dict):
            _m = _running.get(task_id)
            # Mutate in place (see _handle_llm_usage): no write-back, so a cross-thread
            # cancel that popped this task is never resurrected.
            if isinstance(_m, dict):
                if is_progress and not evt.get(HOST_NARRATION):
                    _m["last_progress_at"] = time.time()
                task_row = _m.get("task") if isinstance(_m.get("task"), dict) else {}
                child_meta = subagent_message_meta(task_row, task_id=task_id)
                if child_meta:
                    event_meta = dict(progress_meta or {})
                    progress_meta = dict(child_meta)
                    progress_meta.update(event_meta)
            if is_progress and isinstance(_m, dict):
                # Host-attested RUNNING lineage is the cancel authority.
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
            if is_progress and not isinstance(_m, dict):
                # The in-process direct-chat turn has no RUNNING row; it is a
                # root by construction and owner-addressable through the same
                # ownership seam the cancel ingress uses (workers.direct_chat_turn),
                # so its live card carries the same host-attested marker.
                from supervisor import workers

                if workers.direct_chat_turn(task_id) is not None:
                    progress_meta = dict(progress_meta or {})
                    progress_meta["cancelable"] = True
        if task_id and not is_progress and not task_row and not subagent_message_meta(progress_meta, task_id=task_id):
            try:
                from ouroboros.task_status import load_effective_task_result

                result = load_effective_task_result(ctx.DRIVE_ROOT, task_id, materialize_artifacts=False)
                child_meta = subagent_message_meta(result, task_id=task_id)
                if child_meta:
                    event_meta = dict(progress_meta or {})
                    progress_meta = dict(child_meta)
                    progress_meta.update(event_meta)
            except Exception:
                log.debug("final lineage recovery failed for %s", task_id, exc_info=True)
        meta = progress_meta or {}
        bound_chat = _bound_project_chat_id(
            ctx, task_id,
            meta.get("parent_task_id") or evt.get("parent_task_id"),
            meta.get("root_task_id") or evt.get("root_task_id"),
        )
        system_type = str(evt.get("system_type") or "")
        # Project lifecycle rows pin Main; others keep lineage routing.
        chat_id = int(evt["chat_id"]) if system_type in ("project_started", "project_completion_summary") else bound_chat or int(evt["chat_id"])
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
            system_type=system_type,
        )
        # Register only after send; a failed first copy must not suppress retry.
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



def _delivery_chat_id(evt: Dict[str, Any], ctx: Any) -> "int | None":
    """The bound delivery chat for a media/links/quiz frame (upstream unification):
    post-hoc bound media stays in the project panel even when the task retained
    its original chat id; chat 0 is a real hidden session (same contract across
    photo/video/file/links)."""
    bound_chat = _bound_project_chat_id(
        ctx, evt.get("task_id"), evt.get("parent_task_id"), evt.get("root_task_id")
    )
    raw_chat_id = evt.get("chat_id")
    if not bound_chat and (raw_chat_id is None or raw_chat_id == ""):
        return None
    return bound_chat or int(raw_chat_id)


def _log_error(ctx: Any, event_type: str, **fields: Any) -> None:
    ctx.append_jsonl(
        ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {"ts": utc_now_iso(), "type": event_type, **fields},
    )


def _handle_send_photo(evt: Dict[str, Any], ctx: Any) -> None:
    """Send a photo to the owner's chat."""
    import base64 as b64mod
    try:
        # Binding precedence matches text delivery: post-hoc bound media stays
        # in the project panel even when the task retained its original chat id.
        # chat 0 is a real hidden session (same contract as video/file/links).
        chat_id = _delivery_chat_id(evt, ctx)
        image_b64 = str(evt.get("image_base64") or "")
        caption = str(evt.get("caption") or "")
        mime = str(evt.get("mime") or "image/png")
        if chat_id is None or not image_b64:
            return
        photo_bytes = b64mod.b64decode(image_b64)
        ok, err = ctx.bridge.send_photo(
            chat_id, photo_bytes, caption=caption, mime=mime,
            task_id=str(evt.get("task_id") or ""),
        )
        if not ok:
            _log_error(ctx, "send_photo_error", chat_id=chat_id, error=err)
    except Exception as e:
        _log_error(ctx, "send_photo_event_error", error=repr(e))


def _handle_send_video(evt: Dict[str, Any], ctx: Any) -> None:
    """Send a video to the owner's chat."""
    import base64 as b64mod
    try:
        chat_id = _delivery_chat_id(evt, ctx)
        video_b64 = str(evt.get("video_base64") or "")
        caption = str(evt.get("caption") or "")
        mime = str(evt.get("mime") or "video/mp4")
        if chat_id is None or not video_b64:
            return
        video_bytes = b64mod.b64decode(video_b64)
        ok, err = ctx.bridge.send_video(
            chat_id, video_bytes, caption=caption, mime=mime,
            task_id=str(evt.get("task_id") or ""),
        )
        if not ok:
            _log_error(ctx, "send_video_error", chat_id=chat_id, error=err)
    except Exception as e:
        _log_error(ctx, "send_video_event_error", error=repr(e))


def _handle_send_document(evt: Dict[str, Any], ctx: Any) -> None:
    """Send an arbitrary document/file to the owner's chat."""
    import base64 as b64mod
    try:
        chat_id = _delivery_chat_id(evt, ctx)
        file_b64 = str(evt.get("file_base64") or "")
        if chat_id is None or not file_b64:
            return
        ok, err = ctx.bridge.send_document(
            chat_id,
            b64mod.b64decode(file_b64),
            filename=str(evt.get("filename") or "file"),
            caption=str(evt.get("caption") or ""),
            mime=str(evt.get("mime") or "application/octet-stream"),
            download_url=str(evt.get("download_url") or ""),
            task_id=str(evt.get("task_id") or ""),
        )
        if not ok:
            _log_error(ctx, "send_document_error", chat_id=chat_id, error=err)
    except Exception as e:
        _log_error(ctx, "send_document_event_error", error=repr(e))


def _handle_send_links(evt: Dict[str, Any], ctx: Any) -> None:
    """Send structured HTTP(S) actions to the owner's chat."""
    try:
        chat_id = _delivery_chat_id(evt, ctx)
        actions = evt.get("actions")
        if chat_id is None or not isinstance(actions, list) or not actions:
            return
        ok, err = ctx.bridge.send_links(
            chat_id,
            actions,
            title=str(evt.get("title") or ""),
            task_id=str(evt.get("task_id") or ""),
        )
        if not ok:
            _log_error(ctx, "send_links_error", chat_id=chat_id, error=err)
    except Exception as e:
        _log_error(ctx, "send_links_event_error", error=repr(e))


def _handle_send_quiz(evt: Dict[str, Any], ctx: Any) -> None:
    """Send an owner quiz card to the owner's chat."""
    try:
        chat_id = _delivery_chat_id(evt, ctx)
        options = evt.get("options")
        if chat_id is None or not isinstance(options, list) or not options:
            return
        if chat_id == 0:
            # Deliberate exception to the "0 is a real hidden session" policy:
            # an interactive card in the hidden panel can never be seen or
            # answered, so a headless author's quiz goes to Main.
            chat_id = 1
        ok, err = ctx.bridge.send_quiz(
            chat_id,
            quiz_id=str(evt.get("quiz_id") or ""),
            question=str(evt.get("question") or ""),
            options=options,
            stake=str(evt.get("stake") or ""),
            assumption=str(evt.get("assumption") or ""),
            state=str(evt.get("state") or "open"),
            task_id=str(evt.get("task_id") or ""),
        )
        if not ok:
            _log_error(ctx, "send_quiz_error", chat_id=chat_id, error=err)
    except Exception as exc:
        _log_error(ctx, "send_quiz_event_error", error=repr(exc))


# Merged into supervisor.events.EVENT_HANDLERS (the `**_CEH` pattern): the
# structured chat-delivery event registry this leaf owns.
EVENT_HANDLERS = {
    "send_photo": _handle_send_photo,
    "send_video": _handle_send_video,
    "send_document": _handle_send_document,
    "send_links": _handle_send_links,
    "send_quiz": _handle_send_quiz,
}

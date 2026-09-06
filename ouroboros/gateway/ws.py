"""WebSocket gateway dispatch and broadcast state."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import pathlib
import threading
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from ouroboros.config import DATA_DIR
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_ws_clients: list[WebSocket] = []
_ws_lock = threading.Lock()
_event_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Set the server event loop used by ``broadcast_ws_sync``."""
    global _event_loop
    _event_loop = loop


_MAX_NATIVE_IMAGE_BYTES = 8 * 1024 * 1024


def _first_image_attachment(attachments: Any) -> tuple[str, str, str]:
    """Resolve the first image upload into (base64, mime, caption).

    Attachments reference files already stored by /api/chat/upload under
    data/uploads/ — only a validated basename is accepted (no traversal), so
    the WS frame cannot read arbitrary paths. Non-image/missing/oversized
    attachments keep the text-label-only behavior.
    """
    if not isinstance(attachments, list):
        return "", "", ""
    import base64
    import os as _os

    uploads_dir = (pathlib.Path(DATA_DIR) / "uploads").resolve(strict=False)
    for item in attachments:
        if not isinstance(item, dict):
            continue
        mime = str(item.get("mime") or "").lower()
        if not mime.startswith("image/"):
            continue
        name = _os.path.basename(str(item.get("filename") or "").strip())
        if not name or name in {".", ".."}:
            continue
        path = (uploads_dir / name).resolve(strict=False)
        try:
            path.relative_to(uploads_dir)
        except ValueError:
            continue
        try:
            if not path.is_file() or path.stat().st_size > _MAX_NATIVE_IMAGE_BYTES:
                continue
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
        caption = str(item.get("display_name") or name)
        return encoded, mime, f"[user attachment: {caption}]"
    return "", "", ""


def _chat_attachment_uploads(attachments: Any) -> list[dict]:
    """Resolve EVERY desktop-chat attachment (any type) to a staging spec.

    v6.52.0 (P1, full desktop unify): the paperclip allows multiple files of any
    type. Each references a file already stored by /api/chat/upload under
    data/uploads/, addressed by a validated basename (no traversal) — same
    confinement as ``_first_image_attachment``. Missing files are skipped. The
    returned specs feed ``stage_task_attachments`` in the worker, which routes the
    WHOLE set through the shared artifact_store substrate (images natively + every
    other file via the read_file manifest), so nothing is silently dropped.

    Returns a list of ``{"path": <abs uploads path>, "label": <display>, "mime"}``.
    """
    if not isinstance(attachments, list):
        return []
    import os as _os

    uploads_dir = (pathlib.Path(DATA_DIR) / "uploads").resolve(strict=False)
    specs: list[dict] = []
    for ordinal, item in enumerate(attachments):
        if not isinstance(item, dict):
            specs.append({"path": "", "label": f"attachment {ordinal + 1}", "mime": ""})
            continue
        name = _os.path.basename(str(item.get("filename") or "").strip())
        label = str(item.get("display_name") or item.get("label") or name).strip()
        if not name or name in {".", ".."}:
            specs.append({"path": "", "label": label or f"attachment {ordinal + 1}", "mime": str(item.get("mime") or "")})
            continue
        path = (uploads_dir / name).resolve(strict=False)
        try:
            path.relative_to(uploads_dir)
        except ValueError:
            specs.append({"path": "", "label": label or name, "mime": str(item.get("mime") or "")})
            continue
        if not path.is_file():
            specs.append({"path": "", "label": label or name, "mime": str(item.get("mime") or "")})
            continue
        specs.append({"path": str(path), "label": label or name, "mime": str(item.get("mime") or "")})
    return specs


def has_ws_clients() -> bool:
    with _ws_lock:
        return bool(_ws_clients)


async def close_all_ws(*, code: int = 1012, reason: str = "Server restarting") -> None:
    """Close every connected browser websocket best-effort."""
    with _ws_lock:
        clients = list(_ws_clients)
    for ws in clients:
        try:
            await ws.close(code=code, reason=reason)
        except Exception:
            pass


async def broadcast_ws(msg: dict) -> None:
    """Send a message to all connected WebSocket clients."""
    data = json.dumps(msg, ensure_ascii=False, default=str)
    msg_type = str(msg.get("type", "unknown"))
    with _ws_lock:
        clients = list(_ws_clients)
        total_clients = len(clients)
    dead = []
    # WS4: send to all clients CONCURRENTLY — one slow / half-open client no longer
    # head-of-lines the broadcast (and the heartbeat) to every other client.
    results = await asyncio.gather(
        *(ws.send_text(data) for ws in clients), return_exceptions=True
    )
    for ws, result in zip(clients, results):
        if isinstance(result, BaseException):
            log.info(
                "WebSocket send failed for msg type=%s; dropping client (%s)",
                msg_type,
                type(result).__name__,
            )
            dead.append(ws)
    if dead:
        with _ws_lock:
            for ws in dead:
                try:
                    _ws_clients.remove(ws)
                except ValueError:
                    pass
        try:
            from ouroboros.utils import append_jsonl

            append_jsonl(
                pathlib.Path(DATA_DIR) / "logs" / "events.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "broadcast_partial_failure",
                    "msg_type": msg_type,
                    "dead_clients": len(dead),
                    "total_clients": total_clients,
                },
            )
        except Exception:
            log.debug("Failed to emit broadcast_partial_failure event", exc_info=True)


def broadcast_ws_sync(msg: dict) -> None:
    """Thread-safe sync wrapper for broadcasting."""
    loop = _event_loop
    if loop is None:
        return
    coro = broadcast_ws(msg)
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        # A closed or already-stopped loop never schedules the coroutine. The
        # broadcast is deliberately a no-op then, but the coroutine object must
        # still be closed here: dropped un-awaited it emits "coroutine
        # 'broadcast_ws' was never awaited" from whatever code is running when
        # it is collected, blaming an innocent caller for this line.
        coro.close()


async def _dispatch_extension_message(
    websocket: WebSocket,
    msg: dict[str, Any],
    msg_type: str,
) -> bool:
    """Return True when an extension handler owned this message."""
    parsed_ext_type = None
    if isinstance(msg_type, str):
        try:
            from ouroboros.extension_loader import parse_extension_surface_name

            parsed_ext_type = parse_extension_surface_name(msg_type)
        except Exception:
            parsed_ext_type = None
    if not parsed_ext_type:
        return False

    state = None
    try:
        from ouroboros.config import get_skills_repo_path, load_settings
        from ouroboros.extension_loader import (
            extension_name_prefix,
            list_ws_handlers,
            reconcile_extension,
        )
        from ouroboros.skill_loader import discover_skills

        drive_root = pathlib.Path(
            websocket.app.state.drive_root  # type: ignore[attr-defined]
            if hasattr(websocket.app, "state") and hasattr(websocket.app.state, "drive_root")
            else DATA_DIR
        )
        repo_dir = pathlib.Path(
            websocket.app.state.repo_dir  # type: ignore[attr-defined]
            if hasattr(websocket.app, "state") and hasattr(websocket.app.state, "repo_dir")
            else pathlib.Path(__file__).resolve().parents[2]
        )
        repo_path = get_skills_repo_path()
        handler_spec = list_ws_handlers().get(msg_type)
        skill_name = str((handler_spec or {}).get("skill") or "")
        if not skill_name:
            for skill in discover_skills(drive_root, repo_path=repo_path):
                if msg_type.startswith(extension_name_prefix(skill.name)):
                    skill_name = skill.name
                    break
        if not skill_name:
            raise KeyError(msg_type)
        state = reconcile_extension(skill_name, drive_root, load_settings, repo_path=repo_path)
        if not state.get("desired_live"):
            await websocket.send_text(json.dumps({"type": "log", "data": {"level": "warning", "message": f"extension WS handler {msg_type!r} is not live: {state.get('reason')}"}}))
            return True
        if state.get("action") == "extension_load_error" or not state.get("live_loaded"):
            await websocket.send_text(json.dumps({"type": "log", "data": {"level": "warning", "message": f"extension WS handler {msg_type!r} failed to go live: {state.get('load_error') or state.get('reason')}"}}))
            return True
        handler_spec = list_ws_handlers().get(msg_type)
    except Exception:
        handler_spec = None

    if handler_spec is None:
        extra = ""
        if isinstance(state, dict) and state.get("action") == "extension_load_error":
            extra = f" (load_error={state.get('load_error')})"
        await websocket.send_text(json.dumps({"type": "log", "data": {"level": "warning", "message": f"no extension WS handler for {msg_type!r}{extra}"}}))
        return True

    if handler_spec.get("out_of_process"):
        try:
            from ouroboros.extension_process_runner import dispatch_extension_ws_subprocess

            result = await asyncio.to_thread(
                dispatch_extension_ws_subprocess,
                handler_spec,
                msg,
                drive_root=drive_root,
                repo_dir=repo_dir,
            )
            if result is not None:
                await websocket.send_text(json.dumps({"type": msg_type + ".reply", "data": result}))
        except Exception as exc:
            await websocket.send_text(json.dumps({"type": "log", "data": {"level": "error", "message": f"extension WS handler {msg_type!r} child failed: {type(exc).__name__}: {exc}"}}))
        return True

    handler = handler_spec.get("handler")
    try:
        from ouroboros.extension_process_runner import disclose_inprocess_extension_dispatch

        disclose_inprocess_extension_dispatch(
            handler_spec,
            drive_root=drive_root,
            surface_kind="ws",
            surface=msg_type,
        )
    except Exception as exc:
        await websocket.send_text(json.dumps({
            "type": "log",
            "data": {
                "level": "error",
                "message": (
                    f"extension WS handler {msg_type!r} model-cost disclosure failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            },
        }))
        return True
    try:
        result = handler(msg) if callable(handler) else None
        if inspect.iscoroutine(result):
            result = await result
        if result is not None:
            await websocket.send_text(json.dumps({"type": msg_type + ".reply", "data": result}))
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "log", "data": {"level": "error", "message": f"extension WS handler {msg_type!r} raised: {type(exc).__name__}: {exc}"}}))
    return True


async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    with _ws_lock:
        _ws_clients.append(websocket)
        total = len(_ws_clients)
    log.info("WebSocket client connected (total: %d)", total)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue

            msg_type = str(msg.get("type", "") or "")
            if await _dispatch_extension_message(websocket, msg, msg_type):
                continue

            # Executable gateway ABI (ABI-3, Q7=A): inbound chat/command frames
            # are validated against the derived contract schema at THIS ingress
            # seam only — egress and history replay are never validated.
            if msg_type in ("chat", "command"):
                from ouroboros.gateway.contracts import ChatInbound, CommandInbound
                from ouroboros.gateway.schema import validate_ingress

                schema_errors = validate_ingress(
                    msg, ChatInbound if msg_type == "chat" else CommandInbound)
                if schema_errors:
                    await websocket.send_text(json.dumps({
                        "type": "log",
                        "data": {
                            "level": "warning",
                            "message": ("ingress schema rejected a "
                                        f"{msg_type} message: "
                                        + "; ".join(schema_errors[:5])),
                        },
                    }))
                    continue

            payload = msg.get("content", "") if msg_type == "chat" else msg.get("cmd", "")
            if msg_type in ("chat", "command") and payload:
                try:
                    from ouroboros.client_surface import normalize_client_surface
                    from supervisor.message_bus import get_bridge

                    bridge = get_bridge()
                    if msg_type == "chat":
                        force_plan = bool(msg.get("force_plan"))
                        client_surface = normalize_client_surface(msg.get("client_surface"))
                        image_b64, image_mime, image_caption = _first_image_attachment(
                            msg.get("attachments")
                        )
                        try:
                            thread_id = int(msg.get("chat_id") or 1)
                        except (TypeError, ValueError):
                            thread_id = 1
                        task_metadata: dict[str, Any] = {
                            "force_plan": force_plan,
                            "force_plan_source": "swarm" if force_plan else "",
                        }
                        # v6.52.0 (P1, full desktop unify): forward the WHOLE attachment
                        # set (any type) so the worker stages all of them through the
                        # shared substrate — not just the first image. Flows to
                        # task["metadata"]["chat_attachment_uploads"] like force_plan.
                        uploads = _chat_attachment_uploads(msg.get("attachments"))
                        if uploads:
                            task_metadata["chat_attachment_uploads"] = uploads
                        if client_surface is not None:
                            # Sending-surface observables ride task_metadata
                            # (the force_plan rail): they pass the bus whitelist
                            # nested, reach direct/ephemeral/queued turns, and
                            # land in chat.jsonl at the canonical-row writer.
                            client_surface["received_at"] = utc_now_iso()
                            task_metadata["client_surface"] = client_surface
                        bridge.ui_send(
                            payload,
                            sender_session_id=str(msg.get("sender_session_id", "") or ""),
                            client_message_id=str(msg.get("client_message_id", "") or ""),
                            image_base64=image_b64,
                            image_mime=image_mime,
                            image_caption=image_caption,
                            task_metadata=task_metadata,
                            chat_id=thread_id,
                            project_id=str(msg.get("project_id", "") or ""),
                        )
                    else:
                        bridge.ui_send(payload, broadcast=False)
                except Exception:
                    await websocket.send_text(json.dumps({
                        "type": "chat",
                        "role": "assistant",
                        "content": "⚠️ System is still initializing. Please wait a moment and try again.",
                        "ts": utc_now_iso(),
                    }))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning("WebSocket error: %s", exc)
    finally:
        with _ws_lock:
            try:
                _ws_clients.remove(websocket)
            except ValueError:
                pass
            total = len(_ws_clients)
        log.info("WebSocket client disconnected (total: %d)", total)


__all__ = [
    "broadcast_ws",
    "broadcast_ws_sync",
    "close_all_ws",
    "has_ws_clients",
    "set_event_loop",
    "ws_endpoint",
]

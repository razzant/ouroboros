"""WebSocket gateway dispatch and broadcast state."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import pathlib
import threading
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from ouroboros import chat_uploads
from ouroboros.config import DATA_DIR
from ouroboros.gateway._helpers import run_sync_to_completion, settle_to_completion
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_ws_clients: list[WebSocket] = []
_ws_lock = threading.Lock()
_event_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Set the server event loop used by ``broadcast_ws_sync``."""
    global _event_loop
    _event_loop = loop


def _chat_attachment_uploads(attachments: Any) -> list[dict]:
    """Resolve EVERY desktop-chat attachment (any type) to a staging spec.

    v6.52.0 (P1, full desktop unify): the paperclip allows multiple files of any
    type. Each references a file already stored by /api/chat/upload under
    data/uploads/, addressed by a validated basename (no traversal); a missing or
    invalid one keeps its ordinal with an empty path (rejected at staging), and
    ``_accept_with_attachments`` binds every spec to its measured identity. The
    returned specs feed ``stage_task_attachments`` in the worker, which routes the
    WHOLE set through the shared artifact_store substrate (images natively + every
    other file via the read_file manifest), so nothing is silently dropped.

    Returns a list of ``{"path": <abs uploads path>, "label": <display>, "mime"}``.
    """
    if not isinstance(attachments, list):
        return []
    import os as _os

    uploads_dir = chat_uploads.uploads_dir(DATA_DIR).resolve(strict=False)
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


def _accept_with_attachments(bridge: Any, payload: str, send_kwargs: dict, attachments: Any) -> None:
    """Measure a web frame's attachments off the loop, then accept the message.

    The refs ride ``task_metadata["chat_attachments"]`` (the client_surface rail) to
    the canonical writer, which claims them and records them on the inbound row.
    Each model staging spec (one per frame item, in order) carries its ref's measured
    identity, which ``stage_task_attachments`` verifies while copying: a path swapped
    after this measurement stages nothing else. An unavailable ref stages nothing.
    """
    uploads = _chat_attachment_uploads(attachments)
    refs = chat_uploads.refs_for_frame(attachments, DATA_DIR)
    for spec, ref in zip(uploads, refs):
        spec.update({"path": ""} if ref.get("unavailable") else {"size": ref["size"], "sha256": ref["sha256"]})
    send_kwargs["task_metadata"]["chat_attachment_uploads"] = uploads
    send_kwargs["task_metadata"]["chat_attachments"] = refs
    bridge.ui_send(payload, **send_kwargs)


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
            runtime_state_for_skill_name,
        )
        from ouroboros.skill_peer_inventory import discover_skill_peers

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
            for skill in await asyncio.to_thread(discover_skill_peers, drive_root, repo_path=repo_path):
                if msg_type.startswith(extension_name_prefix(skill.name)):
                    skill_name = skill.name
                    break
        if not skill_name:
            raise KeyError(msg_type)
        state = await asyncio.to_thread(reconcile_extension, skill_name, drive_root, load_settings, repo_path=repo_path)
        state = {**state, **await asyncio.to_thread(runtime_state_for_skill_name, skill_name, drive_root, repo_path=repo_path)}
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
        # Mirror the HTTP dispatcher: a synchronous in-process handler (and its
        # synchronous barrier wait) runs off the ASGI loop, so one skill's
        # blocking callback never stalls unrelated HTTP and WebSocket work.
        if not callable(handler):
            result = None
        elif inspect.iscoroutinefunction(handler):
            result = await handler(msg)
        else:
            result = await asyncio.to_thread(handler, msg)
        if inspect.iscoroutine(result):
            result = await result
        if result is not None:
            await websocket.send_text(json.dumps({"type": msg_type + ".reply", "data": result}))
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "log", "data": {"level": "error", "message": f"extension WS handler {msg_type!r} raised: {type(exc).__name__}: {exc}"}}))
    return True


def _initialization_notice() -> str:
    return json.dumps({
        "type": "chat",
        "role": "system", "system_type": "initialization_notice",
        "content": "⚠️ System is still initializing. Please wait a moment and try again.",
        "ts": utc_now_iso(),
    })


async def _accept_chat_after(websocket: WebSocket, previous: asyncio.Task | None, accept) -> None:
    """Run one chat frame's acceptance once this socket's previous one settled.

    The web acceptance takes the single host's ingress lock and a locked durable
    append (log_chat(require_write=True)); either may wait behind a skill delivery
    scanning retained chat or a slow disk, so it runs off the ASGI loop, and the
    receive loop never awaits it: a command frame on the same socket (Panic,
    Restart) is admitted meanwhile. ``run_sync_to_completion`` keeps custody of
    row → queue → echo; a failure answers the sender with the initialization notice.
    """
    if previous is not None:
        await asyncio.wait((previous,))  # its failure was its own notice
    try:
        await run_sync_to_completion(accept)
    except Exception:
        try:
            await websocket.send_text(_initialization_notice())
        except Exception:
            log.debug("WebSocket closed before its chat failure notice", exc_info=True)


async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    with _ws_lock:
        _ws_clients.append(websocket)
        total = len(_ws_clients)
    log.info("WebSocket client connected (total: %d)", total)
    # Tail of this socket's chat acceptances: each waits for the one before it,
    # so its chat frames settle in receive order while the loop keeps receiving.
    accepting: asyncio.Task | None = None
    controls: set[asyncio.Task] = set()
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
            if msg_type in ("chat", "command") and (payload or (msg_type == "chat" and msg.get("attachments"))):
                try:
                    from ouroboros.client_surface import normalize_client_surface
                    from supervisor.message_bus import try_get_bridge

                    bridge = try_get_bridge()
                    if msg_type == "chat":
                        # The composer sends slash text as a chat frame. Offer
                        # Panic to the authenticated socket's emergency door
                        # before this socket's ordered chat-acceptance tail;
                        # otherwise an earlier blocked append could delay Stop.
                        if bridge is not None and bridge.panic.request(payload):
                            continue
                        force_plan = bool(msg.get("force_plan"))
                        client_surface = normalize_client_surface(msg.get("client_surface"))
                        try:
                            thread_id = int(msg.get("chat_id") or 1)
                        except (TypeError, ValueError):
                            thread_id = 1
                        task_metadata: dict[str, Any] = {
                            "force_plan": force_plan,
                            "force_plan_source": "swarm" if force_plan else "",
                        }
                        # Attachment inspection belongs to the ordered acceptance
                        # worker, so this socket can receive Panic during disk I/O.
                        if client_surface is not None:
                            # Sending-surface observables ride task_metadata
                            # (the force_plan rail): they pass the bus whitelist
                            # nested, reach direct/ephemeral/queued turns, and
                            # land in chat.jsonl at the canonical-row writer.
                            client_surface["received_at"] = utc_now_iso()
                            task_metadata["client_surface"] = client_surface
                        send_kwargs = dict(
                            broadcast=True,
                            sender_session_id=str(msg.get("sender_session_id", "") or ""),
                            client_message_id=str(msg.get("client_message_id", "") or ""),
                            task_metadata=task_metadata,
                            chat_id=thread_id,
                            project_id=str(msg.get("project_id", "") or ""),
                        )
                        if str(payload).strip().lower() == "/restart":
                            from ouroboros.server_control import dispatch_accepted_restart

                            send_kwargs["dispatch"] = functools.partial(
                                dispatch_accepted_restart, bridge,
                                callback=getattr(websocket.app.state, "startup_owner_command", None))
                    else:
                        send_kwargs = {"broadcast": False}
                    if msg_type == "command" and (bridge is None or str(payload).strip().lower() == "/restart"):
                        callback = getattr(websocket.app.state, "startup_owner_command", None)
                        action = callback(payload, send_kwargs=send_kwargs) if callable(callback) else None
                        if action is not None:
                            # A slow checkout cannot hold this socket's Panic behind
                            # Restart. Retain accepted control work through disconnect.
                            control = asyncio.create_task(_accept_chat_after(websocket, None, action))
                            controls.add(control)
                            control.add_done_callback(controls.discard)
                            continue
                    if bridge is None:
                        raise AssertionError("message bus is not initialized")
                    if msg_type == "chat":
                        accept = (functools.partial(_accept_with_attachments, bridge, payload, send_kwargs,
                                                    msg["attachments"]) if msg.get("attachments")
                                  else functools.partial(bridge.ui_send, payload, **send_kwargs))
                        accepting = asyncio.create_task(_accept_chat_after(websocket, accepting, accept))
                    else:
                        bridge.ui_send(payload, **send_kwargs)
                except Exception:
                    await websocket.send_text(_initialization_notice())
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
        if accepting is not None:
            # Received chat frames keep custody through disconnect or cancellation:
            # the socket task returns only after each settled row → queue → echo.
            await settle_to_completion(accepting)
        if controls:
            await settle_to_completion(asyncio.gather(*controls))


__all__ = [
    "broadcast_ws",
    "broadcast_ws_sync",
    "close_all_ws",
    "has_ws_clients",
    "set_event_loop",
    "ws_endpoint",
]

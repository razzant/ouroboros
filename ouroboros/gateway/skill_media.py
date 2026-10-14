"""Narrow, authenticated skill-to-skill audio transcription on the Host Service.

A transport's Host token is never an extension-route credential. This boundary
admits one reviewed target declaration and sends only media bytes and MIME to
the existing extension dispatch; no caller URL, path or headers cross it.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.contracts.skill_manifest import VOICE_INPUT_MAX_BYTES, canonical_skill_name
from ouroboros.extension_loader import runtime_state_for_skill_name
from ouroboros.skill_loader import find_skill, grant_status_for_skill, load_enabled


def _error(code: str, status: int) -> JSONResponse:
    return JSONResponse({"ok": False, "code": code}, status_code=status)


def _target(data_dir, name: str) -> dict:
    """A fresh target snapshot; the gateway independently regates at dispatch."""
    if not name or canonical_skill_name(name) != name:
        raise ValueError("invalid_target")
    skill = find_skill(data_dir, name)
    if (skill is None or skill.identity_collision or not load_enabled(data_dir, name)
            or not skill.review.gate_for(skill.content_hash).get("executable_review")):
        raise ValueError("target_not_ready")
    declaration = skill.manifest.media_transcription
    if not declaration or not grant_status_for_skill(data_dir, skill).get("all_granted"):
        raise ValueError("target_not_ready")
    return {"sha": skill.content_hash, **declaration}


def _caller_origin(data_dir, caller: str, headers) -> None:
    """A Presence transport must name its existing owner-created binding."""
    from ouroboros.presence_bindings import load_presence_binding

    skill = find_skill(data_dir, caller)
    if skill is None:
        raise ValueError("caller_not_ready")
    permissions = set(skill.manifest.permissions)
    granted = set(grant_status_for_skill(data_dir, skill).get("granted_permissions") or [])
    if "presence" in permissions:
        if "presence" not in granted:
            raise ValueError("caller_presence_not_granted")
        binding = load_presence_binding(data_dir, caller, headers.get("x-presence-binding", ""))
        origin = binding.origin
        if (headers.get("x-media-provider") != origin.transport
                or headers.get("x-media-account") != origin.account_id
                or origin.conversation_id not in ("*", headers.get("x-media-conversation"))
                or ((origin.conversation_id != "*" or origin.thread_id)
                    and headers.get("x-media-thread", "") != origin.thread_id)):
            raise ValueError("origin_mismatch")
    elif "inject_chat" not in granted:
        raise ValueError("caller_not_a_transport")


async def _dispatch(request: Request, data_dir, name: str, declaration: dict, audio: bytes, mime: str):
    """Only these synthetic bytes/headers enter the target's ordinary live gate."""
    from ouroboros.gateway.extensions import api_extension_dispatch

    path = f"/api/extensions/{name}/{declaration['route']}"
    # The served module tree is the source for an isolated route child, never
    # a caller path or a possibly unrelated installation's global repo value.
    app = SimpleNamespace(state=SimpleNamespace(drive_root=data_dir, repo_dir=Path(__file__).resolve().parents[2]))
    scope = {"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
             "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 0), "app": app,
             "path": path, "raw_path": quote(path, safe="/").encode("ascii"), "query_string": b"",
             "path_params": {"skill": name, "rest": declaration["route"]},
             "headers": [(b"content-type", mime.encode("ascii")),
                         (b"content-length", str(len(audio)).encode("ascii"))]}
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            # The in-process proxy has no client socket. A lazy streaming
            # extension response must not see a fictitious disconnect.
            await asyncio.Future()
        sent = True
        return {"type": "http.request", "body": audio, "more_body": False}

    response = await api_extension_dispatch(Request(scope, receive))
    status = None
    body = bytearray()

    async def send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            chunk = message.get("body") or b""
            if len(body) + len(chunk) > 16384:
                raise ValueError("target response exceeded audio-result limit")
            body.extend(chunk)

    # RouteStreamResponse starts its child and produces bytes only inside this
    # ASGI call; Response.body on the unopened handle is always empty.
    await response(scope, receive, send)
    return status, bytes(body)


async def api_media_transcribe(request: Request) -> JSONResponse:
    from ouroboros.gateway.host_service import HostServiceAuthError, _authenticated, _bounded_host_read
    from ouroboros.presence_bindings import PresenceBindingError

    ctx = request.app.state.host_service_context
    token = request.headers.get("x-skill-token", "")
    try:
        caller, stamp = await _authenticated(ctx, token, "transcribe_audio")
        await _bounded_host_read(ctx, lambda: _caller_origin(ctx.data_dir, caller, request.headers))
    except (HostServiceAuthError, PresenceBindingError, ValueError):
        return _error("caller_not_authorized", 403)
    if not ctx.rate_limiter.allow(f"{caller}:media_transcribe"):
        return _error("media_rate_limited", 429)
    name = str(request.path_params.get("target_skill") or "")
    try:
        target = await _bounded_host_read(ctx, lambda: _target(ctx.data_dir, name))
    except (OSError, ValueError):
        return _error("target_not_ready", 409)
    mime = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    limit = min(VOICE_INPUT_MAX_BYTES, target["max_bytes"])
    if mime not in target["mime_types"]:
        return _error("media_type_unsupported", 415)
    length = request.headers.get("content-length", "")
    if length and (not length.isdecimal() or int(length) > limit):
        return _error("media_too_large", 413)
    lane = f"{caller}:media_transcribe"
    if not ctx._enter_inflight(lane, limit=2):
        return _error("media_busy", 429)
    held = True
    try:
        async def read_audio():
            audio = bytearray()
            async for chunk in request.stream():
                if len(audio) + len(chunk) > limit:
                    raise ValueError("media_too_large")
                audio.extend(chunk)
            return bytes(audio)

        try:
            audio = await asyncio.wait_for(read_audio(), timeout=30)
        except asyncio.TimeoutError:
            return _error("media_read_timeout", 408)
        except ValueError:
            return _error("media_too_large", 413)
        if not audio:
            return _error("media_empty", 400)
        operation = asyncio.create_task(_dispatch(request, ctx.data_dir, name, target, audio, mime))
        operation.add_done_callback(lambda _done: ctx._leave_inflight(lane))
        held = False  # The operation owns the slot even after HTTP disconnect.
    finally:
        if held:
            ctx._leave_inflight(lane)
    try:
        response = await asyncio.shield(operation)
        renewed_caller, renewed_stamp = await _authenticated(ctx, token, "transcribe_audio")
        renewed_target = await _bounded_host_read(ctx, lambda: _target(ctx.data_dir, name))
        await _bounded_host_read(ctx, lambda: _caller_origin(ctx.data_dir, caller, request.headers))
        state = await _bounded_host_read(ctx, lambda: runtime_state_for_skill_name(name, ctx.data_dir))
        if (renewed_caller != caller or renewed_stamp.get("content_hash") != stamp.get("content_hash")
                or renewed_target != target or not state.get("live_loaded") or not state.get("desired_live")):
            return _error("media_authority_changed", 409)
    except (HostServiceAuthError, PresenceBindingError, OSError, ValueError):
        return _error("media_authority_changed", 409)
    except Exception:
        return _error("media_target_failed", 502)
    status, body = response
    if status != 200 or len(body) > 16384:
        return _error("media_target_unavailable", 502)
    try:
        data = json.loads(body)
        text = data.get("text") if isinstance(data, dict) else None
        if text == "" and data.get("status") == "no_speech":
            return JSONResponse({"ok": True, "text": "", "status": "no_speech"})
        if not isinstance(text, str) or not 0 < len(text.strip()) <= 8000:
            raise ValueError("invalid transcription result")
    except (UnicodeError, ValueError):
        return _error("media_result_invalid", 502)
    return JSONResponse({"ok": True, "text": text.strip()})

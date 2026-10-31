"""Reviewed companions attach as meeting peers under a call-scoped direction grant.

A skill token never admits an owner turn. Only place admission may create a new
skill-opened call, and opening one is a Presence turn: it needs the connector's
``presence`` grant beside ``call_peer``, as ``/presence/turn`` does. Ordinary
attachments name an existing hub and need ``call_peer`` alone. ``binding_id``
names the connected place (its id or a migrated binding id); without it the
source event's provider facts resolve the place.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from ouroboros import call_protocol as cp
from ouroboros.call_peers import CallPeerError, get_hub
from ouroboros.contracts.realtime import RealtimeSetupError
from ouroboros.gateway.call_contracts import HostCallOpenRequest
from ouroboros.gateway.call_ws import read_call_body, refusal, serve_peer
from ouroboros.gateway.host_service import HostServiceAuthError, _authenticated, _bounded_host_read
from ouroboros.gateway.schema import validate_ingress

log = logging.getLogger(__name__)


# HTTP status of a Presence call refusal; any other admission code is a conflict.
_REFUSAL_STATUS = {
    "calls_disabled": 403, "presence_origin_mismatch": 403, "presence_binding_wrong_transport": 403,
    "place_not_connected": 403, "place_silent": 403, "place_paused": 403, "place_off": 403, "place_advises": 403,
    "place_no_wake_match": 403, "place_disconnected": 403, "consciousness_allowance_exhausted": 429,
    "presence_event_invalid": 400, "presence_identity_missing": 400, "presence_call_invalid_setting": 400,
    "presence_call_model_unset": 503,
    "presence_call_hub_unavailable": 503, "presence_call_not_started": 503,
}


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.strip().encode("utf-8")).hexdigest()


def _place(ctx, skill: str, place_ref: str):
    """The connected place of this connector named by its id or a migrated binding id."""
    from ouroboros.places import load_places

    place = next((p for p in load_places(ctx.data_dir) if place_ref in (p.place_id, *p.legacy_binding_ids)), None)
    if place is None or place.connector != skill:
        raise ValueError("call_place_forbidden")
    return place


async def _admit_presence_call(ctx, skill: str, binding_id: str, payload: dict) -> str:
    """Presence owns its gate, turn and hub lifetime; the door only asks.

    The place's connector is authenticated again inside the admission, which
    returns once the call's turn has started; a refusal raises its typed reason.
    """
    from ouroboros.presence_call import open_presence_call
    from supervisor.workers import REPO_DIR, get_event_q

    opened = await _bounded_host_read(ctx, lambda: open_presence_call(
        drive_root=ctx.data_dir, repo_dir=pathlib.Path(REPO_DIR), skill_name=skill, binding_id=binding_id,
        event=payload.get("event") or {}, label=str(payload.get("label") or ""), event_queue=get_event_q()))
    return opened.call_id


async def api_host_call_open(request: Request) -> JSONResponse:
    ctx = request.app.state.host_service_context
    raw_token = request.headers.get("x-skill-token", "")
    try:
        skill, token_payload = await _authenticated(ctx, raw_token, "call_peer")
        payload = await read_call_body(request)
        if validate_ingress(payload, HostCallOpenRequest):
            raise CallPeerError("invalid_call_request")
        call_id, binding_id = payload.get("call_id", ""), payload.get("binding_id", "")
        if not call_id and not (binding_id or payload.get("event")):
            raise CallPeerError("call_or_binding_required")
        if not call_id:
            try:
                await _bounded_host_read(ctx, lambda: ctx.require_permission(skill, token_payload, "presence"))
            except HostServiceAuthError:
                return refusal("presence_forbidden", 403,
                               "Opening a Presence call needs this skill's presence grant as well as call_peer.")
        if not ctx.rate_limiter.allow(f"{skill}:call"):
            return refusal("call_rate_limited", 429)
        if binding_id:
            await _bounded_host_read(ctx, lambda: _place(ctx, skill, binding_id))
        if not call_id:
            from ouroboros.presence_call import PresenceCallRefused

            try:
                call_id = await _admit_presence_call(ctx, skill, binding_id, payload)
            except (PresenceCallRefused, RealtimeSetupError) as exc:
                code = exc.code or "presence_call_refused"
                return refusal(code, _REFUSAL_STATUS.get(code, 409), str(getattr(exc, "message", str(exc))).removeprefix(code + ": "))
        hub = get_hub(call_id)
        if hub is None or hub.closed:
            return refusal("call_not_found", 404)
        if hub.spec.opened_by == cp.OPENED_BY_SKILL:
            # An already-open Presence call retains the same transport boundary.
            place = await _bounded_host_read(ctx, lambda: _place(ctx, skill, hub.spec.presence_binding_id))
            if binding_id and binding_id not in (place.place_id, *place.legacy_binding_ids):
                return refusal("call_binding_mismatch", 403)
        elif binding_id:
            return refusal("call_binding_mismatch", 403)
        hub.grant_peer(skill, _token_digest(raw_token), mic=payload.get("mic", True),
                       speaker=payload.get("speaker", False), video=payload.get("video", False))
        return JSONResponse({"call_id": call_id, "peer_path": f"/call/{call_id}/peer"})
    except HostServiceAuthError:
        return refusal("call_peer_forbidden", 403)
    except CallPeerError as exc:
        return refusal(exc.code, 413 if exc.code == "control_too_large" else 400)
    except ImportError:
        return refusal("presence_call_admission_unavailable", 503)
    except (ValueError, KeyError):
        return refusal("call_binding_forbidden", 403)
    except Exception:
        log.exception("Host call admission failed")
        return refusal("call_admission_failed", 503)


async def ws_host_call(websocket: WebSocket) -> None:
    ctx = websocket.app.state.host_service_context
    raw_token = websocket.headers.get("x-skill-token", "")
    try:
        skill, _ = await _authenticated(ctx, raw_token, "call_peer")
        hub = get_hub(websocket.path_params["call_id"])
        if hub is None or hub.closed:
            raise CallPeerError("call_not_found")
        grant = hub.peer_grant(skill, _token_digest(raw_token))
        if hub.spec.opened_by == cp.OPENED_BY_SKILL:
            await _bounded_host_read(ctx, lambda: _place(ctx, skill, hub.spec.presence_binding_id))
    except (HostServiceAuthError, CallPeerError, ValueError, KeyError):
        await websocket.close(code=1008, reason="call_peer_forbidden")
        return
    await serve_peer(websocket, hub, role=cp.PEER_MEETING, skill=skill,
                     mic=grant.mic, speaker=grant.speaker, video=grant.video)


def host_call_routes() -> list:
    return [Route("/call/open", api_host_call_open, methods=["POST"]),
            WebSocketRoute("/call/{call_id}/peer", ws_host_call)]

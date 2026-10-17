"""Owner call admission and the common framed-socket pump for both call doors.

The enclosing NetworkAuthGate authenticates these routes exactly as it does
``/ws``. Admission is off the server event loop; the turn owns engine lifetime.
A socket owns only its peer and never turns companion context into owner text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import suppress

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from ouroboros import call_protocol as cp
from ouroboros.call_peers import CallPeerError, LiveCallHub, close_hub, get_hub
from ouroboros.gateway.call_contracts import PEER_CONTROLS, CallOpenRequest
from ouroboros.gateway.schema import validate_ingress

log = logging.getLogger(__name__)
# A transport poll, not a conversational/VAD policy. It lets an idle socket
# receive driver-thread output without tying up a worker in a blocking queue.
SOCKET_POLL_SECONDS = 0.01
SOCKET_SEND_TIMEOUT_SECONDS = 5.0


def refusal(code: str, status: int = 400, error: str = "") -> JSONResponse:
    return JSONResponse({"ok": False, "code": code, "error": error or code}, status_code=status)


async def read_call_body(request: Request) -> dict:
    """Bound HTTP control memory before decoding, including chunked bodies."""
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > cp.MAX_CONTROL_BYTES:
            raise CallPeerError("control_too_large")
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        raise CallPeerError("invalid_json") from None
    if not isinstance(value, dict):
        raise CallPeerError("invalid_call_request")
    return value


def build_call_spec(payload: dict) -> cp.CallSpec:
    """Capture configured routing at admission; an explicit voice may be empty."""
    from ouroboros.config import load_settings
    from ouroboros.llm_live import live_key_present

    settings = load_settings()
    if not live_key_present(settings):
        # Said at once, before any turn starts: calls use Google Gemini Live and need its key.
        raise CallPeerError("missing_credential_google")
    model = payload.get("model") or settings.get("OUROBOROS_CALL_MODEL_GOOGLE", "")
    if not isinstance(model, str) or not model.strip():
        raise CallPeerError("call_model_missing")
    voice = payload.get("voice", settings.get("OUROBOROS_CALL_VOICE_GOOGLE", ""))
    limit = settings.get("OUROBOROS_CALL_MAX_MINUTES")
    if limit not in (None, ""):
        try:
            if isinstance(limit, bool) or (not isinstance(limit, (str, int))):
                raise ValueError
            limit = int(limit)
            if limit <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise CallPeerError("invalid_call_max_minutes") from None
    else:
        limit = None
    return cp.CallSpec(uuid.uuid4().hex, payload["chat_id"], "google", model, voice,
                       max_minutes=limit, client_message_id=payload.get("client_message_id", ""))


def _admit_owner_call(payload: dict) -> dict:
    from ouroboros.call_peers import open_owner_hub
    from supervisor.worker_chat_lane import handle_call_direct

    spec = build_call_spec(payload)
    hub, created = open_owner_hub(spec)  # surfaces opening at once share one call, one paid session
    if not created:
        return {"call_id": hub.call_id, "ws_path": f"/ws/call/{hub.call_id}", "joined": True}
    def finished(_task_id: str, ok: bool) -> None:
        # The driver closes its own hub with its reason; a hub still open here never reached it.
        close_hub(spec.call_id, "call_not_started" if ok else "preparation_failed", action=cp.ACTION_RETRY)

    try:
        result = handle_call_direct(spec.chat_id, spec, finished)
    except Exception:
        close_hub(spec.call_id, "admission_failed")
        raise
    if not result.get("admitted"):
        close_hub(spec.call_id, result.get("reason") or "admission_refused")
        return {"refusal": result.get("reason") or "admission_refused"}
    return {"call_id": spec.call_id, "ws_path": f"/ws/call/{spec.call_id}"}


async def api_call_open(request: Request) -> JSONResponse:
    from ouroboros.gateway._helpers import run_sync_to_completion

    try:
        payload = await read_call_body(request)
        if validate_ingress(payload, CallOpenRequest):
            raise CallPeerError("invalid_call_request")
        result = await run_sync_to_completion(_admit_owner_call, payload)
        if "refusal" in result:
            return _owner_refusal(result["refusal"], 409)
        return JSONResponse(result)
    except CallPeerError as exc:
        return _owner_refusal(exc.code, 413 if exc.code == "control_too_large" else 400)
    except ImportError:
        return _owner_refusal("call_admission_unavailable", 503)
    except Exception:
        log.exception("Call admission failed")
        return _owner_refusal("call_admission_failed", 503)


# What the owner's surface shows for an owner-door refusal; the code stays beside it.
_OWNER_REFUSAL_WORDS = {
    "repo_writer_gate_closed": "Ouroboros is applying an update; try the call again in a moment.",
    "budget_exhausted": "The spending limit is reached; raise it in Settings to call.",
    "cost_accounting_unavailable": "Spending cannot be checked right now, so the call did not open.",
    "project_room_unavailable": "This project's room is not available for a call.",
    "missing_credential_google": "Calls need a Google AI key: add it in Settings.",
    "call_model_missing": "No Gemini Live model is set in Settings (Models, Live calls).",
    "invalid_call_max_minutes": "The call's minute limit in Settings is not a positive whole number.",
    "invalid_call_request": "The call request was malformed.",
    "control_too_large": "The call request was too large.",
}


def _owner_refusal(code: str, status: int) -> JSONResponse:
    return refusal(code, status, _OWNER_REFUSAL_WORDS.get(code, "The call could not start; the log has the details."))


def _control(text: str) -> dict:
    try:
        message = cp.decode_control(text)
    except (ValueError, UnicodeError, RecursionError):
        raise CallPeerError("invalid_call_control") from None
    shape = PEER_CONTROLS.get(message["type"])
    if shape is None or validate_ingress(message, shape):
        raise CallPeerError("invalid_call_control")
    if message["type"] == "hello" and type(message["protocol"]) is not int:
        raise CallPeerError("invalid_call_control")
    if message["type"] == "played" and (message["played_ms"] < 0 or not message["item_id"]):
        raise CallPeerError("invalid_playback_ack")
    # Unknown additive fields never cross the trust boundary as control facts.
    return {key: value for key, value in message.items() if key in shape.__annotations__}


async def serve_peer(websocket: WebSocket, hub: LiveCallHub, *, role: str,
                     skill: str = "", mic: bool = True, speaker: bool = True) -> None:
    """One shared parser/pump keeps owner and companion transport behavior equal."""
    await websocket.accept()
    peer_id = uuid.uuid4().hex
    sender = None
    reason = "disconnected"

    async def send_pending() -> None:
        try:
            while True:
                message = hub.next_outgoing(peer_id)
                if isinstance(message, bytes):
                    await asyncio.wait_for(websocket.send_bytes(message), SOCKET_SEND_TIMEOUT_SECONDS)
                elif isinstance(message, str):
                    await asyncio.wait_for(websocket.send_text(message), SOCKET_SEND_TIMEOUT_SECONDS)
                elif hub.closed:
                    await websocket.close(code=1000)
                    return
                else:
                    await asyncio.sleep(SOCKET_POLL_SECONDS)
        except asyncio.TimeoutError:
            await websocket.close(code=1008, reason="peer_send_timeout")
        except CallPeerError as exc:
            await websocket.close(code=1008, reason=exc.code)

    # One receive stays pending across poll ticks: cancelling a receive on a timeout can
    # drop a message it has already taken off the socket (asyncio.wait_for on 3.10).
    receiving = None
    try:
        while True:
            if receiving is None:
                receiving = asyncio.ensure_future(websocket.receive())
            done, _ = await asyncio.wait({receiving}, timeout=SOCKET_POLL_SECONDS)
            if not done:
                if sender is None and hub.closed:
                    reason = "call_closed"
                    break
                if sender is not None and sender.done():
                    await sender
                    break
                continue
            event, receiving = receiving.result(), None
            if event["type"] == "websocket.disconnect":
                break
            if event.get("bytes") is not None:
                if sender is None:
                    raise CallPeerError("hello_required")
                try:
                    frame = cp.decode_frame(event["bytes"])
                except ValueError:
                    raise CallPeerError("invalid_audio_frame") from None
                hub.receive_audio(peer_id, frame)
                continue
            message = _control(event.get("text") or "")
            if sender is None:
                if message["type"] != "hello":
                    raise CallPeerError("hello_required")
                if message["role"] != role:
                    raise CallPeerError("peer_role_forbidden")
                if (message["mic"] and not mic) or (message["speaker"] and not speaker):
                    raise CallPeerError("audio_direction_forbidden")
                hub.join(cp.PeerInfo(peer_id, role, message["mic"], message["speaker"],
                                     message.get("label", ""), skill), message["rate"])
                sender = asyncio.create_task(send_pending())
            else:
                if message["type"] == "hello":
                    raise CallPeerError("peer_already_joined")
                hub.receive_control(peer_id, message)
                if message["type"] == "bye":
                    reason = message.get("reason", "")
                    break
    except (WebSocketDisconnect, RuntimeError):
        pass
    except CallPeerError as exc:
        reason = exc.code
        await websocket.close(code=1008, reason=reason)
    finally:
        if receiving is not None:
            receiving.cancel()
            with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                await receiving
        if sender is not None:
            sender.cancel()
            with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                await sender
        hub.leave(peer_id, reason)
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close(code=1000)


async def ws_call(websocket: WebSocket) -> None:
    hub = get_hub(websocket.path_params["call_id"])
    if hub is None or hub.closed:
        await websocket.close(code=1008, reason="call_not_found")
        return
    if hub.spec.opened_by != cp.OPENED_BY_OWNER:
        await websocket.close(code=1008, reason="owner_call_required")
        return
    await serve_peer(websocket, hub, role=cp.PEER_OWNER)


def call_routes() -> list:
    """The owner surface's call door and audio socket, mounted beside the JSON ``/ws``."""
    return [Route("/api/call/open", endpoint=api_call_open, methods=["POST"]),
            WebSocketRoute("/ws/call/{call_id}", endpoint=ws_call)]

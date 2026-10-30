"""Authenticated owner admission and the preserved peer transport, shared with companions."""
from __future__ import annotations
import asyncio
import json
import uuid
from contextlib import suppress
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
from ouroboros import call_protocol as cp
from ouroboros.call_peers import CallPeerError, LiveCallHub, close_hub, get_hub
from ouroboros.gateway.call_contracts import PEER_CONTROLS
from ouroboros.gateway.schema import validate_ingress
SOCKET_POLL_SECONDS = .01
SOCKET_SEND_TIMEOUT_SECONDS = 5.0

def refusal(code: str, status: int = 400, error: str = "") -> JSONResponse:
    return JSONResponse({"ok": False, "code": code, "error": error or code}, status_code=status)


async def read_call_body(request) -> dict:
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
                     skill: str = "", mic: bool = True, speaker: bool = True, video: bool = False) -> None:
    """One shared parser/pump keeps owner and companion transport behavior equal."""
    await websocket.accept()
    peer_id = uuid.uuid4().hex
    sender = None
    reason = "disconnected"
    ending = asyncio.Event()

    async def send_pending() -> None:
        try:
            while not ending.is_set():
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
                    raise CallPeerError("invalid_call_frame") from None
                if isinstance(frame, cp.VideoFrame):
                    hub.receive_video(peer_id, frame)
                else:
                    while True:
                        try:
                            hub.receive_audio(peer_id, frame)
                            break
                        except CallPeerError as exc:
                            if exc.code != "input_backpressure":
                                raise
                            await asyncio.sleep(SOCKET_POLL_SECONDS)
                continue
            message = _control(event.get("text") or "")
            if sender is None:
                if message["type"] != "hello":
                    raise CallPeerError("hello_required")
                if message["role"] != role:
                    raise CallPeerError("peer_role_forbidden")
                if (message["mic"] and not mic) or (message["speaker"] and not speaker):
                    raise CallPeerError("audio_direction_forbidden")
                if message.get("video", False) and not video:
                    raise CallPeerError("video_direction_forbidden")
                hub.join(cp.PeerInfo(peer_id, role, message["mic"], message["speaker"],
                                     message.get("label", ""), skill, message.get("video", False)), message["rate"])
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
        ending.set()
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


def _admit_owner_call(payload):
    from ouroboros.call_peers import open_owner_hub
    from ouroboros.contracts.realtime import RealtimeSetupError
    from ouroboros.live_conversation import LiveConversationTurn, feature_configuration
    from supervisor.message_bus import try_get_bridge

    if (set(payload) - {"chat_id", "client_message_id", "feature"}
            or type(payload.get("chat_id")) is not int
            or not isinstance(payload.get("client_message_id"), str) or not payload["client_message_id"]):
        raise RealtimeSetupError("invalid_call_request", "chat_id and client_message_id are required")
    with feature_configuration(payload.get("feature", "live_conversation:call")) as (feature, config):
        from ouroboros.live_conversation import call_offer
        offer = call_offer(config)
        if not offer["offered"]:
            raise RealtimeSetupError(offer["reason"], offer["detail"])
        from ouroboros.realtime_setup import loaded_provider
        model = loaded_provider(config.provider)["catalog"].model(config.model)
        call_id = uuid.uuid4().hex
        spec = cp.CallSpec(call_id, payload["chat_id"], config.provider, model.id, config.voice,
                           client_message_id=f"call:{call_id}:open")
        bridge = try_get_bridge()
        if bridge is None:
            # No supervisor in this server generation (still starting, or no model is set up): nothing can run the turn.
            raise RealtimeSetupError("calls_unavailable", "Ouroboros isn’t ready to take calls yet. "
                                     "Try again in a moment, or connect a model in Settings.")
        turn = LiveConversationTurn(feature, config, spec, publish=bridge.broadcast)
        hub, created = open_owner_hub(spec)
        if not created:
            return {"call_id": hub.call_id, "ws_path": f"/ws/call/{hub.call_id}", "joined": True,
                    "vision": turn.supports_images}
        metadata = {"execution_transport": turn.request, "host_fact": {"kind": "call_open", "call_id": spec.call_id}}

        def accepted(text, **message):
            extra = dict(message.get("task_metadata") or {})
            extra["_realtime_turn"] = turn
            try:
                bridge.enqueue_local_message(text, **{**message, "task_metadata": extra})
            except BaseException:
                close_hub(spec.call_id, "admission_failed")
                raise

        try:
            bridge.ui_send(turn.opening, chat_id=spec.chat_id,
                client_message_id=spec.client_message_id, task_metadata=metadata, dispatch=accepted)
        except BaseException:
            close_hub(spec.call_id, "admission_failed")
            raise
        return {"call_id": spec.call_id, "ws_path": f"/ws/call/{spec.call_id}", "vision": turn.supports_images}


async def api_call_open(request):
    from ouroboros.contracts.realtime import RealtimeSetupError
    from ouroboros.gateway._helpers import run_sync_to_completion

    try:
        payload = await read_call_body(request)
        return JSONResponse(await run_sync_to_completion(_admit_owner_call, payload))
    except (RealtimeSetupError, CallPeerError) as exc:
        # `error` is the owner's sentence; the code travels in its own field.
        return refusal(exc.code, 409, str(exc).removeprefix(f"{exc.code}: "))


async def ws_call(websocket):
    hub = get_hub(websocket.path_params["call_id"])
    if hub is None or hub.closed or hub.spec.opened_by != cp.OPENED_BY_OWNER:
        await websocket.close(code=1008, reason="call_not_found")
        return
    await serve_peer(websocket, hub, role=cp.PEER_OWNER, video=True)


def call_routes():
    return [Route("/api/call/open", api_call_open, methods=["POST"]),
            WebSocketRoute("/ws/call/{call_id}", ws_call)]

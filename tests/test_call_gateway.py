"""Exercise both call doors over their real Starlette transports and auth gates."""

import json
import uuid

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ouroboros import call_protocol as cp
from ouroboros.call_peers import close_hub, get_hub, open_hub
from ouroboros.gateway import call_ws, host_call
from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.server_auth import NetworkAuthGate
from ouroboros.skill_loader import (
    SkillReviewState,
    compute_content_hash,
    requested_skill_permissions,
    save_enabled,
    save_review_state,
    save_skill_grants,
)
from ouroboros.utils import atomic_write_json

# An upper bound on waiting for an event that is already on its way; a loaded test
# machine can take more than a second to hand it over, and a prompt event returns at once.
EVENT_WAIT_S = 10


def seed_skill(root, *, name="meeting", token="skill-token", granted=True, reviewed=True,
               permissions=("call_peer",), granted_permissions=None):
    declared = list(permissions)
    directory = root / "skills" / "external" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Audio companion\nversion: 1\n"
        f"type: extension\nentry: plugin.py\npermissions: [{', '.join(declared + ['companion_process'])}]\n"
        "---\n# Audio\n")
    (directory / "plugin.py").write_text("def register(api): pass\n")
    content_hash = compute_content_hash(directory, manifest_entry="plugin.py")
    save_enabled(root, name, True)
    save_review_state(root, name, SkillReviewState(status="pass" if reviewed else "fail", content_hash=content_hash))
    grants = list(declared if granted_permissions is None else granted_permissions) if granted else []
    save_skill_grants(root, name, [], content_hash=content_hash, requested_keys=[],
                      granted_permissions=grants, requested_permissions=requested_skill_permissions(declared))
    atomic_write_json(root / "state" / "skills" / name / "auth_token.json",
                      {"token": token, "content_hash": content_hash})


@pytest.fixture
def hub():
    spec = cp.CallSpec(uuid.uuid4().hex, 1, "google", "test-model")
    result = open_hub(spec)
    yield result
    close_hub(spec.call_id)


@pytest.fixture
def owner(monkeypatch):
    from ouroboros import server_auth

    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "test-password")
    app = Starlette(routes=[Route("/api/call/open", call_ws.api_call_open, methods=["POST"]),
                            WebSocketRoute("/ws/call/{call_id}", call_ws.ws_call)],
                    middleware=[Middleware(NetworkAuthGate)])
    with TestClient(app, headers={"x-ouroboros-password": "test-password"}) as client:
        yield client


@pytest.fixture(params=["owner", "host"])
def door(request, hub, owner, tmp_path):
    if request.param == "owner":
        yield owner, f"/ws/call/{hub.call_id}", "owner", hub
        return
    seed_skill(tmp_path)
    with TestClient(create_host_service_app(tmp_path), headers={"x-skill-token": "skill-token"}) as client:
        response = client.post("/call/open", json={"call_id": hub.call_id, "role": "meeting", "speaker": True})
        assert response.status_code == 200, response.text
        yield client, response.json()["peer_path"], "meeting", hub


def hello(ws, role, **extra):
    ws.send_json({"type": "hello", "protocol": 1, "role": role, "mic": True,
                  "speaker": True, "rate": 16000, "label": "test", **extra})  # Gemini's input rate: no resampling
    assert ws.receive_json()["type"] == "state"
    assert ws.receive_json()["type"] == "flush"


def audio(epoch=0, seq=0, kind=cp.FRAME_AUDIO_IN, rate=16000):
    return cp.encode_frame(cp.AudioFrame(kind, epoch, seq, rate, 0, b"\x64\x00" * (rate // 50)))  # one 20 ms frame


def test_owner_open_captures_settings_and_closes_hub_on_finished(owner, monkeypatch):
    from ouroboros import config
    from supervisor import worker_chat_lane

    calls = []
    monkeypatch.setattr(config, "load_settings", lambda: {
        "GOOGLE_API_KEY": "test-google-key", "OUROBOROS_CALL_MODEL_GOOGLE": "live-model",
        "OUROBOROS_CALL_VOICE_GOOGLE": "configured-voice", "OUROBOROS_CALL_MAX_MINUTES": "9"})

    def admit(chat_id, spec, on_finished):
        assert get_hub(spec.call_id) is not None
        calls.append((chat_id, spec, on_finished))
        return {"admitted": True, "task_id": "turn", "reason": ""}

    monkeypatch.setattr(worker_chat_lane, "handle_call_direct", admit, raising=False)
    response = owner.post("/api/call/open", json={"chat_id": 1, "client_message_id": "message"})
    assert response.status_code == 200, response.text
    chat_id, spec, finished = calls[0]
    assert (chat_id, spec.engine, spec.model, spec.voice, spec.max_minutes, spec.client_message_id) == (
        1, "google", "live-model", "configured-voice", 9, "message")
    assert spec.opened_by == cp.OPENED_BY_OWNER
    with owner.websocket_connect(response.json()["ws_path"]) as ws:
        hello(ws, "owner")
    finished("turn", True)
    assert get_hub(spec.call_id) is None


def test_owner_admission_refusal_removes_hub(owner, monkeypatch):
    from ouroboros import config
    from supervisor import worker_chat_lane

    monkeypatch.setattr(config, "load_settings", lambda: {"GOOGLE_API_KEY": "test-google-key", "OUROBOROS_CALL_MODEL_GOOGLE": "test"})
    seen = []

    def admit(chat_id, spec, callback):
        seen.append(spec.call_id)
        return {"admitted": False, "reason": "budget_exhausted"}

    monkeypatch.setattr(worker_chat_lane, "handle_call_direct", admit, raising=False)
    response = owner.post("/api/call/open", json={"chat_id": 1})
    assert response.status_code == 409 and response.json()["code"] == "budget_exhausted"
    assert response.json()["error"] == "The spending limit is reached; raise it in Settings to call."
    assert get_hub(seen[0]) is None


def test_owner_open_honours_a_chosen_model_and_a_second_surface_joins_the_rooms_call(owner, monkeypatch):
    from ouroboros import config
    from supervisor import worker_chat_lane

    monkeypatch.setattr(config, "load_settings", lambda: {"GOOGLE_API_KEY": "test-google-key", "OUROBOROS_CALL_MODEL_GOOGLE": "settings-model"})
    admitted = []

    def admit(chat_id, spec, on_finished):
        admitted.append(spec)
        return {"admitted": True, "task_id": "turn", "reason": ""}

    monkeypatch.setattr(worker_chat_lane, "handle_call_direct", admit, raising=False)
    first = owner.post("/api/call/open", json={"chat_id": 7, "model": "chosen-model"})
    try:
        assert first.status_code == 200 and admitted[0].model == "chosen-model"
        second = owner.post("/api/call/open", json={"chat_id": 7, "model": "another-model"})
        assert second.status_code == 200 and second.json() == {
            "call_id": admitted[0].call_id, "ws_path": f"/ws/call/{admitted[0].call_id}", "joined": True}
        assert len(admitted) == 1  # one paid call per room; the second surface joined it
        other_room = owner.post("/api/call/open", json={"chat_id": 8})
        assert other_room.status_code == 200 and len(admitted) == 2 and admitted[1].model == "settings-model"
    finally:
        for spec in admitted:
            close_hub(spec.call_id)


def test_two_surfaces_opening_a_room_at_once_share_one_call(monkeypatch):
    import threading

    from ouroboros import config
    from ouroboros.gateway import call_ws
    from supervisor import worker_chat_lane

    monkeypatch.setattr(config, "load_settings", lambda: {"GOOGLE_API_KEY": "test-google-key", "OUROBOROS_CALL_MODEL_GOOGLE": "settings-model"})
    admitted, both_built = [], threading.Barrier(2, timeout=5)
    build = call_ws.build_call_spec

    def build_together(payload):  # both requests finish preparing before either opens a hub
        spec = build(payload)
        both_built.wait()
        return spec

    def admit(chat_id, spec, on_finished):
        admitted.append(spec)
        return {"admitted": True, "task_id": "turn", "reason": ""}

    monkeypatch.setattr(call_ws, "build_call_spec", build_together)
    monkeypatch.setattr(worker_chat_lane, "handle_call_direct", admit, raising=False)
    results: list = []
    threads = [threading.Thread(target=lambda: results.append(call_ws._admit_owner_call({"chat_id": 9})))
               for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert len(admitted) == 1  # one paid session for the room
        assert {result["call_id"] for result in results} == {admitted[0].call_id}
        assert sorted(bool(result.get("joined")) for result in results) == [False, True]
    finally:
        for spec in admitted:
            close_hub(spec.call_id)


@pytest.mark.parametrize("payload", [{}, [], {"chat_id": True}, {"chat_id": "1"},
                                     {"chat_id": 1, "engine": "other"}, {"chat_id": 1, "voice": False}])
def test_owner_rejects_invalid_open_request(owner, payload):
    response = owner.post("/api/call/open", json=payload)
    assert response.status_code == 400 and response.json()["ok"] is False


def test_owner_network_auth_is_the_same_gate_for_http_and_socket(owner, hub):
    owner.headers.pop("x-ouroboros-password")
    assert owner.post("/api/call/open", json={"chat_id": 1}).status_code == 401
    with pytest.raises(WebSocketDisconnect) as exc:
        with owner.websocket_connect(f"/ws/call/{hub.call_id}"):
            pass
    assert exc.value.code == 4401
    owner.headers["x-ouroboros-password"] = "test-password"
    with owner.websocket_connect(f"/ws/call/{hub.call_id}") as ws:
        hello(ws, "owner")


def test_both_doors_audio_controls_output_flush_and_leave(door):
    client, path, role, hub = door
    with client.websocket_connect(path) as ws:
        hello(ws, role)
        event = hub.next_event(EVENT_WAIT_S)
        assert isinstance(event, cp.PeerJoined) and event.peer.role == role
        ws.send_bytes(audio())
        assert hub.next_event(EVENT_WAIT_S) == cp.PeerAudio(event.peer.peer_id, role, b"\x64\x00" * 320)
        for control in [{"type": "played", "item_id": "item", "played_ms": 20},
                        {"type": "mute", "muted": True},
                        {"type": "context", "text": "A slide is visible", "source": "screen"},
                        {"type": "participant", "label": "Guest", "speaking": True}]:
            ws.send_json(control)
            received = hub.next_event(EVENT_WAIT_S)
            assert isinstance(received, cp.PeerControl) and received.role == role
            assert received.message == control
        hub.send_caption("ouroboros", "Hello", True, "item")
        assert ws.receive_json()["item_id"] == "item"
        hub.play(b"\x64\x00" * 480, "item")
        assert cp.decode_frame(ws.receive_bytes()).kind == cp.FRAME_AUDIO_OUT
        epoch = hub.flush()
        assert ws.receive_json() == {"type": "flush", "epoch": epoch}
        ws.send_json({"type": "bye", "reason": "finished"})
        assert hub.next_event(EVENT_WAIT_S).message["type"] == "bye"
        assert isinstance(hub.next_event(EVENT_WAIT_S), cp.PeerLeft)
    assert hub.peers() == ()


@pytest.mark.parametrize("message", ["not json", json.dumps({"type": "state", "state": "live"}),
    json.dumps({"type": "mute", "muted": "false"}),
    json.dumps({"type": "played", "item_id": "item", "played_ms": -1}),
    json.dumps({"type": "context", "text": "x" * cp.MAX_CONTROL_BYTES, "source": "peer"})])
def test_both_doors_reject_invalid_control(door, message):
    client, path, role, hub = door
    with client.websocket_connect(path) as ws:
        hello(ws, role)
        ws.send_text(message)
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 1008
    assert hub.peers() == ()


@pytest.mark.parametrize("frame", [b"bad", audio(kind=cp.FRAME_AUDIO_OUT), audio(rate=24000),  # not the hello's rate
                                    audio()[:cp.HEADER_SIZE] + b"\0" * (cp.MAX_FRAME_PAYLOAD + 2)])
def test_both_doors_refuse_bad_binary_frames(door, frame):
    client, path, role, hub = door
    with client.websocket_connect(path) as ws:
        hello(ws, role)
        ws.send_bytes(frame)
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 1008
    assert hub.peers() == ()


def test_both_doors_discard_stale_epoch_but_accept_current(door):
    client, path, role, hub = door
    with client.websocket_connect(path) as ws:
        hello(ws, role)
        hub.next_event(EVENT_WAIT_S)
        hub.flush()
        ws.receive_json()
        ws.send_bytes(audio(epoch=0))
        ws.send_bytes(audio(epoch=1, seq=1))
        received = hub.next_event(EVENT_WAIT_S)
        assert isinstance(received, cp.PeerAudio)
        assert hub.stats()["stale_frames"] == 1
        assert hub.next_event(0) is None


def test_both_doors_refuse_wrong_call_and_hello_identity(door):
    client, path, role, hub = door
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(path.replace(hub.call_id, "not-admitted")):
            pass
    with client.websocket_connect(path) as ws:
        ws.send_json({"type": "hello", "protocol": 1, "role": "meeting" if role == "owner" else "owner",
                      "mic": True, "speaker": True, "rate": 24000})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
    assert hub.peers() == ()


@pytest.mark.parametrize("mode", ["missing", "invalid", "ungranted", "unreviewed", "disabled", "stale"])
def test_host_requires_current_review_token_and_explicit_call_peer_grant(tmp_path, hub, mode, monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    seed_skill(tmp_path, granted=mode != "ungranted", reviewed=mode != "unreviewed")
    if mode == "disabled":
        save_enabled(tmp_path, "meeting", False)
    if mode == "stale":
        token_path = tmp_path / "state" / "skills" / "meeting" / "auth_token.json"
        atomic_write_json(token_path, {"token": "skill-token", "content_hash": "old"})
    token = "" if mode == "missing" else "incorrect" if mode == "invalid" else "skill-token"
    with TestClient(create_host_service_app(tmp_path), headers={"x-skill-token": token}) as client:
        response = client.post("/call/open", json={"call_id": hub.call_id, "role": "meeting"})
        assert response.status_code == 403 and response.json()["code"] == "call_peer_forbidden"
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/call/{hub.call_id}/peer"):
                pass


def test_host_grant_is_bound_to_call_token_and_directions(tmp_path, hub):
    seed_skill(tmp_path)
    other = open_hub(cp.CallSpec(uuid.uuid4().hex, 1, "google", "test"))
    try:
        with TestClient(create_host_service_app(tmp_path), headers={"x-skill-token": "skill-token"}) as client:
            path = f"/call/{hub.call_id}/peer"
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(path):
                    pass  # The token/grant alone does not grant this call until POST admission.
            assert client.post("/call/open", json={"call_id": hub.call_id, "role": "owner"}).status_code == 400
            response = client.post("/call/open", json={"call_id": hub.call_id, "role": "meeting"})
            assert response.status_code == 200
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(f"/call/{other.call_id}/peer"):
                    pass
            with client.websocket_connect(path) as ws:
                ws.send_json({"type": "hello", "protocol": 1, "role": "meeting", "mic": True,
                              "speaker": True, "rate": 24000})
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_json()
            with client.websocket_connect(path) as ws:
                hello(ws, "meeting", speaker=False)
            rotated = tmp_path / "state" / "skills" / "meeting" / "auth_token.json"
            payload = json.loads(rotated.read_text())
            atomic_write_json(rotated, {**payload, "token": "new-token"})
            client.headers["x-skill-token"] = "new-token"
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(path):
                    pass
    finally:
        close_hub(other.call_id)


def test_presence_admission_uses_binding_and_never_owner_entry(tmp_path, monkeypatch):
    from supervisor import worker_chat_lane

    seed_skill(tmp_path, permissions=("call_peer", "presence"))
    admitted = []
    monkeypatch.setattr(host_call, "_binding", lambda ctx, skill, binding: admitted.append((skill, binding)))
    monkeypatch.setattr(worker_chat_lane, "handle_call_direct",
                        lambda *a: pytest.fail("owner admission"), raising=False)
    value = cp.CallSpec(uuid.uuid4().hex, 1234, "google", "test", opened_by=cp.OPENED_BY_SKILL,
                        presence_binding_id="binding")

    asked = []

    async def admit(ctx, skill, binding, payload):
        asked.append((skill, binding, payload.get("event"), payload.get("label")))
        open_hub(value)
        return value.call_id

    monkeypatch.setattr(host_call, "_admit_presence_call", admit)
    event = {"source_event_id": "zoom-join-1", "provider": "zoom", "text": "Joined the Zoom meeting"}
    try:
        with TestClient(create_host_service_app(tmp_path), headers={"x-skill-token": "skill-token"}) as client:
            response = client.post("/call/open", json={"binding_id": "binding", "role": "meeting",
                                                       "event": event, "label": "Weekly sync"})
            assert response.status_code == 200, response.text
            with client.websocket_connect(response.json()["peer_path"]) as ws:
                hello(ws, "meeting", speaker=False)
            assert admitted and all(row == ("meeting", "binding") for row in admitted)
            assert asked == [("meeting", "binding", event, "Weekly sync")]
    finally:
        close_hub(value.call_id)


@pytest.mark.parametrize("code,status", [("presence_call_busy", 409), ("presence_call_not_enabled", 403),
                                         ("presence_call_model_unset", 503)])
def test_a_presence_refusal_reaches_the_companion_in_plain_words(tmp_path, monkeypatch, code, status):
    from ouroboros.presence_call import PresenceCallRefused

    seed_skill(tmp_path, permissions=("call_peer", "presence"))
    monkeypatch.setattr(host_call, "_binding", lambda ctx, skill, binding: None)

    async def refuse(ctx, skill, binding, payload):
        raise PresenceCallRefused(code, "The call did not open; the reason is in the code.")

    monkeypatch.setattr(host_call, "_admit_presence_call", refuse)
    with TestClient(create_host_service_app(tmp_path), headers={"x-skill-token": "skill-token"}) as client:
        response = client.post("/call/open", json={"binding_id": "binding", "role": "meeting", "event": {}})
    assert response.status_code == status
    assert response.json() == {"ok": False, "code": code, "error": "The call did not open; the reason is in the code."}


@pytest.mark.parametrize("granted", [("call_peer",), ("presence",), ("call_peer", "presence")])
def test_opening_a_presence_call_needs_presence_beside_call_peer_and_attaching_needs_call_peer(
        tmp_path, hub, monkeypatch, granted):
    """Opening a Presence call is a Presence turn (the grant /presence/turn requires); joining an
    existing call, a Presence call included, needs call_peer alone."""
    seed_skill(tmp_path, permissions=("call_peer", "presence"), granted_permissions=granted)
    monkeypatch.setattr(host_call, "_binding", lambda ctx, skill, binding: None)
    presence_call = cp.CallSpec(uuid.uuid4().hex, 1234, "google", "test", opened_by=cp.OPENED_BY_SKILL,
                                presence_binding_id="binding")
    admitted = []

    async def admit(ctx, skill, binding, payload):
        admitted.append(binding)
        return open_hub(presence_call).call_id

    monkeypatch.setattr(host_call, "_admit_presence_call", admit)
    try:
        with TestClient(create_host_service_app(tmp_path), headers={"x-skill-token": "skill-token"}) as client:
            opened = client.post("/call/open", json={"binding_id": "binding", "role": "meeting", "event": {}})
            owner_call = client.post("/call/open", json={"call_id": hub.call_id, "role": "meeting"})
            open_hub(presence_call)
            running = client.post("/call/open", json={"call_id": presence_call.call_id, "role": "meeting"})
    finally:
        close_hub(presence_call.call_id)
    if "call_peer" not in granted:
        for response in (opened, owner_call, running):
            assert response.status_code == 403 and response.json()["code"] == "call_peer_forbidden"
        assert admitted == []
        return
    assert owner_call.status_code == running.status_code == 200, (owner_call.text, running.text)
    if "presence" in granted:
        assert opened.status_code == 200 and admitted == ["binding"]
    else:
        assert opened.status_code == 403 and admitted == []
        assert opened.json() == {"ok": False, "code": "presence_forbidden", "error": (
            "Opening a Presence call needs this skill's presence grant as well as call_peer.")}


def test_published_call_contract_shapes_and_permission_twin():
    from ouroboros.contracts.skill_manifest import VALID_SKILL_PERMISSIONS, parse_skill_manifest_text
    from ouroboros.gateway.call_contracts import CallSocketContract
    from ouroboros.gateway.schema import contract_document, validate_ingress
    from ouroboros.skill_loader import _GRANTABLE_SKILL_PERMISSIONS

    assert "call_peer" in VALID_SKILL_PERMISSIONS & _GRANTABLE_SKILL_PERMISSIONS
    assert requested_skill_permissions(["call_peer"]) == ["call_peer"]
    manifest = parse_skill_manifest_text("---\nname: meeting\ndescription: Audio\npermissions: [call_peer]\n---\n")
    assert not manifest.validate()
    doc = contract_document()
    assert not validate_ingress(doc["call"], CallSocketContract)
    assert set(doc["call"]["peer_messages"]) == set(cp.PEER_MESSAGE_TYPES)
    assert set(doc["call"]["server_messages"]) == set(cp.SERVER_MESSAGE_TYPES)
    assert doc["ws"]["path"] == "/ws"
    assert not set(cp.PEER_MESSAGE_TYPES) & {row["type"] for row in doc["ws"]["messages"]}
    assert doc["call"]["audio_header_bytes"] == cp.HEADER_SIZE


def test_both_doors_close_with_hub_before_hello(door):
    client, path, role, hub = door
    with client.websocket_connect(path) as ws:
        hub.close("finished")
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()
    assert not hub.peers()


def test_both_doors_close_after_terminal_state(door):
    client, path, role, hub = door
    with client.websocket_connect(path) as ws:
        hello(ws, role)
        hub.close("finished")
        assert ws.receive_json() == {"type": "flush", "epoch": hub.epoch}
        assert ws.receive_json() == {"type": "state", "state": "closed", "reason": "finished", "engine": "google"}
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()
    assert not hub.peers()


def test_both_doors_require_hello_and_reject_repeated_hello(door):
    client, path, role, hub = door
    with client.websocket_connect(path) as ws:
        ws.send_bytes(audio())
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()
    with client.websocket_connect(path) as ws:
        hello(ws, role)
        ws.send_json({"type": "hello", "protocol": 1, "role": role, "mic": True,
                      "speaker": True, "rate": 24000})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()
    assert not hub.peers()


def test_http_body_bound_has_allow_and_refusal_paths(owner, hub, tmp_path):
    seed_skill(tmp_path)
    assert owner.post("/api/call/open", content=b"{" * (cp.MAX_CONTROL_BYTES + 1)).status_code == 413
    with TestClient(create_host_service_app(tmp_path), headers={"x-skill-token": "skill-token"}) as client:
        assert client.post("/call/open", content=b"{" * (cp.MAX_CONTROL_BYTES + 1)).status_code == 413
        assert client.post("/call/open", json={"role": "meeting", "call_id": hub.call_id}).status_code == 200
        assert client.post("/call/open", json={"role": "meeting", "call_id": "missing"}).status_code == 404


def test_host_advisory_review_keeps_existing_policy(tmp_path, hub, monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    seed_skill(tmp_path, reviewed=False)
    with TestClient(create_host_service_app(tmp_path), headers={"x-skill-token": "skill-token"}) as client:
        assert client.post("/call/open", json={"role": "meeting", "call_id": hub.call_id}).status_code == 200


def test_presence_binding_failures_cannot_attach_or_open_owner(tmp_path, hub, monkeypatch):
    seed_skill(tmp_path)
    app = create_host_service_app(tmp_path)
    with TestClient(app, headers={"x-skill-token": "skill-token"}) as client:
        # The real binding resolver rejects an unknown/foreign binding before admission.
        response = client.post("/call/open", json={"role": "meeting", "binding_id": "unknown"})
        assert response.status_code == 403
        monkeypatch.setattr(host_call, "_binding", lambda *args: None)
        response = client.post("/call/open", json={"role": "meeting", "binding_id": "b", "call_id": hub.call_id})
        assert response.status_code == 403 and response.json()["code"] == "call_binding_mismatch"


def test_owner_door_refuses_presence_call(owner):
    value = cp.CallSpec(uuid.uuid4().hex, 1234, "google", "test", opened_by="skill", presence_binding_id="b")
    open_hub(value)
    try:
        with pytest.raises(WebSocketDisconnect):
            with owner.websocket_connect(f"/ws/call/{value.call_id}"):
                pass
    finally:
        close_hub(value.call_id)


def test_owner_unsupported_settings_and_admission_crash_are_typed(owner, monkeypatch):
    from ouroboros import config
    from supervisor import worker_chat_lane

    settings = {"GOOGLE_API_KEY": "test-google-key", "OUROBOROS_CALL_MODEL_GOOGLE": "test", "OUROBOROS_CALL_MAX_MINUTES": "-2"}
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setattr(worker_chat_lane, "handle_call_direct", lambda *args: None, raising=False)
    response = owner.post("/api/call/open", json={"chat_id": 1})
    assert response.json()["code"] == "invalid_call_max_minutes"
    settings["OUROBOROS_CALL_MAX_MINUTES"] = ""
    observed = []

    def broken(chat_id, spec, callback):
        observed.append(spec.call_id)
        raise RuntimeError("not started")

    monkeypatch.setattr(worker_chat_lane, "handle_call_direct", broken)
    response = owner.post("/api/call/open", json={"chat_id": 1})
    assert response.status_code == 503 and response.json()["code"] == "call_admission_failed"
    assert get_hub(observed[0]) is None


def test_a_turn_that_ends_before_its_driver_closes_the_hub_with_a_reason(owner, monkeypatch):
    from ouroboros import config
    from supervisor import worker_chat_lane

    calls = []
    monkeypatch.setattr(config, "load_settings", lambda: {"GOOGLE_API_KEY": "g", "OUROBOROS_CALL_MODEL_GOOGLE": "m"})
    monkeypatch.setattr(worker_chat_lane, "handle_call_direct",
                        lambda chat_id, spec, on_finished: calls.append((spec, on_finished)) or {"admitted": True},
                        raising=False)
    response = owner.post("/api/call/open", json={"chat_id": 1})
    spec, finished = calls[0]
    with owner.websocket_connect(response.json()["ws_path"]) as ws:
        hello(ws, "owner")
        finished("turn", False)  # the turn failed while preparing: no driver ever closed the hub
        assert ws.receive_json()["type"] == "flush"
        closed = ws.receive_json()
    assert closed == {"type": "state", "state": "closed", "reason": "preparation_failed", "engine": "google",
                      "action": "retry"}
    assert get_hub(spec.call_id) is None


def test_owner_open_without_a_google_key_is_refused_at_once(owner, monkeypatch):
    from ouroboros import config
    from supervisor import worker_chat_lane

    admitted = []
    monkeypatch.setattr(worker_chat_lane, "handle_call_direct",
                        lambda chat_id, spec, on_finished: admitted.append(spec) or {"admitted": True, "task_id": "t"},
                        raising=False)
    settings = {"OUROBOROS_CALL_MODEL_GOOGLE": "g-model", "OPENAI_API_KEY": "an-openai-key-opens-nothing"}
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    response = owner.post("/api/call/open", json={"chat_id": 1})
    assert response.status_code == 400 and response.json()["code"] == "missing_credential_google"
    assert response.json()["error"] == "Calls need a Google AI key: add it in Settings." and admitted == []
    settings["GOOGLE_API_KEY"] = "g"
    response = owner.post("/api/call/open", json={"chat_id": 1})
    assert response.status_code == 200, response.text
    assert admitted[-1].engine == "google" and admitted[-1].model == "g-model"
    close_hub(admitted[-1].call_id)

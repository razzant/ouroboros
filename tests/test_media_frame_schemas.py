"""Published WS media shapes keep their field set, required payload and discriminator."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from ouroboros.gateway.schema import contract_typed_dict, validate_ingress
from tests.test_gateway_client_contract import _contract_app


_SHARED_REQUIRED = {"type", "role", "mime", "ts"}
_SHARED_OPTIONAL = set("""
    caption download_url download_url_compat content source sender_label sender_session_id
    client_message_id transport chat_id task_id project_thread row_address row_client_message_id
    reply_author_label reply_to relay reply_to_relay kind room_revisions
""".split())


@pytest.mark.parametrize(("name", "frame_type", "payload", "required_extra", "optional_extra"), [
    ("PhotoOutbound", "photo", "image_base64", {}, {"subagents_mode"}),
    ("VideoOutbound", "video", "video_base64", {}, set()),
    ("DocumentOutbound", "document", "file_base64", {"filename": "note.txt"}, {"file_ref", "size_bytes"}),
])
def test_published_media_frame_fields_requiredness_and_type(tmp_path, name, frame_type, payload,
                                                            required_extra, optional_extra):
    with TestClient(_contract_app(tmp_path)) as client:
        reply = client.get("/api/contract")
    assert reply.status_code == 200
    contract = reply.json()
    binding = next(row for row in contract["ws"]["messages"] if row["type"] == frame_type)
    assert binding == {"type": frame_type, "outbound": name}
    schema = contract["schemas"][name]
    required = _SHARED_REQUIRED | {payload} | required_extra.keys()
    assert set(schema["required"]) == required
    assert set(schema["properties"]) == required | _SHARED_OPTIONAL | optional_extra
    assert schema["properties"]["type"] == {"enum": [frame_type]}
    assert schema["properties"]["role"] == {"enum": ["user", "assistant"]}
    assert schema["properties"][payload] == {"type": "string"}

    media = {"type": frame_type, "role": "assistant", "mime": "application/octet-stream",
             "ts": "2026-11-25T12:00:00Z", payload: "YWJj", **required_extra}
    cls = contract_typed_dict(name)
    assert not validate_ingress(media, cls)
    assert validate_ingress({key: value for key, value in media.items() if key != payload}, cls) == [
        f"{payload} is required",
    ]
    assert validate_ingress({**media, "type": "chat"}, cls) == [f"type must be one of: {frame_type}"]
    assert validate_ingress({**media, payload: 42}, cls) == [f"{payload} must be a JSON string"]

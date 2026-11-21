"""HTTP and WebSocket owner sends carry the same host-resolved Reply fact."""
import asyncio
import json

import pytest

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.chat_chain import format_address, row_address
from ouroboros.chat_references import stored_reply
from ouroboros.gateway.control import api_command
from ouroboros.gateway.ws import ws_endpoint
from supervisor import message_bus
from tests import _dialogue as dialogue_test
from tests.test_ws_ingress_off_loop import bridge, _OpenSocket  # noqa: F401


def original(bridge):  # noqa: F811
    row = message_bus.log_chat("out", 1, 0, "Earlier answer", client_message_id="answer")
    return format_address(row_address(row))


@pytest.mark.parametrize("extra", [{}, {"reply_excerpt": {"text": "answer", "start_utf16": 8}}])
def test_http_reply_resolves_and_rejects_client_authored_previews(bridge, extra, monkeypatch):  # noqa: F811
    address = original(bridge)
    app = Starlette(routes=[Route("/api/command", api_command, methods=["POST"])])
    app.state.drive_root = bridge.drive
    with TestClient(app) as client:
        forged = {"address": address, "author_label": "Someone else", "snippet": "Not in the source"}
        body = {"cmd": "about that", "reply_to": address, "accepted_reply": forged, "reply_reference": forged, **extra}
        assert client.post("/api/command", json=body).status_code == 200
        assert client.post("/api/command", json={"cmd": "forged", "reply_to": {"snippet": "fake"}}).status_code == 400
    if extra:
        monkeypatch.setattr("ouroboros.chat_references.resolve_row", lambda *_: pytest.fail("second source read"))
    [item] = [u["message"] for u in bridge.get_updates(offset=0, timeout=0)]
    metadata_read_by_consumer = item["task_metadata"]
    assert metadata_read_by_consumer["reply_to"] == address
    message_bus.record_inbound_message(bridge, item, chat_id=1, user_id=1, text="about that",
        client_message_id="http-reply", ts="2026-10-01T00:00:00Z")
    assert metadata_read_by_consumer["reply_reference"]["address"] == address
    assert metadata_read_by_consumer["reply_reference"].get("excerpt") == extra.get("reply_excerpt")
    assert item["task_metadata"]["reply_reference"]["snippet"] == "Earlier answer"
    rows = dialogue_test.chat_rows(bridge.drive)
    assert len(rows) == 2 and rows[-1]["reply_to"]["address"] == address


@pytest.mark.parametrize("extra", [{}, {"reply_excerpt": {"text": "answer", "start_utf16": 8}}])
def test_websocket_reply_keeps_the_host_preview_on_echo_and_queue(bridge, extra):  # noqa: F811
    address = original(bridge)
    echoes = []
    bridge._broadcast_fn = echoes.append
    socket = _OpenSocket([json.dumps({"type": "chat", "content": "this", "client_message_id": "reply",
                                      "reply_to": address, **extra})])

    async def run():
        task = asyncio.create_task(ws_endpoint(socket))
        try:
            async def echoed():
                while not echoes:
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(echoed(), 5)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run())
    [echo] = echoes
    assert echo["reply_to"].get("excerpt") == extra.get("reply_excerpt")
    assert echo["reply_to"]["address"] == address and echo["reply_to"]["snippet"] == "Earlier answer"
    [item] = [u["message"] for u in bridge.get_updates(offset=0, timeout=0)]
    delivered = item["task_metadata"]["reply_reference"]
    assert stored_reply(delivered) == echo["reply_to"], "the echo keeps the compact reference"
    assert delivered.get("text") == ("Earlier answer" if extra else None), "a selection's source rides the delivery"
    row = item["accepted_source_row"]
    assert echo["row_address"] == format_address(row_address(row))


def test_http_command_example_and_javascript_mirror_share_the_request_schema():
    import re
    import typing
    from pathlib import Path

    from ouroboros.gateway.management_contracts import CommandRequest
    from ouroboros.gateway.schema import validate_ingress
    from tests.test_gateway_parity import _js_typedef_fields

    repo = Path(__file__).resolve().parents[1]
    section = (repo / "docs/CLIENT_CONTRACT.md").read_text(encoding="utf-8").split("**Commands, restart and Panic.**", 1)[1]
    example = next(json.loads(code) for code in re.findall(r"`([^`]+)`", section.splitlines()[0])
                   if code.startswith('{"cmd"') and '"reply_excerpt"' in code)
    assert not validate_ingress(example, CommandRequest)
    assert validate_ingress({**example, "reply_excerpt": {"text": 7}}, CommandRequest)
    mirror = (repo / "web/modules/management_contract_types.js").read_text(encoding="utf-8")
    assert _js_typedef_fields(mirror, "CommandRequest") == set(typing.get_type_hints(CommandRequest))


@pytest.mark.parametrize("kind", ["photo", "video", "document"])
def test_selected_media_words_come_from_explicit_caption_only(bridge, kind):  # noqa: F811
    from ouroboros.chat_references import ingress_reply
    from ouroboros.gateway.history_deliveries import delivery_fields
    from ouroboros.owner_words import reply_target

    source = message_bus.log_chat("out", 1, 0, "Copper lamp", record_type=kind, caption="Copper lamp")
    fallback = message_bus.log_chat("out", 1, 0, "Photo attachment", record_type=kind)
    excerpt = {"text": "lamp", "start_utf16": 7}
    ref = ingress_reply(bridge.drive, {"reply_to": format_address(row_address(source)), "reply_excerpt": excerpt})
    assert ref["excerpt"] == excerpt and reply_target(bridge.drive, ref)["text"] == "Copper lamp"
    assert delivery_fields(source)["caption"] == "Copper lamp"  # still a caption even without a durable URL
    with pytest.raises(ValueError, match="authored chat message"):
        ingress_reply(bridge.drive, {"reply_to": format_address(row_address(fallback)), "reply_excerpt": excerpt})


def test_only_generated_web_attachment_text_is_not_a_reply_source(bridge):  # noqa: F811
    from ouroboros.chat_references import ReplyExcerptNotAuthored, ingress_reply

    tail = "[Attached file: map.pdf]"
    refs = {"attachments": [{"name": "map.pdf", "unavailable": "missing"}]}
    for text, transport, meta, allowed in (
        (tail, "web", refs, False), (tail, "web", {}, True),
        (f"{tail}\n\n{tail}", "web", refs, True), (tail, "telegram", refs, True),
    ):
        source = message_bus.log_chat("in", 1, 1, text, source=transport, message_meta=meta)
        pick = {"reply_to": format_address(row_address(source)), "reply_excerpt": {"text": tail, "start_utf16": 0}}
        if allowed:
            assert ingress_reply(bridge.drive, pick)["excerpt"] == pick["reply_excerpt"]
        else:
            with pytest.raises(ReplyExcerptNotAuthored):
                ingress_reply(bridge.drive, pick)


def test_websocket_names_nonquotable_row_and_conflicting_id(bridge):  # noqa: F811
    from starlette.routing import WebSocketRoute

    source = message_bus.log_chat("out", 1, 0, "Photo attachment", record_type="photo")
    address = original(bridge)
    frame = {"type": "chat", "content": "Keep my draft", "client_message_id": "bound-id", "reply_to": address,
             "reply_excerpt": {"text": "answer", "start_utf16": 8}}
    bridge.ui_send(frame["content"], client_message_id=frame["client_message_id"], task_metadata={
        "reply_to": address, "reply_excerpt": frame["reply_excerpt"]})
    app = Starlette(routes=[WebSocketRoute("/ws", ws_endpoint)])
    with TestClient(app) as client, client.websocket_connect("/ws") as socket:
        for rejected, words in (
            ({**frame, "client_message_id": "not-authored", "reply_to": format_address(row_address(source))}, "no text you can quote"),
            ({**frame, "reply_excerpt": {"text": "Earlier", "start_utf16": 0}}, "belongs to a different message"),
        ):
            socket.send_json(rejected)
            notice = socket.receive_json()
            assert notice["system_type"] == "initialization_notice"  # existing draft recovery door
            assert notice["chat_id"] == 1 and notice["client_message_id"] == rejected["client_message_id"]
            assert words in notice["content"] and "initializing" not in notice["content"]
    assert [row["client_message_id"] for row in dialogue_test.chat_rows(bridge.drive) if row["direction"] == "in"] == ["bound-id"]

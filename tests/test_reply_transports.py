"""HTTP and WebSocket owner sends carry the same host-resolved Reply fact."""
import asyncio
import json

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.chat_chain import format_address, row_address
from ouroboros.gateway.control import api_command
from ouroboros.gateway.ws import ws_endpoint
from supervisor import message_bus
from tests import _dialogue as dialogue_test
from tests.test_ws_ingress_off_loop import bridge, _OpenSocket  # noqa: F401


def original(bridge):  # noqa: F811
    row = message_bus.log_chat("out", 1, 0, "Earlier answer", client_message_id="answer")
    return format_address(row_address(row))


def test_http_reply_resolves_and_rejects_client_authored_previews(bridge):  # noqa: F811
    address = original(bridge)
    app = Starlette(routes=[Route("/api/command", api_command, methods=["POST"])])
    app.state.drive_root = bridge.drive
    with TestClient(app) as client:
        assert client.post("/api/command", json={"cmd": "about that", "reply_to": address}).status_code == 200
        assert client.post("/api/command", json={"cmd": "forged", "reply_to": {"snippet": "fake"}}).status_code == 400
    [item] = [u["message"] for u in bridge.get_updates(offset=0, timeout=0)]
    metadata_read_by_consumer = item["task_metadata"]
    assert metadata_read_by_consumer["reply_to"] == address
    message_bus.record_inbound_message(bridge, item, chat_id=1, user_id=1, text="about that",
        client_message_id="http-reply", ts="2026-10-01T00:00:00Z")
    assert metadata_read_by_consumer["reply_reference"]["address"] == address
    assert item["task_metadata"]["reply_reference"]["snippet"] == "Earlier answer"
    rows = dialogue_test.chat_rows(bridge.drive)
    assert len(rows) == 2 and rows[-1]["reply_to"]["address"] == address


def test_websocket_reply_keeps_the_host_preview_on_echo_and_queue(bridge):  # noqa: F811
    address = original(bridge)
    echoes = []
    bridge._broadcast_fn = echoes.append
    socket = _OpenSocket([json.dumps({"type": "chat", "content": "this", "client_message_id": "reply",
                                      "reply_to": address})])

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
    assert echo["reply_to"]["address"] == address and echo["reply_to"]["snippet"] == "Earlier answer"
    [item] = [u["message"] for u in bridge.get_updates(offset=0, timeout=0)]
    assert item["task_metadata"]["reply_reference"] == echo["reply_to"]
    row = item["accepted_source_row"]
    assert echo["row_address"] == format_address(row_address(row))

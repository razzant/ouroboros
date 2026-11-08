"""Every control message the call page sends crosses the real gateway parser and reaches the hub.

28.10: the page's new share message (Show started or stopped) was known to the protocol and the hub but not to
the gateway's per-type shapes, so the first shared frame closed the owner's call with invalid_call_control. The
hub-level tests went around that parser; these drive ``serve_peer`` with the exact messages the page sends.
"""
from __future__ import annotations

import asyncio
import json
import re
import struct
from pathlib import Path

import pytest

from ouroboros import call_protocol as cp
from ouroboros.call_peers import LiveCallHub
from ouroboros.gateway.call_contracts import PEER_CONTROLS
from ouroboros.gateway.call_ws import serve_peer

UI = Path(__file__).resolve().parents[1] / "skills" / "live_conversation" / "ui"
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"


def test_every_protocol_peer_message_has_a_gateway_shape():
    assert set(cp.PEER_MESSAGE_TYPES) == set(PEER_CONTROLS)


def test_every_message_type_the_page_sends_is_a_protocol_peer_message():
    sent = set()
    for path in UI.glob("*.js"):
        sent |= set(re.findall(r"send\((?:JSON\.stringify\()?\{\s*type:\s*'([a-z_]+)'", path.read_text(encoding="utf-8")))
    assert sent and sent <= set(cp.PEER_MESSAGE_TYPES), sorted(sent - set(cp.PEER_MESSAGE_TYPES))


class _Page:
    """The owner's page as the server sees it: the hello, then each frame and control the page sends."""

    def __init__(self, incoming):
        self.incoming, self.closed, self.sent = list(incoming), [], []

    async def accept(self):
        pass

    async def receive(self):
        await asyncio.sleep(0)
        if not self.incoming:
            return {"type": "websocket.disconnect"}
        item = self.incoming.pop(0)
        return {"type": "websocket.receive", **({"bytes": item} if isinstance(item, bytes) else {"text": json.dumps(item)})}

    async def send_text(self, text):
        self.sent.append(text)

    async def send_bytes(self, data):
        self.sent.append(data)

    async def close(self, **reason):
        self.closed.append(reason)


def test_the_pages_show_mute_playback_and_goodbye_messages_pass_the_real_gateway():
    hub = LiveCallHub(cp.CallSpec("synthetic-page", 1, "synthetic", "synthetic"), rates=(24000, 24000))
    audio = cp.encode_frame(cp.AudioFrame(cp.FRAME_AUDIO_IN, 0, 1, 24000, 0, struct.pack("<480h", *([800] * 480))))
    video = cp.encode_frame(cp.VideoFrame(0, 1, "image/jpeg", JPEG))
    page = _Page([
        {"type": "hello", "protocol": 1, "role": "owner", "mic": True, "speaker": True, "rate": 24000, "video": True,
         "label": "Owner"},
        audio, video,
        {"type": "share", "sharing": True, "kind": "screen"},
        {"type": "mute", "muted": True}, {"type": "mute", "muted": False},
        {"type": "played", "item_id": "item-1", "played_ms": 120},
        {"type": "share", "sharing": False, "kind": "screen"},
        {"type": "bye", "reason": "left"},
    ])
    asyncio.run(serve_peer(page, hub, role="owner", video=True))
    assert all(reason.get("code", 1000) == 1000 for reason in page.closed), page.closed
    controls = []
    while (event := hub.next_event(0.05)) is not None:
        if isinstance(event, cp.PeerControl):
            controls.append((event.message["type"], event.message.get("sharing")))
    assert ("share", True) in controls and ("share", False) in controls
    assert [kind for kind, _ in controls].count("mute") == 2 and ("bye", None) in controls
    hub.close("fixture_cleanup")


@pytest.mark.parametrize("message", [
    {"type": "share", "sharing": "yes", "kind": "screen"},
    {"type": "share", "sharing": True, "kind": "window"},
    {"type": "never_defined"},
])
def test_a_malformed_page_message_still_closes_with_its_reason(message):
    hub = LiveCallHub(cp.CallSpec("synthetic-bad", 1, "synthetic", "synthetic"), rates=(24000, 24000))
    page = _Page([{"type": "hello", "protocol": 1, "role": "owner", "mic": True, "speaker": True, "rate": 24000,
                   "video": True}, message])
    asyncio.run(serve_peer(page, hub, role="owner", video=True))
    assert page.closed and page.closed[0].get("code") == 1008, page.closed
    hub.close("fixture_cleanup")

"""Live call wire protocol: what every audio peer and the live driver agree on.

A call is one admitted turn (ARCHITECTURE §6 "Live calls"); its audio reaches the
server through peers — the owner's own surface (the SPA on any shell) and reviewed
skill companions (system-audio capture, a browser participant, a Zoom SDK client)
— over one framed socket per peer. This module is the shared vocabulary: the call
specification the host admits, the binary audio frame, and the small JSON control
messages. Mixing, resampling and routing live in ``call_peers``; the socket routes
live in the gateway; the turn itself is ``live_conversation.LiveConversationTurn``.

Who spoke is a host fact (the peer's authenticated identity), never a model claim.
An owner surface's microphone is the owner; a meeting capture is "meeting audio"
whose individual speakers stay unknown unless the peer reports them as context.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional

PROTOCOL_VERSION = 1

PEER_OWNER = "owner"        # the owner's own surface: mic is owner speech, speaker plays the call
PEER_MEETING = "meeting"    # a capture or participant companion: meeting audio in, optional speech out
PEER_ROLES = (PEER_OWNER, PEER_MEETING)

OPENED_BY_OWNER = "owner_surface"
OPENED_BY_SKILL = "skill"

# A call is born preparing (its turn reads memory), connects to the engine, then is live.
STATE_PREPARING = "preparing"
STATE_CONNECTING = "connecting"
STATE_LIVE = "live"
STATE_RECONNECTING = "reconnecting"
STATE_CLOSING = "closing"
STATE_CLOSED = "closed"
CALL_STATES = (STATE_PREPARING, STATE_CONNECTING, STATE_LIVE, STATE_RECONNECTING, STATE_CLOSING, STATE_CLOSED)
# Where a live call's turn is: an answer is awaited (the owner finished, a tool result or an addressed fact
# went in), the engine is answering, or neither. The page's Thinking and Speaking follow it.
TURN_AWAITING, TURN_ANSWERING, TURN_IDLE = "awaiting", "answering", "idle"
CALL_TURNS = (TURN_AWAITING, TURN_ANSWERING, TURN_IDLE)
# The one next step a surface offers beside a closed call's reason.

# Binary frame: fixed 16-byte header + PCM16LE mono or JPEG/PNG payload.
#   magic  2s  b"OC"
#   ver    B   PROTOCOL_VERSION
#   kind   B   FRAME_AUDIO_IN | FRAME_AUDIO_OUT | FRAME_VIDEO_IN (peer -> server)
#   epoch  H   output epoch (server bumps it on interruption; peers drop stale epochs)
#   seq    I   per-peer, per-direction sequence
#   rate   I   audio sample rate; for video, 1 = JPEG and 2 = PNG
#   stream H   peer-local stream id (0 = the peer's primary track)
_HEADER = struct.Struct("<2sBBHIIH")
HEADER_SIZE = _HEADER.size
_MAGIC = b"OC"
FRAME_AUDIO_IN = 1
FRAME_AUDIO_OUT = 2
FRAME_VIDEO_IN = 3
MAX_FRAME_PAYLOAD = 64 * 1024
MAX_VIDEO_FRAME_PAYLOAD = 512 * 1024
VIDEO_FORMATS = {"image/jpeg": 1, "image/png": 2}


@dataclass(frozen=True)
class AudioFrame:
    kind: int
    epoch: int
    seq: int
    rate: int
    stream: int
    pcm: bytes


@dataclass(frozen=True)
class VideoFrame:
    epoch: int
    seq: int
    mime: str
    data: bytes


def encode_frame(frame: AudioFrame | VideoFrame) -> bytes:
    if isinstance(frame, VideoFrame):
        if frame.mime not in VIDEO_FORMATS or not 0 < len(frame.data) <= MAX_VIDEO_FRAME_PAYLOAD:
            raise ValueError("video payload requires JPEG/PNG and at most 512 KiB")
        return _HEADER.pack(_MAGIC, PROTOCOL_VERSION, FRAME_VIDEO_IN, frame.epoch & 0xFFFF,
                            frame.seq & 0xFFFFFFFF, VIDEO_FORMATS[frame.mime], 0) + frame.data
    if len(frame.pcm) > MAX_FRAME_PAYLOAD or len(frame.pcm) % 2:
        raise ValueError("audio payload must be PCM16 and at most 64 KiB")
    return _HEADER.pack(_MAGIC, PROTOCOL_VERSION, frame.kind, frame.epoch & 0xFFFF,
                        frame.seq & 0xFFFFFFFF, frame.rate, frame.stream & 0xFFFF) + frame.pcm


def decode_frame(data: bytes) -> AudioFrame | VideoFrame:
    if len(data) < HEADER_SIZE:
        raise ValueError("audio frame shorter than its header")
    magic, version, kind, epoch, seq, rate, stream = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != PROTOCOL_VERSION:
        raise ValueError("not a live call audio frame of this protocol version")
    pcm = data[HEADER_SIZE:]
    if kind == FRAME_VIDEO_IN:
        mime = next((name for name, code in VIDEO_FORMATS.items() if code == rate), "")
        if not mime or stream or not 0 < len(pcm) <= MAX_VIDEO_FRAME_PAYLOAD:
            raise ValueError("malformed live call video frame")
        return VideoFrame(epoch, seq, mime, pcm)
    if len(pcm) > MAX_FRAME_PAYLOAD or len(pcm) % 2 or kind not in (FRAME_AUDIO_IN, FRAME_AUDIO_OUT):
        raise ValueError("malformed live call audio frame")
    return AudioFrame(kind, epoch, seq, rate, stream, pcm)


# JSON control messages (text frames). Peer -> server:
#   {"type": "hello", "protocol": 1, "role": "owner"|"meeting", "mic": bool, "speaker": bool,
#    "rate": int, "video": bool, "label": str}
#   {"type": "mute", "muted": bool}
#   {"type": "played", "item_id": str, "played_ms": int}     # playback acknowledgement
#   {"type": "context", "text": str, "source": str}           # a peer-observed fact for the mind
#   {"type": "participant", "label": str, "speaking": bool}   # who the platform says is speaking
#   {"type": "share", "sharing": bool, "kind": "screen"|"camera"}  # first frame sent / capture stopped
#   {"type": "bye", "reason": str}
# Server -> peer:
#   {"type": "state", "state": <CALL_STATES>, "reason": str, "engine": str,
#    "phase": str, "detail": str, "action": ""|<ACTION_*>}
#        phase names what a preparing call is doing ("memory"); detail is the provider's or the
#        host's own sentence for a closed call; action is its one next step
#   {"type": "activity", "activity": "working"|"idle", "label": str}   # a tool batch runs
#   {"type": "turn", "turn": "awaiting"|"answering"|"idle"}  # where the turn is (CALL_TURNS)
#   {"type": "caption", "speaker": "owner"|"ouroboros", "text": str, "final": bool, "item_id": str}
#   {"type": "flush", "epoch": int}                           # stop playback of older epochs now
PEER_MESSAGE_TYPES = ("hello", "mute", "played", "context", "participant", "share", "bye")
SERVER_MESSAGE_TYPES = ("state", "activity", "turn", "caption", "flush")
MAX_CONTROL_BYTES = 16 * 1024


def encode_control(message: Dict[str, Any]) -> str:
    if message.get("type") not in PEER_MESSAGE_TYPES + SERVER_MESSAGE_TYPES:
        raise ValueError(f"unknown live call control type: {message.get('type')!r}")
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def decode_control(text: str) -> Dict[str, Any]:
    if len(text.encode("utf-8")) > MAX_CONTROL_BYTES:
        raise ValueError("live call control message too large")
    message = json.loads(text)
    if not isinstance(message, dict) or message.get("type") not in PEER_MESSAGE_TYPES + SERVER_MESSAGE_TYPES:
        raise ValueError("unknown live call control message")
    return message


@dataclass(frozen=True)
class CallSpec:
    """What the host admits as one call.

    ``chat_id`` is the room. ``engine``/``model``/``voice`` select the realtime
    route. ``opened_by`` records whether the owner's own surface or a reviewed
    skill opened it; a skill-opened call for a Presence binding carries
    ``presence_binding_id`` and runs under that binding's ceiling. Legacy
    ``max_minutes``, ``instructions``, ``pause`` and ``interruptions`` survive for
    companion source compatibility. The loaded feature config owns current
    limits; provider sessions own turn detection and renewal. Deserialization
    conveys identity only: it cannot mint an admitted turn, funds or owner rights.
    """

    call_id: str
    chat_id: int
    engine: str
    model: str
    voice: str = ""
    opened_by: str = OPENED_BY_OWNER
    presence_binding_id: str = ""
    max_minutes: Optional[int] = None
    label: str = ""
    client_message_id: str = ""
    instructions: str = ""
    pause: str = ""
    interruptions: bool = True


# ---- the peer side of one call, as the live driver sees it ------------------------
#
# The gateway (``call_peers``) owns the sockets; the feature driver owns its provider
# session. They meet in one in-process hub per admitted call. The hub is thread-safe
# and synchronous: the gateway's socket handlers run on the server's event loop, the
# driver runs on the turn's own thread with its own loop.


@dataclass(frozen=True)
class PeerInfo:
    peer_id: str
    role: str            # PEER_OWNER | PEER_MEETING
    mic: bool
    speaker: bool
    label: str = ""
    skill: str = ""      # the reviewed skill a companion peer belongs to ("" for an owner surface)
    video: bool = False


@dataclass(frozen=True)
class PeerJoined:
    peer: PeerInfo


@dataclass(frozen=True)
class PeerLeft:
    peer_id: str
    reason: str = ""


@dataclass(frozen=True)
class PeerAudio:
    """Mic audio of one peer, already converted to the engine input rate (PCM16 mono)."""

    peer_id: str
    role: str
    pcm: bytes


@dataclass(frozen=True)
class PeerVideo:
    peer_id: str
    role: str
    mime: str
    data: bytes


@dataclass(frozen=True)
class PeerControl:
    """A validated control message from one peer (``played``, ``mute``, ``context``,
    ``participant``, ``bye``)."""

    peer_id: str
    role: str
    message: Dict[str, Any]


class CallHub:
    """Interface of the per-call hub (implemented by ``call_peers.LiveCallHub``).

    Methods never block on a socket: ``play``/``flush``/``send_*`` enqueue for the
    peers; ``next_event`` waits at most ``timeout_s`` for the next inbound event.
    """

    call_id: str

    def set_engine_rates(self, input_rate: int, output_rate: int) -> None:
        raise NotImplementedError

    def next_event(self, timeout_s: float):  # -> PeerJoined | PeerLeft | PeerAudio | PeerControl | None
        raise NotImplementedError

    def play(self, pcm: bytes, item_id: str) -> None:
        """Engine-rate output audio for every speaker peer (resampled per peer)."""
        raise NotImplementedError

    def flush(self) -> int:
        """Bump the output epoch, tell peers to drop older playback now; returns the epoch."""
        raise NotImplementedError

    def send_state(self, state: str, reason: str = "", *, phase: str = "", detail: str = "",
                   action: str = "") -> None:
        raise NotImplementedError

    def send_activity(self, activity: str, label: str = "") -> None:
        raise NotImplementedError

    def send_turn(self, turn: str) -> None:
        """Where the turn is (``CALL_TURNS``); a joining surface receives the latest."""
        raise NotImplementedError

    def send_caption(self, speaker: str, text: str, final: bool, item_id: str = "", *, replace: bool = False) -> None:
        raise NotImplementedError

    def peers(self) -> "tuple[PeerInfo, ...]":
        raise NotImplementedError

    @property
    def ended_by_owner(self) -> bool:
        """The owner's surface said bye (a driver that has not opened its engine never opens it)."""
        raise NotImplementedError

    def close(self, reason: str, *, detail: str = "", action: str = "") -> None:
        raise NotImplementedError

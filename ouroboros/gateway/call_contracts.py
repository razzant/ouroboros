"""Call door and audio socket shapes, separate from the ordinary JSON chat socket.

The binary codec remains owned by ``call_protocol``. These descriptive shapes
publish that protocol to replaceable clients and validate the two call doors.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from typing import Literal, NotRequired, TypedDict
except ImportError:  # pragma: no cover - Python 3.10 compatibility.
    from typing_extensions import Literal, NotRequired, TypedDict

from ouroboros import call_protocol as cp


class CallOpenRequest(TypedDict):
    """The loaded feature configuration selects an explicit provider/model."""

    chat_id: int
    feature: NotRequired[str]
    client_message_id: str


class CallOpenResponse(TypedDict):
    call_id: str
    ws_path: str
    vision: bool
    joined: NotRequired[bool]   # the room already had a call: this surface joins it


class CallRefusal(TypedDict):
    ok: Literal[False]
    code: str
    error: str


class HostCallOpenRequest(TypedDict):
    """Attach to an admitted call, or request place admission for a new call.

    call_id attaches; without it the event opens a call in its connected place,
    named by binding_id (a place id or a migrated binding id) or resolved from
    the event's provider facts. With both, the place must be the call's place.
    Directions default to mic=true, speaker=false; a later hello can narrow but
    never widen them.
    """

    role: Literal["meeting"]
    call_id: NotRequired[str]
    binding_id: NotRequired[str]
    event: NotRequired[Dict[str, Any]]   # a Presence call's source event (the /presence/turn shape)
    label: NotRequired[str]
    mic: NotRequired[bool]
    speaker: NotRequired[bool]
    video: NotRequired[bool]


class HostCallOpenResponse(TypedDict):
    call_id: str
    peer_path: str


class CallHello(TypedDict):
    type: Literal["hello"]
    protocol: Literal[1]
    role: Literal["owner", "meeting"]
    mic: bool
    speaker: bool
    video: NotRequired[bool]
    rate: int
    label: NotRequired[str]


class CallMute(TypedDict):
    type: Literal["mute"]
    muted: bool


class CallPlayed(TypedDict):
    type: Literal["played"]
    item_id: str
    played_ms: int


class ConversationFinishedOutbound(TypedDict):
    type: Literal["conversation_finished"]
    chat_id: int
    call_id: str
    task_id: str
    reason: str


class CallContext(TypedDict):
    type: Literal["context"]
    text: str
    source: str


class CallParticipant(TypedDict):
    type: Literal["participant"]
    label: str
    speaking: bool


class CallBye(TypedDict):
    type: Literal["bye"]
    reason: NotRequired[str]


class CallState(TypedDict):
    """A call's state; a closed one carries its reason's words and one next step."""

    type: Literal["state"]
    state: Literal["preparing", "connecting", "live", "reconnecting", "closing", "closed"]
    reason: str
    engine: NotRequired[str]
    phase: NotRequired[str]
    detail: NotRequired[str]
    action: NotRequired[Literal["settings", "retry"]]


class CallActivity(TypedDict):
    type: Literal["activity"]
    activity: Literal["working", "idle"]
    label: str


class CallTurn(TypedDict):
    """Where the call's turn is: an answer awaited, the engine answering, or neither (a joining surface
    receives the latest)."""
    type: Literal["turn"]
    turn: Literal["awaiting", "answering", "idle"]


class CallCaption(TypedDict):
    type: Literal["caption"]
    speaker: Literal["owner", "ouroboros"]
    text: str
    final: bool
    item_id: str
    replace: NotRequired[bool]  # cumulative/revised text replaces, never appends


class CallFlush(TypedDict):
    type: Literal["flush"]
    epoch: int


class CallSocketContract(TypedDict):
    """The separate audio socket, including the companion door on the Host Service.

    Header format is Python struct notation; PCM is mono little-endian int16.
    Sequence is per peer/direction modulo 2**32; epoch is modulo 2**16. A flush
    stops older playback immediately. Binary stream 0 is the primary track;
    item ids for playback acknowledgements come from captions.
    """

    path: str
    host_open_path: str
    host_peer_path: str
    host_auth: str
    protocol: int
    audio_header: str
    audio_header_bytes: int
    max_audio_payload_bytes: int
    max_video_payload_bytes: int
    max_control_bytes: int
    peer_messages: Dict[str, str]
    server_messages: Dict[str, str]


PEER_CONTROLS = {"hello": CallHello, "mute": CallMute, "played": CallPlayed,
                 "context": CallContext, "participant": CallParticipant, "bye": CallBye}
SERVER_CONTROLS = {"state": CallState, "activity": CallActivity, "turn": CallTurn, "caption": CallCaption,
                   "flush": CallFlush}


def socket_contract() -> CallSocketContract:
    return {
        "path": "/ws/call/{call_id}", "host_open_path": "/call/open",
        "host_peer_path": "/call/{call_id}/peer", "host_auth": "X-Skill-Token; call_peer grant",
        "protocol": cp.PROTOCOL_VERSION, "audio_header": "<2sBBHIIH: magic, version, kind, epoch, seq, rate, stream",
        "audio_header_bytes": cp.HEADER_SIZE, "max_audio_payload_bytes": cp.MAX_FRAME_PAYLOAD,
        "max_video_payload_bytes": cp.MAX_VIDEO_FRAME_PAYLOAD,
        "max_control_bytes": cp.MAX_CONTROL_BYTES,
        "peer_messages": {name: cls.__name__ for name, cls in PEER_CONTROLS.items()},
        "server_messages": {name: cls.__name__ for name, cls in SERVER_CONTROLS.items()},
    }

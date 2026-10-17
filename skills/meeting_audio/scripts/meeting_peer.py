"""One meeting peer of an admitted live call, attached through the Host Service door.

The owner presses Call; the call is one admitted turn with its own audio hub. A
companion joins that SAME call as a ``meeting`` peer: ``POST /call/open`` with
the call id and the skill token asks the host for a direction grant, then the
peer socket ``/call/{call_id}/peer`` carries a ``hello`` followed by binary PCM
frames and small JSON control messages. The host decides who may attach; this
module never opens a call.

The wire constants mirror ``ouroboros/call_protocol.py`` (the companion runs
outside the server's import path, so it cannot import it);
``tests/test_skill_meeting_audio.py`` pins the two to each other. Rules the hub
enforces and this peer therefore keeps: every microphone frame carries the
sample rate declared in ``hello`` and the hub's CURRENT output epoch (learned
from ``flush``), sequence numbers only grow, a peer without a speaker never
sends ``played``, and a re-opened grant cannot change its directions.
"""

from __future__ import annotations

import json
import struct
import threading
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional

PROTOCOL_VERSION = 1
PEER_MEETING = "meeting"
FRAME_AUDIO_IN = 1
FRAME_AUDIO_OUT = 2
HEADER = struct.Struct("<2sBBHIIH")
MAGIC = b"OC"
MAX_FRAME_PAYLOAD = 64 * 1024
MAX_CONTROL_BYTES = 16 * 1024
PEER_MESSAGE_TYPES = ("hello", "mute", "played", "context", "participant", "bye")
SERVER_MESSAGE_TYPES = ("state", "activity", "caption", "flush")
STATE_CLOSED = "closed"

HTTP_TIMEOUT_SEC = 10.0
# Waiting for the hub's first flush after hello: a socket handshake bound.
ATTACH_TIMEOUT_SEC = 10.0


def encode_frame(kind: int, epoch: int, seq: int, rate: int, stream: int, pcm: bytes) -> bytes:
    if len(pcm) > MAX_FRAME_PAYLOAD or len(pcm) % 2:
        raise ValueError("audio payload must be PCM16 and at most 64 KiB")
    return HEADER.pack(MAGIC, PROTOCOL_VERSION, kind, epoch & 0xFFFF, seq & 0xFFFFFFFF,
                       rate, stream & 0xFFFF) + pcm


def decode_frame(data: bytes) -> Dict[str, Any]:
    if len(data) < HEADER.size:
        raise ValueError("audio frame shorter than its header")
    magic, version, kind, epoch, seq, rate, stream = HEADER.unpack_from(data)
    pcm = data[HEADER.size:]
    if magic != MAGIC or version != PROTOCOL_VERSION or len(pcm) % 2 or len(pcm) > MAX_FRAME_PAYLOAD:
        raise ValueError("not a live call audio frame of this protocol version")
    return {"kind": kind, "epoch": epoch, "seq": seq, "rate": rate, "stream": stream, "pcm": pcm}


class PeerRefused(RuntimeError):
    """The door or the hub refused this peer; ``code`` is the host's typed reason."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code
        self.message = message or code


def _ws_base(host_url: str) -> str:
    if host_url.startswith("https://"):
        return "wss://" + host_url[len("https://"):]
    if host_url.startswith("http://"):
        return "ws://" + host_url[len("http://"):]
    raise PeerRefused("host_url_invalid", f"unexpected Host Service URL {host_url!r}")


def _post_json(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> tuple:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SEC) as response:
            return response.status, json.loads(response.read(MAX_CONTROL_BYTES) or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read(MAX_CONTROL_BYTES) or b"{}")
        except ValueError:
            payload = {}
        return exc.code, payload if isinstance(payload, dict) else {}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise PeerRefused("host_unreachable", f"the Host Service did not answer: {exc}") from None


def _ws_connect(url: str, headers: Dict[str, str]):
    from websockets.sync.client import connect

    return connect(url, additional_headers=headers, open_timeout=HTTP_TIMEOUT_SEC,
                   max_size=HEADER.size + MAX_FRAME_PAYLOAD, compression=None)


class DoorPeer:
    """This companion's attachment to one admitted call.

    ``open()`` asks the door for the grant once; ``attach(speaker=...)`` opens a
    socket and says hello; ``reattach`` replaces the socket when the speaking
    direction changes (a hello can only narrow the grant, so the grant is asked
    with the widest direction this platform can use). ``close()`` is final: an
    attach that is still connecting when it happens closes its new socket instead
    of staying in the call. Callbacks run on the socket's reader thread and must
    not block on the caller's locks.
    """

    def __init__(self, host_url: str, token: str, call_id: str, *, rate: int, label: str,
                 speaker_grant: bool,
                 on_audio_out: Optional[Callable[[bytes, int, int], None]] = None,
                 on_flush: Optional[Callable[[int], None]] = None,
                 on_closed: Optional[Callable[[int, str], None]] = None,
                 post: Callable[..., tuple] = _post_json,
                 connect: Callable[..., Any] = _ws_connect):
        self.host_url = host_url.rstrip("/")
        self.call_id = call_id
        self.rate = int(rate)
        self.label = label
        self.speaker_grant = bool(speaker_grant)
        self.speaker = False
        self.peer_path = ""
        self.state = ""
        self.closed_reason = ""
        self.stats = {"frames_sent": 0, "frames_unsent": 0, "audio_out_frames": 0,
                      "audio_out_stale": 0}
        self._token = token
        self._on_audio_out = on_audio_out
        self._on_flush = on_flush
        self._on_closed = on_closed
        self._post = post
        self._connect = connect
        self._send_lock = threading.Lock()
        self._ws = None
        self._reader: Optional[threading.Thread] = None
        self._attachment = 0
        self._epoch: Optional[int] = None
        self._epoch_ready = threading.Event()
        self._seq = 0
        self._closed = False

    @property
    def attachment(self) -> int:
        return self._attachment

    @property
    def epoch(self) -> Optional[int]:
        return self._epoch

    def open(self) -> None:
        status, payload = self._post(
            f"{self.host_url}/call/open",
            {"role": PEER_MEETING, "call_id": self.call_id, "mic": True, "speaker": self.speaker_grant},
            {"x-skill-token": self._token},
        )
        if status != 200 or not isinstance(payload.get("peer_path"), str):
            raise PeerRefused(str(payload.get("code") or f"http_{status}"), str(payload.get("error") or ""))
        self.peer_path = payload["peer_path"]

    def attach(self, *, speaker: bool) -> None:
        if self._closed:
            raise PeerRefused("peer_closed", "the meeting peer was closed")
        if not self.peer_path:
            raise PeerRefused("not_opened", "attach requires a granted call")
        if speaker and not self.speaker_grant:
            raise PeerRefused("audio_direction_forbidden", "this peer was not granted a speaker")
        self._attachment += 1
        attachment = self._attachment
        self._epoch, self._seq, self.state, self.closed_reason = None, 0, "", ""
        self._epoch_ready.clear()
        try:
            ws = self._connect(_ws_base(self.host_url) + self.peer_path, {"x-skill-token": self._token})
        except Exception as exc:
            raise PeerRefused("peer_socket_failed", f"could not open the call socket: {exc}") from None
        hello = {"type": "hello", "protocol": PROTOCOL_VERSION, "role": PEER_MEETING, "mic": True,
                 "speaker": bool(speaker), "rate": self.rate, "label": self.label}
        try:
            ws.send(json.dumps(hello, separators=(",", ":")))
        except Exception as exc:
            self._close_socket(ws, "attach_failed")
            raise PeerRefused("peer_socket_failed", f"the call socket closed during hello: {exc}") from None
        self._ws, self.speaker = ws, bool(speaker)
        self._reader = threading.Thread(target=self._read_loop, args=(ws, attachment),
                                        name="meeting-peer-reader", daemon=True)
        self._reader.start()
        answered = self._epoch_ready.wait(ATTACH_TIMEOUT_SEC)
        if self._closed or not answered or self._epoch is None or self._ws is not ws:
            closed = self._closed
            reason = ("the meeting peer was closed" if closed
                      else self.closed_reason or "the call did not answer the hello")
            self._attachment += 1  # a refused or abandoned hello is reported here, not by the reader
            self._ws = None
            self._close_socket(ws, "closed" if closed else "attach_failed")
            raise PeerRefused("peer_closed" if closed else "peer_attach_failed", reason)

    def reattach(self, *, speaker: bool) -> None:
        self.detach("direction_change")
        self.open()
        self.attach(speaker=speaker)

    def send_audio(self, pcm: bytes, stream: int = 0) -> bool:
        """Send capture PCM (mono int16 at ``self.rate``) as frames of the current epoch."""
        ws, epoch = self._ws, self._epoch
        if ws is None or epoch is None:
            self.stats["frames_unsent"] += 1
            return False
        with self._send_lock:
            try:
                for offset in range(0, len(pcm), MAX_FRAME_PAYLOAD):
                    ws.send(encode_frame(FRAME_AUDIO_IN, epoch, self._seq, self.rate, stream,
                                         pcm[offset:offset + MAX_FRAME_PAYLOAD]))
                    self._seq = (self._seq + 1) & 0xFFFFFFFF
                    self.stats["frames_sent"] += 1
            except Exception:
                self.stats["frames_unsent"] += 1
                return False
        return True

    def send_context(self, text: str, source: str = "meeting_audio") -> bool:
        return self._send_control({"type": "context", "text": text, "source": source})

    def _send_control(self, message: Dict[str, Any]) -> bool:
        ws = self._ws
        if ws is None:
            return False
        text = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode("utf-8")) > MAX_CONTROL_BYTES:
            raise ValueError("control message larger than the call protocol allows")
        with self._send_lock:
            try:
                ws.send(text)
            except Exception:
                return False
        return True

    def close(self, reason: str = "stopped") -> None:
        """Leave the call for good: no later or in-flight attach keeps a socket."""
        self._closed = True
        self.detach(reason)

    def detach(self, reason: str = "stopped") -> None:
        """Say bye and close the current socket; the grant stays with the call."""
        ws, self._ws = self._ws, None
        self._attachment += 1  # the closing socket's reader no longer speaks for this peer
        if ws is not None:
            self._close_socket(ws, reason)
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2.0)

    def _close_socket(self, ws: Any, reason: str) -> None:
        with self._send_lock:
            try:
                ws.send(json.dumps({"type": "bye", "reason": reason}, separators=(",", ":")))
            except Exception:
                pass
            try:
                ws.close()
            except Exception:
                pass

    def _read_loop(self, ws: Any, attachment: int) -> None:
        reason = "the call socket closed"
        while True:
            try:
                message = ws.recv()
            except Exception as exc:
                received = getattr(exc, "rcvd", None)
                reason = str(getattr(received, "reason", "") or self.closed_reason or reason)
                break
            if isinstance(message, (bytes, bytearray)):
                self._audio_out(bytes(message))
            else:
                self._control_in(str(message))
        if attachment == self._attachment:
            self.closed_reason = self.closed_reason or reason
        self._epoch_ready.set()
        if attachment == self._attachment and self._ws is ws:
            self._ws = None
        if self._on_closed is not None:
            self._on_closed(attachment, reason)

    def _control_in(self, text: str) -> None:
        if len(text.encode("utf-8")) > MAX_CONTROL_BYTES:
            return
        try:
            message = json.loads(text)
        except ValueError:
            return
        if not isinstance(message, dict):
            return
        kind = message.get("type")
        if kind == "flush" and isinstance(message.get("epoch"), int):
            self._epoch = message["epoch"] & 0xFFFF
            try:
                if self._on_flush is not None:
                    self._on_flush(self._epoch)
            finally:
                self._epoch_ready.set()
        elif kind == "state":
            self.state = str(message.get("state") or "")
            if self.state == STATE_CLOSED:
                self.closed_reason = str(message.get("reason") or "the call ended")

    def _audio_out(self, data: bytes) -> None:
        try:
            frame = decode_frame(data)
        except ValueError:
            return
        if frame["kind"] != FRAME_AUDIO_OUT:
            return
        if self._epoch is not None and frame["epoch"] != self._epoch:
            self.stats["audio_out_stale"] += 1
            return
        self.stats["audio_out_frames"] += 1
        if self._on_audio_out is not None and self.speaker:
            self._on_audio_out(frame["pcm"], frame["rate"], frame["epoch"])

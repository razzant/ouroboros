"""The call side of a joined meeting: the Host door, the peer socket and voice pacing.

When meeting audio flows, the companion asks the Host Service door ``POST /call/open``
to admit the Presence call for its binding. It then attaches as the call's ``meeting``
peer on ``WS /call/{call_id}/peer``, speaking the frozen live-call wire
(``ouroboros/call_protocol.py``: a 16-byte frame header plus PCM16 mono, small JSON
control messages). The companion runs outside the core's import path, so the frame
layout is mirrored here; a test pins it byte for byte to the core's codec.

Meeting audio goes up at the rate the SDK actually delivers, stamped with the hub's
current output epoch: the hub names it in the ``flush`` it sends on attach and after
every interruption, and refuses input of any other epoch. The call's voice comes
down already resampled by the core hub to the rate this peer declared, and goes into
the Zoom virtual microphone in exact 20 ms frames, paced in real time: the SDK plays
what it is given, so sending ahead would compress speech, and a frame cut short would
click. A ``flush`` (the call was interrupted) drops everything queued both ways, and
frames of an older output epoch are dropped on arrival.
"""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import struct
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

log = logging.getLogger("zoom_bot.peer")

PROTOCOL_VERSION = 1
FRAME_AUDIO_IN = 1
FRAME_AUDIO_OUT = 2
MAX_FRAME_PAYLOAD = 64 * 1024
MAX_CONTROL_BYTES = 16 * 1024
_HEADER = struct.Struct("<2sBBHIIH")
HEADER_SIZE = _HEADER.size
_MAGIC = b"OC"
FRAME_MS = 20
# A participant's one-way stream counts as paused after this long without a buffer;
# Zoom delivers raw audio in 10 ms buffers, so this is sixty missed deliveries.
PARTICIPANT_PAUSE_SEC = 0.6
# Queued call voice is bounded to this much audio; older audio is dropped first.
MAX_QUEUED_VOICE_SEC = 30
# Meeting audio waiting for the socket is bounded to about two seconds of 10 ms buffers.
MAX_QUEUED_UPLINK_BUFFERS = 200
# Meeting facts waiting for the socket are bounded like the hub's own control queue.
MAX_QUEUED_CONTROLS = 128
_DOOR_TIMEOUT_SEC = 30.0


def encode_frame(kind: int, epoch: int, seq: int, rate: int, stream: int, pcm: bytes) -> bytes:
    if len(pcm) > MAX_FRAME_PAYLOAD or len(pcm) % 2:
        raise ValueError("audio payload must be PCM16 and at most 64 KiB")
    return _HEADER.pack(_MAGIC, PROTOCOL_VERSION, kind, epoch & 0xFFFF, seq & 0xFFFFFFFF, rate,
                        stream & 0xFFFF) + pcm


def decode_frame(data: bytes) -> Tuple[int, int, int, int, int, bytes]:
    """``(kind, epoch, seq, rate, stream, pcm)``; malformed bytes raise ``ValueError``."""
    if len(data) < HEADER_SIZE:
        raise ValueError("audio frame shorter than its header")
    magic, version, kind, epoch, seq, rate, stream = _HEADER.unpack_from(data)
    pcm = bytes(data[HEADER_SIZE:])
    if magic != _MAGIC or version != PROTOCOL_VERSION or kind not in (FRAME_AUDIO_IN, FRAME_AUDIO_OUT):
        raise ValueError("not a live call audio frame of this protocol version")
    if len(pcm) > MAX_FRAME_PAYLOAD or len(pcm) % 2:
        raise ValueError("malformed live call audio frame")
    return kind, epoch, seq, rate, stream, pcm


def bounded_text(text: str, limit: int) -> str:
    """``text`` whole, or its first ``limit`` characters plus a named count of the rest."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ...[{len(text) - limit} more characters omitted]"


def fit_control(message: Dict[str, Any]) -> Optional[str]:
    """``message`` as protocol JSON within ``MAX_CONTROL_BYTES``.

    A ``text`` too long for one control message keeps its head plus a named count of
    what was cut; a message that cannot fit otherwise is ``None``.
    """
    encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) <= MAX_CONTROL_BYTES:
        return encoded
    text = message.get("text")
    if not isinstance(text, str):
        return None
    best, low, high = None, 0, len(text)
    while low <= high:
        middle = (low + high) // 2
        candidate = json.dumps({**message, "text": bounded_text(text, middle)}, ensure_ascii=False,
                               separators=(",", ":"))
        if len(candidate.encode("utf-8")) <= MAX_CONTROL_BYTES:
            best, low = candidate, middle + 1
        else:
            high = middle - 1
    return best


def epoch_is_newer(candidate: int, current: int) -> bool:
    """Serial comparison of 16-bit output epochs (they wrap)."""
    delta = (candidate - current) & 0xFFFF
    return delta != 0 and delta < 0x8000


class CallOpenRefused(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    # The Host Service is loopback; an inherited proxy must never see the skill token.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _host_json(url: str, token: str, *, body: Optional[Dict[str, Any]] = None,
               timeout: float = _DOOR_TIMEOUT_SEC) -> Tuple[int, Dict[str, Any]]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    request.add_header("x-skill-token", token)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with _no_proxy_opener().open(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
    return status, payload if isinstance(payload, dict) else {}


def open_call(host_url: str, token: str, body: Dict[str, Any]) -> Dict[str, str]:
    """Ask the Host door to admit the Presence call; ``{call_id, peer_path}`` or ``CallOpenRefused``."""
    try:
        status, payload = _host_json(f"{host_url.rstrip('/')}/call/open", token, body=body)
    except (OSError, ValueError) as exc:  # ValueError: no usable Host Service address
        raise CallOpenRefused("call_door_unreachable", f"the Host Service did not answer ({type(exc).__name__})")
    call_id, peer_path = str(payload.get("call_id") or ""), str(payload.get("peer_path") or "")
    if 200 <= status < 300 and call_id and peer_path.startswith("/"):
        return {"call_id": call_id, "peer_path": peer_path}
    code = str(payload.get("code") or f"http_{status}")
    message = str(payload.get("error") or payload.get("message") or payload.get("reason") or code)
    raise CallOpenRefused(code, message)


def identity_name(host_url: str, token: str) -> str:
    """This install's name from the Host Service, or ``""`` when it does not answer."""
    with contextlib.suppress(OSError, ValueError):
        status, payload = _host_json(f"{host_url.rstrip('/')}/identity", token, timeout=5.0)
        if status == 200:
            return str(payload.get("name") or "").strip()
    return ""


class OutboundPacer:
    """Call voice into the virtual microphone, one exact 20 ms frame per 20 ms.

    ``take_frame`` is the whole frame policy: a full frame whenever one is queued; the
    tail of an utterance, padded with silence, once a frame time passed with nothing
    new; otherwise nothing. ``run`` only keeps time: frames go out on a real-time
    schedule that restarts after silence and resynchronizes if the sender fell behind.
    """

    def __init__(self, send: Callable[[bytes], Any], *, rate: int,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.rate = int(rate)
        self.frame_bytes = self.rate * FRAME_MS // 1000 * 2  # 640 samples, 1,280 bytes at 32 kHz
        self.frame_sec = FRAME_MS / 1000.0
        self._send = send
        self._clock = clock
        self._sleep = sleep
        self._max_bytes = self.rate * 2 * MAX_QUEUED_VOICE_SEC
        self._buf = bytearray()
        self._epoch = 0
        self._fed_since_take = False
        self._closed = False
        self._cond = threading.Condition()
        self.sent_frames = 0
        self.flushed_bytes = 0
        self.overflow_bytes = 0
        self.stale_frames = 0

    def feed(self, pcm: bytes, epoch: int) -> None:
        with self._cond:
            if self._closed:
                return
            if epoch_is_newer(epoch, self._epoch):
                self.flushed_bytes += len(self._buf)
                self._buf.clear()
                self._epoch = epoch
            elif epoch != self._epoch:
                self.stale_frames += 1
                return
            self._buf += pcm
            overflow = len(self._buf) - self._max_bytes
            if overflow > 0:
                overflow += overflow % 2
                del self._buf[:overflow]
                self.overflow_bytes += overflow
            self._fed_since_take = True
            self._cond.notify()

    def flush(self, epoch: int) -> int:
        """Drop queued voice now; an older epoch's late flush is ignored."""
        with self._cond:
            if epoch != self._epoch and not epoch_is_newer(epoch, self._epoch):
                return 0
            dropped = len(self._buf)
            self._buf.clear()
            self._epoch = epoch
            self.flushed_bytes += dropped
            return dropped

    def take_frame(self) -> Optional[bytes]:
        with self._cond:
            fed, self._fed_since_take = self._fed_since_take, False
            if len(self._buf) >= self.frame_bytes:
                frame = bytes(self._buf[: self.frame_bytes])
                del self._buf[: self.frame_bytes]
                return frame
            if self._buf and not fed:
                frame = bytes(self._buf) + b"\x00" * (self.frame_bytes - len(self._buf))
                self._buf.clear()
                return frame
            return None

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._buf.clear()
            self._cond.notify_all()

    def run(self, stop: threading.Event) -> None:
        """Keep time; a failing send is logged once per streak, with its count when it ends.

        The microphone takes a frame every 20 ms, so an exception per frame would bury
        the log: the first failure carries its traceback, the rest are counted.
        """
        next_at: Optional[float] = None
        failing = 0  # failed sends in the current streak
        while not stop.is_set() and not self._closed:
            frame = self.take_frame()
            if frame is None:
                with self._cond:
                    idle = not self._buf
                    if idle:
                        next_at = None  # the next utterance starts its own clock
                        self._cond.wait(0.25)
                if not idle:
                    self._sleep(self.frame_sec)  # a tail waits one frame time for more speech
                continue
            now = self._clock()
            if next_at is None or now - next_at > 5 * self.frame_sec:
                next_at = now
            try:
                self._send(frame)
            except Exception:
                if not failing:
                    log.exception("virtual microphone send failed")
                failing += 1
            else:
                self.sent_frames += 1
                if failing:
                    log.warning("virtual microphone send recovered after %d failed frames", failing)
                    failing = 0
            next_at += self.frame_sec
            self._sleep(max(0.0, next_at - self._clock()))
        if failing:
            log.warning("virtual microphone send still failing when the voice stopped: %d failed frames", failing)


class ParticipantTracker:
    """Who Zoom is delivering one-way audio for, as speaking transitions.

    ``speaking`` means the platform is delivering that participant's own audio stream
    right now; it is a delivery fact, not a voice detector, and Zoom decides when a
    stream flows.
    """

    def __init__(self, pause_sec: float = PARTICIPANT_PAUSE_SEC) -> None:
        self._pause = pause_sec
        self._last: Dict[int, float] = {}
        self._speaking: set = set()
        self._lock = threading.Lock()

    def heard(self, node_id: int, now: float) -> None:
        with self._lock:
            self._last[node_id] = now

    def sweep(self, now: float) -> List[Tuple[int, bool]]:
        changes: List[Tuple[int, bool]] = []
        with self._lock:
            for node_id, seen in list(self._last.items()):
                flowing = now - seen <= self._pause
                if flowing and node_id not in self._speaking:
                    self._speaking.add(node_id)
                    changes.append((node_id, True))
                elif not flowing:
                    if node_id in self._speaking:
                        self._speaking.discard(node_id)
                        changes.append((node_id, False))
                    del self._last[node_id]
        return changes


def connect_websocket(url: str, token: str):
    """The peer socket through the core's own WebSocket library (a runtime dependency)."""
    from websockets.sync.client import connect

    return connect(url, additional_headers={"x-skill-token": token}, open_timeout=10,
                   max_size=HEADER_SIZE + MAX_FRAME_PAYLOAD + MAX_CONTROL_BYTES, compression=None)


class PeerSession:
    """One attached ``meeting`` peer: a sender thread and a reader thread.

    Every outbound message goes through one sender thread, so neither the SDK's audio
    callbacks nor the GLib loop ever wait on the socket. Control messages go before
    queued audio; either queue beyond its bound drops its oldest entry first and counts
    it. Audio is stamped with the epoch of the hub's latest ``flush`` (a call starts at
    0); a flush that moves the epoch drops the audio still queued under the old one,
    which the hub would refuse as stale.
    """

    def __init__(self, ws: Any, *, rate: int, on_audio: Callable[[bytes, int], None],
                 on_flush: Callable[[int], None], on_closed: Callable[[str], None]) -> None:
        self._ws = ws
        self.rate = int(rate)
        self._on_audio = on_audio
        self._on_flush = on_flush
        self._on_closed = on_closed
        self._cond = threading.Condition()
        self._audio: Deque[bytes] = collections.deque()
        self._controls: Deque[str] = collections.deque()
        self._closing: Optional[str] = None
        self._closed_reported = False
        self._server_reason = ""
        self._seq = 0
        self._epoch = 0
        self._threads: List[threading.Thread] = []
        self.sent_frames = 0
        self.dropped_uplink = 0
        self.stale_uplink = 0
        self.received_frames = 0
        self.ignored_frames = 0
        self.dropped_controls = 0

    @classmethod
    def attach(cls, host_url: str, token: str, peer_path: str, hello: Dict[str, Any], *,
               connect: Callable[[str, str], Any] = connect_websocket, **callbacks: Any) -> "PeerSession":
        scheme, _sep, rest = host_url.rstrip("/").partition("://")
        ws = connect(("wss" if scheme == "https" else "ws") + "://" + rest + peer_path, token)
        try:
            ws.send(json.dumps(hello, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            with contextlib.suppress(Exception):
                ws.close()
            raise
        session = cls(ws, rate=int(hello["rate"]), **callbacks)
        session.start()
        return session

    def start(self) -> None:
        for target, name in ((self._send_loop, "zoom-peer-send"), (self._read_loop, "zoom-peer-read")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def send_audio(self, pcm: bytes, rate: int) -> None:
        with self._cond:
            if self._closing is not None:
                return
            for start in range(0, len(pcm), MAX_FRAME_PAYLOAD):
                if len(self._audio) >= MAX_QUEUED_UPLINK_BUFFERS:
                    self._audio.popleft()
                    self.dropped_uplink += 1
                self._seq = (self._seq + 1) & 0xFFFFFFFF
                self._audio.append(encode_frame(FRAME_AUDIO_IN, self._epoch, self._seq, rate, 0,
                                                pcm[start:start + MAX_FRAME_PAYLOAD]))
            self._cond.notify()

    def send_control(self, message: Dict[str, Any]) -> None:
        text = fit_control(message)
        if text is None:
            self.dropped_controls += 1
            log.warning("a %s control message does not fit the protocol and was not sent", message.get("type"))
            return
        with self._cond:
            if self._closing is None:
                if len(self._controls) >= MAX_QUEUED_CONTROLS:
                    self._controls.popleft()
                    self.dropped_controls += 1
                self._controls.append(text)
                self._cond.notify()

    def close(self, reason: str) -> None:
        """Say ``bye`` and close; returns at once, the sender thread finishes it."""
        with self._cond:
            if self._closing is None:
                self._closing = reason or "left"
                self._closed_reported = True  # our own close is not a server hang-up
                self._cond.notify()

    def join(self, timeout: float = 2.0) -> None:
        for thread in self._threads:
            thread.join(timeout)

    def _send_loop(self) -> None:
        while True:
            with self._cond:
                while not self._controls and not self._audio and self._closing is None:
                    self._cond.wait(0.5)
                if self._controls:
                    item: Any = self._controls.popleft()
                elif self._closing is not None:
                    break
                else:
                    item = self._audio.popleft()
            try:
                self._ws.send(item)
                if isinstance(item, bytes):
                    self.sent_frames += 1
            except Exception:
                self._report_closed("the call socket closed")
                return
        with contextlib.suppress(Exception):
            self._ws.send(json.dumps({"type": "bye", "reason": self._closing}, ensure_ascii=False))
        with contextlib.suppress(Exception):
            self._ws.close()

    def _read_loop(self) -> None:
        try:
            for message in self._ws:
                self._handle(message)
        except Exception as exc:
            log.info("peer socket ended: %s", type(exc).__name__)
        self._report_closed(self._server_reason or "the call ended")

    def _handle(self, message: Any) -> None:
        if isinstance(message, (bytes, bytearray)):
            try:
                kind, epoch, _seq, rate, _stream, pcm = decode_frame(bytes(message))
            except ValueError:
                self.ignored_frames += 1
                return
            if kind != FRAME_AUDIO_OUT or rate != self.rate:
                self.ignored_frames += 1  # the hub resamples to the declared rate; anything else is not ours
                return
            self.received_frames += 1
            self._on_audio(pcm, epoch)
            return
        try:
            control = json.loads(message)
        except (TypeError, ValueError):
            return
        if not isinstance(control, dict):
            return
        if control.get("type") == "flush":
            epoch = control.get("epoch")
            if type(epoch) is int and 0 <= epoch <= 0xFFFF:
                self._flushed(epoch)
        elif control.get("type") == "state" and control.get("state") in ("closing", "closed"):
            self._server_reason = str(control.get("reason") or "")

    def _flushed(self, epoch: int) -> None:
        with self._cond:
            if epoch != self._epoch:
                self.stale_uplink += len(self._audio)
                self._audio.clear()
                self._epoch = epoch
        self._on_flush(epoch)

    def _report_closed(self, reason: str) -> None:
        with self._cond:
            if self._closed_reported:
                return
            self._closed_reported = True
            self._closing = self._closing or reason
            self._cond.notify()
        self._on_closed(reason)

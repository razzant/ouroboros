"""The synchronous audio boundary shared by socket peers and the live turn.

Queues bound transport memory, not conversation length. Input applies backpressure;
discarded input has counted causes; control overflow refuses the slow peer.
Mixing happens when the driver drains input, so simultaneously buffered microphones
share one engine timeline instead of playing sequentially. A 20 ms mixer tick
collects asynchronously arriving microphone frames before emitting each chunk. A mixed frame keeps a
single peer's identity when possible; otherwise its id is ``mixed`` and its role
is meeting if any contributor is a meeting. Mixing never invents owner provenance.
"""

from __future__ import annotations

import sys
import threading
import time
from array import array
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from ouroboros import call_protocol as cp

# Physical transport bounds: one second of capture per track; ten minutes of
# outbound PCM, because a provider sends a whole answer many times faster than it
# plays and a slow socket drains it over the answer's length (held only while
# queued, under 60 MiB per peer at 48 kHz); bounded controls and peer/track sets.
AUDIO_QUEUE_FRAMES = 50
OUTPUT_QUEUE_FRAMES = 30_000
CONTROL_QUEUE_MESSAGES = 128
MAX_PEERS = 32
MAX_STREAMS = 16
MIX_FRAME_MS = 20
MIN_RATE = 8000
MAX_RATE = 192000


class CallPeerError(ValueError):
    """A transport refusal whose code can cross either socket door."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _rate(rate: int) -> int:
    if type(rate) is not int or not MIN_RATE <= rate <= MAX_RATE:
        raise CallPeerError("unsupported_audio_rate")
    return rate


def _samples(pcm: bytes) -> array:
    if len(pcm) % 2:
        raise CallPeerError("invalid_pcm")
    result = array("h", pcm)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def _pcm(samples: array) -> bytes:
    if sys.byteorder != "little":
        samples = array("h", samples)
        samples.byteswap()
    return samples.tobytes()


class LinearResampler:
    """Carry fractional position across frames to avoid drift and boundary clicks.

    Positions use integer sample clocks; only the final interpolation rounds.
    Keeping one previous sample makes chunked conversion identical to one feed.
    """

    def __init__(self, input_rate: int, output_rate: int):
        self.input_rate, self.output_rate = _rate(input_rate), _rate(output_rate)
        self._position = 0
        self._offset = 0
        self._tail = array("h")

    def feed(self, pcm: bytes) -> bytes:
        samples = _samples(pcm)
        if self.input_rate == self.output_rate:
            return pcm
        data = self._tail + samples
        result = array("h")
        last = self._offset + len(data) - 1
        while data:
            index, fraction = divmod(self._position, self.output_rate)
            if index > last or (fraction and index == last):
                break
            local = index - self._offset
            value = data[local]
            if fraction:
                value += round((data[local + 1] - value) * fraction / self.output_rate)
            result.append(value)
            self._position += self.input_rate
        if data:
            self._tail = data[-1:]
            self._offset = last
        return _pcm(result)


@dataclass
class _Track:
    converter: LinearResampler
    samples: array = field(default_factory=lambda: array("h"))


@dataclass
class _Peer:
    info: cp.PeerInfo
    rate: int
    converter: LinearResampler
    muted: bool = False
    tracks: dict[int, _Track] = field(default_factory=dict)
    outgoing: deque = field(default_factory=deque)
    queued_audio: int = 0  # frames and controls in ``outgoing``, kept so a send never rescans it
    queued_controls: int = 0
    input_seq: int | None = None
    video_seq: int | None = None
    output_seq: int = 0
    failure: str = ""


@dataclass(frozen=True)
class PeerGrant:
    """Host-minted authority on this hub, bound to a skill token and directions."""

    skill: str
    token_digest: str
    mic: bool
    speaker: bool
    video: bool = False


class LiveCallHub(cp.CallHub):
    def __init__(self, spec: cp.CallSpec, *, audio_queue_frames: int = AUDIO_QUEUE_FRAMES,
                 control_queue_messages: int = CONTROL_QUEUE_MESSAGES, output_queue_frames: int = OUTPUT_QUEUE_FRAMES,
                 rates: tuple[int, int] = (24000, 24000)):
        if audio_queue_frames < 1 or control_queue_messages < 1 or output_queue_frames < 1:
            raise ValueError("queue bounds must be positive")
        self.spec, self.call_id = spec, spec.call_id
        self._condition = threading.Condition(threading.RLock())
        self._peers: dict[str, _Peer] = {}
        self._draining_peers: list[_Peer] = []
        self._grants: dict[tuple[str, str], PeerGrant] = {}
        self._events: deque = deque()
        self._video_pending: OrderedDict[str, cp.PeerVideo] = OrderedDict()
        self._last_video_at = 0.0
        self._audio_limit, self._control_limit = audio_queue_frames, control_queue_messages
        self._output_limit = output_queue_frames
        # Placeholder rates before capture begins; the selected provider must declare both at Ready.
        self._input_rate, self._output_rate = _rate(rates[0]), _rate(rates[1])
        self._epoch = 0
        self._mix_at = 0.0
        self._input_dropped_samples = 0
        self._input_dropped_ms = 0.0
        self._input_drop_causes = {}
        self._input_paused = False
        self._closed = False
        self._owner_bye = False
        self._ouroboros_bye = False
        # Born preparing: the call's turn is reading memory before any engine socket exists.
        self._state = {"type": "state", "state": cp.STATE_PREPARING, "reason": "", "engine": spec.engine,
                       "phase": "memory"}
        self._turn = {"type": "turn", "turn": cp.TURN_IDLE}


    @property
    def epoch(self) -> int:
        with self._condition:
            return self._epoch

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def ended_by_owner(self) -> bool:
        """The owner's surface said bye: a driver that has not opened its engine yet never opens it."""
        with self._condition:
            return self._owner_bye

    @property
    def ended_by_ouroboros(self) -> bool:
        """Ouroboros hung up (its ``end_call``): the driver closes once its goodbye has been heard."""
        with self._condition:
            return self._ouroboros_bye

    def end_by_ouroboros(self) -> None:
        with self._condition:
            self._ouroboros_bye = True

    def _live(self) -> None:
        if self._closed:
            raise CallPeerError("call_closed")

    def grant_peer(self, skill: str, token_digest: str, *, mic: bool, speaker: bool, video: bool = False) -> PeerGrant:
        with self._condition:
            self._live()
            grant = PeerGrant(skill, token_digest, mic, speaker, video)
            # Reopening a door cannot widen a previously admitted attachment.
            prior = self._grants.get((skill, token_digest))
            if prior is not None and prior != grant:
                raise CallPeerError("peer_direction_mismatch")
            if prior is None and len(self._grants) >= MAX_PEERS:
                raise CallPeerError("peer_capacity_full")
            self._grants[(skill, token_digest)] = grant
            return grant

    def peer_grant(self, skill: str, token_digest: str) -> PeerGrant:
        with self._condition:
            self._live()
            grant = self._grants.get((skill, token_digest))
            if grant is None:
                raise CallPeerError("call_peer_not_admitted")
            return grant

    def _event(self, event: Any) -> None:
        if len(self._events) >= self._control_limit:
            raise CallPeerError("control_queue_full")
        self._events.append(event)
        self._condition.notify_all()

    def _send(self, peer: _Peer, message: bytes | str) -> None:
        if peer.failure:
            return
        audio = isinstance(message, bytes)
        count = peer.queued_audio if audio else peer.queued_controls
        limit = self._output_limit if audio else self._control_limit
        if count >= limit:
            if not audio:
                peer.failure = "control_queue_full"
                return
            oldest = next(i for i, m in enumerate(peer.outgoing) if isinstance(m, bytes))
            del peer.outgoing[oldest]
            peer.queued_audio -= 1
        peer.outgoing.append(message)
        if audio:
            peer.queued_audio += 1
        else:
            peer.queued_controls += 1

    def join(self, info: cp.PeerInfo, rate: int) -> None:
        with self._condition:
            self._live()
            _rate(rate)
            if info.role not in cp.PEER_ROLES or (info.skill and info.role == cp.PEER_OWNER):
                raise CallPeerError("peer_role_forbidden")
            if info.peer_id in self._peers:
                raise CallPeerError("peer_already_joined")
            if len(self._peers) >= MAX_PEERS:
                raise CallPeerError("peer_capacity_full")
            self._event(cp.PeerJoined(info))
            peer = _Peer(info, rate, LinearResampler(self._output_rate, rate))
            self._peers[info.peer_id] = peer
            self._send(peer, cp.encode_control(self._state))
            self._send(peer, cp.encode_control({"type": "flush", "epoch": self._epoch}))
            if self._turn["turn"] != cp.TURN_IDLE:  # a surface starts idle: only another turn is news
                self._send(peer, cp.encode_control(self._turn))

    def leave(self, peer_id: str, reason: str = "") -> None:
        with self._condition:
            peer = self._peers.pop(peer_id, None)
            if peer is not None:
                if any(track.samples for track in peer.tracks.values()):
                    self._draining_peers.append(peer)
                self._video_pending.pop(peer_id, None)
                # Reserve MAX_PEERS extra lifecycle slots so a full control queue
                # can never suppress the disconnect that frees its producer.
                self._events.append(cp.PeerLeft(peer_id, reason))
                self._condition.notify_all()

    def input_audio_evidence(self) -> dict:
        """Discard totals in samples at their discard rate and milliseconds, by cause."""
        with self._condition:
            return {"dropped_samples": self._input_dropped_samples,
                    "dropped_ms": self._input_dropped_ms,
                    "drop_causes": {cause: dict(counts) for cause, counts in self._input_drop_causes.items()}}

    def _drop_input(self, cause, samples, rate):
        if not samples:
            return
        milliseconds = samples * 1000 / rate
        self._input_dropped_samples += samples
        self._input_dropped_ms += milliseconds
        counts = self._input_drop_causes.setdefault(cause, {"samples": 0, "ms": 0.0})
        counts["samples"] += samples
        counts["ms"] += milliseconds

    def _discard_tracks(self, peer, cause):
        self._drop_input(cause, sum(len(track.samples) for track in peer.tracks.values()), self._input_rate)
        peer.tracks.clear()

    def _audio_peers(self):
        return (*self._peers.values(), *self._draining_peers)

    def pause_input(self):
        """Retain capture while the replacement provider has not declared Ready."""
        with self._condition:
            self._input_paused = True

    def receive_audio(self, peer_id: str, frame: cp.AudioFrame) -> bool:
        with self._condition:
            if self._closed:
                self._drop_input("call_closed", len(frame.pcm) // 2, frame.rate)
            self._live()
            peer = self._peers[peer_id]
            if not peer.info.mic or frame.kind != cp.FRAME_AUDIO_IN:
                self._drop_input("audio_direction_forbidden", len(frame.pcm) // 2, peer.rate)
                raise CallPeerError("audio_direction_forbidden")
            if frame.rate != peer.rate:
                self._drop_input("audio_rate_mismatch", len(frame.pcm) // 2, peer.rate)
                raise CallPeerError("audio_rate_mismatch")
            if len(frame.pcm) > cp.MAX_FRAME_PAYLOAD or len(frame.pcm) % 2:
                self._drop_input("invalid_pcm", len(frame.pcm) // 2, peer.rate)
                raise CallPeerError("invalid_pcm")
            delta = (frame.seq - peer.input_seq) & 0xFFFFFFFF if peer.input_seq is not None else 1
            if frame.epoch != self._epoch:
                self._drop_input("stale_epoch", len(frame.pcm) // 2, peer.rate)
                return False
            if not 0 < delta < 0x80000000:
                self._drop_input("sequence", len(frame.pcm) // 2, peer.rate)
                return False
            if peer.muted:
                peer.input_seq = frame.seq
                self._drop_input("muted", len(frame.pcm) // 2, peer.rate)
                return False
            track = peer.tracks.get(frame.stream)
            limit = self._input_rate * MIX_FRAME_MS * self._audio_limit // 1000
            if track is not None and len(track.samples) >= limit:
                # The socket retains this frame and retries; neither sequence nor
                # resampler advances until there is room. At most one whole
                # bounded input frame extends beyond the queue threshold.
                raise CallPeerError("input_backpressure")
            if track is None:
                if len(peer.tracks) >= MAX_STREAMS:
                    self._drop_input("stream_capacity_full", len(frame.pcm) // 2, peer.rate)
                    raise CallPeerError("stream_capacity_full")
                track = _Track(LinearResampler(frame.rate, self._input_rate))
                peer.tracks[frame.stream] = track
            peer.input_seq = frame.seq
            if not any(t.samples for p in self._audio_peers() for t in p.tracks.values()):
                self._mix_at = time.monotonic() + MIX_FRAME_MS / 1000
            track.samples.extend(_samples(track.converter.feed(frame.pcm)))
            self._condition.notify_all()
            return True

    def receive_video(self, peer_id: str, frame: cp.VideoFrame) -> bool:
        from ouroboros.gateways.image_api import sniff_image_mime

        with self._condition:
            self._live()
            peer = self._peers[peer_id]
            if not peer.info.video or not isinstance(frame, cp.VideoFrame):
                raise CallPeerError("video_direction_forbidden")
            if (not 0 < len(frame.data) <= cp.MAX_VIDEO_FRAME_PAYLOAD
                    or sniff_image_mime(frame.data) != frame.mime):
                raise CallPeerError("invalid_video_frame")
            delta = (frame.seq - peer.video_seq) & 0xFFFFFFFF if peer.video_seq is not None else 1
            if frame.epoch != self._epoch or not 0 < delta < 0x80000000:
                return False
            peer.video_seq = frame.seq
            self._video_pending.pop(peer_id, None)
            self._video_pending[peer_id] = cp.PeerVideo(peer_id, peer.info.role, frame.mime, frame.data)
            self._condition.notify_all()
            return True

    def receive_control(self, peer_id: str, message: dict) -> None:
        with self._condition:
            self._live()
            peer = self._peers[peer_id]
            kind = message.get("type")
            if kind not in cp.PEER_MESSAGE_TYPES or kind == "hello":
                raise CallPeerError("invalid_call_control")
            if kind == "played" and not peer.info.speaker:
                raise CallPeerError("audio_direction_forbidden")
            self._event(cp.PeerControl(peer_id, peer.info.role, dict(message)))
            if kind == "bye" and peer.info.role == cp.PEER_OWNER:
                self._owner_bye = True
            if kind == "mute":
                peer.muted = message["muted"]
                if peer.muted:
                    self._discard_tracks(peer, "muted")

    def _mix(self) -> cp.PeerAudio | None:
        if self._input_paused:
            return None
        tracks = [(p, t) for p in self._audio_peers() for t in p.tracks.values() if t.samples]
        if not tracks or time.monotonic() < self._mix_at:
            return None
        self._mix_at += MIX_FRAME_MS / 1000
        length = min(self._input_rate * MIX_FRAME_MS // 1000, max(len(t.samples) for _, t in tracks))
        mixed = [0] * length
        for _, track in tracks:
            consumed = min(length, len(track.samples))
            for index in range(consumed):
                mixed[index] += track.samples[index]
            del track.samples[:consumed]
        self._draining_peers[:] = [peer for peer in self._draining_peers if any(t.samples for t in peer.tracks.values())]
        peers = {p.info.peer_id for p, _ in tracks}
        role = cp.PEER_MEETING if any(p.info.role == cp.PEER_MEETING for p, _ in tracks) else cp.PEER_OWNER
        return cp.PeerAudio(next(iter(peers)) if len(peers) == 1 else "mixed", role,
                            _pcm(array("h", (max(-32768, min(32767, v)) for v in mixed))))

    def next_event(self, timeout_s: float):
        deadline = time.monotonic() + max(0, timeout_s)
        with self._condition:
            while True:
                if self._events:
                    return self._events.popleft()
                now = time.monotonic()
                if self._video_pending and now - self._last_video_at >= 1.0:
                    self._last_video_at = now
                    return self._video_pending.popitem(last=False)[1]
                audio = self._mix()
                if audio is not None:
                    return audio
                remaining = deadline - time.monotonic()
                if self._closed or remaining <= 0:
                    return None
                buffered = not self._input_paused and any(t.samples for p in self._audio_peers() for t in p.tracks.values())
                wake_in = max(0, self._mix_at - time.monotonic()) if buffered else remaining
                if self._video_pending:
                    wake_in = min(wake_in, max(0, self._last_video_at + 1.0 - time.monotonic()))
                self._condition.wait(min(remaining, wake_in))

    def next_outgoing(self, peer_id: str) -> bytes | str | None:
        """A nonblocking socket poll; the hub owns all queue and epoch checks."""
        with self._condition:
            peer = self._peers.get(peer_id)
            if peer is None:
                raise CallPeerError("peer_left")
            if peer.failure:
                raise CallPeerError(peer.failure)
            if not peer.outgoing:
                return None
            message = peer.outgoing.popleft()
            if isinstance(message, bytes):
                peer.queued_audio -= 1
            else:
                peer.queued_controls -= 1
            return message

    def set_engine_rates(self, input_rate: int, output_rate: int) -> None:
        _rate(input_rate), _rate(output_rate)
        with self._condition:
            self._input_paused = False
            self._condition.notify_all()
            if (input_rate, output_rate) == (self._input_rate, self._output_rate):
                return
            previous_rate = self._input_rate
            self._input_rate, self._output_rate = input_rate, output_rate
            for peer in self._audio_peers():
                if input_rate != previous_rate:
                    for track in peer.tracks.values():
                        track.samples = _samples(LinearResampler(previous_rate, input_rate).feed(_pcm(track.samples)))
                        track.converter = LinearResampler(peer.rate, input_rate)
                peer.converter = LinearResampler(output_rate, peer.rate)
            self.flush()

    def play(self, pcm: bytes, item_id: str) -> None:
        _samples(pcm)
        with self._condition:
            if self._closed:
                return
            for peer in self._peers.values():
                if not peer.info.speaker or peer.failure:
                    continue
                converted = peer.converter.feed(pcm)
                size = min(cp.MAX_FRAME_PAYLOAD, peer.rate * MIX_FRAME_MS // 1000 * 2)
                for offset in range(0, len(converted), size):
                    frame = cp.AudioFrame(cp.FRAME_AUDIO_OUT, self._epoch, peer.output_seq,
                                          peer.rate, 0, converted[offset:offset + size])
                    peer.output_seq = (peer.output_seq + 1) & 0xFFFFFFFF
                    self._send(peer, cp.encode_frame(frame))

    def flush(self) -> int:
        with self._condition:
            self._epoch = (self._epoch + 1) & 0xFFFF
            for peer in self._peers.values():
                peer.outgoing = deque(m for m in peer.outgoing if isinstance(m, str)
                                      and cp.decode_control(m)["type"] != "flush")
                peer.converter = LinearResampler(self._output_rate, peer.rate)
                peer.outgoing.appendleft(cp.encode_control({"type": "flush", "epoch": self._epoch}))
                peer.queued_audio, peer.queued_controls = 0, len(peer.outgoing)
            self._video_pending.clear()
            return self._epoch

    def _broadcast(self, message: dict) -> None:
        text = cp.encode_control(message)
        if len(text.encode("utf-8")) > cp.MAX_CONTROL_BYTES:
            raise CallPeerError("control_too_large")
        for peer in self._peers.values():
            self._send(peer, text)

    def send_state(self, state: str, reason: str = "", *, phase: str = "", detail: str = "",
                   action: str = "") -> None:
        if state not in cp.CALL_STATES:
            raise ValueError("unknown call state")
        extra = {key: value for key, value in (("phase", phase), ("detail", detail), ("action", action)) if value}
        with self._condition:
            self._state = dict(type="state", state=state, reason=reason, engine=self.spec.engine, **extra)
            self._broadcast(self._state)

    def send_activity(self, activity: str, label: str = "") -> None:
        """A tool batch runs (``working``, with the tool's own label) or ended (``idle``)."""
        if activity not in ("working", "idle"):
            raise ValueError("unknown call activity")
        with self._condition:
            self._broadcast(dict(type="activity", activity=activity, label=label))

    def send_turn(self, turn: str) -> None:
        """Where the turn is: the page shows Thinking while an answer is awaited and holds Speaking
        across the pauses of one answer, instead of reading phases off the audio's gaps."""
        if turn not in cp.CALL_TURNS:
            raise ValueError("unknown call turn")
        with self._condition:
            if self._turn["turn"] != turn:
                self._turn = {"type": "turn", "turn": turn}
                self._broadcast(self._turn)

    def send_caption(self, speaker: str, text: str, final: bool, item_id: str = "", *, replace: bool = False) -> None:
        if speaker not in ("owner", "ouroboros"):
            raise ValueError("unknown caption speaker")
        with self._condition:
            self._broadcast(dict(type="caption", speaker=speaker, text=text, final=final,
                                 item_id=item_id, replace=replace))

    def peers(self) -> tuple[cp.PeerInfo, ...]:
        with self._condition:
            return tuple(peer.info for peer in self._peers.values())

    def close(self, reason: str, *, detail: str = "", action: str = "") -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._grants.clear()
            for peer in self._audio_peers():
                self._discard_tracks(peer, "call_closed")
            self._draining_peers.clear()
            self.flush()
            if self._state.get("state") != cp.STATE_CLOSED:  # the driver's own closed state (and words) stands
                self.send_state(cp.STATE_CLOSED, reason, detail=detail, action=action)
            self._condition.notify_all()


_HUBS: dict[str, LiveCallHub] = {}
_REGISTRY_LOCK = threading.Lock()


def open_hub(spec: cp.CallSpec) -> LiveCallHub:
    """Idempotent for one admitted spec so the door and driver share a hub."""
    with _REGISTRY_LOCK:
        hub = _HUBS.get(spec.call_id)
        if hub is not None:
            if hub.spec != spec or hub.closed:
                raise CallPeerError("call_id_conflict")
            return hub
        hub = LiveCallHub(spec)
        _HUBS[spec.call_id] = hub
        return hub


def get_hub(call_id: str) -> LiveCallHub | None:
    with _REGISTRY_LOCK:
        return _HUBS.get(call_id)


def _owner_hub_locked(chat_id: int) -> LiveCallHub | None:
    for hub in _HUBS.values():
        if not hub.closed and hub.spec.chat_id == chat_id and hub.spec.opened_by == cp.OPENED_BY_OWNER:
            return hub
    return None


def open_owner_hub(spec: cp.CallSpec) -> tuple[LiveCallHub, bool]:
    """The room's one owner call in one step: ``(open hub, False)`` to join, else ``(spec's new hub, True)``."""
    with _REGISTRY_LOCK:
        hub = _owner_hub_locked(spec.chat_id)
        if hub is not None:
            return hub, False
        hub = _HUBS[spec.call_id] = LiveCallHub(spec)
        return hub, True


def close_hub(call_id: str, reason: str = "finished", **words: str) -> None:
    """Close and forget a hub; ``words`` are ``detail``/``action`` for its closed state."""
    with _REGISTRY_LOCK:
        hub = _HUBS.pop(call_id, None)
    if hub is not None:
        hub.close(reason, **words)

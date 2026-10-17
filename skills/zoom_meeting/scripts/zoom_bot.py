"""zoom_bot: the companion that takes this install into Zoom meetings.

Host-supervised, started when the skill is enabled and idle until a zoom_meeting tool
asks it to join. It owns the Zoom Meeting SDK on its main thread under the GLib main
loop (``glib_loop``); the tools reach it through the control channel in the skill's
state dir (``companion_link``). Between a join and a leave it:

1. runs Hope's SDK sequence (``zoom_session``) and publishes every state as it is,
   including the wait for the meeting host to allow raw audio;
2. when meeting audio flows, asks the Host door to open the Presence call and attaches
   as the call's ``meeting`` peer (``meeting_peer``);
3. carries audio both ways, plus who Zoom is delivering audio for and the meeting chat
   as peer-observed facts;
4. on zoom_leave, the end of the meeting, removal by the host or the end of the call,
   says ``bye``, closes the socket, leaves the meeting and cleans the SDK up.

A crash or a restart never re-joins by itself: a new companion starts idle and reports
what the previous one was doing when it stopped. Off Linux it starts idle too, with the
platform as the reason every join is refused, rather than crashing into its restart
limit. At start it checks Zoom's raw-audio prerequisites in its own environment
(``raw_audio``) and publishes the result with its state.
"""

from __future__ import annotations

import collections
import contextlib
import importlib
import logging
import os
import re
import secrets
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import companion_link as link
import glib_loop
import meeting_peer as peer_mod
import raw_audio
from zoom_session import JoinRequest, SessionTiming, ZoomSession

log = logging.getLogger("zoom_bot")

DEFAULT_NAME = "Ouroboros"
SWEEP_INTERVAL_MS = 200
HEARTBEAT_MS = 250
# The host stops a companion within seconds; leaving must finish inside that window.
SHUTDOWN_GRACE_MS = 4000
_BINDING_ID = re.compile(r"^[0-9a-f]{32}$")

_STATE_TEXT = {
    "idle": "not in a Zoom meeting",
    "joining": "joining",
    "waiting_for_host": "joining; waiting for the host to start the meeting",
    "waiting_room": "in the waiting room; the host has to admit {name}",
    "in_meeting": "in the meeting, setting up audio",
    "waiting_for_audio": "in the meeting, waiting for the host to allow {name}'s audio",
    "audio_ready": "in the meeting, opening the call",
    "in_call": "in the call",
    "reconnecting": "in the meeting; Zoom is reconnecting",
    "leaving": "leaving the meeting",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SecretRedactor(logging.Filter):
    """Replace known secret values in every log line, tracebacks included.

    The companion never logs a secret on purpose; this is the second line for a value
    that reaches a message through an exception or a library.
    """

    def __init__(self) -> None:
        super().__init__()
        self._values: set = set()
        self._lock = threading.Lock()

    def add(self, value: str) -> None:
        if isinstance(value, str) and len(value) >= 4:
            with self._lock:
                self._values.add(value)

    def scrub(self, text: str) -> str:
        with self._lock:
            values = sorted(self._values, key=len, reverse=True)
        for value in values:
            text = text.replace(value, "[redacted]")
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + logging.Formatter().formatException(record.exc_info)
            record.exc_info = None
            record.exc_text = None
        record.msg, record.args = self.scrub(message), None
        return True


class ZoomBot:
    """State, control requests and the call around one ``ZoomSession`` at a time."""

    def __init__(self, *, state_dir: Path, loop: Any, sdk: Any, sdk_error: str = "",
                 client_id: str = "", client_secret: str = "", host_url: str = "", host_token: str = "",
                 timing: SessionTiming = SessionTiming(), clock: Callable[[], float] = time.monotonic,
                 open_call: Callable[..., Dict[str, str]] = peer_mod.open_call,
                 attach_peer: Callable[..., Any] = peer_mod.PeerSession.attach,
                 identity_name: Callable[[str, str], str] = peer_mod.identity_name,
                 pacer_factory: Optional[Callable[..., Any]] = None,
                 register_secret: Callable[[str], None] = lambda _value: None,
                 audio: Optional[Dict[str, str]] = None) -> None:
        self._state_dir = Path(state_dir)
        self._loop = loop
        self._sdk = sdk if loop is not None else None
        self._sdk_error = sdk_error
        self._client_id, self._client_secret = client_id.strip(), client_secret.strip()
        self._host_url, self._host_token = host_url, host_token
        self._timing = timing
        self._clock = clock
        self._open_call = open_call
        self._attach_peer = attach_peer
        self._identity_name = identity_name
        self._pacer_factory = pacer_factory or (lambda send, rate: peer_mod.OutboundPacer(send, rate=rate))
        self._register_secret = register_secret
        self._audio = dict(audio or {})
        self._account_id = link.app_account_id(self._client_id) if self._client_id else ""
        self._lock = threading.RLock()
        self._publish_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._shutting_down = False
        self.state = "idle"
        self._since = _now_iso()
        self._join: Dict[str, Any] = {}
        self._session: Optional[ZoomSession] = None
        self._before_reconnect = ""
        self._peer: Any = None
        self._pacer: Any = None
        self._pacer_stop: Optional[threading.Event] = None
        self._tracker: Optional[peer_mod.ParticipantTracker] = None
        self._call_opening = False
        self._call_id = ""
        self._audio_rate = 0
        self._mic_sender: Any = None
        self._mic_sending = False
        self._stats: collections.Counter = collections.Counter()
        self._threads: list = []
        self._retired_peers: list = []
        self.last_outcome: Optional[Dict[str, Any]] = None
        self._recover_previous_status()
        self._publish()

    # ---- control channel ------------------------------------------------------

    def handle_control(self, op: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if op == "status":
            return {"ok": True, **self.snapshot()}
        if op == "join":
            return self._control_join(args)
        if op == "leave":
            return self._control_leave()
        return {"ok": False, "code": "ZOOM_UNKNOWN_OP", "error": f"unknown companion request {op!r}"}

    def _refusal(self, code: str, error: str) -> Dict[str, Any]:
        return {"ok": False, "code": code, "error": error, **self.snapshot()}

    def _join_refusal(self, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._shutdown.is_set():
            return self._refusal("ZOOM_STOPPING", "the Zoom companion is stopping")
        if self._sdk is None:
            return self._refusal("ZOOM_SDK_UNAVAILABLE", self._unavailable_reason())
        if not (self._client_id and self._client_secret):
            return self._refusal("ZOOM_CREDENTIALS_MISSING",
                                 "Zoom app credentials are not set in the companion's environment")
        if str(args.get("account_id") or "") != self._account_id:
            return self._refusal("ZOOM_COMPANION_STALE",
                                 "the Zoom app credentials changed after the companion started")
        if not str(args.get("meeting_number") or "").isdigit():
            return self._refusal("ZOOM_MEETING_INVALID", "the meeting number must be digits")
        if not _BINDING_ID.match(str(args.get("binding_id") or "")):
            return self._refusal("ZOOM_BINDING_INVALID", "the binding id must be a Presence binding id")
        if self.state != "idle":
            snapshot = self.snapshot()
            return self._refusal("ZOOM_BUSY", f"{snapshot['message']} (meeting {snapshot['meeting_number']})")
        return None

    def _control_join(self, args: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            refusal = self._join_refusal(args)
        if refusal:
            return refusal
        name = str(args.get("display_name") or "").strip()
        if not name:
            name = self._identity_name(self._host_url, self._host_token) or DEFAULT_NAME
        request = JoinRequest(meeting_number=int(args["meeting_number"]), display_name=name,
                              passcode=str(args.get("passcode") or ""),
                              on_behalf_token=str(args.get("on_behalf_token") or ""))
        self._register_secret(request.passcode)
        self._register_secret(request.on_behalf_token)
        requested_by = args.get("requested_by") if isinstance(args.get("requested_by"), dict) else {}
        with self._lock:
            refusal = self._join_refusal(args)  # another request may have won the race meanwhile
            if refusal:
                return refusal
            self._join = {
                "join_id": secrets.token_hex(8), "meeting_number": str(args["meeting_number"]),
                "display_name": name, "binding_id": str(args["binding_id"]), "topic": "",
                "requested_by": {str(k): str(v) for k, v in requested_by.items()}, "requested_at": _now_iso(),
            }
            session = ZoomSession(self._sdk, self._loop, self, timing=self._timing,
                                  register_secret=self._register_secret)
            self._session = session
            self._set_state("joining")
        self._publish()
        self._loop.idle_add(lambda: session.start(request, self._client_id, self._client_secret))
        return {"ok": True, **self.snapshot()}

    def _control_leave(self) -> Dict[str, Any]:
        with self._lock:
            if self._session is None:
                return {"ok": True, "left": False, **self.snapshot()}
            self._set_state("leaving")
        self._publish()
        self._loop.idle_add(lambda: self._leave("left the meeting"))
        return {"ok": True, "left": True, **self.snapshot()}

    # ---- session listener (see ZoomSession) ------------------------------------

    def session_state(self, state: str) -> None:
        with self._lock:
            if state == "reconnecting":
                if self.state != "reconnecting":  # a repeated notice keeps the state to return to
                    self._before_reconnect = self.state
            elif state == "reconnected":
                state = self._before_reconnect or ("in_call" if self._peer is not None else "in_meeting")
            if self.state == "leaving" and state != "leaving":
                return
            if self.state == "in_call" and state in ("in_meeting", "audio_ready"):
                return
            if state == "in_meeting" and self._session is not None and self._session.topic:
                self._join["topic"] = self._session.topic
            self._set_state(state)
        self._publish()

    def mixed_audio(self, pcm: bytes, rate: int) -> None:
        with self._lock:
            peer = self._peer
            if peer is None:
                if self.state == "audio_ready" and not self._call_opening and not self._shutdown.is_set():
                    self._call_opening = True
                    self._audio_rate = int(rate)
                    self._start_thread(self._open_call_worker, "zoom-call-open")
                else:
                    self._stats["audio_buffers_before_call"] += 1
                return
        peer.send_audio(pcm, rate)

    def participant_audio(self, node_id: int) -> None:
        tracker = self._tracker
        if tracker is not None:
            tracker.heard(node_id, self._clock())

    def mic_ready(self, sender: Any) -> None:
        with self._lock:
            self._mic_sender = sender

    def mic_sending(self, sending: bool) -> None:
        with self._lock:
            self._mic_sending = bool(sending)

    def chat(self, sender: str, text: str, to_all: bool) -> None:
        audience = "to everyone" if to_all else "privately to you"
        # The peer fits a long message to the protocol's control limit with a named omission.
        self._context(f"Zoom chat from {sender or 'a participant'} ({audience}): {text}", "zoom_chat")

    def own_audio(self, muted_by_host: bool) -> None:
        self._context("The meeting host muted you; nobody in the meeting hears you until you are unmuted."
                      if muted_by_host else "The meeting host unmuted you; the meeting can hear you again.",
                      "zoom")

    def ended(self, outcome: str) -> None:
        self._stop_call(outcome)
        with self._lock:
            join = self._join
            self.last_outcome = {"message": outcome, "at": _now_iso(), "join_id": join.get("join_id", ""),
                                 "meeting_number": join.get("meeting_number", "")}
            self._session = None
            self._join = {}
            self._mic_sender, self._mic_sending = None, False
            self._set_state("idle")
        log.info("session ended: %s", outcome)
        self._publish()
        if self._shutting_down:
            self._loop.quit()

    # ---- the call ---------------------------------------------------------------

    def _context(self, text: str, source: str) -> None:
        peer = self._peer
        if peer is None:
            self._stats["facts_before_call"] += 1
            return
        peer.send_control({"type": "context", "text": text, "source": source})

    def call_open_body(self, join: Dict[str, Any], rate: int) -> Dict[str, Any]:
        meeting, topic = join["meeting_number"], join.get("topic") or ""
        label = peer_mod.bounded_text(f"Zoom meeting {topic or meeting}", 200)
        event = {
            "source_event_id": f"zoom-join-{join['join_id']}",
            "provider": link.PROVIDER,
            "account_id": self._account_id,
            "conversation_id": meeting,
            "thread_id": "",
            "conversation_key": link.conversation_key(self._account_id, meeting),
            "actor": dict(join.get("requested_by") or {}) or {"id": "unknown"},
            "conversation": {"kind": "zoom_meeting", "meeting_number": meeting, "topic": topic,
                             "participant_name": join["display_name"]},
            "message": {"kind": "meeting_joined", "joined_at": _now_iso(), "audio_rate": rate},
            "text": f"Joined the Zoom meeting {topic or meeting}",
        }
        return {"binding_id": join["binding_id"], "role": "meeting", "event": event, "label": label,
                "mic": True, "speaker": True}

    def _open_call_worker(self) -> None:
        with self._lock:
            join, rate = dict(self._join), self._audio_rate
        if not join:
            return
        body = self.call_open_body(join, rate)
        name = join["display_name"]
        try:
            opened = self._open_call(self._host_url, self._host_token, body)
            hello = {"type": "hello", "protocol": peer_mod.PROTOCOL_VERSION, "role": "meeting",
                     "mic": True, "speaker": True, "rate": rate, "label": body["label"]}
            peer = self._attach_peer(self._host_url, self._host_token, opened["peer_path"], hello,
                                     on_audio=self._on_call_audio, on_flush=self._on_call_flush,
                                     on_closed=self._on_call_closed)
        except peer_mod.CallOpenRefused as exc:
            self._leave_later(join, f"the call could not open: {exc.message}; {name} left the meeting")
            return
        except Exception as exc:
            log.exception("attaching to the call failed")
            self._leave_later(join, f"the call could not open: {type(exc).__name__}; {name} left the meeting")
            return
        with self._lock:
            # Zoom may reconnect while the call opens: the call still belongs to this meeting.
            reconnecting = self.state == "reconnecting" and self._before_reconnect == "audio_ready"
            if self._join.get("join_id") != join["join_id"] or not (self.state == "audio_ready" or reconnecting):
                peer.close("the meeting ended before the call started")
                return
            self._peer, self._call_id = peer, opened["call_id"]
            self._tracker = peer_mod.ParticipantTracker()
            self._pacer_stop = threading.Event()
            self._pacer = self._pacer_factory(self._mic_send, rate)
            self._start_thread(lambda pacer=self._pacer, stop=self._pacer_stop: pacer.run(stop), "zoom-voice")
            if reconnecting:
                self._before_reconnect = "in_call"  # the reconnect ends in the open call
            else:
                self._set_state("in_call")
        log.info("call %s open", opened["call_id"])
        self._publish()
        self._loop.idle_add(self._start_sweeps)

    def _start_thread(self, target: Callable[[], Any], name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        self._threads = [alive for alive in self._threads if alive.is_alive()] + [thread]
        thread.start()

    def _leave_later(self, join: Dict[str, Any], outcome: str) -> None:
        def run() -> bool:
            if self._join.get("join_id") == join.get("join_id"):
                self._leave(outcome)
            return False

        self._loop.idle_add(run)

    def _mic_send(self, frame: bytes) -> None:
        with self._lock:
            sender, pacer = self._mic_sender, self._pacer
        if sender is None or pacer is None:
            self._stats["voice_frames_without_microphone"] += 1
            return
        sender.send(frame, pacer.rate, self._sdk.ZoomSDKAudioChannel_Mono)

    def _on_call_audio(self, pcm: bytes, epoch: int) -> None:
        pacer = self._pacer
        if pacer is not None:
            pacer.feed(pcm, epoch)

    def _on_call_flush(self, epoch: int) -> None:
        pacer = self._pacer
        if pacer is not None:
            pacer.flush(epoch)

    def _on_call_closed(self, reason: str) -> None:
        with self._lock:
            join = dict(self._join)
        if join:
            self._leave_later(join, f"the call ended ({reason}); {join['display_name']} left the meeting")

    def _start_sweeps(self) -> bool:
        self._loop.timeout_add(SWEEP_INTERVAL_MS, self._sweep)
        return False

    def _sweep(self) -> bool:
        with self._lock:
            tracker, peer, session = self._tracker, self._peer, self._session
        if tracker is None or peer is None or session is None:
            return False
        for node_id, speaking in tracker.sweep(self._clock()):
            peer.send_control({"type": "participant", "label": session.participant_name(node_id),
                               "speaking": speaking})
        return True

    def _stop_call(self, reason: str) -> None:
        with self._lock:
            peer, pacer, stop = self._peer, self._pacer, self._pacer_stop
            self._peer = self._pacer = self._pacer_stop = self._tracker = None
            self._call_opening = False
            self._call_id = ""
        if peer is not None:
            peer.close(reason)
            self._retired_peers = [p for p in self._retired_peers if p is not peer][-3:] + [peer]
        if stop is not None:
            stop.set()
        if pacer is not None:
            pacer.close()

    def _leave(self, outcome: str) -> bool:
        """On the loop thread: end the call first, then leave the meeting."""
        with self._lock:
            session = self._session
            if session is not None:
                self._set_state("leaving")
        self._stop_call(outcome)
        self._publish()
        if session is not None:
            session.leave(outcome)
        return False

    # ---- lifecycle ------------------------------------------------------------

    def request_shutdown(self) -> None:
        """Signal-safe: the heartbeat on the loop thread does the work."""
        self._shutdown.set()

    def heartbeat(self) -> bool:
        if not self._shutdown.is_set() or self._shutting_down:
            return True
        self._shutting_down = True
        with self._lock:
            session, name = self._session, self._join.get("display_name", DEFAULT_NAME)
        if session is None:
            self._loop.quit()
            return False
        self._leave(f"the Zoom companion stopped; {name} left the meeting")
        self._loop.timeout_add(SHUTDOWN_GRACE_MS, self._force_quit)
        return False

    def _force_quit(self) -> bool:
        session = self._session
        if session is not None:
            session.cleanup()
        self._loop.quit()
        return False

    def wait_for_shutdown(self) -> None:
        while not self._shutdown.wait(0.5):
            pass

    def close(self, timeout: float = 2.0) -> None:
        """Last words after the loop stopped: the SDK is cleaned up whatever happened."""
        session = self._session
        if session is not None and not session.ended:
            self._stop_call("the Zoom companion stopped")
            session.cleanup()
        for peer in list(self._retired_peers):
            if hasattr(peer, "join"):
                peer.join(timeout)
        for thread in list(self._threads):
            if thread is not threading.current_thread():
                thread.join(timeout)
        self._publish()

    # ---- status ---------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state, self._since = state, _now_iso()

    def _unavailable_reason(self) -> str:
        return self._sdk_error or "the Zoom Meeting SDK is not loaded"

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            join = dict(self._join)
            name = join.get("display_name") or DEFAULT_NAME
            if self._sdk is None and self.state == "idle":
                message = f"this host cannot run the Zoom Meeting SDK: {self._unavailable_reason()}"
            else:
                message = _STATE_TEXT.get(self.state, self.state).format(name=name)
            pacer, peer = self._pacer, self._peer
            stats = dict(self._stats)
            if pacer is not None:
                stats.update(voice_frames_sent=getattr(pacer, "sent_frames", 0),
                             voice_bytes_flushed=getattr(pacer, "flushed_bytes", 0))
            if peer is not None:
                stats.update(audio_frames_up=getattr(peer, "sent_frames", 0),
                             audio_frames_down=getattr(peer, "received_frames", 0),
                             audio_buffers_dropped_up=getattr(peer, "dropped_uplink", 0),
                             audio_buffers_stale_up=getattr(peer, "stale_uplink", 0),
                             facts_dropped_up=getattr(peer, "dropped_controls", 0))
            return {
                "state": self.state, "message": message, "since": self._since,
                "meeting_number": join.get("meeting_number", ""), "display_name": join.get("display_name", ""),
                "topic": join.get("topic", ""), "binding_id": join.get("binding_id", ""),
                "join_id": join.get("join_id", ""), "call_id": self._call_id,
                "last_outcome": self.last_outcome, "account_id": self._account_id,
                "credentials": bool(self._client_id and self._client_secret),
                "sdk": {"available": self._sdk is not None, "detail": "" if self._sdk is not None
                        else self._unavailable_reason(),
                        "loop": getattr(self._loop, "backend", "")},
                "microphone": {"ready": self._mic_sender is not None, "sending": self._mic_sending},
                "audio": dict(self._audio), "pid": os.getpid(), "stats": stats,
            }

    def _publish(self) -> None:
        """Write the current state; publishers take turns so the newest state lands last."""
        with self._publish_lock:
            try:
                link.write_status(self._state_dir, self.snapshot())
            except OSError:
                log.warning("status file not written")

    def _recover_previous_status(self) -> None:
        previous = link.read_status(self._state_dir) or {}
        if previous.get("pid") == os.getpid():
            return
        if previous.get("state") not in (None, "", "idle"):
            self.last_outcome = {
                "message": (f"the previous zoom_bot stopped while '{previous.get('message', '')}' "
                            f"(meeting {previous.get('meeting_number', '')}); it did not re-join"),
                "at": _now_iso(), "join_id": str(previous.get("join_id") or ""),
                "meeting_number": str(previous.get("meeting_number") or ""),
            }
        elif isinstance(previous.get("last_outcome"), dict):
            self.last_outcome = previous["last_outcome"]


def platform_refusal(platform: Optional[str] = None) -> str:
    """Why this host cannot run the Zoom Meeting SDK at all; ``""`` on Linux."""
    platform = sys.platform if platform is None else platform
    if platform.startswith("linux"):
        return ""
    return f"the Zoom Meeting SDK runs on Linux x86_64 only; this host is {platform}"


def acquire_singleton(state_dir: Path, *, timeout_sec: float = 8.0) -> int:
    """One companion generation owns the SDK; a stopping predecessor gets a few seconds.

    Linux only (POSIX ``fcntl``): elsewhere no SDK loads, so there is nothing to own.
    """
    import fcntl

    fd = os.open(str(state_dir / ".zoom_bot.lock"), os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise RuntimeError("another zoom_bot generation still holds the Zoom SDK")
            time.sleep(0.2)


def watch_parent(on_orphaned: Callable[[], None], *, grace_sec: float = 6.0) -> None:
    """Leave the meeting when the server that spawned us is gone; exit if leaving hangs."""
    parent = os.getppid()
    named = int(os.environ.get("OUROBOROS_PARENT_PID") or 0)

    def watch() -> None:
        while os.getppid() == parent and (not named or named == parent):
            time.sleep(0.5)
        on_orphaned()
        time.sleep(grace_sec)
        os._exit(0)

    threading.Thread(target=watch, name="zoom-bot-lifeline", daemon=True).start()


def load_sdk() -> tuple:
    try:
        return importlib.import_module("zoom_meeting_sdk"), ""
    except Exception as exc:  # ImportError, or a native library the wheel needs is missing
        return None, f"zoom_meeting_sdk is not importable ({type(exc).__name__}: {exc})"


def main() -> int:
    redactor = SecretRedactor()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s zoom_bot %(levelname)s %(message)s"))
    handler.addFilter(redactor)
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    state_raw = os.environ.get("OUROBOROS_SKILL_STATE_DIR", "")
    if not state_raw:
        log.error("OUROBOROS_SKILL_STATE_DIR is missing")
        return 2
    state_dir = Path(state_raw)
    state_dir.mkdir(parents=True, exist_ok=True)
    unsupported = platform_refusal()
    lock_fd = None
    if not unsupported:
        try:
            lock_fd = acquire_singleton(state_dir)
        except RuntimeError as exc:
            log.error("%s", exc)
            return 1
    client_secret = os.environ.get("ZOOM_CLIENT_SECRET", "")
    host_token = os.environ.get("HOST_SERVICE_TOKEN", "")
    redactor.add(client_secret)
    redactor.add(host_token)
    audio: Dict[str, str] = {}
    if not unsupported:
        # Before the SDK loads: the check sets XDG_RUNTIME_DIR the way libpulse will need it.
        status, detail = raw_audio.check(os.environ, home=os.environ.get("HOME", ""), uid=raw_audio.current_uid())
        audio = {"status": status, "detail": detail}
        if status != raw_audio.PASS:
            log.warning("raw meeting audio may not arrive: %s", detail)
    sdk, sdk_error = (None, unsupported) if unsupported else load_sdk()
    loop = None
    if sdk is not None:
        try:
            loop = glib_loop.make_loop()
        except glib_loop.GLibUnavailable as exc:
            sdk_error = str(exc)
    bot = ZoomBot(state_dir=state_dir, loop=loop, sdk=sdk, sdk_error=sdk_error,
                  client_id=os.environ.get("ZOOM_CLIENT_ID", ""), client_secret=client_secret,
                  host_url=os.environ.get("HOST_SERVICE_URL", ""), host_token=host_token,
                  register_secret=redactor.add, audio=audio)
    server = link.ControlServer(state_dir, bot.handle_control)
    server.start()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_args: bot.request_shutdown())
    watch_parent(bot.request_shutdown)
    log.info("zoom_bot ready (sdk %s, loop %s)", "loaded" if loop else "unavailable",
             getattr(loop, "backend", "none"))
    try:
        if loop is None:
            bot.wait_for_shutdown()
        else:
            loop.timeout_add(HEARTBEAT_MS, bot.heartbeat)
            loop.run()
    finally:
        bot.close()
        server.close()
        if lock_fd is not None:
            with contextlib.suppress(OSError):
                os.close(lock_fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())

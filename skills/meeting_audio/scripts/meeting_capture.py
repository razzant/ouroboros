"""Meeting capture companion: the one process that holds meeting audio devices.

The host starts it when the skill is enabled; it waits idle until a meeting tool
asks for something over the private control channel (``meeting_control``). On
``listen_start`` it captures the meeting application's sound and joins the
owner's current call as a ``meeting`` peer through the Host Service door; on
Linux ``speak_start`` adds the "Ouroboros" virtual microphone and plays the
call's output into it. Ouroboros's whispered hints otherwise reach only the
owner's own surface.

It never outlives its work holding a device: capture, speaking mode and any
module it loaded end when the call's socket closes, on ``listen_stop``, on its
own exit (SIGTERM from the host, or the server it belongs to disappearing), and a
crash leftover is unloaded at the next start.
"""

from __future__ import annotations

import os
import pathlib
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from capture_common import CaptureFailed  # noqa: E402
from meeting_apps import Selection, TargetRefused, choose, same_browser_note  # noqa: E402
from meeting_control import ControlServer  # noqa: E402
from meeting_peer import DoorPeer, PeerRefused  # noqa: E402

PARENT_PID_ENV = "OUROBOROS_PARENT_PID"
LIFELINE_SEC = 0.5
# The process that started this companion, read before anything else can delay it.
_START_PARENT = os.getppid()

# Door refusals in plain words; anything else is reported with its code.
_DOOR_REFUSALS = {
    "call_not_found": ("MEETING_CALL_GONE", "The call is no longer open, so there is nothing to join."),
    "call_peer_forbidden": ("MEETING_NOT_GRANTED",
                            "The host refused the meeting peer: this skill needs its reviewed call_peer grant."),
    "call_rate_limited": ("MEETING_DOOR_BUSY", "The host is limiting call attachments right now; try again shortly."),
    "peer_capacity_full": ("MEETING_DOOR_BUSY", "The call has no room for another audio peer."),
}


def _refusal(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"ok": False, "code": code, "message": message, **extra}


@dataclass
class ListenSession:
    call_id: str
    requested: str
    selection: Selection
    capture: Any
    peer: DoorPeer
    started_at: float = field(default_factory=time.time)
    voice: Any = None
    ending: bool = False


def load_backend(platform: str, state_dir: pathlib.Path) -> Any:
    if platform == "darwin":
        from capture_macos import MacBackend

        return MacBackend(state_dir)
    if platform == "win32":
        from capture_windows import WindowsBackend

        return WindowsBackend(state_dir)
    if platform.startswith("linux"):
        from capture_linux import LinuxBackend

        return LinuxBackend(state_dir)
    return None


class MeetingCompanion:
    """Operations behind the meeting tools; one listening session at a time."""

    def __init__(self, state_dir: pathlib.Path, host_url: str, host_token: str, backend: Any,
                 *, peer_factory: Callable[..., DoorPeer] = DoorPeer):
        self.state_dir = pathlib.Path(state_dir)
        self.host_url = host_url
        self.host_token = host_token
        self.backend = backend
        self.peer_factory = peer_factory
        self.stop_event = threading.Event()
        self.last_event = ""
        self.cleaned_up: List[int] = []
        self.permission_owner = ""
        self._lock = threading.RLock()
        self._operations = threading.Lock()
        self._session: Optional[ListenSession] = None
        self._control: Optional[ControlServer] = None

    # -- lifecycle ------------------------------------------------------------------
    def start(self) -> None:
        if self.backend is not None:
            try:
                self.cleaned_up = list(self.backend.cleanup_leftovers())
            except CaptureFailed as exc:
                self.last_event = f"could not clean up an earlier run: {exc.message}"
        if getattr(self.backend, "platform", "") == "darwin":
            self._record_attribution()
        self._control = ControlServer(self.state_dir, self.handle)
        self._control.start()

    def _record_attribution(self) -> None:
        """Who macOS will ask the owner about, read without prompting (see probe_capture)."""
        try:
            from probe_capture import attribution, write_record

            record = attribution()
            write_record(self.state_dir, record)
            self.permission_owner = str(record["responsible"].get("path") or "")
        except Exception as exc:  # a missing fact, never a reason not to start
            self.permission_owner = f"unknown ({type(exc).__name__})"

    def shutdown(self, reason: str = "the meeting companion stopped") -> None:
        with self._lock:
            session = self._session
        if session is not None:
            self._end(session, reason, tell_call=True)
        if self._control is not None:
            self._control.stop()
            self._control = None

    # -- control channel ---------------------------------------------------------------
    def handle(self, op: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if op == "status":
            return self.status()
        operations = {
            "listen_start": lambda: self.listen_start(str(args.get("call_id") or ""), str(args.get("app") or "")),
            "listen_stop": self.listen_stop,
            "speak_start": lambda: self.speak_start(str(args.get("call_id") or "")),
            "speak_stop": self.speak_stop,
        }
        if op not in operations:
            return _refusal("MEETING_CONTROL_REFUSED", f"unknown operation {op!r}")
        with self._operations:  # one change of devices and sockets at a time
            return operations[op]()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            session = self._session
            if session is None:
                message = "not listening"
                if self.last_event:
                    message += f"; last: {self.last_event}"
                return {"ok": True, "state": "idle", "message": message, **self._platform_facts()}
            facts = dict(session.capture.describe())
            speaking = session.voice is not None
            message = f"listening to {session.selection.label}"
            if speaking:
                message += " and speaking into the meeting through the 'Ouroboros' microphone"
            return {"ok": True, "state": "speaking" if speaking else "listening", "message": message,
                    "call_id": session.call_id, "apps": [app.name for app in session.selection.apps],
                    "rate": session.capture.rate, "since": round(session.started_at, 3), **facts,
                    "frames_sent": session.peer.stats["frames_sent"], **self._platform_facts()}

    def _platform_facts(self) -> Dict[str, Any]:
        return {"permission_owner": self.permission_owner} if self.permission_owner else {}

    def listen_start(self, call_id: str, app: str) -> Dict[str, Any]:
        if not call_id:
            return _refusal("MEETING_NO_CALL", "Listening needs the live call it belongs to.")
        if self.backend is None:
            return _refusal("MEETING_PLATFORM_UNSUPPORTED",
                            f"Listening to meetings is not available on {sys.platform}.")
        if not self.host_url or not self.host_token:
            return _refusal("MEETING_HOST_UNAVAILABLE", "The companion has no Host Service address or token.")
        with self._lock:
            current = self._session
            if current is not None and current.call_id == call_id and (not app or app == current.requested):
                return {**self.status(), "message": f"already listening to {current.selection.label}"}
        if current is not None:
            self._end(current, "replaced by a new listening request", tell_call=True)
        try:
            state, message = self.backend.permission()
            if state == "pending":  # never asked on this computer: macOS shows its prompt now
                self.backend.request_permission()
                return _refusal("MEETING_PERMISSION_PENDING", message)
            if state != "granted":  # asked before and still not allowed: no prompt will come
                return _refusal("MEETING_PERMISSION_DENIED", message)
            selection = choose(self.backend.running_apps(), app)
            capture = self.backend.open_capture(selection)
        except TargetRefused as exc:
            return _refusal(exc.code, exc.message, candidates=exc.candidates)
        except CaptureFailed as exc:
            return _refusal(exc.code, exc.message)
        return self._attach(call_id, app, selection, capture)

    def _attach(self, call_id: str, app: str, selection: Selection, capture: Any) -> Dict[str, Any]:
        """Join the call, then capture; a call that closes meanwhile leaves nothing running.

        The session exists before its socket does, so a close that arrives while
        joining or while the capture starts always finds it: ``_end`` marks it and
        stops the capture (a capture's stop is final, even mid-start), and every
        step here re-checks that mark before going on.
        """
        holder: Dict[str, ListenSession] = {}
        peer = self.peer_factory(
            self.host_url, self.host_token, call_id, rate=capture.rate,
            label=f"{selection.label} (meeting audio)",
            speaker_grant=bool(self.backend.speaking_supported),
            on_audio_out=lambda pcm, rate, epoch: self._audio_out(holder.get("session"), pcm, rate, epoch),
            on_flush=lambda epoch: self._flush(holder.get("session"), epoch),
            on_closed=lambda attachment, reason: self._peer_closed(holder.get("session"), attachment, reason),
        )
        session = ListenSession(call_id, app, selection, capture, peer)
        holder["session"] = session
        try:
            peer.open()
            peer.attach(speaker=False)
        except PeerRefused as exc:
            with self._lock:
                session.ending = True
            peer.close("refused")
            self._stop_capture(capture)
            code, message = _DOOR_REFUSALS.get(exc.code, ("MEETING_DOOR_REFUSED",
                                                          f"The host refused the meeting peer: {exc.message}"))
            return _refusal(code, message)
        with self._lock:
            if not session.ending:
                self._session = session
                self.last_event = ""
        if session.ending:
            return self._closed_while_joining(session)
        try:
            capture.start(lambda pcm, stream=0: peer.send_audio(pcm, stream),
                          lambda reason: self._later(self._capture_failed, session, reason))
        except Exception as exc:  # any failed start ends this session, so a retry starts afresh
            if session.ending:
                return self._closed_while_joining(session)
            failed = exc if isinstance(exc, CaptureFailed) else CaptureFailed(
                "MEETING_CAPTURE_FAILED", f"{type(exc).__name__}: {exc}")
            self._end(session, f"capture could not start: {failed.message}", tell_call=True)
            return _refusal(failed.code, failed.message)
        if session.ending:
            return self._closed_while_joining(session)
        # Disclosures describe what the capture really became (a fallback picks its source at start).
        for text in [*capture.disclosures(), same_browser_note(selection)]:
            if text:
                peer.send_context(text)
        return {**self.status(), "message": f"listening to {selection.label}"}

    def _closed_while_joining(self, session: ListenSession) -> Dict[str, Any]:
        session.peer.close("stopped")
        self._stop_capture(session.capture)
        return _refusal("MEETING_CALL_GONE", f"The call closed while listening started: {self.last_event}")

    def listen_stop(self) -> Dict[str, Any]:
        with self._lock:
            session = self._session
        if session is None:
            return {"ok": True, "state": "idle", "message": "not listening"}
        self._end(session, "stopped on request", tell_call=False)
        return {"ok": True, "state": "idle", "message": f"stopped listening to {session.selection.label}"}

    def speak_start(self, call_id: str) -> Dict[str, Any]:
        if self.backend is None or not self.backend.speaking_supported:
            return _refusal("MEETING_SPEAK_UNSUPPORTED", f"Speaking mode is not available on {sys.platform}.")
        with self._lock:
            session = self._session
        if session is None or (call_id and session.call_id != call_id):
            return _refusal("MEETING_NOT_LISTENING",
                            "Speaking into the meeting starts after listening: call meeting_listen_start first.")
        if session.voice is not None:
            return {**self.status(), "message": "already speaking into the meeting"}
        try:
            voice = self.backend.open_voice(session.capture.rate)
            voice.start(lambda reason: self._later(self._voice_failed, session, voice, reason))
        except CaptureFailed as exc:
            return _refusal(exc.code, exc.message)
        with self._lock:
            ended = session.ending
            if not ended:
                session.voice = voice
        if ended:  # listening ended while the microphone was being created
            voice.close()
            return _refusal("MEETING_NOT_LISTENING", f"Listening stopped meanwhile: {self.last_event}")
        try:
            session.peer.reattach(speaker=True)
        except PeerRefused as exc:
            with self._lock:
                if session.voice is voice:
                    session.voice = None
            voice.close()
            return self._restore_listening(session, f"the call refused the speaking direction: {exc.message}")
        return {**self.status(), "message": (
            "speaking into the meeting through the 'Ouroboros' microphone; the owner picks 'Ouroboros' "
            "as the microphone in the meeting application if it is not selected yet")}

    def speak_stop(self) -> Dict[str, Any]:
        with self._lock:
            session = self._session
        with self._lock:
            voice = session.voice if session is not None else None
            if voice is not None:
                session.voice = None
        if voice is None:
            return {**self.status(), "message": "not speaking into the meeting"}
        try:
            session.peer.reattach(speaker=False)
        except PeerRefused as exc:
            voice.close()
            return self._restore_listening(session, f"could not rejoin the call without speaking: {exc.message}")
        voice.close()
        return {**self.status(), "message": f"stopped speaking; still listening to {session.selection.label}"}

    def _restore_listening(self, session: ListenSession, reason: str) -> Dict[str, Any]:
        try:
            session.peer.reattach(speaker=False)
        except PeerRefused:
            self._end(session, reason, tell_call=False)
            return _refusal("MEETING_DOOR_REFUSED", f"{reason}; listening stopped as well")
        return _refusal("MEETING_DOOR_REFUSED", f"{reason}; still listening")

    # -- events from the peer socket and the capture -------------------------------------
    @staticmethod
    def _later(function: Callable[..., None], *args: Any) -> None:
        """Run off the socket/capture thread so it never waits on locks that join that thread."""
        threading.Thread(target=function, args=args, name="meeting-event", daemon=True).start()

    def _audio_out(self, session: Optional[ListenSession], pcm: bytes, rate: int, epoch: int) -> None:
        if session is not None and session.voice is not None:
            session.voice.play(pcm, rate, epoch)

    def _flush(self, session: Optional[ListenSession], epoch: int) -> None:
        if session is not None and session.voice is not None:
            session.voice.flush(epoch)

    def _peer_closed(self, session: Optional[ListenSession], attachment: int, reason: str) -> None:
        if session is None or attachment != session.peer.attachment:
            return
        if session.peer.state == "closed":  # the host announced the end of the call
            text = reason if reason.startswith("the call") else f"the call ended ({reason})"
        else:  # the call may go on; only this peer's socket is gone
            text = f"the host closed the meeting audio socket ({reason or 'no reason given'})"
        self._later(self._end, session, text, False)

    def _voice_failed(self, session: ListenSession, voice: Any, reason: str) -> None:
        """The player died: stop claiming speech, keep listening if the call takes it back."""
        with self._lock:
            if session.ending or session.voice is not voice:
                return
            session.voice = None
            self.last_event = f"speaking stopped: {reason}"
        voice.close()
        try:
            session.peer.reattach(speaker=False)
        except PeerRefused:
            self._end(session, f"speaking stopped ({reason}) and the call refused listening alone", tell_call=False)

    def _capture_failed(self, session: ListenSession, reason: str) -> None:
        self._end(session, f"capture stopped: {reason}", tell_call=True)

    def _end(self, session: ListenSession, reason: str, tell_call: bool) -> None:
        with self._lock:
            if session.ending:
                return
            session.ending = True
            if self._session is session:
                self._session = None
            self.last_event = f"{reason} (was listening to {session.selection.label})"
            voice, session.voice = session.voice, None
        if tell_call:
            session.peer.send_context(f"Meeting audio from {session.selection.label} stopped: {reason}.")
        session.peer.close("stopped")
        self._stop_capture(session.capture)
        if voice is not None:
            voice.close()

    @staticmethod
    def _stop_capture(capture: Any) -> None:
        try:
            capture.stop()
        except Exception:
            pass


def _lifeline_parent() -> int:
    """The process this companion belongs to, or 0 when it was gone before the companion started.

    The host names itself in ``OUROBOROS_PARENT_PID``. A container's server is PID 1,
    and PID 1 also adopts orphans, so a parent pid alone cannot tell a live server
    from an adoption; the name can: the parent at start must be the named one. A
    companion nobody named keeps refusing PID 1. Windows never re-parents a child,
    so there the named process itself is watched.
    """
    try:
        named = int(os.environ.get(PARENT_PID_ENV) or 0)
    except ValueError:
        named = 0
    if sys.platform == "win32":
        return named or _START_PARENT
    if named:
        return _START_PARENT if named == _START_PARENT else 0
    return _START_PARENT if _START_PARENT > 1 else 0


def _parent_alive(pid: int) -> bool:
    """Whether the parent from ``_lifeline_parent`` is still there.

    On POSIX it is gone once it is no longer this process's parent: an orphan is
    re-parented to init, launchd or a subreaper, so a PID-1 server stays a live parent.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":  # pragma: no cover - Windows only
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 0x102  # WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    return os.getppid() == pid


def _watch_parent(stop_event: threading.Event) -> None:
    parent = _lifeline_parent()

    def watch() -> None:
        while not stop_event.wait(LIFELINE_SEC):
            if not _parent_alive(parent):
                stop_event.set()
                return

    threading.Thread(target=watch, name="meeting-lifeline", daemon=True).start()


def main() -> int:
    state_dir = pathlib.Path(os.environ.get("OUROBOROS_SKILL_STATE_DIR") or ".").resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    backend = load_backend(sys.platform, state_dir)
    companion = MeetingCompanion(state_dir, os.environ.get("HOST_SERVICE_URL", ""),
                                 os.environ.get("HOST_SERVICE_TOKEN", ""), backend)
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: companion.stop_event.set())
    companion.start()
    _watch_parent(companion.stop_event)
    try:
        if backend is not None:
            backend.run_main_loop(companion.stop_event)
        else:
            companion.stop_event.wait()
    finally:
        companion.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

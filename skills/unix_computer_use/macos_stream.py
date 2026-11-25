"""One ScreenCaptureKit stream of one exact macOS window that keeps its latest complete frame.

``WindowSessions`` is what the task's MCP server (``window_session_server.py``) calls: ``begin`` selects one window
of one application once and starts one stream of it, ``frame`` hands out the held image with what the stream has said
since, ``release`` ends it. Nothing here runs at import; PyObjC and ScreenCaptureKit are touched by ``begin`` alone.

Selection is the one-shot capture's (``macos_capture``: the launch-free resolver, the window naming rule,
``select_window``, the prompt-free Screen Recording preflight, ``await_handler``) and happens once: the stream stays
bound to that window object, and a title is never looked up again. The stream is bounded: at most ``MAX_WIDTH`` x
``MAX_HEIGHT`` pixels (the screenshot default), ``FRAMES_PER_SECOND``, ``QUEUE_DEPTH`` surfaces, no cursor, no audio;
the store holds ONE sample buffer, the last that carried an image.

What a frame answer may claim. Each image has a ``seq`` and its own content time; the stream's later reports are
separate evidence with their own time (for both, the stream's display time, else the time of arrival, and the basis
is named; a display time that is no number on the host clock is not used). ``idle`` says the window had not changed
at that moment. Everything else that is not a readable image drops the held frame, since it may no longer be the
window's content: ``blank``, ``suspended``, ``stopped``, an image or sample this code cannot read, a callback that
failed (``callback_error``) and the stream's notice that its window closed (``inactive``:
``streamDidBecomeInactive:``, macOS 15.2+). A stream that ended, by ``release`` or by the system
(``stream:didStopWithError:``), gives no image again and is never restarted here. A frame dropped while its PNG was
being written stays dropped for that call: a later valid report cannot revalidate the image that was in flight.
Before starting capture, ``begin`` confirms that WindowServer lists the selected window for its owner PID.
A missing listing or an unreadable lookup starts nothing. The first subsequent confirmed disappearance ends
the session's delivery for good (``window_gone``) and asks the native stream to stop once — in that same call, or in
a jobless moment of the server between calls (``idle_check``). The stop outcome is reported separately. A later
lookup without an answer, or a window that reuses the id, restores nothing,
and only ``release`` and a new ``begin`` watch a window again; while the session lives, a lookup that stops
answering serves no image for that call (``window_lookup_unavailable``) and is an explicit unknown, never proof of
destruction — as little as a missing image is. ``SCWindow.isOnScreen`` is passed on as listed and decides nothing:
a hidden window reads false too.

Ending is confirmed, not assumed. The store closes first, so a callback that arrives later changes nothing (each
session has its own store and sink; none is reopened). The stream's own end is the one exception: it is kept as
``stream_end``, with its own reason, even when the session had closed the store first, and it reopens nothing. The
session leaves its slot, and its native objects are let go, only once the stream is known not to run: it reported its
own end, its start answered with an error, or its stop answered. A stop call that raises proves nothing, and a
"not running" stop error confirms only inside ScreenCaptureKit's own error domain. Until then ``begin`` refuses and
``release`` can be asked again; answers that arrive after their wait ended are still recorded for that. A session
whose window is proven gone keeps its slot, its ``window_gone`` answer and the ``cleanup`` witness of what its
native stop is known to have done until ``release``: confirmed speaks only from a native answer or event, an
unconfirmed stop is never a freed resource, and nothing retries it frantically — ``release`` is the door that asks
again.

Every method runs on the process's main thread — including ``idle_check``, which the server runs in a jobless
moment between calls — which services its run loop on entering ``frame`` and ``release`` and while one of them
waits, not while it idles; sample callbacks arrive on a serial queue of their own. Which queue
ScreenCaptureKit uses for the delegate's calls is not known here: were it the main queue, a system stop would be
taken in at the next call, before that call decides anything, and its ``stop.age_ms`` would count from then.
``StreamBridge`` is the PyObjC boundary (tests replace it): CoreMedia functions, the
sample buffer's CF type, the frame-info keys and both completion handlers are registered and read back before use.
The PNG is written by ImageIO to a file and read from there: ``bytes(NSData)`` keeps each encoded image alive under
PyObjC 12.2.1 (measured: resident memory grew by about the PNG's size per call), which a session would accumulate.
"""

from __future__ import annotations

import datetime
import importlib.util
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import time
from collections import deque
from typing import Any, Callable

MAX_WIDTH, MAX_HEIGHT = 1280, 800  # plugin._MAX_IMAGE_W/H: a session frame is never larger than a default screenshot
FRAMES_PER_SECOND = 5
QUEUE_DEPTH = 4  # one of these surfaces is the held frame; three stay with the stream (its header allows up to 8)
START_TIMEOUT_MS = 8000  # one budget for the window listing and the stream's start
STOP_TIMEOUT_MS = 3000
SHUTDOWN_STOP_MS = 1500  # at the server's end: inside the host's two-second wait for its exit
WAIT_MS_MAX = 10000
BEGIN_OPTIONS = ("app", "window")
FRAME_OPTIONS = ("after_seq", "wait_ms")
FRAME_STATUS = {0: "complete", 1: "idle", 2: "blank", 3: "suspended", 4: "started", 5: "stopped"}  # SCFrameStatus
STOP_REASONS = {-3801: "screen_recording_denied", -3815: "no_capture_source", -3817: "user_stopped",
                -3821: "system_stopped"}  # SCStreamError codes a stream ends with
_NOT_STOPPABLE = -3808  # SCStreamErrorAttemptToStopStreamState: the stream was not running when asked to stop
_CM_PATH = "/System/Library/Frameworks/CoreMedia.framework"
_SB = b"^{opaqueCMSampleBuffer=}"  # CMSampleBufferRef as the runtime encodes it
_PB = b"^{__CVBuffer=}"  # CVImageBufferRef
_CMTIME = b"{?=qiIq}"  # CMTime: value, timescale, flags, epoch
_CMTIME_VALID = 1  # kCMTimeFlags_Valid
_CMTIME_NOT_A_NUMBER = 4 | 8 | 16  # kCMTimeFlags_PositiveInfinity, _NegativeInfinity, _Indefinite
_BGRA = 0x42475241  # kCVPixelFormatType_32BGRA
_FRAME_KEYS = ("SCStreamFrameInfoStatus", "SCStreamFrameInfoDisplayTime", "SCStreamFrameInfoContentRect",
               "SCStreamFrameInfoContentScale", "SCStreamFrameInfoScaleFactor")
_SLICE_S = 0.05
_LOOKUP_SLICE_S = 0.5  # a long wait re-looks at the window server at most this often
_NOTE = ("Capture-only: one window's pixels, not display coordinates. No observation_id is supplied and no input "
         "tool may use this image's coordinates.")


def _error_handler() -> dict[str, Any]:
    """A ``void (^)(NSError *)`` completion handler: argument 0 is the block itself."""
    return {"retval": {"type": b"v"}, "arguments": {0: {"type": b"^v"}, 1: {"type": b"@"}}}


STREAM_HANDLERS: dict[tuple[str, str], dict[str, Any]] = {
    ("SCStream", "startCaptureWithCompletionHandler:"): {"arguments": {2: {"callable": _error_handler()}}},
    ("SCStream", "stopCaptureWithCompletionHandler:"): {"arguments": {2: {"callable": _error_handler()}}},
}


def load_capture(file: str = "macos_capture.py") -> Any:
    """A module beside this file, by its path (``-I`` keeps this folder off ``sys.path``): ``macos_capture.py``, or
    ``mac_input.py`` for the structured refusal of a missing grant."""
    path = pathlib.Path(__file__).resolve().with_name(file)
    spec = importlib.util.spec_from_file_location("_unix_computer_use_" + path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = load_capture()  # standard library only at import; its Bridge imports PyObjC when constructed
mac_input = load_capture("mac_input.py")  # standard library only
# SCStreamError codes — including _NOT_STOPPABLE — prove something only inside ScreenCaptureKit's own error domain;
# the same number from any other domain is an unrelated error.
_SCSTREAM_DOMAIN = capture._TCC_DECLINED[0]  # "com.apple.ScreenCaptureKit.SCStreamErrorDomain"


def stream_size(width: int, height: int, max_width: int = MAX_WIDTH, max_height: int = MAX_HEIGHT) -> tuple[int, int]:
    """The window's pixels fitted inside the bound by one factor for both sides, never enlarged. (0, 0) = none."""
    if width < 1 or height < 1:
        return 0, 0
    shrink = min(1.0, max_width / width, max_height / height)
    return max(1, round(width * shrink)), max(1, round(height * shrink))


def parse_begin(raw: Any) -> tuple[dict[str, Any], str]:
    """``begin``'s arguments -> ({app, want}, error); nothing is resolved here."""
    if not isinstance(raw, dict):
        return {}, "arguments must be an object"
    unknown = sorted(set(raw) - set(BEGIN_OPTIONS))
    if unknown:
        return {}, f"unknown argument(s) {unknown}; known: {list(BEGIN_OPTIONS)}"
    app = raw.get("app")
    if not isinstance(app, str) or not app.strip():
        return {}, "app is required: the application that owns the window"
    want, error = capture.window_selector(raw.get("window"))
    if error:
        return {}, error
    return {"app": app, "want": want}, ""


def parse_frame(raw: Any) -> tuple[dict[str, Any], str]:
    """``frame``'s arguments -> ({after_seq or None, wait_ms}, error); the wait is clamped to ``WAIT_MS_MAX``."""
    if not isinstance(raw, dict):
        return {}, "arguments must be an object"
    unknown = sorted(set(raw) - set(FRAME_OPTIONS))
    if unknown:
        return {}, f"unknown argument(s) {unknown}; known: {list(FRAME_OPTIONS)}"
    out: dict[str, Any] = {}
    for key, default in (("after_seq", None), ("wait_ms", 0)):
        value = raw.get(key, default)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            return {}, f"{key} must be a non-negative integer"
        out[key] = value
    out["wait_ms"] = min(out["wait_ms"] or 0, WAIT_MS_MAX)
    return out, ""


class FrameStore:
    """What one stream's callbacks leave for its session: ONE held frame and the reports around it.

    ``stop`` is None while the stream may deliver. Once set, by the stream's own end or by the session, every later
    callback is dropped, so an ended store never holds a frame again. ``stream_end`` is the stream's own end whenever
    it came, also after the session's: the evidence that the stream no longer runs. ``snapshot`` returns the held
    frame and the facts that describe it from one critical section."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.cond = threading.Condition()
        self._clock = clock  # seconds on the clock the stream's display times are expressed in
        self.seq = 0
        self.frame: dict[str, Any] | None = None  # the last sample that carried an image, with its buffer
        self.withheld: dict[str, Any] | None = None  # the report that dropped the held frame
        self.discards = 0  # monotonic count of held-image drops; a later valid frame does not undo a drop
        self.report: dict[str, Any] | None = None  # the stream's latest report of any kind
        self.counts: dict[str, int] = {}
        self.stop: dict[str, Any] | None = None
        self.stream_end: dict[str, Any] | None = None
        self.errors: deque[str] = deque(maxlen=5)

    def accept(self, status: str, *, display_s: float | None = None, buffer: Any = None, width: int = 0,
               height: int = 0, reported: dict[str, Any] | None = None, rejected: str = "") -> bool:
        """One screen sample's report -> whether it was taken (False once the stream has ended)."""
        # A report speaks of the moment the stream displayed it, which precedes its arrival here; the time of
        # arrival stands in only where the stream attached none, and the basis says which.
        at = {"at_s": self._clock(), "time_basis": "received"} if display_s is None else {"at_s": display_s,
                                                                                         "time_basis": "display"}
        with self.cond:
            if self.stop is not None:
                return False
            self.counts[status] = self.counts.get(status, 0) + 1
            self.report = {"status": status, **at}
            if buffer is not None:
                self.seq += 1
                self.frame = {"seq": self.seq, "buffer": buffer, "status": status, "width": width, "height": height,
                              "reported": reported or {}, **at}
                self.withheld = None
            elif status != "idle":  # the window's content is no longer the held image
                self.frame = None
                self.withheld = {"status": rejected or status, **at}
                self.discards += 1
            self.cond.notify_all()
            return True

    def end(self, reason: str, *, by: str, code: int | None = None, domain: str | None = None) -> bool:
        """The stream ended (``by`` the stream itself or the session) -> whether this call ended it; the first end stands.
        The stream's own first end is kept as ``stream_end`` either way."""
        with self.cond:
            end = {"reason": reason, "by": by, "at_s": self._now(),
                   **({"code": code, "domain": domain} if code is not None else {})}
            if by == "stream" and self.stream_end is None:
                self.stream_end = end
            if self.stop is not None:
                return False
            self.stop = end
            self.frame = None  # the surface goes back; nothing is served after an end
            self.cond.notify_all()
            return True

    def fail(self, exc: BaseException) -> None:
        """A callback this code could not take in: whatever it carried, the held image may no longer be the window."""
        with self.cond:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            if self.stop is not None:
                return
            at = {"at_s": self._now(), "time_basis": "received"}
            self.counts["callback_error"] = self.counts.get("callback_error", 0) + 1
            self.report = {"status": "callback_error", **at}
            self.frame = None
            self.withheld = {"status": "callback_error", **at}
            self.discards += 1
            self.cond.notify_all()

    def note_error(self, exc: BaseException) -> None:
        self.errors.append(f"{type(exc).__name__}: {exc}")

    def _now(self) -> float | None:
        """The clock's time for ``end`` and ``fail``, which must not raise: None when the clock itself fails."""
        try:
            return self._clock()
        except Exception:  # noqa: BLE001
            return None

    def wait(self, floor_seq: int, deadline: float, *, pump: Callable[[float], None],
             clock: Callable[[], float] = time.monotonic) -> None:
        """Until a frame newer than ``floor_seq`` is held, the stream ended or ``deadline`` passed.

        The main run loop is serviced once before the first look and in short slices while waiting (as
        ``macos_capture.await_handler`` does), so an end delivered on the main queue arrives too."""
        pump(0)
        while True:
            with self.cond:
                if self.stop is not None or (self.frame is not None and self.frame["seq"] > floor_seq):
                    return
            left = deadline - clock()
            if left <= 0:
                return
            pump(min(_SLICE_S, left))
            with self.cond:
                if self.stop is None and (self.frame is None or self.frame["seq"] <= floor_seq):
                    self.cond.wait(min(0.01, max(0.0, deadline - clock())))

    def snapshot(self) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """(the held frame with its buffer, the facts at this moment): ages, never a verdict of freshness."""
        now = self._clock()
        age = lambda at: None if at is None else round((now - at) * 1000, 1)  # noqa: E731
        with self.cond:
            frame = dict(self.frame) if self.frame else None
            facts: dict[str, Any] = {"state": "live" if self.stop is None else "stopped", "seq": self.seq}
            if frame:
                facts.update(frame_status=frame["status"], frame_age_ms=age(frame["at_s"]),
                             frame_time_basis=frame["time_basis"], image_width=frame["width"],
                             image_height=frame["height"],
                             **({"stream_reported": frame["reported"]} if frame["reported"] else {}))
            if self.withheld:
                facts["withheld"] = {"status": self.withheld["status"], "age_ms": age(self.withheld["at_s"])}
            stream: dict[str, Any] = {"reports": dict(self.counts)}
            if self.report:
                stream.update(last_report=self.report["status"], last_report_age_ms=age(self.report["at_s"]),
                              last_report_time_basis=self.report["time_basis"])
            if self.errors:
                stream["callback_errors"] = list(self.errors)
            facts["stream"] = stream
            if self.stop:
                facts["stop"] = {**{k: v for k, v in self.stop.items() if k != "at_s"}, "age_ms": age(self.stop["at_s"])}
            if self.stream_end is not None and self.stream_end is not self.stop:  # it came after the session's end
                facts["stream_end"] = {**{k: v for k, v in self.stream_end.items() if k not in ("at_s", "by")},
                                       "age_ms": age(self.stream_end["at_s"])}
            return frame, facts


class _Feed:
    """What the Objective-C sink calls on the stream's queues; nothing may escape into the framework."""

    def __init__(self, bridge: Any, store: FrameStore) -> None:
        self._bridge, self._store = bridge, store

    def sample(self, sample_buffer: Any, output_type: int) -> None:
        try:
            if self._store.stop is not None:  # a late callback: decided again under the lock in accept
                return
            with self._bridge.pool():
                parsed = self._bridge.read_sample(sample_buffer, output_type)
                if parsed is not None:
                    self._store.accept(**parsed)
        except BaseException as exc:  # noqa: BLE001 - an unread sample may have been a new image: the old one goes
            self._store.fail(exc)

    def inactive(self) -> None:
        try:
            self._store.accept("inactive")  # every window the stream shared has closed
        except BaseException as exc:  # noqa: BLE001
            self._store.fail(exc)

    def ended(self, error: Any) -> None:
        try:
            facts = capture._error_facts(error) if error is not None else {}
            code, domain = facts.get("error_code"), facts.get("error_domain")
            reason = STOP_REASONS.get(code, "stream_error") if domain == _SCSTREAM_DOMAIN else "stream_error"
            self._store.end(reason, by="stream", code=code, domain=domain)
        except BaseException as exc:  # noqa: BLE001
            self._store.note_error(exc)
            self._store.end("stream_error", by="stream")


_SINK: Any = None  # the Objective-C class, defined once per process (class names are global there)


def _sink_class(objc: Any) -> Any:
    """SCStreamOutput and SCStreamDelegate in one NSObject; neither protocol exists in the runtime to conform to."""
    global _SINK
    if _SINK is None:
        from Foundation import NSObject

        class OuroborosWindowStreamSink(NSObject):
            def initWithFeed_(self, feed):
                self = objc.super(OuroborosWindowStreamSink, self).init()
                if self is not None:
                    self._feed = feed
                return self

            @objc.typedSelector(b"v@:@" + _SB + b"q")
            def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
                self._feed.sample(sample_buffer, output_type)

            @objc.typedSelector(b"v@:@@")
            def stream_didStopWithError_(self, stream, error):
                self._feed.ended(error)

            @objc.typedSelector(b"v@:@")
            def streamDidBecomeInactive_(self, stream):
                self._feed.inactive()

        _SINK = OuroborosWindowStreamSink
    return _SINK


def numeric_time(cmtime: Any) -> tuple[int, int] | None:
    """(value, timescale) of a CMTime that is a number: valid, neither infinite nor indefinite, a positive timescale."""
    value, timescale, flags, _epoch = cmtime
    if not flags & _CMTIME_VALID or flags & _CMTIME_NOT_A_NUMBER or timescale <= 0:
        return None
    return int(value), int(timescale)


def _seconds(cmtime: Any) -> float | None:
    """Seconds of a CMTime on the host clock; None when it is no number or belongs to another timeline (its epoch)."""
    numeric = numeric_time(cmtime)
    if numeric is None or cmtime[3] != 0:
        return None
    return numeric[0] / numeric[1]


class StreamBridge(capture.Bridge):
    """The one-shot bridge plus what a stream needs; one instance per server process."""

    def __init__(self) -> None:
        super().__init__()
        self._cm: dict[str, Any] = {}
        self._keys: dict[str, Any] = {}
        self._queue: Any = None

    def load(self, handlers: dict[tuple[str, str], dict[str, Any]] | None = None) -> str:
        """Load ScreenCaptureKit and CoreMedia and register what PyObjC does not ship -> '' when ready, else why not."""
        problem = super().load(handlers or {**capture.HANDLERS, **STREAM_HANDLERS})
        if problem:
            return problem
        objc = self.objc
        from Foundation import NSBundle

        # NSError ** is an output only once it is declared one: the call then returns (BOOL, NSError or None).
        for selector, index in ((b"addStreamOutput:type:sampleHandlerQueue:error:", 5), (b"removeStreamOutput:type:error:", 4)):
            objc.registerMetaDataForSelector(b"SCStream", selector, {"arguments": {index: {"type_modifier": b"o"}}})
        wanted = [("CMSampleBufferGetTypeID", b"Q"), ("CMSampleBufferIsValid", b"Z" + _SB),
                  ("CMSampleBufferGetImageBuffer", _PB + _SB),
                  ("CMSampleBufferGetSampleAttachmentsArray", b"^{__CFArray=}" + _SB + b"Z"),
                  ("CMClockGetHostTimeClock", b"@"), ("CMClockGetTime", _CMTIME + b"@"),
                  ("CMClockMakeHostTimeFromSystemUnits", _CMTIME + b"Q")]
        objc.loadBundleFunctions(NSBundle.bundleWithPath_(_CM_PATH), self._cm, wanted)
        objc.loadBundleVariables(NSBundle.bundleWithPath_(capture._SCK_PATH), self._keys,
                                 [(name, b"@") for name in _FRAME_KEYS], True)
        lib: dict[str, Any] = {}  # libdispatch is no bundle: found through the process's own symbols
        objc.loadBundleFunctions(None, lib, [("dispatch_queue_create", b"@*@", "", {"retval": {"already_retained": True}})])
        missing = [name for name, _sig in wanted if name not in self._cm]
        missing += [name for name in _FRAME_KEYS if self._keys.get(name) is None]
        missing += [name for name in ("dispatch_queue_create",) if name not in lib]
        if missing:
            return f"this macOS does not provide {missing} to PyObjC"
        # Without its CF type the delegate would receive a pointer nothing can retain.
        objc.registerCFSignature("CMSampleBufferRef", _SB, self._cm["CMSampleBufferGetTypeID"]())
        self._queue = lib["dispatch_queue_create"](b"ouroboros.unix_computer_use.window_stream", None)  # serial
        _sink_class(objc)
        return "" if self._queue is not None else "the stream's serial queue could not be created"

    def host_now(self) -> float:
        """Seconds on the host-time clock, which frame display times are expressed in."""
        now = _seconds(self._cm["CMClockGetTime"](self._cm["CMClockGetHostTimeClock"]()))
        if now is None:
            raise RuntimeError("the host-time clock gave no numeric time")
        return now

    def pool(self) -> Any:
        """An autorelease pool for a thread that turns no run loop."""
        return self.objc.autorelease_pool()

    def stream_configuration(self, width: int, height: int) -> tuple[Any, dict[str, Any]]:
        """The one-shot's window configuration as a bounded BGRA video stream -> (config, what the object reads back)."""
        config, applied = self.configuration(width, height)
        config.setPixelFormat_(_BGRA)  # the object's default is a video format ImageIO is not given here
        # Without an explicit output space SCK uses the display profile, while
        # CVImageBufferGetColorSpace may be absent. Our PNG fallback is sRGB.
        config.setColorSpaceName_(self.quartz.kCGColorSpaceSRGB)
        config.setMinimumFrameInterval_((1, FRAMES_PER_SECOND, 1, 0))
        config.setQueueDepth_(QUEUE_DEPTH)
        config.setCapturesAudio_(False)
        config.setShouldBeOpaque_(True)  # the image is written without its alpha channel
        interval = numeric_time(config.minimumFrameInterval())
        pixel_format = int(config.pixelFormat())
        return config, {**applied,
                        "frames_per_second_max": round(interval[1] / interval[0], 2) if interval and interval[0] > 0 else None,
                        "queue_depth": int(config.queueDepth()), "audio": bool(config.capturesAudio()),
                        "pixel_format": "BGRA" if pixel_format == _BGRA else f"{pixel_format:#x}"}

    def stream(self, content_filter: Any, config: Any, store: FrameStore) -> tuple[Any, Any]:
        """(stream, sink) for one window; the sink feeds ``store`` and must be kept alive by the caller."""
        sink = _sink_class(self.objc).alloc().initWithFeed_(_Feed(self, store))
        stream = self.classes["SCStream"].alloc().initWithFilter_configuration_delegate_(content_filter, config, sink)
        return stream, sink

    def add_output(self, stream: Any, sink: Any) -> str:
        """Attach ``sink`` as the stream's screen output on the serial queue -> '' or the framework's error."""
        answer = stream.addStreamOutput_type_sampleHandlerQueue_error_(sink, 0, self._queue, None)  # 0: screen
        ok, error = answer if isinstance(answer, tuple) else (answer, None)
        return "" if ok else str(capture._error_facts(error).get("error") if error is not None else "refused")

    def remove_output(self, stream: Any, sink: Any) -> None:
        stream.removeStreamOutput_type_error_(sink, 0, None)

    @staticmethod
    def start(stream: Any, handler: Callable[[Any], None]) -> None:
        stream.startCaptureWithCompletionHandler_(handler)

    @staticmethod
    def stop(stream: Any, handler: Callable[[Any], None]) -> None:
        stream.stopCaptureWithCompletionHandler_(handler)

    def read_sample(self, sample_buffer: Any, output_type: int) -> dict[str, Any] | None:
        """One delivered sample as ``FrameStore.accept``'s arguments; None for what is no screen sample."""
        cm, keys, quartz = self._cm, self._keys, self.quartz
        if output_type != 0:
            return None
        if sample_buffer is None or not cm["CMSampleBufferIsValid"](sample_buffer):
            return {"status": "invalid_sample"}  # a screen sample nothing can be read from
        rows = cm["CMSampleBufferGetSampleAttachmentsArray"](sample_buffer, False)
        info = rows[0] if rows is not None and len(rows) else {}
        raw = info.get(keys["SCStreamFrameInfoStatus"])
        status = "unreported" if raw is None else FRAME_STATUS.get(int(raw), f"unknown:{int(raw)}")
        ticks = info.get(keys["SCStreamFrameInfoDisplayTime"])
        # Native stopped/inactive reports can carry the zero placeholder, not
        # a frame displayed at system boot. Time that report by its receipt.
        if ticks == 0 and status not in ("complete", "started"):
            ticks = None
        out: dict[str, Any] = {"status": status, "display_s": None if ticks is None else _seconds(
            cm["CMClockMakeHostTimeFromSystemUnits"](int(ticks)))}
        if status not in ("complete", "started"):
            return out
        image = cm["CMSampleBufferGetImageBuffer"](sample_buffer)
        pixel_format = None if image is None else int(quartz.CVPixelBufferGetPixelFormatType(image))
        if pixel_format != _BGRA:
            return {**out, "rejected": "no_image" if image is None else f"pixel_format:{pixel_format:#x}"}
        reported: dict[str, Any] = {}
        rect = info.get(keys["SCStreamFrameInfoContentRect"])
        ok, content = quartz.CGRectMakeWithDictionaryRepresentation(rect, None) if rect is not None else (False, None)
        if ok:
            reported["content_rect"] = [round(v, 2) for v in (content.origin.x, content.origin.y, content.size.width,
                                                              content.size.height)]
        for name, key in (("content_scale", "SCStreamFrameInfoContentScale"), ("scale_factor", "SCStreamFrameInfoScaleFactor")):
            if info.get(keys[key]) is not None:
                reported[name] = float(info.get(keys[key]))
        return {**out, "buffer": sample_buffer, "width": int(quartz.CVPixelBufferGetWidth(image)),
                "height": int(quartz.CVPixelBufferGetHeight(image)), "reported": reported}

    def write_png(self, sample_buffer: Any, path: str) -> tuple[int, int]:
        """The held frame's pixels as a PNG file at ``path`` -> (width, height).

        The BGRA bytes are copied under a read-only lock and ImageIO writes the file from that private copy, so a
        frame replaced meanwhile changes nothing and the encoded bytes never become an NSData Python holds."""
        from Foundation import NSURL

        quartz = self.quartz
        image = self._cm["CMSampleBufferGetImageBuffer"](sample_buffer)
        if image is None or quartz.CVPixelBufferGetPixelFormatType(image) != _BGRA:
            raise ValueError("the held frame is not a BGRA pixel buffer")
        width, height = int(quartz.CVPixelBufferGetWidth(image)), int(quartz.CVPixelBufferGetHeight(image))
        stride = quartz.CVPixelBufferGetBytesPerRow(image)
        if quartz.CVPixelBufferLockBaseAddress(image, quartz.kCVPixelBufferLock_ReadOnly) != 0:
            raise RuntimeError("the pixel buffer could not be locked")
        try:
            raw = bytes(quartz.CVPixelBufferGetBaseAddress(image).as_buffer(stride * height))
        finally:
            quartz.CVPixelBufferUnlockBaseAddress(image, quartz.kCVPixelBufferLock_ReadOnly)
        space = quartz.CVImageBufferGetColorSpace(image) or quartz.CGColorSpaceCreateWithName(quartz.kCGColorSpaceSRGB)
        picture = quartz.CGImageCreate(
            width, height, 8, 32, stride, space, quartz.kCGBitmapByteOrder32Little | quartz.kCGImageAlphaNoneSkipFirst,
            quartz.CGDataProviderCreateWithCFData(raw), None, False, quartz.kCGRenderingIntentDefault)
        destination = quartz.CGImageDestinationCreateWithURL(NSURL.fileURLWithPath_(path), "public.png", 1, None)
        if picture is None or destination is None:
            raise RuntimeError("Quartz could not prepare the PNG")
        quartz.CGImageDestinationAddImage(destination, picture, None)
        if not quartz.CGImageDestinationFinalize(destination):
            raise RuntimeError("Quartz did not encode the frame as PNG")
        return width, height

    def window_listed(self, window_id: int, pid: int) -> bool | None:
        """Whether the window server lists this window for this process now; None when it gave no answer."""
        quartz = self.quartz
        rows = quartz.CGWindowListCopyWindowInfo(quartz.kCGWindowListOptionIncludingWindow, window_id)
        if rows is None:
            return None
        return any(row.get(quartz.kCGWindowNumber) == window_id and row.get(quartz.kCGWindowOwnerPID) == pid
                   for row in rows)


def _refusal(reason: str, error: str, **extra: Any) -> dict[str, Any]:
    return mac_input.with_refusal({"ok": False, "reason": reason, "error": error, **extra})


def _grant_stop(facts: dict[str, Any]) -> dict[str, Any]:
    """A stream the system ended for a missing grant (SCStreamError -3801) carries the structured refusal a denied
    start carries: the owner must allow Screen Recording before a new session can work."""
    stop = facts.get("stop") if isinstance(facts.get("stop"), dict) else {}
    return mac_input.permission_refusal("screen_recording") if stop.get("reason") == "screen_recording_denied" else {}


class WindowSessions:
    """``begin`` / ``frame`` / ``release`` for at most one window stream at a time (the module docstring's contract).

    Each answer is ``(facts, png)``: ``png`` is the image's bytes only when the facts say ``image: true``. The
    server's jobless moments may call ``idle_check``, which ends a session whose window is proven destroyed — the
    same proof, and the same single bounded stop, a ``frame`` call would apply."""

    def __init__(self, *, native: Any = None, bridge_factory: Callable[[], Any] = StreamBridge,
                 clock: Callable[[], float] = time.monotonic, mac_version: Callable[[], int | None] = capture.mac_major,
                 platform: str = sys.platform) -> None:
        self._native, self._bridge_factory, self._clock = native, bridge_factory, clock
        self._mac_version, self._platform = mac_version, platform
        self._bridge: Any = None
        self._loaded = False
        self._session: dict[str, Any] | None = None
        self._scratch: str | None = None

    # -- begin --------------------------------------------------------------------------------------------------

    def begin(self, raw: Any) -> tuple[dict[str, Any], None]:
        """Select one window of one application and start its stream; no image is returned."""
        if self._platform != "darwin":
            return _refusal("unsupported_platform", f"a window session is local macOS {capture.MIN_MACOS}+ only "
                            f"(this is {self._platform})"), None
        major = self._mac_version()
        if major is None or major < capture.MIN_MACOS:
            return _refusal("unsupported_os", f"a window session needs macOS {capture.MIN_MACOS} or later"), None
        if self._native is None:
            self._native = capture.load_native()
        spec, error = parse_begin(raw)
        if error:
            return _refusal("bad_option", error), None
        if self._session is not None:
            return _refusal("session_exists", "this task already holds a window session; release it before beginning "
                            "another", session=self._identity(self._session)), None
        target, refusal = self._native.resolve_target(spec["app"])
        if refusal:
            return {**refusal, "ok": False}, None
        bridge, refusal = self._ready_bridge(target)
        if refusal:
            return refusal, None
        return self._start(bridge, target, spec["want"]), None

    def _ready_bridge(self, target: dict[str, Any]) -> tuple[Any, dict[str, Any] | None]:
        """The loaded bridge, once this process holds the Screen Recording grant; nothing is asked without it."""
        if self._bridge is None:
            try:
                self._bridge = self._bridge_factory()
            except ImportError as exc:
                return None, _refusal("capture_unavailable", f"PyObjC is not available to this interpreter ({exc})", app=target)
        grant = self._bridge.screen_recording()
        if grant != "granted":
            return None, _refusal("screen_recording_denied", "this process holds no Screen Recording grant; nothing was "
                                  "asked or captured", app=target, permissions={"screen_recording": grant})
        if not self._loaded:
            try:
                problem = self._bridge.load()
            except Exception as exc:  # noqa: BLE001 - a framework this PyObjC cannot bind is a refusal, not a crash
                problem = f"{type(exc).__name__}: {exc}"
            if problem:
                return None, _refusal("capture_unavailable", problem, app=target)
            self._loaded = True
        return self._bridge, None

    def _start(self, bridge: Any, target: dict[str, Any], want: dict[str, Any]) -> dict[str, Any]:
        clock = self._clock
        started = clock()
        deadline = started + START_TIMEOUT_MS / 1000

        def failed(stage: str, arrived: bool, err: Any, **extra: Any) -> dict[str, Any]:
            if not arrived:
                return _refusal("capture_timeout", f"no ScreenCaptureKit answer within the {START_TIMEOUT_MS} ms the "
                                "listing and the start share", stage=stage, app=target, **extra)
            facts = capture._error_facts(err if err is not None else "no result and no error")
            declined = (facts.get("error_domain"), facts.get("error_code")) == capture._TCC_DECLINED
            return mac_input.with_refusal({"ok": False, "reason": "screen_recording_denied" if declined else "capture_failed",
                                           "stage": stage, "app": target, **extra, **facts})

        arrived, content, err = capture.await_handler(bridge.shareable_content, deadline, pump=bridge.pump, clock=clock)
        if not arrived or err is not None or content is None:
            return failed("shareable_content", arrived, err)
        rows = [(window, bridge.window_facts(window)) for window in bridge.windows(content)]
        window, facts, refusal = capture.select_window(rows, int(target["pid"]), want)
        if refusal:
            return {"ok": False, "app": target, **refusal}
        listed = self._listed(facts["window_id"], target["pid"])
        if listed is not True:
            reason = "window_not_found" if listed is False else "window_lookup_unavailable"
            return _refusal(reason, "WindowServer did not confirm this window for the selected process; "
                            "no capture started", app=target, window=facts)
        content_filter = bridge.content_filter(window)
        width_pt, height_pt, scale = bridge.content_size(content_filter)
        width, height = stream_size(*capture.pixel_size(width_pt, height_pt, scale))
        if not width or not height:
            return _refusal("window_empty", "the window has no content area to capture", app=target, window=facts)
        config, applied = bridge.stream_configuration(width, height)
        store = FrameStore(bridge.host_now)
        stream, sink = bridge.stream(content_filter, config, store)
        problem = bridge.add_output(stream, sink)
        if problem:  # nothing was started: the objects are simply let go
            return _refusal("capture_failed", problem, stage="add_output", app=target, window=facts)
        session = self._session = {
            "app": target, "window": facts, "store": store, "stream": stream, "start": None, "stopped": None,
            "native": (sink, content_filter, config, window),  # kept alive until the stream is known not to run
        }

        def start(handler: Callable[[Any, Any], None]) -> None:
            def answered(error: Any) -> None:
                session["start"] = ("error", error) if error is not None else ("ok", None)  # kept even when late
                handler(None, error)
            try:
                bridge.start(stream, answered)
            except Exception as exc:  # noqa: BLE001 - report the bridge failure, not a native start completion
                handler(None, exc)

        arrived, _value, err = capture.await_handler(start, deadline, pump=bridge.pump, clock=clock)
        if not arrived or err is not None:
            ending = self._end(session, "start_timeout" if not arrived else "start_failed", STOP_TIMEOUT_MS)
            return {**failed("start", arrived, err, window=facts), **({} if arrived else {"reason": "start_timeout"}),
                    "cleanup": ending}
        if store.stop is not None:  # the stream reported its own end before its start answered
            _frame, state = store.snapshot()
            return {**_refusal("stream_stopped", "the stream ended as it started", app=target, window=facts, **state), **_grant_stop(state),
                    "cleanup": self._end(session, "released", STOP_TIMEOUT_MS)}
        return mac_input.proved({"ok": True, "state": "live", "backend": "ScreenCaptureKit", "app": target, "window": facts,
                                 "stream": applied, "point_pixel_scale": scale, "start_ms": round((clock() - started) * 1000, 1),
                                 "window_gone_check": "window_server",
                                 "mapped_input_available": False, "note": _NOTE}, "screen_recording")  # a live stream

    # -- frame --------------------------------------------------------------------------------------------------

    def frame(self, raw: Any) -> tuple[dict[str, Any], bytes | None]:
        """The held frame as PNG bytes with its facts, or the facts alone when there is no newer or no current image."""
        spec, error = parse_frame(raw)
        if error:
            return _refusal("bad_option", error), None
        session = self._session
        if session is None:
            return _refusal("no_session", "no window session is held; begin one first"), None
        bridge, store, clock = self._bridge, session["store"], self._clock
        waited = clock()
        floor = spec["after_seq"] or 0
        deadline = waited + spec["wait_ms"] / 1000
        gone: dict[str, Any] | None = None  # this call's own window_gone proof, with its native-stop witness
        lookup_unknown = False

        def settled() -> bool:
            with store.cond:
                return store.stop is not None or (store.frame is not None and store.frame["seq"] > floor)

        while True:
            # A long pending request must not stay blind to a destroyed target: every bounded slice of the wait
            # is followed by one window-server look, and proven destruction ends the wait and the stream early.
            store.wait(floor, min(deadline, clock() + _LOOKUP_SLICE_S), pump=bridge.pump, clock=clock)
            if settled() or clock() >= deadline:
                break
            if self._listed(*self._ids(session)) is False:
                gone = self._gone(session)
                break
            # a mid-wait look without an answer is unknown, not absence: keep waiting, the final look decides
        if gone is None and store.stop is None:
            listed = self._listed(*self._ids(session))
            if listed is False:
                # The window server's "no such window for this process" is the one destruction proof used here, and
                # it is independent of the stream's delivery: a hidden, occluded, blank or inactive window stays
                # listed, and a missing image proves nothing either way. Delivery ends for good — a later lookup
                # without an answer, or another window under the same id, restores nothing — and the native stream
                # gets one stop request now, while the task and server live; its answer is separate evidence.
                gone = self._gone(session)
            elif listed is None:
                lookup_unknown = True  # an authoritative lookup that stopped answering: unknown, not absence
        with store.cond:  # the held frame and its discard count from one critical section
            held, state = store.snapshot()
            discards = store.discards
        facts = {**state, **self._identity(session), "waited_ms": round((clock() - waited) * 1000, 1),
                 "mapped_input_available": False}

        def withhold(reason: str, error: str) -> tuple[dict[str, Any], None]:
            return {"ok": False, "reason": reason, "error": error, "image": False, **facts, **_grant_stop(facts)}, None

        if state["state"] != "live":
            # The native stream's fate stays answerable until release: the same witness release/_end confirm by,
            # from the answers and events already recorded — nothing is asked again and nothing is retried here.
            facts["cleanup"] = gone if gone is not None else self._stop_witness(session)
            if state["stop"]["reason"] == "window_gone":
                return withhold("window_gone", "the window server stopped listing this window for the application; "
                                "this session serves nothing again: release it")
            return withhold("stream_stopped", f"the stream ended ({state['stop']['reason']}); it is not restarted: "
                            "release this session, then begin again if the window should still be watched")
        if lookup_unknown:
            return withhold("window_lookup_unavailable", "the window server gave no answer about this window for "
                            "this call, so its existence is unknown and no image is served; this is not evidence "
                            "that the window is gone: ask again")
        if held is None:
            if state.get("withheld"):
                return withhold("frame_withheld", f"the stream last reported {state['withheld']['status']}: the earlier "
                                "image is not the window's content now")
            return withhold("no_frame_yet", "the stream has delivered no image yet; ask again with wait_ms")
        if spec["after_seq"] is not None and held["seq"] <= spec["after_seq"]:
            return {"ok": True, "image": False, "reason": "no_newer_frame", "after_seq": spec["after_seq"], **facts}, None
        at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(milliseconds=facts["frame_age_ms"])
        try:
            data = self._encode(bridge, held)
        except Exception as exc:  # noqa: BLE001 - the session stays; the model may ask again
            return withhold("encode_failed", f"{type(exc).__name__}: {exc}")
        bridge.pump(0)
        _again, after = store.snapshot()  # what the stream said while the image was written
        if after["state"] != "live" or after.get("withheld") or store.discards != discards:
            # A drop of the held image while its PNG was being written invalidates that image for this call even
            # when a later valid report already cleared ``withheld``: the drop happened to the frame in flight.
            # The unlocked monotonic read can only see more discards, which withholds too much, never too little.
            facts.update({key: after[key] for key in ("state", "stop", "stream_end", "withheld", "stream") if key in after})
            if after["state"] != "live":
                facts["cleanup"] = self._stop_witness(session)
            return withhold("stream_stopped" if after["state"] != "live" else "frame_withheld",
                            "the stream ended or dropped this frame while it was being written")
        return {"ok": True, "image": True, "frame_at_utc": at.isoformat(timespec="milliseconds"), **facts,
                "note": _NOTE}, data

    def _encode(self, bridge: Any, held: dict[str, Any]) -> bytes:
        if self._scratch is None:
            self._scratch = tempfile.mkdtemp(prefix="ouroboros-window-session-")  # private to this user
        path = os.path.join(self._scratch, f"frame-{held['seq']}.png")
        try:
            with bridge.pool():
                size = bridge.write_png(held["buffer"], path)
            with open(path, "rb") as stream:
                data = stream.read()
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        if not data.startswith(capture._PNG_SIGNATURE) or tuple(size) != (held["width"], held["height"]):
            raise RuntimeError("the written file is not this frame's PNG")
        return data

    # -- release ------------------------------------------------------------------------------------------------

    def release(self, raw: Any = None, *, timeout_ms: int = STOP_TIMEOUT_MS) -> tuple[dict[str, Any], None]:
        """End the held session; without one this is a successful no-op."""
        if raw not in (None, {}):
            return _refusal("bad_option", "release takes no arguments"), None
        session = self._session
        if session is None:
            return {"ok": True, "released": False, "state": "none"}, None
        ending = self._end(session, "released", timeout_ms)
        _frame, state = session["store"].snapshot()
        facts = {"released": ending["confirmed"], **self._identity(session), **state, "cleanup": ending}
        if ending["confirmed"]:
            return {"ok": True, **facts}, None
        return {**_refusal("stop_unconfirmed", "the stream did not confirm its stop; no image is served and no other "
                           "session can begin until release is confirmed or this task's server ends"), **facts}, None

    def shutdown(self) -> None:
        """The server is ending: one bounded stop, then the scratch folder goes. The process's exit ends any stream."""
        try:
            self.release(timeout_ms=SHUTDOWN_STOP_MS)
        finally:
            if self._scratch is not None:
                shutil.rmtree(self._scratch, ignore_errors=True)
                self._scratch = None

    def idle_check(self) -> None:
        """One bounded look at the held session, for a jobless moment of the server's main thread.

        A session whose window the window server no longer lists is ended exactly as the ``frame`` call that would
        prove it ends it: delivery closes and the native stream is asked to stop once, so proven destruction does
        not keep capturing between calls or after the last one. Nothing runs — and nothing native is touched — when
        no session is held or its delivery already ended. A lookup without an answer is unknown, not absence:
        a later idle check looks again. Only an explicit ``release`` retries an unconfirmed stop."""
        session = self._session
        if session is None or session["store"].stop is not None:
            return
        if self._listed(*self._ids(session)) is False:
            self._gone(session)

    def _gone(self, session: dict[str, Any]) -> dict[str, Any]:
        """Proven destruction: end delivery and stop the native stream once, keeping the session's slot.

        The session keeps answering ``window_gone`` — with the ``cleanup`` witness of what the stop is known to
        have done — until ``release`` frees the slot; ``release`` also retries a stop that was not confirmed."""
        return self._end(session, "window_gone", STOP_TIMEOUT_MS, keep_slot=True)

    def _stop_witness(self, session: dict[str, Any]) -> dict[str, Any]:
        """``{confirmed, detail}`` for the native stream's running, from the answers and events already recorded —
        the very grounds ``_end``/``release`` confirm by, asked nothing again. ``confirmed`` speaks only from a
        native answer or event (the stream's own end, a failed start, an answering stop); the absence of a window,
        of an image or of an answer is never confirmation."""
        if session["store"].stream_end is not None:
            return {"confirmed": True, "detail": "the stream reported its own end"}
        if session["start"] is not None and session["start"][0] == "error":
            return {"confirmed": True, "detail": "the stream never started"}
        answer = session["stopped"]
        if answer is not None and self._stop_confirms(session):
            return {"confirmed": True, "detail": "an earlier stop answered"}
        if answer is None:
            return {"confirmed": False, "detail": "no answer to the stop request"}
        if answer[0] == "raised":
            return {"confirmed": False, "detail": f"the stop request raised {type(answer[1]).__name__}: {answer[1]}"}
        return {"confirmed": False, "detail": str(capture._error_facts(answer[1]).get("error"))}

    def _end(self, session: dict[str, Any], reason: str, timeout_ms: int, *, keep_slot: bool = False) -> dict[str, Any]:
        """Close the store, then learn that the stream does not run; only then does the session leave its slot.

        ``keep_slot`` keeps the session — its refusal, its recorded answers and its native objects — after a
        confirmed end: a destroyed window's stream is stopped at once, while only ``release`` frees the slot."""
        bridge, store, clock = self._bridge, session["store"], self._clock
        bridge.pump(0)  # a stream's own end already queued for this thread is taken in before the session's reason
        store.end(reason, by="session")  # from here no callback changes anything; an earlier end keeps its reason
        witness = self._stop_witness(session)
        how = witness["detail"] if witness["confirmed"] else ""
        if not how:
            def stop(handler: Callable[[Any, Any], None]) -> None:
                started = session["start"] is not None  # whether the start had answered when this stop was asked
                before = session["stopped"]

                def answered(error: Any) -> None:
                    session["stopped"] = ("ok" if error is None else "error", error, started)  # kept even when late
                    handler(None, error)
                try:
                    bridge.stop(session["stream"], answered)
                except Exception as exc:  # noqa: BLE001 - whether the stream still runs is unknown: unconfirmed
                    if session["stopped"] is before:  # an answer it gave before raising stands
                        session["stopped"] = ("raised", exc, started)
                    handler(None, exc)

            capture.await_handler(stop, clock() + timeout_ms / 1000, pump=bridge.pump, clock=clock)
            if session["stopped"] is not None and self._stop_confirms(session):
                how = "its stop answered"
            else:  # the stream's own end may have come while its stop was awaited
                witness = self._stop_witness(session)
                how = witness["detail"] if witness["confirmed"] else ""
        if not how:
            return {"confirmed": False, "detail": witness["detail"]}
        if not keep_slot:
            try:
                bridge.remove_output(session["stream"], session["native"][0])
            except Exception:  # noqa: BLE001 - the stream is over either way
                pass
            self._session = None  # the stream, its sink and the window object are let go here
        return {"confirmed": True, "detail": how}

    @staticmethod
    def _stop_confirms(session: dict[str, Any]) -> bool:
        """A stop answer proves the stream is not running: no error, or "not running" when the start had answered
        before that stop was asked (asked earlier, the same error may only mean the start was still under way, and
        the stream may have started since). A stop call that raised is no answer, and a matching code from any
        other error domain proves nothing."""
        kind, error, started = session["stopped"]
        if kind == "ok":
            return True
        if kind != "error" or not started:
            return False
        facts = capture._error_facts(error)
        return (facts.get("error_domain"), facts.get("error_code")) == (_SCSTREAM_DOMAIN, _NOT_STOPPABLE)

    @staticmethod
    def _ids(session: dict[str, Any]) -> tuple[int, int]:
        return int(session["window"]["window_id"]), int(session["app"]["pid"])

    def _listed(self, window_id: int, pid: int) -> bool | None:
        """Whether WindowServer lists this exact window and owner. A failed lookup gives no answer."""
        try:
            return self._bridge.window_listed(window_id, pid)
        except Exception:  # noqa: BLE001
            return None

    def _identity(self, session: dict[str, Any]) -> dict[str, Any]:
        """Which window the session was bound to at ``begin``; its title is that moment's, never read again."""
        window_id, pid = self._ids(session)
        return {"window": {"window_id": window_id, "pid": pid, "title_at_begin": session["window"].get("title")},
                "app": session["app"].get("app")}

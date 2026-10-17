"""macOS meeting capture through ScreenCaptureKit: audio only, the meeting application only.

ScreenCaptureKit (macOS 13+) captures the sound of chosen applications without an
audio driver. The content filter includes only the meeting application (Zoom,
Teams, or the browser that hosts the meeting), and the stream excludes this
process's own audio, so the hints Ouroboros whispers to the owner, played by the
owner's surface, stay out of the meeting audio. A browser filter cannot separate
its tabs: if the Ouroboros page plays in that same browser, the whisper is in the
capture and the companion says so.

ScreenCaptureKit was chosen over Core Audio process taps (macOS 14.4+): it works
from macOS 13, needs no usage-description key in the app bundle, and delivers
sample buffers on a dispatch queue instead of a real-time IOProc thread, which a
Python callback must not hold.

Permission. Recording system audio needs the owner's "Screen & System Audio
Recording" consent, which macOS attributes to the responsible process: the
packaged Ouroboros app when the host spawned this companion, a terminal when run
by hand (``scripts/probe_capture.py`` records which). ``CGPreflightScreenCaptureAccess``
reads the state without prompting; the prompt appears only when listening is
actually requested and consent is missing.
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import platform as _platform
import threading
import time
import types
from typing import Any, Callable, Dict, List, Optional

from capture_common import CaptureFailed, OnError, OnPcm, float32_mono, int16_mono
from meeting_apps import RunningApp, Selection

CAPTURE_RATE = 48000
CONTENT_TIMEOUT_SEC = 10.0
START_TIMEOUT_SEC = 10.0
STOP_TIMEOUT_SEC = 5.0
SC_STREAM_OUTPUT_SCREEN = 0
SC_STREAM_OUTPUT_AUDIO = 1
NS_ACTIVATION_POLICY_REGULAR = 0
SC_ERROR_USER_DECLINED = -3801
K_AUDIO_FORMAT_FLAG_IS_FLOAT = 1
K_AUDIO_FORMAT_FLAG_IS_NON_INTERLEAVED = 1 << 5
CORE_GRAPHICS = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"

PERMISSION_PENDING = (
    "macOS is asking the owner to allow Ouroboros to record system audio (System Settings > "
    "Privacy & Security > Screen & System Audio Recording). Once the owner allows it, ask me to "
    "listen again; macOS may ask to reopen Ouroboros first."
)
PERMISSION_NOT_GRANTED = (
    "macOS has not allowed Ouroboros to record system audio, and it shows its prompt only once. If "
    "the prompt is still open, the owner answers it; otherwise the owner turns Ouroboros on in "
    "System Settings > Privacy & Security > Screen & System Audio Recording. Then ask me to listen "
    "again; macOS may ask to reopen Ouroboros first."
)
# The prompt was shown once on this computer: macOS will not show it again.
REQUESTED_FILE = "permission_requested.json"


def load_frameworks() -> types.SimpleNamespace:
    """The pyobjc modules, installed into this skill's isolated environment."""
    try:
        import AppKit
        import CoreMedia
        import Foundation
        import objc
        import ScreenCaptureKit
    except ImportError as exc:
        raise CaptureFailed("MEETING_DEPENDENCY_MISSING",
                            f"ScreenCaptureKit for Python is not installed for this skill ({exc}); "
                            "its dependency install has not finished") from None
    return types.SimpleNamespace(objc=objc, AppKit=AppKit, Foundation=Foundation, CoreMedia=CoreMedia,
                                 SCK=ScreenCaptureKit)


class ScreenCaptureAccess:
    """The TCC state through CoreGraphics: preflight never prompts, request shows the prompt."""

    def __init__(self, library: Any = None):
        self._library = library

    def _function(self, name: str) -> Callable[[], bool]:
        library = self._library if self._library is not None else ctypes.CDLL(CORE_GRAPHICS)
        function = getattr(library, name)
        function.argtypes = []
        function.restype = ctypes.c_bool
        return function

    def granted(self) -> bool:
        return bool(self._function("CGPreflightScreenCaptureAccess")())

    def request(self) -> bool:
        return bool(self._function("CGRequestScreenCaptureAccess")())


def _macos_supported(version: str) -> bool:
    try:
        return int((version or "0").split(".")[0]) >= 13
    except ValueError:
        return False


def _error_text(error: Any) -> str:
    try:
        return str(error.localizedDescription())
    except Exception:
        return str(error)


def _error_code(error: Any) -> Optional[int]:
    try:
        return int(error.code())
    except Exception:
        return None


_HANDLER_CLASS: Dict[str, Any] = {}


def _handler_class(fw: types.SimpleNamespace) -> Any:
    """One Objective-C class per process (class names are global in the runtime)."""
    if "class" not in _HANDLER_CLASS:
        protocols = [fw.objc.protocolNamed("SCStreamOutput"), fw.objc.protocolNamed("SCStreamDelegate")]

        class OuroborosMeetingCaptureHandler(fw.Foundation.NSObject, protocols=protocols):
            def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
                sink = getattr(self, "sink", None)
                if sink is not None:
                    sink.sample(sample_buffer, output_type)

            def stream_didStopWithError_(self, stream, error):
                sink = getattr(self, "sink", None)
                if sink is not None:
                    sink.stopped_with(error)

        _HANDLER_CLASS["class"] = OuroborosMeetingCaptureHandler
    return _HANDLER_CLASS["class"]


class MacBackend:
    """Capture targets and capture on macOS; speaking needs a driver and is not offered."""

    platform = "darwin"
    speaking_supported = False

    def __init__(self, state_dir: pathlib.Path, *, frameworks: Optional[types.SimpleNamespace] = None,
                 access: Optional[ScreenCaptureAccess] = None, macos_version: Optional[str] = None):
        self.state_dir = pathlib.Path(state_dir)
        self._fw = frameworks
        self._access = access or ScreenCaptureAccess()
        self._version = _platform.mac_ver()[0] if macos_version is None else macos_version

    @property
    def fw(self) -> types.SimpleNamespace:
        if self._fw is None:
            self._fw = load_frameworks()
        return self._fw

    def _require_version(self) -> None:
        if not _macos_supported(self._version):
            raise CaptureFailed("MEETING_PLATFORM_UNSUPPORTED",
                                f"Listening on macOS needs macOS 13 or later; this Mac runs {self._version}.")

    def permission(self) -> tuple:
        """granted, pending (never asked here: the prompt can show) or denied (asked, still no)."""
        self._require_version()
        if self._access.granted():
            return "granted", ""
        if (self.state_dir / REQUESTED_FILE).exists():
            return "denied", PERMISSION_NOT_GRANTED
        return "pending", PERMISSION_PENDING

    def request_permission(self) -> None:
        self._access.request()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / REQUESTED_FILE).write_text(
            json.dumps({"requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), encoding="utf-8")

    def running_apps(self) -> List[RunningApp]:
        workspace = self.fw.AppKit.NSWorkspace.sharedWorkspace()
        apps = []
        for app in workspace.runningApplications() or []:
            if int(app.activationPolicy()) != NS_ACTIVATION_POLICY_REGULAR:
                continue
            apps.append(RunningApp(name=str(app.localizedName() or ""), ident=str(app.bundleIdentifier() or ""),
                                   pid=int(app.processIdentifier())))
        return apps

    def open_capture(self, selection: Selection) -> "MacCapture":
        self._require_version()
        return MacCapture(self.fw, selection)

    def cleanup_leftovers(self) -> List[int]:
        return []

    def open_voice(self, rate: int) -> Any:
        raise CaptureFailed("MEETING_SPEAK_UNSUPPORTED", "Speaking mode is not available on macOS.")

    def run_main_loop(self, stop_event: threading.Event) -> None:
        """Pump the main run loop: some framework callbacks are delivered on it."""
        try:
            foundation = self.fw.Foundation
        except CaptureFailed:
            stop_event.wait()
            return
        loop = foundation.NSRunLoop.currentRunLoop()
        while not stop_event.is_set():
            ran = loop.runMode_beforeDate_(foundation.NSDefaultRunLoopMode,
                                           foundation.NSDate.dateWithTimeIntervalSinceNow_(0.25))
            if not ran:
                stop_event.wait(0.25)


class MacCapture:
    """One ScreenCaptureKit audio stream of the selected applications, mono at 48 kHz.

    ``stop`` is final and waits for a ``start`` in progress, so a call that ends while
    macOS is still starting the stream never leaves it recording.
    """

    rate = CAPTURE_RATE

    def __init__(self, fw: types.SimpleNamespace, selection: Selection):
        self.fw = fw
        self.selection = selection
        self.stats = {"buffers": 0, "unreadable": 0}
        self._stream: Any = None
        self._handler: Any = None
        self._on_pcm: Optional[OnPcm] = None
        self._on_error: Optional[OnError] = None
        self._format: Optional[tuple] = None
        self._lifecycle = threading.Lock()
        self._closed = False

    def describe(self) -> Dict[str, Any]:
        return {"backend": "screencapturekit", "source": "application audio", "device": "",
                "includes_own_voice": self.selection.browser}

    def disclosures(self) -> List[str]:
        return []

    def _completion(self, call: Callable[[Callable[..., None]], Any], timeout: float, what: str) -> tuple:
        done, box = threading.Event(), {}

        def completion(*values: Any) -> None:
            box["values"] = values
            done.set()

        call(completion)
        if not done.wait(timeout):
            raise CaptureFailed("MEETING_CAPTURE_FAILED", f"macOS did not {what} in time")
        return box["values"]

    def _content(self) -> Any:
        content, error = self._completion(
            lambda done: self.fw.SCK.SCShareableContent
            .getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(True, False, done),
            CONTENT_TIMEOUT_SEC, "list the applications that can be captured")
        if error is not None:
            if _error_code(error) == SC_ERROR_USER_DECLINED:
                raise CaptureFailed("MEETING_PERMISSION_DENIED",
                                    "macOS refused to let Ouroboros record system audio; the owner can allow it "
                                    "in System Settings > Privacy & Security > Screen & System Audio Recording.")
            raise CaptureFailed("MEETING_CAPTURE_FAILED", f"macOS could not list capturable content: {_error_text(error)}")
        return content

    def start(self, on_pcm: OnPcm, on_error: OnError) -> None:
        with self._lifecycle:
            if self._closed:
                raise CaptureFailed("MEETING_CAPTURE_STOPPED", "listening stopped before the capture started")
            self._start(on_pcm, on_error)

    def _start(self, on_pcm: OnPcm, on_error: OnError) -> None:
        content = self._content()
        wanted_ids = {app.ident for app in self.selection.apps if app.ident}
        wanted_pids = {app.pid for app in self.selection.apps if app.pid}
        apps = [app for app in (content.applications() or [])
                if str(app.bundleIdentifier() or "") in wanted_ids or int(app.processID()) in wanted_pids]
        if not apps:
            raise CaptureFailed("MEETING_APP_NOT_CAPTURABLE",
                                f"macOS does not offer {self.selection.label} for capture right now.")
        displays = list(content.displays() or [])
        if not displays:
            raise CaptureFailed("MEETING_CAPTURE_FAILED", "macOS reports no display to anchor the capture filter.")
        sck = self.fw.SCK
        content_filter = sck.SCContentFilter.alloc().initWithDisplay_includingApplications_exceptingWindows_(
            displays[0], apps, [])
        config = sck.SCStreamConfiguration.alloc().init()
        config.setCapturesAudio_(True)
        config.setSampleRate_(self.rate)
        config.setChannelCount_(1)
        config.setExcludesCurrentProcessAudio_(True)
        # The stream always produces video too; keep it to a 2x2 frame per second.
        config.setWidth_(2)
        config.setHeight_(2)
        config.setMinimumFrameInterval_(self.fw.CoreMedia.CMTimeMake(1, 1))
        handler = _handler_class(self.fw).alloc().init()
        handler.sink = self
        stream = sck.SCStream.alloc().initWithFilter_configuration_delegate_(content_filter, config, handler)
        for kind in (SC_STREAM_OUTPUT_AUDIO, SC_STREAM_OUTPUT_SCREEN):
            ok, error = stream.addStreamOutput_type_sampleHandlerQueue_error_(handler, kind, None, None)
            if not ok:
                raise CaptureFailed("MEETING_CAPTURE_FAILED",
                                    f"macOS refused the capture output: {_error_text(error)}")
        self._on_pcm, self._on_error = on_pcm, on_error
        self._stream, self._handler = stream, handler
        (error,) = self._completion(stream.startCaptureWithCompletionHandler_, START_TIMEOUT_SEC,
                                    "start the capture")
        if error is not None:
            self._stream, self._on_pcm = None, None
            raise CaptureFailed("MEETING_CAPTURE_FAILED", f"macOS could not start the capture: {_error_text(error)}")

    def _audio_format(self, sample_buffer: Any) -> tuple:
        """(channels, planar, float) of the delivered audio; the configuration when unreadable."""
        if self._format is None:
            fmt = (1, True, True)
            try:
                core_media = self.fw.CoreMedia
                description = core_media.CMSampleBufferGetFormatDescription(sample_buffer)
                asbd = core_media.CMAudioFormatDescriptionGetStreamBasicDescription(description)
                asbd = asbd[0] if isinstance(asbd, (tuple, list)) else asbd
                flags = int(asbd.mFormatFlags)
                fmt = (max(1, int(asbd.mChannelsPerFrame)), bool(flags & K_AUDIO_FORMAT_FLAG_IS_NON_INTERLEAVED),
                       bool(flags & K_AUDIO_FORMAT_FLAG_IS_FLOAT))
                if int(asbd.mSampleRate) != self.rate and self._on_error is not None:
                    self._on_error(f"macOS delivered {int(asbd.mSampleRate)} Hz audio instead of {self.rate} Hz")
            except Exception:
                pass
            self._format = fmt
        return self._format

    def sample(self, sample_buffer: Any, output_type: int) -> None:
        on_pcm = self._on_pcm
        if int(output_type) != SC_STREAM_OUTPUT_AUDIO or on_pcm is None:
            return
        try:
            core_media = self.fw.CoreMedia
            block = core_media.CMSampleBufferGetDataBuffer(sample_buffer)
            if block is None:
                return
            length = int(core_media.CMBlockBufferGetDataLength(block))
            # PyObjC's supported way to read a block buffer (its pointer accessors are not
            # available from Python); it either fills the given buffer or returns a copy.
            buffer = bytearray(length)
            result = core_media.CMBlockBufferCopyDataBytes(block, 0, length, buffer)
            status, data = (result[0], result[1]) if isinstance(result, tuple) else (result, buffer)
            if status != 0:
                self.stats["unreadable"] += 1
                return
            channels, planar, is_float = self._audio_format(sample_buffer)
            raw = bytes(data if data is not None else buffer)
            pcm = float32_mono(raw, channels, planar) if is_float else int16_mono(raw, channels)
        except Exception:
            self.stats["unreadable"] += 1
            return
        self.stats["buffers"] += 1
        if pcm:
            on_pcm(pcm, 0)

    def stopped_with(self, error: Any) -> None:
        on_error = self._on_error
        if self._stream is not None and on_error is not None:
            on_error(f"macOS stopped the capture: {_error_text(error)}")

    def stop(self) -> None:
        with self._lifecycle:
            self._closed = True
            self._teardown()

    def _teardown(self) -> None:
        stream, self._stream = self._stream, None
        self._on_pcm = None
        if stream is None:
            return
        try:
            self._completion(stream.stopCaptureWithCompletionHandler_, STOP_TIMEOUT_SEC, "stop the capture")
        except CaptureFailed:
            pass
        if self._handler is not None:
            self._handler.sink = None
            self._handler = None

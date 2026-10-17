"""Ouroboros's own Chrome window on one meeting page, driven through Playwright.

The window is visible and uses a persistent profile in the skill's state
directory: that profile is Ouroboros's own browser, where the owner signs
Ouroboros's own Zoom, Google or Microsoft account in once. Playwright's
``chrome`` channel launches the Google Chrome already installed on the computer;
nothing is downloaded. Microphone permission is granted only to the meeting
service's origins, and ``participant_page.js`` (an init script) replaces the
page's microphone and taps its remote audio. Chrome starts with fake capture
devices (``--use-fake-device-for-media-stream``), so no path in this browser,
whichever frame or API it uses, reaches the computer's real microphone or camera.

The window's own sound is muted (``--mute-audio``): Ouroboros hears the meeting
through the tap, and a second copy playing from the speakers would echo into
any microphone in the room. Playwright's synchronous API is not thread-safe, so
every Playwright call happens on this session's one thread; other threads hand
it work through a queue.
"""

from __future__ import annotations

import base64
import pathlib
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

PAGE_SCRIPT = pathlib.Path(__file__).with_name("participant_page.js")
PROFILE_DIR = "chrome-profile"
RATE = 48000
LAUNCH_ARGS = ("--autoplay-policy=no-user-gesture-required", "--mute-audio", "--use-fake-device-for-media-stream")
OPEN_TIMEOUT_SEC = 90.0
NAVIGATION_TIMEOUT_MS = 60000
STATUS_EVERY_SEC = 1.0
PUMP_MS = 10
MAX_STREAMS = 16  # the call protocol's per-peer stream bound

STATUS_SCRIPT = "() => window.__ouroborosParticipant ? window.__ouroborosParticipant.status() : null"


class SessionFailed(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def load_sync_playwright() -> Callable[[], Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SessionFailed("MEETING_DEPENDENCY_MISSING",
                            f"Playwright is not installed for this skill ({exc}); its dependency install has "
                            "not finished") from None
    return sync_playwright


def launch_failure(error: Exception) -> SessionFailed:
    text = str(error)
    if "distribution 'chrome' is not found" in text or "Executable doesn't exist" in text:
        return SessionFailed("MEETING_BROWSER_CHROME_MISSING",
                             "The browser participant uses the Google Chrome installed on this computer, and "
                             "Chrome is not installed.")
    if "already in use" in text or "ProcessSingleton" in text:
        return SessionFailed("MEETING_BROWSER_PROFILE_BUSY",
                             "Ouroboros's browser profile is open in another window; close it and try again.")
    return SessionFailed("MEETING_BROWSER_FAILED", f"Chrome could not open the meeting: {text[:400]}")


class BrowserSession:
    """One visible window on one meeting page; created, driven and closed on its own thread."""

    def __init__(self, state_dir: pathlib.Path, url: str, origin: str, *,
                 on_remote_pcm: Callable[[bytes, int], None], on_closed: Callable[[str], None],
                 belongs: Callable[[str], bool], playwright_factory: Optional[Callable[[], Any]] = None):
        self.profile = pathlib.Path(state_dir) / PROFILE_DIR
        self.url = url
        self.origin = origin
        self.page_status: Dict[str, Any] = {}
        self.title = ""
        self.page_url = ""
        self.events: List[str] = []
        self.failure: Optional[SessionFailed] = None
        self.epoch: Optional[int] = None
        self._on_remote_pcm = on_remote_pcm
        self._on_closed = on_closed
        self._belongs = belongs
        self._factory = playwright_factory
        self._commands: "queue.Queue" = queue.Queue()
        self._opened = threading.Event()
        self._stopping = False
        self._closing = False
        self._closed_reason = ""
        self._granted = {origin}
        self._pending_origins: List[str] = []
        self._mic_frames: List[Any] = []
        self._streams: Dict[int, int] = {}
        self._thread = threading.Thread(target=self._run, name="meeting-browser", daemon=True)

    # -- other threads ----------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def open(self) -> None:
        self._thread.start()
        if not self._opened.wait(OPEN_TIMEOUT_SEC):
            self.close("opening timed out")
            raise SessionFailed("MEETING_BROWSER_FAILED", "Chrome did not open the meeting page in time.")
        if self.failure is not None:
            raise self.failure

    def feed(self, pcm: bytes, epoch: Optional[int]) -> None:
        if not self._closing:  # a closed window takes no more call audio
            self._commands.put(("feed", epoch, pcm))

    def flush(self, epoch: int) -> None:
        self.epoch = epoch
        self._commands.put(("flush", epoch, b""))

    def call(self, function: Callable[[Any], Any], timeout: float = 30.0) -> Any:
        """Run ``function(page)`` on the session thread and return its result."""
        done, box = threading.Event(), {}
        self._commands.put(("call", function, (done, box)))
        if not done.wait(timeout):
            raise SessionFailed("MEETING_BROWSER_FAILED", "the browser did not answer in time")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def close(self, reason: str = "closed") -> None:
        self._closing = True
        self._commands.put(("close", reason, b""))
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=15.0)

    # -- the session thread ------------------------------------------------------------
    def _run(self) -> None:
        try:
            factory = self._factory or load_sync_playwright()
            with factory() as playwright:
                try:
                    context = playwright.chromium.launch_persistent_context(
                        str(self.profile), channel="chrome", headless=False, args=list(LAUNCH_ARGS),
                        no_viewport=True)
                except Exception as exc:
                    raise launch_failure(exc) from None
                try:
                    self._drive(context)
                finally:
                    try:
                        context.close()
                    except Exception:
                        pass
        except SessionFailed as exc:
            self.failure = exc
        except Exception as exc:  # the browser died under us: report, never raise into a thread
            self.failure = SessionFailed("MEETING_BROWSER_FAILED", f"the browser stopped: {exc}")
        finally:
            first_open = not self._opened.is_set()
            self._opened.set()
            if not first_open and not self._stopping:
                self._on_closed(self._closed_reason or (self.failure.message if self.failure else "the browser closed"))

    def _drive(self, context: Any) -> None:
        context.grant_permissions(["microphone"], origin=self.origin)
        context.expose_binding("__ouroborosRemotePcm", self._remote_pcm)
        context.expose_binding("__ouroborosParticipantEvent", self._participant_event)
        context.add_init_script(path=str(PAGE_SCRIPT))
        context.on("close", lambda *_: self._mark_closed("the browser window was closed"))
        page = context.pages[0] if context.pages else context.new_page()
        page.on("framenavigated", self._navigated)
        page.on("close", lambda *_: self._mark_closed("the meeting page was closed"))
        try:
            page.goto(self.url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
        except Exception as exc:
            raise SessionFailed("MEETING_BROWSER_FAILED", f"the meeting page did not load: {str(exc)[:300]}") from None
        self._opened.set()
        self._pump(context, page)

    def _mark_closed(self, reason: str) -> None:
        self._closed_reason = self._closed_reason or reason

    def _navigated(self, frame: Any) -> None:
        url = getattr(frame, "url", "") or ""
        if url.startswith("https://") and self._belongs(url):
            origin = "/".join(url.split("/", 3)[:3])
            if origin not in self._granted:
                self._pending_origins.append(origin)

    def _remote_pcm(self, source: Dict[str, Any], text: str, rate: int = RATE) -> None:
        if int(rate) != RATE:
            return
        frame = source.get("frame") if isinstance(source, dict) else None
        key = id(frame)
        if key not in self._streams:
            if len(self._streams) >= MAX_STREAMS:
                return
            self._streams[key] = len(self._streams)
        try:
            pcm = base64.b64decode(text, validate=True)
        except (ValueError, TypeError):
            return
        self._on_remote_pcm(pcm[: len(pcm) - len(pcm) % 2], self._streams[key])

    def _participant_event(self, source: Dict[str, Any], name: str, detail: str = "") -> None:
        self.events.append(str(name))
        del self.events[:-20]
        frame = source.get("frame") if isinstance(source, dict) else None
        if name == "microphone_requested" and frame is not None and frame not in self._mic_frames:
            self._mic_frames.append(frame)

    def _feed_frames(self, script: str, argument: Any = None) -> None:
        for frame in list(self._mic_frames):
            try:
                frame.evaluate(script, argument)
            except Exception:
                self._mic_frames.remove(frame)

    def _pump(self, context: Any, page: Any) -> None:
        next_status = 0.0
        while not self._stopping and not self._closed_reason:
            batch: List[bytes] = []
            while True:
                try:
                    kind, first, second = self._commands.get_nowait()
                except queue.Empty:
                    break
                if kind == "close":
                    self._stopping = True
                    return
                if kind == "flush":
                    batch = []
                    self._feed_frames("() => window.__ouroborosFlush()")
                elif kind == "feed" and (self.epoch is None or first is None or first == self.epoch):
                    batch.append(second)
                elif kind == "call":
                    done, box = second
                    try:
                        box["value"] = first(page)
                    except Exception as exc:
                        box["error"] = exc
                    done.set()
            if batch:
                self._feed_frames("(text) => window.__ouroborosFeed(text)",
                                  base64.b64encode(b"".join(batch)).decode("ascii"))
            while self._pending_origins:
                origin = self._pending_origins.pop()
                context.grant_permissions(["microphone"], origin=origin)
                self._granted.add(origin)
            if time.monotonic() >= next_status:
                next_status = time.monotonic() + STATUS_EVERY_SEC
                try:
                    self.page_status = page.evaluate(STATUS_SCRIPT) or {}
                    self.title, self.page_url = page.title(), page.url
                except Exception:
                    if page.is_closed():
                        self._mark_closed("the meeting page was closed")
            page.wait_for_timeout(PUMP_MS)

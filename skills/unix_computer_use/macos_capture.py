"""One ScreenCaptureKit image of one exact macOS window, run as a short-lived helper process.

``python -I -B macos_capture.py <options json>`` prints one JSON line (``SPEC`` names the options) and exits 0 only
after the PNG at ``output`` is complete; any other ending is a failure the plugin never reads as a capture. Like
``macos_native.py`` it is its own process, so a stuck framework call can be killed and never takes the host down.

The application is resolved launch-free by ``macos_native.resolve_target``. A positive integer names a CGWindowID;
a nonblank string names an exact title, including a title consisting of digits.
``macos_native.py`` is loaded from this folder by its path: ``-I`` keeps the script's folder off
``sys.path``. Only windows ScreenCaptureKit lists for that application's process qualify; none or several is a
refusal, never another window and never a display.

The PyObjC boundary (``Bridge``) is ScreenCaptureKit loaded from the system framework by ``objc.loadBundle`` (the
shipped interpreter carries pyobjc-core, Cocoa and Quartz, no ScreenCaptureKit wrapper). A completion handler can
become an Objective-C block only once its signature is registered (``HANDLERS``), so registration happens before any
use and is read back; an unregistered signature refuses, it is never skipped. Screen Recording is read through the
prompt-free preflight first: without the grant nothing in ScreenCaptureKit is called, so no permission prompt
appears. Each handler is waited for until one deadline shared by both calls (``timeout_ms``) while the main run loop
is serviced in short slices; no call starts after that deadline and a handler running after it is ignored. The image
is the window alone at its own point-to-pixel scale (``SCScreenshotManager``, macOS 14+): no other window, no cursor,
no shadow. Quartz encodes it as PNG, written beside the final name and then hard-linked to it, so an existing file is
never replaced and a partial file never carries the final name. Nothing here activates, raises or clicks; the image
carries no display coordinates.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import pathlib
import platform
import sys
import threading
import time
from typing import Any, Callable

SPEC = {"app": "", "window": None, "output": "", "timeout_ms": 5000}
TIMEOUT_MS = {"min": 500, "max": 20000}  # one budget for both completion handlers
MIN_MACOS = 14  # SCScreenshotManager and SCContentFilter.contentRect/pointPixelScale
MAX_SIDE_PX = 16384  # a larger window is scaled down by one factor for both sides, so its aspect ratio is kept
_SCK_PATH = "/System/Library/Frameworks/ScreenCaptureKit.framework"
_CLASSES = ("SCShareableContent", "SCContentFilter", "SCStreamConfiguration", "SCScreenshotManager")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SHOWN = 20  # the application's windows listed in a refusal
_SLICE_S = 0.05  # one main-run-loop service between checks for a handler
_TCC_DECLINED = ("com.apple.ScreenCaptureKit.SCStreamErrorDomain", -3801)  # SCStreamErrorUserDeclined


def _handler(first: bytes) -> dict[str, Any]:
    """A ``void (^)(first, NSError *)`` completion handler: argument 0 is the block itself."""
    return {"retval": {"type": b"v"}, "arguments": {0: {"type": b"^v"}, 1: {"type": first}, 2: {"type": b"@"}}}


# (class, selector) -> PyObjC metadata. Selector arguments count self and _cmd, so a method's first argument is 2.
HANDLERS: dict[tuple[str, str], dict[str, Any]] = {
    ("SCShareableContent", "getShareableContentExcludingDesktopWindows:onScreenWindowsOnly:completionHandler:"):
        {"arguments": {4: {"callable": _handler(b"@")}}},
    ("SCScreenshotManager", "captureImageWithFilter:configuration:completionHandler:"):
        {"arguments": {4: {"callable": _handler(b"^{CGImage=}")}}},
}


def load_native() -> Any:
    """Load the launch-free application resolver beside this file by path."""
    path = pathlib.Path(__file__).resolve().with_name("macos_native.py")
    spec = importlib.util.spec_from_file_location("_unix_computer_use_macos_native", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registered_handler_problem(classes: dict[str, Any], handlers: dict[tuple[str, str], dict[str, Any]] = HANDLERS) -> str:
    """'' when PyObjC holds every handler signature of ``handlers`` as registered, else what it holds instead."""
    for (cls, selector), meta in handlers.items():
        method = getattr(classes[cls], selector.replace(":", "_"))
        held_arguments = method.__metadata__().get("arguments") or ()
        for index, wanted in meta["arguments"].items():
            if "callable" not in wanted:
                continue
            want = [wanted["callable"]["arguments"][i]["type"] for i in sorted(wanted["callable"]["arguments"])]
            held = (held_arguments[index].get("callable") or {}) if index < len(held_arguments) else {}
            types = [arg.get("type") for arg in held.get("arguments") or ()]
            if types != want:
                return f"{cls} {selector}: completion handler signature not registered (PyObjC holds {types!r})"
    return ""


class Bridge:
    """The PyObjC side of ScreenCaptureKit, one instance per helper process; tests replace it with a fake."""

    def __init__(self) -> None:
        import objc
        import Quartz
        from CoreFoundation import CFRunLoopRunInMode, kCFRunLoopDefaultMode
        self.objc, self.quartz = objc, Quartz
        self._run_loop = lambda seconds: CFRunLoopRunInMode(kCFRunLoopDefaultMode, seconds, True)
        self.classes: dict[str, Any] = {}

    def screen_recording(self) -> str:
        """``granted`` or ``denied`` from the prompt-free preflight; the request that prompts is never made."""
        return "granted" if self.quartz.CGPreflightScreenCaptureAccess() else "denied"

    def load(self, handlers: dict[tuple[str, str], dict[str, Any]] = HANDLERS) -> str:
        """Load ScreenCaptureKit and register ``handlers`` -> '' when ready, else why it is not."""
        objc = self.objc
        objc.loadBundle("ScreenCaptureKit", {}, bundle_path=_SCK_PATH, scan_classes=False)
        for (cls, selector), meta in handlers.items():
            objc.registerMetaDataForSelector(cls.encode(), selector.encode(), meta)
        try:
            self.classes = {name: objc.lookUpClass(name) for name in {*_CLASSES, *(cls for cls, _s in handlers)}}
        except objc.nosuchclass_error as exc:
            return f"ScreenCaptureKit has no class {exc} (macOS {MIN_MACOS}+ is required)"
        # The helper has no NSApplication: CoreGraphics opens this process's window-server session before
        # ScreenCaptureKit is asked anything.
        self.quartz.CGMainDisplayID()
        return registered_handler_problem(self.classes, handlers)

    def pump(self, seconds: float) -> None:
        """Service the main run loop once, so a handler delivered on the main queue arrives too."""
        self._run_loop(seconds)

    def shareable_content(self, handler: Callable[[Any, Any], None]) -> None:
        cls = self.classes["SCShareableContent"]
        cls.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(True, False, handler)

    @staticmethod
    def windows(content: Any) -> list[Any]:
        return list(content.windows() or [])

    @staticmethod
    def window_facts(window: Any) -> dict[str, Any]:
        """window_id, title (None when it has none), owner pid, bounds [x, y, w, h] in global points, on_screen, layer."""
        owner = window.owningApplication()
        frame = window.frame()
        title = window.title()
        return {"window_id": int(window.windowID()), "title": None if title is None else str(title),
                "pid": int(owner.processID()) if owner is not None else None,
                "bounds": [round(frame.origin.x), round(frame.origin.y), round(frame.size.width), round(frame.size.height)],
                "on_screen": bool(window.isOnScreen()), "layer": int(window.windowLayer())}

    def content_filter(self, window: Any) -> Any:
        return self.classes["SCContentFilter"].alloc().initWithDesktopIndependentWindow_(window)

    @staticmethod
    def content_size(content_filter: Any) -> tuple[float, float, float]:
        """(width, height) of the window's content in points and its point-to-pixel scale."""
        rect = content_filter.contentRect()
        return float(rect.size.width), float(rect.size.height), float(content_filter.pointPixelScale())

    def configuration(self, width: int, height: int) -> tuple[Any, dict[str, Any]]:
        """The window at ``width`` x ``height`` pixels without cursor, shadow or display clip -> (config, read back)."""
        config = self.classes["SCStreamConfiguration"].alloc().init()
        config.setWidth_(width)
        config.setHeight_(height)
        config.setShowsCursor_(False)
        config.setIgnoreShadowsSingleWindow_(True)
        config.setIgnoreGlobalClipSingleWindow_(True)
        return config, {"width": int(config.width()), "height": int(config.height()),
                        "cursor_included": bool(config.showsCursor()),
                        "shadow_included": not bool(config.ignoreShadowsSingleWindow())}

    def screenshot(self, content_filter: Any, config: Any, handler: Callable[[Any, Any], None]) -> None:
        cls = self.classes["SCScreenshotManager"]
        cls.captureImageWithFilter_configuration_completionHandler_(content_filter, config, handler)

    def image_size(self, image: Any) -> tuple[int, int]:
        return int(self.quartz.CGImageGetWidth(image)), int(self.quartz.CGImageGetHeight(image))

    def png(self, image: Any) -> bytes:
        """The CGImage as PNG bytes through ImageIO; b'' when Quartz could not encode it."""
        from Foundation import NSMutableData
        data = NSMutableData.data()
        destination = self.quartz.CGImageDestinationCreateWithData(data, "public.png", 1, None)
        if destination is None:
            return b""
        self.quartz.CGImageDestinationAddImage(destination, image, None)
        return bytes(data) if self.quartz.CGImageDestinationFinalize(destination) else b""


def await_handler(start: Callable[[Callable[[Any, Any], None]], None], deadline: float, *,
                  pump: Callable[[float], None], clock: Callable[[], float] = time.monotonic) -> tuple[bool, Any, Any]:
    """Start one completion-handler call and wait for its first answer until ``deadline`` -> (arrived, value, error).

    Nothing is started once ``deadline`` has passed. The handler may run on any queue, also inside ``start`` or a
    ``pump``: the main run loop is serviced meanwhile. An answer counts only when the handler runs before ``deadline``
    (read on ``clock`` under the lock), and then it stands even if the wait notices it later; an answer at or after
    the deadline, a second answer and any answer once the wait has ended are dropped without being read."""
    box: dict[str, Any] = {}
    done = threading.Event()
    lock = threading.Lock()
    closed = [False]

    def handler(value: Any, error: Any) -> None:
        with lock:
            if closed[0] or done.is_set() or clock() >= deadline:
                return
            box["value"], box["error"] = value, error
            done.set()

    if clock() >= deadline:
        return False, None, None
    start(handler)
    while not done.is_set():
        left = deadline - clock()
        if left <= 0:
            break
        pump(min(_SLICE_S, left))
        done.wait(min(0.01, max(0.0, deadline - clock())))
    with lock:
        closed[0] = True
        arrived = done.is_set()
    return arrived, box.get("value"), box.get("error")


def mac_major() -> int | None:
    head = platform.mac_ver()[0].split(".")[0]
    return int(head) if head.isdigit() else None


def window_selector(window: Any) -> tuple[dict[str, Any], str]:
    """The two capture APIs use the JSON type to distinguish an id from an exact title."""
    if isinstance(window, str) and window.strip():
        return {"title": window}, ""
    if type(window) is int and window > 0:
        return {"window_id": window}, ""
    if window is None or isinstance(window, str):
        return {}, "window is required: a positive integer window_id or a nonblank exact title"
    return {}, "window must be a window_id or an exact title; use a positive integer for an ID"


def parse_spec(raw: Any) -> tuple[dict[str, Any], str]:
    """The options -> ({app, want, output, timeout_ms}, error); nothing is resolved or captured here."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError as exc:
        return {}, f"options are not JSON ({exc})"
    if not isinstance(data, dict):
        return {}, "options must be a JSON object"
    unknown = sorted(set(data) - set(SPEC))
    if unknown:
        return {}, f"unknown option(s) {unknown}; known: {sorted(SPEC)}"
    app = data.get("app")
    if not isinstance(app, str) or not app.strip():
        return {}, "app is required: the application that owns the window"
    want, error = window_selector(data.get("window"))
    if error:
        return {}, error
    output = data.get("output")
    if not isinstance(output, str) or not os.path.isabs(output) or not output.endswith(".png"):
        return {}, "output must be an absolute path ending in .png"
    if not os.path.isdir(os.path.dirname(output)) or os.path.lexists(output):
        return {}, "output must name a new file in an existing folder"
    timeout = data.get("timeout_ms", SPEC["timeout_ms"])
    if isinstance(timeout, bool) or not isinstance(timeout, int):
        return {}, "timeout_ms must be an integer"
    timeout = max(TIMEOUT_MS["min"], min(timeout, TIMEOUT_MS["max"]))
    return {"app": app, "want": want, "output": output, "timeout_ms": timeout}, ""


def select_window(rows: list[tuple[Any, dict[str, Any]]], pid: int, want: dict[str, Any]
                  ) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    """The one window of process ``pid`` that ``want`` names -> (window, facts, refusal)."""
    own = [(window, facts) for window, facts in rows if facts.get("pid") == pid]
    key = "window_id" if "window_id" in want else "title"
    picked = [(window, facts) for window, facts in own if facts.get(key) == want[key]]
    if len(picked) == 1:
        return picked[0][0], picked[0][1], None
    listed = {"windows": [facts for _w, facts in own[:_SHOWN]], "windows_total": len(own)}
    if not own:
        return None, {}, {"reason": "no_window", "error": "ScreenCaptureKit lists no window of the application", **listed}
    if picked:
        return None, {}, {"reason": "window_ambiguous", **listed,
                          "error": f"{len(picked)} windows of the application have {key} {want[key]!r}; name one by its window_id"}
    return None, {}, {"reason": "window_not_found", **listed,
                      "error": f"no window of the application has {key} {want[key]!r}"}


def pixel_size(width_pt: float, height_pt: float, scale: float) -> tuple[int, int]:
    """The content in pixels at its own scale; past ``MAX_SIDE_PX`` both sides shrink by one factor. (0, 0) = none."""
    if not all(math.isfinite(v) and v > 0 for v in (width_pt, height_pt, scale)):
        return 0, 0
    width, height = width_pt * scale, height_pt * scale
    shrink = min(1.0, MAX_SIDE_PX / max(width, height))
    return max(1, round(width * shrink)), max(1, round(height * shrink))


def write_new(path: str, data: bytes) -> None:
    """``data`` at ``path``, which must not exist: written beside it, then hard-linked (atomic, never replacing)."""
    part = f"{path}.{os.getpid()}.part"
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.link(part, path)
    finally:
        try:
            os.unlink(part)
        except FileNotFoundError:
            pass


def _error_facts(error: Any) -> dict[str, Any]:
    """An NSError as {error, error_domain, error_code}; any other value as its text."""
    try:
        return {"error": str(error.localizedDescription()), "error_domain": str(error.domain()), "error_code": int(error.code())}
    except Exception:  # noqa: BLE001 - not an NSError
        return {"error": str(error)}


def capture(raw: Any, *, native: Any, bridge_factory: Callable[[], Any] = Bridge,
            clock: Callable[[], float] = time.monotonic, mac_version: Callable[[], int | None] = mac_major) -> dict[str, Any]:
    """The one capture the options ask for -> the helper's answer (``ok`` only with the PNG complete at ``output``)."""
    spec, error = parse_spec(raw)
    if error:
        return {"ok": False, "reason": "bad_option", "error": error}
    major = mac_version()
    if major is None or major < MIN_MACOS:
        return {"ok": False, "reason": "unsupported_os",
                "error": f"window capture needs macOS {MIN_MACOS} or later (this is {platform.mac_ver()[0] or 'unknown'})"}
    target, refusal = native.resolve_target(spec["app"])
    if refusal:
        return refusal
    try:
        bridge = bridge_factory()
    except ImportError as exc:
        return {"ok": False, "reason": "capture_unavailable", "app": target,
                "error": f"PyObjC is not available to this interpreter ({exc})"}
    grant = bridge.screen_recording()
    if grant != "granted":
        return {"ok": False, "reason": "screen_recording_denied", "app": target, "permissions": {"screen_recording": grant},
                "error": "this process holds no Screen Recording grant; nothing was asked or captured"}
    problem = bridge.load()
    if problem:
        return {"ok": False, "reason": "capture_unavailable", "app": target, "error": problem}
    started = clock()
    deadline = started + spec["timeout_ms"] / 1000

    def failed(stage: str, arrived: bool, err: Any, **extra: Any) -> dict[str, Any]:
        if not arrived:
            return {"ok": False, "reason": "capture_timeout", "stage": stage, "app": target, **extra,
                    "error": f"no ScreenCaptureKit answer within the {spec['timeout_ms']} ms both requests share"}
        facts = _error_facts(err if err is not None else "no result and no error")
        declined = (facts.get("error_domain"), facts.get("error_code")) == _TCC_DECLINED
        return {"ok": False, "reason": "screen_recording_denied" if declined else "capture_failed", "stage": stage,
                "app": target, **extra, **facts}

    arrived, content, err = await_handler(bridge.shareable_content, deadline, pump=bridge.pump, clock=clock)
    if not arrived or err is not None or content is None:
        return failed("shareable_content", arrived, err)
    rows = [(window, bridge.window_facts(window)) for window in bridge.windows(content)]
    window, facts, refusal = select_window(rows, int(target["pid"]), spec["want"])
    if refusal:
        return {"ok": False, "app": target, **refusal}
    content_filter = bridge.content_filter(window)
    width_pt, height_pt, scale = bridge.content_size(content_filter)
    width, height = pixel_size(width_pt, height_pt, scale)
    if not width or not height:
        return {"ok": False, "reason": "window_empty", "app": target, "window": facts,
                "error": "the window has no content area to capture"}
    config, applied = bridge.configuration(width, height)
    arrived, image, err = await_handler(lambda handler: bridge.screenshot(content_filter, config, handler), deadline,
                                        pump=bridge.pump, clock=clock)
    if not arrived or err is not None or image is None:
        return failed("screenshot", arrived, err, window=facts)
    image_width, image_height = bridge.image_size(image)
    data = bridge.png(image)
    if not data.startswith(_PNG_SIGNATURE):
        return {"ok": False, "reason": "capture_failed", "stage": "encode", "app": target, "window": facts,
                "error": "Quartz did not encode the window image as PNG"}
    write_new(spec["output"], data)
    return {"ok": True, "backend": "ScreenCaptureKit", "app": target, "window": facts, "output": spec["output"],
            "image_width": image_width, "image_height": image_height, "requested_width": applied["width"],
            "requested_height": applied["height"], "point_pixel_scale": scale,
            "cursor_included": applied["cursor_included"], "shadow_included": applied["shadow_included"],
            "capture_ms": round((clock() - started) * 1000, 1)}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        out: dict[str, Any] = {"ok": False, "reason": "bad_option", "error": "usage: macos_capture.py <options json>"}
    else:
        try:
            out = capture(argv[0], native=load_native())
        except Exception as exc:  # noqa: BLE001 - the plugin renders the failure; nothing here may crash silently
            out = {"ok": False, "reason": "capture_failed", "error": f"{type(exc).__name__}: {exc}"}
    # ASCII-only, one line, flushed: the parent reads exactly this answer under any locale.
    print(json.dumps(out, ensure_ascii=True), flush=True)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    code = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    # A handler dropped after its deadline may still be held by a framework queue: ending without interpreter
    # teardown keeps such a late call from running into a finalizing interpreter.
    os._exit(code)

"""One task's ScreenCaptureKit session of one exact macOS window: what the session decides.

Nothing native runs here: ``macos_stream.WindowSessions`` and its ``FrameStore`` are driven through a fake bridge and
a fake clock, so no test lists windows, asks ScreenCaptureKit for anything, starts a stream, captures, reads a
permission or shows anything. The real boundaries (the MCP SDK, the server process, the manifest contract, PyObjC)
are in ``test_unix_computer_use_window_session_server.py``.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import pathlib
import threading
import time
import zlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SKILL = REPO / "skills" / "unix_computer_use"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stream():
    return _load("unix_window_session_stream", SKILL / "macos_stream.py")


def png_bytes(width: int, height: int) -> bytes:
    """A real, whole PNG (stdlib only)."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")
    rows = b"".join(b"\x00" + b"\x40\x80\xc0" * width for _ in range(height))
    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class _Error:
    def __init__(self, code, text="broken", domain="com.apple.ScreenCaptureKit.SCStreamErrorDomain"):
        self.domain, self.code, self.localizedDescription = (lambda: domain), (lambda: code), (lambda: text)


def _row(window_id, title, pid, on_screen=True):
    return object(), {"window_id": window_id, "title": title, "pid": pid, "bounds": [0, 0, 10, 10],
                      "on_screen": on_screen, "layer": 0}


class FakeBridge:
    """StreamBridge's surface, recording what was asked; nothing native. ``start``/``stop`` answer "ok", an error or
    not at all ("silent"); the handlers stay reachable so a test can answer late."""

    def __init__(self, clock, *, grant="granted", problem="", content=True, windows=None, size=(15.0, 10.0, 2.0),
                 add="", start="ok", stop="ok", listed=True):
        self.clock, self.grant, self.problem, self.content, self.size, self.add = clock, grant, problem, content, size, add
        self.start_answer, self.stop_answer, self.listed = start, stop, listed
        self.rows = windows if windows is not None else [_row(7, "Editor", 11), _row(8, "Other", 11)]
        self.calls, self.stores, self.start_handlers, self.stop_handlers = [], [], [], []
        self.on_pump = None  # run once inside the next pump: what a framework thread would do meanwhile
        self.written = []

    def screen_recording(self):
        self.calls.append("preflight")
        return self.grant

    def load(self):
        self.calls.append("load")
        return self.problem

    def pump(self, seconds):
        self.clock.now += seconds
        hook, self.on_pump = self.on_pump, None
        if hook is not None:
            hook()

    def host_now(self):
        return self.clock.now

    def pool(self):
        return contextlib.nullcontext()

    def shareable_content(self, handler):
        self.calls.append("content")
        if self.content:
            handler("content", None)

    def windows(self, content):
        return [window for window, _facts in self.rows]

    def window_facts(self, window):
        return next(facts for w, facts in self.rows if w is window)

    def content_filter(self, window):
        self.calls.append(("filter", self.window_facts(window)["window_id"]))
        return "filter"

    def content_size(self, content_filter):
        return self.size

    def stream_configuration(self, width, height):
        self.calls.append(("config", width, height))
        return "config", {"width": width, "height": height, "cursor_included": False, "shadow_included": False,
                          "frames_per_second_max": 5.0, "queue_depth": 4, "audio": False, "pixel_format": "BGRA"}

    def stream(self, content_filter, config, store):
        self.stores.append(store)
        return f"stream-{len(self.stores)}", f"sink-{len(self.stores)}"

    def add_output(self, stream, sink):
        self.calls.append("add_output")
        return self.add

    def remove_output(self, stream, sink):
        self.calls.append(("remove_output", stream))

    def _answer(self, answer, handler):
        if answer != "silent":
            handler(None if answer == "ok" else answer)

    def start(self, stream, handler):
        self.calls.append(("start", stream))
        self.start_handlers.append(handler)
        self._answer(self.start_answer, handler)

    def stop(self, stream, handler):
        self.calls.append(("stop", stream))
        self.stop_handlers.append(handler)
        self._answer(self.stop_answer, handler)

    def write_png(self, buffer, path):
        _tag, width, height = buffer
        pathlib.Path(path).write_bytes(png_bytes(width, height))
        self.written.append(path)
        return width, height

    def window_listed(self, window_id, pid):
        self.calls.append(("listed", window_id, pid))
        return self.listed


class FakeNative:
    def __init__(self, window_want, refusal=None):
        self.window_want, self.refusal, self.resolved = window_want, refusal, []

    def resolve_target(self, query):
        self.resolved.append(query)
        return (None, self.refusal) if self.refusal else ({"app": "Editor", "pid": 11, "app_id": "e"}, None)


@pytest.fixture
def desk(stream, monkeypatch):
    """(sessions, bridge, clock) with short budgets: a silent framework costs a test milliseconds."""
    monkeypatch.setattr(stream, "START_TIMEOUT_MS", 300)
    monkeypatch.setattr(stream, "STOP_TIMEOUT_MS", 200)
    monkeypatch.setattr(stream, "SHUTDOWN_STOP_MS", 100)
    clock = _Clock()
    made = {}

    def build(**kwargs):
        version = kwargs.pop("version", 26)
        platform = kwargs.pop("platform", "darwin")
        native = kwargs.pop("native", None) or FakeNative(stream.capture.load_native().window_want)
        bridge = made["bridge"] = FakeBridge(clock, **kwargs)
        sessions = stream.WindowSessions(native=native, bridge_factory=lambda: bridge, clock=clock,
                                         mac_version=lambda: version, platform=platform)
        return sessions, bridge, clock

    yield build
    bridge = made.get("bridge")
    for path in (bridge.written if bridge else []):
        assert not os.path.exists(path), "a written frame was left behind"


def begin(sessions, **kwargs):
    return sessions.begin({"app": "Editor", "window": 7, **kwargs})[0]


def feed(bridge, status="complete", size=(30, 20), **kwargs):
    """What the stream's sink would hand the newest session's store for one sample."""
    if status in ("complete", "started") and "rejected" not in kwargs:
        kwargs = {"buffer": (f"image-{bridge.clock.now}", *size), "width": size[0], "height": size[1],
                  "display_s": bridge.clock.now, **kwargs}
    return bridge.stores[-1].accept(status, **kwargs)


# --- arguments and sizes -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw,error", [
    ([], "arguments must be an object"),
    ({"window": 7}, "app is required"),
    ({"app": "  ", "window": 7}, "app is required"),
    ({"app": "Editor"}, "window is required"),
    ({"app": "Editor", "window": "  "}, "window is required"),
    ({"app": "Editor", "window": True}, "window must be a window_id or an exact title"),
    ({"app": "Editor", "window": 7, "max_width": 4000}, "unknown argument"),
])
def test_begin_arguments_refuse_before_anything_is_resolved(stream, raw, error):
    spec, problem = stream.parse_begin(raw)
    assert spec == {} and error in problem


def test_begin_names_a_window_as_the_one_shot_capture_does(stream):
    assert stream.parse_begin({"app": "A", "window": 42})[0]["want"] == {"window_id": 42}
    assert stream.parse_begin({"app": "A", "window": "0042"})[0]["want"] == {"title": "0042"}
    assert stream.parse_begin({"app": "A", "window": " Notes "})[0]["want"] == {"title": " Notes "}


@pytest.mark.parametrize("raw,error", [
    ({"after_seq": -1}, "after_seq must be a non-negative integer"),
    ({"after_seq": True}, "after_seq must be a non-negative integer"),
    ({"wait_ms": "5"}, "wait_ms must be a non-negative integer"),
    ({"timeout": 5}, "unknown argument"),
    ("frame", "arguments must be an object"),
])
def test_frame_arguments_are_checked_and_the_wait_is_bounded(stream, raw, error):
    assert error in stream.parse_frame(raw)[1]
    assert stream.parse_frame({}) == ({"after_seq": None, "wait_ms": 0}, "")
    assert stream.parse_frame({"after_seq": 0, "wait_ms": 10 ** 9})[0] == {"after_seq": 0, "wait_ms": stream.WAIT_MS_MAX}


def test_a_stream_is_never_larger_than_a_default_screenshot_and_never_enlarged(stream):
    assert (stream.MAX_WIDTH, stream.MAX_HEIGHT) == (1280, 800)
    assert stream.stream_size(2560, 1600) == (1280, 800)
    assert stream.stream_size(3000, 1000) == (1280, 427)  # one factor for both sides
    assert stream.stream_size(1000, 3000) == (267, 800)
    assert stream.stream_size(640, 400) == (640, 400)
    assert stream.stream_size(0, 400) == (0, 0)


@pytest.mark.parametrize("cmtime,seconds", [
    ((3_000_000_000, 1_000_000_000, 1, 0), 3.0),   # a host time: valid, epoch 0
    ((600, 600, 1 | 2, 0), 1.0),                   # "has been rounded" is still a number
    ((0, 1, 1, 0), 0.0),                           # kCMTimeZero
    ((0, 0, 0, 0), None),                          # kCMTimeInvalid
    ((7, 600, 0, 0), None),                        # not valid, whatever its value
    ((0, 0, 1 | 4, 0), None),                      # kCMTimePositiveInfinity
    ((0, 0, 1 | 8, 0), None),                      # kCMTimeNegativeInfinity
    ((0, 0, 1 | 16, 0), None),                     # kCMTimeIndefinite
    ((7, 600, 1 | 4, 0), None),                    # an infinity flag beside a plausible value
    ((7, 0, 1, 0), None),                          # no timescale
    ((7, -600, 1, 0), None),                       # a negative timescale
    ((7, 600, 1, 1), None),                        # another timeline than the host clock's
    ((7, 600, 1, -1), None),
])
def test_only_a_numeric_host_time_becomes_seconds(stream, cmtime, seconds):
    assert stream._seconds(cmtime) == seconds
    assert (stream.numeric_time(cmtime) is None) == (seconds is None and cmtime[3] == 0)  # the epoch is the host rule


def test_a_display_time_that_is_no_host_time_is_not_used(stream):
    """What ``read_sample`` hands on for such a time is None: the report is then timed by its arrival, and says so."""
    clock = _Clock()
    store = stream.FrameStore(clock)
    store.accept("complete", display_s=stream._seconds((0, 0, 1 | 16, 0)), buffer="image", width=1, height=1)
    clock.now += 0.5
    _held, facts = store.snapshot()
    assert (facts["frame_time_basis"], facts["frame_age_ms"]) == ("received", 500.0)


@pytest.mark.parametrize("status", ["idle", "suspended", "stopped"])
def test_native_zero_time_on_nonimage_report_is_received_time(stream, status):
    bridge = object.__new__(stream.StreamBridge)
    bridge.quartz = object()
    bridge._keys = {key: key for key in ("SCStreamFrameInfoStatus", "SCStreamFrameInfoDisplayTime")}
    code = next(code for code, name in stream.FRAME_STATUS.items() if name == status)
    bridge._cm = {
        "CMSampleBufferIsValid": lambda sample: True,
        "CMSampleBufferGetSampleAttachmentsArray": lambda sample, create: [
            {"SCStreamFrameInfoStatus": code, "SCStreamFrameInfoDisplayTime": 0}],
        "CMClockMakeHostTimeFromSystemUnits": lambda ticks: (_ for _ in ()).throw(
            AssertionError("zero status timestamp is not display evidence")),
    }
    assert bridge.read_sample(object(), 0) == {"status": status, "display_s": None}


# --- the frame store ---------------------------------------------------------------------------------------------


def test_only_an_image_advances_seq_and_idle_is_separate_evidence(stream):
    clock = _Clock()
    store = stream.FrameStore(clock)
    assert store.snapshot() == (None, {"state": "live", "seq": 0, "stream": {"reports": {}}})
    assert store.accept("complete", display_s=99.5, buffer="first", width=30, height=20)
    clock.now += 2.0
    assert store.accept("idle", display_s=101.9)
    clock.now += 0.25
    held, facts = store.snapshot()
    assert held["buffer"] == "first" and (facts["seq"], facts["frame_status"]) == (1, "complete")
    assert facts["frame_age_ms"] == 2750.0 and facts["frame_time_basis"] == "display"  # the image's own time
    assert facts["stream"] == {"reports": {"complete": 1, "idle": 1}, "last_report": "idle", "last_report_age_ms": 350.0,
                               "last_report_time_basis": "display"}  # when the stream displayed it, not when it arrived
    assert store.accept("started", buffer="second", width=30, height=20)  # no display time: when it was received
    held, facts = store.snapshot()
    assert (held["buffer"], facts["seq"], facts["frame_time_basis"], facts["frame_age_ms"]) == ("second", 2, "received", 0.0)


@pytest.mark.parametrize("status,extra,named", [
    ("blank", {}, "blank"), ("suspended", {}, "suspended"), ("stopped", {}, "stopped"), ("unknown:9", {}, "unknown:9"),
    ("inactive", {}, "inactive"),
    ("complete", {"rejected": "pixel_format:0x20"}, "pixel_format:0x20"), ("complete", {"rejected": "no_image"}, "no_image"),
])
def test_a_report_that_the_image_is_no_longer_the_window_drops_it(stream, status, extra, named):
    store = stream.FrameStore(_Clock())
    store.accept("complete", buffer="held", width=1, height=1)
    store.accept(status, **extra)
    held, facts = store.snapshot()
    assert held is None and facts["withheld"]["status"] == named and facts["seq"] == 1
    store.accept("idle")  # "unchanged" after a dropped frame restores nothing
    assert store.snapshot()[0] is None
    store.accept("complete", buffer="next", width=1, height=1)
    held, facts = store.snapshot()
    assert held["buffer"] == "next" and facts["seq"] == 2 and "withheld" not in facts


def test_an_ended_store_takes_nothing_again_and_its_first_end_stands(stream):
    store = stream.FrameStore(_Clock())
    store.accept("complete", buffer="held", width=1, height=1)
    assert store.end("user_stopped", by="stream", code=-3817, domain="D") is True
    assert store.end("released", by="session") is False
    assert store.accept("complete", buffer="late", width=1, height=1) is False  # a callback after the end
    held, facts = store.snapshot()
    assert held is None and facts["state"] == "stopped" and facts["seq"] == 1
    assert facts["stop"] == {"reason": "user_stopped", "by": "stream", "code": -3817, "domain": "D", "age_ms": 0.0}
    assert facts["stream"]["reports"] == {"complete": 1} and "stream_end" not in facts  # it is the stop itself


def test_the_streams_own_end_after_the_sessions_is_kept_and_reopens_nothing(stream):
    clock = _Clock()
    store = stream.FrameStore(clock)
    store.accept("complete", buffer="held", width=1, height=1)
    assert store.end("window_gone", by="session") is True
    clock.now += 0.5
    stream._Feed(None, store).ended(_Error(-3817, "stopped by the user"))
    stream._Feed(None, store).ended(_Error(-3821, "stopped by the system"))  # the first stands
    assert store.accept("complete", buffer="late", width=1, height=1) is False
    clock.now += 0.25
    held, facts = store.snapshot()
    assert held is None and facts["state"] == "stopped" and facts["seq"] == 1
    assert facts["stop"] == {"reason": "window_gone", "by": "session", "age_ms": 750.0}
    assert facts["stream_end"] == {"reason": "user_stopped", "code": -3817,
                                    "domain": "com.apple.ScreenCaptureKit.SCStreamErrorDomain", "age_ms": 250.0}


def test_foreign_error_code_does_not_attribute_a_native_stop_to_the_user(stream):
    store = stream.FrameStore(_Clock())
    stream._Feed(None, store).ended(_Error(-3817, domain="OtherDomain"))
    assert store.stop["by"] == "stream" and store.stop["reason"] == "stream_error"
    assert (store.stop["domain"], store.stop["code"]) == ("OtherDomain", -3817)


class _Reader:
    """The part of the bridge a sample callback uses."""

    def __init__(self, answers):
        self.answers = list(answers)

    def pool(self):
        return contextlib.nullcontext()

    def read_sample(self, sample_buffer, output_type):
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def test_a_sample_that_cannot_be_read_drops_the_held_frame(stream):
    """An unread callback may have been a new image: the old one is not served as the window any more."""
    clock = _Clock()
    store = stream.FrameStore(clock)
    image = {"status": "complete", "buffer": "image", "width": 1, "height": 1}
    feed = stream._Feed(_Reader([image, None, ValueError("attachment unreadable"), image, {"status": "invalid_sample"}]), store)
    feed.sample("sample", 0)
    feed.sample("sample", 1)  # no screen sample: nothing changes
    assert store.snapshot()[0]["buffer"] == "image"
    feed.sample("sample", 0)  # raises inside the bridge
    held, facts = store.snapshot()
    assert held is None and facts["withheld"] == {"status": "callback_error", "age_ms": 0.0} and facts["seq"] == 1
    assert facts["stream"]["last_report"] == "callback_error" and facts["stream"]["reports"]["callback_error"] == 1
    assert facts["stream"]["callback_errors"] == ["ValueError: attachment unreadable"] and facts["state"] == "live"
    feed.sample("sample", 0)  # the stream's next readable image is served again
    held, facts = store.snapshot()
    assert held is not None and facts["seq"] == 2 and "withheld" not in facts
    feed.sample("sample", 0)  # a screen sample nothing can be read from
    held, facts = store.snapshot()
    assert held is None and facts["withheld"]["status"] == "invalid_sample"


def test_a_failing_clock_still_drops_the_frame_and_still_ends_the_store(stream):
    """Neither a failed callback nor an end may raise into the framework, whatever broke."""
    broken = []

    def clock():
        if broken:
            raise RuntimeError("the host-time clock gave no numeric time")
        return 100.0
    store = stream.FrameStore(clock)
    feed = stream._Feed(_Reader([{"status": "complete", "buffer": "image", "width": 1, "height": 1}, {"status": "idle"}]), store)
    feed.sample("sample", 0)
    broken.append(True)
    feed.sample("sample", 0)  # accept needs the clock for a report without a display time: it fails
    assert store.frame is None and store.withheld["status"] == "callback_error" and store.withheld["at_s"] is None
    feed.inactive()
    feed.ended(None)
    assert store.stop["reason"] == "stream_error" and store.stop["at_s"] is None and len(store.errors) == 2
    broken.clear()
    _held, facts = store.snapshot()
    assert facts["state"] == "stopped" and facts["stop"]["age_ms"] is None and facts["withheld"]["age_ms"] is None


def test_a_wait_ends_on_a_newer_frame_from_another_thread_or_on_the_end(stream):
    store = stream.FrameStore(lambda: 0.0)
    store.accept("complete", buffer="one", width=1, height=1)
    threading.Timer(0.05, lambda: store.accept("complete", buffer="two", width=1, height=1)).start()
    store.wait(1, time.monotonic() + 10, pump=lambda seconds: None)
    assert store.snapshot()[0]["buffer"] == "two"
    threading.Timer(0.05, lambda: store.end("system_stopped", by="stream")).start()
    store.wait(2, time.monotonic() + 10, pump=lambda seconds: None)
    assert store.snapshot()[1]["state"] == "stopped"


# --- begin -------------------------------------------------------------------------------------------------------


def test_begin_selects_one_window_once_and_starts_one_bounded_stream(desk):
    sessions, bridge, _clock = desk(size=(1600.0, 500.0, 2.0))
    answer = begin(sessions, window="Editor")
    assert answer["ok"] is True and answer["state"] == "live" and answer["backend"] == "ScreenCaptureKit"
    assert answer["permissions_available"] == ["screen_recording"], "a live stream proves the grant"
    assert answer["window"]["window_id"] == 7 and answer["app"]["pid"] == 11
    assert ("config", 1280, 400) in bridge.calls and ("filter", 7) in bridge.calls  # 3200x1000 fitted into 1280x800
    order = [bridge.calls.index(step) for step in ("preflight", "load", "content", "add_output", ("start", "stream-1"))]
    assert order == sorted(order)
    assert answer["stream"]["frames_per_second_max"] == 5.0 and answer["stream"]["queue_depth"] == 4
    assert answer["mapped_input_available"] is False and "observation_id" not in answer
    assert answer["window_gone_check"] == "window_server" and ("listed", 7, 11) in bridge.calls
    assert sessions.begin({"app": "Editor", "window": 7}) == sessions.begin({"app": "Editor", "window": 7})
    assert bridge.calls.count("content") == 1  # the second and third begin asked nothing


def test_a_held_session_refuses_another_begin(desk):
    sessions, bridge, _clock = desk()
    assert begin(sessions)["ok"] is True
    again = begin(sessions, window=8)
    assert again["reason"] == "session_exists"
    assert again["session"] == {"window": {"window_id": 7, "pid": 11, "title_at_begin": "Editor"}, "app": "Editor"}
    assert len(bridge.stores) == 1


def test_on_screen_is_passed_on_and_decides_nothing(desk):
    """A hidden window reads ``on_screen: false`` and is still a valid window to watch."""
    sessions, bridge, _clock = desk(windows=[_row(7, "Editor", 11, on_screen=False)])
    answer = begin(sessions)
    assert answer["ok"] is True and answer["window"]["on_screen"] is False
    feed(bridge)
    assert sessions.frame({})[0]["image"] is True


@pytest.mark.parametrize("kwargs,reason", [
    ({"platform": "linux"}, "unsupported_platform"),
    ({"version": 13}, "unsupported_os"),
    ({"version": None}, "unsupported_os"),
])
def test_other_systems_are_a_typed_refusal_before_anything_native(desk, kwargs, reason):
    sessions, bridge, _clock = desk(**kwargs)
    assert begin(sessions)["reason"] == reason and bridge.calls == []
    assert sessions.frame({})[0]["reason"] == "no_session" and sessions.release()[0]["released"] is False


def test_an_unresolved_application_and_a_missing_grant_ask_nothing_of_screencapturekit(desk, stream):
    refusal = {"ok": False, "stage": "resolve", "reason": "ambiguous", "error": "2 applications"}
    sessions, bridge, _clock = desk(native=FakeNative(stream.capture.load_native().window_want, refusal=refusal))
    assert begin(sessions) == refusal and bridge.calls == []
    sessions, bridge, _clock = desk(grant="denied")
    answer = begin(sessions)
    assert answer["reason"] == "screen_recording_denied" and bridge.calls == ["preflight"]
    assert answer["refusal"] == {"capability_unavailable": True, "permission": "screen_recording", "repair": "owner",
                                 "where": "System Settings > Privacy & Security > Screen & System Audio Recording"}
    assert "refusal" not in refusal, "a refusal for another reason carries no grant field"


def test_an_unready_bridge_refuses_and_a_later_begin_asks_again(desk):
    sessions, bridge, _clock = desk(problem="SCStream startCapture...: completion handler signature not registered")
    answer = begin(sessions)
    assert answer["reason"] == "capture_unavailable" and "not registered" in answer["error"] and "content" not in bridge.calls
    bridge.problem = ""
    assert begin(sessions)["ok"] is True and bridge.calls.count("load") == 2


def test_a_missing_pyobjc_is_named(stream):
    def missing():
        raise ImportError("No module named 'objc'")
    sessions = stream.WindowSessions(native=FakeNative(stream.capture.load_native().window_want), bridge_factory=missing,
                                     mac_version=lambda: 26, platform="darwin")
    answer = begin(sessions)
    assert answer["reason"] == "capture_unavailable" and "objc" in answer["error"]


def test_selection_refusals_and_an_empty_window_start_nothing(desk):
    sessions, bridge, _clock = desk(windows=[_row(7, "Same", 11), _row(8, "Same", 11)])
    answer = begin(sessions, window="Same")
    assert answer["reason"] == "window_ambiguous" and answer["windows_total"] == 2 and bridge.stores == []
    assert begin(sessions, window=9)["reason"] == "window_not_found"
    sessions, bridge, _clock = desk(size=(0.0, 10.0, 2.0))
    assert begin(sessions)["reason"] == "window_empty" and bridge.stores == []
    sessions, bridge, _clock = desk(content=False)
    answer = begin(sessions)
    assert (answer["reason"], answer["stage"]) == ("capture_timeout", "shareable_content")
    sessions, bridge, _clock = desk(add="output refused")
    answer = begin(sessions)
    assert (answer["reason"], answer["stage"], answer["error"]) == ("capture_failed", "add_output", "output refused")
    assert ("start", "stream-1") not in bridge.calls and begin(sessions)["reason"] == "capture_failed"  # no slot taken


def test_a_bridge_exception_is_not_a_native_start_completion(desk):
    sessions, bridge, _clock = desk()
    bridge.start = lambda stream_object, handler: (_ for _ in ()).throw(TypeError("block signature"))
    answer = begin(sessions)
    assert (answer["reason"], answer["stage"], answer["error"]) == ("capture_failed", "start", "block signature")
    assert answer["cleanup"] == {"confirmed": True, "detail": "its stop answered"} and sessions._session is None
    assert _stops(bridge) == [("stop", "stream-1")]
    bridge.stop_answer = "silent"
    assert begin(sessions)["cleanup"]["confirmed"] is False and sessions._session is not None
    assert begin(sessions)["reason"] == "session_exists"
    bridge.stop_handlers[-1](None)
    assert sessions.release()[0]["released"] is True
    bridge.load = lambda: (_ for _ in ()).throw(RuntimeError("no such bundle"))
    sessions._loaded = False
    assert begin(sessions)["error"] == "RuntimeError: no such bundle"


def test_a_start_call_that_raises_after_answering_keeps_the_delivered_answer(desk):
    """The framework may deliver the start callback and then fail the call itself: the delivered answer stands,
    and release must still ask the running stream to stop instead of trusting the raise."""
    sessions, bridge, _clock = desk()

    def start_then_raise(stream_object, handler):
        handler(None)  # the start callback is delivered…
        raise TypeError("block teardown")  # …and the call itself then fails

    bridge.start = start_then_raise
    answer = begin(sessions)
    assert answer["ok"] is True and answer["state"] == "live"
    facts, _png = sessions.release()
    assert facts["cleanup"] == {"confirmed": True, "detail": "its stop answered"}
    assert ("stop", "stream-1") in bridge.calls  # the delivered start was not turned into a failed one


def test_a_start_error_leaves_no_session_and_asks_for_no_stop(desk):
    sessions, bridge, _clock = desk(start=_Error(-3802, "failed to start"))
    answer = begin(sessions)
    assert (answer["reason"], answer["stage"], answer["error_code"]) == ("capture_failed", "start", -3802)
    assert answer["cleanup"] == {"confirmed": True, "detail": "the stream never started"}
    assert not any(call[0] == "stop" for call in bridge.calls if isinstance(call, tuple))
    assert bridge.stores[0].accept("complete", buffer="late", width=1, height=1) is False
    bridge.start_answer = "ok"
    assert begin(sessions)["ok"] is True and len(bridge.stores) == 2  # a new store; the old one stays ended
    sessions, bridge, _clock = desk(start=_Error(-3801, "declined"))
    declined = begin(sessions)
    assert declined["reason"] == "screen_recording_denied" and declined["refusal"]["permission"] == "screen_recording"


def test_a_silent_start_is_stopped_and_only_a_confirmed_stop_frees_the_slot(desk):
    sessions, bridge, _clock = desk(start="silent")
    answer = begin(sessions)
    assert (answer["reason"], answer["stage"]) == ("start_timeout", "start")
    assert answer["cleanup"] == {"confirmed": True, "detail": "its stop answered"} and ("stop", "stream-1") in bridge.calls
    assert bridge.stores[0].snapshot()[1]["stop"]["reason"] == "start_timeout"
    bridge.start_handlers[0](None)  # the start answers after its wait ended: nothing comes back to life
    assert bridge.stores[0].accept("complete", buffer="late", width=1, height=1) is False
    bridge.start_answer = "ok"
    assert begin(sessions)["ok"] is True


def test_an_unconfirmed_end_keeps_the_session_until_release_confirms(desk):
    """Neither the start nor the stop answered: the stream may yet run, so nothing else may begin."""
    sessions, bridge, _clock = desk(start="silent", stop="silent")
    answer = begin(sessions)
    assert answer["reason"] == "start_timeout" and answer["cleanup"]["confirmed"] is False
    assert begin(sessions)["reason"] == "session_exists" and len(bridge.stores) == 1
    facts, png = sessions.frame({})
    assert png is None and facts["reason"] == "stream_stopped" and facts["stop"]["reason"] == "start_timeout"
    facts, _png = sessions.release()
    assert facts["ok"] is False and facts["reason"] == "stop_unconfirmed" and facts["released"] is False
    bridge.stop_handlers[-1](None)  # an earlier stop answers late
    facts, _png = sessions.release()
    assert facts["ok"] is True and facts["cleanup"] == {"confirmed": True, "detail": "an earlier stop answered"}
    bridge.start_answer = bridge.stop_answer = "ok"
    assert begin(sessions)["ok"] is True


def test_not_running_confirms_only_once_the_start_has_answered(desk, stream):
    """Before the start answers, "not running" may mean it is still starting; afterwards it means it is over."""
    sessions, bridge, _clock = desk(start="silent", stop=_Error(stream._NOT_STOPPABLE, "not running"))
    assert begin(sessions)["cleanup"] == {"confirmed": False, "detail": "not running"}
    bridge.start_handlers[0](_Error(-3802, "failed to start"))  # the start reports its failure late
    facts, _png = sessions.release()
    assert facts["ok"] is True and facts["cleanup"]["detail"] == "the stream never started"
    sessions, bridge, _clock = desk(start="silent", stop=_Error(stream._NOT_STOPPABLE, "not running"))
    assert begin(sessions)["cleanup"]["confirmed"] is False
    bridge.start_handlers[0](None)  # the start succeeds late: that stream now runs, whatever the earlier stop said
    bridge.stop_answer = "ok"
    facts, _png = sessions.release()
    assert facts["cleanup"] == {"confirmed": True, "detail": "its stop answered"}
    assert [call for call in bridge.calls if call == ("stop", "stream-1")] == [("stop", "stream-1")] * 2
    sessions, bridge, _clock = desk(stop=_Error(stream._NOT_STOPPABLE, "not running"))
    assert begin(sessions)["ok"] is True
    assert sessions.release()[0]["cleanup"] == {"confirmed": True, "detail": "its stop answered"}
    sessions, bridge, _clock = desk(stop=_Error(-3811, "internal error"))
    assert begin(sessions)["ok"] is True
    assert sessions.release()[0]["cleanup"] == {"confirmed": False, "detail": "internal error"}


def test_a_not_running_error_from_another_domain_confirms_nothing(desk, stream):
    """The bare code -3808 is ScreenCaptureKit's vocabulary only; the same number from another domain proves
    nothing: one error callback must never turn an unconfirmed end into a confirmed one."""
    sessions, bridge, _clock = desk(stop=_Error(stream._NOT_STOPPABLE, "not running", domain="com.example.Other"))
    begin(sessions)
    facts, _png = sessions.release()
    assert facts["cleanup"] == {"confirmed": False, "detail": "not running"} and sessions._session is not None
    facts, _png = sessions.frame({})
    assert facts["reason"] == "stream_stopped" and facts["cleanup"]["confirmed"] is False


# --- frame -------------------------------------------------------------------------------------------------------


def test_a_frame_is_the_held_image_with_its_own_seq_and_age(desk):
    sessions, bridge, clock = desk()
    assert sessions.frame({})[0] == {"ok": False, "reason": "no_session", "error": "no window session is held; begin one first"}
    begin(sessions)
    facts, png = sessions.frame({})
    assert png is None and facts["reason"] == "no_frame_yet" and facts["image"] is False and facts["seq"] == 0
    feed(bridge)
    clock.now += 1.5
    feed(bridge, "idle")
    clock.now += 0.5
    facts, png = sessions.frame({})
    assert facts["ok"] is True and facts["image"] is True and png == png_bytes(30, 20)
    assert (facts["seq"], facts["frame_status"], facts["frame_age_ms"], facts["frame_time_basis"]) == (1, "complete", 2000.0, "display")
    assert facts["stream"]["last_report"] == "idle" and facts["stream"]["last_report_age_ms"] == 500.0
    assert facts["stream"]["last_report_time_basis"] == "received"  # this report carried no display time
    assert (facts["image_width"], facts["image_height"]) == (30, 20) and facts["state"] == "live"
    assert facts["window"] == {"window_id": 7, "pid": 11, "title_at_begin": "Editor"}
    assert facts["mapped_input_available"] is False and facts["frame_at_utc"].endswith("+00:00")
    for key in ("observation_id", "coord_transform", "auto_attach_image", "display"):
        assert key not in facts, key
    assert sessions.frame({})[1] == png  # without after_seq the same image is the answer again


def test_after_seq_sends_no_image_again_and_a_wait_returns_the_next_one(desk):
    sessions, bridge, clock = desk()
    begin(sessions)
    feed(bridge)
    facts, png = sessions.frame({"after_seq": 1, "wait_ms": 200})
    assert png is None and facts["ok"] is True and facts["image"] is False and facts["reason"] == "no_newer_frame"
    assert facts["seq"] == 1 and facts["after_seq"] == 1 and 195 <= facts["waited_ms"] <= 260
    bridge.on_pump = lambda: feed(bridge, size=(40, 20))  # the stream delivers while the call waits
    facts, png = sessions.frame({"after_seq": 1, "wait_ms": 5000})
    assert png == png_bytes(40, 20) and facts["seq"] == 2 and facts["waited_ms"] < 200
    assert sessions.frame({"after_seq": 1})[0]["image"] is True and sessions.frame({"after_seq": 2})[1] is None


def test_a_first_frame_can_be_waited_for(desk):
    sessions, bridge, _clock = desk()
    begin(sessions)
    bridge.on_pump = lambda: feed(bridge, "started")
    facts, png = sessions.frame({"wait_ms": 3000})
    assert png is not None and (facts["seq"], facts["frame_status"]) == (1, "started")


@pytest.mark.parametrize("status", ["blank", "suspended", "stopped", "inactive"])
def test_a_withheld_frame_is_never_served_as_the_window(desk, status):
    sessions, bridge, _clock = desk()
    begin(sessions)
    feed(bridge)
    feed(bridge, status)
    facts, png = sessions.frame({})
    assert png is None and facts["ok"] is False and facts["reason"] == "frame_withheld"
    assert facts["withheld"]["status"] == status and facts["state"] == "live" and bridge.written == []
    feed(bridge)  # the same stream delivers again: no restart was needed or made
    assert sessions.frame({})[1] is not None and len(bridge.stores) == 1


def test_a_stream_the_system_stopped_serves_nothing_and_is_not_restarted(desk):
    sessions, bridge, _clock = desk()
    begin(sessions)
    feed(bridge)
    bridge.stores[0].end("user_stopped", by="stream", code=-3817, domain="D")
    facts, png = sessions.frame({"wait_ms": 5000})
    assert png is None and facts["reason"] == "stream_stopped" and facts["waited_ms"] == 0.0
    assert facts["stop"]["reason"] == "user_stopped" and facts["stop"]["by"] == "stream" and "not restarted" in facts["error"]
    assert facts["cleanup"] == {"confirmed": True, "detail": "the stream reported its own end"}
    assert feed(bridge) is False and sessions.frame({})[1] is None  # a late sample changes nothing
    assert begin(sessions)["reason"] == "session_exists" and bridge.calls.count(("start", "stream-1")) == 1
    facts, _png = sessions.release()
    assert facts["ok"] is True and facts["cleanup"]["detail"] == "the stream reported its own end"
    assert facts["stop"]["reason"] == "user_stopped"  # the stream's own end keeps its reason
    assert not any(call[0] == "stop" for call in bridge.calls if isinstance(call, tuple))


def test_a_lookup_without_an_answer_is_not_an_absence(desk):
    """A lookup that stopped answering serves no image for that call — an explicit unknown — and ends nothing."""
    sessions, bridge, _clock = desk()
    begin(sessions)
    feed(bridge)
    bridge.listed = None
    facts, png = sessions.frame({})
    assert png is None and facts["ok"] is False and facts["reason"] == "window_lookup_unavailable"
    assert facts["state"] == "live" and "stop" not in facts and "cleanup" not in facts and _stops(bridge) == []
    bridge.listed = True
    facts, png = sessions.frame({})  # an answered lookup serves again: nothing was ended or restarted
    assert png is not None and facts["image"] is True and facts["state"] == "live"


def test_a_window_the_window_server_no_longer_lists_is_gone_for_good(desk):
    """One authoritative absence ends delivery AND stops the native stream in the same call: no later lookup,
    recycled id or new sample brings the image back, and nothing is asked a second time per call."""
    sessions, bridge, _clock = desk()
    begin(sessions)
    feed(bridge)
    assert sessions.frame({})[1] is not None
    bridge.listed = False
    facts, png = sessions.frame({})
    assert png is None and facts["ok"] is False and facts["reason"] == "window_gone" and facts["state"] == "stopped"
    assert facts["stop"]["reason"] == "window_gone" and facts["stop"]["by"] == "session"
    assert facts["cleanup"] == {"confirmed": True, "detail": "its stop answered"}  # the native stop answered
    assert _stops(bridge) == [("stop", "stream-1")]
    asked = bridge.calls.count(("listed", 7, 11))
    for later in (None, True):  # the lookup stops answering; another window appears under the same id
        bridge.listed = later
        assert feed(bridge) is False  # and the stream's samples are no longer taken
        facts, png = sessions.frame({"wait_ms": 5000})
        assert png is None and facts["reason"] == "window_gone" and facts["waited_ms"] == 0.0
        assert facts["cleanup"] == {"confirmed": True, "detail": "an earlier stop answered"}  # witness, no new ask
    assert bridge.calls.count(("listed", 7, 11)) == asked and bridge.written == [bridge.written[0]]  # nothing is asked or written again
    assert _stops(bridge) == [("stop", "stream-1")]  # asked exactly once, not per call
    assert begin(sessions)["reason"] == "session_exists"  # the slot stays held: this refusal is checkable
    facts, _png = sessions.release()
    assert facts["ok"] is True and facts["stop"]["reason"] == "window_gone"
    assert facts["cleanup"] == {"confirmed": True, "detail": "an earlier stop answered"}  # no second stop ask
    assert _stops(bridge) == [("stop", "stream-1")] and sessions._session is None
    assert bridge.calls.count(("remove_output", "stream-1")) == 1
    bridge.listed = True
    assert begin(sessions)["ok"] is True and len(bridge.stores) == 2  # only a new begin watches a window again
    feed(bridge)
    assert sessions.frame({})[1] is not None


@pytest.mark.parametrize("listed, reason", [(False, "window_not_found"), (None, "window_lookup_unavailable")])
def test_begin_needs_a_window_that_can_be_monitored(desk, listed, reason):
    sessions, bridge, _clock = desk(listed=listed)
    assert begin(sessions)["reason"] == reason
    sessions.idle_check()
    assert bridge.calls.count(("listed", 7, 11)) == 1
    assert sessions._session is None and not bridge.stores
    assert sessions.frame({})[0]["reason"] == "no_session"
    assert sessions.release()[0]["released"] is False
    bridge.listed = True
    assert begin(sessions)["ok"] and len(bridge.stores) == 1
    feed(bridge)
    assert sessions.frame({})[1] is not None


def test_a_long_wait_notices_a_destroyed_window_before_its_deadline(desk):
    """A prolonged pending request re-looks at the target between bounded wait slices: proven destruction ends
    the wait and the stream early instead of going on blindly until the deadline."""
    sessions, bridge, clock = desk()
    begin(sessions)
    feed(bridge)
    bridge.on_pump = lambda: setattr(bridge, "listed", False)  # the window dies just after the wait begins
    before = clock.now
    facts, png = sessions.frame({"after_seq": 1, "wait_ms": 5000})
    assert png is None and facts["reason"] == "window_gone" and facts["state"] == "stopped"
    assert facts["cleanup"] == {"confirmed": True, "detail": "its stop answered"}
    assert clock.now - before < 2.0  # ended on a mid-wait look, not on the 5 s deadline
    assert _stops(bridge) == [("stop", "stream-1")]


def test_an_end_while_the_image_is_written_withholds_it(desk):
    sessions, bridge, _clock = desk()
    begin(sessions)
    feed(bridge)
    write = bridge.write_png

    def stopped_meanwhile(buffer, path):
        size = write(buffer, path)
        bridge.stores[0].end("system_stopped", by="stream", code=-3821)
        return size
    bridge.write_png = stopped_meanwhile
    facts, png = sessions.frame({})
    assert png is None and facts["reason"] == "stream_stopped" and facts["stop"]["reason"] == "system_stopped"
    assert facts["cleanup"] == {"confirmed": True, "detail": "the stream reported its own end"}


@pytest.mark.parametrize(("reason", "code"), [("screen_recording_denied", -3801), ("system_stopped", -3821)])
def test_a_stream_the_system_ends_for_the_grant_says_which_grant(desk, reason, code):
    """A grant withdrawn while the stream runs ends it with -3801: the frame's refusal carries the same structured
    field a denied start does, so the owner is asked for Screen Recording; another end carries none."""
    sessions, bridge, _clock = desk()
    begin(sessions)
    bridge.stores[0].end(reason, by="stream", code=code)
    facts, png = sessions.frame({})
    assert png is None and facts["reason"] == "stream_stopped" and facts["stop"]["reason"] == reason, facts
    if code == -3801:
        assert facts["refusal"] == {"capability_unavailable": True, "permission": "screen_recording", "repair": "owner",
                                    "where": "System Settings > Privacy & Security > Screen & System Audio Recording"}
    else:
        assert "refusal" not in facts, facts


def test_a_frame_dropped_while_it_is_written_is_not_rescued_by_a_later_image(desk):
    """A held image dropped while its PNG is written stays dropped for that call: a valid report arriving
    before the write returns cannot revalidate the frame that was in flight."""
    sessions, bridge, _clock = desk()
    begin(sessions)
    feed(bridge)
    write = bridge.write_png

    def dropped_meanwhile(buffer, path):
        size = write(buffer, path)
        feed(bridge, "blank")  # the held image is dropped while it is being written…
        feed(bridge)           # …and a later valid report arrives before the write returns
        return size
    bridge.write_png = dropped_meanwhile
    facts, png = sessions.frame({})
    assert png is None and facts["ok"] is False and facts["reason"] == "frame_withheld"
    assert "dropped this frame while it was being written" in facts["error"] and facts["state"] == "live"
    bridge.write_png = write
    facts, png = sessions.frame({})  # the newer image itself is served on the next call
    assert png is not None and facts["image"] is True and facts["seq"] == 2


def test_a_failed_encoding_keeps_the_session_and_leaves_no_file(desk):
    sessions, bridge, _clock = desk()
    begin(sessions)
    feed(bridge)
    write = bridge.write_png
    bridge.write_png = lambda buffer, path: (pathlib.Path(path).write_bytes(b"not a png"), bridge.written.append(path), (30, 20))[2]
    facts, png = sessions.frame({})
    assert png is None and facts["reason"] == "encode_failed" and facts["state"] == "live"
    bridge.write_png = lambda buffer, path: (_ for _ in ()).throw(RuntimeError("Quartz did not encode"))
    assert "Quartz did not encode" in sessions.frame({})[0]["error"]
    bridge.write_png = write
    assert sessions.frame({})[1] == png_bytes(30, 20)


# --- release and the server's end --------------------------------------------------------------------------------


def test_release_ends_the_stream_before_asking_it_to_stop_and_is_idempotent(desk):
    sessions, bridge, _clock = desk()
    assert sessions.release() == ({"ok": True, "released": False, "state": "none"}, None)
    begin(sessions)
    feed(bridge)
    seen = []
    stop = bridge.stop

    def stop_and_look(stream_object, handler):
        seen.append(bridge.stores[0].snapshot()[1]["state"])  # closed before the framework is asked
        stop(stream_object, handler)
    bridge.stop = stop_and_look
    facts, png = sessions.release()
    assert png is None and facts["ok"] is True and facts["released"] is True and seen == ["stopped"]
    assert facts["stop"]["reason"] == "released" and facts["stop"]["by"] == "session"
    assert facts["cleanup"] == {"confirmed": True, "detail": "its stop answered"} and ("remove_output", "stream-1") in bridge.calls
    assert feed(bridge) is False  # a callback after release
    assert sessions.release()[0] == {"ok": True, "released": False, "state": "none"}
    assert sessions.frame({})[0]["reason"] == "no_session" and sessions.release({"x": 1})[0]["reason"] == "bad_option"
    assert begin(sessions)["ok"] is True and len(bridge.stores) == 2
    assert sessions.frame({})[0]["reason"] == "no_frame_yet"  # the new session starts from nothing


def test_a_stop_that_does_not_answer_keeps_the_native_objects(desk):
    sessions, bridge, _clock = desk(stop="silent")
    begin(sessions)
    held = sessions._session
    facts, _png = sessions.release()
    assert facts["reason"] == "stop_unconfirmed" and facts["cleanup"]["detail"] == "no answer to the stop request"
    assert sessions._session is held and held["native"][0] == "sink-1"  # still referenced: a callback may yet come
    assert not any(call[0] == "remove_output" for call in bridge.calls if isinstance(call, tuple))
    bridge.stop_answer = "ok"
    assert sessions.release()[0]["ok"] is True and sessions._session is None


def _stops(bridge):
    return [call for call in bridge.calls if isinstance(call, tuple) and call[0] == "stop"]


def _stream_ends(stream, bridge, code, text):
    """The sink's ``stream:didStopWithError:`` for the newest store, through the real ``_Feed``."""
    return lambda: stream._Feed(bridge, bridge.stores[-1]).ended(_Error(code, text))


def test_a_gone_window_with_an_unanswered_stop_stays_unconfirmed_until_release(desk):
    """Proven destruction asks the stop once; an unanswered stop is no freed resource and is retried by no
    frame or idle call — release is the door that asks again."""
    sessions, bridge, _clock = desk(stop="silent")
    begin(sessions)
    feed(bridge)
    bridge.listed = False
    facts, png = sessions.frame({})
    assert png is None and facts["reason"] == "window_gone"
    assert facts["cleanup"] == {"confirmed": False, "detail": "no answer to the stop request"}
    assert sessions._session is not None
    assert not any(call[0] == "remove_output" for call in bridge.calls if isinstance(call, tuple))
    facts, _png = sessions.frame({})
    assert facts["cleanup"]["confirmed"] is False and _stops(bridge) == [("stop", "stream-1")]  # no frantic retry
    sessions.idle_check()  # a stopped store ends the idle look too: no second ask from there
    assert _stops(bridge) == [("stop", "stream-1")]
    facts, _png = sessions.release()
    assert facts["ok"] is False and facts["reason"] == "stop_unconfirmed"
    assert _stops(bridge) == [("stop", "stream-1")] * 2  # release asks again
    bridge.stop_handlers[-1](None)  # the second stop answers late
    facts, _png = sessions.release()
    assert facts["ok"] is True and facts["cleanup"] == {"confirmed": True, "detail": "an earlier stop answered"}
    assert facts["stop"]["reason"] == "window_gone" and sessions._session is None


def test_an_idle_moment_stops_a_stream_whose_window_was_destroyed(desk):
    """The server's jobless moments end a destroyed window's capture even when no further call ever comes."""
    sessions, bridge, _clock = desk()
    sessions.idle_check()  # no session held: nothing runs and nothing native is touched
    assert bridge.calls == []
    begin(sessions)
    feed(bridge)
    sessions.idle_check()  # the window is still listed: the idle look does nothing
    assert _stops(bridge) == [] and sessions.frame({})[1] is not None
    bridge.listed = None
    sessions.idle_check()  # a lookup without an answer is unknown, not absence
    assert _stops(bridge) == [] and sessions._session["store"].stop is None
    bridge.listed = False
    sessions.idle_check()  # proven destruction with no call in flight
    facts, png = sessions.frame({})
    assert png is None and facts["reason"] == "window_gone"
    assert facts["cleanup"] == {"confirmed": True, "detail": "an earlier stop answered"}
    assert _stops(bridge) == [("stop", "stream-1")]
    asked = bridge.calls.count(("listed", 7, 11))
    sessions.idle_check()  # an ended session is left alone: no further look, no second stop
    assert _stops(bridge) == [("stop", "stream-1")] and bridge.calls.count(("listed", 7, 11)) == asked
    facts, _png = sessions.release()
    assert facts["ok"] is True and facts["stop"]["reason"] == "window_gone"
    assert facts["cleanup"] == {"confirmed": True, "detail": "an earlier stop answered"} and sessions._session is None


def test_a_stream_end_queued_for_the_main_thread_keeps_its_reason_over_release(desk, stream):
    """Taken in before the session's reason: the owner's stop stays ``user_stopped`` and is not asked again."""
    sessions, bridge, _clock = desk(stop="silent")
    begin(sessions)
    feed(bridge)
    bridge.on_pump = _stream_ends(stream, bridge, -3817, "stopped by the user")
    facts, _png = sessions.release()
    assert facts["ok"] is True and facts["cleanup"] == {"confirmed": True, "detail": "the stream reported its own end"}
    assert (facts["stop"]["reason"], facts["stop"]["by"], facts["stop"]["code"]) == ("user_stopped", "stream", -3817)
    assert "stream_end" not in facts and _stops(bridge) == [] and sessions._session is None


def test_a_stream_end_while_its_stop_is_awaited_confirms_release(desk, stream):
    """The stop never answers, or answers with an error that proves nothing: the stream's own end still does."""
    sessions, bridge, _clock = desk(stop="silent")
    begin(sessions)
    feed(bridge)
    stop = bridge.stop

    def stop_then_end(stream_object, handler):
        stop(stream_object, handler)
        bridge.on_pump = _stream_ends(stream, bridge, -3821, "stopped by the system")  # inside the stop's wait
    bridge.stop = stop_then_end
    facts, _png = sessions.release()
    assert facts["ok"] is True and facts["released"] is True
    assert facts["cleanup"] == {"confirmed": True, "detail": "the stream reported its own end"}
    assert facts["stop"]["reason"] == "released" and facts["stream_end"]["reason"] == "system_stopped"
    assert facts["stream_end"]["code"] == -3821 and sessions._session is None
    assert feed(bridge) is False and ("remove_output", "stream-1") in bridge.calls  # nothing reopened
    bridge.stop_handlers[0](None)  # the stop answers after all: nothing changes
    assert begin(sessions)["ok"] is True and len(bridge.stores) == 2

    sessions, bridge, _clock = desk(stop=_Error(-3811, "internal error"))
    begin(sessions)
    stop = bridge.stop
    bridge.stop = lambda stream_object, handler: (_stream_ends(stream, bridge, -3817, "stopped by the user")(),
                                                  stop(stream_object, handler))
    facts, _png = sessions.release()
    assert facts["cleanup"] == {"confirmed": True, "detail": "the stream reported its own end"}
    assert facts["stream_end"]["reason"] == "user_stopped" and facts["stop"]["reason"] == "released"


def test_a_stream_end_after_an_unanswered_stop_confirms_the_next_release_without_asking_again(desk, stream):
    sessions, bridge, _clock = desk(stop="silent")
    begin(sessions)
    facts, _png = sessions.release()
    assert facts["reason"] == "stop_unconfirmed" and "stream_end" not in facts
    _stream_ends(stream, bridge, -3817, "stopped by the user")()
    facts, _png = sessions.release()
    assert facts["ok"] is True and facts["cleanup"] == {"confirmed": True, "detail": "the stream reported its own end"}
    assert facts["stream_end"]["reason"] == "user_stopped" and len(_stops(bridge)) == 1 and sessions._session is None


def test_a_stream_that_ends_itself_after_window_gone_confirms_release_without_a_further_stop(desk, stream):
    """The gone window's stream is asked to stop once, at the proof; an error answer that proves nothing keeps
    the stop unconfirmed until the stream's own end — a native event — confirms the release without a new ask."""
    sessions, bridge, _clock = desk(stop=_Error(-3811, "internal error"))
    begin(sessions)
    feed(bridge)
    bridge.listed = False
    facts, _png = sessions.frame({})
    assert facts["reason"] == "window_gone"
    assert facts["cleanup"] == {"confirmed": False, "detail": "internal error"}  # one error callback confirms nothing
    _stream_ends(stream, bridge, -3815, "no capture source")()
    facts, png = sessions.frame({})
    assert png is None and facts["reason"] == "window_gone" and facts["stream_end"]["reason"] == "no_capture_source"
    assert facts["cleanup"] == {"confirmed": True, "detail": "the stream reported its own end"}
    facts, _png = sessions.release()
    assert facts["ok"] is True and facts["cleanup"] == {"confirmed": True, "detail": "the stream reported its own end"}
    assert facts["stop"]["reason"] == "window_gone" and facts["stream_end"]["code"] == -3815
    assert _stops(bridge) == [("stop", "stream-1")] and sessions._session is None  # the proof asked once


def test_a_stop_call_that_raises_is_unconfirmed_and_shutdown_still_ends(desk, stream):
    """Nothing the stop call raises leaves release or shutdown; the session stays until a stop is confirmed."""
    sessions, bridge, clock = desk()
    begin(sessions)
    feed(bridge)
    assert sessions.frame({})[1] is not None
    scratch = sessions._scratch
    bridge.stop = lambda stream_object, handler: (_ for _ in ()).throw(TypeError("block signature"))
    before = clock.now
    facts, png = sessions.release()
    assert png is None and facts["ok"] is False and facts["reason"] == "stop_unconfirmed" and facts["released"] is False
    assert facts["cleanup"] == {"confirmed": False, "detail": "the stop request raised TypeError: block signature"}
    assert clock.now == before  # the raise ended the wait at once
    assert sessions._session is not None and not any(call[0] == "remove_output" for call in bridge.calls
                                                     if isinstance(call, tuple))
    assert begin(sessions)["reason"] == "session_exists"

    class NotRunning(Exception):  # a raise is no answer, whatever it carries
        domain, code = (lambda self: "D"), (lambda self: stream._NOT_STOPPABLE)
        localizedDescription = lambda self: "not running"  # noqa: E731
    bridge.stop = lambda stream_object, handler: (_ for _ in ()).throw(NotRunning("not running"))
    assert sessions.release()[0]["cleanup"]["confirmed"] is False
    sessions.shutdown()  # the server's end
    assert not os.path.exists(scratch) and sessions._session is not None
    del bridge.stop
    assert sessions.release()[0]["cleanup"] == {"confirmed": True, "detail": "its stop answered"}


def test_shutdown_stops_within_its_own_bound_and_removes_the_scratch_folder(desk, stream):
    sessions, bridge, clock = desk(stop="silent")
    begin(sessions)
    feed(bridge)
    assert sessions.frame({})[1] is not None
    scratch = sessions._scratch
    assert os.path.isdir(scratch) and os.listdir(scratch) == []
    before = clock.now
    sessions.shutdown()
    assert not os.path.exists(scratch) and round(clock.now - before, 3) == stream.SHUTDOWN_STOP_MS / 1000
    stream.WindowSessions(platform="linux").shutdown()  # nothing was ever started: nothing to do

"""One exact macOS window through ScreenCaptureKit: the helper's contract and the screenshot handler's.

No test here enumerates windows, asks ScreenCaptureKit for content, captures, prompts for a permission or touches the
desktop. The helper is driven through a fake bridge; the handler through the registered ``screenshot`` tool with a fake
process boundary. ``test_real_pyobjc_bridge_*`` loads the real framework and registers the real handler signatures,
and stops short of any ScreenCaptureKit request (that live proof needs the owner's consent and stays a named gap).
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import threading
import zlib

import pytest

from skills.unix_computer_use import plugin

SKILL = pathlib.Path(__file__).resolve().parents[1] / "skills" / "unix_computer_use"
HELPER = SKILL / "macos_capture.py"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helper():
    return _load("unix_window_capture_helper", HELPER)


@pytest.fixture(scope="module")
def native(helper):
    return helper.load_native()


def png_bytes(width: int, height: int) -> bytes:
    """A real, whole PNG (stdlib only)."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")
    rows = b"".join(b"\x00" + b"\x40\x80\xc0" * width for _ in range(height))
    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


# --- the helper process boundary -------------------------------------------------------------------------------


def _run_helper(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", "-B", str(HELPER), *args], capture_output=True, text=True, timeout=60,
                          cwd=str(pathlib.Path(os.sep)))


def test_helper_runs_isolated_and_loads_its_native_resolver_by_path(tmp_path):
    """Under -I the helper can load its sibling resolver and refuse a malformed target without any capture."""
    spec = {"app": "Editor", "window": True, "output": str(tmp_path / "w.png")}
    done = _run_helper(json.dumps(spec))
    lines = done.stdout.strip().splitlines()
    assert done.returncode == 1 and len(lines) == 1, (done.stdout, done.stderr)
    answer = json.loads(lines[0])
    assert answer["ok"] is False and answer["reason"] == "bad_option"
    assert "window must be a window_id or an exact title" in answer["error"], answer
    assert not (tmp_path / "w.png").exists()


def test_helper_usage_is_one_refusal_line():
    done = _run_helper()
    assert done.returncode == 1
    assert json.loads(done.stdout.strip()) == {"ok": False, "reason": "bad_option", "error": "usage: macos_capture.py <options json>"}


@pytest.mark.parametrize("change,error", [
    ({"app": ""}, "app is required"),
    ({"app": 7}, "app is required"),
    ({"window": None}, "window is required"),
    ({"window": "   "}, "window is required"),
    ({"window": [1]}, "window must be a window_id or an exact title"),
    ({"output": "w.png"}, "absolute path"),
    ({"output": "/tmp/w.jpg"}, "absolute path"),
    ({"timeout_ms": True}, "timeout_ms must be an integer"),
    ({"timeout_ms": "5"}, "timeout_ms must be an integer"),
    ({"extra": 1}, "unknown option"),
])
def test_options_refuse_before_anything_is_resolved(helper, tmp_path, change, error):
    raw = {"app": "Editor", "window": 7, "output": str(tmp_path / "w.png"), **change}
    spec, problem = helper.parse_spec(json.dumps(raw))
    assert spec == {} and error in problem


def test_options_refuse_an_existing_or_folderless_output_and_clamp_the_budget(helper, tmp_path):
    taken = tmp_path / "taken.png"
    taken.write_bytes(b"x")
    for output in (str(taken), str(tmp_path / "missing" / "w.png")):
        assert "new file in an existing folder" in helper.parse_spec({"app": "A", "window": 1, "output": output})[1]
    spec, problem = helper.parse_spec({"app": "A", "window": "0042", "output": str(tmp_path / "w.png"), "timeout_ms": 10})
    assert problem == "" and spec["want"] == {"title": "0042"} and spec["timeout_ms"] == helper.TIMEOUT_MS["min"]
    spec, _ = helper.parse_spec({"app": "A", "window": " Notes ", "output": str(tmp_path / "w.png"), "timeout_ms": 10 ** 9})
    assert spec["want"] == {"title": " Notes "} and spec["timeout_ms"] == helper.TIMEOUT_MS["max"]


def _row(window_id, title, pid):
    return object(), {"window_id": window_id, "title": title, "pid": pid, "bounds": [0, 0, 10, 10], "on_screen": True,
                      "layer": 0}


def test_selection_is_one_window_of_the_resolved_process(helper):
    rows = [_row(1, "Same", 11), _row(2, "Same", 11), _row(3, "Other", 12), _row(4, None, 11), _row(5, "Solo", 11)]
    window, facts, refusal = helper.select_window(rows, 11, {"window_id": 5})
    assert refusal is None and window is rows[4][0] and facts["title"] == "Solo"
    assert helper.select_window(rows, 11, {"title": "Solo"})[1]["window_id"] == 5
    ambiguous = helper.select_window(rows, 11, {"title": "Same"})[2]
    assert ambiguous["reason"] == "window_ambiguous" and ambiguous["windows_total"] == 4
    foreign = helper.select_window(rows, 11, {"window_id": 3})[2]  # another process's window is never taken
    assert foreign["reason"] == "window_not_found" and {f["pid"] for f in foreign["windows"]} == {11}
    assert helper.select_window(rows, 11, {"title": "Other"})[2]["reason"] == "window_not_found"
    assert helper.select_window(rows, 99, {"window_id": 1})[2]["reason"] == "no_window"


def test_pixel_size_keeps_the_aspect_ratio(helper):
    assert helper.pixel_size(800, 500, 2.0) == (1600, 1000)
    assert helper.pixel_size(10000, 2500, 2.0) == (helper.MAX_SIDE_PX, helper.MAX_SIDE_PX // 4)
    assert helper.pixel_size(0, 500, 2.0) == (0, 0)
    assert helper.pixel_size(float("nan"), 500, 2.0) == (0, 0)


def test_numeric_title_and_window_number_name_different_windows_of_the_same_app(helper, tmp_path):
    rows = [_row(42, "Inbox", 11), _row(7, "42", 11)]
    for selector, expected in ((42, 42), ("42", 7)):
        spec, error = helper.parse_spec({"app": "Editor", "window": selector, "output": str(tmp_path / "w.png")})
        assert not error
        assert helper.select_window(rows, 11, spec["want"])[1]["window_id"] == expected


def test_write_new_never_replaces_and_leaves_no_partial_file(helper, tmp_path):
    target = tmp_path / "w.png"
    helper.write_new(str(target), b"first")
    with pytest.raises(FileExistsError):
        helper.write_new(str(target), b"second")
    assert target.read_bytes() == b"first" and [p.name for p in tmp_path.iterdir()] == ["w.png"]


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_a_handler_is_waited_for_until_the_deadline_and_a_late_answer_is_dropped(helper):
    clock = _Clock()
    held = []
    pumped = []

    def pump(seconds):
        pumped.append(seconds)
        clock.now += seconds

    arrived, value, error = helper.await_handler(held.append, clock.now + 0.2, pump=pump, clock=clock)
    assert (arrived, value, error) == (False, None, None) and 0.19 < sum(pumped) <= 0.2 + 1e-9
    held[0]("late", None)  # after the wait has ended: accepted by the block and dropped without being read


def test_nothing_is_started_once_the_shared_deadline_has_passed(helper):
    clock = _Clock()
    started = []
    result = helper.await_handler(started.append, clock.now, pump=lambda s: pytest.fail("pumped"), clock=clock)
    assert result == (False, None, None) and started == []
    result = helper.await_handler(started.append, clock.now - 1, pump=lambda s: pytest.fail("pumped"), clock=clock)
    assert result == (False, None, None) and started == []


def test_a_synchronous_answer_after_the_deadline_is_dropped(helper):
    """A start that blocks past the deadline and then answers in-line must not deliver a late image."""
    clock = _Clock()

    def slow_start(handler):
        clock.now += 1.0  # the call itself overran the 0.5 s left
        handler("late image", None)

    result = helper.await_handler(slow_start, clock.now + 0.5, pump=lambda s: pytest.fail("pumped"), clock=clock)
    assert result == (False, None, None)


def test_an_answer_inside_a_pump_counts_only_before_the_deadline(helper):
    """A pump that overruns: an answer it carries before the deadline stands; one after it is dropped."""
    for answer_at, expected in ((0.3, (True, "image", None)), (0.6, (False, None, None))):
        clock = _Clock()
        deadline = clock.now + 0.5
        held = []

        def pump(seconds, answer_at=answer_at, held=held, clock=clock):
            clock.now = 100.0 + answer_at
            held[0]("image", None)
            clock.now = 100.0 + 0.9  # the wait itself resumes only after the deadline

        assert helper.await_handler(held.append, deadline, pump=pump, clock=clock) == expected, answer_at


def test_a_handler_answer_from_another_thread_arrives_once(helper):
    def start(handler):
        threading.Thread(target=lambda: (handler("first", None), handler("second", "error"))).start()

    arrived, value, error = helper.await_handler(start, float("inf"), pump=lambda s: None)
    assert (arrived, value, error) == (True, "first", None)


class _Error:
    def __init__(self, domain, code, text):
        self.domain, self.code, self.localizedDescription = (lambda: domain), (lambda: code), (lambda: text)


class FakeBridge:
    """The Bridge's surface, recording what was asked; nothing native."""

    def __init__(self, *, grant="granted", problem="", content=True, image=True, png=None, windows=None, error=None):
        self.grant, self.problem, self.content, self.image, self.error = grant, problem, content, image, error
        self.encoded = png if png is not None else png_bytes(30, 20)
        self.rows = windows if windows is not None else [_row(7, "Editor", 11), _row(8, "Other", 11)]
        self.calls = []

    def screen_recording(self):
        self.calls.append("preflight")
        return self.grant

    def load(self):
        self.calls.append("load")
        return self.problem

    def pump(self, seconds):
        self.calls.append("pump")
        self.clock.now += seconds

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
        return 15.0, 10.0, 2.0

    def configuration(self, width, height):
        self.calls.append(("config", width, height))
        return "config", {"width": width, "height": height, "cursor_included": False, "shadow_included": False}

    def screenshot(self, content_filter, config, handler):
        self.calls.append("screenshot")
        if self.image:
            handler(None if self.error else "image", self.error)

    def image_size(self, image):
        return 30, 20

    def png(self, image):
        return self.encoded


class FakeNative:
    def __init__(self, window_want, target=None, refusal=None):
        self.window_want, self.target, self.refusal, self.resolved = window_want, target, refusal, []

    def resolve_target(self, query):
        self.resolved.append(query)
        return (None, self.refusal) if self.refusal else (self.target or {"app": "Editor", "pid": 11, "app_id": "e"}, None)


def _capture(helper, native, tmp_path, bridge=None, *, window=7, version=26, fake=None):
    fake = fake or FakeNative(native.window_want)
    clock = _Clock()
    if bridge is not None:
        bridge.clock = clock
    spec = json.dumps({"app": "Editor", "window": window, "output": str(tmp_path / "w.png"), "timeout_ms": 1000})
    return helper.capture(spec, native=fake, bridge_factory=lambda: bridge, clock=clock, mac_version=lambda: version), fake


def test_a_capture_is_the_named_window_at_its_own_pixels(helper, native, tmp_path):
    bridge = FakeBridge()
    answer, fake = _capture(helper, native, tmp_path, bridge, window="Editor")
    assert answer["ok"] is True and fake.resolved == ["Editor"]
    assert answer["window"]["window_id"] == 7 and answer["app"]["pid"] == 11
    assert ("config", 30, 20) in bridge.calls and ("filter", 7) in bridge.calls
    assert bridge.calls.index("preflight") < bridge.calls.index("load") < bridge.calls.index("content")
    assert (answer["image_width"], answer["image_height"], answer["point_pixel_scale"]) == (30, 20, 2.0)
    assert answer["cursor_included"] is False and answer["shadow_included"] is False
    assert (tmp_path / "w.png").read_bytes() == bridge.encoded and answer["output"] == str(tmp_path / "w.png")


def test_old_macos_and_unresolved_apps_refuse_before_the_framework(helper, native, tmp_path):
    answer, fake = _capture(helper, native, tmp_path, None, version=13)
    assert answer["reason"] == "unsupported_os" and fake.resolved == []
    refusal = {"ok": False, "stage": "resolve", "reason": "ambiguous", "error": "2 applications"}
    answer, _ = _capture(helper, native, tmp_path, None, fake=FakeNative(native.window_want, refusal=refusal))
    assert answer == refusal


def test_without_the_grant_nothing_in_screencapturekit_is_asked(helper, native, tmp_path):
    bridge = FakeBridge(grant="denied")
    answer, _ = _capture(helper, native, tmp_path, bridge)
    assert answer["reason"] == "screen_recording_denied" and bridge.calls == ["preflight"]
    assert not (tmp_path / "w.png").exists()


def test_an_unregistered_handler_signature_refuses_instead_of_being_skipped(helper, native, tmp_path):
    bridge = FakeBridge(problem="SCScreenshotManager captureImage...: completion handler signature not registered")
    answer, _ = _capture(helper, native, tmp_path, bridge)
    assert answer["reason"] == "capture_unavailable" and "not registered" in answer["error"]
    assert "content" not in bridge.calls


def test_a_missing_pyobjc_is_named(helper, native, tmp_path):
    def missing():
        raise ImportError("No module named 'objc'")
    spec = json.dumps({"app": "Editor", "window": 7, "output": str(tmp_path / "w.png")})
    answer = helper.capture(spec, native=FakeNative(native.window_want), bridge_factory=missing, mac_version=lambda: 26)
    assert answer["reason"] == "capture_unavailable" and "objc" in answer["error"]


def test_silent_content_or_image_is_a_timeout_and_no_file(helper, native, tmp_path):
    bridge = FakeBridge(content=False)
    answer, _ = _capture(helper, native, tmp_path, bridge)
    assert (answer["reason"], answer["stage"]) == ("capture_timeout", "shareable_content") and "screenshot" not in bridge.calls
    bridge = FakeBridge(image=False)
    answer, _ = _capture(helper, native, tmp_path, bridge)
    assert (answer["reason"], answer["stage"]) == ("capture_timeout", "screenshot") and not (tmp_path / "w.png").exists()


def test_framework_errors_are_reported_and_a_declined_grant_is_named(helper, native, tmp_path):
    declined = _Error("com.apple.ScreenCaptureKit.SCStreamErrorDomain", -3801, "The user declined TCCs")
    answer, _ = _capture(helper, native, tmp_path, FakeBridge(error=declined))
    assert answer["reason"] == "screen_recording_denied" and answer["error_code"] == -3801
    answer, _ = _capture(helper, native, tmp_path, FakeBridge(error=_Error("D", 1, "broken")))
    assert (answer["reason"], answer["error"], answer["window"]["window_id"]) == ("capture_failed", "broken", 7)
    answer, _ = _capture(helper, native, tmp_path, FakeBridge(png=b""))
    assert (answer["reason"], answer["stage"]) == ("capture_failed", "encode") and not (tmp_path / "w.png").exists()


def test_a_window_selection_refusal_passes_through(helper, native, tmp_path):
    bridge = FakeBridge(windows=[_row(7, "Same", 11), _row(8, "Same", 11)])
    answer, _ = _capture(helper, native, tmp_path, bridge, window="Same")
    assert answer["reason"] == "window_ambiguous" and "screenshot" not in bridge.calls


def test_handler_registration_is_checked_against_what_pyobjc_holds(helper):
    """PyObjC without the handler's signature (what loadBundle alone gives) refuses; the registered one passes."""
    def method(types):
        block = {"type": b"@?", **({"callable": {"arguments": [{"type": t} for t in types]}} if types else {})}
        meta = {"arguments": ({}, {}, {}, {}, block)}
        return type("Method", (), {"__metadata__": lambda self: meta})()

    def classes(typed):
        return {cls: type(cls, (), {sel.replace(":", "_"): method(typed(meta))})
                for (cls, sel), meta in helper.HANDLERS.items()}

    registered = lambda meta: [meta["arguments"][4]["callable"]["arguments"][i]["type"] for i in range(3)]  # noqa: E731
    assert "not registered" in helper.registered_handler_problem(classes(lambda meta: None))
    assert "not registered" in helper.registered_handler_problem(classes(lambda meta: [b"^v", b"@"]))
    assert helper.registered_handler_problem(classes(registered)) == ""


_REAL_BRIDGE = textwrap.dedent("""
    import importlib.util, json, sys, time
    spec = importlib.util.spec_from_file_location("helper", sys.argv[1])
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    import Quartz
    from Foundation import NSOperationQueue
    bridge = helper.Bridge()
    out = {"load": bridge.load()}
    if out["load"]:
        print(json.dumps(out))
        sys.exit(0)
    config, applied = bridge.configuration(640, 400)
    out["applied"] = applied
    space = Quartz.CGColorSpaceCreateDeviceRGB()
    ctx = Quartz.CGBitmapContextCreate(None, 64, 40, 8, 0, space, Quartz.kCGImageAlphaPremultipliedLast)
    image = Quartz.CGBitmapContextCreateImage(ctx)
    data = bridge.png(image)
    out["png"] = [data[:8].hex(), bridge.image_size(image)]
    background = NSOperationQueue.alloc().init()
    out["background"] = helper.await_handler(lambda h: background.addOperationWithBlock_(lambda: h("bg", None)),
                                             time.monotonic() + 5, pump=bridge.pump)[:2]
    out["main_queue"] = helper.await_handler(lambda h: NSOperationQueue.mainQueue().addOperationWithBlock_(lambda: h("main", None)),
                                             time.monotonic() + 5, pump=bridge.pump)[:2]
    print(json.dumps(out))
""")


def test_real_pyobjc_bridge_registers_handlers_configures_encodes_and_waits(tmp_path):
    """The real boundary short of any ScreenCaptureKit request: no content, no capture, no permission preflight."""
    if sys.platform != "darwin":
        pytest.skip("macOS PyObjC boundary")
    major = _load("unix_window_capture_version", HELPER).mac_major()
    if major is None or major < 14:
        pytest.skip(f"SCScreenshotManager and the single-window options need macOS 14+ (this is {major})")
    pytest.importorskip("objc")
    pytest.importorskip("Quartz")
    script = tmp_path / "bridge.py"
    script.write_text(_REAL_BRIDGE, encoding="utf-8")
    done = subprocess.run([sys.executable, "-I", "-B", str(script), str(HELPER)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    if out["load"].startswith("ScreenCaptureKit has no class"):
        pytest.skip(out["load"])
    assert out["load"] == ""  # both completion handler signatures registered and read back
    assert out["applied"] == {"width": 640, "height": 400, "cursor_included": False, "shadow_included": False}
    assert out["png"] == ["89504e470d0a1a0a", [64, 40]]
    assert out["background"] == [True, "bg"] and out["main_queue"] == [True, "main"]


# --- the screenshot handler ------------------------------------------------------------------------------------


class _API:
    def __init__(self, root: pathlib.Path) -> None:
        self.root, self.tools, self.jobs = root, {}, []

    def get_state_dir(self) -> str:
        return str(self.root / "state")

    def skill_job_dir(self, job_id: str) -> pathlib.Path:
        self.jobs.append(job_id)
        return self.root / "jobs" / job_id

    def register_tool(self, name, handler, **metadata):
        self.tools[name] = {"handler": handler, "metadata": metadata}


@pytest.fixture
def desk(tmp_path, monkeypatch):
    api = _API(tmp_path)
    plugin.register(api)
    monkeypatch.setattr(plugin, "_platform", lambda: "macos")
    monkeypatch.setattr(plugin, "_session_type", lambda: "native")
    monkeypatch.setattr(plugin, "_which", lambda tool: f"/fake/{tool}")
    monkeypatch.setattr(plugin, "_run", lambda *a, **kw: pytest.fail("unexpected process"))
    monkeypatch.setattr(plugin._ComputerUse, "_native", lambda *a, **kw: pytest.fail("unexpected native helper"))
    return api


def shot(api, **kwargs):
    return json.loads(api.tools["screenshot"]["handler"](**kwargs))


def fake_helper(monkeypatch, *, width=30, height=20, answer=None, rc=0, stdout=None, write=True, seen=None):
    """The capture helper's process: checks its argv, writes a PNG at the requested output, prints its answer."""
    def run(cmd, *, timeout):
        assert cmd[1:3] == ["-I", "-B"] and cmd[3].endswith("macos_capture.py") and timeout == 30, cmd
        spec = json.loads(cmd[4])
        assert set(spec) == {"app", "window", "output", "timeout_ms"} and spec["timeout_ms"] == 5000
        if seen is not None:
            seen.append(spec)
        if write:
            pathlib.Path(spec["output"]).write_bytes(png_bytes(width, height))
        body = answer if answer is not None else {
            "ok": True, "backend": "ScreenCaptureKit", "output": spec["output"], "image_width": width, "image_height": height,
            "app": {"app": "Editor", "pid": 11, "app_id": "com.example.editor"},
            "window": {"window_id": 7, "title": "Doc", "pid": 11, "bounds": [0, 0, 15, 10], "on_screen": True, "layer": 0},
            "cursor_included": False, "shadow_included": False, "point_pixel_scale": 2.0, "capture_ms": 12.5}
        return rc, (stdout if stdout is not None else json.dumps(body) + "\n"), ""
    monkeypatch.setattr(plugin, "_run", run)


def test_a_window_capture_without_the_grant_carries_the_one_refusal_field(desk, monkeypatch):
    denied = {"ok": False, "reason": "screen_recording_denied", "app": {"app": "Editor", "pid": 11},
              "permissions": {"screen_recording": "denied"}, "error": "no Screen Recording grant; nothing was asked or captured"}
    fake_helper(monkeypatch, answer=denied, rc=1, write=False)
    out = shot(desk, app="Editor", window=7)
    assert out["ok"] is False and out["reason"] == "screen_recording_denied" and out["refusal"]["permission"] == "screen_recording", out
    fake_helper(monkeypatch, answer={"ok": False, "reason": "window_not_found", "error": "no such window"}, rc=1, write=False)
    other = shot(desk, app="Editor", window=8)
    assert other["reason"] == "window_not_found" and "refusal" not in other


def test_screenshot_schema_names_the_window_parameter(desk):
    meta = desk.tools["screenshot"]["metadata"]
    selector = meta["schema"]["properties"]["window"]
    assert selector["type"] == ["integer", "string"] and "default" not in selector
    assert "window= with app= (local macOS 14+)" in meta["description"] and "capture-only" in meta["description"]


def test_a_window_capture_is_capture_only_and_keyed_to_its_window(desk, monkeypatch, tmp_path):
    seen = []
    fake_helper(monkeypatch, seen=seen)
    first = shot(desk, app="Editor", window=7, job_id="j")
    second = shot(desk, app="Editor", window=7, job_id="j")
    assert first["ok"] is True and first["capture_kind"] == "window" and first["backend"] == "ScreenCaptureKit"
    assert first["permissions_available"] == ["screen_recording"]
    assert first["auto_attach_source"] == second["auto_attach_source"] == "window:11:7"
    assert first["auto_attach_image"] == first["path"] and pathlib.Path(first["path"]).is_file()
    assert first["mapped_input_available"] is False and first["window"]["window_id"] == 7 and first["app"]["pid"] == 11
    for key in ("observation_id", "coord_transform", "display", "input_width", "input_height"):
        assert key not in first, key
    assert (first["image_width"], first["image_height"]) == (30, 20) and first["downscaled"] is False
    assert seen[0]["output"] != seen[1]["output"] and seen[0]["window"] == 7 and seen[0]["app"] == "Editor"
    assert not (tmp_path / "state" / plugin._TRANSFORM_FILE).exists()  # display observations stay as they were


def test_a_large_window_is_downscaled_with_its_aspect_ratio(desk, monkeypatch):
    pytest.importorskip("PIL")
    fake_helper(monkeypatch, width=2000, height=1000)
    result = shot(desk, app="Editor", window="Doc")
    assert (result["capture_width_px"], result["capture_height_px"]) == (2000, 1000)
    assert (result["image_width"], result["image_height"]) == (1280, 640) and result["downscaled"] is True
    assert pathlib.Path(result["full_resolution_path"]).is_file() and result["path"] != result["full_resolution_path"]


@pytest.mark.parametrize("kwargs,reason", [
    ({"window": 7}, "app_required"),
    ({"window": 7, "app": "  "}, "app_required"),
    ({"window": True, "app": "Editor"}, "bad_option"),
    ({"window": "", "app": "Editor"}, "bad_option"),
    ({"window": None, "app": "Editor"}, "bad_option"),
    ({"window": 0, "app": "Editor"}, "bad_option"),
    ({"window": -1, "app": "Editor"}, "bad_option"),
    ({"window": "   ", "app": "Editor"}, "bad_option"),
    ({"window": [7], "app": "Editor"}, "bad_option"),
])
def test_window_requests_that_cannot_name_one_window_refuse_without_a_process(desk, kwargs, reason):
    result = shot(desk, **kwargs)
    assert result["ok"] is False and result["reason"] == reason and desk.jobs == []


def test_window_capture_is_local_macos_only(desk, monkeypatch, tmp_path):
    monkeypatch.setattr(plugin, "_platform", lambda: "linux")
    assert shot(desk, app="Editor", window=7)["reason"] == "unsupported_platform"
    monkeypatch.setattr(plugin, "_platform", lambda: "macos")
    impl = desk.tools["screenshot"]["handler"].__self__
    impl.add_connection(name="guest", backend="osworld_http", target="http://a.invalid", activate=True)
    monkeypatch.setattr(impl, "_osworld_screenshot", lambda *a, **k: pytest.fail("remote capture for window="))
    for selector in (7, "", None):
        result = shot(desk, app="Editor", window=selector)
        assert result["reason"] == "unsupported_backend" and "guest" in result["error"] and desk.jobs == []


def test_a_disabled_connection_still_fails_closed_first(desk, monkeypatch):
    impl = desk.tools["screenshot"]["handler"].__self__
    monkeypatch.setattr(impl, "_connection_snapshot", lambda connection=None: ("gone", {"backend": "local", "disabled": True}))
    result = shot(desk, window=7)  # no app= either: the connection's refusal comes first
    assert result["ok"] is False and result["connection"] == "gone" and "unusable" in result["error"] and desk.jobs == []


def test_an_omitted_window_keeps_the_display_capture(desk, monkeypatch):
    impl = desk.tools["screenshot"]["handler"].__self__
    monkeypatch.setattr(impl, "_capture_display", lambda app: {"index": 1, "main": True, "x": 0, "y": 0, "width": 30, "height": 20})

    def screencapture(cmd, **kw):
        assert cmd[0] == "screencapture"
        pathlib.Path(cmd[-1]).write_bytes(png_bytes(60, 40))
        return 0, "", ""
    monkeypatch.setattr(plugin, "_run", screencapture)
    result = shot(desk)
    assert result["ok"] is True and result["backend"] == "screencapture" and result.get("observation_id")
    assert "capture_kind" not in result


def test_the_helpers_own_refusal_is_returned_and_nothing_kept(desk, monkeypatch, tmp_path):
    refusal = {"ok": False, "reason": "window_ambiguous", "error": "2 windows", "windows": [], "windows_total": 2}
    fake_helper(monkeypatch, rc=1, answer=refusal)
    assert shot(desk, app="Editor", window="Same", job_id="r") == refusal
    assert list((tmp_path / "jobs" / "r" / "output").iterdir()) == []


@pytest.mark.parametrize("case", ["two_lines", "ok_with_failure_exit", "crashed", "no_answer"])
def test_only_a_normal_end_with_one_answer_is_a_capture(desk, monkeypatch, tmp_path, case):
    """Unlike the AX helper's checkpoint, the last line of a broken run is never taken as a capture."""
    if case == "two_lines":
        fake_helper(monkeypatch, stdout='{"ok": false}\n{"ok": true}\n')
    elif case == "ok_with_failure_exit":
        fake_helper(monkeypatch, rc=1)
    elif case == "crashed":
        fake_helper(monkeypatch, rc=-6, stdout="")
    else:
        fake_helper(monkeypatch, stdout="")
    result = shot(desk, app="Editor", window=7, job_id="b")
    assert result["ok"] is False and result["reason"] == "helper_failed", (case, result)
    assert list((tmp_path / "jobs" / "b" / "output").iterdir()) == []


def test_a_stopped_helper_is_no_capture_even_after_printing_ok(desk, monkeypatch, tmp_path):
    def run(cmd, *, timeout):
        output = pathlib.Path(json.loads(cmd[4])["output"])
        output.write_bytes(png_bytes(30, 20))
        pathlib.Path(f"{output}.4242.part").write_bytes(b"\x89PNG")
        raise subprocess.TimeoutExpired(cmd, timeout, output=b'{"ok": true}\n')
    monkeypatch.setattr(plugin, "_run", run)
    result = shot(desk, app="Editor", window=7, job_id="t")
    assert result["reason"] == "helper_stopped" and list((tmp_path / "jobs" / "t" / "output").iterdir()) == []


def test_a_helper_that_cannot_start_is_named(desk, monkeypatch):
    def run(cmd, *, timeout):
        raise FileNotFoundError("python")
    monkeypatch.setattr(plugin, "_run", run)
    assert shot(desk, app="Editor", window=7)["reason"] == "helper_not_started"


@pytest.mark.parametrize("damage", ["missing", "truncated", "size", "foreign", "path"])
def test_an_ok_answer_without_its_whole_png_is_refused(desk, monkeypatch, tmp_path, damage):
    seen = []
    if damage == "missing":
        fake_helper(monkeypatch, write=False, seen=seen)
    elif damage == "size":
        fake_helper(monkeypatch, seen=seen)
        monkeypatch.setattr(plugin, "_png_dimensions", lambda _p: (31, 20))
    elif damage == "foreign":
        def run(cmd, *, timeout):
            spec = json.loads(cmd[4])
            pathlib.Path(spec["output"]).write_bytes(png_bytes(30, 20))
            body = {"ok": True, "output": spec["output"], "image_width": 30, "image_height": 20, "app": {"pid": 11},
                    "window": {"window_id": 7, "pid": 12}}
            return 0, json.dumps(body), ""
        monkeypatch.setattr(plugin, "_run", run)
    elif damage == "path":
        def run(cmd, *, timeout):
            spec = json.loads(cmd[4])
            pathlib.Path(spec["output"]).write_bytes(png_bytes(30, 20))
            body = {"ok": True, "output": "/elsewhere.png", "image_width": 30, "image_height": 20, "app": {"pid": 11},
                    "window": {"window_id": 7, "pid": 11}}
            return 0, json.dumps(body), ""
        monkeypatch.setattr(plugin, "_run", run)
    else:
        def run(cmd, *, timeout):
            spec = json.loads(cmd[4])
            pathlib.Path(spec["output"]).write_bytes(png_bytes(30, 20)[:40])
            body = {"ok": True, "output": spec["output"], "image_width": 30, "image_height": 20, "app": {"pid": 11},
                    "window": {"window_id": 7, "pid": 11}}
            return 0, json.dumps(body), ""
        monkeypatch.setattr(plugin, "_run", run)
    result = shot(desk, app="Editor", window=7, job_id="d")
    assert result["ok"] is False and result["reason"] == "capture_failed", (damage, result)
    assert list((tmp_path / "jobs" / "d" / "output").iterdir()) == []

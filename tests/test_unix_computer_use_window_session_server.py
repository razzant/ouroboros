"""One task's ScreenCaptureKit session of one exact macOS window: its real boundaries.

The MCP result shape through the SDK's in-memory client; the server process over real stdio, where it is asked only
what it refuses or answers without a session; the manifest row through the manifest contract; and
``test_real_pyobjc_*``, which loads the real frameworks and feeds the real sink sample buffers the test paints itself.
No test lists windows, asks ScreenCaptureKit for content, starts a stream, captures, reads or prompts for a
permission, or shows anything: starting and stopping a real stream needs the owner's consent and stays a named gap.
What the session decides is tested against fakes in ``test_unix_computer_use_window_session.py``.
"""
from __future__ import annotations

import base64
import json
import queue
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from tests.test_unix_computer_use_window_session import SKILL, _load, png_bytes

SERVER_ID = "mac_window_session"


@pytest.fixture(scope="module")
def server():
    return _load("unix_window_session_server", SKILL / "window_session_server.py")


@pytest.fixture(scope="module")
def stream():
    return _load("unix_window_session_stream_real", SKILL / "macos_stream.py")


# --- the MCP server ----------------------------------------------------------------------------------------------


def test_the_tools_are_three_and_listing_them_needs_nothing_native(server):
    assert [tool["name"] for tool in server.TOOLS] == ["begin", "frame", "release"]
    begin_schema = server.TOOLS[0]["inputSchema"]
    assert begin_schema["required"] == ["app", "window"] and begin_schema["additionalProperties"] is False
    assert begin_schema["properties"]["window"]["type"] == ["integer", "string"]
    assert set(server.TOOLS[1]["inputSchema"]["properties"]) == {"after_seq", "wait_ms"}
    assert server.TOOLS[2]["inputSchema"] == {"type": "object", "additionalProperties": False, "properties": {}}


_IDLE = textwrap.dedent("""
    import importlib.util, json, sys
    spec = importlib.util.spec_from_file_location("server", sys.argv[1])
    server = importlib.util.module_from_spec(spec); spec.loader.exec_module(server)
    sessions = server.load_stream().WindowSessions()
    answers = [server.dispatch(sessions, "frame", {})[0]["reason"], server.dispatch(sessions, "release", {})[0]["state"],
               server.dispatch(sessions, "begin", {"app": "", "window": 7})[0]["reason"]]
    sessions.shutdown()
    native = sorted(name for name in ("objc", "Quartz", "Foundation", "AppKit", "CoreFoundation") if name in sys.modules)
    print(json.dumps({"answers": answers, "native": native}))
""")


def test_a_server_that_only_lists_idles_or_refuses_loads_nothing_native(tmp_path):
    """A fresh isolated interpreter: the modules, a session object and three refused or empty calls import no PyObjC."""
    script = tmp_path / "idle.py"
    script.write_text(_IDLE, encoding="utf-8")
    done = subprocess.run([sys.executable, "-I", "-B", str(script), str(SKILL / "window_session_server.py")],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["native"] == [] and out["answers"][:2] == ["no_session", "none"]
    assert out["answers"][2] in ("bad_option", "unsupported_platform", "unsupported_os")


def test_dispatch_names_an_unknown_tool_and_contains_a_failure(server):
    class Broken:
        def frame(self, raw):
            raise RuntimeError("boom")

        def release(self, raw):
            return {"ok": True, "released": False, "state": "none"}, None
    assert server.dispatch(Broken(), "shutdown", {})[0]["reason"] == "unknown_tool"  # not a tool, though a method
    facts, png = server.dispatch(Broken(), "frame", {})
    assert png is None and facts == {"ok": False, "reason": "server_error", "error": "RuntimeError: boom"}
    assert server.dispatch(Broken(), "release", None)[0]["released"] is False


def test_calls_run_on_the_thread_that_serves_them_in_order(server):
    main_thread = server.MainThread()
    ran, answers = [], []

    def caller():
        try:
            for index in range(3):
                answers.append(main_thread.call(lambda index=index: (ran.append((index, threading.get_ident())), index)[1]))
            with pytest.raises(ValueError):
                main_thread.call(lambda: (_ for _ in ()).throw(ValueError("refused")))
        finally:
            main_thread.close()
    worker = threading.Thread(target=caller)
    worker.start()
    main_thread.run()  # this thread is the "main thread" here
    worker.join(5)
    assert answers == [0, 1, 2] and [index for index, _thread in ran] == [0, 1, 2]
    assert {thread for _index, thread in ran} == {threading.get_ident()}


def test_a_jobless_moment_runs_the_idle_look_and_jobs_keep_priority(server):
    ticks = []
    main_thread = server.MainThread(idle=lambda: ticks.append(time.monotonic()), idle_s=0.01)
    answers = []

    def caller():
        try:
            for _ in range(3):
                time.sleep(0.05)  # leave the loop jobless moments between the calls
                answers.append(main_thread.call(lambda: "served"))
        finally:
            main_thread.close()

    worker = threading.Thread(target=caller)
    worker.start()
    main_thread.run()
    worker.join(5)
    assert answers == ["served"] * 3 and len(ticks) >= 2  # idle ran between jobs; every job was served


def test_a_raising_idle_look_is_reported_and_the_loop_keeps_serving(server, capsys):
    calls = []

    def idle():
        calls.append(1)
        raise RuntimeError("the lookup broke")

    main_thread = server.MainThread(idle=idle, idle_s=0.01)

    def caller():
        try:
            assert main_thread.call(lambda: "served") == "served"
            time.sleep(0.05)  # leave the loop at least one jobless moment before the close
        finally:
            main_thread.close()

    worker = threading.Thread(target=caller)
    worker.start()
    main_thread.run()
    worker.join(5)
    assert calls and "the lookup broke" in capsys.readouterr().err  # reported, and the loop did not end


def test_a_result_is_one_json_object_and_an_image_only_when_the_facts_say_so(server):
    import anyio
    from mcp.shared.memory import create_connected_server_and_client_session

    png = png_bytes(30, 20)
    answers = {"frame": ({"ok": True, "image": True, "seq": 3, "title": "Заметки"}, png),
               "begin": ({"ok": True, "state": "live"}, None),
               "release": ({"ok": False, "reason": "stop_unconfirmed", "error": "x", "image": True}, png)}
    seen = []

    def call(name, arguments):
        seen.append((name, arguments))
        return answers[name]

    async def talk():
        async with create_connected_server_and_client_session(server.build_server(call)) as client:
            listed = await client.list_tools()
            return listed, [await client.call_tool(name, args) for name, args in
                            (("frame", {"after_seq": 2}), ("begin", {"app": "Editor", "window": 7}), ("release", {}))]

    listed, (frame, begun, released) = anyio.run(talk)
    assert [tool.name for tool in listed.tools] == ["begin", "frame", "release"]
    assert seen[0] == ("frame", {"after_seq": 2}) and seen[1] == ("begin", {"app": "Editor", "window": 7})
    assert frame.isError is False and [part.type for part in frame.content] == ["text", "image"]
    assert json.loads(frame.content[0].text) == answers["frame"][0] and frame.structuredContent is None
    assert frame.content[1].mimeType == "image/png" and base64.b64decode(frame.content[1].data) == png
    assert begun.isError is False and [part.type for part in begun.content] == ["text"]
    assert released.isError is True and [part.type for part in released.content] == ["text", "image"]


class _Process:
    """The server over its real stdio; every read is bounded, so a silent server fails the test instead of hanging it."""

    def __init__(self, *args):
        self.proc = subprocess.Popen([sys.executable, "-I", "-B", str(SKILL / "window_session_server.py"), *args],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     cwd=str(SKILL), text=True, encoding="utf-8")
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: [self.lines.put(line) for line in self.proc.stdout], daemon=True).start()
        self.ids = 0

    def send(self, method, params=None, *, notify=False):
        message = {"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})}
        if not notify:
            self.ids += 1
            message["id"] = self.ids
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        if notify:
            return None
        while True:
            answer = json.loads(self.lines.get(timeout=60))
            if answer.get("id") == self.ids:
                assert "error" not in answer, answer
                return answer["result"]

    def tool(self, name, arguments):
        result = self.send("tools/call", {"name": name, "arguments": arguments})
        return result, json.loads(result["content"][0]["text"])


def test_the_server_process_lists_refuses_and_ends_with_its_stdin(tmp_path):
    """Isolated start (-I -B), the protocol beside a main thread that runs the tools, and a clean end on stdin's close."""
    server = _Process()
    try:
        hello = server.send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "test", "version": "0"}})
        assert hello["serverInfo"]["name"] == SERVER_ID and "capture-only" in hello["instructions"].lower()
        server.send("notifications/initialized", notify=True)
        assert [tool["name"] for tool in server.send("tools/list")["tools"]] == ["begin", "frame", "release"]
        result, facts = server.tool("frame", {"wait_ms": 5000})  # no session: answered at once, nothing waited for
        assert result["isError"] is True and facts["reason"] == "no_session" and len(result["content"]) == 1
        result, facts = server.tool("release", {})
        assert result.get("isError", False) is False and facts == {"ok": True, "released": False, "state": "none"}
        result, facts = server.tool("begin", {"app": "", "window": 7})  # refused before anything is resolved
        wanted = "bad_option" if sys.platform == "darwin" else "unsupported_platform"
        if sys.platform == "darwin" and facts["reason"] == "unsupported_os":
            wanted = "unsupported_os"
        assert result["isError"] is True and facts["reason"] == wanted, facts
        result, facts = server.tool("screenshot", {})
        assert result["isError"] is True and facts["reason"] == "unknown_tool"
        server.proc.stdin.close()
        assert server.proc.wait(timeout=30) == 0
        assert "Traceback" not in server.proc.stderr.read()  # the SDK's own note about the unknown tool is all
    finally:
        if server.proc.poll() is None:
            server.proc.kill()
            server.proc.wait(timeout=30)


_RAISING_STREAM = textwrap.dedent("""
    import atexit, sys
    atexit.register(lambda: sys.stderr.write("INTERPRETER-TEARDOWN\\n"))  # os._exit skips it


    class WindowSessions:
        def shutdown(self):
            raise {error}("the stream's stop raised")
""")


@pytest.mark.parametrize("error", ["RuntimeError", "KeyboardInterrupt"])
def test_a_shutdown_that_raises_still_exits_without_interpreter_teardown(tmp_path, error):
    """The server's own exit is chosen whatever its end raised: a framework callback must not meet a finalizing
    interpreter. The real server file runs beside a stand-in ``macos_stream.py`` whose shutdown raises."""
    (tmp_path / "window_session_server.py").write_bytes((SKILL / "window_session_server.py").read_bytes())
    (tmp_path / "macos_stream.py").write_text(_RAISING_STREAM.format(error=error), encoding="utf-8")
    done = subprocess.run([sys.executable, "-I", "-B", str(tmp_path / "window_session_server.py")], input="",
                          capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    assert done.returncode == 1, done.stderr
    assert f"{error}: the stream's stop raised" in done.stderr  # reported on stderr
    assert "INTERPRETER-TEARDOWN" not in done.stderr, done.stderr


def test_the_manifest_declares_the_server_for_a_task_session():
    """The owner's Connect tools row: one stdio server for the task loop, started by the agent's own Python."""
    import yaml

    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from ouroboros.skill_mcp import entry_problem

    source = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(source.split("---", 2)[1])
    assert [server["id"] for server in frontmatter["optional_mcp_servers"]] == [SERVER_ID]
    manifest = parse_skill_manifest_text(source)
    assert manifest.mcp_servers == [{"id": SERVER_ID, "title": "Local Mac window capture (requires begin)",
                                     "runtime": "agent_python", "entry": "window_session_server.py",
                                     "task_session": True, "browser_policy": False}]
    assert entry_problem(SKILL, "window_session_server.py") == ""
    assert _load("unix_window_session_server_name", SKILL / "window_session_server.py").SERVER_NAME == SERVER_ID


# --- the real PyObjC boundary, short of any ScreenCaptureKit request ------------------------------------------------


_REAL_STREAM = textwrap.dedent("""
    import importlib.util, json, os, sys, threading
    spec = importlib.util.spec_from_file_location("stream", sys.argv[1])
    stream = importlib.util.module_from_spec(spec); spec.loader.exec_module(stream)
    out = {"imports_before_load": sorted(name for name in ("objc", "Quartz", "Foundation") if name in sys.modules)}
    import objc, Quartz
    from Foundation import NSBundle, NSError, NSInvocation, NSNumber, NSOperationQueue
    bridge = stream.StreamBridge()
    out["load"] = bridge.load()
    if out["load"]:
        print(json.dumps(out)); sys.exit(0)
    out["handlers"] = stream.capture.registered_handler_problem(bridge.classes, {**stream.capture.HANDLERS, **stream.STREAM_HANDLERS})
    out["applied"] = bridge.stream_configuration(640, 400)[1]
    make, mach = {}, {}
    objc.loadBundleFunctions(NSBundle.bundleWithPath_(stream._CM_PATH), make, [
        ("CMVideoFormatDescriptionCreateForImageBuffer", b"i^{__CFAllocator=}" + stream._PB + b"^@", "",
         {"arguments": {2: {"type_modifier": b"o", "already_cfretained": True}}}),
        ("CMSampleBufferCreateReadyWithImageBuffer", b"i^{__CFAllocator=}" + stream._PB + b"@^{?=" + stream._CMTIME * 3 + b"}^" + stream._SB, "",
         {"arguments": {3: {"type_modifier": b"n"}, 4: {"type_modifier": b"o", "already_cfretained": True}}})])
    objc.loadBundleFunctions(None, mach, [("mach_absolute_time", b"Q")])
    objc.loadBundleFunctions(NSBundle.bundleWithPath_(stream._CM_PATH), make, [("CMSampleBufferInvalidate", b"i" + stream._SB)])
    cm, keys = bridge._cm, bridge._keys
    special = {}  # CoreMedia's own times that are no number, and its zero
    names = ("kCMTimeInvalid", "kCMTimeIndefinite", "kCMTimePositiveInfinity", "kCMTimeNegativeInfinity", "kCMTimeZero")
    objc.loadBundleVariables(NSBundle.bundleWithPath_(stream._CM_PATH), special, [(name, stream._CMTIME) for name in names])
    out["special_times"] = [stream._seconds(tuple(special[name])) for name in names]
    out["host_times"] = [bridge.host_now() > 0, stream._seconds(tuple(cm["CMClockMakeHostTimeFromSystemUnits"](mach["mach_absolute_time"]()))) > 0]

    def sample(width, height, rgb, status, fmt=Quartz.kCVPixelFormatType_32BGRA, display=True):
        err, pb = Quartz.CVPixelBufferCreate(None, width, height, fmt, {Quartz.kCVPixelBufferIOSurfacePropertiesKey: {}}, None)
        if fmt == Quartz.kCVPixelFormatType_32BGRA:
            Quartz.CVPixelBufferLockBaseAddress(pb, 0)
            stride = Quartz.CVPixelBufferGetBytesPerRow(pb)
            row = bytes((rgb[2], rgb[1], rgb[0], 255)) * width
            row += b"\\x7f" * (stride - len(row))
            Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(stride * height)[:] = row * height
            Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)
        err, desc = make["CMVideoFormatDescriptionCreateForImageBuffer"](None, pb, None)
        now = tuple(cm["CMClockGetTime"](cm["CMClockGetHostTimeClock"]()))
        err, sb = make["CMSampleBufferCreateReadyWithImageBuffer"](None, pb, desc, ((0, 0, 0, 0), now, (0, 0, 0, 0)), None)
        row0 = cm["CMSampleBufferGetSampleAttachmentsArray"](sb, True)[0]
        row0[keys["SCStreamFrameInfoStatus"]] = NSNumber.numberWithInteger_(status)
        if display:
            row0[keys["SCStreamFrameInfoDisplayTime"]] = NSNumber.numberWithUnsignedLongLong_(mach["mach_absolute_time"]())
        return sb

    store = stream.FrameStore(bridge.host_now)
    sink = stream._sink_class(objc).alloc().initWithFeed_(stream._Feed(bridge, store))
    selector = b"stream:didOutputSampleBuffer:ofType:"
    signature = sink.methodSignatureForSelector_(selector)

    def deliver(sb, kind=0):  # the runtime's own call, on a GCD worker thread
        done = threading.Event()
        def work():
            inv = NSInvocation.invocationWithMethodSignature_(signature)
            inv.setTarget_(sink); inv.setSelector_(selector)
            inv.setArgument_atIndex_(None, 2); inv.setArgument_atIndex_(sb, 3); inv.setArgument_atIndex_(kind, 4)
            inv.invoke(); done.set()
        with objc.autorelease_pool():
            NSOperationQueue.alloc().init().addOperationWithBlock_(work)
            assert done.wait(10)

    deliver(sample(131, 67, (200, 10, 20), 4))  # an odd width: padded rows
    deliver(sample(8, 8, (0, 0, 0), 1))         # idle
    deliver(sample(131, 67, (1, 2, 3), 0), 1)   # an audio-typed callback
    held, facts = store.snapshot()
    out["held"] = [facts["seq"], facts["frame_status"], facts["frame_time_basis"], facts["image_width"], facts["image_height"],
                   facts["stream"]["reports"], 0 <= facts["frame_age_ms"] < 60000]
    sessions = stream.WindowSessions(bridge_factory=lambda: bridge)
    sessions._bridge = bridge
    data = sessions._encode(bridge, held)
    out["png"] = [data[:8].hex(), int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big"), os.listdir(sessions._scratch)]
    from AppKit import NSBitmapImageRep
    rep = NSBitmapImageRep.imageRepWithData_(data)
    color = rep.colorAtX_y_(130, 66)
    out["last_pixel"] = [round(color.redComponent() * 255), round(color.greenComponent() * 255), round(color.blueComponent() * 255)]
    sessions.shutdown()
    deliver(sample(16, 16, (0, 0, 0), 0, fmt=Quartz.kCVPixelFormatType_32ARGB))  # an image this code does not read
    held, facts = store.snapshot()
    out["unreadable"] = [held is None, facts["withheld"]["status"]]
    deliver(sample(16, 16, (9, 9, 9), 0, display=False))  # an image without a display time
    held, facts = store.snapshot()
    out["no_display_time"] = [held is not None, facts["seq"], facts["frame_time_basis"], facts["stream"]["last_report_time_basis"]]
    broken = sample(16, 16, (9, 9, 9), 0)
    make["CMSampleBufferInvalidate"](broken)
    deliver(broken)  # a screen sample that is no longer valid
    held, facts = store.snapshot()
    out["invalid_sample"] = [held is None, facts["withheld"]["status"], facts["seq"]]
    deliver(sample(16, 16, (9, 9, 9), 0))
    sink.performSelector_withObject_(b"streamDidBecomeInactive:", None)  # the stream's notice that its window closed
    held, facts = store.snapshot()
    out["inactive"] = [held is None, facts["withheld"]["status"], facts["seq"]]
    deliver(sample(16, 16, (9, 9, 9), 0))  # the same stream delivers again
    error = NSError.errorWithDomain_code_userInfo_("com.apple.ScreenCaptureKit.SCStreamErrorDomain", -3817, None)
    sink.performSelector_withObject_withObject_(b"stream:didStopWithError:", None, error)
    deliver(sample(16, 16, (9, 9, 9), 0))  # after the stream's own end
    held, facts = store.snapshot()
    out["stopped"] = [held is None, facts["state"], facts["seq"], facts["stop"]["reason"], facts["stop"]["code"], list(store.errors)]
    print(json.dumps(out))
""")


def test_real_pyobjc_sink_holds_painted_frames_encodes_them_and_stays_ended(tmp_path, stream):
    """The real boundary short of any ScreenCaptureKit request: no content, no stream, no permission preflight."""
    if sys.platform != "darwin":
        pytest.skip("macOS PyObjC boundary")
    major = stream.capture.mac_major()
    if major is None or major < 14:
        pytest.skip(f"the window stream needs macOS 14+ (this is {major})")
    pytest.importorskip("objc")
    pytest.importorskip("Quartz")
    script = tmp_path / "stream.py"
    script.write_text(_REAL_STREAM, encoding="utf-8")
    done = subprocess.run([sys.executable, "-I", "-B", str(script), str(SKILL / "macos_stream.py")], capture_output=True,
                          text=True, timeout=180)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    if out["load"].startswith("ScreenCaptureKit has no class"):
        pytest.skip(out["load"])
    assert out["imports_before_load"] == []  # importing the module touched no PyObjC
    assert out["load"] == "" and out["handlers"] == ""  # start and stop handlers registered and read back
    assert out["applied"] == {"width": 640, "height": 400, "cursor_included": False, "shadow_included": False,
                              "frames_per_second_max": 5.0, "queue_depth": 4, "audio": False, "pixel_format": "BGRA"}
    assert out["held"] == [1, "started", "display", 131, 67, {"started": 1, "idle": 1}, True]
    assert out["png"] == ["89504e470d0a1a0a", 131, 67, []] and out["last_pixel"] == [200, 10, 20]
    assert out["special_times"] == [None, None, None, None, 0.0] and out["host_times"] == [True, True]
    assert out["unreadable"] == [True, "pixel_format:0x20"]
    assert out["no_display_time"] == [True, 2, "received", "received"]
    assert out["invalid_sample"] == [True, "invalid_sample", 2]
    assert out["inactive"] == [True, "inactive", 3]
    assert out["stopped"] == [True, "stopped", 4, "user_stopped", -3817, []]

"""Opt-in macOS consumer qualification for the skill-bound window stream.

On a disposable macOS 14+ GUI with an existing Screen Recording grant, pass
OUROBOROS_MACOS_STREAM_QA=1 after the safe_test boundary. The display oracle
reads the whole screen. Use a desktop containing only these test windows.
Images stay in the test's private scratch. The one-run entry point and its
manifest are in tests/_run_window_capture_acceptance.py. The managed-worker
scenarios are in tests/system_e2e/test_window_capture_task_native.py.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import pytest

from ouroboros.loop_tool_execution import _execute_with_timeout
from tests._window_capture_observer import (
    capture_server,
    check_permission_result,
    confirm_idle_window_stop,
    overlap,
    pyobjc_mapped,
    require_capture_grant,
    wait_for,
    window_key,
    window_rows,
)
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import owner
from tests.test_mcp_result_images import _batch, _pixels

REPO = pathlib.Path(__file__).resolve().parents[1]
SKILL = "unix_computer_use"
SERVER = "mac_window_session"
GREEN, BLUE, MAGENTA = (24, 179, 74), (37, 99, 235), (229, 41, 163)
OPT_IN = sys.platform == "darwin" and os.environ.get("OUROBOROS_MACOS_STREAM_QA") == "1"
desktop_only = pytest.mark.skipif(not OPT_IN, reason="needs an announced macOS desktop and OUROBOROS_MACOS_STREAM_QA=1")

# Names published by window_session_server.py and checked against the installed catalog.
BEGIN = "begin"
FRAME = "frame"
RELEASE = "release"

FIXTURE = r'''
// Compiled only by the opt-in test. Two instances, two owned PIDs; no input or activation.
import AppKit
import Foundation

let args = CommandLine.arguments
guard args.count == 4, ["target", "occluder"].contains(args[1]) else { exit(2) }
let role = args[1], title = args[2], logPath = args[3]
let logFile: FileHandle = {
    FileManager.default.createFile(atPath: logPath, contents: nil)
    guard let file = FileHandle(forWritingAtPath: logPath) else { exit(2) }
    return file
}()

func log(_ event: String, _ more: [String: Any] = [:]) {
    var row = more
    row["event"] = event
    row["role"] = role
    row["pid"] = Int(getpid())
    row["uptime"] = ProcessInfo.processInfo.systemUptime
    if let data = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]) {
        logFile.write(data + Data([10]))
    }
}

final class ColorView: NSView {
    var color = NSColor(srgbRed: 24/255, green: 179/255, blue: 74/255, alpha: 1)
    override func draw(_ dirtyRect: NSRect) {
        color.setFill()
        bounds.fill()
    }
}

final class ObservedWindow: NSWindow {
    deinit { log("deallocated") }
}

final class Fixture: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var view: ColorView! = ColorView(frame: .zero)
    func applicationDidFinishLaunching(_ notification: Notification) {
        autoreleasepool {
        let rect = NSRect(x: 0, y: 0, width: role == "target" ? 420 : 240,
                          height: role == "target" ? 300 : 160)
        window = ObservedWindow(contentRect: rect, styleMask: [.titled], backing: .buffered, defer: false)
        window.title = title
        window.isReleasedWhenClosed = true // this standalone fixture must release its native surface on close
        window.level = .floating
        view.frame = rect
        view.autoresizingMask = [.width, .height]
        window.contentView = view
        if role == "occluder" {
            view.color = NSColor(srgbRed: 229/255, green: 41/255, blue: 163/255, alpha: 1)
        }
        let screen = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1280, height: 800)
        let x = screen.minX + 80 + (role == "occluder" ? 100 : 0)
        let y = screen.maxY - 80 - (role == "occluder" ? 80 : 0)
        window.setFrameTopLeftPoint(NSPoint(x: x, y: y))
        NSApp.unhideWithoutActivation()
        window.orderFrontRegardless()
        window.displayIfNeeded()
        log("ready", ["title": title, "window_id": window.windowNumber,
                      "visible": window.isVisible, "frame": [window.frame.minX, window.frame.minY,
                                                                  window.frame.width, window.frame.height]])
        DispatchQueue.global().async {
            while let line = readLine() {
                DispatchQueue.main.async { autoreleasepool { self.command(line) } }
            }
            DispatchQueue.main.async { NSApp.terminate(nil) }
        }
        }
    }

    func command(_ line: String) {
        if line == "blue" && role == "target" {
            view.color = NSColor(srgbRed: 37/255, green: 99/255, blue: 235/255, alpha: 1)
            view.needsDisplay = true
            window.displayIfNeeded()
            log("blue", ["window_id": window.windowNumber, "visible": window.isVisible])
        } else if line == "close" {
            let oldID = window.windowNumber
            window.contentView = nil
            view = nil
            // isReleasedWhenClosed consumes a native ownership reference. Supply
            // that reference separately from Swift's property ownership.
            _ = Unmanaged.passRetained(window)
            window.close()
            window = nil
            // Wake this app's event loop so its outer AppKit autorelease pool can
            // drain. This is an app-defined event, not mouse/keyboard/global input.
            if let event = NSEvent.otherEvent(with: .applicationDefined, location: .zero,
                    modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                    windowNumber: 0, context: nil, subtype: 0, data1: 0, data2: 0) {
                NSApp.postEvent(event, atStart: false)
            }
            DispatchQueue.main.async {
                log("closed", ["window_id": oldID, "visible": false])
            }
        } else if line == "quit" {
            log("quit")
            NSApp.terminate(nil)
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular) // the existing launch-free resolver inventories regular apps; never activate
let fixture = Fixture()
app.delegate = fixture
app.run()
'''


def _rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    return [json.loads(line) for line in data[: data.rfind(b"\n") + 1].decode().splitlines() if line]


class OwnedWindow:
    def __init__(self, binary: pathlib.Path, role: str, root: pathlib.Path):
        self.role = role
        self.title = f"ouroboros-stream-qa-{uuid.uuid4().hex[:12]}-{role}"
        self.log = root / f"{role}.jsonl"
        self.stderr = root / f"{role}.stderr"
        with self.stderr.open("w", encoding="utf-8") as err:
            self.proc = subprocess.Popen([str(binary), role, self.title, str(self.log)],
                                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                         stderr=err, text=True, start_new_session=True)
        self.pid = self.proc.pid
        try:
            self.ready = self.wait("ready")
            assert self.ready["pid"] == self.pid and self.ready["visible"], self.ready
        except BaseException:
            self.stop()
            raise

    def wait(self, event: str, timeout: float = 12) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for row in _rows(self.log):
                if row["event"] == event:
                    return row
            if self.proc.poll() is not None:
                break
            time.sleep(0.05)
        raise AssertionError(f"{self.role} did not log {event!r}; exit={self.proc.poll()}")

    def send(self, line: str, *, event: str | None = None) -> dict:
        assert self.proc.poll() is None and self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return self.wait(event or line)

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                if self.proc.stdin and not self.proc.stdin.closed:
                    self.proc.stdin.write("quit\n")
                    self.proc.stdin.flush()
                    self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()  # only this Popen's exact child
                self.proc.wait(timeout=5)


def _pixels_near(source, rgb: tuple[int, int, int], *, around=None) -> int:
    from PIL import Image, ImageCms

    with Image.open(source) as image:
        image.load()
        # The fixture paints explicit sRGB. Display PNGs carry the display's ICC
        # profile: comparing their encoded channels directly is not a color test.
        profile = image.info.get("icc_profile")
        sample = (ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                           ImageCms.createProfile("sRGB"), outputMode="RGB")
                  if profile else image.convert("RGB"))
        if around is not None:
            x, y = (round(f * size) for f, size in zip(around, sample.size))
            assert 5 <= x < sample.width - 5 and 5 <= y < sample.height - 5
            sample = sample.crop((x - 5, y - 5, x + 5, y + 5))
        else:
            sample.thumbnail((1200, 800))
        return sum(1 for r, g, b in sample.getdata()
                   if max(abs(r - rgb[0]), abs(g - rgb[1]), abs(b - rgb[2])) < 35)


def mcp_facts(body):
    # Decode the text part following the host's trust label.
    body = body.split("\n\n", 1)[-1]
    start = body.find("{")
    assert start >= 0, "capture MCP result has no JSON object"
    facts = json.JSONDecoder().raw_decode(body[start:])[0]
    check_permission_result(facts)
    return facts


def _images(row, registry):
    messages, trace = _batch(SimpleNamespace(registry=registry), [row])
    images = _pixels(messages)
    if images:
        assert len(images) == 1 and images[0].startswith(b"\x89PNG\r\n\x1a\n")
        assert trace["tool_calls"][0]["image_attachment"]["status"] == "attached"
    return images


@contextlib.contextmanager
def owned_windows(root):
    require_capture_grant()
    swift, binary = root / "StreamFixture.swift", root / "StreamFixture"
    swift.write_text(FIXTURE, encoding="utf-8")
    subprocess.run(["xcrun", "swiftc", "-swift-version", "5", "-Onone", "-o", str(binary), str(swift)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    with contextlib.ExitStack() as children:
        target = OwnedWindow(binary, "target", root)
        children.callback(target.stop)
        occluder = OwnedWindow(binary, "occluder", root)
        children.callback(occluder.stop)
        assert target.pid != occluder.pid and target.title != occluder.title
        wait_for(lambda: overlap(target, occluder))
        yield target, occluder
    assert target.proc.returncode == occluder.proc.returncode == 0


@contextlib.contextmanager
def connected_skill(drive, seed_repo, settings, monkeypatch, record_property):
    """Use the shipped installer, Connect and MCP catalog in a fresh data root."""
    from ouroboros import config, extension_loader, mcp_client, skill_mcp
    from ouroboros.skill_builtin import _dependency_pass, install_builtin_skill
    from ouroboros.skill_loader import load_enabled
    from tests._shared import clean_extension_runtime_state

    assert mcp_client._MCP_SDK_AVAILABLE
    drive.mkdir(parents=True, exist_ok=True)
    settings_path = drive / "settings.json"
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    with monkeypatch.context() as local:
        local.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
        local.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
        local.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
        local.setenv("OUROBOROS_AGENT_PYTHON", sys.executable)
        local.setenv("OUROBOROS_SKILLS_REPO_PATH", str(drive / "no-extra-skills"))
        local.setattr(config, "DATA_DIR", drive)
        local.setattr(config, "SETTINGS_PATH", settings_path)
        clean_extension_runtime_state()
        mcp_client.reset_manager_for_tests()
        try:
            installed = install_builtin_skill(SKILL, drive_root=drive, seed_dir=seed_repo / "skills",
                                              turn_on_when_ready=True, installed_by="owner")
            assert installed["ok"]
            copied = drive / "skills" / "native" / SKILL
            entry = copied.resolve() / "window_session_server.py"
            source = REPO / "skills" / SKILL
            files = [p.relative_to(source) for p in source.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
            assert files and all((copied / p).read_bytes() == (source / p).read_bytes() for p in files)
            _dependency_pass(drive, [SKILL], logging.getLogger(__name__))
            assert load_enabled(drive, SKILL)
            # Count tool invocations through the real SDK across preview, Connect
            # and discovery. Forward each invocation so a regression is observable.
            calls, observed = [], []
            listings = 0
            original_call = mcp_client.ClientSession.call_tool
            original_list = mcp_client.ClientSession.list_tools

            async def observe_call(client, name, *args, **kwargs):
                calls.append(name)
                return await original_call(client, name, *args, **kwargs)

            async def observe_listing(client, *args, **kwargs):
                nonlocal listings
                listings += 1
                answer = await original_list(client, *args, **kwargs)
                # The SDK still holds the child here. Its caller sees it only
                # after the stdio transport has closed and reaped the process.
                process = capture_server(os.getpid(), entry)
                assert not pyobjc_mapped(process), "discovery loaded the native capture bridge"
                observed.append(process)
                return answer

            local.setattr(mcp_client.ClientSession, "call_tool", observe_call)
            local.setattr(mcp_client.ClientSession, "list_tools", observe_listing)
            preview = skill_mcp.preview_connect(drive, SKILL)
            assert preview.get("ready") is True and preview.get("blockers") == []
            assert len(preview["servers"]) == 1
            declared = preview["servers"][0]
            assert declared["id"] == SERVER and declared["task_session"] and not declared["browser_policy"]
            connected = skill_mcp.connect(drive, SKILL, expected_content_hash=preview["content_hash"],
                                          settings_digest=preview["settings_digest"])
            assert connected.get("ok") and connected.get("saved")
            saved_settings = json.loads(settings_path.read_text())
            assert saved_settings["MCP_ENABLED"] is True
            saved, = saved_settings["MCP_SERVERS"]
            assert saved["skill"] == SKILL and saved["id"] == SERVER
            assert not {"command", "args", "cwd"} & saved.keys()
            cfg, = mcp_client.parse_servers([saved], settings=saved_settings)
            bound = skill_mcp.bind_config(cfg, drive)
            assert not bound.skill_error and bound.task_session
            assert bound.command == os.path.abspath(sys.executable) and bound.args == ["-I", "-B", str(entry)]
            assert pathlib.Path(bound.cwd) == entry.parent
            manager = mcp_client.get_manager()
            manager.reconfigure(saved_settings)
            assert manager.discovery.wait()
            assert manager.refresh_server(SERVER).get("ok")
            assert manager.discovery.wait()
            assert observed and len(observed) == listings and not calls
            local.setattr(mcp_client.ClientSession, "list_tools", original_list)
            local.setattr(mcp_client.ClientSession, "call_tool", original_call)
            for process in observed:
                wait_for(process.gone)
            record_property("discovery_processes", len(observed))
            record_property("discovery_tool_calls", len(calls))
            for name, required in ((BEGIN, {"app", "window"}), (FRAME, {"after_seq", "wait_ms"}), (RELEASE, set())):
                qualified = tool(name)
                assert manager.resolve_tool_name(qualified).status == "callable"
                assert required <= set(manager.get_tool(qualified)["schema"]["properties"])
            assert extension_loader.get_tool(extension_loader.extension_surface_name(SKILL, "screenshot"))
            yield saved_settings, entry
        finally:
            clean_extension_runtime_state()
            mcp_client.reset_manager_for_tests()


def tool(name):
    return f"mcp_{SERVER}__{name}"


def pixels_of_target(png, target_rgb, record_property, prefix):
    own = _pixels_near(io.BytesIO(png), target_rgb)
    front = _pixels_near(io.BytesIO(png), MAGENTA)
    record_property(prefix + "_target_pixels", own)
    record_property(prefix + "_occluder_pixels", front)
    assert own > 5000 and front < 100


def capture_only(facts):
    assert facts.get("mapped_input_available") is False
    assert not {"observation_id", "coord_transform", "display"} & facts.keys()


def _check_mapping(mapper, previous_ring, display, frame):
    capture_only(frame)
    with pytest.raises(ValueError, match="observation_id is required"):
        mapper._map_xy(12, 17, raw=False, observation_id=frame.get("observation_id", ""))
    assert mapper._load_transform() == previous_ring
    if display is not None:
        tf, oid = display["coord_transform"], display["observation_id"]
        x, y, evidence = mapper._map_xy(12, 17, raw=False, observation_id=oid)
        assert (x, y) == (round(12 * tf["sx"] + tf["ox"]), round(17 * tf["sy"] + tf["oy"]))
        assert evidence["observation_id"] == oid and evidence["geometry_verification"] == "live"


@desktop_only
@pytest.mark.parametrize("scene", ["occluded", "independent_lifecycle"])
def test_owned_occluded_window_through_connect_and_task_session(tmp_path, monkeypatch, scene, record_property):
    from ouroboros import extension_loader, mcp_task_session
    from ouroboros.skill_loader import skill_state_dir_path
    from tests.test_unix_computer_use_skill import _API, _load_plugin

    require_capture_grant()
    drive = tmp_path / "private-data"
    with connected_skill(drive, REPO, {}, monkeypatch, record_property) as (_, entry), owned_windows(tmp_path) as (target, occluder):
        registry, _, _ = _registry(drive, monkeypatch)
        registry._ctx.repo_dir = REPO
        registry._ctx.task_metadata = {"work_kind": "owner"}
        shot = extension_loader.extension_surface_name(SKILL, "screenshot")

        def invoke(label, name, args):
            row = _execute_with_timeout(
                registry, {"id": label, "function": {"name": name, "arguments": json.dumps(args)}},
                drive / "logs", 90, task_id=registry._ctx.task_id)
            result = row["tool_result"]
            facts = (mcp_facts(result.producer_text or result.text) if name.startswith("mcp_")
                     else json.loads(result.text.split("\n\n⚠️ ", 1)[0]))
            check_permission_result(facts)
            return row, facts

        selection = {"app": f"pid:{target.pid}", "window": target.title}
        try:
            with owner(registry):
                idle_row, idle = invoke("after_connect", tool(FRAME), {})
                assert idle.get("reason") == "no_session" and not _images(idle_row, registry)
                server = capture_server(os.getpid(), entry)
                assert not pyobjc_mapped(server)
                server.record(record_property, "server")
                area, (x, y), window_spot = wait_for(lambda: overlap(target, occluder))
                record_property("overlap_points2", area)
                row, begun = invoke("begin", tool(BEGIN), selection)
                assert row["tool_result"].status == "ok" and begun.get("ok") and begun["state"] == "live"
                assert (begun["app"]["pid"], begun["window"]["window_id"]) == window_key(target)
                capture_only(begun)
                session = vars(registry._ctx)[mcp_task_session._SLOT].sessions[SERVER]
                mapper = _load_plugin()._ComputerUse(_API(skill_state_dir_path(drive, SKILL)))
                display = None
                if scene == "occluded":
                    assert mapper._capture_display().get("index"), "ordinary display geometry is unavailable"
                    row, display = invoke("display", shot, {"job_id": "stream-display",
                                                            "max_width": 4096, "max_height": 4096})
                    assert display.get("ok") and row["tool_result"].status == "ok"
                    path = pathlib.Path(display["path"])
                    import Quartz

                    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
                    display_spot = ((x - bounds.origin.x) / bounds.size.width,
                                    (y - bounds.origin.y) / bounds.size.height)
                    covered = _pixels_near(path, MAGENTA, around=display_spot)
                    record_property("display_overlap_pixels", covered)
                    assert covered >= 90
                    for label, rgb in (("target", GREEN), ("occluder", MAGENTA)):
                        count = _pixels_near(path, rgb)
                        record_property("display_" + label + "_pixels", count)
                        assert count > 1200
                previous_ring = mapper._load_transform()
                row, one = invoke("one_shot", shot, {"job_id": "stream-one-shot", **selection})
                assert row["tool_result"].status == "ok" and one.get("ok") and one["capture_kind"] == "window"
                assert (one["window"]["pid"], one["window"]["window_id"]) == window_key(target)
                _check_mapping(mapper, previous_ring, display, one)
                image, = _images(row, registry)
                assert image == pathlib.Path(one["path"]).read_bytes()
                pixels_of_target(image, GREEN, record_property, "one_shot")
                under_cover = _pixels_near(io.BytesIO(image), GREEN, around=window_spot)
                record_property("one_shot_overlap_pixels", under_cover)
                assert under_cover >= 90

                row, first = invoke("frame_green", tool(FRAME), {"wait_ms": 3000})
                assert first.get("ok") and first.get("image") is True and first["seq"] >= 1
                assert (first["window"]["pid"], first["window"]["window_id"]) == window_key(target)
                assert row["tool_result"].meta["mcp_image_owner"] == [registry._ctx.task_id, "1"]
                image, = _images(row, registry)
                pixels_of_target(image, GREEN, record_property, "stream")
                assert _pixels_near(io.BytesIO(image), GREEN, around=window_spot) >= 90
                _check_mapping(mapper, previous_ring, display, first)
                if display is not None:
                    record_property("display_mapping_preserved", 1)
                assert capture_server(os.getpid(), entry) == server
                assert pyobjc_mapped(server), "the positive frame must load the observed PyObjC binary"

                target.send("blue")
                cursor = first["seq"]

                def repaint():
                    nonlocal cursor
                    row, second = invoke(f"frame_after_{cursor}", tool(FRAME), {"after_seq": cursor, "wait_ms": 500})
                    assert second.get("ok")
                    if not second.get("image"):
                        return False
                    assert second["seq"] > cursor
                    cursor = second["seq"]
                    image, = _images(row, registry)
                    if _pixels_near(io.BytesIO(image), BLUE) <= 5000:
                        return False
                    assert _pixels_near(io.BytesIO(image), GREEN) < 100
                    _check_mapping(mapper, previous_ring, display, second)
                    return True

                wait_for(repaint)
                record_property("repaint_seq_delta", cursor - first["seq"])
                assert session is vars(registry._ctx)[mcp_task_session._SLOT].sessions[SERVER] and session.usable()
                row, released = invoke("release", tool(RELEASE), {})
                assert row["tool_result"].status == "ok" and released.get("released") is True
                assert released["cleanup"]["confirmed"] is True
                record_property("release_confirmed", 1)
                idle_row, idle = invoke("after_release", tool(FRAME), {})
                assert idle.get("reason") == "no_session" and not _images(idle_row, registry)
                row, foreign = invoke("foreign", tool(BEGIN), {"app": f"pid:{occluder.pid}", "window": target.title})
                assert row["tool_result"].status == "error" and foreign.get("reason") in {"window_not_found", "no_window"}

                _, begun = invoke("begin_before_destruction", tool(BEGIN), selection)
                assert begun.get("ok")
                row, live = invoke("frame_before_destruction", tool(FRAME), {"wait_ms": 3000})
                assert live.get("image") is True and live["state"] == "live"
                assert (live["window"]["pid"], live["window"]["window_id"]) == window_key(target)
                image, = _images(row, registry)
                pixels_of_target(image, BLUE, record_property, "before_close")

                def first_terminal_frame():
                    row, gone = invoke("destroyed_while_idle", tool(FRAME), {})
                    return gone, _images(row, registry)

                confirm_idle_window_stop(target, occluder, first_terminal_frame, record_property)
                assert session.usable() and capture_server(os.getpid(), entry) == server
                _, ended = invoke("release_destroyed", tool(RELEASE), {})
                assert ended.get("ok")
                for name, args in ((tool(BEGIN), selection), (shot, {"job_id": "closed", **selection})):
                    row, closed = invoke("closed_" + name, name, args)
                    assert closed.get("ok") is False and closed.get("reason") in {"window_not_found", "no_window"}
                    assert not _images(row, registry) and "path" not in closed

                _, begun = invoke("begin_before_transport_close", tool(BEGIN),
                                  {"app": f"pid:{occluder.pid}", "window": occluder.title})
                assert begun.get("ok")
                row, last = invoke("frame_before_transport_close", tool(FRAME), {"wait_ms": 3000})
                image, = _images(row, registry)
                assert last.get("image") is True and _pixels_near(io.BytesIO(image), MAGENTA) > 1200
                assert capture_server(os.getpid(), entry) == server

            mcp_task_session.close_context_sessions(registry._ctx, reason="native QA transport close")
            session._thread.join(timeout=5)
            assert session.closure == "closed" and session._transport_closed and not session._thread.is_alive()
            wait_for(server.gone)
            assert window_key(occluder) in window_rows()
            record_property("transport_server_gone", 1)
        finally:
            mcp_task_session.close_context_sessions(registry._ctx, reason="native QA fixture cleanup")


@desktop_only
def test_active_capture_loses_stdin(tmp_path, record_property):
    """Physical EOF while a stream is delivering, without calling release or stopping a task."""
    from base64 import b64decode

    from tests.test_unix_computer_use_window_session_server import _Process

    with owned_windows(tmp_path) as (target, occluder):
        server = _Process()
        try:
            server.send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                       "clientInfo": {"name": "window-qa", "version": "1"}})
            server.send("notifications/initialized", notify=True)
            assert [row["name"] for row in server.send("tools/list")["tools"]] == [BEGIN, FRAME, RELEASE]
            producer = capture_server(os.getpid(), REPO / "skills" / SKILL / "window_session_server.py")
            assert producer.pid == server.proc.pid and not pyobjc_mapped(producer)
            _, started = server.tool(BEGIN, {"app": f"pid:{target.pid}", "window": target.title})
            check_permission_result(started)
            assert started.get("ok") and started["state"] == "live"
            result, frame = server.tool(FRAME, {"wait_ms": 3000})
            assert frame.get("image") is True and (frame["window"]["pid"], frame["window"]["window_id"]) == window_key(target)
            text, image = result["content"]
            assert text["type"] == "text" and image["type"] == "image" and image["mimeType"] == "image/png"
            pixels_of_target(b64decode(image["data"], validate=True), GREEN, record_property, "stream")
            assert pyobjc_mapped(producer)
            producer.record(record_property, "server")
            server.proc.stdin.close()
            assert server.proc.wait(timeout=20) == 0
            wait_for(producer.gone)
            assert {window_key(target), window_key(occluder)} <= window_rows(on_screen=True).keys()
            record_property("eof_exit", server.proc.returncode)
        finally:
            if server.proc.poll() is None:
                server.proc.kill()  # only this Popen child, after the outcome was measured
                server.proc.wait(timeout=10)

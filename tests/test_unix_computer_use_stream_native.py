"""Opt-in macOS consumer qualification for the skill-bound window stream.

This test is intentionally inert unless OUROBOROS_MACOS_STREAM_QA=1 is passed
*after* scripts/safe_test.py's boundary. It creates only two synthetic AppKit
windows, does not activate an app or send input, and closes every owned process.
Run with the app's Python (MCP SDK and PyObjC are required), for example::

    PYTHON=/Applications/Ouroboros.app/Contents/Frameworks/python-standalone/bin/python3
    "$PYTHON" -I -S scripts/safe_test.py --temp-parent /private/var/tmp \
      --evidence-dir .review-evidence --timeout 900 -- env OUROBOROS_MACOS_STREAM_QA=1 \
      OUROBOROS_TEST_EVIDENCE_ROOT=.review-evidence "$PYTHON" -m pytest -q \
      tests/test_unix_computer_use_stream_native.py

Only the parent may run that command after announcing the visible windows. A
missing permission, no pixels, an unconfirmed release or a closed-window result
that is unknown fails with its receipts; it is never converted into a pass.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import pytest

from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import owner
from ouroboros.loop_tool_execution import _execute_with_timeout
from tests.test_mcp_result_images import _batch, _pixels


REPO = pathlib.Path(__file__).resolve().parents[1]
SKILL = "unix_computer_use"
OPT_IN = sys.platform == "darwin" and os.environ.get("OUROBOROS_MACOS_STREAM_QA") == "1"
desktop_only = pytest.mark.skipif(not OPT_IN, reason="needs an announced macOS desktop and OUROBOROS_MACOS_STREAM_QA=1")

# These names come from the builder's window_session_server.py TOOLS declaration.
# Assert them against the installed catalog before starting any window.
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
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .canJoinAllApplications]
        window.level = .floating // keep only these synthetic windows above a full-screen owner app
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


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise AssertionError(f"{self.role} did not log {event!r}; exit={self.proc.poll()}; "
                             f"stderr={self.stderr.read_text()[-800:]!r}; rows={_rows(self.log)[-5:]}")

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


def _pixels_near(path: pathlib.Path, rgb: tuple[int, int, int]) -> int:
    import io
    from PIL import Image, ImageCms

    with Image.open(path) as image:
        image.load()
        # The fixture paints explicit sRGB. Display PNGs carry the display's ICC
        # profile: comparing their encoded channels directly is not a color test.
        profile = image.info.get("icc_profile")
        sample = (ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                           ImageCms.createProfile("sRGB"), outputMode="RGB")
                  if profile else image.convert("RGB"))
        sample.thumbnail((1200, 800))
        return sum(1 for r, g, b in sample.getdata()
                   if max(abs(r - rgb[0]), abs(g - rgb[1]), abs(b - rgb[2])) < 35)


def _answer(result) -> dict:
    body = result.producer_text or result.text
    # The MCP host's trust label precedes the server's JSON text part. An image
    # part may follow it; JSONDecoder consumes only the first complete object.
    body = body.split("\n\n", 1)[-1]
    start = body.find("{")
    assert start >= 0, body[:1200]
    return json.JSONDecoder().raw_decode(body[start:])[0]


def _image_from_mcp(row: dict, dest: pathlib.Path, registry) -> None:
    result = row["tool_result"]
    assert result.meta.get("mcp_image_owner") == [registry._ctx.task_id, "1"], result
    messages, trace = _batch(SimpleNamespace(registry=registry), [row])
    images = _pixels(messages)
    assert len(images) == 1 and trace["tool_calls"][0]["image_attachment"]["status"] == "attached", trace
    assert images[0].startswith(b"\x89PNG\r\n\x1a\n")
    dest.write_bytes(images[0])


@desktop_only
@pytest.mark.parametrize("scene", ["occluded", "independent_lifecycle"])
def test_owned_occluded_window_through_connect_and_task_session(tmp_path, monkeypatch, scene):
    from ouroboros import config, extension_loader, mcp_client, mcp_task_session, skill_mcp
    from ouroboros.skill_builtin import _dependency_pass, install_builtin_skill
    from ouroboros.skill_loader import load_enabled
    from tests._shared import clean_extension_runtime_state

    assert mcp_client._MCP_SDK_AVAILABLE, "real MCP SDK is required"
    assert shutil.which("xcrun"), "Swift/AppKit fixture needs xcrun; no GUI was started"
    assert sys.platform == "darwin"
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    monkeypatch.setenv("OUROBOROS_AGENT_PYTHON", sys.executable)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "no-extra-skills"))
    drive = tmp_path / "private-data"
    drive.mkdir()
    settings_path = drive / "settings.json"
    settings_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)
    clean_extension_runtime_state()
    mcp_client.reset_manager_for_tests()

    evidence = tmp_path / "stream-evidence"
    evidence.mkdir()
    started: list[OwnedWindow] = []
    registry = None
    receipts = evidence / "receipts.jsonl"

    def record(label: str, row: dict | None = None, **facts) -> None:
        result = row.get("tool_result") if row else None
        item = {"label": label, "at": time.time(), **facts}
        if result is not None:
            item.update(status=result.status, code=result.code, text=result.text,
                        meta=result.meta, host_annotations=result.host_annotations)
        with receipts.open("a", encoding="utf-8") as out:
            out.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    try:
        installed = install_builtin_skill(SKILL, drive_root=drive, seed_dir=REPO / "skills",
                                          turn_on_when_ready=True, installed_by="owner")
        assert installed["ok"], installed
        source_skill, copied_skill = REPO / "skills" / SKILL, drive / "skills" / "native" / SKILL
        source_files = [path for path in source_skill.rglob("*") if path.is_file()
                        and "__pycache__" not in path.parts and path.suffix != ".pyc"]
        assert source_files and all((copied_skill / path.relative_to(source_skill)).read_bytes() == path.read_bytes()
                                    for path in source_files), "the private install differs from this checkout's skill"
        _dependency_pass(drive, [SKILL], logging.getLogger(__name__))
        assert load_enabled(drive, SKILL) is True, "the copied skill did not reach its real enabled state"
        preview = skill_mcp.preview_connect(drive, SKILL)
        record("preview", preview=preview)
        assert preview.get("ready") is True and preview.get("blockers") == [], preview
        assert len(preview["servers"]) == 1 and preview["servers"][0]["task_session"] is True, preview
        server_id = preview["servers"][0]["id"]
        assert server_id == "mac_window_session" and preview["servers"][0]["browser_policy"] is False, preview
        connected = skill_mcp.connect(drive, SKILL, expected_content_hash=preview["content_hash"],
                                      settings_digest=preview["settings_digest"])
        record("connect", connect=connected)
        assert connected.get("ok") and connected.get("saved"), connected
        settings = json.loads(settings_path.read_text())
        assert settings["MCP_ENABLED"] is True and len(settings["MCP_SERVERS"]) == 1
        saved = settings["MCP_SERVERS"][0]
        assert saved["skill"] == SKILL and saved["id"] == server_id
        assert not any(key in saved for key in ("command", "args", "cwd")), saved
        bound = skill_mcp.bind_config(mcp_client.parse_servers(settings["MCP_SERVERS"], settings=settings)[0], drive)
        assert bound.skill_error == "" and bound.task_session is True, bound
        assert bound.command == os.path.abspath(sys.executable) and bound.args[:2] == ["-I", "-B"]
        assert pathlib.Path(bound.args[2]) == copied_skill / "window_session_server.py" and bound.cwd == str(copied_skill)
        record("bound", command=bound.command, args=bound.args, revision=bound.skill_revision)
        manager = mcp_client.get_manager()
        manager.reconfigure(settings)
        listed = manager.refresh_server(server_id)
        record("listed", listing=listed)
        assert listed.get("ok"), listed
        for name, required in ((BEGIN, {"app", "window"}), (FRAME, {"after_seq", "wait_ms"}), (RELEASE, set())):
            qualified = f"mcp_{server_id}__{name}"
            resolution = manager.resolve_tool_name(qualified)
            assert resolution.status == "callable", (name, resolution)
            published = manager.get_tool(qualified)
            assert published is not None, qualified
            schema = published["schema"]
            assert required <= set(schema.get("properties", {})), (name, schema)
        assert extension_loader.get_tool(extension_loader.extension_surface_name(SKILL, "screenshot")) is not None

        swift = tmp_path / "StreamFixture.swift"
        swift.write_text(FIXTURE, encoding="utf-8")
        binary = tmp_path / "StreamFixture"
        build = subprocess.run(["xcrun", "swiftc", "-swift-version", "5", "-Onone", "-o", str(binary), str(swift)],
                               capture_output=True, text=True, timeout=300, check=False)
        record("fixture_build", returncode=build.returncode, stderr=build.stderr[-3000:])
        assert build.returncode == 0, build.stderr
        target = OwnedWindow(binary, "target", evidence)
        started.append(target)
        occluder = OwnedWindow(binary, "occluder", evidence)
        started.append(occluder)
        assert target.pid != occluder.pid and target.title != occluder.title
        time.sleep(0.25)  # both AppKit flushes preceded ready; allow WindowServer to composite their new layers

        registry, _, _ = _registry(drive, monkeypatch)
        registry._ctx.repo_dir = REPO
        registry._ctx.task_metadata = {"work_kind": "owner"}
        tool = lambda name: f"mcp_{server_id}__{name}"
        shot = extension_loader.extension_surface_name(SKILL, "screenshot")

        with owner(registry):
            def invoke(label: str, name: str, args: dict, timeout: int = 90) -> tuple[dict, dict]:
                began = time.monotonic()
                row = _execute_with_timeout(
                    registry, {"id": label, "function": {"name": name, "arguments": json.dumps(args)}},
                    registry._ctx.drive_root / "logs", timeout, task_id=registry._ctx.task_id)
                record(label, row, elapsed_ms=round((time.monotonic() - began) * 1000, 2))
                return row, _answer(row["tool_result"]) if name.startswith("mcp_") else json.loads(
                    row["tool_result"].text.split("\n\n⚠️ ", 1)[0])

            record("scene", scene=scene, occlusion_oracle=scene == "occluded")
            if scene == "occluded":
                default_row, default = invoke("display", shot, {"job_id": "stream-display", "max_width": 4096,
                                                               "max_height": 4096})
                assert default.get("ok"), default
                display_file = evidence / "display-with-occluder.png"
                shutil.copy2(default["path"], display_file)
                assert _pixels_near(display_file, (229, 41, 163)) > 1200, "display did not show the owned occluder"
                assert _pixels_near(display_file, (24, 179, 74)) > 1200, "display did not show exposed target pixels"
                assert default_row["tool_result"].status == "ok"

            one_row, one = invoke("one_shot", shot, {"job_id": "stream-one-shot", "app": f"pid:{target.pid}",
                                                       "window": target.title})
            assert one.get("ok") and one.get("capture_kind") == "window", one
            assert one["window"]["window_id"] == target.ready["window_id"] and one["window"]["pid"] == target.pid, one
            assert one.get("mapped_input_available") is False and not any(
                key in one for key in ("observation_id", "coord_transform", "display")), one
            one_file = evidence / "one-shot-covered-target.png"
            shutil.copy2(one["path"], one_file)
            one_messages, one_trace = _batch(SimpleNamespace(registry=registry), [one_row])
            assert _pixels(one_messages) == [one_file.read_bytes()], one_trace
            assert one_trace["tool_calls"][0]["image_attachment"]["status"] == "attached", one_trace
            assert _pixels_near(one_file, (24, 179, 74)) > 5000, "PluginAPI one-shot did not capture target pixels"
            assert _pixels_near(one_file, (229, 41, 163)) < 100, "one-shot included the occluder"
            assert one_row["tool_result"].status == "ok"

            begun_row, begun = invoke("begin", tool(BEGIN), {"app": f"pid:{target.pid}", "window": target.title})
            assert begun_row["tool_result"].status == "ok" and begun.get("ok") and begun["state"] == "live", begun
            assert begun["app"]["pid"] == target.pid and begun["window"]["window_id"] == target.ready["window_id"], begun
            assert begun.get("mapped_input_available") is False and not any(
                key in begun for key in ("observation_id", "coord_transform")), begun
            first_session = vars(registry._ctx)[mcp_task_session._SLOT].sessions[server_id]
            assert first_session.usable()
            server_entry = str(copied_skill / "window_session_server.py")
            census = subprocess.check_output(["ps", "-axo", "pid=,ppid=,command="], text=True)
            rows = [line.strip().split(None, 2) for line in census.splitlines()]
            processes = {int(row[0]): (int(row[1]), row[2]) for row in rows if len(row) == 3}
            descendants = {os.getpid()}
            while True:
                found = {pid for pid, (parent, _cmd) in processes.items() if parent in descendants}
                if found <= descendants:
                    break
                descendants |= found
            servers = [pid for pid in descendants if server_entry in processes.get(pid, (0, ""))[1]]
            assert len(servers) == 1, servers
            server_pid = servers[0]
            def server_alive():
                current = subprocess.run(["ps", "-p", str(server_pid), "-o", "command="],
                                         capture_output=True, text=True, check=False)
                assert current.returncode in (0, 1), current.stderr
                return server_entry in current.stdout
            record("server_process", pid=server_pid)
            first_row, first = invoke("frame_green", tool(FRAME), {"wait_ms": 3000})
            assert first.get("ok") and first.get("image") is True and first["seq"] >= 1, first
            assert first["window"]["window_id"] == target.ready["window_id"], first
            first_file = evidence / "stream-covered-target-green.png"
            _image_from_mcp(first_row, first_file, registry)
            assert _pixels_near(first_file, (24, 179, 74)) > 5000, "stream frame did not show covered target"
            assert _pixels_near(first_file, (229, 41, 163)) < 100, "stream frame included occluder"

            target.send("blue")
            # after_seq means a newer captured frame, not a frame caused by our
            # action. A previously queued green frame can legitimately arrive first.
            deadline, cursor, attempt = time.monotonic() + 4, first["seq"], 0
            second_file = evidence / "stream-covered-target-blue.png"
            while True:
                attempt += 1
                second_row, second = invoke(f"frame_blue_{attempt}", tool(FRAME),
                                            {"after_seq": cursor, "wait_ms": 500})
                assert second.get("ok"), second
                if second.get("image"):
                    assert second["seq"] > cursor, second
                    cursor = second["seq"]
                    observed = evidence / f"repaint-observation-{attempt}.png"
                    _image_from_mcp(second_row, observed, registry)
                    if _pixels_near(observed, (37, 99, 235)) > 5000:
                        shutil.copy2(observed, second_file)
                        break
                assert time.monotonic() < deadline, "no captured frame showed the target repaint within four seconds"
            assert _pixels_near(second_file, (24, 179, 74)) < 100, "the earlier green image was still served"
            assert second.get("mapped_input_available") is False and not any(
                key in second for key in ("observation_id", "coord_transform")), second
            session = vars(registry._ctx)[mcp_task_session._SLOT].sessions[server_id]
            assert session is first_session and session.usable(), "the two frame calls did not share one task session"

            released_row, released = invoke("release", tool(RELEASE), {})
            assert (released_row["tool_result"].status == "ok" and released.get("ok")
                    and released.get("released") and released.get("cleanup", {}).get("confirmed") is True), released
            assert target.proc.poll() is None and target.ready["visible"], "foreign-app check needs a live target window"
            foreign_row, foreign = invoke("foreign", tool(BEGIN), {"app": f"pid:{occluder.pid}",
                                                                "window": target.title})
            assert foreign_row["tool_result"].status == "error" and foreign.get("ok") is False, foreign
            assert foreign.get("reason") in {"window_not_found", "no_window"}, foreign
            _, live_again = invoke("begin_before_destruction", tool(BEGIN),
                                   {"app": f"pid:{target.pid}", "window": target.title})
            assert live_again.get("ok"), live_again
            closed_mark = target.send("close", event="closed")
            assert (closed_mark["window_id"] == target.ready["window_id"]
                    and closed_mark["visible"] is False and target.proc.poll() is None), closed_mark
            target.wait("deallocated", timeout=4)
            # AppKit's close acknowledgement precedes WindowServer retiring the
            # surface. Establish that boundary independently before judging capture.
            import Quartz
            deadline = time.monotonic() + 4
            while True:
                listed = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, 0)
                remaining = [dict(row) for row in listed
                             if int(row.get("kCGWindowNumber", -1)) == closed_mark["window_id"]
                             and int(row.get("kCGWindowOwnerPID", -1)) == target.pid]
                if not remaining:
                    break
                assert time.monotonic() < deadline, ("fixture surface still exists after close", remaining)
                time.sleep(0.05)
            record("window_server_destroyed", pid=target.pid, window_id=closed_mark["window_id"])
            gone_row, gone = invoke("destroyed_while_live", tool(FRAME), {"wait_ms": 1000})
            assert gone.get("ok") is False and gone.get("image") is False, gone
            assert gone.get("reason") in {"window_gone", "stream_stopped", "frame_withheld"}, gone
            _, ended = invoke("release_destroyed", tool(RELEASE), {})
            assert ended.get("ok") and ended.get("cleanup", {}).get("confirmed"), ended
            closed_row, closed = invoke("closed", tool(BEGIN), {"app": f"pid:{target.pid}",
                                                              "window": target.title})
            assert closed_row["tool_result"].status == "error" and closed.get("ok") is False, closed
            assert closed.get("reason") in {"window_not_found", "no_window"}, closed
            assert target.proc.poll() is None, "the closed-window check needs a live target process"
            assert session.usable(), "a refused begin unexpectedly ended the task session"
            _, still_capturing = invoke("begin_before_task_end", tool(BEGIN),
                                        {"app": f"pid:{occluder.pid}", "window": occluder.title})
            assert still_capturing.get("ok"), still_capturing
            _, last_frame = invoke("frame_before_task_end", tool(FRAME), {"wait_ms": 3000})
            assert last_frame.get("ok") and last_frame.get("image"), last_frame
            assert server_alive(), "the task-end witness needs a live server"

        mcp_task_session.close_context_sessions(registry._ctx, reason="native stream QA ended")
        # TaskSession sets _transport_closed only after the SDK stdio context has
        # closed its pipes and joined the child. It does not expose that child's PID.
        session._thread.join(timeout=5)  # transport close may precede its worker thread's final return
        assert session.closure == "closed" and session._transport_closed and not session._thread.is_alive(), (
            session.closure, session._transport_closed)
        assert not server_alive(), "task-session close left its capture process alive"
        record("teardown", server_pid=server_pid, server_exit_confirmed=True,
               active_stream_at_task_end=True, session_closure=session.closure, sdk_stdio_transport_closed=session._transport_closed,
               images={path.name: _sha(path) for path in evidence.glob("*.png")})
    finally:
        if registry is not None:
            mcp_task_session.close_context_sessions(registry._ctx, reason="native stream QA cleanup")
        for child in reversed(started):
            child.stop()
        record("owned_processes_exit", children=[{"role": child.role, "pid": child.pid,
                                                   "returncode": child.proc.poll()} for child in started])
        assert all(child.proc.poll() is not None for child in started)
        clean_extension_runtime_state()
        mcp_client.reset_manager_for_tests()
        root = os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT")
        if root:
            destination = pathlib.Path(root).resolve() / "macos-stream" / uuid.uuid4().hex
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(evidence, destination)

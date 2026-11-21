"""Read-only witnesses used by the opt-in macOS window tests. Runs outside the task."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest


def require_capture_grant():
    import Quartz

    if not Quartz.CGPreflightScreenCaptureAccess():
        pytest.skip("NOT_RUN_SCREEN_RECORDING")
    if not desktop_available():
        pytest.skip("NOT_RUN_DESKTOP")


def desktop_available():
    import Quartz

    major = platform.mac_ver()[0].split(".")[0]
    display = Quartz.CGMainDisplayID()
    return (major.isdigit() and int(major) >= 14 and bool(display)
            and bool(Quartz.CGDisplayIsActive(display)) and Quartz.CGSessionCopyCurrentDictionary() is not None)


def check_permission_result(facts):
    if facts.get("reason") == "screen_recording_denied":
        pytest.skip("NOT_RUN_SCREEN_RECORDING")


def window_rows(on_screen=False):
    import Quartz

    option = Quartz.kCGWindowListOptionOnScreenOnly if on_screen else Quartz.kCGWindowListOptionAll
    rows = Quartz.CGWindowListCopyWindowInfo(option, Quartz.kCGNullWindowID)
    assert rows, "WindowServer did not supply a window list"
    return {(int(row["kCGWindowOwnerPID"]), int(row["kCGWindowNumber"])): row for row in rows}


def window_key(window):
    assert window.proc.poll() is None, "fixture application exited"
    return window.pid, int(window.ready["window_id"])


def overlap(target, occluder):
    # On-screen rows are front to back. Both windows must be present, with the
    # smaller one in front and part of the larger window still exposed.
    rows = window_rows(on_screen=True)
    back, front = window_key(target), window_key(occluder)
    if back not in rows or front not in rows or list(rows).index(front) >= list(rows).index(back):
        return None
    a, b = rows[back]["kCGWindowBounds"], rows[front]["kCGWindowBounds"]
    left, top = max(a["X"], b["X"]), max(a["Y"], b["Y"])
    w = min(a["X"] + a["Width"], b["X"] + b["Width"]) - left
    h = min(a["Y"] + a["Height"], b["Y"] + b["Height"]) - top
    area = int(max(0, w) * max(0, h))
    if not (0 < area < a["Width"] * a["Height"]):
        return None
    x, y = left + w / 2, top + h / 2
    return area, (x, y), ((x - a["X"]) / a["Width"], (y - a["Y"]) / a["Height"])


def wait_for(read, *, seconds=10):
    deadline = time.monotonic() + seconds
    while True:
        result = read()
        if result:
            return result
        if time.monotonic() >= deadline:
            raise AssertionError("observation deadline expired")
        time.sleep(0.05)


def confirm_idle_window_stop(target, occluder, first_terminal_frame, record_property):
    """Date the logical stop on the original stream, then require its cleanup receipt.

    stop.age_ms dates the first terminal decision. cleanup.confirmed says the
    native stop was acknowledged by the time of the reply. The receipt does
    not independently date the native callback or the stopCapture completion.
    """
    key, survivor = window_key(target), window_key(occluder)
    assert {key, survivor} <= window_rows().keys()
    closing = time.monotonic_ns()
    closed = target.send("close", event="closed")
    assert closed["window_id"] == key[1] and closed["visible"] is False
    target.wait("deallocated")

    def retired():
        rows = window_rows()
        assert window_key(target) == key and window_key(occluder) == survivor and survivor in rows
        return key not in rows

    wait_for(retired)
    missing = time.monotonic_ns()
    time.sleep(12)  # several two-second polls, with room for a full worker/model round trip
    assert retired()
    requested = time.monotonic_ns()
    facts, images = first_terminal_frame()
    received = time.monotonic_ns()

    assert facts.get("ok") is False and facts.get("image") is False
    assert len(images) == 0, "the terminal frame attached an image"
    assert facts.get("state") == "stopped"
    assert facts.get("reason") in {"window_gone", "stream_stopped"}
    assert facts.get("cleanup", {}).get("confirmed") is True
    assert (facts["window"]["pid"], facts["window"]["window_id"]) == key and retired()

    age = facts["stop"]["age_ms"]
    assert type(age) in (int, float) and 0 <= age <= (received - closing) / 1_000_000 + 1
    idle_ms = (requested - missing) // 1_000_000
    roundtrip_ms = math.ceil((received - requested) / 1_000_000)
    record_property("idle_after_loss_ms", idle_ms)
    record_property("terminal_roundtrip_ms", roundtrip_ms)
    record_property("stop_age_ms", int(age))
    # Even if the server sampled age at the very end of the round trip,
    # with an extra ms for rounding, its stop must predate the first request.
    assert idle_ms > 2000 and int(age) > roundtrip_ms + 1
    record_property("window_stop_confirmed", 1)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    seconds: int
    microseconds: int

    @classmethod
    def live(cls, pid):
        from ouroboros.platform_darwin_abi import kernel_birth

        observed = kernel_birth(pid)
        assert observed and not observed[2], "no live kernel birth record"
        return cls(pid, observed[0], observed[1])

    def gone(self):
        from ouroboros.platform_darwin_abi import kernel_birth
        from ouroboros.platform_layer import pid_provably_gone

        birth = kernel_birth(self.pid)
        if birth is None:
            return pid_provably_gone(self.pid)  # ESRCH, never an unreadable command line
        return birth[:2] != (self.seconds, self.microseconds)

    def record(self, record_property, name):
        record_property(name + "_pid", self.pid)
        record_property(name + "_birth_s", self.seconds)
        record_property(name + "_birth_us", self.microseconds)


def capture_server(parent_pid, installed_entry):
    """Identify the unique installed entry among this carrier's descendants.

    Check the complete isolated Python invocation read from the kernel. The
    entry imports macos_stream.py by its own resolved path. Its installed
    payload is separately compared with the checkout before any capture.
    """
    from ouroboros.platform_layer import collect_descendant_pids, process_command

    entry = Path(installed_entry).resolve(strict=True)
    suffix = " -I -B " + str(entry)
    matches = []
    for pid in collect_descendant_pids(parent_pid):
        cmd = process_command(pid)
        if cmd.endswith(suffix):
            identity = ProcessIdentity.live(pid)
            cmd = process_command(pid)
            assert cmd.endswith(suffix)
            executable = Path(cmd[:-len(suffix)])
            assert executable.is_absolute() and executable.is_file()
            # A framework build can exec Python.app. Compare with the kernel's
            # spelling for our own interpreter as well as the configured venv.
            assert (os.path.samefile(executable, sys.executable)
                    or process_command(os.getpid()).startswith(str(executable) + " "))
            assert ProcessIdentity.live(pid) == identity
            matches.append(identity)
    assert len(matches) == 1, "expected exactly one live installed capture server under this carrier"
    return matches[0]


def pyobjc_mapped(process):
    """Look for the actual PyObjC extension in this one child's mapped files.

    The capture bridge imports it before calling ScreenCaptureKit. A later
    positive frame supplies the positive control for this same observer.
    """
    from objc import _objc

    extension = Path(_objc.__file__).resolve(strict=True)
    assert ProcessIdentity.live(process.pid) == process
    listing = subprocess.run(["/usr/sbin/lsof", "-nP", "-a", "-p", str(process.pid), "-d", "txt", "-Fn"],
                             capture_output=True, text=True, timeout=15, check=True)
    paths = [Path(line[1:]) for line in listing.stdout.splitlines() if line.startswith("n/")]
    assert paths, "no mapped-file evidence for the selected process"
    assert ProcessIdentity.live(process.pid) == process
    return any(p.name == extension.name and p.resolve() == extension for p in paths)

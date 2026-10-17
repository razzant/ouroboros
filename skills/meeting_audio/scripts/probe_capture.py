"""Probe: which process macOS attributes the meeting-capture permission to.

macOS grants "Screen & System Audio Recording" to a responsible process, not to a
Python script: when the host spawns this skill's companion that is normally the
packaged Ouroboros app; run from a terminal it is the terminal; a bare Python
launched some other way may be Python itself. Without prompting, this probe
records the process chain, the responsible process
(``responsibility_get_pid_responsible_for_pid``) and whether consent is already
given (``CGPreflightScreenCaptureAccess``). The companion records the same facts
at its start (``capture_attribution.json`` in the state directory), so the answer
for the real host-spawned context exists without anyone running this by hand.

``--capture SECONDS`` also performs a real capture of ``--app``; when consent is
missing that shows the macOS prompt, so only the owner runs it. Without it the
real capture is reported as ``NOT_RUN``.

    python3 scripts/probe_capture.py [--state-dir DIR] [--capture SECONDS --app NAME]
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

RECORD_FILE = "capture_attribution.json"
MAX_CHAIN = 12


def process_path(pid: int) -> str:
    try:
        libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        buffer = ctypes.create_string_buffer(4096)
        if libproc.proc_pidpath(ctypes.c_int(pid), buffer, ctypes.c_uint32(len(buffer))) > 0:
            return buffer.value.decode("utf-8", errors="replace")
    except (OSError, AttributeError):
        pass
    return ""


def parent_pid(pid: int, run: Callable[..., Any] = subprocess.run) -> int:
    try:
        result = run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
        return int((result.stdout or "0").strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def responsible_pid(pid: int) -> int:
    try:
        function = ctypes.CDLL(None).responsibility_get_pid_responsible_for_pid
    except (OSError, AttributeError):
        return 0
    function.argtypes = [ctypes.c_int]
    function.restype = ctypes.c_int
    return int(function(pid))


def app_bundle(path: str) -> str:
    marker = ".app/Contents/"
    return path[: path.index(marker) + len(".app")] if marker in path else ""


def classify(path: str) -> str:
    """app_bundle | python | other: what macOS will name in its permission prompt."""
    if app_bundle(path):
        return "app_bundle"
    return "python" if pathlib.Path(path).name.lower().startswith("python") else "other"


def preflight(access: Any = None) -> Optional[bool]:
    try:
        if access is None:
            from capture_macos import ScreenCaptureAccess

            access = ScreenCaptureAccess()
        return access.granted()
    except (OSError, AttributeError, ImportError):
        return None


def attribution(pid: Optional[int] = None, *, path_of: Callable[[int], str] = process_path,
                parent_of: Callable[[int], int] = parent_pid,
                responsible_of: Callable[[int], int] = responsible_pid,
                granted: Callable[[], Optional[bool]] = preflight) -> Dict[str, Any]:
    """The non-prompting attribution facts for ``pid`` (default: this process)."""
    pid = int(pid or os.getpid())
    chain: List[Dict[str, Any]] = []
    current = pid
    while current > 0 and len(chain) < MAX_CHAIN:
        path = path_of(current)
        chain.append({"pid": current, "path": path, "app_bundle": app_bundle(path)})
        current = parent_of(current) if current != 1 else 0
    owner = responsible_of(pid)
    owner_path = path_of(owner) if owner > 0 else ""
    return {
        "probe": "capture_attribution", "platform": sys.platform, "pid": pid,
        "python": sys.executable, "chain": chain,
        "responsible": {"pid": owner, "path": owner_path, "app_bundle": app_bundle(owner_path),
                        "kind": classify(owner_path) if owner_path else "unknown"},
        "preflight_granted": granted(),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_record(state_dir: pathlib.Path, record: Dict[str, Any]) -> pathlib.Path:
    path = pathlib.Path(state_dir) / RECORD_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def real_capture(seconds: float, app: str, state_dir: pathlib.Path) -> Dict[str, Any]:
    """Owner-run only: capture ``app`` for ``seconds`` and report whether audio arrived."""
    from capture_common import CaptureFailed
    from capture_macos import MacBackend
    from meeting_apps import TargetRefused, choose

    backend = MacBackend(state_dir)
    received = {"bytes": 0}
    errors: List[str] = []
    stop = threading.Event()
    try:
        state, message = backend.permission()
        if state != "granted":
            backend.request_permission()
            return {"real_capture": "FAIL", "reason": message}
        capture = backend.open_capture(choose(backend.running_apps(), app))
        capture.start(lambda pcm, stream=0: received.__setitem__("bytes", received["bytes"] + len(pcm)),
                      errors.append)
        threading.Thread(target=lambda: (time.sleep(seconds), stop.set()), daemon=True).start()
        backend.run_main_loop(stop)
        capture.stop()
    except (CaptureFailed, TargetRefused) as exc:
        return {"real_capture": "FAIL", "reason": getattr(exc, "message", str(exc))}
    passed = received["bytes"] > 0 and not errors
    return {"real_capture": "PASS" if passed else "FAIL", "pcm_bytes": received["bytes"], "errors": errors,
            "note": "PASS means audio buffers arrived; silence in the meeting still counts as audio"}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--state-dir", default=os.environ.get("OUROBOROS_SKILL_STATE_DIR", ""))
    parser.add_argument("--capture", type=float, default=0.0,
                        help="seconds of real capture (shows the macOS prompt when consent is missing)")
    parser.add_argument("--app", default="", help="application to capture with --capture")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        print(json.dumps({"probe": "capture_attribution", "platform": sys.platform,
                          "real_capture": "NOT_RUN", "reason": "macOS only"}))
        return 0
    record = attribution()
    if args.capture > 0:
        record.update(real_capture(args.capture, args.app, pathlib.Path(args.state_dir or ".")))
    else:
        record["real_capture"] = "NOT_RUN"
    if args.state_dir:
        write_record(pathlib.Path(args.state_dir), record)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

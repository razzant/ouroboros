"""Windows host facilities: path-free Git absence check, the app's own windows and input, and Git trace sinks.

Contracts:
- Destructive system-Git uninstallation and native input run only on a disposable GitHub-hosted runner.
- ``FileNotFoundError`` (WinError 2/3) is absence; any other read failure raises
  ``Unread`` so the caller settles as ``unconfirmed`` — never folded into empty/absent.
- ``git_lookup_report`` inspects directories via ``os.scandir``, ``shutil.which``, and
  ``System32\\where.exe`` without executing any untrusted bare-name binary probe
  (never runs ``git --version``) and never emits host ``PATH`` or filesystem paths.
- Zero system-wide auditing (no ``auditpol``, no ``wevtutil``, no Security 4688/4689),
  no process census and zero screen captures.
- Windows are the launched program's own: a top-level window counts only when its process
  is that PID, and no other window's title is read. Native pointer/keyboard input is
  desktop-wide, so it also needs that window in the foreground and under the pointer.
- Git trace sinks are read whole, Trace1 before Trace2, and only interpreted in
  ``git_traces``; raw lines, ``argv`` and host paths never leave the runner.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

SYSTEM32 = pathlib.Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
WHERE_EXIT_FOUND_NOT_FOUND = (0, 1)


class Unread(Exception):
    """A link required by a host, window, or Trace2 claim could not be read."""

    def __init__(self, code: str, /, **fields: Any) -> None:
        super().__init__(code)
        self.code = code
        self.fields = fields


def sha256_file(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm(path: str | pathlib.Path) -> str:
    text = str(path or "").strip().strip('"')
    if text.startswith("\\\\?\\"):
        text = text[4:]
    return text.replace("/", "\\").rstrip("\\").lower()


def inside(path: str | pathlib.Path, root: str | pathlib.Path) -> bool:
    p, r = norm(path), norm(root)
    return bool(p and r) and (p == r or p.startswith(r + "\\"))


def is_windows() -> bool:
    return sys.platform == "win32"


def disposable_hosted_runner() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted"


def _plain(value: Any, limit: int = 40) -> str:
    return re.sub(r"[^0-9A-Za-z._ -]", "?", str(value or ""))[:limit]


def host_image() -> dict[str, Any]:
    version = getattr(sys, "getwindowsversion", None)
    return {
        "image_os": _plain(os.environ.get("ImageOS")),
        "image_version": _plain(os.environ.get("ImageVersion")),
        "windows_build": int(version().build) if version is not None else None,
    }


# ---------------------------------------------------------------------------
# 1. System Git uninstallation & path-free absence verification (no bare-name exec)
# ---------------------------------------------------------------------------

REGISTRY_PATH_KEYS = (
    ("machine", "HKLM", r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ("user", "HKCU", r"Environment"),
)
GIT_INSTALL_KEYS = (
    ("HKLM", r"SOFTWARE\GitForWindows"),
    ("HKLM", r"SOFTWARE\WOW6432Node\GitForWindows"),
    ("HKCU", r"SOFTWARE\GitForWindows"),
    ("HKCU", r"SOFTWARE\WOW6432Node\GitForWindows"),
)


def _hive(name: str) -> Any:
    import winreg

    return {"HKLM": winreg.HKEY_LOCAL_MACHINE, "HKCU": winreg.HKEY_CURRENT_USER}[name]


def read_registry_string(hive: str, key: str, name: str) -> str | None:
    import winreg

    try:
        with winreg.OpenKey(_hive(hive), key) as handle:
            value, kind = winreg.QueryValueEx(handle, name)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise Unread(
            "registry_unreadable",
            hive=hive,
            problem=type(exc).__name__,
            winerror=getattr(exc, "winerror", None),
        ) from exc
    if kind not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) or not isinstance(value, str):
        raise Unread("registry_value_not_a_string", hive=hive, kind=kind)
    return value


def registry_path_dirs(reader: Callable[[str, str, str], str | None] = read_registry_string) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    problems: list[dict[str, Any]] = []
    for label, hive, key in REGISTRY_PATH_KEYS:
        try:
            raw = reader(hive, key, "Path")
        except Unread as unread:
            problems.append({"which": label, "code": unread.code, **unread.fields})
            continue
        entries = [os.path.expandvars(part.strip()) for part in (raw or "").split(";") if part.strip()]
        if unexpanded := sum(1 for entry in entries if "%" in entry):
            problems.append({"which": label, "code": "path_entry_unexpanded", "entries": unexpanded})
        out[label] = entries
    if problems:
        raise Unread("registry_path_unreadable", links=problems)
    return out


def git_for_windows_install_paths(reader: Callable[[str, str, str], str | None] = read_registry_string) -> list[str]:
    found: list[str] = []
    problems: list[dict[str, Any]] = []
    for hive, key in GIT_INSTALL_KEYS:
        try:
            value = reader(hive, key, "InstallPath")
        except Unread as unread:
            problems.append({"code": unread.code, **unread.fields})
            continue
        if value is not None:
            found.append(value)
    if problems:
        raise Unread("git_install_registry_unreadable", links=problems)
    return sorted(set(found))


def runner_git_installation() -> pathlib.Path | None:
    root = pathlib.Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git"
    registered = any(norm(path) == norm(root) for path in git_for_windows_install_paths())
    return root if registered and (root / "cmd" / "git.exe").is_file() else None


def uninstall_runner_git(root: pathlib.Path) -> dict[str, Any]:
    uninstaller = root / "unins000.exe"
    if not uninstaller.is_file():
        return {"done": False, "exit_code": None, "problem": "no_uninstaller_beside_the_installation"}
    try:
        done = subprocess.run(
            [str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
            capture_output=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return {"done": False, "exit_code": None, "problem": "uninstaller_timed_out"}

    def present() -> bool:
        return (root / "cmd" / "git.exe").exists() or any(norm(p) == norm(root) for p in git_for_windows_install_paths())

    deadline = time.monotonic() + 600
    while time.monotonic() < deadline and present():
        time.sleep(5)
    gone = not present()
    return {"done": gone, "exit_code": done.returncode, "problem": None if gone else "git_still_installed_after_uninstall"}


def default_git_dirs() -> list[str]:
    local = os.environ.get("LOCALAPPDATA", "")
    if not local:
        raise Unread("environment_variable_missing", name="LOCALAPPDATA")
    roots = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", ""),
        os.path.join(local, "Programs"),
    ]
    leaves = ("cmd", "bin", os.path.join("mingw64", "bin"), os.path.join("usr", "bin"))
    return [os.path.join(root, "Git", leaf) for root in roots if root for leaf in leaves]


def candidate_tool_dirs(candidate_src: pathlib.Path, python: str, env: dict[str, str]) -> list[str]:
    code = (
        "import json,sys; sys.path.insert(0, sys.argv[1]); "
        "from ouroboros.platform_layer import common_tool_directories as c; print(json.dumps(c('')))"
    )
    try:
        done = subprocess.run(
            [python, "-I", "-B", "-c", code, str(candidate_src)],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Unread("candidate_tool_dirs_unreadable", problem=type(exc).__name__) from exc
    if done.returncode != 0:
        raise Unread("candidate_tool_dirs_unreadable", exit_code=done.returncode)
    try:
        folders = json.loads(done.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise Unread("candidate_tool_dirs_output_unreadable") from exc
    if not isinstance(folders, list) or not all(isinstance(folder, str) for folder in folders):
        raise Unread("candidate_tool_dirs_output_unreadable")
    return folders


def lookup_dirs(launch_path: list[str], tool_dirs: list[str], app_dir: pathlib.Path) -> list[tuple[str, str]]:
    windows = os.environ.get("SystemRoot", r"C:\Windows")
    registry = registry_path_dirs()
    labelled = [
        (str(app_dir), "app"),
        (str(SYSTEM32), "system"),
        (os.path.join(windows, "System"), "system"),
        (windows, "system"),
        *((item, "launch_path") for item in launch_path),
        *((item, "registry_machine") for item in registry["machine"]),
        *((item, "registry_user") for item in registry["user"]),
        *((item, "candidate_tools") for item in tool_dirs),
        *((item, "git_default") for item in default_git_dirs()),
    ]
    for install in git_for_windows_install_paths():
        labelled += [(os.path.join(install, leaf), "git_registered") for leaf in ("cmd", "bin", os.path.join("mingw64", "bin"))]
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for item, source in labelled:
        if item and norm(item) not in seen:
            seen.add(norm(item))
            out.append((item, source))
    return out


def _probe_dir_for_git(directory: str) -> tuple[int, str]:
    exts = {e.lower() for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";") if e} | {".exe", ""}
    names = {"git" + ext for ext in exts}
    try:
        with os.scandir(directory) as entries:
            return sum(1 for e in entries if e.name.lower() in names and e.is_file()), "read"
    except (FileNotFoundError, NotADirectoryError):
        return 0, "absent"
    except OSError as exc:
        return 0, f"unreadable:{type(exc).__name__}"


def git_lookup_report(
    dirs: list[tuple[str, str]],
    package_root: pathlib.Path,
    launch_env: dict[str, str],
    cwd: pathlib.Path,
    *,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    which_fn: Callable[..., str | None] = shutil.which,
) -> dict[str, Any]:
    """Verify zero host ``git`` across lookup dirs without invoking any bare-name program or leaking paths."""
    by_source: dict[str, Counter[str]] = {}
    git_hit_count = unreadable_count = checked_dirs = 0
    non_package_dirs: list[str] = []
    for directory, source in dirs:
        if inside(directory, package_root):
            continue
        non_package_dirs.append(directory)
        checked_dirs += 1
        hits, state = _probe_dir_for_git(directory)
        bucket = by_source.setdefault(source, Counter())
        bucket[state.split(":", 1)[0]] += 1
        if hits:
            bucket["git_hits"] += hits
            git_hit_count += hits
        if state.startswith("unreadable"):
            unreadable_count += 1

    which_hit = which_fn("git", path=os.pathsep.join(non_package_dirs)) is not None
    if sys.platform == "win32":
        where_exe = SYSTEM32 / "where.exe"
        if not where_exe.is_file():
            raise Unread("where_exe_missing")
        try:
            where = run_fn(
                [str(where_exe), "git"],
                capture_output=True,
                text=True,
                env=launch_env,
                cwd=str(cwd),
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise Unread("where_exe_not_run", problem=type(exc).__name__) from exc
        if where.returncode not in WHERE_EXIT_FOUND_NOT_FOUND:
            raise Unread("where_exe_failed", exit_code=where.returncode)
        where_found = where.returncode == 0
    else:
        where_found = False

    return {
        "checked_dir_count": checked_dirs,
        "by_source": {src: dict(counts) for src, counts in sorted(by_source.items())},
        "git_hit_count": git_hit_count,
        "unreadable_dir_count": unreadable_count,
        "shutil_which_found": which_hit,
        "where_found": where_found,
        "bare_name_execution_probed": False,
        "absent": git_hit_count == 0 and unreadable_count == 0 and not which_hit and not where_found,
    }


def webview2_runtime(reader: Callable[[str, str, str], str | None] = read_registry_string) -> str:
    client = r"Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    problems: list[dict[str, Any]] = []
    for hive, key in (
        ("HKLM", rf"SOFTWARE\WOW6432Node\{client}"),
        ("HKLM", rf"SOFTWARE\{client}"),
        ("HKCU", rf"Software\{client}"),
    ):
        try:
            value = reader(hive, key, "pv")
        except Unread as unread:
            problems.append({"code": unread.code, **unread.fields})
            continue
        if value and value != "0.0.0.0":
            return _plain(value, limit=32)
    if problems:
        raise Unread("webview2_registry_unreadable", links=problems)
    return ""


# ---------------------------------------------------------------------------
# 2. Win32 Top-Level Window Custody & Owned/Focused Native SendInput
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopWindow:
    hwnd: int
    pid: int
    title: str


def _user32() -> Any:
    return ctypes.WinDLL("user32", use_last_error=True)


def make_dpi_aware() -> str:
    if sys.platform != "win32":
        return "non_windows"
    user32 = _user32()
    try:
        context = ctypes.c_void_p((1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 4)  # PER_MONITOR_AWARE_V2
        if user32.SetProcessDpiAwarenessContext(context):
            return "per_monitor_v2"
    except AttributeError:
        pass
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
        return "per_monitor"
    except (AttributeError, OSError):
        return "unaware"


def window_pid(hwnd: int) -> int:
    """The process that owns ``hwnd`` (0 when none)."""
    from ctypes import wintypes

    pid = wintypes.DWORD(0)
    _user32().GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    return int(pid.value)


def list_own_windows(pid: int) -> list[TopWindow]:
    """Visible titled top-level windows of process ``pid``; another process's title is never read."""
    if sys.platform != "win32":
        raise Unread("enum_windows_unavailable", platform=sys.platform)
    from ctypes import wintypes

    user32 = _user32()
    found: list[TopWindow] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _visit(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd) or window_pid(hwnd) != pid:
            return True
        length = int(user32.GetWindowTextLengthW(hwnd))
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if title := buf.value.strip():
                found.append(TopWindow(hwnd=int(hwnd), pid=pid, title=title[:120]))
        return True

    if not user32.EnumWindows(_visit, 0):
        raise Unread("enum_windows_failed", winerror=ctypes.get_last_error())
    return found


def is_window_open(hwnd: int) -> bool:
    if sys.platform != "win32" or not hwnd:
        return False
    user32 = _user32()
    return bool(user32.IsWindow(hwnd) and user32.IsWindowVisible(hwnd))


def window_owned_by(hwnd: int, pid: int) -> bool:
    if sys.platform != "win32" or not hwnd or not pid:
        return False
    user32 = _user32()
    if not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
        return False
    return window_pid(hwnd) == pid


def foreground_window() -> int:
    if sys.platform != "win32":
        return 0
    from ctypes import wintypes

    user32 = _user32()
    user32.GetForegroundWindow.argtypes = ()
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
    user32.GetAncestor.restype = wintypes.HWND
    fg = user32.GetForegroundWindow()
    if not fg:
        return 0
    return int(user32.GetAncestor(fg, 2) or fg)


def ensure_foreground_owned(hwnd: int, root_pid: int) -> bool:
    """Focus ``hwnd`` only when process ``root_pid`` owns it and confirm it holds foreground. Every native input
    passes here first: it is desktop-wide, so it runs only on a disposable hosted runner."""
    if not disposable_hosted_runner():
        raise Unread("native_input_needs_a_disposable_runner")
    if sys.platform != "win32" or not window_owned_by(hwnd, root_pid):
        return False
    if foreground_window() == hwnd:
        return True
    from ctypes import wintypes

    user32 = _user32()
    user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.BringWindowToTop.argtypes = (wintypes.HWND,)
    user32.BringWindowToTop.restype = wintypes.BOOL
    for _ in range(5):
        if not window_owned_by(hwnd, root_pid):
            return False
        user32.BringWindowToTop(wintypes.HWND(hwnd))
        user32.SetForegroundWindow(wintypes.HWND(hwnd))
        time.sleep(0.12)
        if foreground_window() == hwnd and window_owned_by(hwnd, root_pid):
            return True
    return foreground_window() == hwnd and window_owned_by(hwnd, root_pid)


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    from ctypes import wintypes

    rect = wintypes.RECT()
    if ctypes.WinDLL("dwmapi").DwmGetWindowAttribute(
        wintypes.HWND(hwnd), 9, ctypes.byref(rect), ctypes.sizeof(rect)
    ) != 0:
        _user32().GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
    return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)


def screen_info() -> dict[str, Any]:
    if sys.platform != "win32":
        return {"width": None, "height": None, "dpi": None, "foreground_present": False}
    user32 = _user32()
    return {
        "width": int(user32.GetSystemMetrics(0)),
        "height": int(user32.GetSystemMetrics(1)),
        "dpi": int(user32.GetDpiForSystem()) if hasattr(user32, "GetDpiForSystem") else None,
        "foreground_present": foreground_window() != 0,
    }


def root_window_at(x: int, y: int) -> int:
    from ctypes import wintypes

    user32 = _user32()
    user32.WindowFromPoint.argtypes = (wintypes.POINT,)
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
    user32.GetAncestor.restype = wintypes.HWND
    hit = user32.WindowFromPoint(wintypes.POINT(int(x), int(y)))
    return int(user32.GetAncestor(hit, 2) or 0) if hit else 0


@functools.lru_cache(maxsize=1)
def _input_types() -> tuple[Any, Any, Any]:
    from ctypes import wintypes

    ulong_ptr = ctypes.c_size_t

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ulong_ptr),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ulong_ptr),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

    class _UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _UNION)]

    return INPUT, MOUSEINPUT, KEYBDINPUT


def _send_inputs(*inputs: Any) -> None:
    INPUT, _m, _k = _input_types()
    array = (INPUT * len(inputs))(*inputs)
    sent = _user32().SendInput(len(inputs), array, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        raise Unread("send_input_incomplete", sent=int(sent), requested=len(inputs), winerror=ctypes.get_last_error())


def _mouse_event(flags: int, data: int = 0) -> Any:
    INPUT, MOUSEINPUT, _k = _input_types()
    ev = INPUT(type=0)
    ev.u.mi = MOUSEINPUT(0, 0, ctypes.c_uint32(data).value, flags, 0, 0)
    return ev


def _key_vk(vk: int, up: bool = False) -> Any:
    INPUT, _m, KEYBDINPUT = _input_types()
    ev = INPUT(type=1)
    ev.u.ki = KEYBDINPUT(vk, 0, 0x0002 if up else 0, 0, 0)
    return ev


def _prepare_pointer(x: int, y: int, hwnd: int, root_pid: int) -> None:
    if not ensure_foreground_owned(hwnd, root_pid):
        raise Unread("window_focus_or_ownership_unavailable")
    if root_window_at(x, y) != hwnd:
        raise Unread("target_point_obscured_or_unowned")
    if not _user32().SetCursorPos(int(x), int(y)):
        raise Unread("set_cursor_pos_failed", winerror=ctypes.get_last_error())
    time.sleep(0.08)
    if not window_owned_by(hwnd, root_pid) or foreground_window() != hwnd or root_window_at(x, y) != hwnd:
        raise Unread("window_changed_before_pointer_input")


def click(x: int, y: int, hwnd: int, root_pid: int) -> None:
    _prepare_pointer(x, y, hwnd, root_pid)
    _send_inputs(_mouse_event(0x0002))
    time.sleep(0.05)
    _send_inputs(_mouse_event(0x0004))
    time.sleep(0.22)


def wheel(x: int, y: int, hwnd: int, root_pid: int, notches: int) -> None:
    """Scroll at ``(x, y)`` inside ``hwnd``; positive ``notches`` scrolls down."""
    _prepare_pointer(x, y, hwnd, root_pid)
    _send_inputs(_mouse_event(0x0800, -120 * int(notches)))
    time.sleep(0.28)


def select_all_and_replace_text(text: str, hwnd: int, root_pid: int) -> None:
    """Replace focused control contents via Ctrl+A, Backspace, and Unicode SendInput."""
    if not ensure_foreground_owned(hwnd, root_pid):
        raise Unread("window_focus_or_ownership_unavailable")
    # VK_CONTROL (0x11) + 'A' (0x41), then VK_BACK (0x08)
    _send_inputs(_key_vk(0x11, False), _key_vk(0x41, False), _key_vk(0x41, True), _key_vk(0x11, True))
    time.sleep(0.05)
    _send_inputs(_key_vk(0x08, False), _key_vk(0x08, True))
    time.sleep(0.05)
    INPUT, _m, KEYBDINPUT = _input_types()
    for ch in text:
        if not window_owned_by(hwnd, root_pid) or foreground_window() != hwnd:
            raise Unread("window_focus_lost_during_typing")
        code = ord(ch)
        units = [code] if code <= 0xFFFF else [0xD800 + ((code - 0x10000) >> 10), 0xDC00 + ((code - 0x10000) & 0x3FF)]
        events = []
        for unit in units:
            down, up = INPUT(type=1), INPUT(type=1)
            down.u.ki = KEYBDINPUT(0, unit, 0x0004, 0, 0)
            up.u.ki = KEYBDINPUT(0, unit, 0x0004 | 0x0002, 0, 0)
            events.extend((down, up))
        _send_inputs(*events)
        time.sleep(0.006)
    time.sleep(0.12)

# ---------------------------------------------------------------------------
# 3. Git trace sinks: read on the runner, interpreted only by git_traces
# ---------------------------------------------------------------------------

def trace_env(trace1: pathlib.Path, trace2_dir: pathlib.Path) -> dict[str, str]:
    """The variables that ask every Git of a process tree for its identity line and its session events."""
    return {"GIT_TRACE": str(trace1), "GIT_TRACE2_EVENT": str(trace2_dir), "GIT_TRACE2_MAX_FILES": "100000"}


def read_trace_sinks(trace1: pathlib.Path, trace2_dir: pathlib.Path) -> tuple[str, list[str], bool]:
    """Trace1's text, then every Trace2 file's text, and whether Git left its discard sentinel. Trace1 is read
    first, so a process starting in between can only add a session without its line: a mismatch, never a match."""
    try:
        first = trace1.read_bytes().decode("utf-8", errors="replace") if trace1.exists() else ""
        files = sorted(entry for entry in trace2_dir.iterdir() if entry.is_file()) if trace2_dir.is_dir() else []
        texts = [entry.read_bytes().decode("utf-8", errors="replace") for entry in files]
    except OSError as exc:
        raise Unread("git_trace_unreadable", problem=type(exc).__name__) from exc
    return first, texts, any(entry.name == "git-trace2-discard" for entry in files)


def exe_digests(package_root: pathlib.Path, rel_folder: str) -> dict[str, str]:
    """Name -> SHA-256 of every ``.exe`` now in this package folder."""
    try:
        with os.scandir(package_root / pathlib.PurePosixPath(rel_folder)) as entries:
            exes = [entry for entry in entries if entry.is_file() and entry.name.lower().endswith(".exe")]
        return {entry.name: sha256_file(entry.path) for entry in exes}
    except OSError as exc:
        raise Unread("bundled_git_folder_unreadable", problem=type(exc).__name__) from exc


def traced_git_version(git_exe: pathlib.Path, sinks: pathlib.Path, env: dict[str, str]) -> int | None:
    """Run the package's own ``git version`` once with both sinks in ``sinks``: the instrument's positive control
    on this build. Returns the exit code, None when it did not run or answer."""
    (sinks / "trace2").mkdir(parents=True, exist_ok=True)
    try:
        done = subprocess.run([str(git_exe), "version"], cwd=str(sinks), capture_output=True, timeout=60,
                              env={**env, **trace_env(sinks / "trace1.log", sinks / "trace2")})
    except (OSError, subprocess.SubprocessError):
        return None
    return done.returncode

"""Windows primitives through documented APIs only: ctypes Win32 reads, WMI, the registry and the PowerShell probes in ps/.

No injection, no debugger, no hooks, no reading of another process's memory, and NO physical input: nothing here moves the pointer or
types. The one message this module posts is WM_CLOSE to one window the caller owns (what that window's close button sends). Windows
modules are imported when a function runs, so this file imports anywhere. A read that fails returns None / raises
``windows_host.Unread``; it is never read as "absent".
"""

from __future__ import annotations

import ctypes
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import windows_host as wh  # vendored #4220 module

ROOT = pathlib.Path(os.environ.get("SystemRoot", r"C:\Windows"))
POWERSHELL = ROOT / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
PS_DIR = pathlib.Path(__file__).resolve().parent / "ps"
NO_WINDOW, NEW_GROUP = 0x08000000, 0x00000200


def _k32() -> Any:
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes, k.OpenProcess.restype = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD), wintypes.HANDLE
    k.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    return k


def pid_alive(pid: int) -> bool:
    if sys.platform != "win32" or int(pid) <= 0:
        return False
    from ctypes import wintypes

    k = _k32()
    handle = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() == 5  # access denied: it exists; any other error: it does not
    try:
        code = wintypes.DWORD(0)
        return bool(k.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
    finally:
        k.CloseHandle(handle)


def image_path(pid: int) -> str | None:
    """The image as Windows reports it (QueryFullProcessImageNameW); None when unreadable."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    k = _k32()
    handle = k.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    try:
        size, buf = wintypes.DWORD(32768), ctypes.create_unicode_buffer(32768)
        return buf.value if k.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)) else None
    finally:
        k.CloseHandle(handle)


def lock_pid(app_root: Any) -> int | None:
    """The PID in the installation's own lock file, ``<profile>/ouroboros.pid``; None when absent or unreadable."""
    for _ in range(3):
        try:
            value = int(pathlib.Path(app_root, "ouroboros.pid").read_text(encoding="utf-8").strip())
            return value if value > 0 else None
        except (OSError, ValueError):
            time.sleep(0.05)
    return None


def cim_process(pid: int) -> dict[str, Any]:
    """WMI Win32_Process (documented) for one process: parent id and name, and the command line. {} when unreadable."""
    ps = ("$p=Get-CimInstance Win32_Process -Filter 'ProcessId=%d'; if($p){$q=Get-CimInstance Win32_Process -Filter ('ProcessId='+$p.ParentProcessId); "
          "@{ppid=[int]$p.ParentProcessId;parent=[string]$q.Name;cmd=[string]$p.CommandLine}|ConvertTo-Json -Compress}" % int(pid))
    try:
        done = subprocess.run([str(POWERSHELL), "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True, text=True, timeout=90)
        return json.loads(done.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def descendants(root_pid: int) -> set[int] | None:
    """Every process below ``root_pid``, from one WMI read of the process table (documented): the windows of an installer we started
    belong to its extracted child, so its own tree is the only honest scope. None when the table could not be read."""
    ps = "Get-CimInstance Win32_Process | ForEach-Object { '{0} {1}' -f $_.ProcessId, $_.ParentProcessId }"
    try:
        done = subprocess.run([str(POWERSHELL), "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True, text=True,
                              timeout=60, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode:
        return None
    children: dict[int, list[int]] = {}
    for line in done.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found: set[int] = set()
    stack = [int(root_pid)]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def loopback_listener_owned(port: int, root_pid: int) -> bool:
    """Prove the exact 127.0.0.1 listener belongs to the launched app's tree before sending fixture writes."""
    if not 1 <= port <= 65535:
        return False
    ps = ("Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort %d -State Listen -ErrorAction Stop | "
          "Select-Object -ExpandProperty OwningProcess" % port)
    try:
        done = subprocess.run([str(POWERSHELL), "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                              text=True, timeout=15, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return False
    pids = done.stdout.split()
    return (not done.returncode and len(pids) == 1 and pids[0].isdigit()
            and int(pids[0]) in ({root_pid} | (descendants(root_pid) or set())))


def post_close(hwnd: int, pid: int) -> None:
    """WM_CLOSE to a window of ``pid``: what pressing its close button sends."""
    if not wh.window_owned_by(hwnd, pid):
        raise wh.Unread("window_not_owned_for_close")
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.PostMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    if not user32.PostMessageW(wintypes.HWND(hwnd), 0x0010, 0, 0):
        raise wh.Unread("post_close_failed", winerror=ctypes.get_last_error())


def run_ps(script: pathlib.Path, args: list[Any], *, timeout: float, env: dict[str, str] | None = None) -> tuple[int | None, str]:
    """A PowerShell script under the normal local execution policy: (exit code or None on timeout, stderr tail). The operator's own
    OUROBOROS_QA_DISPOSABLE=1 passes through; this code never sets it."""
    cmd = [str(POWERSHELL), "-NoProfile", "-NonInteractive", "-Sta", "-File", str(script), *[str(a) for a in args]]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env if env is not None else os.environ.copy(), creationflags=NO_WINDOW)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return None, type(exc).__name__
    return done.returncode, (done.stderr or done.stdout or "")[-300:]


def read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def start_shortcut(lnk: pathlib.Path, route: str, env: dict[str, str] | None) -> None:
    """Open the real .lnk. ``shellexecute``: ``cmd /c start`` under ``env``, so the app inherits the Git trace sinks.
    ``explorer``: the running shell opens it as Start does; the shell's environment applies and no sink reaches the app."""
    quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if route == "explorer":
        subprocess.Popen([str(ROOT / "explorer.exe"), str(lnk)], **quiet)
    else:
        subprocess.Popen([str(ROOT / "System32" / "cmd.exe"), "/c", "start", "", str(lnk)], env=env, creationflags=NO_WINDOW, **quiet)


def uninstall_entry(key_name: str) -> dict[str, str] | None:
    """The per-user Uninstall entry the installer writes (HKCU, 64-bit view): its values, or None when the key is absent."""
    import winreg

    path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall" + "\\" + key_name
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            out = {}
            for name in ("DisplayName", "UninstallString", "QuietUninstallString", "InstallLocation", "DisplayIcon"):
                try:
                    out[name] = str(winreg.QueryValueEx(key, name)[0])
                except OSError:
                    out[name] = ""
            return out
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise wh.Unread("uninstall_key_unreadable", problem=type(exc).__name__) from exc


def split_command(text: str) -> list[str]:
    """``"C:\\x\\unins000.exe" /SILENT`` as [program, *arguments]."""
    text = text.strip()
    if text.startswith('"') and '"' in text[1:]:
        end = text.index('"', 1)
        return [text[1:end], *text[end + 1:].split()]
    head, _, rest = text.partition(" ")
    return [head, *rest.split()]

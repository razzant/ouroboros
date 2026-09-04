"""Cross-platform process, locking, path, and runtime helpers."""

from __future__ import annotations

import logging
import os
import pathlib
import platform
import re
import signal
import subprocess
import sys
import time
from typing import Any, Callable, List, Optional

log = logging.getLogger(__name__)

# Platform flags.
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

PATH_SEP = ";" if IS_WINDOWS else ":"
_SUBPROCESS_NO_WINDOW = (
    getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if IS_WINDOWS else 0
)
_PATH_BOOTSTRAPPED = False


def executable_name_candidates(name: str) -> List[str]:
    """Platform spellings for an executable stored in a known directory."""
    base = str(name or "").strip()
    if not base:
        return []
    if IS_WINDOWS:
        return [f"{base}.cmd", f"{base}.exe", f"{base}.bat", base]
    return [base]


def local_zoneinfo():
    """Best-effort DST-aware local timezone.

    ``astimezone().tzinfo`` is a *fixed* offset that drifts across DST; resolve the IANA
    zone (``TZ`` or ``/etc/localtime``), falling back to the fixed offset.
    """
    import datetime
    from zoneinfo import ZoneInfo

    tz_env = os.environ.get("TZ", "").strip()
    if tz_env:
        try:
            return ZoneInfo(tz_env)
        except Exception:
            log.debug("Invalid TZ env %r for local timezone", tz_env)
    try:
        link = os.readlink("/etc/localtime")
        if "zoneinfo/" in link:
            return ZoneInfo(link.split("zoneinfo/", 1)[1])
    except (OSError, ValueError):
        pass
    return datetime.datetime.now().astimezone().tzinfo or datetime.timezone.utc


def is_container_env() -> bool:
    """Return whether explicit env or Docker sentinel indicates a container."""
    if os.environ.get("OUROBOROS_CONTAINER") == "1":
        return True
    # /.dockerenv is Docker's Linux sentinel.
    if IS_LINUX and pathlib.Path("/.dockerenv").exists():
        return True
    return False


def bootstrap_process_path() -> list[str]:
    """Add existing common user tool directories to this process PATH once."""

    global _PATH_BOOTSTRAPPED
    if _PATH_BOOTSTRAPPED:
        return []
    _PATH_BOOTSTRAPPED = True

    candidates: list[pathlib.Path] = []
    home = pathlib.Path.home()
    if IS_MACOS or IS_LINUX:
        candidates.extend([
            pathlib.Path("/opt/homebrew/bin"),
            pathlib.Path("/opt/homebrew/sbin"),
            pathlib.Path("/usr/local/bin"),
            pathlib.Path("/usr/local/sbin"),
            pathlib.Path("/opt/local/bin"),
            home / ".local" / "bin",
            home / ".cargo" / "bin",
            home / ".npm-global" / "bin",
            home / "go" / "bin",
        ])
    if IS_WINDOWS:
        def _env_path(name: str, default: str = "") -> pathlib.Path | None:
            text = os.environ.get(name, default)
            if not text:
                return None
            path = pathlib.Path(text)
            return path if path.is_absolute() else None

        program_files = _env_path("ProgramFiles", r"C:\Program Files")
        local_app_data = _env_path("LOCALAPPDATA")
        app_data = _env_path("APPDATA")
        user_profile = _env_path("USERPROFILE")
        if program_files:
            candidates.extend([program_files / "Git" / "cmd", program_files / "nodejs"])
        if local_app_data:
            candidates.append(local_app_data / "Programs" / "Git" / "cmd")
        if app_data:
            candidates.append(app_data / "npm")
        if user_profile:
            candidates.append(user_profile / ".cargo" / "bin")

    existing = [part for part in os.environ.get("PATH", "").split(PATH_SEP) if part]
    existing_norm = {str(pathlib.Path(part)).lower() if IS_WINDOWS else str(pathlib.Path(part)) for part in existing}
    added: list[str] = []
    for candidate in candidates:
        try:
            if not candidate.is_dir():
                continue
            text = str(candidate)
            norm = text.lower() if IS_WINDOWS else text
            if norm in existing_norm:
                continue
            existing_norm.add(norm)
            added.append(text)
        except OSError:
            continue
    if added:
        os.environ["PATH"] = PATH_SEP.join([*added, *existing])
    return added


def scrub_repo_from_pythonpath(env: dict[str, str], repo_dir: "str | pathlib.Path | None") -> dict[str, str]:
    """Return a copy of *env* with any ``PYTHONPATH`` entry resolving to the Ouroboros
    system repo dir removed.

    An EXTERNAL-workspace command inherits the worker's ``PYTHONPATH`` repo entry, which
    makes the target's ``import web``/``server``/``ouroboros`` resolve to OUROBOROS's modules.
    Dropping ONLY the repo entry isolates the target; no-op without one."""
    out = dict(env)
    raw = out.get("PYTHONPATH", "")
    if not raw or not repo_dir:
        return out
    try:
        repo_resolved = pathlib.Path(repo_dir).resolve(strict=False)
    except Exception:
        return out
    kept: list[str] = []
    for part in raw.split(os.pathsep):
        if not part:
            continue
        try:
            if pathlib.Path(part).resolve(strict=False) == repo_resolved:
                continue
        except Exception:
            pass
        kept.append(part)
    if kept:
        out["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        out.pop("PYTHONPATH", None)
    return out


def _lock_owner_pid(lock_path: pathlib.Path) -> int:
    """The ``pid=`` recorded in a lockfile's metadata, or 0 when absent/unreadable."""
    try:
        for field in lock_path.read_text(encoding="utf-8", errors="replace").split():
            if field.startswith("pid="):
                return int(field[4:])
    except (OSError, ValueError):
        pass
    return 0


def _reclaim_lock_of_dead_owner(lock_path: pathlib.Path) -> bool:
    """Unlink ``lock_path`` if its recorded owner pid is provably gone.

    The inode is re-checked right before the unlink so a lock that a new owner
    re-created between our read and our unlink is left alone.
    """
    try:
        before = lock_path.stat()
    except OSError:
        return False
    owner_pid = _lock_owner_pid(lock_path)
    if owner_pid <= 0 or owner_pid == os.getpid() or not pid_provably_gone(owner_pid):
        return False
    try:
        if lock_path.stat().st_ino != before.st_ino:
            return False
        lock_path.unlink()
    except OSError:
        return False
    log.warning(
        "Reclaimed lock %s held by dead process pid=%d", lock_path, owner_pid,
    )
    return True


def acquire_exclusive_file_lock(
    lock_path: pathlib.Path,
    *,
    timeout_sec: float = 4.0,
    stale_sec: float = 90.0,
    metadata: str = "",
    poll_sec: float = 0.05,
    owner_aware_stale: bool = False,
    reclaim_dead_owner: bool = False,
) -> Optional[int]:
    """Acquire a portable lockfile using O_EXCL and return its file descriptor.

    Authority streams opt into ``owner_aware_stale`` so elapsed time alone can
    never steal a lock from a live writer.  A dead/malformed legacy owner still
    recovers through the existing stale-age path.

    ``reclaim_dead_owner`` reclaims a lock whose recorded ``pid=`` owner the OS
    positively reports as gone, without waiting out ``stale_sec``.  A worker
    killed by cancel/timeout custody while it held the usage ledger lock
    (``proc.terminate()`` — no unwinding, no ``finally``) otherwise orphans the
    lock for the full stale window: every 45 s ledger transaction behind it
    fails and the task fails as ``infra_failed`` (Tier-2 load repro on
    02b99c71: one cancel -> 90 s outage -> 17 accounting failures, 10 healthy
    tasks written off).  Only a provably dead pid is reclaimed; a live or
    unknown owner still goes through the stale-age path.
    """
    lock_path = pathlib.Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while (time.time() - started) < timeout_sec:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                text = metadata or f"pid={os.getpid()} ts={time.time()}\n"
                os.write(fd, text.encode("utf-8"))
            except Exception:
                log.debug("Failed to write lock metadata to %s", lock_path, exc_info=True)
            return fd
        except (FileExistsError, PermissionError):
            try:
                if reclaim_dead_owner and _reclaim_lock_of_dead_owner(lock_path):
                    continue
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_sec:
                    if owner_aware_stale:
                        owner_pid = _lock_owner_pid(lock_path)
                        if owner_pid > 0 and pid_is_alive(owner_pid):
                            time.sleep(poll_sec)
                            continue
                    lock_path.unlink()
                    continue
            except Exception:
                log.debug("Failed to inspect/remove stale lock %s", lock_path, exc_info=True)
            time.sleep(poll_sec)
        except Exception:
            log.warning("Failed to acquire lock at %s", lock_path, exc_info=True)
            break
    return None


def release_exclusive_file_lock(lock_path: pathlib.Path, lock_fd: Optional[int]) -> None:
    """Release a lock acquired by :func:`acquire_exclusive_file_lock`."""
    lock_path = pathlib.Path(lock_path)
    if lock_fd is None:
        return
    try:
        os.close(lock_fd)
    except Exception:
        log.debug("Failed to close lock fd %s for %s", lock_fd, lock_path, exc_info=True)
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        log.debug("Failed to unlink lock file %s", lock_path, exc_info=True)


def unlink_lockfile(lock_path: pathlib.Path) -> None:
    """Best-effort cleanup for path-only locks whose fd was closed after acquire."""
    lock_path = pathlib.Path(lock_path)
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        log.debug("Failed to unlink lock file %s", lock_path, exc_info=True)


def open_path_external(path: pathlib.Path) -> None:
    """Open a local path with the platform default application."""

    target = pathlib.Path(path)
    if IS_MACOS:
        subprocess.Popen(["open", str(target)])
    elif IS_WINDOWS:
        os.startfile(str(target))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(target)])


def is_unstable_macos_app_path(path: pathlib.Path) -> bool:
    """Return whether a macOS app path is likely a DMG/AppTranslocation mount."""
    raw = str(path).replace("\\", "/")
    resolved = str(path.resolve()).replace("\\", "/")
    return (
        "AppTranslocation" in raw
        or "AppTranslocation" in resolved
        or raw.startswith("/Volumes/")
        or resolved.startswith("/Volumes/")
    )


def ensure_windows_user_path(path: pathlib.Path) -> None:
    """Add a directory to the current Windows user's PATH and notify shells."""
    if not IS_WINDOWS:
        return
    import winreg  # type: ignore[import-not-found]

    path_text = str(path)
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE) as key:
        try:
            current, value_type = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            current, value_type = "", winreg.REG_EXPAND_SZ
        parts = [p for p in str(current).split(";") if p]
        if any(p.lower() == path_text.lower() for p in parts):
            return
        updated = ";".join(parts + [path_text])
        winreg.SetValueEx(key, "Path", 0, value_type, updated)
    _broadcast_windows_environment_change()


def _broadcast_windows_environment_change() -> None:
    if not IS_WINDOWS:
        return
    try:
        import ctypes

        result = ctypes.c_ulong()
        ctypes.windll.user32.SendMessageTimeoutW(
            0xFFFF,  # HWND_BROADCAST
            0x001A,  # WM_SETTINGCHANGE
            0,
            "Environment",
            0x0002,  # SMTO_ABORTIFHUNG
            5000,
            ctypes.byref(result),
        )
    except Exception:
        pass


def _hidden_run(command: list[str], **kwargs):
    if _SUBPROCESS_NO_WINDOW:
        kwargs = dict(kwargs)
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | _SUBPROCESS_NO_WINDOW
    return subprocess.run(command, **kwargs)


# PID file locking.
_lock_fd: Any = None


def pid_lock_acquire(path: str) -> bool:
    """Acquire an exclusive PID lock, closing the fd on lock failure."""
    global _lock_fd
    fd_obj = None
    try:
        fd_obj = open(path, "w")
        if IS_WINDOWS:
            _win32_lock(fd_obj.fileno(), exclusive=True, blocking=False)
        else:
            import fcntl
            fcntl.flock(fd_obj, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd_obj.write(str(os.getpid()))
        fd_obj.flush()
        # Promote to global only after lock and PID write both succeed.
        _lock_fd = fd_obj
        return True
    except (IOError, OSError):
        if fd_obj is not None:
            try:
                fd_obj.close()
            except Exception:
                pass
        return False


def pid_lock_release(path: str) -> None:
    """Release the PID lock."""
    global _lock_fd
    if _lock_fd is not None:
        if IS_WINDOWS:
            try:
                _win32_unlock(_lock_fd.fileno())
            except Exception:
                pass
        else:
            import fcntl
            try:
                fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            except Exception:
                pass
        try:
            _lock_fd.close()
        except Exception:
            pass
        _lock_fd = None
    try:
        os.unlink(path)
    except Exception:
        pass


# File locking.

def file_lock_exclusive(fd: int) -> None:
    """Acquire an exclusive (write) lock on a file descriptor. Blocks."""
    if IS_WINDOWS:
        _win32_lock(fd, exclusive=True, blocking=True)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)


def file_lock_shared(fd: int) -> None:
    """Acquire a shared (read) lock on a file descriptor. Blocks."""
    if IS_WINDOWS:
        _win32_lock(fd, exclusive=False, blocking=True)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_SH)


def file_lock_exclusive_nb(fd: int) -> None:
    """Try to acquire an exclusive lock, non-blocking. Raises OSError on failure."""
    if IS_WINDOWS:
        _win32_lock(fd, exclusive=True, blocking=False)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def file_unlock(fd: int) -> None:
    """Release a file lock."""
    if IS_WINDOWS:
        _win32_unlock(fd)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


def pid_is_alive(pid: int) -> bool:
    """Return whether a PID appears alive without exposing os.kill to callers."""

    if pid <= 0:
        return False
    if IS_WINDOWS:
        # os.kill(pid, 0) is WRONG on Windows: signal 0 is CTRL_C_EVENT, so
        # os.kill sends Ctrl+C to the target pid's CONSOLE PROCESS GROUP instead
        # of probing liveness — when the pid shares this process's console (e.g.
        # our own pid, or a sibling under the same runner console) it delivers a
        # KeyboardInterrupt to the whole group. Probe with OpenProcess +
        # GetExitCodeProcess, which never signals anything.
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        _STILL_ACTIVE = 259
        _ERROR_ACCESS_DENIED = 5
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            # A live but access-protected process reads as alive; anything else
            # (invalid parameter -> no such pid) reads as dead.
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True  # opened but unreadable -> fail SAFE toward alive
            return int(code.value) == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pid_provably_gone(pid: int) -> bool:
    """True only when the OS positively answers that ``pid`` does not exist.

    Stricter than ``not pid_is_alive``: the POSIX branch there folds EVERY
    OSError into 'dead', but EPERM means the process EXISTS and merely refuses
    our signal — a caller deciding whether a killed process is really gone
    must treat that (and anything else undeterminable) as still present."""
    if pid <= 0:
        return True
    if IS_WINDOWS:
        return not pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


# Windows file locking via LockFileEx/UnlockFileEx; unlike msvcrt.locking(),
# this works on empty files by locking a range beyond current size.

# Per-fd OVERLAPPED storage for unlock.
_win32_overlapped: dict = {}


_OVERLAPPED_CLS = None  # cached once per process


def _win32_overlapped_class():
    """Return cached portable OVERLAPPED; ctypes requires one class identity."""
    global _OVERLAPPED_CLS
    if _OVERLAPPED_CLS is not None:
        return _OVERLAPPED_CLS

    import ctypes
    from ctypes import wintypes

    class OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _OVERLAPPED_CLS = OVERLAPPED
    return OVERLAPPED


def _win32_lock(fd: int, *, exclusive: bool = True, blocking: bool = True) -> None:
    """Lock a file descriptor using Win32 LockFileEx. Works on empty files."""
    import ctypes
    from ctypes import wintypes
    import msvcrt as _msvcrt

    _LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
    _LOCKFILE_EXCLUSIVE_LOCK = 0x00000002

    OVERLAPPED = _win32_overlapped_class()

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LockFileEx.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(OVERLAPPED),
    ]
    kernel32.LockFileEx.restype = wintypes.BOOL

    hfile = _msvcrt.get_osfhandle(fd)
    flags = 0
    if exclusive:
        flags |= _LOCKFILE_EXCLUSIVE_LOCK
    if not blocking:
        flags |= _LOCKFILE_FAIL_IMMEDIATELY

    ov = OVERLAPPED()
    # Win32 whole-file lock pattern: huge range from offset 0.
    if not kernel32.LockFileEx(hfile, flags, 0, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(ov)):
        err = ctypes.get_last_error()
        raise OSError(f"LockFileEx failed (error {err})")

    _win32_overlapped[fd] = (hfile, ov)


def _win32_unlock(fd: int) -> None:
    """Unlock a file descriptor previously locked by _win32_lock."""
    import ctypes
    from ctypes import wintypes

    entry = _win32_overlapped.pop(fd, None)
    if entry is None:
        return

    OVERLAPPED = _win32_overlapped_class()

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.UnlockFileEx.argtypes = [
        wintypes.HANDLE, wintypes.DWORD,
        wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(OVERLAPPED),
    ]
    kernel32.UnlockFileEx.restype = wintypes.BOOL

    hfile, ov = entry
    try:
        kernel32.UnlockFileEx(hfile, 0, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(ov))
    except OSError:
        pass


# Process management.

def kill_process_tree(proc: subprocess.Popen) -> None:
    """Force-kill a subprocess and its entire process tree.

    On POSIX the immediate process group is SIGKILLed first, then descendants that
    escaped into their own session/group are swept by PID — without that sweep a
    cancelled child which spawned grandchildren in new groups leaks orphans.
    Descendants are collected BEFORE the kill: once the parent dies its children are
    reparented and the ppid links disappear.
    """
    pid = proc.pid
    if IS_WINDOWS:
        try:
            _hidden_run(["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=10)
        except Exception:
            pass
        return
    descendants: list[int] = []
    try:
        _collect_descendants(pid, descendants)
    except Exception:
        descendants = []
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    for dpid in reversed(descendants):
        try:
            os.kill(dpid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def terminate_process_tree(proc: subprocess.Popen) -> None:
    """Gracefully terminate a subprocess and its process tree."""
    if IS_WINDOWS:
        proc.terminate()
    else:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def terminate_process_group_id(pgid: int) -> None:
    """Gracefully terminate a Unix process group by id."""
    if IS_WINDOWS:
        return
    try:
        os.killpg(int(pgid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError, ValueError):
        pass


def kill_process_group_id(pgid: int) -> None:
    """Force-kill a Unix process group by id."""
    if IS_WINDOWS:
        return
    try:
        os.killpg(int(pgid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError, ValueError):
        pass


def process_group_id(pid: int) -> int:
    """Return the Unix process group id for ``pid`` or 0 when unavailable."""
    if IS_WINDOWS:
        return 0
    try:
        return int(os.getpgid(int(pid)))
    except (ProcessLookupError, PermissionError, OSError, ValueError):
        return 0


def process_group_is_alive(pgid: int) -> bool:
    """Return whether a Unix process group still has at least one member."""
    if IS_WINDOWS or int(pgid or 0) <= 0:
        return False
    try:
        os.killpg(int(pgid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False


def current_process_group_id() -> int:
    """Return the current Unix process group id or 0 when unavailable."""
    if IS_WINDOWS:
        return 0
    try:
        return int(os.getpgrp())
    except (PermissionError, OSError, ValueError):
        return 0


_BOOT_ID = ""  # full hex of the /proc boot id; empty until a successful read, then latched


def process_start_time(pid: int) -> str:
    """Best-effort stable start-time token for (pid, start_time) fingerprints.

    A bare pid is not an identity (kernels reuse pids) and boot-relative ticks RECUR across
    reboots, so a bare tick + a recycled pid + the same command line is a real collision;
    refusing that kill is the job. Linux mints IN THIS ORDER: ``"<ticks>.<boot_hex>"`` when the
    boot id is readable (no subprocess); else the ``ps`` wall-clock token, which does not recur
    across boots in practice; and only once ``ps`` has ALSO failed, ``"<ticks>."`` as a
    disclosed last resort — two of THOSE from different boots do string-match, hence last, not
    first. No ``/proc``: legacy throughout; Windows and a dead pid return "". Disclosed: the
    FORM changes if the boot id starts or stops being readable mid-generation, so a row
    recorded across it can mismatch its own live process — safe: it prunes, never kills, and
    the cheap reap path skips live rows."""
    global _BOOT_ID
    if pid <= 0 or os.name == "nt":
        return ""
    if not (ticks := _proc_start_ticks(pid)):
        return process_start_time_legacy(pid)  # no /proc here (macOS, BSD): ps is the only source
    if not _BOOT_ID:  # a failed read is transient: retry next call, never downgrade the generation
        try:
            _BOOT_ID = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip().replace("-", "")
        except (OSError, ValueError):  # matches _proc_start_ticks; an escapee would abort a custody sweep
            pass
    if _BOOT_ID:
        return f"{ticks}.{_BOOT_ID}"
    # legacy degrades to str(ticks) when ps ALSO failed; only then is the separator form the best we have.
    return legacy if (legacy := process_start_time_legacy(pid)) and legacy != str(ticks) else f"{ticks}."


def process_start_time_legacy(pid: int) -> str:
    """The historical ``ps -o lstart=`` token (bare ``/proc`` ticks when ``ps`` fails). Two jobs:
    the DOWNGRADE-SAFE spelling the custody ledger keeps writing into ``fingerprint.start_time``
    (an N−1 reader understands it), and the compatibility comparison a boot-qualified current
    token falls back to (see ``process_custody._legacy_start_matches``)."""
    if pid <= 0 or os.name == "nt":
        return ""
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5)
        text = (out.stdout or "").strip()
        if out.returncode == 0 and text:
            return text
    except Exception:
        pass
    ticks = _proc_start_ticks(pid)
    return str(ticks) if ticks else ""


def _proc_start_ticks(pid: int) -> int:
    """Boot-relative start time (``/proc/<pid>/stat`` field 22), or 0 when it cannot be read."""
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as handle:
            fields = handle.read().rpartition(b")")[2].split()
        return int(fields[19]) if len(fields) >= 20 else 0  # rpartition dropped fields 1-2
    except (OSError, ValueError):
        return 0


def process_command(pid: int) -> str:
    """Return a best-effort command line for a Unix process."""
    if IS_WINDOWS:
        return ""
    try:
        # -ww: unlimited width. BSD ps truncates to the terminal/128 cols
        # otherwise, and consumers match exact argv tokens — a packaged
        # interpreter path is long enough to push the script argument off the
        # end of a truncated line.
        result = subprocess.run(["ps", "-ww", "-p", str(int(pid)), "-o", "command="],
                                capture_output=True, text=True, timeout=3)
        return result.stdout.strip()
    except Exception:
        return ""


def force_kill_pid(pid: int) -> None:
    """Force-kill a single process by PID."""
    if IS_WINDOWS:
        try:
            _hidden_run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
        except Exception:
            pass
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def kill_pid_tree(pid: int, exclude_pids: "set[int] | None" = None) -> None:
    """Force-kill a PID tree recursively.

    ``exclude_pids`` are spared along with their own descendants, keeping
    ``service_teardown=keep`` services reachable for a verifier when a worker is
    force-killed; spared children reparent to init and fall to the custody reaper.
    """
    exclude = {int(p) for p in (exclude_pids or set())}
    if IS_WINDOWS:
        # exclude_pids is a POSIX-only nicety: descendant enumeration relies on
        # `pgrep -P`, which Windows lacks, so honouring exclusions here would
        # enumerate nothing and LEAK the worker's whole tree (only the root would
        # die). taskkill /T always tree-kills; sparing is unsupported on Windows.
        try:
            _hidden_run(["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=10)
        except Exception:
            pass
        return

    descendants: list[int] = []
    _collect_descendants(pid, descendants)
    spared: set[int] = set()
    for ep in exclude:
        spared.add(ep)
        sub: list[int] = []
        _collect_descendants(ep, sub)
        spared.update(sub)
    for dpid in reversed(descendants):
        if dpid in spared:
            continue
        try:
            os.kill(dpid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if pid in spared:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _collect_descendants(pid: int, result: list[int]) -> None:
    """Recursively collect all descendant PIDs via pgrep."""
    try:
        out = subprocess.run(["pgrep", "-P", str(pid)],
                             capture_output=True, text=True, timeout=3)
        for line in out.stdout.strip().splitlines():
            line = line.strip()
            if line:
                child_pid = int(line)
                _collect_descendants(child_pid, result)
                result.append(child_pid)
    except Exception:
        pass


def collect_descendant_pids(pid: int) -> List[int]:
    """Public: all descendant PIDs of ``pid`` (depth-first, children last).

    Keeps tree discovery in the platform layer, off the private recursive helper."""
    result: List[int] = []
    try:
        _collect_descendants(int(pid), result)
    except (TypeError, ValueError):
        pass
    return result


def kill_processes_referencing(marker: str) -> None:
    """Force-kill any process whose command line references ``marker``.

    Sweeps children that double-forked to init, escaping both ``killpg`` and the
    ``pgrep -P`` walk. ``marker`` is matched literally (regex specials escaped) so a
    temp path containing ``.``/``+`` cannot over-match unrelated command lines."""
    if IS_WINDOWS or not marker:
        return
    try:
        out = subprocess.run(
            ["pgrep", "-f", re.escape(marker)], capture_output=True, text=True, timeout=3
        )
    except Exception:
        return
    my_pid = os.getpid()
    for line in (out.stdout or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid == my_pid:
            continue
        force_kill_pid(pid)


def tcp_keepalive_socket_options() -> List[tuple]:
    """Cross-platform TCP keepalive options for long-lived remote sockets.

    A NAT/VPN gateway that silently drops an idle connection's mapping leaves
    the local socket half-open: without keepalive probes the process only
    learns at the (deliberately long) transport read timeout. Kernel probes
    detect the dead peer within minutes instead.

    Every platform gets ``SO_KEEPALIVE``; the probe-tuning constants are set
    only where the platform exposes them (Linux spells the idle threshold
    ``TCP_KEEPIDLE``, Darwin spells it ``TCP_KEEPALIVE``), each behind a
    ``hasattr`` guard so an older interpreter still gets the safe minimum.
    """
    import socket

    from ouroboros.config import (
        TCP_KEEPALIVE_IDLE_SEC,
        TCP_KEEPALIVE_INTERVAL_SEC,
        TCP_KEEPALIVE_PROBE_COUNT,
    )

    options: List[tuple] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if IS_LINUX:
        if hasattr(socket, "TCP_KEEPIDLE"):
            options.append(
                (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, TCP_KEEPALIVE_IDLE_SEC)
            )
        if hasattr(socket, "TCP_KEEPINTVL"):
            options.append(
                (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, TCP_KEEPALIVE_INTERVAL_SEC)
            )
        if hasattr(socket, "TCP_KEEPCNT"):
            options.append(
                (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, TCP_KEEPALIVE_PROBE_COUNT)
            )
    elif IS_MACOS:
        if hasattr(socket, "TCP_KEEPALIVE"):
            options.append(
                (socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, TCP_KEEPALIVE_IDLE_SEC)
            )
    return options


def kill_process_on_port(port: int) -> None:
    """Kill any process listening on the given TCP port."""
    try:
        if IS_WINDOWS:
            res = _hidden_run(
                ["netstat", "-ano"],
                capture_output=True, text=True, timeout=5,
            )
            for line in res.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.strip().split()
                    if parts:
                        try:
                            pid = int(parts[-1])
                            if pid != os.getpid():
                                _hidden_run(
                                    ["taskkill", "/F", "/PID", str(pid)],
                                    capture_output=True,
                                )
                        except (ValueError, ProcessLookupError, PermissionError):
                            pass
        else:
            # -sTCP:LISTEN scopes the sweep to the listener, mirroring the
            # Windows branch's LISTENING filter: a bare tcp:PORT selector also
            # matches ESTABLISHED client sockets, so on browser-mode installs
            # the sweep would SIGKILL the owner's own browser mid-session.
            # -nP skips host/port name resolution so a slow resolver cannot
            # eat the 5s timeout.
            res = subprocess.run(
                ["lsof", "-nP", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=5,
            )
            for pid_str in res.stdout.strip().split():
                try:
                    pid = int(pid_str)
                    if pid != os.getpid():
                        os.kill(pid, 9)
                except (ValueError, ProcessLookupError, PermissionError):
                    pass
    except Exception:
        pass


# Embedded Python paths.

def embedded_python_candidates(base_dir: pathlib.Path) -> List[pathlib.Path]:
    """Return candidate embedded python-build-standalone paths."""
    if IS_WINDOWS:
        return [
            base_dir / "python-standalone" / "python.exe",
            base_dir / "python-standalone" / "python3.exe",
        ]
    return [
        base_dir / "python-standalone" / "bin" / "python3",
        base_dir / "python-standalone" / "bin" / "python",
    ]


EMBEDDED_PYTHON_DIR_NAME = "python-standalone"


def interpreter_is_embedded(interpreter: str) -> bool:
    """True when ``interpreter`` is the packaged ``python-standalone`` runtime."""
    try:
        return EMBEDDED_PYTHON_DIR_NAME in pathlib.Path(interpreter).resolve().parts
    except (OSError, ValueError):
        return False


def pip_install_target_args(interpreter: str) -> List[str]:
    """Extra pip flags so an install never writes INSIDE the packaged bundle.

    The embedded interpreter lives in the signed bundle, so its own
    ``site-packages`` is the wrong install target: writing there breaks the code
    signature and fails outright on a read-only install. ``--user`` redirects to
    the user site under ``PYTHONUSERBASE`` (set by
    ``launcher_bootstrap.embedded_python_env`` for every process that runs the
    embedded interpreter). A non-embedded interpreter — a dev venv, a system
    python — gets NO flag: ``--user`` is refused inside a virtualenv, so a blanket
    flag would trade one broken install for another.
    """
    return ["--user"] if interpreter_is_embedded(interpreter) else []


def project_venv_python(project_root: pathlib.Path) -> str:
    """Return the executable for a valid project ``.venv`` on this platform.

    Keep the lexical venv path (rather than resolving its symlink) so Python
    discovers the adjacent ``pyvenv.cfg`` and activates the environment.
    """
    env_root = pathlib.Path(project_root) / ".venv"
    if not (env_root / "pyvenv.cfg").is_file():
        return ""
    candidates = (
        (env_root / "Scripts" / "python.exe",)
        if IS_WINDOWS
        else (env_root / "bin" / "python", env_root / "bin" / "python3")
    )
    for candidate in candidates:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return os.path.abspath(os.fspath(candidate))
        except OSError:
            continue
    return ""


def embedded_node_candidates(base_dir: pathlib.Path) -> List[pathlib.Path]:
    """Return candidate bundled Node.js runtime paths."""
    if IS_WINDOWS:
        return [base_dir / "node-standalone" / "node.exe"]
    return [base_dir / "node-standalone" / "bin" / "node"]


def node_distribution_platform() -> str:
    """Return the Node.js archive platform key supported by Ouroboros."""
    machine = platform.machine().strip().lower()
    architecture = {
        "amd64": "x64",
        "x86_64": "x64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(machine, "")
    if IS_WINDOWS:
        return "win32-x64" if architecture == "x64" else ""
    if IS_MACOS and architecture:
        return f"darwin-{architecture}"
    if IS_LINUX and architecture:
        return f"linux-{architecture}"
    return ""


def embedded_ripgrep_candidates(base_dir: pathlib.Path) -> List[pathlib.Path]:
    """Return candidate bundled ripgrep paths."""
    if IS_WINDOWS:
        return [base_dir / "ripgrep-standalone" / "rg.exe"]
    return [base_dir / "ripgrep-standalone" / "bin" / "rg"]


BUNDLE_DIR_ENV = "OUROBOROS_BUNDLE_DIR"


def bundled_resource_ancestor_bases(executable: "str | pathlib.Path | None" = None) -> List[pathlib.Path]:
    """Bundle roots recoverable from an embedded interpreter path.

    Managed updates replace the server checkout but not the frozen launcher.
    Launchers predating ``OUROBOROS_BUNDLE_DIR`` still start the updated server
    with ``.../Resources/python-standalone/...`` (macOS) or
    ``.../_internal/python-standalone/...`` (portable builds).  The interpreter
    path therefore remains a durable, cross-platform pointer to the old app's
    resource root.
    """
    try:
        start = pathlib.Path(executable or sys.executable).resolve()
    except (OSError, ValueError):
        return []
    found: List[pathlib.Path] = []
    chain = [start.parent, *start.parents]
    for ancestor in chain:
        if ancestor.name == EMBEDDED_PYTHON_DIR_NAME:
            found.append(ancestor.parent)
        for child_name in ("Resources", "_internal"):
            candidate = ancestor / child_name
            try:
                if any(path.is_file() for path in embedded_python_candidates(candidate)):
                    found.append(candidate)
            except OSError:
                continue
    return found


def bundled_resource_bases() -> List[pathlib.Path]:
    """Roots to search for a resource shipped INSIDE the packaged bundle.

    SSOT for every bundled-payload lookup, because the process that consumes a
    bundled payload is usually NOT the frozen launcher. In a packaged install the
    launcher runs the server/CLI as a SEPARATE child of the embedded interpreter,
    out of the launcher-managed repo under the data dir: that child has no
    ``sys._MEIPASS`` and its ``__file__`` parent is the managed repo, so both
    historical bases miss and every bundled payload silently reads as absent.
    The launcher therefore hands the bundle root down by value in
    ``OUROBOROS_BUNDLE_DIR`` and it is searched FIRST. The other two bases stay
    for the frozen process itself and for the dev/source layout (payloads sit at
    the repo root, two levels up from this module).
    """
    bases: List[pathlib.Path] = []
    env_base = str(os.environ.get(BUNDLE_DIR_ENV) or "").strip()
    if env_base:
        bases.append(pathlib.Path(env_base))
    frozen_base = getattr(sys, "_MEIPASS", None)
    if frozen_base:
        bases.append(pathlib.Path(frozen_base))
    bases.extend(bundled_resource_ancestor_bases())
    bases.append(pathlib.Path(__file__).resolve().parent.parent)
    unique: List[pathlib.Path] = []
    seen = set()
    for base in bases:
        try:
            key = base.resolve()
        except OSError:
            key = base
        if key in seen:
            continue
        seen.add(key)
        unique.append(base)
    return unique


def _resolve_bundled_payload(candidates_for: Callable[[pathlib.Path], List[pathlib.Path]]) -> Optional[str]:
    """First existing candidate across the bundle bases, or None."""
    for base in bundled_resource_bases():
        for candidate in candidates_for(base):
            try:
                if candidate.is_file():
                    return str(candidate)
            except OSError:
                continue
    return None


def resolve_bundled_node() -> Optional[str]:
    """Return the path to the bundled, signed Node.js runtime if present.

    The packaged app ships an official notarized node under ``node-standalone``
    (re-signed under the hardened runtime by the build's signing pass). Prefer it
    over a PATH (e.g. Homebrew) node, which macOS code-signing enforcement can
    SIGKILL when launched from the packaged process tree.
    """
    return _resolve_bundled_payload(embedded_node_candidates)


def resolve_bundled_ripgrep() -> Optional[str]:
    """Return the bundled rg path if present."""
    return _resolve_bundled_payload(embedded_ripgrep_candidates)


def get_system_memory() -> str:
    """Return total system memory as a human-readable string."""
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            mem_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
            ).strip())
            return f"{mem_bytes / (1024**3):.1f} GB"
        elif os_name == "Linux":
            out = subprocess.check_output(
                ["awk", '/MemTotal/ {print $2/1024/1024 " GB"}', "/proc/meminfo"],
            ).strip().decode()
            return out
        elif os_name == "Windows":
            out = _hidden_run(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/value"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            for line in out.splitlines():
                if "=" in line:
                    mem_bytes = int(line.split("=")[1])
                    return f"{mem_bytes / (1024**3):.1f} GB"
    except Exception:
        pass
    return "Unknown"


def get_cpu_info() -> str:
    """Return CPU model string."""
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
            ).strip().decode()
        elif os_name == "Windows":
            out = _hidden_run(
                ["wmic", "cpu", "get", "Name", "/value"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            for line in out.splitlines():
                if "=" in line:
                    return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


# Process session isolation.

def create_new_session() -> None:
    """Create a new process session (Unix: setsid). No-op on Windows."""
    if not IS_WINDOWS:
        os.setsid()


def subprocess_new_group_kwargs() -> dict:
    """Return subprocess kwargs for killable process-group/session isolation."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def install_shutdown_signal_handlers(handler) -> None:
    """Register ``handler`` for the signals that ask a console process to shut
    down: SIGINT everywhere, SIGTERM on POSIX. The platform-specific signal
    surface lives HERE, never in callers (checklist 15). The handler must only
    set a flag/event — real teardown belongs on the caller's main thread, not
    inside a signal frame."""
    signal.signal(signal.SIGINT, handler)
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, handler)


def subprocess_hidden_kwargs() -> dict:
    """Return kwargs to suppress Windows console windows."""
    if IS_WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def merge_hidden_kwargs(kwargs: dict) -> dict:
    """Merge Windows hidden-window flags without dropping caller flags."""
    hidden = subprocess_hidden_kwargs()
    if not hidden:
        return dict(kwargs)
    result = dict(kwargs)
    result["creationflags"] = result.get("creationflags", 0) | hidden.get("creationflags", 0)
    return result


# Git installation hint.

def git_install_hint() -> str:
    """Return platform-appropriate instructions for installing Git."""
    if IS_MACOS:
        return "Install Git via Xcode CLI Tools: xcode-select --install"
    elif IS_WINDOWS:
        return "Download Git from https://git-scm.com/download/win or run: winget install Git.Git"
    else:
        return "Install Git via your package manager, e.g.: sudo apt install git"


# Windows Job Object helpers.

if IS_WINDOWS:
    import ctypes
    import ctypes.wintypes

    # `use_last_error=True` so `ctypes.get_last_error()` reads the code the CALL set: without it
    # ctypes does not snapshot the thread's last error, and the failure text below would quote
    # whatever ctypes' own bookkeeping left behind. Same pattern as the file-lock helpers above.
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]

    # Explicit ABI declarations: without restype=HANDLE, ctypes truncates 64-bit
    # HANDLEs to c_int — a job/process handle above 2^31 comes back corrupted and
    # every later Job Object call silently operates on garbage.
    _kernel32.CreateJobObjectW.restype = ctypes.wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = (ctypes.wintypes.LPVOID, ctypes.wintypes.LPCWSTR)
    _kernel32.SetInformationJobObject.restype = ctypes.wintypes.BOOL
    _kernel32.SetInformationJobObject.argtypes = (
        ctypes.wintypes.HANDLE, ctypes.c_int, ctypes.wintypes.LPVOID, ctypes.wintypes.DWORD,
    )
    _kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = (ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD)
    _kernel32.AssignProcessToJobObject.restype = ctypes.wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = (ctypes.wintypes.HANDLE, ctypes.wintypes.HANDLE)
    _kernel32.TerminateJobObject.restype = ctypes.wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = (ctypes.wintypes.HANDLE, ctypes.wintypes.UINT)
    _kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (ctypes.wintypes.HANDLE,)

    # .value, not the HANDLE instance: with restype=HANDLE the calls return plain
    # ints (or None for NULL), and an int never equals a ctypes instance.
    _INVALID_HANDLE_VALUE = ctypes.wintypes.HANDLE(-1).value
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JOBOBJECTINFOCLASS_EXTENDED = 9
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SUSPEND_RESUME = 0x0800
    _CREATE_SUSPENDED = 0x4

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", ctypes.wintypes.DWORD),
            ("SchedulingClass", ctypes.wintypes.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _ExtendedLimitInfo(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def create_kill_on_close_job() -> Optional[Any]:
    """Create a Windows kill-on-close Job Object, or None."""
    if not IS_WINDOWS:
        return None
    try:
        handle = _kernel32.CreateJobObjectW(None, None)
        if not handle or handle == _INVALID_HANDLE_VALUE:
            log.warning("CreateJobObjectW failed")
            return None
        info = _ExtendedLimitInfo()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = _kernel32.SetInformationJobObject(
            handle,
            _JOBOBJECTINFOCLASS_EXTENDED,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            log.warning("SetInformationJobObject failed")
            _kernel32.CloseHandle(handle)
            return None
        return handle
    except Exception as exc:
        log.warning("Job Object creation failed: %s", exc)
        return None


def assign_pid_to_job(job_handle: Any, pid: int) -> bool:
    """Assign a running process (by PID) to a Job Object. Windows only."""
    if not IS_WINDOWS or job_handle is None:
        return False
    try:
        proc_handle = _kernel32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid,
        )
        if not proc_handle:
            log.warning("OpenProcess(%d) failed for Job Object assignment", pid)
            return False
        ok = _kernel32.AssignProcessToJobObject(job_handle, proc_handle)
        _kernel32.CloseHandle(proc_handle)
        if not ok:
            log.warning("AssignProcessToJobObject failed for pid %d", pid)
            return False
        return True
    except Exception as exc:
        log.warning("Job Object assign failed: %s", exc)
        return False


def terminate_job(job_handle: Any, exit_code: int = 1) -> str:
    """Terminate all processes in a Job Object; "" on success, else the reason it is unproven.

    A FALSE Win32 BOOL is a failure exactly like a raised call, and swallowing either let
    ``ProcessContainer.reap`` report a clean teardown while job members were still running."""
    if not IS_WINDOWS or job_handle is None:
        return ""
    try:
        if not _kernel32.TerminateJobObject(job_handle, exit_code):
            return (f"TerminateJobObject returned false (Win32 error {ctypes.get_last_error()}), "
                    "so the processes held by the job cannot be assumed dead")
    except Exception as exc:
        return f"TerminateJobObject failed ({exc}), so the job's processes are unaccounted for"
    return ""


def close_job(job_handle: Any) -> str:
    """Close a Job Object handle (triggers kill-on-close if set); "" on success, else the reason.

    The handle is the last thing holding kill-on-close, so a close that did not happen leaves
    survivors AND leaks the handle; the caller reports it rather than discarding it."""
    if not IS_WINDOWS or job_handle is None:
        return ""
    try:
        if not _kernel32.CloseHandle(job_handle):
            return (f"CloseHandle on the Job Object returned false (Win32 error "
                    f"{ctypes.get_last_error()}), so kill-on-close never fired")
    except Exception as exc:
        return f"CloseHandle on the Job Object failed ({exc}), so kill-on-close never fired"
    return ""


def resume_process(pid: int) -> bool:
    """Resume all threads of a suspended process. Windows only."""
    if not IS_WINDOWS:
        return False
    try:
        _ntdll = ctypes.windll.ntdll  # type: ignore[attr-defined]
        # Same 64-bit ABI rule as the kernel32 block: an undeclared HANDLE
        # argument is truncated to c_int, corrupting handles above 2^31.
        _ntdll.NtResumeProcess.restype = ctypes.c_int32
        _ntdll.NtResumeProcess.argtypes = (ctypes.wintypes.HANDLE,)
        handle = _kernel32.OpenProcess(_PROCESS_SUSPEND_RESUME, False, pid)
        if not handle:
            log.warning("OpenProcess(%d) failed for resume", pid)
            return False
        status = _ntdll.NtResumeProcess(handle)
        _kernel32.CloseHandle(handle)
        if status != 0:
            log.warning("NtResumeProcess(%d) returned NTSTATUS 0x%08x", pid, status)
            return False
        return True
    except Exception as exc:
        log.warning("resume_process failed: %s", exc)
        return False


# Node runtime health/policy moved to ouroboros/node_runtime.py (its own module:
# the policy grew past what a cross-platform primitives file should hold, and
# the 1600-line module gate agrees). The re-export is a PEP 562 module
# __getattr__ rather than an eager from-import: node_runtime itself imports
# this module at module level, and an eager import back from HERE re-entered a
# partially initialized node_runtime whenever node_runtime was imported first
# (triad finding, all three phase-C reviewers). Lazy resolution keeps both
# import orders sound while every existing importer (preflight_node, skill
# surfaces, the interpreter resolver, claudexor_runtime) keeps its
# `from ouroboros.platform_layer import <name>` spelling unchanged.
_NODE_RUNTIME_REEXPORTS = (
    "NodeRuntimeHealth",
    "node_runtime_health",
    "probe_node_version",
    "select_skill_node_runtime",
    "skill_node_emergency_path_dir",
)


def __getattr__(name: str):
    if name in _NODE_RUNTIME_REEXPORTS:
        from ouroboros import node_runtime

        return getattr(node_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Cross-platform process, locking, path, and runtime helpers."""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import pathlib
import platform
import re
import signal
import subprocess
import sys
import threading
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

# POSIX signal numbers -> names, the same on every platform. ``signal.Signals``
# knows only the host's own signals, so a child that died of SIGKILL would be
# named "SIG9" wherever the reader lacks that signal (Windows).
_POSIX_SIGNAL_NAMES = {
    1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 6: "SIGABRT", 9: "SIGKILL",
    11: "SIGSEGV", 13: "SIGPIPE", 14: "SIGALRM", 15: "SIGTERM",
}


def posix_signal_name(signum: int) -> str:
    """Name of a POSIX signal number (``SIGKILL``); ``SIG<n>`` when unknown."""
    number = int(signum)
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = ""
    if not name or re.fullmatch(r"SIG\d+", name):
        name = _POSIX_SIGNAL_NAMES.get(number, name or f"SIG{number}")
    return name


def executable_name_candidates(name: str) -> List[str]:
    """Platform spellings for an executable stored in a known directory."""
    base = str(name or "").strip()
    if not base:
        return []
    if IS_WINDOWS:
        return [f"{base}.cmd", f"{base}.exe", f"{base}.bat", base]
    return [base]


def local_zoneinfo():
    """Best-effort DST-aware local timezone: the IANA zone from ``TZ`` or
    ``/etc/localtime``, else the fixed offset (which drifts across DST)."""
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
    # /.dockerenv is Docker's Linux sentinel.
    return os.environ.get("OUROBOROS_CONTAINER") == "1" or (
        IS_LINUX and pathlib.Path("/.dockerenv").exists()
    )


def bootstrap_process_path() -> list[str]:
    """Add existing common user tool directories to this process PATH once."""

    global _PATH_BOOTSTRAPPED
    if _PATH_BOOTSTRAPPED:
        return []
    _PATH_BOOTSTRAPPED = True

    candidates: list[pathlib.Path] = []
    home = pathlib.Path.home()
    if IS_MACOS or IS_LINUX:
        candidates.extend(pathlib.Path(p) for p in (
            "/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin",
            "/usr/local/sbin", "/opt/local/bin",
        ))
        candidates.extend((
            home / ".local" / "bin", home / ".cargo" / "bin",
            home / ".npm-global" / "bin", home / "go" / "bin",
        ))
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
    """Copy of *env* without ``PYTHONPATH`` entries resolving to the system repo.

    An EXTERNAL-workspace command inheriting that entry resolves its own
    ``import web``/``server``/``ouroboros`` to OUROBOROS's modules; dropping
    ONLY the repo entry isolates the target (no-op without one)."""
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


def _lock_identity(target: "int | pathlib.Path") -> tuple:
    """``(ino, dev, mtime_ns)`` of a lock file by descriptor or path; ``()`` if gone."""
    try:
        info = os.fstat(target) if isinstance(target, int) else os.stat(str(target))
    except OSError:
        return ()
    return (info.st_ino, info.st_dev, info.st_mtime_ns)


# Contention retries; unsupported locks/ENOLCK select the recorded name tier.
# Other errors fail closed (ARCHITECTURE §1 "Platform substrate").
_LOCK_HELD_ERRNOS = frozenset({errno.EAGAIN, errno.EWOULDBLOCK})
_LOCK_UNSUPPORTED_ERRNOS = frozenset({errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOSYS})
_WIN32_LOCK_ERRNOS = {33: errno.EAGAIN, 1: errno.ENOSYS, 50: errno.EOPNOTSUPP}  # violation = held; invalid function / not supported = no byte-range locks here
_KERNEL_LOCK_TIER: dict = {}  # lock directory -> (kernel locks enforced there, the errno that selected the name tier or None)
_KERNEL_LOCK_TIER_LOCK = threading.Lock()  # one probe per directory, one verdict for every thread


def kernel_file_locks_enforced(lock_path: pathlib.Path) -> bool:
    """Probe and cache kernel-lock capability per directory, never on a live refusal.

    Unprobeable directories stay enforced and retry later; unsupported/ENOLCK
    select the recorded name-only tier. See ARCHITECTURE §1 "Platform substrate"."""
    directory = os.path.realpath(str(pathlib.Path(lock_path).parent))
    with _KERNEL_LOCK_TIER_LOCK:
        tier, refused = _KERNEL_LOCK_TIER.get(directory, (None, None))
        if tier is None:
            probe = os.path.join(directory, f".kernel-lock-probe.{os.getpid()}.{time.time_ns()}")
            try:
                fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            except OSError:
                log.debug("Cannot probe kernel lock support under %s", directory, exc_info=True)
                return True  # undecided: enforced now, probed again next time
            try:
                file_lock_exclusive_nb(fd)
                file_unlock(fd)
                tier, refused = True, None
            except OSError as exc:
                refused, tier = exc.errno, exc.errno not in _LOCK_UNSUPPORTED_ERRNOS and exc.errno != errno.ENOLCK
            finally:
                os.close(fd)
                try:
                    os.unlink(probe)
                except OSError:
                    log.debug("Kernel-lock probe %s left behind", probe, exc_info=True)
            if not tier:
                log.warning("No kernel file locks under %s (errno %s): locks there use the name protocol only", directory, refused)
            _KERNEL_LOCK_TIER[directory] = (tier, refused)
    return tier


def acquire_exclusive_file_lock(
    lock_path: pathlib.Path,
    *,
    timeout_sec: float = 4.0,
    stale_sec: float = 90.0,
    metadata: str = "",
    poll_sec: float = 0.05,
    owner_aware_stale: bool = False,
    refuse_name_tier_errnos: frozenset = frozenset(),
) -> Optional[int]:
    """Acquire a descriptor-owned lock whose path still names the held inode.

    Kernel errors fail closed except contention; owner-aware stale recovery
    cannot evict a live writer. Name-tier refusal is caller policy. Platform
    eviction/release ordering and limitations: ARCHITECTURE §1 "Platform substrate"."""
    lock_path = pathlib.Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    enforced = kernel_file_locks_enforced(lock_path)
    if not enforced and _KERNEL_LOCK_TIER.get(os.path.realpath(str(lock_path.parent)), (False, None))[1] in refuse_name_tier_errnos:
        log.warning("Name-tier lock refused by caller policy at %s: no lock taken", lock_path)
        return None
    started = time.time()
    while (time.time() - started) < timeout_sec:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            stamp = (metadata or f"pid={os.getpid()} ts={time.time()}\n").encode("utf-8")
            try:  # the owner pid goes in BEFORE the kernel lock, so an owner-aware
                os.write(fd, stamp)
            except Exception:  # reclaimer never judges a live creator's fresh file empty
                log.debug("Failed to write lock metadata to %s", lock_path, exc_info=True)
            try:
                if enforced:
                    file_lock_exclusive_nb(fd)
            except OSError as exc:
                if exc.errno not in _LOCK_HELD_ERRNOS:
                    release_exclusive_file_lock(lock_path, fd)  # ours, yet never a hold
                    log.warning("Kernel lock refused at %s (errno %s): no lock taken", lock_path, exc.errno)
                    return None
            else:
                # Only a readable fd/path identity proves the creator still owns this name.
                won = _lock_identity(fd)[:2]
                if won and won == _lock_identity(lock_path)[:2]:
                    return fd
                if not won:
                    with contextlib.suppress(OSError):
                        if lock_path.read_bytes() == stamp:
                            os.unlink(str(lock_path))
                    os.close(fd)
                    log.warning("Lock identity unreadable at %s: no lock taken", lock_path)
                    return None
            if IS_WINDOWS:  # a lock goes before its handle (see _win32_unlock)
                file_unlock(fd)
            os.close(fd)  # the file we created was kernel-locked by a racing
            time.sleep(poll_sec)  # evictor's probe, or evicted: the name alone is
            continue  # not ownership — stand down and re-contend
        except (FileExistsError, PermissionError):
            stale = refused = None
            try:
                probe = os.open(str(lock_path), os.O_RDONLY)
                try:
                    judged = _lock_identity(probe)
                    owner_pid = 0
                    for field in os.read(probe, 512).decode("utf-8", "replace").split():
                        if field.startswith("pid=") and field[4:].isdigit():
                            owner_pid = int(field[4:])
                    stale = bool(judged) and (time.time() - judged[2] / 1e9) > stale_sec
                    if judged and owner_aware_stale and owner_pid > 0:
                        stale = not pid_is_alive(owner_pid)  # Proven death needs no age grace.
                    # Judge and evict the same inode under a kernel hold.
                    if stale and enforced:
                        try:
                            file_lock_exclusive_nb(probe)
                        except OSError as exc:  # a live kernel hold, or a refusal:
                            stale = False  # either way, never evict without the hold
                            refused = None if exc.errno in _LOCK_HELD_ERRNOS else exc
                        else:
                            if not IS_WINDOWS:  # Windows deletes no open file: it unlinks
                                if _lock_identity(lock_path) == judged:  # below, after
                                    os.unlink(str(lock_path))  # under the held flock
                                continue
                finally:
                    if IS_WINDOWS:  # the probe's hold goes before its handle
                        file_unlock(probe)
                    os.close(probe)
                if stale and _lock_identity(lock_path) == judged:
                    lock_path.unlink()
                    continue
            except Exception:
                log.debug("Failed to inspect/remove stale lock %s", lock_path, exc_info=True)
            if refused is not None:
                log.warning("Kernel lock refused on stale %s (%s): no lock taken", lock_path, refused)
                return None
            time.sleep(poll_sec)
        except Exception:
            log.warning("Failed to acquire lock at %s", lock_path, exc_info=True)
            break
    return None


def refresh_exclusive_file_lock(lock_path: pathlib.Path, lock_fd: Optional[int]) -> bool:
    """Refresh only our held inode; False requires abandoning protected work.

    Extends the stale-age clock without renewing a stolen/replaced lock."""
    if lock_fd is None:
        return False
    held = _lock_identity(lock_fd)
    if not held or held[:2] != _lock_identity(lock_path)[:2]:
        return False
    try:
        os.utime(lock_fd if os.utime in getattr(os, "supports_fd", ()) else str(lock_path))
    except OSError:
        log.debug("Failed to refresh lock %s", lock_path, exc_info=True)
        return False
    return True


def _unlink_lock_path(lock_path: pathlib.Path, held: Optional[tuple]) -> None:
    """Unlink only the held identity (None means unconditional path-only cleanup).

    Retry transient Windows sharing refusal within the bounded release window;
    leave a replaced path alone. See ARCHITECTURE §1 "Platform substrate"."""
    deadline = time.monotonic() + 2.0
    while True:
        try:
            if held is None or (held and _lock_identity(lock_path)[:2] == held):
                os.unlink(str(lock_path))
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if not IS_WINDOWS or time.monotonic() >= deadline:
                log.debug("Failed to unlink lock file %s", lock_path, exc_info=True)
                return
            time.sleep(0.005)
        except Exception:
            log.debug("Failed to unlink lock file %s", lock_path, exc_info=True)
            return


def release_exclusive_file_lock(lock_path: pathlib.Path, lock_fd: Optional[int]) -> None:
    """Release only this descriptor's lock path, never a successor's.

    POSIX unlinks under the hold; Windows unlocks/closes before identity-checked
    unlink. See ARCHITECTURE §1 "Platform substrate" for the sharing gap."""
    lock_path = pathlib.Path(lock_path)
    if lock_fd is None:
        return
    held = _lock_identity(lock_fd)[:2]
    if IS_WINDOWS:
        file_unlock(lock_fd)
    else:
        _unlink_lock_path(lock_path, held)
    try:
        os.close(lock_fd)
    except Exception:
        log.debug("Failed to close lock fd %s for %s", lock_fd, lock_path, exc_info=True)
    if IS_WINDOWS:
        _unlink_lock_path(lock_path, held)


def unlink_lockfile(lock_path: pathlib.Path) -> None:
    """Best-effort cleanup for path-only locks whose fd was closed after acquire
    (the same transient Windows refusal is retried, see :func:`_unlink_lock_path`)."""
    _unlink_lock_path(pathlib.Path(lock_path), None)


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
    return any("AppTranslocation" in p or p.startswith("/Volumes/") for p in (raw, resolved))


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
        ctypes.windll.user32.SendMessageTimeoutW(  # HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG
            0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, ctypes.byref(result),
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
        file_lock_exclusive_nb(fd_obj.fileno())
        fd_obj.write(str(os.getpid()))
        fd_obj.flush()
        # Promote to global only after lock and PID write both succeed.
        _lock_fd = fd_obj
        return True
    except OSError:
        if fd_obj is not None:
            with contextlib.suppress(Exception):
                fd_obj.close()
        return False


def pid_lock_release(path: str) -> None:
    """Release the PID lock."""
    global _lock_fd
    if _lock_fd is not None:
        with contextlib.suppress(Exception):
            file_unlock(_lock_fd.fileno())
        with contextlib.suppress(Exception):
            _lock_fd.close()
        _lock_fd = None
    with contextlib.suppress(Exception):
        os.unlink(path)


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
    """Observe process presence; access denial remains alive, not signal authority.

    Windows uses OpenProcess/GetExitCodeProcess, never a signal-zero probe."""

    if pid <= 0:
        return False
    if IS_WINDOWS:
        # os.kill(pid, 0) is WRONG here: signal 0 is CTRL_C_EVENT, delivered to the pid's
        # whole console group. Probe with OpenProcess + GetExitCodeProcess, which never signals anything.
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
            # A live but access-protected process reads as alive; anything else (invalid parameter -> no such pid) reads as dead.
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
    except ProcessLookupError:
        return False
    except OSError:  # EPERM: it exists and refuses us; anything else undeterminable reads as present
        pass
    return True


def pid_is_signalable(pid: int) -> bool:
    """Whether this caller can signal a PID: POSIX signal-zero, Windows presence.

    This is distinct from identity/ownership and from an access-denied live PID."""
    if pid <= 0:
        return False
    if IS_WINDOWS:
        return pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def pid_provably_gone(pid: int) -> bool:
    """Positive absence from the platform presence reader; denial is not death."""
    return not pid_is_alive(pid)


# Windows locking via LockFileEx: unlike msvcrt.locking(), works on empty files.

# Keep mandatory LockFileEx bytes beyond the readable owner stamp.
# Range/release rationale: ARCHITECTURE §1 "Platform substrate".
_WIN32_LOCK_OFFSET = 0x7FFFFFFF00000000
_WIN32_LOCK_LENGTH = 1


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
    """Lock the fixed out-of-stamp byte range with LockFileEx (including empty files)."""
    import ctypes
    from ctypes import wintypes
    import msvcrt as _msvcrt

    _LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
    _LOCKFILE_EXCLUSIVE_LOCK = 0x00000002

    OVERLAPPED = _win32_overlapped_class()

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(OVERLAPPED)]
    kernel32.LockFileEx.restype = wintypes.BOOL

    hfile = _msvcrt.get_osfhandle(fd)
    flags = (_LOCKFILE_EXCLUSIVE_LOCK if exclusive else 0) | (0 if blocking else _LOCKFILE_FAIL_IMMEDIATELY)

    ov = OVERLAPPED()
    ov.Offset, ov.OffsetHigh = _WIN32_LOCK_OFFSET & 0xFFFFFFFF, _WIN32_LOCK_OFFSET >> 32
    if not kernel32.LockFileEx(hfile, flags, 0, _WIN32_LOCK_LENGTH, 0, ctypes.byref(ov)):
        raise _win32_lock_error(ctypes.get_last_error())


def _win32_lock_error(err: int) -> OSError:
    """Map contention/unsupported Win32 codes explicitly; preserve all other errors.

    Four-argument OSError derives errno from winerror, hence the explicit mapping."""
    code = _WIN32_LOCK_ERRNOS.get(err)
    if code is None:
        return OSError(0, f"LockFileEx failed (error {err})", None, err)
    refused = OSError(code, f"LockFileEx refused (error {err})")
    refused.winerror = err  # kept for diagnostics; the errno carries the verdict
    return refused


def _win32_unlock(fd: int) -> None:
    """Release the fixed range best-effort before closing the handle.

    Rebuild OVERLAPPED; no recycled-fd map. ERROR_NOT_LOCKED needs no recovery."""
    import ctypes
    from ctypes import wintypes
    import msvcrt as _msvcrt

    OVERLAPPED = _win32_overlapped_class()

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.DWORD, ctypes.POINTER(OVERLAPPED)]
    kernel32.UnlockFileEx.restype = wintypes.BOOL

    ov = OVERLAPPED()
    ov.Offset, ov.OffsetHigh = _WIN32_LOCK_OFFSET & 0xFFFFFFFF, _WIN32_LOCK_OFFSET >> 32
    with contextlib.suppress(Exception):
        kernel32.UnlockFileEx(_msvcrt.get_osfhandle(fd), 0, _WIN32_LOCK_LENGTH, 0, ctypes.byref(ov))


# Process management.

def kill_process_tree(proc: subprocess.Popen, *, exclude_pids: "set[int] | None" = None) -> None:
    """Capture descendants before termination and spare retained branches.

    POSIX kills the group only when it contains no spared PID, then escaped
    descendants. Windows uses selective PID termination when exclusions exist."""
    pid = proc.pid
    if IS_WINDOWS and not exclude_pids:
        try:
            _hidden_run(["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=10)
        except Exception:
            pass
        return
    targets, spared = _tree_kill_targets(pid, exclude_pids)
    if not IS_WINDOWS:
        pgid = process_group_id(pid)
        if pgid > 0 and not any(process_group_id(p) == pgid for p in spared):
            kill_process_group_id(pgid)
    for dpid in targets:
        force_kill_pid(dpid)


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


def terminate_process_group_id(pgid: int, *, exclude_pids: "set[int] | None" = None) -> None:
    """Gracefully terminate a Unix process group by id."""
    if IS_WINDOWS:
        return
    if _group_has_spared_process(pgid, exclude_pids):
        return
    try:
        os.killpg(int(pgid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError, ValueError):
        pass


def kill_process_group_id(pgid: int, *, exclude_pids: "set[int] | None" = None) -> None:
    """Force-kill a Unix process group by id."""
    if IS_WINDOWS:
        return
    if _group_has_spared_process(pgid, exclude_pids):
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


def _group_has_spared_process(pgid: int, roots: "set[int] | None") -> bool:
    for pid in roots or ():
        if any(process_group_id(p) == pgid for p in [pid, *collect_descendant_pids(pid)]):
            return True
    return False


_BOOT_ID = ""  # full hex of the /proc boot id; empty until a successful read, then latched


def process_start_time(pid: int) -> str:
    """Stable birth token: Linux boot-qualified ticks, legacy fallback, or Windows FILETIME.

    Mint order, cross-boot limits and downgrade-compatible ledger fields:
    ARCHITECTURE §1 "Platform substrate"."""
    global _BOOT_ID
    if pid <= 0:
        return ""
    if IS_WINDOWS:
        return _windows_process_start_time(pid)
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
    """Legacy ps wall-clock token (bare ticks as last resort); Windows FILETIME.

    Used for downgrade-safe ledger writes and _legacy_start_matches comparison."""
    if pid <= 0:
        return ""
    if IS_WINDOWS:
        return _windows_process_start_time(pid)
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
    """Return the live command line, canonically quoting native Windows argv."""
    if IS_WINDOWS:
        try:
            import psutil

            return subprocess.list2cmdline(psutil.Process(int(pid)).cmdline())
        except Exception:
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
            os.kill(pid, getattr(signal, "SIGKILL", 9))  # spelled portably: a test may drive this branch on Windows
        except (ProcessLookupError, PermissionError, OSError):
            pass


def kill_pid_tree(pid: int, exclude_pids: "set[int] | None" = None) -> None:
    """Kill a captured PID tree, sparing excluded roots and their descendants.

    The caller owns daemon/service retention policy; this helper only signals."""
    if IS_WINDOWS and not exclude_pids:
        try:
            _hidden_run(["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=10)
        except Exception:
            pass
        return

    targets, _ = _tree_kill_targets(pid, exclude_pids)
    for dpid in targets:
        force_kill_pid(dpid)


def _tree_kill_targets(pid: int, exclude_pids: "set[int] | None") -> tuple[list[int], set[int]]:
    """Capture before signalling; a spared root keeps its entire branch alive."""
    exclude = {int(p) for p in (exclude_pids or ())}
    if IS_WINDOWS:
        children = _windows_process_children()
        descendants = _snapshot_descendants(pid, children)
        spared = exclude | {p for root in exclude for p in _snapshot_descendants(root, children)}
        return [p for p in [*descendants, pid] if p not in spared], spared
    descendants = collect_descendant_pids(pid)
    spared: set[int] = set()
    for ep in exclude:
        spared.add(ep)
        spared.update(collect_descendant_pids(ep))
    return [p for p in [*descendants, pid] if p not in spared], spared


def _windows_process_children() -> dict[int, list[int]]:
    """One PID/PPID observation for both the target and retained subtrees."""
    import psutil

    children: dict[int, list[int]] = {}
    for process in psutil.process_iter(["pid", "ppid"]):
        parent = process.info.get("ppid")
        if parent is not None:
            children.setdefault(int(parent), []).append(process.pid)
    return children


def _snapshot_descendants(pid: int, children: dict[int, list[int]]) -> list[int]:
    """Children before parents, excluding the root itself."""
    result: list[int] = []
    seen = {pid}

    def visit(parent: int) -> None:
        for child in children.get(parent, ()):
            if child not in seen:
                seen.add(child)
                visit(child)
                result.append(child)

    visit(pid)
    return result


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


def collect_descendant_pids(pid: int, *, exclude_pids: "set[int] | None" = None) -> List[int]:
    """Descendant PIDs in postorder, excluding retained branches when requested."""
    if exclude_pids:
        targets, _ = _tree_kill_targets(int(pid), exclude_pids)
        return [target for target in targets if target != int(pid)]
    if IS_WINDOWS:
        return _snapshot_descendants(int(pid), _windows_process_children())
    result: List[int] = []
    try:
        _collect_descendants(int(pid), result)
    except (TypeError, ValueError):
        pass
    return result


def tcp_keepalive_socket_options() -> List[tuple]:
    """Platform-guarded TCP keepalive options from config; unknown options are omitted.

    Dead-socket rationale and per-platform fallback: ARCHITECTURE §6 transport."""
    import socket

    from ouroboros.config import (
        TCP_KEEPALIVE_IDLE_SEC,
        TCP_KEEPALIVE_INTERVAL_SEC,
        TCP_KEEPALIVE_PROBE_COUNT,
    )

    options: List[tuple] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle_name = "TCP_KEEPIDLE" if IS_LINUX else ("TCP_KEEPALIVE" if IS_MACOS else "")
    if not idle_name:
        return options
    for name, value in (
        (idle_name, TCP_KEEPALIVE_IDLE_SEC),
        ("TCP_KEEPINTVL", TCP_KEEPALIVE_INTERVAL_SEC),
        ("TCP_KEEPCNT", TCP_KEEPALIVE_PROBE_COUNT),
    ):
        if hasattr(socket, name):
            options.append((socket.IPPROTO_TCP, getattr(socket, name), value))
    return options


def kill_process_on_port(port: int) -> None:
    """Kill any process listening on the given TCP port."""
    try:
        if IS_WINDOWS:
            res = _hidden_run(["netstat", "-ano"], capture_output=True, text=True, timeout=5)
            listeners = [
                line.split()[-1] for line in res.stdout.splitlines()
                if f":{port}" in line and "LISTENING" in line and line.split()
            ]
        else:
            # -sTCP:LISTEN scopes the sweep to the listener (a bare tcp:PORT
            # selector also matches ESTABLISHED client sockets — on browser
            # installs that would SIGKILL the owner's browser mid-session);
            # -nP skips name resolution so a slow resolver can't eat the 5s.
            res = subprocess.run(
                ["lsof", "-nP", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=5,
            )
            listeners = res.stdout.split()
        for pid_str in listeners:
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid != os.getpid():
                force_kill_pid(pid)
    except Exception:
        pass


# Embedded Python paths.

def embedded_python_candidates(base_dir: pathlib.Path) -> List[pathlib.Path]:
    """Return candidate embedded python-build-standalone paths."""
    root = base_dir / "python-standalone"
    if IS_WINDOWS:
        return [root / "python.exe", root / "python3.exe"]
    return [root / "bin" / "python3", root / "bin" / "python"]


EMBEDDED_PYTHON_DIR_NAME = "python-standalone"


def interpreter_is_embedded(interpreter: str) -> bool:
    """True when ``interpreter`` is the packaged ``python-standalone`` runtime."""
    try:
        return EMBEDDED_PYTHON_DIR_NAME in pathlib.Path(interpreter).resolve().parts
    except (OSError, ValueError):
        return False


def pip_install_target_args(interpreter: str) -> List[str]:
    """Use the embedded interpreter's userbase; never add --user for a dev venv.

    Bundle-signature and target policy: ARCHITECTURE §1 CLI / Headless Boundary."""
    invocation = pathlib.Path(interpreter)
    if invocation.parent.name.lower() in {"bin", "scripts"} and (
        invocation.parent.parent / "pyvenv.cfg"
    ).is_file():
        return []  # Resolve only after Python's lexical venv selection is known.
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
    architecture = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, "")
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
    """Recover bundle roots from embedded-interpreter ancestors for older launchers.

    Search Resources/_internal too; managed checkout is not the packaged root."""
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
    """Resource lookup SSOT: explicit bundle root, frozen root, ancestors, source.

    Managed child and old-launcher rationale: ARCHITECTURE §1 CLI / Headless Boundary."""
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
    """Locate the bundled signed Node; callers own health and PATH preference (§1)."""
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


def subprocess_new_group_kwargs(*, breakaway_from_job: bool = False, suspended: bool = False) -> dict:
    """Return subprocess kwargs for killable process-group/session isolation."""
    if IS_WINDOWS:
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
        if breakaway_from_job:
            flags |= _windows_breakaway_flags()
        if suspended:
            flags |= getattr(subprocess, "CREATE_SUSPENDED", 0x4)
        return {"creationflags": flags}
    return {"start_new_session": True}


def install_shutdown_signal_handlers(handler) -> None:
    """Install SIGINT and POSIX SIGTERM handlers that only set a flag/event.

    The caller's main thread owns teardown, never the signal frame."""
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

    # Snapshot the actual call error; declare full-width HANDLE ABI (ARCHITECTURE §1).
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]

    _kernel32.CreateJobObjectW.restype = ctypes.wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = (ctypes.wintypes.LPVOID, ctypes.wintypes.LPCWSTR)
    _kernel32.SetInformationJobObject.restype = ctypes.wintypes.BOOL
    _kernel32.SetInformationJobObject.argtypes = (
        ctypes.wintypes.HANDLE, ctypes.c_int, ctypes.wintypes.LPVOID, ctypes.wintypes.DWORD,
    )
    _kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = (ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD)
    _kernel32.GetProcessTimes.restype = ctypes.wintypes.BOOL
    _kernel32.GetProcessTimes.argtypes = (
        ctypes.wintypes.HANDLE, *(ctypes.POINTER(ctypes.wintypes.FILETIME),) * 4,
    )
    _kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    _kernel32.GetCurrentProcess.argtypes = ()
    _kernel32.IsProcessInJob.restype = ctypes.wintypes.BOOL
    _kernel32.IsProcessInJob.argtypes = (
        ctypes.wintypes.HANDLE, ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.wintypes.BOOL),
    )
    _kernel32.QueryInformationJobObject.restype = ctypes.wintypes.BOOL
    _kernel32.QueryInformationJobObject.argtypes = (
        ctypes.wintypes.HANDLE, ctypes.c_int, ctypes.wintypes.LPVOID,
        ctypes.wintypes.DWORD, ctypes.POINTER(ctypes.wintypes.DWORD),
    )
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
    _JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x800
    _JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x1000
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


def _windows_process_start_time(pid: int) -> str:
    handle = _kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        created, exited, kernel, user = (ctypes.wintypes.FILETIME() for _ in range(4))
        if not _kernel32.GetProcessTimes(
            handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user),
        ):
            return ""
        return f"win-filetime:{(created.dwHighDateTime << 32) | created.dwLowDateTime}"
    finally:
        _kernel32.CloseHandle(handle)


def _windows_breakaway_flags() -> int:
    """Request breakaway from a permitting immediate Job; retain old-launcher spawn.

    This reads Job capability, never psutil/stop readiness (ARCHITECTURE §1/§9)."""
    in_job = ctypes.wintypes.BOOL()
    if _kernel32.IsProcessInJob(_kernel32.GetCurrentProcess(), None, ctypes.byref(in_job)):
        if not in_job.value:
            return 0
        info = _ExtendedLimitInfo()
        if _kernel32.QueryInformationJobObject(
            None, _JOBOBJECTINFOCLASS_EXTENDED, ctypes.byref(info), ctypes.sizeof(info), None,
        ):
            flags = info.BasicLimitInformation.LimitFlags
            if flags & _JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK:
                return 0
            if flags & _JOB_OBJECT_LIMIT_BREAKAWAY_OK:
                return getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
    log.warning("The current Windows Job does not confirm breakaway; shared daemon survival "
                "after launcher close requires an updated launcher or a permitting Job")
    return 0


def create_kill_on_close_job(*, allow_breakaway: bool = False) -> Optional[Any]:
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
        if allow_breakaway:
            info.BasicLimitInformation.LimitFlags |= _JOB_OBJECT_LIMIT_BREAKAWAY_OK
        ok = _kernel32.SetInformationJobObject(
            handle, _JOBOBJECTINFOCLASS_EXTENDED, ctypes.byref(info), ctypes.sizeof(info),
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
    """Terminate Job members; a false Win32 BOOL or exception is an unconfirmed stop."""
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
    """Close the Job handle; return a reason when kill-on-close remains unconfirmed."""
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
        # Full-width HANDLE, as in the kernel32 declarations above.
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


# Preserve import order without the node_runtime -> platform_layer eager cycle.
# PEP 562 re-exports retain callers' spellings (ARCHITECTURE §1 "Platform substrate").
_NODE_RUNTIME_REEXPORTS = (
    "NodeRuntimeHealth",
    "node_runtime_health",
    "probe_node_version",
    "prepend_skill_node_emergency_path",
    "select_skill_node_runtime",
    "skill_manifest_owns_path",
    "skill_node_argv",
    "skill_node_emergency_path_dir",
)


def __getattr__(name: str):
    if name in _NODE_RUNTIME_REEXPORTS:
        from ouroboros import node_runtime

        return getattr(node_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

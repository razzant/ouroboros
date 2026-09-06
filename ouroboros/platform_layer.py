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


# What a kernel refusal MEANS. Held by someone (flock's EWOULDBLOCK): stand down and re-contend.
# No kernel locks on this filesystem (EOPNOTSUPP/ENOSYS; LockFileEx winerrors mapped below) or no
# lock service (ENOLCK: lockd-less NFS, exhausted lock table): the name tier, errno RECORDED so a
# caller may refuse it (the monetary lock does). Anything else fails closed.
_LOCK_HELD_ERRNOS = frozenset({errno.EAGAIN, errno.EWOULDBLOCK})
_LOCK_UNSUPPORTED_ERRNOS = frozenset({errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOSYS})
_WIN32_LOCK_ERRNOS = {33: errno.EAGAIN, 1: errno.ENOSYS, 50: errno.EOPNOTSUPP}  # violation = held; invalid function / not supported = no byte-range locks here
_KERNEL_LOCK_TIER: dict = {}  # lock directory -> (kernel locks enforced there, the errno that selected the name tier or None)
_KERNEL_LOCK_TIER_LOCK = threading.Lock()  # one probe per directory, one verdict for every thread


def kernel_file_locks_enforced(lock_path: pathlib.Path) -> bool:
    """Capability predicate: are locks in ``lock_path``'s directory kernel-enforced (flock /
    LockFileEx held on the fd) or name-only?  Decided ONCE per directory under one module
    lock (racing threads share one probe and verdict) by locking a scratch file there — never
    by a refusal on a live acquisition.  The kernel's "this filesystem cannot" and ENOLCK ("no
    lock service") select the name tier, the errno recorded beside the verdict so a caller may
    refuse that tier; an unprobeable directory answers enforced for that call and is probed
    again next time (not cached); every other answer is the enforced tier, where a refused live
    lock fails closed.  Name-tier locks run the O_EXCL name protocol alone (re-check-then-unlink
    eviction, no kernel exclusion): disclosed best effort — the monetary compaction pass refuses to run there, appends continue.
    Windows answers this probe like POSIX since 7.0: the LockFileEx range sits beyond the owner
    stamp (:data:`_WIN32_LOCK_OFFSET`), so a mandatory hold no longer refuses a contender the read
    that lets it judge the hold — the defect that made this predicate answer False there."""
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
    """Acquire a portable lockfile.  On the enforced tier
    (:func:`kernel_file_locks_enforced`) the returned descriptor HOLDS a
    kernel lock (flock / LockFileEx): exclusion rests on the fd, not on the
    O_EXCL name alone, and a kernel refusal that is not contention fails
    CLOSED — no descriptor, our own file removed.  The name tier is selected
    by that predicate, never by a refusal (``refuse_name_tier_errnos``: probe answers,
    ENOLCK, on which THIS caller fails closed there instead).  On either tier a won lock is
    returned only while the path PROVABLY still names it: an evicted creator
    re-contends, and a descriptor whose own identity cannot be read is not a
    hold either — that fails closed too, taking our stamp off the path with it.

    Authority streams opt into ``owner_aware_stale`` so elapsed time alone
    never steals a lock from a live writer; a dead/malformed legacy owner
    still recovers through the stale-age path.  POSIX evicts a stale lock
    only UNDER a held flock on the very fd it judged, re-checking that the
    path still names that inode: of two racing reclaimers at most one can
    evict.  Windows takes the SAME kernel hold on the judged fd before it may
    evict, but cannot unlink an open file, so it unlinks after closing its
    probe — the hold it just released is what proves nobody else holds the
    file, and the identity re-check plus Windows' own refusal to delete a file
    the new owner holds open is what keeps the shape exclusive across that gap
    (it does not give the POSIX "at most one may evict" by the kernel alone:
    two reclaimers may both re-check, and the loser's unlink is refused by the
    winner's open handle rather than by the lock).  The name tier re-checks
    then unlinks with no hold at all: best effort, disclosed."""
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
                # A creator stalled between its create and its lock (SIGSTOP, suspend,
                # clock skew) is judged abandoned — aged, with no hold yet to refuse the
                # evictor — and evicted: its lock lands on an unlinked inode. Not a hold.
                # An identity we cannot READ (ESTALE/EIO) is no proof either — two empty
                # answers would compare equal — so it fails closed, and the file we
                # stamped with our LIVE pid goes with it: left behind, no owner-aware
                # reclaimer could ever remove it. Only bytes still exactly ours are ours.
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
                    stale = bool(judged) and (time.time() - judged[2] / 1e9) > stale_sec and not (
                        owner_aware_stale and owner_pid > 0 and pid_is_alive(owner_pid)
                    )
                    # Evict ONLY the exact file just judged abandoned, and only
                    # while flock-holding it: between judgement and unlink the
                    # owner may release and a third writer re-create the lock —
                    # removing THAT file puts two writers on one authority.
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
    """Renew a HELD lock's staleness clock; report whether it is still OURS.

    A critical section that legitimately outlives ``stale_sec`` (a monetary
    compaction pass) keeps the lockfile young for acquirers that judge by age
    alone.  The return value is an OWNERSHIP verdict, not a courtesy: ``False``
    means the path no longer names the descriptor we hold (evicted, deleted,
    replaced — however atomically), so the caller must abandon its work rather
    than finish beside a second writer.  A stolen lock is never refreshed."""
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
    """Unlink a lock file while the path still names ``held`` (``None``: unconditionally).
    Windows refuses to delete a file any other handle has open (CPython opens without
    FILE_SHARE_DELETE), and the name protocol's contenders open the lock on every poll to
    read its identity and owner stamp — a refusal at the owner's release is therefore
    routine and TRANSIENT (each such handle lives microseconds), so it is retried for a
    bounded window rather than swallowed: a swallowed refusal orphaned the lock with the
    owner's LIVE pid stamped in it, which no owner-aware acquirer would ever evict (the
    Windows matrices after the C6 merge, last 33663258606 on 35b82db0: monetary writers
    refused until restart, chat appends falling to the unlocked lane).  POSIX never refuses for a reader, so it does not retry."""
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
    """Release a lock acquired by :func:`acquire_exclusive_file_lock`: unlink
    OUR lock file or nothing at all (a hold evicted as stale and re-taken must
    not delete the new owner's lock on its way out).  POSIX unlinks BEFORE the
    close — under the still-held kernel lock on the enforced tier, so no
    reclaimer can be evicting the same file between re-check and unlink.
    Windows cannot unlink an open file: it releases the kernel hold FIRST (a
    handle closed with an outstanding lock leaves the release undefined), then
    closes, then re-checks the path — protected by the new owner's open
    handle — retrying a contender's transient sharing refusal
    (:func:`_unlink_lock_path`)."""
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
    """Return whether a PID appears alive without exposing os.kill to callers.

    Only a positive "no such process" is dead: EPERM means the process EXISTS
    and merely refuses our signal (another uid's pid on a shared host, a pid
    recycled onto one) — alive, like Windows' access-denied answer below."""

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
    """The KILL-decision question, distinct from :func:`pid_is_alive`: can THIS
    process signal ``pid``?  POSIX answers with signal 0 — EPERM (another
    user's process, pid 1) is "not ours", unlike the liveness reading where it is
    "alive"; Windows has no signal probe, so liveness stands in for it."""
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
    """True only when the OS positively answers that ``pid`` does not exist: the
    negation of :func:`pid_is_alive`, which reads EPERM — the process EXISTS and
    merely refuses our signal — and anything else undeterminable as still present;
    a killed-process check gets that fail-safe answer under the name that says what it proves."""
    return not pid_is_alive(pid)


# Windows locking via LockFileEx: unlike msvcrt.locking(), works on empty files.

# WHERE the lock is taken, and why it is not the whole file.  A Win32 byte-range lock is
# MANDATORY: bytes inside it cannot even be READ by another handle.  The whole-file range this
# used to take (offset 0, length 0xFFFFFFFFFFFFFFFF) therefore refused every contender the read
# of the owner stamp the lock protocol needs to judge a hold, and every wait ran to its timeout
# (the C6 Windows matrix, run 33654743857: eight monetary writers answered "lock unavailable").
# So the hold is ONE byte at an offset no lock file can reach — the common Win32 idiom — leaving
# the stamp bytes [0, 512) readable by anyone.  A lock file this long is not writable by this
# protocol (its stamp is one short line), and a lock BEYOND end-of-file is legal on Windows.
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
    """Lock a file descriptor using Win32 LockFileEx, on the one byte at
    :data:`_WIN32_LOCK_OFFSET` — never the whole file, whose stamp bytes a
    mandatory lock would make unreadable to the contenders that must judge it."""
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
    """The OSError a refused LockFileEx raises. ERROR_LOCK_VIOLATION means HELD BY SOMEONE
    (busy: re-contend); ERROR_INVALID_FUNCTION / ERROR_NOT_SUPPORTED are what a redirector
    answers when the volume takes no byte-range locks AT ALL, and read as the unsupported
    errnos — without them the name tier is unreachable on Windows and a lock-less volume
    fails every monetary append closed instead of degrading to it. Anything else keeps its
    winerror-derived errno (access denied, sharing violation -> EACCES) and fails closed.
    The 4-argument form derives errno FROM the winerror on Windows and ignores the one
    passed, so a classified code carries its own errno instead."""
    code = _WIN32_LOCK_ERRNOS.get(err)
    if code is None:
        return OSError(0, f"LockFileEx failed (error {err})", None, err)
    refused = OSError(code, f"LockFileEx refused (error {err})")
    refused.winerror = err  # kept for diagnostics; the errno carries the verdict
    return refused


def _win32_unlock(fd: int) -> None:
    """Release the fixed range _win32_lock takes on this descriptor, if any.

    The range is a constant, so the OVERLAPPED is rebuilt rather than remembered
    per fd: no map to leak, and none to answer for a descriptor number the
    process has since recycled onto another file.  An fd holding nothing is
    ERROR_NOT_LOCKED, which this ignores like every other refusal — a release
    is best effort by construction, and the handle's close is the backstop."""
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

def kill_process_tree(proc: subprocess.Popen) -> None:
    """Force-kill a subprocess and its entire process tree.

    POSIX SIGKILLs the process group first, then sweeps by PID descendants
    that escaped into their own session/group — collected BEFORE the kill,
    because a dead parent's children are reparented and the ppid links vanish.
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
    for dpid in [*reversed(descendants), pid]:
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
            os.kill(pid, getattr(signal, "SIGKILL", 9))  # spelled portably: a test may drive this branch on Windows
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
    for dpid in [*reversed(descendants), pid]:
        if dpid not in spared:
            force_kill_pid(dpid)


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

    Every platform gets ``SO_KEEPALIVE``; the probe-tuning constants — idle
    threshold, probe interval, probe count — are set only where the platform
    exposes them (Linux spells the idle threshold ``TCP_KEEPIDLE``, Darwin
    spells it ``TCP_KEEPALIVE``; both take ``TCP_KEEPINTVL``/``TCP_KEEPCNT``,
    which CPython exports on Darwin too, against XNU's 75 s × 8 defaults),
    each behind its own ``hasattr`` guard so the tuning degrades per option on
    an older interpreter. Every other platform (Windows included) keeps
    ``SO_KEEPALIVE`` alone.
    """
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
    """Extra pip flags so an install never writes INSIDE the packaged bundle.

    The embedded interpreter's own ``site-packages`` would break the code
    signature (and a read-only install outright): ``--user`` redirects to the
    ``PYTHONUSERBASE`` user site set by ``launcher_bootstrap``.  A dev venv or
    system python gets NO flag — ``--user`` is refused inside a virtualenv.
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

    SSOT for every bundled-payload lookup, because the consumer is usually NOT
    the frozen launcher: a packaged install runs the server/CLI as a SEPARATE
    child of the embedded interpreter out of the launcher-managed repo under
    the data dir — no ``sys._MEIPASS``, a ``__file__`` parent that is the
    managed repo — so both historical bases miss and every bundled payload
    silently reads as absent. The launcher therefore hands the bundle root
    down by value in ``OUROBOROS_BUNDLE_DIR``, searched FIRST; the other bases
    serve the frozen process itself and the dev/source layout (payloads sit at
    the repo root, two levels up from this module)."""
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
# the policy outgrew a cross-platform primitives file). The re-export is a PEP
# 562 module __getattr__ rather than an eager from-import: node_runtime imports
# this module at module level, and an eager import back from HERE re-entered a
# partially initialized node_runtime whenever node_runtime was imported first
# (triad finding, all three phase-C reviewers). Lazy resolution keeps both
# import orders sound while every importer keeps its
# `from ouroboros.platform_layer import <name>` spelling unchanged.
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

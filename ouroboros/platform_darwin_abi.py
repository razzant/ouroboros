"""Private Darwin ABI: a process's kernel birth record and a folder's kept spelling, read without a subprocess.

``sysctl({CTL_KERN, KERN_PROC, KERN_PROC_PID, pid})`` fills one LP64
``struct kinfo_proc`` (648 bytes on arm64 and x86_64). ``kp_proc.p_starttime``
is the timeval arm of ``extern_proc``'s leading union (``tv_sec`` int64 at 0,
``tv_usec`` int32 at 8), ``p_stat`` sits at 36 (``SZOMB`` = 5) and ``p_pid`` at
40. XNU writes the start once at fork and keeps it across exec; a calendar step
moves no process's start. The call answers for every user's processes and for
zombies. libproc's ``PROC_PIDTBSDINFO`` is not used: it refuses another user's
pid with EPERM, so a pid reused by a system daemon would stay unreadable.

Every surprise reads as unreadable (None), never as another birth: an absent
pid answers rc 0 with nothing filled, and a layout this module does not know
fails the filled-size or the echoed-pid check. Importing the module loads
nothing native; the library handle opens on the first macOS call.
"""

from __future__ import annotations

import functools
import struct
import sys

IS_DARWIN = sys.platform == "darwin"

_KERN_PROC_PID_MIB = (1, 14, 1)  # CTL_KERN, KERN_PROC, KERN_PROC_PID
_KINFO_PROC_SIZE = 648
_SZOMB = 5


@functools.lru_cache(maxsize=1)
def _sysctl():
    import ctypes

    function = ctypes.CDLL(None).sysctl  # libSystem is mapped into every process
    function.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
                         ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t)
    function.restype = ctypes.c_int
    return function


def kernel_birth(pid: int) -> tuple[int, int, bool] | None:
    """``(tv_sec, tv_usec, is_zombie)`` of ``pid``'s kernel start record, or None.

    None means absent, unreadable or not macOS: never a guessed birth.
    """
    if not IS_DARWIN:
        return None
    try:
        import ctypes

        pid = int(pid)
        if not 0 < pid < 2**31:  # pid_t; a wider value would be silently truncated
            return None
        mib = (ctypes.c_int * 4)(*_KERN_PROC_PID_MIB, pid)
        buffer = ctypes.create_string_buffer(_KINFO_PROC_SIZE)
        size = ctypes.c_size_t(_KINFO_PROC_SIZE)
        if _sysctl()(mib, 4, buffer, ctypes.byref(size), None, 0) != 0 or size.value != _KINFO_PROC_SIZE:
            return None  # rc 0 with nothing filled = no such pid
        raw = buffer.raw
        sec, usec = struct.unpack_from("<qi", raw, 0)
        (echoed_pid,) = struct.unpack_from("<i", raw, 40)
    except Exception:
        return None
    if echoed_pid != pid or sec <= 0 or not 0 <= usec < 1_000_000:
        return None
    return sec, usec, raw[36] == _SZOMB


def kept_spelling(path: str, *, directory_fd: int | None = None) -> str | None:
    """The spelling macOS keeps for the existing folder ``path``, or None.

    ``fcntl(F_GETPATH)`` on the borrowed directory descriptor or one opened for this call names every
    component as the volume stores it: its case and Unicode form, through symlinks and firmlinks
    (``/System/Volumes/Data/Users/a`` reads ``/Users/a``). None means absent, not a folder, unreadable
    (a folder that may be searched but not read), not macOS or any other error: never a guessed spelling.
    """
    if not IS_DARWIN:
        return None
    import fcntl  # Darwin's own; imported here, as the platform guard requires outside platform_layer
    import os

    try:
        fd = directory_fd if directory_fd is not None else os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NONBLOCK)
    except (OSError, ValueError):
        return None
    try:
        raw = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))  # MAXPATHLEN
    except OSError:
        return None
    finally:
        if directory_fd is None:
            os.close(fd)
    kept = os.fsdecode(raw.split(b"\0", 1)[0])
    return kept if kept.startswith("/") else None


def wait_child_status(pid: int, *, nohang: bool) -> tuple[int, int]:
    """Darwin waitid(WEXITED | WNOWAIT), absent from CPython's macOS os module.

    LP64 siginfo_t is 104 bytes: six int32s, three pointer/long words,
    seven reserved words (SDK sys/signal.h). Return waitpid-shaped status
    without consuming the child. Errors, including ECHILD, stay errors.
    """
    import ctypes
    import os

    class Siginfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_int) for name in
                    ("signo", "error", "code", "pid", "uid", "status")] + [
                        ("tail", ctypes.c_ulong * 10)]

    libc = ctypes.CDLL(None, use_errno=True)
    waitid = libc.waitid
    waitid.argtypes = (ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Siginfo), ctypes.c_int)
    waitid.restype = ctypes.c_int
    info = Siginfo()
    # P_PID=1, WEXITED=4, WNOWAIT=32, WNOHANG=1 (SDK sys/wait.h).
    while waitid(1, pid, ctypes.byref(info), 4 | 32 | int(nohang)) != 0:
        code = ctypes.get_errno()
        if code == 4:  # EINTR: match os.waitpid's interrupted-wait retry
            continue
        raise OSError(code, os.strerror(code))
    return info.pid, (info.status << 8 if info.code == 1 else info.status)

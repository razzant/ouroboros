"""Private Darwin ABI: a process's kernel birth record, read without a subprocess.

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

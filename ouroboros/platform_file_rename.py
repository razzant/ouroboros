"""Native atomic publication without replacement, and exchange of two existing names.

Used for Project registry commits when a concurrently returned original must win,
and for ownership of the compatible POSIX writer name. A missing primitive is
reported rather than emulated by a separate existence check and replacement.
"""
from __future__ import annotations

import errno
import functools
import os
import pathlib
import sys

from ouroboros.utils import assert_test_data_path


@functools.lru_cache(maxsize=1)
def _native_rename():
    import ctypes

    path, fd, flags = ctypes.c_char_p, ctypes.c_int, ctypes.c_uint
    if sys.platform == "darwin":
        name, args = "renamex_np", (path, path, flags)
    elif sys.platform.startswith("linux"):
        name, args = "renameat2", (fd, path, fd, path, flags)
    else:
        return None
    rename = getattr(ctypes.CDLL(None, use_errno=True), name, None)
    if rename is not None:
        rename.argtypes, rename.restype = args, ctypes.c_int
    return rename


def rename_exclusive(src: pathlib.Path, dst: pathlib.Path, *, exchange=False) -> None:
    """Publish only to a missing name, or exchange two existing names atomically."""
    assert_test_data_path(src)
    assert_test_data_path(dst)
    if os.name == "nt" and not exchange:
        os.rename(src, dst)
        return
    import ctypes

    rename = _native_rename()
    if rename is None:
        raise OSError(errno.ENOTSUP, "Exclusive rename unavailable")
    a, b = os.fsencode(src), os.fsencode(dst)
    if sys.platform == "darwin":
        result = rename(a, b, 2 if exchange else 4)  # RENAME_SWAP / RENAME_EXCL
    else:
        result = rename(-100, a, -100, b, 2 if exchange else 1)  # AT_FDCWD, RENAME_EXCHANGE / NOREPLACE
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(dst))

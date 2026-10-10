"""Open ONE regular file directly inside a trusted directory, never through a link.

The caller resolves the trusted directory and validates the plain name; this leaf
proves that the descriptor it returns is that directory's own regular file, as it
was when opened, and hands back the open handle plus the fstat of that descriptor.
Nothing re-opens a pathname after the check.

POSIX: the directory is opened ``O_DIRECTORY | O_NOFOLLOW`` and the file
``O_NOFOLLOW | O_NONBLOCK`` relative to it (a link is ELOOP, a planted FIFO never
blocks), and the descriptor must fstat as a regular file. ``task_archive`` shares
these flags for its multi-segment descent.

Windows has no directory-relative open in the Win32 API, so confinement is proved
by handles: the directory is opened with ``FILE_FLAG_OPEN_REPARSE_POINT`` for
listing — a data access, so its share mode is enforced (an attributes-only open
takes no part in sharing) — and held WITHOUT delete sharing (it cannot be renamed,
replaced or deleted while held), the file is opened the same way, neither may carry
a reparse point (symlink, junction, mount point), the file must be an ordinary disk
file (``GetFileType``), and the file's final path must be exactly the held
directory's final path plus the plain name, so a file moved out while opened is refused.
Only then does the handle become a CRT descriptor. The plain-name rule refuses
separators, drive/stream colons, wildcards, control characters, a trailing dot or
space and the reserved device names, so no request reaches a device, a stream or
another share. Platform modules are imported function-locally under the Windows
guard (docs/development/10-platform-abstraction-rule.md).
"""

from __future__ import annotations

import errno
import os
import pathlib
import stat
import sys
from typing import Any, BinaryIO, Tuple

IS_WINDOWS = sys.platform == "win32"
POSIX_CONFINED = bool(os.open in os.supports_dir_fd and os.stat in os.supports_dir_fd
                      and hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"))
DIR_FLAGS = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_CLOEXEC", 0))
FILE_FLAGS = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
              | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOCTTY", 0))
# A confined open exists on this platform: the POSIX descent or the Windows handle proof.
CONFINED_OPEN_AVAILABLE = POSIX_CONFINED or IS_WINDOWS

_WINDOWS_FORBIDDEN = frozenset('<>:"/\\|?*')
_WINDOWS_DEVICE_NAMES = frozenset({"con", "prn", "aux", "nul", "conin$", "conout$",
                                   *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))})


def plain_name(name: Any) -> str:
    """NAME when it is one plain file name on every platform, else ``""``."""
    text = str(name or "")
    if not text or text in {".", ".."} or len(text.encode("utf-8")) > 255:
        return ""
    if any(ord(char) < 32 or char in _WINDOWS_FORBIDDEN for char in text) or text[-1] in ". ":
        return ""
    if text.split(".", 1)[0].strip().lower() in _WINDOWS_DEVICE_NAMES:
        return ""
    return text


def open_regular_at(dir_fd: int, name: str) -> Tuple[BinaryIO, os.stat_result]:
    """POSIX: the regular file NAME below an open confined directory descriptor."""
    fd = os.open(name, FILE_FLAGS, dir_fd=dir_fd)
    try:
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode):
            raise OSError(f"not a regular file: {name}")  # no errno: refused like a swap
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "rb"), observed


def open_regular_file(directory: Any, name: Any, *, win32: Any = None) -> Tuple[BinaryIO, os.stat_result]:
    """A binary handle on DIRECTORY/NAME plus the fstat of that descriptor.

    Raises ``OSError``: ENOENT/ENOTDIR (absent), ELOOP (a link or reparse point),
    an errno-less error (not a regular file, or the final path left the
    directory), ENOTSUP where neither proof exists, or the platform's I/O error.
    ``win32`` substitutes the Windows file API (``_Win32Files`` shape) in tests.
    """
    plain = plain_name(name)
    if not plain or plain != name:
        raise OSError(errno.EINVAL, "not a plain file name")
    if win32 is not None or IS_WINDOWS:
        return _open_windows(pathlib.Path(directory), plain, win32 or _Win32Files.load())
    if not POSIX_CONFINED:
        raise OSError(errno.ENOTSUP, "confined file open is unavailable on this platform")
    dir_fd = os.open(directory, DIR_FLAGS)
    try:
        return open_regular_at(dir_fd, plain)
    finally:
        os.close(dir_fd)


def _open_windows(directory: pathlib.Path, name: str, api: Any) -> Tuple[BinaryIO, os.stat_result]:
    held = api.open(str(directory), directory=True)
    try:
        api.require_plain(held, directory=True)
        expected = api.final_path(held).rstrip("\\") + "\\" + name
        handle = api.open(str(directory / name), directory=False)
        try:
            api.require_plain(handle, directory=False)
            if api.final_path(handle).casefold() != expected.casefold():
                raise OSError(f"file left its directory while opened: {name}")
            fd = api.to_fd(handle)  # the descriptor owns the handle from here on
        except BaseException:
            api.close(handle)
            raise
    finally:
        api.close(held)
    try:
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode):
            raise OSError(f"not a regular file: {name}")
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "rb"), observed


class _Win32Files:
    """The five kernel32 calls the Windows proof needs (bound once, lazily)."""

    _instance = None
    FILE_LIST_DIRECTORY = 0x1
    FILE_READ_ATTRIBUTES = 0x80
    GENERIC_READ = 0x80000000
    SHARE_READ, SHARE_WRITE, SHARE_DELETE = 0x1, 0x2, 0x4
    OPEN_EXISTING = 3
    FLAG_BACKUP_SEMANTICS = 0x02000000
    FLAG_OPEN_REPARSE_POINT = 0x00200000
    ATTRIBUTE_DIRECTORY = 0x10
    ATTRIBUTE_REPARSE_POINT = 0x400
    FILE_TYPE_DISK = 1
    _ERRNO = {2: errno.ENOENT, 3: errno.ENOENT, 5: errno.EACCES, 32: errno.EACCES, 123: errno.EINVAL}

    @classmethod
    def load(cls) -> "_Win32Files":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes
        import msvcrt

        class _Info(ctypes.Structure):
            _fields_ = [("attributes", wintypes.DWORD), ("times", wintypes.DWORD * 6),
                        ("volume", wintypes.DWORD), ("size_high", wintypes.DWORD),
                        ("size_low", wintypes.DWORD), ("links", wintypes.DWORD),
                        ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ctypes, self._msvcrt, self._info_type = ctypes, msvcrt, _Info
        self._create = kernel32.CreateFileW
        self._create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                 wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        self._create.restype = wintypes.HANDLE
        self._info = kernel32.GetFileInformationByHandle
        self._info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Info)]
        self._info.restype = wintypes.BOOL
        self._type = kernel32.GetFileType
        self._type.argtypes = [wintypes.HANDLE]
        self._type.restype = wintypes.DWORD
        self._final = kernel32.GetFinalPathNameByHandleW
        self._final.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
        self._final.restype = wintypes.DWORD
        self._close = kernel32.CloseHandle
        self._close.argtypes = [wintypes.HANDLE]
        self._close.restype = wintypes.BOOL
        self._invalid = wintypes.HANDLE(-1).value

    def _error(self, what: str) -> OSError:
        code = self._ctypes.get_last_error()
        return OSError(self._ERRNO.get(code, errno.EIO), f"{what} failed (Windows error {code})")

    def open(self, path: str, *, directory: bool) -> int:
        # Listing is a data access, so the held directory's share mode binds other opens.
        access = (self.FILE_LIST_DIRECTORY | self.FILE_READ_ATTRIBUTES) if directory else self.GENERIC_READ
        # The held directory refuses delete sharing; the file shares delete like POSIX unlink.
        share = self.SHARE_READ | self.SHARE_WRITE | (0 if directory else self.SHARE_DELETE)
        flags = self.FLAG_OPEN_REPARSE_POINT | (self.FLAG_BACKUP_SEMANTICS if directory else 0)
        handle = self._create(path, access, share, None, self.OPEN_EXISTING, flags, None)
        if handle in (None, self._invalid):
            raise self._error("CreateFileW")
        return handle

    def require_plain(self, handle: int, *, directory: bool) -> None:
        info = self._info_type()
        if not self._info(handle, self._ctypes.byref(info)):
            raise self._error("GetFileInformationByHandle")
        if info.attributes & self.ATTRIBUTE_REPARSE_POINT:
            raise OSError(errno.ELOOP, "a reparse point is not followed")
        if bool(info.attributes & self.ATTRIBUTE_DIRECTORY) != directory:
            raise OSError(errno.ENOTDIR if directory else errno.EISDIR, "unexpected file kind")
        if not directory and self._type(handle) != self.FILE_TYPE_DISK:
            raise OSError("not an ordinary disk file")

    def final_path(self, handle: int) -> str:
        size = 512
        while True:
            buffer = self._ctypes.create_unicode_buffer(size)
            length = self._final(handle, buffer, size, 0)  # FILE_NAME_NORMALIZED | VOLUME_NAME_DOS
            if not length:
                raise self._error("GetFinalPathNameByHandleW")
            if length < size:
                return buffer.value
            size = length + 1

    def to_fd(self, handle: int) -> int:
        return self._msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))

    def close(self, handle: int) -> None:
        self._close(handle)

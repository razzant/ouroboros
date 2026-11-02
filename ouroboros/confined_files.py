"""Confined regular-file reads and artifact directory walks.

``confined_directory`` holds the trusted root and descends by one plain component
at a time. POSIX uses dirfd/no-follow; Windows uses NtCreateFile RootDirectory,
OBJ_DONT_REPARSE and handle type checks, with NtQueryDirectoryFile enumeration.
Only the configured root may follow an owner-provided junction. Descendants are
never resolved or reopened by absolute path. A returned file owns its descriptor,
except ``open_for_read``'s, the store's raw-reader contract a deliverable walk keeps.
Like a POSIX dirfd, a held walk directory shares rename and delete: others may
move or remove it, and relative opens still address the held object. Walk names
follow this platform's entry rule (``native_name``); HTTP selectors keep the
stricter cross-platform ``relative_parts``.

The existing ``open_regular_file`` single-name upload contract is unchanged:

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
import contextlib
import json
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
                                   *(f"com{i}" for i in "0123456789¹²³"), *(f"lpt{i}" for i in "0123456789¹²³")})


def relative_parts(value: Any) -> list[str]:
    """Lexical store selector. Windows names cannot address devices or alternate streams."""
    text = str(value or "")
    parts = text.split("/")
    if "\\" in text or "\x00" in text or {"", ".", ".."} & set(parts):
        return []
    # Artifact selectors use native UTF-16 lengths; uploads keep their existing
    # cross-platform UTF-8 byte limit in plain_name.
    return [] if IS_WINDOWS and any(len(part.encode("utf-16-le", "surrogatepass")) > 510
                                   or not _plain_spelling(part) for part in parts) else parts


def native_name(name: Any) -> bool:
    """One directory entry name this platform's file APIs address as itself: POSIX
    refuses only ``/`` and NUL; Windows keeps its Win32 naming rules (no separator,
    stream, device name, trailing dot or space), as its HTTP selectors do."""
    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name or "\x00" in name:
        return False
    return not IS_WINDOWS or relative_parts(name) == [name]


def root_spellings(root: Any) -> Tuple[pathlib.Path, ...]:
    """A host-configured ROOT as reads descend from it (physical), then as producers
    recorded it (configured) when that differs. Only this trusted root is resolved."""
    configured = pathlib.Path(root).absolute()
    physical = configured.resolve()
    return (physical,) if physical == configured else (physical, configured)


def physical_path(spellings: Any, path: Any) -> pathlib.Path:
    """PATH recorded through the configured root, re-spelled lexically under the physical one."""
    candidate = pathlib.Path(str(path or ""))
    for alias in tuple(spellings)[1:]:
        if candidate.is_absolute() and candidate.is_relative_to(alias):
            return spellings[0].joinpath(candidate.relative_to(alias))
    return candidate


def lexical_relative(root: Any, path: Any, spellings: Any = ()) -> str:
    """Attribute an absolute recorded path without touching any descendant."""
    candidate, base = physical_path(spellings, path), pathlib.Path(root)
    if not candidate.is_absolute() or ".." in candidate.parts:
        return ""
    try:
        text = candidate.relative_to(base).as_posix()
    except ValueError:
        return ""
    return text if all(native_name(part) for part in text.split("/")) else ""


class ConfinedDirectory:
    """Held directory, descended only by plain names. The root is host-owned and may
    be a configured junction; descendants never follow links. No path is reopened.
    A child must be closed before its parent (use ExitStack for a multi-part walk).
    """

    def __init__(self, root: Any = None, *, _parent: Any = None, _name: str = "") -> None:
        self.api = _parent.api if _parent is not None else (_Win32Files.load() if IS_WINDOWS else None)
        self.handle = None
        if _parent is not None and not native_name(_name):
            raise OSError(errno.EINVAL, "not a plain directory component")
        if self.api:
            handle = (self.api.open_relative(_parent.handle, _name, directory=True) if _parent is not None
                      else self.api.open_root(str(root)))
        elif POSIX_CONFINED:
            # Only the configured root may follow a link, never a descendant.
            handle = (os.open(_name, DIR_FLAGS, dir_fd=_parent.handle) if _parent is not None else
                      os.open(root, DIR_FLAGS & ~os.O_NOFOLLOW))
        else:
            raise OSError(errno.ENOTSUP, "confined directory reads unavailable")
        self.handle = handle

    def __enter__(self) -> "ConfinedDirectory":
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.handle is not None:
            handle, self.handle = self.handle, None
            self.api.close(handle) if self.api else os.close(handle)

    def child(self, name: str) -> "ConfinedDirectory":
        return ConfinedDirectory(_parent=self, _name=name)

    def open(self, name: str) -> Tuple[BinaryIO, os.stat_result]:
        if not native_name(name):
            raise OSError(errno.EINVAL, "not a plain file component")
        if not self.api:
            return open_regular_at(self.handle, name)
        handle = self.api.open_relative(self.handle, name, directory=False)
        try:
            fd = self.api.to_fd(handle)
        except BaseException:
            self.api.close(handle)
            raise
        try:
            observed = os.fstat(fd)
            if not stat.S_ISREG(observed.st_mode):
                raise OSError("not a regular file")
            return os.fdopen(fd, "rb"), observed
        except BaseException:
            os.close(fd)
            raise

    def open_for_read(self, name: str) -> Tuple[Any, os.stat_result]:
        """``open`` as ``store.open_for_read`` reads, for a walk in a process that may hold a store: a
        SQLite WAL index is refused before any data access (``WalIndexRefused``). POSIX asks the rule
        by ``fstatat`` (no descriptor of the index exists to close) and reads every other file through
        the store's shared descriptor, whose close drops no lock of this process's SQLite connections.
        Windows locks belong to their handle; the index gets attribute-only handles (``_wal_index``)."""
        from ouroboros.store import WalIndexRefused, open_for_read

        if not native_name(name):
            raise OSError(errno.EINVAL, "not a plain file component")
        if self.api:
            if self._wal_index(name):
                raise WalIndexRefused(errno.EACCES, "a SQLite WAL index is not read as a file", name)
            return self.open(name)
        handle = open_for_read(name, dir_fd=self.handle)
        try:
            observed = os.fstat(handle.fileno())
            if not stat.S_ISREG(observed.st_mode):
                raise OSError("not a regular file")
            return handle, observed
        except BaseException:
            handle.close()
            raise

    def _wal_index(self, name: str) -> bool:
        """``store.is_wal_index`` by handle (Windows has no ``dir_fd``): NAME and its database NAME[:-4]
        are regular, the database is SQLite by its header and no index itself, and ``<database>-shm``
        names NAME's own file. NAME and that spelling get attribute-only handles (``identity``), so no
        data handle of the index is opened; only the database is opened, to read its header."""
        if len(name) <= 4 or name[-4:].casefold() != "-shm" or self._wal_index(name[:-4]):
            return False
        database = name[:-4]
        try:
            own = self.api.identity(self.handle, name)
            with self.open(database)[0] as base:
                if base.read(16) != b"SQLite format 3\x00":
                    return False
            return database + "-shm" == name or self.api.identity(self.handle, database + "-shm") == own
        except OSError:
            return False

    def entries(self, *, include_links: bool = False) -> list[tuple[str, bool]]:
        """Directories and regular files with directory hints; opens recheck kind. Links,
        FIFOs, sockets and devices are no walk's material. Metadata can observe them all."""
        if self.api:
            return self.api.entries(self.handle, include_links=include_links)
        with os.scandir(self.handle) as entries:
            return [(entry.name, entry.is_dir(follow_symlinks=False)) for entry in entries
                    if include_links or entry.is_dir(follow_symlinks=False) or entry.is_file(follow_symlinks=False)]


@contextlib.contextmanager
def confined_directory(root: Any, parts: Any = ()):
    """Hold the root and every directory component until the caller finishes."""
    with contextlib.ExitStack() as held:
        directory = held.enter_context(ConfinedDirectory(root))
        for name in parts:
            directory = held.enter_context(directory.child(name))
        yield directory


def open_confined_file(root: Any, parts: Any) -> Tuple[BinaryIO, os.stat_result]:
    parts = list(parts)
    if not parts or not all(native_name(part) for part in parts):
        raise OSError(errno.EINVAL, "invalid confined file selector")
    with confined_directory(root, parts[:-1]) as directory:
        return directory.open(parts[-1])


def read_confined_json(root: Any, parts: Any, *, strict: bool = False) -> dict | None:
    """Metadata bytes from the same confined walk as downloads; no writes or repair."""
    try:
        handle, _ = open_confined_file(root, parts)
        with handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("metadata is not an object")
        return value
    except FileNotFoundError:
        return None
    except (OSError, ValueError, UnicodeError) as exc:
        if strict:
            raise OSError("confined metadata is unreadable") from exc
        return None


def plain_name(name: Any) -> str:
    """NAME when it is one plain file name on every platform, else ``""``."""
    text = str(name or "")
    if not text or text in {".", ".."} or len(text.encode("utf-8")) > 255:
        return ""
    return text if _plain_spelling(text) else ""


def _plain_spelling(text: str) -> bool:
    if any(ord(char) < 32 or char in _WINDOWS_FORBIDDEN for char in text) or text[-1] in ". ":
        return False
    if text.split(".", 1)[0].strip().lower() in _WINDOWS_DEVICE_NAMES:
        return False
    return True


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
    """Lazy kernel32/NT bindings for uploads and handle-relative artifact reads."""

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
        # NT's RootDirectory is the authority, not a reconstructed absolute path.
        class UnicodeString(ctypes.Structure):
            _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
                        ("Buffer", wintypes.LPWSTR)]

        class ObjectAttributes(ctypes.Structure):
            _fields_ = [("Length", wintypes.ULONG), ("RootDirectory", wintypes.HANDLE),
                        ("ObjectName", ctypes.POINTER(UnicodeString)), ("Attributes", wintypes.ULONG),
                        ("SecurityDescriptor", wintypes.LPVOID), ("SecurityQualityOfService", wintypes.LPVOID)]

        class IoStatus(ctypes.Structure):
            _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

        self._unicode, self._attributes, self._io_status = UnicodeString, ObjectAttributes, IoStatus
        ntdll = ctypes.WinDLL("ntdll")
        self._nt_create = ntdll.NtCreateFile
        self._nt_create.argtypes = [ctypes.POINTER(wintypes.HANDLE), wintypes.ULONG,
                                   ctypes.POINTER(ObjectAttributes), ctypes.POINTER(IoStatus),
                                   wintypes.LPVOID, wintypes.ULONG, wintypes.ULONG, wintypes.ULONG,
                                   wintypes.ULONG, wintypes.LPVOID, wintypes.ULONG]
        self._nt_create.restype = wintypes.LONG
        self._nt_query = ntdll.NtQueryDirectoryFile
        self._nt_query.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.LPVOID, wintypes.LPVOID,
                                  ctypes.POINTER(IoStatus), wintypes.LPVOID, wintypes.ULONG,
                                  wintypes.ULONG, ctypes.c_ubyte, wintypes.LPVOID, ctypes.c_ubyte]
        self._nt_query.restype = wintypes.LONG
        self._nt_error = ntdll.RtlNtStatusToDosError
        self._nt_error.argtypes, self._nt_error.restype = [wintypes.LONG], wintypes.ULONG

    def _status_error(self, status: int) -> OSError:
        code = status & 0xffffffff
        # Includes a held directory changed into a reparse point (Microsoft MS-ERREF).
        if code in {0xC000050B, 0xC0000280, 0xC0000279}:
            number = errno.ELOOP
        elif code == 0xC0000103:
            number = errno.ENOTDIR
        elif code == 0xC00000BA:
            number = errno.EISDIR
        else:
            number = self._ERRNO.get(self._nt_error(status), errno.EIO)
        return OSError(number, f"confined NT read failed (NTSTATUS 0x{code:08X})")

    def open_root(self, path: str) -> int:
        """Only the host-configured anchor follows junctions, once, on acquisition."""
        # Walk handles share delete like a POSIX dirfd; only the upload proof (``open``) denies it.
        handle = self._create(path, self.FILE_LIST_DIRECTORY | self.FILE_READ_ATTRIBUTES,
                              self.SHARE_READ | self.SHARE_WRITE | self.SHARE_DELETE, None, self.OPEN_EXISTING,
                              self.FLAG_BACKUP_SEMANTICS, None)
        if handle in (None, self._invalid):
            raise self._error("CreateFileW(root)")
        try:
            self.require_plain(handle, directory=True)
            return handle
        except BaseException:
            self.close(handle)
            raise

    def open_relative(self, parent: int, name: str, *, directory: bool, metadata: bool = False) -> int:
        """``metadata``: no data access, so the handle reads no byte and NT checks no share mode."""
        if not native_name(name):
            raise OSError(errno.EINVAL, "not a plain Windows component")
        ctypes = self._ctypes
        self.require_plain(parent, directory=True)
        buffer = ctypes.create_unicode_buffer(name)
        length = len(name.encode("utf-16-le", "surrogatepass"))
        text = self._unicode(length, length + 2, ctypes.cast(buffer, ctypes.c_wchar_p))
        attributes = self._attributes(ctypes.sizeof(self._attributes), parent, ctypes.pointer(text),
                                      0x40 | 0x1000, None, None)  # CASE_INSENSITIVE | DONT_REPARSE
        handle, io = ctypes.c_void_p(), self._io_status()
        access = 0x100000 | 0x80 | (0 if metadata else 0x1)  # SYNCHRONIZE | READ_ATTRIBUTES [| READ_DATA]
        status = self._nt_create(ctypes.byref(handle), access, ctypes.byref(attributes),
                                 ctypes.byref(io), None, 0, self.SHARE_READ | self.SHARE_WRITE | self.SHARE_DELETE,
                                 1, 0x20 | 0x200000 | (0x1 if directory else 0x40), None, 0)
        # FILE_OPEN, SYNCHRONOUS_IO_NONALERT, OPEN_REPARSE_POINT, DIRECTORY/NON_DIRECTORY_FILE.
        if status < 0:
            raise self._status_error(status)
        try:
            self.require_plain(handle.value, directory=directory)
            return handle.value
        except BaseException:
            self.close(handle.value)
            raise

    def identity(self, parent: int, name: str) -> Tuple[int, int, int]:
        """A plain regular entry's (volume, file index) by an attribute-only relative handle: the
        access ``os.stat`` takes here, which reads nothing and meets no share mode."""
        handle = self.open_relative(parent, name, directory=False, metadata=True)
        try:
            info = self._info_type()
            if not self._info(handle, self._ctypes.byref(info)):
                raise self._error("GetFileInformationByHandle")
            return info.volume, info.index_high, info.index_low
        finally:
            self.close(handle)

    def entries(self, handle: int, *, include_links: bool = False) -> list[tuple[str, bool]]:
        """FileDirectoryInformation through the held handle; never FindFirstFile(path)."""
        import struct

        ctypes, result, restart = self._ctypes, [], True
        buffer, io = ctypes.create_string_buffer(65536), self._io_status()
        while True:
            self.require_plain(handle, directory=True)
            status = self._nt_query(handle, None, None, None, ctypes.byref(io), buffer, len(buffer),
                                     1, False, None, restart)
            restart = False
            if status & 0xffffffff == 0x80000006:  # STATUS_NO_MORE_FILES
                return result
            if status < 0:
                raise self._status_error(status)
            raw, offset = buffer.raw[:io.Information], 0
            if not raw:
                return result
            while True:
                if offset + 64 > len(raw):
                    raise OSError("truncated directory information")
                following = struct.unpack_from("<I", raw, offset)[0]
                attributes, length = struct.unpack_from("<II", raw, offset + 56)
                if length % 2 or offset + 64 + length > len(raw):
                    raise OSError("invalid directory name extent")
                name = raw[offset + 64:offset + 64 + length].decode("utf-16-le", "surrogatepass")
                if native_name(name) and (include_links or not attributes & self.ATTRIBUTE_REPARSE_POINT):
                    result.append((name, bool(attributes & self.ATTRIBUTE_DIRECTORY)))
                if not following:
                    break
                if following < 64 + length or offset + following >= len(raw):
                    raise OSError("invalid directory entry extent")
                offset += following

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

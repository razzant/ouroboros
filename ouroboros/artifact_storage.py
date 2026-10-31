"""Lossless filesystem storage for artifacts. Clone-on-copy gives a new inode sharing blocks where it
can give one (macOS APFS ``clonefile(2)``, Linux ``FICLONE`` on Btrfs and reflink-enabled XFS); any
other platform, filesystem or volume boundary answers ``False`` and the caller streams a copy as before.
Never a hard link: an in-place write to the source would change the stored result, while a clone
diverges at the first write to either side. ``clone_or_copy2`` is ``shutil.copy2`` over it, for a folder
that is read rather than kept: APFS leaves a clone's may-share mark after its peer is gone, so a file meant
to be compressed later stays a stream. Transparent compression keeps the one floor its callers share
(``compression_outcome``: the patch writers and the ``record_compression`` maintenance job). The platform
calls are function-local under ``sys.platform`` guards (docs/development/10-platform-abstraction-rule.md
names this module's exception)."""
from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import zlib
from typing import Any, Callable, Optional

from ouroboros.temp_sweep import atomic_temp_path as temporary_sibling, is_atomic_temp

_FICLONE = 0x40049409  # _IOW(0x94, 9, int), <linux/fs.h>
FLOOR_BYTES = 64 * 1024  # under it a file gains little and costs one rewrite (decmpfs blocks, FSEvents)
_BLOCK = 4096  # one allocation block: the least a compression must save
STAT_SKIPS = ("not_regular", "dataless", "small", "linked", "compressed_already")  # decided by one lstat, no read
_SIBLING_TAG = "decmpfs"  # the compression sibling's own tag inside the atomic signature
_SF_DATALESS = 0x40000000  # <sys/stat.h>: an evicted (File Provider) file; reading it would download it
_UTILITY_DISK = ("/usr/sbin/taskpolicy", "-d", "utility")  # a program run at the utility disk tier (macOS)
_REFUSES_TIMES = stat.UF_IMMUTABLE | stat.UF_APPEND | stat.SF_IMMUTABLE | stat.SF_APPEND


def clone_file(source: pathlib.Path, destination: pathlib.Path) -> bool:
    """Make DESTINATION (which must not exist) a block-sharing clone of SOURCE; ``False`` when this
    platform or filesystem cannot, with nothing left at DESTINATION, so the caller copies instead.
    Only a regular file is cloned: a directory would be cloned whole and a FIFO's open would block,
    so anything else is left to the streamed copy's own typed refusal. The clone's times are its
    own: ``clonefile`` copies the source's, and a temporary that looks as old as the result it copies
    would be reaped as a hard kill's orphan by the temp sweep's age guard while the copy is in flight."""
    source, destination = pathlib.Path(source), pathlib.Path(destination)
    try:
        if not stat.S_ISREG(os.stat(source).st_mode):
            return False
        if sys.platform == "darwin":
            cloned = _clonefile(source, destination)
        elif sys.platform.startswith("linux"):
            cloned = _ficlone(source, destination)
        else:
            return False
    except OSError:
        return False
    if cloned:
        with contextlib.suppress(OSError):  # the clone stands either way; the caller sets the published times
            os.utime(destination)
    return cloned


def clone_or_copy2(source: Any, destination: Any) -> Any:
    """``shutil.copy2`` (bytes, mode, times; a ``copytree`` copy function) as a clone where ``clone_file`` can
    make one, ``copy2`` itself otherwise, its refusals included. Its ACL is the destination's inherited one,
    as with ``copy2`` (the clone passes no ``CLONE_ACL``). Unlike macOS ``copy2`` it keeps the source's
    compression and extended attributes, which no reader here looks at. A flag that refuses new times
    (immutable, append-only) would travel too, so that source is left to ``copy2``."""
    if not getattr(os.stat(source), "st_flags", 0) & _REFUSES_TIMES and clone_file(source, destination):
        shutil.copystat(source, destination)
        return destination
    return shutil.copy2(source, destination)


def _clonefile(source: pathlib.Path, destination: pathlib.Path) -> bool:
    import ctypes

    try:
        clonefile = ctypes.CDLL(None, use_errno=True).clonefile
    except (OSError, AttributeError):
        return False
    clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    clonefile.restype = ctypes.c_int
    # A refusal (another volume, a non-APFS filesystem, an unreadable source) leaves no file behind:
    # the streamed copy then succeeds or reports the real fault with its own message.
    return clonefile(os.fsencode(source), os.fsencode(destination), 0) == 0


def _ficlone(source: pathlib.Path, destination: pathlib.Path) -> bool:
    import fcntl

    try:
        # Non-blocking open: a FIFO swapped in since the regular-file check never parks this thread.
        reader = os.open(source, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(reader).st_mode):
            return False
        with open(destination, "xb") as writer:
            try:
                fcntl.ioctl(writer.fileno(), _FICLONE, reader)
                return True
            except OSError:
                pass
    except OSError:
        return False
    finally:
        os.close(reader)
    destination.unlink(missing_ok=True)  # the empty file of a refused clone
    return False


def compression_sibling(path: pathlib.Path) -> pathlib.Path:
    """The sibling ``_compress_macos`` writes: ``.{name}.decmpfs.tmp.<pid>.<tid>.<uuid>``, the atomic
    signature under this module's own tag, so ``temp_sweep`` still reaps it where it walks."""
    return temporary_sibling(path, tag=_SIBLING_TAG)


def is_compression_sibling(name: str) -> bool:
    """A name ``compression_sibling`` made: the only temporaries the ``record_compression`` job reaps
    (any other atomic-signature name is another writer's, or a record that merely looks like one)."""
    return is_atomic_temp(name) and name.rsplit(".tmp.", 1)[0].endswith("." + _SIBLING_TAG)


def compression_supported() -> bool:
    """Transparent compression exists here: decmpfs on macOS, NTFS on Windows. Linux, Docker and Colab
    keep ordinary files: Btrfs needs its compression attribute BEFORE the data write."""
    return sys.platform in ("darwin", "win32")


def compress_in_place(path: pathlib.Path, *, owned: Optional[Callable[[], bool]] = None) -> bool:
    """Best-effort transparent compression (``compression_outcome``): True when PATH is now compressed.
    A refusal is only a missed space saving, never a failed patch publication."""
    return compression_outcome(path, owned=owned) == "compressed"


def compression_outcome(path: pathlib.Path, *, owned: Optional[Callable[[], bool]] = None,
                        background: bool = False) -> str:
    """Compress PATH in place, logical size, bytes and sha256 preserved, or say why it stayed: the one
    floor every caller shares. ``unsupported`` off macOS and Windows; one ``lstat`` decides
    ``not_regular``, ``dataless`` (macOS: evicted, a read would download it), ``small`` (under 64 KiB),
    ``linked`` (another name shares the inode) and ``compressed_already``; macOS keeps a file that may
    share blocks with a clone (``shared``;
    ``share_unknown`` when the volume does not say: compressing a shared file would unshare it), Windows
    a read-only one (``read_only``: its file control needs a writable handle); a first 64 KiB block zlib
    cannot shrink by one allocation block is ``incompressible``; free space below the file's size is
    ``low_space``. macOS hashes, writes a ``ditto --hfsCompression`` sibling (``compression_sibling``;
    BACKGROUND, the ``record_compression`` job's, runs it at the utility disk tier through ``taskpolicy``),
    verifies it, asks OWNED (else ``not_owned``), then re-stats the original right before the replace
    (device, inode, size, mtime: else ``changed``); NTFS asks OWNED, compresses in place through its
    file control and verifies the bytes after. Anything else is ``failed``."""
    path = pathlib.Path(path)
    if not compression_supported():
        return "unsupported"
    try:
        before = os.lstat(path)
        reason = _floor(path, before)
        if reason:
            return reason
        if sys.platform == "darwin":
            return _compress_macos(path, before, owned, background)
        return _compress_ntfs(path, owned)
    except (OSError, subprocess.SubprocessError):
        return "failed"


def _floor(path: pathlib.Path, info: os.stat_result) -> str:
    """Why PATH stays as it is before any rewrite, or ``""``."""
    if not stat.S_ISREG(info.st_mode):
        return "not_regular"
    if getattr(info, "st_flags", 0) & _SF_DATALESS:
        return "dataless"
    if info.st_size < FLOOR_BYTES:
        return "small"
    if info.st_nlink != 1:
        return "linked"
    attributes = getattr(info, "st_file_attributes", 0)
    if getattr(info, "st_flags", 0) & stat.UF_COMPRESSED or attributes & stat.FILE_ATTRIBUTE_COMPRESSED:
        return "compressed_already"
    if sys.platform == "darwin":
        shared = _may_share(path)
        if shared is not False:
            return "share_unknown" if shared is None else "shared"
    elif attributes & stat.FILE_ATTRIBUTE_READONLY:
        return "read_only"
    with open(path, "rb") as source:
        block = source.read(FLOOR_BYTES)
    if len(zlib.compress(block, 1)) + _BLOCK > len(block):
        return "incompressible"
    if shutil.disk_usage(path.parent).free < info.st_size:
        return "low_space"
    return ""


def _may_share(path: pathlib.Path) -> Optional[bool]:
    """macOS: whether PATH may share blocks with another file (an APFS clone, which ``nlink`` cannot
    see): ``EF_MAY_SHARE_BLOCKS`` of the ``getattrlist`` extended flags; None when the volume does not say."""
    import ctypes
    import struct

    class AttrList(ctypes.Structure):  # <sys/attr.h> struct attrlist
        _fields_ = [("bitmapcount", ctypes.c_ushort), ("reserved", ctypes.c_uint16), ("commonattr", ctypes.c_uint32),
                    ("volattr", ctypes.c_uint32), ("dirattr", ctypes.c_uint32), ("fileattr", ctypes.c_uint32),
                    ("forkattr", ctypes.c_uint32)]

    try:
        getattrlist = ctypes.CDLL(None, use_errno=True).getattrlist
    except (OSError, AttributeError):
        return None
    getattrlist.argtypes = (ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
    getattrlist.restype = ctypes.c_int
    # ATTR_CMN_RETURNED_ATTRS, then ATTR_CMNEXT_EXT_FLAGS (forkattr under FSOPT_ATTR_CMN_EXTENDED), FSOPT_NOFOLLOW.
    request, answer = AttrList(5, 0, 0x80000000, 0, 0, 0, 0x200), ctypes.create_string_buffer(64)
    if getattrlist(os.fsencode(path), ctypes.byref(request), answer, ctypes.sizeof(answer), 0x1 | 0x20) != 0:
        return None
    if not struct.unpack_from("=5I", answer.raw, 4)[4] & 0x200:  # the volume returned no extended flags
        return None
    return bool(struct.unpack_from("=Q", answer.raw, 24)[0] & 0x1)  # EF_MAY_SHARE_BLOCKS, <sys/stat.h>


def _compress_macos(path: pathlib.Path, before: os.stat_result, owned: Optional[Callable[[], bool]],
                    background: bool = False) -> str:
    from ouroboros.artifacts import stream_artifact_file
    from ouroboros.config import ARTIFACT_COMPRESSION_TIMEOUT_SEC

    expected = stream_artifact_file(path)
    temporary = compression_sibling(path)  # a hard kill's leftover is reaped (own tag), never a deliverable
    try:
        subprocess.run(
            [*(_UTILITY_DISK if background else ()), "/usr/bin/ditto", "--hfsCompression", str(path), str(temporary)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=ARTIFACT_COMPRESSION_TIMEOUT_SEC,
        )
        stream_artifact_file(temporary, expected=expected)
        if not getattr(os.lstat(temporary), "st_flags", 0) & stat.UF_COMPRESSED:
            return "incompressible"
        if owned is not None and not owned():  # it may wait on the store: before the last look, never after
            return "not_owned"
        now = os.lstat(path)
        if (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
            return "changed"  # a writer touched it since: its bytes win
        os.replace(temporary, path)
        return "compressed"
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _compress_ntfs(path: pathlib.Path, owned: Optional[Callable[[], bool]]) -> str:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    from ouroboros.artifacts import stream_artifact_file

    expected = stream_artifact_file(path)
    if owned is not None and not owned():
        return "not_owned"
    control = ctypes.WinDLL("kernel32", use_last_error=True).DeviceIoControl
    control.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                        wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID)
    control.restype = wintypes.BOOL
    mode, returned = ctypes.c_ushort(1), wintypes.DWORD()  # COMPRESSION_FORMAT_DEFAULT
    with path.open("r+b") as source:
        done = bool(control(msvcrt.get_osfhandle(source.fileno()), 0x9C040,  # FSCTL_SET_COMPRESSION
                            ctypes.byref(mode), ctypes.sizeof(mode), None, 0, ctypes.byref(returned), None))
    if not done:
        return "failed"
    stream_artifact_file(path, expected=expected)  # the filesystem compressed in place: the bytes are unchanged
    return "compressed"

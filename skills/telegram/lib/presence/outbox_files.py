"""The skill's own copies of files queued for Presence sending.

``telegram_send`` copies each file here when it queues it, unchanged and under its own name, and records the copy's
size and SHA-256 and the path it came from in the row as ``_frozen``. The outbox sends that copy, checked against the
record before each attempt. The copy stays with the row's custody history, as the row does, whatever the row's end:
an uncertain send keeps the exact bytes it may have delivered. The caller's file is read once and never changed,
moved or removed. Only a copy that no row names, made by a ``telegram_send`` that queued nothing, is removed.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import pathlib
import re
import stat
import threading
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

DIRECTORY = "outbox_files"
_CHUNK = 1024 * 1024
MAX_OUTBOX_FILES_BYTES = 512 * 1024 * 1024
_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: Dict[str, threading.Lock] = {}


class CopyTooLarge(ValueError):
    """The file grew past the upload limit while it was being copied; nothing was kept."""


class OutboxStorageFull(OSError):
    """The skill's frozen outbox_files store reached its disk cap."""


class StagingStorageFull(OSError):
    """The skill's inbound staging store reached its disk cap."""


class FrozenFileMissing(ValueError):
    """Media/album outbox row lacks a verified frozen copy inside outbox_files/."""


_MEDIA_KINDS = frozenset({"photo", "document", "video", "audio", "voice", "animation"})


@contextlib.contextmanager
def _quota_lock(state_dir: Any, name: str) -> Iterator[None]:
    base = pathlib.Path(state_dir)
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = base / f".{name}.quota.lock"
    key = os.path.realpath(lock_path)
    with _LOCKS_GUARD:
        tlock = _THREAD_LOCKS.setdefault(key, threading.Lock())
    with tlock:
        if fcntl is not None:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        else:
            from ...scripts.platform_support import acquire_file_lock, release_file_lock

            fd = acquire_file_lock(lock_path, timeout_sec=30.0)
            try:
                yield
            finally:
                release_file_lock(fd)


def _dir_bytes(root: pathlib.Path) -> int:
    total = 0
    if not root.is_dir():
        return 0
    for item in root.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def stage_budget_remaining(state_dir: Any, cap_bytes: int) -> int:
    """Current remaining staging byte budget under the atomic staging quota lock."""
    if cap_bytes <= 0:
        return 0
    staging_root = pathlib.Path(state_dir) / "staging"
    with _quota_lock(state_dir, "staging"):
        return max(0, int(cap_bytes) - _dir_bytes(staging_root))


def commit_staged_file(state_dir: Any, temp_path: pathlib.Path, destination: pathlib.Path, cap_bytes: int) -> None:
    """Atomically verify actual downloaded bytes against ``cap_bytes`` and move into ``staging/``."""
    temp_path, destination = pathlib.Path(temp_path), pathlib.Path(destination)
    staging_root = pathlib.Path(state_dir) / "staging"
    with _quota_lock(state_dir, "staging"):
        actual = temp_path.stat().st_size if temp_path.is_file() else 0
        current = _dir_bytes(staging_root)
        if cap_bytes > 0 and current + actual > cap_bytes:
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise StagingStorageFull(f"staging disk cap ({cap_bytes:,} bytes) reached")
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temp_path.replace(destination)


def sweep_staging_tmp(state_dir: Any) -> None:
    """Remove leftover interrupted staging downloads in ``staging_tmp/`` on startup."""
    tmp_root = pathlib.Path(state_dir) / "staging_tmp"
    if not tmp_root.is_dir():
        return
    for item in tmp_root.iterdir():
        if item.is_file():
            with contextlib.suppress(OSError):
                item.unlink(missing_ok=True)


def remove_staged_files(
    state_dir: Any, staged: Sequence[pathlib.Path], *, on_error: Callable[[str], None] = lambda _m: None
) -> None:
    staging_root = (pathlib.Path(state_dir) / "staging").resolve(strict=False)
    parents = set()
    for path in staged:
        try:
            path.unlink(missing_ok=True)
            parents.add(path.parent)
        except OSError:
            on_error(f"telegram presence: could not remove staged file {path.name}")
    for parent in parents:
        resolved = parent.resolve(strict=False)
        if resolved != staging_root and resolved.is_relative_to(staging_root):
            with contextlib.suppress(OSError):
                parent.rmdir()


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:160] or "event"


def declared_size(attachment: Dict[str, Any]) -> int:
    try:
        return int(attachment.get("file_size") or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def safe_filename(value: str) -> str:
    name = pathlib.PurePath(value.replace("\\", "/")).name
    clean = re.sub(r"[^A-Za-z0-9_. -]+", "_", name).strip(" .")
    if len(clean) > 160:
        suffix = pathlib.PurePath(clean).suffix
        clean = clean[: 160 - len(suffix)] + suffix if len(suffix) < 160 else clean[:160]
    return clean or "attachment"


def keep_fd(
    src_fd: int, source: pathlib.Path, state_dir: Any, max_bytes: int, *, storage_cap: int = MAX_OUTBOX_FILES_BYTES
) -> Dict[str, Any]:
    """Copy bytes directly from an already-opened, root-confined ``src_fd`` into ``outbox_files/``."""
    source = pathlib.Path(source)
    root = pathlib.Path(state_dir) / DIRECTORY
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _quota_lock(state_dir, "outbox_files"):
        current = _dir_bytes(root)
        if storage_cap > 0 and current >= storage_cap:
            raise OutboxStorageFull(f"outbox_files storage limit ({storage_cap:,} bytes) reached")
        folder = root / uuid.uuid4().hex
        folder.mkdir(mode=0o700)
        target = folder / safe_filename(source.name)
        digest, size = hashlib.sha256(), 0
        try:
            st = os.fstat(src_fd)
            if not stat.S_ISREG(st.st_mode):
                raise PermissionError(f"{source.name} is not a regular file")
            src_size = st.st_size
            if src_size > max_bytes:
                raise CopyTooLarge(f"{source.name} exceeds {max_bytes:,} bytes")
            if storage_cap > 0 and current + src_size > storage_cap:
                raise OutboxStorageFull(f"outbox_files storage limit ({storage_cap:,} bytes) exceeded")
            os.lseek(src_fd, 0, os.SEEK_SET)
            created = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
                0o600,
            )
            with os.fdopen(created, "wb") as writer:
                while chunk := os.read(src_fd, _CHUNK):
                    size += len(chunk)
                    if size > max_bytes:
                        raise CopyTooLarge(f"{source.name} grew past {max_bytes:,} bytes while it was copied")
                    if storage_cap > 0 and current + size > storage_cap:
                        raise OutboxStorageFull(f"outbox_files storage limit ({storage_cap:,} bytes) exceeded")
                    digest.update(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
        except BaseException:
            discard([{"path": str(target)}], state_dir)
            raise
    return {"path": str(target), "sha256": digest.hexdigest(), "size": size, "source": str(source)}


def keep(source: pathlib.Path, state_dir: Any, max_bytes: int, *, storage_cap: int = MAX_OUTBOX_FILES_BYTES) -> Dict[str, Any]:
    """Copy ``source`` into a new folder of its own and return the row's record of the copy."""
    source = pathlib.Path(source)
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    src_fd = os.open(source, read_flags)
    try:
        return keep_fd(src_fd, source, state_dir, max_bytes, storage_cap=storage_cap)
    finally:
        os.close(src_fd)


def records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The copies a row names."""
    kept = payload.get("_frozen")
    return [item for item in kept if isinstance(item, dict) and item.get("path")] if isinstance(kept, list) else []


def _open_frozen_copy_fd(path_str: str, state_dir: Optional[Any]) -> int:
    leaf_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    if state_dir is None:
        return os.open(path_str, leaf_flags)
    root = (pathlib.Path(state_dir) / DIRECTORY).resolve(strict=True)
    target = pathlib.Path(path_str)
    try:
        rel = target.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise FrozenFileMissing("frozen copy lies outside outbox_files/") from exc
    if len(rel.parts) != 2:
        raise FrozenFileMissing("frozen copy path is not directly inside an outbox_files/ copy folder")
    if os.name != "nt" and os.open in getattr(os, "supports_dir_fd", ()):
        dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        root_fd = os.open(root, dir_flags)
        try:
            sub_fd = os.open(rel.parts[0], dir_flags | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
        finally:
            os.close(root_fd)
        try:
            return os.open(rel.parts[1], leaf_flags, dir_fd=sub_fd)
        finally:
            os.close(sub_fd)
    from ouroboros.confined_files import open_confined_file

    handle, _st = open_confined_file(root, rel.parts)
    try:
        return os.dup(handle.fileno())
    finally:
        handle.close()


def close_verified(verified_fds: Optional[Dict[str, Tuple[int, int]]]) -> None:
    """Close all open verified descriptors returned by :func:`open_verified`."""
    if not verified_fds:
        return
    for fd, _size in verified_fds.values():
        try:
            os.close(fd)
        except OSError:
            pass
    verified_fds.clear()


async def run_owned_in_thread(fn: Callable[[], Any], closer: Callable[[Any], None]) -> Any:
    """Run ``fn`` in a worker thread with deterministic resource cleanup across cancel and loop shutdown."""
    lock = threading.Lock()
    state: Dict[str, Any] = {"abandoned": False, "done": False, "result": None}

    def _worker() -> Any:
        with lock:
            if state["abandoned"]:
                return None
        res = fn()
        with lock:
            if state["abandoned"]:
                must_close = True
            else:
                state["result"], state["done"], must_close = res, True, False
        if must_close:
            closer(res)
        return res

    try:
        return await asyncio.to_thread(_worker)
    except BaseException:
        with lock:
            state["abandoned"] = True
            res_to_close = state["result"] if state["done"] else None
            state["result"] = None
        if res_to_close is not None:
            closer(res_to_close)
        raise


def open_verified(
    payload: Dict[str, Any], *, state_dir: Optional[Any] = None, require_for_media: bool = False
) -> Dict[str, Tuple[int, int]]:
    """Open, verify (size, SHA-256, confinement), rewind, and return ``{path: (fd, size)}`` for frozen copies.

    Callers own the returned descriptors and MUST close them via :func:`close_verified`. On any failure,
    every opened descriptor is closed before raising so zero-TOCTOU uploads never reopen by pathname.
    """
    kept = records(payload)
    kind = str(payload.get("kind") or "message")
    if require_for_media:
        if kind in _MEDIA_KINDS:
            expected_path = str(payload.get("file_path") or "")
            if len(kept) != 1 or not expected_path or str(kept[0].get("path") or "") != expected_path:
                raise FrozenFileMissing("media row lacks a verified frozen copy in outbox_files; held")
        elif kind == "album":
            items = payload.get("_rendered_items") or payload.get("items") or []
            if not isinstance(items, list) or not items or len(kept) != len(items):
                raise FrozenFileMissing("album row lacks complete verified frozen copies in outbox_files; held")
            for idx, item in enumerate(items):
                if not isinstance(item, dict) or str(item.get("file_path") or "") != str(kept[idx].get("path") or ""):
                    raise FrozenFileMissing("album item path does not match its verified frozen copy; held")
    opened: Dict[str, Tuple[int, int]] = {}
    verified_meta: Dict[str, Tuple[Any, Any]] = {}
    try:
        for record in kept:
            path_str = str(record["path"])
            name = pathlib.PurePath(path_str).name
            if path_str in opened:
                if verified_meta[path_str] != (record.get("size"), record.get("sha256")):
                    raise ValueError(f"the copy of {name} kept when it was queued has changed since; not sent")
                continue
            digest, size = hashlib.sha256(), 0
            try:
                fd = _open_frozen_copy_fd(path_str, state_dir)
            except FrozenFileMissing:
                raise
            except OSError as exc:
                raise ValueError(
                    f"the copy of {name} kept when it was queued cannot be read ({type(exc).__name__}); not sent"
                ) from exc
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError(f"the copy of {name} is not a regular file; not sent")
                while True:
                    chunk = os.read(fd, _CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    digest.update(chunk)
                if size != record.get("size") or digest.hexdigest() != record.get("sha256"):
                    raise ValueError(f"the copy of {name} kept when it was queued has changed since; not sent")
                os.lseek(fd, 0, os.SEEK_SET)
            except BaseException:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
            opened[path_str] = (fd, size)
            verified_meta[path_str] = (size, digest.hexdigest())
        return opened
    except BaseException:
        close_verified(opened)
        raise


def verify(payload: Dict[str, Any], *, state_dir: Optional[Any] = None, require_for_media: bool = False) -> None:
    """Verify every frozen copy matches its size, SHA-256, and (when ``state_dir`` is given) lives in ``outbox_files/``."""
    opened = open_verified(payload, state_dir=state_dir, require_for_media=require_for_media)
    close_verified(opened)


def discard(kept: List[Dict[str, Any]], state_dir: Any) -> None:
    """Remove copies that no row names: a ``telegram_send`` that queued nothing made them.

    Only a file directly inside one of this skill's copy folders is removed, never anything outside them.
    """
    root = os.path.realpath(pathlib.Path(state_dir) / DIRECTORY)
    for record in kept:
        path = str(record.get("path") or "")
        folder = os.path.realpath(os.path.dirname(path))
        if not path or os.path.dirname(folder) != root:
            continue
        try:
            os.unlink(os.path.join(folder, os.path.basename(path)))
        except FileNotFoundError:
            pass
        try:
            os.rmdir(folder)
        except OSError:
            pass

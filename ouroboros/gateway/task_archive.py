"""Confined reads of a task's recorded files: one file, or one directory as a ZIP (TZ-1 V12).

A task's files live in its own stores only (``task_custody.task_artifact_stores``: the
canonical store, then each own child drive's); a recorded path is attributed to one of
them once, lexically (``task_artifact_location``; a path recorded through the configured
root's spelling is re-spelled onto its physical directory), and the bytes that leave are
then read from ONE open descriptor reached by a confined descent from the host-configured canonical
root. ``confined_files`` opens each plain component relative to a held directory:
POSIX no-follow descriptors or Windows NT RootDirectory/DONT_REPARSE handles.
Descendant symlinks, junctions and nonregular files are refused; the configured
root itself may be a legitimate junction. On this route, admission facts authorize a
child's metadata store before its path-based SQLite reader; scratch existence is not admission.
Payload reads never reopen a checked pathname. A captured file (an immutable row,
content-addressed chat media
whose name is its sha256, or a delegated run's retained journal range named by its digest)
is verified INTO a private spool and only the spool is served, so the bytes on the wire
are exactly the verified ones. The same holds for any row that records a digest: a
mutable file's bytes that no longer match it are refused typed (409
``artifact_identity_changed``, naming the recorded digest) rather than served
under an identity a parent's disposition may have bound; a listing without a digest
streams its current bytes and says so (``x-ouroboros-artifact-identity: unmeasured``).

A directory archive includes only recorded regular files lexically under its
own-store directory, with no failed/missing capture. One relpath is one member
(canonical store first), named relative to the directory's parent. It
spools in 1 MiB chunks through one anonymous temporary file (bounded memory, exact
``Content-Length``, status settled before the first byte); a member that records a
digest must match it while read, or the answer is a typed refusal. Wrap up's saved-work
document (``?wrap_up=``) is written on read into such a spool the same way.

Without a supported confined backend, ordinary files, archives and delegated
journals return typed HTTP 503, never an unconfined fallback. Published result and
host-acceptance sources retain their separate verified-source reader contract.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import mimetypes
import os
import pathlib
import re
import stat
import tempfile
import time
import zipfile
from email.utils import formatdate
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote

import anyio
from starlette.responses import Response, StreamingResponse

from ouroboros import artifacts as artifact_store
from ouroboros.confined_files import (DIR_FLAGS, CONFINED_OPEN_AVAILABLE, IS_WINDOWS, confined_directory,
                                     lexical_relative, open_regular_at, relative_parts, root_spellings)
from ouroboros.gateway._helpers import json_error
from ouroboros.task_custody import task_artifact_stores
from ouroboros.task_status import load_effective_task_result

log = logging.getLogger(__name__)
ARCHIVE_SUFFIX = ".zip"
_CHUNK = 1024 * 1024
# A recorded status that still offers bytes: ``ready`` or none (legacy and unmeasured
# listings state nothing; their bytes are what is on disk now).
_SERVABLE_STATUSES = frozenset({"", "ready"})
# Gone, lost a directory, or turned into a link (ELOOP; EMLINK on some BSDs): not an I/O fault.
_MEMBER_GONE = frozenset({errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EMLINK})
CONFINED, _DIR_FLAGS = CONFINED_OPEN_AVAILABLE, DIR_FLAGS
_CHAT_MEDIA_DIGEST_RE = re.compile(r"chat-media-([0-9a-f]{64})\.[a-z0-9]+")
_DELEGATED_SOURCE_RE = re.compile(r"source_handles/delegated_activity/([A-Za-z0-9][A-Za-z0-9_.-]*-([0-9a-f]{64})\.jsonl)")
_ACCEPTANCE_SOURCE_RE = re.compile(r"source_handles/context_checkpoints/(acceptance-([0-9a-f]{64})\.json)")
# An SVG is an image a browser also runs as a document. Its exact bytes, type and static
# view stay; opened directly it gets the chat-upload inline policy (gateway/files.py):
# no script or application origin, no sniffing. Other types keep their headers.
_SVG_POLICY = {"content-security-policy": "sandbox", "x-content-type-options": "nosniff"}

Route = Tuple[pathlib.Path, List[str]]  # (directory opened by absolute path, segments below it)


def task_artifact_location(stores: List[pathlib.Path], raw_path: Any, spellings: Any = ()) -> Optional[tuple]:
    """Attribute lexically (``spellings``: ``root_spellings`` of the canonical root the
    stores were built from); only the subsequent held walk may touch descendants."""
    return next(((index, pathlib.Path(raw_path), relative) for index, store in enumerate(stores)
                 if (relative := lexical_relative(store, raw_path, spellings))), None)


def plain_segments(relpath: Any) -> List[str]:
    """The segments of a store-relative path, or [] when any is empty/``.``/``..`` or the text
    carries a backslash or NUL."""
    return relative_parts(relpath)


def recorded_identity(row: Any) -> Optional[Dict[str, Any]]:
    """The identity a row records for its bytes (a measured digest, immutable or not); None for
    an unmeasured listing, which states nothing about them."""
    return row if isinstance(row, dict) and row.get("sha256") and row.get("measured") is not False else None


def _eligible(row: Dict[str, Any]) -> bool:
    """A row whose capture still offers bytes (an immutable one only with the identity to verify)."""
    if row.get("immutable") and (not isinstance(row.get("size"), int) or not row.get("sha256")):
        return False
    return not row.get("errors") and not row.get("copy_status") \
        and str(row.get("status") or "").strip().lower() in _SERVABLE_STATUSES


def _route(anchor: Optional[pathlib.Path], store: pathlib.Path, relpath: str) -> Optional[Route]:
    segments = plain_segments(relpath)
    if not segments:
        return None
    if anchor is not None and store != anchor and store.is_relative_to(anchor):
        return anchor, [*store.relative_to(anchor).parts, *segments]
    return store, segments


class _Parents:
    """One confined directory descriptor at a time; consecutive members under the same
    directory reuse it instead of repeating the descent. A failed descent is not kept."""

    def __init__(self) -> None:
        self._key: Optional[tuple] = None
        self._fd = -1
        self._held = self.directory = None

    def __enter__(self) -> "_Parents":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def get(self, root: pathlib.Path, directories: List[str]) -> int:
        key = (root, tuple(directories))
        if key != self._key:
            self.close()
            held = confined_directory(root, directories)
            directory = held.__enter__()
            self._held, self.directory = held, directory
            self._fd, self._key = directory.handle, key
        return self._fd

    def close(self) -> None:
        if self._fd >= 0:
            self._fd, self._key = -1, None
            held, self._held, self.directory = self._held, None, None
            held.__exit__(None, None, None)


def _member_stat(parents: _Parents, route: Optional[Route]) -> Optional[os.stat_result]:
    """No-follow stat through the confined descent; None unless a regular file is there."""
    if route is None or not CONFINED:
        return None
    root, segments = route
    try:
        if IS_WINDOWS:
            handle, observed = _open_member(parents, route)
            handle.close()
            return observed
        observed = os.stat(segments[-1], dir_fd=parents.get(root, segments[:-1]), follow_symlinks=False)
    except OSError:
        return None
    return observed if stat.S_ISREG(observed.st_mode) else None


def _open_member(parents: _Parents, route: Route) -> Tuple[Any, os.stat_result]:
    """A binary handle on the member itself plus the fstat of that descriptor; anything but a
    regular file is refused (an OSError without errno, like a failed verification)."""
    root, segments = route
    parent = parents.get(root, segments[:-1])
    return (parents.directory.open(segments[-1]) if IS_WINDOWS else open_regular_at(parent, segments[-1]))


def directory_archives(stores: List[pathlib.Path], rows: Any, *, anchor: Any = None) -> Dict[str, Dict[str, Any]]:
    """Per top-level result directory, what ``?archive=<dir>`` would stream now: ``files``/``size``
    of its members from one confined no-follow stat each (no hashing), the recorded rows it leaves
    out (``excluded``) and ``available`` iff it has a member (never where the platform cannot confine)."""
    spellings = root_spellings(anchor) if anchor is not None else ()
    anchor = spellings[0] if spellings else None
    view: Dict[str, Dict[str, Any]] = {}
    picked: Dict[str, Tuple[tuple, int]] = {}
    with _Parents() as parents:
        for order, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            location = task_artifact_location(stores, row.get("path"), spellings)
            relpath = location[2] if location else str(row.get("relpath") or "")
            parts = plain_segments(relpath)
            if len(parts) < 2:
                continue  # a root file (or an unattributable row) belongs to no directory
            entry = view.setdefault(parts[0], {"name": parts[0] + ARCHIVE_SUFFIX, "files": 0, "size": 0,
                                               "excluded": 0, "available": False})
            observed = (_member_stat(parents, _route(anchor, stores[location[0]], relpath))
                        if location and _eligible(row) else None)
            if observed is None:
                entry["excluded"] += 1
            elif relpath not in picked or (location[0], order) < picked[relpath][0]:
                picked[relpath] = ((location[0], order), observed.st_size)
    for relpath, (_rank, size) in picked.items():
        entry = view[relpath.split("/", 1)[0]]
        entry["files"] += 1
        entry["size"] += size
    for entry in view.values():
        entry["available"] = entry["files"] > 0
    return view


def _archive_members(stores: List[pathlib.Path], rows: Any, directory: str, *,
                     anchor: Any) -> List[Tuple[str, Route, Dict[str, Any]]]:
    """``(member name, route, row)`` of every eligible recorded file under DIRECTORY that a
    confined stat finds, once per relpath (canonical store first), sorted by name."""
    spellings = root_spellings(anchor)
    anchor = spellings[0]
    prefix, base = directory + "/", (directory.rsplit("/", 1)[0] + "/" if "/" in directory else "")
    picked: Dict[str, Tuple[tuple, Route, Dict[str, Any]]] = {}
    with _Parents() as parents:
        for order, row in enumerate(rows if isinstance(rows, list) else []):
            location = task_artifact_location(stores, row.get("path"), spellings) if isinstance(row, dict) else None
            if location is None or not location[2].startswith(prefix) or not _eligible(row):
                continue
            route = _route(anchor, stores[location[0]], location[2])
            if route is not None and _member_stat(parents, route) is not None and (
                    location[2] not in picked or (location[0], order) < picked[location[2]][0]):
                picked[location[2]] = ((location[0], order), route, row)
    return [(relpath[len(base):], route, row) for relpath, (_rank, route, row) in sorted(picked.items())]


def _zip_info(name: str, observed: os.stat_result) -> zipfile.ZipInfo:
    """What ``ZipInfo.from_file(strict_timestamps=False)`` records, from the open handle's fstat."""
    date_time = time.localtime(observed.st_mtime)[:6]
    date_time = max((1980, 1, 1, 0, 0, 0), min(date_time, (2107, 12, 31, 23, 59, 59)))
    info = zipfile.ZipInfo(name, date_time)
    info.external_attr = (observed.st_mode & 0xFFFF) << 16
    info.file_size = observed.st_size
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _drain(spool: Any) -> Iterator[bytes]:
    try:
        spool.seek(0)
        yield from iter(lambda: spool.read(_CHUNK), b"")
    finally:
        spool.close()


def serve_directory_archive(drive_root: Any, task_id: str, name: str, directory: str, *,
                            other_selectors: bool) -> Response:
    """``GET /api/tasks/{task_id}/artifacts/{basename}.zip?archive=<directory>``: the recorded
    directory's ZIP, or a typed refusal - 400 ``artifact_archive_invalid``, 404 ``task not found``
    / ``artifact_archive_empty`` / ``artifact_archive_unverified`` (a member changed, vanished,
    became a link or failed its capture while read), 503 ``artifact_archive_unavailable`` (no
    confinement on this platform, or the spool or a read failed)."""
    parts = plain_segments(directory)
    refusal = ("archive must name a store-relative directory in plain segments, without relpath, source or wrap_up"
               if other_selectors or not parts else
               f"archive name must be the directory's basename plus {ARCHIVE_SUFFIX}"
               if name != parts[-1] + ARCHIVE_SUFFIX else "")
    if refusal:
        return json_error(refusal, 400, reason_code="artifact_archive_invalid", task_id=task_id, artifact=name)
    unavailable = {"reason_code": "artifact_archive_unavailable", "task_id": task_id, "artifact": name,
                   "directory": directory}
    if not CONFINED:
        return json_error("archive confinement needs directory-relative no-follow opens, which this platform "
                          "lacks (issue #1297)", 503, **unavailable)
    result = load_effective_task_result(drive_root, task_id) or {}
    if not result:
        return json_error("task not found", 404)
    members = _archive_members(task_artifact_stores(drive_root, task_id), result.get("artifacts"), directory,
                               anchor=drive_root)
    if not members:
        return json_error("no recorded ready file of this task lies in that directory", 404,
                          reason_code="artifact_archive_empty", task_id=task_id, artifact=name, directory=directory)
    member = ""
    try:
        spool = tempfile.TemporaryFile()
    except OSError:
        return json_error("archive could not be built", 503, member=member, **unavailable)
    try:
        with zipfile.ZipFile(spool, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive, \
                _Parents() as parents:
            for member, route, row in members:
                handle, observed = _open_member(parents, route)  # the handle, never the path, from here on
                with handle, archive.open(_zip_info(member, observed), "w") as sink:
                    artifact_store.stream_artifact_file(handle, sink, expected=recorded_identity(row))
    except OSError as exc:
        spool.close()
        if exc.errno is None or exc.errno in _MEMBER_GONE:  # verification failures carry no errno
            return json_error("archive member is missing, changed while read, or failed its capture verification",
                              404, reason_code="artifact_archive_unverified", task_id=task_id, artifact=name,
                              directory=directory, member=member)
        return json_error("archive could not be built", 503, member=member, **unavailable)
    except BaseException:
        spool.close()
        raise
    size = spool.tell()
    quoted = quote(name)
    disposition = f"attachment; filename*=utf-8''{quoted}" if quoted != name else f'attachment; filename="{name}"'
    return StreamingResponse(_drain(spool), media_type="application/zip",
                             headers={"Content-Length": str(size), "Content-Disposition": disposition})


def serve_wrap_up_document(drive_root: Any, task_id: str, name: str, request_id: str, *,
                           other_selectors: bool) -> Response:
    """``GET /api/tasks/{task_id}/artifacts/{name}?wrap_up=<Stop request id>``: Wrap up's
    saved-work document named by the task detail's ``wrap_up_document``, written on read from
    the stored receipt and the exact sources it names (``owner_stop_receipt.write_wrap_up_document``)
    into one anonymous spool, then served like a task file (ranges, HEAD, ``unmeasured``: no
    identity was recorded). Nothing is stored. Typed refusals: 400 ``wrap_up_document_invalid``
    (relpath or source; with ``archive`` its own 400 answers first), 404 ``task not found`` /
    ``wrap_up_document_unavailable`` (no such receipt, name and Stop), 503
    ``wrap_up_document_unavailable`` (the spool or the receipt is unwritable)."""
    from ouroboros.owner_stop_receipt import wrap_up_document_view, write_wrap_up_document

    refusal = {"reason_code": "wrap_up_document_unavailable", "task_id": task_id, "artifact": name}
    if other_selectors:
        return json_error("wrap_up takes no relpath, source or archive", 400,
                          **{**refusal, "reason_code": "wrap_up_document_invalid"})
    result = load_effective_task_result(drive_root, task_id, materialize_artifacts=False) or {}
    if not result:
        return json_error("task not found", 404)
    if wrap_up_document_view(result) != {"name": name, "request_id": request_id}:
        return json_error("this task keeps no Wrap up saved work by that name and Stop", 404, **refusal)
    try:
        spool = tempfile.TemporaryFile()
    except OSError:
        return json_error("saved work could not be written", 503, **refusal)
    try:
        write_wrap_up_document(drive_root, result, spool)
        spool.flush()
        observed = os.fstat(spool.fileno())
    except Exception:
        spool.close()
        log.warning("Wrap up saved work could not be written for %s", task_id, exc_info=True)
        return json_error("saved work could not be written", 503, **refusal)
    return DescriptorResponse(spool, name, observed, media_type="text/markdown")


def chat_media_identity(name: str) -> Dict[str, Any]:
    """The capture identity a content-addressed chat-media name promises."""
    match = _CHAT_MEDIA_DIGEST_RE.fullmatch(str(name or ""))
    if match is None:
        raise ValueError(f"not a content-addressed chat media name: {name!r}")
    return {"sha256": match.group(1)}


def serve_task_source(drive_root: Any, stores: List[pathlib.Path], result: Dict[str, Any], task_id: str,
                      name: str, source: str) -> Response:
    """``?source=<store-relative path>``: one source a task's own records name.

    A delegated run's retained journal range (``delegate_activity``) is content-addressed
    like chat media: only ``source_handles/delegated_activity/<id>-<sha256>.jsonl`` whose
    basename is ``name`` qualifies, read from the task's own stores (canonical first)
    through the confined descent and verified against the digest its name carries. Any
    other path must be a source the task result publishes, returned as its verified JSON,
    or this author's own host acceptance record (history: the retired task acceptance
    wrote it) named by its digest: an old late notice keeps linking the record it was
    sent with after a newer publication replaced its panel's ref (#1369), resolved by
    the same membership ``get_task_result(review_source_sha256=)`` verifies
    (``task_finalization.host_acceptance_source``).
    """
    match = _DELEGATED_SOURCE_RE.fullmatch(str(source or ""))
    if match:
        if match.group(1) != name:
            return json_error("source must end in the requested name", 400, reason_code="task_source_invalid",
                              task_id=task_id, artifact=name)
        # New delegated JSONL sources retain the ordinary file route's fail-closed
        # platform boundary (#1297); the result-published review-source branch below is unchanged.
        with _Parents() as parents:
            present = [store for store in stores if _member_stat(parents, _route(
                pathlib.Path(drive_root).resolve(), store, source)) is not None]
        return serve_task_file(drive_root, (present or stores)[0], source, name, {"sha256": match.group(2)},
                               task_id=task_id)
    try:
        return Response(artifact_store.read_task_result_source_bytes(drive_root, result, name, source),
                        media_type="application/json")
    except (OSError, ValueError, RuntimeError):
        pass
    record = _ACCEPTANCE_SOURCE_RE.fullmatch(str(source or ""))
    try:
        if record and record.group(1) == name:
            from ouroboros.task_finalization import host_acceptance_source

            return Response(host_acceptance_source(drive_root, task_id, record.group(2), anchor=drive_root)[1],
                            media_type="application/json")
    except (OSError, ValueError, RuntimeError):
        pass
    return json_error("task source is unavailable or does not match its recorded identity", 404)


class DescriptorResponse(Response):
    """The bytes of ONE already-open binary handle: GET/HEAD, ``Accept-Ranges`` and a single
    ``Range`` (206; an unsatisfiable one 416; several are served whole), headers from the fstat
    of that descriptor. Never opens a path; the handle closes with the response. ``media_type``
    defaults to the name's guess; ``headers`` add a caller's policy (chat uploads)."""

    def __init__(self, handle: Any, name: str, observed: os.stat_result, verified: Optional[str] = None, *,
                 media_type: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> None:
        super().__init__(content=None, media_type=media_type or mimetypes.guess_type(name)[0]
                         or "application/octet-stream", headers=headers)
        self._handle, self._size = handle, observed.st_size
        etag = hashlib.md5(f"{observed.st_mtime}-{observed.st_size}".encode(), usedforsecurity=False).hexdigest()
        self.headers.update({"content-length": str(self._size), "accept-ranges": "bytes", "etag": f'"{etag}"',
                             "last-modified": formatdate(observed.st_mtime, usegmt=True),
                             "x-ouroboros-artifact-identity": "verified" if verified else "unmeasured"})
        if verified:
            self.headers["x-ouroboros-artifact-sha256"] = verified

    def _range(self, header: str) -> Optional[Tuple[int, int]]:
        spec = header.strip().lower()
        if not spec.startswith("bytes=") or "," in spec:
            return None
        first, _, last = spec[6:].strip().partition("-")
        try:
            start, end = ((int(first), int(last) + 1 if last else self._size) if first
                          else (max(0, self._size - int(last)), self._size))
        except ValueError:
            return None
        return (start, min(end, self._size)) if start < min(end, self._size) else (-1, -1)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            start, end, status = 0, self._size, 200
            wanted = self._range(dict(scope.get("headers") or []).get(b"range", b"").decode("latin-1"))
            if wanted == (-1, -1):
                await Response(status_code=416, headers={"content-range": f"bytes */{self._size}"})(scope, receive, send)
                return
            if wanted is not None:
                (start, end), status = wanted, 206
                self.headers["content-range"] = f"bytes {start}-{end - 1}/{self._size}"
                self.headers["content-length"] = str(end - start)
            await send({"type": "http.response.start", "status": status, "headers": self.raw_headers})
            if scope.get("method") == "HEAD":
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
            await anyio.to_thread.run_sync(self._handle.seek, start)
            sent = False
            while start < end:
                chunk = await anyio.to_thread.run_sync(self._handle.read, min(_CHUNK, end - start))
                if not chunk:
                    break  # a short file leaves the declared length unmet, never padded
                start += len(chunk)
                sent = True
                await send({"type": "http.response.body", "body": chunk, "more_body": start < end})
            if start < end or not sent:  # an empty or short body still completes the response
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            self._handle.close()


def serve_task_file(drive_root: Any, store: pathlib.Path, relpath: str, name: str,
                    expected: Optional[Dict[str, Any]], *, task_id: str, mutable: bool = False) -> Response:
    """The ONE file the handler attributed to ``relpath`` of ``store``, read through the confined
    descent: the descriptor response (``x-ouroboros-artifact-identity`` verified/unmeasured), else
    409 ``artifact_identity_changed`` (a ``mutable`` row's bytes no longer match the digest it
    records: the recorded identity rides the refusal, the row wants re-recording), 404
    ``artifact_unverified`` (missing, swapped for a link or FIFO, changed while read, or a capture
    that did not verify) or 503 ``artifact_unavailable`` (including a platform without
    confinement, issue #1297)."""
    try:
        if not CONFINED:
            raise OSError(errno.ENOTSUP, "confined file open unavailable (issue #1297)")
        route = _route(pathlib.Path(drive_root).resolve(strict=False), store, relpath)
        if route is None:
            raise OSError("artifact relpath is not plain segments")
        with _Parents() as parents:
            handle, observed = _open_member(parents, route)
        if expected is not None:
            with handle:  # closed whatever the spool allocation or verification does
                spool = tempfile.TemporaryFile()
                try:
                    measured = artifact_store.stream_artifact_file(handle, spool, expected=expected)
                    if measured["size"] != observed.st_size:
                        raise OSError("task file changed between its open and its verification")
                except BaseException:
                    spool.close()
                    raise
            handle = spool
    except OSError as exc:
        if mutable and exc.errno is None and expected and "verification" in str(exc):
            return json_error("artifact bytes no longer match the identity the task result records for this file",
                              409, reason_code="artifact_identity_changed", task_id=task_id, artifact=name,
                              recorded_sha256=str(expected.get("sha256") or ""), recorded_size=expected.get("size"))
        if exc.errno is None or exc.errno in _MEMBER_GONE:
            return json_error("artifact file is missing, changed while read, or failed its capture verification",
                              404, reason_code="artifact_unverified", task_id=task_id, artifact=name)
        return json_error("artifact could not be read", 503, reason_code="artifact_unavailable",
                          task_id=task_id, artifact=name)
    return DescriptorResponse(handle, name, observed, str((expected or {}).get("sha256") or "") or None,
                              headers=_SVG_POLICY if mimetypes.guess_type(name)[0] == "image/svg+xml" else None)

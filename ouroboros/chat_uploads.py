"""Owner chat attachments: the one upload store, its measured refs and their views.

Every owner attachment — the web paperclip, a skill's inbound file, a transport's
inline photo — is ONE file in ``data/uploads`` named ``<32 hex>_<plain name>``.
That stored name is the upload id. An accepted chat row records, per attachment,
a ref the SERVER measured: ``{"upload", "name", "mime", "kind", "size", "sha256",
"mtime_ns"}`` (``name`` is the stored name without its prefix; ``kind`` is
image/video/audio/file from the bytes, never from the client or the extension alone;
``size`` and ``mtime_ns`` are the stored file's stat witness). No absolute path,
base64 or client URL is ever recorded. A frame naming an upload that does not
exist records ``{"name", "kind": "file", "unavailable": "missing"}`` instead.

``sha256`` is what the bytes were when measured (retry identity, staging
verification); nothing re-proves it on replay or download. Replay compares only
the stat witness: a delete, or a replace or rewrite that changes the size or the
mtime, shows unavailable; a same-size replacement that restores the mtime is not
detected (the witness is not a hash).

The view is a pure function of a ref: the same dict for the sender's own bubble
(the upload response), the echo to other tabs and the history replay. Its ``url``
is ``/api/files/download?upload=<id>`` — an authority independent of the Files
root and of any task's lifetime, served by ``gateway.files``.

Pending guard: an upload this process stored for the web composer is "pending"
until a message accepts it. Only a pending upload may be deleted (the composer's
cleanup after a failed send); acceptance claims it inside the message bus's
ingress critical section, so DELETE and acceptance are serialized and an
accepted original is never removed. After a restart nothing is pending: DELETE
fails closed and an unaccepted copy stays (no second store tracks it).
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import pathlib
import re
import stat
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from ouroboros.confined_files import open_regular_file, plain_name

DOWNLOAD_ROUTE = "/api/files/download?upload="
UPLOAD_ID_RE = re.compile(r"[0-9a-f]{32}_[^\x00-\x1f/\\]{1,200}")
_NAME_BYTES = 222  # 255-byte file-name limit minus the "<32 hex>_" prefix
_HEAD_BYTES = 64
_PENDING_MAX = 512
_PENDING: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_PENDING_LOCK = threading.Lock()
# Host placeholders logged when a message carries attachments but no words; the row that
# carries one says so (``text_placeholder``), so an owner who typed the same words keeps them.
PLACEHOLDER_TEXTS = ("(image attached)", "(file attached)")


def attachment_placeholder(image_base64: Any, metadata: dict | None) -> str:
    """The same canonical empty-caption text at acceptance and batch dispatch."""
    metadata = metadata or {}
    image_ref = any(isinstance(ref, dict) and ref.get("kind") == "image"
                    for ref in metadata.get("chat_attachments") or ())
    return ("(image attached)" if image_base64 or image_ref else
            "(file attached)" if metadata.get("chat_attachment_uploads") or metadata.get("chat_attachments") else "")


def uploads_dir(data_dir: Any = None) -> pathlib.Path:
    """``<data>/uploads``; without ``data_dir`` the data root is resolved per call."""
    if data_dir is None:
        from ouroboros.config import resolve_data_dir

        data_dir = resolve_data_dir()
    return pathlib.Path(data_dir) / "uploads"


def safe_upload_name(raw_name: Any) -> str:
    """The plain name part of a stored upload: basename, spaces as ``_``, Windows-unsafe
    characters as ``_``, no trailing dot/space, at most 200 characters and 222 UTF-8 bytes
    (extension kept)."""
    name = re.split(r"[/\\]", str(raw_name or ""))[-1].rstrip(". ").replace(" ", "_")
    name = "".join("_" if ord(c) < 32 or c in '<>:"|?*' else c for c in name).rstrip(". ") or "upload"
    stem, dot, ext = name.rpartition(".")
    if not dot or len(ext) > 16:
        stem, dot, ext = name, "", ""
    budget = _NAME_BYTES - len((dot + ext).encode("utf-8"))
    stem = stem[:200 - len(dot + ext)].encode("utf-8")[:budget].decode("utf-8", "ignore")
    name = (stem + dot + ext).rstrip(". ") or "upload"
    return name if plain_name(name) else f"upload{dot}{ext}" if plain_name(f"upload{dot}{ext}") else "upload"


def new_upload_id(raw_name: Any) -> str:
    """``<uuid32>_<safe name>`` — the ONE stored-name shape (``artifacts.stage_task_attachments``
    strips the 32-hex prefix to judge the secret-name rule on the original name)."""
    return f"{uuid.uuid4().hex}_{safe_upload_name(raw_name)}"


def upload_id(value: Any) -> str:
    """VALUE when it is a well-formed stored upload id, else ``""``."""
    text = str(value or "")
    return text if UPLOAD_ID_RE.fullmatch(text) and plain_name(text) == text else ""


def _label(value: Any) -> str:
    return " ".join("".join(c for c in str(value or "") if c.isprintable()).split())[:200] or "attachment"


# ---------------------------------------------------------------------------
# Byte detection (conservative: an unknown or ambiguous signature is a file)
# ---------------------------------------------------------------------------

_MP4_BRANDS = frozenset({b"isom", b"iso2", b"iso3", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42",
                         b"avc1", b"dash", b"M4V ", b"mp71", b"MSNV"})
_3GP_BRANDS = frozenset({b"3gp4", b"3gp5", b"3gp6", b"3g2a"})
_HEIF_BRANDS = frozenset({b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"})


def _iso_bmff(head: bytes, ext: str) -> Optional[Tuple[str, str]]:
    size = int.from_bytes(head[0:4], "big")
    major = head[8:12]
    brands = {major} | {head[i:i + 4] for i in range(16, min(max(size, 16), len(head)) - 3, 4)}
    if brands & {b"avif", b"avis"}:
        return "image/avif", "image"
    if brands & _HEIF_BRANDS:
        return ("image/heic" if brands & {b"heic", b"heix"} else "image/heif"), "image"
    if major in {b"M4A ", b"M4B ", b"M4P "} or (ext in {".m4a", ".m4b"} and brands & _MP4_BRANDS):
        return "audio/mp4", "audio"
    if major == b"qt  ":
        return "video/quicktime", "video"
    if brands & _3GP_BRANDS:
        return "video/3gpp", "video"
    if brands & _MP4_BRANDS:
        return "video/mp4", "video"
    return None


def file_media(path: pathlib.Path) -> Tuple[str, str]:
    """Classify a staged file by bytes, as for its chat view, never by suffix alone.

    The caller owns and has verified this staged copy; this is not a download opener.
    HTML named .png remains a file, while JPEG named .mp4 remains an image.
    """
    with path.open("rb") as handle:
        return detect_media(handle.read(64), path.name)


def detect_media(head: bytes, name: str) -> Tuple[str, str]:
    """``(mime, kind)`` from the first bytes; ``kind`` is image/video/audio only for a
    recognized media signature. Anything else — SVG, HTML, PDF, archives, unknown —
    is ``file`` with its extension's MIME as a label (never served inline)."""
    head = bytes(head or b"")[:_HEAD_BYTES]
    ext = pathlib.PurePosixPath(str(name or "")).suffix.lower()
    signatures = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"),
                  (b"GIF87a", "image/gif"), (b"GIF89a", "image/gif"))
    for magic, mime in signatures:
        if head.startswith(magic):
            return mime, "image"
    detected: Optional[Tuple[str, str]] = None
    if head[:4] == b"RIFF" and head[8:12] in {b"WEBP", b"WAVE"}:
        detected = ("image/webp", "image") if head[8:12] == b"WEBP" else ("audio/wav", "audio")
    elif head[:2] == b"BM" and ext == ".bmp" and int.from_bytes(head[14:18], "little") in {12, 40, 52, 56, 108, 124}:
        detected = "image/bmp", "image"
    elif head[4:8] == b"ftyp":
        detected = _iso_bmff(head, ext)
    elif head[:4] == b"\x1aE\xdf\xa3" and b"webm" in head:
        detected = "video/webm", "video"
    elif head[:4] == b"OggS":
        detected = ("video/ogg", "video") if ext == ".ogv" else ("audio/ogg", "audio")
    elif head[:4] == b"fLaC":
        detected = "audio/flac", "audio"
    elif head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0
                               and head[1] & 0x06 and ext in {".mp3", ".mpga"}):
        detected = "audio/mpeg", "audio"
    elif len(head) > 1 and head[0] == 0xFF and head[1] & 0xF6 == 0xF0 and ext == ".aac":
        detected = "audio/aac", "audio"
    if detected:
        return detected
    label = (mimetypes.guess_type("x" + ext)[0] if ext else None) or "application/octet-stream"
    # Bytes that did not prove a media type never keep a media label (an HTML ``.png``).
    media_label = label.split("/", 1)[0] in {"image", "video", "audio"} and label != "image/svg+xml"
    return ("application/octet-stream" if media_label else label), "file"


# ---------------------------------------------------------------------------
# Store and measure
# ---------------------------------------------------------------------------

def _ref(upload: str, measured: Dict[str, Any], head: bytes, observed: os.stat_result) -> Dict[str, Any]:
    name = upload[33:]
    mime, kind = detect_media(head, name)
    if observed.st_size != measured["size"]:
        raise OSError(f"upload changed while it was measured: {upload}")
    return {"upload": upload, "name": name, "mime": mime, "kind": kind, "size": int(measured["size"]),
            "sha256": str(measured["sha256"]), "mtime_ns": int(observed.st_mtime_ns)}


def store_upload(source: Any, display_name: Any, *, data_dir: Any = None,
                 pending: bool = False) -> Tuple[pathlib.Path, Dict[str, Any]]:
    """Copy a confined path, a completed borrowed spool, or ``bytes`` into the store.

    Returns ``(stored path, measured ref)``; ``pending`` registers a web composer
    upload as deletable until a message claims it.
    """
    from ouroboros.artifacts import copy_artifact_file

    upload = new_upload_id(display_name)
    dest = uploads_dir(data_dir) / upload
    if isinstance(source, (bytes, bytearray)):
        from ouroboros.utils import write_bytes_atomic

        dest.parent.mkdir(parents=True, exist_ok=True)
        write_bytes_atomic(dest, bytes(source))
        measured = {"size": len(source), "sha256": hashlib.sha256(source).hexdigest()}
    else:
        measured = copy_artifact_file(source, dest)
    try:  # the stat witness of the file just published
        with dest.open("rb") as handle:
            ref = _ref(upload, measured, handle.read(_HEAD_BYTES), os.fstat(handle.fileno()))
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if pending:
        with _PENDING_LOCK:
            _PENDING[upload] = dict(ref)
            while len(_PENDING) > _PENDING_MAX:
                _PENDING.popitem(last=False)  # the evicted copy is simply no longer deletable
    return dest, ref


def open_upload(upload: str, data_dir: Any = None):
    """``(handle, fstat)`` of one stored upload through the confined open."""
    if not upload_id(upload):
        raise OSError(2, "not a stored upload id")
    return open_regular_file(uploads_dir(data_dir).resolve(strict=True), upload)


def measure_upload(upload: str, data_dir: Any = None) -> Dict[str, Any]:
    """Hash an existing upload into its ref (server-side, confined, never followed)."""
    from ouroboros.artifacts import stream_artifact_file

    handle, observed = open_upload(upload, data_dir)
    with handle:
        head = handle.read(_HEAD_BYTES)
        measured = stream_artifact_file(handle)  # proves the descriptor unchanged through EOF
        if (os.fstat(handle.fileno()).st_mtime_ns, measured["size"]) != (observed.st_mtime_ns, observed.st_size):
            raise OSError(f"upload changed while it was measured: {upload}")
    return _ref(upload, measured, head, observed)


def unavailable_ref(label: Any) -> Dict[str, Any]:
    return {"name": _label(label), "kind": "file", "unavailable": "missing"}


def refs_for_frame(attachments: Any, data_dir: Any = None) -> List[Dict[str, Any]]:
    """Measured refs for a web chat frame's ``attachments``: every one, in order.

    A pending upload reuses the facts measured when it was stored; any other
    named upload is measured from disk; a malformed or absent one is unavailable.
    """
    refs: List[Dict[str, Any]] = []
    for item in attachments if isinstance(attachments, list) else []:
        item = item if isinstance(item, dict) else {}
        upload = upload_id(item.get("filename"))
        label = item.get("display_name") or item.get("filename")
        with _PENDING_LOCK:
            ref = dict(_PENDING[upload]) if upload in _PENDING else None
        if ref is None and upload:
            try:
                ref = measure_upload(upload, data_dir)
            except OSError:
                ref = None
        refs.append(ref or unavailable_ref(label))
    return refs


def claim_refs(refs: Any, data_dir: Any = None) -> List[Dict[str, Any]]:
    """Accept REFS: call inside the ingress critical section, before the row write.

    A claimed pending upload can no longer be deleted; an upload that vanished
    since it was measured (deleted while pending) is recorded unavailable.
    """
    claimed: List[Dict[str, Any]] = []
    for ref in refs if isinstance(refs, list) else []:
        ref = dict(ref) if isinstance(ref, dict) else unavailable_ref("")
        upload = upload_id(ref.get("upload"))
        with _PENDING_LOCK:
            was_pending = _PENDING.pop(upload, None) is not None if upload else False
        if upload and not was_pending and not os.path.lexists(uploads_dir(data_dir) / upload):
            ref = unavailable_ref(ref.get("name"))
        claimed.append(ref)
    return claimed


def delete_pending_upload(upload: str, data_dir: Any = None) -> str:
    """Remove a still-pending upload: ``"deleted"``, ``"missing"`` or ``"accepted"`` (refused:
    claimed, from another source, or stored before this process started)."""
    upload = upload_id(upload)
    with _PENDING_LOCK:
        if not upload or upload not in _PENDING:
            return "accepted" if upload and os.path.lexists(uploads_dir(data_dir) / upload) else "missing"
        del _PENDING[upload]
        try:
            (uploads_dir(data_dir) / upload).unlink()
        except FileNotFoundError:
            return "missing"
    return "deleted"


# ---------------------------------------------------------------------------
# Identity and presentation
# ---------------------------------------------------------------------------

def attachment_identity(refs: Any) -> List[List[str]]:
    """What makes two deliveries the SAME message's attachments: ordered content and
    names, never the regenerated stored id."""
    return [[str(ref.get("sha256") or ""), str(ref.get("name") or "")]
            for ref in (refs if isinstance(refs, list) else []) if isinstance(ref, dict)]


def source_identity(path: Any, name: Any) -> List[str]:
    """The identity entry a not-yet-stored source file will have once stored."""
    from ouroboros.artifacts import stream_artifact_file

    return [stream_artifact_file(pathlib.Path(path))["sha256"], safe_upload_name(name)]


def bytes_identity(data: bytes, name: Any) -> List[str]:
    return [hashlib.sha256(data).hexdigest(), safe_upload_name(name)]


_INLINE_EXTENSIONS = {"image/jpeg": "jpg", "video/quicktime": "mov", "video/3gpp": "3gp",
                      "audio/mpeg": "mp3", "audio/mp4": "m4a"}


def inline_image_name(data: bytes) -> str:
    """The stored name of a transport's inline bytes, from the bytes alone (never the
    transport's label): ``photo.<ext>`` for a proven image, ``attachment.<ext>`` for proven
    video/audio, else ``attachment.bin``."""
    mime, kind = detect_media(data[:_HEAD_BYTES], "")
    ext = _INLINE_EXTENSIONS.get(mime) or (mime.split("/", 1)[1] if kind != "file" else "bin")
    return safe_upload_name(f"{'photo' if kind == 'image' else 'attachment'}.{ext[:10]}")


def same_message(row: Any, text: str, refs: Any) -> bool:
    """A retried delivery is the accepted one: same normalized text AND attachments."""
    from ouroboros.project_dialogue import _text_sha256

    row = row if isinstance(row, dict) else {}
    return (_text_sha256(row.get("text")) == _text_sha256(text)
            and attachment_identity(row.get("attachments")) == attachment_identity(refs))


def attachment_view(ref: Any) -> Dict[str, Any]:
    """The one presentation of a stored ref (own bubble, echo and history alike)."""
    ref = ref if isinstance(ref, dict) else {}
    upload = upload_id(ref.get("upload"))
    name = _label(ref.get("name"))
    if not upload or ref.get("unavailable"):
        return {"name": name, "kind": "file", "mime": "", "available": False}
    kind = ref.get("kind") if ref.get("kind") in {"image", "video", "audio", "file"} else "file"
    size = ref.get("size")
    return {"name": name, "kind": kind, "mime": str(ref.get("mime") or "application/octet-stream"),
            "size": size if type(size) is int else None, "available": True,
            "url": DOWNLOAD_ROUTE + quote(upload, safe="")}


def _still_stored(ref: Dict[str, Any], data_dir: Any) -> bool:
    """The upload is still the regular file its ref witnessed: the same size AND mtime (one
    ``lstat``, never a hash). A ref without its witness proves nothing and is not shown."""
    try:
        observed = os.lstat(uploads_dir(data_dir) / str(ref.get("upload")))
    except OSError:
        return False
    size, mtime_ns = ref.get("size"), ref.get("mtime_ns")
    return (stat.S_ISREG(observed.st_mode) and type(size) is int and type(mtime_ns) is int
            and (observed.st_size, observed.st_mtime_ns) == (size, mtime_ns))


def attachment_views(refs: Any, data_dir: Any = None) -> List[Dict[str, Any]]:
    """Every ref's view. With ``data_dir`` (history replay), an upload deleted since, or
    replaced or rewritten so its size or mtime changed (the Files API reaches ``uploads/`` too),
    fails its stat witness and shows unavailable, not stale actions; the recorded sha256 is not
    re-proven here, so a same-size replacement that restores the mtime still shows."""
    views = []
    for ref in refs if isinstance(refs, list) else []:
        view = attachment_view(ref)
        if data_dir is not None and view["available"] and not _still_stored(ref, data_dir):
            view = attachment_view({**ref, "unavailable": "missing"})
        views.append(view)
    return views


def stored_refs(value: Any) -> List[Dict[str, Any]]:
    """The refs a canonical writer persists: every one, as plain JSON dicts, nothing else."""
    keys = ("upload", "name", "mime", "kind", "size", "sha256", "mtime_ns", "unavailable")
    return [{key: ref[key] for key in keys if key in ref}
            for ref in (value if isinstance(value, list) else []) if isinstance(ref, dict)]


__all__ = [
    "DOWNLOAD_ROUTE", "PLACEHOLDER_TEXTS", "attachment_identity", "attachment_view",
    "attachment_views", "bytes_identity", "claim_refs", "delete_pending_upload", "detect_media",
    "inline_image_name", "measure_upload", "new_upload_id", "open_upload", "refs_for_frame",
    "safe_upload_name", "same_message", "source_identity", "store_upload", "stored_refs",
    "unavailable_ref", "upload_id", "uploads_dir",
]

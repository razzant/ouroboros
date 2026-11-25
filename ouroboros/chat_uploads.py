"""Owner chat attachments: the one upload store, its measured refs and their views.

Every owner attachment — the web paperclip, a skill's inbound file, a transport's
inline photo — is ONE file in ``data/uploads`` named ``<32 hex>_<plain name>``.
That stored name is the upload id. An accepted chat row records, per attachment,
a ref the SERVER measured: ``{"upload", "name", "mime", "kind", "size", "sha256",
"mtime_ns"}`` (``name`` is the stored name without its prefix; ``kind`` is
image/video/audio/file from the bytes, never from the client or the extension alone;
``size`` and ``mtime_ns`` are the stored file's stat witness). No absolute path,
base64 or client URL is ever recorded. A frame naming an upload that does not
exist records ``{"name", "kind": "file", "unavailable": "missing"}`` instead; media a
transport could not receive records its own pathless fact (``delivery_ref``).

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

A task that copies a held row's files reads them here too: staging specs under
the root the row was read from (``upload_specs``), and rows verified against their
saved identity for a successor's own copy (``quoted_inputs``).
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import pathlib
import re
import stat
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Tuple
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
PLACEHOLDER_TEXTS = ("(image attached)", "(file attached)", "(attachment unavailable)")
# Media a transport never received (over its download cap, or a failed download): no bytes, no upload.
DELIVERY_FAILURES = ("too_large", "download_failed")
_MIME_RE = re.compile(r"[\w.+-]{1,100}/[\w.+-]{1,100}", re.ASCII)


def attachment_placeholder(image_base64: Any, metadata: Any) -> str:
    """The same canonical empty-caption text at acceptance and batch dispatch."""
    metadata = metadata if isinstance(metadata, dict) else {}
    refs = [ref for ref in metadata.get("chat_attachments") or () if isinstance(ref, dict)]
    if refs and not image_base64 and all(ref.get("unavailable") in DELIVERY_FAILURES for ref in refs):
        return "(attachment unavailable)"
    image_ref = any(ref.get("kind") == "image" for ref in refs)
    return ("(image attached)" if image_base64 or image_ref else
            "(file attached)" if metadata.get("chat_attachment_uploads") or metadata.get("chat_attachments") else "")


def local_message_words(source: str, text: str, caption: str) -> str:
    """Named Telegram media keep their exact caption; other local doors keep their text convention."""
    if source == "skill:telegram" and caption.strip() and (not text.strip() or text == caption):
        return caption
    return text.strip() or caption.strip()


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
# Labels the platform table may lack (Python 3.10 knows no .md, and its built-in table no .docx,
# which only some system tables add): the same file gets one label on every interpreter.
# Only new measurements use it; recorded refs and manifests keep theirs.
_DOCUMENT_LABELS = {".md": "text/markdown", ".markdown": "text/markdown",
                    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


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
    label = _DOCUMENT_LABELS.get(ext) or (mimetypes.guess_type("x" + ext)[0] if ext else None) or "application/octet-stream"
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


def delivery_ref(item: Dict[str, Any]) -> Dict[str, Any]:
    """The ref a row records for a transport's pathless fact ``{name, mime?, size?, unavailable, limit_bytes?}``:
    media it never received, so nothing is read, stored or downloadable. ``limit_bytes`` (the transport's cap)
    comes with ``too_large`` only. Anything else is refused (``ValueError``)."""
    reason, size, limit = item.get("unavailable"), item.get("size"), item.get("limit_bytes")
    if (reason not in DELIVERY_FAILURES or set(item) - {"name", "mime", "size", "unavailable", "limit_bytes"}
            or "size" in item and (type(size) is not int or size < 0)
            or ("limit_bytes" in item) != (reason == "too_large")
            or "limit_bytes" in item and (type(limit) is not int or limit < 1)):
        raise ValueError("is not a valid unavailable-media fact")
    mime = item.get("mime")
    return {"name": _label(re.split(r"[/\\]", str(item.get("name") or ""))[-1]), "kind": "file",
            "mime": mime if isinstance(mime, str) and _MIME_RE.fullmatch(mime) else "application/octet-stream",
            "unavailable": reason, **({"size": size} if "size" in item else {}),
            **({"limit_bytes": limit} if "limit_bytes" in item else {})}


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

def delivery_failures(refs: Any) -> list[dict]:
    """Only observed never-received files, not vanished uploads or user-supplied URLs."""
    return [{**ref, "ordinal": ordinal} for ordinal, ref in enumerate(refs if isinstance(refs, list) else []) if isinstance(ref, dict)
            and ref.get("unavailable") in DELIVERY_FAILURES and not ref.get("upload")]


def delivery_manifest_rows(attachments: Any) -> list[dict]:
    """Reports of bytes the transport never received, not staging failures."""
    return [row for row in (attachments if isinstance(attachments, list) else []) if isinstance(row, dict)
            and row.get("status") == "rejected" and row.get("reason") in DELIVERY_FAILURES]


def delivery_manifest_refs(manifest: Any) -> list[dict]:
    """The mailbox's retained facts as corpus attachment refs, independent of any other stored row."""
    return [{"name": row.get("label") or "attachment", "kind": "file", "unavailable": row["reason"],
             **{key: row[key] for key in ("mime", "size", "limit_bytes") if key in row}}
            for row in delivery_manifest_rows(manifest)]


def delivery_failure_line(name: str, reason: str, ordinal: int, facts: dict) -> str:
    said = "".join(f", {key}={facts[key]}" for key in ("mime", "size", "limit_bytes") if facts.get(key) is not None)
    return f"- {name}: not received (reason={reason}{said}, ordinal={ordinal})"


def _delivery_block(lines) -> str:
    body = "\n".join(lines)
    return ("[Host fact: attachment delivery; not the owner's words]\n"
            f"[ATTACHMENTS]\n{body}\n[END_ATTACHMENTS]") if body else ""


def attachment_delivery_block(manifest: Any) -> str:
    """Attributed model input on a new delivery, independent of the caption's words."""
    return _delivery_block(delivery_failure_line(row.get("label") or "attachment", row["reason"], row.get("ordinal", 0), row)
                           for row in delivery_manifest_rows(manifest))


def attachment_ref_delivery_block(refs: Any) -> str:
    """Same facts from a verified chat row, including each declaration's original position."""
    return _delivery_block(delivery_failure_line(ref.get("name") or "attachment", ref["unavailable"], ref.get("ordinal", ordinal), ref)
                           for ordinal, ref in enumerate(refs if isinstance(refs, list) else [])
                           if isinstance(ref, dict) and ref.get("unavailable") in DELIVERY_FAILURES and not ref.get("upload"))


def attachment_reply_context(target: Any) -> str:
    """An ordinary Reply's exact resolved source, only when it records a transport refusal."""
    facts = attachment_ref_delivery_block(target.get("attachments"))
    if not facts or not target.get("address"):
        return ""
    return "\n".join(part for part in (
        f"[Reply source from {target.get('author') or 'author not recorded'} at {target['address']}]",
        target.get("text"), facts) if part)


def retained_attachments_note(files: Any) -> str:
    """Files named by the host under the row's words: name, type, size, staged path or delivered download URL,
    or the fact that the file was not given to this work. ``""`` for a row with none."""
    named = []
    for ref in files if isinstance(files, list) else []:
        if isinstance(ref, Mapping) and ref.get("unavailable") not in DELIVERY_FAILURES:
            facts = ([str(ref["mime"])] * bool(ref.get("mime")) + [f"{ref['size']} bytes"] * (ref.get("size") is not None)
                     + [str(ref[key]) for key in ("staged", "download_url") if ref.get(key)])
            if ref.get("unavailable"):
                facts.append("unavailable")
            named.append(str(ref.get("name") or "attachment") + (f" ({', '.join(facts)})" if facts else ""))
    return "\n".join(part for part in (
        f"[Host-retained attachments: {'; '.join(named)}]" if named else "", attachment_ref_delivery_block(files)) if part)


def attachment_identity(refs: Any) -> list:
    """What makes two deliveries the SAME message's attachments: ordered content and
    names, never the regenerated stored id."""
    return [[ref.get(key) for key in ("unavailable", "name", "mime", "size", "limit_bytes")]
            if ref.get("unavailable") in DELIVERY_FAILURES else [str(ref.get("sha256") or ""), str(ref.get("name") or "")]
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
    same_words = (row.get("text") == text if row.get("source") == "skill:telegram"
                  else _text_sha256(row.get("text")) == _text_sha256(text))
    return (same_words
            and attachment_identity(row.get("attachments")) == attachment_identity(refs))


def attachment_view(ref: Any) -> Dict[str, Any]:
    """The one presentation of a stored ref (own bubble, echo and history alike)."""
    ref = ref if isinstance(ref, dict) else {}
    upload = upload_id(ref.get("upload"))
    name = _label(ref.get("name"))
    if not upload or ref.get("unavailable"):
        facts = ({key: ref[key] for key in ("unavailable", "size", "limit_bytes") if key in ref}
                 if ref.get("unavailable") in DELIVERY_FAILURES and not upload else {})
        return {"name": name, "kind": "file", "mime": "", "available": False, **facts}
    kind = ref.get("kind") if ref.get("kind") in {"image", "video", "audio", "file"} else "file"
    size = ref.get("size")
    return {"name": name, "kind": kind, "mime": str(ref.get("mime") or "application/octet-stream"),
            "size": size if type(size) is int else None, "available": True,
            "url": DOWNLOAD_ROUTE + quote(upload, safe="")}


def web_attachment_caption(row: dict) -> str:
    """Words before the exact web composer tail (``attachmentCaption``); no file access."""
    text = str(row.get("text") or "")
    refs = row.get("attachments")
    if row.get("source") != "web" or not isinstance(refs, list) or not refs:
        return text
    lines = [f"[Attached file: {attachment_view(ref)['name']}]" for ref in refs[:25]]
    if len(refs) > 25:
        lines.append(f"[{len(refs) - 25} more attached files]")
    tail = "\n".join(lines)
    # An author who typed the same line still has the generated tail after it. Remove only one.
    return "" if text == tail else text.removesuffix("\n\n" + tail)


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
    keys = ("upload", "name", "mime", "kind", "size", "sha256", "mtime_ns", "unavailable", "limit_bytes")
    return [{key: ref[key] for key in keys if key in ref}
            for ref in (value if isinstance(value, list) else []) if isinstance(ref, dict)]


def answer_files(refs: Any) -> List[Dict[str, Any]]:
    """A recorded answer's files as its card and receipt state them: name, kind, type, size, digest, or why one was
    never received. No stored id or stat witness: these facts open no download."""
    return [{key: ref[key] for key in ("name", "kind", "mime", "size", "sha256", "unavailable", "limit_bytes")
             if key in ref} for ref in stored_refs(refs)]


def _project_attachments(mapping: Dict[str, Any], key: str = "attachments") -> Dict[str, Any]:
    """Shallow-copy ``mapping`` with ``key`` projected via ``answer_files`` (dropped when empty)."""
    out = dict(mapping)
    if key in out:
        files = answer_files(out[key])
        if files:
            out[key] = files
        else:
            out.pop(key, None)
    return out


def _is_secret_quiz(block: Any) -> bool:
    """Whether ``block`` carries protected secret-form metadata (never altered by display projection)."""
    if not isinstance(block, dict):
        return False
    from ouroboros.owner_quiz import SECRET_FIELDS, secret_outcome_block

    return any(block.get(key) for key in SECRET_FIELDS) or secret_outcome_block(block) is not None


def public_quiz_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a stored quiz/answer block with top-level and ``delivery.answer_row`` attachments projected via
    ``answer_files`` (shallow-copying only touched branches, never mutating ``block``)."""
    out = _project_attachments(block)
    delivery = out.get("delivery")
    answer_row = delivery.get("answer_row") if isinstance(delivery, dict) else None
    if isinstance(answer_row, dict) and ("attachments" in answer_row or (
            isinstance(answer_row.get("quiz"), dict) and "attachments" in answer_row["quiz"])):
        row_copy = _project_attachments(answer_row)
        if isinstance(row_copy.get("quiz"), dict) and "attachments" in row_copy["quiz"]:
            row_copy["quiz"] = _project_attachments(row_copy["quiz"])
        out["delivery"] = {**delivery, "answer_row": row_copy}
    return out


def _public_late_answer(late: Any) -> Any:
    """Copy a ``late_answer`` carrier with ``answer_snapshot.quiz`` projected via ``public_quiz_block``."""
    snapshot = late.get("answer_snapshot") if isinstance(late, dict) else None
    if not (isinstance(snapshot, dict) and isinstance(snapshot.get("quiz"), dict)) or _is_secret_quiz(snapshot["quiz"]):
        return late
    return {**late, "answer_snapshot": {**snapshot, "quiz": public_quiz_block(snapshot["quiz"])}}


def _late_quiz_file_snapshot(late: Any) -> bool:
    """Pure typed-shape check for an accepted late-answer snapshot of an ordinary (non-secret)
    quiz carrying recorded file refs — the exact shape ``gateway/task_decision._forward_late_quiz_answer``
    and ``supervisor/message_ingress._chat_record`` pair with ``metadata.chat_attachments`` /
    ``metadata.chat_attachment_uploads`` and a held late-answer row's ``attachments``. Never reads a store."""
    if not isinstance(late, dict):
        return False
    snapshot = late.get("answer_snapshot")
    quiz = snapshot.get("quiz") if isinstance(snapshot, dict) else None
    if not isinstance(quiz, dict):
        return False
    if not (quiz.get("answered_after_terminal") is True or quiz.get("delivery_after_terminal") is True
            or isinstance(quiz.get("delivery"), dict)):
        return False
    from ouroboros.owner_quiz import late_answer_ref

    ref = late_answer_ref(late)
    block = (ref.get("answer_snapshot") or {}).get("quiz") if isinstance(ref, dict) else None
    if not isinstance(block, dict) or _is_secret_quiz(block):
        return False
    return any(stored_refs(block.get("attachments")))


def _public_late_carrier(
    mapping: Dict[str, Any], *, attachments_key: str, drop_uploads: bool = False,
) -> Dict[str, Any]:
    """Project ``late_answer.answer_snapshot.quiz`` and, when ``_late_quiz_file_snapshot`` holds,
    the carrier's companion file refs (and optional transport-only uploads) without mutating ``mapping``."""
    late = mapping.get("late_answer")
    projected_late = _public_late_answer(late)
    late_files = _late_quiz_file_snapshot(late)
    if projected_late is late and not late_files:
        return mapping
    out = _project_attachments(mapping, attachments_key) if late_files else dict(mapping)
    if projected_late is not late:
        out["late_answer"] = projected_late
    if late_files and drop_uploads:
        out.pop("chat_attachment_uploads", None)
    return out


def public_owner_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """A held owner row's display copy: its ``quiz`` block and/or accepted ``late_answer`` snapshot as their cards
    state them (``public_quiz_block`` / ``_public_late_carrier``), plus its own ``attachments`` when the row carries
    a ``quiz`` block or a typed late ordinary quiz file snapshot, shallow-copying only the touched branches. A row
    carrying neither stays as it is."""
    if not isinstance(row, dict):
        return row
    out = _public_late_carrier(row, attachments_key="attachments")
    if not isinstance(row.get("quiz"), dict):
        return out
    return {**public_quiz_block(out), "quiz": public_quiz_block(row["quiz"])}


def _public_held_rows(held: Dict[str, Any]) -> Dict[str, Any]:
    return {address: public_owner_row(row) if isinstance(row, dict) else row
            for address, row in held.items()}


def _public_mailbox_row(line: Any) -> Any:
    """Project ``quiz_delivery.answer_snapshot`` on one serialized ordinary ``quiz_answer`` mailbox line,
    keeping its string wire type and leaving every other line byte-for-byte untouched."""
    if not isinstance(line, str):
        return line
    try:
        entry = json.loads(line)
    except (ValueError, TypeError):
        return line
    if not isinstance(entry, dict) or entry.get("kind") != "quiz_answer":
        return line
    delivery = entry.get("quiz_delivery")
    snapshot = delivery.get("answer_snapshot") if isinstance(delivery, dict) else None
    if not isinstance(snapshot, dict) or _is_secret_quiz(snapshot) or _is_secret_quiz(entry):
        return line
    projected = public_quiz_block(snapshot)
    if projected == snapshot:
        return line
    from ouroboros.utils import machine_json

    return machine_json({**entry, "quiz_delivery": {**delivery, "answer_snapshot": projected}})


def _public_mailbox_capture(capture: Any) -> Any:
    """Project serialized ``quiz_answer`` lines inside a mailbox capture dict without mutating ``capture``."""
    rows = capture.get("rows") if isinstance(capture, dict) else None
    if not isinstance(rows, list):
        return capture
    projected = [_public_mailbox_row(line) for line in rows]
    if all(new is old for new, old in zip(projected, rows)):
        return capture
    return {**capture, "rows": projected}


def public_quiz_copies(payload: Dict[str, Any]) -> None:
    """Project all named public task-result and task-event carriers of quiz answer files in place on ``payload``:
    a task-scoped ``quiz_answer`` event row's top-level ``quiz`` (and ``attachments`` if present),
    ``owner_quiz``, top-level ``held_owner_messages`` (both ``quiz`` rows and held inbound ``late_answer`` rows),
    serialized ``quiz_answer`` rows inside ``owner_mailbox.rows`` / ``unread_mailbox.rows``,
    ``metadata.late_answer.answer_snapshot.quiz`` (plus sibling ``metadata.chat_attachments`` and transport-only
    ``metadata.chat_attachment_uploads`` when ``metadata.late_answer`` establishes a typed late ordinary quiz file
    snapshot) and ``metadata.continuation.owner_sources.held_owner_messages``. Only these named carriers are touched —
    never a walk of the whole document, and non-quiz rows stay verbatim — shallow-copying touched containers so the
    stored result, the canonical chat row, exact owner reads and every other carrier keep their own bytes."""
    if not isinstance(payload, dict):
        return
    quiz = payload.get("quiz")
    if (payload.get("type") == "quiz_answer" and isinstance(quiz, dict)
            and not _is_secret_quiz(quiz) and not _is_secret_quiz(payload)):
        payload["quiz"] = public_quiz_block(quiz)
        if "attachments" in payload:
            files = answer_files(payload["attachments"])
            if files:
                payload["attachments"] = files
            else:
                payload.pop("attachments", None)
    owner_quiz = payload.get("owner_quiz")
    if isinstance(owner_quiz, dict):
        payload["owner_quiz"] = {
            qid: public_quiz_block(block) if isinstance(block, dict) else block
            for qid, block in owner_quiz.items()
        }
    held = payload.get("held_owner_messages")
    if isinstance(held, dict):
        payload["held_owner_messages"] = _public_held_rows(held)
    for key in ("owner_mailbox", "unread_mailbox"):
        capture = payload.get(key)
        projected_capture = _public_mailbox_capture(capture)
        if projected_capture is not capture:
            payload[key] = projected_capture
    meta = payload.get("metadata")
    if not isinstance(meta, dict):
        return
    meta_out = _public_late_carrier(meta, attachments_key="chat_attachments", drop_uploads=True)
    continuation = meta_out.get("continuation")
    sources = continuation.get("owner_sources") if isinstance(continuation, dict) else None
    inherited = sources.get("held_owner_messages") if isinstance(sources, dict) else None
    if isinstance(inherited, dict):
        meta_out = {
            **meta_out,
            "continuation": {
                **continuation,
                "owner_sources": {**sources, "held_owner_messages": _public_held_rows(inherited)},
            },
        }
    if meta_out is not meta:
        payload["metadata"] = meta_out


def staging_specs(refs: Any, data_dir: Any = None) -> List[Dict[str, Any]]:
    """Recorded refs as ``artifacts.stage_task_attachments`` specs, built when delivered: a stored upload by its path
    and measured size and digest, verified while copied (a file changed or gone since is refused, never replaced); a
    file never received or gone at acceptance, by its reason and facts, with no path."""
    specs = []
    for ref in stored_refs(refs):
        upload = "" if ref.get("unavailable") else upload_id(ref.get("upload"))
        specs.append({"label": str(ref.get("name") or "attachment"),
                      **{key: ref[key] for key in ("mime", "size", "sha256", "unavailable", "limit_bytes") if key in ref},
                      **({"path": str(uploads_dir(data_dir) / upload)} if upload else {})})
    return specs


def answer_evidence(refs: Any) -> List[Dict[str, Any]]:
    """The recorded refs that are a file's evidence: a stored upload id, or the typed reason none was held. A ref
    with neither (an empty or display-only entry) evidences nothing and answers nothing."""
    return [ref for ref in stored_refs(refs) if upload_id(ref.get("upload"))
            or ref.get("unavailable") in (*DELIVERY_FAILURES, "missing", "outside_store")]


def row_attachments(row: Any, root: Any = None) -> List[Dict[str, Any]]:
    """The stored refs of the files an owner row carries, by its own record: an inbound row's ``attachments``; a card's
    answer row (``type == quiz_answer``), the files its recorded winner holds (``quiz.attachments``); a late answer's
    row recorded without them, its accepted snapshot's. Display facts (``answer_files``) never come here."""
    if not isinstance(row, dict):
        return []
    if row.get("attachments") or (row.get("type") != "quiz_answer" and not row.get("late_answer")):
        return stored_refs(row.get("attachments"))
    if row.get("type") == "quiz_answer":  # the host's own answer row (``owner_relay.quotable_owner_row``), never a secret outcome
        from ouroboros.owner_quiz import secret_outcome_block
        quiz = row.get("quiz")
        return (stored_refs(quiz.get("attachments")) if row.get("source") == "owner_quiz_answer" and isinstance(quiz, dict)
                and quiz.get("state") == "answered" and secret_outcome_block(row) is None else [])
    from ouroboros.owner_quiz import late_answer_ref, resolved_late_quiz
    late = late_answer_ref(row.get("late_answer"))
    return stored_refs(resolved_late_quiz(root, late).get("attachments")) if late else []


def upload_specs(row: Any, root: Any) -> List[Dict[str, Any]]:
    """A held row's files as staging specs for ``artifacts.stage_task_attachments``: each saved upload under the root
    the row was read from, with its saved identity (a file with no saved upload keeps its name and stages nothing).
    Retains the full stored ref fields and an explicit ``path`` (empty when unavailable), distinct from ``staging_specs``."""
    specs = []
    for ref in row_attachments(row, root):
        upload = upload_id(ref.get("upload"))
        path = str(uploads_dir(root) / upload) if upload and not ref.get("unavailable") else ""
        specs.append({**ref, "path": path, "label": str(ref.get("name") or "attachment")})
    return specs


def quoted_inputs(root, refs: list) -> list:
    """The owner files a relay's quoted rows carried (their saved chat uploads, which ``owner_relay.quote_content``
    stages on delivery), as rows another task's own copy reads (``artifacts.materialize_inherited_attachment_manifest``):
    each upload verified against its saved identity and named by its unique upload id, so two same-named files stay
    two. One gone or changed is a rejected ``source_missing`` row under its name, never other bytes; a secret-named one
    is refused as staging refuses it (``secret_source``)."""
    from ouroboros.artifacts import stream_artifact_file
    from ouroboros.workspace_patch_rules import _sensitive_untracked_reason

    rows = []
    for ordinal, ref in enumerate(refs):
        uid = upload_id(ref.get("upload"))
        source_root = ref.get("_owner_source_root") or root
        path, original = uploads_dir(source_root) / uid, re.sub(r"^[0-9a-f]{32}_", "", uid)
        row = {"ordinal": ordinal, "label": str(ref.get("name") or "attachment"),
               **{key: ref[key] for key in ("mime", "size", "sha256") if key in ref}}
        if uid and (reason := _sensitive_untracked_reason(original)):
            rows.append({**row, "status": "rejected", "reason": "secret_source",
                         "rule": f"uploaded file name {original!r}: {reason}"})
            continue
        try:
            if not uid:
                raise OSError("the quoted row names no saved upload")
            stream_artifact_file(path, expected=ref)
            mime, kind = file_media(path)
        except OSError:
            rows.append({**row, "status": "rejected", "reason": "source_missing"})
            continue
        rows.append({**row, "status": "staged", "root": "artifact_store", "relpath": f"attachments/{uid}",
                     "abs_path": str(path), "mime": mime, "is_image": kind == "image"})
    return rows


__all__ = [
    "DELIVERY_FAILURES", "DOWNLOAD_ROUTE", "PLACEHOLDER_TEXTS", "answer_files", "attachment_identity",
    "attachment_view", "attachment_views", "bytes_identity", "claim_refs", "delete_pending_upload", "delivery_ref",
    "detect_media", "inline_image_name", "measure_upload", "new_upload_id", "open_upload", "public_owner_row",
    "public_quiz_copies", "quoted_inputs", "refs_for_frame", "row_attachments", "safe_upload_name", "same_message", "source_identity",
    "staging_specs", "store_upload", "stored_refs", "unavailable_ref", "upload_id", "upload_specs", "uploads_dir",
]

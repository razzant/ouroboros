"""Small read model for canonical Project dialogue and routing annotations.

Project conversion stores a reference to the original owner row on the immutable
task binding. A Project room projects that row instead of copying it into
``chat.jsonl``. ``chat_annotations.jsonl`` is presentation-only: it never owns a
routing decision or Project state.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import uuid
from typing import Any, Dict, Iterable, List

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.utils import iter_jsonl_objects, jsonl_append_lock_path, utc_now_iso

_ANNOTATIONS_NAME = "chat_annotations.jsonl"
_COMPACT_AT_BYTES = 800_000
_RETAINED_ARCHIVES = 3


def _chat_paths(drive_root: Any) -> List[pathlib.Path]:
    root = pathlib.Path(drive_root)
    archives = sorted(
        (root / "archive").glob("chat_*.jsonl"),
        key=lambda path: path.name,
        reverse=True,
    )[:_RETAINED_ARCHIVES]
    return [*reversed(archives), root / "logs" / "chat.jsonl"]


def _text_sha256(value: Any) -> str:
    normalized = " ".join(str(value or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_owner_message_ref(
    *,
    chat_id: int,
    client_message_id: str,
    ts: str,
    text: str,
) -> Dict[str, Any]:
    """The canonical owner-row identity, built AT INGRESS from host-known facts.

    Identity is captured where the host writes the canonical row and passed by
    value downstream (BIBLE P2/P5); it is never re-derived from content later.
    ``text_sha256`` rides along as an integrity checksum, not a lookup key."""
    return {
        "chat_id": int(chat_id or 0),
        "client_message_id": str(client_message_id or ""),
        "ts": str(ts or ""),
        "text_sha256": _text_sha256(text),
    }


def source_refs_for_project(drive_root: Any, project_chat_id: int) -> List[Dict[str, Any]]:
    """Canonical owner-row references held by bindings for one Project lens."""
    from ouroboros.projects_registry import project_task_bindings

    refs: List[Dict[str, Any]] = []
    for row in project_task_bindings(drive_root).values():
        try:
            same_chat = int(row.get("project_chat_id") or 0) == int(project_chat_id or 0)
        except (TypeError, ValueError):
            same_chat = False
        ref = row.get("source_ref")
        if same_chat and isinstance(ref, dict) and ref:
            refs.append(dict(ref))
    return refs


def origin_rows_by_chat(
    drive_root: Any, project_chat_ids: Iterable[Any]
) -> Dict[int, List[Dict[str, Any]]]:
    """Origin rows for SEVERAL project chats, bucketed, in ONE bindings read.

    A fork's history needs its own origins AND every ancestor's; asking
    :func:`project_origin_rows` per chat re-read
    ``state/project_task_bindings.json`` once per link in the chain. Identity
    dedupe is GLOBAL across the requested chats and uses the same identity
    tuple, so ONE owner message several threads bind to yields ONE row rather
    than one per ancestor.

    Only bindings carrying ``source_text`` qualify (cross-thread origins — the
    binding is the retention-proof copy of the message that started the
    project).
    """
    from ouroboros.projects_registry import project_task_bindings

    wanted: set = set()
    for value in project_chat_ids or ():
        try:
            wanted.add(int(value or 0))
        except (TypeError, ValueError):
            continue
    out: Dict[int, List[Dict[str, Any]]] = {}
    if not wanted:
        return out
    seen: set = set()
    for row in project_task_bindings(drive_root).values():
        try:
            owner = int(row.get("project_chat_id") or 0)
        except (TypeError, ValueError):
            continue
        if owner not in wanted:
            continue
        ref = row.get("source_ref")
        text = row.get("source_text")
        if not (isinstance(ref, dict) and ref and isinstance(text, str) and text):
            continue
        identity = (
            str(ref.get("chat_id") or ""),
            str(ref.get("client_message_id") or ""),
            str(ref.get("ts") or ""),
            str(ref.get("text_sha256") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        out.setdefault(owner, []).append({"ref": dict(ref), "text": text})
    return out


def project_origin_rows(drive_root: Any, project_chat_id: int) -> List[Dict[str, Any]]:
    """Origin rows a Project lens can SYNTHESIZE when the canonical row is gone.

    Single-chat projection of :func:`origin_rows_by_chat` (kept as the stable
    name for callers that read exactly one lens)."""
    try:
        chat = int(project_chat_id or 0)
    except (TypeError, ValueError):
        return []
    return origin_rows_by_chat(drive_root, (chat,)).get(chat, [])


def entry_matches_source_ref(entry: Dict[str, Any], refs: Iterable[Dict[str, Any]]) -> bool:
    """Whether ``entry`` is the original row identified by one binding ref."""
    if str(entry.get("direction") or "") != "in":
        return False
    try:
        entry_chat_id = int(entry.get("chat_id", 1) or 1)
    except (TypeError, ValueError):
        entry_chat_id = 1
    entry_client_id = str(entry.get("client_message_id") or "")
    entry_ts = str(entry.get("ts") or "")
    entry_hash = _text_sha256(entry.get("text"))
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        try:
            if int(ref.get("chat_id") or 0) != entry_chat_id:
                continue
        except (TypeError, ValueError):
            continue
        client_id = str(ref.get("client_message_id") or "")
        if client_id and client_id != entry_client_id:
            continue
        if str(ref.get("ts") or "") and str(ref.get("ts")) != entry_ts:
            continue
        if str(ref.get("text_sha256") or "") != entry_hash:
            continue
        return True
    return False


def _latest_annotations(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl_objects(path):
        message_id = str(row.get("client_message_id") or "")
        if message_id and row.get("type") == "chat_annotation":
            latest[message_id] = dict(row)
    return latest


def latest_chat_annotations(drive_root: Any) -> Dict[str, Dict[str, Any]]:
    """Latest presentation annotation per message; a torn tail is ignored."""
    path = pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME
    return _latest_annotations(path)


def chat_annotation_receipt(
    drive_root: Any, client_message_id: str, routing_token: str,
) -> Dict[str, Any]:
    """Return the exact token-bound annotation for one routing attempt."""
    row = latest_chat_annotations(drive_root).get(str(client_message_id or ""), {})
    if str(row.get("routing_token") or "") != str(routing_token or ""):
        return {}
    return dict(row)


def _compact_annotations_locked(drive_root: Any, path: pathlib.Path) -> None:
    if not path.is_file() or path.stat().st_size < _COMPACT_AT_BYTES:
        return
    retained_ids = {
        str(row.get("client_message_id") or "")
        for chat_path in _chat_paths(drive_root)
        for row in iter_jsonl_objects(chat_path)
        if row.get("client_message_id")
    }
    rows = [
        row for message_id, row in _latest_annotations(path).items()
        if message_id in retained_ids
    ]
    rows.sort(key=lambda row: str(row.get("ts") or ""))
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def append_chat_annotation(
    drive_root: Any,
    client_message_id: str,
    *,
    action: str,
    target: str = "",
    status: str,
    routing_token: str = "",
    reason: str = "",
    detail: str = "",
    options: Any = None,
) -> bool:
    """Append one compact UI annotation; no semantic routing state is stored."""
    message_id = str(client_message_id or "").strip()
    if not message_id:
        return False
    row = {
        "ts": utc_now_iso(),
        "type": "chat_annotation",
        "client_message_id": message_id[:200],
        "action": str(action or "")[:80],
        "target": str(target or "")[:200],
        "status": str(status or "")[:80],
    }
    if str(routing_token or ""):
        row["routing_token"] = str(routing_token)[:128]
    if str(reason or ""):
        row["reason"] = str(reason)[:200]
    if str(detail or ""):
        row["detail"] = str(detail)[:1000]
    if isinstance(options, list):
        row["options"] = [dict(item) for item in options[:100] if isinstance(item, dict)]
    path = pathlib.Path(drive_root) / "logs" / _ANNOTATIONS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = jsonl_append_lock_path(path)
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=10.0)
    if lock_fd is None:
        return False
    try:
        data = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        _compact_annotations_locked(drive_root, path)
        return True
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


__all__ = [
    "append_chat_annotation",
    "build_owner_message_ref",
    "chat_annotation_receipt",
    "entry_matches_source_ref",
    "latest_chat_annotations",
    "origin_rows_by_chat",
    "project_origin_rows",
    "source_refs_for_project",
]

"""Revision-checked publication of shared plain memory files.

Inference happens before this seam. Only the source check, history append and
atomic replacement share the short per-file lock, like knowledge publication:
the history row is written first, so a version this function publishes is in its
journal before the file holds it, and a row that cannot be written leaves the
file untouched.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterator

from ouroboros.platform_layer import file_lock_exclusive, file_unlock
from ouroboros.utils import append_jsonl, utc_now_iso, write_bytes_atomic


@dataclass(frozen=True)
class Publication:
    ok: bool
    reason: str
    revision: str
    old_content: str
    content: str


@contextmanager
def _publication_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+b") as lock:
        file_lock_exclusive(lock.fileno())
        try:
            yield
        finally:
            file_unlock(lock.fileno())


def publish_memory_file(path: Path, expected_revision: str | None, render: Callable[[str], str],
                        *, journal: Path, author: dict[str, Any], source_type: str) -> Publication:
    """A missing file accepts None/empty; an existing file requires its byte revision.

    A read error is never treated as an empty source. Conflicts publish neither
    source nor journal. Successful history retains both versions and origin; the
    row precedes the replacement and its append is checked (``history_unavailable``
    publishes nothing). A row whose replacement then fails names a version that
    never landed, as a knowledge capture does.
    """
    with _publication_lock(path):
        raw = path.read_bytes() if path.exists() else None
        revision = sha256(raw).hexdigest() if raw is not None else ""
        old = raw.decode("utf-8") if raw is not None else ""
        if raw is not None and expected_revision is None:
            return Publication(False, "revision_required", revision, old, old)
        if (expected_revision or "") != revision:
            return Publication(False, "revision_conflict", revision, old, old)
        content = render(old)
        new_raw = content.encode("utf-8")
        new_revision = sha256(new_raw).hexdigest()
        if not append_jsonl(journal, {
            "ts": utc_now_iso(), "task_id": author["task_id"], "source_type": source_type,
            "focus": author["focus"], "route": author["route"],
            "old_len": len(old), "new_len": len(content),
            "old_sha256": revision, "new_sha256": new_revision,
            "old_content": old, "new_content": content,
            "old_preview": old[:500], "new_preview": content[:500],
        }, ensure_record_boundary=True, require_lock=True):
            return Publication(False, "history_unavailable", revision, old, old)
        write_bytes_atomic(path, new_raw)
        return Publication(True, "saved", new_revision, old, content)


def publication_refusal(result: Publication, path: str) -> str:
    if result.reason == "history_unavailable":
        return (f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: history_unavailable; current_revision={result.revision!r}. "
                f"Its history journal could not record this version, so {path} was not changed; "
                "the same call can be retried.")
    return (f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: {result.reason}; current_revision={result.revision!r}. "
            f"Read {path}, reconcile your change, then retry with expected_revision from that read.")

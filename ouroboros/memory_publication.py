"""Revision-checked publication of shared plain memory files.

Inference happens before this seam. Only the source check, history append and
atomic replacement share the short per-file lock, like knowledge publication:
the history row is written first, so a version this function publishes is in its
journal before the file holds it, and a row that cannot be written leaves the
file untouched. ``publish_memory_file`` keeps the registry's whole-file rows;
``publish_text`` is the writer of identity, the world map and a project's workpad
(``memory_texts``), whose rows also carry the mode, reason and what changed.
"""
from __future__ import annotations

from collections import Counter
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
    existed: bool = True  # the file held a version before this write (absence is no version, not sha256(b""))
    previous: dict[str, Any] | None = None  # who wrote the replaced version: knowledge_versions.row_facts, revision included
    writer: dict[str, Any] | None = None  # the row that wrote it, or None when no row in the journal's tail did
    held: dict[str, Any] | None = None  # a stale revision's version and the rows since (knowledge_versions.held_version)
    delta: dict[str, Any] | None = None  # what the write changed, as its row records it
    detail: str = ""  # a refusal's own words


class Refused(ValueError):
    """A write a plain text's own rules refuse once its current bytes are known (an edit item, an empty result, the
    host's block, a version that does not read): ``reason`` is the refusal's code, the message says why."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


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


# One read of the journal's tail per write, the note writer's budget (knowledge._JOURNAL_TAIL_BYTES): the replaced
# version's writer and its base are looked for there; a miss says "no row wrote it" and stores both texts whole.
_TAIL_BYTES = 8 * 1024 * 1024


def _parsed(text: str) -> Any:
    """A text as the history helpers read one: its bytes and its Markdown headings (``None`` when it does not parse)."""
    from types import SimpleNamespace

    from ouroboros.markdown_source import MarkdownSourceError, parse_markdown_source

    raw = text.encode("utf-8")
    try:
        source = parse_markdown_source(raw, "")
    except MarkdownSourceError:
        source = None
    return SimpleNamespace(raw=raw, source=source)


def text_delta(old: str | None, new: str, change: list) -> dict[str, Any]:
    """What a write changed, as a plain text's row records it: sizes, characters removed and added (``text_change``),
    the headings that left and came and the sections that left (``None`` where a side does not parse)."""
    from ouroboros import knowledge_versions, text_change

    removed, added = text_change.change_chars(change)
    before, after = (_parsed(old) if old is not None else None), _parsed(new)
    known = (before is None or before.source is not None) and after.source is not None
    counts = [Counter((h.level, h.title) for h in item.source.headings) if item is not None and item.source else Counter()
              for item in (before, after)]
    return {"old_chars": len(old or ""), "new_chars": len(new), "change_chars": len(new) - len(old or ""),
            "removed_chars": removed, "added_chars": added,
            "removed_headings": sorted("#" * level + " " + title for level, title in (counts[0] - counts[1]).elements())
            if known else None,
            "added_headings": sorted("#" * level + " " + title for level, title in (counts[1] - counts[0]).elements())
            if known else None,
            "removed_sections": knowledge_versions.removed_sections(before, after)}


def publish_text(source: Any, expected_revision: str | None, prepare: Callable[[str], str], *, author: dict[str, Any],
                 source_type: str, mode: str, reason: str = "", reverted_from: str | None = None) -> Publication:
    """One write of a plain memory text (``memory_texts.TextSource``) in one critical section.

    Under the per-file lock the host's startup refresh of the map also takes: read the current bytes;
    check the revision (an existing text needs its revision, unless the source lets an overwrite be blind;
    an empty ``expected_revision`` is create-only; edit and revert need an existing text); ``prepare`` the
    new text from the current one (``Refused`` writes nothing); answer ``unchanged`` for identical bytes of
    an EXISTING text (no row, no file); find the row that wrote the replaced version (the newest row of the
    journal's tail that published it) and, for a source that stores changes, its verified base
    (``knowledge_rows``); append the row (checked, locked, record boundary) and only then replace the file
    atomically. A row whose replacement fails names a version that never landed, as a knowledge capture
    does; the file keeps its previous bytes and the same call can be retried. No new durability:
    ``append_jsonl`` does not fsync and the atomic replacement's fsync is not asked for.
    """
    from ouroboros import knowledge_rows, knowledge_versions, text_change

    path, journal = Path(source.path), Path(source.journal)
    with _publication_lock(path):
        raw = path.read_bytes() if path.exists() else None
        revision = sha256(raw).hexdigest() if raw is not None else ""
        old = raw.decode("utf-8") if raw is not None else ""
        if raw is None and mode in ("edit", "revert"):
            return Publication(False, f"{mode}_source_missing", "", "", "", existed=False)
        if raw is not None and expected_revision is None and not (source.blind and mode == "overwrite"):
            return Publication(False, "revision_required", revision, old, old)
        if expected_revision is not None and expected_revision != revision:
            held = (knowledge_versions.held_version(journal, source, expected_revision, _TAIL_BYTES, old)
                    if expected_revision and raw is not None else None)
            return Publication(False, "revision_conflict", revision, old, old, existed=raw is not None, held=held)
        try:
            content = prepare(old)
        except Refused as exc:
            return Publication(False, exc.reason, revision, old, old, existed=raw is not None, detail=str(exc))
        new_raw = content.encode("utf-8")
        if raw is not None and new_raw == raw:
            return Publication(True, "unchanged", revision, old, old)
        rows = list(knowledge_versions.tail_rows(journal, source, _TAIL_BYTES)) if raw is not None else []
        writer = next((row for _offset, row in rows if row.get("new_sha256") == revision), None)
        change = text_change.exact_change(old, content)
        delta = text_delta(old if raw is not None else None, content, change)
        if source.changes:
            base = knowledge_rows.verified_base(journal, rows, revision) if raw is not None else None
            texts = knowledge_rows.encode(old, content, base, change)
        else:  # identity and the map keep both whole texts, as their journals always did
            texts = {"old_content": old, "new_content": content, "old_preview": old[:500], "new_preview": content[:500]}
        new_revision = sha256(new_raw).hexdigest()
        if not append_jsonl(journal, {
            "ts": utc_now_iso(), "task_id": author["task_id"], "source_type": source_type,
            "focus": author["focus"], "route": author["route"], **({"topic": source.topic} if source.topic else {}),
            "mode": mode, "old_len": len(old), "new_len": len(content), "old_sha256": revision, "new_sha256": new_revision,
            **texts, "delta": delta, **({"reason": reason} if reason.strip() else {}),
            **({"reverted_from": reverted_from} if reverted_from else {}),
        }, ensure_record_boundary=True, require_lock=True):
            return Publication(False, "history_unavailable", revision, old, old, existed=raw is not None)
        try:
            write_bytes_atomic(path, new_raw)
        except OSError as exc:
            return Publication(False, "replace_failed", revision, old, old, existed=raw is not None,
                               detail=f"{type(exc).__name__}: {exc}")
        previous = {**knowledge_versions.row_facts(writer or {}), "revision": revision} if raw is not None else None
        return Publication(True, "saved", new_revision, old, content, existed=raw is not None, previous=previous,
                           writer=writer, delta=delta)

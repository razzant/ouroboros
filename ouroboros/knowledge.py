"""Linked Markdown knowledge on the existing global and project shelves.

The files are the authority. This module owns address resolution, source reads,
generated inventory and the short source/history/index write transaction. It
does not select memories, call models or decide what an observation means.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import quote, unquote, urlsplit

import yaml

from ouroboros.markdown_source import MarkdownSource, parse_markdown_source
from ouroboros.platform_layer import file_lock_exclusive, file_unlock
from ouroboros.utils import append_jsonl, utc_now_iso, write_bytes_atomic

INDEX_FILE = "index-full.md"
OVERVIEW_TOPIC = "overview"
_INDEX_HEADER = "# Knowledge Base Index\n<!-- ouroboros:knowledge-index:1 -->\n\n"
_LEGACY_INDEX_MARKER = "\n<!-- ouroboros:legacy-knowledge-index -->\n"


def sanitize_topic(topic: str) -> str:
    """Keep a shelf-relative topic identity, including useful nested names."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Topic must be a non-empty string")
    topic = topic.strip()
    path = PurePosixPath(topic)
    if (path.is_absolute() or "\\" in topic or "\x00" in topic
            or any(part in ("", ".", "..") for part in topic.split("/"))):
        raise ValueError("Topic must be a relative path within its knowledge shelf")
    if topic in {"index-full", "_index"}:
        raise ValueError(f"Reserved generated topic: {topic}")
    return topic


@dataclass(frozen=True)
class KnowledgeAddress:
    canonical_root: Path
    shelf: Path
    topic: str
    project_id: str = ""

    @property
    def scope(self) -> str:
        return f"project:{self.project_id}" if self.project_id else "global"

    @property
    def path(self) -> Path:
        return self.shelf / f"{self.topic}.md"

    def as_dict(self) -> dict[str, str]:
        return {"scope": self.scope, "topic": self.topic,
                "path": str(self.path), "canonical_root": str(self.canonical_root)}


def resolve_knowledge_address(
    canonical_root: Path, topic: str, scope: str = "", project_id: str = "",
) -> KnowledgeAddress:
    """Resolve a real shelf; an explicit scope never falls back to another one."""
    from ouroboros.project_facts import explicit_project_id_ok

    if not isinstance(scope, str):
        raise ValueError("scope must be global or project:<exact project id>")
    selected = project_id if not scope else ""
    if scope.startswith("project:"):
        selected = scope[len("project:"):]
    elif scope not in ("", "global"):
        raise ValueError("scope must be global or project:<exact project id>")
    if selected and not explicit_project_id_ok(selected):
        raise ValueError("Project scope requires an exact existing project id spelling")
    if scope == "project:":
        raise ValueError("Project scope requires a project id")
    root = Path(canonical_root).resolve()
    shelf = root / "projects" / selected / "knowledge" if selected else root / "memory" / "knowledge"
    address = KnowledgeAddress(root, shelf, sanitize_topic(topic), selected)
    address.path.resolve().relative_to(shelf.resolve())
    return address


@dataclass(frozen=True)
class KnowledgeNote:
    address: KnowledgeAddress
    raw: bytes
    source: MarkdownSource | None
    parse_error: str = ""

    @property
    def revision(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()

    @property
    def text(self) -> str:
        return self.raw.decode("utf-8")

    @property
    def metadata(self) -> Mapping[str, Any]:
        return (self.source.frontmatter or {}) if self.source is not None else {}

    @property
    def summary(self) -> str:
        value = self.metadata.get("summary")
        return value if isinstance(value, str) else ""

    @property
    def title(self) -> str:
        value = self.metadata.get("title")
        if isinstance(value, str) and value.strip():
            return value
        if self.source and self.source.headings:
            return self.source.headings[0].title
        return self.address.topic

    def source_ref(self) -> dict[str, Any]:
        address = self.address
        canonical_shelf = (address.canonical_root / "projects" / address.project_id / "knowledge"
                           if address.project_id else address.canonical_root / "memory" / "knowledge")
        reader = {"tool": "knowledge_read", "arguments": {"topic": address.topic, "scope": address.scope}}
        if address.shelf != canonical_shelf:
            reader = {"tool": "read_file", "arguments": {
                "root": "runtime_data", "path": str(address.path.relative_to(address.canonical_root)),
            }}
        return {"read": reader, "canonical_root": str(self.address.canonical_root),
            "path": str(self.address.path), "revision": self.revision,
            "start_byte": 0, "end_byte": len(self.raw),
            "start_line": 1, "end_line": self.raw.count(b"\n") + int(bool(self.raw) and not self.raw.endswith(b"\n"))}


def _note(address: KnowledgeAddress, raw: bytes) -> KnowledgeNote:
    raw.decode("utf-8")
    try:
        source = parse_markdown_source(raw, str(address.path))
    except (yaml.YAMLError, ValueError) as exc:
        # A malformed legacy preamble stays readable. Its metadata and links
        # are unknown; no synthetic parsed model is allowed to replace it.
        return KnowledgeNote(address, raw, None, f"{type(exc).__name__}: {exc}")
    return KnowledgeNote(address, raw, source)


def read_knowledge_note(address: KnowledgeAddress) -> KnowledgeNote:
    return _note(address, address.path.read_bytes())


def note_descriptor(note: KnowledgeNote) -> dict[str, Any]:
    return {**note.address.as_dict(), "title": note.title, "type": note.metadata.get("type"),
            "summary": note.summary, "revision": note.revision,
            "source_ref": note.source_ref(), "parse_error": note.parse_error}


def inventory_knowledge(address: KnowledgeAddress) -> tuple[dict[str, Any], ...]:
    """Enumerate the complete current shelf without creating any files."""
    if not address.shelf.exists():
        return ()
    rows = []
    for path in sorted(address.shelf.rglob("*.md")):
        if path == address.shelf / INDEX_FILE:
            continue
        topic = path.relative_to(address.shelf).as_posix()[:-3]
        try:
            item = replace(address, topic=sanitize_topic(topic))
            item.path.resolve().relative_to(address.shelf.resolve())
        except ValueError:
            continue
        try:
            rows.append(note_descriptor(read_knowledge_note(item)))
        except (OSError, UnicodeDecodeError) as exc:
            rows.append({**item.as_dict(), "title": topic, "summary": "", "revision": None,
                         "read_error": type(exc).__name__})
    return tuple(rows)


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("*", "\\*")


def render_knowledge_index(rows: tuple[dict[str, Any], ...], legacy_context: str = "",
                           *, include_summaries: bool = True) -> str:
    """Generated navigation is separate from authored understanding."""
    entries = []
    for row in rows:
        topic, title = str(row["topic"]), str(row.get("title") or row["topic"])
        line = f"- **{_label(topic)}**: [{_label(title)}](<{quote(topic + '.md', safe='/')}>)"
        summary = row.get("summary")
        if include_summaries and isinstance(summary, str) and summary:
            line += "\n" + "\n".join("  " + part for part in summary.splitlines())
        if row.get("read_error") or row.get("parse_error"):
            line += "\n  (source metadata unavailable; read the original note)"
        entries.append(line)
    rendered = _INDEX_HEADER + ("\n".join(entries) if entries else "(empty)") + "\n"
    if legacy_context:
        rendered += ("\n## Earlier generated context\n\n"
                     "These earlier index previews are retained until an authored global overview exists. "
                     "They are historical context, not current authored summaries.\n"
                     + _LEGACY_INDEX_MARKER + legacy_context)
    return rendered


@contextmanager
def knowledge_write_lock(shelf: Path):
    """One stable sidecar serializes existing note/index/history writers."""
    shelf.mkdir(parents=True, exist_ok=True)
    fd = os.open(shelf / f"{INDEX_FILE}.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        file_lock_exclusive(fd)
        try:
            yield
        finally:
            file_unlock(fd)
    finally:
        os.close(fd)


def rebuild_knowledge_index(address: KnowledgeAddress) -> None:
    """Caller holds the shelf lock; all producers use the same inventory."""
    path = address.shelf / INDEX_FILE
    old = path.read_bytes() if path.exists() else b""
    text = old.decode("utf-8")
    modern = text.startswith(_INDEX_HEADER)
    legacy = text.partition(_LEGACY_INDEX_MARKER)[2] if modern else text
    # Preserve the exact old source before changing its first generated view.
    # Later rebuilds retain that same source, never a recursively nested index.
    if old and not modern and not append_jsonl(address.shelf.parent / "knowledge_history.jsonl", {
        "ts": utc_now_iso(), "type": "knowledge_index_source", "topic": "index-full",
        "scope": address.scope, "old_content": text,
        "old_sha256": hashlib.sha256(old).hexdigest(),
    }, ensure_record_boundary=True, require_lock=True):
        raise OSError("Previous knowledge index source could not be preserved")
    try:
        overview = read_knowledge_note(resolve_knowledge_address(
            address.canonical_root, OVERVIEW_TOPIC, "global"))
        if overview.source and overview.source.text_at(overview.source.body_span).strip():
            legacy = ""
    except (OSError, UnicodeDecodeError):
        pass
    write_bytes_atomic(path, render_knowledge_index(inventory_knowledge(address), legacy).encode("utf-8"))


def knowledge_links(note: KnowledgeNote) -> tuple[dict[str, Any], ...]:
    """Resolve authored links from their physical source; missing notes are gaps."""
    if note.source is None:
        return ()
    rows = []
    for link in note.source.links:
        url = urlsplit(link.destination)
        row: dict[str, Any] = {"label": link.label, "target": link.destination,
                               "start_byte": link.span.start_byte, "end_byte": link.span.end_byte}
        if url.scheme or url.netloc:
            row["status"] = "external"
        elif not url.path:
            row.update(status="same_source", path=str(note.address.path))
        else:
            target = (note.address.path.parent / unquote(url.path)).resolve()
            row.update(path=str(target), status="present" if target.is_file() else "unwritten")
            try:
                parts = target.relative_to(note.address.canonical_root).parts
                if parts[:2] == ("memory", "knowledge"):
                    scope, topic = "global", "/".join(parts[2:])
                elif len(parts) > 3 and parts[0] == "projects" and parts[2] == "knowledge":
                    scope, topic = f"project:{parts[1]}", "/".join(parts[3:])
                else:
                    scope, topic = "", ""
                if topic.endswith(".md"):
                    address = resolve_knowledge_address(note.address.canonical_root, topic[:-3], scope)
                    row["address"] = address.as_dict()
                    row["read"] = {"tool": "knowledge_read", "arguments": {
                        "topic": address.topic, "scope": address.scope,
                    }}
            except ValueError:
                pass  # An ordinary source link need not be a KB address.
        rows.append(row)
    return tuple(rows)


def apply_knowledge_edits(original: str, edits: Any) -> tuple[str, str]:
    """Compile explicit disjoint edits against the complete note the actor read.

    The caller binds its own read and the existing writer CAS checks that
    revision at publication. Unmentioned text survives byte-for-byte; this
    helper never judges whether the authored basis is semantically correct.
    """
    if not isinstance(edits, list) or not edits:
        return "", "existing_note_requires_edits"
    spans = []
    for edit in edits:
        if not isinstance(edit, dict) or not all(isinstance(edit.get(k), str) for k in
                                                  ("old_text", "new_text", "basis")):
            return "", "invalid_knowledge_edit"
        old, new, basis = edit["old_text"], edit["new_text"], edit["basis"]
        start = original.find(old)
        if (not old or not basis.strip() or old == new or start < 0
                or original.find(old, start + 1) >= 0):
            return "", "unanchored_knowledge_edit"
        spans.append((start, start + len(old), new))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        return "", "overlapping_knowledge_edits"
    updated = original
    for start, end, new in reversed(spans):
        updated = updated[:start] + new + updated[end:]
    return updated, ""


def _write_content(current: KnowledgeNote | None, content: str, mode: str, *, exact: bool = False) -> bytes:
    proposed = content.encode("utf-8")
    if mode == "append" and current is not None:
        return current.raw + (b"\n" if current.raw and not current.raw.endswith(b"\n") else b"") + proposed
    source = parse_markdown_source(proposed, str(current.address.path) if current else "")
    if exact:
        if current is None or current.parse_error or mode != "overwrite":
            raise ValueError("Exact edit requires a readable existing note and overwrite mode")
        return proposed  # An automatic anchored edit may remove YAML keys; no merge or re-dump.
    old = current.metadata if current is not None else {}
    if source.frontmatter is None:
        if current is not None:
            if current.parse_error:
                raise ValueError("The current preamble is malformed; preserve its complete source when editing")
            if current.source.frontmatter_span:
                return current.raw[:current.source.body_span.start_byte] + proposed
            return proposed  # No forced migration of an existing plain Markdown note.
        return b"---\ntype: note\n---\n\n" + proposed
    merged = {**old, **source.frontmatter}
    merged.setdefault("type", "note")
    if current is None and (not isinstance(merged["type"], str) or not merged["type"].strip()):
        raise ValueError("A new note's type must be a non-empty string; any authored type is allowed")
    if merged == source.frontmatter:
        return proposed
    # Unknown fields survive even when an author only supplies changed fields.
    front = yaml.safe_dump(merged, allow_unicode=True, sort_keys=False).encode("utf-8")
    return b"---\n" + front + b"---\n" + proposed[source.body_span.start_byte:]


@dataclass(frozen=True)
class KnowledgeWriteResult:
    ok: bool
    reason: str
    current: KnowledgeNote | None
    previous_revision: str | None = None


def write_knowledge_note(
    address: KnowledgeAddress, content: str, mode: str = "overwrite",
    expected_revision: str | None = None, task_id: str = "", *, exact: bool = False,
) -> KnowledgeWriteResult:
    """Publish a note against the actual current source, with no inference lock."""
    if mode not in {"overwrite", "append"} or not isinstance(content, str):
        raise ValueError("content must be Markdown text; mode must be overwrite or append")
    with knowledge_write_lock(address.shelf):
        # Re-resolve inside the lock; a changed symlink cannot redirect a write.
        address.path.resolve().relative_to(address.shelf.resolve())
        try:
            current = read_knowledge_note(address)
        except FileNotFoundError:
            current = None
        # Blank is a create-only expectation, checked under the same source lock.
        # Never turn it into an unconditional overwrite/append of an existing note.
        if current is None and expected_revision == "":
            expected_revision = None
        revision = current.revision if current else None
        if current is not None and mode == "overwrite" and expected_revision is None:
            return KnowledgeWriteResult(False, "revision_required", current, revision)
        if expected_revision is not None and expected_revision != revision:
            return KnowledgeWriteResult(False, "revision_conflict", current, revision)
        try:
            raw = _write_content(current, content, mode, exact=exact)
        except (ValueError, yaml.YAMLError) as exc:
            return KnowledgeWriteResult(False, f"invalid_note: {exc}", current, revision)
        if current is not None and raw == current.raw:
            return KnowledgeWriteResult(True, "unchanged", current, revision)
        updated = _note(address, raw)
        old_text = current.text if current else ""
        # Capture both complete versions before replacing source bytes, as the
        # Pattern Register already does. A capture is not a commit receipt; a
        # failed publication returns its actual current source, never success.
        history = {"ts": utc_now_iso(), "task_id": task_id, "topic": address.topic, "mode": mode,
                   "address": address.as_dict(), "publication": "source_capture",
                   "old_sha256": hashlib.sha256(current.raw).hexdigest() if current and current.raw else "",
                   "new_sha256": updated.revision if raw else "", "old_content": old_text,
                   "new_content": updated.text, "source_ref": updated.source_ref()}
        if not append_jsonl(address.shelf.parent / "knowledge_history.jsonl", history,
                            ensure_record_boundary=True, require_lock=True):
            return KnowledgeWriteResult(False, "history_unavailable", current, revision)
        try:
            address.path.parent.mkdir(parents=True, exist_ok=True)
            write_bytes_atomic(address.path, raw)
            rebuild_knowledge_index(address)
        except (OSError, UnicodeDecodeError):
            try:
                observed = read_knowledge_note(address)
            except (OSError, UnicodeDecodeError):
                observed = None
            return KnowledgeWriteResult(False, "publication_incomplete", observed, revision)
        try:
            append_jsonl(address.shelf.parent / "knowledge_journal.jsonl", {
                "ts": utc_now_iso(), "task_id": task_id, "topic": address.topic, "mode": mode,
                "address": address.as_dict(), "revision": updated.revision,
                "file_kb": len(raw) / 1024,
                "total_knowledge_kb": round(sum(p.stat().st_size for p in address.shelf.rglob("*.md")) / 1024, 2),
            }, ensure_record_boundary=True)
        except OSError:
            pass  # Size telemetry is not source/history publication authority.
        return KnowledgeWriteResult(True, "saved", updated, revision)

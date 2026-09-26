"""Linked Markdown knowledge on the existing global and project shelves.

The files are the authority. This module owns address resolution, source reads,
generated inventory and the short source/history/index write transaction. It
does not select memories, call models or decide what an observation means.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping
from urllib.parse import quote, unquote, urlsplit

import yaml

from ouroboros.markdown_source import MarkdownSource, parse_markdown_source
from ouroboros.platform_layer import file_lock_exclusive, file_unlock
from ouroboros.utils import append_jsonl, utc_now_iso, write_bytes_atomic

INDEX_FILE = "index-full.md"
UNKNOWN_STAMP = "unknown"  # a history stamp the writer could not name; legacy rows read the same way
OVERVIEW_TOPIC = "overview"
_INDEX_HEADER = "# Knowledge Base Index\n<!-- ouroboros:knowledge-index:1 -->\n\n"
_LEGACY_INDEX_MARKER = "\n<!-- ouroboros:legacy-knowledge-index -->\n"


def observed_route_stamp(usage: Any) -> Any:
    """The route a physical usage row says ANSWERED, as the history ``route`` stamp.

    Every wire lane stamps ``provider`` and ``resolved_model`` on the usage it
    returns (a model-wait override or account rotation changes them, the
    configured route does not), and the Claudexor lane adds its ``route`` with
    the serving ``source``/``account``. Only those physical facts are read: a
    usage without any — a released send, a fake in a test — is the honest
    ``unknown``, and a partial one leaves the missing field ``unknown``; the
    configured or requested route never fills a gap, so no caller argument can.
    An already derived stamp (``_observed_route``, forwarded by usage merges as
    the LAST call's stamp only) is returned as is.
    """
    if not isinstance(usage, dict):
        return UNKNOWN_STAMP
    prior = usage.get("_observed_route")
    if isinstance(prior, dict):
        return prior
    provider, resolved = usage.get("provider"), usage.get("resolved_model")
    if not provider and not resolved:
        return UNKNOWN_STAMP
    stamp: Dict[str, Any] = {"provider": str(provider or UNKNOWN_STAMP), "model": str(resolved or UNKNOWN_STAMP)}
    served = usage.get("claudexor")
    if isinstance(served, dict) and isinstance(served.get("route"), dict):
        route = served["route"]
        if route.get("source"):
            stamp["source"] = route["source"]
        # The engine's served route names the account as ``credentialProfileId``
        # (+ ``accountFingerprint``); ``account`` is the legacy/request spelling.
        account = route.get("credentialProfileId") or route.get("account")
        if account:
            stamp["account"] = str(account)
        if route.get("accountFingerprint"):
            stamp["account_fingerprint"] = str(route["accountFingerprint"])
    return stamp


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


def compile_anchored_edits(text: str, pairs: Any, anchor: str = "old_str") -> str:
    """Replace exact ``(old, new)`` spans of one source text atomically.

    Every old text is located in the ORIGINAL ``text`` before anything changes:
    it must be non-empty and occur exactly once (an overlapping repeat is a
    second occurrence), and no two spans may overlap, so one edit can neither
    create, consume nor shift another's anchor. Unmentioned characters stay
    identical. A refusal raises ``ValueError`` naming the caller's ``anchor``
    field and applies nothing. Manual ``mode=edit`` is the one-pair case.
    """
    pairs = list(pairs)
    spans = []
    for number, (old, new) in enumerate(pairs, 1):
        where = f" (edit {number})" if len(pairs) > 1 else ""
        if not isinstance(old, str) or not old:
            raise ValueError(f"edit requires a non-empty {anchor}{where}")
        start = text.find(old)
        if start < 0 or text.find(old, start + 1) >= 0:
            raise ValueError(f"edit {anchor} must occur exactly once in the note body{where}")
        spans.append((start, start + len(old), new))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("edits must not overlap")
    for start, end, new in reversed(spans):
        text = text[:start] + new + text[end:]
    return text


def _authored_edit_pairs(edits: Any) -> list[tuple[str, str]]:
    """An automatic edit states its old text, replacement and basis; the basis is required, never judged."""
    if not isinstance(edits, list):
        raise ValueError("edits must be a list of {old_text, new_text, basis}")
    for number, edit in enumerate(edits, 1):
        fields = [edit.get(key) for key in ("old_text", "new_text", "basis")] if isinstance(edit, dict) else []
        if len(fields) != 3 or not all(isinstance(value, str) for value in fields) or not fields[2].strip():
            raise ValueError(f"edit {number} needs string old_text and new_text and a non-empty basis")
    return [(edit["old_text"], edit["new_text"]) for edit in edits]


def nomination_write_form(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The one automatic nomination contract, checked before any source is read.

    Either new-note ``content`` (complete Markdown; ``edits`` absent or ``[]``)
    or the anchored form: an ``edits`` list (empty only beside a summary) plus
    an optional ``summary``. Present keys are judged by shape, never truthiness:
    a non-list ``edits`` (``null``, ``""``, ``{}``) or a non-text/blank
    ``summary`` is malformed, the legacy generic ``frontmatter`` is refused
    rather than dropped, and non-blank content beside edits or a summary is
    ambiguous. Returns ``write_knowledge_note`` arguments; a refusal raises
    ``ValueError`` carrying its typed reason."""
    edits, content = entry.get("edits", []), entry.get("content")
    if "frontmatter" in entry:
        raise ValueError("invalid_nomination: frontmatter is not an automatic field; revise the summary with summary")
    if not isinstance(edits, list):
        raise ValueError("invalid_nomination: edits must be a list of {old_text, new_text, basis}")
    if "summary" in entry and (not isinstance(entry["summary"], str) or not entry["summary"].strip()):
        raise ValueError("invalid_nomination: summary must be non-empty text")
    if edits or "summary" in entry:
        if content is not None and not (isinstance(content, str) and not content.strip()):
            raise ValueError("ambiguous_nomination: content creates a new note; edits and summary change a read one")
        return {"content": "", "mode": "edit", "edits": edits, "summary": entry.get("summary")}
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty_nomination")
    return {"content": content, "mode": "overwrite"}


def _yaml_form(metadata: Mapping[str, Any]) -> str:
    """Loaded frontmatter spelled exactly, for comparison without ``==``.

    Legal YAML may alias a node inside itself (``custom: &loop [*loop]``); ``==``
    walks that cycle until RecursionError, while the dump spells a shared node
    once and aliases it. Sorted keys keep dict equality's order blindness."""
    return yaml.safe_dump(metadata, allow_unicode=True, sort_keys=True)


def _write_content(current: KnowledgeNote | None, content: str, mode: str,
                   old_str: str | None = None, edits: Any = None, summary: str | None = None) -> bytes:
    proposed = content.encode("utf-8")
    if mode == "edit":
        if current is None or current.parse_error:
            raise ValueError("edit requires an existing readable note")
        body_start = current.source.body_span.start_byte
        body = current.raw[body_start:].decode("utf-8")
        body = (compile_anchored_edits(body, [(old_str, content)]) if edits is None
                else compile_anchored_edits(body, _authored_edit_pairs(edits), "old_text"))
        if summary is None or current.metadata.get("summary") == summary:
            return current.raw[:body_start] + body.encode("utf-8")
        # A revised summary takes the ordinary overwrite merge below: it replaces
        # its own field, every other field (unknown ones too) survives, and the
        # YAML preamble is re-rendered; the edited body bytes are kept as-is.
        front = yaml.safe_dump({"summary": summary}, allow_unicode=True)
        proposed = f"---\n{front}---\n{body}".encode("utf-8")
    elif mode == "append" and current is not None:
        return current.raw + (b"\n" if current.raw and not current.raw.endswith(b"\n") else b"") + proposed
    source = parse_markdown_source(proposed, str(current.address.path) if current else "")
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
    delta: dict[str, Any] | None = None


def write_knowledge_note(
    address: KnowledgeAddress, content: str, mode: str = "overwrite",
    expected_revision: str | None = None, task_id: str = "", old_str: str | None = None,
    *, writer: str = "", route: Any = None, writer_input_ref: Any = None,
    edits: Any = None, summary: str | None = None,
) -> KnowledgeWriteResult:
    """Publish a note against the actual current source, with no inference lock.

    ``writer`` names the seam that authored ``content`` (turn, consolidation,
    scratchpad_consolidation, reflection, knowledge_maintenance), ``route`` the
    model route it ran on and ``writer_input_ref`` what it saw. They are host
    facts stamped on the history row, never on the note body; a caller that
    cannot name one leaves the honest ``unknown``, which is also how rows written
    before the stamp existed read.

    ``edits`` (mode=edit, instead of ``old_str``/``content``) is the automatic
    form: a list of ``{old_text, new_text, basis}`` applied to the body only by
    the same anchored compiler; beside it, ``summary`` optionally revises that
    one field in the same locked write through the ordinary metadata merge.
    History retains the authored ``edits`` and, only when supplied, ``summary``.
    """
    if mode not in {"overwrite", "append", "edit"} or not isinstance(content, str):
        raise ValueError("content must be Markdown text; mode must be overwrite, append or edit")
    if mode != "edit" and old_str is not None:
        raise ValueError("old_str is used only with mode=edit")
    if mode != "edit" and (edits is not None or summary is not None):
        raise ValueError("edits and summary are used only with mode=edit")
    if edits is not None and (old_str is not None or content):
        raise ValueError("edits replace old_str and content; pass one edit form")
    if summary is not None and (edits is None or not isinstance(summary, str) or not summary.strip()):
        raise ValueError("summary is non-empty text beside edits")
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
        if mode == "edit" and current is None:
            return KnowledgeWriteResult(False, "edit_source_missing", current, revision)
        if current is not None and mode in {"overwrite", "edit"} and expected_revision is None:
            return KnowledgeWriteResult(False, "revision_required", current, revision)
        if expected_revision is not None and expected_revision != revision:
            return KnowledgeWriteResult(False, "revision_conflict", current, revision)
        try:
            raw = _write_content(current, content, mode, old_str, edits, summary)
        except (ValueError, yaml.YAMLError) as exc:
            return KnowledgeWriteResult(False, f"invalid_note: {exc}", current, revision)
        if current is not None and raw == current.raw:
            return KnowledgeWriteResult(True, "unchanged", current, revision,
                                        {"old_chars": len(current.text), "new_chars": len(current.text),
                                         "change_chars": 0, "removed_headings": []})
        updated = _note(address, raw)
        # A body edit keeps the preamble bytes; a revised summary may re-render
        # them only if every other field keeps its value (``type`` defaults, in
        # the merge's key order), compared by exact YAML spelling.
        if mode == "edit" and (updated.parse_error or (
                bool(updated.source.frontmatter_span) != bool(current.source.frontmatter_span) or
                updated.raw[:current.source.body_span.start_byte] !=
                current.raw[:current.source.body_span.start_byte]) and (
                summary is None or _yaml_form(updated.metadata) != _yaml_form(
                    {**current.metadata, "summary": summary, "type": current.metadata.get("type", "note")}))):
            return KnowledgeWriteResult(False, "invalid_note: edit cannot change frontmatter", current, revision)
        old_text = current.text if current else ""
        before = (Counter((heading.level, heading.title) for heading in current.source.headings)
                  if current and current.source else Counter())
        after = (Counter((heading.level, heading.title) for heading in updated.source.headings)
                 if updated.source else Counter())
        removed_headings = (sorted("#" * level + " " + title for (level, title) in (before - after).elements())
                            if (current is None or current.source) and updated.source else None)
        delta = {"old_chars": len(old_text), "new_chars": len(updated.text),
                 "change_chars": len(updated.text) - len(old_text), "removed_headings": removed_headings}
        # Capture both complete versions before replacing source bytes, as the
        # Pattern Register already does. A capture is not a commit receipt; a
        # failed publication returns its actual current source, never success.
        history = {"ts": utc_now_iso(), "task_id": task_id, "topic": address.topic, "mode": mode,
                   "address": address.as_dict(), "publication": "source_capture",
                   "writer": writer or UNKNOWN_STAMP, "route": route or UNKNOWN_STAMP,
                   "writer_input_ref": writer_input_ref or UNKNOWN_STAMP,
                   "old_chars": len(old_text), "new_chars": len(updated.text),
                   "old_sha256": hashlib.sha256(current.raw).hexdigest() if current and current.raw else "",
                   "new_sha256": updated.revision if raw else "", "old_content": old_text,
                   "new_content": updated.text, "source_ref": updated.source_ref(), "delta": delta,
                   **({"edits": edits} if edits is not None else {}),
                   **({"summary": summary} if summary is not None else {})}
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
            return KnowledgeWriteResult(False, "publication_incomplete", observed, revision,
                                        delta if observed is not None and observed.raw == raw else None)
        return KnowledgeWriteResult(True, "saved", updated, revision, delta)

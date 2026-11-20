"""Linked Markdown knowledge on the existing global and project shelves.

The files are the authority. This module owns address resolution, source reads,
generated inventory and the short source/history/index write transaction. It
does not select memories, call models or decide what an observation means.
Generated ``memory/knowledge/index-full.md`` (``INDEX_FILE``) is reserved,
never a topic. A stale revision the journal tail holds gets all changes since,
whole; otherwise the writer returns the current source.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Mapping
from urllib.parse import quote, unquote, unquote_to_bytes, urlsplit

import yaml

from ouroboros import knowledge_rows, knowledge_versions, text_change, utils
# The edit compiler lives beside its one caller's module; ``compile_anchored_edits`` stays importable here.
from ouroboros.knowledge_edits import EditRefused, compile_anchored_edits, compile_edit_items, edit_items, yaml_form
from ouroboros.knowledge_versions import body_chars as body_chars
from ouroboros.markdown_source import MarkdownSource, parse_markdown_navigation, parse_markdown_source
from ouroboros.platform_layer import file_lock_exclusive, file_unlock
from ouroboros.utils import append_jsonl, utc_now_iso, write_bytes_atomic

INDEX_FILE = "index-full.md"
UNKNOWN_STAMP = "unknown"  # a history stamp the writer could not name; legacy rows read the same way
OVERVIEW_TOPIC = "overview"
PERSON_TYPE = "person"  # Authored note type; its summary survives the outline.
# Only the overview, the shared orientation every integrating context loads, must stay active.
ALWAYS_ACTIVE_TOPICS = frozenset({OVERVIEW_TOPIC})
# Topics with one global home whichever room addresses them. ``patterns`` and
# ``improvement-backlog`` are ordinary notes; they keep their old global address
# so an earlier call from a project room still reaches the one existing note.
GLOBAL_ADDRESS_TOPICS = frozenset({OVERVIEW_TOPIC, "patterns", "improvement-backlog"})
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


def focus_signature(ctx: Any) -> Dict[str, Any]:
    """The host's signature of the focus that writes a memory record, never the writer's own claim.

    ``{"kind": "mind", "focus": {role, task_id, parent_task_id, root_task_id,
    chat_id}, "task_id", "route"}``. The role is read from host-copied task
    metadata, first match wins: ``nanny`` (the configured agent-session route,
    the one dispatch fact), ``child`` (``delegation_role == "subagent"``),
    ``consciousness`` (a wake's own ledger category; work a wake starts is a
    ``root``), ``presence``, ``main`` (the direct chat turn), else ``root``.
    The model never supplies it.
    """
    from ouroboros.consciousness_authority import CONSCIOUSNESS_CATEGORY, WAKE_CAUSE_FOCUS, wake_cause
    from ouroboros.dialogue_provenance import is_presence_task
    from ouroboros.subagent_dispatch_notes import _nanny_route_dispatched_for  # D15->D07 is lazy-only

    raw = getattr(ctx, "task_metadata", None)
    meta = dict(raw) if isinstance(raw, Mapping) else {}
    task_id = str(getattr(ctx, "task_id", "") or "")
    if _nanny_route_dispatched_for(meta, None):
        role = "nanny"
    elif str(meta.get("delegation_role") or "").strip().lower() == "subagent":
        role = "child"
    elif wake_cause(meta) == WAKE_CAUSE_FOCUS:
        role = "presence"
    elif meta.get("usage_category") == CONSCIOUSNESS_CATEGORY:
        role = "consciousness"
    elif is_presence_task({"metadata": meta}):
        role = "presence"
    elif getattr(ctx, "is_direct_chat", False):
        role = "main"
    else:
        role = "root"
    chat_id = getattr(ctx, "current_chat_id", None)
    focus = {"role": role, "task_id": task_id, "parent_task_id": str(meta.get("parent_task_id") or ""),
             "root_task_id": str(meta.get("root_task_id") or task_id),
             "chat_id": chat_id if chat_id is not None else meta.get("chat_id")}
    assignment = meta.get("focus") if isinstance(meta.get("focus"), Mapping) else {}
    room = focus["chat_id"]
    if getattr(ctx, "drive_root", None) is not None:
        from ouroboros.dialogue_evidence import own_room_chat
        from ouroboros.tool_access_paths import canonical_data_root

        resolved = own_room_chat(ctx, canonical_data_root(ctx))
        room = resolved if resolved is not None else room
    room = assignment.get("room_chat_id", room)
    focus["room_id"] = str(room if room is not None else 1)
    from ouroboros.dialogue_provenance import presence_provenance_from_task

    origin = {**presence_provenance_from_task({"metadata": meta}), **assignment}
    for key in ("place_id", "conversation_key"):
        if origin.get(key):
            focus[key] = str(origin[key])
    return {"kind": "mind", "focus": focus, "task_id": task_id,
            "route": observed_route_stamp(getattr(ctx, "_accumulated_usage", None))}


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


def _folded(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def shelf_spelling(address: KnowledgeAddress) -> KnowledgeAddress:
    """Keep the directory-entry spelling when it can be read."""
    parent = address.shelf
    for part in address.path.relative_to(address.shelf).parts:
        requested = parent / part
        try:
            if not requested.exists():
                parent = requested
                continue
            entries = os.listdir(parent)
            if part not in entries:
                wanted = _folded(part)
                part = next((name for name in entries if _folded(name) == wanted
                             and requested.samefile(parent / name)), part)
        except OSError:
            return address
        parent /= part
    topic = parent.relative_to(address.shelf).as_posix()[:-3]
    return replace(address, topic=sanitize_topic(topic)) if topic != address.topic else address


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

    @property
    def archive_error(self) -> str:
        return _archive_error(self.address.topic, self.metadata)

    @property
    def state(self) -> str:
        # Unknown metadata never hides a source from ordinary navigation.
        return "archived" if "archive" in self.metadata and not self.archive_error else "active"

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


def read_knowledge_note(address: KnowledgeAddress, *, captured=None) -> KnowledgeNote:
    if captured is not None:
        identity, raw = captured
        if utils.file_identity(address.path.stat()) == identity:
            return _note(address, raw)
    return _note(address, address.path.read_bytes())


def note_descriptor(note: KnowledgeNote) -> dict[str, Any]:
    return {**note.address.as_dict(), "title": note.title, "type": note.metadata.get("type"),
            "summary": note.summary, "revision": note.revision,
            "claims": knowledge_versions.claim_names(note.metadata.get("claims")),
            "source_ref": note.source_ref(), "parse_error": note.parse_error,
            "state": note.state, "archive": note.metadata.get("archive"), "archive_error": note.archive_error}


def _archive_error(topic: str, metadata: Mapping[str, Any]) -> str:
    if "archive" not in metadata:
        return ""
    value = metadata["archive"]
    if topic in ALWAYS_ACTIVE_TOPICS:
        return "This shared root must stay active"
    if not isinstance(value, dict) or not all(
            isinstance(value.get(key), str) and value[key].strip() for key in ("at", "reason")):
        return "archive must contain non-empty at and reason strings"
    return ""


def _navigation_descriptor(address: KnowledgeAddress, raw: bytes | None = None) -> dict[str, Any]:
    raw = address.path.read_bytes() if raw is None else raw
    raw.decode("utf-8")  # unreadable bytes remain a read_error, as in full inventory
    try:
        metadata, title = parse_markdown_navigation(raw, str(address.path))
    except (yaml.YAMLError, ValueError) as exc:
        return {**address.as_dict(), "claims": [], "parse_error": f"{type(exc).__name__}: {exc}"}
    error = _archive_error(address.topic, metadata)
    claims = knowledge_versions.claim_names(metadata.get("claims"))
    return {**address.as_dict(), "title": title, "type": metadata.get("type"), "summary": metadata.get("summary"),
            "state": "archived" if "archive" in metadata and not error else "active",
            "archive": metadata.get("archive"), "archive_error": error, "keys": metadata.get("keys"),
            "claims": claims, **({"revision": hashlib.sha256(raw).hexdigest()} if claims else {})}


# a note's address -> (what proves its bytes unchanged, its navigation row); a listing drops deleted notes
_NAVIGATION: Dict[KnowledgeAddress, tuple] = {}


def _navigation_row(address: KnowledgeAddress, bodies: dict | None = None,
                    names: frozenset[str] = frozenset()) -> dict[str, Any]:
    """``_navigation_descriptor``, parsed again only when the note's bytes may have changed: while its
    file identity holds (``utils.file_identity``, the rule ``utils.json_snapshot`` keeps on every OS;
    Windows' ``st_ctime`` is the creation time, its in-place writes move the mtime), settled beyond a
    clock tick. A row with a parse error is never kept."""
    started = time.time_ns()
    info = os.stat(address.path)  # before the read: a write after it moves the identity
    key, settled = utils.file_identity(info), max(info.st_mtime_ns, info.st_ctime_ns) < started - utils._RACY_NS
    held = _NAVIGATION.get(address)
    if held is not None and held[0] == key:
        return dict(held[1])
    raw = address.path.read_bytes()
    row = _navigation_descriptor(address, raw)
    if bodies is not None and row.get("state") == "active" and not names.isdisjoint(row.get("claims", ())):
        bodies[address] = (key, raw)
    if settled and not row.get("parse_error"):
        _NAVIGATION[address] = (key, row)
    return dict(row)


def inventory_knowledge(address: KnowledgeAddress, *, navigation: bool = False,
                        known_note: KnowledgeNote | None = None, after: str = "",
                        limit: int | None = None, bodies: dict | None = None,
                        body_claims: tuple[str, ...] = ()) -> tuple[dict[str, Any], ...]:
    """Enumerate the current shelf without writes; navigation omits source proofs.

    A note already read by this operation can be reused. A note's navigation is
    reused only while its identity proves the bytes unchanged: external edits and
    missing/failed index publications stay visible. ``after`` (a topic) and ``limit``
    read one page in the order of the shelf-relative parts, the order the page cursor
    is compared in on every platform (``WindowsPath`` would sort them case-folded):
    only that page's notes are opened.
    """
    if not address.shelf.exists():
        return ()
    rows, listed, past = [], set(), tuple(f"{after}.md".split("/")) if after else ()
    names = frozenset(body_claims)
    for path in sorted(address.shelf.rglob("*.md"), key=lambda found: found.relative_to(address.shelf).parts):
        if path == address.shelf / INDEX_FILE or path.relative_to(address.shelf).parts <= past:
            continue
        if limit is not None and len(rows) >= limit:
            break
        topic = path.relative_to(address.shelf).as_posix()[:-3]
        try:
            item = replace(address, topic=sanitize_topic(topic))
            item.path.resolve().relative_to(address.shelf.resolve())
        except ValueError:
            continue
        listed.add(item)
        try:
            rows.append(note_descriptor(known_note) if known_note is not None and item.path == known_note.address.path
                        else _navigation_row(item, bodies, names) if navigation else note_descriptor(read_knowledge_note(item)))
        except (OSError, UnicodeDecodeError) as exc:
            rows.append({**item.as_dict(), "title": topic, "summary": "", "revision": None,
                         "read_error": type(exc).__name__})
    if navigation and not after and limit is None:  # only a whole listing proves a note gone
        # A snapshot: an extension's paged read may keep a row from another thread meanwhile.
        for gone in [held for held in list(_NAVIGATION) if held.shelf == address.shelf and held not in listed]:
            _NAVIGATION.pop(gone, None)
    return tuple(rows)


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("*", "\\*")


def render_knowledge_index(rows: tuple[dict[str, Any], ...], legacy_context: str = "",
                           *, include_summaries: bool = True, view: str = "active", scope: str = "global",
                           selection: tuple[str, ...] = ()) -> str:
    """Generated navigation is separate from authored understanding."""
    if view not in {"active", "archived", "all"}:
        raise ValueError("view must be active, archived or all")
    entries = []
    for row in rows:
        state = row.get("state", "active")
        if view != "all" and state != view:
            continue
        topic, title = str(row["topic"]), str(row.get("title") or row["topic"])
        line = f"- **{_label(topic)}**: [{_label(title)}](<{quote(topic + '.md', safe='/')}>)"
        if state == "archived":
            line += " (archived)"
            if archive := row.get("archive"):
                line += "\n  Archived at: " + "\n  ".join(archive["at"].splitlines())
                line += "\n  Reason: " + "\n  ".join(archive["reason"].splitlines())
        summary = row.get("summary")
        if include_summaries and isinstance(summary, str) and summary:
            line += "\n" + "\n".join("  " + part for part in summary.splitlines())
        elif include_summaries and state != "archived" and not row.get("read_error") and not row.get("parse_error"):
            line += " (no summary)"  # a gap the author sees; nothing is synthesized from the body
        if row.get("read_error") or row.get("parse_error") or row.get("archive_error"):
            line += "\n  (source metadata unavailable; read the original note)"
        entries.append(line)
    selection_note = ""
    if selection:
        selection_note = "Selection: " + json.dumps(list(selection), ensure_ascii=False) + "\n"
        unmatched = [item for item in selection if not any(
            _selected(str(row["topic"]), (item,)) and (view == "all" or row.get("state", "active") == view)
            for row in rows)]
        if unmatched:
            selection_note += f"Unmatched selectors in {view} view: " + json.dumps(unmatched, ensure_ascii=False) + "\n"
        selection_note += "\n"
    rendered = _INDEX_HEADER + selection_note + ("\n".join(entries) if entries else "(empty)") + "\n"
    archived = sum(row.get("state") == "archived" for row in rows)
    archive_label = "Archived notes in this selection" if selection else "Archived notes"
    selected_arg = f", topics={list(selection)!r}" if selection else ""
    rendered += (f"\n{archive_label}: {archived}. Open with knowledge_list(scope={scope!r}, view='archived'{selected_arg}); "
                 f"complete inventory: knowledge_list(scope={scope!r}, view='all').\n")
    if legacy_context:
        rendered += ("\n## Earlier generated context\n\n"
                     "These earlier index previews are retained until an authored global overview exists. "
                     "They are historical context, not current authored summaries; they may still mention archived notes.\n"
                     + _LEGACY_INDEX_MARKER + legacy_context)
    return rendered


def outline_keeps_summary(note_type: Any) -> bool:
    """Whether the outline keeps a note's summary: a note typed person does, the type read as a family (its first
    word, split at hyphens, underscores and any whitespace: ``person``, ``person-note``). The receipt reads it too."""
    return str(note_type or "").casefold().replace("_", " ").replace("-", " ").split()[:1] == [PERSON_TYPE]


def render_knowledge_outline(rows: tuple[dict[str, Any], ...], legacy_context: str = "", *,
                             scope: str = "global") -> str:
    """The index at a coarser resolution, never a selection: every active note keeps its line.

    A line is the note's address and its title, grouped by the shelf directory the note was
    written into (its authored path, never an inferred category). A note whose authored
    type is ``person`` or of its family (``outline_keeps_summary``) keeps its summary: understanding
    of people is the continuity an outline must not thin. Every summary and body stays one read away.
    """
    active = [row for row in rows if row.get("state", "active") == "active"]
    groups: dict[str, list[str]] = {}
    for row in active:
        topic, title = str(row["topic"]), str(row.get("title") or row["topic"]).strip()
        line = f"- {topic}" + (f" — {_label(title)}" if title and title != topic else "")
        summary = row.get("summary")
        if outline_keeps_summary(row.get("type")) and isinstance(summary, str) and summary.strip():
            line += "\n" + "\n".join("  " + part for part in summary.splitlines())
        if row.get("read_error") or row.get("parse_error") or row.get("archive_error"):
            line += " (source metadata unavailable; read the original note)"
        groups.setdefault(topic.rsplit("/", 1)[0] + "/" if "/" in topic else "", []).append(line)
    body = "\n\n".join((f"{group}\n" if group else "") + "\n".join(lines)
                        for group, lines in sorted(groups.items(), key=lambda item: (item[0] != "", item[0])))
    archived = sum(row.get("state") == "archived" for row in rows)
    rendered = (f"Outline: every active note ({len(active)}) by its address and title; notes typed person keep their "
                f"summaries. Full summaries: knowledge_list(scope={scope!r}); one note: "
                f"knowledge_read(topic=..., scope={scope!r}).\n\n" + (body or "(empty)") + "\n")
    rendered += f"\nArchived notes: {archived}. Open with knowledge_list(scope={scope!r}, view='archived').\n"
    if legacy_context:
        rendered += f"Earlier generated context is retained; knowledge_list(scope={scope!r}) shows it.\n"
    return rendered


def _selected(topic: str, topics: tuple[str, ...]) -> bool:
    return any(topic == item or (item.endswith("/") and topic.startswith(item)) for item in topics)


def _legacy_prose(address: KnowledgeAddress, view: str, overview: KnowledgeNote | None) -> str:
    """Old authored index prose cannot be filtered by lifecycle. Keep it visibly
    historical until the existing global-overview retirement condition applies."""
    path = address.shelf / INDEX_FILE
    text = path.read_bytes().decode("utf-8") if path.exists() else ""
    legacy = text.partition(_LEGACY_INDEX_MARKER)[2] if text.startswith(_INDEX_HEADER) else text
    if legacy and view == "active":
        try:
            if overview is None:
                overview = read_knowledge_note(resolve_knowledge_address(address.canonical_root, OVERVIEW_TOPIC, "global"))
            if overview.source and overview.source.text_at(overview.source.body_span).strip():
                legacy = ""
        except (OSError, UnicodeDecodeError):
            pass
    return legacy if view == "active" else ""


def knowledge_index_view(address: KnowledgeAddress, view: str = "active", *,
                         overview: KnowledgeNote | None = None, topics: tuple[str, ...] = ()) -> str:
    """Read-only, current-source projection shared by context, list and index writers.

    ``topics`` narrows it to exact topics or groups (an entry ending with ``/``), summaries kept.
    """
    rows = inventory_knowledge(address, navigation=True, known_note=overview)
    if topics:
        return render_knowledge_index(tuple(row for row in rows if _selected(str(row["topic"]), tuple(topics))),
                                      view=view, scope=address.scope, selection=topics)
    return render_knowledge_index(rows, _legacy_prose(address, view, overview), view=view, scope=address.scope)


def knowledge_index_views(address: KnowledgeAddress, *, overview: KnowledgeNote | None = None,
                          whole: bool = True, inventory: tuple[dict[str, Any], ...] | None = None) -> tuple[str, str]:
    """The active index whole (``""`` for a reader of the outline alone) and as its outline, from one inventory."""
    rows = inventory_knowledge(address, navigation=True, known_note=overview) if inventory is None else inventory
    legacy = _legacy_prose(address, "active", overview)
    return (render_knowledge_index(rows, legacy, view="active", scope=address.scope) if whole else "",
            render_knowledge_outline(rows, legacy, scope=address.scope))


def active_claims(drive_root: Path, *, inventory: tuple[dict[str, Any], ...] | None = None) -> dict[str, list[KnowledgeAddress]]:
    """Exact names to global note addresses, reusing the start's navigation inventory when supplied."""
    shelf = resolve_knowledge_address(drive_root, OVERVIEW_TOPIC, "global")
    rows = inventory_knowledge(shelf, navigation=True) if inventory is None else inventory
    found: dict[str, list[KnowledgeAddress]] = {}
    eligible = (row for row in rows if row.get("scope") == "global" and row.get("state") == "active" and row.get("claims"))
    for topic, names in knowledge_versions.inventory_claim_names(shelf.shelf, eligible):
        address = replace(shelf, topic=topic)
        for name in names:
            found.setdefault(name, []).append(address)
    return found


def _key_part(value: Any) -> Any:
    """A key part as written: a YAML integer (an unquoted numeric account id) is its decimal string; nothing else."""
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else value


def key_pages(canonical_root: Path) -> dict[tuple[Any, Any, Any], dict[str, list[str]]]:
    """Every exact key the global notes' ``keys`` lists hold, to their readers by state (``active``, ``archived``):
    one pass over the shelf's navigation, which writes nothing.

    A key is an outside name of a note's subject, ``{provider, namespace, id}`` (an account id in its account scope,
    an email, a repository), three strings only the mind writes in the note's ``keys`` frontmatter list. It joins by
    equality alone: case kept, nothing trimmed, an empty namespace meets only an empty one, a display name never.
    Several notes holding one key are all named, never merged or ranked; a key grants no authority."""
    found: dict[tuple[Any, Any, Any], dict[str, list[str]]] = {}
    for row in inventory_knowledge(resolve_knowledge_address(canonical_root, OVERVIEW_TOPIC, "global"), navigation=True):
        for entry in row.get("keys") if isinstance(row.get("keys"), list) else []:
            if isinstance(entry, Mapping):
                key = tuple(_key_part(entry.get(name)) for name in ("provider", "namespace", "id"))
                try:
                    hash(key)
                except TypeError:  # a part that is no scalar (``id: [a, b]``) equals no arrival's string: no key
                    continue
                readers = found.setdefault(key, {"active": [], "archived": []})[row["state"]]
                reader = f"knowledge_read(topic={row['topic']!r}, scope='global')"
                if reader not in readers:
                    readers.append(reader)
    return found


def speaker_pages(canonical_root: Path | None, provider: str, namespace: str, actor_id: str, *,
                  unverified: bool = False, pages: dict | None = None) -> dict[str, Any]:
    """A place speaker's exact key and the global notes whose ``keys`` hold it (``key_pages``, or ``pages`` read
    once for many speakers), as host facts of the turn; archived ones are named apart. Without a data root or an
    actor id there is no key and no fact."""
    if canonical_root is None or not actor_id:
        return {}
    key = {"provider": provider, "namespace": namespace, "id": actor_id}
    found = (key_pages(canonical_root) if pages is None else pages).get((provider, namespace, actor_id)) or {}
    active, archived = found.get("active") or [], found.get("archived") or []
    facts: dict[str, Any] = {"key": key}
    if len(active) == 1:
        facts["page"] = active[0]
    elif active:
        facts["pages"] = active
    if archived:
        facts["archived_pages"] = archived
    if unverified and (active or archived):
        facts["matched_by"] = "an unverified address"
    return facts


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
    """Caller holds the shelf lock; all producers use the same inventory.

    The file is replaced only when the view's bytes differ from it, so a body-only edit leaves the
    index file, its bytes and its identity, untouched."""
    path = address.shelf / INDEX_FILE
    old = path.read_bytes() if path.exists() else b""
    text = old.decode("utf-8")
    modern = text.startswith(_INDEX_HEADER)
    # Preserve the exact old source before changing its first generated view.
    # Later rebuilds retain that same source, never a recursively nested index.
    if old and not modern and not append_jsonl(address.shelf.parent / "knowledge_history.jsonl", {
        "ts": utc_now_iso(), "type": knowledge_versions.INDEX_SOURCE_TYPE, "topic": "index-full",
        "scope": address.scope, "old_content": text,
        "old_sha256": hashlib.sha256(old).hexdigest(),
    }, ensure_record_boundary=True, require_lock=True):
        raise OSError("Previous knowledge index source could not be preserved")
    view = knowledge_index_view(address).encode("utf-8")
    if view != old:
        write_bytes_atomic(path, view)


def knowledge_links(note: KnowledgeNote, *, before_byte: int | None = None) -> tuple[dict[str, Any], ...]:
    """Resolve authored links from their physical source; missing notes are gaps."""
    if note.source is None:
        return ()
    rows = []
    for link in note.source.links:
        if before_byte is not None and link.span.end_byte > before_byte:
            continue
        row: dict[str, Any] = {"label": link.label, "target": link.destination,
                               "start_byte": link.span.start_byte, "end_byte": link.span.end_byte}
        try:
            url = urlsplit(link.destination)
        except ValueError as exc:
            rows.append({**row, "status": "unavailable", "error": type(exc).__name__})
            continue
        if url.scheme or url.netloc:
            row["status"] = "external"
        elif not url.path:
            row.update(status="same_source", path=str(note.address.path))
        else:
            try:
                target = (note.address.path.parent / unquote(url.path)).resolve()
                row.update(path=str(target), status="present" if target.is_file() else "unwritten")
            except (OSError, ValueError, RuntimeError) as exc:
                rows.append({**row, "status": "unavailable", "error": type(exc).__name__})
                continue
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
                    if scope == "global" and row["status"] == "present":
                        address = shelf_spelling(address)
                    row["address"] = address.as_dict()
                    row["read"] = {"tool": "knowledge_read", "arguments": {
                        "topic": address.topic, "scope": address.scope,
                    }}
            except (OSError, ValueError, RuntimeError):
                pass  # An ordinary source link need not be a KB address.
        rows.append(row)
    return tuple(rows)


def knowledge_backlinks(note: KnowledgeNote, shelves: tuple[KnowledgeAddress, ...]) -> tuple[dict[str, Any], ...]:
    """Current sources on the readable shelves, joined by resolved scope and topic, both in the shelf's spelling
    (``shelf_spelling``; a note whose shelf spelling is reserved keeps the caller's): ``Hub.md`` names ``hub.md``
    only on a volume that folds case.

    File-name text in any case (also percent-encoded) only selects candidates to parse; the
    existing Markdown resolver decides identity. No graph or disk index is kept.
    """
    try:
        target = shelf_spelling(note.address)
    except (OSError, ValueError):  # a reserved on-disk name (the generated index): the caller's spelling, as before
        target = note.address
    filename = _folded(target.path.name)

    def names_target(where: dict) -> bool:  # only a spelling of the target's own name asks the shelf
        if where.get("scope") != target.scope or _folded(where["topic"]) != _folded(target.topic):
            return False
        try:
            return where["topic"] == target.topic or shelf_spelling(
                replace(target, topic=where["topic"])).topic == target.topic
        except (OSError, ValueError):  # a reserved or unresolvable spelling names no note here
            return False

    rows = []
    for shelf in {item.shelf: item for item in shelves}.values():
        for row in inventory_knowledge(shelf, navigation=True):
            address = replace(shelf, topic=row["topic"])
            try:
                raw = address.path.read_bytes()
                text = raw.decode("utf-8", "replace")
                spelled = (text, unquote_to_bytes(raw).decode("utf-8", "replace"),
                           unquote(html.unescape(text).replace("\\", "")))
                if not any(filename in _folded(form) for form in spelled):
                    continue  # also entity- or backslash-spelled names, as the Markdown parser reads them
                source = _note(address, raw)
                links = knowledge_links(source)
            except (OSError, UnicodeDecodeError, ValueError):
                continue  # Unreadable sources or unresolvable URLs supply no link facts.
            if any(names_target(link.get("address", {})) for link in links):
                rows.append({"scope": address.scope, "topic": address.topic, "state": source.state})
    return tuple(rows)


def yaml_fields(content: str) -> dict[str, Any] | None:
    """The fields of a content that is only a YAML block (whitespace may follow it), else ``None``; a block that
    does not parse is a ``ValueError`` naming it. As an edit's content it sets those fields and keeps the body."""
    source = parse_markdown_source(content.encode("utf-8"), "content")
    if source.frontmatter is None or source.text_at(source.body_span).strip():
        return None
    return dict(source.frontmatter)


def _merged_fields(old: Mapping[str, Any], supplied: Mapping[str, Any]) -> dict[str, Any]:
    """Each supplied field replaces its own in place, every other field survives, and ``type`` defaults last: the
    order the writer renders, which ``yaml_form`` compares when PyYAML cannot sort mixed keys (``on:`` is ``True``)."""
    merged = {**old, **supplied}
    merged.setdefault("type", "note")
    return merged


def _write_content(current: KnowledgeNote | None, content: str, mode: str,
                   old_str: str | None = None, summary: str | None = None,
                   items: list | None = None, fields: dict[str, Any] | None = None) -> tuple[bytes, bool]:
    """Prepare the bytes and say whether the supplied frontmatter set claims."""
    proposed = content.encode("utf-8")
    if mode == "edit":
        if current is None or current.parse_error:
            raise ValueError("edit requires an existing readable note")
        body_start = current.source.body_span.start_byte
        body = current.raw[body_start:].decode("utf-8")
        if items is not None:
            body = compile_edit_items(current.raw, body_start, items).decode("utf-8")
        elif old_str is not None or (summary is None and fields is None):  # a field revision keeps every body byte
            body = compile_anchored_edits(body, [(old_str, content)])
        revised = {**(fields or {}), **({"summary": summary} if summary is not None else {})}
        if yaml_form({**current.metadata, **revised}) == yaml_form(current.metadata):
            return current.raw[:body_start] + body.encode("utf-8"), "claims" in revised
        # Revised fields (a YAML-only edit's, a summary) take the ordinary overwrite
        # merge below: each replaces its own field, every other field (unknown ones
        # too) survives, and the YAML preamble is re-rendered; the body bytes are kept as-is.
        front = yaml.safe_dump(revised, allow_unicode=True, sort_keys=False)
        proposed = f"---\n{front}---\n{body}".encode("utf-8")
    elif mode == "append" and current is not None:
        separator = b"\n" if current.raw and not current.raw.endswith(b"\n") else b""
        return current.raw + separator + proposed, False
    source = parse_markdown_source(proposed, str(current.address.path) if current else "")
    old = current.metadata if current is not None else {}
    if source.frontmatter is None:
        if current is not None:
            if current.parse_error:
                raise ValueError("The current preamble is malformed; preserve its complete source when editing")
            if current.source.frontmatter_span:
                return current.raw[:current.source.body_span.start_byte] + proposed, False
            return proposed, False  # No forced migration of an existing plain Markdown note.
        return b"---\ntype: note\n---\n\n" + proposed, False
    merged = _merged_fields(old, source.frontmatter)
    if current is None and (not isinstance(merged["type"], str) or not merged["type"].strip()):
        raise ValueError("A new note's type must be a non-empty string; any authored type is allowed")
    explicit_claims = "claims" in source.frontmatter
    if yaml_form(merged) == yaml_form(source.frontmatter):
        return proposed, explicit_claims
    # Unknown fields survive even when an author only supplies changed fields.
    front = yaml.safe_dump(merged, allow_unicode=True, sort_keys=False).encode("utf-8")
    return b"---\n" + front + b"---\n" + proposed[source.body_span.start_byte:], explicit_claims


def _lifecycle_content(current: KnowledgeNote, mode: str, reason: str) -> bytes:
    if current.parse_error or (mode == "archive" and current.archive_error):
        raise ValueError("lifecycle requires readable, valid metadata")
    metadata = dict(current.metadata)
    if mode == "archive":
        metadata["archive"] = {**metadata.get("archive", {"at": utc_now_iso()}), "reason": reason}
    else:
        metadata.pop("archive", None)
    if yaml_form(metadata) == yaml_form(current.metadata):
        return current.raw
    front = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).encode("utf-8") if metadata else b""
    return (b"---\n" + front + b"---\n" if metadata else b"") + current.raw[current.source.body_span.start_byte:]


@dataclass(frozen=True)
class KnowledgeWriteResult:
    ok: bool
    reason: str
    current: KnowledgeNote | None
    previous_revision: str | None = None
    delta: dict[str, Any] | None = None
    history_ref: dict[str, Any] | None = None
    previous: dict[str, Any] | None = None  # who wrote the replaced version: knowledge_versions.row_facts
    held: dict[str, Any] | None = None  # a stale revision this note's tail holds: knowledge_versions.held_version
    show_source: bool = True  # a refused edit item or revert says why instead of resending the note
    claims: knowledge_versions.ClaimWrite | None = None


def _body_text(note: KnowledgeNote) -> str:
    """The stored body ``body_chars`` counts: the text after the parsed frontmatter, the whole text without one."""
    return note.source.text_at(note.source.body_span) if note.source else note.text


def overview_residency(note: KnowledgeNote) -> str:
    """A revision-stable cost line, independent of route, clock or accepted session."""
    chars = body_chars(note)
    return (f"Residency: overview body {chars} chars; estimate {chars / 4:.2f} tokens (chars/4); "
            "loaded whole in every integrating context (Main, tasks, wakes and Presence); helpers read on demand.")


def write_knowledge_note(
    address: KnowledgeAddress, content: str, mode: str = "overwrite",
    expected_revision: str | None = None, task_id: str = "", old_str: str | None = None,
    *, writer: str = "", route: Any = None, writer_input_ref: Any = None,
    summary: str | None = None, focus: Any = None,
    reason: str = "", capture_previous: Callable[[KnowledgeNote], dict[str, Any]] | None = None,
    edits: Any = None, version: str | None = None,
) -> KnowledgeWriteResult:
    """Publish a note against the actual current source, with no inference lock.

    ``writer`` names the seam that authored ``content`` (the acting mind's
    ``turn``; earlier rows also name retired automatic writers), ``route`` the
    model route it ran on, ``writer_input_ref`` what it saw and ``focus`` which
    focus of the subject wrote it (``focus_signature(ctx)["focus"]``). They are
    host facts stamped on the history row, never on the note body; a caller that
    cannot name one leaves the honest ``unknown``, which is also how rows written
    before the stamp existed read. A non-blank ``reason`` (optional on every mode,
    required for archive, which also writes it into the frontmatter) is stored on
    the row, never in the body.

    ``mode=edit`` replaces one ``old_str`` by ``content``, or applies ``edits``
    (``knowledge_edits.compile_edit_items``: exact ``{old_str, new_str}`` items in the
    original body, all or nothing; a refused item is ``edit_refused``, named). A
    ``content`` alone that is only a YAML block (``yaml_fields``) sets its fields
    through the overwrite's merge and keeps every body byte; its ``claims`` field is
    the turn's explicit write, adopted as an overwrite's.
    ``mode=revert`` publishes the stored bytes of an earlier ``version`` of this
    note exactly (``knowledge_versions.find_version``: verified by sha256), so the
    new revision is that version; a version with other archive metadata is
    ``revert_changes_archive_state``, which names both states (a revert refused
    for its version never resends the note). ``summary`` (overwrite, append or edit)
    revises that one field in the same locked write through the ordinary metadata
    merge, after an overwrite's or append's own text, beside an edit or alone;
    replacing an existing note's summary needs its current revision, as every
    replacement does. History retains the authored ``summary`` only when supplied;
    the delta carries ``old_chars``/``new_chars``/``change_chars``, the characters
    ``removed_chars``/``added_chars`` from the exact change of the two texts
    (``text_change``), ``removed_headings``/``added_headings``, ``removed_sections`` and
    ``body_changed``/``summary_changed`` (heading and changed facts are ``None``
    when a side has no parsed source). The row names its base, this note's newest
    row in the journal's tail read that holds the replaced version and reads back,
    and carries the change from it, with the whole new text by the byte rule
    (``knowledge_rows``); without a verified base, and on lifecycle rows, both texts
    are stored whole. That same read gives ``previous`` (who wrote the replaced
    version) and, for a stale revision one of its rows wrote, ``held`` (that version
    and what came since).

    ``archive``/``restore`` require an existing revision and own only the
    ``archive: {at, reason}`` field, never body bytes. The tool's optional
    ``capture_previous`` exports an actor-readable source under the same lock,
    before history and publication. A failed source capture publishes nothing;
    a failed index publication reports the observed source. Lifecycle no-ops
    rebuild the index so a retry cannot mistake equal note bytes for recovery.
    """
    lifecycle = mode in {"archive", "restore"}
    if mode not in {"overwrite", "append", "edit", "revert", "archive", "restore"} or not isinstance(content, str):
        raise ValueError("content must be Markdown text; mode must be overwrite, append, edit, revert, archive or restore")
    if ((lifecycle or mode == "revert") and content) or (mode == "archive" and address.topic in ALWAYS_ACTIVE_TOPICS):
        raise ValueError("archive/restore/revert take no content; the overview must stay active")
    if not isinstance(reason, str) or (mode == "archive" and not reason.strip()):
        raise ValueError("reason is required for archive and optional text on every other mode")
    if mode != "edit" and (old_str is not None or edits is not None):
        raise ValueError(f"{'old_str' if old_str is not None else 'edits'} is used only with mode=edit")
    if edits is not None and (old_str is not None or content):
        raise ValueError("edits carries its own replacements: pass edits without old_str and content")
    if summary is not None and mode not in {"overwrite", "append", "edit"}:
        raise ValueError("summary is used only with mode=overwrite, append or edit")
    if summary is not None and (not isinstance(summary, str) or not summary.strip()):
        raise ValueError("summary must be non-empty text")
    fields = yaml_fields(content) if mode == "edit" and old_str is None and edits is None and content else None
    if fields is None and mode == "edit" and old_str is None and edits is None and content:
        raise ValueError("content replaces old_str; an edit's content alone is a YAML block, which sets its fields")
    if (mode == "revert") != (version is not None):
        raise ValueError("version is used with mode=revert, which requires it")
    items = edit_items(edits) if edits is not None else None
    target: Any = None
    if mode == "revert":  # outside the lock: a recorded row never changes
        try:
            _, target, _ = knowledge_versions.find_note_version(shelf_spelling(address), address, version)
        except knowledge_versions.VersionUnavailable as exc:
            target = exc
    journal = address.shelf.parent / knowledge_versions.JOURNAL_NAME
    with knowledge_write_lock(address.shelf):
        # Re-resolve inside the lock; a changed symlink cannot redirect a write.
        address.path.resolve().relative_to(address.shelf.resolve())
        address = shelf_spelling(address)
        if mode == "archive" and address.topic in ALWAYS_ACTIVE_TOPICS:
            raise ValueError("the overview must stay active")
        try:
            current = read_knowledge_note(address)
        except FileNotFoundError:
            current = None
        # Blank is a create-only expectation, checked under the same source lock.
        # Never turn it into an unconditional overwrite/append of an existing note.
        if current is None and expected_revision == "":
            expected_revision = None
        revision = current.revision if current else None
        if (mode in {"edit", "revert"} or lifecycle) and current is None:
            return KnowledgeWriteResult(False, "lifecycle_source_missing" if lifecycle else f"{mode}_source_missing", current, revision)
        # Only adding text is blind; a summary replaced beside it is a replacement.
        if current is not None and (mode != "append" or summary is not None) and expected_revision is None:
            return KnowledgeWriteResult(False, "revision_required", current, revision)
        if expected_revision is not None and expected_revision != revision:
            return KnowledgeWriteResult(False, "revision_conflict", current, revision, held=None if not expected_revision
                                        else knowledge_versions.held_version(journal, address, expected_revision, knowledge_versions.WRITE_TAIL_BYTES,
                                                                             current.text if current else None))
        # A refused revert says why; its caller holds the current revision (checked above), so no note is resent.
        if mode == "revert" and version != revision and isinstance(target, Exception):
            return KnowledgeWriteResult(False, f"version_unavailable: {target}", current, revision, show_source=False)
        try:
            explicit_claims = False
            if mode == "revert":
                raw = current.raw if version == revision else target.encode("utf-8")
            elif lifecycle:
                raw = _lifecycle_content(current, mode, reason)
            else:
                raw, explicit_claims = _write_content(current, content, mode, old_str, summary, items, fields)
            if mode in {"append", "overwrite"} and summary is not None:  # the written note's summary-only edit
                raw, _ = _write_content(_note(address, raw), "", "edit", None, summary)
        except EditRefused as exc:
            return KnowledgeWriteResult(False, f"edit_refused: {exc}", current, revision, show_source=False)
        except (ValueError, yaml.YAMLError) as exc:
            return KnowledgeWriteResult(False, f"invalid_note: {exc}", current, revision)
        updated = _note(address, raw)
        if (mode == "overwrite" and writer == "turn" and explicit_claims and current is not None
                and _body_text(current).strip() and not _body_text(updated).strip()):
            # An overwrite replaces the whole note: claims sent alone would erase it; a YAML-only edit keeps the body.
            return KnowledgeWriteResult(False, "invalid_note: this overwrite sets claims and carries no body, so it would "
                                        f"erase the note's body ({body_chars(current)} characters); an overwrite replaces the "
                                        "whole note: send its whole text with the claims list in its YAML, or the YAML alone "
                                        "with mode=edit, which keeps the body", current, revision)
        old_archive = {"archive": current.metadata["archive"]} if current and "archive" in current.metadata else {}
        new_archive = {"archive": updated.metadata["archive"]} if "archive" in updated.metadata else {}
        if not lifecycle and yaml_form(old_archive) != yaml_form(new_archive):
            if mode != "revert":
                return KnowledgeWriteResult(False, "invalid_note: archive is owned by archive/restore modes", current, revision)
            states = (f"is {updated.state}, the current note is {current.state}; archive and restore change that state"
                      if updated.state != current.state else
                      f"and the current note are both {current.state}, with different archive fields; "
                      "archive and restore own that field")
            return KnowledgeWriteResult(False, f"revert_changes_archive_state: version {version} {states}", current,
                                        revision, show_source=False)
        claims = knowledge_versions.prepare_claim_write(address, current, updated, explicit=writer == "turn" and explicit_claims)
        unchanged = current is not None and raw == current.raw
        if unchanged and not (claims and claims.changed):
            try:
                if lifecycle or (claims and claims.needs_publication(revision)):
                    # Equal note bytes are no proof a prior index publication landed.
                    rebuild_knowledge_index(address)
                if claims is not None:
                    claims.publish(address, {}, revision)
            except (OSError, UnicodeDecodeError):
                return KnowledgeWriteResult(False, "publication_incomplete", current, revision)
            return KnowledgeWriteResult(True, "unchanged", current, revision,
                                        {"old_chars": len(current.text), "new_chars": len(current.text),
                                         "old_body_chars": body_chars(current), "new_body_chars": body_chars(current),
                                         "change_chars": 0, "removed_chars": 0, "added_chars": 0, "removed_headings": [],
                                         "added_headings": [], "removed_sections": [], "body_changed": False,
                                         "summary_changed": False}, claims=claims)
        # A body edit keeps the preamble bytes; revised fields (a YAML-only edit's,
        # a summary beside an edit or append) may re-render them only if every other
        # field keeps its value (``type`` defaults, as the merge does), compared by exact YAML spelling.
        revised = {**(fields or {}), **({"summary": summary} if summary is not None else {})}
        if current is not None and (mode == "edit" or (mode == "append" and summary is not None)) and (
                updated.parse_error or (
                bool(updated.source.frontmatter_span) != bool(current.source.frontmatter_span) or
                updated.raw[:current.source.body_span.start_byte] !=
                current.raw[:current.source.body_span.start_byte]) and (
                not revised or yaml_form(updated.metadata) != yaml_form(_merged_fields(current.metadata, revised)))):
            return KnowledgeWriteResult(False, f"invalid_note: {mode} cannot change frontmatter", current, revision)
        old_text = current.text if current else ""
        change = text_change.exact_change(old_text, updated.text)
        delta = knowledge_versions.note_delta(current, updated, change)
        # Capture the new version, and the one it replaces (by its change from a
        # verified base, else whole), before replacing source bytes. A capture
        # is not a commit receipt; a failed publication returns its actual
        # current source, never success.
        newest, base = knowledge_versions.write_tail(journal, address, None if lifecycle else revision) if current else (None, None)
        previous = None if current is None else {**knowledge_versions.row_facts(
            newest[1] if newest is not None and newest[1].get("new_sha256") == revision else {}), "revision": revision}
        history = {"ts": utc_now_iso(), "task_id": task_id, "topic": address.topic, "mode": mode,
                   "address": address.as_dict(), "publication": "source_capture",
                   "writer": writer or UNKNOWN_STAMP, "route": route or UNKNOWN_STAMP,
                   "writer_input_ref": writer_input_ref or UNKNOWN_STAMP, "focus": focus or UNKNOWN_STAMP,
                   "old_chars": len(old_text), "new_chars": len(updated.text),
                   # An existing empty file is the version sha256(b""); only a note's absence has none.
                   "old_sha256": current.revision if current is not None else "", "new_sha256": updated.revision,
                   **({"old_content": old_text, "new_content": updated.text} if lifecycle
                      else knowledge_rows.encode(old_text, updated.text, base, change)),
                   "source_ref": updated.source_ref(), "delta": delta,
                   # Only a write that carries or removes recorded names: a YAML field nobody adopted adds nothing.
                   **({"active_claims": claims.active} if claims is not None and claims.path is not None
                      and (claims.active or claims.recorded) else {}),
                   **({"summary": summary} if summary is not None else {}),
                   **({"reason": reason} if reason.strip() else {}),
                   **({"reverted_from": version} if mode == "revert" else {})}
        history_ref = None
        if lifecycle:
            history_ref = {"history_path": str(journal),
                           "history_field": "old_content", "revision": revision, "new_revision": updated.revision}
            if capture_previous is not None:
                try:
                    history_ref.update(capture_previous(current))
                except (OSError, ValueError) as exc:
                    return KnowledgeWriteResult(False, f"source_capture_unavailable: {exc}", current, revision)
        if not append_jsonl(journal, history, ensure_record_boundary=True, require_lock=True):
            return KnowledgeWriteResult(False, "history_unavailable", current, revision)
        try:
            if not unchanged:
                address.path.parent.mkdir(parents=True, exist_ok=True)
                write_bytes_atomic(address.path, raw)
            rebuild_knowledge_index(address)
            if claims is not None:
                claims.publish(address, history, updated.revision)
            knowledge_versions.held_write(address, history)
        except (OSError, UnicodeDecodeError):
            try:
                observed = read_knowledge_note(address)
            except (OSError, UnicodeDecodeError):
                observed = None
            return KnowledgeWriteResult(False, "publication_incomplete", observed, revision,
                                        delta if observed is not None and observed.raw == raw else None, history_ref)
        return KnowledgeWriteResult(True, "saved", updated, revision, delta, history_ref, previous, claims=claims)

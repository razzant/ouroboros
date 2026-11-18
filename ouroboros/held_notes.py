"""Captured heads of the global notes that claim the acting focus's situation."""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ouroboros import knowledge, knowledge_versions
from ouroboros.chronicle_store import DELEGATED_FOCUS_ROLES
from ouroboros.markdown_source import MarkdownSource
from ouroboros.memory_view_events import situation_names
from ouroboros.section_floor import HostSection

log = logging.getLogger(__name__)
_HEADING = "## Held knowledge notes"
_HEAD_END = re.compile(rb"^## ", re.MULTILINE)
CAPTURE_KEY = "held_note_sections"


@dataclass(frozen=True, eq=False, repr=False)
class HeldNoteSection(HostSection):
    """One entry within the held-notes section, with its captured window forms."""

    topic: str = ""
    revision: str = ""
    window_label: str = ""  # a head this actor has an exact reader for
    summary: str = ""
    outline_has_summary: bool = False

    def __new__(cls, section_id, full_text, *_args, **_kwargs):
        return super().__new__(cls, section_id, full_text)

    def __getnewargs__(self):
        return (*super().__getnewargs__(), self.topic, self.revision, self.window_label,
                self.summary, self.outline_has_summary)

    def with_index(self, row: Mapping, navigation: Mapping) -> HeldNoteSection:
        """Locate the captured summary in this request's index, including after a cold restore."""
        same = row.get("summary") == self.summary and row.get("state") == "active"
        outline = navigation.get("outline", False) and same and knowledge.outline_keeps_summary(row.get("type"))
        resident = same if navigation.get("whole") else outline
        return replace(self, operative_view=self._summary_view(resident), outline_has_summary=outline)

    def _summary_view(self, resident: bool) -> str | None:
        if not self.window_label or not self.summary.strip():
            return None
        marker = "[Head shortened to its authored summary for the physical window.]"
        if resident:
            return f"{self.window_label}\n\n{marker} The summary is in the knowledge index."
        return f"{self.window_label}\n\n{self.summary}\n\n{marker}"

    def view_for_receipt(self, row: Mapping, addressed: Mapping) -> str | None:
        if row.get("held_note_form") == "reader":
            return (self.window_label + "\n\n[Head omitted at the physical window's emergency floor.]"
                    if self.window_label else None)
        return self._summary_view(self.outline_has_summary) if "knowledge_index" in addressed else self.operative_view

    def window_receipt(self, form: str) -> dict:
        return {"section_id": self.section_id, "held_note_form": form, "reader": self.reader,
                "source_ref": {"kind": "knowledge_version", "revision": self.revision,
                               "read": {"tool": "knowledge_read", "arguments": {
                                   "topic": self.topic, "scope": "global", "version": self.revision}}}}


def continuation_capture(plan: Any) -> dict:
    """Carry the original entries by value in the ordinary continuation."""
    core = getattr(plan, "core", None)
    entries = [asdict(part) for part in getattr(core, "dynamic_sections", ()) if isinstance(part, HeldNoteSection)]
    return {CAPTURE_KEY: entries} if entries else {}


def note_head(source: MarkdownSource) -> str:
    """Read physical lines of the body, using the Markdown parser's fenced ranges."""
    start = cursor = source.body_span.start_byte
    end = source.body_span.end_byte
    for fence in source.code_blocks:
        if heading := _HEAD_END.search(source.raw, cursor, fence.start_byte):
            return source.raw[start:heading.start()].decode("utf-8")
        cursor = fence.end_byte
    heading = _HEAD_END.search(source.raw, cursor, end)
    return source.raw[start:heading.start() if heading else end].decode("utf-8")


def _can_read(ctx: Any, topic: str, revision: str) -> bool:
    from ouroboros.presence_authority import (PresenceAuthorityError, apply_presence_argument_bindings,
                                             presence_ceiling_allows_tool, presence_ceiling_from_context)

    try:
        ceiling = presence_ceiling_from_context(ctx)
        args = {"topic": topic, "scope": "global", "version": revision}
        return ceiling is None or (
            presence_ceiling_allows_tool(ceiling, "knowledge_read")
            and apply_presence_argument_bindings(ctx, "knowledge_read", args) == args)
    except PresenceAuthorityError:
        return False


def _entry(note: knowledge.KnowledgeNote, matched: list[str], *, first: bool, ctx: Any) -> HeldNoteSection:
    topic, revision = note.address.topic, note.revision
    names = ", ".join(repr(name) for name in matched)
    label = (f"{_HEADING}\n\n" if first else "") + (
        f"### {note.title}\n\nGlobal note: memory/knowledge/{topic}.md\n"
        f"Matched claims: {names}\nRevision: {revision}\n")
    reader = f"knowledge_read(topic={topic!r}, scope='global', version={revision!r})"
    try:
        knowledge_versions.retain_held_version(note, revision)
        label += f"Exact reader: {reader}"
    except OSError:
        log.warning("Could not retain held note %s at %s", topic, revision, exc_info=True)
        reader = ""
        label += "Exact reader unavailable: this revision could not be retained."

    window_label = ""
    if topic == knowledge.OVERVIEW_TOPIC:
        full = label + "\nShared understanding reference."
    else:
        full = label + "\n\n" + note_head(note.source)
        if reader and _can_read(ctx, topic, revision):
            window_label = label
    return HeldNoteSection("held_note:" + topic, full, reader=reader, topic=topic, revision=revision,
                           window_label=window_label, summary=note.summary)


def turn_sections(root: Path, task: Mapping[str, Any], spec: Any, navigation: Mapping,
                  *, ctx: Any = None) -> list[HeldNoteSection]:
    """Select recorded global claims from the same inventory the knowledge index used."""
    if spec.role in DELEGATED_FOCUS_ROLES:
        return []
    inventory = navigation.get("inventory", ())
    if (captured := getattr(ctx, "_held_note_sections", None)) is not None:
        rows = {row["topic"]: row for row in inventory}
        return [HeldNoteSection(**entry).with_index(rows.get(entry["topic"], {}), navigation) for entry in captured]

    claims = knowledge.active_claims(root, inventory=inventory)
    matches: dict[knowledge.KnowledgeAddress, list[str]] = {}
    for name in situation_names(task, spec.room_id):
        for address in claims.get(name, ()):
            matches.setdefault(address, []).append(name)
    if not matches:
        return []

    rows = {row["topic"]: row for row in inventory}
    entries = []
    for address in sorted(matches, key=lambda item: item.topic):
        try:
            note = navigation.get("overview") if address.topic == knowledge.OVERVIEW_TOPIC else None
            if note is None:
                note = knowledge.read_knowledge_note(address)
        except (ValueError, OSError):
            continue  # the source changed or disappeared between navigation and capture
        if note.source is None or note.state != "active":
            continue
        written = set(knowledge_versions.claim_names(note.metadata.get("claims")))
        matched = [name for name in matches[address] if name in written]
        if matched:
            entry = _entry(note, matched, first=not entries, ctx=ctx)
            entries.append(entry.with_index(rows[address.topic], navigation))
    return entries

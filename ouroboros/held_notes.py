"""Captured heads of the global notes that claim the acting focus's situation, and of the notes their heads link."""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass, fields, replace
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from ouroboros import knowledge, knowledge_versions
from ouroboros.contracts.task_contract import task_input_sources
from ouroboros.markdown_source import MarkdownSource
from ouroboros.memory_view_events import situation_names
from ouroboros.section_floor import HostSection

log = logging.getLogger(__name__)
_HEADING = "## Held knowledge notes"
_HEAD_END = re.compile(rb"^## ", re.MULTILINE)
_INDEX_FIELDS = ("topic", "title", "state", "archive_error", "read_error", "parse_error")
CAPTURE_KEY = "held_note_capture"


@dataclass(frozen=True, eq=False, repr=False)
class HeldNoteSection(HostSection):
    """One entry within the held-notes section, with its captured window forms."""

    topic: str = ""
    revision: str = ""
    window_label: str = ""  # the label backed by a verified revision
    summary: str = ""
    outline_has_summary: bool = False
    index_has_summary: bool = False
    source_chars: int = 0
    index_row: Mapping[str, Any] | None = None
    reader_restriction: str = ""
    carried_by: tuple[str, ...] = ()  # the held heads linking a note no claim of its own holds here: it yields first

    def __new__(cls, section_id, full_text, *_args, **_kwargs):
        return super().__new__(cls, section_id, full_text)

    def __getnewargs__(self):
        return (*super().__getnewargs__(), self.topic, self.revision, self.window_label,
                self.summary, self.outline_has_summary, self.index_has_summary, self.source_chars, self.index_row,
                self.reader_restriction, self.carried_by)

    @classmethod
    def restored(cls, entry: Mapping, carried: Mapping) -> HeldNoteSection:
        """A saved entry, read by the fields this build knows (a capture a later build saved resumes here too), with
        its carriers from the entry, else from the map ``continuation_capture`` keeps beside the entries."""
        known = {item.name: entry[item.name] for item in fields(cls) if item.name in entry}
        return cls(**{**known, "carried_by": tuple(entry.get("carried_by") or carried.get(entry["section_id"], ()))})

    def with_index(self, row: Mapping, navigation: Mapping) -> HeldNoteSection:
        """Locate the authored summary in either captured form of the index."""
        same = row.get("summary") == self.summary and row.get("state") == "active"
        outline = navigation.get("outline", False) and same and knowledge.outline_keeps_summary(row.get("type"))
        resident = same if navigation.get("whole") else outline
        index_row = self.index_row
        if row:
            index_row = {key: row[key] for key in _INDEX_FIELDS if key in row}
            index_row["type"] = "person" if knowledge.outline_keeps_summary(row.get("type")) else None
            index_row["summary"] = row.get("summary") if isinstance(row.get("summary"), str) else None
        return replace(self, operative_view=self._summary_view(resident), outline_has_summary=outline,
                       index_has_summary=resident, index_row=index_row)

    def recheck_reader(self, ctx: Any, root: Path) -> HeldNoteSection:
        if not self.reader:
            return self
        restriction = _reader_restriction(ctx, root, self.topic, self.revision, self.source_chars)
        if restriction == self.reader_restriction:
            return self
        previous = _reader_line(self.reader, self.reader_restriction)
        prefix = (self.window_label.removesuffix(_reader_line(self.reader, "")) if self.window_label else
                  self.full_text.rsplit("\n", 1)[0].removesuffix(previous))  # the overview ends with one reference line
        return replace(self, full_text=prefix + _reader_line(self.reader, restriction) +
                       self.full_text[len(prefix) + len(previous):], operative_view=None,
                       reader_restriction=restriction)

    @property
    def available_window_label(self) -> str:
        return "" if self.reader_restriction else self.window_label

    def summary_is_resident(self, addressed: Mapping) -> bool:
        return self.outline_has_summary if "knowledge_index" in addressed else self.index_has_summary

    def _summary_view(self, resident: bool) -> str | None:
        if not (label := self.available_window_label) or not self.summary.strip():
            return None
        marker = "[Head shortened to its authored summary for the physical window.]"
        if resident:
            return f"{label}\n\n{marker} The summary is in the knowledge index."
        return f"{label}\n\n{self.summary}\n\n{marker}"

    def summary_saves(self, addressed: Mapping) -> bool:
        """Whether the summary form, as ``addressed`` places the summary, is shorter than the whole entry."""
        from ouroboros.memory_floor import view_tokens

        summary = self.view_for_receipt({}, addressed)
        return summary is not None and view_tokens(summary) < view_tokens(self.full_text)

    def view_for_receipt(self, row: Mapping, addressed: Mapping) -> str | None:
        if row.get("held_note_form") == "reader":  # a carried head no summary shortens takes it first (_held_note_steps)
            why = (" at the physical window's emergency floor" if not self.carried_by or self.summary_saves(addressed)
                   else ": its summary form saves no room" if self.summary.strip()
                   else " for the physical window: this note has no authored summary")
            return f"{label}\n\n[Head omitted{why}.]" if (label := self.available_window_label) else None
        return self._summary_view(self.outline_has_summary) if "knowledge_index" in addressed else self.operative_view

    def window_receipt(self, form: str) -> dict:
        return {"section_id": self.section_id, "held_note_form": form, "reader": self.reader,
                "source_ref": {"kind": "knowledge_version", "revision": self.revision,
                               "read": {"tool": "knowledge_read", "arguments": {
                                   "topic": self.topic, "scope": "global", "version": self.revision}}}}


def frame_key(mode: str, owner: str, window: int, record: Any, reserve: int, ratio: float, request_tokens: int) -> list:
    """The mode, its physical frame and the uncut request price used to fit it."""
    if record and record.usable is not None:
        window = replace(record, reserve=reserve).total_equivalent
    return [mode, owner, window, record.boundary if record else None, reserve, ratio, request_tokens]


def can_replay_frame(core: Any, saved: list, current: list, receipts: Mapping) -> bool:
    if saved != current or any(not part.available_window_label for part in core.dynamic_sections
                               if isinstance(part, HeldNoteSection) and part.section_id in receipts):
        return False
    if index_receipt := receipts.get("knowledge_index"):
        index = next(part for part in core.dynamic_head_sections if part.section_id == "knowledge_index")
        return index_receipt["source_ref"]["sha256"] == hashlib.sha256(index.full_text.encode("utf-8")).hexdigest()
    return True


def restored_capture(state: Mapping) -> dict:
    capture = state.get(CAPTURE_KEY) or {"entries": state.get("held_note_sections", [])}
    if capture["entries"]:
        return {**capture, "messages": state.get("messages"), "tool_schemas": state.get("tool_schemas")}
    return capture


def continuation_capture(plan: Any, mode: str | None) -> dict:
    """Carry the sent forms and their captured index and overview in the ordinary continuation. An entry holds only
    the fields a build since P1 restores, which refuses any other key; carriage stands beside the entries, only where
    a head is carried, so a work saved here still resumes after a Rollback."""
    from ouroboros.context_budget import context_mode_limits

    core = getattr(plan, "core", None)
    entries = [part for part in getattr(core, "dynamic_sections", ()) if isinstance(part, HeldNoteSection)]
    if not entries:
        return {}
    mode = mode if mode in {"max", "low", "nano"} else plan.initial_mode
    companions = {}
    if any(part.topic == knowledge.OVERVIEW_TOPIC for part in entries):
        companions["shared_understanding"] = asdict(core.dynamic_head_sections[0])
    _, reserve = context_mode_limits(mode, plan.preferred_mode, plan.output_reserve_tokens,
                                     working_reserve_tokens=plan.working_reserve_tokens)
    projection = plan.projection(mode)
    facts = projection.memory_facts
    request_tokens = facts.get("held_note_request_tokens")
    carried = {part.section_id: list(part.carried_by) for part in entries if part.carried_by}
    return {CAPTURE_KEY: {
        "entries": [{key: value for key, value in asdict(part).items() if key != "carried_by"} for part in entries],
        **({"carried": carried} if carried else {}), "companions": companions,
        "frame": frame_key(mode, plan.preferred_mode, plan.window_tokens, plan.window_record, reserve,
                           projection.calibration_ratio, request_tokens) if request_tokens is not None else None,
        "receipts": [row for row in facts.get("host_sections_addressed", ())
                     if row.get("held_note_form") or row["section_id"] == "knowledge_index"],
    }}


# Stands after the heads a helper holds, native or external; the reader line of a head an external session receives.
HELPER_NOTE = ("[Host: the held notes above are my working notes, my account and not evidence: apply their facts, but "
               "do not quote or copy them into code, commits, files or messages, and report any contradiction.]")
EXTERNAL_READER = "an external session cannot call knowledge_read."


def claimed_topics(root: Path, names: tuple[str, ...], inventory: tuple | None = None,
                   claims: Mapping | None = None) -> list[str]:
    """The topics of the active global notes whose recorded claims (``knowledge.active_claims``) name one of ``names``."""
    if names and claims is None:
        claims = knowledge.active_claims(root, inventory=inventory)
    return sorted({address.topic for name in names for address in (claims or {}).get(name, ())})


def carried_situation(ctx: Any, root: Path, *, declared: bool) -> tuple[dict, str]:
    """``({CARRIED_NAMES: names}, the parent's line)`` for a helper this work schedules; ``({}, "")`` on an install
    that records no claim, where the helper's record stays as it was. The line names the notes that claim the names
    now: a queued helper reads their heads, at their revisions then, in its own start."""
    from ouroboros.memory_view_events import CARRIED_NAMES, helper_names

    try:
        claims = knowledge.active_claims(root)
        names = helper_names(ctx, root) if claims else ()
    except (OSError, ValueError):  # an unreadable shelf or registry schedules the helper as before, without names
        log.warning("Could not read the situation names a scheduled helper would carry", exc_info=True)
        return {}, ""
    if not names:
        return {}, ""
    holds = ("its input is declared, so it holds no head" if declared else
             "it holds the heads of the notes that claim them and of the notes their heads link, read at its start")
    topics = ", ".join(claimed_topics(root, names, claims=claims)) or "none"
    return {CARRIED_NAMES: list(names)}, (f"\nheld notes: the child carries my situation names ({', '.join(names)}); "
                                          f"{holds} (claiming now: {topics}).")


def sent_heads(entries: list[HeldNoteSection]) -> list[dict]:
    """Each head an external session receives whole (its reader ``EXTERNAL_READER``), by address and revision, a
    zero-character one (a note that starts with ``## ``) included; the overview stands as a reference, no head."""
    line, heads = _reader_line("", EXTERNAL_READER) + "\n\n", []
    for part in entries:
        _label, after_reader, head = part.full_text.partition(line)
        if after_reader:
            heads.append({"topic": part.topic, "revision": part.revision, "chars": len(head), "text": head})
    return heads


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


def linked_heads(note: knowledge.KnowledgeNote) -> tuple[dict[str, knowledge.KnowledgeNote], list[str]]:
    """The active global notes this note's head links, by topic, and its head's other links to a knowledge address,
    each named with why it holds nothing. The head is ``note_head``'s: a link below its first `## ` line, in a fenced
    block or in a code span is not in it (``MarkdownSource.links``); a link to the note itself, or to the overview,
    which stands in its own place, is neither."""
    end = note.source.body_span.start_byte + len(note_head(note.source).encode("utf-8"))
    carried, unheld = {}, {}
    for link in knowledge.knowledge_links(note, before_byte=end):
        where = link.get("address")
        if where is None or (where["scope"], where["topic"]) in {(note.address.scope, note.address.topic),
                                                                  ("global", knowledge.OVERVIEW_TOPIC)}:
            continue
        if where["scope"] != "global":
            why = "project shelf"
        elif link["status"] != "present":
            why = "no such note"
        else:
            try:
                target = knowledge.read_knowledge_note(replace(note.address, topic=where["topic"]))
                why = "malformed" if target.source is None else "archived" if target.state != "active" else ""
            except (OSError, ValueError):
                why = "unreadable"
            if not why:
                carried.setdefault(where["topic"], target)
                continue
        unheld.setdefault(f"{Path(where['path']).relative_to(where['canonical_root']).as_posix()} ({why})", None)
    return carried, list(unheld)


def standing_heads(root: Path, names, *, written: knowledge.KnowledgeNote | None = None,
                   holding: dict[str, dict[str, list[str]]] | None = None) -> dict[str, tuple[int, int]]:
    """``(heads, host tokens)`` standing whole wherever each name is present: the head of every active global note
    whose own recorded claims hold that name, and of every one a held head links where that head's claims hold it
    (``standing``, the selection ``turn_sections`` makes), each note once; the overview, which stands in its own
    place, aside. ``written``, a note just written, counts from its own bytes; ``holding`` is ``standing(root)``
    when the caller already read it. A write receipt's measure (``memory_floor.view_tokens``, as the held heads fact
    counts); no start reads it."""
    from ouroboros.memory_floor import view_tokens

    shelf, sizes = knowledge.resolve_knowledge_address(root, knowledge.OVERVIEW_TOPIC, "global"), {}
    holding = standing(root) if holding is None else holding

    def size(topic: str) -> int | None:
        if topic not in sizes:
            try:
                note = (written if written is not None and written.address.topic == topic
                        else knowledge.read_knowledge_note(replace(shelf, topic=topic)))
            except (ValueError, OSError):
                note = None  # the source changed or disappeared since the shelf was read
            sizes[topic] = (view_tokens(note_head(note.source))
                            if note is not None and note.source is not None and note.state == "active" else None)
        return sizes[topic]

    found = {}
    for name in names:
        held = [tokens for topic, holders in holding.items() if topic != knowledge.OVERVIEW_TOPIC
                and any(name in held_by for held_by in holders.values()) and (tokens := size(topic)) is not None]
        found[name] = (len(held), sum(held))
    return found


def _reader_restriction(ctx: Any, root: Path, topic: str, revision: str, length: int) -> str:
    from ouroboros.tool_access_paths import workspace_mode_block_reason
    from ouroboros.tools.knowledge import _address, _knowledge_read, _range
    from ouroboros.tools.registry_core import _presence_bound_args
    from ouroboros.tools.registry_guards import _capability_resource_guard_result

    if unavailable := getattr(ctx, "held_reader_unavailable", ""):  # an actor without the tool (``EXTERNAL_READER``)
        return unavailable
    args = {"topic": topic, "scope": "global", "version": revision}
    bound, refusal = _presence_bound_args(ctx, "knowledge_read", args)
    if refusal:
        return refusal
    if denied := _capability_resource_guard_result(ctx, "knowledge_read", bound):
        return denied.text
    try:
        if problem := workspace_mode_block_reason(ctx):
            return f"Workspace restriction ({problem})."
        call = signature(_knowledge_read).bind(ctx, **bound)
        call.apply_defaults()
        effective = call.arguments
        address = _address(ctx, effective["topic"], effective["scope"])
        start, end, _ = _range(length, effective["start_char"], effective["end_char"])
        if (address.canonical_root == root.resolve() and address.scope == "global" and address.topic == topic
                and effective["version"] == revision and effective["view"] in ("note", None, "")
                and effective["fingerprint"] in (None, "") and (start, end) == (0, length)):
            return ""
    except (OSError, TypeError, ValueError):
        pass
    return "Presence argument bindings do not request the complete note at this address and revision."


def _reader_line(reader: str, restriction: str) -> str:
    return f"Exact reader unavailable: {restriction}" if restriction else f"Exact reader: {reader}"


def _label_title(note: knowledge.KnowledgeNote, head: str) -> str:
    """The YAML title, else an H1 the head itself carries, else the topic: never a heading below the head."""
    value = note.metadata.get("title")
    if isinstance(value, str) and value.strip():
        return value
    start = note.source.body_span.start_byte
    end = start + len(head.encode("utf-8"))
    return next((heading.title for heading in note.source.headings
                 if heading.level == 1 and start <= heading.span.start_byte < end), note.address.topic)


def _reasons(carriers: tuple, unheld: tuple) -> str:
    """The label lines naming each held head that links this note, and under a carrier its head links held nowhere."""
    return "".join(f"Linked from the head of {title} (memory/knowledge/{topic}.md), held by "
                   + ", ".join(repr(name) for name in names) + "\n" for topic, title, names in carriers) + (
        f"Linked from this head, not held: {', '.join(unheld)}\n" if unheld else "")


def _entry(note: knowledge.KnowledgeNote, matched: list[str], *, first: bool, ctx: Any,
           carriers: tuple = (), unheld: tuple = ()) -> HeldNoteSection:
    topic, revision, head = note.address.topic, note.revision, note_head(note.source)
    names = ", ".join(repr(name) for name in matched)
    label = (f"{_HEADING}\n\n" if first else "") + (
        f"### {_label_title(note, head)}\n\nGlobal note: memory/knowledge/{topic}.md\n"
        + (f"Matched claims: {names}\n" if matched else "") + _reasons(carriers, unheld) + f"Revision: {revision}\n")
    reader = f"knowledge_read(topic={topic!r}, scope='global', version={revision!r})"
    length = len(note.text)
    restriction = _reader_restriction(ctx, note.address.canonical_root, topic, revision, length)
    if not restriction:
        try:
            knowledge_versions.retain_held_version(note, revision)
        except OSError:
            log.warning("Could not retain held note %s at %s", topic, revision, exc_info=True)
            restriction = "This revision could not be retained and is unavailable in the journal."
    if restriction:
        reader = ""
    label += _reader_line(reader, restriction)

    window_label = ""
    if topic == knowledge.OVERVIEW_TOPIC:
        full = label + "\nShared understanding reference."
    else:
        full = label + "\n\n" + head
        if reader:
            window_label = label
    return HeldNoteSection("held_note:" + topic, full, reader=reader, topic=topic, revision=revision,
                           window_label=window_label, summary=note.summary, source_chars=length,
                           carried_by=() if matched else tuple(carrier for carrier, _title, _names in carriers))


def turn_sections(root: Path, task: Mapping[str, Any], spec: Any, navigation: Mapping,
                  *, ctx: Any = None) -> list[HeldNoteSection]:
    """Select recorded global claims from the same inventory the knowledge index used; a declared input holds none."""
    if task_input_sources(task) == "declared":
        return []
    inventory = navigation.get("inventory", ())
    if (captured := getattr(ctx, "_held_note_capture", None)) is not None:
        if not captured["entries"]:
            return []
        rows, carried = {row["topic"]: row for row in inventory}, captured.get("carried") or {}
        return [HeldNoteSection.restored(entry, carried).recheck_reader(ctx, root)
                .with_index(rows.get(entry["topic"], {}), navigation) for entry in captured["entries"]]

    claims = knowledge.active_claims(root, inventory=inventory)
    matches: dict[knowledge.KnowledgeAddress, list[str]] = {}
    for name in navigation["names"] if "names" in navigation else situation_names(task, spec.room_id):
        for address in claims.get(name, ()):
            matches.setdefault(address, []).append(name)
    if not matches:
        return []

    if ctx is None:
        from ouroboros.contracts.task_contract import build_task_contract

        ctx = SimpleNamespace(drive_root=root, task_metadata=task.get("metadata") or task,
                              task_contract=build_task_contract(dict(task)), project_id=task.get("project_id"))
    rows = {row["topic"]: row for row in inventory}
    bodies = navigation.get("bodies", {})
    held = []
    for address in sorted(matches, key=lambda item: item.topic):
        try:
            note = navigation.get("overview") if address.topic == knowledge.OVERVIEW_TOPIC else None
            if note is None:
                note = knowledge.read_knowledge_note(address, captured=bodies.get(address))
        except (ValueError, OSError):
            continue  # the source changed or disappeared between navigation and capture
        if note.source is None or note.state != "active":
            continue
        written = set(knowledge_versions.claim_names(note.metadata.get("claims")))
        matched = [name for name in matches[address] if name in written]
        if note.revision != rows[address.topic].get("revision"):
            current = knowledge.active_claims(root, inventory=(knowledge.note_descriptor(note),))
            matched = [name for name in matched if address in current.get(name, ())]
        if matched:
            held.append((note, matched))
    return _with_links(held, rows, navigation, ctx)


def _with_links(held: list, rows: Mapping, navigation: Mapping, ctx: Any) -> list[HeldNoteSection]:
    """The claimed heads in address order, then the active global notes their heads link, one hop: a linked head links
    nothing further and ``overview`` carries nothing. A note stands once and its label names every reason."""
    carriers, linked, unheld = {}, {}, {}
    for carrier, matched in held:
        if carrier.address.topic == knowledge.OVERVIEW_TOPIC:
            continue
        targets, unheld[carrier.address.topic] = linked_heads(carrier)
        reason = (carrier.address.topic, _label_title(carrier, note_head(carrier.source)), matched)
        for topic, note in targets.items():
            carriers.setdefault(topic, []).append(reason)
            linked.setdefault(topic, note)
    entries = []
    for note, matched in held:
        topic = note.address.topic
        entry = _entry(note, matched, first=not entries, ctx=ctx, carriers=tuple(carriers.pop(topic, ())),
                       unheld=tuple(unheld.get(topic, ())))
        entries.append(entry.with_index(rows[topic], navigation))
    for topic in sorted(carriers):
        entry = _entry(linked[topic], [], first=not entries, ctx=ctx, carriers=tuple(carriers[topic]))
        entries.append(entry.with_index(rows.get(topic, {}), navigation))
    return entries


def standing(root: Path) -> dict[str, dict[str, list[str]]]:
    """Every global note held somewhere, to what holds it: under ``""`` its own recorded claims, under each carrier's
    topic the claims of that active held note whose head links it (``linked_heads``). Reads the shelf as it stands."""
    claimed: dict[knowledge.KnowledgeAddress, list[str]] = {}
    for name, addresses in knowledge.active_claims(root).items():
        for address in addresses:
            claimed.setdefault(address, []).append(name)
    held = {address.topic: {"": names} for address, names in claimed.items()}
    for address, names in claimed.items():
        try:
            carrier = knowledge.read_knowledge_note(address)
        except (OSError, ValueError):
            continue
        if carrier.source is not None and carrier.state == "active" and address.topic != knowledge.OVERVIEW_TOPIC:
            for topic in linked_heads(carrier)[0]:
                held.setdefault(topic, {})[address.topic] = names
    return held

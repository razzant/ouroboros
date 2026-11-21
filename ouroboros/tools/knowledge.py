"""Knowledge tool adapters over the common linked Markdown source owner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import List

from ouroboros import knowledge as knowledge_store
from ouroboros import knowledge_versions
from ouroboros.project_facts import placed_project_id
from ouroboros.tool_capabilities import tool_result_limit
from ouroboros.tool_access_paths import canonical_data_root
from ouroboros.knowledge import INDEX_FILE as INDEX_FILE, OVERVIEW_TOPIC as OVERVIEW_TOPIC
from ouroboros.knowledge import sanitize_topic as _sanitize_topic
from ouroboros.knowledge_edits import check_arguments, mode_received, normalize_arguments
from ouroboros.memory_view_events import SituationReceipt, receipt_situation
from ouroboros.tools.arg_feedback import ignored_argument_note
from ouroboros.tools.registry import ToolEntry, ToolContext
from ouroboros.tools.tool_result import ToolResult, _MAX_META_BYTES, _publish_tool_result, completed_local_read
from ouroboros.tools.tool_result import publish_no_effect

KNOWLEDGE_DIR = "memory/knowledge"
_MODES = ("overwrite", "append", "edit", "revert", "archive", "restore")
BACKLOG_TOPIC = "improvement-backlog"
# Topics with exactly one home. Whichever room asks for them, they resolve to the
# global shelf: the overview is the shared orientation every integrating context
# loads; patterns and improvement-backlog are ordinary notes kept at their old
# global address, so an earlier call from a project room reaches the existing
# note instead of starting a second one.
GLOBAL_ONLY_TOPICS = knowledge_store.GLOBAL_ADDRESS_TOPICS


def _address(ctx: ToolContext, topic: str, scope: str = "") -> knowledge_store.KnowledgeAddress:
    # One root for every shelf: the canonical data root the scratchpad and the room
    # binding (placed_project_id) already use, never the ambient DATA_DIR.
    if not isinstance(scope, str):
        raise ValueError("scope must be global or project:<exact project id>")
    sanitized = _sanitize_topic(topic)
    if sanitized in GLOBAL_ONLY_TOPICS:
        return knowledge_store.resolve_knowledge_address(canonical_data_root(ctx), sanitized, "global")
    return knowledge_store.resolve_knowledge_address(
        canonical_data_root(ctx), sanitized, scope, str(placed_project_id(ctx) or ""))


def _range(total: int, start_char: int | None, end_char: int | None,
           noun: str = "note") -> tuple[int, int, list[str]]:
    """One exact character range of a ``total``-character note or view; a bound that asks for nothing is not an error.

    A missing bound is its own edge, ``(0, 0)`` selects nothing and reads all of it,
    and an ``end_char`` past the end is lowered to it (like ``read_file``).
    Each is said in the header; the returned range is always the range delivered.
    """
    if any(bound is not None and type(bound) is not int for bound in (start_char, end_char)):
        raise ValueError(f"Knowledge source range start_char={start_char!r}, end_char={end_char!r} "
                         f"must be integers; this {noun} has complete_chars={total}")
    start, end = (0 if start_char is None else start_char), (total if end_char is None else end_char)
    notes = []
    if (start, end) == (0, 0) and total:
        notes.append(ignored_argument_note("end_char", 0, f"a 0..0 range selects nothing; returned the whole {noun}"))
        end = total
    elif end > total:
        notes.append(ignored_argument_note("end_char", end, f"the {noun} ends at {total}; returned {start}..{total}"))
        end = total
    if not 0 <= start <= end:
        raise ValueError(f"Knowledge source range start_char={start_char!r}, end_char={end_char!r} is outside "
                         f"this {noun}: complete_chars={total}; use 0 <= start_char <= end_char")
    return start, end, notes


def _argument_notes(notes) -> str:
    return "".join(f"[Argument note] {item}\n" for item in notes)


def _source_view(note: knowledge_store.KnowledgeNote, start_char: int | None = None,
                 end_char: int | None = None, earlier: dict | None = None, notes=(), shelves=(),
                 situation: str = "", *, version: str | None = None) -> tuple[str, dict]:
    """One exact character range of the note (``_range``), or of an ``earlier`` version of it."""
    ref = note.source_ref()
    ref["state"] = note.state
    text = note.text
    total = len(text)
    start, end, range_notes = _range(total, start_char, end_char)
    if version is not None:
        ref["read"]["arguments"]["version"] = version
    if earlier is not None:
        ref["earlier_version"] = earlier
    ref.update(start_char=start, end_char=end, complete_chars=total)
    body = text[start:end]
    header = "[Knowledge source] " + json.dumps(ref, ensure_ascii=False, sort_keys=True) + "\n"
    header += "".join(f"[Range note] {item}\n" for item in range_notes) + _argument_notes(notes)
    if earlier is not None:
        written = knowledge_versions.written_text(earlier)
        current = earlier["current_revision"]
        where = (f"The current revision is {current}." if current else
                 f"The live note is {earlier['live_status']} at this address.")
        header += f"[Earlier version, not the current note: {written}. {where}]\n"
    if note.parse_error:
        header += "[Metadata unavailable; the complete original Markdown follows.]\n"
    if note.archive_error:
        header += f"[Archive metadata invalid; kept active: {note.archive_error}.]\n"
    if shelves and not start_char:  # link facts once, on a read from the note's start, not on every range page
        links = {}
        try:
            outgoing = knowledge_store.knowledge_links(note)
        except (ValueError, OSError) as exc:
            links["unavailable"] = type(exc).__name__
            outgoing = ()
        for link in outgoing:
            status = link["status"]
            if status == "external":
                links[status] = links.get(status, 0) + 1
            elif status == "unavailable":
                links.setdefault(status, link["error"])
            else:
                target = link.get("address", note.address.as_dict() if status == "same_source" else {})
                links.setdefault(status, []).append(
                    {key: target[key] for key in ("scope", "topic")} if target else link["target"])
        if links:
            header += "[Knowledge links] " + json.dumps(links, ensure_ascii=False) + "\n"
        backlinks = knowledge_store.knowledge_backlinks(note, shelves)
        if backlinks:
            header += "[Knowledge backlinks (readable shelves now)] " + json.dumps(backlinks, ensure_ascii=False) + "\n"
    header += (situation + "\n" if situation else "") + "\n"
    return header + body, {
        "knowledge_source": ref, "knowledge_body_start": len(header),
        "knowledge_body_chars": len(body), "knowledge_source_complete": start == 0 and end == total,
    }


def _rendered_view(view: str, text: str, facts: dict, start_char: int | None, end_char: int | None,
                   notes=(), situation: str = "") -> tuple[str, dict]:
    """An outline or history listing, paged by the same character ranges as a note."""
    start, end, range_notes = _range(len(text), start_char, end_char, "view")
    head = {**facts, "view": view, "start_char": start, "end_char": end, "complete_chars": len(text)}
    header = (f"[Knowledge {view}] " + json.dumps(head, ensure_ascii=False, sort_keys=True) + "\n"
              + "".join(f"[Range note] {item}\n" for item in range_notes) + _argument_notes(notes)
              + (situation + "\n" if situation else "") + "\n")
    return header + text[start:end], {"knowledge_view": head, "knowledge_body_start": len(header),
                                      "knowledge_body_chars": end - start}


class ListChanged(Exception):
    """A history page's continuation named another list than the one rendered now (a write landed since)."""


def _note_view(address: knowledge_store.KnowledgeAddress, view: str, version: str | None,
               start_char: int | None, end_char: int | None, notes=(), fingerprint: str = "", shelves=(),
               situation: str = "") -> tuple[str, dict]:
    if view not in ("note", "outline", "history"):
        raise ValueError(f"view={view!r}: use note (default), outline or history")
    if fingerprint and view != "history":
        raise ValueError("fingerprint continues a list of versions (view='history'), as its first page names it")
    if version is not None and view == "history":
        raise ValueError("version selects one version for view=note or view=outline; view=history lists them all")
    requested = address
    if version is not None or view == "history":
        address = knowledge_store.shelf_spelling(address)
    live_status = "readable"
    try:
        note = knowledge_store.read_knowledge_note(address)
    except (OSError, UnicodeDecodeError) as exc:
        if version is None:
            raise
        note = None
        live_status = "absent" if isinstance(exc, FileNotFoundError) else "unreadable"
    current_revision = note.revision if note is not None else None
    shown, earlier = note, None
    if version is not None:
        if version == current_revision:
            if requested != address:
                try:
                    pin, _, _ = knowledge_versions.find_note_version(address, requested, version, live=None)
                except knowledge_versions.VersionUnavailable:
                    pass  # The live text can still supply a head with no recoverable journal entry.
                else:
                    shown = replace(note, address=pin)
        else:
            live = note.text if note is not None else None
            pin, text, row = knowledge_versions.find_note_version(address, requested, version, live=live)
            shown = knowledge_store._note(pin, text.encode("utf-8"))
            earlier = {**knowledge_versions.row_facts(row), "revision": version, "current_revision": current_revision,
                       **({"live_status": live_status} if note is None else {})}
    if view == "note":
        return _source_view(shown, start_char, end_char, earlier, notes, shelves, situation, version=version)
    topic = requested.topic if view == "history" else shown.address.topic
    facts = {"topic": topic, "scope": shown.address.scope, "revision": shown.revision,
             **({"current_revision": current_revision, "earlier_version": earlier} if earlier else {})}
    if view == "history":  # newest first: a later page continues the list its first page read, by its fingerprint
        for name, bound in (("start_char", start_char), ("end_char", end_char)):  # an argument error, not a list's
            if bound is not None and (type(bound) is not int or bound < 0):
                raise ValueError(f"{name}={bound!r} is not a position in the list: a non-negative integer")
        listing = knowledge_versions.render_history(note, requested)
        digest = hashlib.sha256(listing.encode("utf-8")).hexdigest()[:16]
        if (start_char or fingerprint) and fingerprint != digest:
            raise ListChanged("that page belongs to another list of versions (a write landed since, or no fingerprint "
                              "named the list it continues): read it again from start_char=0" if start_char else
                              "the fingerprint names another list of versions (a write landed since): a first page "
                              "takes none, so read it without the fingerprint")
        return _rendered_view(view, listing, {**facts, "fingerprint": digest}, start_char, end_char, notes, situation)
    label = ""
    if earlier is not None:
        current = f"current revision {current_revision}" if current_revision else f"live note {live_status} at this address"
        label = f" — an earlier version, not the current note ({current})"
    if version is not None:
        facts["read"] = {"tool": "knowledge_read", "arguments": {
            "topic": shown.address.topic, "scope": shown.address.scope, "version": version}}
    return _rendered_view(view, knowledge_versions.render_outline(shown, label, version=version), facts,
                          start_char, end_char, notes, situation)


def _line(situation: SituationReceipt | None) -> str:
    return situation.line if situation is not None else ""


def _with_lines(text: str, *lines: str) -> str:
    return "\n".join((text, *(line for line in lines if line)))


def _knowledge_read(ctx: ToolContext, topic: str, scope: str = "",
                    start_char: int | None = None, end_char: int | None = None,
                    view: str = "note", version: str | None = None, fingerprint: str = "") -> str:
    notes = []  # a value that names the default takes it, said in the header (tools/arg_feedback)
    if view in (None, ""):
        notes.append(ignored_argument_note("view", view, "the default view, the note itself"))
        view = "note"
    if isinstance(version, str) and not version.strip():  # empty or whitespace: a blank value names no revision
        notes.append(ignored_argument_note("version", version, "no revision named; this is the current note"))
        version = None
    situation = _line(receipt_situation(ctx, canonical_data_root(ctx)))
    try:
        sanitized = _sanitize_topic(topic)
        address = _address(ctx, sanitized, scope)
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise ValueError("fingerprint is the text a history page's header names")
        shelves = (address,
                   knowledge_store.resolve_knowledge_address(address.canonical_root, sanitized, "global"),
                   knowledge_store.resolve_knowledge_address(address.canonical_root, sanitized,
                                                             project_id=str(placed_project_id(ctx) or ""))) if view == "note" else ()
        text, meta = _note_view(address, view, version, start_char, end_char, notes, fingerprint or "", shelves, situation)
    except ListChanged as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_REPORTED_FAILURE", text=_with_lines(f"⚠️ TOOL_ERROR: position_mismatch: {exc}", situation),
            meta={"operation_outcome": "completed_no_effect"}))
    except knowledge_versions.VersionUnavailable as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_REPORTED_FAILURE", text=_with_lines(f"⚠️ TOOL_ERROR: version_unavailable: {exc}", situation),
            meta={"operation_outcome": "completed_no_effect"}))
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=_with_lines(f"⚠️ TOOL_ARG_ERROR: {exc}", situation),
            meta={"operation_outcome": "completed_no_effect"}))
    except FileNotFoundError:
        elsewhere = ("" if address.scope == "global"
                     else f" A global note may exist: knowledge_read(topic={sanitized!r}, scope='global').")
        return _publish_tool_result(ctx, ToolResult(
            status="ok", code="LEGACY_WARNING", text=_with_lines(
                f"Topic {topic!r} not found in {address.scope}. Use knowledge_list to see available topics." + elsewhere,
                situation),
            meta={"knowledge_address": address.as_dict(), "knowledge_missing": True,
                  "operation_outcome": "completed_no_effect"}))
    except (OSError, UnicodeDecodeError) as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_REPORTED_FAILURE", text=_with_lines(
                f"⚠️ TOOL_ERROR: Knowledge source unavailable: {type(exc).__name__}", situation),
            meta={"operation_outcome": "completed_no_effect"}))
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=text, meta={**meta, "operation_outcome": "completed_no_effect"}))


def _meta_bytes(meta: dict) -> int:
    return len(json.dumps(meta, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _omitted(text: str) -> dict:
    return {"omitted": True, "chars": len(text), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


_DELTA_LISTS = ("removed_headings", "added_headings", "removed_sections")


def _bound_delta_meta(meta: dict) -> dict:
    """Keep the tool receipt inside its existing metadata limit and return the host meta; history holds the full delta.

    Over the limit, the largest of what can give way goes first, until the receipt fits: a delta
    list (``_DELTA_LISTS``; an empty one has nothing to give) becomes its count, an omission flag and
    its sha256, and one ``omitted_read`` names for them all the ``view='history'`` read of the note
    ``knowledge_source`` names; the bare append's outline and the replaced version's reason become
    their size, sha256 and the read that shows them. A transport bound on the receipt, never a
    judgment of what matters: a small list is not dropped while a larger one overflows. Still over
    with nothing left to give (each call repeats the topic, 6 or 12 bytes a character outside ASCII),
    the host meta keeps only the write's reason, state and revision, the captured previous source of
    an archive or restore, and the receipt's size and sha256: a ✅ reply carries the receipt as its
    JSON line, so a landed write is never reported as failed.
    """
    arguments = ((meta.get("knowledge_source") or {}).get("read") or {}).get("arguments", {})

    def read(view: str) -> dict:
        return {"tool": "knowledge_read", "arguments": {**arguments, "view": view}}

    while _meta_bytes(meta) > _MAX_META_BYTES:
        delta, previous = meta.get("knowledge_delta") or {}, meta.get("knowledge_previous")
        options = [(field, delta[field]) for field in _DELTA_LISTS
                   if isinstance(delta.get(field), list) and delta[field]]
        if isinstance(meta.get("knowledge_outline"), str):
            options.append(("outline", meta["knowledge_outline"]))
        if isinstance(previous, dict) and isinstance(previous.get("reason"), str):
            options.append(("reason", previous["reason"]))
        if not options:
            return {**{key: meta[key] for key in ("knowledge_write_reason", "knowledge_state",
                                                  "knowledge_previous_source") if key in meta},
                    "knowledge_revision": (meta.get("knowledge_source") or {}).get("revision"),
                    "knowledge_receipt": _omitted(json.dumps(meta, ensure_ascii=False, sort_keys=True))}
        field, value = max(options, key=lambda option: len(json.dumps(option[1], ensure_ascii=True)))
        if field == "outline":
            meta["knowledge_outline"] = {**_omitted(value), "read": read("outline")}
        elif field == "reason":
            meta["knowledge_previous"] = {**previous, "reason": {**_omitted(value), "read": read("history")}}
        else:
            meta["knowledge_delta"] = {**delta, field: [], f"{field}_count": len(value), f"{field}_omitted": True,
                                       f"{field}_sha256": hashlib.sha256(
                                           json.dumps(value, ensure_ascii=False).encode("utf-8")).hexdigest(),
                                       "omitted_read": {"tool": "knowledge_read", "view": "history"}}
    return meta


def _residency(note: knowledge_store.KnowledgeNote, delta: dict | None = None) -> str:
    """Where the written note now sits in my contexts: a fact about its cost, never a limit."""
    if note.state == "archived":
        return "Residency: archived, so it is not in the active knowledge index; knowledge_read still opens it."
    body = note.source.text_at(note.source.body_span) if note.source else note.text
    if note.address.scope == "global" and note.address.topic == OVERVIEW_TOPIC:
        before = (delta or {}).get('old_body_chars', len(body))
        after = len(body)
        return (knowledge_store.overview_residency(note)
                + f" Body {before} → {after} chars; estimate {before / 4:.2f} → {after / 4:.2f} tokens; "
                f"delta {(after - before) / 4:+.2f} tokens (chars/4). "
                "Effect applies at the next context build; an accepted session keeps its earlier capture. "
                "Current route room: unknown here; compact_context inspects its actual observation.")
    project = f"the {note.address.scope} knowledge index, loaded in that project's contexts"
    tail = f"the body ({len(body)} chars) is read on demand with knowledge_read."
    if not (isinstance(note.summary, str) and note.summary.strip()):
        where = "the global knowledge index" if note.address.scope == "global" else project + ","
        return f"Residency: it has no summary, so {where} shows its address and title, and its whole view marks it (no summary); {tail}"
    loaders = ("Main, tasks, wakes and Presence, and the outline a tight window, a helper and a session start with"
               if knowledge_store.outline_keeps_summary(note.metadata.get("type"))
               else "Main, tasks, wakes and Presence; a tight window and a helper's or session's start show only its address and title "
                    "(a note typed person or person-… keeps its summary there)")
    where = f"the global knowledge index ({loaders})" if note.address.scope == "global" else project
    return (f"Residency: its summary ({len(note.summary)} chars) is shown in {where} from each context's next build; "
            f"works already running keep the view they started with; {tail}")


def _capture_previous(ctx: ToolContext, note: knowledge_store.KnowledgeNote) -> dict:
    """Expose exact pre-transition bytes through existing actor source custody.

    Project history is not readable via runtime_data file tools. The ordinary
    artifact reader works on both shelves. Publish to the actor's drive and the
    canonical drive before terminal copyback. The writer calls this under its
    source lock, before publication; durable shelf history remains authoritative.
    """
    from ouroboros.artifacts import persist_exact_text_source

    # read_file's text ABI normalizes newlines. JSON escapes retain the exact
    # Markdown (CRLF included) through that reader; decode old_content as UTF-8.
    source = json.dumps({"old_content": note.text, "revision": note.revision}, ensure_ascii=True)
    for root in dict.fromkeys((Path(ctx.drive_root).resolve(strict=False), canonical_data_root(ctx))):
        _text, ref, issue = persist_exact_text_source(
            root, ctx.task_id, source_id="knowledge-previous", text=source)
        if issue:
            raise OSError(issue["reason"])
    return {**ref, "format": "json", "field": "old_content"}


def _mode_refusal(ctx: ToolContext, mode: object) -> str:
    """A write names its operation; without an exact mode no note is addressed or written.

    An omitted mode once meant overwrite, so a call meant to add one section replaced the whole note.
    Null and blank text say what omission says (``tools/arg_feedback``), now refused like it; any other
    value is refused as received, never trimmed, case-folded or matched to the nearest operation.
    """
    text = (
        f"⚠️ TOOL_ARG_ERROR: {mode_received(mode)}. knowledge_write has no default operation: pass mode as exactly "
        "one of "
        "overwrite (content replaces the whole body), append (content goes at the end), edit (old_str with "
        "content, or edits, replaces exact text; summary may go alone), revert (puts back the earlier version "
        "named by version), archive or restore (the archive state only; archive requires reason). version names "
        "an earlier version to put back; expected_revision is the current revision a replacing write checks. "
        "Nothing was written.")
    situation = _line(receipt_situation(ctx, canonical_data_root(ctx)))
    return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=_with_lines(text, situation)))


def _holds_names(result: knowledge_store.KnowledgeWriteResult) -> bool:
    """The written note holds names: an active global note whose recorded names are not empty after the write. The
    overview is not one here: it stands in its own place, named in the held section, not copied."""
    note, claims = result.current, result.claims
    return bool(note is not None and claims is not None and claims.active and note.state == "active"
                and note.source is not None and note.address.topic != OVERVIEW_TOPIC)


# Where a head yields to the physical window (``memory_floor._held_note_steps``), keyed by how the present situations
# holding the note hold it: by its own names (True) it yields last; only by a link (False), before every claimed head.
_YIELDS = {
    frozenset({True}): "A held head takes its space ahead of the memory of other rooms and yields to the window last.",
    frozenset({True, False}): ("A held head takes its space ahead of the memory of other rooms and yields to the window "
                               "last; where only a link holds it, before every claimed head."),
    frozenset({False}): ("A head a held head links takes its space ahead of the memory of other rooms and yields to the "
                         "window before every claimed head."),
}


def _claim_lines(result: knowledge_store.KnowledgeWriteResult, situation: SituationReceipt, root: Path,
                 holding: dict[str, dict[str, list[str]]]) -> tuple[list[str], frozenset[bool]]:
    """What a write says of where this note stands: on a write that sets or changes names, what became of them
    (ignored entries, names no present situation matches or whose evidence was unreadable, a removed name); on every
    write of a note that holds names (``_holds_names``) or that a held head links (``holding``,
    ``held_notes.standing``), one line naming each present situation where it stands, by its own names or carried,
    with the heads standing there and their tokens (``held_notes.standing_heads``), situations alike in both and in
    why named together. Beside the lines, how each present situation holding the head holds it (``_YIELDS``): empty
    where none does, so the head's size and where it yields are said only where one does."""
    from ouroboros.held_notes import standing_heads

    claims, note, lines, heads, own = result.claims, result.current, [], {}, []
    matching, unverified = situation.matching(claims.written) if claims is not None else (set(), set())
    if claims is not None and claims.report:
        if claims.ignored:
            lines.append("Ignored claims entries (expected a list of strings): " + ", ".join(claims.ignored))
        names = dict.fromkeys(claims.written)
        if unmatched := [repr(name) for name in names if name not in matching and name not in unverified]:
            lines.append("Claims with no matching situation found: " + ", ".join(unmatched))
        if unverified:
            lines.append("Claims with unverified situations: " + ", ".join(repr(name) for name in names if name in unverified))
    if note.address.scope != "global":
        lines.append("Claims on a project-shelf note hold nowhere.")
    else:
        active = claims.active if claims is not None and note.state == "active" else []
        carriers = [(by, names) for by, names in sorted(holding.get(note.address.topic, {}).items()) if by]
        reached, doubtful = situation.matching([name for _, names in carriers for name in names]) if carriers else ((), ())
        own = [name for name in active if name in matching]  # own names first, then each carrier's, in that order
        where = list(dict.fromkeys([*own, *(name for _, names in carriers for name in names if name in reached)]))
        if where and note.state == "active" and note.source is not None and note.address.topic != OVERVIEW_TOPIC:
            heads = standing_heads(root, where, written=note, holding=holding)  # the overview stands only by name
        alike: dict[tuple, list[str]] = {}  # situations holding the same heads for the same reasons are named together
        for name in where:
            reasons = (name in own, tuple(by for by, names in carriers if name in names))
            alike.setdefault((heads.get(name), *reasons), []).append(name)
        parts = []
        for (total, mine, by), names in alike.items():
            size = (f" {'each hold' if len(names) > 1 else 'holds'} {total[0]} head{'s' * (total[0] != 1)}, "
                    f"{total[1]:,} tokens" if total else "")
            why = f" ({'also ' if mine else ''}carried by {', '.join(by)})" if by else ""
            parts.append(", ".join(names) + size + why)
        carried = f" (carried by {', '.join(by for by, _ in carriers)})" if carriers else ""
        nowhere = ("unverified" if unverified.intersection(active) or doubtful else "none") + carried
        lines.append("Present situations holding this note: " + ("; ".join(parts) or nowhere))
    if claims is not None and claims.report and claims.removed:
        lines.append("A removed name stops holding this head in works that start from now on; a work already running "
                     "keeps the heads it started with.")
    return lines, frozenset(name in own for name in heads)  # only where a present situation holds the head


def _head_line(note: knowledge_store.KnowledgeNote) -> str:
    """How large a head this write leaves where a present situation holds the note, by its own names or carried by a
    held head: there it stands whole, and a note with no `## ` line is all head; in characters and in host tokens
    (``memory_floor.view_tokens``)."""
    from ouroboros.held_notes import note_head
    from ouroboros.memory_floor import view_tokens

    head, body = note_head(note.source), note.source.body_span
    size = f"{len(head):,} characters, {view_tokens(head):,} tokens"
    if len(head.encode("utf-8")) < body.end_byte - body.start_byte:
        return f"Held head: {size}, the body above the first `## ` line."
    return f"Held head: {size}: this note has no `## ` line outside a fence, so its whole body is the head."


def _link_lines(result: knowledge_store.KnowledgeWriteResult, holding: dict[str, dict[str, list[str]]],
                evidence: SituationReceipt) -> list[str]:
    """Where this note's own head carried or carries notes, beside who holds each after the write (``holding``,
    ``held_notes.standing``): each one newly carried or released with its head size, and its head links held
    nowhere. Facts only: no limit and no choice."""
    from ouroboros.held_notes import linked_heads, note_head

    note, claims, lines = result.current, result.claims, []
    topic = note.address.topic

    def present(names) -> str:
        matching, unverified = evidence.matching(list(names))
        return ", ".join(name for name in dict.fromkeys(names) if name in matching) or ("unverified" if unverified else "none")

    def carries(version, names) -> tuple[dict, list]:
        held = version.source is not None and version.state == "active" and names and topic != OVERVIEW_TOPIC
        return linked_heads(version) if held else ({}, [])

    def listed(targets, names, where=lambda _name: "") -> str:
        parts, total = [], 0
        for name in names:
            head, body = note_head(targets[name].source), targets[name].source.body_span
            whole = ("; no `## ` line outside a fence: the whole body"
                     if len(head.encode("utf-8")) >= body.end_byte - body.start_byte else "")
            parts.append(f"{name} ({len(head):,} characters{whole}{where(name)})")
            total += len(head)
        return f"{', '.join(parts)}; {total:,} characters in all"

    own = claims.active if claims is not None else []
    after, unheld = carries(note, own)
    before = {}
    if claims is not None and claims.recorded and result.previous_revision:
        try:
            old = note if result.previous_revision == note.revision else knowledge_store._note(note.address, (
                knowledge_versions.find_version(note.address, result.previous_revision, note.text)[0].encode("utf-8")))
            before = carries(old, set(claims.recorded) & set(knowledge_versions.claim_names(old.metadata.get("claims"))))[0]
        except (knowledge_versions.VersionUnavailable, OSError) as exc:
            before = None
            lines.append(f"What this head carried before the write is unknown: {exc}.")
    if before is not None and (newly := [name for name in after if name not in before]):
        lines.append(f"Newly carried by this head: {listed(after, newly)}, standing where this note's own names hold it: "
                     f"{present(own)}.")  # one hop: where a link holds this note, its links hold nothing
    if before is not None and (released := [name for name in before if name not in after]):
        lines.append("Released by this head: " + listed(before, released, lambda name: "; now held: " + present(
            [held for names in holding.get(name, {}).values() for held in names])) + ".")
    if unheld:
        lines.append(f"Linked from this head, not held: {', '.join(unheld)}.")
    return lines


def _knowledge_write(
    ctx: ToolContext, topic: str, content: str | None = None, mode: str | None = None,
    scope: str = "", expected_revision: str | None = None, old_str: str | None = None,
    summary: str | None = None, reason: str = "", edits: list | None = None, version: str | None = None,
    new_str: str | None = None,
) -> str:
    if not isinstance(mode, str) or mode not in _MODES:  # no default operation: the call must choose one
        return _mode_refusal(ctx, mode)
    root = canonical_data_root(ctx)
    situation = receipt_situation(ctx, root)
    situation_line = _line(situation)
    notes = []  # a value that means what omission means takes the omitted path, said in the result (tools/arg_feedback)
    # The argument rules every exact editor shares (knowledge_edits); archive still requires its reason.
    version, edits, old_str, reason = normalize_arguments(mode, version, expected_revision, edits, old_str, reason,
                                                          notes, reason_required=("archive",))
    try:
        sanitized = _sanitize_topic(topic)
        lifecycle = mode in ("archive", "restore")
        if mode == "archive" and sanitized in knowledge_store.ALWAYS_ACTIVE_TOPICS:
            raise ValueError("the overview must stay active")
        summary = None if summary == "" else summary  # an empty summary asks for nothing
        if summary is not None and (not isinstance(summary, str) or not summary.strip()):
            raise ValueError(f"summary={summary!r} is not summary text; pass the revised summary, "
                             "or omit summary to keep the current one")
        if mode not in ("overwrite", "append", "edit") and summary is not None and mode in _MODES:
            raise ValueError(f"summary is used only with mode=overwrite, append or edit, not mode={mode}")
        fields = (mode == "edit" and old_str in (None, "") and edits is None and isinstance(content, str) and bool(content)
                  and knowledge_store.yaml_fields(content) is not None)  # a YAML block alone sets fields, keeps the body
        content, old_str, edits = check_arguments(
            mode, content, old_str, new_str, edits, version, reason, notes, modes=_MODES,
            no_content=("revert", "archive", "restore"), reason_required=("archive",), alone="summary",
            alone_given=summary is not None, fields=fields, version_reader="knowledge_read(version=...)")
        address = _address(ctx, sanitized, scope)
        # The turn is the writer; the route stamp is the route that ANSWERED the
        # loop's last round (provider + resolved model, account when Claudexor
        # served it), recorded by the loop, otherwise honestly unknown. The host
        # signs which focus wrote it (``focus_signature``), never the writer's own claim.
        result = knowledge_store.write_knowledge_note(
            address, content, mode, expected_revision,
            str(getattr(ctx, "task_id", "") or ""), old_str, writer="turn",
            route=(getattr(ctx, "_accumulated_usage", None) or {}).get("_observed_route") or None,
            summary=summary, focus=knowledge_store.focus_signature(ctx)["focus"], reason=reason,
            capture_previous=(lambda note: _capture_previous(ctx, note)) if lifecycle else None,
            edits=edits, version=version,
        )
    except ValueError as exc:
        said = f" ({'; '.join(notes)})" if notes else ""
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=_with_lines(f"⚠️ TOOL_ARG_ERROR: {exc}{said}", situation_line)))
    except (OSError, UnicodeDecodeError) as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_REPORTED_FAILURE", text=_with_lines(
                f"⚠️ TOOL_ERROR: Knowledge write failed: {type(exc).__name__}", situation_line)))
    meta = {"knowledge_write_reason": result.reason, **({"ignored_arguments": notes} if notes else {})}
    if result.history_ref is not None:
        meta["knowledge_previous_source"] = result.history_ref
    if result.delta is not None:
        meta["knowledge_delta"] = result.delta
    if result.current is not None:
        meta["knowledge_source"] = result.current.source_ref()
        meta["knowledge_state"] = result.current.state
    if result.ok:
        if result.current is not None:  # before the bound, which then makes room for it
            meta["knowledge_residency"] = _residency(result.current, result.delta)
        if result.reason == "saved" and result.previous is not None:  # who wrote what this replaced, and its calls
            where = {"topic": result.current.address.topic, "scope": result.current.address.scope}
            meta["knowledge_previous"] = result.previous
            meta["knowledge_read_previous"] = {"tool": "knowledge_read",
                                               "arguments": {**where, "version": result.previous_revision}}
            if not lifecycle:  # a revert across archive states refuses; the opposite lifecycle mode is the way back
                meta["knowledge_revert"] = {"tool": "knowledge_write", "arguments": {
                    **where, "mode": "revert", "version": result.previous_revision,
                    "expected_revision": result.current.revision}}
            if not expected_revision:  # a bare append: earlier text stood unseen
                meta["knowledge_outline"] = knowledge_versions.render_outline(result.current)
                meta["knowledge_note"] = "Earlier text was not revised by this write."
        outline = meta.get("knowledge_outline")
        published = _bound_delta_meta(meta)
        text = f"✅ Knowledge '{sanitized}' {result.reason} ({mode}).\n" + json.dumps(meta, ensure_ascii=False, sort_keys=True)
        from ouroboros.held_notes import standing

        claim_lines, link_lines, held, head = [], [], frozenset(), []
        note = result.current  # links hold only once some claim was recorded on this shelf
        linked = (note is not None and note.address.scope == "global"
                  and knowledge_versions.claim_record_path(note.address.shelf, note.address.topic).parent.is_dir())
        holding = standing(root) if linked else {}  # who holds each note, read once for every line below
        carried = linked and any(carrier for carrier in holding.get(note.address.topic, {}))
        named = carried or result.claims is not None and (result.claims.report or _holds_names(result))
        if named or linked:
            evidence = situation or receipt_situation(ctx, root, claims_written=True)
            if named:
                claim_lines, held = _claim_lines(result, evidence, root, holding)
            if linked:
                link_lines = _link_lines(result, holding, evidence)
        if held:  # a size and where it yields are said only where a present situation holds the head
            head = [_head_line(note), _YIELDS[held]]
        text = _with_lines(text, situation_line, *claim_lines, *link_lines, *head)
        if isinstance(outline, str) and not isinstance(meta["knowledge_outline"], str):
            # The bare appender's only look at the note: an outline the meta cannot hold follows
            # the JSON line whole; the meta keeps its pointer (size, sha256, the read).
            text += "\n\n" + outline
        return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=text, meta=published))
    text = f"⚠️ TOOL_ERROR: Knowledge write not completed: {result.reason}."
    changes = None
    if result.reason == "revision_conflict" and result.held is not None and result.current is not None:
        # The held version plus what came since is the whole current note: the note is not resent.
        reply = (text + f" Nothing was written. Reconcile your change with what came since, then pass "
                 f"expected_revision={result.current.revision!r}.")
        diff = knowledge_versions.render_changes(result.current, result.held)
        if len(reply) + 2 + len(diff) <= tool_result_limit("knowledge_write"):
            changes = _with_lines(reply, situation_line) + "\n\n" + diff
    if changes is not None:
        text = changes
        meta["knowledge_changes_since"] = result.held["facts"]["revision"]
    elif not result.show_source:  # a refused edit item or revert says why; the note the caller holds is not resent
        text = _with_lines(text + " Nothing was written.", situation_line)
    else:
        if result.reason in ("revision_required", "revision_conflict"):
            text += " Read the current source and pass its revision as expected_revision when replacing it."
        text = _with_lines(text, situation_line)
        if result.current is not None:
            view, current_meta = _source_view(result.current)
            meta.update(current_meta)
            # The complete current source is useful immediately, without a hidden
            # retry or another model deciding whether the stale write was safe.
            text += "\n\n" + view
            meta["knowledge_body_start"] = len(text) - len(result.current.text)
    return _publish_tool_result(ctx, ToolResult(
        status="error", code="TOOL_REPORTED_FAILURE", text=text, meta=_bound_delta_meta(meta)))


@completed_local_read
def _knowledge_list(ctx: ToolContext, scope: str = "", view: str = "active",
                    topics: List[str] | None = None) -> str:
    try:
        address = _address(ctx, "topic", scope)
        if topics is not None and (not isinstance(topics, list) or any(not isinstance(t, str) for t in topics)):
            raise ValueError("topics must be an array of topic paths or directory prefixes ending in /")
        return knowledge_store.knowledge_index_view(address, view, topics=tuple(topics or ()))
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR: {exc}"))


def get_tools() -> List[ToolEntry]:
    # The chronicle tools ride this module's export: the packaged build's frozen
    # module list already names it, and tools/chronicle.py has no get_tools of its own.
    from ouroboros.tools.chronicle import chronicle_tools

    topic = {"type": "string", "description": "Shelf-relative topic path without .md; nested paths and Unicode names are supported; no scope prefixes (global/, project/)."}
    scope = {"type": "string", "description": "global or project:<exact project id>. Omitted uses this task's project shelf, otherwise global. Global knowledge remains explicitly reachable from a project. Understanding of people and relationships, and anything that should outlive the project, belongs in global. The topics overview, patterns and improvement-backlog always resolve to global."}
    return [
        ToolEntry("knowledge_read", {
            "name": "knowledge_read",
            "description": ("Read the current complete knowledge note, active or archived, with its state, canonical address "
                            "and exact source revision. Paths and relative Markdown links survive archival. Read current "
                            "understanding before revising it. view='outline' shows its summary, revision, state, size and "
                            "every heading with its line and the start_char..end_char range of its section; view='history' "
                            "lists its versions newest first: who wrote each, when, in which mode, why, and what left; a "
                            "later page passes the fingerprint its first page names. "
                            "version=<revision> reads that earlier version exactly (verified by its sha256), as the note or "
                            "with view='outline' as its outline."),
            "parameters": {"type": "object", "properties": {"topic": topic, "scope": scope,
                "start_char": {"type": "integer", "description": "Optional half-open character range start in the exact complete note (omitted = 0). Use ranges to read a large source in parts."},
                "end_char": {"type": "integer", "description": "Exclusive range end (omitted = the end; a value past the end is lowered to it); complete_chars and revision are returned with every view. Omit both bounds, or pass 0 and 0, for the full note."},
                "view": {"type": "string", "enum": ["note", "outline", "history"], "description": "note (default): the note itself; outline: its headings with section ranges, to read one section with start_char/end_char; history: its versions, newest first. start_char/end_char page the outline and history text like a note's."},
                "version": {"type": "string", "description": "The revision of an earlier version of this note (view='history' or a write receipt names it): read that version instead of the current note."},
                "fingerprint": {"type": "string", "description": "view='history', a later page: the fingerprint its first page's header names; a changed list is refused."}},
                "required": ["topic"]},
        }, _knowledge_read),
        ToolEntry("knowledge_write", {
            "name": "knowledge_write",
            "description": (
                "Create, revise, append, revert or reversibly archive durable Markdown understanding. YAML carries type, "
                "an authored summary and optional title, keys and claims; unknown metadata survives. keys are exact outside "
                "names of the note's subject (an account, an email), each {provider, namespace, id}: a place turn states its "
                "speaker's key and the global notes holding it; equality alone joins them and grants no authority. claims "
                "lists exact situation names as knowledge_read and knowledge_write state them (room:<id>, place:<id>, "
                "conversation:…, thread:…; not the label in parentheses): where one is present, every integrating focus "
                "holds this active global note's head, the body above its first `## ` line. No selected mode shortens a "
                "held head; only the window can, to my summary and an exact reader I can open, or in an emergency to "
                "the reader alone. A name holds once a create, an overwrite or a YAML-only edit sets it. Each active "
                "summary stays resident in "
                "the index's whole view (a global note's in every room and place I work in, a project note's in that "
                "project), never in a Safety check; body text never replaces it. A write does not refresh a built context: a "
                "running work keeps the index it started with. Under physical window pressure, the outline names every active "
                "topic and keeps type person summaries; others stay readable with knowledge_list. A summary names the "
                "note's enduring scope and when to open it, and is not a dated log of the latest edit. "
                "mode is required and has no default; a call without it writes nothing. "
                "overwrite replaces the note with content; its frontmatter fields merge with the retained ones. "
                "append adds content at the end. edit replaces exact body text of the revision you read: one old_str by "
                "content (or new_str), or edits, a list of {old_str, new_str} items; YAML-only content sets only its "
                "fields. Each old_str must occur exactly "
                "once in that revision's body (never in an earlier item's result); items may touch but not overlap; all "
                "apply or none; an empty new_str deletes; a refused item is named by its number. "
                "revert puts back an earlier version byte for byte: version is its revision. Text a write removes stays "
                "in the note's history, which I can read (knowledge_read with view='history' or version=...) and put "
                "back (mode='revert'). archive and restore own only the archive field and keep the body bytes; archive "
                "requires reason. Archive hides default navigation rows; direct reads and explicit Presence topics stay "
                "available. Restore removes even malformed archive metadata from readable YAML in CURRENT text, "
                "including later edits. Other writes keep the archive field; a revert across "
                "archive states refuses. summary may ride with overwrite, append or edit (alone with edit). reason, "
                "optional on every mode, is kept in the note's history, never in its body. expected_revision (from "
                "knowledge_read, a receipt or a refusal) is required to replace an existing note: overwrite, edit, "
                "revert, archive, restore and an append with summary; creating a note and a bare append need none. A "
                "stale revision writes nothing and returns every change since that version in whole lines when the note's "
                "recent history holds it and one reply carries them all, else the current note. A receipt names the version "
                "it replaced, who wrote it, and the calls that read it and, for content writes, revert to it. The "
                "overview always stays active. Legacy index prose stays until an authored global overview retires it."),
            "parameters": {"type": "object", "properties": {
                "topic": topic, "scope": scope,
                "content": {"type": "string", "description": "Markdown, optionally with YAML frontmatter; supplied fields merge with retained ones. The text of overwrite and append, or the replacement of one old_str (empty deletes it). Omit for archive, restore, revert, an edits edit and a summary-only edit. No summary is generated from the body."},
                "mode": {"type": "string", "enum": list(_MODES), "description": "Required, with no default: overwrite, append, edit, revert, archive or restore, as this tool's description explains. Missing notes are created by overwrite/append only. The overview always stays active."},
                "reason": {"type": "string", "description": "Why this write: optional on every mode and required for archive, which also records it in the archive field. Kept in the note's history, never written into its body; no automatic classification or summarization occurs."},
                "old_str": {"type": "string", "description": "mode=edit, one pair: non-empty exact body text that occurs once; content (or new_str) replaces it (empty text deletes it). Omit both to change only the summary; edits replaces several places at once."},
                "new_str": {"type": "string", "description": "mode=edit with old_str: its replacement, the same as content (empty text deletes it). Pass one of the two."},
                "edits": {"type": "array", "minItems": 1, "description": "mode=edit: several exact replacements in one revision-checked write, all or nothing. Each item is {old_str, new_str}: old_str is exact body text found once in the revision you read; new_str replaces it, empty deletes. Spans may touch, not overlap.",
                          "items": {"type": "object", "additionalProperties": False, "required": ["old_str", "new_str"],
                                    "properties": {"old_str": {"type": "string"}, "new_str": {"type": "string"}}}},
                "summary": {"type": "string", "description": "A summary names the note's enduring scope and says when to open this note, not a dated log of its latest edit: it is shown in the resident knowledge index and can stand for a held head that the window reduces; the whole note remains readable on demand. Passed with overwrite, append or edit, it replaces the current summary in the same revision-checked write (mode=edit may pass it alone); omit it to keep the current one."},
                "expected_revision": {"type": "string", "description": "The note's current revision, as knowledge_read, a receipt or a refusal returns it. Required to replace an existing note: overwrite, edit, revert, archive, restore and an append with summary. Omit it for a bare append or to create a note; pass empty only to create a missing note (on an existing note an empty value is a stale revision). A stale revision writes nothing."},
                "version": {"type": "string", "description": "mode=revert: the revision of an earlier version of this note (knowledge_read(view='history') lists them); its exact stored bytes become the current note."},
            }, "required": ["topic", "mode"]},
        }, _knowledge_write),
        ToolEntry("knowledge_list", {
            "name": "knowledge_list",
            "description": "List current knowledge with authored summaries and source links. Optional topics select exact topics or directory prefixes ending in /. Default active view includes the archive count and address; archived/all reveal archived notes with archive time and reason. Legacy index prose can still mention archived notes until an authored global overview retires it.",
            "parameters": {"type": "object", "properties": {"scope": scope,
                "topics": {"type": "array", "items": {"type": "string"}, "description": "Optional exact topic paths or directory prefixes ending in /; omitted lists the whole shelf."},
                "view": {"type": "string", "enum": ["active", "archived", "all"], "description": "active (default), archived only, or complete inventory."}}, "required": []},
        }, _knowledge_list),
    ] + chronicle_tools()

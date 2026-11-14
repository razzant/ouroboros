"""Knowledge tool adapters over the common linked Markdown source owner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

from ouroboros import knowledge as knowledge_store
from ouroboros import knowledge_versions
from ouroboros.project_facts import placed_project_id
from ouroboros.tool_capabilities import tool_result_limit
from ouroboros.tool_access_paths import canonical_data_root
from ouroboros.knowledge import INDEX_FILE as INDEX_FILE, OVERVIEW_TOPIC as OVERVIEW_TOPIC
from ouroboros.knowledge import sanitize_topic as _sanitize_topic
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
                 end_char: int | None = None, earlier: dict | None = None, notes=()) -> tuple[str, dict]:
    """One exact character range of the note (``_range``), or of an ``earlier`` version of it."""
    ref = note.source_ref()
    ref["state"] = note.state
    text = note.text
    total = len(text)
    start, end, range_notes = _range(total, start_char, end_char)
    if earlier is not None:  # the bytes of a recorded version, read through its own call
        ref["read"] = {"tool": "knowledge_read", "arguments": {
            "topic": note.address.topic, "scope": note.address.scope, "version": note.revision}}
        ref["earlier_version"] = earlier
    ref.update(start_char=start, end_char=end, complete_chars=total)
    body = text[start:end]
    header = "[Knowledge source] " + json.dumps(ref, ensure_ascii=False, sort_keys=True) + "\n"
    header += "".join(f"[Range note] {item}\n" for item in range_notes) + _argument_notes(notes)
    if earlier is not None:
        written = ("no history row wrote it (an outside edit, or a write before history)"
                   if earlier["ts"] == knowledge_versions.UNKNOWN else
                   f"written {earlier['ts']} by focus {knowledge_versions.focus_text(earlier['focus'])} "
                   f"({earlier['mode']}; reason: {knowledge_versions.reason_text(earlier['reason'])})")
        header += f"[Earlier version, not the current note: {written}. The current revision is {earlier['current_revision']}.]\n"
    if note.parse_error:
        header += "[Metadata unavailable; the complete original Markdown follows.]\n"
    if note.archive_error:
        header += f"[Archive metadata invalid; kept active: {note.archive_error}.]\n"
    header += "\n"
    return header + body, {
        "knowledge_source": ref, "knowledge_body_start": len(header),
        "knowledge_body_chars": len(body), "knowledge_source_complete": start == 0 and end == total,
    }


def _rendered_view(view: str, text: str, facts: dict, start_char: int | None, end_char: int | None,
                   notes=()) -> tuple[str, dict]:
    """An outline or history listing, paged by the same character ranges as a note."""
    start, end, range_notes = _range(len(text), start_char, end_char, "view")
    head = {**facts, "view": view, "start_char": start, "end_char": end, "complete_chars": len(text)}
    header = (f"[Knowledge {view}] " + json.dumps(head, ensure_ascii=False, sort_keys=True) + "\n"
              + "".join(f"[Range note] {item}\n" for item in range_notes) + _argument_notes(notes) + "\n")
    return header + text[start:end], {"knowledge_view": head, "knowledge_body_start": len(header),
                                      "knowledge_body_chars": end - start}


def _note_view(note: knowledge_store.KnowledgeNote, view: str, version: str | None,
               start_char: int | None, end_char: int | None, notes=()) -> tuple[str, dict]:
    if view not in ("note", "outline", "history"):
        raise ValueError(f"view={view!r}: use note (default), outline or history")
    if version is not None and view == "history":
        raise ValueError("version selects one version for view=note or view=outline; view=history lists them all")
    shown, earlier = note, None
    if version is not None and version != note.revision:
        text, row = knowledge_versions.find_version(note.address, version)
        shown = knowledge_store._note(note.address, text.encode("utf-8"))
        earlier = {**knowledge_versions.row_facts(row), "revision": version, "current_revision": note.revision}
    if view == "note":
        return _source_view(shown, start_char, end_char, earlier, notes)
    facts = {"topic": note.address.topic, "scope": note.address.scope, "revision": shown.revision,
             **({"current_revision": note.revision, "earlier_version": earlier} if earlier else {})}
    if view == "history":
        return _rendered_view(view, knowledge_versions.render_history(note), facts, start_char, end_char, notes)
    label = f" — an earlier version, not the current note (current revision {note.revision})" if earlier else ""
    return _rendered_view(view, knowledge_versions.render_outline(shown, label), facts, start_char, end_char, notes)


def _knowledge_read(ctx: ToolContext, topic: str, scope: str = "",
                    start_char: int | None = None, end_char: int | None = None,
                    view: str = "note", version: str | None = None) -> str:
    notes = []  # a value that names the default takes it, said in the header (tools/arg_feedback)
    if view in (None, ""):
        notes.append(ignored_argument_note("view", view, "the default view, the note itself"))
        view = "note"
    if isinstance(version, str) and not version.strip():  # empty or whitespace: a blank value names no revision
        notes.append(ignored_argument_note("version", version, "no revision named; this is the current note"))
        version = None
    try:
        sanitized = _sanitize_topic(topic)
        address = _address(ctx, sanitized, scope)
        note = knowledge_store.read_knowledge_note(address)
        text, meta = (_source_view(note, start_char, end_char, notes=notes) if view == "note" and version is None
                      else _note_view(note, view, version, start_char, end_char, notes))
    except knowledge_versions.VersionUnavailable as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_REPORTED_FAILURE", text=f"⚠️ TOOL_ERROR: version_unavailable: {exc}",
            meta={"operation_outcome": "completed_no_effect"}))
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR: {exc}",
            meta={"operation_outcome": "completed_no_effect"}))
    except FileNotFoundError:
        elsewhere = ("" if address.scope == "global"
                     else f" A global note may exist: knowledge_read(topic={sanitized!r}, scope='global').")
        return _publish_tool_result(ctx, ToolResult(
            status="ok", code="LEGACY_WARNING", text=f"Topic {topic!r} not found in {address.scope}. Use knowledge_list to see available topics." + elsewhere,
            meta={"knowledge_address": address.as_dict(), "knowledge_missing": True,
                  "operation_outcome": "completed_no_effect"}))
    except (OSError, UnicodeDecodeError) as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_REPORTED_FAILURE", text=f"⚠️ TOOL_ERROR: Knowledge source unavailable: {type(exc).__name__}",
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
               else "Main, tasks, wakes and Presence; a tight window and a helper's or session's start show only its address and title")
    where = f"the global knowledge index ({loaders})" if note.address.scope == "global" else project
    return f"Residency: its summary ({len(note.summary)} chars) is shown in {where}; {tail}"


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
    """A write names its operation; without an exact mode nothing is written, and nothing else is read first.

    An omitted mode once meant overwrite, so a call meant to add one section replaced the whole note.
    Null and blank text say what omission says (``tools/arg_feedback``), now refused like it; any other
    value is refused as received, never trimmed, case-folded or matched to the nearest operation.
    """
    if mode is None:
        received = "mode is missing or null"
    elif not isinstance(mode, str):
        received = f"mode={mode!r} is not text"
    elif not mode.strip():
        # Reuse the canonical blank-value representation; keep the established
        # refusal wording for non-blank invalid values.
        received = ignored_argument_note("mode", mode, "is not an operation").replace(
            " ignored: is not an operation", " is not an operation")
    else:
        received = f"mode={mode!r} is not an operation"
    return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
        f"⚠️ TOOL_ARG_ERROR: {received}. knowledge_write has no default operation: pass mode as exactly one of "
        "overwrite (content replaces the whole body), append (content goes at the end), edit (old_str with "
        "content, or edits, replaces exact text; summary may go alone), revert (puts back the earlier version "
        "named by version), archive or restore (the archive state only; archive requires reason). version names "
        "an earlier version to put back; expected_revision is the current revision a replacing write checks. "
        "Nothing was written.")))


def _knowledge_write(
    ctx: ToolContext, topic: str, content: str | None = None, mode: str | None = None,
    scope: str = "", expected_revision: str | None = None, old_str: str | None = None,
    summary: str | None = None, reason: str = "", edits: list | None = None, version: str | None = None,
    new_str: str | None = None,
) -> str:
    if not isinstance(mode, str) or mode not in _MODES:  # no default operation: the call must choose one
        return _mode_refusal(ctx, mode)
    notes = []  # a value that means what omission means takes the omitted path, said in the result (tools/arg_feedback)
    blank_version = isinstance(version, str) and not version.strip()  # empty or whitespace: it names no revision
    # Outside revert, a blank version or the revision this call already names as expected_revision asks for nothing.
    if mode != "revert" and (blank_version or isinstance(version, str) and version == expected_revision):
        notes.append(ignored_argument_note("version", version, ("" if blank_version else "the same as expected_revision; ")
                                           + "only mode=revert takes a version"))
        version = None
    if isinstance(edits, list) and not edits:
        notes.append(ignored_argument_note("edits", edits, "an empty list asks for no replacement"))
        edits = None
    if old_str == "" and mode != "edit":
        notes.append(ignored_argument_note("old_str", old_str, "only mode=edit replaces text"))
        old_str = None
    if reason is None and mode != "archive":  # optional text: null is an omitted reason; archive still requires one
        notes.append(ignored_argument_note("reason", reason, "no reason is recorded"))
        reason = ""
    try:
        sanitized = _sanitize_topic(topic)
        lifecycle = mode in ("archive", "restore")
        if not isinstance(content, (str, type(None))):
            raise ValueError("content must be Markdown")
        if new_str is not None:  # the one-pair edit's replacement as other editors spell it: content
            if not isinstance(new_str, str):
                raise ValueError("new_str must be text, the replacement of old_str (empty text deletes it)")
            if mode == "edit" and edits is not None and new_str:
                raise ValueError("each edits item carries its own new_str; a top-level new_str belongs to the "
                                 "one-pair spelling with old_str")
            if mode == "edit" and isinstance(old_str, str) and old_str:
                if content and new_str and content != new_str:
                    raise ValueError("new_str and content both give old_str's replacement and differ; pass one")
                if content is not None and bool(content) != bool(new_str):  # an empty one beside the other is filler
                    empty, given = ("content", "new_str") if new_str else ("new_str", "content")
                    notes.append(ignored_argument_note(empty, "", f"{given} gives the replacement"))
                content = content or new_str  # both empty: the replacement is empty text, which deletes
            elif new_str:
                raise ValueError("new_str is the replacement of old_str in mode=edit, the same as content; "
                                 + ("pass old_str with it" if mode == "edit" else f"mode={mode} takes its text as content"))
            else:
                notes.append(ignored_argument_note("new_str", new_str, "only an edit with old_str takes a replacement"))
        if mode == "archive" and sanitized in knowledge_store.ALWAYS_ACTIVE_TOPICS:
            raise ValueError("the overview must stay active")
        if not isinstance(reason, str) or (mode == "archive" and not reason.strip()):
            raise ValueError("reason is required for archive and optional text on every other mode")
        summary = None if summary == "" else summary  # an empty summary asks for nothing
        if summary is not None and (not isinstance(summary, str) or not summary.strip()):
            raise ValueError(f"summary={summary!r} is not summary text; pass the revised summary, "
                             "or omit summary to keep the current one")
        if mode != "edit" and (old_str is not None or edits is not None):
            raise ValueError("old_str is used only with mode=edit" if old_str is not None
                             else "edits is used only with mode=edit")
        if mode not in ("overwrite", "append", "edit") and summary is not None:
            raise ValueError(f"summary is used only with mode=overwrite, append or edit, not mode={mode}")
        if mode == "revert" and (version is None or blank_version):
            raise ValueError("mode=revert requires version, the revision to put back")
        if mode != "revert" and version is not None:
            raise ValueError("version is used only with mode=revert; knowledge_read(version=...) reads one")
        if mode == "revert" and content:
            raise ValueError("mode=revert takes version, expected_revision and an optional reason; no content")
        if mode not in ("edit", "revert") and not lifecycle and content is None:
            raise ValueError(f"mode={mode} requires content")
        if (lifecycle or mode == "revert") and content is None:
            content = ""
        if mode == "edit" and edits is not None:
            if old_str not in (None, "") or content:
                raise ValueError("edits carries its own replacements: pass edits without old_str and content")
            content, old_str = "", None
        elif mode == "edit" and summary is not None and old_str in (None, ""):  # "" asks for nothing
            if content:
                raise ValueError(f"content ({len(content)} chars) has no old_str to replace; pass old_str "
                                 "for a body edit, or omit content to revise only the summary")
            content, old_str = "", None
        elif mode == "edit" and (not isinstance(old_str, str) or not old_str):
            raise ValueError("mode=edit requires edits, or a non-empty old_str, or a summary")
        elif mode == "edit" and content is None:
            raise ValueError("mode=edit with old_str requires content, its replacement "
                             "(empty text deletes the old_str span)")
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
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR: {exc}{said}"))
    except (OSError, UnicodeDecodeError) as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_REPORTED_FAILURE", text=f"⚠️ TOOL_ERROR: Knowledge write failed: {type(exc).__name__}"))
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
            where = {"topic": address.topic, "scope": address.scope}
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
        if isinstance(outline, str) and not isinstance(meta["knowledge_outline"], str):
            # The bare appender's only look at the note: an outline the meta cannot hold follows
            # the JSON line whole; the meta keeps its pointer (size, sha256, the read).
            text += "\n\n" + outline
        return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=text, meta=published))
    text = f"⚠️ TOOL_ERROR: Knowledge write not completed: {result.reason}."
    changes = None
    if result.reason == "revision_conflict" and result.held is not None and result.current is not None:
        # The held version plus what came since is the whole current note: the note is not resent.
        changes = (text + f" Nothing was written. Reconcile your change with what came since, then pass "
                   f"expected_revision={result.current.revision!r}.\n\n"
                   + knowledge_versions.render_changes(result.current, result.held))
        if len(changes) > tool_result_limit("knowledge_write"):
            changes = None  # all or whole: never a clipped diff; the complete current source follows instead
    if changes is not None:
        text = changes
        meta["knowledge_changes_since"] = result.held["facts"]["revision"]
    elif not result.show_source:  # a refused edit item or revert says why; the note the caller holds is not resent
        text += " Nothing was written."
    else:
        if result.reason in ("revision_required", "revision_conflict"):
            text += " Read the current source and pass its revision as expected_revision when replacing it."
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
                            "lists its versions newest first: who wrote each, when, in which mode, why, and what left. "
                            "version=<revision> reads that earlier version exactly (verified by its sha256), as the note or "
                            "with view='outline' as its outline."),
            "parameters": {"type": "object", "properties": {"topic": topic, "scope": scope,
                "start_char": {"type": "integer", "description": "Optional half-open character range start in the exact complete note (omitted = 0). Use ranges to read a large source in parts."},
                "end_char": {"type": "integer", "description": "Exclusive range end (omitted = the end; a value past the end is lowered to it); complete_chars and revision are returned with every view. Omit both bounds, or pass 0 and 0, for the full note."},
                "view": {"type": "string", "enum": ["note", "outline", "history"], "description": "note (default): the note itself; outline: its headings with section ranges, to read one section with start_char/end_char; history: its versions, newest first. start_char/end_char page the outline and history text like a note's."},
                "version": {"type": "string", "description": "The revision of an earlier version of this note (view='history' or a write receipt names it): read that version instead of the current note."}},
                "required": ["topic"]},
        }, _knowledge_read),
        ToolEntry("knowledge_write", {
            "name": "knowledge_write",
            "description": (
                "Create, revise, append, revert or reversibly archive durable Markdown understanding. YAML carries type, "
                "optional title and an authored summary; unknown metadata survives. Each active summary stays resident in "
                "the index's whole view (a global note's in every room and place I work in, a project note's in that "
                "project), never in a Safety check; body text never replaces it. Under physical window pressure, the outline names every active "
                "topic and keeps type person summaries; others remain readable with knowledge_list. A summary names the "
                "note's enduring scope and when to open it, and is not a dated log of the latest edit. "
                "mode is required and has no default; a call without it writes nothing. "
                "overwrite replaces the note with content; its frontmatter fields merge with the retained ones. "
                "append adds content at the end. edit replaces exact body text of the revision you read: one old_str by "
                "content (or new_str), or edits, a list of {old_str, new_str} items. Each old_str must occur exactly "
                "once in that revision's body (never in an earlier item's result); items may touch but not overlap; all "
                "apply or none; an empty new_str deletes; a refused item is named by its number. "
                "revert puts back an earlier version byte for byte: version is its revision. Text a write removes stays "
                "in the note's history, which I can read (knowledge_read with view='history' or version=...) and put "
                "back (mode='revert'). archive and restore own only the archive field and keep the body bytes; archive "
                "requires reason. Archive hides default navigation rows; direct reads and explicit Presence topics stay "
                "available. Restore removes even malformed archive metadata from readable YAML in CURRENT text, "
                "including later edits. Ordinary writes preserve the lifecycle-owned archive field, and a revert across "
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
                "summary": {"type": "string", "description": "A summary names the note's enduring scope and says when to open this note, not a dated log of its latest edit: it is shown in the resident knowledge index, while the body is read on demand. Passed with overwrite, append or edit, it replaces the current summary in the same revision-checked write (mode=edit may pass it alone); omit it to keep the current one."},
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
